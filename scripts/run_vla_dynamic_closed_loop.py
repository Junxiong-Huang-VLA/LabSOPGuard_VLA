from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.action.policy import ActionPolicy
from project_name.alerting.notifier import AlertNotifier
from project_name.common.config import load_yaml
from project_name.common.schemas import PerceptionResult
from project_name.detection.multi_level_detector import MultiLevelDetector
from project_name.language.instruction_parser import InstructionParser
from project_name.monitoring.sop_engine import SOPComplianceEngine
from project_name.reporting.pdf_report import generate_compliance_report
from project_name.video.capture import FramePacket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic VLA closed loop and export a 60s visual demo video."
    )
    parser.add_argument("--video", required=True, help="Input RGB video path")
    parser.add_argument(
        "--instruction",
        default="Pick the sample container and place it to the right zone carefully while avoiding obstacles.",
    )
    parser.add_argument("--sample-id", default="vla_dynamic_demo")
    parser.add_argument("--rules", default="configs/sop/rules.yaml")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--loop-input", action="store_true", default=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--out-video", default="outputs/predictions/vla_dynamic_60s.mp4")
    parser.add_argument("--out-json", default="outputs/predictions/vla_dynamic_60s.json")
    parser.add_argument("--out-report", default="outputs/reports/vla_dynamic_60s_report.pdf")
    return parser.parse_args()


def _pick_target(objects: List[Dict[str, Any]], target_object: str) -> Dict[str, Any] | None:
    if not objects:
        return None
    for obj in objects:
        if str(obj.get("label", "")).lower() == target_object.lower():
            return obj
    return max(objects, key=lambda x: float(x.get("score", 0.0)))


def _build_perception(obj: Dict[str, Any], target_name: str) -> PerceptionResult:
    bbox = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
    x1, y1, x2, y2 = bbox
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    confidence = float(obj.get("score", 0.0))
    depth_info = {
        "center_depth": 0.0,
        "region_depth_mean": 0.0,
        "region_depth_median": 0.0,
        "valid_depth_ratio": 0.0,
    }
    return PerceptionResult(
        target_name=target_name,
        bbox=bbox,
        center_point=[cx, cy],
        depth_info=depth_info,
        confidence=confidence,
        segmentation=None,
        region_reference={"type": "bbox", "bbox": bbox},
        xyz=None,
        target_representation={
            "target_name": target_name,
            "bbox": bbox,
            "center_point": [cx, cy],
            "depth_info": depth_info,
            "source": "dynamic_rgb_bootstrap",
        },
    )


def _draw_overlay(
    frame_bgr,
    instruction: str,
    det_objects: List[Dict[str, Any]],
    action_seq: List[str],
    violations: List[Dict[str, Any]],
    layer_outputs: Dict[str, Any],
) -> Any:
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    pose = (
        layer_outputs.get("layer1_realtime_pose", {}).get("pose_keypoints_17")
        if isinstance(layer_outputs, dict)
        else None
    )

    # COCO 17-keypoint skeleton edges (0-based index)
    skeleton_edges = [
        (5, 7), (7, 9), (6, 8), (8, 10),  # arms
        (5, 6), (5, 11), (6, 12), (11, 12),  # torso
        (11, 13), (13, 15), (12, 14), (14, 16),  # legs
        (0, 1), (0, 2), (1, 3), (2, 4),  # head
    ]

    if isinstance(pose, list) and len(pose) >= 17:
        pts = []
        for kp in pose[:17]:
            if not isinstance(kp, list) or len(kp) < 3:
                pts.append(None)
                continue
            x, y, c = float(kp[0]), float(kp[1]), float(kp[2])
            if c < 0.15:
                pts.append(None)
                continue
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= w or yi < 0 or yi >= h:
                pts.append(None)
                continue
            pts.append((xi, yi))
            cv2.circle(canvas, (xi, yi), 3, (255, 200, 0), -1, cv2.LINE_AA)
        for a, b in skeleton_edges:
            pa = pts[a] if a < len(pts) else None
            pb = pts[b] if b < len(pts) else None
            if pa is not None and pb is not None:
                cv2.line(canvas, pa, pb, (255, 170, 0), 2, cv2.LINE_AA)

    for obj in det_objects:
        x1, y1, x2, y2 = [int(v) for v in obj.get("bbox", [0, 0, 0, 0])]
        label = str(obj.get("label", "obj"))
        score = float(obj.get("score", 0.0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            canvas,
            f"{label}:{score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    panel_h = 160
    cv2.rectangle(canvas, (0, h - panel_h), (w, h), (20, 20, 20), -1)
    cv2.putText(
        canvas,
        f"Instruction: {instruction[:90]}",
        (10, h - 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Action: {' -> '.join(action_seq[:6])}",
        (10, h - 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    layer_tag = (
        f"L1:{layer_outputs.get('layer1_realtime_pose', {}).get('model','NA')} | "
        f"L2:{layer_outputs.get('layer2_action_analysis', {}).get('model','NA')} | "
        f"L3:{layer_outputs.get('layer3_vlm_semantic', {}).get('model','NA')} | "
        f"L4:{layer_outputs.get('layer4_step_anomaly', {}).get('framework','NA')}"
    )
    cv2.putText(
        canvas,
        layer_tag[:120],
        (10, h - 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 240, 255),
        1,
        cv2.LINE_AA,
    )
    alert_text = "ALERT: none" if not violations else f"ALERT: {violations[0].get('rule_id','violation')}"
    alert_color = (80, 220, 80) if not violations else (30, 30, 255)
    cv2.putText(canvas, alert_text, (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, alert_color, 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    rules = load_yaml(args.rules)
    detector = MultiLevelDetector(confidence_threshold=args.confidence_threshold)
    parser = InstructionParser()
    policy = ActionPolicy()
    engine = SOPComplianceEngine(rules=rules)
    notifier = AlertNotifier(alert_file="outputs/predictions/vla_dynamic_alerts.jsonl")

    parsed = parser.parse(args.instruction)
    max_frames = max(1, int(round(args.duration_sec * args.target_fps)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    step = max(1, int(round(src_fps / args.target_fps)))

    writer = None
    src_frame_id = 0
    global_frame_id = 0
    processed = 0
    trace: List[Dict[str, Any]] = []
    report_violations: List[Dict[str, Any]] = []

    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            if args.loop_input:
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    break
                src_frame_id = 0
                continue
            break
        if src_frame_id % step != 0:
            src_frame_id += 1
            continue

        pkt = FramePacket(
            frame_id=global_frame_id,
            timestamp_sec=float(processed / max(args.target_fps, 1e-6)),
            frame_bgr=frame,
            source=str(video_path),
        )
        det = detector.detect(pkt)
        target = _pick_target(det.objects, parsed.target_object)

        if target is not None:
            perception = _build_perception(target, parsed.target_object)
            action_plan = policy.plan(parsed, perception)
            action_seq = action_plan.action_sequence
            action_payload = {
                "action_sequence": action_plan.action_sequence,
                "grasp_point_xyz": action_plan.grasp_point_xyz,
                "end_effector_target_xyz": action_plan.end_effector_target_xyz,
                "robot_command": action_plan.robot_command,
                "metadata": action_plan.metadata,
            }
        else:
            action_seq = ["observe", "search_target"]
            action_payload = {"action_sequence": action_seq}

        violations = engine.update(det)
        alert_rows = notifier.notify(violations) if violations else []
        if violations:
            notifier.send_console(violations)
            report_violations.extend(alert_rows)

        frame_vis = _draw_overlay(
            frame_bgr=frame,
            instruction=args.instruction,
            det_objects=det.objects,
            action_seq=action_seq,
            violations=alert_rows,
            layer_outputs=det.layer_outputs,
        )

        if writer is None:
            out_video = Path(args.out_video)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame_vis.shape[:2]
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), float(args.target_fps), (w, h), True)
        writer.write(frame_vis)

        trace.append(
            {
                "sample_id": args.sample_id,
                "frame_id": det.frame_id,
                "timestamp_sec": det.timestamp_sec,
                "instruction": args.instruction,
                "parsed_instruction": parser.to_dict(parsed),
                "detection": {
                    "ppe": det.ppe,
                    "objects": det.objects,
                    "actions": det.actions,
                    "confidence": det.confidence,
                    "layer_outputs": det.layer_outputs,
                },
                "action_plan": action_payload,
                "alerts": alert_rows,
            }
        )
        processed += 1
        src_frame_id += 1
        global_frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    status = engine.build_status()
    report_data = {
        "title": "Lab SOP VLA Dynamic Demo Report",
        "session_id": args.sample_id,
        "compliance_ratio": float(status.get("compliance_ratio", 0.0)),
        "violations": report_violations,
    }
    report_path = generate_compliance_report(report_data, args.out_report)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "sample_id": args.sample_id,
                "video": str(video_path),
                "instruction": args.instruction,
                "target_fps": args.target_fps,
                "duration_sec_requested": args.duration_sec,
                "frames_processed": processed,
                "status": status,
                "report_path": report_path,
                "trace": trace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"vla demo video: {args.out_video}")
    print(f"vla trace json: {args.out_json}")
    print(f"vla report: {report_path}")
    print(f"frames processed: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
