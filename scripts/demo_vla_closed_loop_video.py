from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.action.policy import ActionPolicy
from project_name.common.schemas import PerceptionResult
from project_name.detection.multi_level_detector import MultiLevelDetector
from project_name.language.instruction_parser import InstructionParser
from project_name.video.capture import FramePacket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a visualized VLA closed-loop demo video with bbox + action plan overlay."
    )
    parser.add_argument("--video", required=True, help="Input RGB video path.")
    parser.add_argument(
        "--instruction",
        default="Pick the sample container and place it to the right zone carefully.",
        help="Natural language instruction for VLA planning.",
    )
    parser.add_argument("--sample-id", default="demo_session")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument(
        "--out-video",
        default="outputs/predictions/vla_demo_60s.mp4",
        help="Output visualization video path.",
    )
    parser.add_argument(
        "--out-json",
        default="outputs/predictions/vla_demo_60s.json",
        help="Output structured frame-level VLA trace json path.",
    )
    return parser.parse_args()


def _pick_target_object(
    objects: List[Dict[str, Any]],
    expected_label: str,
) -> Dict[str, Any] | None:
    if not objects:
        return None
    for obj in objects:
        if str(obj.get("label", "")).lower() == expected_label.lower():
            return obj
    return max(objects, key=lambda x: float(x.get("score", 0.0)))


def _to_perception_result(target_obj: Dict[str, Any], expected_label: str) -> PerceptionResult:
    bbox = [int(v) for v in target_obj.get("bbox", [0, 0, 0, 0])]
    x1, y1, x2, y2 = bbox
    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    confidence = float(target_obj.get("score", 0.0))
    depth_info = {
        "center_depth": 0.0,
        "region_depth_mean": 0.0,
        "region_depth_median": 0.0,
        "valid_depth_ratio": 0.0,
    }
    target_representation = {
        "target_name": expected_label,
        "bbox": bbox,
        "center_point": [cx, cy],
        "depth_info": depth_info,
        "source": "rgb_only_bootstrap",
    }
    return PerceptionResult(
        target_name=expected_label,
        bbox=bbox,
        center_point=[cx, cy],
        depth_info=depth_info,
        confidence=confidence,
        segmentation=None,
        region_reference={"type": "bbox", "bbox": bbox},
        xyz=None,
        target_representation=target_representation,
    )


def _draw_event_overlay(
    frame_bgr,
    detection: Dict[str, Any],
    action_sequence: List[str],
    instruction: str,
) -> Any:
    h, w = frame_bgr.shape[:2]
    canvas = frame_bgr.copy()

    for obj in detection.get("objects", []):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        score = float(obj.get("score", 0.0))
        label = str(obj.get("label", "object"))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            canvas,
            f"{label}:{score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    panel_h = 120
    cv2.rectangle(canvas, (0, h - panel_h), (w, h), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        f"Instruction: {instruction[:95]}",
        (12, h - 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    actions_text = " -> ".join(action_sequence[:5])
    cv2.putText(
        canvas,
        f"Planned Actions: {actions_text}",
        (12, h - 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "VLA Closed-Loop: Vision + Language + Action (dynamic stream)",
        (12, h - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _as_detection_dict(det: Any) -> Dict[str, Any]:
    return {
        "frame_id": int(det.frame_id),
        "timestamp_sec": float(det.timestamp_sec),
        "ppe": dict(det.ppe),
        "objects": list(det.objects),
        "actions": list(det.actions),
        "confidence": float(det.confidence),
    }


def _serialize_plan(plan: Any) -> Dict[str, Any]:
    return {
        "action_sequence": list(plan.action_sequence),
        "grasp_point_xyz": plan.grasp_point_xyz,
        "end_effector_target_xyz": plan.end_effector_target_xyz,
        "metadata": dict(plan.metadata),
    }


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = MultiLevelDetector(confidence_threshold=args.confidence_threshold)
    parser = InstructionParser()
    policy = ActionPolicy()
    parsed_instruction = parser.parse(args.instruction)

    max_frames = max(1, int(round(args.duration_sec * args.target_fps)))

    writer = None
    traces: List[Dict[str, Any]] = []
    processed = 0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    step = max(1, int(round(src_fps / args.target_fps)))

    frame_id = 0
    while processed < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_id % step != 0:
            frame_id += 1
            continue
        packet = FramePacket(
            frame_id=frame_id,
            timestamp_sec=float(frame_id / src_fps),
            frame_bgr=frame_bgr,
            source=str(video_path),
        )
        det = detector.detect(packet)
        det_dict = _as_detection_dict(det)
        target_obj = _pick_target_object(det_dict["objects"], parsed_instruction.target_object)

        if target_obj is None:
            action_sequence = ["observe", "search_target"]
            visual = _draw_event_overlay(packet.frame_bgr, det_dict, action_sequence, args.instruction)
            trace = {
                "sample_id": args.sample_id,
                "frame_id": det_dict["frame_id"],
                "timestamp_sec": det_dict["timestamp_sec"],
                "instruction": args.instruction,
                "parsed_instruction": parser.to_dict(parsed_instruction),
                "detection": det_dict,
                "perception": None,
                "action_plan": {"action_sequence": action_sequence},
            }
        else:
            perception = _to_perception_result(target_obj, parsed_instruction.target_object)
            plan = policy.plan(parsed_instruction, perception)
            visual = _draw_event_overlay(packet.frame_bgr, det_dict, plan.action_sequence, args.instruction)
            trace = {
                "sample_id": args.sample_id,
                "frame_id": det_dict["frame_id"],
                "timestamp_sec": det_dict["timestamp_sec"],
                "instruction": args.instruction,
                "parsed_instruction": parser.to_dict(parsed_instruction),
                "detection": det_dict,
                "perception": {
                    "target_name": perception.target_name,
                    "bbox": perception.bbox,
                    "center_point": perception.center_point,
                    "depth_info": perception.depth_info,
                    "confidence": perception.confidence,
                    "target_representation": perception.target_representation,
                },
                "action_plan": _serialize_plan(plan),
            }

        if writer is None:
            out_video = Path(args.out_video)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            h, w = visual.shape[:2]
            writer = cv2.VideoWriter(
                str(out_video),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.target_fps),
                (w, h),
                True,
            )
        writer.write(visual)
        traces.append(trace)
        processed += 1
        frame_id += 1

    cap.release()

    if writer is not None:
        writer.release()

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
                "trace": traces,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"demo video: {args.out_video}")
    print(f"trace json: {args.out_json}")
    print(f"frames processed: {processed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
