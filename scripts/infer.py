from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from project_name.common.config import load_yaml
from project_name.common.logging_utils import setup_logger
from project_name.pipelines.sop_monitor_pipeline import SOPMonitorPipeline
from project_name.vision.pose_depth_mvp import (
    PoseConfig,
    PoseDepthMVP,
    export_pose_csv,
    export_pose_jsonl,
    load_camera_info,
    load_extrinsics,
    run_pose_export_batch,
)

CSV_FIELDS = [
    "sample_id",
    "camera_id",
    "frame_id",
    "timestamp",
    "class_name",
    "confidence",
    "event_type",
    "sop_step",
    "violation_flag",
    "severity_level",
]


def export_events_jsonl(path: str | Path, events: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def export_events_csv(path: str | Path, events: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for e in events:
            writer.writerow({k: e.get(k) for k in CSV_FIELDS})


def _load_offsets(path: str | None) -> dict | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"camera offsets json not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_one(
    pipeline: SOPMonitorPipeline,
    video: str,
    sample_id: str,
    camera_id: str,
    max_frames: int,
    target_fps: float,
    camera_offsets_ms: dict | None,
) -> Dict[str, Any]:
    return pipeline.run(
        video_source=video,
        max_frames=max_frames,
        target_fps=target_fps,
        sample_id=sample_id,
        camera_id=camera_id,
        camera_offsets_ms=camera_offsets_ms,
    )


def _read_manifest(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _resolve_manifest_csv(manifest_csv: str | None, data_config: str) -> str:
    if manifest_csv:
        return manifest_csv
    cfg = load_yaml(data_config)
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    path = dataset_cfg.get("manifest_csv")
    if not path:
        raise ValueError(
            f"No --manifest-csv provided and no dataset.manifest_csv in {data_config}."
        )
    return str(path)


def _is_row_valid(row: Dict[str, str]) -> bool:
    valid_status = str(row.get("valid_status", "")).strip().lower()
    if valid_status:
        return valid_status == "valid"
    return str(row.get("pair_status", "")).strip().lower() == "paired"


def _build_pose_mvp(args: argparse.Namespace) -> PoseDepthMVP:
    pose_cfg_raw = load_yaml("configs/vision_pose.yaml")
    cfg = PoseConfig(
        model_path=str(args.pose_model or pose_cfg_raw.get("model_path", "yolo26s-pose.pt")),
        device=str(pose_cfg_raw.get("device", "cuda:0")),
        conf=float(pose_cfg_raw.get("conf", 0.25)),
        iou=float(pose_cfg_raw.get("iou", 0.45)),
        imgsz=int(pose_cfg_raw.get("imgsz", 960)),
        depth_window_size=int(pose_cfg_raw.get("depth_window_size", 5)),
        min_valid_depth_ratio=float(pose_cfg_raw.get("min_valid_depth_ratio", 0.2)),
        max_depth_m=float(pose_cfg_raw.get("max_depth_m", 3.0)),
        depth_unit=str(pose_cfg_raw.get("depth_unit", "auto")),
    )
    camera_intrinsics = load_camera_info(args.camera_info, pose_cfg_raw)
    extrinsics = load_extrinsics(args.extrinsics, pose_cfg_raw)
    return PoseDepthMVP(
        cfg=cfg,
        keypoint_names_cfg=pose_cfg_raw.get("keypoint_names", {}),
        class_alias_cfg=pose_cfg_raw.get("class_alias", {}),
        camera_intrinsics=camera_intrinsics,
        extrinsics=extrinsics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SOP monitoring inference on one video or batch from manifest"
    )
    parser.add_argument("--video", default=None, help="Single video path.")
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Batch manifest csv. Expected columns include sample_id,rgb_path,valid_status.",
    )
    parser.add_argument(
        "--valid-only",
        action="store_true",
        help="When using --manifest-csv, only run rows with valid_status=valid.",
    )
    parser.add_argument("--rules", default="configs/sop/rules.yaml")
    parser.add_argument("--data-config", default="configs/data/dataset.yaml")
    parser.add_argument("--sample-id", default="session_demo")
    parser.add_argument("--camera-id", default="cam0")
    parser.add_argument("--camera-offsets-json", default=None)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--output", default="outputs/predictions/infer_result.json")
    parser.add_argument(
        "--batch-output-dir",
        default="outputs/predictions/batch_infer",
        help="Per-sample result folder for batch mode.",
    )
    parser.add_argument("--enable-pose", action="store_true", help="Enable pose + depth + 3D export.")
    parser.add_argument("--pose-model", default=None, help="Pose model path, default from configs/vision_pose.yaml")
    parser.add_argument("--depth-path", default=None, help="Optional depth image/video for single-source inference.")
    parser.add_argument("--camera-info", default=None, help="Camera intrinsics yaml/json path.")
    parser.add_argument("--extrinsics", default=None, help="Camera->base extrinsics yaml/json path.")
    parser.add_argument("--export-3d", action="store_true", help="Export infer_events_pose.jsonl/.csv")
    parser.add_argument("--export-base-frame", action="store_true", help="Export keypoints in robot base frame.")
    parser.add_argument("--debug-overlay", action="store_true", help="Export debug overlay images with bbox/kps.")
    args = parser.parse_args()

    if not args.video and not args.manifest_csv:
        args.manifest_csv = _resolve_manifest_csv(args.manifest_csv, args.data_config)

    logger = setup_logger("infer")
    rules = load_yaml(args.rules)
    pipeline = SOPMonitorPipeline(rules=rules)
    offsets = _load_offsets(args.camera_offsets_json)
    pose_mvp: PoseDepthMVP | None = None
    if args.enable_pose or args.export_3d:
        try:
            pose_mvp = _build_pose_mvp(args)
            for w in pose_mvp.init_warnings:
                logger.warning("pose init warning: %s", w)
        except Exception as exc:
            logger.warning("pose module init failed, keep base infer only: %s", exc)
            pose_mvp = None

    if args.video:
        result = _run_one(
            pipeline=pipeline,
            video=args.video,
            sample_id=args.sample_id,
            camera_id=args.camera_id,
            max_frames=args.max_frames,
            target_fps=args.target_fps,
            camera_offsets_ms=offsets,
        )

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

        export_events_jsonl("outputs/predictions/infer_events.jsonl", result["events"])
        export_events_csv("outputs/reports/infer_events.csv", result["events"])
        logger.info("saved %s", out)
        logger.info("saved outputs/predictions/infer_events.jsonl")
        logger.info("saved outputs/reports/infer_events.csv")

        if pose_mvp is not None and args.export_3d:
            pose_rows = run_pose_export_batch(
                mvp=pose_mvp,
                sources=[
                    {
                        "sample_id": args.sample_id,
                        "camera_id": args.camera_id,
                        "video_path": args.video,
                        "depth_path": args.depth_path,
                    }
                ],
                max_frames=args.max_frames,
                target_fps=args.target_fps,
                export_base_frame=bool(args.export_base_frame),
                debug_overlay=bool(args.debug_overlay),
                overlay_dir="outputs/predictions/debug_overlay",
                logger=logger,
            )
            export_pose_jsonl("outputs/predictions/infer_events_pose.jsonl", pose_rows)
            export_pose_csv("outputs/reports/infer_events_pose.csv", pose_rows)
            logger.info("saved outputs/predictions/infer_events_pose.jsonl")
            logger.info("saved outputs/reports/infer_events_pose.csv")
        return 0

    rows = _read_manifest(args.manifest_csv)
    if args.valid_only:
        rows = [r for r in rows if _is_row_valid(r)]

    batch_dir = Path(args.batch_output_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    merged_events: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        sample_id = row.get("sample_id") or f"sample_{idx:04d}"
        video = row.get("rgb_path") or row.get("video_path")
        if not video:
            logger.warning("skip sample without rgb/video path: %s", sample_id)
            continue

        result = _run_one(
            pipeline=pipeline,
            video=video,
            sample_id=sample_id,
            camera_id=args.camera_id,
            max_frames=args.max_frames,
            target_fps=args.target_fps,
            camera_offsets_ms=offsets,
        )
        sample_out = batch_dir / f"{sample_id}.json"
        sample_out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        merged_events.extend(result.get("events", []))
        summary.append(
            {
                "sample_id": sample_id,
                "video": video,
                "events": len(result.get("events", [])),
                "violations": len(result.get("violations", [])),
                "output": str(sample_out).replace("\\", "/"),
            }
        )
        logger.info("batch [%d/%d] saved %s", idx + 1, len(rows), sample_out)

    summary_out = batch_dir / "summary.json"
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    export_events_jsonl(batch_dir / "all_events.jsonl", merged_events)
    export_events_csv(batch_dir / "all_events.csv", merged_events)

    if pose_mvp is not None and args.export_3d:
        pose_sources: List[Dict[str, Any]] = []
        for idx, row in enumerate(rows):
            sample_id = row.get("sample_id") or f"sample_{idx:04d}"
            video = row.get("rgb_path") or row.get("video_path")
            if not video:
                continue
            pose_sources.append(
                {
                    "sample_id": sample_id,
                    "camera_id": args.camera_id,
                    "video_path": video,
                    "depth_path": row.get("depth_path") or args.depth_path,
                }
            )
        pose_rows = run_pose_export_batch(
            mvp=pose_mvp,
            sources=pose_sources,
            max_frames=args.max_frames,
            target_fps=args.target_fps,
            export_base_frame=bool(args.export_base_frame),
            debug_overlay=bool(args.debug_overlay),
            overlay_dir=batch_dir / "debug_overlay",
            logger=logger,
        )
        export_pose_jsonl(batch_dir / "infer_events_pose.jsonl", pose_rows)
        export_pose_csv(batch_dir / "infer_events_pose.csv", pose_rows)
        logger.info("saved %s", batch_dir / "infer_events_pose.jsonl")
        logger.info("saved %s", batch_dir / "infer_events_pose.csv")

    logger.info("saved %s", summary_out)
    logger.info("saved %s", batch_dir / "all_events.jsonl")
    logger.info("saved %s", batch_dir / "all_events.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
