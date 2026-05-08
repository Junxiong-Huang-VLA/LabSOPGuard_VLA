from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LABSOPGUARD = ROOT / "LabSOPGuard"
BACKEND = LABSOPGUARD / "backend"
EXPERIMENT_ID = "2190fe06-3619-45fc-96ef-1bb8afb9bdf9"
EXPERIMENT_DIR = LABSOPGUARD / "outputs" / "experiments" / EXPERIMENT_ID
OUTPUT_DIR = EXPERIMENT_DIR / "key_action_index"
RAW_DIR = EXPERIMENT_DIR / "raw"

THIRD_PERSON_VIDEO = RAW_DIR / "top_view.browser_h264.mp4"
FIRST_PERSON_VIDEO = RAW_DIR / "bottom_view.browser_h264.mp4"
THIRD_PERSON_MODEL = LABSOPGUARD / "models" / "yolo" / "third_person" / "best.pt"
FIRST_PERSON_MODEL = LABSOPGUARD / "models" / "yolo" / "first_person" / "best.pt"


for path in (str(ROOT / "src"), str(LABSOPGUARD), str(LABSOPGUARD / "src"), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def video_info(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": frame_count / fps if fps > 0 else 0.0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def build_manifest() -> Path:
    third_info = video_info(THIRD_PERSON_VIDEO)
    first_info = video_info(FIRST_PERSON_VIDEO)
    exp = read_json(EXPERIMENT_DIR / "experiment.json")
    session_start_time = "2026-04-24T16:57:18+08:00"
    detection_config = {
        "sample_fps": 2.0,
        "parent_sample_fps": 2.0,
        "micro_refine_sample_fps": 6.0,
        "enable_micro_refine_rescan": True,
        "start_threshold": 0.18,
        "end_threshold": 0.08,
        "start_min_duration_sec": 1.0,
        "end_min_duration_sec": 2.0,
        "merge_gap_sec": 3.0,
        "min_segment_duration_sec": 2.0,
        "buffer_sec": 1.0,
        "motion_normalization": "adaptive",
        "roi_mode": "manifest_or_default",
        "detector_backend": "yolo",
        "yolo_preferred_view": "first_person",
        "yolo_scan_both_views": True,
        "yolo_first_person_model_path": str(FIRST_PERSON_MODEL),
        "yolo_third_person_model_path": str(THIRD_PERSON_MODEL),
        "yolo_conf": 0.25,
        "yolo_iou": 0.45,
        "yolo_device": os.environ.get("KEY_ACTION_YOLO_DEVICE", "cpu"),
        "yolo_fallback_to_motion": False,
        "yolo_config": {
            "first_person_model_path": str(FIRST_PERSON_MODEL),
            "third_person_model_path": str(THIRD_PERSON_MODEL),
        },
    }
    manifest = {
        "session_id": EXPERIMENT_ID,
        "session_start_time": session_start_time,
        "videos": {
            "third_person": {
                "path": str(THIRD_PERSON_VIDEO),
                "start_time": session_start_time,
                "fps": third_info["fps"],
                "offset_sec": 0.0,
                "role": "third_person",
                "camera_id": "top_view",
            },
            "first_person": {
                "path": str(FIRST_PERSON_VIDEO),
                "start_time": session_start_time,
                "fps": first_info["fps"],
                "offset_sec": 0.0,
                "role": "first_person",
                "camera_id": "bottom_view",
            },
        },
        "detection_config": detection_config,
        "micro_segment_config": {
            "micro_min_duration_sec": 0.5,
            "micro_merge_gap_sec": 0.35,
        },
        "config": {
            "capability_gap_project_root": str(ROOT),
            "experiment_title": exp.get("title") or "固体称量实验",
        },
        "output_dir": str(OUTPUT_DIR),
    }
    manifest_path = EXPERIMENT_DIR / "key_action_yolo_rerun_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def update_experiment(summary: dict[str, Any], candidate_summary: dict[str, Any], report_summary: dict[str, Any]) -> None:
    import main

    exp_path = EXPERIMENT_DIR / "experiment.json"
    exp = read_json(exp_path)
    output_paths = {
        **(exp.get("output_paths") or {}),
        "key_action_index": str(OUTPUT_DIR),
    }
    output_paths = main._attach_professional_report_output_paths(EXPERIMENT_ID, output_paths, report_summary)
    exp["status"] = "analyzed"
    exp["processing_stage"] = "output_generation"
    exp["processing_error"] = None
    exp["completed_at"] = now_iso()
    exp["analyzed_at"] = exp["completed_at"]
    exp["output_paths"] = output_paths
    exp["key_action_index"] = {
        "status": "completed",
        "output_dir": str(OUTPUT_DIR),
        "segment_count": int(summary.get("segment_count") or summary.get("detected_segment_count") or 0),
        "material_candidates": candidate_summary,
        "model_paths_by_view": {
            "first_person": str(FIRST_PERSON_MODEL),
            "third_person": str(THIRD_PERSON_MODEL),
        },
    }
    exp.setdefault("metadata", {})["yolo_rerun"] = {
        "status": "completed",
        "completed_at": exp["completed_at"],
        "first_person_model_path": str(FIRST_PERSON_MODEL),
        "third_person_model_path": str(THIRD_PERSON_MODEL),
        "top_view_as": "third_person",
        "bottom_view_as": "first_person",
    }
    exp.setdefault("metadata", {})["professional_report"] = report_summary
    write_json(exp_path, exp)


def main() -> int:
    import main as backend_main
    from key_action_indexer.material_references import build_yolo_material_candidates, build_yolo_material_references
    from key_action_indexer.pipeline import run_pipeline

    for required in (THIRD_PERSON_VIDEO, FIRST_PERSON_VIDEO, THIRD_PERSON_MODEL, FIRST_PERSON_MODEL):
        if not required.exists():
            raise FileNotFoundError(required)

    backend_main._write_key_action_status(
        EXPERIMENT_ID,
        {
            "status": "running",
            "progress": 0.05,
            "message": "Running dual-view YOLO rerun from raw videos",
            "third_person_video_path": str(THIRD_PERSON_VIDEO),
            "first_person_video_path": str(FIRST_PERSON_VIDEO),
            "third_person_model_path": str(THIRD_PERSON_MODEL),
            "first_person_model_path": str(FIRST_PERSON_MODEL),
            "started_at": now_iso(),
            "completed_at": None,
            "error": None,
            "traceback": None,
        },
    )
    try:
        manifest_path = build_manifest()
        backend_main._write_key_action_status(EXPERIMENT_ID, {"progress": 0.15, "message": "YOLO pipeline started"})
        summary = run_pipeline(manifest_path, dry_run=False)
        backend_main._write_key_action_status(EXPERIMENT_ID, {"progress": 0.75, "message": "Building YOLO material references"})
        reference_summary = build_yolo_material_references(OUTPUT_DIR, archive_existing=True)
        backend_main._write_key_action_status(EXPERIMENT_ID, {"progress": 0.82, "message": "Building frontend review candidates"})
        candidate_summary = build_yolo_material_candidates(OUTPUT_DIR, archive_existing=False)
        candidate_summary["material_references"] = reference_summary
        backend_main._write_key_action_status(EXPERIMENT_ID, {"progress": 0.9, "message": "Generating professional report"})
        report_summary = backend_main._generate_professional_report_for_experiment(
            EXPERIMENT_ID,
            output_paths={"key_action_index": str(OUTPUT_DIR)},
        )
        update_experiment(summary, candidate_summary, report_summary)
        backend_main._write_key_action_status(
            EXPERIMENT_ID,
            {
                "status": "completed",
                "progress": 1.0,
                "message": "Dual-view YOLO evidence chain rebuilt",
                "completed_at": now_iso(),
                "summary": summary,
                "material_candidates": candidate_summary,
                "professional_report": report_summary,
            },
        )
        print(json.dumps({"status": "completed", "summary": summary, "candidates": candidate_summary, "report": report_summary}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        backend_main._write_key_action_status(
            EXPERIMENT_ID,
            {
                "status": "failed",
                "progress": 1.0,
                "message": str(exc),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at": now_iso(),
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
