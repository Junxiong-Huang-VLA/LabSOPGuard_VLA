from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = ROOT / "outputs" / "experiments"
DEFAULT_INPUT_DIR = Path("C:/Users/Xx7/Desktop") / "固体称量双视角实验-5.8"
DEFAULT_EXPERIMENT_ID = "solid-weighing-dual-view-20260508-153648"
DEFAULT_TITLE = "固体称量双视角实验-5.8"


for path in (ROOT, ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _copy2(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _find_required_input(input_dir: Path, pattern: str) -> Path:
    matches = sorted(input_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Missing input matching {pattern!r} in {input_dir}")
    return matches[0]


def _probe(path: Path, index: int, role: str, camera_id: str) -> dict[str, Any]:
    from backend import main as backend_main

    metadata = backend_main._probe_video_metadata(path, index)
    metadata["view_type"] = role
    metadata["role"] = role
    metadata["camera_id"] = camera_id
    metadata["video_path"] = str(path)
    metadata["source"] = str(path)
    metadata["source_type"] = "file"
    metadata["ingest_mode"] = "file"
    metadata["start_offset_sec"] = 0.0
    return metadata


def _prepare_experiment(*, experiment_id: str, title: str, input_dir: Path) -> dict[str, Any]:
    from backend import main as backend_main

    session = _read_json(input_dir / "session.json")
    session_start_time = str(session.get("start_utc") or _now_iso())
    third_source = _find_required_input(input_dir, "camera_00*_rgb_1920x1080_30fps.mp4")
    first_source = _find_required_input(input_dir, "camera_01*_rgb_1920x1080_30fps.mp4")
    third_meta_source = _find_required_input(input_dir, "camera_00*_rgb_1920x1080_30fps.json")
    first_meta_source = _find_required_input(input_dir, "camera_01*_rgb_1920x1080_30fps.json")
    alignment_source = _find_required_input(input_dir, "*.alignment.csv")

    exp_dir = EXPERIMENTS_ROOT / experiment_id
    raw_dir = exp_dir / "raw"
    third_path = raw_dir / "top_view.browser_h264.mp4"
    first_path = raw_dir / "bottom_view.browser_h264.mp4"
    _copy2(third_source, third_path)
    _copy2(first_source, first_path)
    _copy2(third_meta_source, raw_dir / "top_view.source.json")
    _copy2(first_meta_source, raw_dir / "bottom_view.source.json")
    _copy2(alignment_source, raw_dir / "dual_view_alignment.csv")
    if (input_dir / "session.json").exists():
        _copy2(input_dir / "session.json", raw_dir / "session.json")

    third_metadata = _probe(third_path, 0, "third_person", "top_view")
    first_metadata = _probe(first_path, 1, "first_person", "bottom_view")
    exp = {
        "experiment_id": experiment_id,
        "title": title,
        "description": (
            "New dual-view solid weighing analysis. camera_00 is mapped to third_person/top_view; "
            "camera_01 is mapped to first_person/bottom_view."
        ),
        "status": "video_uploaded",
        "processing_stage": "key_action_index",
        "created_at": session_start_time,
        "started_at": _now_iso(),
        "completed_at": None,
        "analyzed_at": None,
        "video_paths": [str(third_path), str(first_path)],
        "video_inputs": [
            {
                "video_index": 0,
                "video_path": str(third_path),
                "source": str(third_path),
                "source_type": "file",
                "ingest_mode": "file",
                "camera_id": "top_view",
                "role": "third_person",
                "view_type": "third_person",
                "start_offset_sec": 0.0,
            },
            {
                "video_index": 1,
                "video_path": str(first_path),
                "source": str(first_path),
                "source_type": "file",
                "ingest_mode": "file",
                "camera_id": "bottom_view",
                "role": "first_person",
                "view_type": "first_person",
                "start_offset_sec": 0.0,
            },
        ],
        "video_metadata": [third_metadata, first_metadata],
        "output_paths": {
            "source_video": str(third_path),
            "key_action_index": str(exp_dir / "key_action_index"),
            "experiment_json": str(exp_dir / "experiment.json"),
        },
        "metadata": {
            "source_session": session,
            "view_mapping": {
                "camera_00": "third_person/top_view",
                "camera_01": "first_person/bottom_view",
            },
        },
        "key_action_index": {
            "status": "queued",
            "third_person_video_path": str(third_path),
            "first_person_video_path": str(first_path),
            "session_start_time": session_start_time,
            "output_dir": str(exp_dir / "key_action_index"),
        },
        "processing_error": None,
    }
    backend_main._ensure_experiment_run_metadata(exp)
    exp["analysis_job_id"] = exp.get("analysis_job_id") or f"key_action_{experiment_id}"
    backend_main._save_experiment(exp)
    return {
        "experiment_id": experiment_id,
        "experiment_dir": str(exp_dir),
        "third_person_video_path": str(third_path),
        "first_person_video_path": str(first_path),
        "session_start_time": session_start_time,
    }


def _write_post_run_context(session_dir: Path) -> dict[str, Any]:
    from key_action_indexer.lightweight_context_import import import_lightweight_context

    return import_lightweight_context(
        session_dir,
        sop_text=(
            "1. 准备称量区域、称量纸和电子天平，确认手套手与称量纸、容器的接触证据。\n"
            "2. 使用药匙或刮勺取样并转移固体，关注 spatula 与手套手接触。\n"
            "3. 在电子天平上完成称量或读数确认，关注 balance 与样品、称量纸的同框证据。\n"
            "4. 操作试剂瓶或样品容器并收尾，关注 reagent_bottle、sample_bottle 与手套手接触。"
        ),
        note_text="本次用户指定双视角映射：camera_00 为第三人称视角，camera_01 为第一人称视角。",
        record_text="实验记录来自 2026-05-08 15:36:48 左右采集的双 Orbbec RGB 会话。",
    )


def _resolve_yolo_device(requested: str) -> str:
    value = str(requested or "auto").strip().lower()
    if value and value not in {"auto", "cuda"}:
        return value
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def _run_post_refresh(experiment_id: str, session_dir: Path) -> dict[str, Any]:
    import main as backend_main
    from key_action_indexer.derived_refresh import refresh_derived_artifacts
    from key_action_indexer.health_report import build_run_health_report
    from key_action_indexer.micro_quality_enrichment import enrich_micro_quality
    from key_action_indexer.quality_gate import build_quality_gate
    from key_action_indexer.reviewed_dataset import freeze_reviewed_dataset
    from key_action_indexer.video_understanding import build_video_understanding

    queries = ["固体称量", "balance weighing", "称量纸 手套 天平", "hand object interaction"]
    post: dict[str, Any] = {}
    post["context_text"] = _write_post_run_context(session_dir)
    post["micro_quality"] = enrich_micro_quality(session_dir)
    post["video_understanding"] = build_video_understanding(session_dir)
    post["derived_refresh"] = refresh_derived_artifacts(session_dir, query_texts=queries)
    post["health"] = build_run_health_report(
        session_dir,
        query_texts=queries,
        output_json=session_dir / "reports" / "run_health_report.json",
        output_md=session_dir / "reports" / "run_health_report.md",
    )
    post["reviewed_dataset"] = freeze_reviewed_dataset(session_dir)
    post["quality_gate"] = build_quality_gate(session_dir, output_path=session_dir / "reports" / "quality_gate.json")

    status = backend_main._read_key_action_status(experiment_id)
    status["post_refresh"] = {
        "micro_quality_report": post["micro_quality"].get("artifacts", {}),
        "derived_refresh_summary": post["derived_refresh"].get("paths", {}),
        "reviewed_release": (post["reviewed_dataset"].get("release") or {}).get("version"),
        "health_status": post["health"].get("status"),
        "health_gate_status": post["health"].get("gate_status"),
        "quality_gate_status": post["quality_gate"].get("status") or post["quality_gate"].get("gate_status"),
    }
    backend_main._write_key_action_status(experiment_id, status)
    return post


def run(args: argparse.Namespace) -> dict[str, Any]:
    from backend import main as backend_main

    input_dir = Path(args.input_dir)
    info = _prepare_experiment(experiment_id=args.experiment_id, title=args.title, input_dir=input_dir)
    yolo_device = _resolve_yolo_device(args.yolo_device)
    detection_config = backend_main._with_default_key_action_yolo_config(
        {
            "sample_fps": args.sample_fps,
            "parent_sample_fps": args.sample_fps,
            "micro_refine_sample_fps": args.micro_refine_sample_fps,
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
            "yolo_fallback_to_motion": False,
            "yolo_device": yolo_device,
        }
    )
    backend_main._run_key_action_index_task(
        args.experiment_id,
        third_person_video_path=info["third_person_video_path"],
        first_person_video_path=info["first_person_video_path"],
        session_start_time=info["session_start_time"],
        detection_config=detection_config,
    )
    task_status = backend_main._read_key_action_status(args.experiment_id)
    if str(task_status.get("status") or "").lower() == "failed":
        raise RuntimeError(f"Key-action task failed before post refresh: {task_status.get('message') or task_status.get('error')}")
    session_dir = EXPERIMENTS_ROOT / args.experiment_id / "key_action_index"
    post = _run_post_refresh(args.experiment_id, session_dir)
    result = {
        "status": "completed",
        "experiment": info,
        "job_status": backend_main._read_key_action_status(args.experiment_id),
        "post_refresh": post,
    }
    _write_json(session_dir / "reports" / "dual_view_run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 2026-05-08 dual-view solid weighing key-action workflow.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--sample-fps", type=float, default=float(os.environ.get("KEY_ACTION_SAMPLE_FPS", "2.0")))
    parser.add_argument("--yolo-device", default="auto")
    parser.add_argument(
        "--micro-refine-sample-fps",
        type=float,
        default=float(os.environ.get("KEY_ACTION_MICRO_REFINE_SAMPLE_FPS", "6.0")),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        try:
            from backend import main as backend_main

            backend_main._write_key_action_status(
                args.experiment_id,
                {
                    "status": "failed",
                    "progress": 1.0,
                    "message": str(exc),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_at": _now_iso(),
                },
            )
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
