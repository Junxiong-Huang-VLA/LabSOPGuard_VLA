from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schemas import SessionManifest


PLAN_SCHEMA_VERSION = "key_action_long_video_plan.v1"
CHECKPOINT_SCHEMA_VERSION = "key_action_long_video_checkpoint.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _duration_for_view(
    manifest: SessionManifest,
    validation_result: Mapping[str, Any] | None,
    view: str,
    *,
    dry_run: bool,
) -> float:
    video_sources = (validation_result or {}).get("video_sources")
    if isinstance(video_sources, Mapping):
        info = video_sources.get(view)
        if isinstance(info, Mapping):
            duration = _as_float(info.get("duration_sec"), 0.0)
            if duration > 0:
                return duration
    if dry_run:
        return 960.0
    source = manifest.videos.first_person if view == "first_person" else manifest.videos.third_person
    fps = float(source.fps or 30.0)
    path = Path(source.path)
    if path.exists():
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:
                video_fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
                frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
                if video_fps > 0 and frame_count > 0:
                    return frame_count / video_fps
            finally:
                cap.release()
        except Exception:
            pass
    return 0.0


def _chunk_id(view: str, index: int, start_sec: float, end_sec: float) -> str:
    raw = f"{view}|{index}|{start_sec:.3f}|{end_sec:.3f}"
    return f"chunk_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _cache_key(view: str, path: str, start_sec: float, end_sec: float, fps: float, stage: str) -> str:
    raw = f"{view}|{path}|{start_sec:.3f}|{end_sec:.3f}|{fps:.6f}|{stage}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _existing_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _completed_chunk_ids(checkpoint: Mapping[str, Any]) -> set[str]:
    chunks = checkpoint.get("chunks")
    if not isinstance(chunks, list):
        return set()
    return {
        str(item.get("chunk_id"))
        for item in chunks
        if isinstance(item, Mapping) and item.get("chunk_id") and str(item.get("status") or "") == "completed"
    }


def _chunk_rows(
    *,
    view: str,
    video_path: str,
    duration_sec: float,
    chunk_sec: float,
    stage1_sample_fps: float,
    stage2_sample_fps: float,
    cache_root: Path,
    completed_ids: set[str],
) -> list[dict[str, Any]]:
    if duration_sec <= 0:
        return []
    rows: list[dict[str, Any]] = []
    start = 0.0
    index = 0
    while start < duration_sec - 1e-9:
        end = min(duration_sec, start + chunk_sec)
        chunk_id = _chunk_id(view, index, start, end)
        stage1_key = _cache_key(view, video_path, start, end, stage1_sample_fps, "coarse")
        stage2_key = _cache_key(view, video_path, start, end, stage2_sample_fps, "refine")
        rows.append(
            {
                "chunk_id": chunk_id,
                "view": view,
                "chunk_index": index,
                "start_sec": round(start, 6),
                "end_sec": round(end, 6),
                "duration_sec": round(end - start, 6),
                "status": "completed" if chunk_id in completed_ids else "pending",
                "cache": {
                    "coarse_frame_rows": str(cache_root / f"{stage1_key}.jsonl"),
                    "refine_frame_rows": str(cache_root / f"{stage2_key}.jsonl"),
                    "summary": str(cache_root / f"{chunk_id}_summary.json"),
                },
                "sampling": {
                    "coarse_sample_fps": stage1_sample_fps,
                    "refine_sample_fps": stage2_sample_fps,
                },
            }
        )
        index += 1
        start = end
    return rows


def build_long_video_processing_plan(
    manifest: SessionManifest,
    validation_result: Mapping[str, Any] | None,
    detector_config: Any,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
    output_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a resumable chunk/cache plan used by long-video pipeline runs.

    The current detectors still own frame extraction, but this plan is the stable
    contract for chunked processing, cache paths, checkpoint resume, and two-stage
    sampling. It is intentionally runnable in dry-run sessions with no video files.
    """

    root = Path(output_dir)
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    target = Path(output_path) if output_path is not None else metadata / "long_video_processing_plan.json"
    checkpoint_target = Path(checkpoint_path) if checkpoint_path is not None else metadata / "long_video_checkpoint.json"
    checkpoint = _existing_checkpoint(checkpoint_target)
    completed_ids = _completed_chunk_ids(checkpoint) if bool(getattr(detector_config, "long_video_resume", True)) else set()

    chunk_sec = max(60.0, _as_float(getattr(detector_config, "long_video_chunk_sec", 1800.0), 1800.0))
    stage1_sample_fps = _as_float(
        getattr(detector_config, "long_video_stage1_sample_fps", None),
        _as_float(getattr(detector_config, "parent_sample_fps", None), _as_float(getattr(detector_config, "sample_fps", 2.0), 2.0)),
    )
    stage2_sample_fps = _as_float(
        getattr(detector_config, "long_video_stage2_sample_fps", None),
        _as_float(getattr(detector_config, "micro_refine_sample_fps", 8.0), 8.0),
    )
    cache_dir_value = getattr(detector_config, "long_video_cache_dir", None)
    cache_root = Path(cache_dir_value) if cache_dir_value else root / ".cache" / "key_action_indexer"
    cache_root.mkdir(parents=True, exist_ok=True)

    views = ["third_person"]
    if manifest.videos.first_person is not None:
        views.append("first_person")

    chunks: list[dict[str, Any]] = []
    video_summaries = []
    for view in views:
        source = manifest.videos.first_person if view == "first_person" else manifest.videos.third_person
        if source is None:
            continue
        duration = _duration_for_view(manifest, validation_result, view, dry_run=dry_run)
        view_chunks = _chunk_rows(
            view=view,
            video_path=source.path,
            duration_sec=duration,
            chunk_sec=chunk_sec,
            stage1_sample_fps=stage1_sample_fps,
            stage2_sample_fps=stage2_sample_fps,
            cache_root=cache_root / view,
            completed_ids=completed_ids,
        )
        chunks.extend(view_chunks)
        video_summaries.append(
            {
                "view": view,
                "path": source.path,
                "duration_sec": duration,
                "chunk_count": len(view_chunks),
                "sample_fps": {
                    "coarse": stage1_sample_fps,
                    "refine": stage2_sample_fps,
                },
            }
        )

    pending = sum(1 for item in chunks if item["status"] != "completed")
    completed = len(chunks) - pending
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_at": _now(),
        "session_id": manifest.session_id,
        "dry_run": bool(dry_run),
        "enabled": True,
        "chunk_sec": chunk_sec,
        "cache_enabled": bool(getattr(detector_config, "long_video_cache_enabled", True)),
        "resume_enabled": bool(getattr(detector_config, "long_video_resume", True)),
        "two_stage_sampling": bool(getattr(detector_config, "long_video_two_stage_sampling", True)),
        "cache_dir": str(cache_root),
        "checkpoint_path": str(checkpoint_target),
        "video_count": len(video_summaries),
        "chunk_count": len(chunks),
        "completed_chunk_count": completed,
        "pending_chunk_count": pending,
        "videos": video_summaries,
        "chunks": chunks,
    }
    target.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    checkpoint_rows = [
        {
            "chunk_id": item["chunk_id"],
            "view": item["view"],
            "chunk_index": item["chunk_index"],
            "status": item["status"],
            "cache": item["cache"],
            "updated_at": plan["created_at"],
        }
        for item in chunks
    ]
    checkpoint_doc = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "updated_at": plan["created_at"],
        "session_id": manifest.session_id,
        "plan_path": str(target),
        "chunk_count": len(checkpoint_rows),
        "completed_chunk_count": completed,
        "pending_chunk_count": pending,
        "chunks": checkpoint_rows,
    }
    checkpoint_target.write_text(json.dumps(checkpoint_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


__all__ = ["build_long_video_processing_plan"]
