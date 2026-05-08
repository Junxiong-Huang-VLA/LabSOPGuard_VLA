from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl, write_jsonl


SESSION_CONTEXT_SCHEMA_VERSION = "key_action_session_context_event.v1"
SESSION_CONTEXT_SUMMARY_SCHEMA_VERSION = "key_action_session_context_seed.v1"
SESSION_CONTEXT_FILENAME = "session_context_events.jsonl"
SESSION_CONTEXT_SUMMARY_FILENAME = "session_context_seed_summary.json"
SEED_EVENT_ID = "session_context_seed_000001"


def seed_session_context(
    session_dir: str | Path,
    *,
    force: bool = False,
    output_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Write a non-label operational context event derived from session metadata."""

    session = Path(session_dir)
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    target = metadata_dir / SESSION_CONTEXT_FILENAME
    summary_target = Path(output_summary_path) if output_summary_path else metadata_dir / SESSION_CONTEXT_SUMMARY_FILENAME

    existing_rows = _read_jsonl_if_exists(target)
    if existing_rows and not force:
        summary = _summary(
            session=session,
            target=target,
            rows=existing_rows,
            written=False,
            skipped=True,
            reason="session_context_events_already_present",
        )
        _write_summary(summary_target, summary)
        return summary

    manifest_path = session / "manifest.json"
    video_info_path = session / "video_info.json"
    pipeline_summary_path = session / "pipeline_summary.json"
    manifest = _read_json(manifest_path)
    video_info = _read_json(video_info_path)
    pipeline_summary = _read_json(pipeline_summary_path)
    if not manifest and not video_info and not pipeline_summary:
        raise FileNotFoundError(f"No manifest, video_info, or pipeline_summary found under {session}")

    seed_row = _seed_row(
        session=session,
        manifest_path=manifest_path if manifest_path.exists() else None,
        manifest=manifest,
        video_info=video_info,
        pipeline_summary=pipeline_summary,
    )
    preserved_rows = [row for row in existing_rows if not _is_seed_row(row)]
    rows = [*preserved_rows, seed_row]
    write_jsonl(target, rows)

    summary = _summary(
        session=session,
        target=target,
        rows=rows,
        written=True,
        skipped=False,
        reason=None,
        seed_event_id=seed_row["event_id"],
    )
    _write_summary(summary_target, summary)
    return summary


def _seed_row(
    *,
    session: Path,
    manifest_path: Path | None,
    manifest: Mapping[str, Any],
    video_info: Mapping[str, Any],
    pipeline_summary: Mapping[str, Any],
) -> dict[str, Any]:
    session_id = str(
        manifest.get("session_id")
        or pipeline_summary.get("session_id")
        or session.parent.name
        or session.name
    )
    session_start_time = _session_start_time(manifest)
    videos = _video_summaries(manifest, video_info)
    duration_sec = _max_video_duration(videos)
    detector_backend = _detector_backend(manifest, pipeline_summary)
    segment_count = pipeline_summary.get("segment_count")
    total_action_duration_sec = pipeline_summary.get("total_action_duration_sec")
    text = _context_text(
        session_id=session_id,
        videos=videos,
        detector_backend=detector_backend,
        duration_sec=duration_sec,
        segment_count=segment_count,
        total_action_duration_sec=total_action_duration_sec,
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SESSION_CONTEXT_SCHEMA_VERSION,
        "event_id": SEED_EVENT_ID,
        "session_id": session_id,
        "event_type": "session_context",
        "modality": "text",
        "source": "session_context",
        "source_path": str(manifest_path) if manifest_path else None,
        "source_row_index": 1,
        "synthetic": False,
        "non_label_context": True,
        "context_kind": "operational_session_metadata",
        "raw_timestamp": session_start_time,
        "global_time": session_start_time,
        "session_time_sec": 0.0 if session_start_time else None,
        "duration_sec": 0.0,
        "anchor_strategy": "session_start" if session_start_time else "unanchored_session_metadata",
        "anchor_confidence": 1.0 if session_start_time else 0.0,
        "text": text,
        "links": [],
        "payload": {
            "generated_at": generated_at,
            "session_dir": str(session),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "session_start_time": session_start_time,
            "video_duration_sec": duration_sec,
            "videos": videos,
            "detector_backend": detector_backend,
            "segment_count": segment_count,
            "total_action_duration_sec": total_action_duration_sec,
            "evidence_policy": {
                "is_manual_label": False,
                "is_sop_record": False,
                "is_database_record": False,
                "supports_strong_action_confirmation": False,
            },
        },
    }


def _context_text(
    *,
    session_id: str,
    videos: list[dict[str, Any]],
    detector_backend: str | None,
    duration_sec: float | None,
    segment_count: Any,
    total_action_duration_sec: Any,
) -> str:
    view_parts = []
    for video in videos:
        view_id = str(video.get("view_id") or "")
        camera_id = str(video.get("camera_id") or "") if video.get("camera_id") else ""
        role = str(video.get("role") or "") if video.get("role") else ""
        label = view_id
        if role and role != view_id:
            label = f"{label}/{role}" if label else role
        if camera_id:
            label = f"{label} camera={camera_id}" if label else f"camera={camera_id}"
        if label:
            view_parts.append(label)
    parts = [
        f"Session context seed for experiment {session_id}.",
        "Non-label operational metadata only; no human annotation, SOP, or database claim is encoded.",
    ]
    if view_parts:
        parts.append(f"Video inputs: {', '.join(view_parts)}.")
    if duration_sec is not None:
        parts.append(f"Max video duration {duration_sec:.3f} seconds.")
    if detector_backend:
        parts.append(f"Detector backend {detector_backend}.")
    if segment_count is not None:
        parts.append(f"Pipeline segment count {segment_count}.")
    if total_action_duration_sec is not None:
        parts.append(f"Total detected action duration {total_action_duration_sec} seconds.")
    return " ".join(parts)


def _video_summaries(manifest: Mapping[str, Any], video_info: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest_videos = manifest.get("videos") if isinstance(manifest.get("videos"), Mapping) else {}
    info_videos = video_info.get("video_sources") if isinstance(video_info.get("video_sources"), Mapping) else {}
    summaries = []
    for view_id in sorted({*manifest_videos.keys(), *info_videos.keys()}):
        manifest_row = manifest_videos.get(view_id) if isinstance(manifest_videos.get(view_id), Mapping) else {}
        info_row = info_videos.get(view_id) if isinstance(info_videos.get(view_id), Mapping) else {}
        summaries.append(
            {
                "view_id": str(view_id),
                "role": manifest_row.get("role") or view_id,
                "camera_id": manifest_row.get("camera_id"),
                "path": manifest_row.get("path") or info_row.get("path"),
                "start_time": manifest_row.get("start_time"),
                "offset_sec": manifest_row.get("offset_sec"),
                "fps": info_row.get("fps") or manifest_row.get("fps"),
                "width": info_row.get("width"),
                "height": info_row.get("height"),
                "duration_sec": info_row.get("duration_sec"),
                "exists": info_row.get("exists"),
                "can_open": info_row.get("can_open"),
            }
        )
    return summaries


def _max_video_duration(videos: list[Mapping[str, Any]]) -> float | None:
    durations = []
    for video in videos:
        try:
            value = video.get("duration_sec")
            if value is not None:
                durations.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(durations) if durations else None


def _detector_backend(manifest: Mapping[str, Any], pipeline_summary: Mapping[str, Any]) -> str | None:
    detector_summary = pipeline_summary.get("detector_summary") if isinstance(pipeline_summary.get("detector_summary"), Mapping) else {}
    detection_config = manifest.get("detection_config") if isinstance(manifest.get("detection_config"), Mapping) else {}
    value = detector_summary.get("detector_backend") or detection_config.get("detector_backend")
    return str(value) if value else None


def _session_start_time(manifest: Mapping[str, Any]) -> str | None:
    value = manifest.get("session_start_time")
    if value:
        return str(value)
    videos = manifest.get("videos") if isinstance(manifest.get("videos"), Mapping) else {}
    starts = [
        str(row.get("start_time"))
        for row in videos.values()
        if isinstance(row, Mapping) and row.get("start_time")
    ]
    return sorted(starts)[0] if starts else None


def _summary(
    *,
    session: Path,
    target: Path,
    rows: list[dict[str, Any]],
    written: bool,
    skipped: bool,
    reason: str | None,
    seed_event_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SESSION_CONTEXT_SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "artifact": str(target),
        "row_count": len(rows),
        "written": written,
        "skipped": skipped,
        "reason": reason,
        "seed_event_id": seed_event_id,
        "non_label_context": True,
    }


def _is_seed_row(row: Mapping[str, Any]) -> bool:
    return row.get("event_id") == SEED_EVENT_ID or row.get("context_kind") == "operational_session_metadata"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return []


def _write_summary(path: Path, summary: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "SESSION_CONTEXT_FILENAME",
    "SESSION_CONTEXT_SUMMARY_FILENAME",
    "seed_session_context",
]
