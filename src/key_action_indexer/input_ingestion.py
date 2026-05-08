from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import InputEventSource, SessionManifest, VideoSource, read_jsonl, write_jsonl
from .time_alignment import parse_time
from .unified_timeline import build_timeline_event


INPUT_EVENT_SCHEMA_VERSION = "key_action_input_event.v1"
VIDEO_SOURCE_SCHEMA_VERSION = "key_action_video_source.v1"

SOURCE_OUTPUTS = {
    "user_text": "user_text_events.jsonl",
    "ai_reply": "ai_reply_events.jsonl",
    "upload": "upload_events.jsonl",
}


def write_video_source_metadata(
    manifest: SessionManifest,
    session_dir: str | Path,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    root = Path(session_dir)
    target = Path(output_path) if output_path is not None else root / "metadata" / "video_sources.jsonl"
    rows = [
        _video_source_row(
            manifest=manifest,
            session_dir=root,
            view_id=view_id,
            source=source,
            is_primary=view_id in {"third_person", "first_person"},
        )
        for view_id, source in manifest.videos.all_sources().items()
    ]
    write_jsonl(target, rows)
    return rows


def ingest_manifest_inputs(
    manifest: SessionManifest,
    session_dir: str | Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = Path(session_dir)
    metadata_dir = root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, Any]]] = {
        "user_text": [],
        "ai_reply": [],
        "upload": [],
    }
    source_summaries: dict[str, Any] = {}
    for name, source in manifest.input_sources.items():
        normalized_source = _canonical_source_type(source.source_type)
        if normalized_source not in grouped:
            continue
        rows = _read_event_rows(source.path)
        source_summaries[name] = {
            "source_type": normalized_source,
            "path": source.path,
            "exists": Path(source.path).exists(),
            "row_count": len(rows),
        }
        grouped[normalized_source].extend(
            normalize_input_events(
                rows,
                manifest=manifest,
                source=source,
                session_dir=root,
            )
        )

    synthetic_sources = _synthetic_sources(manifest) if dry_run else {}
    for source_type, rows in synthetic_sources.items():
        if grouped[source_type]:
            continue
        grouped[source_type].extend(
            normalize_input_events(
                rows,
                manifest=manifest,
                source=InputEventSource(
                    path=f"dry_run:{source_type}",
                    source_type=source_type,
                    event_type=source_type if source_type != "ai_reply" else "ai_reply",
                    modality="text" if source_type != "upload" else None,
                ),
                session_dir=root,
                synthetic=True,
            )
        )

    artifacts: dict[str, str] = {}
    counts: dict[str, int] = {}
    for source_type, filename in SOURCE_OUTPUTS.items():
        path = metadata_dir / filename
        rows = grouped[source_type]
        write_jsonl(path, rows)
        artifacts[source_type] = str(path)
        counts[source_type] = len(rows)

    summary = {
        "schema_version": "key_action_input_ingestion.v1",
        "session_id": manifest.session_id,
        "dry_run": dry_run,
        "artifacts": artifacts,
        "counts": counts,
        "sources": source_summaries,
    }
    (metadata_dir / "input_ingestion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def normalize_input_events(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest: SessionManifest,
    source: InputEventSource,
    session_dir: str | Path,
    synthetic: bool = False,
) -> list[dict[str, Any]]:
    normalized = []
    for index, row in enumerate(rows, start=1):
        normalized.append(
            normalize_input_event(
                row,
                manifest=manifest,
                source=source,
                session_dir=session_dir,
                index=index,
                synthetic=synthetic,
            )
        )
    return normalized


def normalize_input_event(
    row: Mapping[str, Any],
    *,
    manifest: SessionManifest,
    source: InputEventSource,
    session_dir: str | Path,
    index: int,
    synthetic: bool = False,
) -> dict[str, Any]:
    source_type = _canonical_source_type(source.source_type)
    prepared = dict(row)
    prepared.setdefault("source", source_type)
    prepared.setdefault("event_type", source.event_type)
    if source.modality is not None:
        prepared.setdefault("modality", source.modality)
    timeline_event = build_timeline_event(
        prepared,
        manifest=manifest,
        session_id=manifest.session_id,
        source=source_type,
        index=index,
        source_start_time=source.start_time,
        offset_sec=source.offset_sec,
        latency_sec=source.latency_sec,
    )
    event_id = str(prepared.get("event_id") or prepared.get("id") or timeline_event["timeline_event_id"])
    result = {
        "schema_version": INPUT_EVENT_SCHEMA_VERSION,
        "event_id": event_id,
        "session_id": manifest.session_id,
        "event_type": timeline_event["event_type"],
        "modality": timeline_event["modality"],
        "source": source_type,
        "source_path": source.path,
        "source_row_index": index,
        "synthetic": synthetic,
        "raw_timestamp": _raw_timestamp(prepared),
        "global_time": timeline_event["global_time"],
        "session_time_sec": timeline_event["session_time_sec"],
        "duration_sec": timeline_event["duration_sec"],
        "anchor_strategy": timeline_event["anchor_strategy"],
        "anchor_confidence": timeline_event["anchor_confidence"],
        "text": timeline_event["text"],
        "links": timeline_event["links"],
        "payload": dict(row),
    }
    if source_type == "user_text":
        result.update(_user_fields(prepared))
    elif source_type == "ai_reply":
        result.update(_ai_fields(prepared, timeline_event["text"]))
    elif source_type == "upload":
        result.update(_upload_fields(prepared, timeline_event, session_dir))
    return result


def _video_source_row(
    *,
    manifest: SessionManifest,
    session_dir: Path,
    view_id: str,
    source: VideoSource,
    is_primary: bool,
) -> dict[str, Any]:
    path = Path(source.path)
    absolute = path if path.is_absolute() else (Path.cwd() / path)
    try:
        relative_to_session = str(path.relative_to(session_dir))
    except ValueError:
        try:
            relative_to_session = str(absolute.resolve().relative_to(session_dir.resolve()))
        except Exception:
            relative_to_session = None
    start = parse_time(source.start_time)
    session_start = parse_time(manifest.session_start_time)
    return {
        "schema_version": VIDEO_SOURCE_SCHEMA_VERSION,
        "session_id": manifest.session_id,
        "view_id": view_id,
        "name": source.name,
        "role": source.role or view_id,
        "camera_id": source.camera_id,
        "path": source.path,
        "absolute_path": str(absolute),
        "relative_path": relative_to_session,
        "start_time": source.start_time,
        "session_start_delta_sec": (start - session_start).total_seconds(),
        "fps": source.fps,
        "offset_sec": source.offset_sec,
        "exists": path.exists() or absolute.exists(),
        "availability_status": "available" if path.exists() or absolute.exists() else "missing",
        "is_primary": is_primary,
        "time_basis": "global_time = start_time + offset_sec + local_time_sec",
    }


def _read_event_rows(path_value: str | Path) -> list[dict[str, Any]]:
    path = Path(path_value)
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping):
        rows = data.get("events") or data.get("rows") or data.get("items")
        if isinstance(rows, list):
            return [dict(item) for item in rows if isinstance(item, Mapping)]
        return [dict(data)]
    return []


def _synthetic_sources(manifest: SessionManifest) -> dict[str, list[dict[str, Any]]]:
    return {
        "user_text": [
            {
                "event_id": "dry_user_001",
                "session_sec": 621.2,
                "role": "user",
                "text": "dry-run operator asks whether this is the 200 uL addition step.",
            }
        ],
        "ai_reply": [
            {
                "event_id": "dry_ai_001",
                "session_sec": 622.4,
                "role": "assistant",
                "reply_type": "ai_suggestion",
                "message": "dry-run assistant suggests checking the pipette contact frame.",
            }
        ],
        "upload": [
            {
                "event_id": "dry_upload_001",
                "session_sec": 628.0,
                "upload_type": "text",
                "text": f"dry-run upload note for {manifest.session_id}",
                "file_path": "uploads/dry_run_note.txt",
            }
        ],
    }


def _canonical_source_type(value: str) -> str:
    text = str(value or "").strip().lower()
    if text in {"user", "user_events", "user_text_events", "manual_note", "manual_notes"}:
        return "user_text"
    if text in {"ai", "assistant", "ai_events", "ai_reply_events"}:
        return "ai_reply"
    if text in {"uploads", "upload_events"}:
        return "upload"
    return text


def _raw_timestamp(row: Mapping[str, Any]) -> Any:
    for key in ("timestamp", "timestamp_ms", "global_time", "time", "session_sec", "start_sec", "time_sec"):
        if key in row:
            return row[key]
    return None


def _user_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "user_id": row.get("user_id") or row.get("user") or row.get("author"),
        "role": row.get("role") or "user",
    }


def _ai_fields(row: Mapping[str, Any], text: str) -> dict[str, Any]:
    explicit = row.get("reply_type") or row.get("ai_reply_type")
    lowered = str(text or "").lower()
    if explicit:
        reply_type = str(explicit)
    elif any(token in lowered for token in ("correction", "correct", "revise")):
        reply_type = "ai_correction"
    elif any(token in lowered for token in ("conclusion", "confirmed", "result")):
        reply_type = "ai_conclusion"
    elif any(token in lowered for token in ("suggest", "recommend", "check")):
        reply_type = "ai_suggestion"
    else:
        reply_type = "ai_reply"
    return {
        "role": row.get("role") or "assistant",
        "reply_type": reply_type,
    }


def _upload_fields(row: Mapping[str, Any], timeline_event: Mapping[str, Any], session_dir: str | Path) -> dict[str, Any]:
    file_path = _first_path(row, timeline_event.get("links") or [])
    resolved = _resolve_upload_path(file_path, Path(session_dir)) if file_path else None
    sha256, hash_status = _sha256_for_upload(resolved, row, file_path)
    return {
        "upload_type": row.get("upload_type") or timeline_event.get("modality"),
        "file_path": file_path,
        "sha256": sha256,
        "hash_status": hash_status,
        "uploaded_at": row.get("uploaded_at") or row.get("timestamp") or row.get("global_time") or timeline_event.get("global_time"),
        "parsed_text": row.get("parsed_text") or row.get("text") or row.get("content") or timeline_event.get("text"),
        "thumbnail_path": row.get("thumbnail_path") or row.get("thumbnail") or row.get("image_path"),
    }


def _first_path(row: Mapping[str, Any], links: Iterable[Mapping[str, Any]]) -> str | None:
    for key in ("file_path", "path", "media_path", "image_path"):
        if row.get(key):
            return str(row[key])
    for link in links:
        if link.get("path"):
            return str(link["path"])
    return None


def _resolve_upload_path(value: str, session_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    session_candidate = session_dir / path
    if session_candidate.exists():
        return session_candidate
    return Path.cwd() / path


def _sha256_for_upload(path: Path | None, row: Mapping[str, Any], file_path: str | None) -> tuple[str, str]:
    explicit = row.get("sha256") or row.get("hash")
    if explicit:
        return str(explicit), "provided"
    if path is not None and path.exists() and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), "file"
    synthetic_payload = json.dumps({"file_path": file_path, "row": row}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(synthetic_payload.encode("utf-8")).hexdigest(), "synthetic_missing_file"
