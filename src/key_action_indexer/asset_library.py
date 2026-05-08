from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl, write_jsonl


CATALOG_FILENAME = "material_asset_catalog.jsonl"
SUMMARY_FILENAME = "material_library_summary.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".jsonl", ".csv", ".tsv"}

ASSET_FIELDS = (
    "asset_id",
    "session_id",
    "asset_type",
    "path",
    "exists",
    "size_bytes",
    "dry_run_placeholder",
    "source_type",
    "source_id",
    "segment_id",
    "micro_segment_id",
    "global_start_time",
    "global_end_time",
    "objects",
    "actions",
    "state_tags",
    "evidence_level",
    "event_type",
    "confirmation_level",
    "search_text",
    "quality",
    "evidence_refs",
    "payload_ref",
    "source_path",
    "sha256",
    "created_at",
    "schema_version",
    "privacy_level",
    "audit_trail",
)

ASSET_SCHEMA_VERSION = "key_action_asset.v1"


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def load_material_assets(path: str | Path) -> list[dict[str, Any]]:
    """Load a material asset catalog JSONL file."""

    return _read_jsonl_if_present(Path(path))


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _unique_strings(*values: Any) -> list[str]:
    items: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            for key in ("label", "name", "object_label", "object_name", "primary_object", "interaction", "action_type"):
                if value.get(key) is not None:
                    add(value.get(key))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        text = str(value).strip()
        if text and text not in items:
            items.append(text)

    for value in values:
        add(value)
    return items


def _join_search_text(*values: Any) -> str:
    parts: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            for key in (
                "index_text",
                "summary",
                "text",
                "message",
                "content",
                "action_type",
                "interaction_type",
                "interaction",
                "primary_object",
                "object_label",
                "object_name",
            ):
                if key in value:
                    add(value.get(key))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        text = str(value).strip()
        if text and text not in parts:
            parts.append(text)

    for value in values:
        add(value)
    return " ".join(parts)


def _manifest_session_id(session_dir: Path) -> str:
    manifest = session_dir / "manifest.json"
    if not manifest.exists():
        return ""
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("session_id") or "")


def _infer_session_id(session_dir: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    for row in rows:
        session_id = row.get("session_id")
        if session_id:
            return str(session_id)
        payload = row.get("payload")
        if isinstance(payload, Mapping) and payload.get("session_id"):
            return str(payload["session_id"])
    return _manifest_session_id(session_dir)


def _vector_maps(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    by_segment: dict[str, Mapping[str, Any]] = {}
    by_micro: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        micro_id = row.get("micro_segment_id")
        if micro_id:
            by_micro.setdefault(str(micro_id), row)
            continue
        segment_id = row.get("segment_id") or row.get("parent_segment_id")
        if segment_id:
            by_segment.setdefault(str(segment_id), row)
    return by_segment, by_micro


def _is_external_uri(path: str) -> bool:
    lowered = path.lower()
    return lowered.startswith(("http://", "https://", "s3://", "gs://", "az://"))


def _resolve_asset_path(session_dir: Path, path: str) -> Path | None:
    if not path or _is_external_uri(path):
        return None
    root = session_dir.resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    for index, part in enumerate(parts):
        if part == session_dir.name and index < len(parts) - 1:
            return root / Path(*parts[index + 1 :])
    if candidate.exists():
        return candidate
    return root / candidate


def _dry_run_placeholder(path: Path) -> bool:
    try:
        return path.read_bytes()[:32].upper().startswith(b"DRY RUN")
    except OSError:
        return False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_if_present(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_created_at(path: Path | None) -> str:
    if path is not None and path.exists():
        try:
            return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            pass
    return _now()


def _privacy_level(path: str, asset_type: str) -> str:
    text = f"{path} {asset_type}".lower()
    if _is_external_uri(path):
        return "external_reference"
    if any(token in text for token in ("raw", ".mp4", ".mov", ".avi", ".mkv", "video_clip")):
        return "restricted"
    if any(token in text for token in ("transcript", "dialogue", "confirmation", "audit")):
        return "confidential"
    return "internal"


def _quality_for_path(session_dir: Path, path: str, asset_type: str, source_quality: Any = None) -> tuple[bool, int, bool, dict[str, Any]]:
    if path and _is_external_uri(path):
        quality = {"status": "external_link", "warnings": [], "resolved_path": None}
        if source_quality is not None:
            quality["source_quality"] = source_quality
        return False, 0, False, quality

    resolved = _resolve_asset_path(session_dir, path)
    if resolved is None:
        status = "inline_text" if asset_type == "text_asset" else "no_path"
        quality = {"status": status, "warnings": [], "resolved_path": None}
        if source_quality is not None:
            quality["source_quality"] = source_quality
        return False, 0, False, quality

    exists = resolved.exists()
    size_bytes = int(resolved.stat().st_size) if exists else 0
    dry_run = bool(exists and _dry_run_placeholder(resolved))
    if dry_run:
        status = "dry_run_placeholder"
        warnings: list[str] = []
    elif exists:
        status = "present"
        warnings = []
    else:
        status = "missing"
        warnings = ["missing_file"]
    quality = {
        "status": status,
        "warnings": warnings,
        "resolved_path": str(resolved),
    }
    if source_quality is not None:
        quality["source_quality"] = source_quality
    return exists, size_bytes, dry_run, quality


def _asset_id(
    *,
    session_id: str,
    source_type: str,
    source_id: str,
    path: str,
    role: str,
    segment_id: str | None,
    micro_segment_id: str | None,
) -> str:
    raw = "|".join(
        [
            session_id,
            source_type,
            source_id,
            segment_id or "",
            micro_segment_id or "",
            role,
            path,
        ]
    )
    return f"asset_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _make_asset_row(
    *,
    session_dir: Path,
    session_id: str,
    asset_type: str,
    path: Any,
    source_type: str,
    source_id: Any,
    segment_id: Any = None,
    micro_segment_id: Any = None,
    global_start_time: Any = None,
    global_end_time: Any = None,
    objects: Any = None,
    actions: Any = None,
    state_tags: Any = None,
    evidence_level: Any = None,
    event_type: Any = None,
    confirmation_level: Any = None,
    search_text: Any = "",
    evidence_refs: Any = None,
    payload_ref: Mapping[str, Any] | None = None,
    source_quality: Any = None,
) -> dict[str, Any]:
    path_text = str(path or "")
    source_id_text = str(source_id or "")
    payload = dict(payload_ref or {})
    role = str(payload.get("role") or payload.get("path_field") or asset_type)
    exists, size_bytes, dry_run, quality = _quality_for_path(session_dir, path_text, asset_type, source_quality)
    resolved_text = str(quality.get("resolved_path") or "") if isinstance(quality, Mapping) else ""
    resolved_path = Path(resolved_text) if resolved_text else None
    source_path = resolved_text or path_text
    created_at = _file_created_at(resolved_path if exists else None)
    row = {
        "asset_id": _asset_id(
            session_id=session_id,
            source_type=source_type,
            source_id=source_id_text,
            path=path_text,
            role=role,
            segment_id=str(segment_id) if segment_id else None,
            micro_segment_id=str(micro_segment_id) if micro_segment_id else None,
        ),
        "session_id": session_id,
        "asset_type": asset_type,
        "path": path_text,
        "exists": exists,
        "size_bytes": size_bytes,
        "dry_run_placeholder": dry_run,
        "source_type": source_type,
        "source_id": source_id_text,
        "segment_id": str(segment_id) if segment_id else None,
        "micro_segment_id": str(micro_segment_id) if micro_segment_id else None,
        "global_start_time": str(global_start_time) if global_start_time else None,
        "global_end_time": str(global_end_time) if global_end_time else None,
        "objects": _unique_strings(objects),
        "actions": _unique_strings(actions),
        "state_tags": _unique_strings(state_tags),
        "evidence_level": str(evidence_level) if evidence_level else None,
        "event_type": str(event_type) if event_type else None,
        "confirmation_level": str(confirmation_level) if confirmation_level else None,
        "search_text": _join_search_text(search_text, objects, actions, state_tags),
        "quality": quality,
        "evidence_refs": _as_list(evidence_refs),
        "payload_ref": payload,
        "source_path": source_path,
        "sha256": _sha256_if_present(resolved_path),
        "created_at": created_at,
        "schema_version": ASSET_SCHEMA_VERSION,
        "privacy_level": _privacy_level(path_text, asset_type),
        "audit_trail": [
            {
                "event_type": "asset_cataloged",
                "created_at": created_at,
                "source_type": source_type,
                "source_id": source_id_text,
                "path_field": payload.get("path_field"),
                "source_path": source_path,
            }
        ],
    }
    return {field: row[field] for field in ASSET_FIELDS}


def _evidence_level(row: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> str | None:
    for source in (row, vector or {}):
        if source.get("evidence_level"):
            return str(source["evidence_level"])
        evidence = source.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("evidence_level"):
            return str(evidence["evidence_level"])
    return None


def _source_quality(row: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> Any:
    quality = row.get("quality")
    if quality is not None:
        return quality
    if vector and vector.get("quality") is not None:
        return vector.get("quality")
    return None


def _segment_objects(segment: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> list[str]:
    text = _as_dict(segment.get("text_description"))
    objects: list[Any] = [
        text.get("objects"),
        text.get("tools"),
        vector.get("detected_objects") if vector else None,
        vector.get("primary_object") if vector else None,
    ]
    for interaction in _as_list(segment.get("interaction_events")) + _as_list(segment.get("yolo_interactions")):
        if isinstance(interaction, Mapping):
            objects.extend([interaction.get("object_label"), interaction.get("object_name"), interaction.get("labels")])
    return _unique_strings(objects)


def _segment_actions(segment: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> list[str]:
    text = _as_dict(segment.get("text_description"))
    actions: list[Any] = [
        text.get("action_type"),
        vector.get("action_type") if vector else None,
        vector.get("visual_keywords") if vector else None,
    ]
    for interaction in _as_list(segment.get("interaction_events")) + _as_list(segment.get("yolo_interactions")):
        if isinstance(interaction, Mapping):
            actions.append(interaction.get("interaction") or interaction.get("interaction_type"))
    return _unique_strings(actions)


def _segment_search_text(segment: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> str:
    text = _as_dict(segment.get("text_description"))
    index_info = _as_dict(segment.get("index"))
    return _join_search_text(
        vector.get("index_text") if vector else None,
        index_info.get("index_text"),
        text,
        segment.get("dialogue_context"),
        segment.get("interaction_events"),
        segment.get("yolo_interactions"),
    )


def _micro_objects(micro: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> list[str]:
    interaction = _as_dict(micro.get("interaction"))
    return _unique_strings(
        interaction.get("primary_object"),
        interaction.get("secondary_objects"),
        interaction.get("detected_objects"),
        interaction.get("primary_object_family"),
        vector.get("primary_object") if vector else None,
        vector.get("detected_objects") if vector else None,
    )


def _micro_actions(micro: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> list[str]:
    interaction = _as_dict(micro.get("interaction"))
    text = _as_dict(micro.get("text_description"))
    return _unique_strings(
        text.get("action_type"),
        interaction.get("interaction_type"),
        vector.get("action_type") if vector else None,
        vector.get("interaction_type") if vector else None,
    )


def _micro_search_text(micro: Mapping[str, Any], vector: Mapping[str, Any] | None = None) -> str:
    text = _as_dict(micro.get("text_description"))
    return _join_search_text(
        vector.get("index_text") if vector else None,
        text,
        micro.get("dialogue_context"),
        micro.get("interaction"),
    )


def _timeline_asset_type(row: Mapping[str, Any], path: str, rel: str | None = None) -> str:
    modality = str(row.get("modality") or "").lower()
    event_type = str(row.get("event_type") or "").lower()
    role_text = f"{rel or ''} {event_type} {path}".lower()
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        if "keyframe" in role_text or "keyframes" in role_text or event_type in {"yolo_interaction", "micro_contact_anchor"}:
            return "keyframe"
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video_clip"
    if suffix in TEXT_EXTENSIONS:
        return "text_asset"
    if modality == "image":
        return "image"
    if modality == "text":
        return "text_asset"
    if modality == "video":
        return "video_clip"
    return "unknown"


def _timeline_text(row: Mapping[str, Any]) -> str:
    payload = _as_dict(row.get("payload"))
    return _join_search_text(
        row.get("text"),
        payload.get("text"),
        payload.get("message"),
        payload.get("content"),
        payload.get("filename"),
    )


def _nested_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(row.get("payload"))
    nested = _as_dict(payload.get("payload"))
    return nested


def _short_action_label(value: Any, max_len: int = 80) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or "\n" in text or len(text) > max_len:
        return None
    return text


def _timeline_objects(row: Mapping[str, Any]) -> list[str]:
    payload = _as_dict(row.get("payload"))
    nested = _nested_payload(row)
    interaction = _as_dict(payload.get("interaction")) or _as_dict(nested.get("interaction"))
    text_description = _as_dict(payload.get("text_description"))
    return _unique_strings(
        row.get("primary_object"),
        payload.get("primary_object"),
        nested.get("primary_object"),
        interaction.get("primary_object"),
        interaction.get("secondary_objects"),
        interaction.get("detected_objects"),
        interaction.get("primary_object_family"),
        payload.get("object_label"),
        payload.get("object_name"),
        nested.get("object_label"),
        nested.get("object_name"),
        text_description.get("objects"),
    )


def _timeline_actions(row: Mapping[str, Any]) -> list[str]:
    payload = _as_dict(row.get("payload"))
    nested = _nested_payload(row)
    interaction = _as_dict(payload.get("interaction")) or _as_dict(nested.get("interaction"))
    text_description = _as_dict(payload.get("text_description"))
    return _unique_strings(
        row.get("event_type"),
        payload.get("event_type"),
        text_description.get("action_type"),
        payload.get("action_type"),
        nested.get("action_type"),
        _short_action_label(payload.get("interaction")),
        _short_action_label(nested.get("interaction")),
        interaction.get("interaction_type"),
        _short_action_label(row.get("text")),
        _short_action_label(payload.get("text")),
    )


def _timeline_links(row: Mapping[str, Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(rel: str, value: Any) -> None:
        if not value:
            return
        path = str(value)
        key = (rel, path)
        if key in seen:
            return
        seen.add(key)
        links.append({"rel": rel, "path": path})

    raw_links = row.get("links")
    if isinstance(raw_links, list):
        for item in raw_links:
            if isinstance(item, Mapping):
                add(str(item.get("rel") or "link"), item.get("path") or item.get("url") or item.get("href"))
            else:
                add("link", item)

    payload = _as_dict(row.get("payload"))
    for source in (row, payload):
        for key in ("path", "image_path", "media_path", "file_path", "video_path", "source_image_path", "filename"):
            if source.get(key):
                add(key, source[key])
    return links


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _timeline_end_time(row: Mapping[str, Any]) -> str | None:
    if row.get("global_end_time"):
        return str(row["global_end_time"])
    start = _parse_iso_datetime(row.get("global_time") or row.get("global_start_time"))
    if start is None:
        return str(row.get("global_time") or row.get("global_start_time") or "") or None
    try:
        duration = float(row.get("duration_sec") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        return start.isoformat()
    return (start + timedelta(seconds=duration)).isoformat()


def _add_row(rows: list[dict[str, Any]], seen: set[tuple[Any, ...]], row: dict[str, Any]) -> None:
    key = (
        row.get("session_id"),
        row.get("segment_id"),
        row.get("micro_segment_id"),
        row.get("source_id"),
        row.get("path"),
        row.get("asset_type"),
    )
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def _build_segment_assets(
    *,
    session_dir: Path,
    session_id: str,
    segments: list[dict[str, Any]],
    vector_by_segment: Mapping[str, Mapping[str, Any]],
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
) -> None:
    for row_index, segment in enumerate(segments, start=1):
        segment_id = str(segment.get("segment_id") or f"segment_{row_index:06d}")
        vector = vector_by_segment.get(segment_id)
        objects = _segment_objects(segment, vector)
        actions = _segment_actions(segment, vector)
        evidence_level = _evidence_level(segment, vector)
        search_text = _segment_search_text(segment, vector)

        for view_key in ("third_person", "first_person"):
            view = _as_dict(segment.get(view_key))
            clip_path = view.get("clip_path")
            if not clip_path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id,
                    asset_type="video_clip",
                    path=clip_path,
                    source_type="segment_clip",
                    source_id=segment_id,
                    segment_id=segment_id,
                    global_start_time=segment.get("global_start_time"),
                    global_end_time=segment.get("global_end_time"),
                    objects=objects,
                    actions=actions,
                    state_tags=["segment_clip", view_key, evidence_level],
                    evidence_level=evidence_level,
                    search_text=search_text,
                    payload_ref={
                        "metadata_file": "metadata/key_action_segments.jsonl",
                        "row_index": row_index,
                        "path_field": f"{view_key}.clip_path",
                        "role": view_key,
                    },
                ),
            )

        for keyframe_index, keyframe in enumerate(_as_list(segment.get("interaction_keyframes")), start=1):
            if not isinstance(keyframe, Mapping):
                continue
            path = keyframe.get("path") or keyframe.get("keyframe_path")
            if not path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id,
                    asset_type="keyframe",
                    path=path,
                    source_type="interaction_keyframe",
                    source_id=keyframe.get("event_id") or segment_id,
                    segment_id=segment_id,
                    global_start_time=keyframe.get("global_time") or segment.get("global_start_time"),
                    global_end_time=keyframe.get("global_time") or segment.get("global_start_time"),
                    objects=_unique_strings(objects, keyframe.get("labels")),
                    actions=_unique_strings(actions, keyframe.get("interaction")),
                    state_tags=["interaction_keyframe", keyframe.get("view"), evidence_level],
                    evidence_level=evidence_level,
                    search_text=_join_search_text(search_text, keyframe),
                    payload_ref={
                        "metadata_file": "metadata/key_action_segments.jsonl",
                        "row_index": row_index,
                        "path_field": f"interaction_keyframes[{keyframe_index}].path",
                        "role": "interaction_keyframe",
                    },
                ),
            )

        for event_index, event in enumerate(_as_list(segment.get("interaction_events")), start=1):
            if not isinstance(event, Mapping):
                continue
            path = event.get("keyframe_path")
            if not path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id,
                    asset_type="keyframe",
                    path=path,
                    source_type="interaction_keyframe",
                    source_id=event.get("event_id") or segment_id,
                    segment_id=segment_id,
                    global_start_time=event.get("global_time") or segment.get("global_start_time"),
                    global_end_time=event.get("global_time") or segment.get("global_start_time"),
                    objects=_unique_strings(objects, event.get("object_label"), event.get("object_name")),
                    actions=_unique_strings(actions, event.get("interaction")),
                    state_tags=["interaction_keyframe", event.get("view"), evidence_level],
                    evidence_level=evidence_level,
                    search_text=_join_search_text(search_text, event),
                    payload_ref={
                        "metadata_file": "metadata/key_action_segments.jsonl",
                        "row_index": row_index,
                        "path_field": f"interaction_events[{event_index}].keyframe_path",
                        "role": "interaction_keyframe",
                    },
                ),
            )


def _build_micro_assets(
    *,
    session_dir: Path,
    session_id: str,
    micros: list[dict[str, Any]],
    vector_by_micro: Mapping[str, Mapping[str, Any]],
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
) -> None:
    phase_tags = {
        "contact_frame": "contact_started",
        "peak_frame": "peak_interaction",
        "release_frame": "contact_released",
    }
    for row_index, micro in enumerate(micros, start=1):
        micro_id = str(micro.get("micro_segment_id") or f"micro_{row_index:06d}")
        segment_id = micro.get("parent_segment_id") or micro.get("segment_id")
        vector = vector_by_micro.get(micro_id)
        objects = _micro_objects(micro, vector)
        actions = _micro_actions(micro, vector)
        evidence_level = _evidence_level(micro, vector)
        search_text = _micro_search_text(micro, vector)
        source_quality = _source_quality(micro, vector)
        interaction = _as_dict(micro.get("interaction"))
        interaction_type = interaction.get("interaction_type")

        for view_key in ("third_person", "first_person"):
            view = _as_dict(micro.get(view_key))
            clip_path = view.get("clip_path")
            if not clip_path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id,
                    asset_type="video_clip",
                    path=clip_path,
                    source_type="micro_clip",
                    source_id=micro_id,
                    segment_id=segment_id,
                    micro_segment_id=micro_id,
                    global_start_time=micro.get("global_start_time"),
                    global_end_time=micro.get("global_end_time"),
                    objects=objects,
                    actions=actions,
                    state_tags=["micro_clip", view_key, interaction_type, evidence_level],
                    evidence_level=evidence_level,
                    search_text=search_text,
                    payload_ref={
                        "metadata_file": "metadata/micro_segments.jsonl",
                        "row_index": row_index,
                        "path_field": f"{view_key}.clip_path",
                        "role": view_key,
                    },
                    source_quality=source_quality,
                ),
            )

        keyframes = _as_dict(micro.get("keyframes"))
        for path_field, state_tag in phase_tags.items():
            keyframe_path = keyframes.get(path_field)
            if not keyframe_path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id,
                    asset_type="keyframe",
                    path=keyframe_path,
                    source_type="micro_keyframe",
                    source_id=micro_id,
                    segment_id=segment_id,
                    micro_segment_id=micro_id,
                    global_start_time=micro.get("global_start_time"),
                    global_end_time=micro.get("global_end_time"),
                    objects=objects,
                    actions=actions,
                    state_tags=["micro_keyframe", state_tag, interaction_type, evidence_level],
                    evidence_level=evidence_level,
                    search_text=_join_search_text(search_text, state_tag),
                    payload_ref={
                        "metadata_file": "metadata/micro_segments.jsonl",
                        "row_index": row_index,
                        "path_field": f"keyframes.{path_field}",
                        "role": path_field,
                    },
                    source_quality=source_quality,
                ),
            )


def _build_timeline_assets(
    *,
    session_dir: Path,
    session_id: str,
    timeline_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
) -> None:
    for row_index, timeline in enumerate(timeline_rows, start=1):
        event_type = str(timeline.get("event_type") or "")
        modality = str(timeline.get("modality") or "")
        source_id = str(timeline.get("timeline_event_id") or timeline.get("event_id") or timeline.get("id") or f"timeline_{row_index:06d}")
        text = _timeline_text(timeline)
        start_time = timeline.get("global_time") or timeline.get("global_start_time")
        end_time = _timeline_end_time(timeline)
        links = _timeline_links(timeline)
        objects = _timeline_objects(timeline)
        actions = _timeline_actions(timeline)
        state_tags = _unique_strings(event_type, modality, "upload" if event_type == "upload" else None)

        for link_index, link in enumerate(links, start=1):
            path = link.get("path") or ""
            asset_type = _timeline_asset_type(timeline, path, link.get("rel"))
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id or str(timeline.get("session_id") or ""),
                    asset_type=asset_type,
                    path=path,
                    source_type="timeline_link",
                    source_id=source_id,
                    segment_id=timeline.get("segment_id"),
                    micro_segment_id=timeline.get("micro_segment_id"),
                    global_start_time=start_time,
                    global_end_time=end_time,
                    objects=objects,
                    actions=actions,
                    state_tags=_unique_strings(state_tags, link.get("rel")),
                    evidence_level=_evidence_level(timeline),
                    search_text=_join_search_text(text, path, link.get("rel")),
                    payload_ref={
                        "metadata_file": "metadata/unified_multimodal_timeline.jsonl",
                        "row_index": row_index,
                        "path_field": f"links[{link_index}].path",
                        "role": link.get("rel") or "timeline_link",
                    },
                ),
            )

        if not links and text and (event_type == "upload" or modality == "text"):
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id or str(timeline.get("session_id") or ""),
                    asset_type="text_asset",
                    path="",
                    source_type="timeline_text",
                    source_id=source_id,
                    segment_id=timeline.get("segment_id"),
                    micro_segment_id=timeline.get("micro_segment_id"),
                    global_start_time=start_time,
                    global_end_time=end_time,
                    objects=objects,
                    actions=actions,
                    state_tags=state_tags,
                    evidence_level=_evidence_level(timeline),
                    search_text=text,
                    payload_ref={
                        "metadata_file": "metadata/unified_multimodal_timeline.jsonl",
                        "row_index": row_index,
                        "path_field": "text",
                        "role": "timeline_text",
                    },
                ),
            )


def _vector_asset_objects(vector: Mapping[str, Any]) -> list[str]:
    return _unique_strings(vector.get("primary_object"), vector.get("detected_objects"), vector.get("visual_keywords"))


def _vector_asset_actions(vector: Mapping[str, Any]) -> list[str]:
    return _unique_strings(vector.get("action_type"), vector.get("interaction_type"), vector.get("visual_keywords"))


def _build_vector_fallback_assets(
    *,
    session_dir: Path,
    session_id: str,
    vectors: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
) -> None:
    for row_index, vector in enumerate(vectors, start=1):
        source_id = str(vector.get("micro_segment_id") or vector.get("segment_id") or vector.get("embedding_id") or f"vector_{row_index:06d}")
        segment_id = vector.get("segment_id") or vector.get("parent_segment_id")
        micro_id = vector.get("micro_segment_id")
        objects = _vector_asset_objects(vector)
        actions = _vector_asset_actions(vector)
        evidence_level = _evidence_level(vector)
        search_text = str(vector.get("index_text") or "")
        source_quality = _source_quality(vector)

        for path_field in ("third_person_clip", "first_person_clip"):
            path = vector.get(path_field)
            if not path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id or str(vector.get("session_id") or ""),
                    asset_type="video_clip",
                    path=path,
                    source_type="vector_metadata",
                    source_id=source_id,
                    segment_id=segment_id,
                    micro_segment_id=micro_id,
                    global_start_time=vector.get("global_start_time"),
                    global_end_time=vector.get("global_end_time"),
                    objects=objects,
                    actions=actions,
                    state_tags=["vector_metadata", path_field, vector.get("index_level"), evidence_level],
                    evidence_level=evidence_level,
                    search_text=search_text,
                    payload_ref={
                        "metadata_file": "metadata/vector_metadata.jsonl",
                        "row_index": row_index,
                        "path_field": path_field,
                        "role": path_field,
                    },
                    source_quality=source_quality,
                ),
            )

        keyframe_values = list(_as_list(vector.get("keyframes")))
        keyframe_values.extend(_as_list(vector.get("interaction_keyframes")))
        for keyframe_index, item in enumerate(keyframe_values, start=1):
            path = item.get("path") if isinstance(item, Mapping) else item
            if not path:
                continue
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id or str(vector.get("session_id") or ""),
                    asset_type="keyframe",
                    path=path,
                    source_type="vector_metadata",
                    source_id=source_id,
                    segment_id=segment_id,
                    micro_segment_id=micro_id,
                    global_start_time=vector.get("global_start_time"),
                    global_end_time=vector.get("global_end_time"),
                    objects=objects,
                    actions=actions,
                    state_tags=["vector_metadata", "keyframe", vector.get("index_level"), evidence_level],
                    evidence_level=evidence_level,
                    search_text=search_text,
                    payload_ref={
                        "metadata_file": "metadata/vector_metadata.jsonl",
                        "row_index": row_index,
                        "path_field": f"keyframes[{keyframe_index}]",
                        "role": f"keyframe_{keyframe_index}",
                    },
                    source_quality=source_quality,
                ),
            )


def _build_advanced_evidence_assets(
    *,
    session_dir: Path,
    session_id: str,
    evidence_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
) -> None:
    for row_index, evidence in enumerate(evidence_rows, start=1):
        if not isinstance(evidence, Mapping):
            continue
        evidence_id = str(evidence.get("evidence_id") or f"advanced_evidence_{row_index:06d}")
        object_label = evidence.get("object_label")
        action_type = evidence.get("action_type")
        evidence_type = evidence.get("evidence_type")
        confirmation_level = evidence.get("confirmation_level")
        visual_level = evidence.get("visual_confirmation_level")
        refs = evidence.get("evidence_refs") or evidence.get("asset_refs") or []
        for ref_index, ref in enumerate(_as_list(refs), start=1):
            if not isinstance(ref, Mapping):
                continue
            path = ref.get("path") or ref.get("video_path") or ref.get("image_path")
            if not path:
                continue
            asset_type = str(ref.get("asset_type") or "")
            if asset_type in {"frame", "video_frame"}:
                asset_type = "keyframe"
            if not asset_type or asset_type in {"yolo_frame_row", "annotation_dataset"}:
                asset_type = _timeline_asset_type(evidence, str(path), str(ref.get("rel") or ""))
            _add_row(
                rows,
                seen,
                _make_asset_row(
                    session_dir=session_dir,
                    session_id=session_id or str(evidence.get("session_id") or ""),
                    asset_type=asset_type,
                    path=path,
                    source_type="advanced_vision_evidence",
                    source_id=evidence_id,
                    segment_id=evidence.get("segment_id"),
                    micro_segment_id=evidence.get("micro_segment_id"),
                    global_start_time=evidence.get("global_start_time"),
                    global_end_time=evidence.get("global_end_time"),
                    objects=_unique_strings(object_label, _as_dict(evidence.get("metrics")).get("object_label")),
                    actions=_unique_strings(action_type, evidence_type, visual_level),
                    state_tags=_unique_strings(evidence_type, confirmation_level, visual_level),
                    evidence_level=confirmation_level or visual_level,
                    event_type=evidence_type,
                    confirmation_level=confirmation_level,
                    search_text=_join_search_text(
                        evidence.get("evidence_reasons"),
                        evidence.get("confidence_reasons"),
                        evidence.get("limitations"),
                        evidence.get("metrics"),
                    ),
                    evidence_refs=refs,
                    payload_ref={
                        "metadata_file": "metadata/advanced_vision_evidence.jsonl",
                        "row_index": row_index,
                        "path_field": f"evidence_refs[{ref_index}].path",
                        "role": evidence_type or "advanced_vision_evidence",
                    },
                    source_quality=ref.get("quality"),
                ),
            )


def summarize_material_assets(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)

    def count_values(field: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in materialized:
            for value in _unique_strings(row.get(field)):
                counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    asset_type_counts: dict[str, int] = {}
    for row in materialized:
        asset_type = str(row.get("asset_type") or "unknown")
        asset_type_counts[asset_type] = asset_type_counts.get(asset_type, 0) + 1
    asset_type_counts = dict(sorted(asset_type_counts.items()))

    missing_assets = [
        {
            "asset_id": row.get("asset_id"),
            "path": row.get("path"),
            "source_type": row.get("source_type"),
            "source_id": row.get("source_id"),
        }
        for row in materialized
        if _as_dict(row.get("quality")).get("status") == "missing"
    ]
    dry_run_assets = [
        {
            "asset_id": row.get("asset_id"),
            "path": row.get("path"),
            "source_type": row.get("source_type"),
            "source_id": row.get("source_id"),
        }
        for row in materialized
        if bool(row.get("dry_run_placeholder"))
    ]
    return {
        "asset_count": len(materialized),
        "asset_type_counts": asset_type_counts,
        "object_counts": count_values("objects"),
        "action_counts": count_values("actions"),
        "state_tag_counts": count_values("state_tags"),
        "missing_count": len(missing_assets),
        "dry_run_count": len(dry_run_assets),
        "missing_assets": missing_assets,
        "dry_run_assets": dry_run_assets,
    }


def build_material_asset_catalog(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build material asset rows from session metadata and write catalog artifacts."""

    root = Path(session_dir)
    metadata_dir = root / "metadata"
    segment_rows = _read_jsonl_if_present(metadata_dir / "key_action_segments.jsonl")
    micro_rows = _read_jsonl_if_present(metadata_dir / "micro_segments.jsonl")
    vector_rows = _read_jsonl_if_present(metadata_dir / "vector_metadata.jsonl")
    timeline_rows = _read_jsonl_if_present(metadata_dir / "unified_multimodal_timeline.jsonl")
    advanced_evidence_rows = _read_jsonl_if_present(metadata_dir / "advanced_vision_evidence.jsonl")

    session_id = _infer_session_id(root, [*segment_rows, *micro_rows, *vector_rows, *timeline_rows, *advanced_evidence_rows])
    vector_by_segment, vector_by_micro = _vector_maps(vector_rows)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    _build_segment_assets(
        session_dir=root,
        session_id=session_id,
        segments=segment_rows,
        vector_by_segment=vector_by_segment,
        rows=rows,
        seen=seen,
    )
    _build_micro_assets(
        session_dir=root,
        session_id=session_id,
        micros=micro_rows,
        vector_by_micro=vector_by_micro,
        rows=rows,
        seen=seen,
    )
    _build_timeline_assets(
        session_dir=root,
        session_id=session_id,
        timeline_rows=timeline_rows,
        rows=rows,
        seen=seen,
    )
    _build_vector_fallback_assets(
        session_dir=root,
        session_id=session_id,
        vectors=vector_rows,
        rows=rows,
        seen=seen,
    )
    _build_advanced_evidence_assets(
        session_dir=root,
        session_id=session_id,
        evidence_rows=advanced_evidence_rows,
        rows=rows,
        seen=seen,
    )

    catalog_path = Path(output_path) if output_path is not None else metadata_dir / CATALOG_FILENAME
    library_summary_path = Path(summary_path) if summary_path is not None else metadata_dir / SUMMARY_FILENAME
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    library_summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(catalog_path, rows)
    summary = summarize_material_assets(rows)
    library_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "session_dir": str(root),
        "session_id": session_id,
        "catalog_path": str(catalog_path),
        "summary_path": str(library_summary_path),
        "asset_count": len(rows),
        "summary": summary,
    }

