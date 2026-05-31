from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl, write_jsonl
from .time_alignment import parse_time


MICRO_PHASES = (
    ("contact_started", "contact_start_sec", "contact_frame"),
    ("peak_interaction", "peak_interaction_sec", "peak_frame"),
    ("contact_released", "contact_end_sec", "release_frame"),
)

LOW_EVIDENCE_LEVELS = {"weak_visual_evidence", "insufficient_evidence", "low"}
TIMED_STATE_ORDER = {
    "contact_started": 10,
    "object_contact": 20,
    "peak_interaction": 30,
    "contact_released": 40,
    "dialogue_context_available": 50,
}
STABLE_FIELDS = (
    "state_change_id",
    "session_id",
    "state_type",
    "global_time",
    "session_time_sec",
    "segment_id",
    "micro_segment_id",
    "primary_object",
    "interaction_type",
    "action_type",
    "evidence_level",
    "state_tags",
    "asset_refs",
    "text",
    "payload",
)


def build_state_change_index(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata_dir = session / "metadata"
    micro_path = metadata_dir / "micro_segments.jsonl"
    segment_path = metadata_dir / "key_action_segments.jsonl"
    timeline_path = metadata_dir / "unified_multimodal_timeline.jsonl"
    asset_catalog_path = metadata_dir / "material_asset_catalog.jsonl"
    target_path = Path(output_path) if output_path is not None else metadata_dir / "state_change_index.jsonl"

    micro_rows = _read_jsonl_if_exists(micro_path)
    segment_rows = _read_jsonl_if_exists(segment_path)
    timeline_rows = _read_jsonl_if_exists(timeline_path)
    asset_lookup = _asset_lookup(session, _read_jsonl_if_exists(asset_catalog_path))
    segment_by_id = {str(row.get("segment_id")): row for row in segment_rows if row.get("segment_id")}
    session_start = _infer_session_start(micro_rows, segment_rows, timeline_rows)

    events: list[dict[str, Any]] = []
    for micro in micro_rows:
        if not isinstance(micro, Mapping):
            continue
        events.extend(_micro_phase_events(micro, segment_by_id, session_start))
        dialogue_event = _micro_dialogue_event(micro, segment_by_id, session_start)
        if dialogue_event is not None:
            events.append(dialogue_event)

    for timeline_index, row in enumerate(timeline_rows, start=1):
        if _is_yolo_interaction_event(row):
            events.append(_timeline_object_contact_event(row, timeline_index))

    events = [_enrich_asset_refs(event, asset_lookup) for event in events]
    events = sorted(events, key=_state_sort_key)
    write_jsonl(target_path, [_stable_row(event) for event in events])

    counts = Counter(str(event["state_type"]) for event in events)
    session_id = _first_text(
        [
            *(row.get("session_id") for row in micro_rows if isinstance(row, Mapping)),
            *(row.get("session_id") for row in segment_rows if isinstance(row, Mapping)),
            *(row.get("session_id") for row in timeline_rows if isinstance(row, Mapping)),
        ]
    )
    return {
        "session_id": session_id,
        "state_change_count": len(events),
        "state_type_counts": dict(sorted(counts.items())),
        "input_counts": {
            "micro_segments": len(micro_rows),
            "key_action_segments": len(segment_rows),
            "timeline_events": len(timeline_rows),
            "material_assets": len(asset_lookup.get("__rows__", [])),
        },
        "asset_ref_count": sum(len(event.get("asset_refs") or []) for event in events),
        "asset_id_ref_count": sum(
            1
            for event in events
            for ref in (event.get("asset_refs") or [])
            if isinstance(ref, Mapping) and ref.get("asset_id")
        ),
        "state_change_index": str(target_path),
    }


def load_state_changes(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def _stable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in STABLE_FIELDS}


def _micro_phase_events(
    micro: Mapping[str, Any],
    segment_by_id: Mapping[str, Mapping[str, Any]],
    session_start: datetime | None,
) -> list[dict[str, Any]]:
    interaction = _as_mapping(micro.get("interaction"))
    events: list[dict[str, Any]] = []
    for state_type, time_key, keyframe_key in MICRO_PHASES:
        session_time = _phase_session_time(micro, interaction, state_type, time_key)
        global_time, timing_source = _phase_global_time(micro, session_time, state_type, session_start)
        events.append(
            _build_micro_state(
                micro,
                segment_by_id,
                state_type=state_type,
                session_time_sec=session_time,
                global_time=global_time,
                timing_source=timing_source,
                extra_tags=_missing_time_tags(session_time, global_time),
                asset_refs=_micro_asset_refs(micro, preferred_keyframe=keyframe_key),
                payload={
                    "source": "micro_segment",
                    "phase_time_key": time_key,
                    "phase_time_sec": session_time,
                    "timing_source": timing_source,
                    "micro_segment": dict(micro),
                },
            )
        )
    return events


def _micro_dialogue_event(
    micro: Mapping[str, Any],
    segment_by_id: Mapping[str, Mapping[str, Any]],
    session_start: datetime | None,
) -> dict[str, Any] | None:
    dialogue = _dialogue_items(micro)
    if not dialogue:
        return None
    session_time = _as_float(micro.get("start_sec"))
    global_time, timing_source = _phase_global_time(micro, session_time, "contact_started", session_start)
    return _build_micro_state(
        micro,
        segment_by_id,
        state_type="dialogue_context_available",
        session_time_sec=session_time,
        global_time=global_time,
        timing_source=timing_source,
        extra_tags=["transcript_available", *_missing_time_tags(session_time, global_time)],
        asset_refs=_micro_asset_refs(micro),
        text=_dialogue_text(dialogue),
        payload={
            "source": "micro_segment_dialogue",
            "timing_source": timing_source,
            "dialogue_context": dialogue,
            "micro_segment": dict(micro),
        },
    )


def _build_micro_state(
    micro: Mapping[str, Any],
    segment_by_id: Mapping[str, Mapping[str, Any]],
    *,
    state_type: str,
    session_time_sec: float | None,
    global_time: str | None,
    timing_source: str,
    extra_tags: Iterable[str] = (),
    asset_refs: list[dict[str, Any]] | None = None,
    text: str | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    interaction = _as_mapping(micro.get("interaction"))
    text_description = _as_mapping(micro.get("text_description"))
    segment_id = _segment_id(micro)
    parent_segment = segment_by_id.get(str(segment_id)) if segment_id is not None else None
    evidence_level, limitations = _evidence_for_micro(micro)
    primary_object = _first_text(
        [
            interaction.get("primary_object"),
            micro.get("primary_object"),
            _as_mapping(parent_segment).get("primary_object"),
        ]
    )
    interaction_type = _first_text([interaction.get("interaction_type"), micro.get("interaction_type")])
    action_type = _first_text([text_description.get("action_type"), micro.get("action_type")])
    micro_id = _first_text([micro.get("micro_segment_id"), micro.get("display_id")])
    tags = _state_tags(
        state_type=state_type,
        primary_object=primary_object,
        interaction_type=interaction_type,
        action_type=action_type,
        evidence_level=evidence_level,
        limitations=limitations,
        extra=[timing_source, *extra_tags],
    )
    return {
        "state_change_id": _state_id(micro.get("session_id"), micro_id, state_type),
        "session_id": _first_text([micro.get("session_id"), _as_mapping(parent_segment).get("session_id")]),
        "state_type": state_type,
        "global_time": global_time,
        "session_time_sec": session_time_sec,
        "segment_id": segment_id,
        "micro_segment_id": micro_id,
        "primary_object": primary_object,
        "interaction_type": interaction_type,
        "action_type": action_type,
        "evidence_level": evidence_level,
        "state_tags": tags,
        "asset_refs": asset_refs or [],
        "text": text if text is not None else _micro_text(micro, state_type, primary_object, interaction_type),
        "payload": payload or {"source": "micro_segment", "micro_segment": dict(micro)},
    }


def _timeline_object_contact_event(row: Mapping[str, Any], timeline_index: int) -> dict[str, Any]:
    payload = _as_mapping(row.get("payload"))
    source_payload = payload if payload else row
    session_id = _first_text([row.get("session_id"), payload.get("session_id")])
    segment_id = _first_text([row.get("segment_id"), payload.get("segment_id")])
    primary_object = _first_text(
        [
            row.get("primary_object"),
            payload.get("primary_object"),
            payload.get("object_label"),
            payload.get("object_name"),
            payload.get("target_label"),
            payload.get("object"),
        ]
    )
    interaction_type = _first_text(
        [
            row.get("interaction_type"),
            payload.get("interaction_type"),
            payload.get("interaction"),
            f"hand_{primary_object}_contact" if primary_object else None,
        ]
    )
    action_type = _first_text([row.get("action_type"), payload.get("action_type"), "yolo_interaction"])
    evidence_level, limitations = _evidence_for_timeline_yolo(row, payload)
    global_time = _first_text([row.get("global_time"), payload.get("global_time")])
    session_time = _as_float(row.get("session_time_sec"))
    if session_time is None:
        session_time = _as_float(payload.get("session_time_sec"))
    timeline_event_id = _first_text([row.get("timeline_event_id"), payload.get("event_id"), f"timeline_{timeline_index:06d}"])
    asset_refs = _timeline_asset_refs(row, source_payload)
    return {
        "state_change_id": _state_id(session_id, timeline_event_id, "object_contact"),
        "session_id": session_id,
        "state_type": "object_contact",
        "global_time": global_time,
        "session_time_sec": session_time,
        "segment_id": segment_id,
        "micro_segment_id": _first_text([row.get("micro_segment_id"), payload.get("micro_segment_id")]),
        "primary_object": primary_object,
        "interaction_type": interaction_type,
        "action_type": action_type,
        "evidence_level": evidence_level,
        "state_tags": _state_tags(
            state_type="object_contact",
            primary_object=primary_object,
            interaction_type=interaction_type,
            action_type=action_type,
            evidence_level=evidence_level,
            limitations=limitations,
            extra=["source:yolo_interaction", *_missing_time_tags(session_time, global_time)],
        ),
        "asset_refs": asset_refs,
        "text": _first_text([row.get("text"), payload.get("text"), payload.get("interaction"), primary_object]),
        "payload": {
            "source": "unified_multimodal_timeline",
            "timeline_event": dict(row),
        },
    }


def _is_yolo_interaction_event(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    event_type = str(row.get("event_type") or "").casefold()
    source = str(row.get("source") or "").casefold()
    payload = _as_mapping(row.get("payload"))
    payload_type = str(payload.get("event_type") or payload.get("source") or "").casefold()
    return "yolo_interaction" in {event_type, source, payload_type}


def _infer_session_start(
    micro_rows: Iterable[Mapping[str, Any]],
    segment_rows: Iterable[Mapping[str, Any]],
    timeline_rows: Iterable[Mapping[str, Any]],
) -> datetime | None:
    for rows in (micro_rows, segment_rows, timeline_rows):
        for row in rows:
            for key in ("session_start_time", "session_start", "global_origin_time"):
                parsed = _parse_datetime(row.get(key))
                if parsed is not None:
                    return parsed

    for row in micro_rows:
        start = _parse_datetime(row.get("global_start_time") or row.get("global_start"))
        start_sec = _as_float(row.get("start_sec"))
        if start is not None and start_sec is not None:
            return start - timedelta(seconds=start_sec)
        end = _parse_datetime(row.get("global_end_time") or row.get("global_end"))
        end_sec = _as_float(row.get("end_sec"))
        if end is not None and end_sec is not None:
            return end - timedelta(seconds=end_sec)

    for row in timeline_rows:
        global_time = _parse_datetime(row.get("global_time"))
        session_time = _as_float(row.get("session_time_sec"))
        if global_time is not None and session_time is not None:
            return global_time - timedelta(seconds=session_time)

    for row in segment_rows:
        global_start = _parse_datetime(row.get("global_start_time") or row.get("global_start"))
        start_sec = _as_float(row.get("start_sec"))
        if global_start is not None and start_sec is not None:
            return global_start - timedelta(seconds=start_sec)
    return None


def _phase_session_time(
    micro: Mapping[str, Any],
    interaction: Mapping[str, Any],
    state_type: str,
    time_key: str,
) -> float | None:
    explicit = _as_float(interaction.get(time_key))
    if explicit is not None:
        return explicit
    start = _as_float(micro.get("start_sec"))
    end = _as_float(micro.get("end_sec"))
    if state_type == "contact_started":
        return start
    if state_type == "contact_released":
        return end
    if start is not None and end is not None:
        return (start + end) / 2.0
    return None


def _phase_global_time(
    micro: Mapping[str, Any],
    session_time: float | None,
    state_type: str,
    session_start: datetime | None,
) -> tuple[str | None, str]:
    global_start = _parse_datetime(micro.get("global_start_time") or micro.get("global_start"))
    global_end = _parse_datetime(micro.get("global_end_time") or micro.get("global_end"))
    start_sec = _as_float(micro.get("start_sec"))
    end_sec = _as_float(micro.get("end_sec"))

    if session_time is not None and global_start is not None and start_sec is not None:
        return (global_start + timedelta(seconds=session_time - start_sec)).isoformat(), "time_from_global_start"
    if session_time is not None and global_end is not None and end_sec is not None:
        return (global_end + timedelta(seconds=session_time - end_sec)).isoformat(), "time_from_global_end"
    if session_time is not None and session_start is not None:
        return (session_start + timedelta(seconds=session_time)).isoformat(), "time_from_session_start"

    if state_type == "contact_started" and global_start is not None:
        return global_start.isoformat(), "time_from_global_start"
    if state_type == "contact_released" and global_end is not None:
        return global_end.isoformat(), "time_from_global_end"
    if state_type == "peak_interaction" and global_start is not None and global_end is not None:
        return (global_start + (global_end - global_start) / 2).isoformat(), "time_from_global_midpoint"
    return None, "missing_time"


def _evidence_for_micro(micro: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    evidence = _as_mapping(micro.get("evidence"))
    level = _first_text([micro.get("evidence_level"), evidence.get("evidence_level")])
    limitations = _string_list(micro.get("limitations")) + _string_list(evidence.get("limitations"))
    return level, _dedupe(limitations)


def _evidence_for_timeline_yolo(row: Mapping[str, Any], payload: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    evidence = _as_mapping(row.get("evidence")) or _as_mapping(payload.get("evidence"))
    level = _first_text([row.get("evidence_level"), payload.get("evidence_level"), evidence.get("evidence_level")])
    limitations = _dedupe(
        _string_list(row.get("limitations"))
        + _string_list(payload.get("limitations"))
        + _string_list(evidence.get("limitations"))
    )
    if level:
        return level, limitations
    confidence = _as_float(row.get("confidence"))
    if confidence is None:
        confidence = _as_float(payload.get("confidence"))
    if confidence is not None and confidence < 0.5:
        return "weak_visual_evidence", limitations
    return "visual_confirmed", limitations


def _micro_asset_refs(micro: Mapping[str, Any], preferred_keyframe: str | None = None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    keyframes = _as_mapping(micro.get("keyframes"))
    if preferred_keyframe:
        _append_ref(refs, "keyframe", preferred_keyframe, keyframes.get(preferred_keyframe))
    for key in ("contact_frame", "peak_frame", "release_frame"):
        if key != preferred_keyframe:
            _append_ref(refs, "keyframe", key, keyframes.get(key))
    for view_key in ("first_person", "third_person"):
        view = _as_mapping(micro.get(view_key))
        _append_ref(refs, "clip", f"{view_key}.clip_path", view.get("clip_path"))
        _append_ref(refs, "clip", f"{view_key}.annotated_clip_path", view.get("annotated_clip_path"))
    return _dedupe_refs(refs)


def _timeline_asset_refs(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    links = row.get("links")
    if isinstance(links, list):
        for link in links:
            if isinstance(link, Mapping):
                _append_ref(refs, "timeline_link", str(link.get("rel") or "link"), link.get("path"))
    for key in ("keyframe_path", "source_image_path", "path", "image_path", "media_path", "file_path"):
        _append_ref(refs, "timeline_payload", key, payload.get(key))
    return _dedupe_refs(refs)


def _append_ref(refs: list[dict[str, Any]], asset_type: str, rel: str, path: Any) -> None:
    if path is None or path == "":
        return
    refs.append({"asset_type": asset_type, "rel": rel, "path": str(path)})


def _dedupe_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for ref in refs:
        key = (str(ref.get("asset_type")), str(ref.get("rel")), str(ref.get("path")))
        if key in seen:
            continue
        seen.add(key)
        output.append(ref)
    return output


def _asset_lookup(session_dir: Path, catalog_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup: dict[str, Any] = {"__rows__": catalog_rows}
    for row in catalog_rows:
        if not isinstance(row, Mapping):
            continue
        asset_id = row.get("asset_id")
        path = row.get("path")
        if not asset_id or not path:
            continue
        for key in _path_lookup_keys(session_dir, path):
            lookup.setdefault(key, row)
    return lookup


def _path_lookup_keys(session_dir: Path, path: Any) -> list[str]:
    text = str(path or "").strip()
    if not text:
        return []
    keys = {text, text.replace("/", "\\")}
    try:
        candidate = Path(text)
        if candidate.is_absolute():
            keys.add(str(candidate.resolve()))
        else:
            parts = candidate.parts
            for index, part in enumerate(parts):
                if part == session_dir.name and index < len(parts) - 1:
                    keys.add(str(Path(*parts[index + 1 :])))
                    keys.add(str((session_dir / Path(*parts[index + 1 :])).resolve()))
            keys.add(str(candidate))
            keys.add(str((session_dir / candidate).resolve()))
    except (OSError, ValueError):
        pass
    return [key for key in keys if key]


def _enrich_asset_refs(event: dict[str, Any], asset_lookup: Mapping[str, Any]) -> dict[str, Any]:
    if not asset_lookup:
        return event
    enriched_refs = []
    for ref in event.get("asset_refs") or []:
        if not isinstance(ref, Mapping):
            continue
        enriched = dict(ref)
        asset = _lookup_asset_for_ref(ref, asset_lookup)
        if isinstance(asset, Mapping):
            enriched.setdefault("asset_id", asset.get("asset_id"))
            enriched.setdefault("source_type", asset.get("source_type"))
        enriched_refs.append(enriched)
    event["asset_refs"] = enriched_refs
    return event


def _lookup_asset_for_ref(ref: Mapping[str, Any], asset_lookup: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in _path_lookup_keys(Path("."), ref.get("path")):
        asset = asset_lookup.get(key)
        if isinstance(asset, Mapping):
            return asset
    return None


def _state_tags(
    *,
    state_type: str,
    primary_object: str | None,
    interaction_type: str | None,
    action_type: str | None,
    evidence_level: str | None,
    limitations: list[str],
    extra: Iterable[str] = (),
) -> list[str]:
    tags = [
        f"state:{state_type}",
        f"object:{primary_object}" if primary_object else "",
        f"interaction:{interaction_type}" if interaction_type else "",
        f"action:{action_type}" if action_type else "",
        f"evidence_level:{evidence_level}" if evidence_level else "",
        *[str(item) for item in extra if item],
    ]
    normalized_level = str(evidence_level or "").casefold()
    if normalized_level in LOW_EVIDENCE_LEVELS or normalized_level == "transcript_supported":
        tags.append("limited_evidence")
        tags.extend(f"limitation:{item}" for item in limitations if item)
    return _dedupe([tag for tag in tags if tag])


def _missing_time_tags(session_time: float | None, global_time: str | None) -> list[str]:
    tags: list[str] = []
    if session_time is None:
        tags.append("missing_session_time")
    if not global_time:
        tags.append("missing_global_time")
    return tags


def _state_sort_key(row: Mapping[str, Any]) -> tuple[int, float, float, int, str]:
    global_time = _parse_datetime(row.get("global_time"))
    if global_time is not None:
        return (0, _timestamp(global_time), 0.0, TIMED_STATE_ORDER.get(str(row.get("state_type")), 99), str(row.get("state_change_id") or ""))
    session_time = _as_float(row.get("session_time_sec"))
    if session_time is not None:
        return (1, float(session_time), 0.0, TIMED_STATE_ORDER.get(str(row.get("state_type")), 99), str(row.get("state_change_id") or ""))
    return (2, 0.0, 0.0, TIMED_STATE_ORDER.get(str(row.get("state_type")), 99), str(row.get("state_change_id") or ""))


def _timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).timestamp()
    return value.timestamp()


def _state_id(session_id: Any, source_id: Any, state_type: str) -> str:
    session = str(session_id or "session")
    source = str(source_id or "unknown")
    return f"{session}:{source}:{state_type}"


def _segment_id(micro: Mapping[str, Any]) -> str | None:
    return _first_text([micro.get("segment_id"), micro.get("parent_segment_id")])


def _micro_text(
    micro: Mapping[str, Any],
    state_type: str,
    primary_object: str | None,
    interaction_type: str | None,
) -> str:
    description = _as_mapping(micro.get("text_description"))
    summary = _first_text([description.get("summary"), description.get("index_text"), micro.get("text")])
    parts = [state_type]
    if interaction_type:
        parts.append(interaction_type)
    if primary_object:
        parts.append(f"object={primary_object}")
    if summary:
        parts.append(summary)
    return " | ".join(parts)


def _dialogue_items(micro: Mapping[str, Any]) -> list[Any]:
    dialogue = micro.get("dialogue_context")
    if isinstance(dialogue, list) and dialogue:
        return dialogue
    related = micro.get("related_dialogue")
    if isinstance(related, list) and related:
        return related
    if _truthy(micro.get("dialogue_context_available")):
        return [{"text": "dialogue context available"}]
    return []


def _dialogue_text(dialogue: Iterable[Any]) -> str:
    parts: list[str] = []
    for item in dialogue:
        if isinstance(item, Mapping):
            text = _first_text([item.get("text"), item.get("utterance"), item.get("content")])
        else:
            text = str(item) if item else ""
        if text:
            parts.append(text)
    return " ".join(parts)


def _first_text(values: Iterable[Any]) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text != "":
            return text
    return None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return parse_time(str(value))
    except Exception:
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item) != ""]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item) != ""]
    text = str(value)
    return [text] if text else []


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y", "on"}
    return bool(value)
