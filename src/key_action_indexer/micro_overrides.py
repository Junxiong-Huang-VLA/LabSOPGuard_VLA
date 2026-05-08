from __future__ import annotations

import json
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any

from .schemas import read_jsonl, write_jsonl
from .time_alignment import parse_time
from .vector_index import VectorIndex


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_manifest(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _override_value(override: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if override.get(key) is not None:
            return override[key]
    return None


def _set_times(item: dict[str, Any], session_start_time: str | None) -> None:
    if item.get("start_sec") is None or item.get("end_sec") is None:
        return
    item["start_sec"] = float(item["start_sec"])
    item["end_sec"] = float(item["end_sec"])
    item["duration_sec"] = max(0.0, float(item["end_sec"]) - float(item["start_sec"]))
    if session_start_time:
        start = parse_time(session_start_time) + timedelta(seconds=float(item["start_sec"]))
        end = parse_time(session_start_time) + timedelta(seconds=float(item["end_sec"]))
        item["global_start_time"] = start.isoformat()
        item["global_end_time"] = end.isoformat()


def _micro_to_vector_metadata(micro: dict[str, Any]) -> dict[str, Any]:
    interaction = micro.get("interaction") or {}
    text_description = micro.get("text_description") or {}
    index_info = micro.get("index") or {}
    first = micro.get("first_person") or {}
    third = micro.get("third_person") or {}
    keyframes_obj = micro.get("keyframes") or {}
    keyframes = [value for value in keyframes_obj.values() if value]
    return {
        "index_level": "micro_segment",
        "embedding_id": index_info.get("embedding_id") or f"emb_{micro.get('micro_segment_id')}",
        "segment_id": micro.get("parent_segment_id"),
        "micro_segment_id": micro.get("micro_segment_id"),
        "parent_segment_id": micro.get("parent_segment_id"),
        "display_order": micro.get("display_order"),
        "display_id": micro.get("display_id"),
        "session_id": micro.get("session_id"),
        "index_text": text_description.get("index_text") or "",
        "global_start_time": micro.get("global_start_time"),
        "global_end_time": micro.get("global_end_time"),
        "third_person_clip": third.get("clip_path") if isinstance(third, dict) else None,
        "first_person_clip": first.get("clip_path") if isinstance(first, dict) else None,
        "related_dialogue": [item.get("text", "") for item in micro.get("dialogue_context", []) if isinstance(item, dict)],
        "action_type": text_description.get("action_type"),
        "interaction_type": interaction.get("interaction_type"),
        "primary_object": interaction.get("primary_object"),
        "detected_objects": interaction.get("detected_objects") or [],
        "keyframes": keyframes,
        "quality": (micro.get("quality") or {}).get("confidence"),
        "class_threshold": micro.get("class_threshold") or {},
        "dialogue_context_available": bool(micro.get("dialogue_context_available") or micro.get("dialogue_context")),
        "manual_corrected": bool(micro.get("manual_corrected")),
        "manual_correction_note": micro.get("manual_correction_note"),
        "interaction": {
            "avg_interaction_score": interaction.get("avg_interaction_score"),
            "max_interaction_score": interaction.get("max_interaction_score"),
            "evidence_frame_indices": interaction.get("evidence_frame_indices") or [],
        },
    }


def _apply_one_override(micro: dict[str, Any], override: dict[str, Any], session_start_time: str | None) -> dict[str, Any]:
    item = deepcopy(micro)
    start_sec = _override_value(override, "override_start_sec", "start_sec")
    end_sec = _override_value(override, "override_end_sec", "end_sec")
    if start_sec is not None:
        item["start_sec"] = float(start_sec)
    if end_sec is not None:
        item["end_sec"] = float(end_sec)
    _set_times(item, session_start_time)

    interaction = dict(item.get("interaction") or {})
    primary = _override_value(override, "override_primary_object", "primary_object")
    interaction_type = _override_value(override, "override_interaction_type", "interaction_type")
    if primary:
        primary = str(primary)
        interaction["primary_object"] = primary
        interaction["interaction_type"] = str(interaction_type or f"hand_{primary}_contact")
        detected = list(interaction.get("detected_objects") or [])
        if primary not in detected:
            detected.append(primary)
        interaction["detected_objects"] = detected
    elif interaction_type:
        interaction["interaction_type"] = str(interaction_type)
    item["interaction"] = interaction

    text = dict(item.get("text_description") or {})
    action_type = _override_value(override, "override_action_type", "action_type")
    if action_type:
        text["action_type"] = str(action_type)
    if override.get("summary") is not None:
        text["summary"] = str(override["summary"])
    note = str(override.get("note") or "")
    index_text = str(text.get("index_text") or "")
    text["index_text"] = (
        f"{index_text.rstrip()}\n"
        "manual_corrected: true\n"
        f"manual_correction_note: {note}\n"
        f"override_primary_object: {interaction.get('primary_object')}\n"
        f"override_action_type: {text.get('action_type')}\n"
    )
    item["text_description"] = text

    quality = dict(item.get("quality") or {})
    if override.get("quality") is not None:
        quality["confidence"] = str(override["quality"])
    item["quality"] = quality
    item["manual_corrected"] = True
    item["manual_correction_note"] = note
    item["dialogue_context_available"] = bool(item.get("dialogue_context"))
    return item


def _next_micro_id(parent_segment_id: str, micros: list[dict[str, Any]]) -> str:
    prefix = f"{parent_segment_id}_micro_"
    max_index = 0
    for micro in micros:
        micro_id = str(micro.get("micro_segment_id") or "")
        if not micro_id.startswith(prefix):
            continue
        try:
            max_index = max(max_index, int(micro_id.removeprefix(prefix)))
        except ValueError:
            continue
    return f"{prefix}{max_index + 1:03d}"


def _insert_micro_from_override(
    override: dict[str, Any],
    micros: list[dict[str, Any]],
    session_start_time: str | None,
) -> dict[str, Any]:
    parent_segment_id = str(override.get("parent_segment_id") or override.get("segment_id") or "seg_000001")
    primary = str(_override_value(override, "override_primary_object", "primary_object") or "unknown_object")
    action_type = str(_override_value(override, "override_action_type", "action_type") or "unknown_operation")
    interaction_type = str(_override_value(override, "override_interaction_type", "interaction_type") or f"hand_{primary}_contact")
    start_sec = float(_override_value(override, "override_start_sec", "start_sec") or 0.0)
    end_sec = float(_override_value(override, "override_end_sec", "end_sec") or start_sec)
    micro_id = str(override.get("micro_segment_id") or _next_micro_id(parent_segment_id, micros))
    item = {
        "micro_segment_id": micro_id,
        "parent_segment_id": parent_segment_id,
        "session_id": override.get("session_id"),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "duration_sec": max(0.0, end_sec - start_sec),
        "global_start_time": None,
        "global_end_time": None,
        "first_person": None,
        "third_person": None,
        "interaction": {
            "interaction_type": interaction_type,
            "primary_object": primary,
            "secondary_objects": [],
            "detected_objects": [primary],
            "avg_interaction_score": None,
            "max_interaction_score": None,
            "evidence_frame_indices": [],
        },
        "keyframes": {},
        "dialogue_context": [],
        "text_description": {
            "action_type": action_type,
            "summary": str(override.get("summary") or ""),
            "index_text": f"manual inserted micro segment {micro_id} {primary} {interaction_type} {action_type}",
        },
        "index": {"index_level": "micro_segment", "embedding_id": f"emb_{micro_id}"},
        "quality": {"confidence": str(override.get("quality") or "manual"), "warnings": []},
    }
    _set_times(item, session_start_time)
    return _apply_one_override(item, override, session_start_time)


def apply_micro_overrides(
    session_dir: str | Path,
    overrides_path: str | Path | None = None,
    *,
    source_path: str | Path | None = None,
    rebuild_index: bool = True,
) -> dict[str, Any]:
    root = Path(session_dir)
    metadata_dir = root / "metadata"
    source = Path(source_path) if source_path else metadata_dir / "micro_segments.jsonl"
    if overrides_path:
        overrides = Path(overrides_path)
    else:
        overrides = metadata_dir / "micro_segments_overrides.jsonl"
        manual = metadata_dir / "manual_micro_segments.jsonl"
        if not overrides.exists() and manual.exists():
            overrides = manual
    if not source.exists():
        raise FileNotFoundError(f"micro segments not found: {source}")

    micros = read_jsonl(source)
    override_rows = read_jsonl(overrides) if overrides.exists() else []
    manifest = _load_manifest(root)
    session_start_time = manifest.get("session_start_time")
    overrides_by_id = {str(item.get("micro_segment_id")): item for item in override_rows if item.get("micro_segment_id")}

    updated = 0
    deleted = 0
    corrected: list[dict[str, Any]] = []
    for micro in micros:
        micro_id = str(micro.get("micro_segment_id") or "")
        override = overrides_by_id.get(micro_id)
        if override is None:
            corrected.append(micro)
            continue
        operation = str(override.get("operation") or "update").strip().lower()
        if operation == "delete":
            deleted += 1
            continue
        corrected.append(_apply_one_override(micro, override, session_start_time))
        updated += 1

    inserted = 0
    for override in override_rows:
        if str(override.get("operation") or "").strip().lower() != "insert":
            continue
        micro_id = str(override.get("micro_segment_id") or "")
        if micro_id and any(str(item.get("micro_segment_id") or "") == micro_id for item in corrected):
            continue
        corrected.append(_insert_micro_from_override(override, corrected, session_start_time))
        inserted += 1

    counters: dict[str, int] = {}
    corrected.sort(key=lambda item: (str(item.get("parent_segment_id") or ""), float(item.get("start_sec") or 0.0), str(item.get("micro_segment_id") or "")))
    for item in corrected:
        parent_id = str(item.get("parent_segment_id") or "seg_000001")
        counters[parent_id] = counters.get(parent_id, 0) + 1
        item["display_order"] = counters[parent_id]
        item["display_id"] = f"micro_{counters[parent_id]:03d}"

    corrected_path = metadata_dir / "micro_segments_corrected.jsonl"
    write_jsonl(corrected_path, corrected)
    micro_vector = [_micro_to_vector_metadata(item) for item in corrected]
    write_jsonl(metadata_dir / "micro_vector_metadata.jsonl", micro_vector)

    existing_vector = read_jsonl(metadata_dir / "vector_metadata.jsonl") if (metadata_dir / "vector_metadata.jsonl").exists() else []
    segment_vector = [item for item in existing_vector if item.get("index_level", "segment") == "segment"]
    combined = segment_vector + micro_vector
    write_jsonl(metadata_dir / "vector_metadata.jsonl", combined)

    index_dir: str | None
    if rebuild_index:
        index_dir = str(root / "index")
        index = VectorIndex()
        index.build([str(item.get("index_text") or "") for item in combined], combined)
        index.save(root / "index")
        micro_index = VectorIndex()
        micro_index.build([str(item.get("index_text") or "") for item in micro_vector], micro_vector)
        micro_index.save(root / "index" / "micro_segments")
    else:
        index_dir = None

    summary = {
        "session_dir": str(root),
        "overrides_path": str(overrides),
        "num_micro_segments": len(corrected),
        "num_overrides": len(override_rows),
        "updated": updated,
        "inserted": inserted,
        "deleted": deleted,
        "corrected_path": str(corrected_path),
        "index_dir": index_dir,
    }
    _write_json(root / "metadata" / "micro_override_summary.json", summary)
    return summary
