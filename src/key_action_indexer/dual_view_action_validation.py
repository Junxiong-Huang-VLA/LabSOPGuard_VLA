from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DUAL_VIEW_ACTION_SCHEMA_VERSION = "dual_view_action_alignment.v1"
REQUIRED_FORMAL_MATERIALS = (
    ("first_person", "keyframe"),
    ("first_person", "keyclip"),
    ("third_person", "keyframe"),
    ("third_person", "keyclip"),
)

_ACTION_ALIASES = {
    "hand_balance": "hand-balance",
    "balance": "hand-balance",
    "scale": "hand-balance",
    "weighing": "hand-balance",
    "weighing_operation": "hand-balance",
    "balance_weighing": "hand-balance",
    "hand_object_balance": "hand-balance",
    "hand_balance_contact": "hand-balance",
    "hand_bottle": "hand-bottle",
    "hand_bottle_contact": "hand-bottle",
    "bottle_interaction": "hand-bottle",
    "reagent_bottle_interaction": "hand-bottle",
    "sample_bottle_interaction": "hand-bottle",
    "hand_reagent_bottle_contact": "hand-bottle",
    "hand_sample_bottle_contact": "hand-bottle",
    "hand_paper": "hand-paper",
    "hand_paper_contact": "hand-paper",
    "paper_interaction": "hand-paper",
    "weighing_paper_transfer": "hand-paper",
    "hand_spatula": "hand-spatula",
    "spatula_interaction": "hand-spatula",
    "hand_spatula_contact": "hand-spatula",
    "spatula_sampling": "hand-spatula",
    "solid_transfer": "hand-spatula",
    "hand_pipette": "hand-pipette",
    "pipetting": "hand-pipette",
    "sample_adding": "hand-pipette",
    "liquid_transfer": "hand-pipette",
    "hand_pipette_contact": "hand-pipette",
    "hand_pipette_tip_contact": "hand-pipette",
}

_OBJECT_ACTIONS = {
    "balance": "hand-balance",
    "scale": "hand-balance",
    "reagent_bottle": "hand-bottle",
    "sample_bottle": "hand-bottle",
    "sample_bottle_blue": "hand-bottle",
    "bottle": "hand-bottle",
    "paper": "hand-paper",
    "weighing_paper": "hand-paper",
    "spatula": "hand-spatula",
    "pipette": "hand-pipette",
    "pipette_tip": "hand-pipette",
}

_VIEW_ALIASES = {
    "first": "first_person",
    "fp": "first_person",
    "firstperson": "first_person",
    "first_person": "first_person",
    "egocentric": "first_person",
    "third": "third_person",
    "tp": "third_person",
    "thirdperson": "third_person",
    "third_person": "third_person",
    "external": "third_person",
}


@dataclass(frozen=True)
class DualViewActionEvent:
    event_id: str
    canonical_action_type: str
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float
    source_candidate_ids: list[str]
    views: dict[str, dict[str, Any]]
    material_refs: dict[str, dict[str, str]]
    timing: dict[str, Any]
    status: str = "formal"
    schema_version: str = DUAL_VIEW_ACTION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonicalize_action(value: Any = None, row: Mapping[str, Any] | None = None) -> str:
    row = row or {}
    raw = _first_text(
        value,
        row.get("canonical_action_type"),
        row.get("physical_action_type"),
        row.get("semantic_action"),
        row.get("action_type"),
        row.get("action_name"),
        row.get("event_type"),
    )
    token = _norm_token(raw)
    if token in _ACTION_ALIASES:
        return _ACTION_ALIASES[token]
    if token.startswith("hand_"):
        tail = token.removeprefix("hand_")
        if tail in _OBJECT_ACTIONS:
            return _OBJECT_ACTIONS[tail]
        return token.replace("_", "-")
    object_token = _norm_token(
        _first_text(
            row.get("canonical_object"),
            row.get("primary_object"),
            row.get("object_label"),
            _mapping(row.get("interaction")).get("primary_object"),
        )
    )
    if object_token in _OBJECT_ACTIONS:
        return _OBJECT_ACTIONS[object_token]
    if "weigh" in token or "balance" in token:
        return "hand-balance"
    if "bottle" in token:
        return "hand-bottle"
    if "paper" in token:
        return "hand-paper"
    if "spatula" in token or "scoop" in token:
        return "hand-spatula"
    if "pipette" in token or "liquid" in token:
        return "hand-pipette"
    return token.replace("_", "-") if token else ""


def summarize_gpu_config(
    config: Mapping[str, Any] | None = None,
    timing_rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    resolved_device: Any = None,
) -> dict[str, Any]:
    config = config or {}
    timing_rows = timing_rows or []
    requested = _first_text(
        config.get("yolo_device"),
        _mapping(config.get("detection")).get("yolo_device"),
        _mapping(config.get("yolo_config")).get("yolo_device"),
        "auto",
    )
    resolved = str(resolved_device) if resolved_device is not None else requested
    actual_devices = _unique(
        _first_text(row.get("actual_yolo_device"), row.get("actual_device"))
        for row in timing_rows
        if isinstance(row, Mapping)
    )
    requested_devices = _unique(
        _first_text(row.get("requested_yolo_device"), row.get("requested_device"))
        for row in timing_rows
        if isinstance(row, Mapping)
    )
    gpu_observed = any(_is_gpu_device(device) for device in [resolved, *actual_devices])
    gpu_auto = _is_auto_device(requested)
    gpu_requested = _is_gpu_request(requested)
    if gpu_observed:
        status = "gpu_observed"
    elif gpu_requested and actual_devices:
        status = "gpu_requested_but_not_observed"
    elif gpu_requested:
        status = "gpu_requested_no_timing"
    elif gpu_auto:
        status = "auto_no_gpu_observed"
    else:
        status = "gpu_not_requested"
    return {
        "requested_yolo_device": requested,
        "resolved_yolo_device": resolved,
        "requested_devices_in_timing": requested_devices,
        "actual_devices_in_timing": actual_devices,
        "gpu_auto_select": gpu_auto,
        "gpu_requested": gpu_requested,
        "gpu_observed": gpu_observed,
        "status": status,
        "timing_row_count": len(timing_rows),
    }


def validate_dual_view_action_alignment(
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    material_rows: Sequence[Mapping[str, Any]] | None = None,
    timing_rows: Sequence[Mapping[str, Any]] | None = None,
    config: Mapping[str, Any] | None = None,
    max_alignment_delta_sec: float = 1.0,
    require_formal_event: bool = False,
) -> dict[str, Any]:
    normalized = [_normalize_candidate(row) for row in candidate_rows if isinstance(row, Mapping)]
    material_index = _material_index([*normalized, *(material_rows or [])])
    formal_events: list[DualViewActionEvent] = []
    rejected: list[dict[str, Any]] = []

    weak = [row for row in normalized if not row["strong_evidence"]]
    for row in weak:
        rejected.append(_rejection("weak_evidence", [row], "candidate evidence is below formal threshold"))

    strong = [row for row in normalized if row["strong_evidence"] and row["view"] in {"first_person", "third_person"}]
    pairs, paired_ids = _pair_dual_view_rows(strong, max_alignment_delta_sec=max_alignment_delta_sec)
    for row in strong:
        if row["candidate_id"] not in paired_ids:
            rejected.append(_rejection("missing_complementary_view", [row], "strong evidence exists in only one required view"))

    for pair_index, (left, right) in enumerate(pairs, start=1):
        missing = _missing_materials(left, right, material_index)
        if missing:
            rejected.append(_rejection("missing_formal_material", [left, right], ",".join(missing), missing_materials=missing))
            continue
        event = _event_from_pair(pair_index, left, right, material_index)
        formal_events.append(event)

    events = [event.to_dict() for event in formal_events]
    errors: list[str] = []
    if require_formal_event and not events:
        errors.append("no_formal_dual_view_action_event")
    for event in events:
        if not isinstance(event.get("timing"), dict):
            errors.append(f"{event.get('event_id')}:missing_timing")
        else:
            missing_timing = [key for key in ("start_sec", "end_sec", "duration_sec", "view_delta_sec") if key not in event["timing"]]
            errors.extend(f"{event.get('event_id')}:missing_timing.{key}" for key in missing_timing)

    timing_summary = summarize_timing_rows(timing_rows or [])
    gpu_summary = summarize_gpu_config(config or {}, timing_rows or [])
    status = "fail" if errors else "pass"
    return {
        "schema_version": DUAL_VIEW_ACTION_SCHEMA_VERSION,
        "status": status,
        "errors": errors,
        "summary": {
            "candidate_count": len(normalized),
            "strong_candidate_count": len(strong),
            "dual_view_pair_count": len(pairs),
            "formal_event_count": len(events),
            "rejected_count": len(rejected),
        },
        "events": events,
        "rejected_candidates": rejected,
        "timing_rows_summary": timing_summary,
        "gpu_config": gpu_summary,
    }


def validate_dual_view_action_alignment_files(
    candidates_path: str | Path,
    *,
    materials_path: str | Path | None = None,
    timing_path: str | Path | None = None,
    config_path: str | Path | None = None,
    max_alignment_delta_sec: float = 1.0,
    require_formal_event: bool = False,
) -> dict[str, Any]:
    return validate_dual_view_action_alignment(
        read_jsonl(candidates_path),
        material_rows=read_jsonl(materials_path) if materials_path else [],
        timing_rows=read_jsonl(timing_path) if timing_path else [],
        config=_read_json(config_path) if config_path else {},
        max_alignment_delta_sec=max_alignment_delta_sec,
        require_formal_event=require_formal_event,
    )


def summarize_timing_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("stage", "wall_sec", "requested_device", "actual_device")
    normalized = []
    missing_by_row: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        item = {
            "stage": _first_text(row.get("stage"), row.get("pipeline_stage")),
            "wall_sec": row.get("wall_sec"),
            "requested_device": _first_text(row.get("requested_device"), row.get("requested_yolo_device")),
            "actual_device": _first_text(row.get("actual_device"), row.get("actual_yolo_device")),
        }
        missing = [key for key in required if item.get(key) in (None, "")]
        if missing:
            missing_by_row.append({"row_index": index, "missing_fields": missing})
        normalized.append(item)
    return {
        "row_count": len(normalized),
        "required_fields": list(required),
        "missing_field_rows": missing_by_row,
        "has_required_timing_fields": not missing_by_row,
    }


def read_jsonl(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    rows: list[dict[str, Any]] = []
    if not source.exists():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    view = _normalize_view(_first_text(row.get("view"), row.get("source_view"), row.get("camera_view")))
    start = _time_value(row, "start")
    end = _time_value(row, "end")
    if end is None and start is not None:
        end = start
    if start is None and end is not None:
        start = end
    canonical_action = canonicalize_action(row=row)
    confidence = _confidence(row)
    candidate_id = _first_text(
        row.get("candidate_id"),
        row.get("event_id"),
        row.get("micro_segment_id"),
        row.get("segment_id"),
        f"candidate_{abs(hash(json.dumps(dict(row), sort_keys=True, default=str))) % 1000000}",
    )
    return {
        **dict(row),
        "candidate_id": candidate_id,
        "view": view,
        "canonical_action_type": canonical_action,
        "start_sec": float(start or 0.0),
        "end_sec": float(end if end is not None else start or 0.0),
        "center_sec": _center_sec(start, end),
        "confidence": confidence,
        "strong_evidence": _strong_evidence(row, confidence),
        "group_id": _group_id(row),
    }


def _pair_dual_view_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_alignment_delta_sec: float,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], set[str]]:
    first_rows = [dict(row) for row in rows if row.get("view") == "first_person" and row.get("canonical_action_type")]
    third_rows = [dict(row) for row in rows if row.get("view") == "third_person" and row.get("canonical_action_type")]
    used_third: set[str] = set()
    paired_ids: set[str] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for first in sorted(first_rows, key=lambda row: (row.get("canonical_action_type"), float(row.get("center_sec") or 0.0))):
        candidates = []
        for third in third_rows:
            third_id = str(third.get("candidate_id"))
            if third_id in used_third:
                continue
            if third.get("canonical_action_type") != first.get("canonical_action_type"):
                continue
            if first.get("group_id") and third.get("group_id") and first.get("group_id") != third.get("group_id"):
                continue
            delta = abs(float(first.get("center_sec") or 0.0) - float(third.get("center_sec") or 0.0))
            if delta <= float(max_alignment_delta_sec):
                candidates.append((delta, third))
        if not candidates:
            continue
        _delta, selected = sorted(candidates, key=lambda item: item[0])[0]
        used_third.add(str(selected.get("candidate_id")))
        paired_ids.update({str(first.get("candidate_id")), str(selected.get("candidate_id"))})
        pairs.append((first, selected))
    return pairs, paired_ids


def _event_from_pair(
    pair_index: int,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    material_index: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> DualViewActionEvent:
    by_view = {str(left["view"]): left, str(right["view"]): right}
    first = by_view["first_person"]
    third = by_view["third_person"]
    start = min(float(first["start_sec"]), float(third["start_sec"]))
    end = max(float(first["end_sec"]), float(third["end_sec"]), start)
    group = _event_group_key(left, right)
    action = str(left["canonical_action_type"])
    material_refs = {
        view: {
            kind: material_index.get(f"{group}::{action}", {}).get(view, {}).get(kind, "")
            for kind in ("keyframe", "keyclip")
        }
        for view in ("first_person", "third_person")
    }
    timing = {
        "start_sec": round(start, 6),
        "end_sec": round(end, 6),
        "duration_sec": round(max(0.0, end - start), 6),
        "first_person_start_sec": round(float(first["start_sec"]), 6),
        "third_person_start_sec": round(float(third["start_sec"]), 6),
        "view_delta_sec": round(abs(float(first["center_sec"]) - float(third["center_sec"])), 6),
        "alignment_source": "dual_view_candidate_jsonl",
    }
    confidence = round(min(float(first["confidence"]), float(third["confidence"])), 6)
    return DualViewActionEvent(
        event_id=f"dual_view_action_{pair_index:06d}",
        canonical_action_type=action,
        start_sec=timing["start_sec"],
        end_sec=timing["end_sec"],
        duration_sec=timing["duration_sec"],
        confidence=confidence,
        source_candidate_ids=[str(first["candidate_id"]), str(third["candidate_id"])],
        views={
            "first_person": _view_payload(first),
            "third_person": _view_payload(third),
        },
        material_refs=material_refs,
        timing=timing,
    )


def _material_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, str]]]:
    index: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("exists") is False:
            continue
        normalized = _normalize_candidate(row)
        group = _event_group_key(normalized)
        action = str(normalized.get("canonical_action_type") or "")
        if not group or not action:
            continue
        bucket = index.setdefault(f"{group}::{action}", {})
        view = str(normalized.get("view") or "")
        if view not in {"first_person", "third_person"}:
            continue
        refs = bucket.setdefault(view, {})
        for kind, path in _material_paths(row).items():
            if path:
                refs[kind] = path
    return index


def _missing_materials(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    material_index: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> list[str]:
    group = _event_group_key(left, right)
    action = str(left.get("canonical_action_type") or "")
    refs = material_index.get(f"{group}::{action}", {})
    missing = []
    for view, kind in REQUIRED_FORMAL_MATERIALS:
        if not refs.get(view, {}).get(kind):
            missing.append(f"{view}:{kind}")
    return missing


def _material_paths(row: Mapping[str, Any]) -> dict[str, str]:
    kind = _material_kind(row)
    source = _first_text(
        row.get("source_file"),
        row.get("path"),
        row.get("file_path"),
        row.get("absolute_path"),
        row.get("keyframe_path"),
        row.get("keyclip_path"),
        row.get("clip_path"),
    )
    paths: dict[str, str] = {}
    if kind in {"keyframe", "keyclip"} and source:
        paths[kind] = source
    for key in ("keyframe_path", "representative_keyframe_path", "thumbnail_path"):
        if row.get(key):
            paths.setdefault("keyframe", str(row[key]))
    for key in ("keyclip_path", "clip_path", "annotated_clip_path", "video_clip_path"):
        if row.get(key):
            paths.setdefault("keyclip", str(row[key]))
    if isinstance(row.get("keyframes"), Mapping):
        for value in row["keyframes"].values():
            if value:
                paths.setdefault("keyframe", str(value))
                break
    return paths


def _material_kind(row: Mapping[str, Any]) -> str:
    raw = _norm_token(_first_text(row.get("asset_kind"), row.get("material_type"), row.get("asset_type"), row.get("kind")))
    if raw in {"keyframe", "key_frame", "frame", "thumbnail"}:
        return "keyframe"
    if raw in {"keyclip", "key_clip", "clip", "video", "segment_clip"}:
        return "keyclip"
    return raw


def _event_group_key(*rows: Mapping[str, Any]) -> str:
    for row in rows:
        group = _first_text(row.get("group_id"), _group_id(row))
        if group:
            return group
    return ""


def _group_id(row: Mapping[str, Any]) -> str:
    return _first_text(
        row.get("dual_view_group_id"),
        row.get("physical_action_material_id"),
        row.get("material_group_id"),
        row.get("micro_segment_id"),
        row.get("parent_micro_segment_id"),
        row.get("segment_id"),
        row.get("parent_segment_id"),
        row.get("candidate_group_id"),
    )


def _view_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "start_sec": row.get("start_sec"),
        "end_sec": row.get("end_sec"),
        "confidence": row.get("confidence"),
        "evidence_level": row.get("evidence_level") or row.get("evidence_grade") or row.get("evidence_strength"),
    }


def _rejection(reason: str, rows: Sequence[Mapping[str, Any]], message: str, **extra: Any) -> dict[str, Any]:
    return {
        "reason": reason,
        "message": message,
        "candidate_ids": [str(row.get("candidate_id") or "") for row in rows],
        "views": sorted({str(row.get("view") or "") for row in rows if row.get("view")}),
        "canonical_action_types": sorted({str(row.get("canonical_action_type") or "") for row in rows if row.get("canonical_action_type")}),
        **extra,
    }


def _strong_evidence(row: Mapping[str, Any], confidence: float) -> bool:
    level = _norm_token(_first_text(row.get("evidence_level"), row.get("evidence_grade"), row.get("evidence_strength"), row.get("status")))
    if level in {"strong", "high", "confirmed", "formal", "auto_confirmed", "dual_view"}:
        return True
    if level in {"weak", "low", "rejected", "candidate_only"}:
        return False
    return confidence >= 0.75


def _confidence(row: Mapping[str, Any]) -> float:
    values = [
        row.get("confidence"),
        row.get("score"),
        row.get("interaction_score"),
        row.get("active_score"),
        row.get("probability"),
        row.get("prob"),
    ]
    for value in values:
        number = _float_or_none(value)
        if number is not None:
            return max(0.0, min(1.0, number))
    return 0.0


def _time_value(row: Mapping[str, Any], edge: str) -> float | None:
    keys = (
        ("start_sec", "global_start_sec", "alignment_start_sec", "local_start_sec", "time_sec", "alignment_time_sec")
        if edge == "start"
        else ("end_sec", "global_end_sec", "alignment_end_sec", "local_end_sec", "time_sec", "alignment_time_sec")
    )
    for key in keys:
        number = _float_or_none(row.get(key))
        if number is not None:
            return number
    timing = row.get("timing")
    if isinstance(timing, Mapping):
        return _time_value(timing, edge)
    return None


def _center_sec(start: float | None, end: float | None) -> float:
    if start is None and end is None:
        return 0.0
    if start is None:
        return float(end or 0.0)
    if end is None:
        return float(start)
    return (float(start) + float(end)) / 2.0


def _normalize_view(value: Any) -> str:
    token = _norm_token(value)
    return _VIEW_ALIASES.get(token, token)


def _is_gpu_request(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if token in {"cuda", "gpu", "mps"}:
        return True
    if token.startswith(("cuda", "mps")):
        return True
    return token.isdigit()


def _is_auto_device(value: Any) -> bool:
    token = str(value or "").strip().lower()
    return token in {"", "none", "auto"}


def _is_gpu_device(value: Any) -> bool:
    token = str(value or "").strip().lower()
    if token.startswith(("cuda", "mps")):
        return True
    return token.isdigit()


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _norm_token(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


__all__ = [
    "DUAL_VIEW_ACTION_SCHEMA_VERSION",
    "DualViewActionEvent",
    "canonicalize_action",
    "read_jsonl",
    "summarize_gpu_config",
    "summarize_timing_rows",
    "validate_dual_view_action_alignment",
    "validate_dual_view_action_alignment_files",
    "write_json",
]
