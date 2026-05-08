from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .health_report import build_run_health_report
from .model_observations import MODEL_INPUT_FILE_ALIASES
from .schemas import read_jsonl


ADAPTER_VALIDATION_SCHEMA_VERSION = "key_action_evidence_adapter_validation.v1"
ADAPTER_VALIDATION_FILENAME = "evidence_adapter_validation.json"
POINT_EVENT_NEAR_MATCH_TOLERANCE_SEC = 1.0

CANONICAL_ADAPTERS: dict[str, dict[str, Any]] = {
    "object_tracks": {
        "source_type": "object_track",
        "canonical_file": "object_tracks.jsonl",
        "required_any": (("object_label", "label", "class_name", "object_name"), ("track_id", "object_track_id", "tracklet_id"), ("bbox", "points", "trajectory", "detections", "track")),
    },
    "panel_ocr": {
        "source_type": "equipment_panel_state",
        "canonical_file": "panel_ocr.jsonl",
        "required_any": (("equipment_label", "equipment_id", "panel_label", "object_label", "label"), ("display_text", "ocr_text", "readout", "reading", "value", "button_state", "knob_state", "switch_state", "control_state", "state")),
    },
    "liquid_state": {
        "source_type": "liquid_segmentation",
        "canonical_file": "liquid_state.jsonl",
        "required_any": (("object_label", "container_label", "container_id", "target_object", "label"), ("liquid_level_y_norm", "meniscus_y_norm", "level_y_norm", "liquid_area_px", "mask_area_px", "volume_ml", "volume_ul", "flow_direction", "stream_width_px", "mask_path", "state", "liquid_state")),
    },
    "container_state": {
        "source_type": "container_state",
        "canonical_file": "container_state.jsonl",
        "required_any": (("container_label", "container_id", "object_label", "label"), ("state", "before_state", "after_state", "lid_state", "cap_state", "open_closed_state", "open_close_state", "color_before", "color_after", "liquid_level_y_norm", "volume_ml", "volume_ul")),
    },
}


def validate_evidence_adapters(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    health = build_run_health_report(session)
    metrics = health.get("metrics") if isinstance(health.get("metrics"), Mapping) else {}
    duration_sec = _float(metrics.get("video_duration_sec"))
    manifest_session_id = _manifest_session_id(session)
    context = _validation_context(metadata)

    adapters: dict[str, Any] = {}
    for adapter_name, spec in CANONICAL_ADAPTERS.items():
        adapters[adapter_name] = _validate_adapter(
            metadata,
            adapter_name,
            spec,
            duration_sec=duration_sec,
            manifest_session_id=manifest_session_id,
            context=context,
        )

    totals = {
        "adapter_count": len(adapters),
        "present_adapter_count": sum(1 for item in adapters.values() if item["present"]),
        "row_count": sum(int(item["row_count"]) for item in adapters.values()),
        "error_count": sum(int(item["error_count"]) for item in adapters.values()),
        "warning_count": sum(int(item["warning_count"]) for item in adapters.values()),
        "semantic_issue_count": sum(int(item.get("semantic_issue_count") or 0) for item in adapters.values()),
        "missing_adapter_count": sum(1 for item in adapters.values() if not item["present"]),
    }
    status = "fail" if totals["error_count"] else "warning" if totals["warning_count"] or totals["missing_adapter_count"] else "pass"
    payload = {
        "schema_version": ADAPTER_VALIDATION_SCHEMA_VERSION,
        "generated_at": _now(),
        "session_dir": str(session),
        "metadata_dir": str(metadata),
        "status": status,
        "duration_sec": duration_sec,
        "manifest_session_id": manifest_session_id,
        "summary": totals,
        "adapters": adapters,
    }
    target = Path(output_path) if output_path is not None else metadata / ADAPTER_VALIDATION_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["validation_path"] = str(target)
    return payload


def _validate_adapter(
    metadata: Path,
    adapter_name: str,
    spec: Mapping[str, Any],
    *,
    duration_sec: float | None,
    manifest_session_id: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    source_type = str(spec["source_type"])
    aliases = list(MODEL_INPUT_FILE_ALIASES.get(source_type, (spec["canonical_file"],)))
    canonical = str(spec["canonical_file"])
    if canonical not in aliases:
        aliases.insert(0, canonical)
    paths = [metadata / name for name in aliases]
    present_paths = [path for path in paths if path.exists()]
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    line_errors = 0

    if not present_paths:
        issues.append(_issue("warning", "adapter_missing", f"{canonical} is not present", path=str(metadata / canonical)))
    for path in present_paths:
        parsed, errors = _read_jsonl_with_errors(path)
        rows.extend({**row, "_source_path": str(path), "_source_line": index + 1} for index, row in enumerate(parsed))
        for error in errors:
            issues.append(error)
        line_errors += len(errors)

    starts: list[float] = []
    ends: list[float] = []
    views: set[str] = set()
    session_ids: set[str] = set()
    valid_rows = 0
    semantic_summary = {"missing_fields": 0, "time_mismatch": 0, "action_mismatch": 0}
    linked_segments: set[str] = set()
    linked_micros: set[str] = set()
    for row_index, row in enumerate(rows, start=1):
        row_issues = _validate_row(adapter_name, row, spec, row_index, duration_sec, manifest_session_id)
        semantic = _semantic_support_issues(adapter_name, row, row_index, context=context)
        row_issues.extend(semantic)
        for issue in semantic:
            category = str(issue.get("semantic_category") or "")
            if category in semantic_summary:
                semantic_summary[category] += 1
        relation = _row_relation(row, context, time_tolerance_sec=_semantic_time_tolerance_sec(adapter_name, row))
        if relation.get("segment_id"):
            linked_segments.add(str(relation["segment_id"]))
        if relation.get("micro_segment_id"):
            linked_micros.add(str(relation["micro_segment_id"]))
        issues.extend(row_issues)
        if not any(issue["severity"] == "error" for issue in row_issues):
            valid_rows += 1
        start, end = _row_time_window(row)
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
        view = str(row.get("view") or row.get("source_view") or row.get("camera") or row.get("camera_id") or "").strip()
        if view:
            views.add(view)
        session_id = str(row.get("session_id") or "").strip()
        if session_id:
            session_ids.add(session_id)

    if rows and not views:
        issues.append(_issue("warning", "adapter_view_missing", f"{canonical} has rows but no view/source_view/camera field"))
    if len(views) == 1:
        issues.append(_issue("warning", "single_view_adapter", f"{canonical} only reports one view: {sorted(views)[0]}"))
    if manifest_session_id and session_ids and session_ids != {manifest_session_id}:
        issues.append(_issue("error", "adapter_session_mismatch", f"{canonical} session_id values do not match manifest", details={"expected": manifest_session_id, "actual": sorted(session_ids)}))

    error_count = sum(1 for item in issues if item["severity"] == "error")
    warning_count = sum(1 for item in issues if item["severity"] == "warning")
    semantic_issue_count = sum(1 for item in issues if str(item.get("code") or "").startswith("semantic_"))
    return {
        "adapter": adapter_name,
        "canonical_file": canonical,
        "accepted_aliases": aliases,
        "present": bool(present_paths),
        "source_paths": [str(path) for path in present_paths],
        "row_count": len(rows),
        "valid_row_count": valid_rows,
        "error_count": error_count,
        "warning_count": warning_count,
        "semantic_issue_count": semantic_issue_count,
        "semantic_summary": semantic_summary,
        "supported_action_types": _supported_action_types(adapter_name),
        "status": "fail" if error_count else "warning" if warning_count or not present_paths else "pass",
        "coverage": {
            "start_sec": min(starts) if starts else None,
            "end_sec": max(ends) if ends else None,
            "duration_sec": round(max(ends) - min(starts), 4) if starts and ends and max(ends) >= min(starts) else None,
            "session_duration_sec": duration_sec,
        },
        "views": sorted(views),
        "session_ids": sorted(session_ids),
        "line_error_count": line_errors,
        "linked_segment_ids": sorted(linked_segments),
        "linked_micro_segment_ids": sorted(linked_micros),
        "issues": issues[:80],
    }


def _validate_row(
    adapter_name: str,
    row: Mapping[str, Any],
    spec: Mapping[str, Any],
    row_index: int,
    duration_sec: float | None,
    manifest_session_id: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for group in spec.get("required_any") or []:
        if not any(_has_value(row, key) for key in group):
            issues.append(
                _issue(
                    "error",
                    "required_field_group_missing",
                    f"{adapter_name} row {row_index} needs one of: {', '.join(group)}",
                    row=row_index,
                    details={"fields": list(group)},
                )
            )
    start, end = _row_time_window(row)
    if start is None and end is None:
        issues.append(_issue("warning", "time_window_missing", f"{adapter_name} row {row_index} has no local/session time", row=row_index))
    elif start is not None and end is not None and end < start:
        issues.append(_issue("error", "time_window_reversed", f"{adapter_name} row {row_index} end_sec is before start_sec", row=row_index))
    if duration_sec is not None:
        for label, value in (("start_sec", start), ("end_sec", end)):
            if value is not None and (value < -2.0 or value > duration_sec + 2.0):
                issues.append(
                    _issue(
                        "error",
                        "time_window_out_of_session",
                        f"{adapter_name} row {row_index} {label}={value} is outside session duration {duration_sec}",
                        row=row_index,
                        details={"field": label, "value": value, "duration_sec": duration_sec},
                    )
                )
    session_id = str(row.get("session_id") or "").strip()
    if manifest_session_id and session_id and session_id != manifest_session_id:
        issues.append(_issue("error", "row_session_mismatch", f"{adapter_name} row {row_index} session_id={session_id} does not match {manifest_session_id}", row=row_index))
    confidence = row.get("confidence", row.get("score", row.get("model_confidence")))
    if confidence is not None:
        numeric = _float(confidence)
        if numeric is None or numeric < 0.0 or numeric > 1.0:
            issues.append(_issue("warning", "confidence_out_of_range", f"{adapter_name} row {row_index} confidence should be 0-1", row=row_index))
    return issues


def _semantic_support_issues(
    adapter_name: str,
    row: Mapping[str, Any],
    row_index: int,
    *,
    context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    relation = _row_relation(row, context, time_tolerance_sec=_semantic_time_tolerance_sec(adapter_name, row))
    support = _semantic_support(adapter_name, row)
    if not support["supported"]:
        issues.append(
            _issue(
                "warning",
                "semantic_missing_fields",
                f"{adapter_name} row {row_index} is structurally valid but lacks semantic evidence fields for {', '.join(_supported_action_types(adapter_name))}",
                row=row_index,
                details={
                    "adapter": adapter_name,
                    "supported_action_types": _supported_action_types(adapter_name),
                    "missing_semantic_fields": support["missing_fields"],
                    "relation": relation,
                },
                semantic_category="missing_fields",
            )
        )
    if not relation.get("time_overlap_ok"):
        issues.append(
            _issue(
                "warning",
                "semantic_time_mismatch",
                f"{adapter_name} row {row_index} does not overlap a known segment or micro time window",
                row=row_index,
                details={"adapter": adapter_name, "relation": relation},
                semantic_category="time_mismatch",
            )
        )
    mismatch = _semantic_action_mismatch(adapter_name, row, relation)
    if mismatch:
        issues.append(
            _issue(
                "warning",
                "semantic_action_mismatch",
                f"{adapter_name} row {row_index} does not match the linked action/object labels",
                row=row_index,
                details={"adapter": adapter_name, "relation": relation, **mismatch},
                semantic_category="action_mismatch",
            )
        )
    return issues


def _semantic_support(adapter_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if adapter_name == "object_tracks":
        groups = [
            ("object identity", ("object_label", "label", "class_name", "object_name")),
            ("trajectory", ("bbox", "points", "trajectory", "detections", "track")),
        ]
    elif adapter_name == "panel_ocr":
        groups = [
            (
                "readout/control/interaction state",
                (
                    "display_text",
                    "ocr_text",
                    "readout",
                    "reading",
                    "value",
                    "button_state",
                    "button_pressed",
                    "knob_state",
                    "knob_angle_deg",
                    "switch_state",
                    "control_state",
                    "panel_state",
                    "interaction_score",
                    "interaction",
                    "candidate_type",
                ),
            ),
        ]
    elif adapter_name == "liquid_state":
        groups = [
            (
                "liquid level/flow/transfer geometry",
                (
                    "liquid_level_y_norm",
                    "meniscus_y_norm",
                    "level_y_norm",
                    "liquid_area_px",
                    "mask_area_px",
                    "volume_ml",
                    "volume_ul",
                    "flow_direction",
                    "stream_width_px",
                    "mask_path",
                    "liquid_state",
                    "tool_label",
                    "container_label",
                    "tool_container_distance_px",
                    "candidate_type",
                ),
            ),
        ]
    elif adapter_name == "container_state":
        groups = [
            ("open/closed/container state", ("state", "before_state", "after_state", "lid_state", "cap_state", "open_closed_state", "open_close_state", "color_before", "color_after", "liquid_level_y_norm", "volume_ml", "volume_ul")),
        ]
    else:
        groups = []
    missing = [label for label, keys in groups if not any(_has_value(row, key) for key in keys)]
    return {"supported": not missing, "missing_fields": missing}


def _supported_action_types(adapter_name: str) -> list[str]:
    return {
        "object_tracks": ["hand_object_interaction", "object_presence", "motion"],
        "panel_ocr": ["recording_or_reading", "panel_readout", "equipment_control"],
        "liquid_state": ["liquid_level", "liquid_flow", "liquid_state_change"],
        "container_state": ["open_close", "container_color", "container_state_change"],
    }.get(adapter_name, [])


def _read_jsonl_with_errors(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        return [], [_issue("error", "adapter_file_unreadable", str(exc), path=str(path))]
    for line_number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            errors.append(_issue("error", "jsonl_parse_error", str(exc), path=str(path), row=line_number))
            continue
        if not isinstance(row, dict):
            errors.append(_issue("error", "jsonl_row_not_object", "JSONL row must be an object", path=str(path), row=line_number))
            continue
        rows.append(row)
    return rows, errors


def _row_time_window(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = _first_float(row, "start_sec", "start_time_sec", "local_start_sec", "session_start_sec", "time_sec", "local_time_sec", "session_time_sec", "timestamp_sec")
    end = _first_float(row, "end_sec", "end_time_sec", "local_end_sec", "session_end_sec")
    duration = _first_float(row, "duration_sec")
    if end is None and start is not None and duration is not None:
        end = start + duration
    if end is None and start is not None:
        end = start
    if start is None or end is None:
        points = row.get("points") or row.get("trajectory") or row.get("detections") or row.get("track")
        if isinstance(points, list):
            times = [_first_float(point, "time_sec", "local_time_sec", "session_time_sec", "timestamp_sec") for point in points if isinstance(point, Mapping)]
            times = [value for value in times if value is not None]
            if times:
                start = min(times) if start is None else start
                end = max(times) if end is None else end
    return start, end


def _validation_context(metadata: Path) -> dict[str, Any]:
    raw_segments = _read_jsonl_if_exists(metadata / "key_action_segments.jsonl")
    raw_micros = _read_jsonl_if_exists(metadata / "micro_segments.jsonl")
    reviewed_segments = _read_jsonl_if_exists(metadata / "reviewed_segments.jsonl")
    reviewed_micros = _read_jsonl_if_exists(metadata / "reviewed_micro_segments.jsonl")
    segments = [*raw_segments, *reviewed_segments]
    micros = [*raw_micros, *reviewed_micros]
    return {
        "segments": segments,
        "micros": micros,
        "segments_by_id": {str(row.get("segment_id") or ""): row for row in segments if row.get("segment_id")},
        "micros_by_id": {str(row.get("micro_segment_id") or ""): row for row in micros if row.get("micro_segment_id")},
    }


def _row_relation(row: Mapping[str, Any], context: Mapping[str, Any], *, time_tolerance_sec: float = 0.0) -> dict[str, Any]:
    start, end = _row_time_window(row)
    segment_id = str(row.get("segment_id") or "")
    micro_id = str(row.get("micro_segment_id") or "")
    segments_by_id = context.get("segments_by_id") if isinstance(context.get("segments_by_id"), Mapping) else {}
    micros_by_id = context.get("micros_by_id") if isinstance(context.get("micros_by_id"), Mapping) else {}
    linked_micro = micros_by_id.get(micro_id) if micro_id else None
    linked_segment = segments_by_id.get(segment_id) if segment_id else None
    if not isinstance(linked_micro, Mapping):
        linked_micro = _best_overlap(row, context.get("micros") if isinstance(context.get("micros"), list) else [], time_tolerance_sec=time_tolerance_sec)
    if not isinstance(linked_segment, Mapping):
        linked_segment = _best_overlap(row, context.get("segments") if isinstance(context.get("segments"), list) else [], time_tolerance_sec=time_tolerance_sec)
    if isinstance(linked_micro, Mapping):
        micro_id = str(linked_micro.get("micro_segment_id") or micro_id)
        segment_id = str(linked_micro.get("parent_segment_id") or linked_micro.get("segment_id") or segment_id)
    elif isinstance(linked_segment, Mapping):
        segment_id = str(linked_segment.get("segment_id") or segment_id)
    related = linked_micro if isinstance(linked_micro, Mapping) else linked_segment if isinstance(linked_segment, Mapping) else {}
    related_window = _row_time_window(related) if isinstance(related, Mapping) else (None, None)
    overlap = _interval_overlap((start, end), related_window) if isinstance(related, Mapping) else 0.0
    gap = _interval_gap((start, end), related_window) if isinstance(related, Mapping) else None
    time_overlap_ok = bool(
        overlap > 0.0
        or (start is None and end is None)
        or (time_tolerance_sec > 0.0 and gap is not None and gap <= time_tolerance_sec)
    )
    return {
        "segment_id": segment_id or None,
        "micro_segment_id": micro_id or None,
        "start_sec": start,
        "end_sec": end,
        "linked_start_sec": related_window[0],
        "linked_end_sec": related_window[1],
        "time_overlap_sec": round(overlap, 4),
        "time_gap_sec": round(gap, 4) if gap is not None else None,
        "time_tolerance_sec": time_tolerance_sec,
        "time_overlap_ok": time_overlap_ok,
        "linked_action_type": _first_text(related, "action_type", "interaction_type") if isinstance(related, Mapping) else "",
        "linked_objects": sorted(_object_terms(related)) if isinstance(related, Mapping) else [],
    }


def _best_overlap(row: Mapping[str, Any], candidates: list[Any], *, time_tolerance_sec: float = 0.0) -> Mapping[str, Any] | None:
    row_interval = _row_time_window(row)
    best: tuple[float, Mapping[str, Any]] | None = None
    nearest: tuple[float, Mapping[str, Any]] | None = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_interval = _row_time_window(candidate)
        overlap = _interval_overlap(row_interval, candidate_interval)
        if overlap <= 0:
            gap = _interval_gap(row_interval, candidate_interval)
            if time_tolerance_sec > 0.0 and gap is not None and gap <= time_tolerance_sec:
                if nearest is None or gap < nearest[0]:
                    nearest = (gap, candidate)
            continue
        if best is None or overlap > best[0]:
            best = (overlap, candidate)
    return best[1] if best else nearest[1] if nearest else None


def _interval_overlap(left: tuple[float | None, float | None], right: tuple[float | None, float | None]) -> float:
    left_start, left_end = left
    right_start, right_end = right
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return 0.0
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _interval_gap(left: tuple[float | None, float | None], right: tuple[float | None, float | None]) -> float | None:
    left_start, left_end = left
    right_start, right_end = right
    if left_start is None or left_end is None or right_start is None or right_end is None:
        return None
    if _interval_overlap(left, right) > 0.0:
        return 0.0
    if left_end <= right_start:
        return right_start - left_end
    return left_start - right_end


def _semantic_time_tolerance_sec(adapter_name: str, row: Mapping[str, Any]) -> float:
    if adapter_name != "panel_ocr":
        return 0.0
    start, end = _row_time_window(row)
    if start is None or end is None:
        return 0.0
    if abs(end - start) <= 0.5:
        return POINT_EVENT_NEAR_MATCH_TOLERANCE_SEC
    return 0.0


def _semantic_action_mismatch(adapter_name: str, row: Mapping[str, Any], relation: Mapping[str, Any]) -> dict[str, Any] | None:
    linked_action = _norm(relation.get("linked_action_type"))
    linked_objects = {_norm(item) for item in relation.get("linked_objects") or []}
    row_action = _norm(_first_text(row, "action_type", "interaction_type", "event_type", "observation_type"))
    row_objects = _object_terms(row)
    expected_family = _adapter_action_family(adapter_name, row)
    if row_action and expected_family and not _action_family_matches(row_action, expected_family):
        return {"mismatch_type": "adapter_event_type", "row_action": row_action, "expected_family": sorted(expected_family)}
    if linked_action and expected_family and not _action_family_matches(linked_action, expected_family) and _requires_strict_action_match(adapter_name, row):
        return {"mismatch_type": "linked_action_type", "linked_action": linked_action, "expected_family": sorted(expected_family)}
    if linked_objects and row_objects and linked_objects.isdisjoint(row_objects) and _requires_strict_object_match(adapter_name, row):
        return {"mismatch_type": "object_label", "row_objects": sorted(row_objects), "linked_objects": sorted(linked_objects)}
    return None


def _adapter_action_family(adapter_name: str, row: Mapping[str, Any]) -> set[str]:
    text = _row_text(row)
    if adapter_name == "object_tracks":
        return {"object", "motion", "hand", "track", "interaction"}
    if adapter_name == "panel_ocr":
        if any(token in text for token in ("ocr", "readout", "reading", "display", "panel", "control", "button", "knob", "switch", "equipment")):
            return {"panel", "readout", "recording", "control", "equipment", "interaction"}
    if adapter_name == "liquid_state":
        if any(token in text for token in ("liquid", "flow", "transfer", "pipette", "meniscus", "volume", "stream")):
            return {"liquid", "flow", "transfer", "pipetting"}
    if adapter_name == "container_state":
        return {"container", "open", "close", "cap", "lid", "color", "level"}
    return set()


def _action_family_matches(value: str, families: set[str]) -> bool:
    value = _norm(value)
    return any(family and family in value for family in families)


def _requires_strict_action_match(adapter_name: str, row: Mapping[str, Any]) -> bool:
    confirmation = _norm(row.get("confirmation_level"))
    explicit = _first_text(row, "action_type", "interaction_type")
    return bool(explicit and confirmation in {"confirmed", "measured", "trusted"})


def _requires_strict_object_match(adapter_name: str, row: Mapping[str, Any]) -> bool:
    return adapter_name in {"container_state"} and _norm(row.get("confirmation_level")) in {"confirmed", "measured", "trusted"}


def _object_terms(row: Mapping[str, Any]) -> set[str]:
    terms = {
        _norm(_first_text(row, "object_label", "label", "class_name", "object_name", "container_label", "equipment_label", "target_object")),
    }
    measurement = row.get("measurement") if isinstance(row.get("measurement"), Mapping) else {}
    for key in ("object_label", "container_label", "tool_label", "equipment_label", "panel_label", "cap_label", "hand_label"):
        value = measurement.get(key)
        if value:
            terms.add(_norm(value))
    interaction = measurement.get("interaction") if isinstance(measurement.get("interaction"), Mapping) else {}
    for key in ("object_label", "hand_label"):
        value = interaction.get(key)
        if value:
            terms.add(_norm(value))
    for key in ("detected_objects", "visual_keywords"):
        for item in row.get(key) or []:
            terms.add(_norm(item))
    terms.discard("")
    return terms


def _row_text(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, default=str).casefold().replace("-", "_").replace(" ", "_")


def _manifest_session_id(session: Path) -> str:
    for path in (session / "manifest.json", session / "metadata" / "manifest.json"):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, Mapping) and data.get("session_id"):
            return str(data["session_id"])
    return ""


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    path: str | None = None,
    row: int | None = None,
    details: Mapping[str, Any] | None = None,
    semantic_category: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if path:
        payload["path"] = path
    if row is not None:
        payload["row"] = row
    if details:
        payload["details"] = dict(details)
    if semantic_category:
        payload["semantic_category"] = semantic_category
    return payload


def _has_value(row: Mapping[str, Any], key: str) -> bool:
    value = _nested_value(row, key)
    if value in (None, ""):
        return False
    if isinstance(value, list) and not value:
        return False
    if isinstance(value, Mapping) and not value:
        return False
    return True


def _nested_value(row: Mapping[str, Any], key: str) -> Any:
    if key in row:
        return row.get(key)
    for container_key in ("measurement", "metrics"):
        nested = row.get(container_key)
        if isinstance(nested, Mapping):
            if key in nested:
                return nested.get(key)
            measurement = nested.get("measurement")
            if isinstance(measurement, Mapping) and key in measurement:
                return measurement.get(key)
    return None


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = _nested_value(row, key)
        if value not in (None, ""):
            return str(value)
    return ""


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "ADAPTER_VALIDATION_FILENAME",
    "ADAPTER_VALIDATION_SCHEMA_VERSION",
    "CANONICAL_ADAPTERS",
    "validate_evidence_adapters",
]
