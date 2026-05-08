from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .health_report import build_run_health_report
from .model_observations import MODEL_INPUT_FILE_ALIASES


ADAPTER_VALIDATION_SCHEMA_VERSION = "key_action_evidence_adapter_validation.v1"
ADAPTER_VALIDATION_FILENAME = "evidence_adapter_validation.json"

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

    adapters: dict[str, Any] = {}
    for adapter_name, spec in CANONICAL_ADAPTERS.items():
        adapters[adapter_name] = _validate_adapter(
            metadata,
            adapter_name,
            spec,
            duration_sec=duration_sec,
            manifest_session_id=manifest_session_id,
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
    for row_index, row in enumerate(rows, start=1):
        row_issues = _validate_row(adapter_name, row, spec, row_index, duration_sec, manifest_session_id)
        row_issues.extend(_semantic_support_issues(adapter_name, row, row_index))
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


def _semantic_support_issues(adapter_name: str, row: Mapping[str, Any], row_index: int) -> list[dict[str, Any]]:
    support = _semantic_support(adapter_name, row)
    if support["supported"]:
        return []
    return [
        _issue(
            "warning",
            "semantic_support_missing",
            f"{adapter_name} row {row_index} is structurally valid but does not expose evidence fields for {', '.join(_supported_action_types(adapter_name))}",
            row=row_index,
            details={
                "adapter": adapter_name,
                "supported_action_types": _supported_action_types(adapter_name),
                "missing_semantic_fields": support["missing_fields"],
            },
        )
    ]


def _semantic_support(adapter_name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    if adapter_name == "object_tracks":
        groups = [
            ("object identity", ("object_label", "label", "class_name", "object_name")),
            ("trajectory", ("bbox", "points", "trajectory", "detections", "track")),
        ]
    elif adapter_name == "panel_ocr":
        groups = [
            ("readout/control state", ("display_text", "ocr_text", "readout", "reading", "value", "button_state", "knob_state", "switch_state", "control_state")),
        ]
    elif adapter_name == "liquid_state":
        groups = [
            ("liquid level/flow", ("liquid_level_y_norm", "meniscus_y_norm", "level_y_norm", "liquid_area_px", "mask_area_px", "volume_ml", "volume_ul", "flow_direction", "stream_width_px", "mask_path", "liquid_state")),
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
) -> dict[str, Any]:
    payload: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if path:
        payload["path"] = path
    if row is not None:
        payload["row"] = row
    if details:
        payload["details"] = dict(details)
    return payload


def _has_value(row: Mapping[str, Any], key: str) -> bool:
    value = row.get(key)
    if value in (None, ""):
        return False
    if isinstance(value, list) and not value:
        return False
    if isinstance(value, Mapping) and not value:
        return False
    return True


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
