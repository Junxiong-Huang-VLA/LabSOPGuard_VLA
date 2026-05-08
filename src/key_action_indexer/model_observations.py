from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl, write_jsonl


MODEL_OBSERVATION_FIELDS = (
    "observation_id",
    "session_id",
    "segment_id",
    "micro_segment_id",
    "source_file",
    "source_type",
    "observation_type",
    "event_type",
    "global_start_time",
    "global_end_time",
    "start_sec",
    "end_sec",
    "view",
    "object_label",
    "track_id",
    "state",
    "measurement",
    "confirmation_level",
    "confidence",
    "confidence_reasons",
    "evidence_reasons",
    "limitations",
    "metrics",
    "asset_refs",
    "evidence_refs",
    "payload",
)

MODEL_INPUT_FILES = {
    "liquid_segmentation": "liquid_segmentation.jsonl",
    "equipment_panel_state": "equipment_panel_states.jsonl",
    "container_state": "container_state_events.jsonl",
    "object_track": "object_tracks.jsonl",
}

MODEL_INPUT_FILE_ALIASES = {
    "liquid_segmentation": (
        "liquid_segmentation.jsonl",
        "liquid_state.jsonl",
        "liquid_level_events.jsonl",
        "liquid_flow_events.jsonl",
    ),
    "equipment_panel_state": (
        "equipment_panel_states.jsonl",
        "panel_ocr.jsonl",
        "equipment_panel_ocr.jsonl",
        "equipment_control_states.jsonl",
        "button_knob_switch_states.jsonl",
    ),
    "container_state": (
        "container_state_events.jsonl",
        "container_state.jsonl",
        "container_state_candidates.jsonl",
        "container_open_close_events.jsonl",
        "container_color_events.jsonl",
        "container_liquid_level_events.jsonl",
    ),
    "object_track": ("object_tracks.jsonl",),
}


def build_model_observation_events(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = Path(output_path) if output_path is not None else metadata / "model_observation_events.jsonl"
    summary_path = _summary_path_for(target)
    micro_rows = _read_jsonl_if_exists(metadata / "micro_segments.jsonl")

    events: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    for source_type, file_name in MODEL_INPUT_FILES.items():
        paths = _input_paths(metadata, source_type)
        rows_by_path = [(path, _read_jsonl_if_exists(path)) for path in paths]
        input_counts[source_type] = sum(len(rows) for _path, rows in rows_by_path)
        converter = _converter_for(source_type)
        source_index = 1
        for path, rows in rows_by_path:
            for row in rows:
                if not isinstance(row, Mapping):
                    source_index += 1
                    continue
                event = converter(row, source_index, path, micro_rows)
                if event is not None:
                    events.append(event)
                source_index += 1

    events.sort(key=lambda row: (str(row.get("global_start_time") or ""), str(row.get("observation_id") or "")))
    write_jsonl(target, [_stable(row) for row in events])

    source_counts = Counter(str(row.get("source_type") or "unknown") for row in events)
    event_counts = Counter(str(row.get("event_type") or "unknown") for row in events)
    level_counts = Counter(str(row.get("confirmation_level") or "unknown") for row in events)
    summary = {
        "session_id": _first_text(
            [
                *(row.get("session_id") for row in events if isinstance(row, Mapping)),
                *(row.get("session_id") for row in micro_rows if isinstance(row, Mapping)),
                _manifest_session_id(session),
            ]
        ),
        "event_count": len(events),
        "source_type_counts": dict(sorted(source_counts.items())),
        "event_type_counts": dict(sorted(event_counts.items())),
        "confirmation_level_counts": dict(sorted(level_counts.items())),
        "input_counts": input_counts,
        "model_observation_events": str(target),
        "summary_path": str(summary_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_model_observation_events(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return read_jsonl(source) if source.exists() else []


def load_or_build_model_observation_events(session_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = metadata / "model_observation_events.jsonl"
    raw_inputs_exist = any(path.exists() for source_type in MODEL_INPUT_FILES for path in _input_paths(metadata, source_type))
    if raw_inputs_exist or not target.exists():
        summary = build_model_observation_events(session)
        return load_model_observation_events(target), summary
    rows = load_model_observation_events(target)
    summary = {
        "event_count": len(rows),
        "input_counts": {source_type: 0 for source_type in MODEL_INPUT_FILES},
        "model_observation_events": str(target),
        "loaded_existing": True,
    }
    return rows, summary


def model_observation_input_paths(session_dir: str | Path) -> dict[str, Path]:
    metadata = Path(session_dir) / "metadata"
    return {source_type: metadata / file_name for source_type, file_name in MODEL_INPUT_FILES.items()}


def _input_paths(metadata: Path, source_type: str) -> list[Path]:
    names = MODEL_INPUT_FILE_ALIASES.get(source_type, (MODEL_INPUT_FILES[source_type],))
    return [metadata / name for name in names]


def _summary_path_for(target: Path) -> Path:
    if target.name == "model_observation_events.jsonl":
        return target.with_name("model_observation_events_summary.json")
    return target.with_name(f"{target.stem}_summary.json")


def _converter_for(source_type: str):
    return {
        "liquid_segmentation": _liquid_observation,
        "equipment_panel_state": _equipment_observation,
        "container_state": _container_observation,
        "object_track": _track_observation,
    }[source_type]


def _liquid_observation(
    row: Mapping[str, Any],
    row_index: int,
    source_path: Path,
    micro_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _base_observation(row, row_index, source_path, "liquid_segmentation", micro_rows)
    measurement = _measurement(
        row,
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
            "mask_id",
            "bbox",
        ),
    )
    has_level = any(key in measurement for key in ("liquid_level_y_norm", "meniscus_y_norm", "level_y_norm", "volume_ml", "volume_ul"))
    event_type = str(row.get("event_type") or ("liquid_level_measured" if has_level else "liquid_flow_observed"))
    confirmation_level = _confirmation_level(row, "measured" if has_level else "confirmed")
    reasons = _strings(row.get("evidence_reasons")) or [
        "external liquid_segmentation model output",
        "segmentation mask, stream, or level metadata provided",
    ]
    return {
        **base,
        "observation_type": "liquid_segmentation",
        "event_type": event_type,
        "object_label": _first_text_value(
            row,
            "object_label",
            "container_label",
            "container_id",
            "target_object",
            "label",
            "class_name",
            default="liquid",
        ),
        "track_id": _first_non_empty(row.get("track_id"), row.get("object_track_id")),
        "state": _first_non_empty(row.get("state"), row.get("liquid_state"), row.get("stream_state")),
        "measurement": measurement,
        "confirmation_level": confirmation_level,
        "confidence": _confidence(row, 0.82 if confirmation_level in {"confirmed", "measured"} else 0.5),
        "evidence_reasons": _dedupe(reasons),
        "limitations": _strings(row.get("limitations")),
        "metrics": _metrics(row, row_index, source_path, measurement),
        "asset_refs": _asset_refs(row),
        "payload": _payload(row),
    }


def _equipment_observation(
    row: Mapping[str, Any],
    row_index: int,
    source_path: Path,
    micro_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _base_observation(row, row_index, source_path, "equipment_panel_state", micro_rows)
    measurement = _measurement(
        row,
        (
            "display_text",
            "ocr_text",
            "ocr_confidence",
            "readout",
            "reading",
            "value",
            "unit",
            "control_type",
            "control_label",
            "button_state",
            "button_label",
            "button_pressed",
            "knob_state",
            "knob_label",
            "knob_angle_deg",
            "knob_position",
            "switch_state",
            "switch_label",
            "switch_position",
            "switch_on",
            "toggle_state",
            "panel_state",
            "control_state",
            "state",
            "bbox",
            "panel_bbox",
            "control_bbox",
        ),
    )
    is_numeric_or_ocr = any(key in measurement for key in ("display_text", "ocr_text", "readout", "reading", "value"))
    has_control_state = any(
        key in measurement
        for key in (
            "button_state",
            "button_pressed",
            "knob_state",
            "knob_angle_deg",
            "knob_position",
            "switch_state",
            "switch_position",
            "switch_on",
            "toggle_state",
            "control_state",
        )
    )
    default_event_type = "equipment_panel_state_measured" if is_numeric_or_ocr else "equipment_control_state_confirmed" if has_control_state else "equipment_panel_state_candidate"
    event_type = str(row.get("event_type") or default_event_type)
    confirmation_level = _confirmation_level(row, "measured" if is_numeric_or_ocr else "confirmed" if has_control_state else "candidate")
    return {
        **base,
        "observation_type": "equipment_panel_state",
        "event_type": event_type,
        "object_label": _first_text_value(row, "equipment_label", "equipment_id", "panel_label", "object_label", "label", default="equipment_panel"),
        "track_id": _first_non_empty(row.get("track_id"), row.get("panel_track_id")),
        "state": _first_non_empty(
            row.get("panel_state"),
            row.get("control_state"),
            row.get("state"),
            row.get("button_state"),
            row.get("knob_state"),
            row.get("switch_state"),
            row.get("switch_position"),
            row.get("toggle_state"),
        ),
        "measurement": measurement,
        "confirmation_level": confirmation_level,
        "confidence": _confidence(row, 0.82 if confirmation_level in {"confirmed", "measured"} else 0.5),
        "evidence_reasons": _dedupe(
            _strings(row.get("evidence_reasons"))
            or [
                "external equipment panel OCR/control-state model output"
                if is_numeric_or_ocr or has_control_state
                else "equipment panel candidate row without OCR/control-state measurement",
            ]
        ),
        "limitations": _strings(row.get("limitations")),
        "metrics": _metrics(row, row_index, source_path, measurement),
        "asset_refs": _asset_refs(row),
        "payload": _payload(row),
    }


def _container_observation(
    row: Mapping[str, Any],
    row_index: int,
    source_path: Path,
    micro_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _base_observation(row, row_index, source_path, "container_state", micro_rows)
    measurement = _measurement(
        row,
        (
            "state",
            "before_state",
            "after_state",
            "lid_state",
            "cap_state",
            "open_closed_state",
            "open_close_state",
            "color_before",
            "color_after",
            "dominant_color_before",
            "dominant_color_after",
            "color_delta",
            "liquid_level_y_norm",
            "liquid_level_before",
            "liquid_level_after",
            "meniscus_y_norm",
            "volume_ml",
            "volume_ul",
            "event",
            "bbox",
        ),
    )
    has_liquid_measurement = any(
        key in measurement
        for key in ("liquid_level_y_norm", "liquid_level_before", "liquid_level_after", "meniscus_y_norm", "volume_ml", "volume_ul")
    )
    has_color_measurement = any(key in measurement for key in ("color_before", "color_after", "dominant_color_before", "dominant_color_after", "color_delta"))
    state = _first_non_empty(
        row.get("after_state"),
        row.get("open_closed_state"),
        row.get("open_close_state"),
        row.get("lid_state"),
        row.get("cap_state"),
        row.get("state"),
        row.get("event"),
    )
    default_event_type = "liquid_level_measured" if has_liquid_measurement else "container_color_change_confirmed" if has_color_measurement else "container_state_confirmed"
    confirmation_level = _confirmation_level(row, "measured" if has_liquid_measurement else "confirmed")
    return {
        **base,
        "observation_type": "container_state",
        "event_type": str(row.get("event_type") or default_event_type),
        "object_label": _first_text_value(row, "container_label", "container_id", "object_label", "label", default="container"),
        "track_id": _first_non_empty(row.get("track_id"), row.get("container_track_id")),
        "state": state,
        "measurement": measurement,
        "confirmation_level": confirmation_level,
        "confidence": _confidence(row, 0.82 if confirmation_level in {"confirmed", "measured"} else 0.5),
        "evidence_reasons": _dedupe(
            _strings(row.get("evidence_reasons"))
            or [
                "external container liquid-level model output"
                if has_liquid_measurement
                else "external container color-change model output"
                if has_color_measurement
                else "external container open/closed state model output",
            ]
        ),
        "limitations": _strings(row.get("limitations")),
        "metrics": _metrics(row, row_index, source_path, measurement),
        "asset_refs": _asset_refs(row),
        "payload": _payload(row),
    }


def _track_observation(
    row: Mapping[str, Any],
    row_index: int,
    source_path: Path,
    micro_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    base = _base_observation(row, row_index, source_path, "object_track", micro_rows)
    track_points = _track_points(row)
    track_metrics = _track_metrics(track_points)
    measurement = _measurement(
        row,
        (
            "track_id",
            "object_track_id",
            "path_length_px",
            "displacement_px",
            "movement_score",
            "identity_confidence",
            "bbox",
        ),
    )
    measurement.update(track_metrics)
    confirmation_level = _confirmation_level(row, "measured")
    displacement = _as_float(measurement.get("displacement_px"))
    path_length = _as_float(measurement.get("path_length_px"))
    moving = max(displacement or 0.0, path_length or 0.0) > 0.0
    return {
        **base,
        "observation_type": "object_track",
        "event_type": str(row.get("event_type") or ("object_movement_measured" if moving else "object_track_observed")),
        "object_label": _first_text_value(row, "object_label", "label", "class_name", "object_name", default="object"),
        "track_id": _first_non_empty(row.get("track_id"), row.get("object_track_id"), row.get("tracklet_id")),
        "state": _first_non_empty(row.get("state"), row.get("motion_state")),
        "measurement": measurement,
        "confirmation_level": confirmation_level,
        "confidence": _confidence(row, 0.82 if confirmation_level in {"confirmed", "measured"} else 0.5),
        "evidence_reasons": _dedupe(_strings(row.get("evidence_reasons")) or ["external object_tracks model output"]),
        "limitations": _strings(row.get("limitations")),
        "metrics": {**_metrics(row, row_index, source_path, measurement), "track_point_count": len(track_points)},
        "asset_refs": _asset_refs(row),
        "payload": {**_payload(row), "normalized_track_points": track_points[:50]},
    }


def _base_observation(
    row: Mapping[str, Any],
    row_index: int,
    source_path: Path,
    source_type: str,
    micro_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    micro = _matching_micro(row, micro_rows)
    start_sec = _first_float(row, "start_sec", "start_time_sec", "local_start_sec", "session_start_sec", "time_sec", "local_time_sec", "session_time_sec")
    end_sec = _first_float(row, "end_sec", "end_time_sec", "local_end_sec", "session_end_sec")
    duration_sec = _first_float(row, "duration_sec")
    if end_sec is None and start_sec is not None and duration_sec is not None:
        end_sec = start_sec + duration_sec
    if end_sec is None and start_sec is not None:
        end_sec = start_sec
    if micro:
        start_sec = start_sec if start_sec is not None else _as_float(micro.get("start_sec"))
        end_sec = end_sec if end_sec is not None else _as_float(micro.get("end_sec"))
    session_id = _first_non_empty(row.get("session_id"), micro.get("session_id") if micro else None)
    segment_id = _first_non_empty(row.get("segment_id"), row.get("parent_segment_id"), micro.get("parent_segment_id") if micro else None, micro.get("segment_id") if micro else None)
    micro_id = _first_non_empty(row.get("micro_segment_id"), micro.get("micro_segment_id") if micro else None)
    global_start = _first_non_empty(
        row.get("global_start_time"),
        row.get("start_global_time"),
        row.get("global_time"),
        micro.get("global_start_time") if micro else None,
    )
    global_end = _first_non_empty(
        row.get("global_end_time"),
        row.get("end_global_time"),
        row.get("global_time"),
        micro.get("global_end_time") if micro else None,
    )
    object_key = _first_non_empty(
        row.get("track_id"),
        row.get("object_track_id"),
        row.get("container_id"),
        row.get("equipment_id"),
        row.get("object_label"),
        row.get("label"),
    )
    observation_id = _first_non_empty(
        row.get("observation_id"),
        row.get("event_id"),
        row.get("id"),
        f"model_obs:{source_type}:{micro_id or segment_id or 'unassigned'}:{object_key or 'item'}:{row_index:06d}",
    )
    return {
        "observation_id": str(observation_id),
        "session_id": session_id,
        "segment_id": segment_id,
        "micro_segment_id": micro_id,
        "source_file": str(source_path),
        "source_type": source_type,
        "global_start_time": global_start,
        "global_end_time": global_end,
        "start_sec": round(start_sec, 4) if start_sec is not None else None,
        "end_sec": round(end_sec, 4) if end_sec is not None else None,
        "view": _first_non_empty(row.get("view"), row.get("source_view"), row.get("camera"), row.get("camera_id")),
    }


def _matching_micro(row: Mapping[str, Any], micro_rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    micro_id = _first_non_empty(row.get("micro_segment_id"), row.get("source_id"))
    if micro_id:
        for micro in micro_rows:
            if str(micro.get("micro_segment_id") or "") == str(micro_id):
                return micro
    start_sec = _first_float(row, "start_sec", "start_time_sec", "local_start_sec", "session_start_sec", "time_sec", "local_time_sec", "session_time_sec")
    end_sec = _first_float(row, "end_sec", "end_time_sec", "local_end_sec", "session_end_sec")
    if end_sec is None:
        duration_sec = _first_float(row, "duration_sec")
        if start_sec is not None and duration_sec is not None:
            end_sec = start_sec + duration_sec
        else:
            end_sec = start_sec
    if start_sec is None or end_sec is None:
        return None
    best: Mapping[str, Any] | None = None
    best_score = 0.0
    for micro in micro_rows:
        micro_start = _as_float(micro.get("start_sec"))
        micro_end = _as_float(micro.get("end_sec"))
        if micro_start is None or micro_end is None:
            continue
        overlap = max(0.0, min(end_sec, micro_end) - max(start_sec, micro_start))
        if overlap <= 0.0 and micro_start <= start_sec <= micro_end:
            overlap = 0.0001
        if overlap > best_score:
            best_score = overlap
            best = micro
    return best


def _measurement(row: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in keys:
        if key in row and row.get(key) is not None:
            result[key] = row.get(key)
    nested_metrics = row.get("measurement")
    if isinstance(nested_metrics, Mapping):
        for key, value in nested_metrics.items():
            if value is not None:
                result.setdefault(str(key), value)
    return result


def _metrics(row: Mapping[str, Any], row_index: int, source_path: Path, measurement: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(row.get("metrics") or {}) if isinstance(row.get("metrics"), Mapping) else {}
    metrics.update(
        {
            "source_row_index": row_index,
            "source_file_name": source_path.name,
            "model_name": _first_non_empty(row.get("model_name"), row.get("model"), row.get("model_id")),
            "measurement_keys": sorted(str(key) for key in measurement),
        }
    )
    return metrics


def _track_points(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_points = row.get("points") or row.get("trajectory") or row.get("detections") or row.get("track")
    points: list[dict[str, Any]] = []
    if isinstance(raw_points, list):
        for index, point in enumerate(raw_points):
            if not isinstance(point, Mapping):
                continue
            bbox = _bbox(point.get("bbox") or point.get("box"))
            normalized = {
                "index": index,
                "time_sec": _first_float(point, "time_sec", "local_time_sec", "session_time_sec", "timestamp_sec"),
                "global_time": _first_non_empty(point.get("global_time"), point.get("global_start_time")),
                "bbox": bbox,
                "confidence": _confidence(point, None),
            }
            if bbox is not None:
                normalized["center"] = _center(bbox)
            points.append(normalized)
    if points:
        return points
    bbox = _bbox(row.get("bbox") or row.get("box"))
    if bbox is None:
        return []
    return [
        {
            "index": 0,
            "time_sec": _first_float(row, "time_sec", "local_time_sec", "session_time_sec", "start_sec"),
            "global_time": _first_non_empty(row.get("global_time"), row.get("global_start_time")),
            "bbox": bbox,
            "center": _center(bbox),
            "confidence": _confidence(row, None),
        }
    ]


def _track_metrics(points: list[dict[str, Any]]) -> dict[str, Any]:
    centers = [point.get("center") for point in points if isinstance(point.get("center"), list) and len(point["center"]) >= 2]
    result: dict[str, Any] = {"point_count": len(points)}
    if len(centers) >= 2:
        first = centers[0]
        last = centers[-1]
        displacement = math.dist((float(first[0]), float(first[1])), (float(last[0]), float(last[1])))
        path_length = 0.0
        for prev, current in zip(centers, centers[1:]):
            path_length += math.dist((float(prev[0]), float(prev[1])), (float(current[0]), float(current[1])))
        result.update(
            {
                "first_center": [round(float(first[0]), 4), round(float(first[1]), 4)],
                "last_center": [round(float(last[0]), 4), round(float(last[1]), 4)],
                "displacement_px": round(displacement, 4),
                "path_length_px": round(path_length, 4),
            }
        )
    return result


def _asset_refs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    raw_refs = row.get("asset_refs")
    if isinstance(raw_refs, list):
        for ref in raw_refs:
            if isinstance(ref, Mapping):
                refs.append(dict(ref))
    for key, asset_type in (
        ("image_path", "frame"),
        ("source_image_path", "frame"),
        ("frame_path", "frame"),
        ("mask_path", "segmentation_mask"),
        ("clip_path", "video_clip"),
    ):
        value = row.get(key)
        if value:
            refs.append({"asset_type": asset_type, "rel": key, "path": value})
    return _dedupe_refs(refs)


def _payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"source": "model_observations", "source_row": dict(row)}


def _bbox(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            items = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
        elif all(key in value for key in ("x", "y", "w", "h")):
            x = _as_float(value.get("x"))
            y = _as_float(value.get("y"))
            w = _as_float(value.get("w"))
            h = _as_float(value.get("h"))
            if None in (x, y, w, h):
                return None
            items = [x, y, x + w, y + h]
        else:
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        items = list(value[:4])
    else:
        return None
    try:
        return [float(items[0]), float(items[1]), float(items[2]), float(items[3])]
    except (TypeError, ValueError):
        return None


def _center(bbox: list[float]) -> list[float]:
    return [round((bbox[0] + bbox[2]) / 2.0, 4), round((bbox[1] + bbox[3]) / 2.0, 4)]


def _confirmation_level(row: Mapping[str, Any], default: str) -> str:
    value = _first_non_empty(row.get("confirmation_level"), row.get("visual_confirmation_level"), row.get("status"))
    if not value:
        return default
    text = str(value).strip().lower()
    if text in {"measured", "measurement", "quantified"}:
        return "measured"
    if text in {"confirmed", "detected", "observed", "true", "verified"}:
        return "confirmed"
    if "candidate" in text or "review" in text:
        return "candidate"
    return text


def _confidence(row: Mapping[str, Any], default: float | None) -> float | None:
    value = _first_non_empty(row.get("confidence"), row.get("score"), row.get("probability"), row.get("model_confidence"))
    if value is None:
        return default
    numeric = _as_float(value)
    if numeric is None:
        return default
    return round(max(0.0, min(1.0, numeric)), 4)


def _first_float(row: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            value = _as_float(row.get(key))
            if value is not None:
                return value
    return None


def _first_text_value(row: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return default


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)] if str(value) else []


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _dedupe_refs(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for ref in refs:
        key = (str(ref.get("asset_id") or ""), str(ref.get("asset_type") or ""), str(ref.get("path") or ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(ref))
    return output


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value) != "":
            return value
    return None


def _first_text(values: Iterable[Any]) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _manifest_session_id(session: Path) -> str:
    manifest = session / "manifest.json"
    if not manifest.exists():
        return ""
    try:
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("session_id") or "")


def _stable(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {field: row.get(field) for field in MODEL_OBSERVATION_FIELDS}
    if output.get("confidence_reasons") is None:
        output["confidence_reasons"] = row.get("evidence_reasons") or []
    if output.get("evidence_refs") is None:
        output["evidence_refs"] = row.get("asset_refs") or []
    return output


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


__all__ = [
    "MODEL_INPUT_FILES",
    "MODEL_INPUT_FILE_ALIASES",
    "MODEL_OBSERVATION_FIELDS",
    "build_model_observation_events",
    "load_model_observation_events",
    "load_or_build_model_observation_events",
    "model_observation_input_paths",
]
