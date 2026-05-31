from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl, write_jsonl
from .yolo_detector import canonical_yolo_label, find_hand_object_interactions


GENERATED_SOURCE = "lab_model_signal_inputs"

CONTAINER_LABELS = {
    "beaker",
    "bottle",
    "container",
    "cup",
    "jar",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "tube",
    "vial",
}
CAP_LABELS = {"cap", "lid", "tube_cap", "bottle_cap", "container_cap"}
EQUIPMENT_LABELS = {"balance", "equipment_panel", "panel", "display", "screen", "button", "knob", "switch"}
PIPETTE_LABELS = {"pipette", "pipette_tip"}
HAND_LABELS = {"hand", "gloved_hand"}
PHASE_ORDER = {"start": 0.0, "middle": 0.5, "mid": 0.5, "end": 1.0}


def build_lab_model_signal_inputs(
    session_dir: str | Path,
    *,
    min_confidence: float = 0.25,
    output_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Convert existing lab YOLO/model signals into canonical observation input files.

    The generated rows are conservative candidates. They bridge available lab model
    detections into the downstream evidence pipeline without pretending that YOLO
    boxes are OCR, liquid segmentation, or direct open/closed state labels.
    """

    session = Path(session_dir)
    metadata = session / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    target_summary = Path(output_summary_path) if output_summary_path else metadata / "lab_model_signal_inputs_summary.json"

    micro_rows = _read_jsonl(metadata / "micro_segments.jsonl")
    segment_rows = _read_jsonl(metadata / "key_action_segments.jsonl")
    yolo_rows, source_path, source_kind = _load_yolo_rows(session)
    model_path = _model_path(session)
    session_id = _session_id(session, micro_rows, segment_rows)

    normalized_rows = _normalize_yolo_rows(
        yolo_rows,
        micro_rows=micro_rows,
        segment_rows=segment_rows,
        session_id=session_id,
        source_path=source_path,
        source_kind=source_kind,
        min_confidence=min_confidence,
    )
    candidate_rows = _candidate_rows(normalized_rows, model_path=model_path)

    output_files = {
        "liquid_transfer_candidate": metadata / "liquid_segmentation.jsonl",
        "equipment_panel_candidate": metadata / "equipment_panel_states.jsonl",
        "container_state_candidate": metadata / "container_state_events.jsonl",
    }
    written_counts: dict[str, int] = {}
    total_preserved = 0
    for candidate_type, path in output_files.items():
        generated = candidate_rows.get(candidate_type, [])
        preserved = _preserved_existing_rows(path)
        total_preserved += len(preserved)
        write_jsonl(path, [_stable(row) for row in [*preserved, *generated]])
        written_counts[candidate_type] = len(generated)

    event_counts = Counter(
        str(row.get("event_type") or "unknown")
        for rows in candidate_rows.values()
        for row in rows
    )
    summary = {
        "metadata_version": "key_action_lab_model_signal_inputs.v1",
        "session_id": session_id,
        "source_kind": source_kind,
        "source_path": str(source_path) if source_path else None,
        "model_path": model_path,
        "normalized_frame_count": len(normalized_rows),
        "generated_candidate_count": sum(written_counts.values()),
        "preserved_external_row_count": total_preserved,
        "candidate_counts": dict(sorted(written_counts.items())),
        "event_type_counts": dict(sorted(event_counts.items())),
        "settings": {"min_confidence": min_confidence},
        "outputs": {key: str(path) for key, path in output_files.items()},
        "summary_path": str(target_summary),
    }
    target_summary.parent.mkdir(parents=True, exist_ok=True)
    target_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _normalize_yolo_rows(
    rows: list[Mapping[str, Any]],
    *,
    micro_rows: list[Mapping[str, Any]],
    segment_rows: list[Mapping[str, Any]],
    session_id: str,
    source_path: Path | None,
    source_kind: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    micros_by_id = {str(row.get("micro_segment_id") or ""): row for row in micro_rows if row.get("micro_segment_id")}
    segments_by_id = {str(row.get("segment_id") or ""): row for row in segment_rows if row.get("segment_id")}
    normalized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        micro = _micro_for_row(row, micro_rows)
        segment = _segment_for_row(row, segment_rows)
        if micro is None and source_kind == "keyframe_detections":
            segment_id = str(row.get("segment_id") or "")
            segment = segments_by_id.get(segment_id)
        micro_id = str(micro.get("micro_segment_id") or "") if isinstance(micro, Mapping) else ""
        segment_id = (
            str(micro.get("parent_segment_id") or micro.get("segment_id") or "")
            if isinstance(micro, Mapping)
            else str(row.get("segment_id") or (segment or {}).get("segment_id") or "")
        )
        detections = [
            item
            for item in (_normalize_detection(detection) for detection in _detections(row))
            if item and float(item.get("confidence") or 0.0) >= min_confidence
        ]
        if not detections:
            continue
        interactions = _interactions(row, detections)
        start_sec = _row_time(row)
        end_sec = start_sec
        global_time = row.get("global_time")
        if isinstance(micro, Mapping):
            start_sec = _float(micro.get("start_sec")) if start_sec is None else start_sec
            end_sec = _float(micro.get("end_sec"))
            global_time = global_time or micro.get("global_start_time")
        elif isinstance(segment, Mapping):
            start_sec = _phase_time(segment, str(row.get("phase") or "")) if start_sec is None else start_sec
            end_sec = _float(segment.get("end_sec"))
            global_time = global_time or _phase_global_time(segment, str(row.get("phase") or ""))
        normalized.append(
            {
                "session_id": session_id,
                "segment_id": segment_id or None,
                "micro_segment_id": micro_id or None,
                "start_sec": start_sec,
                "end_sec": end_sec if end_sec is not None else start_sec,
                "global_time": global_time,
                "view": str(row.get("source_view") or row.get("view") or row.get("camera") or "unknown"),
                "detections": detections,
                "interactions": interactions,
                "frame_index": row.get("frame_index"),
                "sample_index": row.get("sample_index"),
                "image_path": row.get("image_path") or row.get("source_image_path") or row.get("frame_path"),
                "source_row_index": row_index,
                "source_file": str(source_path) if source_path else None,
                "source_kind": source_kind,
                "micro": dict(micros_by_id.get(micro_id, {})) if micro_id else {},
            }
        )
    return normalized


def _candidate_rows(rows: list[Mapping[str, Any]], *, model_path: str | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[tuple[str, str, str, str], list[dict[str, Any]]]] = {
        "liquid_transfer_candidate": defaultdict(list),
        "equipment_panel_candidate": defaultdict(list),
        "container_state_candidate": defaultdict(list),
    }
    for row in rows:
        detections = [item for item in row.get("detections") or [] if isinstance(item, Mapping)]
        interactions = [item for item in row.get("interactions") or [] if isinstance(item, Mapping)]
        containers = [item for item in detections if _label(item) in CONTAINER_LABELS]
        caps = [item for item in detections if _label(item) in CAP_LABELS]
        equipment = [item for item in detections if _label(item) in EQUIPMENT_LABELS]
        pipettes = [item for item in detections if _label(item) in PIPETTE_LABELS]

        for equipment_detection in equipment:
            interaction = _best_interaction_for(interactions, {_label(equipment_detection)})
            if not interaction:
                continue
            _add_grouped(
                grouped["equipment_panel_candidate"],
                row,
                object_label=_label(equipment_detection),
                event_type="equipment_panel_interaction_candidate",
                state="interaction_candidate",
                confidence=_combined_confidence(equipment_detection, interaction),
                measurement={
                    "bbox": equipment_detection.get("bbox"),
                    "interaction_score": interaction.get("score"),
                    "interaction": interaction,
                },
                model_path=model_path,
                reasons=[
                    "Lab YOLO model detected hand-equipment interaction",
                    "candidate only; no OCR/readout/button-state model output attached",
                ],
                limitations=["no panel OCR, button state, or knob angle measurement in source row"],
            )

        if caps and containers:
            cap = max(caps, key=lambda item: float(item.get("confidence") or 0.0))
            container, distance = _nearest_detection(cap, containers)
            state = "cap_visible_near_container" if distance is not None and distance <= _distance_threshold(container) else "cap_visible"
            _add_grouped(
                grouped["container_state_candidate"],
                row,
                object_label=_label(container),
                event_type="container_state_candidate",
                state=state,
                confidence=_mean_confidence([cap, container], default=0.5),
                measurement={
                    "cap_label": _label(cap),
                    "container_label": _label(container),
                    "cap_bbox": cap.get("bbox"),
                    "container_bbox": container.get("bbox"),
                    "cap_container_distance_px": round(distance, 4) if distance is not None else None,
                },
                model_path=model_path,
                reasons=[
                    "Lab YOLO model detected cap/lid and container in the same evidence frame",
                    "candidate state signal from cap-container geometry",
                ],
                limitations=["does not directly classify open/closed state"],
            )

        if pipettes and containers and _pipette_transfer_signal(interactions, pipettes, containers):
            pipette = max(pipettes, key=lambda item: float(item.get("confidence") or 0.0))
            container, distance = _nearest_detection(pipette, containers)
            _add_grouped(
                grouped["liquid_transfer_candidate"],
                row,
                object_label=_label(container),
                event_type="liquid_flow_candidate",
                state="transfer_candidate",
                confidence=_mean_confidence([pipette, container], default=0.5),
                measurement={
                    "tool_label": _label(pipette),
                    "container_label": _label(container),
                    "tool_bbox": pipette.get("bbox"),
                    "container_bbox": container.get("bbox"),
                    "tool_container_distance_px": round(distance, 4) if distance is not None else None,
                },
                model_path=model_path,
                reasons=[
                    "Lab YOLO model detected pipette/tool with target container during interaction",
                    "candidate liquid-transfer signal from object/contact geometry",
                ],
                limitations=["no liquid mask, flow segmentation, or liquid-level measurement in source row"],
            )

    return {name: [_merge_group(name, items) for items in groups.values()] for name, groups in grouped.items()}


def _add_grouped(
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]],
    row: Mapping[str, Any],
    *,
    object_label: str,
    event_type: str,
    state: str,
    confidence: float,
    measurement: Mapping[str, Any],
    model_path: str | None,
    reasons: list[str],
    limitations: list[str],
) -> None:
    segment_id = str(row.get("segment_id") or "")
    micro_id = str(row.get("micro_segment_id") or "")
    view = str(row.get("view") or "unknown")
    key = (micro_id or segment_id or "unassigned", view, object_label, event_type)
    groups[key].append(
        {
            "event_id": f"{GENERATED_SOURCE}:{event_type}:{key[0]}:{view}:{object_label}",
            "session_id": row.get("session_id"),
            "segment_id": row.get("segment_id"),
            "micro_segment_id": row.get("micro_segment_id"),
            "start_sec": row.get("start_sec"),
            "end_sec": row.get("end_sec"),
            "global_start_time": row.get("global_time"),
            "global_end_time": row.get("global_time"),
            "view": view,
            "object_label": object_label,
            "state": state,
            "event_type": event_type,
            "confirmation_level": "candidate",
            "confidence": round(max(0.0, min(1.0, confidence)), 4),
            "measurement": {key: value for key, value in measurement.items() if value is not None},
            "model_name": Path(model_path).name if model_path else None,
            "model_path": model_path,
            "source_file": row.get("source_file"),
            "evidence_reasons": reasons,
            "limitations": limitations,
            "metrics": {
                "source_row_index": row.get("source_row_index"),
                "source_kind": row.get("source_kind"),
                "frame_index": row.get("frame_index"),
                "sample_index": row.get("sample_index"),
            },
            "asset_refs": _asset_refs(row),
            "payload": {
                "source": GENERATED_SOURCE,
                "source_row_index": row.get("source_row_index"),
                "source_kind": row.get("source_kind"),
                "source_file": row.get("source_file"),
            },
        }
    )


def _merge_group(candidate_type: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (_none_last_float(row.get("start_sec")), str(row.get("event_id") or "")))
    first = ordered[0]
    last = ordered[-1]
    confidence = mean(float(row.get("confidence") or 0.0) for row in ordered)
    measurement = dict(first.get("measurement") or {})
    measurement["candidate_frame_count"] = len(ordered)
    measurement["candidate_type"] = candidate_type
    metrics = dict(first.get("metrics") or {})
    metrics["candidate_frame_count"] = len(ordered)
    metrics["source_row_indices"] = [row.get("metrics", {}).get("source_row_index") for row in ordered if isinstance(row.get("metrics"), Mapping)]
    return {
        **first,
        "end_sec": last.get("end_sec") if last.get("end_sec") is not None else first.get("end_sec"),
        "global_end_time": last.get("global_end_time") or first.get("global_end_time"),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "measurement": measurement,
        "metrics": metrics,
        "asset_refs": _dedupe_refs(ref for row in ordered for ref in row.get("asset_refs") or []),
    }


def _load_yolo_rows(session: Path) -> tuple[list[dict[str, Any]], Path | None, str]:
    candidates = [
        (session / "cv_outputs" / "yolo_micro_frame_rows.jsonl", "frame_rows"),
        (session / "cv_outputs" / "yolo_frame_rows.jsonl", "frame_rows"),
        (session / "metadata" / "yolo_frame_rows.jsonl", "frame_rows"),
        (session / "metadata" / "yolo_detections.jsonl", "keyframe_detections"),
    ]
    for path, source_kind in candidates:
        if path.exists():
            return read_jsonl(path), path, source_kind
    return [], None, "missing"


def _model_path(session: Path) -> str | None:
    for path in (
        session / "metadata" / "yolo_micro_scan_summary.json",
        session / "metadata" / "yolo_scan_summary.json",
        session / "metadata" / "yolo_summary.json",
        session / "metadata" / "model_inventory.json",
    ):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("model_path"):
            return str(data["model_path"])
        primary = data.get("primary_model")
        if isinstance(primary, Mapping) and primary.get("path"):
            return str(primary["path"])
    return None


def _detections(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    detections = row.get("detections")
    return [item for item in detections if isinstance(item, Mapping)] if isinstance(detections, list) else []


def _normalize_detection(detection: Mapping[str, Any]) -> dict[str, Any] | None:
    label = canonical_yolo_label(detection.get("label") or detection.get("object_label") or detection.get("raw_label") or detection.get("name"))
    bbox = _bbox(detection.get("bbox") or detection.get("box") or detection.get("xyxy"))
    if not label or bbox is None:
        return None
    return {
        "label": label,
        "raw_label": str(detection.get("raw_label") or detection.get("label") or label),
        "class_id": detection.get("class_id") or detection.get("cls"),
        "confidence": _confidence(detection, 0.0),
        "bbox": bbox,
    }


def _interactions(row: Mapping[str, Any], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = row.get("hand_object_interactions") or row.get("interactions")
    interactions = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            interaction = dict(item)
            if interaction.get("object_label"):
                interaction["object_label"] = canonical_yolo_label(interaction.get("object_label"))
            interactions.append(interaction)
    if interactions:
        return interactions
    frame_width = _int_or_none(row.get("frame_width") or row.get("width"))
    frame_height = _int_or_none(row.get("frame_height") or row.get("height"))
    return find_hand_object_interactions(detections, frame_width=frame_width, frame_height=frame_height)


def _best_interaction_for(interactions: list[Mapping[str, Any]], labels: set[str]) -> Mapping[str, Any] | None:
    matching = [
        interaction
        for interaction in interactions
        if canonical_yolo_label(interaction.get("object_label") or interaction.get("object_name")) in labels
    ]
    if not matching:
        return None
    return max(matching, key=lambda item: float(item.get("score") or item.get("confidence") or 0.0))


def _pipette_transfer_signal(
    interactions: list[Mapping[str, Any]],
    pipettes: list[Mapping[str, Any]],
    containers: list[Mapping[str, Any]],
) -> bool:
    interaction_labels = {
        canonical_yolo_label(item.get("object_label") or item.get("object_name"))
        for item in interactions
        if isinstance(item, Mapping)
    }
    if interaction_labels & PIPETTE_LABELS:
        return True
    if interaction_labels & CONTAINER_LABELS and pipettes:
        return True
    pipette = max(pipettes, key=lambda item: float(item.get("confidence") or 0.0))
    container, distance = _nearest_detection(pipette, containers)
    return bool(container and distance is not None and distance <= _distance_threshold(container))


def _nearest_detection(source: Mapping[str, Any], candidates: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any], float | None]:
    if not candidates:
        return {}, None
    source_bbox = source.get("bbox")
    distances = [
        (candidate, _bbox_distance(source_bbox, candidate.get("bbox")))
        for candidate in candidates
        if candidate.get("bbox")
    ]
    if not distances:
        return candidates[0], None
    return min(distances, key=lambda item: item[1])


def _bbox_distance(box_a: Any, box_b: Any) -> float:
    if not isinstance(box_a, list) or not isinstance(box_b, list) or len(box_a) < 4 or len(box_b) < 4:
        return float("inf")
    ax = (float(box_a[0]) + float(box_a[2])) / 2.0
    ay = (float(box_a[1]) + float(box_a[3])) / 2.0
    bx = (float(box_b[0]) + float(box_b[2])) / 2.0
    by = (float(box_b[1]) + float(box_b[3])) / 2.0
    return math.dist((ax, ay), (bx, by))


def _distance_threshold(detection: Mapping[str, Any]) -> float:
    bbox = detection.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return 64.0
    width = abs(float(bbox[2]) - float(bbox[0]))
    height = abs(float(bbox[3]) - float(bbox[1]))
    return max(40.0, (width * width + height * height) ** 0.5 * 1.5)


def _micro_for_row(row: Mapping[str, Any], micro_rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    time_value = _row_time(row)
    if time_value is None:
        return None
    best: Mapping[str, Any] | None = None
    best_margin = float("inf")
    for micro in micro_rows:
        start = _float(micro.get("start_sec"))
        end = _float(micro.get("end_sec"))
        if start is None or end is None:
            continue
        if start <= time_value <= end:
            return micro
        margin = min(abs(time_value - start), abs(time_value - end))
        if margin < best_margin and margin <= 0.25:
            best = micro
            best_margin = margin
    return best


def _segment_for_row(row: Mapping[str, Any], segment_rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    segment_id = str(row.get("segment_id") or "")
    if segment_id:
        for segment in segment_rows:
            if str(segment.get("segment_id") or "") == segment_id:
                return segment
    time_value = _row_time(row)
    if time_value is None:
        return None
    for segment in segment_rows:
        start = _float(segment.get("start_sec"))
        end = _float(segment.get("end_sec"))
        if start is not None and end is not None and start <= time_value <= end:
            return segment
    return None


def _row_time(row: Mapping[str, Any]) -> float | None:
    return _first_float(row.get("alignment_time_sec"), row.get("session_time_sec"), row.get("local_time_sec"), row.get("time_sec"))


def _phase_time(segment: Mapping[str, Any], phase: str) -> float | None:
    start = _float(segment.get("start_sec"))
    end = _float(segment.get("end_sec"))
    if start is None or end is None:
        return None
    return round(start + (end - start) * PHASE_ORDER.get(phase, 0.5), 4)


def _phase_global_time(segment: Mapping[str, Any], phase: str) -> str | None:
    start = _parse_datetime(segment.get("global_start_time"))
    end = _parse_datetime(segment.get("global_end_time"))
    if start is None or end is None:
        return None
    return (start + (end - start) * PHASE_ORDER.get(phase, 0.5)).isoformat()


def _asset_refs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = []
    if row.get("image_path"):
        refs.append({"asset_type": "frame", "rel": "image_path", "path": row.get("image_path")})
    return refs


def _preserved_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    preserved = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        payload = row.get("payload")
        if isinstance(payload, Mapping) and payload.get("source") == GENERATED_SOURCE:
            continue
        preserved.append(dict(row))
    return preserved


def _session_id(session: Path, micro_rows: list[Mapping[str, Any]], segment_rows: list[Mapping[str, Any]]) -> str:
    for row in [*micro_rows, *segment_rows]:
        if isinstance(row, Mapping) and row.get("session_id"):
            return str(row["session_id"])
    manifest = session / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if data.get("session_id"):
                return str(data["session_id"])
        except (OSError, json.JSONDecodeError):
            pass
    return session.name


def _stable(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _bbox(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            value = [value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2")]
        elif all(key in value for key in ("x", "y", "w", "h")):
            x = _float(value.get("x"))
            y = _float(value.get("y"))
            w = _float(value.get("w"))
            h = _float(value.get("h"))
            if None in (x, y, w, h):
                return None
            value = [x, y, x + w, y + h]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [round(float(item), 4) for item in value[:4]]
    except (TypeError, ValueError):
        return None


def _label(detection: Mapping[str, Any]) -> str:
    return canonical_yolo_label(detection.get("label") or detection.get("object_label"))


def _confidence(row: Mapping[str, Any], default: float) -> float:
    value = row.get("confidence", row.get("score", row.get("probability", row.get("model_confidence"))))
    numeric = _float(value)
    return round(max(0.0, min(1.0, numeric if numeric is not None else default)), 4)


def _combined_confidence(detection: Mapping[str, Any], interaction: Mapping[str, Any]) -> float:
    values = [
        _float(detection.get("confidence")),
        _float(interaction.get("score") or interaction.get("confidence")),
    ]
    values = [item for item in values if item is not None]
    return float(mean(values)) if values else 0.5


def _mean_confidence(rows: Iterable[Mapping[str, Any]], *, default: float) -> float:
    values = [_float(row.get("confidence")) for row in rows]
    values = [item for item in values if item is not None]
    return float(mean(values)) if values else default


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _float(value)
        if numeric is not None:
            return numeric
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _none_last_float(value: Any) -> float:
    numeric = _float(value)
    return numeric if numeric is not None else float("inf")


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _dedupe_refs(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for ref in refs:
        key = (str(ref.get("asset_type") or ""), str(ref.get("rel") or ""), str(ref.get("path") or ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(ref))
    return output


__all__ = ["build_lab_model_signal_inputs"]
