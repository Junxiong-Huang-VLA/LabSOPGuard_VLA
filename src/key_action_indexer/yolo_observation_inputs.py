from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .physical_event_gate import gate_object_move, summarize_gate_decisions
from .schemas import read_jsonl, write_jsonl
from .track_normalizer import track_evidence_from_points


IGNORED_TRACK_LABELS = {
    "hand",
    "gloved_hand",
    "person",
    "paper",
    "lab_coat",
}

PHASE_ORDER = {"start": 0.0, "middle": 0.5, "mid": 0.5, "end": 1.0}


def build_yolo_observation_inputs(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    min_points: int = 2,
    min_motion_px: float = 5.0,
    max_points_per_track: int = 160,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = Path(output_path) if output_path is not None else metadata / "object_tracks.jsonl"
    summary_path = target.with_name("yolo_observation_inputs_summary.json")

    micro_rows = _read_jsonl(metadata / "micro_segments.jsonl")
    segment_rows = _read_jsonl(metadata / "key_action_segments.jsonl")
    yolo_rows, source_path, source_kind = _load_yolo_rows(session)
    model_path = _model_path(session)
    session_id = _session_id(session, micro_rows, segment_rows)

    generated_tracks = _tracks_from_frame_rows(
        yolo_rows,
        micro_rows,
        session_id=session_id,
        source_path=source_path,
        model_path=model_path,
        min_points=min_points,
        min_motion_px=min_motion_px,
        max_points_per_track=max_points_per_track,
    )
    if not generated_tracks and source_kind == "keyframe_detections":
        generated_tracks = _tracks_from_keyframe_detections(
            yolo_rows,
            segment_rows,
            session_id=session_id,
            source_path=source_path,
            model_path=model_path,
            min_points=min_points,
            min_motion_px=min_motion_px,
            max_points_per_track=max_points_per_track,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = _preserved_existing_tracks(target)
    all_tracks = [*existing_rows, *generated_tracks]
    write_jsonl(target, [_stable(row) for row in all_tracks])
    gate_decisions = [
        row["physical_event_gate"]
        for row in generated_tracks
        if isinstance(row.get("physical_event_gate"), dict)
    ]
    if gate_decisions:
        gate_path = metadata / "physical_event_gate_decisions.jsonl"
        rejected_path = metadata / "rejected_physical_event_candidates.jsonl"
        write_jsonl(gate_path, [_stable_gate(row) for row in gate_decisions])
        rejected_rows = []
        for row in generated_tracks:
            gate = row.get("physical_event_gate") if isinstance(row.get("physical_event_gate"), dict) else {}
            if gate.get("status") == "rejected":
                rejected_rows.append(
                    {
                        "candidate_id": row.get("event_id"),
                        "event_type": "object_move",
                        "status": "rejected",
                        "time_start": row.get("start_sec"),
                        "time_end": row.get("end_sec"),
                        "source_view": row.get("view"),
                        "actor_track_id": None,
                        "object_track_ids": [row.get("track_id")] if row.get("track_id") else [],
                        "object_labels": [row.get("object_label")] if row.get("object_label") else [],
                        "reject_reasons": gate.get("reject_reasons") or [],
                        "evidence_detail": gate.get("evidence") or {},
                        "limitations": gate.get("limitations") or row.get("limitations") or [],
                    }
                )
        write_jsonl(rejected_path, [_stable_rejected(row) for row in rejected_rows])

    source_counts = Counter(str(row.get("source_type") or "unknown") for row in all_tracks)
    state_counts = Counter(str(row.get("motion_state") or row.get("state") or "unknown") for row in all_tracks)
    label_counts = Counter(str(row.get("object_label") or "unknown") for row in all_tracks)
    event_counts = Counter(str(row.get("event_type") or "unknown") for row in all_tracks)
    summary = {
        "session_id": session_id,
        "source_kind": source_kind,
        "source_path": str(source_path) if source_path is not None else None,
        "model_path": model_path,
        "generated_track_count": len(generated_tracks),
        "preserved_track_count": len(existing_rows),
        "track_count": len(all_tracks),
        "source_type_counts": dict(sorted(source_counts.items())),
        "motion_state_counts": dict(sorted(state_counts.items())),
        "object_label_counts": dict(sorted(label_counts.items())),
        "event_type_counts": dict(sorted(event_counts.items())),
        "physical_event_gate_summary": summarize_gate_decisions(gate_decisions) if gate_decisions else {},
        "settings": {
            "min_points": min_points,
            "min_motion_px": min_motion_px,
            "max_points_per_track": max_points_per_track,
            "ignored_track_labels": sorted(IGNORED_TRACK_LABELS),
        },
        "object_tracks": str(target),
        "summary_path": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _tracks_from_frame_rows(
    rows: list[Mapping[str, Any]],
    micro_rows: list[Mapping[str, Any]],
    *,
    session_id: str,
    source_path: Path | None,
    model_path: str | None,
    min_points: int,
    min_motion_px: float,
    max_points_per_track: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    micros_by_id = {str(row.get("micro_segment_id") or ""): row for row in micro_rows if row.get("micro_segment_id")}
    for row_index, row in enumerate(rows, start=1):
        micro = _micro_for_frame_row(row, micro_rows)
        if micro is None:
            continue
        micro_id = str(micro.get("micro_segment_id") or "")
        view = str(row.get("view") or row.get("source_view") or row.get("camera") or "unknown")
        row_time = _row_time(row)
        for detection in _detections(row):
            label = _label(detection)
            if not label or label.lower() in IGNORED_TRACK_LABELS:
                continue
            bbox = _bbox(detection.get("bbox") or detection.get("box"))
            if bbox is None:
                continue
            groups[(micro_id, view, label)].append(
                {
                    "index": len(groups[(micro_id, view, label)]),
                    "time_sec": row_time,
                    "global_time": row.get("global_time"),
                    "frame_index": row.get("frame_index"),
                    "sample_index": row.get("sample_index"),
                    "bbox": bbox,
                    "center": _center(bbox),
                    "confidence": _confidence(detection, default=None),
                    "source_row_index": row_index,
                }
            )

    tracks = []
    for (micro_id, view, label), points in sorted(groups.items()):
        if len(points) < min_points:
            continue
        micro = micros_by_id.get(micro_id, {})
        tracks.append(
            _track_row(
                session_id=micro.get("session_id") or session_id,
                segment_id=micro.get("parent_segment_id") or micro.get("segment_id"),
                micro_segment_id=micro_id,
                view=view,
                label=label,
                points=points,
                source_path=source_path,
                model_path=model_path,
                start_sec=_float(micro.get("start_sec")),
                end_sec=_float(micro.get("end_sec")),
                global_start_time=micro.get("global_start_time"),
                global_end_time=micro.get("global_end_time"),
                min_motion_px=min_motion_px,
                max_points=max_points_per_track,
                source_mode="yolo_frame_rows",
            )
        )
    return tracks


def _tracks_from_keyframe_detections(
    rows: list[Mapping[str, Any]],
    segment_rows: list[Mapping[str, Any]],
    *,
    session_id: str,
    source_path: Path | None,
    model_path: str | None,
    min_points: int,
    min_motion_px: float,
    max_points_per_track: int,
) -> list[dict[str, Any]]:
    segments_by_id = {str(row.get("segment_id") or ""): row for row in segment_rows if row.get("segment_id")}
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row_index, row in enumerate(rows, start=1):
        segment_id = str(row.get("segment_id") or "")
        view = str(row.get("view") or row.get("source_view") or "unknown")
        phase = str(row.get("phase") or "")
        segment = segments_by_id.get(segment_id, {})
        time_sec = _phase_time(segment, phase)
        global_time = _phase_global_time(segment, phase)
        for detection in _detections(row):
            label = _label(detection)
            if not label or label.lower() in IGNORED_TRACK_LABELS:
                continue
            bbox = _bbox(detection.get("bbox") or detection.get("box"))
            if bbox is None:
                continue
            groups[(segment_id, view, label)].append(
                {
                    "index": len(groups[(segment_id, view, label)]),
                    "time_sec": time_sec,
                    "global_time": global_time,
                    "phase": phase,
                    "image_path": row.get("image_path"),
                    "annotated_image_path": row.get("annotated_image_path"),
                    "bbox": bbox,
                    "center": _center(bbox),
                    "confidence": _confidence(detection, default=None),
                    "source_row_index": row_index,
                }
            )

    tracks = []
    for (segment_id, view, label), points in sorted(groups.items()):
        if len(points) < min_points:
            continue
        segment = segments_by_id.get(segment_id, {})
        tracks.append(
            _track_row(
                session_id=segment.get("session_id") or session_id,
                segment_id=segment_id,
                micro_segment_id=None,
                view=view,
                label=label,
                points=points,
                source_path=source_path,
                model_path=model_path,
                start_sec=_float(segment.get("start_sec")),
                end_sec=_float(segment.get("end_sec")),
                global_start_time=segment.get("global_start_time"),
                global_end_time=segment.get("global_end_time"),
                min_motion_px=min_motion_px,
                max_points=max_points_per_track,
                source_mode="yolo_keyframe_detections",
            )
        )
    return tracks


def _track_row(
    *,
    session_id: Any,
    segment_id: Any,
    micro_segment_id: Any,
    view: str,
    label: str,
    points: list[dict[str, Any]],
    source_path: Path | None,
    model_path: str | None,
    start_sec: float | None,
    end_sec: float | None,
    global_start_time: Any,
    global_end_time: Any,
    min_motion_px: float,
    max_points: int,
    source_mode: str,
) -> dict[str, Any]:
    metrics = _track_metrics(points)
    displacement = float(metrics.get("displacement_px") or 0.0)
    path_length = float(metrics.get("path_length_px") or 0.0)
    confidence_values = [float(point["confidence"]) for point in points if point.get("confidence") is not None]
    confidence = round(mean(confidence_values), 4) if confidence_values else 0.5
    track_id = f"yolo_track:{micro_segment_id or segment_id or 'unassigned'}:{view}:{label}"
    track_evidence = track_evidence_from_points(
        track_id=track_id,
        object_label=label,
        points=points,
        track_type="label_level_pseudo_track",
        source_view=view,
        identity_confidence=0.62,
        id_switch_risk=0.5,
        can_confirm_motion=False,
        limitations=["label-level pseudo-track; no external re-identification tracker id; cannot confirm object movement"],
    )
    gate = gate_object_move(
        event_candidate={
            "candidate_id": track_id,
            "event_type": "object_move",
            "time_start": start_sec,
            "time_end": end_sec,
            "source_view": view,
            "object_labels": [label],
        },
        track=track_evidence,
        scene_motion={"method": "none", "limitations": ["no_scene_stabilization"]},
    )
    moving = gate.get("status") == "confirmed"
    candidate_motion = gate.get("status") == "candidate"
    limitations = list(dict.fromkeys([*track_evidence.limitations, *(gate.get("limitations") or [])]))
    if not moving:
        limitations.append("object movement not confirmed by physical_event_gate")
    reasons = [
        "YOLO detection rows converted to standard object track observation",
        f"point_count={len(points)}",
        f"source_mode={source_mode}",
        f"physical_event_gate_status={gate.get('status')}",
    ]
    if moving or candidate_motion:
        reasons.append(f"motion_px={max(displacement, path_length):.3f}")
    evidence_detail = gate.get("evidence") or {}
    confirmation_level = "measured" if moving else ("candidate" if candidate_motion else "rejected")
    event_type = "object_movement_measured" if moving else ("object_movement_candidate" if candidate_motion else "object_track_observed")
    return {
        "event_id": track_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "micro_segment_id": micro_segment_id,
        "start_sec": round(start_sec, 4) if start_sec is not None else _first_point_time(points),
        "end_sec": round(end_sec, 4) if end_sec is not None else _last_point_time(points),
        "global_start_time": global_start_time or points[0].get("global_time"),
        "global_end_time": global_end_time or points[-1].get("global_time"),
        "view": view,
        "object_label": label,
        "track_id": track_id,
        "state": "moving" if moving else ("candidate_motion" if candidate_motion else "tracked_static"),
        "motion_state": "moving" if moving else ("candidate_motion" if candidate_motion else "tracked_static"),
        "event_type": event_type,
        "physical_event_type": "object_move",
        "status": gate.get("status"),
        "hard_gate": gate.get("hard_gate"),
        "reject_reasons": gate.get("reject_reasons") or [],
        "evidence_detail": evidence_detail,
        "motion_threshold_px": evidence_detail.get("motion_threshold_px"),
        "jitter_sigma_px": evidence_detail.get("jitter_sigma_px"),
        "track_type": evidence_detail.get("track_type") or track_evidence.track_type,
        "can_confirm_motion": evidence_detail.get("can_confirm_motion", track_evidence.can_confirm_motion),
        "physical_event_gate": gate,
        "confirmation_level": confirmation_level,
        "confidence": confidence if moving else min(confidence, float(gate.get("confidence") or 0.5)),
        "points": points[:max_points],
        "path_length_px": metrics.get("path_length_px"),
        "displacement_px": metrics.get("displacement_px"),
        "movement_score": round(min(1.0, max(displacement, path_length) / max(min_motion_px, 1.0)), 4),
        "identity_confidence": evidence_detail.get("identity_confidence", track_evidence.identity_confidence),
        "id_switch_risk": evidence_detail.get("id_switch_risk", track_evidence.id_switch_risk),
        "model_name": Path(model_path).name if model_path else None,
        "model_path": model_path,
        "source_mode": source_mode,
        "source_file": str(source_path) if source_path is not None else None,
        "evidence_reasons": reasons,
        "limitations": limitations,
        "metrics": {
            **metrics,
            "source_mode": source_mode,
            "min_motion_px": min_motion_px,
            "raw_point_count": len(points),
            "emitted_point_count": min(len(points), max_points),
            "physical_event_gate": gate,
        },
        "payload": {
            "source": "yolo_observation_inputs",
            "source_mode": source_mode,
            "source_file": str(source_path) if source_path is not None else None,
        },
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
    ):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        value = data.get("model_path")
        if value:
            return str(value)
    return None


def _micro_for_frame_row(row: Mapping[str, Any], micro_rows: list[Mapping[str, Any]]) -> Mapping[str, Any] | None:
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


def _row_time(row: Mapping[str, Any]) -> float | None:
    return _first_float(
        row.get("alignment_time_sec"),
        row.get("session_time_sec"),
        row.get("local_time_sec"),
        row.get("time_sec"),
    )


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
    fraction = PHASE_ORDER.get(phase, 0.5)
    return (start + (end - start) * fraction).isoformat()


def _detections(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    detections = row.get("detections")
    return [item for item in detections if isinstance(item, Mapping)] if isinstance(detections, list) else []


def _label(detection: Mapping[str, Any]) -> str:
    return str(detection.get("label") or detection.get("object_label") or detection.get("raw_label") or "").strip()


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return [round(float(value[0]), 4), round(float(value[1]), 4), round(float(value[2]), 4), round(float(value[3]), 4)]
    except (TypeError, ValueError):
        return None


def _center(bbox: list[float]) -> list[float]:
    return [round((bbox[0] + bbox[2]) / 2.0, 4), round((bbox[1] + bbox[3]) / 2.0, 4)]


def _track_metrics(points: list[Mapping[str, Any]]) -> dict[str, Any]:
    centers = [point.get("center") for point in points if isinstance(point.get("center"), list) and len(point["center"]) >= 2]
    result: dict[str, Any] = {"point_count": len(points)}
    if len(centers) >= 2:
        first = centers[0]
        last = centers[-1]
        displacement = math.dist((float(first[0]), float(first[1])), (float(last[0]), float(last[1])))
        path_length = sum(
            math.dist((float(prev[0]), float(prev[1])), (float(current[0]), float(current[1])))
            for prev, current in zip(centers, centers[1:])
        )
        result.update(
            {
                "first_center": [round(float(first[0]), 4), round(float(first[1]), 4)],
                "last_center": [round(float(last[0]), 4), round(float(last[1]), 4)],
                "displacement_px": round(displacement, 4),
                "path_length_px": round(path_length, 4),
            }
        )
    return result


def _preserved_existing_tracks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    preserved = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        payload = row.get("payload")
        if isinstance(payload, Mapping) and payload.get("source") == "yolo_observation_inputs":
            continue
        preserved.append(dict(row))
    return preserved


def _session_id(session: Path, micro_rows: list[Mapping[str, Any]], segment_rows: list[Mapping[str, Any]]) -> str:
    for row in [*micro_rows, *segment_rows]:
        value = row.get("session_id") if isinstance(row, Mapping) else None
        if value:
            return str(value)
    manifest = session / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if data.get("session_id"):
                return str(data["session_id"])
        except (OSError, json.JSONDecodeError):
            pass
    return session.name


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _confidence(row: Mapping[str, Any], default: float | None) -> float | None:
    value = _first_float(row.get("confidence"), row.get("score"), row.get("probability"))
    if value is None:
        return default
    return round(max(0.0, min(1.0, value)), 4)


def _first_point_time(points: list[Mapping[str, Any]]) -> float | None:
    return _float(points[0].get("time_sec")) if points else None


def _last_point_time(points: list[Mapping[str, Any]]) -> float | None:
    return _float(points[-1].get("time_sec")) if points else None


def _first_float(*values: Any) -> float | None:
    for value in values:
        numeric = _float(value)
        if numeric is not None:
            return numeric
    return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


TRACK_FIELDS = (
    "event_id",
    "session_id",
    "segment_id",
    "micro_segment_id",
    "start_sec",
    "end_sec",
    "global_start_time",
    "global_end_time",
    "view",
    "object_label",
    "track_id",
    "state",
    "motion_state",
    "event_type",
    "confirmation_level",
    "confidence",
    "points",
    "path_length_px",
    "displacement_px",
    "movement_score",
    "identity_confidence",
    "model_name",
    "model_path",
    "source_mode",
    "source_file",
    "evidence_reasons",
    "limitations",
    "metrics",
    "payload",
    "physical_event_type",
    "status",
    "hard_gate",
    "reject_reasons",
    "evidence_detail",
    "motion_threshold_px",
    "jitter_sigma_px",
    "track_type",
    "can_confirm_motion",
    "physical_event_gate",
)


def _stable(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in TRACK_FIELDS}


def _stable_gate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": row.get("status"),
        "event_type": row.get("event_type"),
        "confidence": row.get("confidence"),
        "hard_gate": row.get("hard_gate") or {},
        "evidence": row.get("evidence") or {},
        "reject_reasons": row.get("reject_reasons") or [],
        "limitations": row.get("limitations") or [],
        "audit": row.get("audit") or {},
    }


def _stable_rejected(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "event_type": row.get("event_type"),
        "status": row.get("status"),
        "time_start": row.get("time_start"),
        "time_end": row.get("time_end"),
        "source_view": row.get("source_view"),
        "actor_track_id": row.get("actor_track_id"),
        "object_track_ids": row.get("object_track_ids") or [],
        "object_labels": row.get("object_labels") or [],
        "reject_reasons": row.get("reject_reasons") or [],
        "evidence_detail": row.get("evidence_detail") or {},
        "limitations": row.get("limitations") or [],
    }


__all__ = ["build_yolo_observation_inputs"]
