from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from .physical_event_types import TrackEvidence


def bbox_center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def bbox_size(bbox: Sequence[float]) -> float:
    return max(abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1])), 1.0)


def track_evidence_from_points(
    *,
    track_id: str,
    object_label: str,
    points: Sequence[Mapping[str, Any]],
    track_type: str,
    source_view: str = "",
    identity_confidence: float | None = None,
    id_switch_risk: float | None = None,
    background_shift_px: float = 0.0,
    can_confirm_motion: bool | None = None,
    limitations: Sequence[str] | None = None,
) -> TrackEvidence:
    usable = []
    for point in points:
        bbox = point.get("bbox") or point.get("box")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            usable.append({**dict(point), "bbox": [float(v) for v in bbox]})
    centers = [bbox_center(point["bbox"]) for point in usable]
    sizes = [bbox_size(point["bbox"]) for point in usable]
    raw_displacement = math.dist(centers[0], centers[-1]) if len(centers) >= 2 else 0.0
    path_length = sum(math.dist(a, b) for a, b in zip(centers, centers[1:]))
    step_lengths = [math.dist(a, b) for a, b in zip(centers, centers[1:])]
    median_size = float(median(sizes)) if sizes else 0.0
    persistent_steps = sum(1 for step in step_lengths if step >= max(3.0, 0.03 * median_size))
    normalized_type = str(track_type or "").strip() or "inferred_track"
    limits = list(limitations or [])
    if normalized_type == "label_level_pseudo_track":
        limits.append("label-level pseudo-track; cannot confirm object movement")
    can_confirm = normalized_type != "label_level_pseudo_track" if can_confirm_motion is None else bool(can_confirm_motion)
    if normalized_type == "label_level_pseudo_track":
        can_confirm = False
    return TrackEvidence(
        track_id=track_id,
        track_type=normalized_type,
        object_label=str(object_label or ""),
        source_view=source_view,
        point_count=len(usable),
        identity_confidence=float(identity_confidence if identity_confidence is not None else (0.82 if normalized_type in {"tracker_track", "instance_track"} else 0.68)),
        id_switch_risk=float(id_switch_risk if id_switch_risk is not None else (0.12 if normalized_type in {"tracker_track", "instance_track"} else 0.35)),
        median_bbox_size=round(median_size, 3),
        raw_displacement_px=round(raw_displacement, 3),
        path_length_px=round(path_length, 3),
        stabilized_displacement_px=round(max(0.0, raw_displacement - float(background_shift_px or 0.0)), 3),
        motion_persistent=persistent_steps >= 2 or raw_displacement >= max(18.0, 0.4 * median_size),
        can_confirm_motion=can_confirm,
        limitations=sorted(set(limits)),
    )


def normalize_track_evidence(track: Mapping[str, Any] | TrackEvidence | None) -> TrackEvidence | None:
    if track is None:
        return None
    if isinstance(track, TrackEvidence):
        return track
    points = track.get("points") or []
    if points:
        return track_evidence_from_points(
            track_id=str(track.get("track_id") or ""),
            object_label=str(track.get("object_label") or track.get("label") or track.get("class_name") or ""),
            points=points,
            track_type=str(track.get("track_type") or track.get("source_type") or "inferred_track"),
            source_view=str(track.get("source_view") or track.get("view") or ""),
            identity_confidence=_optional_float(track.get("identity_confidence")),
            id_switch_risk=_optional_float(track.get("id_switch_risk")),
            background_shift_px=_float(track.get("background_shift_px")),
            can_confirm_motion=track.get("can_confirm_motion"),
            limitations=track.get("limitations") or [],
        )
    return TrackEvidence(
        track_id=str(track.get("track_id") or ""),
        track_type=str(track.get("track_type") or "inferred_track"),
        object_label=str(track.get("object_label") or track.get("label") or track.get("class_name") or ""),
        source_view=str(track.get("source_view") or track.get("view") or ""),
        point_count=int(track.get("point_count") or 0),
        identity_confidence=_float(track.get("identity_confidence"), 0.0),
        id_switch_risk=_float(track.get("id_switch_risk"), 1.0),
        median_bbox_size=_float(track.get("median_bbox_size"), 0.0),
        raw_displacement_px=_float(track.get("raw_displacement_px") or track.get("displacement_px"), 0.0),
        path_length_px=_float(track.get("path_length_px"), 0.0),
        stabilized_displacement_px=_optional_float(track.get("stabilized_displacement_px")),
        motion_persistent=bool(track.get("motion_persistent")),
        can_confirm_motion=bool(track.get("can_confirm_motion", True)),
        limitations=list(track.get("limitations") or []),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

