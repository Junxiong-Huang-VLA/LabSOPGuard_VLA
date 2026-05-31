from __future__ import annotations

import math
from statistics import median
from typing import Any, Iterable, Mapping

from .physical_event_types import SceneMotionEvidence


def estimate_scene_motion_from_points(rows: Iterable[Mapping[str, Any]] | None = None) -> SceneMotionEvidence:
    """Lightweight fallback scene-motion estimate from detection centers.

    This intentionally stays conservative. If rows do not contain enough stable
    anchor-like detections, it returns method="none" and records the limitation.
    """

    if rows is None:
        return SceneMotionEvidence(method="none", limitations=["no_scene_stabilization"])
    deltas: list[float] = []
    previous_by_label: dict[str, tuple[float, float]] = {}
    for row in rows:
        detections = row.get("detections") or []
        if not isinstance(detections, list):
            continue
        for det in detections:
            if not isinstance(det, Mapping):
                continue
            label = str(det.get("label") or det.get("object_label") or "").strip().lower()
            if label in {"hand", "gloved_hand", "lab_coat"}:
                continue
            bbox = det.get("bbox") or det.get("box")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            center = ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)
            previous = previous_by_label.get(label)
            if previous is not None:
                deltas.append(math.dist(previous, center))
            previous_by_label[label] = center
    if len(deltas) < 4:
        return SceneMotionEvidence(method="none", limitations=["no_scene_stabilization"])
    shift = float(median(deltas))
    is_camera_motion = shift >= 20.0
    return SceneMotionEvidence(
        is_camera_motion=is_camera_motion,
        background_shift_px=round(shift, 3),
        homography_confidence=0.0,
        global_motion_ratio=1.0 if is_camera_motion else 0.0,
        method="static_anchor",
        limitations=[] if not is_camera_motion else ["global detection shift suggests camera motion"],
    )


def no_scene_stabilization(reason: str = "no_scene_stabilization") -> SceneMotionEvidence:
    return SceneMotionEvidence(method="none", limitations=[reason])

