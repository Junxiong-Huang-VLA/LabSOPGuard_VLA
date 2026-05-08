"""Compute temporal offsets between dual-view YOLO frame timelines."""

from __future__ import annotations

from typing import Any

from .yolo_detector import INTERACTION_OBJECT_LABELS, canonical_yolo_label


def _row_time(row: dict[str, Any]) -> float:
    return float(row.get("time_sec", row.get("local_time_sec", 0.0)) or 0.0)


def _first_glove_time(rows: list[dict[str, Any]], min_confidence: float = 0.35) -> float | None:
    for row in sorted(rows, key=_row_time):
        for detection in row.get("detections") or []:
            label = canonical_yolo_label(detection.get("label", ""))
            if label == "gloved_hand" and float(detection.get("confidence", 0.0) or 0.0) >= min_confidence:
                return _row_time(row)
    return None


def _first_interaction_time(rows: list[dict[str, Any]]) -> float | None:
    for row in sorted(rows, key=_row_time):
        interactions = row.get("hand_object_interactions") or []
        if interactions and float(interactions[0].get("score", 0.0) or 0.0) >= 0.2:
            return _row_time(row)
    return None


def _first_object_time(rows: list[dict[str, Any]], target_labels: set[str] | None = None) -> float | None:
    labels = target_labels or set(INTERACTION_OBJECT_LABELS)
    for row in sorted(rows, key=_row_time):
        for detection in row.get("detections") or []:
            label = canonical_yolo_label(detection.get("label", ""))
            if label in labels and float(detection.get("confidence", 0.0) or 0.0) >= 0.3:
                return _row_time(row)
    return None


def compute_view_offset(
    yolo_frame_rows: list[dict[str, Any]],
    reference_view: str = "third_person",
    target_view: str = "first_person",
) -> dict[str, Any]:
    """Estimate seconds to add to target-view timestamps for alignment."""

    reference_rows = [row for row in yolo_frame_rows if row.get("source_view") == reference_view]
    target_rows = [row for row in yolo_frame_rows if row.get("source_view") == target_view]
    no_offset = {
        "offset_sec": 0.0,
        "method": "none",
        "confidence": 0.0,
        "reference_landmark_sec": 0.0,
        "target_landmark_sec": 0.0,
        "reference_view": reference_view,
        "target_view": target_view,
    }
    if not reference_rows or not target_rows:
        return no_offset

    candidates: list[tuple[float, str, float, float, float]] = []
    landmarks = [
        ("glove_appearance", 0.8, _first_glove_time(reference_rows), _first_glove_time(target_rows)),
        ("first_interaction", 0.6, _first_interaction_time(reference_rows), _first_interaction_time(target_rows)),
        ("first_object", 0.4, _first_object_time(reference_rows), _first_object_time(target_rows)),
    ]
    for method, weight, reference_time, target_time in landmarks:
        if reference_time is None or target_time is None:
            continue
        candidates.append((weight, method, target_time - reference_time, reference_time, target_time))
    if not candidates:
        return no_offset

    if len(candidates) >= 2 and max(item[2] for item in candidates) - min(item[2] for item in candidates) < 5.0:
        total_weight = sum(item[0] for item in candidates)
        best_offset = sum(weight * offset for weight, _, offset, _, _ in candidates) / total_weight
        _, _, _, best_ref, best_target = candidates[0]
        best_confidence = 0.9
        best_method = "multi_landmark_consensus"
    else:
        best_confidence, best_method, best_offset, best_ref, best_target = max(candidates, key=lambda item: item[0])

    return {
        "offset_sec": round(best_offset, 3),
        "method": best_method,
        "confidence": round(best_confidence, 3),
        "reference_landmark_sec": round(best_ref, 3),
        "target_landmark_sec": round(best_target, 3),
        "reference_view": reference_view,
        "target_view": target_view,
        "all_candidates": [
            {"method": method, "offset": round(offset, 3), "ref_t": round(ref_t, 3), "tgt_t": round(target_t, 3)}
            for _, method, offset, ref_t, target_t in candidates
        ],
    }


__all__ = ["compute_view_offset"]
