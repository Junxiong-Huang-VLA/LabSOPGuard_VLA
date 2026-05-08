"""Detect the true experiment start from YOLO frame evidence.

The detector is intentionally small and dependency-free so dry-run sessions can
exercise the same API without video files or ffmpeg.
"""

from __future__ import annotations

from typing import Any

from .yolo_detector import INTERACTION_OBJECT_LABELS, canonical_yolo_label


def _row_time(row: dict[str, Any]) -> float:
    return float(row.get("time_sec", row.get("local_time_sec", 0.0)) or 0.0)


def _glove_confidence_in_row(row: dict[str, Any], min_confidence: float) -> float | None:
    for detection in row.get("detections") or []:
        label = canonical_yolo_label(detection.get("label", ""))
        if label != "gloved_hand":
            continue
        confidence = float(detection.get("confidence", 0.0) or 0.0)
        if confidence >= min_confidence:
            return confidence
    return None


def _first_interaction_object_in_window(
    rows: list[dict[str, Any]],
    start_idx: int,
    window_start_sec: float,
    window_sec: float,
) -> tuple[str | None, float]:
    window_end = window_start_sec + window_sec
    for row in rows[start_idx:]:
        time_sec = _row_time(row)
        if time_sec > window_end:
            break
        for detection in row.get("detections") or []:
            label = canonical_yolo_label(detection.get("label", ""))
            if label in INTERACTION_OBJECT_LABELS:
                return label, time_sec
    return None, 0.0


def detect_experiment_start(
    yolo_frame_rows: list[dict[str, Any]],
    *,
    min_glove_confidence: float = 0.35,
    min_consecutive_frames: int = 2,
    object_confirmation_window_sec: float = 15.0,
    buffer_sec: float = 2.0,
) -> dict[str, Any]:
    """Find the earliest sustained glove appearance confirmed by lab objects."""

    not_detected: dict[str, Any] = {
        "detected": False,
        "start_time_sec": 0.0,
        "glove_first_seen_sec": 0.0,
        "confirmation_object": "",
        "confirmation_time_sec": 0.0,
        "confidence": 0.0,
    }
    if not yolo_frame_rows:
        return not_detected

    ordered = sorted(yolo_frame_rows, key=_row_time)
    streak = 0
    streak_start_idx = 0
    streak_start_sec = 0.0
    streak_confidence = 0.0
    gap_tolerance = 1

    for idx, row in enumerate(ordered):
        confidence = _glove_confidence_in_row(row, min_glove_confidence)
        if confidence is not None:
            if streak == 0:
                streak_start_idx = idx
                streak_start_sec = _row_time(row)
                streak_confidence = confidence
            streak += 1
            gap_tolerance = 1
        elif gap_tolerance > 0 and streak > 0:
            gap_tolerance -= 1
        else:
            streak = 0
            gap_tolerance = 1

        if streak < min_consecutive_frames:
            continue

        object_label, object_time = _first_interaction_object_in_window(
            ordered,
            streak_start_idx,
            streak_start_sec,
            object_confirmation_window_sec,
        )
        if object_label is None:
            object_label, object_time = _first_interaction_object_in_window(
                ordered,
                max(0, streak_start_idx - 5),
                max(0.0, streak_start_sec - 5.0),
                object_confirmation_window_sec + 5.0,
            )
        if object_label is None:
            continue

        trim_time = max(0.0, streak_start_sec - buffer_sec)
        return {
            "detected": True,
            "start_time_sec": round(trim_time, 6),
            "glove_first_seen_sec": round(streak_start_sec, 6),
            "confirmation_object": object_label,
            "confirmation_time_sec": round(object_time, 6),
            "confidence": round(streak_confidence, 6),
        }

    return not_detected


def compute_video_trim(yolo_frame_rows: list[dict[str, Any]], **kwargs: Any) -> float:
    """Return seconds to trim from the beginning, or 0.0 if not detected."""

    result = detect_experiment_start(yolo_frame_rows, **kwargs)
    return float(result["start_time_sec"])


__all__ = ["compute_video_trim", "detect_experiment_start"]
