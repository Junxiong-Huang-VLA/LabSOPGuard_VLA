from __future__ import annotations

from key_action_indexer.experiment_start_detector import compute_video_trim, detect_experiment_start
from key_action_indexer.view_alignment import compute_view_offset


def _det(label: str, confidence: float = 0.8) -> dict:
    return {"label": label, "confidence": confidence, "bbox": [0, 0, 10, 10]}


def test_detect_experiment_start_requires_sustained_glove_and_object() -> None:
    rows = [
        {"time_sec": 1.0, "detections": [_det("gloved_hand", 0.9)]},
        {"time_sec": 1.5, "detections": [_det("gloved_hand", 0.85)]},
        {"time_sec": 2.0, "detections": [_det("container", 0.8)]},
    ]

    result = detect_experiment_start(rows, buffer_sec=0.25)

    assert result["detected"] is True
    assert result["start_time_sec"] == 0.75
    assert result["glove_first_seen_sec"] == 1.0
    assert result["confirmation_object"] == "container"
    assert compute_video_trim(rows, buffer_sec=0.25) == 0.75


def test_compute_view_offset_uses_multi_landmark_consensus() -> None:
    rows = [
        {"source_view": "third_person", "time_sec": 1.0, "detections": [_det("gloved_hand")]},
        {"source_view": "third_person", "time_sec": 3.0, "detections": [_det("container")], "hand_object_interactions": [{"score": 0.5}]},
        {"source_view": "first_person", "time_sec": 2.5, "detections": [_det("gloved_hand")]},
        {"source_view": "first_person", "time_sec": 4.5, "detections": [_det("container")], "hand_object_interactions": [{"score": 0.5}]},
    ]

    result = compute_view_offset(rows)

    assert result["method"] == "multi_landmark_consensus"
    assert result["confidence"] == 0.9
    assert result["offset_sec"] == 1.5
