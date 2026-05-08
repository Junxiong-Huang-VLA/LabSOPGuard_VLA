from __future__ import annotations

from key_action_indexer.yolo_analysis import (
    TemporalDetectionSmoother,
    filter_detections_by_allowed_labels,
    filter_detections_by_class_threshold,
    parse_class_thresholds,
)


def test_parse_class_thresholds_accepts_json_and_key_value_pairs() -> None:
    from_json = parse_class_thresholds('{"container": 0.4}')
    from_pairs = parse_class_thresholds("container=0.4, pipette=0.5")

    assert from_json["container"] == 0.4
    assert from_pairs["container"] == 0.4
    assert from_pairs["pipette"] == 0.5


def test_filter_detections_by_threshold_and_allowed_labels() -> None:
    detections = [
        {"label": "container", "confidence": 0.45},
        {"label": "pipette", "confidence": 0.2},
    ]

    filtered = filter_detections_by_class_threshold(detections, {"container": 0.4, "pipette": 0.3})
    allowed = filter_detections_by_allowed_labels(filtered, ["container"])

    assert filtered == [{"label": "container", "confidence": 0.45}]
    assert allowed == filtered


def test_temporal_detection_smoother_keeps_recent_confirmed_label() -> None:
    smoother = TemporalDetectionSmoother(hold_frames=2, min_hits=2)

    first = smoother.update([{"label": "container", "confidence": 0.8, "bbox": [0, 0, 10, 10]}])
    second = smoother.update([{"label": "container", "confidence": 0.82, "bbox": [0, 0, 10, 10]}])
    third = smoother.update([])

    assert first == []
    assert second and second[0]["label"] == "container"
    assert third and third[0]["label"] == "container"
