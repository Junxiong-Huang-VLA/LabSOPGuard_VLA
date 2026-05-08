from __future__ import annotations

from key_action_indexer.yolo_detector import (
    canonical_yolo_label,
    filter_implausible_detections,
    find_hand_object_interactions,
    normalize_yolo_detection,
)


def test_canonical_yolo_label_normalizes_lab_synonyms() -> None:
    assert canonical_yolo_label("glove") == "gloved_hand"
    assert canonical_yolo_label("sample bottle") == "sample_bottle"
    assert canonical_yolo_label("tube-cap") == "tube_cap"


def test_find_hand_object_interactions_scores_nearby_hand_and_container() -> None:
    detections = [
        normalize_yolo_detection({"label": "gloved_hand", "confidence": 0.9, "bbox": [10, 10, 40, 40]}),
        normalize_yolo_detection({"label": "container", "confidence": 0.8, "bbox": [35, 12, 70, 42]}),
        normalize_yolo_detection({"label": "balance", "confidence": 0.8, "bbox": [200, 200, 260, 260]}),
    ]

    interactions = find_hand_object_interactions(detections, frame_width=320, frame_height=240)

    assert interactions
    assert interactions[0]["object_label"] == "container"
    assert interactions[0]["score"] > 0.1


def test_implausible_detection_filter_suppresses_flat_blue_background_for_all_classes() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)  # BGR: blue/cyan workbench-like region.
    detections = [
        {"label": "gloved_hand", "confidence": 0.82, "bbox": [20, 20, 210, 130]},
        {"label": "beaker", "confidence": 0.81, "bbox": [25, 30, 230, 145]},
        {"label": "sample_bottle_blue", "confidence": 0.84, "bbox": [250, 40, 282, 108]},
    ]

    kept, ignored = filter_implausible_detections(
        detections,
        frame=frame,
        source_view="third_person",
    )

    assert [item["label"] for item in kept] == ["sample_bottle_blue"]
    assert {item["label"] for item in ignored} == {"gloved_hand", "beaker"}


def test_third_person_low_confidence_edge_hand_does_not_drive_interaction() -> None:
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    detections = [
        {"label": "gloved_hand", "confidence": 0.51, "bbox": [2, 212, 151, 354]},
        {"label": "container", "confidence": 0.76, "bbox": [0, 154, 90, 285]},
    ]

    interactions = find_hand_object_interactions(
        detections,
        frame=frame,
        source_view="third_person",
    )

    assert interactions == []
