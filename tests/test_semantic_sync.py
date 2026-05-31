from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.semantic_sync import MultimodalSemanticSyncEngine
from labsopguard.time_sync import TimeSyncCalibrator


def test_semantic_sync_generates_anchors_from_cross_view_events():
    frames = [
        {
            "camera_id": "usb0",
            "view_type": "first_person",
            "local_timestamp_sec": 10.0,
            "frame_id": 10,
            "scene_description": "operator starts setup and prepares the sample",
            "object_labels": ["gloves"],
            "confidence": 0.9,
        },
        {
            "camera_id": "top_view",
            "view_type": "third_person",
            "local_timestamp_sec": 7.5,
            "frame_id": 15,
            "scene_description": "operator starts setup at the bench",
            "object_labels": ["person"],
            "confidence": 0.88,
        },
        {
            "camera_id": "usb0",
            "view_type": "first_person",
            "local_timestamp_sec": 20.0,
            "frame_id": 20,
            "scene_description": "operator uses pipette to transfer liquid into tube",
            "object_labels": ["pipette", "tube"],
            "detected_activities": ["liquid transfer"],
            "confidence": 0.92,
        },
        {
            "camera_id": "top_view",
            "view_type": "third_person",
            "local_timestamp_sec": 17.5,
            "frame_id": 25,
            "scene_description": "liquid transfer with pipette is visible",
            "object_labels": ["pipette"],
            "detected_activities": ["transfer"],
            "confidence": 0.86,
        },
    ]

    result = MultimodalSemanticSyncEngine.build(
        experiment_id="exp_sem",
        run_id="run_sem",
        frame_items=frames,
    )

    assert result["status"] == "calibrated"
    assert result["reference_stream"]["camera_id"] == "usb0"
    assert {event["event_type"] for event in result["semantic_events"]} >= {"experiment_start", "liquid_transfer"}
    anchors = result["sync_anchors"]
    assert len(anchors) >= 2
    assert all(anchor["method"] == "multimodal_semantic" for anchor in anchors)

    sync_anchors = MultimodalSemanticSyncEngine.anchors_as_sync_anchors(result)
    profile = TimeSyncCalibrator.fit_profile_from_anchors("top_view", sync_anchors, reference_camera_id="usb0")
    assert profile.offset_sec == 2.5
    assert profile.local_to_global(17.5) == 20.0


def test_semantic_sync_reports_insufficient_overlap():
    result = MultimodalSemanticSyncEngine.build(
        experiment_id="exp_sem",
        run_id="run_sem",
        frame_items=[
            {
                "camera_id": "usb0",
                "view_type": "first_person",
                "local_timestamp_sec": 1.0,
                "scene_description": "operator uses pipette to transfer liquid",
            },
            {
                "camera_id": "top_view",
                "view_type": "third_person",
                "local_timestamp_sec": 2.0,
                "scene_description": "empty bench with no matching action",
            },
        ],
    )

    assert result["status"] == "insufficient_overlap"
    assert result["sync_anchors"] == []
