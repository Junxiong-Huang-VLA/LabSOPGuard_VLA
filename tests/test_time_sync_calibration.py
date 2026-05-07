from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.time_sync import SyncAnchor, TimeSyncCalibrator


def test_single_anchor_fits_offset_only():
    anchors = [
        SyncAnchor(
            camera_id="cam_b",
            local_time_sec=4.0,
            reference_time_sec=5.25,
            method="manual",
            confidence=0.9,
        )
    ]

    profile = TimeSyncCalibrator.fit_profile_from_anchors(
        "cam_b",
        anchors,
        reference_camera_id="cam_a",
    )

    assert profile.offset_sec == 1.25
    assert profile.clock_scale == 1.0
    assert profile.local_to_global(10.0) == 11.25


def test_multiple_anchors_fit_clock_drift():
    scale = 1.0005
    offset = 2.0
    anchors = [
        SyncAnchor(camera_id="cam_b", local_time_sec=0.0, reference_time_sec=offset, method="sync_board"),
        SyncAnchor(camera_id="cam_b", local_time_sec=120.0, reference_time_sec=120.0 * scale + offset, method="sync_board"),
        SyncAnchor(camera_id="cam_b", local_time_sec=240.0, reference_time_sec=240.0 * scale + offset, method="sync_board"),
    ]

    profile = TimeSyncCalibrator.fit_profile_from_anchors(
        "cam_b",
        anchors,
        reference_camera_id="cam_a",
    )

    assert profile.offset_sec == pytest.approx(offset, abs=1e-6)
    assert profile.clock_scale == pytest.approx(scale, abs=1e-9)
    assert profile.drift_ppm == pytest.approx(500.0, abs=0.001)
    assert profile.local_to_global(60.0) == pytest.approx(62.03, abs=1e-6)


def _summary(timestamp: float, delta: float) -> dict:
    return {"timestamp_sec": timestamp, "brightness_delta": delta}


def test_auto_visual_flash_and_event_matching_generates_sync_anchors():
    scale = 1.001
    offset = 2.0

    def cam_b_local(reference_time: float) -> float:
        return (reference_time - offset) / scale

    streams = {
        "cam_a": {
            "frame_summaries": [
                _summary(9.0, 0.0),
                _summary(10.0, 100.0),
                _summary(11.0, 0.0),
                _summary(29.0, 0.0),
                _summary(30.0, 120.0),
                _summary(31.0, 0.0),
            ],
            "events": [
                {"timestamp_sec": 50.0, "event_id": "operator_clap", "type": "sync_event", "confidence": 0.9}
            ],
        },
        "cam_b": {
            "frame_summaries": [
                _summary(cam_b_local(10.0) - 1.0, 0.0),
                _summary(cam_b_local(10.0), 95.0),
                _summary(cam_b_local(10.0) + 1.0, 0.0),
                _summary(cam_b_local(30.0) - 1.0, 0.0),
                _summary(cam_b_local(30.0), 130.0),
                _summary(cam_b_local(30.0) + 1.0, 0.0),
            ],
            "events": [
                {
                    "timestamp_sec": cam_b_local(50.0),
                    "event_id": "operator_clap",
                    "type": "sync_event",
                    "confidence": 0.9,
                }
            ],
        },
    }

    anchors = TimeSyncCalibrator.generate_visual_sync_anchors(
        streams,
        reference_camera_id="cam_a",
        z_threshold=1.0,
    )
    profile = TimeSyncCalibrator.fit_profile_from_anchors(
        "cam_b",
        anchors,
        reference_camera_id="cam_a",
    )

    assert [anchor.reference_time_sec for anchor in anchors] == [10.0, 30.0, 50.0]
    assert all(anchor.method.startswith("auto_visual:") for anchor in anchors)
    assert profile.offset_sec == pytest.approx(offset, abs=2e-6)
    assert profile.clock_scale == pytest.approx(scale, abs=1e-9)
