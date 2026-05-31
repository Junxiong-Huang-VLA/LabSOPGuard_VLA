from __future__ import annotations

from typing import Any

from key_action_indexer.physical_event_gate import gate_object_move


def test_static_reagent_bottle_bbox_jitter_is_rejected() -> None:
    decision = gate_object_move(track=_track(raw=6.0, stabilized=5.0))

    assert decision["status"] == "rejected"
    assert "displacement_below_threshold" in decision["reject_reasons"]
    assert "bbox_jitter_or_static_object" in decision["reject_reasons"]


def test_label_level_pseudo_track_cannot_confirm_object_move() -> None:
    decision = gate_object_move(track={**_track(raw=40.0, stabilized=40.0), "track_type": "label_level_pseudo_track"})

    assert decision["status"] != "confirmed"
    assert "label_level_pseudo_track" in decision["reject_reasons"]


def test_camera_motion_rejects_otherwise_large_raw_displacement() -> None:
    decision = gate_object_move(
        track=_track(raw=32.0, stabilized=3.0),
        scene_motion={"is_camera_motion": True, "is_scene_cut": False, "background_shift_px": 29.0},
    )

    assert decision["status"] == "rejected"
    assert "camera_motion" in decision["reject_reasons"]


def test_real_stabilized_tracker_motion_confirms() -> None:
    decision = gate_object_move(
        track=_track(raw=42.0, stabilized=36.0, identity=0.93, risk=0.04),
        scene_motion={"is_camera_motion": False, "is_scene_cut": False, "background_shift_px": 2.0, "method": "feature_homography"},
    )

    assert decision["status"] == "confirmed"
    assert decision["hard_gate"]["passed"] is True


def test_change_score_only_cannot_confirm_object_move() -> None:
    decision = gate_object_move(event_candidate={"change_score": 0.9}, track=None)

    assert decision["status"] == "rejected"
    assert "missing_track" in decision["reject_reasons"]


def _track(
    *,
    raw: float,
    stabilized: float,
    identity: float = 0.9,
    risk: float = 0.1,
) -> dict[str, Any]:
    return {
        "track_id": "trk_bottle",
        "track_type": "tracker_track",
        "object_label": "reagent_bottle",
        "point_count": 6,
        "identity_confidence": identity,
        "id_switch_risk": risk,
        "median_bbox_size": 80.0,
        "raw_displacement_px": raw,
        "path_length_px": max(raw, stabilized),
        "stabilized_displacement_px": stabilized,
        "motion_persistent": True,
        "can_confirm_motion": True,
    }
