from __future__ import annotations

from typing import Any

import pytest

from key_action_indexer.material_references import _event_action_gate_reasons
from key_action_indexer.physical_event_gate import (
    gate_container_state_change,
    gate_hand_object_contact,
    gate_liquid_transfer,
    gate_object_move,
    gate_panel_operation,
    merge_gate_with_qwen_audit,
    parse_qwen_audit,
)
from key_action_indexer.physical_evidence import (
    validate_yolo_physical_evidence,
    yolo_physical_evidence_diagnostics,
)


def test_object_move_label_level_pseudotrack_requires_active_hand_object_evidence() -> None:
    reasons = _event_action_gate_reasons(
        {
            "event_type": "object_trajectory_movement",
            "metrics": {
                "measurement": {
                    "point_count": 15,
                    "identity_confidence": 0.72,
                    "displacement_px": 48.0,
                    "source_mode": "yolo_frame_rows",
                }
            },
            "confidence_reasons": ["YOLO detection rows converted to standard object track observation"],
            "limitations": ["label-level pseudo-track; no external re-identification tracker id"],
        },
        event_type="object_trajectory_movement",
        physical_action_type="object_movement",
        primary_object="reagent_bottle",
        contact_valid_count=0,
        contact_usable_count=1,
        contact_peak_score=0.74,
    )

    assert "label_level_pseudotrack_not_physical_movement" in reasons
    assert "object_movement_requires_active_hand_object_evidence" in reasons
    assert "sparse_movement_contact_evidence_not_default_result" in reasons


def test_object_move_candidate_without_measurement_is_not_treated_as_movement() -> None:
    reasons = _event_action_gate_reasons(
        {"event_type": "object_movement_candidate", "metrics": {}},
        event_type="object_movement_candidate",
        physical_action_type="object_movement",
        primary_object="sample_bottle",
        contact_valid_count=2,
        contact_usable_count=2,
        contact_peak_score=0.91,
    )

    assert "object_movement_not_measured" in reasons


def test_hand_proximity_gate_rejects_same_frame_detection_without_contact() -> None:
    ok, reasons = validate_yolo_physical_evidence(
        _evidence_row(
            hand_bbox=[40, 40, 100, 120],
            object_bbox=[700, 380, 760, 440],
            interactions=[],
        ),
        "pipette",
    )

    assert ok is False
    assert "missing_matching_hand_object_interaction" in reasons


def test_hand_proximity_gate_accepts_nearby_yolo_hand_object_evidence() -> None:
    ok, reasons = validate_yolo_physical_evidence(
        _evidence_row(
            hand_bbox=[180, 90, 250, 170],
            object_bbox=[220, 112, 320, 126],
            interactions=[],
        ),
        "pipette",
    )

    assert ok is True
    assert reasons == []


def test_physical_evidence_near_only_cannot_confirm_contact() -> None:
    ok, reasons = validate_yolo_physical_evidence(
        _evidence_row(
            hand_bbox=[100, 100, 150, 150],
            object_bbox=[160, 100, 220, 150],
            interactions=[
                {
                    "hand_label": "gloved_hand",
                    "object_label": "pipette",
                    "score": 0.92,
                    "hand_bbox": [100, 100, 150, 150],
                    "object_bbox": [160, 100, 220, 150],
                    "distance_px": 10.0,
                }
            ],
        ),
        "pipette",
    )

    assert ok is False
    assert set(reasons) & {"near_only", "single_frame_near"}


def test_static_liquid_candidate_stays_blocked_without_visual_flow_or_level_change() -> None:
    reasons = _event_action_gate_reasons(
        {
            "event_type": "liquid_transfer_candidate",
            "primary_object": "beaker",
            "confidence": 0.62,
            "conclusion_status": "candidate",
            "anomaly_flags": ["not_visual_liquid_flow_confirmed", "visual_confirmation_limited"],
        },
        event_type="liquid_transfer_candidate",
        physical_action_type="liquid_movement",
        primary_object="beaker",
        contact_valid_count=0,
        contact_usable_count=0,
        contact_peak_score=0.0,
    )

    assert reasons == ["liquid_movement_not_visually_supported"]


@pytest.mark.parametrize(
    ("event", "event_type", "physical_action_type", "primary_object"),
    [
        (
            {
                "event_type": "equipment_panel_operation_candidate",
                "primary_object": "panel",
                "confidence": 0.7,
            },
            "equipment_panel_operation_candidate",
            "equipment_panel_operation",
            "panel",
        ),
        (
            {
                "event_type": "container_state_change_candidate",
                "primary_object": "tube",
                "metrics": {"cap_lid_tokens": ["cap_open"]},
            },
            "container_state_change_candidate",
            "container_state_change",
            "tube",
        ),
    ],
)
def test_panel_and_container_presence_are_accepted_only_when_the_specific_gate_signal_exists(
    event: dict[str, Any],
    event_type: str,
    physical_action_type: str,
    primary_object: str,
) -> None:
    reasons = _event_action_gate_reasons(
        event,
        event_type=event_type,
        physical_action_type=physical_action_type,
        primary_object=primary_object,
        contact_valid_count=0,
        contact_usable_count=0,
        contact_peak_score=0.0,
    )

    assert reasons == []


def test_container_presence_only_signal_stays_review_blocked() -> None:
    reasons = _event_action_gate_reasons(
        {
            "event_type": "container_state_change_candidate",
            "primary_object": "tube",
            "metrics": {"container_state_indicators": {"state_signal": "container_interaction_only"}},
        },
        event_type="container_state_change_candidate",
        physical_action_type="container_state_change",
        primary_object="tube",
        contact_valid_count=0,
        contact_usable_count=0,
        contact_peak_score=0.0,
    )

    assert reasons == ["container_state_change_not_confirmed"]


def test_diagnostics_report_object_move_false_positive_reasons() -> None:
    diagnostics = yolo_physical_evidence_diagnostics(
        [
            _evidence_row(
                hand_bbox=[40, 40, 100, 120],
                object_bbox=[700, 380, 760, 440],
                interactions=[],
            )
        ],
        "pipette",
    )

    assert diagnostics["valid_evidence_count"] == 0
    assert diagnostics["invalid_reason_counts"]["missing_matching_hand_object_interaction"] == 1


def test_gate_rejects_static_reagent_bottle_bbox_jitter() -> None:
    decision = gate_object_move(track=_track(raw=6.0, stabilized=5.5, label="reagent_bottle"))

    assert decision["status"] == "rejected"
    assert "displacement_below_threshold" in decision["reject_reasons"]
    assert "bbox_jitter_or_static_object" in decision["reject_reasons"]


def test_gate_rejects_label_level_pseudo_track_even_with_large_displacement() -> None:
    decision = gate_object_move(track={**_track(raw=30.0, stabilized=30.0), "track_type": "label_level_pseudo_track"})

    assert decision["status"] == "rejected"
    assert "label_level_pseudo_track" in decision["reject_reasons"]


def test_gate_rejects_camera_motion_after_background_compensation() -> None:
    decision = gate_object_move(
        track=_track(raw=30.0, stabilized=2.0),
        scene_motion={"is_camera_motion": True, "background_shift_px": 28.0, "method": "feature_homography"},
    )

    assert decision["status"] == "rejected"
    assert "camera_motion" in decision["reject_reasons"]


def test_gate_confirms_real_tracker_motion() -> None:
    decision = gate_object_move(
        track=_track(raw=38.0, stabilized=35.0, identity=0.92, risk=0.05),
        scene_motion={"is_camera_motion": False, "is_scene_cut": False, "background_shift_px": 2.0, "method": "feature_homography"},
    )

    assert decision["status"] == "confirmed"
    assert decision["hard_gate"]["passed"] is True


def test_gate_rejects_change_score_without_object_motion() -> None:
    decision = gate_object_move(event_candidate={"change_score": 0.9}, track=_track(raw=4.0, stabilized=4.0))

    assert decision["status"] == "rejected"
    assert "change_score_only" in decision["reject_reasons"]


def test_gate_does_not_confirm_near_only_hand_object_contact() -> None:
    decision = gate_hand_object_contact(
        frame_evidence_list=[],
        external_observation={
            "has_hand": True,
            "has_object": True,
            "near_only": True,
            "min_distance_px": 8.0,
            "contact_frames": 0,
            "overlap_frames": 0,
        },
    )

    assert decision["status"] in {"candidate", "rejected"}
    assert decision["status"] != "confirmed"


def test_gate_confirms_sustained_hand_object_overlap() -> None:
    decision = gate_hand_object_contact(
        frame_evidence_list=[],
        external_observation={
            "has_hand": True,
            "has_object": True,
            "contact_frames": 3,
            "continuous_contact_frames": 3,
            "overlap_frames": 3,
            "max_iou": 0.08,
        },
    )

    assert decision["status"] == "confirmed"


def test_gate_rejects_static_liquid_presence() -> None:
    decision = gate_liquid_transfer(liquid_observation={"has_liquid_region": True})

    assert decision["status"] == "rejected"
    assert "only_liquid_present" in decision["reject_reasons"]


def test_gate_confirms_liquid_level_change() -> None:
    decision = gate_liquid_transfer(
        liquid_observation={
            "has_liquid_region": True,
            "liquid_level_before": 0.2,
            "liquid_level_after": 0.45,
            "liquid_level_delta": 0.25,
        },
        tool_track={"track_id": "trk_pipette"},
    )

    assert decision["status"] == "confirmed"


def test_gate_rejects_device_presence_only_panel_operation() -> None:
    decision = gate_panel_operation(device_track={"track_id": "trk_balance"})

    assert decision["status"] == "rejected"
    assert "device_presence_only" in decision["reject_reasons"]


def test_gate_confirms_panel_state_change() -> None:
    decision = gate_panel_operation(
        hand_track={"track_id": "trk_hand"},
        device_track={"track_id": "trk_balance"},
        control_roi={"x1": 0},
        display_state={"display_changed": True, "hand_in_control_roi_frames": 3},
    )

    assert decision["status"] == "confirmed"


def test_gate_rejects_container_presence_only() -> None:
    decision = gate_container_state_change(container_track={"track_id": "trk_bottle"})

    assert decision["status"] == "rejected"
    assert "container_presence_only" in decision["reject_reasons"]


def test_gate_confirms_container_cap_open_state_change() -> None:
    decision = gate_container_state_change(
        container_track={"track_id": "trk_bottle"},
        pre_state={"cap_state": "closed"},
        post_state={"cap_state": "open"},
    )

    assert decision["status"] == "confirmed"
    assert "cap_state" in decision["evidence"]["changed_fields"]


def test_container_state_change_no_track_cannot_confirm() -> None:
    decision = gate_container_state_change(
        container_track=None,
        pre_state={"cap_state": "closed"},
        post_state={"cap_state": "open"},
    )

    assert decision["status"] != "confirmed"
    assert "missing_same_container_track" in decision["reject_reasons"]


def test_container_state_change_different_container_ids_rejected() -> None:
    decision = gate_container_state_change(
        container_track={"track_id": "trk_bottle"},
        pre_state={"container_id": "bottle_a", "cap_state": "closed"},
        post_state={"container_id": "bottle_b", "cap_state": "open"},
    )

    assert decision["status"] == "rejected"
    assert "different_container_instances" in decision["reject_reasons"]


def test_container_state_change_same_track_with_changed_fields_confirmed() -> None:
    decision = gate_container_state_change(
        container_track={"track_id": "trk_bottle", "track_type": "tracker_track"},
        pre_state={"track_id": "trk_bottle", "liquid_level": 0.1},
        post_state={"track_id": "trk_bottle", "liquid_level": 0.4},
    )

    assert decision["status"] == "confirmed"
    assert "liquid_level" in decision["evidence"]["changed_fields"]


def test_container_state_change_lighting_only_rejected() -> None:
    decision = gate_container_state_change(
        container_track={"track_id": "trk_bottle"},
        pre_state={"content_color": "clear"},
        post_state={"content_color": "blue"},
        frame_pair_evidence={"lighting_change_risk": True},
    )

    assert decision["status"] == "rejected"
    assert "lighting_change_only" in decision["reject_reasons"]


def test_qwen_cannot_upgrade_non_confirmed_gate() -> None:
    merged = merge_gate_with_qwen_audit(
        {"status": "candidate", "event_type": "object_move", "confidence": 0.5, "hard_gate": {}, "evidence": {}},
        {"decision": "accept", "should_write_confirmed_event": True},
    )

    assert merged["final_status"] == "candidate"
    assert merged["should_write_confirmed_event"] is False
    assert merged["audit"]["qwen_upgrade_blocked"] is True


def test_qwen_can_downgrade_confirmed_gate() -> None:
    merged = merge_gate_with_qwen_audit(
        {"status": "confirmed", "event_type": "object_move", "confidence": 0.9, "hard_gate": {}, "evidence": {}},
        {"decision": "reject", "should_write_confirmed_event": False},
    )

    assert merged["final_status"] == "rejected_by_audit"
    assert merged["should_write_confirmed_event"] is False


def test_qwen_parse_failure_never_defaults_to_accept() -> None:
    audit = parse_qwen_audit("not-json", event_type="object_move")

    assert audit["status"] == "parse_failed"
    assert audit["decision"] == "uncertain"
    assert audit["should_write_confirmed_event"] is False


def _evidence_row(
    *,
    hand_bbox: list[int],
    object_bbox: list[int],
    interactions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "view": "front",
        "frame_width": 960,
        "frame_height": 540,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.92, "bbox": hand_bbox},
            {"label": "pipette", "confidence": 0.91, "bbox": object_bbox},
        ],
        "hand_object_interactions": interactions,
    }


def _track(
    *,
    raw: float,
    stabilized: float,
    label: str = "reagent_bottle",
    identity: float = 0.9,
    risk: float = 0.1,
) -> dict[str, Any]:
    return {
        "track_id": "trk_001",
        "track_type": "tracker_track",
        "object_label": label,
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
