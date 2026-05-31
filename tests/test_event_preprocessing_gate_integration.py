from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABSOPGUARD_SRC = ROOT / "LabSOPGuard" / "src"
if str(LABSOPGUARD_SRC) not in sys.path:
    sys.path.insert(0, str(LABSOPGUARD_SRC))

from labsopguard.event_preprocessing.engine import EventPreprocessingEngine  # noqa: E402
from labsopguard.event_preprocessing.event_segmentation import EventSegmenter  # noqa: E402
from labsopguard.event_preprocessing.physical_event_gate import gate_object_move, summarize_gate_decisions  # noqa: E402
from labsopguard.event_preprocessing.schemas import EventProposal  # noqa: E402


def test_rejected_formal_proposal_is_not_sent_to_final_events(tmp_path: Path) -> None:
    proposal = _proposal("object_move")
    decision = gate_object_move(
        track={
            "track_id": "trk_bottle",
            "track_type": "tracker_track",
            "object_label": "reagent_bottle",
            "point_count": 6,
            "identity_confidence": 0.9,
            "id_switch_risk": 0.1,
            "median_bbox_size": 80.0,
            "raw_displacement_px": 4.0,
            "path_length_px": 4.0,
            "stabilized_displacement_px": 4.0,
            "motion_persistent": True,
            "can_confirm_motion": True,
        }
    )

    EventPreprocessingEngine._apply_gate_to_proposal(proposal, decision)
    confirmed = [item for item in [proposal] if item.status == "confirmed"]
    events = EventSegmenter().segment(
        confirmed,
        experiment_id="exp1",
        source_video_id="video1",
        video_duration_sec=10.0,
        experiment_name="实验",
    )

    assert proposal.status == "rejected"
    assert "displacement_below_threshold" in proposal.reject_reasons
    assert events == []

    summary = summarize_gate_decisions([decision])
    EventPreprocessingEngine._write_gate_artifacts(tmp_path, [decision], [EventPreprocessingEngine._rejected_candidate_row(proposal, decision)], summary)
    assert json.loads((tmp_path / "physical_event_gate_summary.json").read_text(encoding="utf-8"))["rejected"] == 1
    assert "displacement_below_threshold" in (tmp_path / "rejected_physical_event_candidates.jsonl").read_text(encoding="utf-8")


def test_confirmed_formal_proposal_preserves_gate_fields_in_physical_event() -> None:
    proposal = _proposal("object_move")
    decision = gate_object_move(
        track={
            "track_id": "trk_bottle",
            "track_type": "tracker_track",
            "object_label": "reagent_bottle",
            "point_count": 6,
            "identity_confidence": 0.92,
            "id_switch_risk": 0.05,
            "median_bbox_size": 80.0,
            "raw_displacement_px": 38.0,
            "path_length_px": 40.0,
            "stabilized_displacement_px": 36.0,
            "motion_persistent": True,
            "can_confirm_motion": True,
        },
        scene_motion={"method": "feature_homography", "background_shift_px": 2.0},
    )

    EventPreprocessingEngine._apply_gate_to_proposal(proposal, decision)
    events = EventSegmenter().segment(
        [proposal],
        experiment_id="exp1",
        source_video_id="video1",
        video_duration_sec=10.0,
        experiment_name="实验",
    )

    assert proposal.status == "confirmed"
    assert len(events) == 1
    payload = events[0].to_dict()
    assert payload["status"] == "confirmed"
    assert payload["hard_gate"]["gate_name"] == "gate_object_move"
    assert payload["evidence_detail"]["stabilized_displacement_px"] == 36.0


def _proposal(event_type: str) -> EventProposal:
    return EventProposal(
        proposal_id="proposal_001",
        event_type=event_type,
        start_frame_idx=1,
        end_frame_idx=6,
        start_time_sec=1.0,
        end_time_sec=3.0,
        evidence_frame_indices=[1, 2, 3, 4, 5, 6],
        involved_objects=["reagent_bottle"],
        dominant_object="reagent_bottle",
        involved_track_ids=["trk_bottle"],
        primary_track_id="trk_bottle",
        source_container=None,
        target_container=None,
        track_motion_summary={},
        actor_track_id="trk_hand",
        tool_track_id=None,
        related_tracks=["trk_bottle", "trk_hand"],
        transfer_mode=None,
        action_resolution_source="unit_test",
        action_resolution_notes="unit test",
        supporting_relation_ids=[],
        direction_confidence=None,
        direction_status=None,
        direction_evidence=[],
        state_before=None,
        state_after=None,
        state_change_type=None,
        state_confidence=None,
        state_evidence=[],
        evidence_grade="strong",
        review_status="auto_confirmed",
        evidence_summary="unit test",
        related_detection_classes=["gloved_hand", "reagent_bottle"],
        confidence=0.9,
    )
