from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.confirmation_loop import (
    apply_confirmation_decision,
    build_confirmation_queue,
    build_confirmation_review_summary,
    list_confirmation_queue,
    _video_events_requiring_confirmation,
)
from key_action_indexer.schemas import read_jsonl, write_jsonl


def test_confirmation_queue_and_decision_updates_process(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    process_path = metadata / "experiment_process.json"
    process_path.write_text(
        json.dumps(
            {
                "session_id": "confirm_session",
                "steps": [
                    {
                        "step_id": "step_001",
                        "name": "Pipette",
                        "status": "inferred_missing",
                        "inferred": True,
                        "completed": True,
                        "skipped": False,
                        "abnormal": False,
                        "confidence": 0.35,
                        "requires_human_confirmation": True,
                        "missing_completion_reason": "no direct observation",
                        "evidence_refs": [{"type": "asset", "asset_id": "asset_1"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = build_confirmation_queue(tmp_path)
    rows = list_confirmation_queue(tmp_path)
    result = apply_confirmation_decision(tmp_path, "confirm_session:step_001", "approved", reviewer="qa", note="matches notebook")
    updated = json.loads(process_path.read_text(encoding="utf-8"))

    assert summary["pending_count"] == 1
    assert rows[0]["confirmation_id"] == "confirm_session:step_001"
    assert result["queue_summary"]["pending_count"] == 0
    assert updated["steps"][0]["status"] == "human_confirmed"
    assert updated["steps"][0]["requires_human_confirmation"] is False
    assert updated["steps"][0]["confirmation_decision"]["reviewer"] == "qa"
    assert updated["pending_confirmation_step_ids"] == []


def test_confirmation_review_bundle_audit_and_recap_outputs(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    process_path = metadata / "experiment_process.json"
    process_path.write_text(
        json.dumps(
            {
                "session_id": "audit_session",
                "steps": [
                    {
                        "step_id": "step_002",
                        "name": "Transfer sample",
                        "expected_action": "pipetting",
                        "status": "inferred_missing",
                        "observed": False,
                        "inferred": True,
                        "completed": True,
                        "skipped": False,
                        "abnormal": False,
                        "confidence": 0.52,
                        "confidence_reasons": ["inferred from surrounding steps"],
                        "requires_human_confirmation": True,
                        "confirmation_status": "pending",
                        "missing_completion_reason": "step inferred from neighboring evidence",
                        "evidence_refs": [
                            {
                                "type": "video_event",
                                "video_event_id": "micro_001:hand_object_contact",
                                "confidence": 0.82,
                            },
                            {"type": "asset", "asset_id": "asset_peak", "path": "keyframes/micro/micro_001/peak.jpg"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    micro = {
        "session_id": "audit_session",
        "parent_segment_id": "seg_001",
        "micro_segment_id": "micro_001",
        "global_start_time": "2026-04-29T10:00:02+08:00",
        "global_end_time": "2026-04-29T10:00:05+08:00",
        "first_person": {"clip_path": "clips/micro_001_first.mp4"},
        "third_person": {"clip_path": "clips/micro_001_third.mp4"},
        "interaction": {
            "interaction_type": "hand_pipette_contact",
            "primary_object": "pipette",
            "detected_objects": ["hand", "pipette"],
            "max_interaction_score": 0.91,
        },
        "keyframes": {
            "contact_frame": "keyframes/micro/micro_001/contact.jpg",
            "peak_frame": "keyframes/micro/micro_001/peak.jpg",
            "release_frame": "keyframes/micro/micro_001/release.jpg",
        },
        "text_description": {"action_type": "pipetting", "summary": "hand contacts pipette"},
        "quality": {"confidence": "high"},
        "evidence": {"evidence_level": "visual_confirmed"},
    }
    write_jsonl(metadata / "micro_segments.jsonl", [micro])
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "micro_001:hand_object_contact",
                "session_id": "audit_session",
                "segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "event_type": "hand_object_contact",
                "global_start_time": "2026-04-29T10:00:02+08:00",
                "global_end_time": "2026-04-29T10:00:05+08:00",
                "primary_object": "pipette",
                "action_type": "pipetting",
                "confidence": 0.82,
                "confidence_reasons": ["visual_confirmed"],
                "anomaly_flags": [],
                "asset_refs": [
                    {
                        "asset_id": "asset_peak",
                        "asset_type": "keyframe",
                        "rel": "peak_frame",
                        "path": "keyframes/micro/micro_001/peak.jpg",
                    }
                ],
                "text": "hand contacts pipette",
                "payload": {"source": "micro_segment", "micro_segment": micro},
            }
        ],
    )
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "session_id": "audit_session",
                "segment_id": "seg_001",
                "global_start_time": "2026-04-29T10:00:00+08:00",
                "global_end_time": "2026-04-29T10:00:08+08:00",
                "first_person": {"clip_path": "clips/seg_001_first.mp4"},
                "third_person": {"clip_path": "clips/seg_001_third.mp4"},
                "interaction_keyframes": [
                    {
                        "path": "keyframes/seg_001/interaction_001.jpg",
                        "view": "first_person",
                        "event_id": "event_001",
                    }
                ],
            }
        ],
    )
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [
            {
                "asset_id": "asset_peak",
                "asset_type": "keyframe",
                "source_type": "micro_keyframe",
                "source_id": "micro_001",
                "segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "path": "keyframes/micro/micro_001/peak.jpg",
            }
        ],
    )

    summary = build_confirmation_queue(tmp_path)
    queue_rows = list_confirmation_queue(tmp_path)
    bundle = json.loads(Path(summary["review_bundle_path"]).read_text(encoding="utf-8"))
    item = bundle["items"][0]

    assert queue_rows[0]["evidence_summary"]["event_count"] == 1
    assert item["evidence_summary"]["hand_object_interaction_count"] == 1
    assert item["suggested_action"]["decision"] == "approved"
    assert any(ref["path"].endswith("peak.jpg") for ref in item["keyframe_refs"])
    assert any(ref["path"].endswith("micro_001_first.mp4") for ref in item["clip_refs"])

    result = apply_confirmation_decision(
        tmp_path,
        "audit_session:step_002",
        "rejected",
        reviewer="worker_d",
        note="clip does not match SOP note",
    )
    audit_rows = read_jsonl(metadata / "human_confirmation_audit_trail.jsonl")
    updated = json.loads(process_path.read_text(encoding="utf-8"))
    recap = build_confirmation_review_summary(tmp_path)

    assert result["audit"]["before_state"]["status"] == "inferred_missing"
    assert result["audit"]["after_state"]["status"] == "human_rejected"
    assert audit_rows[0]["reviewer"] == "worker_d"
    assert audit_rows[0]["note"] == "clip does not match SOP note"
    assert updated["steps"][0]["requires_human_confirmation"] is False
    assert recap["steps"][0]["latest_decision"]["decision"] == "rejected"
    assert recap["steps"][0]["last_audit"]["after_status"] == "human_rejected"
    assert recap["evidence"][0]["step_ids"] == ["step_002"]


def test_video_confirmation_queue_skips_low_signal_capability_candidates() -> None:
    context = {
        "video_events": [
            {
                "video_event_id": "micro_001:liquid_transfer_candidate",
                "event_type": "liquid_transfer_candidate",
                "conclusion_status": "candidate",
                "confidence": 0.62,
                "anomaly_flags": ["heuristic_candidate", "not_visual_liquid_flow_confirmed", "requires_human_confirmation"],
            },
            {
                "video_event_id": "micro_002:panel_conflict",
                "event_type": "equipment_panel_operation_candidate",
                "conclusion_status": "candidate",
                "confidence": 0.62,
                "anomaly_flags": ["heuristic_candidate", "conflicting_physical_event_evidence", "requires_human_confirmation"],
            },
            {
                "video_event_id": "advanced_liquid_001",
                "event_type": "liquid_flow_candidate_visual",
                "conclusion_status": "candidate",
                "confidence": 0.47,
                "anomaly_flags": ["requires_human_confirmation", "low_confidence_candidate_event"],
                "payload": {"source": "advanced_vision_evidence"},
            },
            {
                "video_event_id": "track_001",
                "event_type": "object_movement_detected",
                "conclusion_status": "measured",
                "confidence": 0.32,
                "anomaly_flags": ["requires_human_confirmation", "low_confidence_confirmed_or_measured_event"],
            },
        ]
    }

    rows = _video_events_requiring_confirmation(context)

    assert [row["video_event_id"] for row in rows] == ["micro_002:panel_conflict", "track_001"]


def test_confirmation_queue_moves_step_covered_video_events_to_machine_backlog(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "covered_video",
                "steps": [
                    {
                        "step_id": "step_001",
                        "name": "Weigh",
                        "status": "completed",
                        "observed": True,
                        "completed": True,
                        "requires_human_confirmation": True,
                        "evidence_refs": [
                            {"type": "video_event", "video_event_id": "track_001"},
                            {"type": "asset", "micro_segment_id": "micro_001"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "track_001",
                "event_type": "object_movement_detected",
                "conclusion_status": "measured",
                "confidence": 0.32,
                "anomaly_flags": ["requires_human_confirmation", "low_confidence_confirmed_or_measured_event"],
            },
            {
                "video_event_id": "track_002",
                "micro_segment_id": "micro_001",
                "event_type": "object_movement_detected",
                "conclusion_status": "measured",
                "confidence": 0.32,
                "anomaly_flags": ["requires_human_confirmation", "low_confidence_confirmed_or_measured_event"],
            }
        ],
    )

    summary = build_confirmation_queue(tmp_path)
    queue_rows = list_confirmation_queue(tmp_path)
    backlog_rows = read_jsonl(metadata / "human_confirmation_machine_backlog.jsonl")

    assert summary["pending_count"] == 1
    assert summary["standalone_video_item_count"] == 0
    assert summary["machine_backlog_count"] == 2
    assert [row["item_type"] for row in queue_rows] == ["experiment_step"]
    assert [row["item_id"] for row in backlog_rows] == ["track_001", "track_002"]
    assert {row["reason"] for row in backlog_rows} == {"covered_by_step_review"}


def test_confirmation_queue_replays_existing_decisions_into_process(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    process_path = metadata / "experiment_process.json"
    process_path.write_text(
        json.dumps(
            {
                "session_id": "replay_session",
                "steps": [
                    {
                        "step_id": "step_001",
                        "name": "Pipette",
                        "status": "inferred_missing",
                        "inferred": True,
                        "completed": True,
                        "requires_human_confirmation": True,
                        "confirmation_status": "pending",
                        "evidence_refs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "human_confirmation_decisions.jsonl",
        [
            {
                "confirmation_id": "replay_session:step_001",
                "decision": "approved",
                "reviewer": "qa",
                "note": "replayed after process rebuild",
                "decided_at": "2026-04-29T10:00:00+08:00",
            }
        ],
    )

    summary = build_confirmation_queue(tmp_path)
    updated = json.loads(process_path.read_text(encoding="utf-8"))
    queue_rows = list_confirmation_queue(tmp_path)

    assert summary["pending_count"] == 0
    assert queue_rows == []
    assert updated["steps"][0]["status"] == "human_confirmed"
    assert updated["steps"][0]["requires_human_confirmation"] is False
    assert updated["steps"][0]["confirmation_decision"]["reviewer"] == "qa"
    assert updated["pending_confirmation_step_ids"] == []
