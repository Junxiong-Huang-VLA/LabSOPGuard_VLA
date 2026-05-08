from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.schemas import write_jsonl
from key_action_indexer.video_understanding import build_video_understanding, load_video_understanding


def test_build_video_understanding_events_confidence_and_anomalies(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "interaction": {
                    "interaction_type": "hand_pipette_contact",
                    "primary_object": "pipette",
                    "detected_objects": ["hand", "pipette", "tube"],
                    "max_interaction_score": 0.82,
                    "avg_interaction_score": 0.71,
                },
                "keyframes": {"contact_frame": "keyframes/contact.jpg", "peak_frame": "keyframes/peak.jpg"},
                "third_person": {"clip_path": "clips/micro.mp4"},
                "text_description": {"action_type": "pipetting", "summary": "transfer 200 微升 liquid"},
                "quality": {"confidence": "high"},
                "evidence": {"evidence_level": "transcript_supported", "limitations": ["missing pipette or tube visual evidence"]},
            }
        ],
    )
    write_jsonl(
        metadata / "state_change_index.jsonl",
        [
            {
                "state_change_id": "state_contact",
                "micro_segment_id": "micro_001",
                "state_type": "contact_started",
                "asset_refs": [{"asset_id": "asset_contact", "asset_type": "keyframe", "path": "keyframes/contact.jpg"}],
            },
            {"state_change_id": "state_release", "micro_segment_id": "micro_001", "state_type": "contact_released", "asset_refs": []},
        ],
    )
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [
            {
                "asset_id": "asset_contact",
                "asset_type": "keyframe",
                "source_id": "micro_001",
                "path": "keyframes/contact.jpg",
                "quality": {"status": "present"},
            }
        ],
    )
    write_jsonl(
        metadata / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "adv_move",
                "session_id": "s1",
                "evidence_type": "object_trajectory_movement",
                "micro_segment_id": "micro_001",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "object_label": "pipette",
                "action_type": "pipetting",
                "visual_confirmation_level": "trajectory_confirmed",
                "confidence": 0.82,
                "evidence_reasons": ["tracked pipette"],
                "limitations": [],
                "asset_refs": [{"asset_id": "asset_contact", "path": "keyframes/contact.jpg"}],
            }
        ],
    )

    summary = build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    event_types = {row["event_type"] for row in rows}

    assert summary["video_event_count"] == len(rows)
    assert {"hand_object_contact", "liquid_transfer_candidate", "experiment_action_classification", "object_movement_detected"}.issubset(event_types)
    movement = next(row for row in rows if row["event_type"] == "object_movement_detected")
    assert movement["confidence"] == 0.82
    assert "advanced_vision_level=trajectory_confirmed" in movement["confidence_reasons"]
    liquid = next(row for row in rows if row["event_type"] == "liquid_transfer_candidate")
    assert liquid["confidence"] <= 0.62
    assert "not_visual_liquid_flow_confirmed" in liquid["anomaly_flags"]
    assert "transcript_supported_without_strong_visual" in liquid["anomaly_flags"]
    assert any(ref.get("asset_id") == "asset_contact" for ref in liquid["asset_refs"])
    assert json.loads((metadata / "video_understanding_summary.json").read_text(encoding="utf-8"))["event_type_counts"]["hand_object_contact"] == 1


def test_video_understanding_fuses_panel_container_color_level_and_entities(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s_panel",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 4.0,
                "global_start_time": "2026-05-03T09:00:01+08:00",
                "global_end_time": "2026-05-03T09:00:04+08:00",
                "interaction": {"primary_object": "balance_panel", "max_interaction_score": 0.8},
                "text_description": {"action_type": "recording", "summary": "record balance readout"},
                "keyframes": {"peak_frame": "keyframes/panel.jpg"},
                "evidence": {"evidence_level": "visual_confirmed"},
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    write_jsonl(
        metadata / "equipment_panel_ocr.jsonl",
        [
            {
                "event_id": "ocr_001",
                "session_id": "s_panel",
                "micro_segment_id": "micro_001",
                "equipment_label": "balance_panel",
                "display_text": "12.4 g",
                "confidence": 0.91,
            }
        ],
    )
    write_jsonl(
        metadata / "button_knob_switch_states.jsonl",
        [
            {
                "event_id": "switch_001",
                "session_id": "s_panel",
                "micro_segment_id": "micro_001",
                "equipment_label": "balance_panel",
                "switch_state": "on",
                "knob_angle_deg": 45,
                "confidence": 0.84,
            }
        ],
    )
    write_jsonl(
        metadata / "container_open_close_events.jsonl",
        [
            {
                "event_id": "container_001",
                "session_id": "s_panel",
                "micro_segment_id": "micro_001",
                "container_label": "tube_A",
                "before_state": "closed",
                "after_state": "open",
                "confidence": 0.86,
            }
        ],
    )
    write_jsonl(
        metadata / "container_color_events.jsonl",
        [
            {
                "event_id": "color_001",
                "session_id": "s_panel",
                "micro_segment_id": "micro_001",
                "container_label": "tube_A",
                "color_before": "clear",
                "color_after": "blue",
                "color_delta": 0.4,
                "confidence": 0.82,
            }
        ],
    )
    write_jsonl(
        metadata / "container_liquid_level_events.jsonl",
        [
            {
                "event_id": "level_001",
                "session_id": "s_panel",
                "micro_segment_id": "micro_001",
                "container_label": "tube_A",
                "liquid_level_before": 0.2,
                "liquid_level_after": 0.5,
                "confidence": 0.9,
            }
        ],
    )

    summary = build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    event_types = {row["event_type"] for row in rows}

    assert summary["input_counts"]["model_observation_inputs"]["equipment_panel_state"] == 2
    assert summary["input_counts"]["model_observation_inputs"]["container_state"] == 3
    assert {
        "equipment_panel_operation_detected",
        "container_state_change_detected",
        "container_color_change_detected",
        "liquid_level_change_detected",
    }.issubset(event_types)
    panel = next(row for row in rows if row["event_type"] == "equipment_panel_operation_detected" and row["conclusion_status"] == "measured")
    assert panel["object_category"] == "equipment_control"
    assert panel["normalized_object"]["canonical_label"] == "equipment_panel"
    assert panel["extracted_entities"]["equipment"] == ["equipment_panel"]
    assert any(ref["type"] == "model_observation_event" for ref in panel["evidence_refs"])
    level = next(row for row in rows if row["event_type"] == "liquid_level_change_detected")
    assert level["conclusion_status"] == "measured"
    assert any(param["name"] == "liquid_level_y_norm" or param["name"] == "liquid_level_after" for param in level["extracted_entities"]["parameters"])


def test_video_understanding_downgrades_confirmed_without_evidence(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(
        metadata / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "unsupported_panel",
                "session_id": "s1",
                "evidence_type": "equipment_control_change",
                "micro_segment_id": "micro_unsupported",
                "object_label": "balance_panel",
                "visual_confirmation_level": "equipment_panel_state_confirmed",
                "confidence": 0.88,
                "evidence_reasons": [],
                "limitations": [],
                "metrics": {},
                "asset_refs": [],
            }
        ],
    )

    build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    event = rows[0]

    assert event["event_type"] == "equipment_panel_operation_candidate"
    assert event["conclusion_status"] == "candidate"
    assert event["confidence"] <= 0.62
    assert "confirmed_without_model_or_visual_evidence" in event["anomaly_flags"]
    assert "requires_human_confirmation" in event["anomaly_flags"]


def test_video_understanding_flags_conflicting_container_states(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s_conflict",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "global_start_time": "2026-05-03T09:00:01+08:00",
                "global_end_time": "2026-05-03T09:00:02+08:00",
                "interaction": {"primary_object": "tube_A"},
                "text_description": {"action_type": "sample_handling"},
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    write_jsonl(
        metadata / "container_state_events.jsonl",
        [
            {
                "event_id": "container_open",
                "session_id": "s_conflict",
                "micro_segment_id": "micro_001",
                "container_label": "tube_A",
                "after_state": "open",
                "confidence": 0.87,
            },
            {
                "event_id": "container_closed",
                "session_id": "s_conflict",
                "micro_segment_id": "micro_001",
                "container_label": "tube_A",
                "after_state": "closed",
                "confidence": 0.86,
            },
        ],
    )

    build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    container_rows = [row for row in rows if row["event_type"] == "container_state_change_detected"]

    assert len(container_rows) == 2
    assert all("conflicting_physical_event_evidence" in row["anomaly_flags"] for row in container_rows)
    assert all("requires_human_confirmation" in row["anomaly_flags"] for row in container_rows)


def test_video_understanding_rolls_up_adjacent_candidates_without_merging_confirmed(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(
        metadata / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "candidate_panel_1",
                "session_id": "s_rollup",
                "segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "evidence_type": "equipment_control_change",
                "object_label": "balance_panel",
                "action_type": "recording",
                "visual_confirmation_level": "candidate_requires_panel_ocr_or_control_detector",
                "global_start_time": "2026-05-03T09:00:01+08:00",
                "global_end_time": "2026-05-03T09:00:02+08:00",
                "confidence": 0.55,
                "evidence_reasons": ["panel context"],
                "limitations": ["requires OCR confirmation"],
                "asset_refs": [{"asset_id": "asset_1", "path": "panel_1.jpg"}],
            },
            {
                "evidence_id": "candidate_panel_2",
                "session_id": "s_rollup",
                "segment_id": "seg_001",
                "micro_segment_id": "micro_002",
                "evidence_type": "equipment_control_change",
                "object_label": "balance_panel",
                "action_type": "recording",
                "visual_confirmation_level": "candidate_requires_panel_ocr_or_control_detector",
                "global_start_time": "2026-05-03T09:00:03+08:00",
                "global_end_time": "2026-05-03T09:00:04+08:00",
                "confidence": 0.61,
                "evidence_reasons": ["readout-like region"],
                "limitations": ["requires OCR confirmation"],
                "asset_refs": [{"asset_id": "asset_2", "path": "panel_2.jpg"}],
            },
            {
                "evidence_id": "confirmed_move",
                "session_id": "s_rollup",
                "segment_id": "seg_001",
                "micro_segment_id": "micro_002",
                "evidence_type": "object_trajectory_movement",
                "object_label": "balance_panel",
                "action_type": "recording",
                "visual_confirmation_level": "trajectory_confirmed",
                "global_start_time": "2026-05-03T09:00:03+08:00",
                "global_end_time": "2026-05-03T09:00:04+08:00",
                "confidence": 0.82,
                "evidence_reasons": ["tracked object"],
                "asset_refs": [{"asset_id": "asset_3", "path": "move.jpg"}],
            },
        ],
    )

    summary = build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    rollup_rows = [row for row in rows if "candidate_rollup" in row["anomaly_flags"]]

    assert summary["video_event_count"] == 2
    assert summary["candidate_rollup"]["input_candidate_count"] == 2
    assert summary["candidate_rollup"]["output_candidate_count"] == 1
    assert summary["candidate_rollup"]["rollup_group_count"] == 1
    assert len(rollup_rows) == 1
    rollup = rollup_rows[0]
    assert rollup["conclusion_status"] == "candidate"
    assert rollup["event_type"] == "equipment_panel_operation_candidate"
    assert {ref["asset_id"] for ref in rollup["asset_refs"]} == {"asset_1", "asset_2"}
    assert {
        ref["video_event_id"]
        for ref in rollup["evidence_refs"]
        if ref.get("type") == "video_event"
    } == {
        "candidate_panel_1:equipment_panel_operation_candidate",
        "candidate_panel_2:equipment_panel_operation_candidate",
    }
    assert set(rollup["payload"]["rollup"]["source_micro_segment_ids"]) == {"micro_001", "micro_002"}
    assert any(row["event_type"] == "object_movement_detected" and row["conclusion_status"] == "confirmed" for row in rows)


def test_video_understanding_rolls_up_same_micro_cross_family_candidates(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "micro_segments.jsonl", [])
    base = {
        "session_id": "s_cross_family",
        "segment_id": "seg_001",
        "micro_segment_id": "micro_001",
        "object_label": "pipette",
        "action_type": "pipetting",
        "global_start_time": "2026-05-03T09:00:01+08:00",
        "global_end_time": "2026-05-03T09:00:02+08:00",
        "confidence": 0.58,
        "evidence_reasons": ["same micro candidate evidence"],
        "limitations": ["requires confirmation"],
    }
    write_jsonl(
        metadata / "advanced_vision_evidence.jsonl",
        [
            {
                **base,
                "evidence_id": "candidate_panel",
                "evidence_type": "equipment_control_change",
                "visual_confirmation_level": "candidate_requires_panel_ocr_or_control_detector",
                "asset_refs": [{"asset_id": "asset_panel", "path": "panel.jpg"}],
            },
            {
                **base,
                "evidence_id": "candidate_liquid",
                "evidence_type": "liquid_flow_candidate_visual",
                "visual_confirmation_level": "candidate_visual_liquid_flow",
                "asset_refs": [{"asset_id": "asset_liquid", "path": "liquid.jpg"}],
            },
            {
                **base,
                "evidence_id": "candidate_container",
                "evidence_type": "container_open_close",
                "visual_confirmation_level": "candidate_requires_container_state_detector",
                "asset_refs": [{"asset_id": "asset_container", "path": "container.jpg"}],
            },
            {
                **base,
                "evidence_id": "confirmed_motion",
                "evidence_type": "object_trajectory_movement",
                "visual_confirmation_level": "trajectory_confirmed",
                "confidence": 0.82,
                "asset_refs": [{"asset_id": "asset_motion", "path": "motion.jpg"}],
            },
        ],
    )

    summary = build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    rollup_rows = [row for row in rows if "candidate_cross_family_rollup" in row["anomaly_flags"]]

    assert summary["video_event_count"] == 2
    assert summary["candidate_rollup"]["input_candidate_count"] == 3
    assert summary["candidate_rollup"]["output_candidate_count"] == 1
    assert summary["candidate_rollup"]["primary_group_count"] == 0
    assert summary["candidate_rollup"]["cross_family_group_count"] == 1
    assert len(rollup_rows) == 1
    rollup = rollup_rows[0]
    assert rollup["conclusion_status"] == "candidate"
    assert set(rollup["payload"]["rollup"]["source_event_families"]) == {
        "container_state",
        "equipment_panel",
        "liquid_flow",
    }
    assert {ref["asset_id"] for ref in rollup["asset_refs"]} == {"asset_panel", "asset_liquid", "asset_container"}
    assert any(row["event_type"] == "object_movement_detected" and row["conclusion_status"] == "confirmed" for row in rows)


def test_video_understanding_bundles_weak_same_micro_candidates_across_actions(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "micro_segments.jsonl", [])
    base = {
        "session_id": "s_weak_bundle",
        "segment_id": "seg_001",
        "micro_segment_id": "micro_001",
        "object_label": "paper",
        "global_start_time": "2026-05-03T09:00:01+08:00",
        "global_end_time": "2026-05-03T09:00:02+08:00",
        "confidence": 0.58,
        "evidence_reasons": ["weak same-micro hypothesis"],
        "limitations": ["requires confirmation"],
    }
    write_jsonl(
        metadata / "advanced_vision_evidence.jsonl",
        [
            {
                **base,
                "evidence_id": "candidate_panel",
                "evidence_type": "equipment_control_change",
                "action_type": "recording",
                "visual_confirmation_level": "candidate_requires_panel_ocr_or_control_detector",
            },
            {
                **base,
                "evidence_id": "candidate_liquid",
                "evidence_type": "liquid_flow_candidate_visual",
                "action_type": "pipetting",
                "visual_confirmation_level": "candidate_visual_liquid_flow",
            },
            {
                **base,
                "evidence_id": "candidate_container",
                "evidence_type": "container_open_close",
                "action_type": "open container",
                "visual_confirmation_level": "candidate_requires_container_state_detector",
            },
            {
                **base,
                "evidence_id": "confirmed_motion",
                "evidence_type": "object_trajectory_movement",
                "action_type": "moving",
                "visual_confirmation_level": "trajectory_confirmed",
                "confidence": 0.82,
                "asset_refs": [{"asset_id": "asset_motion", "path": "motion.jpg"}],
            },
        ],
    )

    summary = build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")
    bundle_rows = [row for row in rows if "candidate_weak_bundle_rollup" in row["anomaly_flags"]]

    assert summary["video_event_count"] == 2
    assert summary["candidate_rollup"]["input_candidate_count"] == 3
    assert summary["candidate_rollup"]["output_candidate_count"] == 1
    assert summary["candidate_rollup"]["weak_bundle_group_count"] == 1
    assert len(bundle_rows) == 1
    bundle = bundle_rows[0]
    assert bundle["conclusion_status"] == "candidate"
    assert set(bundle["payload"]["rollup"]["source_action_families"]) == {
        "container_state",
        "liquid_transfer",
        "weighing_or_readout",
    }
    assert any(row["event_type"] == "object_movement_detected" and row["conclusion_status"] == "confirmed" for row in rows)


def test_video_understanding_compresses_segment_level_backfill_micro(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s_backfill",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "seg_001_micro_001",
                "global_start_time": "2026-05-08T10:00:00+08:00",
                "global_end_time": "2026-05-08T10:00:03+08:00",
                "interaction": {"primary_object": "balance", "detected_objects": ["gloved_hand", "balance"]},
                "text_description": {"action_type": "weighing", "summary": "segment backfill"},
                "quality": {"confidence": "low", "warnings": ["segment_level_retrieval_backfill"]},
                "evidence": {
                    "evidence_level": "weak_visual_evidence",
                    "segment_level_coverage_backfill": True,
                    "force_retrieval_candidate": True,
                    "limitations": ["retrieval only"],
                },
                "keyframes": {"peak_frame": "peak.jpg"},
            }
        ],
    )

    summary = build_video_understanding(tmp_path)
    rows = load_video_understanding(metadata / "video_understanding.jsonl")

    assert summary["video_event_count"] == 1
    assert rows[0]["event_type"] == "segment_level_retrieval_candidate"
    assert rows[0]["conclusion_status"] == "candidate"
    assert "segment_level_retrieval_backfill" in rows[0]["anomaly_flags"]
