from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.artifact_schema import validate_artifact_file
from key_action_indexer.confirmation_loop import build_confirmation_queue
from key_action_indexer.process_reasoner import build_experiment_process
from key_action_indexer.process_record import build_process_record
from key_action_indexer.schemas import write_jsonl


def test_process_record_exports_inferred_provenance_bidirectional_index_and_audit(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "record_session", "procedure_candidates": []}),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "evt_weigh",
                "session_id": "record_session",
                "segment_id": "seg_weigh",
                "micro_segment_id": "micro_weigh",
                "event_type": "experiment_action_classification",
                "action_type": "weighing",
                "global_start_time": "2026-05-03T10:00:00+08:00",
                "global_end_time": "2026-05-03T10:00:04+08:00",
                "confidence": 0.88,
                "asset_refs": [{"asset_id": "asset_weigh_peak", "path": "keyframes/weigh_peak.jpg"}],
                "anomaly_flags": [],
                "text": "sample weighed on balance",
                "payload": {"source": "test"},
            },
            {
                "video_event_id": "evt_record",
                "session_id": "record_session",
                "segment_id": "seg_record",
                "micro_segment_id": "micro_record",
                "event_type": "experiment_action_classification",
                "action_type": "recording",
                "global_start_time": "2026-05-03T10:00:20+08:00",
                "global_end_time": "2026-05-03T10:00:22+08:00",
                "confidence": 0.84,
                "asset_refs": [],
                "anomaly_flags": [],
                "text": "record balance readout",
                "payload": {"source": "test"},
            },
        ],
    )
    write_jsonl(
        metadata / "state_change_index.jsonl",
        [
            {
                "state_change_id": "state_weigh_peak",
                "segment_id": "seg_weigh",
                "micro_segment_id": "micro_weigh",
                "state_type": "peak_interaction",
                "global_time": "2026-05-03T10:00:02+08:00",
                "confidence": 0.8,
                "asset_refs": [{"asset_id": "asset_weigh_peak", "path": "keyframes/weigh_peak.jpg"}],
            }
        ],
    )
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [
            {
                "asset_id": "asset_weigh_peak",
                "session_id": "record_session",
                "asset_type": "keyframe",
                "path": "keyframes/weigh_peak.jpg",
                "exists": False,
                "size_bytes": 0,
                "dry_run_placeholder": True,
                "source_type": "micro_keyframe",
                "source_id": "micro_weigh",
                "segment_id": "seg_weigh",
                "micro_segment_id": "micro_weigh",
                "objects": ["balance"],
                "actions": ["weighing"],
                "state_tags": ["peak_interaction"],
                "evidence_level": "visual_confirmed",
                "search_text": "balance weighing keyframe",
                "quality": {},
                "payload_ref": {},
            }
        ],
    )
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "micro_weigh",
                "parent_segment_id": "seg_weigh",
                "first_person": {"clip_path": "clips/micro_weigh_first.mp4"},
                "third_person": {"clip_path": "clips/micro_weigh_third.mp4"},
                "keyframes": {"peak_frame": "keyframes/weigh_peak.jpg"},
            }
        ],
    )
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_weigh",
                "first_person": {"clip_path": "clips/seg_weigh_first.mp4"},
                "third_person": {"clip_path": "clips/seg_weigh_third.mp4"},
            }
        ],
    )
    (metadata / "history_model.json").write_text(
        json.dumps(
            {
                "session_count": 3,
                "action_counts": {"weighing": 3, "pipetting": 3, "recording": 3},
                "transition_probabilities": {"weighing": {"pipetting": 1.0}, "pipetting": {"recording": 1.0}},
            }
        ),
        encoding="utf-8",
    )
    sop = tmp_path / "sop.json"
    sop.write_text(
        json.dumps(
            {
                "steps": [
                    {"step_id": "weigh", "name": "Weigh sample", "expected_action": "weighing"},
                    {"step_id": "pipette", "name": "Pipette sample", "expected_action": "pipetting"},
                    {"step_id": "record", "name": "Record result", "expected_action": "recording"},
                ]
            }
        ),
        encoding="utf-8",
    )

    process = build_experiment_process(tmp_path, sop_path=sop)
    build_confirmation_queue(tmp_path)
    record = build_process_record(tmp_path)
    validation = validate_artifact_file(tmp_path / "exports" / "process_record.json", "process_record")

    process_steps = {step["step_id"]: step for step in process["steps"]}
    record_steps = {step["step_id"]: step for step in record["steps"]}

    assert process_steps["pipette"]["inferred"] is True
    assert process_steps["pipette"]["inference_source"] == ["surrounding_observed_steps", "sop_order", "history_prior"]
    assert process_steps["pipette"]["inference_reason"]
    assert process_steps["pipette"]["inference_confidence"] == process_steps["pipette"]["confidence"]
    assert record_steps["pipette"]["inference"]["evidence_ids"]
    assert "history:pipette:1" in record["evidence_index"]
    assert "pipette" in record["evidence_index"]["inference:pipette"]
    assert record_steps["weigh"]["evidence_summary"]["has_keyframe_trace"] is True
    assert record_steps["weigh"]["evidence_summary"]["has_video_trace"] is True
    assert record_steps["pipette"]["confirmation"]["policy"] == "manual_review"
    assert validation["valid"] is True
    assert (tmp_path / "reports" / "process_audit_report.md").exists()
