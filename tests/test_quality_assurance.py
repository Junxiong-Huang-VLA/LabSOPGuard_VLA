from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.quality_assurance import build_quality_assurance_report
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.scope_config import build_stage_scope


def test_quality_assurance_report_scores_artifacts(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "t1", "event_type": "segment", "session_id": "s1"}])
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1"}])
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "micro_1",
                "keyframes": {
                    "contact_frame": "keyframes/contact.jpg",
                    "peak_frame": "keyframes/peak.jpg",
                    "release_frame": "keyframes/release.jpg",
                },
            }
        ],
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {"video_event_id": "v1", "event_type": "experiment_action_classification", "action_type": "weighing", "confidence": 0.9},
            {"video_event_id": "v2", "event_type": "object_state_change", "confidence": 0.8},
        ],
    )
    write_jsonl(metadata / "material_asset_catalog.jsonl", [{"asset_id": "a1", "asset_type": "keyframe", "search_text": "weighing"}])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"embedding_id": "e1", "index_text": "weighing"}])
    write_jsonl(metadata / "human_confirmation_queue.jsonl", [])
    (metadata / "time_calibration_report.json").write_text(
        json.dumps({"sources": {}, "event_count": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "s1", "procedure_candidates": [{"action_type": "weighing"}], "materials": [{"name": "sample"}], "parameters": []}),
        encoding="utf-8",
    )
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "current_step_id": "step_1",
                "next_step_id": None,
                "steps": [
                    {
                        "step_id": "step_1",
                        "observed": True,
                        "completed": True,
                        "confidence": 0.9,
                        "requires_human_confirmation": False,
                        "evidence_refs": [{"type": "video_event", "video_event_id": "v1"}],
                    }
                ],
                "evidence_index": {"v1": ["step_1"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata / "advanced_vision_evidence_summary.json").write_text(
        json.dumps({"visual_confirmation_level_counts": {"trajectory_confirmed": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metadata / "model_inventory.json").write_text(
        json.dumps({"model_count": 1, "dataset_count": 1, "capabilities": {"object_detection": {"available": True}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metadata / "history_model.json").write_text(
        json.dumps({"event_count": 2, "transition_counts": {"weighing->recording": 1}}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = build_quality_assurance_report(tmp_path)

    assert report["session_id"] == "s1"
    assert report["artifact_counts"]["video_events"] == 2
    assert report["status_counts"]["pass"] >= 10
    assert "next_round_scheduler" in report
    assert all("blocking_tasks" in check for check in report["checks"])
    assert all("suggested_commands" in check for check in report["checks"])
    assert all("required_inputs" in check for check in report["checks"])
    assert (metadata / "process_quality_report.json").exists()


def test_quality_assurance_maps_needs_review_checks_to_scheduler_tasks(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "t1", "event_type": "segment", "session_id": "s1"}])
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1"}])
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {"video_event_id": "v1", "event_type": "experiment_action_classification", "action_type": "weighing", "confidence": 0.4},
            {"video_event_id": "v2", "event_type": "object_state_change", "confidence": 0.8},
        ],
    )
    write_jsonl(metadata / "material_asset_catalog.jsonl", [{"asset_id": "a1", "asset_type": "keyframe", "search_text": "weighing"}])
    write_jsonl(metadata / "vector_metadata.jsonl", [])
    write_jsonl(
        metadata / "human_confirmation_queue.jsonl",
        [{"confirmation_id": "s1:step_1", "item_id": "step_1", "status": "pending"}],
    )
    (metadata / "time_calibration_report.json").write_text(
        json.dumps({"sources": {"transcript": {"input_event_count": 1, "anchor_count": 0, "residual_max_abs_sec": 0.0}}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "s1", "procedure_candidates": [{"action_type": "weighing"}], "materials": [{"name": "sample"}], "parameters": []}),
        encoding="utf-8",
    )
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "current_step_id": None,
                "next_step_id": None,
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "weigh sample",
                        "observed": True,
                        "completed": True,
                        "requires_human_confirmation": True,
                        "evidence_refs": [],
                    },
                    {
                        "step_id": "recording",
                        "name": "record reading",
                        "status": "not_observed",
                        "observed": False,
                        "completed": False,
                        "inferred": True,
                        "requires_human_confirmation": False,
                    },
                ],
                "evidence_index": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata / "advanced_vision_evidence_summary.json").write_text(
        json.dumps({"visual_confirmation_level_counts": {"candidate": 2}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (metadata / "model_inventory.json").write_text(
        json.dumps(
            {
                "model_count": 1,
                "dataset_count": 1,
                "capabilities": {
                    "object_detection": {"available": True},
                    "liquid_stream_segmentation": {"available": False},
                    "equipment_control_state_detection": {"available": False},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata / "capability_gap_report.json").write_text(
        json.dumps(
            {
                "gap_count": 2,
                "missing_capabilities": ["liquid_stream_segmentation", "equipment_control_state_detection"],
                "recommended_labels": ["liquid", "stream", "button", "knob", "display"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata / "history_model.json").write_text(json.dumps({"event_count": 0, "transition_counts": {}}, ensure_ascii=False), encoding="utf-8")

    report = build_quality_assurance_report(tmp_path)
    needs_review = [check for check in report["checks"] if check["status"] == "needs_review"]

    assert needs_review
    assert all(check["blocking_tasks"] for check in needs_review)
    assert all(check["suggested_commands"] for check in needs_review)
    assert all(check["required_inputs"] for check in needs_review)
    assert all(str(task["task_id"]).startswith("P-") for check in needs_review for task in check["blocking_tasks"])
    assert report["artifact_counts"]["capability_gap_reports"] == 1
    assert report["artifact_counts"]["capability_gap_items"] == 2

    step_check = next(check for check in report["checks"] if check["check_id"] == "step_reasoning")
    assert step_check["details"]["pending_confirmation_ids"] == ["s1:step_1"]
    assert "confirmation_id:s1:step_1" in step_check["required_inputs"]

    completion_check = next(check for check in report["checks"] if check["check_id"] == "process_completion")
    assert "step_id:recording" in completion_check["required_inputs"]

    model_check = next(check for check in report["checks"] if check["check_id"] == "model_coverage")
    assert model_check["details"]["capability_gap_report_present"] is True
    assert "liquid_stream_segmentation" in model_check["details"]["reported_missing_capabilities"]
    assert "recommended_label:button" in model_check["required_inputs"]

    scheduler = report["next_round_scheduler"]
    assert scheduler["task_count"] >= len(needs_review)
    assert any(task["source_check_id"] == "model_coverage" for task in scheduler["tasks"])


def test_quality_assurance_scope_marks_deferred_capabilities_nonblocking(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "t1", "event_type": "segment", "session_id": "s1"}])
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1"}])
    write_jsonl(metadata / "micro_segments.jsonl", [{"micro_segment_id": "m1", "parent_segment_id": "seg_1", "keyframes": {"peak_frame": "peak.jpg"}}])
    write_jsonl(metadata / "video_understanding.jsonl", [{"video_event_id": "v1", "event_type": "experiment_action_classification", "confidence": 0.9}])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [{"asset_id": "a1", "asset_type": "keyframe", "search_text": "weighing"}])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"embedding_id": "e1", "index_text": "weighing"}])
    write_jsonl(metadata / "human_confirmation_queue.jsonl", [])
    (metadata / "time_calibration_report.json").write_text(json.dumps({"sources": {}, "event_count": 1}), encoding="utf-8")
    (metadata / "experiment_context.json").write_text(json.dumps({"session_id": "s1", "procedure_candidates": [{"action_type": "weighing"}], "materials": [{"name": "sample"}]}), encoding="utf-8")
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "current_step_id": "step_1",
                "steps": [{"step_id": "step_1", "completed": True, "requires_human_confirmation": False, "evidence_refs": [{"type": "video_event", "video_event_id": "v1"}]}],
                "evidence_index": {"v1": ["step_1"]},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "advanced_vision_evidence_summary.json").write_text(json.dumps({"visual_confirmation_level_counts": {"trajectory_confirmed": 1}}), encoding="utf-8")
    (metadata / "model_inventory.json").write_text(
        json.dumps(
            {
                "model_count": 1,
                "dataset_count": 1,
                "capabilities": {
                    "object_detection": {"available": True},
                    "liquid_stream_segmentation": {"available": False},
                    "equipment_control_state_detection": {"available": False},
                },
            }
        ),
        encoding="utf-8",
    )
    (metadata / "capability_gap_report.json").write_text(
        json.dumps(
            {
                "missing_capabilities": ["liquid_stream_segmentation", "equipment_control_state_detection"],
                "recommended_labels": ["liquid", "button", "balance"],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "history_model.json").write_text(json.dumps({"event_count": 2, "transition_counts": {"a->b": 1}}), encoding="utf-8")
    build_stage_scope(tmp_path)

    report = build_quality_assurance_report(tmp_path)
    model_check = next(check for check in report["checks"] if check["check_id"] == "model_coverage")

    assert model_check["status"] == "pass"
    assert model_check["blocking_tasks"] == []
    assert model_check["details"]["unavailable_capabilities"] == []
    assert "liquid_stream_segmentation" in model_check["details"]["out_of_scope_unavailable_capabilities"]
    assert model_check["details"]["recommended_labels"] == ["balance"]


def test_quality_assurance_treats_rejected_step_as_resolved(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "t1", "event_type": "segment", "session_id": "s1"}])
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1"}])
    write_jsonl(metadata / "micro_segments.jsonl", [{"micro_segment_id": "m1", "parent_segment_id": "seg_1", "keyframes": {"peak_frame": "peak.jpg"}}])
    write_jsonl(metadata / "video_understanding.jsonl", [{"video_event_id": "v1", "event_type": "experiment_action_classification", "confidence": 0.9}])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [{"asset_id": "a1", "asset_type": "keyframe", "search_text": "weighing"}])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"embedding_id": "e1", "index_text": "weighing"}])
    write_jsonl(metadata / "human_confirmation_queue.jsonl", [])
    (metadata / "time_calibration_report.json").write_text(json.dumps({"sources": {}, "event_count": 1}), encoding="utf-8")
    (metadata / "experiment_context.json").write_text(json.dumps({"session_id": "s1", "procedure_candidates": [{"action_type": "weighing"}], "materials": []}), encoding="utf-8")
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "current_step_id": "step_1",
                "next_step_id": None,
                "steps": [
                    {"step_id": "step_1", "completed": True, "requires_human_confirmation": False, "evidence_refs": [{"type": "video_event", "video_event_id": "v1"}]},
                    {"step_id": "step_2", "completed": False, "status": "human_rejected", "confirmation_status": "rejected", "requires_human_confirmation": False, "evidence_refs": []},
                ],
                "evidence_index": {"v1": ["step_1"]},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "advanced_vision_evidence_summary.json").write_text(json.dumps({"visual_confirmation_level_counts": {"trajectory_confirmed": 1}}), encoding="utf-8")
    (metadata / "model_inventory.json").write_text(json.dumps({"model_count": 1, "capabilities": {"object_detection": {"available": True}}}), encoding="utf-8")
    (metadata / "history_model.json").write_text(json.dumps({"event_count": 2, "transition_counts": {"a->b": 1}}), encoding="utf-8")

    report = build_quality_assurance_report(tmp_path)
    step_check = next(check for check in report["checks"] if check["check_id"] == "step_reasoning")
    confirmation_check = next(check for check in report["checks"] if check["check_id"] == "human_confirmation")

    assert step_check["status"] == "pass"
    assert step_check["details"]["human_rejected"] == 1
    assert confirmation_check["status"] == "pass"


def test_time_alignment_accepts_absolute_time_anchor_artifact(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "t1", "event_type": "session_context", "session_id": "s1"}])
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1"}])
    write_jsonl(metadata / "micro_segments.jsonl", [{"micro_segment_id": "m1", "parent_segment_id": "seg_1", "keyframes": {"peak_frame": "peak.jpg"}}])
    write_jsonl(metadata / "video_understanding.jsonl", [{"video_event_id": "v1", "event_type": "experiment_action_classification", "confidence": 0.9}])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [{"asset_id": "a1", "asset_type": "keyframe", "search_text": "weighing"}])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"embedding_id": "e1", "index_text": "weighing"}])
    write_jsonl(metadata / "human_confirmation_queue.jsonl", [])
    write_jsonl(
        metadata / "time_anchors.jsonl",
        [
            {
                "source_event": {"source": "session_context"},
                "target_event": {"global_time": "2026-04-29T10:00:00+08:00"},
                "confidence": 1.0,
                "reason": "absolute_time",
            }
        ],
    )
    (metadata / "time_calibration_report.json").write_text(
        json.dumps({"sources": {"session_context": {"input_event_count": 1, "anchor_count": 0, "residual_max_abs_sec": 0.0}}}),
        encoding="utf-8",
    )
    (metadata / "experiment_context.json").write_text(json.dumps({"session_id": "s1", "procedure_candidates": [{"action_type": "weighing"}], "materials": []}), encoding="utf-8")
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "current_step_id": "step_1",
                "next_step_id": None,
                "steps": [{"step_id": "step_1", "completed": True, "requires_human_confirmation": False, "evidence_refs": [{"type": "video_event", "video_event_id": "v1"}]}],
                "evidence_index": {"v1": ["step_1"]},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "advanced_vision_evidence_summary.json").write_text(json.dumps({"visual_confirmation_level_counts": {"trajectory_confirmed": 1}}), encoding="utf-8")
    (metadata / "model_inventory.json").write_text(json.dumps({"model_count": 1, "capabilities": {"object_detection": {"available": True}}}), encoding="utf-8")
    (metadata / "history_model.json").write_text(json.dumps({"event_count": 2, "transition_counts": {"a->b": 1}}), encoding="utf-8")

    report = build_quality_assurance_report(tmp_path)
    time_check = next(check for check in report["checks"] if check["check_id"] == "time_alignment")

    assert time_check["status"] == "pass"
    assert time_check["details"]["artifact_anchored_sources"] == ["session_context"]
