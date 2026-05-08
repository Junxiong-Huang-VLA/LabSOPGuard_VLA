from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.process_reasoner import build_experiment_process, load_experiment_process
from key_action_indexer.schemas import write_jsonl


def test_process_reasoner_observed_inferred_and_evidence_chain(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "s1", "procedure_candidates": [{"action_type": "weighing"}, {"action_type": "recording"}]}),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "vu_weigh",
                "session_id": "s1",
                "micro_segment_id": "micro_1",
                "event_type": "experiment_action_classification",
                "action_type": "weighing",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:04+08:00",
                "confidence": 0.86,
                "asset_refs": [{"asset_id": "asset_weigh", "path": "weigh.jpg"}],
                "anomaly_flags": [],
            },
            {
                "video_event_id": "vu_record",
                "session_id": "s1",
                "micro_segment_id": "micro_3",
                "event_type": "experiment_action_classification",
                "action_type": "recording",
                "global_start_time": "2026-04-29T17:25:20+08:00",
                "global_end_time": "2026-04-29T17:25:21+08:00",
                "confidence": 0.72,
                "asset_refs": [{"asset_id": "asset_record", "path": "record.jpg"}],
                "anomaly_flags": ["heuristic_candidate"],
            },
        ],
    )
    write_jsonl(
        metadata / "state_change_index.jsonl",
        [
            {
                "state_change_id": "state_1",
                "micro_segment_id": "micro_1",
                "state_type": "peak_interaction",
                "asset_refs": [{"asset_id": "asset_weigh", "path": "weigh.jpg"}],
            }
        ],
    )
    write_jsonl(metadata / "material_asset_catalog.jsonl", [{"asset_id": "asset_weigh", "path": "weigh.jpg"}])
    (metadata / "history_model.json").write_text(
        json.dumps(
            {
                "session_count": 2,
                "action_counts": {"weighing": 2, "pipetting": 2, "recording": 2},
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
                    {"step_id": "s1", "name": "Weigh", "expected_action": "weighing"},
                    {"step_id": "s2", "name": "Pipette", "expected_action": "pipetting"},
                    {"step_id": "s3", "name": "Record", "expected_action": "recording"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_experiment_process(tmp_path, sop_path=sop)
    loaded = load_experiment_process(metadata / "experiment_process.json")
    steps = {step["step_id"]: step for step in result["steps"]}
    timeline_rows = [json.loads(line) for line in (metadata / "experiment_process_timeline.jsonl").read_text(encoding="utf-8").splitlines()]

    assert loaded["session_id"] == "s1"
    assert steps["s1"]["observed"] is True
    assert steps["s1"]["completed"] is True
    assert any(ref.get("asset_id") == "asset_weigh" for ref in steps["s1"]["evidence_refs"])
    assert steps["s2"]["inferred"] is True
    assert steps["s2"]["requires_human_confirmation"] is True
    assert any(item["type"] == "transition_prior" for item in steps["s2"]["history_basis"])
    assert result["history_prior"]["transition_score"] == 1.0
    assert result["history_deviation"]["rare_transition_count"] == 0
    assert steps["s3"]["abnormal"] is False
    assert steps["s3"]["requires_human_confirmation"] is False
    assert result["evidence_index"]["asset_weigh"] == ["s1"]
    assert len(timeline_rows) == 3


def test_process_reasoner_handles_branches_repeats_and_order_conflicts(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "s2", "materials": [{"name": "sample_bottle"}]}),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "record_early",
                "session_id": "s2",
                "micro_segment_id": "micro_record",
                "event_type": "experiment_action_classification",
                "action_type": "recording",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:02+08:00",
                "confidence": 0.8,
                "asset_refs": [],
                "anomaly_flags": [],
            },
            {
                "video_event_id": "pipette_1",
                "session_id": "s2",
                "micro_segment_id": "micro_pipette_1",
                "event_type": "experiment_action_classification",
                "action_type": "pipetting",
                "global_start_time": "2026-04-29T17:25:10+08:00",
                "global_end_time": "2026-04-29T17:25:12+08:00",
                "confidence": 0.8,
                "asset_refs": [],
                "anomaly_flags": [],
            },
            {
                "video_event_id": "pipette_2",
                "session_id": "s2",
                "micro_segment_id": "micro_pipette_2",
                "event_type": "experiment_action_classification",
                "action_type": "pipetting",
                "global_start_time": "2026-04-29T17:25:20+08:00",
                "global_end_time": "2026-04-29T17:25:21+08:00",
                "confidence": 0.8,
                "asset_refs": [],
                "anomaly_flags": [],
            },
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    sop = tmp_path / "sop.json"
    sop.write_text(
        json.dumps(
            {
                "steps": [
                    {"step_id": "record", "expected_action": "recording"},
                    {"step_id": "pipette", "expected_action": "pipetting", "repeatable": False, "max_repeats": 1},
                    {
                        "step_id": "cleanup",
                        "expected_action": "cleanup",
                        "branch_condition": {"when_any_action_observed": ["cleanup"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_experiment_process(tmp_path, sop_path=sop)
    steps = {step["step_id"]: step for step in result["steps"]}

    assert steps["pipette"]["repeated"] is True
    assert "unexpected_repeat" in steps["pipette"]["conflict_flags"]
    assert steps["pipette"]["requires_human_confirmation"] is True
    assert steps["cleanup"]["status"] == "branch_not_taken"
    assert steps["cleanup"]["skipped"] is True
    assert steps["cleanup"]["requires_human_confirmation"] is False


def test_process_reasoner_does_not_treat_empty_history_as_negative_evidence(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "empty_history", "procedure_candidates": [{"action_type": "weighing"}]}),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "weigh_1",
                "session_id": "empty_history",
                "micro_segment_id": "micro_weigh",
                "event_type": "experiment_action_classification",
                "action_type": "weighing",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:02+08:00",
                "confidence": 0.9,
                "asset_refs": [],
                "anomaly_flags": ["heuristic_candidate", "requires_human_confirmation"],
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    (metadata / "history_model.json").write_text(
        json.dumps({"event_count": 0, "session_count": 0, "action_counts": {}, "transition_probabilities": {}}),
        encoding="utf-8",
    )
    sop = tmp_path / "sop.json"
    sop.write_text(json.dumps({"steps": [{"step_id": "weigh", "expected_action": "weighing"}]}), encoding="utf-8")

    result = build_experiment_process(tmp_path, sop_path=sop)
    step = result["steps"][0]

    assert step["status"] == "completed"
    assert step["requires_human_confirmation"] is False
    assert step["history_prior"]["available"] is False
    assert step["history_deviation"]["flags"] == []
    assert step["conflict_flags"] == []


def test_process_reasoner_evaluates_serializable_sop_conditions(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_context.json").write_text(
        json.dumps({"session_id": "s3", "materials": [{"name": "pipette"}]}),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "weigh",
                "session_id": "s3",
                "micro_segment_id": "micro_weigh",
                "event_type": "experiment_action_classification",
                "action_type": "weighing",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:02+08:00",
                "confidence": 0.9,
                "asset_refs": [],
                "anomaly_flags": [],
            },
            {
                "video_event_id": "pipette",
                "session_id": "s3",
                "micro_segment_id": "micro_pipette",
                "event_type": "experiment_action_classification",
                "action_type": "pipetting",
                "global_start_time": "2026-04-29T17:25:10+08:00",
                "global_end_time": "2026-04-29T17:25:12+08:00",
                "confidence": 0.82,
                "asset_refs": [],
                "anomaly_flags": [],
            },
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    sop = tmp_path / "sop_conditions.json"
    sop.write_text(
        json.dumps(
            {
                "steps": [
                    {"step_id": "weigh", "expected_action": "weighing"},
                    {
                        "step_id": "pipette",
                        "expected_action": "pipetting",
                        "completion_conditions": [
                            {
                                "all_actions": ["weighing"],
                                "any_actions": ["pipetting", "sample_adding_candidate"],
                                "not_actions": ["cleanup"],
                                "required_material": "pipette",
                                "min_confidence": 0.75,
                                "max_elapsed_sec": 5,
                            }
                        ],
                    },
                    {
                        "step_id": "forbidden_after_pipette",
                        "expected_action": "recording",
                        "branch_condition": {"not_actions": ["pipetting"]},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_experiment_process(tmp_path, sop_path=sop)
    steps = {step["step_id"]: step for step in result["steps"]}

    assert steps["pipette"]["status"] == "condition_failed"
    assert steps["pipette"]["observed"] is True
    assert steps["pipette"]["completed"] is False
    assert "max_elapsed_sec_exceeded" in steps["pipette"]["conflict_flags"]
    assert steps["pipette"]["condition_results"]["completion"]["passed"] is False
    assert steps["forbidden_after_pipette"]["status"] == "branch_not_taken"
