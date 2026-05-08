from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.confirmation_loop import apply_confirmation_decision, build_confirmation_queue
from key_action_indexer.process_reasoner import build_experiment_process
from key_action_indexer.schemas import read_jsonl, write_jsonl
from key_action_indexer.sop_state_machine import build_sop_state_machine


def test_sop_state_machine_parses_text_branches_wait_repeat_and_parallel(tmp_path: Path) -> None:
    sop = tmp_path / "sop.txt"
    sop.write_text(
        "\n".join(
            [
                "1. Weigh sample on balance.",
                "2. If the tube is present, pipette liquid and repeat until target volume is reached.",
                "3. Wait until readout is stable while parallel observer records the value.",
            ]
        ),
        encoding="utf-8",
    )

    machine = build_sop_state_machine(sop)
    steps = {step["step_id"]: step for step in machine["steps"]}

    assert machine["schema_version"] == "sop_state_machine/v1"
    assert machine["step_count"] == 3
    assert steps["step_002"]["expected_action"] == "pipetting"
    assert steps["step_002"]["repeatable"] is True
    assert steps["step_002"]["repeat_until"]
    assert steps["step_003"]["wait_conditions"]
    assert steps["step_003"]["parallel_observations"]
    assert any(row["transition_type"] == "repeat_until" for row in machine["transitions"])


def test_process_reasoner_fuses_context_history_and_conflict_confirmation(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "experiment_context.json").write_text(
        json.dumps(
            {
                "session_id": "complex_sop",
                "procedure_candidates": [{"action_type": "weighing", "score": 2, "source_types": ["text"]}],
                "related_records": [{"record_id": "hist_1", "matched_actions": ["weighing"], "transition_sequence": ["weighing", "recording"], "score": 3}],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "weigh_1",
                "session_id": "complex_sop",
                "event_type": "experiment_action_classification",
                "action_type": "weighing",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:02+08:00",
                "confidence": 0.82,
                "asset_refs": [],
                "anomaly_flags": ["low_light"],
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    (metadata / "history_model.json").write_text(
        json.dumps(
            {
                "session_count": 1,
                "action_counts": {"weighing": 3, "recording": 3},
                "transition_probabilities": {"weighing": {"recording": 1.0}},
                "history_records": [
                    {
                        "record_id": "hist_1",
                        "source_session_id": "hist_1",
                        "summary": {"actions": ["weighing", "recording"]},
                        "process": {"steps": [{"expected_action": "weighing"}, {"expected_action": "recording"}]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sop = tmp_path / "sop.json"
    sop.write_text(
        json.dumps(
            {
                "steps": [
                    {"step_id": "weigh", "expected_action": "weighing"},
                    {"step_id": "record", "expected_action": "recording", "wait_conditions": {"any_actions": ["recording"]}},
                ]
            }
        ),
        encoding="utf-8",
    )

    process = build_experiment_process(tmp_path, sop_path=sop)
    steps = {step["step_id"]: step for step in process["steps"]}
    queue = build_confirmation_queue(tmp_path)
    decision = apply_confirmation_decision(tmp_path, "complex_sop:weigh", "needs_review", reviewer="qa", note="check low light")
    audit_rows = read_jsonl(metadata / "human_confirmation_audit_trail.jsonl")

    assert (metadata / "sop_state_machine.json").exists()
    assert steps["weigh"]["reasoning"]
    assert "video_anomaly:low_light" in steps["weigh"]["conflict_flags"]
    assert steps["weigh"]["requires_human_confirmation"] is True
    assert any(ref["type"] == "text_context" for ref in steps["weigh"]["evidence_refs"])
    assert any(ref["type"] == "database_record" for ref in steps["weigh"]["evidence_refs"])
    assert process["conflict_report"]["flags"]["video_anomaly:low_light"] == ["weigh"]
    assert steps["record"]["status"] in {"waiting", "not_observed"}
    assert queue["pending_count"] >= 1
    assert decision["audit"]["after_state"]["reasoning"]
    assert audit_rows[0]["before_state"]["reasoning"]
