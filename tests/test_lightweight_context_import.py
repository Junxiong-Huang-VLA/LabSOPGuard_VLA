from __future__ import annotations

from pathlib import Path

from key_action_indexer.context_fusion import build_experiment_context
from key_action_indexer.lightweight_context_import import import_lightweight_context
from key_action_indexer.process_reasoner import build_experiment_process
from key_action_indexer.schemas import read_jsonl, write_jsonl


def test_import_lightweight_context_feeds_context_and_process_without_visual_claims(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "video_understanding.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    write_jsonl(metadata / "state_change_index.jsonl", [])

    summary = import_lightweight_context(
        tmp_path,
        sop_text="\n".join(
            [
                "1. Weigh sample on balance.",
                "2. Transfer 200 uL sample with pipette.",
                "3. Record balance readout.",
            ]
        ),
        note_text="Operator note: pipette transfer uses 200 uL sample.",
        record_text="Experiment record says balance readout should be recorded after weighing.",
    )

    sop_rows = read_jsonl(metadata / "sop_records.jsonl")
    user_rows = read_jsonl(metadata / "user_text_events.jsonl")
    context = build_experiment_context(tmp_path)
    process = build_experiment_process(tmp_path)
    actions = {item["action_type"] for item in context["procedure_candidates"]}

    assert summary["imported_sop_record_count"] == 3
    assert summary["imported_user_text_event_count"] == 2
    assert {row["expected_action"] for row in sop_rows} == {"weighing", "pipetting", "recording"}
    assert all(row["not_visual_evidence"] is True and row["process_eligible"] is False for row in sop_rows)
    assert all(row["not_visual_evidence"] is True and row["process_eligible"] is False for row in user_rows)
    assert {"weighing", "pipetting", "recording"}.issubset(actions)
    assert context["source_counts"]["user_text_events"] == 2
    assert context["source_counts"]["sop_rows"] == 3
    assert process["step_count"] == 3
    assert all(ref["evidence_level"] == "text_support" for step in process["steps"] for ref in step["sop_refs"])
    assert any(step["text_context_refs"] for step in process["steps"])
