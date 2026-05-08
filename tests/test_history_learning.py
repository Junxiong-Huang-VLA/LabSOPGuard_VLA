from __future__ import annotations

from pathlib import Path

import json

from key_action_indexer.history_learning import (
    build_history_model,
    build_history_process_record,
    load_history_model,
    load_history_records,
    score_process_with_history,
    search_similar_history,
    write_back_process_data,
)
from key_action_indexer.schemas import write_jsonl


def test_history_learning_builds_transition_and_duration_model(tmp_path: Path) -> None:
    session_a = tmp_path / "session_a" / "metadata"
    session_b = tmp_path / "session_b" / "metadata"
    session_a.mkdir(parents=True)
    session_b.mkdir(parents=True)
    for metadata in (session_a, session_b):
        write_jsonl(
            metadata / "video_understanding.jsonl",
            [
                {"session_id": metadata.parent.name, "action_type": "weighing", "global_start_time": "2026-04-29T17:25:01+08:00", "duration_sec": 4, "primary_object": "balance"},
                {"session_id": metadata.parent.name, "action_type": "pipetting", "global_start_time": "2026-04-29T17:25:10+08:00", "duration_sec": 6, "primary_object": "pipette"},
                {"session_id": metadata.parent.name, "action_type": "recording", "global_start_time": "2026-04-29T17:25:20+08:00", "duration_sec": 2, "primary_object": "balance"},
            ],
        )
    output = tmp_path / "history_model.json"

    model = build_history_model([session_a.parent, session_b.parent], output_path=output)
    loaded = load_history_model(output)
    scored = score_process_with_history(
        {"steps": [{"expected_action": "weighing"}, {"expected_action": "recording"}]},
        model,
    )

    assert loaded["session_count"] == 2
    assert model["action_counts"]["pipetting"] == 2
    assert model["action_probabilities"]["pipetting"] == 0.333333
    assert model["transition_probabilities"]["weighing"]["pipetting"] == 1.0
    assert model["duration_stats"]["pipetting"]["median_sec"] == 6
    assert model["recommended_sop"][0]["expected_action"] in {"weighing", "pipetting", "recording"}
    assert scored["rare_transition_flags"][0]["transition"] == "weighing->recording"
    assert scored["history_prior"]["action_score"] > 0
    assert scored["history_deviation"]["rare_transition_count"] == 1
    assert model["schema_version"] == "key_action_history_model.v2"
    assert model["history_record_count"] == 0
    assert model["audit_trail"]


def test_history_learning_reads_legacy_labsopguard_event_dirs(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy_session"
    legacy.mkdir()
    (legacy / "physical_events.json").write_text(
        json.dumps(
            {
                "schema_version": "physical_events.v4",
                "experiment_id": "legacy_session",
                "physical_events": [
                    {"event_type": "hand_object_interaction", "duration_sec": 2.0, "primary_object": "pipette"},
                    {"event_type": "liquid_transfer", "duration_sec": 3.0, "primary_object": "beaker"},
                ],
            }
        ),
        encoding="utf-8",
    )

    model = build_history_model([legacy])

    assert model["session_count"] == 1
    assert model["source_session_ids"] == ["legacy_session"]
    assert model["action_counts"]["liquid_transfer"] == 1
    assert model["duration_stats"]["liquid_transfer"]["median_sec"] == 3


def test_history_records_preserve_version_source_and_audit_trail(tmp_path: Path) -> None:
    session = tmp_path / "session_history"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    process = {
        "session_id": "source_session_1",
        "process_status": "completed",
        "status_counts": {"completed": 2},
        "steps": [
            {"step_id": "s1", "expected_action": "weighing", "status": "completed", "required_material": "balance"},
            {"step_id": "s2", "expected_action": "pipetting", "status": "completed", "required_material": "pipette"},
        ],
    }
    (metadata / "experiment_process.json").write_text(json.dumps(process, ensure_ascii=False), encoding="utf-8")
    write_jsonl(
        metadata / "experiment_process_timeline.jsonl",
        [
            {"timeline_event_id": "s1", "event_type": "experiment_step", "global_time": "2026-05-03T09:00:00+08:00"},
            {"timeline_event_id": "s2", "event_type": "experiment_step", "global_time": "2026-05-03T09:00:10+08:00"},
        ],
    )
    store = tmp_path / "history_store.jsonl"

    record = build_history_process_record(session)
    first = write_back_process_data(session, store, actor="tester")
    second = write_back_process_data(session, store, actor="tester", note="manual_update")
    records = load_history_records(store)
    similar = search_similar_history({"steps": [{"expected_action": "weighing"}, {"expected_action": "pipetting"}]}, records)

    assert record["source_session_id"] == "source_session_1"
    assert record["version"] == 1
    assert record["steps"][0]["source_session_id"] == "source_session_1"
    assert record["steps"][0]["audit_trail"]
    assert first["version"] == 1
    assert second["version"] == 2
    assert len(records) == 2
    assert records[-1]["audit_trail"][0]["action"] == "manual_update"
    assert similar[0]["source_session_id"] == "source_session_1"
    assert similar[0]["score"] > 0.8
