from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.record_ingestion import ingest_sop_and_database_records, search_records
from key_action_indexer.schemas import SessionManifest, read_jsonl


def test_ingests_sop_and_database_records_and_searches_fields(tmp_path: Path) -> None:
    sop_path = tmp_path / "sop.json"
    sop_path.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "step_id": "s1",
                        "step_order": 1,
                        "expected_action": "weighing",
                        "materials": ["balance", "sample"],
                        "parameters": {"mass": "10 mg"},
                        "text": "weigh sample on balance",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    database_path = tmp_path / "history.jsonl"
    database_path.write_text(
        json.dumps(
            {
                "record_id": "hist_1",
                "experiment_type": "assay",
                "action_type": "pipetting",
                "materials": ["pipette", "tube"],
                "parameters": {"volume": "200 uL"},
                "duration_sec": 12.5,
                "text": "historical pipette transfer",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = SessionManifest.from_dict(
        {
            "session_id": "record_ingestion_session",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third_person.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                }
            },
            "sop_path": str(sop_path),
            "database_paths": [str(database_path)],
            "output_dir": str(tmp_path / "session"),
        }
    )

    summary = ingest_sop_and_database_records(manifest, tmp_path / "session" / "metadata")
    sop_rows = read_jsonl(tmp_path / "session" / "metadata" / "sop_records.jsonl")
    database_rows = read_jsonl(tmp_path / "session" / "metadata" / "database_records.jsonl")
    results = search_records([*sop_rows, *database_rows], action="pipetting", material="pipette")

    assert summary["sop_record_count"] == 1
    assert summary["database_record_count"] == 1
    assert sop_rows[0]["event_type"] == "sop_record"
    assert sop_rows[0]["global_start_time"] is None
    assert database_rows[0]["record_id"] == "hist_1"
    assert database_rows[0]["evidence_source"] == "historical_database"
    assert results[0]["record_id"] == "hist_1"
