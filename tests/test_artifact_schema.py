from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.artifact_schema import (
    get_artifact_schema,
    list_artifact_specs,
    validate_artifact_file,
    validate_session_artifacts,
)
from key_action_indexer.schemas import write_jsonl


def test_validate_session_artifacts_accepts_video_understanding_core_output(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "evt_1",
                "event_type": "hand_object_contact",
                "confidence": 0.82,
                "confidence_reasons": ["visual evidence"],
                "anomaly_flags": [],
                "asset_refs": [],
                "payload": {},
            }
        ],
    )

    result = validate_session_artifacts(session, artifact_types=["video_understanding"])

    assert result["valid"] is True
    assert result["artifact_count"] == 1
    assert get_artifact_schema("video_understanding")["title"] == "Video understanding event"
    assert any(item["artifact_type"] == "video_understanding" for item in list_artifact_specs())


def test_validate_artifact_file_reports_schema_errors(tmp_path: Path) -> None:
    path = tmp_path / "video_understanding.jsonl"
    write_jsonl(path, [{"video_event_id": "evt_bad", "confidence": 1.5, "confidence_reasons": [], "anomaly_flags": [], "asset_refs": [], "payload": {}}])

    result = validate_artifact_file(path, "video_understanding")

    assert result["valid"] is False
    assert any("$.event_type" in issue["path"] for issue in result["issues"])
    assert any("expected <=" in issue["message"] for issue in result["issues"])
