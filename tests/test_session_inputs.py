from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.input_ingestion import ingest_manifest_inputs, write_video_source_metadata
from key_action_indexer.schemas import SessionManifest, read_jsonl
from key_action_indexer.session_layout import STANDARD_SESSION_DIRS, initialize_session_dir


def test_session_layout_creates_standard_directories(tmp_path: Path) -> None:
    paths = initialize_session_dir(tmp_path / "session")

    assert set(STANDARD_SESSION_DIRS).issubset(paths)
    assert all(path.exists() for path in paths.values())


def test_manifest_accepts_extra_video_views_and_writes_video_sources(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    manifest = SessionManifest.from_dict(
        {
            "session_id": "multi_view",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(session_dir / "raw" / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                },
                "first_person": {
                    "path": str(session_dir / "raw" / "first.mp4"),
                    "start_time": "2026-04-29T17:25:02+08:00",
                    "fps": 30,
                },
                "bench_closeup": {
                    "path": str(session_dir / "raw" / "close.mp4"),
                    "start_time": "2026-04-29T17:25:01+08:00",
                    "fps": 25,
                    "role": "optional_closeup",
                },
            },
            "output_dir": str(session_dir),
        }
    )

    rows = write_video_source_metadata(manifest, session_dir)

    assert set(manifest.videos.all_sources()) == {"third_person", "first_person", "bench_closeup"}
    assert {row["view_id"] for row in rows} == {"third_person", "first_person", "bench_closeup"}
    closeup = next(row for row in rows if row["view_id"] == "bench_closeup")
    assert closeup["role"] == "optional_closeup"
    assert closeup["time_basis"] == "global_time = start_time + offset_sec + local_time_sec"


def test_ingests_user_ai_and_upload_events_into_metadata(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    upload_path = session_dir / "uploads" / "note.txt"
    upload_path.parent.mkdir(parents=True)
    upload_path.write_text("sample note", encoding="utf-8")
    user_events = tmp_path / "user.jsonl"
    ai_events = tmp_path / "ai.jsonl"
    upload_events = tmp_path / "uploads.jsonl"
    user_events.write_text(json.dumps({"event_id": "user_1", "session_sec": 1.0, "text": "add 200 uL"}) + "\n", encoding="utf-8")
    ai_events.write_text(json.dumps({"event_id": "ai_1", "session_sec": 2.0, "message": "suggest checking pipette"}) + "\n", encoding="utf-8")
    upload_events.write_text(
        json.dumps({"event_id": "up_1", "session_sec": 3.0, "upload_type": "text", "file_path": str(upload_path)})
        + "\n",
        encoding="utf-8",
    )
    manifest = SessionManifest.from_dict(
        {
            "session_id": "inputs",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(session_dir / "raw" / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                }
            },
            "input_sources": {
                "user_text_events": {"path": str(user_events), "source_type": "user_text"},
                "ai_reply_events": {"path": str(ai_events), "source_type": "ai_reply"},
                "upload_events": {"path": str(upload_events), "source_type": "upload"},
            },
            "output_dir": str(session_dir),
        }
    )

    summary = ingest_manifest_inputs(manifest, session_dir)
    user_rows = read_jsonl(summary["artifacts"]["user_text"])
    ai_rows = read_jsonl(summary["artifacts"]["ai_reply"])
    upload_rows = read_jsonl(summary["artifacts"]["upload"])

    assert summary["counts"] == {"user_text": 1, "ai_reply": 1, "upload": 1}
    assert user_rows[0]["event_type"] == "user_text"
    assert user_rows[0]["global_time"] == "2026-04-29T17:25:01+08:00"
    assert ai_rows[0]["reply_type"] == "ai_suggestion"
    assert upload_rows[0]["hash_status"] == "file"
    assert len(upload_rows[0]["sha256"]) == 64


def test_dry_run_synthesizes_input_events_when_sources_are_absent(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    manifest = SessionManifest.from_dict(
        {
            "session_id": "dry_inputs",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(session_dir / "raw" / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                }
            },
            "output_dir": str(session_dir),
        }
    )

    summary = ingest_manifest_inputs(manifest, session_dir, dry_run=True)

    assert summary["counts"] == {"user_text": 1, "ai_reply": 1, "upload": 1}
    assert read_jsonl(summary["artifacts"]["user_text"])[0]["synthetic"] is True
