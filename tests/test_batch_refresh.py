from __future__ import annotations

from pathlib import Path

from key_action_indexer import batch_refresh
from key_action_indexer.batch_refresh import batch_refresh_sessions


def test_batch_refresh_orchestrates_sessions_and_writes_summary(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    (session / "metadata").mkdir(parents=True)
    calls: list[tuple[str, str]] = []

    def fake_micro(path: Path):
        calls.append(("micro", str(path)))
        return {"micro_segment_count": 2, "strong_process_micro_count": 1, "retrieval_candidate_micro_count": 1}

    def fake_coverage(path: Path):
        calls.append(("coverage", str(path)))
        return {"added_micro_count": 1, "output_micro_count": 2, "skipped_counts": {"already_has_micro": 1}}

    def fake_video(path: Path):
        calls.append(("video", str(path)))
        return {
            "session_id": "batch_s1",
            "video_event_count": 8,
            "conclusion_status_counts": {"candidate": 2, "confirmed": 6},
            "candidate_rollup": {"removed_candidate_event_count": 3},
        }

    def fake_refresh(path: Path, *, query_texts):
        calls.append(("derived", ",".join(query_texts)))
        return {
            "steps": {
                "health": {"gate_status": "pass"},
                "quality": {"overall_status": "pass"},
                "artifact_validation": {"valid": True},
            },
            "paths": {"summary": str(path / "reports" / "derived_refresh_summary.json")},
        }

    def fake_load_scope(path: Path):
        calls.append(("load_scope", str(path)))
        return {"scope_name": "test_scope", "stage": "test", "status": "active", "out_of_scope_capabilities": ["liquid"]}

    monkeypatch.setattr(batch_refresh, "backfill_micro_coverage", fake_coverage)
    monkeypatch.setattr(batch_refresh, "enrich_micro_quality", fake_micro)
    monkeypatch.setattr(batch_refresh, "build_video_understanding", fake_video)
    monkeypatch.setattr(batch_refresh, "load_stage_scope", fake_load_scope)
    monkeypatch.setattr(batch_refresh, "refresh_derived_artifacts", fake_refresh)

    summary = batch_refresh_sessions(
        [tmp_path / "experiment"],
        query_texts=["balance weighing"],
        output_summary_path=tmp_path / "batch.json",
    )

    assert summary["refreshed_count"] == 1
    assert summary["error_count"] == 0
    assert summary["sessions"][0]["session_id"] == "batch_s1"
    assert summary["sessions"][0]["steps"]["micro_coverage"]["added_micro_count"] == 1
    assert summary["sessions"][0]["steps"]["stage_scope"]["scope_name"] == "test_scope"
    assert summary["sessions"][0]["steps"]["video_understanding"]["candidate_rollup"]["removed_candidate_event_count"] == 3
    assert calls == [
        ("coverage", str(session)),
        ("micro", str(session)),
        ("video", str(session)),
        ("load_scope", str(session)),
        ("derived", "balance weighing"),
    ]
    assert (tmp_path / "batch.json").exists()


def test_batch_refresh_records_errors_without_stopping(tmp_path: Path) -> None:
    summary = batch_refresh_sessions([tmp_path / "missing"], stop_on_error=False)

    assert summary["refreshed_count"] == 0
    assert summary["error_count"] == 1
    assert summary["sessions"][0]["status"] == "error"
