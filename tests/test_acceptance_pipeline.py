from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.acceptance_pipeline import PipelineOptions, run_acceptance_pipeline
from key_action_indexer.cli import main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_acceptance_pipeline_generates_review_and_reports(tmp_path: Path) -> None:
    session = tmp_path / "session"
    write_jsonl(
        session / "metadata" / "model_observation_events.jsonl",
        [
            {
                "event_id": "evt-1",
                "start_s": 1.0,
                "end_s": 2.0,
                "description": "\u624b\u78b0\u74f6\u5b50\u5e76\u52a0\u6837",
                "confirmation_status": "candidate",
            }
        ],
    )
    write_jsonl(
        session / "index" / "retrieval_index.jsonl",
        [{"id": "seg-1", "text": "\u79f0\u91cf \u4f7f\u7528\u522e\u52fa \u6253\u5f00\u5bb9\u5668"}],
    )
    (session / "transcript").mkdir(parents=True)
    (session / "transcript" / "audio.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nok\n", encoding="utf-8")

    result = run_acceptance_pipeline(PipelineOptions(session_dir=session, strict=True))

    assert result.ok
    assert (session / "exports" / "review_bundle.json").exists()
    assert (session / "metadata" / "confirmation_decisions.template.jsonl").exists()
    assert (session / "reports" / "query_acceptance_report.json").exists()
    quality = json.loads((session / "metadata" / "process_quality_report.json").read_text(encoding="utf-8"))
    assert quality["summary"]["labels_and_training_blocking"] is False
    assert quality["summary"]["human_review_candidate_count"] == 1


def test_acceptance_pipeline_dry_run_allows_empty_session(tmp_path: Path) -> None:
    session = tmp_path / "empty"

    result = run_acceptance_pipeline(PipelineOptions(session_dir=session, strict=True, dry_run=True))

    assert result.ok
    assert (session / "reports" / "acceptance_snapshot.json").exists()
    snapshot = json.loads((session / "reports" / "acceptance_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot["dry_run"] is True
    assert snapshot["labels_and_training_blocking"] is False


def test_acceptance_pipeline_strict_fails_invalid_jsonl(tmp_path: Path) -> None:
    session = tmp_path / "broken"
    bad = session / "metadata" / "model_observation_events.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text("{bad\n", encoding="utf-8")

    result = run_acceptance_pipeline(PipelineOptions(session_dir=session, strict=True))

    assert result.status == "fail"
    assert result.errors


def test_cli_acceptance_pipeline_smoke(tmp_path: Path) -> None:
    session = tmp_path / "cli-session"

    exit_code = main(["acceptance-pipeline", "--session-dir", str(session), "--dry-run", "--strict"])

    assert exit_code == 0
    assert (session / "reports" / "boss_report.md").exists()
