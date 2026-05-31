"""Integration test: verify run_manifest.json is written on dry-run."""
from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.pipeline import run_pipeline


def test_run_manifest_written_on_dry_run(tmp_path: Path) -> None:
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "test"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "test_manifest",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30, "offset_sec": 0},
            "first_person": {"path": str(output_dir / "raw" / "fp.mp4"), "start_time": "2026-04-29T17:25:02+08:00", "fps": 30, "offset_sec": 0},
        },
        "transcript": {"path": str(transcript_path), "start_time": "2026-04-29T17:25:00+08:00", "offset_sec": 0},
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)

    run_manifest_path = output_dir / "run_manifest.json"
    assert run_manifest_path.exists(), "run_manifest.json not written"
    rm = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert rm["run_id"] == summary["run_id"]
    assert rm["session_id"] == "test_manifest"
    assert "model_versions" in rm
    assert "parameters" in rm
    assert "timing" in rm
    assert rm["timing"]["stage_count"] >= 4
    assert summary["run_id"] == rm["run_id"]
    assert summary["artifacts"]["debug_report"].endswith("mvp_validation_report.md")
    assert summary["artifacts"]["formal_report"].endswith("formal_validation_report.md")
    assert summary["artifacts"]["user_text_events"].endswith("user_text_events.jsonl")


def test_pipeline_log_written_on_dry_run(tmp_path: Path) -> None:
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "test"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "test_log",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30, "offset_sec": 0},
            "first_person": {"path": str(output_dir / "raw" / "fp.mp4"), "start_time": "2026-04-29T17:25:02+08:00", "fps": 30, "offset_sec": 0},
        },
        "transcript": {"path": str(transcript_path), "start_time": "2026-04-29T17:25:00+08:00", "offset_sec": 0},
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)

    log_path = output_dir / "pipeline.log"
    assert log_path.exists(), "pipeline.log not written"
    log_content = log_path.read_text(encoding="utf-8")
    assert summary["run_id"][:8] in log_content
    assert "Pipeline started" in log_content
    assert "Pipeline completed" in log_content


def test_failed_stages_in_summary(tmp_path: Path) -> None:
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "x"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "test_failed",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30, "offset_sec": 0},
            "first_person": {"path": str(output_dir / "raw" / "fp.mp4"), "start_time": "2026-04-29T17:25:02+08:00", "fps": 30, "offset_sec": 0},
        },
        "transcript": {"path": str(transcript_path), "start_time": "2026-04-29T17:25:00+08:00", "offset_sec": 0},
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)
    assert "failed_stages" in summary
    assert isinstance(summary["failed_stages"], list)


def test_pipeline_summary_file_matches_returned_dry_run_summary(tmp_path: Path) -> None:
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "test"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "test_pipeline_summary",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30, "offset_sec": 0},
            "first_person": {"path": str(output_dir / "raw" / "fp.mp4"), "start_time": "2026-04-29T17:25:02+08:00", "fps": 30, "offset_sec": 0},
        },
        "transcript": {"path": str(transcript_path), "start_time": "2026-04-29T17:25:00+08:00", "offset_sec": 0},
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)

    summary_path = output_dir / "pipeline_summary.json"
    written_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written_summary["run_id"] == summary["run_id"]
    assert written_summary["artifacts"] == summary["artifacts"]
    assert written_summary["module_runs"] == summary["module_runs"]
    assert written_summary["error_diagnostics"] == summary["error_diagnostics"]
