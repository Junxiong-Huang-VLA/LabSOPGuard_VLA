from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.schemas import write_jsonl
from key_action_indexer.session_audit import build_session_audit_report, render_session_audit_markdown


def test_session_audit_summarizes_core_metrics_and_risks(tmp_path: Path) -> None:
    session = tmp_path / "session" / "key_action_index"
    metadata = session / "metadata"
    reports = session / "reports"
    metadata.mkdir(parents=True)
    reports.mkdir()
    (session / "manifest.json").write_text(json.dumps({"session_id": "audit_s1"}), encoding="utf-8")
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_001"}])
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "micro_001",
                "evidence": {
                    "process_evidence_role": "strong_process_evidence",
                    "process_eligible": True,
                    "retrieval_priority_bucket": "high_physical_continuity",
                },
            },
            {
                "micro_segment_id": "micro_002",
                "evidence": {
                    "process_evidence_role": "retrieval_candidate",
                    "retrieval_candidate_only": True,
                    "coverage_signal_grade": "single_frame_yolo_candidate",
                    "retrieval_priority_bucket": "segment_level_backfill",
                    "warnings": ["segment_level_retrieval_backfill"],
                },
            },
        ],
    )
    write_jsonl(metadata / "vector_metadata.jsonl", [{"segment_id": "seg_001"}])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [{"micro_segment_id": "micro_001"}])
    (metadata / "video_understanding_summary.json").write_text(
        json.dumps(
            {
                "session_id": "audit_s1",
                "video_event_count": 10,
                "conclusion_status_counts": {"candidate": 2, "confirmed": 5, "measured": 3},
                "human_review_candidate_count": 2,
                "candidate_rollup": {"removed_candidate_event_count": 4},
            }
        ),
        encoding="utf-8",
    )
    (metadata / "process_quality_report.json").write_text(
        json.dumps({"overall_status": "pass", "overall_score": 0.91, "status_counts": {"pass": 15}}),
        encoding="utf-8",
    )
    (metadata / "experiment_process.json").write_text(
        json.dumps({"process_status": "completed", "step_count": 2, "status_counts": {"completed": 2}}),
        encoding="utf-8",
    )
    (metadata / "history_model.json").write_text(
        json.dumps({"session_count": 3, "event_count": 42, "source_session_ids": ["a", "b", "c"]}),
        encoding="utf-8",
    )
    (reports / "run_health_report.json").write_text(
        json.dumps({"status": "pass", "gate_status": "pass", "error_count": 0, "warning_count": 0}),
        encoding="utf-8",
    )
    write_jsonl(metadata / "human_confirmation_queue.jsonl", [])
    write_jsonl(metadata / "human_confirmation_machine_backlog.jsonl", [{"id": "m1"}])

    report = build_session_audit_report([tmp_path / "session"], output_json=tmp_path / "audit.json", output_md=tmp_path / "audit.md")
    row = report["sessions"][0]
    markdown = render_session_audit_markdown(report)

    assert report["summary"]["health_pass_count"] == 1
    assert report["summary"]["qa_pass_count"] == 1
    assert report["summary"]["strong_process_micro_count"] == 1
    assert row["session_id"] == "audit_s1"
    assert row["metrics"]["video_understanding"]["candidate_ratio"] == 0.2
    assert row["metrics"]["micro_evidence"]["retrieval_candidate"] == 1
    assert row["metrics"]["micro_evidence"]["segment_level_backfill_count"] == 1
    assert row["metrics"]["micro_evidence"]["segment_level_backfill_promoted_count"] == 0
    assert row["risks"][0]["code"] == "history_under_sampled"
    assert "P4 Session Audit Summary" in markdown
    assert "Retrieval Acceptance" in markdown
    assert "Backfill Evidence Guard" in markdown
    assert (tmp_path / "audit.json").exists()
    assert (tmp_path / "audit.md").exists()
