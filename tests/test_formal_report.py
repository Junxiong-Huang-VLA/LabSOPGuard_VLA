from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.report import generate_formal_validation_report
from key_action_indexer.schemas import write_jsonl


def test_formal_report_contains_metric_mode_and_incomplete_gt_warning(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "evaluation").mkdir()
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "session_start_time": "2026-04-29T17:25:00+08:00",
                "videos": {"third_person": {"path": "third.mp4", "start_time": "2026-04-29T17:25:00+08:00"}},
                "output_dir": str(session),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(
        session / "metadata" / "key_action_segments.jsonl",
        [{"segment_id": "seg_000001", "duration_sec": 10.0, "dialogue_context": []}],
    )
    write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_001",
                "parent_segment_id": "seg_000001",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "duration_sec": 1.0,
                "interaction": {"primary_object": "balance", "interaction_type": "hand_balance_contact"},
                "evidence_level": "visual_confirmed",
                "quality": {"confidence": "high"},
            }
        ],
    )
    (session / "evaluation" / "micro_segment_eval.json").write_text(
        json.dumps(
            {
                "metric_mode": "debugging",
                "precision_is_formal": False,
                "gt_completeness": "unknown",
                "precision": 0.1,
                "recall": 1.0,
                "f1": 0.18,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (session / "evaluation" / "query_validation.json").write_text(
        json.dumps({"query_hit_rate": 1.0, "queries": []}, ensure_ascii=False),
        encoding="utf-8",
    )

    report = generate_formal_validation_report(session)
    text = report.read_text(encoding="utf-8")

    assert "metric_mode: debugging" in text
    assert "Current precision is not a final quality metric" in text
    assert "Query Validation" in text
