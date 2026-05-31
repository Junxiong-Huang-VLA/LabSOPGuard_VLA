from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.boss_report import generate_boss_acceptance_report
from key_action_indexer.schemas import write_jsonl


def test_generate_boss_acceptance_report_writes_nine_sections(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"session_id": "s1", "videos": {}}), encoding="utf-8")
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "process_status": "needs_review",
                "step_count": 1,
                "steps": [
                    {
                        "step_id": "step_1",
                        "name": "Weigh sample",
                        "status": "completed",
                        "confidence": 0.9,
                        "observed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (metadata / "process_quality_report.json").write_text(
        json.dumps({"overall_status": "needs_review", "overall_score": 0.8, "scorecard": {"evidence_chain": {"status": "needs_review", "score": 0.5}}}),
        encoding="utf-8",
    )
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1", "duration_sec": 2.0}])
    write_jsonl(metadata / "micro_segments.jsonl", [{"micro_segment_id": "micro_1"}])

    result = generate_boss_acceptance_report(session)
    text = Path(result["output_path"]).read_text(encoding="utf-8")

    assert result["section_count"] == 9
    assert result["step_count"] == 1
    assert "Experiment Acceptance Report" in text
    assert "evidence_chain" in text
