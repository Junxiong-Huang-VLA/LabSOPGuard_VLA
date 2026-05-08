from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.report import generate_report
from key_action_indexer.schemas import write_jsonl


def _base_session(tmp_path: Path, *, precision_is_formal: bool, completeness: str) -> Path:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "evaluation").mkdir()
    manifest = {
        "session_id": "s1",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {"third_person": {"path": "third.mp4", "start_time": "2026-04-29T17:25:00+08:00"}},
        "output_dir": str(session),
    }
    (session / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    write_jsonl(
        session / "metadata" / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_000001",
                "duration_sec": 10.0,
                "global_start_time": "t0",
                "global_end_time": "t10",
                "third_person": {"clip_path": "third_clip.mp4"},
                "first_person": None,
                "text_description": {"action_type": "unknown_operation"},
                "index": {"index_text": "parent"},
                "dialogue_context": [],
            }
        ],
    )
    write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_001",
                "display_id": "micro_001",
                "parent_segment_id": "seg_000001",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "duration_sec": 1.0,
                "interaction": {"primary_object": "balance", "interaction_type": "hand_balance_contact", "max_interaction_score": 0.8},
                "keyframes": {"peak_frame": "peak.jpg"},
                "quality": {"confidence": "high", "warnings": []},
            }
        ],
    )
    (session / "evaluation" / "micro_segment_eval.json").write_text(
        json.dumps(
            {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "gt_completeness": completeness,
                "labeled_window_count": 1,
                "labeled_duration_sec": 10.0,
                "predictions_inside_labeled_windows": 1,
                "predictions_outside_labeled_windows": 0,
                "precision_is_formal": precision_is_formal,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return session


def test_report_marks_partial_gt_precision_as_debug(tmp_path: Path) -> None:
    session = _base_session(tmp_path, precision_is_formal=False, completeness="partial")
    text = generate_report(session).read_text(encoding="utf-8")

    assert "## GT Coverage" in text
    assert "gt_completeness: partial" in text
    assert "precision_is_formal: False" in text
    assert "precision is for debugging only" in text


def test_report_marks_complete_gt_precision_as_formal(tmp_path: Path) -> None:
    session = _base_session(tmp_path, precision_is_formal=True, completeness="complete")
    text = generate_report(session).read_text(encoding="utf-8")

    assert "gt_completeness: complete" in text
    assert "precision_is_formal: True" in text
    assert "precision is formal within labeled windows" in text
