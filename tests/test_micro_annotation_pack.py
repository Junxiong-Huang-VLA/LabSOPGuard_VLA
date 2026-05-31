from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.annotation import generate_micro_annotation_pack
from key_action_indexer.schemas import read_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_generate_micro_annotation_pack_writes_review_files(tmp_path: Path) -> None:
    session = tmp_path / "session"
    source = session / "metadata" / "micro_segments.jsonl"
    _write_jsonl(
        source,
        [
            {
                "micro_segment_id": "seg_001_micro_001",
                "parent_segment_id": "seg_001",
                "display_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 2.5,
                "duration_sec": 1.5,
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:02.500000+08:00",
                "interaction": {"primary_object": "pipette", "interaction_type": "hand_pipette_contact", "max_interaction_score": 0.9},
                "text_description": {"action_type": "pipetting", "summary": "pipette contact"},
                "quality": {"confidence": "high", "warnings": []},
                "keyframes": {"peak_frame": "peak.jpg"},
                "third_person": {"clip_path": "third.mp4"},
            }
        ],
    )

    summary = generate_micro_annotation_pack(session)

    review_dir = session / "annotation" / "micro_review"
    assert summary["micro_segment_count"] == 1
    assert (review_dir / "micro_review.md").exists()
    assert (review_dir / "micro_review.jsonl").exists()
    assert (review_dir / "manual_micro_segments.template.jsonl").exists()
    review = read_jsonl(review_dir / "micro_review.jsonl")
    template = read_jsonl(review_dir / "manual_micro_segments.template.jsonl")
    assert review[0]["review_id"] == "review_000001"
    assert review[0]["primary_object"] == "pipette"
    assert template[0]["operation"] == "update"
    assert "seg_001_micro_001" in (review_dir / "micro_review.md").read_text(encoding="utf-8")


def test_full_window_annotation_pack_writes_eval_config_template(tmp_path: Path) -> None:
    session = tmp_path / "session"
    source = session / "metadata" / "micro_segments.jsonl"
    _write_jsonl(
        source,
        [
            {
                "micro_segment_id": "seg_001_micro_001",
                "parent_segment_id": "seg_001",
                "start_sec": 1.0,
                "end_sec": 2.5,
                "duration_sec": 1.5,
                "interaction": {"primary_object": "pipette", "interaction_type": "hand_pipette_contact"},
                "text_description": {"action_type": "pipetting", "summary": "pipette contact"},
                "quality": {"confidence": "high", "warnings": []},
            }
        ],
    )

    summary = generate_micro_annotation_pack(session, full_window=True, window_start_sec=0.0, window_end_sec=68.7)
    review_dir = session / "annotation" / "micro_review"
    template = read_jsonl(review_dir / "manual_micro_segments.template.jsonl")
    eval_config = json.loads((review_dir / "manual_micro_eval_config.template.json").read_text(encoding="utf-8"))
    review_md = (review_dir / "micro_review.md").read_text(encoding="utf-8")

    assert summary["full_window"] is True
    assert template[0]["micro_segment_id"] == "gt_micro_001"
    assert template[0]["source_prediction_id"] == "seg_001_micro_001"
    assert eval_config["gt_completeness"] == "complete"
    assert eval_config["labeled_windows"][0]["end_sec"] == 68.7
    assert "Do not only label predicted micro-segments" in review_md
