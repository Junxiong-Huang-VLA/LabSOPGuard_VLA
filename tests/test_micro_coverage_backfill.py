from __future__ import annotations

from pathlib import Path

from key_action_indexer.micro_coverage_backfill import backfill_micro_coverage
from key_action_indexer.micro_quality_enrichment import enrich_micro_quality
from key_action_indexer.schemas import read_jsonl, write_jsonl


def test_backfill_micro_coverage_adds_retrieval_only_parent_micro(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "session_id": "s1",
                "segment_id": "seg_001",
                "global_start_time": "2026-05-08T10:00:00+08:00",
                "global_end_time": "2026-05-08T10:00:03+08:00",
                "duration_sec": 3.0,
                "cv_detection": {"start_sec": 10.0, "end_sec": 13.0},
                "first_person": {"clip_path": "seg_001/first.mp4", "local_start_sec": 10.0, "local_end_sec": 13.0},
                "third_person": {"clip_path": "seg_001/third.mp4", "local_start_sec": 10.0, "local_end_sec": 13.0},
                "interaction_events": [
                    {
                        "view": "first_person",
                        "local_time_sec": 11.0,
                        "global_time": "2026-05-08T10:00:01+08:00",
                        "hand_label": "gloved_hand",
                        "object_label": "balance",
                        "confidence": 0.48,
                        "detections": [
                            {"label": "gloved_hand", "bbox": [0, 0, 10, 10]},
                            {"label": "balance", "bbox": [8, 8, 25, 25]},
                        ],
                    }
                ],
                "interaction_keyframes": [{"path": "seg_001/interaction.jpg", "local_time_sec": 11.0}],
                "yolo_label_counts": {"gloved_hand": 5, "balance": 4},
                "text_description": {"action_type": "weighing", "summary": "balance interaction"},
            }
        ],
    )
    write_jsonl(metadata / "micro_segments.jsonl", [])

    report = backfill_micro_coverage(tmp_path)
    rows = read_jsonl(metadata / "micro_segments.jsonl")
    segments = read_jsonl(metadata / "key_action_segments.jsonl")

    assert report["added_micro_count"] == 1
    assert len(rows) == 1
    assert rows[0]["parent_segment_id"] == "seg_001"
    assert rows[0]["interaction"]["primary_object"] == "balance"
    assert rows[0]["evidence"]["force_retrieval_candidate"] is True
    assert rows[0]["evidence"]["segment_level_coverage_backfill"] is True
    assert segments[0]["micro_segments"][0]["micro_segment_id"] == rows[0]["micro_segment_id"]

    enriched = enrich_micro_quality(tmp_path)
    enriched_rows = read_jsonl(metadata / "micro_segments.jsonl")
    evidence = enriched_rows[0]["evidence"]
    assert enriched["retrieval_candidate_micro_count"] == 1
    assert evidence["process_evidence_role"] == "retrieval_candidate"
    assert evidence["strong_process_evidence"] is False
    assert evidence["retrieval_priority_bucket"] == "segment_level_backfill"


def test_backfill_micro_coverage_is_idempotent_and_skips_existing_parent(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "session_id": "s1",
                "segment_id": "seg_001",
                "global_start_time": "2026-05-08T10:00:00+08:00",
                "global_end_time": "2026-05-08T10:00:03+08:00",
                "duration_sec": 3.0,
                "asset_bindings": [{"keyframe_path": "middle.jpg"}],
                "yolo_label_counts": {"beaker": 3},
            }
        ],
    )
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [{"session_id": "s1", "parent_segment_id": "seg_001", "micro_segment_id": "seg_001_micro_001"}],
    )

    report = backfill_micro_coverage(tmp_path)

    assert report["added_micro_count"] == 0
    assert report["output_micro_count"] == 1
    assert report["skipped_counts"]["already_has_micro"] == 1
