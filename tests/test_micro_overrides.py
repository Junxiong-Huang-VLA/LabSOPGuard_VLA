from __future__ import annotations

from pathlib import Path

from key_action_indexer.micro_overrides import apply_micro_overrides
from key_action_indexer.schemas import read_jsonl, write_jsonl


def test_micro_override_updates_bounds_object_and_vector_metadata(tmp_path: Path) -> None:
    session = tmp_path
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (session / "manifest.json").write_text(
        '{"session_id":"s1","session_start_time":"2026-04-29T17:25:00+08:00"}',
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_019",
                "parent_segment_id": "seg_000001",
                "session_id": "s1",
                "display_order": 1,
                "display_id": "micro_001",
                "start_sec": 20.0,
                "end_sec": 22.0,
                "duration_sec": 2.0,
                "global_start_time": "2026-04-29T17:25:20+08:00",
                "global_end_time": "2026-04-29T17:25:22+08:00",
                "first_person": {"clip_path": "first.mp4"},
                "third_person": {"clip_path": "third.mp4"},
                "interaction": {"primary_object": "balance", "interaction_type": "hand_balance_contact", "detected_objects": ["balance"]},
                "keyframes": {"peak_frame": "peak.jpg"},
                "dialogue_context": [],
                "text_description": {"action_type": "weighing", "index_text": "balance"},
                "index": {"embedding_id": "emb_seg_000001_micro_019"},
                "quality": {"confidence": "low"},
            }
        ],
    )
    write_jsonl(
        metadata / "micro_segments_overrides.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_019",
                "override_start_sec": 24.1,
                "override_end_sec": 31.8,
                "override_primary_object": "spatula",
                "override_action_type": "spatula_sampling",
                "note": "manual correction",
            }
        ],
    )
    summary = apply_micro_overrides(session)
    corrected = read_jsonl(metadata / "micro_segments_corrected.jsonl")[0]
    vector = read_jsonl(metadata / "micro_vector_metadata.jsonl")[0]
    assert summary["num_overrides"] == 1
    assert corrected["start_sec"] == 24.1
    assert corrected["end_sec"] == 31.8
    assert corrected["interaction"]["primary_object"] == "spatula"
    assert corrected["manual_corrected"] is True
    assert vector["primary_object"] == "spatula"
    assert vector["manual_corrected"] is True

