from __future__ import annotations

from key_action_indexer.micro_postprocess import merge_same_object_adjacent_micro_segments
from key_action_indexer.schemas import MicroSegmentConfig


def _row(micro_id: str, start: float, end: float, obj: str, score: float = 0.8) -> dict:
    return {
        "micro_segment_id": micro_id,
        "parent_segment_id": "seg_000001",
        "session_id": "s1",
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "global_start_time": f"t{start}",
        "global_end_time": f"t{end}",
        "first_person": {"clip_path": f"{micro_id}_first.mp4", "local_start_sec": start, "local_end_sec": end},
        "third_person": {"clip_path": f"{micro_id}_third.mp4", "local_start_sec": start, "local_end_sec": end},
        "interaction": {
            "primary_object": obj,
            "interaction_type": f"hand_{obj}_contact",
            "detected_objects": ["hand", obj],
            "avg_interaction_score": score,
            "max_interaction_score": score,
            "contact_start_sec": start,
            "peak_interaction_sec": (start + end) / 2,
            "contact_end_sec": end,
            "evidence_frame_indices": [int(start * 10)],
        },
        "keyframes": {
            "contact_frame": f"{micro_id}_contact.jpg",
            "peak_frame": f"{micro_id}_peak.jpg",
            "release_frame": f"{micro_id}_release.jpg",
        },
        "quality": {"confidence": "high", "warnings": []},
        "dialogue_context": [],
        "text_description": {
            "action_type": "bottle_interaction" if "bottle" in obj else f"{obj}_interaction",
            "index_text": f"micro_segment_id: {micro_id}\nprimary_object: {obj}",
        },
        "index": {"index_level": "micro_segment", "embedding_id": f"emb_{micro_id}"},
    }


def test_adjacent_same_object_micro_segments_are_merged() -> None:
    rows = [_row("m1", 1.0, 2.0, "sample_bottle", 0.7), _row("m2", 2.4, 3.0, "sample_bottle", 0.95)]
    merged, stats = merge_same_object_adjacent_micro_segments(
        rows,
        MicroSegmentConfig(same_object_merge_gap_sec=1.0, max_merged_micro_duration_sec=8.0),
    )

    assert len(merged) == 1
    assert stats["merge_count"] == 1
    assert merged[0]["merged_from_micro_segment_ids"] == ["m1", "m2"]
    assert merged[0]["keyframes"]["contact_frame"] == "m1_contact.jpg"
    assert merged[0]["keyframes"]["peak_frame"] == "m2_peak.jpg"
    assert merged[0]["keyframes"]["release_frame"] == "m2_release.jpg"


def test_different_objects_or_large_gap_do_not_merge() -> None:
    config = MicroSegmentConfig(same_object_merge_gap_sec=1.0)
    different, _ = merge_same_object_adjacent_micro_segments(
        [_row("m1", 1.0, 2.0, "sample_bottle"), _row("m2", 2.3, 3.0, "balance")],
        config,
    )
    large_gap, _ = merge_same_object_adjacent_micro_segments(
        [_row("m1", 1.0, 2.0, "sample_bottle"), _row("m2", 4.0, 5.0, "sample_bottle")],
        config,
    )

    assert len(different) == 2
    assert len(large_gap) == 2
