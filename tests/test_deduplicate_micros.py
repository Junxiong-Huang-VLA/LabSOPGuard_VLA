"""Unit tests for deduplicate_micro_segments."""
from __future__ import annotations

from key_action_indexer.micro_segmenter import deduplicate_micro_segments
from key_action_indexer.schemas import (
    MicroSegment,
    MicroSegmentIndexInfo,
    MicroSegmentInteraction,
    MicroSegmentKeyframes,
    MicroSegmentQuality,
    MicroSegmentTextDescription,
    MicroSegmentView,
)


def _make_micro(micro_id: str, start: str, end: str, score: float) -> MicroSegment:
    return MicroSegment(
        micro_segment_id=micro_id,
        parent_segment_id="seg_000001",
        session_id="test",
        display_order=1,
        display_id="micro_001",
        start_sec=0.0,
        end_sec=10.0,
        duration_sec=10.0,
        global_start_time=start,
        global_end_time=end,
        first_person=None,
        third_person=MicroSegmentView(clip_path=None, local_start_sec=0.0, local_end_sec=10.0),
        interaction=MicroSegmentInteraction(
            interaction_type="hand_sample_bottle_contact",
            primary_object="sample_bottle",
            secondary_objects=["hand"],
            detected_objects=["sample_bottle", "hand"],
            avg_interaction_score=score,
            max_interaction_score=score,
            contact_start_sec=0.0,
            peak_interaction_sec=5.0,
            contact_end_sec=10.0,
            evidence_frame_indices=[0, 1, 2],
        ),
        keyframes=MicroSegmentKeyframes(),
        dialogue_context=[],
        text_description=MicroSegmentTextDescription(action_type="test", summary="test", index_text="test"),
        index=MicroSegmentIndexInfo(index_level="micro_segment", embedding_id=f"emb_{micro_id}"),
    )


def test_no_duplicates():
    micros = [
        _make_micro("m1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:05+00:00", 0.8),
        _make_micro("m2", "2026-01-01T00:00:10+00:00", "2026-01-01T00:00:15+00:00", 0.7),
    ]
    kept, log = deduplicate_micro_segments(micros)
    assert len(kept) == 2
    assert len(log) == 0


def test_high_overlap_removes_lower_score():
    micros = [
        _make_micro("m1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:10+00:00", 0.9),
        _make_micro("m2", "2026-01-01T00:00:01+00:00", "2026-01-01T00:00:09+00:00", 0.5),
    ]
    kept, log = deduplicate_micro_segments(micros, overlap_threshold=0.80)
    assert len(kept) == 1
    assert kept[0].micro_segment_id == "m1"
    assert len(log) == 1
    assert log[0]["removed_micro_id"] == "m2"
    assert log[0]["kept_micro_id"] == "m1"


def test_partial_overlap_below_threshold_keeps_both():
    micros = [
        _make_micro("m1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:10+00:00", 0.9),
        _make_micro("m2", "2026-01-01T00:00:07+00:00", "2026-01-01T00:00:17+00:00", 0.8),
    ]
    kept, log = deduplicate_micro_segments(micros, overlap_threshold=0.80)
    assert len(kept) == 2
    assert len(log) == 0


def test_empty_input():
    kept, log = deduplicate_micro_segments([])
    assert kept == []
    assert log == []


def test_single_input():
    micros = [_make_micro("m1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:05+00:00", 0.8)]
    kept, log = deduplicate_micro_segments(micros)
    assert len(kept) == 1
    assert len(log) == 0


def test_dedup_log_format():
    micros = [
        _make_micro("m1", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:10+00:00", 0.9),
        _make_micro("m2", "2026-01-01T00:00:00+00:00", "2026-01-01T00:00:10+00:00", 0.5),
    ]
    kept, log = deduplicate_micro_segments(micros)
    assert len(log) == 1
    entry = log[0]
    assert "removed_micro_id" in entry
    assert "kept_micro_id" in entry
    assert "overlap_ratio" in entry
    assert "removed_score" in entry
    assert "kept_score" in entry
    assert entry["overlap_ratio"] >= 0.8
