from __future__ import annotations

from key_action_indexer.action_detector import build_segments_from_scores
from key_action_indexer.config import DetectorConfig
from key_action_indexer.schemas import FrameScore, VideoSource


def test_segment_merge_filter_and_buffer() -> None:
    scores = []
    for t in range(0, 45):
        active = 0.05
        if 10 <= t <= 15:
            active = 0.8
        if 18 <= t <= 25:
            active = 0.8
        if 35 <= t <= 36:
            active = 0.8
        scores.append(FrameScore(time_sec=float(t), motion_score=active, active_score=active))

    config = DetectorConfig(
        sample_fps=1,
        start_threshold=0.6,
        start_min_duration_sec=1,
        end_threshold=0.3,
        end_min_duration_sec=1,
        merge_gap_sec=5,
        min_segment_duration_sec=5,
        buffer_sec=2,
    )
    source = VideoSource("third_person", "third.mp4", "2026-04-29T17:25:00+08:00", fps=30)
    segments = build_segments_from_scores(scores, source, duration_sec=44, config=config)
    assert len(segments) == 1
    assert segments[0].start_sec == 8
    assert segments[0].end_sec == 28
