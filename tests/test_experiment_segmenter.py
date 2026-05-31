"""Tests for experiment boundary detection and segmentation."""
from __future__ import annotations

from pathlib import Path

import pytest

from labsopguard.event_preprocessing.activity_presegmenter import ActivitySegment
from labsopguard.event_preprocessing.experiment_segmenter import (
    ExperimentBoundary,
    ExperimentSegmentation,
    ExperimentSegmenter,
    SegmentationConfig,
)


def _make_segments(ranges: list) -> list:
    """Helper: create ActivitySegments from (start, end) tuples."""
    return [
        ActivitySegment(start_sec=s, end_sec=e, peak_score=1.0, avg_score=0.8, trigger="motion")
        for s, e in ranges
    ]


class TestExperimentSegmenter:
    def test_single_experiment_no_split(self):
        """Short video with one continuous experiment should not be split."""
        segments = _make_segments([(5, 30), (35, 80), (85, 120)])
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=600)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=130)
        assert result.total_segments == 1
        assert result.segments[0].start_sec == 5
        assert result.segments[0].end_sec == 120

    def test_two_experiments_detected(self):
        """Two activity clusters separated by >3min gap should be split."""
        # Experiment 1: 0-45s, Experiment 2: 300-400s (gap = 255s > 180s)
        segments = _make_segments([(5, 20), (25, 45), (300, 340), (345, 400)])
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=100)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=450)
        assert result.total_segments == 2
        assert result.segments[0].end_sec == 45
        assert result.segments[1].start_sec == 300
        assert len(result.boundaries) == 1
        assert "long_gap" in result.boundaries[0].signals

    def test_three_experiments_detected(self):
        """Three experiments with sufficient gaps."""
        segments = _make_segments([
            (10, 50),       # Exp 1
            (300, 380),     # Exp 2 (gap 250s)
            (650, 720),     # Exp 3 (gap 270s)
        ])
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=100)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=800)
        assert result.total_segments == 3
        assert len(result.boundaries) == 2

    def test_short_gap_no_split(self):
        """Gaps shorter than threshold should not trigger split."""
        segments = _make_segments([(5, 30), (90, 120), (150, 200)])
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=100)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=250)
        assert result.total_segments == 1

    def test_merge_short_experiment(self):
        """Very short segment should be merged into neighbor."""
        segments = _make_segments([
            (10, 50),       # Exp 1 (40s)
            (300, 320),     # Too short (20s) - should merge
            (500, 600),     # Exp 2 (100s)
        ])
        config = SegmentationConfig(
            min_gap_sec=180,
            min_experiment_duration_sec=60,
            skip_if_video_shorter_than=100,
        )
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=700)
        # The 20s segment merges with exp1 → 2 total
        assert result.total_segments == 2

    def test_disabled_config(self):
        """When disabled, always returns single segment."""
        segments = _make_segments([(10, 50), (500, 600)])
        config = SegmentationConfig(enabled=False)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=700)
        assert result.total_segments == 1

    def test_skip_short_video(self):
        """Videos shorter than skip threshold should not be segmented."""
        segments = _make_segments([(10, 50), (300, 400)])
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=600)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=450)
        assert result.total_segments == 1

    def test_max_experiments_limit(self):
        """Should not exceed max_experiments."""
        segments = _make_segments([
            (10, 30), (300, 320), (600, 620),
            (900, 920), (1200, 1220), (1500, 1520),
        ])
        config = SegmentationConfig(
            min_gap_sec=180,
            max_experiments=3,
            skip_if_video_shorter_than=100,
        )
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=1600)
        assert result.total_segments <= 3

    def test_empty_segments(self):
        """Empty input should return empty result."""
        config = SegmentationConfig()
        segmenter = ExperimentSegmenter(config)
        result = segmenter.segment([], video_duration_sec=100)
        assert result.total_segments == 0

    def test_to_dict(self):
        """Serialization should produce valid dict."""
        segments = _make_segments([(10, 50), (400, 500)])
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=100)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=600)
        d = result.to_dict()
        assert d["schema_version"] == "experiment_segmentation.v1"
        assert d["total_segments"] == 2
        assert all("start_sec" in s for s in d["segments"])

    def test_boundary_confidence_scales_with_gap(self):
        """Longer gaps should produce higher confidence."""
        segments = _make_segments([(10, 50), (400, 500)])  # gap = 350s
        config = SegmentationConfig(min_gap_sec=180, skip_if_video_shorter_than=100)
        segmenter = ExperimentSegmenter(config)

        result = segmenter.segment(segments, video_duration_sec=600)
        assert len(result.boundaries) == 1
        assert result.boundaries[0].confidence > 0.5
