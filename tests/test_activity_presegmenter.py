"""Tests for the ActivityPreSegmenter module."""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from labsopguard.event_preprocessing.activity_presegmenter import (
    ActivityPreSegmenter,
    ActivitySegment,
    PresegmentConfig,
)


def _create_synthetic_video(path: Path, fps: int = 30, duration_sec: float = 60.0) -> None:
    """Create a synthetic video with alternating static and motion segments."""
    width, height = 320, 240
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(total_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Static segments: 0-10s, 20-30s, 40-50s
        # Active segments: 10-20s, 30-40s, 50-60s
        segment_idx = int(t / 10)
        is_active = segment_idx % 2 == 1

        if is_active:
            # Draw moving circle to simulate activity
            cx = int((t % 10) / 10 * width)
            cy = height // 2 + int(30 * np.sin(t * 3))
            cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
            cv2.rectangle(frame, (50, 50), (100 + int(t * 5) % 100, 100), (255, 0, 0), -1)
        else:
            # Static background
            frame[:] = 40

        writer.write(frame)
    writer.release()


def _create_short_video(path: Path, fps: int = 30, duration_sec: float = 5.0) -> None:
    """Create a short video that should bypass presegmentation."""
    width, height = 320, 240
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        cx = int(i / total_frames * width)
        cv2.circle(frame, (cx, height // 2), 20, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


class TestActivityPreSegmenter:
    def test_segment_synthetic_video(self, tmp_path: Path) -> None:
        video_path = tmp_path / "test_video.mp4"
        _create_synthetic_video(video_path, fps=30, duration_sec=60.0)

        config = PresegmentConfig(
            enabled=True,
            scan_fps=2.0,
            skip_if_video_shorter_than=30.0,
            min_segment_sec=3.0,
            merge_gap_sec=5.0,
            padding_sec=2.0,
        )
        segmenter = ActivityPreSegmenter(config)
        segments = segmenter.segment(video_path, stream_id="test")

        assert len(segments) > 0
        total_active = sum(seg.duration_sec for seg in segments)
        # Should not cover entire video (some reduction expected)
        assert total_active < 60.0
        # Should have found at least some activity
        assert total_active > 0.5

        for seg in segments:
            assert seg.start_sec >= 0.0
            assert seg.end_sec <= 60.0
            assert seg.end_sec > seg.start_sec
            assert seg.stream_id == "test"

    def test_short_video_bypass(self, tmp_path: Path) -> None:
        video_path = tmp_path / "short.mp4"
        _create_short_video(video_path, fps=30, duration_sec=5.0)

        config = PresegmentConfig(skip_if_video_shorter_than=30.0)
        segmenter = ActivityPreSegmenter(config)
        segments = segmenter.segment(video_path)

        assert len(segments) == 1
        assert segments[0].trigger == "short_video_bypass"
        assert segments[0].start_sec == 0.0

    def test_nonexistent_video(self) -> None:
        config = PresegmentConfig()
        segmenter = ActivityPreSegmenter(config)
        segments = segmenter.segment(Path("/nonexistent/video.mp4"))
        assert segments == []

    def test_segment_merging(self, tmp_path: Path) -> None:
        video_path = tmp_path / "merge_test.mp4"
        _create_synthetic_video(video_path, fps=30, duration_sec=60.0)

        # With large merge gap, nearby segments should merge
        config = PresegmentConfig(
            skip_if_video_shorter_than=30.0,
            merge_gap_sec=15.0,
            padding_sec=3.0,
        )
        segmenter = ActivityPreSegmenter(config)
        segments_merged = segmenter.segment(video_path)

        config_strict = PresegmentConfig(
            skip_if_video_shorter_than=30.0,
            merge_gap_sec=1.0,
            padding_sec=0.5,
        )
        segmenter_strict = ActivityPreSegmenter(config_strict)
        segments_strict = segmenter_strict.segment(video_path)

        # Merged should have fewer or equal segments
        assert len(segments_merged) <= len(segments_strict)

    def test_to_dict(self) -> None:
        seg = ActivitySegment(
            start_sec=10.5,
            end_sec=25.3,
            peak_score=0.15,
            avg_score=0.08,
            trigger="motion",
            stream_id="cam_1",
        )
        d = seg.to_dict()
        assert d["start_sec"] == 10.5
        assert d["end_sec"] == 25.3
        assert d["duration_sec"] == 14.8
        assert d["trigger"] == "motion"
        assert d["stream_id"] == "cam_1"

    def test_disabled_config(self, tmp_path: Path) -> None:
        """When disabled, presegmenter is not called by the stream builder."""
        config = PresegmentConfig(enabled=False)
        segmenter = ActivityPreSegmenter(config)
        # Even if called directly, it still works
        video_path = tmp_path / "vid.mp4"
        _create_short_video(video_path, fps=30, duration_sec=5.0)
        segments = segmenter.segment(video_path)
        assert len(segments) == 1
