"""Integration test: presegment + detection stream + cache working together."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from labsopguard.event_preprocessing.activity_presegmenter import ActivitySegment, PresegmentConfig
from labsopguard.event_preprocessing.detection_cache import DetectionCache
from labsopguard.event_preprocessing.frame_detection_stream import DetectionFrameStreamBuilder


def _create_test_video(path: Path, fps: int = 30, duration_sec: float = 45.0) -> None:
    """Create a test video with alternating activity patterns."""
    width, height = 320, 240
    total_frames = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(total_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if 10 < t < 20 or 30 < t < 40:
            cx = int((t % 10) / 10 * width)
            cy = height // 2
            cv2.circle(frame, (cx, cy), 25, (0, 200, 100), -1)
        else:
            frame[:] = 30
        writer.write(frame)
    writer.release()


class _MockPipeline:
    """Minimal mock of VideoAnalysisPipeline for testing."""

    class _Settings:
        yolo_model_path = "fake_model.pt"
        yolo_imgsz = 640
        confidence_threshold = 0.25

    settings = _Settings()

    def _run_yolo(self, frame, frame_idx, ts):
        return []

    def _run_yolo_batch(self, frames, frame_indices, timestamps):
        return [[] for _ in frames]


class TestPresegmentIntegration:
    def test_short_video_bypasses_presegment(self, tmp_path: Path) -> None:
        video_path = tmp_path / "short.mp4"
        _create_test_video(video_path, fps=30, duration_sec=10.0)

        config = PresegmentConfig(enabled=True, skip_if_video_shorter_than=30.0)
        builder = DetectionFrameStreamBuilder(
            pipeline=_MockPipeline(),
            interval_sec=0.5,
            presegment_config=config,
        )
        frames, tracklets = builder.build(video_path)
        # Short video: all frames processed (no presegment filter)
        assert len(frames) > 15

    def test_long_video_reduces_frames(self, tmp_path: Path) -> None:
        video_path = tmp_path / "long.mp4"
        _create_test_video(video_path, fps=30, duration_sec=45.0)

        # Without presegment
        config_off = PresegmentConfig(enabled=False)
        builder_off = DetectionFrameStreamBuilder(
            pipeline=_MockPipeline(),
            interval_sec=0.5,
            presegment_config=config_off,
        )
        frames_all, _ = builder_off.build(video_path)

        # With presegment
        config_on = PresegmentConfig(
            enabled=True,
            skip_if_video_shorter_than=30.0,
            min_segment_sec=2.0,
            padding_sec=1.0,
        )
        builder_on = DetectionFrameStreamBuilder(
            pipeline=_MockPipeline(),
            interval_sec=0.5,
            presegment_config=config_on,
        )
        frames_filtered, _ = builder_on.build(video_path)

        # Presegment should reduce frames processed
        assert len(frames_filtered) < len(frames_all)
        # But should still have some frames
        assert len(frames_filtered) > 0

    def test_cache_prevents_reprocessing(self, tmp_path: Path) -> None:
        video_path = tmp_path / "cached.mp4"
        _create_test_video(video_path, fps=30, duration_sec=10.0)
        cache_dir = tmp_path / "cache"

        config = PresegmentConfig(enabled=False)
        pipeline = _MockPipeline()
        builder = DetectionFrameStreamBuilder(
            pipeline=pipeline,
            interval_sec=1.0,
            presegment_config=config,
            cache_dir=cache_dir,
        )

        # First run: should process and cache
        frames1, _ = builder.build(video_path)
        assert len(frames1) > 0
        assert any(cache_dir.glob("*.manifest.json"))

        # Second run: should hit cache (mock pipeline shouldn't be called)
        call_count = [0]
        original_run = pipeline._run_yolo

        def counting_run(*args, **kwargs):
            call_count[0] += 1
            return original_run(*args, **kwargs)

        pipeline._run_yolo = counting_run
        frames2, _ = builder.build(video_path)

        assert len(frames2) == len(frames1)
        assert call_count[0] == 0  # Cache hit, no YOLO calls

    def test_time_range_uses_isolated_cache_and_stays_in_range(self, tmp_path: Path) -> None:
        video_path = tmp_path / "range_cached.mp4"
        _create_test_video(video_path, fps=30, duration_sec=20.0)
        cache_dir = tmp_path / "cache"

        builder = DetectionFrameStreamBuilder(
            pipeline=_MockPipeline(),
            interval_sec=1.0,
            presegment_config=PresegmentConfig(enabled=False),
            cache_dir=cache_dir,
        )

        full_frames, _ = builder.build(video_path)
        ranged_frames, _ = builder.build(video_path, time_range=(5.0, 10.0))

        assert len(full_frames) > len(ranged_frames)
        assert ranged_frames
        assert min(frame.timestamp_sec for frame in ranged_frames) >= 5.0
        assert max(frame.timestamp_sec for frame in ranged_frames) <= 10.0
        assert len(list(cache_dir.glob("*.manifest.json"))) == 2

    def test_presegment_result_available(self, tmp_path: Path) -> None:
        video_path = tmp_path / "segmented.mp4"
        _create_test_video(video_path, fps=30, duration_sec=45.0)

        config = PresegmentConfig(enabled=True, skip_if_video_shorter_than=30.0)
        builder = DetectionFrameStreamBuilder(
            pipeline=_MockPipeline(),
            interval_sec=0.5,
            presegment_config=config,
        )
        builder.build(video_path)

        segments = builder.last_presegment_result
        assert len(segments) > 0
        for seg in segments:
            assert seg.start_sec >= 0
            assert seg.end_sec > seg.start_sec

    def test_batch_size_respected(self, tmp_path: Path) -> None:
        video_path = tmp_path / "batch.mp4"
        _create_test_video(video_path, fps=30, duration_sec=10.0)

        batch_calls = []

        class BatchTrackingPipeline(_MockPipeline):
            def _run_yolo_batch(self, frames, frame_indices, timestamps):
                batch_calls.append(len(frames))
                return [[] for _ in frames]

        config = PresegmentConfig(enabled=False)
        builder = DetectionFrameStreamBuilder(
            pipeline=BatchTrackingPipeline(),
            interval_sec=1.0,
            presegment_config=config,
            batch_size=4,
        )
        builder.build(video_path)

        # Should have called batch with size <= 4
        assert all(size <= 4 for size in batch_calls)
        assert len(batch_calls) > 0

    def test_three_hour_frame_plan_only_samples_active_windows(self, tmp_path: Path) -> None:
        builder = DetectionFrameStreamBuilder(
            pipeline=_MockPipeline(),
            interval_sec=0.5,
            max_frames=360,
            presegment_config=PresegmentConfig(enabled=True, skip_if_video_shorter_than=30.0),
        )
        builder.presegmenter.segment = lambda _path: [
            ActivitySegment(100.0, 700.0, 1.0, 1.0, "motion"),
            ActivitySegment(4000.0, 4600.0, 1.0, 1.0, "motion"),
        ]

        indices = builder._compute_frame_indices(
            tmp_path / "synthetic_3h.mp4",
            fps=30.0,
            total_frames=int(10_800 * 30),
            duration=10_800.0,
        )
        timestamps = [idx / 30.0 for idx in indices]

        assert indices
        assert len(indices) <= 362
        assert all((100.0 <= ts <= 700.0) or (4000.0 <= ts <= 4600.0) for ts in timestamps)
        assert any(690.0 <= ts <= 700.0 for ts in timestamps)
        assert any(4590.0 <= ts <= 4600.0 for ts in timestamps)
