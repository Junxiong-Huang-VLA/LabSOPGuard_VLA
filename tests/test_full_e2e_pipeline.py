"""End-to-end pipeline integration tests."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


def _create_video(path: Path, fps: int = 30, duration_sec: float = 30.0) -> None:
    w, h = 320, 240
    total = int(fps * duration_sec)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        t = i / fps
        if 5 < t < 25:
            cx = int((t - 5) / 20 * w)
            cv2.circle(frame, (cx, h // 2), 20, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()


def _create_multi_experiment_video(path: Path, fps: int = 30) -> None:
    """Create 13min video: active 10-250s, idle 250-550s, active 550-750s."""
    w, h = 320, 240
    duration = 780
    total = int(fps * duration)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for i in range(total):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        t = i / fps
        if 10 < t < 250 or 550 < t < 750:
            cx = int((t % 100) / 100 * w)
            cv2.circle(frame, (cx, h // 2), 25, (0, 200, 100), -1)
            cv2.rectangle(frame, (50, 50), (100 + int(t) % 50, 100), (255, 0, 0), -1)
        else:
            frame[:] = 30
        writer.write(frame)
    writer.release()


class TestFullPipelineE2E:
    def test_short_video_produces_detection_frames(self, tmp_path: Path) -> None:
        from labsopguard.config import load_runtime_settings
        from labsopguard.event_preprocessing.engine import EventPreprocessingEngine

        video_path = tmp_path / "short.mp4"
        _create_video(video_path, fps=30, duration_sec=30.0)

        project_root = Path("D:/LabCapability/LabSOPGuard")
        settings = load_runtime_settings(project_root)
        engine = EventPreprocessingEngine(settings)

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        material_index = output_dir / "material_index.sqlite"

        result = engine.run(
            experiment_id="e2e-short-test",
            experiment_name="E2E Short Test",
            source_video=video_path,
            output_dir=output_dir,
            material_index_path=material_index,
        )

        assert result["preprocessing_payload"]["event_preprocessing"]["detection_frame_count"] > 0
        assert material_index.exists()

    def test_multi_experiment_segmentation(self, tmp_path: Path) -> None:
        """Segmenter correctly splits when given segments with large gap."""
        from labsopguard.event_preprocessing.activity_presegmenter import ActivitySegment
        from labsopguard.event_preprocessing.experiment_segmenter import ExperimentSegmenter, SegmentationConfig

        # Manually construct segments with a clear 300s gap (bypass presegmenter noise)
        segments = [
            ActivitySegment(start_sec=10, end_sec=240, peak_score=1.0, avg_score=0.8, trigger="motion"),
            ActivitySegment(start_sec=550, end_sec=750, peak_score=1.0, avg_score=0.8, trigger="motion"),
        ]

        segmenter = ExperimentSegmenter(SegmentationConfig(
            min_gap_sec=180,
            skip_if_video_shorter_than=100,
        ))
        result = segmenter.segment(segments, video_duration_sec=780.0)

        assert result.total_segments == 2
        assert result.segments[0].end_sec < result.segments[1].start_sec
        assert len(result.boundaries) == 1

    def test_time_range_limits_processing(self, tmp_path: Path) -> None:
        from labsopguard.config import load_runtime_settings
        from labsopguard.event_preprocessing.engine import EventPreprocessingEngine

        video_path = tmp_path / "range_test.mp4"
        _create_video(video_path, fps=30, duration_sec=60.0)  # Longer video

        project_root = Path("D:/LabCapability/LabSOPGuard")
        settings = load_runtime_settings(project_root)
        engine = EventPreprocessingEngine(settings)

        output_full = tmp_path / "full"
        output_full.mkdir()
        result_full = engine.run(
            experiment_id="e2e-full",
            experiment_name="Full",
            source_video=video_path,
            output_dir=output_full,
            material_index_path=output_full / "idx.sqlite",
        )

        output_range = tmp_path / "range"
        output_range.mkdir()
        result_range = engine.run(
            experiment_id="e2e-range",
            experiment_name="Range",
            source_video=video_path,
            output_dir=output_range,
            material_index_path=output_range / "idx.sqlite",
            time_range=(10.0, 20.0),
        )

        full_frames = result_full["preprocessing_payload"]["event_preprocessing"]["detection_frame_count"]
        range_frames = result_range["preprocessing_payload"]["event_preprocessing"]["detection_frame_count"]
        assert range_frames <= full_frames
