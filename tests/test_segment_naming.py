from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from labsopguard.event_preprocessing.activity_presegmenter import ActivitySegment
from labsopguard.event_preprocessing.experiment_segmenter import (
    ExperimentSegmentation,
    ExperimentSegment,
)
from labsopguard.event_preprocessing.segment_naming import name_experiment_segments


def _create_video(path: Path, fps: int = 10, duration_sec: float = 5.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (240, 160))
    total = int(fps * duration_sec)
    for index in range(total):
        frame = np.zeros((160, 240, 3), dtype=np.uint8)
        frame[:] = (50, 120, 200) if index < total // 2 else (80, 180, 100)
        writer.write(frame)
    writer.release()


def test_segment_naming_falls_back_without_vlm(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LABSOPGUARD_SEGMENT_NAMING_VLM_ENABLED", "0")
    video_path = tmp_path / "source.mp4"
    _create_video(video_path)
    segmentation = ExperimentSegmentation(
        video_duration_sec=5.0,
        total_segments=2,
        boundaries=[],
        unassigned_time_sec=0.0,
        segments=[
            ExperimentSegment(
                segment_id="seg_0",
                index=0,
                start_sec=0.0,
                end_sec=2.0,
                duration_sec=2.0,
                activity_segments=[ActivitySegment(0.0, 2.0, 1.0, 1.0, "motion")],
            ),
            ExperimentSegment(
                segment_id="seg_1",
                index=1,
                start_sec=3.0,
                end_sec=5.0,
                duration_sec=2.0,
                activity_segments=[ActivitySegment(3.0, 5.0, 1.0, 1.0, "motion")],
            ),
        ],
    )

    manifest = name_experiment_segments(
        video_path=video_path,
        segmentation=segmentation,
        output_dir=tmp_path / "artifacts",
    )

    assert manifest["segment_count"] == 2
    assert manifest["vlm_enabled"] is False
    assert segmentation.segments[0].display_name == "实验 1"
    assert segmentation.segments[1].display_name == "实验 2"
    assert segmentation.segments[0].naming_source == "fallback"
    assert manifest["segments"][0]["representative_frame_path"]
