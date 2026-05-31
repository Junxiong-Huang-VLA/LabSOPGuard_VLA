from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from labsopguard.event_preprocessing.activity_presegmenter import ActivitySegment
from labsopguard.event_preprocessing.experiment_segmenter import (
    ExperimentSegmentation,
    ExperimentSegment,
)
from labsopguard.event_preprocessing.segmentation_preview import render_segmentation_preview


def _create_video(path: Path, fps: int = 12, duration_sec: float = 6.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 180))
    total = int(fps * duration_sec)
    for index in range(total):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        t = index / fps
        color = (40, 130, 220) if t < duration_sec / 2 else (80, 200, 120)
        frame[:] = color
        cv2.circle(frame, (int((t / duration_sec) * 320), 90), 18, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_render_segmentation_preview_video(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    _create_video(video_path)
    segmentation = ExperimentSegmentation(
        video_duration_sec=6.0,
        total_segments=2,
        boundaries=[],
        unassigned_time_sec=0.0,
        segments=[
            ExperimentSegment(
                segment_id="seg_0",
                index=0,
                start_sec=0.0,
                end_sec=2.8,
                duration_sec=2.8,
                activity_segments=[ActivitySegment(0.0, 2.8, 1.0, 1.0, "motion")],
            ),
            ExperimentSegment(
                segment_id="seg_1",
                index=1,
                start_sec=3.2,
                end_sec=6.0,
                duration_sec=2.8,
                activity_segments=[ActivitySegment(3.2, 6.0, 1.0, 1.0, "motion")],
            ),
        ],
    )

    output_path = tmp_path / "segmentation_preview.mp4"
    manifest = render_segmentation_preview(
        video_path=video_path,
        segmentation=segmentation,
        output_path=output_path,
        sample_interval_sec=1.0,
        max_frames=12,
        output_fps=4.0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert manifest["experiment_segment_count"] == 2
    assert manifest["frame_count"] > 0


def test_render_segmentation_preview_time_range(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    _create_video(video_path)
    segmentation = ExperimentSegmentation(
        video_duration_sec=6.0,
        total_segments=1,
        boundaries=[],
        unassigned_time_sec=0.0,
        segments=[
            ExperimentSegment(
                segment_id="seg_0",
                index=0,
                start_sec=3.0,
                end_sec=6.0,
                duration_sec=3.0,
                activity_segments=[ActivitySegment(3.0, 6.0, 1.0, 1.0, "motion")],
                display_name="实验 1",
            ),
        ],
    )

    output_path = tmp_path / "segment_preview.mp4"
    manifest = render_segmentation_preview(
        video_path=video_path,
        segmentation=segmentation,
        output_path=output_path,
        time_range=(3.0, 6.0),
        sample_interval_sec=1.0,
        max_frames=8,
        output_fps=4.0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert manifest["render_start_sec"] == 3.0
    assert manifest["render_end_sec"] == 6.0
    assert manifest["frame_count"] > 0


class _FakeDetector:
    def detect_frame(self, frame):
        h, w = frame.shape[:2]
        return [
            {
                "label": "gloved_hand",
                "confidence": 0.86,
                "bbox": [int(w * 0.34), int(h * 0.34), int(w * 0.54), int(h * 0.64)],
            },
            {
                "label": "pipette",
                "confidence": 0.81,
                "bbox": [int(w * 0.50), int(h * 0.38), int(w * 0.70), int(h * 0.62)],
            },
        ]


def test_render_segmentation_preview_yolo_evidence_overlay(tmp_path: Path) -> None:
    video_path = tmp_path / "source.mp4"
    _create_video(video_path)
    segmentation = ExperimentSegmentation(
        video_duration_sec=6.0,
        total_segments=1,
        boundaries=[],
        unassigned_time_sec=0.0,
        segments=[
            ExperimentSegment(
                segment_id="seg_0",
                index=0,
                start_sec=0.0,
                end_sec=6.0,
                duration_sec=6.0,
                activity_segments=[ActivitySegment(0.0, 6.0, 1.0, 1.0, "motion")],
                display_name="实验 1",
            ),
        ],
    )

    output_path = tmp_path / "evidence_preview.mp4"
    manifest = render_segmentation_preview(
        video_path=video_path,
        segmentation=segmentation,
        output_path=output_path,
        detector=_FakeDetector(),
        yolo_overlay=True,
        sample_interval_sec=1.0,
        max_frames=8,
        output_fps=4.0,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert manifest["annotation_mode"] == "yolo_hand_object_key_action"
    assert manifest["yolo_overlay_enabled"] is True
    assert manifest["yolo_detection_count"] > 0
    assert manifest["hand_object_interaction_count"] > 0
    assert "Key action: tool handling" in manifest["key_action_labels"]
