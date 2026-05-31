from __future__ import annotations

from dataclasses import dataclass

import pytest

from key_action_indexer.frame_time_map import (
    capture_sec_to_video_sec,
    frame_time_map_summary,
    should_use_frame_time_map,
    video_sec_to_capture_sec,
)


@dataclass
class _Source:
    path: str
    fps: float
    duration_sec: float
    frames_csv_path: str | None = None


def test_frame_time_map_defaults_to_source_video_capture_seconds(tmp_path):
    video = tmp_path / "rgb.mp4"
    video.write_bytes(b"placeholder")
    frames = tmp_path / "frames.csv"
    frames.write_text(
        "\n".join(
            [
                "local_time_us,stream_type,frame_id",
                "1000000,depth,0",
                "1000000,rgb,0",
                "1300000,rgb,1",
                "2500000,rgb,2",
                "3000000,depth,1",
            ]
        ),
        encoding="utf-8",
    )
    source = _Source(path=str(video), fps=1.0, duration_sec=2.0, frames_csv_path=str(frames))

    assert video_sec_to_capture_sec(source, 0.0) == 0.0
    assert video_sec_to_capture_sec(source, 1.0) == 1.0
    assert video_sec_to_capture_sec(source, 2.0) == 2.0
    assert capture_sec_to_video_sec(source, 1.4) == 1.4

    assert video_sec_to_capture_sec(source, 1.0, use_frame_time_map=True) == pytest.approx(0.3)
    assert capture_sec_to_video_sec(source, 1.4, use_frame_time_map=True) == 2.0
    assert should_use_frame_time_map(source) is False
    assert capture_sec_to_video_sec(source, 1.4, use_frame_time_map="auto") == pytest.approx(1.4)

    summary = frame_time_map_summary(source)
    assert summary is not None
    assert summary["frame_count"] == 3
    assert summary["capture_span_sec"] == 1.5
    assert summary["playback_time_basis"] == "source-video playback windows use frames.csv when capture/video drift is significant"
    assert summary["auto_frame_time_map_applied"] is False


def test_frame_time_map_auto_applies_when_capture_and_video_duration_drift(tmp_path):
    video = tmp_path / "rgb.mp4"
    video.write_bytes(b"placeholder")
    frames = tmp_path / "frames.csv"
    frames.write_text(
        "\n".join(
            [
                "local_time_us,stream_type,frame_id",
                "1000000,rgb,0",
                "5000000,rgb,1",
                "9000000,rgb,2",
                "13000000,rgb,3",
            ]
        ),
        encoding="utf-8",
    )
    source = _Source(path=str(video), fps=30.0, duration_sec=1.0, frames_csv_path=str(frames))

    assert should_use_frame_time_map(source) is True
    assert capture_sec_to_video_sec(source, 8.2, use_frame_time_map="auto") == pytest.approx(2 / 3)


def test_frame_time_map_identity_fallback_without_sidecar(tmp_path):
    video = tmp_path / "rgb.mp4"
    video.write_bytes(b"placeholder")
    source = _Source(path=str(video), fps=30.0, duration_sec=10.0)

    assert video_sec_to_capture_sec(source, 4.25) == 4.25
    assert capture_sec_to_video_sec(source, 4.25) == 4.25
    assert frame_time_map_summary(source) is None
