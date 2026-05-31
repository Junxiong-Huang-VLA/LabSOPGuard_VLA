from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from key_action_indexer import time_axis_health
from key_action_indexer.time_axis_health import analyze_dual_view_time_axis, analyze_video_time_axis


@dataclass
class _Source:
    path: str
    frames_csv_path: str
    role: str | None = None
    name: str | None = None


def _write_frames(path: Path, times_us: list[int]) -> None:
    lines = ["local_time_us,stream_type,frame_id"]
    lines.extend(f"{time_us},rgb,{index}" for index, time_us in enumerate(times_us))
    path.write_text("\n".join(lines), encoding="utf-8")


def test_single_view_healthy_time_axis(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "third.mp4"
    frames = tmp_path / "frames.csv"
    _write_frames(frames, [1_000_000, 6_000_000, 11_000_000])
    monkeypatch.setattr(time_axis_health, "_ffprobe_duration_sec", lambda path: 10.0)

    result = analyze_video_time_axis(video, frames, role="third_person")

    assert result["status"] == "healthy"
    assert result["capture_span_sec"] == pytest.approx(10.0)
    assert result["mp4_duration_sec"] == pytest.approx(10.0)
    assert result["duration_delta_sec"] == pytest.approx(0.0)
    assert result["duration_ratio"] == pytest.approx(1.0)
    assert result["gap_count"] == 0
    assert result["largest_gap_sec"] == pytest.approx(5.0)
    assert result["gap_total_sec"] == pytest.approx(0.0)
    assert result["reasons"] == []
    assert result["can_publish_formal_materials"] is True
    assert result["can_write_video_memory"] is True


def test_single_view_marks_large_duration_drift_unreliable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "first.mp4"
    frames = tmp_path / "frames.csv"
    _write_frames(frames, [0, 20_000_000, 80_000_000])
    monkeypatch.setattr(time_axis_health, "_ffprobe_duration_sec", lambda path: 10.0)

    result = analyze_video_time_axis(_Source(str(video), str(frames), role="first_person"))

    assert result["status"] == "time_axis_unreliable"
    assert result["capture_span_sec"] == pytest.approx(80.0)
    assert result["duration_delta_sec"] == pytest.approx(70.0)
    assert result["gap_count"] == 2
    assert result["largest_gap_sec"] == pytest.approx(60.0)
    assert any(reason.startswith("capture_mp4_duration_delta_70.000s") for reason in result["reasons"])
    assert "capture_gap_largest_60.000s_exceeds_30.000s" in result["reasons"]
    assert result["can_publish_formal_materials"] is False
    assert result["can_write_video_memory"] is False


def test_ffprobe_missing_is_graceful_warning(tmp_path: Path) -> None:
    video = tmp_path / "missing-video.mp4"
    frames = tmp_path / "frames.csv"
    _write_frames(frames, [0, 1_000_000, 2_000_000])

    result = analyze_video_time_axis(video, frames)

    assert result["status"] == "warning"
    assert result["mp4_duration_sec"] is None
    assert result["capture_span_sec"] == pytest.approx(2.0)
    assert "ffprobe_duration_unavailable" in result["reasons"]
    assert result["can_publish_formal_materials"] is False
    assert result["can_write_video_memory"] is True


def test_frames_csv_filters_rgb_and_warns_for_gaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    video = tmp_path / "third.mp4"
    frames = tmp_path / "frames.csv"
    frames.write_text(
        "\n".join(
            [
                "local_time_us,stream_type,frame_id",
                "0,depth,0",
                "0,rgb,0",
                "6_000_000,bad,ignored",
                "7_000_000,rgb,1",
                "8_000_000,depth,1",
                "12_000_000,rgb,2",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(time_axis_health, "_ffprobe_duration_sec", lambda path: 12.0)

    result = analyze_video_time_axis(video, frames)

    assert result["status"] == "warning"
    assert result["frame_count"] == 3
    assert result["gap_count"] == 1
    assert result["largest_gap_sec"] == pytest.approx(7.0)
    assert result["gap_total_sec"] == pytest.approx(7.0)
    assert "capture_gap_count_1_largest_7.000s" in result["reasons"]
    assert result["can_write_video_memory"] is True


def test_dual_view_start_delta_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    third_video = tmp_path / "third.mp4"
    first_video = tmp_path / "first.mp4"
    third_frames = tmp_path / "third_frames.csv"
    first_frames = tmp_path / "first_frames.csv"
    _write_frames(third_frames, [1_000_000, 6_000_000, 11_000_000])
    _write_frames(first_frames, [4_500_000, 9_500_000, 14_500_000])
    monkeypatch.setattr(time_axis_health, "_ffprobe_duration_sec", lambda path: 10.0)

    result = analyze_dual_view_time_axis(
        _Source(str(third_video), str(third_frames), name="third_person"),
        _Source(str(first_video), str(first_frames), name="first_person"),
    )

    assert result["status"] == "warning"
    assert result["dual_start_delta_sec"] == pytest.approx(3.5)
    assert "dual_start_delta_3.500s_exceeds_2.000s" in result["reasons"]
    assert result["third_person"]["status"] == "healthy"
    assert result["first_person"]["status"] == "healthy"
    assert result["can_publish_formal_materials"] is False
    assert result["can_write_video_memory"] is True


def test_dual_view_duration_delta_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    third_video = tmp_path / "third.mp4"
    first_video = tmp_path / "first.mp4"
    third_frames = tmp_path / "third_frames.csv"
    first_frames = tmp_path / "first_frames.csv"
    _write_frames(third_frames, [1_000_000, 6_000_000, 11_000_000])
    _write_frames(first_frames, [1_000_000 + step * 5_000_000 for step in range(11)])

    def _duration(path: Path) -> float:
        return 10.0 if path == third_video else 50.0

    monkeypatch.setattr(time_axis_health, "_ffprobe_duration_sec", _duration)

    result = analyze_dual_view_time_axis(
        _Source(str(third_video), str(third_frames), role="third_person"),
        _Source(str(first_video), str(first_frames), role="first_person"),
    )

    assert result["status"] == "warning"
    assert result["dual_capture_span_delta_sec"] == pytest.approx(40.0)
    assert result["dual_mp4_duration_delta_sec"] == pytest.approx(40.0)
    assert "dual_capture_span_delta_40.000s_exceeds_30.000s" in result["reasons"]
    assert "dual_mp4_duration_delta_40.000s_exceeds_30.000s" in result["reasons"]
    assert result["can_publish_formal_materials"] is False
    assert result["can_write_video_memory"] is True


def test_dual_view_start_delta_unreliable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    third_video = tmp_path / "third.mp4"
    first_video = tmp_path / "first.mp4"
    third_frames = tmp_path / "third_frames.csv"
    first_frames = tmp_path / "first_frames.csv"
    _write_frames(third_frames, [0, 5_000_000, 10_000_000])
    _write_frames(first_frames, [35_500_000, 40_500_000, 45_500_000])
    monkeypatch.setattr(time_axis_health, "_ffprobe_duration_sec", lambda path: 10.0)

    result = analyze_dual_view_time_axis((third_video, third_frames), (first_video, first_frames))

    assert result["status"] == "time_axis_unreliable"
    assert result["dual_start_delta_sec"] == pytest.approx(35.5)
    assert "dual_start_delta_35.500s_exceeds_30.000s" in result["reasons"]
    assert result["can_publish_formal_materials"] is False
    assert result["can_write_video_memory"] is False
