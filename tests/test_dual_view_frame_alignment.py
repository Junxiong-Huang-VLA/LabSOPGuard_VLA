from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import sys
import types

import pytest

from key_action_indexer.dual_view_frame_alignment import (
    ViewFrameTimeline,
    analyze_dual_view_frame_alignment,
    alignment_quality_report,
    build_alignment_samples,
    _extract_view_frames_by_index,
    run_dual_view_alignment_pipeline,
)


def _timeline(role: str, fps: float, times: list[float]) -> ViewFrameTimeline:
    return ViewFrameTimeline(
        role=role,
        video_path=f"{role}.mp4",
        frames_csv_path=f"{role}_frames.csv",
        duration_sec=max(times) if times else 0.0,
        fps=fps,
        absolute_times_sec=tuple(times),
        local_video_secs=tuple(time - times[0] for time in times),
        source="test_frames_csv",
    )


def test_lower_fps_view_is_base_and_frames_outside_common_range_are_dropped():
    third = _timeline("third_person", 4.0, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    first = _timeline("first_person", 1.0, [1.0, 2.0, 3.0, 4.0])

    samples = build_alignment_samples(
        third=third,
        first=first,
        start_abs_sec=max(third.start_abs_sec, first.start_abs_sec),
        end_abs_sec=min(third.end_abs_sec, first.end_abs_sec),
        target_fps=4.0,
    )
    quality = alignment_quality_report(third=third, first=first, samples=samples, target_fps=4.0, output_paths={})

    assert quality["base_view"] == "first_person"
    assert [sample.absolute_time_sec for sample in samples] == [1.0, 2.0, 3.0]
    assert all(1.0 <= sample.absolute_time_sec < 3.5 for sample in samples)
    assert 0.0 not in [sample.absolute_time_sec for sample in samples]
    assert 4.0 not in [sample.absolute_time_sec for sample in samples]


def test_target_fps_below_base_fps_downsamples_from_base_frames():
    third = _timeline("third_person", 10.0, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1])
    first = _timeline("first_person", 4.0, [1.0, 1.25, 1.5, 1.75, 2.0, 2.25])

    samples = build_alignment_samples(
        third=third,
        first=first,
        start_abs_sec=1.0,
        end_abs_sec=2.3,
        target_fps=2.0,
    )

    assert [sample.absolute_time_sec for sample in samples] == [1.0, 1.5, 2.0]
    assert [sample.first_frame_index for sample in samples] == [0, 2, 4]


def test_preflight_allows_formal_results_for_reliable_nearest_neighbor_alignment(monkeypatch):
    third = _timeline("third_person", 4.0, [0.0, 0.5, 1.0, 1.5, 2.0])
    first = _timeline("first_person", 2.0, [0.0, 0.5, 1.0, 1.5, 2.0])

    def fake_load_view_timeline(role, video_path, frames_csv_path=None):
        return third if role == "third_person" else first

    monkeypatch.setattr(
        "key_action_indexer.dual_view_frame_alignment.load_view_timeline",
        fake_load_view_timeline,
    )

    payload = analyze_dual_view_frame_alignment(
        third_video=Path("third.mp4"),
        first_video=Path("first.mp4"),
        target_fps=2.0,
    )

    assert payload["status"] == "aligned_frame_time_reliable"
    assert payload["formal_results_allowed"] is True
    assert payload["video_memory_allowed"] is True
    assert payload["reasons"] == []


def test_preflight_blocks_formal_results_when_no_common_frame_range(monkeypatch):
    third = _timeline("third_person", 2.0, [0.0, 0.5, 1.0])
    first = _timeline("first_person", 2.0, [2.0, 2.5, 3.0])

    def fake_load_view_timeline(role, video_path, frames_csv_path=None):
        return third if role == "third_person" else first

    monkeypatch.setattr(
        "key_action_indexer.dual_view_frame_alignment.load_view_timeline",
        fake_load_view_timeline,
    )

    payload = analyze_dual_view_frame_alignment(
        third_video=Path("third.mp4"),
        first_video=Path("first.mp4"),
        target_fps=2.0,
    )

    assert payload["status"] == "frame_time_alignment_unreliable"
    assert payload["formal_results_allowed"] is False
    assert payload["video_memory_allowed"] is False
    assert payload["reasons"] == ["no_common_dual_view_frame_time_range"]
    assert payload["sample_count"] == 0


def test_delta_above_threshold_is_dropped_from_sync_samples():
    third = _timeline("third_person", 2.0, [0.0, 0.5, 1.0])
    first = _timeline("first_person", 2.0, [0.4, 0.9, 1.4])

    samples = build_alignment_samples(
        third=third,
        first=first,
        start_abs_sec=0.0,
        end_abs_sec=1.1,
        target_fps=2.0,
        max_pair_delta_sec=0.050,
    )

    assert samples == []


def test_alignment_pipeline_writes_required_stage_artifacts(monkeypatch, tmp_path: Path):
    @dataclass
    class Source:
        name: str
        path: str
        frames_csv_path: str
        start_time: str = "2026-05-28T00:00:00+08:00"
        camera_id: str | None = None

    class Videos:
        def __init__(self, third, first):
            self.third_person = third
            self.first_person = first

        def all_sources(self):
            return {"third_person": self.third_person, "first_person": self.first_person}

    @dataclass
    class Manifest:
        session_id: str
        videos: Videos

    third_video = tmp_path / "third.mp4"
    first_video = tmp_path / "first.mp4"
    third_video.write_bytes(b"third")
    first_video.write_bytes(b"first")
    third_frames = tmp_path / "third_frames.csv"
    first_frames = tmp_path / "first_frames.csv"
    third_frames.write_text(
        "packet_system_timestamp_us,stream_type,frame_id\n0,rgb,0\n500000,rgb,1\n1000000,rgb,2\n",
        encoding="utf-8",
    )
    first_frames.write_text(
        "packet_system_timestamp_us,stream_type,frame_id\n0,rgb,0\n500000,rgb,1\n1000000,rgb,2\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("key_action_indexer.dual_view_frame_alignment._ffprobe_video_fps", lambda path: 2.0)
    monkeypatch.setattr("key_action_indexer.dual_view_frame_alignment._safe_ffprobe_float", lambda path, entry: 1.0)

    manifest = Manifest(
        "artifact_smoke",
        Videos(
            Source("third_person", str(third_video), str(third_frames), camera_id="cam01"),
            Source("first_person", str(first_video), str(first_frames), camera_id="cam02"),
        ),
    )

    summary = run_dual_view_alignment_pipeline(
        manifest,
        tmp_path / "out",
        timestamp_field="packet_system_timestamp_us",
        make_aligned_videos=False,
        target_fps=2.0,
    )

    root = tmp_path / "out" / "dual_view_alignment"
    assert summary["status"] == "alignment_ready_pending_yolo_phase"
    for name in [
        "video_registration.jsonl",
        "time_axis_report.json",
        "state_scan_segments.json",
        "trim_window.json",
        "alignment_units.json",
        "local_offset_report.json",
        "sync_index.csv",
        "sync_index.jsonl",
        "alignment_quality_report.json",
        "phase_consistency_report.json",
        "formal_experiment_windows.json",
        "publish_gate_report.json",
    ]:
        assert (root / name).exists()
    assert third_video.read_bytes() == b"third"
    assert first_video.read_bytes() == b"first"


def test_monotonic_frame_extraction_uses_sequential_read(monkeypatch, tmp_path: Path):
    calls = {"set": 0, "read": 0, "write": 0}

    class FakeFrame:
        shape = (1080, 1920, 3)

    class FakeCapture:
        def __init__(self, _path):
            self.index = 0

        def isOpened(self):
            return True

        def set(self, *_args):
            calls["set"] += 1

        def read(self):
            calls["read"] += 1
            if self.index > 5:
                return False, None
            self.index += 1
            return True, FakeFrame()

        def release(self):
            return None

    def fake_resize(frame, _size):
        return frame

    def fake_imwrite(path, _frame):
        calls["write"] += 1
        Path(path).write_bytes(b"jpg")
        return True

    fake_cv2 = types.SimpleNamespace(
        VideoCapture=FakeCapture,
        CAP_PROP_POS_FRAMES=1,
        resize=fake_resize,
        imwrite=fake_imwrite,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    result = _extract_view_frames_by_index(
        "fake.mp4",
        [1, 3, 5],
        tmp_path,
        max_width=640,
    )

    assert result["read_mode"] == "sequential"
    assert calls["set"] == 0
    assert calls["write"] == 3
    assert (tmp_path / "frame_000000.jpg").exists()
