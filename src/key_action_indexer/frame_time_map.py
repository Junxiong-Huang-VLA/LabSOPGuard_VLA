from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_TIMESTAMP_COLUMNS = ("local_time_us", "packet_system_timestamp_us", "frame_system_timestamp_us", "timestamp_us")


@dataclass(frozen=True)
class FrameTimeMap:
    video_path: str
    frames_csv_path: str
    timestamp_column: str
    frame_count: int
    encoded_fps: float
    capture_span_sec: float
    capture_rel_times_sec: tuple[float, ...]

    def video_sec_to_capture_sec(self, video_sec: float) -> float:
        if not self.capture_rel_times_sec or self.encoded_fps <= 0:
            return max(0.0, float(video_sec))
        index = int(round(max(0.0, float(video_sec)) * self.encoded_fps))
        index = max(0, min(len(self.capture_rel_times_sec) - 1, index))
        return float(self.capture_rel_times_sec[index])

    def capture_sec_to_video_sec(self, capture_sec: float) -> float:
        if not self.capture_rel_times_sec or self.encoded_fps <= 0:
            return max(0.0, float(capture_sec))
        target = max(0.0, float(capture_sec))
        index = bisect.bisect_left(self.capture_rel_times_sec, target)
        if index <= 0:
            nearest = 0
        elif index >= len(self.capture_rel_times_sec):
            nearest = len(self.capture_rel_times_sec) - 1
        else:
            before = self.capture_rel_times_sec[index - 1]
            after = self.capture_rel_times_sec[index]
            nearest = index if abs(after - target) < abs(before - target) else index - 1
        return float(nearest) / self.encoded_fps


def discover_frame_time_map(source: Any) -> FrameTimeMap | None:
    video_path = Path(str(getattr(source, "path", "") or ""))
    if not video_path.exists():
        return None
    frames_path_value = str(getattr(source, "frames_csv_path", "") or "").strip()
    frames_path = Path(frames_path_value) if frames_path_value else video_path.parent / "frames.csv"
    if not frames_path.exists():
        return None
    fps = _safe_float(getattr(source, "fps", None))
    duration_sec = _safe_float(getattr(source, "duration_sec", None))
    try:
        return _load_frame_time_map(str(video_path), str(frames_path), fps, duration_sec)
    except Exception:
        return None


def should_use_frame_time_map(source: Any) -> bool:
    """Return whether frames.csv should drive capture/video time conversion.

    Most single-file recordings have playback seconds close enough to capture
    seconds that identity mapping is less surprising. Concatenated or dropped
    frame recordings can diverge by minutes, and those must use frames.csv or
    the two camera views will be numerically aligned while showing different
    physical moments.
    """

    frame_map = discover_frame_time_map(source)
    if frame_map is None:
        return False
    duration_sec = _safe_float(getattr(source, "duration_sec", None))
    if duration_sec is None or duration_sec <= 0:
        return True
    drift_sec = abs(float(frame_map.capture_span_sec) - float(duration_sec))
    return drift_sec >= max(5.0, float(duration_sec) * 0.01)


def video_sec_to_capture_sec(source: Any, video_sec: float, *, use_frame_time_map: bool | str = False) -> float:
    """Map source-video playback seconds onto capture/session seconds.

    Playback windows use the unified capture/session clock by default. The
    frames.csv nearest-frame map is intentionally opt-in because frame capture
    jitter can otherwise compress or stretch source-video windows.
    """
    if use_frame_time_map == "auto":
        use_frame_time_map = should_use_frame_time_map(source)
    if not use_frame_time_map:
        return max(0.0, float(video_sec))
    frame_map = discover_frame_time_map(source)
    if frame_map is None:
        return max(0.0, float(video_sec))
    return frame_map.video_sec_to_capture_sec(float(video_sec))


def capture_sec_to_video_sec(source: Any, capture_sec: float, *, use_frame_time_map: bool | str = False) -> float:
    """Map capture/session seconds onto source-video playback seconds.

    The default is identity for source-video playback. Pass
    ``use_frame_time_map=True`` only for diagnostics that explicitly need the
    nearest RGB frames.csv mapping.
    """
    if use_frame_time_map == "auto":
        use_frame_time_map = should_use_frame_time_map(source)
    if not use_frame_time_map:
        return max(0.0, float(capture_sec))
    frame_map = discover_frame_time_map(source)
    if frame_map is None:
        return max(0.0, float(capture_sec))
    return frame_map.capture_sec_to_video_sec(float(capture_sec))


def frame_time_map_summary(source: Any) -> dict[str, Any] | None:
    frame_map = discover_frame_time_map(source)
    if frame_map is None:
        return None
    return {
        "schema_version": "key_action_frame_time_map.v1",
        "video_path": frame_map.video_path,
        "frames_csv_path": frame_map.frames_csv_path,
        "timestamp_column": frame_map.timestamp_column,
        "frame_count": frame_map.frame_count,
        "encoded_fps": round(frame_map.encoded_fps, 6),
        "capture_span_sec": round(frame_map.capture_span_sec, 6),
        "playback_time_basis": "source-video playback windows use frames.csv when capture/video drift is significant",
        "frame_time_map_basis": "diagnostic nearest RGB frames.csv capture timestamp by encoded frame index",
        "auto_frame_time_map_applied": should_use_frame_time_map(source),
    }


@lru_cache(maxsize=16)
def _load_frame_time_map(video_path: str, frames_csv_path: str, fps: float | None, duration_sec: float | None) -> FrameTimeMap:
    rows: list[float] = []
    timestamp_column: str | None = None
    with Path(frames_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"frames.csv has no header: {frames_csv_path}")
        stream_column = "stream_type" if "stream_type" in reader.fieldnames else None
        for candidate in _TIMESTAMP_COLUMNS:
            if candidate in reader.fieldnames:
                timestamp_column = candidate
                break
        if timestamp_column is None:
            raise ValueError(f"frames.csv has no usable timestamp column: {frames_csv_path}")
        for row in reader:
            if stream_column and str(row.get(stream_column) or "").strip().lower() != "rgb":
                continue
            value = _safe_float(row.get(timestamp_column))
            if value is None:
                continue
            rows.append(value / 1_000_000.0)
    if len(rows) < 2:
        raise ValueError(f"frames.csv has fewer than two RGB timestamp rows: {frames_csv_path}")
    first = rows[0]
    rel_times = tuple(max(0.0, value - first) for value in rows)
    capture_span = rel_times[-1]
    encoded_fps = _resolve_encoded_fps(row_count=len(rel_times), fps=fps, duration_sec=duration_sec, capture_span_sec=capture_span)
    return FrameTimeMap(
        video_path=video_path,
        frames_csv_path=frames_csv_path,
        timestamp_column=timestamp_column,
        frame_count=len(rel_times),
        encoded_fps=encoded_fps,
        capture_span_sec=capture_span,
        capture_rel_times_sec=rel_times,
    )


def _resolve_encoded_fps(*, row_count: int, fps: float | None, duration_sec: float | None, capture_span_sec: float) -> float:
    if duration_sec is not None and duration_sec > 0 and row_count > 1:
        return max(0.001, float(row_count - 1) / float(duration_sec))
    if fps is not None and fps > 0:
        return float(fps)
    if capture_span_sec > 0 and row_count > 1:
        return max(0.001, float(row_count - 1) / float(capture_span_sec))
    return 30.0


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number
