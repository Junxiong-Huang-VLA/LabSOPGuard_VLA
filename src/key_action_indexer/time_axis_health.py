from __future__ import annotations

import csv
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STATUS_HEALTHY = "healthy"
STATUS_WARNING = "warning"
STATUS_UNRELIABLE = "time_axis_unreliable"

LARGE_GAP_SEC = 5.0
UNRELIABLE_GAP_SEC = 30.0
DURATION_DELTA_FLOOR_SEC = 30.0
DURATION_DELTA_RATIO = 0.02
DUAL_START_WARNING_SEC = 2.0
DUAL_START_UNRELIABLE_SEC = 30.0
DUAL_DURATION_WARNING_SEC = 30.0


@dataclass(frozen=True)
class TimeAxisHealth:
    status: str
    role: str | None
    video_path: str | None
    frames_csv_path: str | None
    capture_span_sec: float | None
    mp4_duration_sec: float | None
    duration_delta_sec: float | None
    duration_ratio: float | None
    gap_count: int
    largest_gap_sec: float
    gap_total_sec: float
    reasons: list[str]
    can_publish_formal_materials: bool
    can_write_video_memory: bool
    frame_count: int = 0
    capture_start_local_time_us: int | None = None
    capture_end_local_time_us: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _FrameStats:
    frame_count: int
    capture_span_sec: float
    gap_count: int
    largest_gap_sec: float
    gap_total_sec: float
    capture_start_local_time_us: int
    capture_end_local_time_us: int


def analyze_video_time_axis(
    video_path: Any,
    frames_csv_path: str | os.PathLike[str] | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Assess one video time axis using ffprobe duration and RGB frames.csv time.

    The function is intentionally side-effect free and tolerant of missing
    video tooling so dry-run flows can still produce a diagnostic payload.
    """

    source = _coerce_source(video_path, frames_csv_path, role)
    reasons: list[str] = []

    frame_stats = _read_frame_stats(source["frames_csv_path"], reasons)
    mp4_duration_sec = _safe_probe_duration(source["video_path"], reasons)

    capture_span_sec = frame_stats.capture_span_sec if frame_stats else None
    duration_delta_sec: float | None = None
    duration_ratio: float | None = None
    if capture_span_sec is not None and mp4_duration_sec is not None and mp4_duration_sec > 0:
        duration_delta_sec = abs(float(capture_span_sec) - float(mp4_duration_sec))
        duration_ratio = float(capture_span_sec) / float(mp4_duration_sec)
        limit = max(DURATION_DELTA_FLOOR_SEC, DURATION_DELTA_RATIO * float(mp4_duration_sec))
        if duration_delta_sec > limit:
            reasons.append(f"capture_mp4_duration_delta_{duration_delta_sec:.3f}s_exceeds_{limit:.3f}s")

    if frame_stats and frame_stats.gap_count:
        reasons.append(
            f"capture_gap_count_{frame_stats.gap_count}_largest_{frame_stats.largest_gap_sec:.3f}s"
        )
        if frame_stats.largest_gap_sec > UNRELIABLE_GAP_SEC:
            reasons.append(f"capture_gap_largest_{frame_stats.largest_gap_sec:.3f}s_exceeds_30.000s")

    status = _single_view_status(reasons)
    health = TimeAxisHealth(
        status=status,
        role=source["role"],
        video_path=str(source["video_path"]) if source["video_path"] is not None else None,
        frames_csv_path=str(source["frames_csv_path"]) if source["frames_csv_path"] is not None else None,
        capture_span_sec=_round_or_none(capture_span_sec),
        mp4_duration_sec=_round_or_none(mp4_duration_sec),
        duration_delta_sec=_round_or_none(duration_delta_sec),
        duration_ratio=_round_or_none(duration_ratio),
        gap_count=frame_stats.gap_count if frame_stats else 0,
        largest_gap_sec=_round_or_zero(frame_stats.largest_gap_sec if frame_stats else 0.0),
        gap_total_sec=_round_or_zero(frame_stats.gap_total_sec if frame_stats else 0.0),
        reasons=reasons,
        can_publish_formal_materials=status == STATUS_HEALTHY,
        can_write_video_memory=status != STATUS_UNRELIABLE,
        frame_count=frame_stats.frame_count if frame_stats else 0,
        capture_start_local_time_us=frame_stats.capture_start_local_time_us if frame_stats else None,
        capture_end_local_time_us=frame_stats.capture_end_local_time_us if frame_stats else None,
    )
    return health.to_dict()


def analyze_dual_view_time_axis(third: Any, first: Any) -> dict[str, Any]:
    """Assess two views and flag start-clock disagreements before formal use."""

    third_health = analyze_video_time_axis(third, role=_role_or_default(third, "third_person"))
    first_health = analyze_video_time_axis(first, role=_role_or_default(first, "first_person"))
    reasons: list[str] = []
    reasons.extend(_prefix_reasons("third_person", third_health["reasons"]))
    reasons.extend(_prefix_reasons("first_person", first_health["reasons"]))

    start_delta_sec: float | None = None
    third_start = third_health.get("capture_start_local_time_us")
    first_start = first_health.get("capture_start_local_time_us")
    if third_start is None or first_start is None:
        reasons.append("dual_start_delta_unavailable")
    else:
        start_delta_sec = abs(float(third_start) - float(first_start)) / 1_000_000.0
        if start_delta_sec > DUAL_START_UNRELIABLE_SEC:
            reasons.append(f"dual_start_delta_{start_delta_sec:.3f}s_exceeds_30.000s")
        elif start_delta_sec > DUAL_START_WARNING_SEC:
            reasons.append(f"dual_start_delta_{start_delta_sec:.3f}s_exceeds_2.000s")

    capture_span_diff_sec = _abs_optional_delta(
        third_health.get("capture_span_sec"),
        first_health.get("capture_span_sec"),
    )
    if capture_span_diff_sec is not None and capture_span_diff_sec > DUAL_DURATION_WARNING_SEC:
        reasons.append(f"dual_capture_span_delta_{capture_span_diff_sec:.3f}s_exceeds_30.000s")
    mp4_duration_diff_sec = _abs_optional_delta(
        third_health.get("mp4_duration_sec"),
        first_health.get("mp4_duration_sec"),
    )
    if mp4_duration_diff_sec is not None and mp4_duration_diff_sec > DUAL_DURATION_WARNING_SEC:
        reasons.append(f"dual_mp4_duration_delta_{mp4_duration_diff_sec:.3f}s_exceeds_30.000s")

    status = _combined_status([third_health["status"], first_health["status"]], reasons)
    gap_count = int(third_health["gap_count"]) + int(first_health["gap_count"])
    largest_gap_sec = max(float(third_health["largest_gap_sec"]), float(first_health["largest_gap_sec"]))
    gap_total_sec = float(third_health["gap_total_sec"]) + float(first_health["gap_total_sec"])

    return {
        "status": status,
        "third_person": third_health,
        "first_person": first_health,
        "capture_span_sec": {
            "third_person": third_health["capture_span_sec"],
            "first_person": first_health["capture_span_sec"],
        },
        "mp4_duration_sec": {
            "third_person": third_health["mp4_duration_sec"],
            "first_person": first_health["mp4_duration_sec"],
        },
        "duration_delta_sec": {
            "third_person": third_health["duration_delta_sec"],
            "first_person": first_health["duration_delta_sec"],
        },
        "duration_ratio": {
            "third_person": third_health["duration_ratio"],
            "first_person": first_health["duration_ratio"],
        },
        "dual_start_delta_sec": _round_or_none(start_delta_sec),
        "dual_capture_span_delta_sec": _round_or_none(capture_span_diff_sec),
        "dual_mp4_duration_delta_sec": _round_or_none(mp4_duration_diff_sec),
        "gap_count": gap_count,
        "largest_gap_sec": _round_or_zero(largest_gap_sec),
        "gap_total_sec": _round_or_zero(gap_total_sec),
        "reasons": reasons,
        "can_publish_formal_materials": status == STATUS_HEALTHY,
        "can_write_video_memory": status != STATUS_UNRELIABLE,
    }


def _coerce_source(
    video_path: Any,
    frames_csv_path: str | os.PathLike[str] | None,
    role: str | None,
) -> dict[str, Any]:
    if isinstance(video_path, dict):
        return {
            "video_path": _path_or_none(video_path.get("path") or video_path.get("video_path")),
            "frames_csv_path": _path_or_none(frames_csv_path or video_path.get("frames_csv_path")),
            "role": role or _str_or_none(video_path.get("role") or video_path.get("name")),
        }
    if isinstance(video_path, (tuple, list)) and len(video_path) >= 2:
        return {
            "video_path": _path_or_none(video_path[0]),
            "frames_csv_path": _path_or_none(frames_csv_path or video_path[1]),
            "role": role,
        }
    if not isinstance(video_path, (str, os.PathLike)):
        return {
            "video_path": _path_or_none(getattr(video_path, "path", None)),
            "frames_csv_path": _path_or_none(frames_csv_path or getattr(video_path, "frames_csv_path", None)),
            "role": role or _str_or_none(getattr(video_path, "role", None) or getattr(video_path, "name", None)),
        }
    return {
        "video_path": _path_or_none(video_path),
        "frames_csv_path": _path_or_none(frames_csv_path),
        "role": role,
    }


def _read_frame_stats(frames_csv_path: Path | None, reasons: list[str]) -> _FrameStats | None:
    if frames_csv_path is None:
        reasons.append("frames_csv_missing")
        return None
    if not frames_csv_path.exists():
        reasons.append("frames_csv_missing")
        return None

    timestamps: list[int] = []
    try:
        with frames_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                reasons.append("frames_csv_missing_header")
                return None
            if "local_time_us" not in reader.fieldnames:
                reasons.append("frames_csv_missing_local_time_us")
                return None
            stream_column = "stream_type" if "stream_type" in reader.fieldnames else None
            for row in reader:
                if stream_column and str(row.get(stream_column) or "").strip().lower() != "rgb":
                    continue
                value = _safe_int(row.get("local_time_us"))
                if value is not None:
                    timestamps.append(value)
    except OSError:
        reasons.append("frames_csv_unreadable")
        return None

    timestamps.sort()
    if len(timestamps) < 2:
        reasons.append("frames_csv_has_fewer_than_two_rgb_rows")
        return None

    gaps = [(right - left) / 1_000_000.0 for left, right in zip(timestamps, timestamps[1:])]
    large_gaps = [gap for gap in gaps if gap > LARGE_GAP_SEC]
    return _FrameStats(
        frame_count=len(timestamps),
        capture_span_sec=(timestamps[-1] - timestamps[0]) / 1_000_000.0,
        gap_count=len(large_gaps),
        largest_gap_sec=max(gaps) if gaps else 0.0,
        gap_total_sec=sum(large_gaps),
        capture_start_local_time_us=timestamps[0],
        capture_end_local_time_us=timestamps[-1],
    )


def _safe_probe_duration(video_path: Path | None, reasons: list[str]) -> float | None:
    if video_path is None:
        reasons.append("video_path_missing")
        return None
    try:
        duration = _ffprobe_duration_sec(video_path)
    except Exception:
        duration = None
    if duration is None:
        reasons.append("ffprobe_duration_unavailable")
        return None
    return duration


def _ffprobe_duration_sec(video_path: Path) -> float | None:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return _safe_float(completed.stdout.strip())


def _single_view_status(reasons: list[str]) -> str:
    if _has_unreliable_reason(reasons):
        return STATUS_UNRELIABLE
    if reasons:
        return STATUS_WARNING
    return STATUS_HEALTHY


def _combined_status(view_statuses: list[str], reasons: list[str]) -> str:
    if STATUS_UNRELIABLE in view_statuses or _has_unreliable_reason(reasons):
        return STATUS_UNRELIABLE
    if STATUS_WARNING in view_statuses or reasons:
        return STATUS_WARNING
    return STATUS_HEALTHY


def _has_unreliable_reason(reasons: list[str]) -> bool:
    return any(
        "capture_mp4_duration_delta_" in reason
        or "capture_gap_largest_" in reason
        or "dual_start_delta_" in reason and "_exceeds_30.000s" in reason
        for reason in reasons
    )


def _prefix_reasons(role: str, reasons: list[str]) -> list[str]:
    return [f"{role}:{reason}" for reason in reasons]


def _role_or_default(source: Any, default: str) -> str:
    if isinstance(source, dict):
        return str(source.get("role") or source.get("name") or default)
    value = getattr(source, "role", None) or getattr(source, "name", None)
    return str(value or default)


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    return Path(text) if text else None


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError, OverflowError):
        return None


def _abs_optional_delta(left: Any, right: Any) -> float | None:
    left_number = _safe_float(left)
    right_number = _safe_float(right)
    if left_number is None or right_number is None:
        return None
    return abs(left_number - right_number)


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _round_or_zero(value: float) -> float:
    return round(float(value), 6)
