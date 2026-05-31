from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import StreamConfig, TimestampConfig


class FrameDataError(ValueError):
    """Raised when a frames.csv file is malformed."""


@dataclass(frozen=True)
class CameraFrames:
    camera_id: str
    stream_path: Path
    frames_path: Path
    timestamp_col: str
    frame_count: int
    median_frame_interval_us: float | None
    df: pd.DataFrame
    warnings: tuple[str, ...]


def load_camera_frames(stream: StreamConfig, timestamp: TimestampConfig) -> CameraFrames:
    frames_path = stream.frames_path
    if not frames_path.exists():
        raise FrameDataError(f"{stream.camera_id}: frames.csv does not exist: {frames_path}")
    df = pd.read_csv(frames_path)
    if df.empty:
        raise FrameDataError(f"{stream.camera_id}: frames.csv is empty: {frames_path}")
    df = _apply_filters(df, stream)
    timestamp_col = choose_timestamp_col(df.columns, timestamp, camera_id=stream.camera_id)
    warnings: list[str] = []
    original_ts = df[timestamp_col]
    numeric_ts = pd.to_numeric(original_ts, errors="coerce")
    bad_count = int(numeric_ts.isna().sum())
    if bad_count:
        raise FrameDataError(f"{stream.camera_id}: timestamp column {timestamp_col!r} contains {bad_count} non-numeric values")

    out = df.copy()
    if "frame_index" in out.columns:
        frame_index = out["frame_index"]
    elif "frame_id" in out.columns:
        frame_index = out["frame_id"]
    else:
        frame_index = pd.Series(range(len(out)), index=out.index)
    out["frame_index"] = frame_index
    out["timestamp_us"] = numeric_ts.astype("int64")

    if out["timestamp_us"].duplicated().any():
        duplicate_count = int(out["timestamp_us"].duplicated().sum())
        warnings.append(f"{stream.camera_id}: timestamp column {timestamp_col!r} has {duplicate_count} duplicate values")
    if not out["timestamp_us"].is_monotonic_increasing:
        warnings.append(f"{stream.camera_id}: frames.csv was not sorted by {timestamp_col!r}; rows were sorted before sync")
    out = out.sort_values(["timestamp_us", "frame_index"], kind="mergesort").reset_index(drop=True)

    intervals = out["timestamp_us"].diff().dropna()
    median_interval = float(intervals.median()) if not intervals.empty else None
    return CameraFrames(
        camera_id=stream.camera_id,
        stream_path=stream.stream_path,
        frames_path=frames_path,
        timestamp_col=timestamp_col,
        frame_count=len(out),
        median_frame_interval_us=median_interval,
        df=out[["frame_index", "timestamp_us"]],
        warnings=tuple(warnings),
    )


def _apply_filters(df: pd.DataFrame, stream: StreamConfig) -> pd.DataFrame:
    if not stream.filters:
        return df
    filtered = df
    for column, values in stream.filters.items():
        if column not in filtered.columns:
            raise FrameDataError(f"{stream.camera_id}: frame filter column {column!r} does not exist in frames.csv")
        before = len(filtered)
        allowed = {str(value) for value in values}
        filtered = filtered[filtered[column].astype(str).isin(allowed)]
        if filtered.empty:
            allowed_text = ", ".join(sorted(allowed))
            raise FrameDataError(
                f"{stream.camera_id}: frame filter {column} in [{allowed_text}] removed all {before} rows"
            )
    return filtered.copy()


def choose_timestamp_col(columns: Any, timestamp: TimestampConfig, *, camera_id: str = "") -> str:
    available = {str(col) for col in columns}
    for col in timestamp.preferred_cols:
        if col in available:
            return col
    prefix = f"{camera_id}: " if camera_id else ""
    preferred = ", ".join(timestamp.preferred_cols)
    present = ", ".join(sorted(available))
    raise FrameDataError(f"{prefix}none of timestamp.preferred_cols exists in frames.csv; preferred=[{preferred}], columns=[{present}]")
