from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ConfigError, SyncConfig
from .frames import CameraFrames, load_camera_frames
from .manifest import resolve_streams
from .report import SyncReport, write_outputs


@dataclass(frozen=True)
class SyncResult:
    config: SyncConfig
    long_pairs: pd.DataFrame
    wide_pairs: pd.DataFrame
    report: SyncReport
    output_dir: Path


def run_sync(config: SyncConfig) -> SyncResult:
    effective_config, streams, manifest_warnings = resolve_streams(config)
    camera_frames = [load_camera_frames(stream, effective_config.timestamp) for stream in streams]
    by_camera = {item.camera_id: item for item in camera_frames}
    if effective_config.reference_camera not in by_camera:
        raise ConfigError(f"reference_camera {effective_config.reference_camera!r} is not present in streams")
    reference = by_camera[effective_config.reference_camera]
    long_pairs = build_long_pairs(effective_config, reference, camera_frames)
    wide_pairs = build_wide_pairs(reference, long_pairs)
    report = SyncReport.from_result(
        config=effective_config,
        camera_frames=camera_frames,
        long_pairs=long_pairs,
        wide_pairs=wide_pairs,
        warnings=[*manifest_warnings, *(warning for cam in camera_frames for warning in cam.warnings)],
    )
    write_outputs(effective_config, long_pairs, wide_pairs, report)
    return SyncResult(effective_config, long_pairs, wide_pairs, report, effective_config.output_dir)


def build_long_pairs(config: SyncConfig, reference: CameraFrames, cameras: list[CameraFrames]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    ref = reference.df.rename(
        columns={
            "frame_index": "reference_frame_index",
            "timestamp_us": "reference_timestamp_us",
        }
    ).sort_values("reference_timestamp_us")
    for camera in cameras:
        if camera.camera_id == reference.camera_id:
            continue
        target = camera.df.rename(
            columns={
                "frame_index": "matched_frame_index",
                "timestamp_us": "matched_timestamp_us",
            }
        ).sort_values("matched_timestamp_us")
        merged = pd.merge_asof(
            ref,
            target,
            left_on="reference_timestamp_us",
            right_on="matched_timestamp_us",
            direction="nearest",
            tolerance=config.tolerance_us,
        )
        merged.insert(0, "run_id", config.run_id)
        merged.insert(1, "experiment_name", config.experiment_name)
        merged.insert(2, "batch_id", config.batch_id)
        merged.insert(3, "reference_camera", reference.camera_id)
        merged["camera_id"] = camera.camera_id
        merged["time_diff_us"] = merged["matched_timestamp_us"] - merged["reference_timestamp_us"]
        merged["abs_time_diff_us"] = merged["time_diff_us"].abs()
        merged["matched_ok"] = merged["matched_timestamp_us"].notna()
        rows.append(
            merged[
                [
                    "run_id",
                    "experiment_name",
                    "batch_id",
                    "reference_camera",
                    "reference_frame_index",
                    "reference_timestamp_us",
                    "camera_id",
                    "matched_frame_index",
                    "matched_timestamp_us",
                    "time_diff_us",
                    "abs_time_diff_us",
                    "matched_ok",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "run_id",
                "experiment_name",
                "batch_id",
                "reference_camera",
                "reference_frame_index",
                "reference_timestamp_us",
                "camera_id",
                "matched_frame_index",
                "matched_timestamp_us",
                "time_diff_us",
                "abs_time_diff_us",
                "matched_ok",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def build_wide_pairs(reference: CameraFrames, long_pairs: pd.DataFrame) -> pd.DataFrame:
    wide = reference.df.rename(
        columns={
            "frame_index": "reference_frame_index",
            "timestamp_us": "reference_timestamp_us",
        }
    ).copy()
    wide.insert(0, "reference_camera", reference.camera_id)
    for camera_id, group in long_pairs.groupby("camera_id", sort=True):
        subset = group[
            [
                "reference_frame_index",
                "matched_frame_index",
                "matched_timestamp_us",
                "time_diff_us",
                "abs_time_diff_us",
                "matched_ok",
            ]
        ].rename(
            columns={
                "matched_frame_index": f"{camera_id}_matched_frame_index",
                "matched_timestamp_us": f"{camera_id}_matched_timestamp_us",
                "time_diff_us": f"{camera_id}_time_diff_us",
                "abs_time_diff_us": f"{camera_id}_abs_time_diff_us",
                "matched_ok": f"{camera_id}_matched_ok",
            }
        )
        wide = wide.merge(subset, on="reference_frame_index", how="left")
    return wide
