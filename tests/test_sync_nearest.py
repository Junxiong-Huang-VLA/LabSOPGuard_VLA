from __future__ import annotations

import pandas as pd
import pytest

from realityloop_sync.config import QualityGateConfig, SyncConfig, StreamConfig, TimestampConfig
from realityloop_sync.frames import FrameDataError
from realityloop_sync.sync import run_sync


def _write_frames(run_dir, timestamps, *, col="packet_system_timestamp_us", frame_order=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_order = list(range(len(timestamps))) if frame_order is None else frame_order
    pd.DataFrame({"frame_index": frame_order, col: timestamps}).to_csv(run_dir / "frames.csv", index=False)


def _config(tmp_path, *, tolerance_us=None, timestamp_cols=("packet_system_timestamp_us",)):
    return SyncConfig(
        run_id="run_a",
        experiment_name="solid weighing",
        batch_id="batch_a",
        reference_camera="cam01",
        output_dir=tmp_path / "out",
        timestamp=TimestampConfig(tuple(timestamp_cols)),
        tolerance_us=tolerance_us,
        streams=(
            StreamConfig("cam01", tmp_path / "cam01"),
            StreamConfig("cam02", tmp_path / "cam02"),
        ),
    )


def test_two_camera_exact_timestamp_match(tmp_path):
    _write_frames(tmp_path / "cam01", [1000, 2000, 3000])
    _write_frames(tmp_path / "cam02", [1000, 2000, 3000])

    result = run_sync(_config(tmp_path))

    assert result.long_pairs["matched_ok"].tolist() == [True, True, True]
    assert result.long_pairs["time_diff_us"].tolist() == [0, 0, 0]
    assert result.report.payload["match_rate"]["cam02"] == 1.0
    assert result.report.payload["quality_gate_status"] == "pass"
    assert result.report.payload["quality_gate"]["passed"] is True
    assert (tmp_path / "out" / "sync_pairs_long.csv").exists()
    assert (tmp_path / "out" / "sync_pairs_wide.csv").exists()
    assert (tmp_path / "out" / "sync_report.json").exists()
    assert (tmp_path / "out" / "run_summary.md").exists()


def test_nearest_neighbor_with_small_offset(tmp_path):
    _write_frames(tmp_path / "cam01", [1000, 2000, 3000])
    _write_frames(tmp_path / "cam02", [960, 2040, 3090])

    result = run_sync(_config(tmp_path))

    assert result.long_pairs["matched_frame_index"].tolist() == [0, 1, 2]
    assert result.long_pairs["time_diff_us"].tolist() == [-40, 40, 90]
    assert result.report.payload["max_abs_time_diff_us"]["cam02"] == 90.0
    assert result.report.payload["output_files"]["sync_pairs_long"]["row_count"] == 3


def test_tolerance_marks_over_threshold_as_unmatched(tmp_path):
    _write_frames(tmp_path / "cam01", [1000, 2000, 3000])
    _write_frames(tmp_path / "cam02", [1000, 2080, 3300])

    result = run_sync(_config(tmp_path, tolerance_us=100))

    assert result.long_pairs["matched_ok"].tolist() == [True, True, False]
    assert result.report.payload["matched_count"]["cam02"] == 2
    assert result.report.payload["unmatched_count"]["cam02"] == 1
    assert result.report.payload["quality_gate_status"] == "warning"
    assert result.report.payload["quality_gate"]["issues"][0]["code"] == "unmatched_frames_over_warning_threshold"


def test_quality_gate_warns_on_configured_time_diff_threshold_when_tolerance_is_null(tmp_path):
    _write_frames(tmp_path / "cam01", [1000, 2000, 3000])
    _write_frames(tmp_path / "cam02", [960, 2040, 3090])
    config = SyncConfig(
        run_id="run_quality",
        reference_camera="cam01",
        output_dir=tmp_path / "out",
        quality_gate=QualityGateConfig(max_abs_time_diff_us_warning=80),
        streams=(
            StreamConfig("cam01", tmp_path / "cam01"),
            StreamConfig("cam02", tmp_path / "cam02"),
        ),
    )

    result = run_sync(config)

    assert result.report.payload["tolerance_us"] is None
    assert result.report.payload["quality_gate_status"] == "warning"
    assert result.report.payload["quality_gate"]["thresholds"]["max_abs_time_diff_us_warning"] == 80
    assert result.report.payload["quality_gate"]["issues"][0]["code"] == "max_abs_time_diff_us_over_warning_threshold"


def test_unsorted_frames_are_sorted_before_sync(tmp_path):
    _write_frames(tmp_path / "cam01", [3000, 1000, 2000], frame_order=[30, 10, 20])
    _write_frames(tmp_path / "cam02", [1000, 2000, 3000])

    result = run_sync(_config(tmp_path))

    assert result.long_pairs["reference_frame_index"].tolist() == [10, 20, 30]
    assert result.long_pairs["time_diff_us"].tolist() == [0, 0, 0]
    assert any("not sorted" in warning for warning in result.report.payload["warnings"])


def test_stream_filters_remove_mixed_depth_rows_before_sync(tmp_path):
    for camera_id in ("cam01", "cam02"):
        run_dir = tmp_path / camera_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "frame_id": [0, 0, 1, 1],
                "stream_type": ["depth", "rgb", "depth", "rgb"],
                "packet_system_timestamp_us": [1000, 1010, 2000, 2010],
            }
        ).to_csv(run_dir / "frames.csv", index=False)
    config = SyncConfig(
        run_id="run_filtered",
        reference_camera="cam01",
        output_dir=tmp_path / "out",
        streams=(
            StreamConfig("cam01", tmp_path / "cam01", filters={"stream_type": ("rgb",)}),
            StreamConfig("cam02", tmp_path / "cam02", filters={"stream_type": ("rgb",)}),
        ),
    )

    result = run_sync(config)

    assert result.report.payload["frame_count"] == {"cam01": 2, "cam02": 2}
    assert result.long_pairs["reference_timestamp_us"].tolist() == [1010, 2010]
    assert not any("duplicate" in warning for warning in result.report.payload["warnings"])


def test_missing_timestamp_field_has_clear_error(tmp_path):
    _write_frames(tmp_path / "cam01", [1000, 2000], col="other_timestamp_us")
    _write_frames(tmp_path / "cam02", [1000, 2000])

    with pytest.raises(FrameDataError, match="none of timestamp.preferred_cols"):
        run_sync(_config(tmp_path))
