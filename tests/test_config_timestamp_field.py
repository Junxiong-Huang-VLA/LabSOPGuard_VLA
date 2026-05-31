from __future__ import annotations

import pandas as pd

from realityloop_sync.config import SyncConfig, StreamConfig, TimestampConfig
from realityloop_sync.sync import run_sync


def _write_frames(run_dir, timestamps, *, col):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame_index": range(len(timestamps)), col: timestamps}).to_csv(run_dir / "frames.csv", index=False)


def test_timestamp_field_can_switch_to_synced_timestamp_without_code_change(tmp_path):
    _write_frames(tmp_path / "cam01", [1000, 2000, 3000], col="synced_timestamp_us")
    _write_frames(tmp_path / "cam02", [1010, 1990, 3010], col="synced_timestamp_us")

    config = SyncConfig(
        run_id="synced_col",
        reference_camera="cam01",
        output_dir=tmp_path / "out",
        timestamp=TimestampConfig(("synced_timestamp_us", "packet_system_timestamp_us")),
        streams=(
            StreamConfig("cam01", tmp_path / "cam01"),
            StreamConfig("cam02", tmp_path / "cam02"),
        ),
    )

    result = run_sync(config)

    assert result.report.payload["timestamp_col_used"] == {
        "cam01": "synced_timestamp_us",
        "cam02": "synced_timestamp_us",
    }
    assert result.long_pairs["time_diff_us"].tolist() == [10, -10, 10]


def test_config_parses_quality_gate_warning_thresholds(tmp_path):
    config = SyncConfig.from_mapping(
        {
            "run_id": "quality_config",
            "output_dir": str(tmp_path / "out"),
            "quality_gate": {
                "max_abs_time_diff_us_warning": "5000",
                "unmatched_frames_warning": 2,
                "unmatched_rate_warning": "0.1",
            },
            "streams": [
                {"camera_id": "cam01", "stream_path": str(tmp_path / "cam01")},
                {"camera_id": "cam02", "stream_path": str(tmp_path / "cam02")},
            ],
        },
        base_dir=tmp_path,
    )

    assert config.quality_gate.max_abs_time_diff_us_warning == 5000
    assert config.quality_gate.unmatched_count_warning == 2
    assert config.quality_gate.unmatched_rate_warning == 0.1
