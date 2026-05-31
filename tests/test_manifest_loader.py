from __future__ import annotations

import pandas as pd

from realityloop_sync.config import SyncConfig, TimestampConfig
from realityloop_sync.manifest import load_manifest_rows, resolve_streams
from realityloop_sync.sync import run_sync


def _write_frames(run_dir, timestamps):
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame_index": range(len(timestamps)), "packet_system_timestamp_us": timestamps}).to_csv(
        run_dir / "frames.csv",
        index=False,
    )


def test_manifest_csv_loads_experiment_name_batch_and_streams(tmp_path):
    cam01 = tmp_path / "cam01"
    cam02 = tmp_path / "cam02"
    _write_frames(cam01, [1000, 2000])
    _write_frames(cam02, [1005, 1995])
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "run_id": "run_manifest",
                "experiment_name": "固体称量实验",
                "batch_id": "batch_001",
                "camera_id": "cam01",
                "stream_path": str(cam01),
            },
            {
                "run_id": "run_manifest",
                "experiment_name": "固体称量实验",
                "batch_id": "batch_001",
                "camera_id": "cam02",
                "stream_path": str(cam02),
            },
        ]
    ).to_csv(manifest, index=False)

    config = SyncConfig(
        run_id="run_manifest",
        reference_camera="cam01",
        output_dir=tmp_path / "out",
        timestamp=TimestampConfig(("packet_system_timestamp_us",)),
        manifest_csv=manifest,
    )

    rows = load_manifest_rows(manifest)
    effective, streams, warnings = resolve_streams(config)
    result = run_sync(config)

    assert len(rows) == 2
    assert [stream.camera_id for stream in streams] == ["cam01", "cam02"]
    assert warnings == []
    assert effective.experiment_name == "固体称量实验"
    assert effective.batch_id == "batch_001"
    assert result.report.payload["experiment_name"] == "固体称量实验"
    assert result.report.payload["batch_id"] == "batch_001"

