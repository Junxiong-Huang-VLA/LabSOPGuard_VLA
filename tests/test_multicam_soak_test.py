from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.soak_test import load_soak_test_config, make_dry_run_config, run_soak_test


def test_load_soak_test_config_filters_disabled_cameras(tmp_path: Path):
    config_path = tmp_path / "soak.yaml"
    config_path.write_text(
        """
run:
  experiment_id: test_soak
  duration_sec: 1
defaults:
  expected_fps: 5
cameras:
  - camera_id: disabled_cam
    enabled: false
    source_type: rtsp
    source: rtsp://example.invalid/stream
  - camera_id: enabled_cam
    enabled: true
    source_type: synthetic
    source: synthetic://enabled
""",
        encoding="utf-8",
    )

    config = load_soak_test_config(config_path)

    assert config.experiment_id == "test_soak"
    assert len(config.cameras) == 1
    assert config.cameras[0].camera_id == "enabled_cam"
    assert config.cameras[0].expected_fps == 5.0


def test_multicam_soak_dry_run_writes_report_and_material_stream(tmp_path: Path):
    config = make_dry_run_config(duration_sec=2.2, output_root=str(tmp_path / "soak_tests"))
    report = run_soak_test(config)

    report_json = Path(report["artifact_paths"]["report_json"])
    report_md = Path(report["artifact_paths"]["report_md"])
    stream_health_csv = Path(report["artifact_paths"]["stream_health_csv"])
    material_stream_path = Path(report["artifact_paths"]["material_stream"])
    video_inputs_path = Path(report["artifact_paths"]["video_inputs"])

    assert report["status"] == "passed"
    assert report_json.exists()
    assert report_md.exists()
    assert stream_health_csv.exists()
    assert material_stream_path.exists()
    assert video_inputs_path.exists()

    material_stream = json.loads(material_stream_path.read_text(encoding="utf-8"))
    assert {item["camera_id"] for item in material_stream} == {"dry_front", "dry_side"}
    assert all(item["file_exists"] for item in material_stream)
    assert all(Path(item["recorded_file_path"]).exists() for item in material_stream)

    saved_report = json.loads(report_json.read_text(encoding="utf-8"))
    assert saved_report["schema_version"] == "multicam_soak_test.v1"
    assert len(saved_report["camera_results"]) == 2
