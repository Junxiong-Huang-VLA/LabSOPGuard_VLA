from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import backend.main as backend_main


def _setup_project_root(tmp_path: Path) -> None:
    backend_main.PROJECT_ROOT = tmp_path
    backend_main._EXPERIMENTS.clear()
    (tmp_path / "outputs" / "experiments").mkdir(parents=True, exist_ok=True)


def test_material_diagnostics_counts_asset_gaps_and_warnings(tmp_path: Path):
    _setup_project_root(tmp_path)
    client = TestClient(backend_main.app)
    experiment_id = "diag_exp"
    exp_dir = backend_main._experiment_output_dir(experiment_id)
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": experiment_id}), encoding="utf-8")

    published_dir = exp_dir / "published_materials" / "operator" / "transfer" / "event_a"
    published_dir.mkdir(parents=True)
    playable_clip = published_dir / "clip.mp4"
    preview = published_dir / "preview.jpg"
    keyframe = published_dir / "keyframe_01.jpg"
    playable_clip.write_bytes(b"video")
    preview.write_bytes(b"image")
    keyframe.write_bytes(b"image")
    old_root_clip = f"D:/OldRoot/LabSOPGuard/outputs/experiments/{experiment_id}/published_materials/operator/transfer/event_a/clip.mp4"

    payload = {
        "schema_version": "published_materials.v1",
        "experiment_id": experiment_id,
        "total": 3,
        "items": [
            {
                "event_id": "event_a",
                "time_start": 0,
                "time_end": 10,
                "published_paths": {
                    "clip": old_root_clip,
                    "preview": str(preview),
                    "keyframes": [str(keyframe)],
                },
                "warnings": ["direction_not_confirmed"],
            },
            {
                "event_id": "event_b",
                "time_start": 10,
                "time_end": 45,
                "published_paths": {
                    "clip": str(exp_dir / "published_materials" / "missing.mp4"),
                    "preview": str(exp_dir / "published_materials" / "missing.jpg"),
                    "keyframes": [str(exp_dir / "published_materials" / "missing_keyframe.jpg")],
                },
                "warnings": ["missing_clip", "missing_preview"],
            },
            {
                "event_id": "event_c",
                "duration_sec": 5,
                "published_paths": {
                    "clip": str(tmp_path.parent / "escaped.mp4"),
                    "preview": str(preview),
                    "keyframes": [],
                },
                "warnings": ["missing_clip"],
            },
        ],
    }
    (exp_dir / "published_materials.json").write_text(json.dumps(payload), encoding="utf-8")

    response = client.get(f"/api/v1/experiments/{experiment_id}/materials/diagnostics")

    assert response.status_code == 200
    diagnostics = response.json()
    assert diagnostics["experiment_id"] == experiment_id
    assert diagnostics["published_total"] == 3
    assert diagnostics["clip_count"] == 3
    assert diagnostics["missing_clip_count"] == 2
    assert diagnostics["missing_preview_count"] == 1
    assert diagnostics["missing_keyframe_count"] == 2
    assert diagnostics["playable_clip_count"] == 1
    assert diagnostics["long_clip_count"] == 1
    assert diagnostics["long_clip_threshold_sec"] == 30.0
    assert diagnostics["avg_duration_sec"] == 16.667
    assert diagnostics["warnings_count"] == 4
    assert diagnostics["warnings_by_type"] == {
        "direction_not_confirmed": 1,
        "missing_clip": 2,
        "missing_preview": 1,
    }
    assert len(diagnostics["broken_clip_paths"]) == 2
