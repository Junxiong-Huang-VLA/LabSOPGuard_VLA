import importlib.util
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _load_backend_main():
    path = Path("LabSOPGuard") / "backend" / "main.py"
    spec = importlib.util.spec_from_file_location("lab_backend_main_preview_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_sub_experiment_payload_exposes_preview_urls_and_trace_paths(tmp_path, monkeypatch) -> None:
    backend = _load_backend_main()
    project_root = tmp_path / "LabSOPGuard"
    experiment_id = "exp-preview-contract"
    output_dir = project_root / "outputs" / "experiments" / experiment_id
    metadata_dir = output_dir / "key_action_index" / "metadata"
    window_dir = output_dir / "windows" / "formal_window_001"
    metadata_dir.mkdir(parents=True)
    window_dir.mkdir(parents=True)
    monkeypatch.setattr(backend, "PROJECT_ROOT", project_root)

    for name in (
        "third_view_realtime_preview.mp4",
        "first_view_realtime_preview.mp4",
        "window_preview.browser.mp4",
        "fast_preview.mp4",
        "sample_grid.jpg",
        "window_sync_index.csv",
    ):
        (window_dir / name).write_bytes(b"preview")

    (window_dir / "window_report.json").write_text(
        json.dumps(
            {
                "window_preview_duration_s": 2.1,
                "window_preview_mode": "realtime_preview",
                "window_preview_playback_speed_ratio": 1.0,
                "window_preview_output_fps": 15,
                "actual_experiment_duration_s": 1.7,
                "third_view_realtime_preview": str(window_dir / "third_view_realtime_preview.mp4"),
                "first_view_realtime_preview": str(window_dir / "first_view_realtime_preview.mp4"),
                "side_by_side_realtime_preview": str(window_dir / "window_preview.browser.mp4"),
                "fast_preview": str(window_dir / "fast_preview.mp4"),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (metadata_dir / "formal_experiment_windows.json").write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "window_id": "formal_window_001",
                        "start_sec": 10,
                        "end_sec": 11,
                        "duration_sec": 1,
                        "start_global_timestamp_us": 1_000_000,
                        "end_global_timestamp_us": 3_100_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    segments = backend._candidate_experiment_window_segments(experiment_id, output_dir)

    assert len(segments) == 1
    segment = segments[0]
    assert segment["third_view_realtime_preview_url"].endswith("/files/windows/formal_window_001/third_view_realtime_preview.mp4")
    assert segment["first_view_realtime_preview_url"].endswith("/files/windows/formal_window_001/first_view_realtime_preview.mp4")
    assert segment["side_by_side_realtime_preview_url"].endswith("/files/windows/formal_window_001/window_preview.browser.mp4")
    assert segment["fast_preview_url"].endswith("/files/windows/formal_window_001/fast_preview.mp4")
    assert segment["third_view_realtime_preview"] == segment["third_view_realtime_preview_url"]
    assert segment["first_view_realtime_preview"] == segment["first_view_realtime_preview_url"]
    assert segment["third_view_realtime_preview_path"] == str(window_dir / "third_view_realtime_preview.mp4")
    assert segment["source_window_sync_index"] == str(window_dir / "window_sync_index.csv")
    assert segment["playback_speed_ratio"] == 1.0
    assert segment["experiment_window_duration_s"] == 2.1
