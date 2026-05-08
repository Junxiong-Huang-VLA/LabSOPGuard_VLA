from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.schemas import FrameScore, write_jsonl
from key_action_indexer.tuning import tune_detector


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _make_manifest(tmp_path: Path) -> Path:
    session_dir = tmp_path / "session"
    manifest = {
        "session_id": "tune_test",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {
                "path": str(session_dir / "raw" / "third_person.mp4"),
                "start_time": "2026-04-29T17:25:00+08:00",
                "fps": 30,
                "offset_sec": 0,
            }
        },
        "detection_config": {
            "sample_fps": 1,
            "start_threshold": 0.6,
            "end_threshold": 0.3,
            "start_min_duration_sec": 2.0,
            "end_min_duration_sec": 5.0,
            "merge_gap_sec": 1.0,
            "min_segment_duration_sec": 5.0,
            "buffer_sec": 2.0,
        },
        "output_dir": str(session_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def _write_frame_scores(session_dir: Path) -> None:
    scores = []
    for t in range(0, 21):
        active = 0.65 if 5 <= t <= 10 else 0.05
        scores.append(FrameScore(time_sec=float(t), motion_score=active, active_score=active))
    write_jsonl(session_dir / "cv_outputs" / "frame_scores.jsonl", scores)


def test_tune_detector_selects_best_config(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    session_dir = tmp_path / "session"
    _write_frame_scores(session_dir)
    gt_path = tmp_path / "manual.jsonl"
    _write_jsonl(gt_path, [{"segment_id": "gt_001", "start_sec": 3.0, "end_sec": 13.0, "label": "action"}])

    result = tune_detector(
        manifest_path,
        gt_path,
        start_thresholds=[0.6, 0.7],
        end_thresholds=[0.3],
        merge_gap_secs=[1.0],
        min_segment_duration_secs=[5.0],
    )

    assert result["best_config"]["start_threshold"] == 0.6
    assert result["best_config"]["f1"] == 1.0
    assert result["best_config"]["segment_count"] == 1
    assert (session_dir / "evaluation" / "tuning_results.json").exists()


def test_tune_cli_writes_basic_output(tmp_path: Path) -> None:
    manifest_path = _make_manifest(tmp_path)
    session_dir = tmp_path / "session"
    _write_frame_scores(session_dir)
    gt_path = tmp_path / "manual.jsonl"
    _write_jsonl(gt_path, [{"segment_id": "gt_001", "start_sec": 3.0, "end_sec": 13.0, "label": "action"}])

    exit_code = main(
        [
            "tune",
            "--manifest",
            str(manifest_path),
            "--ground-truth",
            str(gt_path),
            "--start-threshold",
            "0.6,0.7",
        ]
    )

    output_path = session_dir / "evaluation" / "tuning_results.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["best_config"]["start_threshold"] == 0.6
    assert len(payload["results"]) == 2
    assert {"precision", "recall", "f1", "mean_iou", "segment_count", "total_duration"} <= set(payload["results"][0])
