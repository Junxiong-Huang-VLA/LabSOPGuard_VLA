from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

import backend.main as backend_main


def _setup_project_root(tmp_path: Path) -> None:
    backend_main.PROJECT_ROOT = tmp_path
    backend_main._EXPERIMENTS.clear()
    (tmp_path / "outputs" / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads" / "experiments").mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_key_action_results_prefers_real_index_metadata_over_root_manifest(tmp_path: Path) -> None:
    _setup_project_root(tmp_path)
    client = TestClient(backend_main.app)
    experiment_id = "exp_key_actions_001"
    exp_dir = tmp_path / "outputs" / "experiments" / experiment_id
    key_action_dir = exp_dir / "key_action_index"
    metadata_dir = key_action_dir / "metadata"
    cv_dir = key_action_dir / "cv_outputs"
    exp_dir.mkdir(parents=True)
    (key_action_dir / "clips" / "seg_000001").mkdir(parents=True)
    (key_action_dir / "keyframes" / "micro" / "seg_000001_micro_001").mkdir(parents=True)

    (exp_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "title": "Recovered key actions",
                "status": "analyzed",
                "processing_stage": "output_generation",
                "created_at": "2026-05-07T12:00:00+08:00",
            }
        ),
        encoding="utf-8",
    )
    (exp_dir / "manifest.json").write_text(
        json.dumps({"actions": [{"id": f"legacy_{idx}", "start_sec": idx, "end_sec": idx + 1} for idx in range(4)]}),
        encoding="utf-8",
    )
    (key_action_dir / "job_status.json").write_text(
        json.dumps({"status": "completed", "summary": {"segment_count": 1, "micro_segment_count": 2}}),
        encoding="utf-8",
    )
    (key_action_dir / "pipeline_summary.json").write_text(
        json.dumps({"detector_summary": {"frame_rows": 2, "interaction_count": 3}}),
        encoding="utf-8",
    )

    clip_path = key_action_dir / "clips" / "seg_000001" / "third_person.mp4"
    peak_frame = key_action_dir / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    segment = {
        "session_id": experiment_id,
        "segment_id": "seg_000001",
        "duration_sec": 4.0,
        "third_person": {"clip_path": str(clip_path), "local_start_sec": 1.0, "local_end_sec": 5.0},
        "first_person": None,
        "interaction_events": [{"event_id": "picked_keyframe", "keyframe_path": str(peak_frame)}],
        "interaction_keyframes": [{"event_id": "picked_keyframe", "path": str(peak_frame)}],
    }
    micros = [
        {
            "micro_segment_id": "seg_000001_micro_001",
            "parent_segment_id": "seg_000001",
            "third_person": {"clip_path": str(key_action_dir / "clips" / "micro_001.mp4")},
            "keyframes": {"peak_frame": str(peak_frame)},
        },
        {
            "micro_segment_id": "seg_000001_micro_002",
            "parent_segment_id": "seg_000001",
            "third_person": {"clip_path": str(key_action_dir / "clips" / "micro_002.mp4")},
            "keyframes": {"peak_frame": str(peak_frame)},
        },
    ]
    yolo_rows = [
        {
            "source_view": "third_person",
            "frame_index": 10,
            "local_time_sec": 2.0,
            "hand_object_interactions": [
                {"hand_label": "gloved_hand", "object_label": "spatula", "score": 0.9},
                {"hand_label": "gloved_hand", "object_label": "balance", "score": 0.7},
            ],
        },
        {
            "source_view": "third_person",
            "frame_index": 20,
            "local_time_sec": 4.0,
            "hand_object_interactions": [
                {"hand_label": "gloved_hand", "object_label": "paper", "score": 0.8},
            ],
        },
    ]
    _write_jsonl(metadata_dir / "key_action_segments.jsonl", [segment])
    _write_jsonl(metadata_dir / "micro_segments.jsonl", micros)
    _write_jsonl(cv_dir / "yolo_frame_rows.jsonl", yolo_rows)

    response = client.get(f"/api/v1/experiments/{experiment_id}/key-actions/results")
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["source"] == "key_action_index_metadata"
    assert payload["summary"]["segment_count"] == 1
    assert payload["summary"]["micro_segment_count"] == 2
    assert payload["summary"]["interaction_event_count"] == 3
    assert len(payload["segments"]) == 1
    assert len(payload["micro_segments"]) == 2
    assert len(payload["interaction_events"]) == 3
    assert payload["segments"][0]["third_person"]["clip_url"].startswith(f"/api/v1/experiments/{experiment_id}/files/")
    assert payload["micro_segments"][0]["keyframes"]["peak_frame_url"].startswith(
        f"/api/v1/experiments/{experiment_id}/files/"
    )
