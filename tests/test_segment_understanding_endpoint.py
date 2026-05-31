"""Contract test for GET /experiments/{id}/segment-understanding.

Verifies the endpoint maps per-window understanding text, prefers real Qwen
vlm_semantics over template rows, and degrades gracefully when no data exists.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import main as backend_main  # noqa: E402


def _setup_isolated_root(tmp_path: Path):
    backend_main.PROJECT_ROOT = tmp_path
    backend_main._EXPERIMENTS.clear()
    (tmp_path / "outputs" / "experiments").mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_segment_understanding_prefers_qwen_then_template(tmp_path, monkeypatch):
    _setup_isolated_root(tmp_path)

    experiment_id = "exp-understanding"
    output_dir = tmp_path / "outputs" / "experiments" / experiment_id
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    backend_main._EXPERIMENTS[experiment_id] = {
        "experiment_id": experiment_id,
        "status": "completed",
    }

    # Two windows. w1 has a real Qwen material; w2 only has a template row.
    monkeypatch.setattr(
        backend_main,
        "_candidate_experiment_window_segments",
        lambda eid, out: [
            {
                "segment_id": "formal_window_001",
                "index": 0,
                "display_name": "实验片段一",
                "start_sec": 0.0,
                "end_sec": 100.0,
                "raw_window_start_global_timestamp_us": 0,
                "raw_window_end_global_timestamp_us": 100_000_000,
            },
            {
                "segment_id": "formal_window_002",
                "index": 1,
                "display_name": "实验片段二",
                "start_sec": 100.0,
                "end_sec": 200.0,
                "raw_window_start_global_timestamp_us": 100_000_000,
                "raw_window_end_global_timestamp_us": 200_000_000,
            },
        ],
    )

    _write_jsonl(
        metadata_dir / "key_material_references.jsonl",
        [
            {
                "window_id": "formal_window_001",
                "vlm_semantics": {
                    "description": "操作者将称量纸放在天平上并加入试剂。",
                    "physical_action": "weighing",
                    "confirmed_objects": ["balance", "weighing_paper"],
                },
            }
        ],
    )
    _write_jsonl(
        metadata_dir / "video_understanding.jsonl",
        [
            {
                "global_start_time": "2026-05-22T00:00:02+00:00",
                "global_end_time": "2026-05-22T00:00:50+00:00",
                "text": "模板：天平称量片段。",
            },
            {
                "global_start_time": "2026-05-22T00:02:05+00:00",
                "global_end_time": "2026-05-22T00:02:55+00:00",
                "text": "模板：移液片段。",
            },
        ],
    )
    # Make the template timestamps line up with the window epoch ranges above.
    monkeypatch.setattr(backend_main, "_iso_to_epoch_us", lambda v: {
        "2026-05-22T00:00:02+00:00": 2_000_000,
        "2026-05-22T00:00:50+00:00": 50_000_000,
        "2026-05-22T00:02:05+00:00": 125_000_000,
        "2026-05-22T00:02:55+00:00": 175_000_000,
    }.get(str(v)))

    client = TestClient(backend_main.app)
    resp = client.get(f"/api/v1/experiments/{experiment_id}/segment-understanding")
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["total"] == 2
    segments = {s["segment_id"]: s for s in payload["segments"]}

    w1 = segments["formal_window_001"]
    assert w1["source"] == "qwen"
    assert "称量纸" in w1["understanding_text"]
    assert "weighing" in w1["understanding_text"]

    w2 = segments["formal_window_002"]
    assert w2["source"] == "template"
    assert "移液" in w2["understanding_text"]


def test_segment_understanding_empty_when_no_artifacts(tmp_path, monkeypatch):
    _setup_isolated_root(tmp_path)
    experiment_id = "exp-empty-understanding"
    (tmp_path / "outputs" / "experiments" / experiment_id / "metadata").mkdir(parents=True, exist_ok=True)
    backend_main._EXPERIMENTS[experiment_id] = {"experiment_id": experiment_id, "status": "completed"}
    monkeypatch.setattr(backend_main, "_candidate_experiment_window_segments", lambda eid, out: [])

    client = TestClient(backend_main.app)
    resp = client.get(f"/api/v1/experiments/{experiment_id}/segment-understanding")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
