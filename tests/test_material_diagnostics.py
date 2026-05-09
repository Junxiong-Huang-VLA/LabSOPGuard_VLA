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


def test_material_diagnostics_exposes_formal_evidence_chain(tmp_path: Path):
    _setup_project_root(tmp_path)
    client = TestClient(backend_main.app)
    experiment_id = "formal_diag_exp"
    exp_dir = backend_main._experiment_output_dir(experiment_id)
    delivery_root = tmp_path / "outputs" / "material_references" / "正式诊断测试_20260508"
    frame_dir = delivery_root / "关键帧"
    clip_dir = delivery_root / "关键片段"
    frame_dir.mkdir(parents=True, exist_ok=True)
    clip_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "title": "正式诊断测试",
                "created_at": "2026-05-08T09:30:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    frame = frame_dir / "手与烧杯操作_20260508.jpg"
    clip = clip_dir / "手与烧杯操作_20260508.mp4"
    frame.write_bytes(b"jpg")
    clip.write_bytes(b"mp4")
    rows = [
        {
            "schema_version": "material_reference.item.v1",
            "asset_kind": "关键帧",
            "material_type": "关键帧",
            "candidate_id": "candidate_frame",
            "candidate_group_id": "candidate_group",
            "action_name": "手与烧杯操作",
            "primary_object": "beaker",
            "stored_file": str(frame),
            "stored_filename": frame.name,
            "source_candidate_file": str(exp_dir / "_material_review_queue" / "关键帧" / frame.name),
            "review_status": "accepted",
            "approved_by": "pytest",
            "approved_at": "2026-05-08T09:40:00+08:00",
            "formal_material_reference": True,
            "yolo_recheck": {"status": "passed", "valid_evidence_count": 3},
            "vlm_semantics": {"status": "aligned", "model": "qwen3.6-plus", "description": "戴手套操作烧杯"},
        },
        {
            "schema_version": "material_reference.item.v1",
            "asset_kind": "关键片段",
            "material_type": "关键片段",
            "candidate_id": "candidate_clip",
            "candidate_group_id": "candidate_group",
            "action_name": "手与烧杯操作",
            "primary_object": "beaker",
            "stored_file": str(clip),
            "stored_filename": clip.name,
            "source_candidate_file": str(exp_dir / "_material_review_queue" / "关键片段" / clip.name),
            "review_status": "accepted",
            "approved_by": "pytest",
            "approved_at": "2026-05-08T09:40:00+08:00",
            "formal_material_reference": True,
            "yolo_recheck": {"status": "passed", "valid_evidence_count": 3},
            "vlm_semantics": {"status": "aligned", "model": "qwen3.6-plus", "description": "戴手套操作烧杯"},
        },
    ]
    (delivery_root / "素材索引.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    response = client.get(f"/api/v1/experiments/{experiment_id}/materials/diagnostics")

    assert response.status_code == 200
    diagnostics = response.json()
    assert diagnostics["published_total"] == 2
    assert diagnostics["formal_material_reference_count"] == 2
    assert diagnostics["url_accessible_count"] == 2
    assert {item["candidate_id"] for item in diagnostics["evidence_items"]} == {"candidate_frame", "candidate_clip"}
    assert {item["yolo_recheck_status"] for item in diagnostics["evidence_items"]} == {"passed"}
    assert {item["vlm_model"] for item in diagnostics["evidence_items"]} == {"qwen3.6-plus"}
    for item in diagnostics["evidence_items"]:
        assert item["approved_by"] == "pytest"
        assert item["material_exists"] is True
        assert item["url_accessible"] is True
        assert item["material_url"].startswith(f"/api/v1/experiments/{experiment_id}/material-references/files/")
        assert client.get(item["material_url"]).status_code == 200
