from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_cli_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_material_library_cli_ready.py"
    spec = importlib.util.spec_from_file_location("check_material_library_cli_ready", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cli_ready_report_accepts_review_candidates_without_memory_eligibility(tmp_path):
    module = _load_cli_module()
    root = tmp_path / "exp001"
    root.mkdir()
    sync = root / "window_sync_index.csv"
    first_frame = root / "first_keyframe.jpg"
    third_frame = root / "third_keyframe.jpg"
    first_clip = root / "first_keyclip.mp4"
    third_clip = root / "third_keyclip.mp4"
    for path in [sync, first_frame, third_frame, first_clip, third_clip]:
        path.write_text("x", encoding="utf-8")

    row = {
        "material_id": "mat001",
        "evidence_bundle_id": "bundle001",
        "official_status": "needs_review",
        "action_type": "hand_object_contact",
        "experiment_window_id": "window001",
        "global_timestamp_us": 123456,
        "source_window_sync_index": str(sync),
        "first_keyframe": str(first_frame),
        "third_keyframe": str(third_frame),
        "first_keyclip": str(first_clip),
        "third_keyclip": str(third_clip),
        "keyframe_quality_score": 0.82,
        "cli_ready_folder": str(root),
        "memory_eligible": False,
    }
    (root / "material_stream.jsonl").write_text(
        __import__("json").dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "review_candidate_materials.jsonl").write_text("", encoding="utf-8")
    (root / "official_materials.jsonl").write_text("", encoding="utf-8")

    report = module.build_cli_ready_report(root)

    assert report["status"] == "ready"
    assert report["material_stream"]["row_count"] == 1
    assert report["checks"]["memory_policy_violations"] == []


def test_cli_ready_report_rejects_needs_review_memory_eligible(tmp_path):
    module = _load_cli_module()
    root = tmp_path / "exp001"
    root.mkdir()
    row = {
        "material_id": "mat001",
        "evidence_bundle_id": "bundle001",
        "official_status": "needs_review",
        "action_type": "hand_object_contact",
        "experiment_window_id": "window001",
        "global_timestamp_us": 123456,
        "source_window_sync_index": str(root / "missing.csv"),
        "first_keyframe": str(root / "missing.jpg"),
        "third_keyframe": str(root / "missing.jpg"),
        "first_keyclip": str(root / "missing.mp4"),
        "third_keyclip": str(root / "missing.mp4"),
        "keyframe_quality_score": 0.82,
        "cli_ready_folder": str(root),
        "memory_eligible": True,
    }
    (root / "material_stream.jsonl").write_text(
        __import__("json").dumps(row, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report = module.build_cli_ready_report(root)

    assert report["status"] == "needs_attention"
    assert report["checks"]["memory_policy_violations"]
