from __future__ import annotations

import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_SRC = PROJECT_ROOT.parent / "src"
if ROOT_SRC.exists() and str(ROOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROOT_SRC))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.material_best_score import (
    MATERIAL_BEST_REASON_SCHEMA_VERSION,
    MATERIAL_BEST_SCORE_SCHEMA_VERSION,
    enrich_material_best_score,
)
from labsopguard.material_maintenance import rebuild_workspace_published_materials_index
from labsopguard.material_retrieval_eval import DEFAULT_MATERIAL_RETRIEVAL_QUERIES, evaluate_material_retrieval_quality
from labsopguard.material_taxonomy import MATERIAL_TAXONOMY_SCHEMA_VERSION, STANDARD_ACTION_TAXONOMY, enrich_material_taxonomy
try:
    from key_action_indexer.material_references import (
        MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION as _KEY_ACTION_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
        MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION as _KEY_ACTION_REFERENCE_TRACE_SCHEMA_VERSION,
    )
except Exception:
    _KEY_ACTION_CANDIDATE_DISPOSITION_SCHEMA_VERSION = "material_candidate_disposition.v1"
    _KEY_ACTION_REFERENCE_TRACE_SCHEMA_VERSION = "material_reference_trace.v1"


MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION = _KEY_ACTION_CANDIDATE_DISPOSITION_SCHEMA_VERSION
MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION = _KEY_ACTION_REFERENCE_TRACE_SCHEMA_VERSION


EXPECTED_ACTIONS = {"hand-paper", "hand-bottle", "hand-spatula", "hand-container", "hand-balance"}


def test_material_taxonomy_and_best_score_schema_fields() -> None:
    taxonomy = enrich_material_taxonomy({"primary_object": "spatula"})
    assert set(STANDARD_ACTION_TAXONOMY) == EXPECTED_ACTIONS
    assert taxonomy["taxonomy_schema_version"] == MATERIAL_TAXONOMY_SCHEMA_VERSION
    assert taxonomy["canonical_action_type"] == "hand-spatula"
    assert taxonomy["canonical_action_label"] == "Hand-spatula"

    scored = enrich_material_best_score({**taxonomy, "quality_score": 0.9, "yolo_evidence_count": 3})
    assert scored["best_score_schema_version"] == MATERIAL_BEST_SCORE_SCHEMA_VERSION
    assert scored["best_reason_schema_version"] == MATERIAL_BEST_REASON_SCHEMA_VERSION
    assert scored["best_score"] > 0
    assert "hand-spatula" in scored["best_reason"]


def test_root_material_reference_contract_when_available() -> None:
    material_references = pytest.importorskip("key_action_indexer.material_references")
    signature = inspect.signature(material_references.approve_material_candidates)
    assert "reason_code" in signature.parameters
    assert "reason" in signature.parameters
    assert material_references.MATERIAL_TAXONOMY_SCHEMA_VERSION == MATERIAL_TAXONOMY_SCHEMA_VERSION
    assert material_references.MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION == "material_candidate_disposition.v1"
    assert material_references.MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION == "material_reference_trace.v1"


def test_material_retrieval_quality_eval_reports_canonical_hits(tmp_path: Path) -> None:
    experiments_root = tmp_path / "experiments"
    exp_dir = experiments_root / "eval_exp"
    exp_dir.mkdir(parents=True)
    items = [
        ("balance", "天平称量", "hand-balance", "balance"),
        ("paper", "称量纸", "hand-paper", "paper"),
        ("spatula", "药匙取样", "hand-spatula", "spatula"),
        ("bottle", "试剂瓶倾倒", "hand-bottle", "bottle"),
        ("container", "烧杯承接", "hand-container", "container"),
    ]
    payload = {
        "schema_version": "published_materials.v1",
        "items": [
            {
                "material_id": f"mat_{name}",
                "event_id": f"evt_{name}",
                "experiment_id": "eval_exp",
                "display_name": display,
                "event_type": display,
                "canonical_action_type": action,
                "canonical_object": obj,
                "sop_phase": f"{obj}-phase",
                "review_status": "accepted",
                "evidence_grade": "strong",
                "published_paths": {},
            }
            for name, display, action, obj in items
        ],
        "total": len(items),
    }
    (exp_dir / "published_materials.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    index_path = tmp_path / "published.sqlite"

    rebuild_workspace_published_materials_index(experiments_root, index_path)
    report = evaluate_material_retrieval_quality(index_path, top_k=3)

    assert report["schema_version"] == "material_retrieval_quality_eval.v2"
    assert report["query_count"] == len(DEFAULT_MATERIAL_RETRIEVAL_QUERIES)
    assert report["canonical_hit_rate"] >= 0.9
    assert report["top_k_hit_rate"] >= 0.9
    assert report["top1_hit_rate"] >= 0.75
    assert report["canonical_action_hit_rate"] >= 0.9
    assert report["canonical_object_hit_rate"] >= 0.9
    assert report["top_k_hit_count"] == report["canonical_hit_count"]
    assert report["top3_hit_count"] <= report["query_count"]
    assert {row["expected_canonical_action_type"] for row in report["queries"]} == {
        "hand-balance",
        "hand-paper",
        "hand-spatula",
        "hand-bottle",
        "hand-container",
    }


def test_professional_reports_have_independent_endpoint_and_stay_out_of_grid() -> None:
    import backend.main as main

    exp_id = "_pytest_professional_report_entry"
    exp_dir = main.PROJECT_ROOT / "outputs" / "experiments" / exp_id
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    main._EXPERIMENTS.pop(exp_id, None)
    try:
        ref_root = exp_dir / "material_references"
        keyframe_dir = ref_root / "关键帧"
        report_dir = ref_root / "专业报告"
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        frame = keyframe_dir / "hand_paper.jpg"
        report = report_dir / "professional_report_qwen36max.pdf"
        frame.write_bytes(b"jpg")
        report.write_bytes(b"%PDF")
        (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": exp_id, "title": "report entry"}), encoding="utf-8")
        rows = [
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": "关键帧",
                "material_type": "关键帧",
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "primary_object": "paper",
                "review_status": "accepted",
                "approved_at": "2026-05-09T00:00:00+08:00",
                "formal_material_reference": True,
            },
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": "专业报告",
                "material_type": "专业报告",
                "role": "professional_report_pdf",
                "stored_file": str(report),
                "stored_filename": report.name,
                "review_status": "accepted",
            },
        ]
        (ref_root / "素材索引.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

        client = TestClient(main.app)
        published = client.get(f"/api/v1/experiments/{exp_id}/materials/published")
        assert published.status_code == 200
        assert published.json()["total"] == 1
        assert all(item["asset_kind"] != "专业报告" for item in published.json()["items"])

        reports = client.get(f"/api/v1/experiments/{exp_id}/materials/professional-reports")
        assert reports.status_code == 200
        report_payload = reports.json()
        assert report_payload["total"] >= 1
        assert any(item["file_name"] == report.name for item in report_payload["items"])
        assert {item["grid_policy"] for item in report_payload["items"]} == {"professional_report_only"}
    finally:
        main._EXPERIMENTS.pop(exp_id, None)
        if exp_dir.exists():
            shutil.rmtree(exp_dir)


def test_formal_material_records_keep_complete_evidence_chain_metadata(tmp_path: Path) -> None:
    import backend.main as main

    exp_id = "_pytest_formal_material_chain_59"
    original_root = main.PROJECT_ROOT
    original_experiments = dict(main._EXPERIMENTS)

    try:
        main.PROJECT_ROOT = tmp_path
        main._EXPERIMENTS.clear()

        exp_dir = tmp_path / "outputs" / "experiments" / exp_id
        published_root = exp_dir
        source_file = exp_dir / "materials" / "events" / "evt_formal_paper" / "preview.jpg"
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_bytes(b"jpg")

        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": exp_id, "title": "Formal evidence chain", "created_at": "2026-05-09T10:00:00+08:00"}),
            encoding="utf-8",
        )
        (exp_dir / "published_materials.json").write_text(
            json.dumps(
                {
                    "schema_version": "published_materials.v1",
                    "experiment_id": exp_id,
                    "total": 1,
                    "items": [
                        {
                            "schema_version": "material_reference.item.v1",
                            "asset_kind": "frame",
                            "material_type": "frame",
                            "event_type": "hand-paper",
                            "canonical_action_type": "hand-paper",
                            "canonical_object": "paper",
                            "sop_phase": "weighing-paper-prep",
                            "display_name": "Formal paper frame",
                            "stable_name": "formal_paper_frame",
                            "review_status": "accepted",
                            "event_id": "evt_formal_paper",
                            "time_start": 12.0,
                            "time_end": 15.0,
                            "quality_score": 0.91,
                            "yolo_evidence_count": 3,
                            "camera_view": "third_person",
                            "source_clip": "third_person.mp4",
                            "source_file": "third_person.mp4",
                            "source_clip_path": str(source_file),
                            "preview_path": str(source_file),
                            "source_container": {"class_name": "paper"},
                            "published_paths": {"preview": str(source_file)},
                            "candidate_disposition_schema_version": MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
                            "candidate_status": "approved",
                            "evidence_chain": {
                                "schema_version": MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
                                "source_clip": "third_person.mp4",
                                "camera_view": "third_person",
                                "time_start": 12.0,
                                "time_end": 15.0,
                                "yolo_evidence_count": 3,
                                "canonical_action_type": "hand-paper",
                                "candidate_disposition": "approved",
                            },
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        rebuild_workspace_published_materials_index(
            tmp_path / "outputs" / "experiments",
            main._workspace_published_materials_index_path(),
        )
        client = TestClient(main.app)
        response = client.get("/api/v1/materials/published?limit=10", headers={"X-Operator-Role": "admin"})
        assert response.status_code == 200
        payload = response.json()
        item = next((row for row in payload["items"] if row.get("event_id") == "evt_formal_paper"), None)
        assert item is not None
        item_payload = item.get("payload")
        assert isinstance(item_payload, dict)
        chain = item_payload.get("evidence_chain")
        assert chain["schema_version"] == MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION
        assert chain["source_clip"] == "third_person.mp4"
        assert chain["camera_view"] == "third_person"
        assert chain["time_start"] == 12.0
        assert chain["time_end"] == 15.0
        assert chain["yolo_evidence_count"] == 3
        assert chain["candidate_disposition"] == "approved"
        assert item_payload["candidate_disposition_schema_version"] == MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION
        assert item_payload["best_score_schema_version"] == MATERIAL_BEST_SCORE_SCHEMA_VERSION
        assert item_payload["best_reason_schema_version"] == MATERIAL_BEST_REASON_SCHEMA_VERSION
    finally:
        main._EXPERIMENTS.clear()
        main._EXPERIMENTS.update(original_experiments)
        main.PROJECT_ROOT = original_root
