from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as main
import key_action_indexer.material_references as root_material_references
from labsopguard.material_best_score import MATERIAL_BEST_REASON_SCHEMA_VERSION, MATERIAL_BEST_SCORE_SCHEMA_VERSION
from labsopguard.material_taxonomy import MATERIAL_TAXONOMY_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if payload:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_root_reference_payload_contract_is_preserved_for_labsopguard_sync(tmp_path: Path) -> None:
    original_root = main.PROJECT_ROOT
    original_experiments = dict(main._EXPERIMENTS)
    try:
        main.PROJECT_ROOT = tmp_path
        main._EXPERIMENTS.clear()

        experiment_id = "contract-exp"
        exp_dir = tmp_path / "outputs" / "experiments" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": experiment_id, "title": "contract-material-test"}, ensure_ascii=False),
            encoding="utf-8",
        )

        formal_root = exp_dir / "material_references"
        keyframe_dir = formal_root / root_material_references.KEYFRAME_DIR_NAME
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        source_file = keyframe_dir / "third_person.mp4"
        stored_file = keyframe_dir / "paper_frame.jpg"
        source_file.write_bytes(b"mp4")
        stored_file.write_bytes(b"jpg")
        _write_json(formal_root / "manifest.json", {"formal_material_references": str(formal_root)})
        formal_rows = [
            {
                "schema_version": "material_reference.item.v1",
                "material_type": root_material_references.KEYFRAME_DIR_NAME,
                "asset_kind": root_material_references.KEYFRAME_DIR_NAME,
                "action_name": "hand-paper-transfer",
                "primary_object": "paper",
                "micro_segment_id": "micro_001",
                "parent_segment_id": "parent_001",
                "segment_id": "segment_001",
                "view": "third_person",
                "camera_view": "third_person",
                "time_start": 12.6,
                "time_end": 15.9,
                "source_file": str(source_file),
                "source_clip": str(source_file),
                "source_clip_path": str(source_file),
                "stored_file": str(stored_file),
                "stored_filename": stored_file.name,
                "file_name": stored_file.name,
                "review_status": "accepted",
                "candidate_status": "approved",
                "quality_score": 0.88,
                "yolo_evidence_count": 9,
                "candidate_disposition_schema_version": root_material_references.MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION,
                "trace_schema_version": root_material_references.MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
                "evidence_chain": {
                    "schema_version": root_material_references.MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION,
                    "source_clip": str(source_file),
                    "camera_view": "third_person",
                    "time_start": 12.6,
                    "time_end": 15.9,
                    "yolo_evidence_count": 9,
                    "canonical_action_type": "hand-paper",
                    "candidate_disposition": "approved",
                },
            }
        ]
        _write_jsonl(formal_root / f"{root_material_references.MATERIAL_INDEX_BASENAME}.jsonl", formal_rows)

        assert main._formal_material_reference_root_for_exp(exp_dir) == formal_root

        published = main._sync_published_materials_from_references(exp_dir, experiment_id)
        assert published["schema_version"] == "published_materials.approved_material_references.v1"
        assert published["total"] == 1
        sync_item = published["items"][0]
        assert sync_item["candidate_disposition_schema_version"] == root_material_references.MATERIAL_CANDIDATE_DISPOSITION_SCHEMA_VERSION
        assert sync_item["evidence_chain"]["schema_version"] == root_material_references.MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION
        assert sync_item["evidence_chain"]["candidate_disposition"] == "approved"
        assert sync_item["evidence_chain"]["source_clip"] == str(source_file)
        assert sync_item["source_file"] == str(source_file)
        assert sync_item["canonical_object"] == "paper"
        assert sync_item["canonical_action_type"] == "hand-paper"
        assert sync_item["taxonomy_schema_version"] == root_material_references.MATERIAL_TAXONOMY_SCHEMA_VERSION

        payload = main._published_material_items(exp_dir, experiment_id)
        items = payload.get("items")
        if not isinstance(items, list):
            items = []
        all_items = payload.get("all_items")
        if isinstance(all_items, list):
            payload_items = all_items
        else:
            payload_items = items
        assert isinstance(items, list)
        assert len(payload_items) == 1
        item = payload_items[0]
        assert item["schema_version"] == "material_reference.item.v1"
        assert item["best_score_schema_version"] == MATERIAL_BEST_SCORE_SCHEMA_VERSION
        assert item["best_reason_schema_version"] == MATERIAL_BEST_REASON_SCHEMA_VERSION

        assert MATERIAL_TAXONOMY_SCHEMA_VERSION == root_material_references.MATERIAL_TAXONOMY_SCHEMA_VERSION
        assert item["schema_version"] == "material_reference.item.v1"
        assert "evidence_chain" in item
        assert item["evidence_chain"]["schema_version"] == root_material_references.MATERIAL_REFERENCE_TRACE_SCHEMA_VERSION
    finally:
        main.PROJECT_ROOT = original_root
        main._EXPERIMENTS.clear()
        main._EXPERIMENTS.update(original_experiments)


def test_secondary_material_actions_are_visible_to_published_search(tmp_path: Path) -> None:
    original_root = main.PROJECT_ROOT
    original_experiments = dict(main._EXPERIMENTS)
    try:
        from labsopguard.material_maintenance import rebuild_workspace_published_materials_index, query_workspace_published_materials

        main.PROJECT_ROOT = tmp_path
        main._EXPERIMENTS.clear()

        experiment_id = "secondary-actions-exp"
        exp_dir = tmp_path / "outputs" / "experiments" / experiment_id
        formal_root = exp_dir / "material_references"
        keyframe_dir = formal_root / root_material_references.KEYFRAME_DIR_NAME
        keyframe_dir.mkdir(parents=True, exist_ok=True)
        frame = keyframe_dir / "paper_balance.jpg"
        frame.write_bytes(b"jpg")
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": experiment_id, "title": "secondary material action test"}, ensure_ascii=False),
            encoding="utf-8",
        )
        formal_rows = [
            {
                "schema_version": "material_reference.item.v1",
                "asset_kind": root_material_references.KEYFRAME_DIR_NAME,
                "material_type": root_material_references.KEYFRAME_DIR_NAME,
                "action_name": "手与称量纸/天平操作",
                "primary_object": "paper",
                "canonical_action_type": "hand-paper",
                "canonical_object": "paper",
                "secondary_actions": ["hand-balance"],
                "secondary_objects": ["balance"],
                "stored_file": str(frame),
                "stored_filename": frame.name,
                "review_status": "accepted",
                "candidate_status": "approved",
                "formal_material_reference": True,
            }
        ]
        _write_jsonl(formal_root / f"{root_material_references.MATERIAL_INDEX_BASENAME}.jsonl", formal_rows)

        published = main._sync_published_materials_from_references(exp_dir, experiment_id)
        item = published["items"][0]
        assert "balance" in item["object_labels"]
        assert "hand-balance" in item["actions"]

        index_path = tmp_path / "published_materials.sqlite"
        rebuild = rebuild_workspace_published_materials_index(tmp_path / "outputs" / "experiments", index_path)
        assert rebuild["total"] == 1
        queried = query_workspace_published_materials(index_path, text="天平", limit=10)
        assert queried["total"] == 1
        assert queried["items"][0]["payload"]["secondary_actions"] == ["hand-balance"]
    finally:
        main.PROJECT_ROOT = original_root
        main._EXPERIMENTS.clear()
        main._EXPERIMENTS.update(original_experiments)


def test_material_candidates_payload_keeps_blocked_status_out_of_pending_logic(tmp_path: Path) -> None:
    original_root = main.PROJECT_ROOT
    original_experiments = dict(main._EXPERIMENTS)
    try:
        main.PROJECT_ROOT = tmp_path
        main._EXPERIMENTS.clear()

        experiment_id = "contract-candidates"
        exp_dir = tmp_path / "outputs" / "experiments" / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": experiment_id, "title": "Contract candidates"}, ensure_ascii=False),
            encoding="utf-8",
        )

        candidate_root = exp_dir / "_material_review_queue"
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate_rows = [
            {"candidate_id": "p1", "candidate_group_id": "g-pending", "candidate_status": "pending", "asset_kind": root_material_references.KEYFRAME_DIR_NAME},
            {"candidate_id": "p2", "candidate_group_id": "g-pending", "candidate_status": "pending", "asset_kind": root_material_references.KEY_CLIP_DIR_NAME},
            {"candidate_id": "r1", "candidate_group_id": "g-reject", "candidate_status": "rejected", "asset_kind": root_material_references.KEYFRAME_DIR_NAME},
            {"candidate_id": "d1", "candidate_group_id": "g-defer", "candidate_status": "deferred", "asset_kind": root_material_references.KEYFRAME_DIR_NAME},
            {"candidate_id": "n1", "candidate_group_id": "g-notsel", "candidate_status": "not_selected", "asset_kind": root_material_references.KEYFRAME_DIR_NAME},
        ]
        _write_jsonl(candidate_root / f"{root_material_references.MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl", candidate_rows)

        payload = main._material_candidates_payload(experiment_id)
        assert payload["total"] == 4
        assert payload["pending_total"] == 2
        assert payload["rejected_total"] == 1
        assert payload["deferred_total"] == 1
        assert payload["not_selected_total"] == 1

        statuses = {item["candidate_group_id"]: str(item["status"]) for item in payload["items"]}
        assert statuses["g-pending"] == "pending"
        assert statuses["g-reject"] == "rejected"
        assert statuses["g-defer"] == "deferred"
        assert statuses["g-notsel"] == "not_selected"

        client = TestClient(main.app)
        response = client.get(f"/api/v1/experiments/{experiment_id}/materials/candidates")
        assert response.status_code == 200
        response_payload = response.json()
        assert response_payload["pending_total"] == 2
        blocked = {item["status"] for item in response_payload["items"] if item.get("candidate_group_id") in {"g-reject", "g-defer", "g-notsel"}}
        assert blocked == {"rejected", "deferred", "not_selected"}
    finally:
        main.PROJECT_ROOT = original_root
        main._EXPERIMENTS.clear()
        main._EXPERIMENTS.update(original_experiments)


def test_candidate_window_api_exposes_separate_preview_artifacts(tmp_path: Path) -> None:
    original_root = main.PROJECT_ROOT
    original_experiments = dict(main._EXPERIMENTS)
    try:
        main.PROJECT_ROOT = tmp_path
        main._EXPERIMENTS.clear()

        experiment_id = "window-preview-contract"
        exp_dir = tmp_path / "outputs" / "experiments" / experiment_id
        metadata_dir = exp_dir / "key_action_index" / "metadata"
        window_dir = exp_dir / "windows" / "formal_window_001"
        window_dir.mkdir(parents=True, exist_ok=True)
        third_preview = window_dir / "third_view_realtime_preview.mp4"
        first_preview = window_dir / "first_view_realtime_preview.mp4"
        side_preview = window_dir / "window_preview.browser.mp4"
        for path in (third_preview, first_preview, side_preview, window_dir / "sample_grid.jpg", window_dir / "window_sync_index.csv"):
            path.write_bytes(b"x")
        window_report = {
            "third_view_realtime_preview": str(third_preview),
            "first_view_realtime_preview": str(first_preview),
            "side_by_side_realtime_preview": str(side_preview),
            "window_preview_browser": str(side_preview),
            "window_preview_duration_s": 30.0,
            "window_preview_mode": "realtime_preview",
            "raw_window_start_global_timestamp_us": 1000,
            "actual_experiment_start_global_timestamp_us": 2000,
            "focus_preview_start_global_timestamp_us": 2000,
            "actual_experiment_duration_s": 42.0,
        }
        _write_json(window_dir / "window_report.json", window_report)
        _write_json(
            metadata_dir / "formal_experiment_windows.json",
            {
                "windows": [
                    {
                        "experiment_window_id": "formal_window_001",
                        "start_sec": 1.0,
                        "end_sec": 61.0,
                        "duration_sec": 60.0,
                        "window_report": str(window_dir / "window_report.json"),
                        "source_window_sync_index": str(window_dir / "window_sync_index.csv"),
                    }
                ]
            },
        )

        segments = main._candidate_experiment_window_segments(experiment_id, exp_dir)
        assert len(segments) == 1
        item = segments[0]
        assert item["third_view_realtime_preview"]
        assert item["first_view_realtime_preview"]
        assert item["side_by_side_realtime_preview"]
        assert item["third_person_video_url"] != item["side_by_side_realtime_preview"]
        assert item["first_person_video_url"] != item["side_by_side_realtime_preview"]
        assert item["actual_experiment_start_global_timestamp_us"] == 2000
        assert item["raw_window_start_global_timestamp_us"] == 1000
    finally:
        main.PROJECT_ROOT = original_root
        main._EXPERIMENTS.clear()
        main._EXPERIMENTS.update(original_experiments)


def test_published_material_api_returns_window_linkage_and_orphan_report(tmp_path: Path) -> None:
    import shutil

    original_root = main.PROJECT_ROOT
    original_experiments = dict(main._EXPERIMENTS)
    try:
        main.PROJECT_ROOT = tmp_path
        main._EXPERIMENTS.clear()

        experiment_id = "contract-window-linkage-exp"
        exp_dir = tmp_path / "outputs" / "experiments" / experiment_id
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "experiment.json").write_text(
            json.dumps({"experiment_id": experiment_id, "title": "Contract window linkage API"},
                       ensure_ascii=False),
            encoding="utf-8",
        )

        (exp_dir / "window_preview.jpg").write_bytes(b"preview")
        (exp_dir / "sample_grid.jpg").write_bytes(b"grid")

        published_materials = {
            "schema_version": "published_materials.v1",
            "total": 4,
            "items": [
                {
                    "material_id": "mat-with-exp-window",
                    "item_id": "item-exp-window",
                    "experiment_window_id": "segment-exp",
                    "window_sync_index": "101",
                    "window_preview": "window_preview.jpg",
                    "sample_grid": "sample_grid.jpg",
                    "review_status": "needs_review",
                    "official_status": "candidate_review",
                    "keyframe_quality_score": 0.93,
                    "cli_ready_folder": "cli/mat-with-exp-window",
                    "evidence_bundle_id": "bundle-exp-window",
                    "display_title": "valid window id",
                    "event_type": "hand-container",
                    "time_start": 1.0,
                    "time_end": 2.0,
                    "exists": True,
                },
                {
                    "material_id": "mat-missing-sync",
                    "item_id": "item-missing-sync",
                    "window_id": "segment-missing-sync",
                    "display_title": "orphan missing source sync",
                    "event_type": "hand-container",
                    "time_start": 2.0,
                    "time_end": 3.0,
                    "exists": True,
                },
                {
                    "material_id": "mat-missing-both",
                    "item_id": "item-missing-both",
                    "display_title": "orphan missing window",
                    "event_type": "hand-paper",
                    "time_start": 4.0,
                    "time_end": 5.0,
                    "exists": True,
                },
                {
                    "material_id": "mat-segment-only",
                    "item_id": "item-segment-only",
                    "segment_id": "segment-only",
                    "source_window_sync_index": "303",
                    "display_title": "valid alias from segment_id",
                    "event_type": "hand-paper",
                    "time_start": 3.0,
                    "time_end": 4.0,
                    "exists": True,
                },
            ],
        }
        _write_json(exp_dir / "published_materials.json", published_materials)

        client = TestClient(main.app)
        response = client.get(f"/api/v1/experiments/{experiment_id}/materials/published")
        assert response.status_code == 200
        payload = response.json()
        all_items = payload.get("all_items")
        items = payload.get("items")
        if isinstance(all_items, list):
            material_items = all_items
        else:
            material_items = items if isinstance(items, list) else []
        assert isinstance(material_items, list)
        assert len(material_items) == 4

        assert isinstance(items, list)
        if items is not None:
            assert len(items) <= 4
        by_material_id = {item["material_id"]: item for item in material_items}

        valid_item = by_material_id["mat-with-exp-window"]
        assert valid_item["window_id"] == "segment-exp"
        assert valid_item["source_window_sync_index"] == "101"
        assert valid_item["window_preview"].endswith("window_preview.jpg")
        assert valid_item["sample_grid"].endswith("sample_grid.jpg")
        assert valid_item["review_status"] == "needs_review"
        assert valid_item["official_status"] == "candidate_review"
        assert valid_item["keyframe_quality_score"] == 0.93
        assert valid_item["cli_ready_folder"] == "cli/mat-with-exp-window"
        assert valid_item["evidence_bundle_id"] == "bundle-exp-window"
        assert valid_item["orphan_material"] is False

        missing_sync_item = by_material_id["mat-missing-sync"]
        assert missing_sync_item["window_id"] == "segment-missing-sync"
        assert missing_sync_item.get("source_window_sync_index") in ("", None)
        assert missing_sync_item["orphan_material"] is True

        missing_both_item = by_material_id["mat-missing-both"]
        assert missing_both_item.get("window_id") in ("", None)
        assert missing_both_item.get("source_window_sync_index") in ("", None)
        assert missing_both_item["orphan_material"] is True

        aliased_segment_item = by_material_id["mat-segment-only"]
        assert aliased_segment_item["window_id"] == "segment-only"
        assert aliased_segment_item["source_window_sync_index"] == "303"
        assert aliased_segment_item["segment_id"] == "segment-only"
        assert aliased_segment_item["orphan_material"] is False

        report_path = exp_dir / "material_window_dependency_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["material_count"] == 4
        assert report["material_with_window_id_count"] == 3
        assert report["material_with_source_window_sync_index_count"] == 2
        assert report["material_with_window_preview_count"] >= 1
        assert report["material_with_sample_grid_count"] >= 1
        assert report["orphan_material_count"] == 2
        assert len(report["materials_missing_window_refs"]) == 2
        assert report["status"] == "needs_attention"
    finally:
        main.PROJECT_ROOT = original_root
        main._EXPERIMENTS.clear()
        main._EXPERIMENTS.update(original_experiments)
