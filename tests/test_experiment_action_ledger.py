from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.experiment_action_ledger import (
    apply_material_candidate_feedback,
    build_experiment_action_ledger,
    query_materials,
    refresh_labvideo_memory_corpus,
    run_labvideo_backfill,
    sync_candidate_review_outputs,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _make_review_candidate_material(exp_root: Path, *, material_id: str = "m-review", evidence_bundle_id: str = "b-review") -> dict:
    source_root = exp_root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    for name in ("first.jpg", "third.jpg", "first.mp4", "third.mp4"):
        (source_root / name).write_bytes(b"media")
    sync_index = exp_root / "windows" / "w1" / "window_sync_index.csv"
    sync_index.parent.mkdir(parents=True, exist_ok=True)
    sync_index.write_text("window_sync_index,global_timestamp_us\n0,1\n", encoding="utf-8")
    return {
        "material_id": material_id,
        "evidence_bundle_id": evidence_bundle_id,
        "action_type": "hand_object_contact",
        "official_status": "needs_review",
        "review_status": "needs_review",
        "candidate_status": "needs_review",
        "candidate_group_id": "g-review",
        "candidate_id": "c-review-1",
        "experiment_id": exp_root.name,
        "window_id": "w1",
        "source_window_sync_index": str(sync_index),
        "dual_view_action_phase_status": "dual_view_valid",
        "first_keyframe": str(source_root / "first.jpg"),
        "third_keyframe": str(source_root / "third.jpg"),
        "first_keyclip": str(source_root / "first.mp4"),
        "third_keyclip": str(source_root / "third.mp4"),
        "keyframe_quality_score": 0.82,
        "selected_keyframe_reason": "test-quality",
        "confidence": 0.9,
        "cli_ready_folder": str(exp_root),
        "lineage": {"source": "unit-test"},
        "memory_eligible": False,
    }


def test_experiment_action_ledger_keeps_review_candidates_out_of_memory(tmp_path: Path) -> None:
    material_root = tmp_path / "library"
    exp_root = material_root / "exp-001"
    _write_jsonl(
        exp_root / "material_stream.jsonl",
        [
            {
                "material_id": "m1",
                "evidence_bundle_id": "b1",
                "action_type": "hand_object_contact",
                "official_status": "needs_review",
                "object_refs": ["weighing_paper"],
                "first_keyframe": "first.jpg",
                "third_keyframe": "third.jpg",
                "cli_ready_folder": str(exp_root),
            }
        ],
    )

    ledger = build_experiment_action_ledger(material_root, "exp-001")

    assert ledger["ledger_status"] == "needs_review_only"
    assert ledger["memory_eligible"] is False
    assert ledger["official_material_count"] == 0
    assert ledger["review_candidate_count"] == 1
    assert (exp_root / "experiment_action_ledger.json").exists()
    assert (exp_root / "material_index.sqlite").exists()
    assert (exp_root / "evidence_trace_index.json").exists()


def test_legacy_material_references_are_not_auto_promoted_to_official(tmp_path: Path) -> None:
    material_root = tmp_path / "library"
    exp_root = material_root / "legacy-exp"
    _write_jsonl(
        exp_root / "key_material_references.jsonl",
        [
            {
                "material_id": "legacy-m1",
                "evidence_bundle_id": "legacy-b1",
                "action_type": "hand_object_contact",
                "stored_file": "legacy.jpg",
                "official_material": True,
                "memory_eligible": True,
            }
        ],
    )

    ledger = build_experiment_action_ledger(material_root, "legacy-exp")

    assert ledger["ledger_status"] == "needs_review_only"
    assert ledger["memory_eligible"] is False
    assert ledger["official_material_count"] == 0
    assert ledger["review_candidate_count"] == 1
    assert ledger["confirmed_action_summary"] == []


def test_corpus_marks_existing_labvideo_backfill_as_not_real_30_day_memory(tmp_path: Path) -> None:
    material_root = tmp_path / "library"
    exp_root = material_root / "exp-002"
    _write_jsonl(
        exp_root / "material_stream.jsonl",
        [
            {
                "material_id": "m2",
                "evidence_bundle_id": "b2",
                "action_type": "device_panel_interaction",
                "official_status": "official",
                "memory_eligible": True,
                "instrument_refs": ["balance"],
                "cli_ready_folder": str(exp_root),
            }
        ],
    )

    corpus = refresh_labvideo_memory_corpus(material_root, labvideo_root=tmp_path / "LabVideo")
    rows = query_materials(material_root, action_type="device_panel_interaction")

    assert corpus["source_mode"] == "existing_labvideo_backfill"
    assert corpus["is_real_30_day_memory"] is False
    assert corpus["official_ready_ledger_count"] == 1
    assert corpus["total_official_materials"] == 1
    assert rows and rows[0]["memory_eligible"] is True
    assert (material_root / "global_experiment_ledger_index.jsonl").exists()
    assert (material_root / "global_material_search_index.jsonl").exists()
    assert (material_root / "global_evidence_trace_index.jsonl").exists()


def test_candidate_feedback_sync_refreshes_officials_ledger_and_corpus(tmp_path: Path) -> None:
    material_root = tmp_path / "library"
    session = tmp_path / "outputs" / "experiments" / "exp-003" / "key_action_index"
    candidate_root = tmp_path / "outputs" / "experiments" / "exp-003" / "material_candidates"
    frame = candidate_root / "关键帧" / "candidate.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"image")
    rows = [
        {
            "candidate_id": "c1",
            "candidate_group_id": "g1",
            "evidence_bundle_id": "b3",
            "asset_kind": "关键帧",
            "view": "third_person",
            "stored_file": str(frame),
            "action_type": "hand_object_contact",
            "candidate_status": "approved",
            "review_status": "accepted",
            "official_material": True,
            "memory_write_allowed": True,
        }
    ]

    summary = sync_candidate_review_outputs(
        session,
        candidate_root,
        rows,
        {"decision": "approved", "candidate_group_id": "g1", "candidate_ids": ["c1"], "reviewer": "tester"},
        material_root=material_root,
    )
    exp_root = material_root / "exp-003"
    official = [json.loads(line) for line in (exp_root / "official_materials.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ledger = json.loads((exp_root / "experiment_action_ledger.json").read_text(encoding="utf-8"))
    corpus = json.loads((material_root / "labvideo_memory_corpus.json").read_text(encoding="utf-8"))

    assert summary["official_material_count"] == 1
    assert official[0]["official_status"] == "official"
    assert ledger["ledger_status"] == "official_ready"
    assert ledger["memory_eligible"] is True
    assert corpus["is_real_30_day_memory"] is False
    assert (exp_root / "human_feedback.jsonl").exists()
    assert (exp_root / "feedback_update_jobs.jsonl").exists()
    assert (exp_root / "corrected_material_stream.jsonl").exists()


def test_needs_review_candidate_can_be_promoted_to_official_material_db(tmp_path: Path) -> None:
    material_root = tmp_path / "library"
    exp_root = material_root / "exp-promote"
    row = _make_review_candidate_material(exp_root)
    _write_jsonl(exp_root / "material_stream.jsonl", [row])
    _write_jsonl(exp_root / "review_candidate_materials.jsonl", [row])

    report = apply_material_candidate_feedback(
        material_root,
        "exp-promote",
        candidate_group_id="g-review",
        action="upgrade_to_official",
        reviewer="tester",
        notes="promote one candidate",
    )
    official = [json.loads(line) for line in (exp_root / "official_materials.jsonl").read_text(encoding="utf-8").splitlines() if line]
    stream = [json.loads(line) for line in (exp_root / "material_stream.jsonl").read_text(encoding="utf-8").splitlines() if line]
    review = [json.loads(line) for line in (exp_root / "review_candidate_materials.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ledger = json.loads((exp_root / "experiment_action_ledger.json").read_text(encoding="utf-8"))

    assert report["after"]["official_material_count"] == 1
    assert official[0]["official_status"] == "official"
    assert official[0]["memory_eligible"] is True
    assert Path(official[0]["quality_report"]).is_file()
    assert stream[0]["official_status"] == "official"
    assert review[0]["review_status"] == "upgraded"
    assert review[0]["upgraded_to_official"] is True
    assert (exp_root / "human_feedback.jsonl").exists()
    assert (exp_root / "corrected_material_stream.jsonl").exists()
    assert ledger["ledger_status"] == "official_ready"
    assert ledger["confirmed_action_summary"]
    rows = query_materials(material_root, action_type="hand_object_contact")
    assert rows and rows[0]["official_status"] == "official"
    assert rows[0]["memory_eligible"] is True


def test_rejected_candidate_does_not_become_official(tmp_path: Path) -> None:
    material_root = tmp_path / "library"
    exp_root = material_root / "exp-reject"
    row = _make_review_candidate_material(exp_root, material_id="m-reject", evidence_bundle_id="b-reject")
    _write_jsonl(exp_root / "material_stream.jsonl", [row])
    _write_jsonl(exp_root / "review_candidate_materials.jsonl", [row])

    report = apply_material_candidate_feedback(
        material_root,
        "exp-reject",
        candidate_group_id="g-review",
        action="reject",
        reviewer="tester",
        reason_code="wrong_action",
        reason="not the expected action",
    )
    official_text = (exp_root / "official_materials.jsonl").read_text(encoding="utf-8") if (exp_root / "official_materials.jsonl").exists() else ""
    stream = [json.loads(line) for line in (exp_root / "material_stream.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ledger = json.loads((exp_root / "experiment_action_ledger.json").read_text(encoding="utf-8"))

    assert report["after"]["official_material_count"] == 0
    assert official_text == ""
    assert stream[0]["official_status"] == "rejected"
    assert stream[0]["memory_eligible"] is False
    assert ledger["ledger_status"] == "insufficient_evidence"
    assert ledger["confirmed_action_summary"] == []


def test_labvideo_backfill_scan_is_incremental_and_skips_incomplete_sources(tmp_path: Path) -> None:
    labvideo = tmp_path / "LabVideo"
    good = labvideo / "raw_uploads" / "by_import" / "good_run"
    (good / "cam01").mkdir(parents=True)
    (good / "cam02").mkdir(parents=True)
    (good / "cam01" / "rgb.mp4").write_bytes(b"third")
    (good / "cam02" / "rgb.mp4").write_bytes(b"first")
    (good / "cam01" / "frames.csv").write_text("frame_index,packet_system_timestamp_us\n0,1\n", encoding="utf-8")
    (good / "cam02" / "frames.csv").write_text("frame_index,packet_system_timestamp_us\n0,1\n", encoding="utf-8")
    incomplete = labvideo / "raw_uploads" / "by_import" / "incomplete"
    incomplete.mkdir(parents=True)

    material_root = tmp_path / "library"
    report = run_labvideo_backfill(labvideo, material_root)
    plan = json.loads((material_root / "batch_backfill_plan.json").read_text(encoding="utf-8"))

    assert report["scanned_experiment_count"] == 2
    modes = {entry["source_name"]: entry["mode"] for entry in plan["entries"]}
    assert modes["good_run"] == "full_analysis"
    assert modes["incomplete"] == "skipped"
    assert (material_root / "batch_backfill_run_report.json").exists()
    assert (material_root / "batch_backfill_quality_report.json").exists()
