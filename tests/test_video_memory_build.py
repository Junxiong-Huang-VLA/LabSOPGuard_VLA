from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.material_library_store import sync_material_library
from key_action_indexer.video_memory import (
    build_video_memory,
    get_memory_snapshot,
    query_video_memory,
    record_human_feedback,
)


def _write_fake_material_package(root: Path) -> None:
    package_root = root / "material_references" / "pkg_20260525"
    (package_root / "keyframes").mkdir(parents=True)
    (package_root / "clips").mkdir(parents=True)
    (package_root / "keyframes" / "paper_third.jpg").write_bytes(b"\xff\xd8\xff\xe0real keyframe")
    (package_root / "clips" / "paper_first.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42real clip")
    rows = [
        {
            "experiment_id": "exp1",
            "asset_type": "keyframe",
            "action_name": "hand_object_contact",
            "canonical_action_type": "hand_object_contact",
            "primary_object": "paper",
            "view": "third_person",
            "start_sec": 10,
            "end_sec": 12,
            "stored_file": "keyframes/paper_third.jpg",
            "file_name": "paper_third.jpg",
            "micro_segment_id": "micro1",
            "segment_id": "seg1",
            "quality_score": 0.82,
            "yolo_evidence_count": 7,
        },
        {
            "experiment_id": "exp1",
            "asset_type": "video_clip",
            "action_name": "hand_object_contact",
            "canonical_action_type": "hand_object_contact",
            "primary_object": "paper",
            "view": "first_person",
            "start_sec": 10,
            "end_sec": 12,
            "stored_file": "clips/paper_first.mp4",
            "file_name": "paper_first.mp4",
            "micro_segment_id": "micro1",
            "segment_id": "seg1",
            "quality_score": 0.78,
            "yolo_evidence_count": 6,
        },
    ]
    (package_root / "key_material_references.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


def test_build_video_memory_from_material_library_creates_partial_snapshot_and_query_trace(tmp_path: Path) -> None:
    library_root = tmp_path / "LabMaterialLibrary"
    _write_fake_material_package(library_root)
    sync_material_library(library_root)

    result = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        job_type="backfill",
    )

    snapshot = result["snapshot"]
    assert result["counts"]["materials"] == 2
    assert result["counts"]["evidence_bundles"] == 1
    assert result["counts"]["clusters"] == 1
    assert snapshot["window_completeness"] == "partial"
    assert snapshot["window_completeness_ratio"] == "1/30"
    assert snapshot["cluster_ids"]
    assert snapshot["material_ids"]
    assert snapshot["sha256s"]
    assert snapshot["micro_segment_ids"] == ["micro1"]
    assert snapshot["keyframe_refs"]
    assert snapshot["keyclip_refs"]
    assert snapshot["timestamps"]
    assert snapshot["evidence_trace"]["trace_complete"] is True
    assert not snapshot["recurring_workflow_patterns"]

    bundles = _read_jsonl(Path(result["memory_index_root"]) / "evidence_bundles.jsonl")
    ledgers = _read_jsonl(Path(result["memory_index_root"]) / "daily_event_ledgers.jsonl")
    clusters = _read_jsonl(Path(result["memory_index_root"]) / "memory_clusters.jsonl")
    for row in (bundles[0], ledgers[0], clusters[0]):
        assert row["material_ids"]
        assert row["sha256s"]
        assert row["micro_segment_ids"] == ["micro1"]
        assert row["keyframe_refs"]
        assert row["keyclip_refs"]
        assert row["evidence_trace"]["trace_complete"] is True

    loaded = get_memory_snapshot(library_root=library_root)
    assert loaded is not None
    assert loaded["snapshot_id"] == snapshot["snapshot_id"]

    answer = query_video_memory("paper hand contact", library_root=library_root)
    assert answer["claims"]
    claim = answer["claims"][0]
    assert claim["cluster_id"] == snapshot["cluster_ids"][0]
    assert claim["ledger_event_id"]
    assert claim["evidence_bundle_id"]
    assert claim["material_id"]
    assert claim["session_id"] == "pkg_20260525"
    assert claim["experiment_id"] == "exp1"
    assert claim["micro_segment_id"] == "micro1"
    assert claim["confidence"] > 0
    assert claim["human_confirmation_status"]
    assert claim["evidence_bundle_ids"]
    assert claim["keyframe"] or claim["keyclip"]
    assert claim["timestamp"]
    assert claim["evidence_trace"]["ledger_event_ids"]
    assert claim["keyframe_refs"] or claim["keyclip_refs"]
    assert claim["micro_segment_ids"] == ["micro1"]
    assert "1/30" in answer["answer_summary"]
    assert answer["partial_window"] is True
    assert answer["is_full_30_day_memory"] is False

    feedback = record_human_feedback(
        target_type="cluster",
        target_id=snapshot["cluster_ids"][0],
        feedback_type="confirm",
        context_fields={"project_name": "confirmed project"},
        library_root=library_root,
    )
    assert feedback["feedback_type"] == "confirm"

    rebuilt = build_video_memory(
        library_root=library_root,
        window_end_date="2026-05-25",
        job_type="feedback_update",
    )
    assert rebuilt["snapshot"]["human_confirmed_contexts"]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
