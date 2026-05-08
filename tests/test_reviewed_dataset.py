from __future__ import annotations

from pathlib import Path

from key_action_indexer.review_queue import apply_review_decision
from key_action_indexer.reviewed_dataset import freeze_reviewed_dataset, rollback_reviewed_release
from key_action_indexer.schemas import read_jsonl, write_jsonl
from key_action_indexer.vector_index import VectorIndex


def _build_source_index(session: Path) -> None:
    rows = [
        {
            "index_level": "segment",
            "segment_id": "seg_1",
            "action_type": "weighing",
            "primary_object": "balance",
            "index_text": "balance weighing sample",
            "keyframes": ["seg.jpg"],
        },
        {
            "index_level": "micro_segment",
            "segment_id": "seg_1",
            "micro_segment_id": "micro_1",
            "action_type": "weighing",
            "primary_object": "balance",
            "index_text": "balance weighing hand contact",
            "keyframes": ["micro.jpg"],
        },
    ]
    index = VectorIndex()
    index.build([row["index_text"] for row in rows], rows)
    index.save(session / "index")


def test_freeze_reviewed_dataset_applies_decisions_boundaries_and_index(tmp_path: Path) -> None:
    session = tmp_path
    metadata = session / "metadata"
    segment = {
        "segment_id": "seg_1",
        "start_sec": 0.0,
        "end_sec": 20.0,
        "duration_sec": 20.0,
        "text_description": {"summary": "balance weighing"},
        "index": {"index_text": "balance weighing segment"},
    }
    micro = {
        "micro_segment_id": "micro_1",
        "parent_segment_id": "seg_1",
        "start_sec": 2.0,
        "end_sec": 8.0,
        "duration_sec": 6.0,
        "primary_object": "balance",
        "index": {"index_text": "balance weighing micro"},
    }
    write_jsonl(metadata / "key_action_segments.jsonl", [segment])
    write_jsonl(metadata / "micro_segments.jsonl", [micro])
    write_jsonl(metadata / "model_observation_events.jsonl", [{"observation_id": "obs_1", "segment_id": "seg_1", "micro_segment_id": "micro_1"}])
    _build_source_index(session)

    apply_review_decision(
        session,
        item_id="micro:micro_1",
        decision="approved",
        reviewer="qa",
        note="tightened",
        boundary_start_sec=3.0,
        boundary_end_sec=7.0,
    )
    manifest = freeze_reviewed_dataset(session)

    reviewed_micros = read_jsonl(metadata / "reviewed_micro_segments.jsonl")
    assert manifest["reviewed_counts"]["micro_segments"] == 1
    assert reviewed_micros[0]["review_status"] == "approved"
    assert reviewed_micros[0]["start_sec"] == 3.0
    assert reviewed_micros[0]["end_sec"] == 7.0
    assert reviewed_micros[0]["duration_sec"] == 4.0
    assert reviewed_micros[0]["reviewed_boundary"]["adjusted_start_sec"] == 3.0
    assert (session / "reviewed_index" / "fallback_index.pkl").exists()
    assert (metadata / "reviewed_export.json").exists()


def test_freeze_reviewed_dataset_excludes_rejected_rows(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_1", "index": {"index_text": "bad"}}])
    write_jsonl(metadata / "micro_segments.jsonl", [])

    apply_review_decision(tmp_path, item_id="segment:seg_1", decision="rejected", reviewer="qa")
    manifest = freeze_reviewed_dataset(tmp_path)

    assert manifest["reviewed_counts"]["segments"] == 0
    assert read_jsonl(metadata / "reviewed_segments.jsonl") == []


def test_freeze_reviewed_dataset_converges_long_segment_and_creates_release(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [{"segment_id": "seg_long", "start_sec": 0.0, "end_sec": 100.0, "duration_sec": 100.0, "index": {"index_text": "coarse action"}}],
    )
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {"micro_segment_id": "micro_a", "parent_segment_id": "seg_long", "start_sec": 10.0, "end_sec": 15.0, "index": {"index_text": "micro a"}},
            {"micro_segment_id": "micro_b", "parent_segment_id": "seg_long", "start_sec": 60.0, "end_sec": 65.0, "index": {"index_text": "micro b"}},
        ],
    )

    first = freeze_reviewed_dataset(tmp_path)
    second = freeze_reviewed_dataset(tmp_path)
    rollback = rollback_reviewed_release(tmp_path)

    assert first["auto_convergence"]["applied"] is True
    assert first["reviewed_counts"]["segments"] == 2
    assert first["reviewed_metrics"]["total_action_coverage_ratio"] == 0.1
    assert first["reviewed_metrics"]["unreviewed_count"] == 0
    assert first["release"]["version"] == "v001"
    assert second["release"]["version"] == "v002"
    assert rollback["active_version"] == "v001"
    assert (tmp_path / "reviewed_releases" / "v001" / "reviewed_release_export.zip").exists()
