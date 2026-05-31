from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer.retrieval_eval import confirm_gold_query_benchmark
from key_action_indexer.review_queue import apply_review_decision
from key_action_indexer.reviewed_dataset import (
    active_reviewed_release,
    freeze_reviewed_dataset,
    load_reviewed_export,
    promote_reviewed_release,
    reviewed_index_dir,
    rollback_reviewed_release,
)
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
            {
                "micro_segment_id": "micro_b",
                "parent_segment_id": "seg_long",
                "start_sec": 60.0,
                "end_sec": 65.0,
                "global_start_time": "2026-05-08T10:01:00+08:00",
                "global_end_time": "2026-05-08T10:01:05+08:00",
                "index_text": "review_status: pending",
                "text_description": {"index_text": "rich micro b sample handling text"},
            },
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
    reviewed_segments = read_jsonl(metadata / "reviewed_segments.jsonl")
    assert reviewed_segments[1]["micro_segment_id"] == "micro_b"
    assert reviewed_segments[1]["global_start_time"] == "2026-05-08T10:01:00+08:00"
    assert "rich micro b sample handling text" in reviewed_segments[1]["index_text"]
    assert second["release"]["version"] == "v002"
    assert rollback["active_version"] == "v001"
    assert (tmp_path / "reviewed_releases" / "v001" / "reviewed_release_export.zip").exists()


def test_promote_reviewed_release_requires_gate_eval_and_becomes_default(tmp_path: Path) -> None:
    session = tmp_path
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    cv_outputs.mkdir(parents=True)
    segment = {
        "segment_id": "seg_1",
        "start_sec": 0.0,
        "end_sec": 10.0,
        "duration_sec": 10.0,
        "boundary_confidence": 0.9,
        "boundary_source": "yolo_physical_evidence",
        "boundary_support_count": 3,
        "action_type": "weighing",
        "primary_object": "balance",
        "index_text": "balance weighing sample",
        "keyframes": ["seg.jpg"],
        "evidence_level": "visual_confirmed",
    }
    micro = {
        "micro_segment_id": "micro_1",
        "parent_segment_id": "seg_1",
        "start_sec": 1.0,
        "end_sec": 3.0,
        "duration_sec": 2.0,
        "action_type": "weighing",
        "primary_object": "balance",
        "index_text": "balance weighing sample micro",
        "keyframes": ["micro.jpg"],
        "evidence_level": "visual_confirmed",
    }
    write_jsonl(cv_outputs / "detected_segments.jsonl", [segment])
    write_jsonl(metadata / "key_action_segments.jsonl", [segment])
    write_jsonl(metadata / "micro_segments.jsonl", [micro])
    write_jsonl(metadata / "vector_metadata.jsonl", [{**segment, "index_level": "segment"}])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [{**micro, "index_level": "micro_segment", "segment_id": "seg_1"}])
    write_jsonl(
        metadata / "object_tracks.jsonl",
        [{"session_id": "sess_1", "view": "first_person", "start_sec": 1.0, "end_sec": 2.0, "object_label": "balance", "track_id": "trk_1", "bbox": [0, 0, 10, 10]}],
    )
    write_jsonl(
        metadata / "panel_ocr.jsonl",
        [{"session_id": "sess_1", "view": "first_person", "start_sec": 1.0, "end_sec": 2.0, "measurement": {"equipment_label": "balance", "display_text": "0.124 g"}}],
    )
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])
    _build_source_index(session)

    apply_review_decision(session, item_id="segment:seg_1", decision="approved", reviewer="qa")
    apply_review_decision(session, item_id="micro:micro_1", decision="approved", reviewer="qa")
    release = freeze_reviewed_dataset(session)
    decisions_path = metadata / "gold_query_decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "query_id": f"gold_cn_{index:03d}",
                        "decision": "approved",
                        "expected_segment_ids": ["seg_1"],
                        "expected_micro_segment_ids": ["micro_1"],
                        "expected_index_level": "micro_segment",
                        "reviewer": "qa",
                    }
                    for index in range(1, 4)
                ]
            }
        ),
        encoding="utf-8",
    )
    gold = confirm_gold_query_benchmark(session, query_count=3, reviewer="qa", decisions_path=decisions_path)
    promoted = promote_reviewed_release(session, version=release["release"]["version"], reviewer="qa", note="ship", query_count=3)

    assert gold["reviewed_release"] == "v001"
    assert gold["human_verified_query_count"] == 3
    assert promoted["active_version"] == "v001"
    assert promoted["promotion_requirements"]["gold_benchmark_reviewed_release"] == "v001"
    assert active_reviewed_release(session)["version"] == "v001"
    assert reviewed_index_dir(session) == session / "reviewed_releases" / "v001" / "reviewed_index"
    assert load_reviewed_export(session)["manifest"]["release"]["version"] == "v001"
    assert (session / "reviewed_releases" / "promoted_release.json").exists()
    assert (metadata / "promoted_release.json").exists()


def test_failed_promotion_writes_candidate_artifacts_without_overwriting_defaults(tmp_path: Path) -> None:
    session = tmp_path
    metadata = session / "metadata"
    segment = {
        "segment_id": "seg_1",
        "start_sec": 0.0,
        "end_sec": 10.0,
        "duration_sec": 10.0,
        "boundary_confidence": 0.9,
        "action_type": "weighing",
        "primary_object": "balance",
        "index_text": "balance weighing sample",
        "keyframes": ["seg.jpg"],
    }
    write_jsonl(metadata / "key_action_segments.jsonl", [segment])
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(metadata / "vector_metadata.jsonl", [{**segment, "index_level": "segment"}])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [])
    _build_source_index(session)
    release = freeze_reviewed_dataset(session)
    default_gate = metadata / "quality_gate.json"
    default_eval = session / "evaluation" / "default_chinese_query_validation.json"
    default_eval.parent.mkdir(parents=True)
    default_gate.write_text(json.dumps({"status": "pass", "sentinel": "promoted_default"}), encoding="utf-8")
    default_eval.write_text(json.dumps({"status": "pass", "sentinel": "promoted_default"}), encoding="utf-8")

    try:
        promote_reviewed_release(session, version=release["release"]["version"], reviewer="qa", query_count=2)
    except ValueError as exc:
        assert "reviewed release cannot be promoted" in str(exc)
    else:
        raise AssertionError("promotion should fail without fully human-verified gold benchmark")

    assert json.loads(default_gate.read_text(encoding="utf-8"))["sentinel"] == "promoted_default"
    assert json.loads(default_eval.read_text(encoding="utf-8"))["sentinel"] == "promoted_default"
    assert (metadata / "quality_gate.v001.candidate.json").exists()
    assert (session / "evaluation" / "default_chinese_query_validation.v001.candidate.json").exists()


def test_promote_reviewed_release_rejects_gold_bound_to_other_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = tmp_path
    release_dir = session / "reviewed_releases" / "v002"
    release_dir.mkdir(parents=True)
    (release_dir / "reviewed_release_manifest.json").write_text(
        json.dumps({"version": "v002", "release_dir": str(release_dir)}),
        encoding="utf-8",
    )

    from key_action_indexer import quality_gate, retrieval_eval

    monkeypatch.setattr(
        quality_gate,
        "build_quality_gate",
        lambda *args, **kwargs: {"status": "pass", "can_mark_complete": True, "summary": {"blocking_count": 0}},
    )
    monkeypatch.setattr(
        retrieval_eval,
        "run_default_chinese_query_eval",
        lambda *args, **kwargs: {
            "status": "pass",
            "query_count": 1,
            "total_query_count": 1,
            "applicable_query_count": 1,
            "excluded_query_count": 0,
            "human_verified_query_count": 1,
            "human_reviewed_query_count": 1,
            "benchmark_binding_mode": "human_verified_review_file",
            "reviewed_release": "v001",
            "top1_hit_rate": 1.0,
            "topk_hit_rate": 1.0,
            "expected_id_hit_rate": 1.0,
        },
    )

    with pytest.raises(ValueError, match="reviewed_release_mismatch"):
        promote_reviewed_release(session, version="v002", reviewer="qa", query_count=1)
