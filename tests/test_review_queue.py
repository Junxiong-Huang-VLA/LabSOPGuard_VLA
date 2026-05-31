from __future__ import annotations

from pathlib import Path

from key_action_indexer.review_queue import apply_review_decision, build_review_queue
from key_action_indexer.schemas import write_jsonl


def test_review_queue_collects_quality_segment_micro_and_material_items(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    cv_outputs = tmp_path / "cv_outputs"
    (tmp_path / "index").mkdir(parents=True)
    (tmp_path / "index" / "fallback_index.pkl").write_bytes(b"index")
    cv_outputs.mkdir(parents=True)
    segment = {
        "segment_id": "seg_1",
        "duration_sec": 90.0,
        "boundary_confidence": 0.0,
        "third_person": {"local_start_sec": 1.0, "local_end_sec": 91.0},
        "text_description": {"summary": "coarse segment", "action_type": "weighing"},
    }
    micro = {
        "micro_segment_id": "micro_1",
        "parent_segment_id": "seg_1",
        "start_sec": 2.0,
        "end_sec": 8.0,
        "confidence": "low",
        "evidence_level": "weak_visual_evidence",
        "primary_object": "balance",
    }
    write_jsonl(cv_outputs / "detected_segments.jsonl", [segment])
    write_jsonl(metadata / "key_action_segments.jsonl", [segment])
    write_jsonl(metadata / "micro_segments.jsonl", [micro])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"segment_id": "seg_1"}])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [{"micro_segment_id": "micro_1"}])

    queue = build_review_queue(
        tmp_path,
        material_candidates={
            "items": [
                {
                    "candidate_group_id": "mat_1",
                    "status": "pending",
                    "files": [],
                    "quality_score": 0.7,
                }
            ]
        },
    )

    item_types = {item["item_type"] for item in queue["items"]}
    assert {"qa_warning", "segment", "micro_segment", "material_candidate"}.issubset(item_types)
    assert queue["summary"]["pending"] == queue["summary"]["total"]
    assert queue["quality"]["core_metrics"]["unreviewed_count"] >= 1

    decision = apply_review_decision(tmp_path, item_id="micro:micro_1", decision="approved", reviewer="tester", note="ok")
    assert decision["decision"] == "approved"
    updated = build_review_queue(tmp_path)
    approved = [item for item in updated["items"] if item["item_id"] == "micro:micro_1"][0]
    assert approved["review_status"] == "approved"


def test_review_queue_links_semantic_adapter_failures_to_timeline_target(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    cv_outputs = tmp_path / "cv_outputs"
    (tmp_path / "index").mkdir(parents=True)
    (tmp_path / "index" / "fallback_index.pkl").write_bytes(b"index")
    cv_outputs.mkdir(parents=True)
    segment = {
        "segment_id": "seg_1",
        "start_sec": 0.0,
        "end_sec": 10.0,
        "duration_sec": 10.0,
        "boundary_confidence": 0.9,
        "boundary_source": "yolo_physical_evidence",
        "action_type": "weighing",
        "primary_object": "balance",
    }
    write_jsonl(cv_outputs / "detected_segments.jsonl", [segment])
    write_jsonl(metadata / "key_action_segments.jsonl", [segment])
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"segment_id": "seg_1", "index_text": "balance weighing"}])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [])
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(
        metadata / "panel_ocr.jsonl",
        [{"session_id": "sess_1", "view": "first_person", "start_sec": 1.0, "end_sec": 2.0, "equipment_label": "balance", "state": "visible"}],
    )
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])

    queue = build_review_queue(tmp_path)

    semantic_items = [item for item in queue["items"] if item["item_type"] == "evidence_semantic"]
    assert semantic_items
    item = semantic_items[0]
    assert item["segment_id"] == "seg_1"
    assert item["start_sec"] == 1.0
    assert item["end_sec"] == 2.0
    assert item["payload"]["semantic_category"] == "missing_fields"
