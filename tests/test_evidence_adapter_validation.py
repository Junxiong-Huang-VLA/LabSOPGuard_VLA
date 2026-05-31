from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.evidence_adapter_validation import validate_evidence_adapters
from key_action_indexer.quality_gate import build_quality_gate
from key_action_indexer.reviewed_dataset import freeze_reviewed_dataset
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.vector_index import VectorIndex


def _base_session(session: Path) -> None:
    metadata = session / "metadata"
    (session / "manifest.json").write_text(json.dumps({"session_id": "sess_1"}, ensure_ascii=False), encoding="utf-8")
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "duration_sec": 10.0,
                "boundary_confidence": 0.9,
                "action_type": "weighing",
                "primary_object": "balance",
            }
        ],
    )
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(metadata / "vector_metadata.jsonl", [{"segment_id": "seg_1", "index_text": "balance weighing"}])
    index = VectorIndex()
    index.build(["balance weighing"], [{"segment_id": "seg_1", "index_text": "balance weighing"}])
    index.save(session / "index")


def test_validate_evidence_adapters_reports_counts_coverage_and_issues(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(
        metadata / "object_tracks.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "object_label": "tube",
                "track_id": "track_1",
                "bbox": [0, 0, 10, 10],
                "confidence": 0.9,
            }
        ],
    )
    write_jsonl(metadata / "panel_ocr.jsonl", [])
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])

    result = validate_evidence_adapters(tmp_path)

    assert result["summary"]["present_adapter_count"] == 4
    assert result["adapters"]["object_tracks"]["row_count"] == 1
    assert result["adapters"]["object_tracks"]["coverage"]["start_sec"] == 1.0
    assert result["adapters"]["object_tracks"]["warning_count"] >= 1


def test_quality_gate_blocks_missing_adapter_inputs_and_unreviewed_items(tmp_path: Path) -> None:
    _base_session(tmp_path)
    freeze_reviewed_dataset(tmp_path)

    gate = build_quality_gate(tmp_path)

    assert gate["can_mark_complete"] is False
    names = {item["name"] for item in gate["blocking_checks"]}
    assert "missing_adapter_count" in names


def test_quality_gate_accepts_approved_medium_visual_segment_near_threshold(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(metadata / "panel_ocr.jsonl", [])
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])
    write_jsonl(
        metadata / "reviewed_segments.jsonl",
        [
            {
                "segment_id": "seg_1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "boundary_confidence": 0.548,
                "evidence_level": "visual_confirmed",
                "quality": {"confidence": "medium"},
                "visual_keywords": ["balance"],
                "review": {"decision": "approved"},
            }
        ],
    )

    gate = build_quality_gate(tmp_path)

    assert gate["summary"]["low_confidence_segment_count"] == 0
    assert not any(item["name"] == "low_confidence_segment_count" for item in gate["blocking_checks"])


def test_quality_gate_accepts_approved_visual_confirmed_segment_with_assets(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(metadata / "panel_ocr.jsonl", [])
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])
    write_jsonl(
        metadata / "reviewed_segments.jsonl",
        [
            {
                "segment_id": "seg_1",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "boundary_confidence": 0.41,
                "evidence_level": "visual_confirmed",
                "keyframes": {"peak_frame": "keyframes/seg_1.jpg"},
                "yolo_interactions": [{"object_label": "balance", "confidence": 0.83}],
                "review": {"decision": "approved"},
            }
        ],
    )

    gate = build_quality_gate(tmp_path)

    assert gate["summary"]["low_confidence_segment_count"] == 0
    assert not any(item["name"] == "low_confidence_segment_count" for item in gate["review_required_checks"])


def test_validate_evidence_adapters_reports_semantic_support_issues(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(metadata / "panel_ocr.jsonl", [{"session_id": "sess_1", "view": "first_person", "equipment_label": "balance", "state": "visible"}])
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])

    result = validate_evidence_adapters(tmp_path)

    panel = result["adapters"]["panel_ocr"]
    assert panel["semantic_issue_count"] == 1
    assert panel["semantic_summary"]["missing_fields"] == 1
    assert result["summary"]["semantic_issue_count"] == 1
    assert any(issue["code"] == "semantic_missing_fields" for issue in panel["issues"])


def test_validate_evidence_adapters_accepts_nested_measurement_semantics(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(
        metadata / "panel_ocr.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "measurement": {"equipment_label": "balance", "display_text": "0.124 g"},
            }
        ],
    )
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])

    result = validate_evidence_adapters(tmp_path)

    assert result["adapters"]["panel_ocr"]["semantic_issue_count"] == 0
    assert result["adapters"]["panel_ocr"]["linked_segment_ids"] == ["seg_1"]


def test_validate_evidence_adapters_allows_nearby_panel_point_event(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "segment_id": "seg_1",
                "micro_segment_id": "micro_1",
                "start_sec": 9.0,
                "end_sec": 10.2,
                "action_type": "recording_or_reading",
                "primary_object": "balance",
            }
        ],
    )
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(
        metadata / "panel_ocr.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "time_sec": 10.7,
                "equipment_label": "balance",
                "display_text": "0.124 g",
            }
        ],
    )
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])

    result = validate_evidence_adapters(tmp_path)

    panel = result["adapters"]["panel_ocr"]
    assert panel["semantic_issue_count"] == 0
    assert panel["linked_micro_segment_ids"] == ["micro_1"]


def test_validate_evidence_adapters_allows_point_event_on_segment_boundary(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(metadata / "panel_ocr.jsonl", [])
    write_jsonl(
        metadata / "liquid_state.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "micro_segment_id": "micro_1",
                "start_sec": 10.0,
                "end_sec": 10.0,
                "measurement": {"container_label": "tube", "liquid_state": "meniscus_visible"},
            }
        ],
    )
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "segment_id": "seg_1",
                "micro_segment_id": "micro_1",
                "start_sec": 8.0,
                "end_sec": 10.0,
                "action_type": "liquid_state_change",
                "primary_object": "tube",
            }
        ],
    )
    write_jsonl(metadata / "container_state.jsonl", [])

    result = validate_evidence_adapters(tmp_path)

    liquid = result["adapters"]["liquid_state"]
    assert liquid["semantic_issue_count"] == 0
    assert liquid["linked_micro_segment_ids"] == ["micro_1"]


def test_validate_evidence_adapters_ignores_unlinked_candidate_context_signal(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(
        metadata / "panel_ocr.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "time_sec": 90.0,
                "equipment_label": "balance",
                "state": "interaction_candidate",
                "event_type": "equipment_panel_interaction_candidate",
                "confirmation_level": "candidate",
                "measurement": {"candidate_type": "equipment_panel_candidate"},
            }
        ],
    )
    write_jsonl(metadata / "liquid_state.jsonl", [])
    write_jsonl(metadata / "container_state.jsonl", [])

    result = validate_evidence_adapters(tmp_path)

    panel = result["adapters"]["panel_ocr"]
    assert panel["semantic_issue_count"] == 0
    assert panel["semantic_summary"]["time_mismatch"] == 0


def test_validate_evidence_adapters_classifies_time_and_action_semantic_failures(tmp_path: Path) -> None:
    _base_session(tmp_path)
    metadata = tmp_path / "metadata"
    write_jsonl(metadata / "object_tracks.jsonl", [])
    write_jsonl(metadata / "panel_ocr.jsonl", [])
    write_jsonl(
        metadata / "liquid_state.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "start_sec": 11.0,
                "end_sec": 11.2,
                "measurement": {"container_label": "tube", "liquid_state": "meniscus_visible"},
            }
        ],
    )
    write_jsonl(
        metadata / "container_state.jsonl",
        [
            {
                "session_id": "sess_1",
                "view": "first_person",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "container_label": "tube",
                "state": "open",
                "action_type": "open",
                "confirmation_level": "confirmed",
            }
        ],
    )

    result = validate_evidence_adapters(tmp_path)

    liquid = result["adapters"]["liquid_state"]
    container = result["adapters"]["container_state"]
    assert liquid["semantic_summary"]["time_mismatch"] == 1
    assert container["semantic_summary"]["action_mismatch"] == 1
    assert any(issue["semantic_category"] == "time_mismatch" for issue in liquid["issues"])
    assert any(issue["semantic_category"] == "action_mismatch" for issue in container["issues"])
