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
    assert result["summary"]["semantic_issue_count"] == 1
    assert any(issue["code"] == "semantic_support_missing" for issue in panel["issues"])
