from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.material_references import apply_formal_dual_view_material_publish_gate
from key_action_indexer.schemas import write_jsonl


def _write_reliable_timeline(session: Path) -> None:
    metadata = session / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "pre_coarse_timeline_alignment.json").write_text(
        json.dumps(
            {
                "status": "aligned",
                "alignment_status": "aligned",
                "alignment_reliable_for_dual_view_pairing": True,
                "formal_results_allowed": True,
                "video_memory_allowed": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _evidence(view: str) -> list[dict]:
    return [
        {
            "view": view,
            "time_sec": 46.4 + index * 0.2,
            "local_time_sec": 46.4 + index * 0.2,
            "primary_object": "balance",
            "detections": [
                {"label": "gloved_hand", "confidence": 0.82, "bbox": [110, 120, 190, 230]},
                {"label": "balance", "confidence": 0.84, "bbox": [165, 155, 335, 315]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "balance",
                    "score": 0.82,
                    "hand_bbox": [110, 120, 190, 230],
                    "object_bbox": [165, 155, 335, 315],
                    "iou": 0.08,
                }
            ],
        }
        for index in range(2)
    ]


def _physical_diagnostics(view: str) -> dict:
    return {
        "primary_object": "balance",
        "total_evidence_count": 2,
        "valid_evidence_count": 2,
        "evidence_by_view": {view: 2},
        "valid_evidence_by_view": {view: 2},
        "invalid_reason_counts": {},
    }


def _write_formal_dual_event(session: Path, *, dual_event_id: str = "dual_event_000123") -> None:
    write_jsonl(
        session / "metadata" / "dual_view_action_events.jsonl",
        [
            {
                "dual_event_id": dual_event_id,
                "status": "matched_dual_view",
                "formal_event_promoted": True,
                "micro_segment_ids": ["micro_bottle"],
                "first_evidence_id": "first_evidence_001",
                "third_evidence_id": "third_evidence_001",
                "views": {
                    "first_person": {
                        "evidence_id": "first_evidence_001",
                        "view": "first_person",
                        "frame_count": 2,
                    },
                    "third_person": {
                        "evidence_id": "third_evidence_001",
                        "view": "third_person",
                        "frame_count": 2,
                    },
                },
            }
        ],
    )


def _write_blocked_action_alignment_summary(session: Path) -> None:
    metadata = session / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "dual_view_action_alignment_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "dual_view_action_alignment.v1",
                "dual_view_action_event_count": 1,
                "formal_event_count": 0,
                "formal_results_allowed": False,
                "decision": "no_dual_view_action_events",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _complete_material_rows() -> list[dict]:
    rows: list[dict] = []
    for view in ("first_person", "third_person"):
        for asset_kind in ("keyframe", "keyclip"):
            rows.append(
                {
                    "candidate_id": f"{view}-{asset_kind}",
                    "material_group_id": "group_bottle",
                    "dual_event_id": "dual_event_000123",
                    "micro_segment_id": "micro_bottle",
                    "canonical_action_type": "hand-bottle",
                    "view": view,
                    "asset_kind": asset_kind,
                    "exists": True,
                    "primary_object": "balance",
                    "source_yolo_evidence": _evidence(view),
                    "physical_evidence_diagnostics": _physical_diagnostics(view),
                    "evidence_chain": {
                        "camera_view": view,
                        "micro_segment_id": "micro_bottle",
                    },
                }
            )
    return rows


def test_publish_gate_rejects_complete_assets_when_no_formal_dual_event(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    write_jsonl(session / "metadata" / "dual_view_action_events.jsonl", [])

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, _complete_material_rows())

    assert kept == []
    assert len(rejected) == 4
    assert {row["suppression_reason"] for row in rejected} == {"missing_explicit_dual_view_action_event"}


def test_publish_gate_binds_complete_assets_to_formal_dual_event(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    _write_formal_dual_event(session)

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, _complete_material_rows())

    assert rejected == []
    assert len(kept) == 4
    assert {row["dual_event_id"] for row in kept} == {"dual_event_000123"}
    assert {row["dual_event_binding_source"] for row in kept} == {"explicit_dual_event_id"}
    assert all(row["formal_publish_gate"]["status"] == "passed" for row in kept)


def test_publish_gate_blocks_stale_event_rows_when_action_alignment_forbids_formal_results(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    _write_formal_dual_event(session)
    _write_blocked_action_alignment_summary(session)

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, _complete_material_rows())

    assert kept == []
    assert len(rejected) == 4
    assert {row["blocked_reason"] for row in rejected} == {"formal_results_not_allowed"}
    assert all(row["formal_dual_view_action_gate"]["formal_event_count"] == 0 for row in rejected)


def test_publish_gate_rejects_derived_complete_group_binding(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    _write_formal_dual_event(session, dual_event_id="dual_event_derived")
    rows = _complete_material_rows()
    for row in rows:
        row["dual_event_id"] = "dual_event_derived"
        row["dual_event_binding_source"] = "derived_from_complete_dual_view_material_group"

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, rows)

    assert kept == []
    assert len(rejected) == 4
    assert {row["suppression_reason"] for row in rejected} == {"missing_explicit_dual_view_action_event"}


def test_publish_gate_rejects_group_missing_keyclip(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    _write_formal_dual_event(session)
    rows = [
        row
        for row in _complete_material_rows()
        if not (row["view"] == "third_person" and row["asset_kind"] == "keyclip")
    ]

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, rows)

    assert kept == []
    assert len(rejected) == 3
    assert {row["suppression_reason"] for row in rejected} == {"incomplete_dual_view_material_group"}
    assert all("third_person:keyclip" in row["missing_dual_view_assets"] for row in rejected)


def test_publish_gate_rejects_material_bound_to_different_event_micro(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    _write_formal_dual_event(session, dual_event_id="dual_event_000123")
    rows = _complete_material_rows()
    for row in rows:
        row["micro_segment_id"] = "micro_other"
        row["evidence_chain"]["micro_segment_id"] = "micro_other"

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, rows)

    assert kept == []
    assert len(rejected) == 4
    assert {row["suppression_reason"] for row in rejected} == {"missing_explicit_dual_view_action_event"}


def test_publish_gate_rejects_cross_view_physical_evidence(tmp_path: Path) -> None:
    session = tmp_path / "key_action_index"
    _write_reliable_timeline(session)
    _write_formal_dual_event(session)
    rows = _complete_material_rows()
    for row in rows:
        if row["view"] == "first_person":
            row["source_yolo_evidence"] = _evidence("third_person")
            row["physical_evidence_diagnostics"] = _physical_diagnostics("third_person")

    kept, rejected = apply_formal_dual_view_material_publish_gate(session, rows)

    assert kept == []
    assert len(rejected) == 4
    assert {row["suppression_reason"] for row in rejected} == {"missing_same_view_physical_evidence"}
    first_rejections = [row for row in rejected if row["view"] == "first_person"]
    assert all(
        row["same_view_physical_evidence_gate"]["reason"]
        == "source_yolo_evidence_missing_same_view_valid_physical_evidence"
        for row in first_rejections
    )
