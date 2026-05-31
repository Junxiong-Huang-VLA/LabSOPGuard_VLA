from __future__ import annotations

import copy
from typing import Any

from key_action_indexer.video_memory_workflow import (
    build_workflow_context_reasoning,
    workflow_reasoning_fingerprint,
)


def _source_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bundles = []
    ledgers = []
    dates = ["2026-05-23", "2026-05-24", "2026-05-25"]
    for index, date_value in enumerate(dates, start=1):
        bundle_id = f"bundle_weigh_{index}"
        material_id = f"material_weigh_{index}"
        ledger_id = f"ledger_weigh_{index}"
        bundles.append(
            {
                "bundle_id": bundle_id,
                "material_ids": [material_id],
                "sha256s": [f"sha256_weigh_{index}"],
                "micro_segment_ids": [f"micro_weigh_{index}"],
                "timestamp": {"start_sec": 10.0 + index, "end_sec": 16.0 + index},
                "views": ["first_person", "third_person"],
                "keyframes": [f"package://pkg/keyframes/weigh_{index}.jpg"],
                "keyclips": [f"package://pkg/clips/weigh_{index}.mp4"],
            }
        )
        ledgers.append(
            {
                "ledger_event_id": ledger_id,
                "date": date_value,
                "session_id": f"session_{index}",
                "experiment_id": "do_not_promote_experiment_name",
                "material_ids": [material_id],
                "sha256s": [f"sha256_weigh_{index}"],
                "micro_segment_id": f"micro_weigh_{index}",
                "micro_segment_ids": [f"micro_weigh_{index}"],
                "keyframe_refs": [f"package://pkg/keyframes/weigh_{index}.jpg"],
                "keyclip_refs": [f"package://pkg/clips/weigh_{index}.mp4"],
                "timestamp": {"start_sec": 10.0 + index, "end_sec": 16.0 + index},
                "time_range": {"start_sec": 10.0 + index, "end_sec": 16.0 + index},
                "canonical_action_type": "weigh_material",
                "action_name": "weigh material",
                "primary_object": "balance",
                "detected_objects": ["balance", "tube", "spatula", "gloved_hand"],
                "evidence_bundle_ids": [bundle_id],
                "confidence": 0.82,
            }
        )
    clusters = [
        {
            "cluster_id": "cluster_balance_weighing",
            "status": "primary",
            "confirmation_status": "auto_inferred",
            "canonical_actions": ["weigh_material"],
            "key_objects": ["balance", "tube", "spatula"],
            "key_instruments": ["balance"],
            "related_dates": dates,
            "related_sessions": [f"session_{index}" for index in range(1, 4)],
            "related_experiments": ["do_not_promote_experiment_name"],
            "ledger_event_ids": [row["ledger_event_id"] for row in ledgers],
            "evidence_bundle_ids": [row["evidence_bundle_ids"][0] for row in ledgers],
            "material_ids": [bundle["material_ids"][0] for bundle in bundles],
            "occurrence_count": 3,
            "day_count": 3,
            "view_coverage": ["first_person", "third_person"],
            "confidence": 0.82,
            "cluster_signature": {
                "sequence_signature": ["weigh_material", "balance"],
            },
            "unresolved_questions": ["Need human SOP, sample, and project context."],
        }
    ]
    return clusters, ledgers, bundles


def _assert_complete_trace(row: dict[str, Any]) -> None:
    trace = row["evidence_trace"]
    assert trace["cluster_id"]
    assert trace["ledger_event_id"]
    assert trace["bundle_id"]
    assert trace["material_id"]
    assert trace["sha256s"]
    assert trace["micro_segment_ids"]
    assert trace["keyframe_refs"]
    assert trace["keyclip_refs"]
    assert trace["timestamps"]
    assert trace["trace_complete"] is True


def test_v2_reasoning_finds_workflow_instrument_and_project_hints_without_invented_names() -> None:
    clusters, ledgers, bundles = _source_rows()

    reasoning = build_workflow_context_reasoning(
        clusters=clusters,
        ledger_events=ledgers,
        bundles=bundles,
        human_feedback_entries=[],
    )

    assert reasoning["schema_version"] == "video_memory.workflow_reasoning.v2"
    assert reasoning["workflow_patterns"]
    assert reasoning["instrument_usage_patterns"]
    assert reasoning["project_hints"]
    assert reasoning["project_or_context_hints"] == reasoning["project_hints"]
    assert not reasoning["human_confirmed_contexts"]
    assert "do_not_promote_experiment_name" not in repr(reasoning["project_hints"])
    assert any("SOP" in question or "project" in question for question in reasoning["unresolved_questions"])

    for collection_name in ("workflow_patterns", "instrument_usage_patterns", "project_hints"):
        for row in reasoning[collection_name]:
            assert 0.0 < row["confidence"] <= 1.0
            _assert_complete_trace(row)


def test_v3_human_context_generates_candidate_only_sop_reasoning() -> None:
    clusters, ledgers, bundles = _source_rows()
    feedback = [
        {
            "feedback_id": "feedback_confirm_context",
            "target_type": "cluster",
            "target_id": "cluster_balance_weighing",
            "feedback_type": "confirm",
            "context_fields": {
                "sop_name": "SOP-42 Balance Weighing",
                "sample_name": "Sample A",
                "project_name": "Project Iris",
            },
            "created_at": "2026-05-25T09:00:00+00:00",
            "user_id": "reviewer",
        }
    ]

    reasoning = build_workflow_context_reasoning(
        clusters=clusters,
        ledger_events=ledgers,
        bundles=bundles,
        human_feedback_entries=feedback,
    )

    context = reasoning["human_confirmed_contexts"][0]
    assert context["confirmation_status"] == "human_confirmed"
    assert context["sop_name"] == "SOP-42 Balance Weighing"
    assert context["sample_name"] == "Sample A"
    assert context["project_name"] == "Project Iris"
    _assert_complete_trace(context)

    candidate_collections = (
        "step_reasoning_candidates",
        "process_completion_candidates",
        "rule_candidates",
        "reminder_candidates",
    )
    for collection_name in candidate_collections:
        assert reasoning[collection_name], collection_name
        for candidate in reasoning[collection_name]:
            assert candidate["candidate_status"] == "candidate"
            assert candidate.get("compliance_fact_status", "not_evaluated") in {"not_evaluated", "not_asserted"}
            assert candidate.get("enforcement_status", "not_enforced") != "enforced"
            assert candidate.get("force_real_time_alert", False) is False
            _assert_complete_trace(candidate)


def test_workflow_reasoning_fingerprint_ignores_audit_timestamps_but_tracks_context() -> None:
    clusters, ledgers, bundles = _source_rows()
    reasoning = build_workflow_context_reasoning(
        clusters=clusters,
        ledger_events=ledgers,
        bundles=bundles,
        human_feedback_entries=[
            {
                "feedback_id": "feedback_a",
                "target_type": "cluster",
                "target_id": "cluster_balance_weighing",
                "feedback_type": "confirm",
                "context_fields": {"project_name": "Project Iris"},
                "created_at": "2026-05-25T09:00:00+00:00",
            }
        ],
    )
    same_reasoning = copy.deepcopy(reasoning)
    same_reasoning["human_confirmed_contexts"][0]["source_feedback_id"] = "feedback_b"
    same_reasoning["human_confirmed_contexts"][0]["confirmed_at"] = "2026-05-25T10:00:00+00:00"
    same_reasoning["human_confirmed_contexts"][0]["note"] = "same semantic context"

    changed_reasoning = copy.deepcopy(reasoning)
    changed_reasoning["human_confirmed_contexts"][0]["project_name"] = "Project Lark"

    assert workflow_reasoning_fingerprint(reasoning) == workflow_reasoning_fingerprint(same_reasoning)
    assert workflow_reasoning_fingerprint(reasoning) != workflow_reasoning_fingerprint(changed_reasoning)
