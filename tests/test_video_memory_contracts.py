from __future__ import annotations

import copy
import importlib
from dataclasses import asdict, is_dataclass
from typing import Any

import pytest


NOW = "2026-05-25T10:00:00+08:00"
OLD = "2026-04-20T10:00:00+08:00"


def _video_memory_module() -> Any:
    try:
        return importlib.import_module("key_action_indexer.video_memory")
    except ModuleNotFoundError as exc:
        if exc.name == "key_action_indexer.video_memory":
            pytest.skip("30-Day Video Memory module is not implemented yet; contract tests are ready for the new module.")
        raise


def _required(name: str) -> Any:
    video_memory = _video_memory_module()
    if not hasattr(video_memory, name):
        pytest.fail(f"key_action_indexer.video_memory must expose {name} for the 30-Day Video Memory T1-T14 contract")
    return getattr(video_memory, name)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_json_dict"):
        return _jsonable(value.to_json_dict())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_list(value: Any) -> list[dict[str, Any]]:
    data = _jsonable(value)
    if isinstance(data, dict) and "clusters" in data:
        data = data["clusters"]
    assert isinstance(data, list)
    return data


def _build_evidence_item(payload: dict[str, Any]) -> Any:
    EvidenceItem = _required("EvidenceItem")
    if hasattr(EvidenceItem, "from_dict"):
        return EvidenceItem.from_dict(payload)
    return EvidenceItem(**payload)


def _evidence_item(
    evidence_id: str,
    *,
    cluster_id: str = "cluster_pipette_tube_contact",
    observed_at: str = NOW,
    action: str = "hand_object_interaction",
    primary_object: str = "pipette",
    secondary_objects: list[str] | None = None,
    views: list[str] | None = None,
    score: float = 0.89,
) -> dict[str, Any]:
    secondary_objects = secondary_objects if secondary_objects is not None else ["tube"]
    views = views if views is not None else ["front", "side"]
    view_payloads = []
    for index, view in enumerate(views, start=1):
        view_payloads.append(
            {
                "view_id": view,
                "source_video_id": f"video_{view}_001",
                "local_start_sec": 124.0 + index,
                "local_end_sec": 127.0 + index,
                "clip_uri": f"package://pkg_30d/clips/{evidence_id}_{view}.mp4",
                "keyframe_uri": f"package://pkg_30d/keyframes/{evidence_id}_{view}.jpg",
                "frame_ids": [f"{evidence_id}_{view}_f001", f"{evidence_id}_{view}_f002"],
                "yolo_refs": [
                    {
                        "frame_id": f"{evidence_id}_{view}_f001",
                        "label": primary_object,
                        "bbox": [238, 112, 340, 126],
                        "confidence": 0.91,
                    }
                ],
            }
        )
    return {
        "schema_version": "video_memory.evidence_item.v1",
        "evidence_id": evidence_id,
        "cluster_id": cluster_id,
        "session_id": "session_30d_001",
        "experiment_id": "exp_30d_001",
        "material_id": f"material_{evidence_id}",
        "sha256": f"{evidence_id}_sha256",
        "created_at": observed_at,
        "observed_at": observed_at,
        "time_range": {
            "global_start_time": observed_at,
            "global_end_time": observed_at,
            "start_sec": 124.0,
            "end_sec": 128.0,
        },
        "action": {
            "type": action,
            "primary_object": primary_object,
            "secondary_objects": secondary_objects,
            "description": "gloved hand brings pipette into contact with tube",
        },
        "physical_evidence": {
            "hand_object_contact": action == "hand_object_interaction",
            "object_state_change": action == "object_move",
            "confidence": score,
            "uncertainty_reasons": [],
        },
        "views": view_payloads,
        "vlm": {
            "cache_key": f"vlm_cache_{evidence_id}",
            "model": "qwen2.5-vl",
            "prompt_version": "physical-evidence-v1",
            "summary": "Pipette tip is visibly contacting the tube opening.",
            "confidence": 0.84,
        },
        "retrieval": {
            "index_text": "pipette tube contact hand object interaction",
            "embedding_id": f"emb_{evidence_id}",
            "score": score,
        },
        "material_reference": {
            "material_id": f"material_{evidence_id}",
            "sha256": f"{evidence_id}_sha256",
        },
        "trace": {
            "decision_path": "yolo.micro_segment.vlm",
            "decision_trace": ["yolo_candidate", "micro_segment_confirmed", "vlm_supported"],
            "source_micro_segment_ids": ["micro_001"],
        },
        "retention": {
            "retention_days": 30,
            "expires_at": "2026-06-24T10:00:00+08:00",
        },
    }


def _snapshot_payload() -> dict[str, Any]:
    evidence_items = [
        _evidence_item("ev_touch_front_side"),
        _evidence_item("ev_touch_followup", score=0.81),
    ]
    return {
        "schema_version": "video_memory.partial_snapshot.v1",
        "snapshot_id": "snapshot_20260525_100000_partial",
        "snapshot_kind": "partial",
        "generated_at": NOW,
        "is_partial": True,
        "coverage": {
            "retention_days": 30,
            "start_time": "2026-04-25T10:00:00+08:00",
            "end_time": NOW,
            "truncation_reason": "max_items",
        },
        "evidence_items": evidence_items,
        "clusters": [
            {
                "schema_version": "video_memory.cluster.v1",
                "cluster_id": "cluster_pipette_tube_contact",
                "lifecycle_state": "promoted",
                "score": 0.92,
                "evidence_item_ids": [item["evidence_id"] for item in evidence_items],
                "score_reasons": ["multi_view_support", "vlm_supported", "accepted_feedback"],
                "last_observed_at": NOW,
            }
        ],
    }


def test_evidence_item_contract_preserves_dual_view_traceable_payload() -> None:
    validate_evidence_item = _required("validate_evidence_item")
    payload = _evidence_item("ev_touch_front_side")

    normalized = _jsonable(validate_evidence_item(_build_evidence_item(payload)))

    assert normalized["schema_version"] == "video_memory.evidence_item.v1"
    assert normalized["evidence_id"] == "ev_touch_front_side"
    assert normalized["cluster_id"] == "cluster_pipette_tube_contact"
    assert normalized["action"]["type"] == "hand_object_interaction"
    assert normalized["action"]["primary_object"] == "pipette"
    assert normalized["action"]["secondary_objects"] == ["tube"]
    assert normalized["physical_evidence"]["hand_object_contact"] is True
    assert normalized["retrieval"]["index_text"]
    assert normalized["trace"]["decision_path"] == "yolo.micro_segment.vlm"
    assert set(normalized["trace"]["decision_trace"]) >= {"yolo_candidate", "micro_segment_confirmed", "vlm_supported"}
    assert {view["view_id"] for view in normalized["views"]} == {"front", "side"}
    assert all(view["clip_uri"].startswith("package://") for view in normalized["views"])
    assert all(view["yolo_refs"] for view in normalized["views"])


def test_vlm_cache_uses_stable_content_key_and_records_cache_trace(tmp_path) -> None:
    VLMCache = _required("VLMCache")
    build_vlm_cache_key = _required("build_vlm_cache_key")
    evidence_payload = _evidence_item("ev_touch_front_side")

    key_a = build_vlm_cache_key(
        model="qwen2.5-vl",
        prompt_version="physical-evidence-v1",
        evidence_item=evidence_payload,
        image_refs=["package://pkg_30d/keyframes/ev_touch_front_side_front.jpg"],
    )
    key_b = build_vlm_cache_key(
        prompt_version="physical-evidence-v1",
        image_refs=["package://pkg_30d/keyframes/ev_touch_front_side_front.jpg"],
        evidence_item=copy.deepcopy(evidence_payload),
        model="qwen2.5-vl",
    )

    assert isinstance(key_a, str)
    assert len(key_a) >= 16
    assert key_a == key_b

    cache = VLMCache(tmp_path / "vlm_cache")
    cache.put(
        key_a,
        {
            "schema_version": "video_memory.vlm_cache_entry.v1",
            "description": "Pipette tip contacts the tube.",
            "physical_action": "pipette_tube_contact",
            "confirmed_objects": ["pipette", "tube", "gloved_hand"],
            "confidence": 0.84,
        },
        metadata={
            "model": "qwen2.5-vl",
            "prompt_version": "physical-evidence-v1",
            "source_evidence_ids": ["ev_touch_front_side"],
        },
    )

    hit = _jsonable(cache.get(key_a))
    miss = _jsonable(cache.get("missing-cache-key"))

    assert hit["status"] == "hit"
    assert hit["cache_key"] == key_a
    assert hit["response"]["physical_action"] == "pipette_tube_contact"
    assert hit["metadata"]["source_evidence_ids"] == ["ev_touch_front_side"]
    assert miss["status"] == "miss"


def test_cluster_scoring_and_lifecycle_promotes_supported_recent_clusters() -> None:
    score_evidence_clusters = _required("score_evidence_clusters")
    update_cluster_lifecycle = _required("update_cluster_lifecycle")
    evidence_items = [
        _evidence_item("ev_touch_front", views=["front"], score=0.86),
        _evidence_item("ev_touch_side", views=["side"], score=0.88),
        _evidence_item(
            "ev_old_bottle_move",
            cluster_id="cluster_old_bottle_move",
            observed_at=OLD,
            action="object_move",
            primary_object="reagent_bottle",
            secondary_objects=[],
            views=["front"],
            score=0.63,
        ),
    ]

    scored = score_evidence_clusters(evidence_items=evidence_items, now=NOW)
    updated = update_cluster_lifecycle(
        clusters=scored,
        feedback_events=[
            {
                "feedback_id": "fb_accept_contact",
                "target_type": "cluster",
                "target_id": "cluster_pipette_tube_contact",
                "feedback_type": "accepted",
                "weight": 0.15,
                "created_at": NOW,
            },
            {
                "feedback_id": "fb_reject_old_bottle",
                "target_type": "evidence_item",
                "target_id": "ev_old_bottle_move",
                "feedback_type": "rejected",
                "weight": -0.2,
                "created_at": NOW,
            },
        ],
        now=NOW,
        policy={"promote_score": 0.75, "stale_after_days": 14, "archive_after_days": 30},
    )
    clusters = {cluster["cluster_id"]: cluster for cluster in _as_list(updated)}

    promoted = clusters["cluster_pipette_tube_contact"]
    archived = clusters["cluster_old_bottle_move"]
    assert promoted["lifecycle_state"] == "promoted"
    assert promoted["score"] > archived["score"]
    assert "multi_view_support" in promoted["score_reasons"]
    assert "accepted_feedback" in promoted["score_reasons"]
    assert set(promoted["evidence_item_ids"]) == {"ev_touch_front", "ev_touch_side"}
    assert archived["lifecycle_state"] == "archived"
    assert archived["archive_reason"] == "older_than_retention_window"


def test_partial_snapshot_marks_truncation_and_keeps_package_relative_assets() -> None:
    build_partial_snapshot = _required("build_partial_snapshot")
    evidence_items = [
        _evidence_item("ev_touch_front_side"),
        _evidence_item("ev_touch_followup", score=0.81),
        _evidence_item("ev_expired", observed_at=OLD, score=0.7),
    ]
    clusters = [
        {
            "cluster_id": "cluster_pipette_tube_contact",
            "lifecycle_state": "promoted",
            "score": 0.92,
            "evidence_item_ids": ["ev_touch_front_side", "ev_touch_followup"],
        }
    ]

    snapshot = _jsonable(
        build_partial_snapshot(
            evidence_items=evidence_items,
            clusters=clusters,
            now=NOW,
            retention_days=30,
            max_items=2,
        )
    )

    assert snapshot["schema_version"] == "video_memory.partial_snapshot.v1"
    assert snapshot["snapshot_kind"] == "partial"
    assert snapshot["is_partial"] is True
    assert snapshot["coverage"]["retention_days"] == 30
    assert snapshot["coverage"]["truncation_reason"] == "max_items"
    assert snapshot["counts"]["source_evidence_items"] == 3
    assert snapshot["counts"]["included_evidence_items"] == 2
    assert {item["evidence_id"] for item in snapshot["evidence_items"]} == {"ev_touch_front_side", "ev_touch_followup"}
    assert any(skipped["evidence_id"] == "ev_expired" for skipped in snapshot["skipped_items"])
    assert "D:\\" not in repr(snapshot)


def test_query_answer_requires_claim_level_evidence_trace() -> None:
    answer_video_memory_query = _required("answer_video_memory_query")

    answer = _jsonable(
        answer_video_memory_query(
            query="Did the pipette touch the tube during the last 30 days?",
            snapshot=_snapshot_payload(),
            top_k=2,
        )
    )

    assert answer["schema_version"] == "video_memory.query_answer.v1"
    assert answer["answer"]["status"] == "supported"
    assert answer["answer"]["text"]
    trace = answer["evidence_trace"]
    assert trace["query"] == "Did the pipette touch the tube during the last 30 days?"
    assert trace["retrieved_evidence"]
    assert trace["claims"]
    for claim in trace["claims"]:
        assert claim["support"] == "supported"
        assert claim["claim_type"] != "fact"
        assert claim["fact_status"] == "not_a_strong_fact_without_evidence_bundle"
        assert claim["has_evidence_bundle"] is False
        assert claim["evidence_item_ids"]
        assert claim["cluster_ids"] == ["cluster_pipette_tube_contact"]
        for key in (
            "evidence_bundle_id",
            "material_id",
            "session_id",
            "experiment_id",
            "micro_segment_id",
            "timestamp",
            "keyframe",
            "keyclip",
            "confidence",
            "human_confirmation_status",
        ):
            assert key in claim
        assert claim["session_id"] == "session_30d_001"
        assert claim["experiment_id"] == "exp_30d_001"
        assert claim["material_id"].startswith("material_ev_touch_")
        assert claim["sha256"].endswith("_sha256")
        assert claim["micro_segment_id"] == "micro_001"
        assert claim["keyframe"].startswith("package://")
        assert claim["keyclip"].startswith("package://")
    for evidence in trace["retrieved_evidence"]:
        assert evidence["evidence_id"].startswith("ev_touch_")
        assert evidence["time_range"]["start_sec"] <= evidence["time_range"]["end_sec"]
        assert evidence["score_breakdown"]["final_score"] > 0
        assert evidence["vlm_cache_key"].startswith("vlm_cache_")
    assert answer["partial_window"] is True
    assert answer["is_full_30_day_memory"] is False


def test_feedback_update_job_plans_auditable_dry_run_without_mutating_snapshot() -> None:
    run_feedback_update_job = _required("run_feedback_update_job")
    snapshot = _snapshot_payload()
    before = copy.deepcopy(snapshot)

    result = _jsonable(
        run_feedback_update_job(
            snapshot=snapshot,
            feedback_events=[
                {
                    "feedback_id": "fb_reject_side",
                    "target_type": "evidence_item",
                    "target_id": "ev_touch_followup",
                    "feedback_type": "rejected",
                    "correction": {
                        "reason": "side view keyframe was a near miss, not contact",
                        "label": "near_miss",
                    },
                    "created_at": NOW,
                    "actor": "reviewer",
                },
                {
                    "feedback_id": "fb_accept_cluster",
                    "target_type": "cluster",
                    "target_id": "cluster_pipette_tube_contact",
                    "feedback_type": "accepted",
                    "created_at": NOW,
                    "actor": "reviewer",
                },
            ],
            now=NOW,
            dry_run=True,
        )
    )

    assert snapshot == before
    assert result["schema_version"] == "video_memory.feedback_update_job.v1"
    assert result["dry_run"] is True
    assert result["job_status"] == "planned"
    operation_types = {operation["operation"] for operation in result["operations"]}
    assert {"deprioritize_evidence_item", "rescore_cluster", "write_feedback_audit"} <= operation_types
    for operation in result["operations"]:
        assert operation["target_id"]
        assert operation["audit"]["feedback_ids"]
        assert operation["audit"]["created_at"] == NOW
