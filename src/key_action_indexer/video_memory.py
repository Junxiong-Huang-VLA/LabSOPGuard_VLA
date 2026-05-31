from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .material_library_store import (
    default_material_library_root,
    global_material_library_db_path,
    sync_material_library,
)
from .schemas import write_jsonl
from .video_memory_vlm import (
    SUPPORTED_QWEN_MODELS,
    VLMResultCache,
    build_vlm_result_cache_key,
    enhance_bundles_sync,
    enhance_items_sync,
    reusable_vlm_sources,
)
from .video_memory_workflow import (
    build_workflow_context_reasoning,
    workflow_reasoning_fingerprint,
)


SCHEMA_VERSION = "video_memory.v1"
EVIDENCE_ITEM_SCHEMA_VERSION = "video_memory.evidence_item.v1"
PROMPT_VERSION = "video_memory_prompt.v1"
VLM_MODEL_VERSION = "metadata_grounded_vlm_stub.v1"
VLM_MODE_OFFLINE = "offline_metadata"
VLM_MODE_REUSE_EXISTING = "reuse_existing_vlm"
VLM_MODE_REAL_QWEN_ASYNC = "real_qwen_async"
DEFAULT_ITEM_VLM_MODEL = "qwen3.5-flash"
DEFAULT_BUNDLE_VLM_MODEL = "qwen3.5-plus"
CLUSTER_ALGORITHM_VERSION = "mvp_signature_weighted_merge.v1"
MEMORY_DB_NAME = "video_memory.sqlite"
MEMORY_INDEX_DIR = "memory_index"
WINDOW_DAYS = 30
_BACKGROUND_REBUILD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="video-memory-rebuild")
_BACKGROUND_REBUILD_JOBS: dict[str, Future[dict[str, Any]]] = {}

JSONL_FILES = {
    "evidence_items": "evidence_items.jsonl",
    "vlm_item_results": "vlm_item_results.jsonl",
    "vlm_bundle_results": "vlm_bundle_results.jsonl",
    "evidence_bundles": "evidence_bundles.jsonl",
    "daily_event_ledgers": "daily_event_ledgers.jsonl",
    "memory_clusters": "memory_clusters.jsonl",
    "memory_snapshots": "memory_snapshots.jsonl",
    "memory_build_jobs": "memory_build_jobs.jsonl",
    "human_feedback_entries": "human_feedback_entries.jsonl",
    "vlm_result_cache": "vlm_result_cache.jsonl",
}

SIGNATURE_WEIGHTS = {
    "action_signature": 0.35,
    "object_signature": 0.25,
    "instrument_signature": 0.20,
    "visual_semantic_signature": 0.10,
    "temporal_signature": 0.07,
    "sequence_signature": 0.03,
}

AUTO_MERGE_THRESHOLD = 0.78
REVIEW_MERGE_THRESHOLD = 0.65
MIN_ACTION_SIMILARITY = 0.55
MIN_OBJECT_INSTRUMENT_SIMILARITY = 0.50
INSTRUMENT_LABELS = {
    "balance",
    "scale",
    "panel",
    "display",
    "pipette",
    "beaker",
    "tube",
    "tube_rack",
    "rack",
    "bottle",
    "reagent_bottle",
    "weighing_paper",
    "paper",
    "spatula",
    "container",
    "flask",
    "vial",
}


def memory_index_root(library_root: str | Path | None = None) -> Path:
    root = Path(library_root) if library_root is not None else default_material_library_root()
    if root.name.lower() == MEMORY_INDEX_DIR:
        return root
    if root.name.lower() == "material_references":
        root = root.parent
    return root / MEMORY_INDEX_DIR


def video_memory_db_path(
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> Path:
    if sqlite_path is not None:
        return Path(sqlite_path)
    return memory_index_root(library_root) / MEMORY_DB_NAME


def start_video_memory_rebuild_background(**build_kwargs: Any) -> dict[str, Any]:
    """Start a Video Memory rebuild in the current process for API/server integrations."""

    job_id = _stable_id("video-memory-bg-job", {"kwargs": _strip_unserializable_client(build_kwargs), "created_at": _now_iso()})
    future = _BACKGROUND_REBUILD_EXECUTOR.submit(build_video_memory, **build_kwargs)
    _BACKGROUND_REBUILD_JOBS[job_id] = future
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "job_status": "running",
        "background": True,
        "created_at": _now_iso(),
    }


def get_video_memory_rebuild_background_status(job_id: str) -> dict[str, Any]:
    future = _BACKGROUND_REBUILD_JOBS.get(str(job_id))
    if future is None:
        return {"schema_version": SCHEMA_VERSION, "job_id": job_id, "job_status": "missing", "background": True}
    if future.running():
        return {"schema_version": SCHEMA_VERSION, "job_id": job_id, "job_status": "running", "background": True}
    if not future.done():
        return {"schema_version": SCHEMA_VERSION, "job_id": job_id, "job_status": "pending", "background": True}
    exc = future.exception()
    if exc is not None:
        return {"schema_version": SCHEMA_VERSION, "job_id": job_id, "job_status": "failed", "background": True, "error_message": str(exc)}
    result = future.result()
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job_id,
        "job_status": "succeeded",
        "background": True,
        "result": result,
    }


def build_video_memory(
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    window_end_date: str | date | None = None,
    window_days: int = WINDOW_DAYS,
    job_type: str = "incremental",
    force_material_sync: bool = False,
    prompt_version: str = PROMPT_VERSION,
    vlm_model_version: str = VLM_MODEL_VERSION,
    cluster_algorithm_version: str = CLUSTER_ALGORITHM_VERSION,
    vlm_mode: str = VLM_MODE_OFFLINE,
    item_vlm_model: str = DEFAULT_ITEM_VLM_MODEL,
    bundle_vlm_model: str = DEFAULT_BUNDLE_VLM_MODEL,
    vlm_client: Any | None = None,
    max_real_vlm_items: int | None = None,
    max_real_vlm_bundles: int | None = None,
) -> dict[str, Any]:
    """Build a rolling video-memory snapshot from published material references.

    The builder consumes the existing material library only. It does not decode
    source videos, does not call ffmpeg, and does not copy original videos.
    """

    root = Path(library_root) if library_root is not None else default_material_library_root()
    index_root = memory_index_root(root)
    db_path = video_memory_db_path(root, sqlite_path)
    index_root.mkdir(parents=True, exist_ok=True)
    ensure_video_memory_schema(db_path)
    normalized_vlm_mode = _normalize_vlm_mode(vlm_mode)
    item_model_version = item_vlm_model or vlm_model_version or DEFAULT_ITEM_VLM_MODEL
    bundle_model_version = bundle_vlm_model or vlm_model_version or DEFAULT_BUNDLE_VLM_MODEL
    pipeline_vlm_model_version = vlm_model_version
    if normalized_vlm_mode != VLM_MODE_OFFLINE:
        pipeline_vlm_model_version = f"{normalized_vlm_mode}:{item_model_version}|{bundle_model_version}"

    window_end = _coerce_date(window_end_date) or _today_utc()
    window_start = window_end - timedelta(days=max(window_days, 1) - 1)
    material_index_path = global_material_library_db_path(root)
    if force_material_sync or not material_index_path.exists():
        sync_material_library(root)
    material_index_version = _material_index_version(material_index_path)
    idempotency_key = _stable_id(
        "memory-job",
        {
            "job_type": job_type,
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "material_index_version": material_index_version,
            "prompt_version": prompt_version,
            "vlm_model_version": pipeline_vlm_model_version,
            "vlm_mode": normalized_vlm_mode,
            "item_vlm_model": item_model_version,
            "bundle_vlm_model": bundle_model_version,
            "cluster_algorithm_version": cluster_algorithm_version,
        },
    )
    job_id = _stable_id("job", idempotency_key)
    started_at = _now_iso()
    running_job = {
        "job_id": job_id,
        "schema_version": SCHEMA_VERSION,
        "job_type": job_type,
        "job_status": "running",
        "triggered_by": "system",
        "trigger_reason": job_type,
        "target_window_start_date": window_start.isoformat(),
        "target_window_end_date": window_end.isoformat(),
        "window_days_expected": int(window_days),
        "window_days_available": 0,
        "window_completeness": "empty",
        "source_material_library_root": str(root),
        "source_index_paths": [str(material_index_path)],
        "material_index_version": material_index_version,
        "material_index_snapshot_hash": _file_fingerprint(material_index_path),
        "vlm_model_version": pipeline_vlm_model_version,
        "vlm_mode": normalized_vlm_mode,
        "item_vlm_model": item_model_version,
        "bundle_vlm_model": bundle_model_version,
        "max_real_vlm_items": max_real_vlm_items,
        "max_real_vlm_bundles": max_real_vlm_bundles,
        "prompt_version": prompt_version,
        "cluster_algorithm_version": cluster_algorithm_version,
        "memory_schema_version": SCHEMA_VERSION,
        "ledger_schema_version": SCHEMA_VERSION,
        "snapshot_schema_version": SCHEMA_VERSION,
        "idempotency_key": idempotency_key,
        "started_at": started_at,
        "finished_at": "",
        "duration_ms": None,
        "error_message": "",
        "retry_count": 0,
    }
    _upsert_rows(db_path, "memory_build_jobs", [running_job], key="job_id")

    try:
        material_rows = _load_material_rows(root)
        invalid_experiment_ids = _load_invalid_time_axis_experiment_ids(root)
        evidence_items = [
            _material_row_to_evidence_item(row)
            for row in material_rows
            if _is_publishable_material_row(row, invalid_experiment_ids=invalid_experiment_ids)
        ]
        item_results, item_cache_stats = _build_item_vlm_results(
            db_path,
            evidence_items,
            prompt_version=prompt_version,
            model_version=item_model_version,
            vlm_mode=normalized_vlm_mode,
            vlm_client=vlm_client,
            max_real_vlm_items=max_real_vlm_items,
        )
        bundles = _build_evidence_bundles(evidence_items, item_results)
        bundle_results, bundle_cache_stats = _build_bundle_vlm_results(
            db_path,
            bundles,
            item_results,
            prompt_version=prompt_version,
            model_version=bundle_model_version,
            vlm_mode=normalized_vlm_mode,
            vlm_client=vlm_client,
            max_real_vlm_bundles=max_real_vlm_bundles,
        )
        ledger_events = _build_daily_event_ledgers(bundles, bundle_results)
        window_ledger_events = [
            row for row in ledger_events if _date_in_window(row.get("date"), window_start, window_end)
        ]
        feedback_entries = load_human_feedback(library_root=root, sqlite_path=db_path)
        clusters = _build_memory_clusters(window_ledger_events, bundles, feedback_entries)
        snapshot = _build_memory_snapshot(
            clusters,
            window_ledger_events,
            bundles,
            root=root,
            job_id=job_id,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
            job_type=job_type,
            prompt_version=prompt_version,
            vlm_model_version=pipeline_vlm_model_version,
            vlm_mode=normalized_vlm_mode,
            item_vlm_model=item_model_version,
            bundle_vlm_model=bundle_model_version,
            cluster_algorithm_version=cluster_algorithm_version,
            feedback_entries=feedback_entries,
        )

        _replace_rows(db_path, "evidence_items", evidence_items, key="evidence_id")
        _replace_rows(db_path, "vlm_item_results", item_results, key="vlm_result_id")
        _replace_rows(db_path, "evidence_bundles", bundles, key="bundle_id")
        _replace_rows(db_path, "vlm_bundle_results", bundle_results, key="vlm_result_id")
        _replace_rows(db_path, "daily_event_ledgers", ledger_events, key="ledger_event_id")
        _replace_rows(db_path, "memory_clusters", clusters, key="cluster_id")
        _upsert_rows(db_path, "memory_snapshots", [snapshot], key="snapshot_id")

        _write_outputs(
            index_root,
            {
                "evidence_items": evidence_items,
                "vlm_item_results": item_results,
                "evidence_bundles": bundles,
                "vlm_bundle_results": bundle_results,
                "daily_event_ledgers": ledger_events,
                "memory_clusters": clusters,
                "memory_snapshots": [snapshot],
                "human_feedback_entries": feedback_entries,
                "vlm_result_cache": _load_cache_rows(db_path),
            },
        )

        finished_at = _now_iso()
        duration_ms = _duration_ms(started_at, finished_at)
        completed_job = {
            **running_job,
            "job_status": "succeeded",
            "window_days_available": snapshot["window_days_available"],
            "window_completeness": snapshot["window_completeness"],
            "source_session_ids": sorted({row.get("session_id", "") for row in window_ledger_events if row.get("session_id")}),
            "source_experiment_ids": sorted({row.get("experiment_id", "") for row in window_ledger_events if row.get("experiment_id")}),
            "input_ledger_event_count": len(window_ledger_events),
            "input_bundle_count": len(bundles),
            "input_material_count": len(evidence_items),
            "vlm_cache_hit_count": item_cache_stats["hit"] + bundle_cache_stats["hit"],
            "vlm_cache_miss_count": item_cache_stats["miss"] + bundle_cache_stats["miss"],
            "vlm_qwen_cache_reuse_count": item_cache_stats.get("qwen_cache_reused", 0) + bundle_cache_stats.get("qwen_cache_reused", 0),
            "vlm_reused_existing_source_count": item_cache_stats.get("reused_existing", 0) + bundle_cache_stats.get("reused_existing", 0),
            "vlm_real_call_count": item_cache_stats.get("qwen_calls", 0) + bundle_cache_stats.get("qwen_calls", 0),
            "cluster_count": len(clusters),
            "output_snapshot_id": snapshot["snapshot_id"],
            "output_paths": {
                name: str(index_root / filename)
                for name, filename in JSONL_FILES.items()
                if name != "memory_build_jobs"
            },
            "finished_at": finished_at,
            "duration_ms": duration_ms,
        }
        _upsert_rows(db_path, "memory_build_jobs", [completed_job], key="job_id")
        _write_outputs(index_root, {"memory_build_jobs": [completed_job]})
        return {
            "schema_version": SCHEMA_VERSION,
            "job": completed_job,
            "snapshot": snapshot,
            "counts": {
                "materials": len(evidence_items),
                "item_vlm_results": len(item_results),
                "evidence_bundles": len(bundles),
                "bundle_vlm_results": len(bundle_results),
                "ledger_events": len(ledger_events),
                "window_ledger_events": len(window_ledger_events),
                "clusters": len(clusters),
            },
            "sqlite_path": str(db_path),
            "memory_index_root": str(index_root),
        }
    except Exception as exc:
        failed_job = {
            **running_job,
            "job_status": "failed",
            "finished_at": _now_iso(),
            "duration_ms": _duration_ms(started_at, _now_iso()),
            "error_message": str(exc),
        }
        _upsert_rows(db_path, "memory_build_jobs", [failed_job], key="job_id")
        _write_outputs(index_root, {"memory_build_jobs": [failed_job]})
        raise


def query_video_memory(
    query: str,
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    snapshot_id: str | None = None,
    filters: Mapping[str, Any] | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    db_path = video_memory_db_path(library_root, sqlite_path)
    ensure_video_memory_schema(db_path)
    snapshot = get_memory_snapshot(library_root=library_root, sqlite_path=db_path, snapshot_id=snapshot_id)
    if snapshot is None:
        return {
            "answer_id": _stable_id("answer", {"query": query, "empty": True}),
            "schema_version": SCHEMA_VERSION,
            "query": query,
            "generated_at": _now_iso(),
            "answer_summary": "No Video Memory snapshot is available yet; no 30-day claim can be made.",
            "claims": [],
            "unresolved_questions": ["Build Video Memory before querying evidence-linked answers."],
            "confidence": 0.0,
            "human_confirmation_status": "unconfirmed",
            "partial_window": True,
            "is_full_30_day_memory": False,
            "partial_window_notice": "No snapshot is available.",
            "window_scope": {
                "window_completeness": "missing",
                "window_days_expected": WINDOW_DAYS,
                "window_days_available": 0,
            },
        }

    filter_payload = dict(filters or {})
    clusters = [
        cluster
        for cluster in _load_table(db_path, "memory_clusters")
        if _memory_cluster_matches_filters(cluster, filter_payload)
    ]
    ledgers = {row["ledger_event_id"]: row for row in _load_table(db_path, "daily_event_ledgers")}
    bundles = {row["bundle_id"]: row for row in _load_table(db_path, "evidence_bundles")}
    tokens = set(_tokens(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    for cluster in clusters:
        text = _join_text(
            [
                cluster.get("cluster_title"),
                cluster.get("cluster_summary"),
                cluster.get("query_ready_text"),
                " ".join(_ensure_list(cluster.get("key_objects"))),
                " ".join(_ensure_list(cluster.get("canonical_actions"))),
            ]
        )
        text_tokens = set(_tokens(text))
        score = _jaccard(tokens, text_tokens) if tokens else 0.0
        if cluster.get("status") in {"primary", "human_confirmed"}:
            score += 0.20
        elif cluster.get("status") == "active":
            score += 0.10
        if score > 0 or not tokens:
            scored.append((score, cluster))
    scored.sort(key=lambda item: (item[0], float(item[1].get("confidence") or 0.0)), reverse=True)
    selected = [cluster for _, cluster in scored[: max(limit, 1)]]

    claims = [
        _cluster_to_answer_claim(cluster, ledgers=ledgers, bundles=bundles, filters=filter_payload)
        for cluster in selected
        if cluster.get("status") not in {"human_rejected", "suppressed", "archived", "expired_from_current_window"}
    ]
    if filter_payload:
        claims = [
            claim
            for claim in claims
            if claim.get("ledger_event_id") or claim.get("evidence_bundle_id") or claim.get("evidence_links")
        ]
    evidence_linked_claims = [
        claim
        for claim in claims
        if claim.get("evidence_bundle_id") and claim.get("support") != "unsupported_no_evidence_bundle"
    ]
    window_days_expected = int(snapshot.get("window_days_expected") or WINDOW_DAYS)
    window_days_available = int(snapshot.get("window_days_available") or 0)
    window_completeness = str(snapshot.get("window_completeness") or "unknown")
    is_full_window = window_completeness == "complete" and window_days_available >= window_days_expected
    partial_window_notice = ""
    if not is_full_window:
        partial_window_notice = (
            f"Partial window only: available material dates cover "
            f"{window_days_available}/{window_days_expected} days; this is not a complete 30-day memory."
        )
    prefix = partial_window_notice
    summary_parts = [prefix] if prefix else []
    if evidence_linked_claims:
        summary_parts.append("; ".join(str(claim.get("claim_text") or "") for claim in evidence_linked_claims[:3]))
    else:
        summary_parts.append("No matching evidence-linked claim was found.")
    return {
        "answer_id": _stable_id("answer", {"query": query, "snapshot_id": snapshot.get("snapshot_id"), "claims": [c.get("claim_id") for c in claims]}),
        "schema_version": SCHEMA_VERSION,
        "query": query,
        "generated_at": _now_iso(),
        "window_start_date": snapshot.get("window_start_date"),
        "window_end_date": snapshot.get("window_end_date"),
        "window_completeness": snapshot.get("window_completeness"),
        "window_days_available": snapshot.get("window_days_available"),
        "window_days_expected": snapshot.get("window_days_expected"),
        "partial_window": not is_full_window,
        "is_full_30_day_memory": is_full_window and window_days_expected == WINDOW_DAYS,
        "partial_window_notice": partial_window_notice,
        "window_scope": {
            "window_start_date": snapshot.get("window_start_date"),
            "window_end_date": snapshot.get("window_end_date"),
            "window_days_expected": window_days_expected,
            "window_days_available": window_days_available,
            "window_completeness": window_completeness,
            "window_completeness_ratio": snapshot.get("window_completeness_ratio"),
            "partial_window_reason": snapshot.get("partial_window_reason") or "",
        },
        "snapshot_id": snapshot.get("snapshot_id"),
        "filters": filter_payload,
        "answer_summary": " ".join(part for part in summary_parts if part),
        "claims": claims,
        "evidence_linked_claim_count": len(evidence_linked_claims),
        "unsupported_claim_count": len(claims) - len(evidence_linked_claims),
        "unresolved_questions": snapshot.get("unresolved_questions") or [],
        "confidence": _mean([float(claim.get("confidence") or 0.0) for claim in evidence_linked_claims]),
        "human_confirmation_status": "human_confirmed"
        if evidence_linked_claims and all(claim.get("human_confirmation_status") == "human_confirmed" for claim in evidence_linked_claims)
        else "unconfirmed",
    }


def _memory_cluster_matches_filters(cluster: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    if not filters:
        return True

    def _text_set(*keys: str) -> set[str]:
        values: set[str] = set()
        for key in keys:
            raw = cluster.get(key)
            for value in _ensure_list(raw):
                if isinstance(value, Mapping):
                    values.update(str(v).strip().lower() for v in value.values() if str(v or "").strip())
                elif str(value or "").strip():
                    values.add(str(value).strip().lower())
        return values

    def _match_any(expected: Any, values: set[str], *, contains: bool = False) -> bool:
        expected_values = [str(value).strip().lower() for value in _ensure_list(expected) if str(value or "").strip()]
        if not expected_values:
            return True
        if contains:
            return any(any(expected_value in value for value in values) for expected_value in expected_values)
        return any(expected_value in values for expected_value in expected_values)

    if not _match_any(filters.get("cluster_id"), _text_set("cluster_id")):
        return False
    if not _match_any(filters.get("session_id"), _text_set("session_id", "related_sessions")):
        return False
    if not _match_any(filters.get("experiment_id"), _text_set("experiment_id", "related_experiments")):
        return False
    if not _match_any(filters.get("date"), _text_set("date", "related_dates")):
        return False
    if not _match_any(filters.get("status"), _text_set("status")):
        return False

    action_filter = filters.get("action") or filters.get("canonical_action") or filters.get("action_name")
    if not _match_any(action_filter, _text_set("canonical_actions", "cluster_title", "query_ready_text"), contains=True):
        return False
    object_filter = filters.get("object") or filters.get("primary_object")
    if not _match_any(object_filter, _text_set("key_objects", "cluster_title", "query_ready_text"), contains=True):
        return False
    instrument_filter = filters.get("instrument")
    if not _match_any(instrument_filter, _text_set("key_instruments", "cluster_title", "query_ready_text"), contains=True):
        return False

    date_values = sorted(_text_set("date", "related_dates"))
    start_date = str(filters.get("start_date") or "").strip()
    end_date = str(filters.get("end_date") or "").strip()
    if start_date and not any(value >= start_date for value in date_values):
        return False
    if end_date and not any(value <= end_date for value in date_values):
        return False
    return True


def _memory_payload_matches_filters(payload: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    if not filters:
        return True

    def values(*keys: str) -> set[str]:
        result: set[str] = set()
        for key in keys:
            for value in _ensure_list(payload.get(key)):
                if isinstance(value, Mapping):
                    result.update(str(v).strip().lower() for v in value.values() if str(v or "").strip())
                elif str(value or "").strip():
                    result.add(str(value).strip().lower())
        return result

    def match_exact(expected: Any, current: set[str]) -> bool:
        expected_values = [str(value).strip().lower() for value in _ensure_list(expected) if str(value or "").strip()]
        return not expected_values or any(value in current for value in expected_values)

    def match_contains(expected: Any, current: set[str]) -> bool:
        expected_values = [str(value).strip().lower() for value in _ensure_list(expected) if str(value or "").strip()]
        return not expected_values or any(any(value in item for item in current) for value in expected_values)

    if not match_exact(filters.get("session_id"), values("session_id")):
        return False
    if not match_exact(filters.get("experiment_id"), values("experiment_id")):
        return False
    if not match_exact(filters.get("segment_id"), values("segment_id")):
        return False
    if not match_exact(filters.get("micro_segment_id"), values("micro_segment_id")):
        return False
    if not match_exact(filters.get("date"), values("date")):
        return False

    action_filter = filters.get("action") or filters.get("canonical_action") or filters.get("action_name")
    if not match_contains(action_filter, values("action_name", "canonical_action_type", "query_ready_text")):
        return False
    object_filter = filters.get("object") or filters.get("primary_object")
    if not match_contains(object_filter, values("primary_object", "detected_objects", "key_objects", "query_ready_text")):
        return False

    date_values = sorted(values("date"))
    start_date = str(filters.get("start_date") or "").strip()
    end_date = str(filters.get("end_date") or "").strip()
    if start_date and not any(value >= start_date for value in date_values):
        return False
    if end_date and not any(value <= end_date for value in date_values):
        return False
    return True


class EvidenceItem:
    """JSON-compatible evidence item wrapper used by contract tests and APIs."""

    def __init__(self, **payload: Any) -> None:
        self.payload = dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceItem":
        return cls(**dict(payload))

    def to_json_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def to_dict(self) -> dict[str, Any]:
        return self.to_json_dict()


def validate_evidence_item(item: EvidenceItem | Mapping[str, Any]) -> dict[str, Any]:
    payload = item.to_json_dict() if hasattr(item, "to_json_dict") else dict(item)  # type: ignore[arg-type]
    required = ("evidence_id", "session_id", "experiment_id", "time_range", "action", "views", "trace")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"EvidenceItem missing required fields: {', '.join(missing)}")
    if not isinstance(payload.get("views"), list) or not payload["views"]:
        raise ValueError("EvidenceItem.views must contain at least one view payload")
    for view in payload["views"]:
        if not isinstance(view, Mapping):
            raise ValueError("EvidenceItem.views rows must be objects")
        for path_key in ("clip_uri", "keyframe_uri"):
            value = str(view.get(path_key) or "")
            if re.match(r"^[A-Za-z]:\\", value):
                raise ValueError(f"EvidenceItem {path_key} must not be an absolute workstation path")
    return payload


def build_vlm_cache_key(
    *,
    model: str,
    prompt_version: str,
    evidence_item: Mapping[str, Any],
    image_refs: Sequence[str] | None = None,
) -> str:
    return _stable_id(
        "vlm-cache",
        {
            "model": model,
            "prompt_version": prompt_version,
            "evidence_item": evidence_item,
            "image_refs": sorted(str(ref) for ref in (image_refs or [])),
        },
    )


class VLMCache:
    """Small filesystem cache for dry-run and offline VLM result reuse."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, cache_key: str, response: Mapping[str, Any], *, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": "video_memory.vlm_cache_entry.v1",
            "status": "hit",
            "cache_key": str(cache_key),
            "response": dict(response),
            "metadata": dict(metadata or {}),
            "created_at": _now_iso(),
        }
        (self.root / f"{_safe_cache_filename(cache_key)}.json").write_text(_json_dumps(payload), encoding="utf-8")
        return payload

    def get(self, cache_key: str) -> dict[str, Any]:
        path = self.root / f"{_safe_cache_filename(cache_key)}.json"
        if not path.exists():
            return {"status": "miss", "cache_key": str(cache_key)}
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "hit"
        payload["cache_key"] = str(cache_key)
        return payload


def score_evidence_clusters(
    *,
    evidence_items: Sequence[Mapping[str, Any]],
    now: str | datetime | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in evidence_items:
        grouped.setdefault(str(item.get("cluster_id") or item.get("evidence_id") or ""), []).append(item)
    clusters: list[dict[str, Any]] = []
    for cluster_id, items in grouped.items():
        views = sorted({str(view.get("view_id") or "") for item in items for view in _ensure_list(item.get("views")) if isinstance(view, Mapping)})
        material_ids = _unique_strings(
            [
                *[item.get("material_id") for item in items],
                *[
                    (item.get("material_reference") or {}).get("material_id")
                    for item in items
                    if isinstance(item.get("material_reference"), Mapping)
                ],
            ]
        )
        sha256s = _unique_strings(
            [
                *[item.get("sha256") for item in items],
                *[
                    (item.get("material_reference") or {}).get("sha256")
                    for item in items
                    if isinstance(item.get("material_reference"), Mapping)
                ],
            ]
        )
        micro_segment_ids = _unique_strings(
            [
                *[item.get("micro_segment_id") for item in items],
                *[
                    micro_id
                    for item in items
                    if isinstance(item.get("trace"), Mapping)
                    for micro_id in _ensure_list(item.get("trace", {}).get("source_micro_segment_ids"))
                ],
            ]
        )
        keyframe_refs = _unique_strings(
            view.get("keyframe_uri") or view.get("keyframe_path")
            for item in items
            for view in _ensure_list(item.get("views"))
            if isinstance(view, Mapping)
        )
        keyclip_refs = _unique_strings(
            view.get("clip_uri") or view.get("keyclip_uri") or view.get("keyclip_path")
            for item in items
            for view in _ensure_list(item.get("views"))
            if isinstance(view, Mapping)
        )
        timestamps = [item.get("time_range") for item in items if isinstance(item.get("time_range"), Mapping)]
        confidences = [
            float(((item.get("physical_evidence") or {}).get("confidence") if isinstance(item.get("physical_evidence"), Mapping) else None) or ((item.get("retrieval") or {}).get("score") if isinstance(item.get("retrieval"), Mapping) else 0.0) or 0.0)
            for item in items
        ]
        score = _bounded(_mean(confidences) + (0.08 if len(views) >= 2 else 0.0) + min(len(items), 5) * 0.02)
        reasons = ["trace_support"]
        if len(views) >= 2:
            reasons.append("multi_view_support")
        if any(((item.get("vlm") or {}).get("confidence") if isinstance(item.get("vlm"), Mapping) else 0.0) for item in items):
            reasons.append("vlm_supported")
        observed = [_parse_datetime(item.get("observed_at") or item.get("created_at")) for item in items]
        observed = [value for value in observed if value is not None]
        latest = max(observed).isoformat() if observed else str(now or _now_iso())
        clusters.append(
            {
                "schema_version": "video_memory.cluster.v1",
                "cluster_id": cluster_id,
                "lifecycle_state": "active" if score >= 0.65 else "candidate",
                "score": score,
                "score_reasons": reasons,
                "evidence_item_ids": [str(item.get("evidence_id") or "") for item in items],
                "material_id": material_ids[0] if material_ids else "",
                "material_ids": material_ids,
                "sha256": _aggregate_sha256(sha256s),
                "sha256s": sha256s,
                "micro_segment_id": micro_segment_ids[0] if micro_segment_ids else "",
                "micro_segment_ids": micro_segment_ids,
                "keyframe": keyframe_refs[0] if keyframe_refs else "",
                "keyframe_refs": keyframe_refs,
                "keyclip": keyclip_refs[0] if keyclip_refs else "",
                "keyclip_refs": keyclip_refs,
                "timestamp": timestamps[0] if timestamps else {},
                "timestamps": timestamps,
                "evidence_trace": {
                    "cluster_id": cluster_id,
                    "evidence_item_ids": [str(item.get("evidence_id") or "") for item in items],
                    "material_ids": material_ids,
                    "sha256": _aggregate_sha256(sha256s),
                    "sha256s": sha256s,
                    "micro_segment_ids": micro_segment_ids,
                    "keyframe_refs": keyframe_refs,
                    "keyclip_refs": keyclip_refs,
                    "timestamps": timestamps,
                    "trace_complete": bool(material_ids and (keyframe_refs or keyclip_refs) and timestamps),
                },
                "last_observed_at": latest,
                "archive_reason": "",
                "evidence_items": [dict(item) for item in items],
            }
        )
    return clusters


def update_cluster_lifecycle(
    *,
    clusters: Sequence[Mapping[str, Any]],
    feedback_events: Sequence[Mapping[str, Any]] | None = None,
    now: str | datetime | None = None,
    policy: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    current = _parse_datetime(now) or datetime.now(timezone.utc)
    feedback_events = list(feedback_events or [])
    policy = dict(policy or {})
    promote_score = float(policy.get("promote_score") or 0.75)
    stale_after_days = int(policy.get("stale_after_days") or 14)
    archive_after_days = int(policy.get("archive_after_days") or 30)
    updated: list[dict[str, Any]] = []
    for cluster in clusters:
        item = dict(cluster)
        score = float(item.get("score") or 0.0)
        reasons = _ensure_list(item.get("score_reasons"))
        evidence_ids = set(str(value) for value in _ensure_list(item.get("evidence_item_ids")))
        for event in feedback_events:
            feedback_type = str(event.get("feedback_type") or "")
            target_type = str(event.get("target_type") or "")
            target_id = str(event.get("target_id") or "")
            if target_type == "cluster" and target_id == item.get("cluster_id"):
                if feedback_type in {"accepted", "confirm"}:
                    score += float(event.get("weight") or 0.10)
                    reasons.append("accepted_feedback")
                elif feedback_type in {"rejected", "reject"}:
                    score -= abs(float(event.get("weight") or -0.20))
                    reasons.append("rejected_feedback")
            if target_type == "evidence_item" and target_id in evidence_ids and feedback_type in {"rejected", "reject"}:
                score -= abs(float(event.get("weight") or -0.20))
                reasons.append("rejected_feedback")
        score = _bounded(score)
        observed = _parse_datetime(item.get("last_observed_at"))
        age_days = (current - observed).days if observed else 0
        if age_days > archive_after_days:
            lifecycle = "archived"
            archive_reason = "older_than_retention_window"
        elif age_days > stale_after_days:
            lifecycle = "stale"
            archive_reason = ""
        elif score >= promote_score:
            lifecycle = "promoted"
            archive_reason = ""
        elif score >= 0.50:
            lifecycle = "active"
            archive_reason = ""
        else:
            lifecycle = "candidate"
            archive_reason = ""
        item["score"] = score
        item["score_reasons"] = _unique_strings(reasons)
        item["lifecycle_state"] = lifecycle
        item["human_confirmation_status"] = "human_confirmed" if "accepted_feedback" in item["score_reasons"] else item.get("human_confirmation_status", "unconfirmed")
        item["archive_reason"] = archive_reason
        updated.append(item)
    return updated


def build_partial_snapshot(
    *,
    evidence_items: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
    now: str | datetime | None = None,
    retention_days: int = WINDOW_DAYS,
    max_items: int | None = None,
) -> dict[str, Any]:
    current = _parse_datetime(now) or datetime.now(timezone.utc)
    included: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in evidence_items:
        observed = _parse_datetime(item.get("observed_at") or item.get("created_at"))
        expired = bool(observed and (current - observed).days > retention_days)
        if expired:
            skipped.append({"evidence_id": item.get("evidence_id"), "reason": "outside_retention_window"})
            continue
        if max_items is not None and len(included) >= max_items:
            skipped.append({"evidence_id": item.get("evidence_id"), "reason": "max_items"})
            continue
        included.append(_strip_absolute_paths(dict(item)))
    truncation_reason = (
        "max_items"
        if max_items is not None and len(evidence_items) > max_items
        else "retention_window" if skipped else ""
    )
    material_ids = _unique_strings(
        [
            *[item.get("material_id") for item in included],
            *[
                (item.get("material_reference") or {}).get("material_id")
                for item in included
                if isinstance(item.get("material_reference"), Mapping)
            ],
        ]
    )
    sha256s = _unique_strings(
        [
            *[item.get("sha256") for item in included],
            *[
                (item.get("material_reference") or {}).get("sha256")
                for item in included
                if isinstance(item.get("material_reference"), Mapping)
            ],
        ]
    )
    micro_segment_ids = _unique_strings(
        [
            *[item.get("micro_segment_id") for item in included],
            *[
                micro_id
                for item in included
                if isinstance(item.get("trace"), Mapping)
                for micro_id in _ensure_list(item.get("trace", {}).get("source_micro_segment_ids"))
            ],
        ]
    )
    keyframe_refs = _unique_strings(
        view.get("keyframe_uri") or view.get("keyframe_path")
        for item in included
        for view in _ensure_list(item.get("views"))
        if isinstance(view, Mapping)
    )
    keyclip_refs = _unique_strings(
        view.get("clip_uri") or view.get("keyclip_uri") or view.get("keyclip_path")
        for item in included
        for view in _ensure_list(item.get("views"))
        if isinstance(view, Mapping)
    )
    timestamps = [item.get("time_range") for item in included if isinstance(item.get("time_range"), Mapping)]
    return {
        "schema_version": "video_memory.partial_snapshot.v1",
        "snapshot_id": _stable_id("partial-snapshot", {"now": str(now), "included": [item.get("evidence_id") for item in included], "max_items": max_items}),
        "snapshot_kind": "partial",
        "generated_at": current.isoformat(),
        "is_partial": True,
        "coverage": {
            "retention_days": retention_days,
            "start_time": (current - timedelta(days=retention_days)).isoformat(),
            "end_time": current.isoformat(),
            "truncation_reason": truncation_reason,
        },
        "counts": {
            "source_evidence_items": len(evidence_items),
            "included_evidence_items": len(included),
            "skipped_items": len(skipped),
        },
        "evidence_items": included,
        "clusters": [dict(cluster) for cluster in clusters],
        "skipped_items": skipped,
        "material_ids": material_ids,
        "sha256s": sha256s,
        "micro_segment_ids": micro_segment_ids,
        "keyframe_refs": keyframe_refs,
        "keyclip_refs": keyclip_refs,
        "timestamps": timestamps,
        "evidence_trace": {
            "material_ids": material_ids,
            "sha256": _aggregate_sha256(sha256s),
            "sha256s": sha256s,
            "micro_segment_ids": micro_segment_ids,
            "keyframe_refs": keyframe_refs,
            "keyclip_refs": keyclip_refs,
            "timestamps": timestamps,
            "cluster_ids": _unique_strings(cluster.get("cluster_id") for cluster in clusters),
            "trace_complete": bool(material_ids and (keyframe_refs or keyclip_refs) and timestamps),
        },
    }


def answer_video_memory_query(
    *,
    query: str,
    snapshot: Mapping[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    evidence_items = [dict(item) for item in _ensure_list(snapshot.get("evidence_items")) if isinstance(item, Mapping)]
    clusters = [dict(cluster) for cluster in _ensure_list(snapshot.get("clusters")) if isinstance(cluster, Mapping)]
    tokens = set(_tokens(query))
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in evidence_items:
        text = _join_text(
            [
                item.get("evidence_id"),
                (item.get("action") or {}).get("type") if isinstance(item.get("action"), Mapping) else "",
                (item.get("action") or {}).get("primary_object") if isinstance(item.get("action"), Mapping) else "",
                " ".join(_ensure_list((item.get("action") or {}).get("secondary_objects") if isinstance(item.get("action"), Mapping) else [])),
                (item.get("vlm") or {}).get("summary") if isinstance(item.get("vlm"), Mapping) else "",
                (item.get("retrieval") or {}).get("index_text") if isinstance(item.get("retrieval"), Mapping) else "",
            ]
        )
        retrieval_score = float(((item.get("retrieval") or {}).get("score") if isinstance(item.get("retrieval"), Mapping) else 0.0) or 0.0)
        token_score = _jaccard(tokens, set(_tokens(text))) if tokens else 0.0
        scored.append((_bounded(token_score + retrieval_score * 0.5), item))
    scored.sort(key=lambda row: row[0], reverse=True)
    selected = [(score, item) for score, item in scored[: max(top_k, 1)] if score > 0]
    cluster_by_id = {str(cluster.get("cluster_id") or ""): cluster for cluster in clusters}
    claims: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    for score, item in selected:
        cluster_id = str(item.get("cluster_id") or "")
        time_range = item.get("time_range") if isinstance(item.get("time_range"), Mapping) else {}
        views = [dict(view) for view in _ensure_list(item.get("views")) if isinstance(view, Mapping)]
        keyframes = _unique_strings([view.get("keyframe_uri") or view.get("keyframe_path") for view in views])
        keyclips = _unique_strings([view.get("clip_uri") or view.get("keyclip_uri") or view.get("keyclip_path") for view in views])
        trace = item.get("trace") if isinstance(item.get("trace"), Mapping) else {}
        micro_ids = _unique_strings([item.get("micro_segment_id"), *_ensure_list(trace.get("source_micro_segment_ids"))])
        cluster = cluster_by_id.get(cluster_id) or {}
        human_status = _first_text(
            item.get("human_confirmation_status"),
            cluster.get("human_confirmation_status"),
            cluster.get("confirmation_status"),
            "unconfirmed",
        )
        material_ref = item.get("material_reference") if isinstance(item.get("material_reference"), Mapping) else {}
        material_id = _first_text(item.get("material_id"), material_ref.get("material_id"))
        sha256 = _first_text(item.get("sha256"), material_ref.get("sha256"))
        evidence_bundle_id = _first_text(item.get("evidence_bundle_id"), item.get("bundle_id"))
        has_bundle = bool(evidence_bundle_id)
        retrieved.append(
            {
                "evidence_id": item.get("evidence_id"),
                "cluster_id": cluster_id,
                "time_range": time_range,
                "score_breakdown": {"final_score": score},
                "vlm_cache_key": ((item.get("vlm") or {}).get("cache_key") if isinstance(item.get("vlm"), Mapping) else ""),
                "views": views,
            }
        )
        claims.append(
            {
                "claim_id": _stable_id("contract-claim", {"query": query, "evidence": item.get("evidence_id")}),
                "support": "supported",
                "claim_type": "bundle_supported_fact" if has_bundle else "evidence_item_observation",
                "fact_status": "bundle_supported_fact" if has_bundle else "not_a_strong_fact_without_evidence_bundle",
                "has_evidence_bundle": has_bundle,
                "evidence_bundle_id": evidence_bundle_id,
                "evidence_bundle_ids": _unique_strings([evidence_bundle_id]),
                "material_id": material_id,
                "material_ids": _unique_strings([material_id, *_ensure_list(item.get("material_ids"))]),
                "sha256": sha256,
                "sha256s": _unique_strings([sha256, *_ensure_list(item.get("sha256s"))]),
                "session_id": _first_text(item.get("session_id")),
                "session_ids": _unique_strings([item.get("session_id")]),
                "experiment_id": _first_text(item.get("experiment_id")),
                "experiment_ids": _unique_strings([item.get("experiment_id")]),
                "micro_segment_id": micro_ids[0] if micro_ids else "",
                "micro_segment_ids": micro_ids,
                "timestamp": time_range,
                "timestamps": [time_range] if time_range else [],
                "keyframe": keyframes[0] if keyframes else "",
                "keyframe_refs": keyframes,
                "keyclip": keyclips[0] if keyclips else "",
                "keyclip_refs": keyclips,
                "confidence": score,
                "human_confirmation_status": human_status,
                "evidence_strength": "strong" if has_bundle else "item_only",
                "evidence_item_ids": [item.get("evidence_id")],
                "cluster_ids": [cluster_id],
                "claim_text": cluster.get("cluster_id") or cluster_id,
                "limitations": [] if has_bundle else ["Evidence item has no evidence_bundle_id; do not use as a strong fact."],
            }
        )
    return {
        "schema_version": "video_memory.query_answer.v1",
        "answer": {
            "status": "supported" if claims else "unsupported",
            "text": "Found traceable evidence for this query." if claims else "No sufficient evidence was found for this query.",
        },
        "partial_window": bool(snapshot.get("is_partial") or snapshot.get("snapshot_kind") == "partial"),
        "is_full_30_day_memory": False if snapshot.get("snapshot_kind") == "partial" else bool(snapshot.get("is_full_30_day_memory")),
        "partial_window_notice": (
            "Partial snapshot only; do not present these claims as a complete 30-day memory."
            if snapshot.get("is_partial") or snapshot.get("snapshot_kind") == "partial"
            else ""
        ),
        "window_scope": snapshot.get("coverage") or {},
        "evidence_trace": {
            "query": query,
            "retrieved_evidence": retrieved,
            "claims": claims,
            "limitations": [] if claims else ["Claims with no evidence item IDs are unsupported."],
        },
    }


def run_feedback_update_job(
    *,
    snapshot: Mapping[str, Any],
    feedback_events: Sequence[Mapping[str, Any]],
    now: str | datetime | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    operations: list[dict[str, Any]] = []
    for event in feedback_events:
        target_type = str(event.get("target_type") or "")
        target_id = str(event.get("target_id") or "")
        feedback_id = str(event.get("feedback_id") or "")
        feedback_type = str(event.get("feedback_type") or "")
        if target_type == "evidence_item" and feedback_type in {"rejected", "reject"}:
            operations.append(_feedback_operation("deprioritize_evidence_item", target_id, feedback_id, now))
            operations.append(_feedback_operation("rescore_cluster", _cluster_for_evidence(snapshot, target_id), feedback_id, now))
        elif target_type == "cluster":
            operations.append(_feedback_operation("rescore_cluster", target_id, feedback_id, now))
        operations.append(_feedback_operation("write_feedback_audit", target_id, feedback_id, now))
    return {
        "schema_version": "video_memory.feedback_update_job.v1",
        "job_id": _stable_id("feedback-job", {"snapshot": snapshot.get("snapshot_id"), "feedback": [event.get("feedback_id") for event in feedback_events], "now": str(now)}),
        "dry_run": bool(dry_run),
        "job_status": "planned" if dry_run else "applied",
        "operations": operations,
        "created_at": str(now or _now_iso()),
    }


def record_human_feedback(
    *,
    target_type: str,
    target_id: str,
    feedback_type: str,
    context_fields: Mapping[str, Any] | None = None,
    note: str = "",
    user_id: str = "local_user",
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> dict[str, Any]:
    db_path = video_memory_db_path(library_root, sqlite_path)
    ensure_video_memory_schema(db_path)
    entry = {
        "feedback_id": _stable_id(
            "feedback",
            {
                "target_type": target_type,
                "target_id": target_id,
                "feedback_type": feedback_type,
                "context_fields": dict(context_fields or {}),
                "note": note,
                "created_at": _now_iso(),
            },
        ),
        "schema_version": SCHEMA_VERSION,
        "target_type": str(target_type),
        "target_id": str(target_id),
        "feedback_type": str(feedback_type),
        "context_fields": dict(context_fields or {}),
        "note": str(note or ""),
        "user_id": str(user_id or "local_user"),
        "created_at": _now_iso(),
    }
    _upsert_rows(db_path, "human_feedback_entries", [entry], key="feedback_id")
    _write_outputs(memory_index_root(library_root), {"human_feedback_entries": load_human_feedback(library_root=library_root, sqlite_path=db_path)})
    return entry


def get_memory_snapshot(
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    snapshot_id: str | None = None,
) -> dict[str, Any] | None:
    db_path = video_memory_db_path(library_root, sqlite_path)
    if not db_path.exists():
        return None
    ensure_video_memory_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if snapshot_id:
            row = conn.execute("SELECT payload_json FROM memory_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        else:
            row = conn.execute("SELECT payload_json FROM memory_snapshots ORDER BY updated_at DESC LIMIT 1").fetchone()
        return json.loads(row["payload_json"]) if row else None
    finally:
        conn.close()


def load_human_feedback(
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    db_path = video_memory_db_path(library_root, sqlite_path)
    if not db_path.exists():
        return []
    ensure_video_memory_schema(db_path)
    return _load_table(db_path, "human_feedback_entries")


def ensure_video_memory_schema(path: str | Path) -> None:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        for table, id_column in (
            ("evidence_items", "evidence_id"),
            ("vlm_item_results", "vlm_result_id"),
            ("vlm_bundle_results", "vlm_result_id"),
            ("evidence_bundles", "bundle_id"),
            ("daily_event_ledgers", "ledger_event_id"),
            ("memory_clusters", "cluster_id"),
            ("memory_snapshots", "snapshot_id"),
            ("memory_build_jobs", "job_id"),
            ("human_feedback_entries", "feedback_id"),
            ("vlm_result_cache", "cache_id"),
        ):
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    {id_column} TEXT PRIMARY KEY,
                    schema_version TEXT,
                    material_id TEXT,
                    session_id TEXT,
                    experiment_id TEXT,
                    segment_id TEXT,
                    micro_segment_id TEXT,
                    sha256 TEXT,
                    date TEXT,
                    status TEXT,
                    confidence REAL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT,
                    updated_at TEXT
                )
                """
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_item_material ON vlm_item_results(material_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_bundle_micro ON evidence_bundles(micro_segment_id, segment_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_ledger_date ON daily_event_ledgers(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_cluster_status ON memory_clusters(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_snapshot_date ON memory_snapshots(date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_cache_sha ON vlm_result_cache(sha256)")
        conn.commit()
    finally:
        conn.close()


def cluster_similarity(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    scores = {
        "action_signature": _jaccard(_signature_set(a, "action_signature"), _signature_set(b, "action_signature")),
        "object_signature": _jaccard(_signature_set(a, "object_signature"), _signature_set(b, "object_signature")),
        "instrument_signature": _jaccard(_signature_set(a, "instrument_signature"), _signature_set(b, "instrument_signature")),
        "visual_semantic_signature": _jaccard(_signature_set(a, "visual_semantic_signature"), _signature_set(b, "visual_semantic_signature")),
        "temporal_signature": _jaccard(_signature_set(a, "temporal_signature"), _signature_set(b, "temporal_signature")),
        "sequence_signature": _jaccard(_signature_set(a, "sequence_signature"), _signature_set(b, "sequence_signature")),
    }
    weighted = sum(scores[key] * SIGNATURE_WEIGHTS[key] for key in SIGNATURE_WEIGHTS)
    object_instrument = max(scores["object_signature"], scores["instrument_signature"])
    hard_blocked = (
        scores["action_signature"] < MIN_ACTION_SIMILARITY
        or object_instrument < MIN_OBJECT_INSTRUMENT_SIMILARITY
    )
    decision = "merge" if weighted >= AUTO_MERGE_THRESHOLD and not hard_blocked else "review" if weighted >= REVIEW_MERGE_THRESHOLD and not hard_blocked else "separate"
    return {
        "score": round(weighted, 4),
        "component_scores": scores,
        "decision": decision,
        "hard_blocked": hard_blocked,
    }


def _load_material_rows(library_root: Path) -> list[dict[str, Any]]:
    db_path = global_material_library_db_path(library_root)
    if not db_path.exists():
        sync_material_library(library_root)
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in conn.execute("SELECT * FROM material_refs").fetchall()]
    finally:
        conn.close()
    return [_parse_material_row(row) for row in rows]


def _parse_material_row(row: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(row)
    for key in ("secondary_objects", "secondary_actions", "objects", "actions"):
        item[key] = _loads_json(item.get(key), default=[])
    if item.get("payload_json"):
        item["payload"] = _loads_json(item.get("payload_json"), default={})
    else:
        item["payload"] = {}
    item["exists"] = bool(item.get("exists"))
    return item


def _load_invalid_time_axis_experiment_ids(root: Path) -> set[str]:
    candidates = [
        memory_index_root(root) / "invalid_time_axis_experiments.json",
        root / "invalid_time_axis_experiments.json",
    ]
    invalid: set[str] = set()
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("experiments") if isinstance(payload, Mapping) else payload
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                value = row.get("experiment_id")
            else:
                value = row
            if value is not None and str(value).strip():
                invalid.add(str(value).strip())
    return invalid


def _is_publishable_material_row(row: Mapping[str, Any], *, invalid_experiment_ids: set[str] | None = None) -> bool:
    if row.get("exists") is not True:
        return False
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    experiment_id = _first_text(row.get("experiment_id"), payload.get("experiment_id"))
    if invalid_experiment_ids and experiment_id in invalid_experiment_ids:
        return False
    if bool(row.get("time_axis_unreliable")) or bool(payload.get("time_axis_unreliable")):
        return False
    time_axis = payload.get("time_axis_health") if isinstance(payload.get("time_axis_health"), Mapping) else {}
    if bool(time_axis.get("time_axis_unreliable")) or str(time_axis.get("status") or "").strip().lower() == "time_axis_unreliable":
        return False
    if str(row.get("asset_type") or "") not in {"keyframe", "video_clip"}:
        return False
    text = _join_text(
        [
            row.get("stored_file"),
            row.get("file_name"),
            row.get("source_file"),
            row.get("searchable_text"),
            row.get("candidate_status"),
            row.get("review_status"),
        ]
    ).lower()
    return not any(marker in text for marker in ("placeholder", "poster", "synthetic", "dry_run", "dry-run"))


def _material_row_to_evidence_item(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
    start_sec = _float(row.get("start_sec"))
    end_sec = _float(row.get("end_sec"), start_sec)
    event_date = _material_date(row)
    material_id = str(row.get("material_id") or "")
    sha256 = str(row.get("sha256") or "")
    evidence_id = _stable_id(
        "evidence",
        {
            "material_id": material_id,
            "sha256": sha256,
            "start_sec": start_sec,
            "end_sec": end_sec,
        },
    )
    source_video = payload.get("source_video_path") or payload.get("source_video") or row.get("source_file") or ""
    view = _first_text(row.get("view"), payload.get("view"))
    action_name = _first_text(row.get("action_name"), payload.get("action_name"))
    canonical_action = _first_text(row.get("canonical_action_type"), payload.get("canonical_action_type"))
    primary_object = _first_text(row.get("primary_object"), row.get("canonical_object"), payload.get("primary_object"))
    detected_objects = _unique_strings([*_ensure_list(row.get("objects")), *_ensure_list(row.get("secondary_objects")), primary_object])
    time_alignment = _extract_time_alignment_context(row, payload, start_sec=start_sec, end_sec=end_sec, view=view)
    micro_context = _extract_micro_context(row, payload, start_sec=start_sec, end_sec=end_sec)
    existing_vlm_sources = _extract_existing_vlm_sources(row, payload)
    package_uri = str(row.get("package_uri") or "")
    view_payload = _evidence_view_payload(
        row,
        payload,
        view=view,
        start_sec=start_sec,
        end_sec=end_sec,
        package_uri=package_uri,
    )
    micro_segment_id = _first_text(row.get("micro_segment_id"), payload.get("micro_segment_id"))
    micro_segment_ids = _unique_strings([micro_segment_id, row.get("micro_segment_id"), payload.get("micro_segment_id")])
    time_range = {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "global_start_sec": time_alignment.get("global_start_sec", start_sec),
        "global_end_sec": time_alignment.get("global_end_sec", end_sec),
        "display": _timestamp_display(start_sec, end_sec),
    }
    keyframe_ref = str(view_payload.get("keyframe_uri") or "")
    keyclip_ref = str(view_payload.get("clip_uri") or "")
    return {
        "schema_version": EVIDENCE_ITEM_SCHEMA_VERSION,
        "evidence_id": evidence_id,
        "session_id": _first_text(row.get("session_id"), payload.get("session_id"), row.get("package_name")),
        "experiment_id": _first_text(row.get("experiment_id"), payload.get("experiment_id")),
        "segment_id": _first_text(row.get("segment_id"), payload.get("segment_id")),
        "micro_segment_id": micro_segment_id,
        "micro_segment_ids": micro_segment_ids,
        "material_id": material_id,
        "material_ids": [material_id] if material_id else [],
        "date": event_date,
        "view": view,
        "camera_role": _camera_role(view),
        "global_start_sec": start_sec,
        "global_end_sec": end_sec,
        "local_video_start_sec": start_sec,
        "local_video_end_sec": end_sec,
        "timestamp_display": _timestamp_display(start_sec, end_sec),
        "timestamp": time_range,
        "timestamps": [time_range],
        "asset_type": row.get("asset_type") or "",
        "keyframe_path": str(row.get("absolute_path") or "") if row.get("asset_type") == "keyframe" else "",
        "keyclip_path": str(row.get("absolute_path") or "") if row.get("asset_type") == "video_clip" else "",
        "keyframe": keyframe_ref,
        "keyframe_refs": [keyframe_ref] if keyframe_ref else [],
        "keyclip": keyclip_ref,
        "keyclip_refs": [keyclip_ref] if keyclip_ref else [],
        "source_video_path": str(source_video or ""),
        "action_name": action_name,
        "canonical_action_type": canonical_action,
        "primary_object": primary_object,
        "detected_objects": detected_objects,
        "yolo_evidence_count": int(_float(row.get("yolo_evidence_count"), 0.0) or 0),
        "physical_event_type": _first_text(canonical_action, action_name),
        "interaction_score": _float(payload.get("interaction_score"), _float(row.get("quality_score"), 0.0)) or 0.0,
        "quality_score": _float(row.get("quality_score"), 0.0) or 0.0,
        "time_alignment_confidence": _float(time_alignment.get("confidence"), 1.0) or 1.0,
        "material_gate_status": "publishable",
        "searchable_text_from_preprocess": _first_text(row.get("searchable_text"), payload.get("searchable_text")),
        "human_text_context": "",
        "vlm_prompt_context": {},
        "time_range": time_range,
        "action": {
            "type": _first_text(canonical_action, action_name, "unknown_action"),
            "name": action_name,
            "primary_object": primary_object,
            "secondary_objects": _ensure_list(row.get("secondary_objects")),
            "description": _first_text(payload.get("display_title"), row.get("display_name"), action_name),
        },
        "physical_evidence": {
            "mode": _first_text(payload.get("physical_evidence_mode"), row.get("candidate_source")),
            "hand_object_contact": "hand" in _first_text(canonical_action, action_name).lower(),
            "yolo_evidence_count": int(_float(row.get("yolo_evidence_count"), 0.0) or 0),
            "quality_score": _float(row.get("quality_score"), 0.0) or 0.0,
            "confidence": _bounded((_float(row.get("quality_score"), 0.5) or 0.5) + min(int(_float(row.get("yolo_evidence_count"), 0) or 0), 10) * 0.02),
            "uncertainty_reasons": _ensure_list(payload.get("quality_reasons")),
        },
        "views": [view_payload] if view_payload else [],
        "material_reference": {
            "material_id": material_id,
            "source_material_id": row.get("source_material_id") or payload.get("material_id") or "",
            "package_name": row.get("package_name") or "",
            "package_uri": package_uri,
            "stored_file": row.get("stored_file") or "",
            "asset_type": row.get("asset_type") or "",
            "asset_kind": row.get("asset_kind") or payload.get("asset_kind") or "",
            "sha256": sha256,
        },
        "micro_segment": micro_context,
        "time_alignment": time_alignment,
        "retrieval": {
            "index_text": _join_text([row.get("searchable_text"), action_name, canonical_action, primary_object, detected_objects]),
            "embedding_id": "",
            "score": _float(row.get("quality_score"), 0.0) or 0.0,
        },
        "trace": {
            "decision_path": "material_library.key_material_references.micro.time_alignment.video_memory",
            "decision_trace": _unique_strings(
                [
                    "material_library_row",
                    "key_material_references",
                    "micro_segment" if micro_context else "",
                    "time_alignment" if time_alignment else "",
                    "reused_existing_vlm" if existing_vlm_sources else "",
                ]
            ),
            "material_id": material_id,
            "material_ids": [material_id] if material_id else [],
            "sha256": sha256,
            "sha256s": [sha256] if sha256 else [],
            "micro_segment_id": micro_segment_id,
            "micro_segment_ids": micro_segment_ids,
            "timestamp": time_range,
            "timestamps": [time_range],
            "keyframe": keyframe_ref,
            "keyframe_refs": [keyframe_ref] if keyframe_ref else [],
            "keyclip": keyclip_ref,
            "keyclip_refs": [keyclip_ref] if keyclip_ref else [],
            "source_material_id": row.get("source_material_id") or "",
            "source_material_library_id": material_id,
            "source_micro_segment_ids": micro_segment_ids,
            "source_paths": {
                "stored_file": row.get("stored_file") or "",
                "package_uri": package_uri,
                "source_file": row.get("source_file") or "",
            },
        },
        "vlm_existing_sources": existing_vlm_sources,
        "existing_vlm_result": _extract_existing_vlm_payload(row, payload),
        "confidence": _bounded((_float(row.get("quality_score"), 0.5) or 0.5) + min(int(_float(row.get("yolo_evidence_count"), 0) or 0), 10) * 0.02),
        "evidence_strength": _evidence_strength(row),
        "related_evidence_ids": [],
        "stored_file": str(row.get("stored_file") or ""),
        "source_file": str(row.get("source_file") or ""),
        "sha256": sha256,
        "sha256s": [sha256] if sha256 else [],
        "asset_sha256": sha256,
        "package_name": str(row.get("package_name") or ""),
        "created_at": _now_iso(),
    }


def _normalize_vlm_mode(value: str | None) -> str:
    mode = str(value or VLM_MODE_OFFLINE).strip().lower().replace("-", "_")
    aliases = {
        "offline": VLM_MODE_OFFLINE,
        "metadata": VLM_MODE_OFFLINE,
        "metadata_only": VLM_MODE_OFFLINE,
        "reuse": VLM_MODE_REUSE_EXISTING,
        "reuse_existing": VLM_MODE_REUSE_EXISTING,
        "reuse_existing_cache": VLM_MODE_REUSE_EXISTING,
        "reuse_existing_vlm_cache": VLM_MODE_REUSE_EXISTING,
        "cache_only": VLM_MODE_REUSE_EXISTING,
        "existing": VLM_MODE_REUSE_EXISTING,
        "real": VLM_MODE_REAL_QWEN_ASYNC,
        "real_qwen": VLM_MODE_REAL_QWEN_ASYNC,
        "qwen": VLM_MODE_REAL_QWEN_ASYNC,
        "qwen_async": VLM_MODE_REAL_QWEN_ASYNC,
    }
    mode = aliases.get(mode, mode)
    if mode not in {VLM_MODE_OFFLINE, VLM_MODE_REUSE_EXISTING, VLM_MODE_REAL_QWEN_ASYNC}:
        return VLM_MODE_OFFLINE
    return mode


def _extract_time_alignment_context(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    start_sec: float | None,
    end_sec: float | None,
    view: str,
) -> dict[str, Any]:
    source = _first_mapping(
        payload.get("time_alignment"),
        payload.get("paired_view_time_alignment"),
        payload.get("alignment"),
        row.get("time_alignment"),
    )
    confidence = _float(
        source.get("confidence")
        or source.get("alignment_confidence")
        or payload.get("time_alignment_confidence")
        or row.get("time_alignment_confidence"),
        1.0,
    )
    global_start = _float(source.get("global_start_sec") or source.get("start_sec"), start_sec)
    global_end = _float(source.get("global_end_sec") or source.get("end_sec"), end_sec)
    return {
        "schema_version": "video_memory.time_alignment_ref.v1",
        "mode": _first_text(source.get("mode"), payload.get("physical_evidence_mode"), "material_row_time_range"),
        "view": view,
        "global_start_sec": global_start,
        "global_end_sec": global_end,
        "local_start_sec": start_sec,
        "local_end_sec": end_sec,
        "offset_sec": _float(source.get("offset_sec"), 0.0) or 0.0,
        "confidence": _bounded(confidence or 0.0),
        "source": _first_text(source.get("source"), payload.get("candidate_source"), "material_library"),
        "failure_reason": _first_text(source.get("failure_reason"), source.get("reason")),
        "views": source.get("views") if isinstance(source.get("views"), Mapping) else {},
    }


def _extract_micro_context(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    start_sec: float | None,
    end_sec: float | None,
) -> dict[str, Any]:
    micro = _first_mapping(payload.get("micro_segment"), payload.get("micro"), row.get("micro_segment"))
    micro_id = _first_text(row.get("micro_segment_id"), payload.get("micro_segment_id"), micro.get("micro_segment_id"))
    if not micro_id and not micro:
        return {}
    return {
        "schema_version": "video_memory.micro_segment_ref.v1",
        "micro_segment_id": micro_id,
        "parent_segment_id": _first_text(row.get("parent_segment_id"), payload.get("parent_segment_id"), row.get("segment_id"), micro.get("parent_segment_id")),
        "start_sec": _float(micro.get("start_sec"), start_sec),
        "end_sec": _float(micro.get("end_sec"), end_sec),
        "action_label": _first_text(micro.get("action_label"), micro.get("action_name"), payload.get("action_name"), row.get("action_name")),
        "primary_object": _first_text(micro.get("primary_object"), payload.get("primary_object"), row.get("primary_object")),
        "source_view": _first_text(micro.get("source_view"), micro.get("view"), payload.get("view"), row.get("view")),
        "evidence_refs": _ensure_list(micro.get("evidence_refs") or payload.get("source_yolo_evidence") or payload.get("yolo_evidence")),
    }


def _evidence_view_payload(
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    view: str,
    start_sec: float | None,
    end_sec: float | None,
    package_uri: str,
) -> dict[str, Any]:
    if not view and not package_uri:
        return {}
    asset_type = str(row.get("asset_type") or "")
    clip_uri = package_uri if asset_type == "video_clip" else _first_text(payload.get("clip_uri"), payload.get("keyclip_uri"))
    keyframe_uri = package_uri if asset_type == "keyframe" else _first_text(payload.get("keyframe_uri"), payload.get("frame_uri"))
    yolo_refs = _ensure_list(payload.get("source_yolo_evidence") or payload.get("yolo_evidence") or row.get("source_yolo_evidence"))
    return {
        "view_id": view or "unknown_view",
        "camera_role": _camera_role(view),
        "source_video_id": _first_text(payload.get("source_video_id"), payload.get("source_video"), row.get("source_file")),
        "local_start_sec": start_sec,
        "local_end_sec": end_sec,
        "clip_uri": clip_uri,
        "keyframe_uri": keyframe_uri,
        "frame_ids": _unique_strings([item.get("frame_id") for item in yolo_refs if isinstance(item, Mapping)]),
        "yolo_refs": yolo_refs[:20],
        "asset_type": asset_type,
    }


def _extract_existing_vlm_sources(row: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for source_type, value in (
        ("vlm_semantics", _first_value("vlm_semantics", row, payload)),
        ("qwen_event_audits", _first_value("qwen_event_audits", row, payload)),
        ("qwen_event_audit", _first_value("qwen_event_audit", row, payload)),
        ("advanced_vision_evidence", _first_value("advanced_vision_evidence", row, payload)),
        ("vlm_result", _first_value("vlm_result", row, payload)),
        ("vlm_review", _first_value("vlm_review", row, payload)),
        ("semantic_review", _first_value("semantic_review", row, payload)),
    ):
        for index, item in enumerate(_ensure_list(_loads_json(value, default=value))):
            if isinstance(item, Mapping):
                payload_item = dict(item)
            elif isinstance(item, str) and item.strip():
                payload_item = {"description": item.strip()}
            else:
                continue
            sources.append(
                {
                    "source_type": source_type,
                    "source_id": _first_text(payload_item.get("audit_id"), payload_item.get("event_id"), payload_item.get("vlm_result_id"), f"{source_type}_{index}"),
                    "status": _first_text(payload_item.get("status"), payload_item.get("decision")),
                    "confidence": _float(payload_item.get("confidence"), 0.0) or 0.0,
                    "model": _first_text(payload_item.get("model"), payload_item.get("model_version")),
                    "payload": payload_item,
                }
            )
    return sources


def _extract_existing_vlm_payload(row: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    sources = _extract_existing_vlm_sources(row, payload)
    if sources:
        first = sources[0]
        return dict(first.get("payload") or {})
    for key in (
        "vlm_semantics",
        "vlm_result",
        "vlm_review",
        "qwen_event_audit",
        "semantic_review",
        "yolo_vlm_review",
        "vlm",
    ):
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if isinstance(value, Mapping) and value:
            return dict(value)
        if isinstance(value, str) and value.strip():
            return {"description": value.strip()}
        value = row.get(key)
        if isinstance(value, Mapping) and value:
            return dict(value)
        if isinstance(value, str) and value.strip():
            parsed = _loads_json(value, default=None)
            return dict(parsed) if isinstance(parsed, Mapping) else {"description": value.strip()}
    return {}


def _item_understanding_from_existing_vlm(
    item: Mapping[str, Any],
    existing_vlm: Mapping[str, Any],
    *,
    prompt_version: str,
    model_version: str,
) -> dict[str, Any]:
    fallback = _infer_item_understanding(item, prompt_version=prompt_version, model_version=model_version)
    description = _first_text(
        existing_vlm.get("description"),
        existing_vlm.get("summary"),
        existing_vlm.get("visual_scene_summary"),
        fallback.get("visual_scene_summary"),
    )
    activities = _unique_strings(
        [
            *_ensure_list(existing_vlm.get("detected_activities")),
            *_ensure_list(existing_vlm.get("activities")),
            existing_vlm.get("possible_lab_action"),
        ]
    )
    objects = _unique_strings(
        [
            *_ensure_list(existing_vlm.get("object_labels")),
            *_ensure_list(existing_vlm.get("visible_objects")),
            *_ensure_list(fallback.get("visible_objects")),
        ]
    )
    confidence = _bounded(_float(existing_vlm.get("confidence"), fallback.get("confidence")) or 0.0)
    fallback.update(
        {
            "model_version": model_version,
            "visual_scene_summary": description,
            "visible_objects": objects,
            "possible_lab_action": activities[0] if activities else fallback.get("possible_lab_action", ""),
            "evidence_for_action": _unique_strings([description, *_ensure_list(fallback.get("evidence_for_action"))]),
            "confidence": confidence,
            "searchable_keywords": _unique_strings([*_ensure_list(fallback.get("searchable_keywords")), *objects, *activities]),
            "strong_facts": _unique_strings([description, *_ensure_list(fallback.get("strong_facts"))]),
            "vlm_source": "existing_material_vlm",
            "raw_existing_vlm": dict(existing_vlm),
        }
    )
    return fallback


def _run_real_item_vlm(
    item: Mapping[str, Any],
    *,
    vlm_client: Any,
    prompt_version: str,
    model_version: str,
) -> dict[str, Any]:
    fallback = _infer_item_understanding(item, prompt_version=prompt_version, model_version=model_version)
    asset_path = str(item.get("keyframe_path") or "")
    if not asset_path:
        return fallback
    prompt = _item_vlm_prompt(item)
    response = _call_vlm_client(vlm_client, asset_path=asset_path, prompt=prompt, model_version=model_version)
    description = _first_text(response.get("description"), response.get("visual_scene_summary"), fallback.get("visual_scene_summary"))
    objects = _unique_strings([*_ensure_list(response.get("object_labels")), *_ensure_list(response.get("visible_objects")), *_ensure_list(fallback.get("visible_objects"))])
    activities = _unique_strings([*_ensure_list(response.get("detected_activities")), response.get("possible_lab_action")])
    confidence = _bounded(_float(response.get("confidence"), fallback.get("confidence")) or 0.0)
    fallback.update(
        {
            "model_version": model_version,
            "visual_scene_summary": description,
            "visible_objects": objects,
            "manipulated_objects": _unique_strings([*_ensure_list(fallback.get("manipulated_objects")), *_ensure_list(response.get("manipulated_objects"))]),
            "possible_lab_action": activities[0] if activities else fallback.get("possible_lab_action", ""),
            "evidence_for_action": _unique_strings([description, *_ensure_list(fallback.get("evidence_for_action"))]),
            "confidence": confidence,
            "searchable_keywords": _unique_strings([*_ensure_list(fallback.get("searchable_keywords")), *objects, *activities]),
            "strong_facts": _unique_strings([description, *_ensure_list(fallback.get("strong_facts"))]),
            "vlm_source": "real_qwen_item",
            "raw_vlm_response": response,
        }
    )
    return fallback


def _run_real_bundle_vlm(
    bundle: Mapping[str, Any],
    item_by_id: Mapping[str, Mapping[str, Any]],
    *,
    vlm_client: Any,
    prompt_version: str,
    model_version: str,
) -> dict[str, Any]:
    fallback = _infer_bundle_understanding(bundle, item_by_id, prompt_version=prompt_version, model_version=model_version)
    keyframes = [path for path in _ensure_list(bundle.get("keyframes")) if str(path).strip()]
    if not keyframes:
        return fallback
    prompt = _bundle_vlm_prompt(bundle, item_by_id)
    # DashScopeVLClient accepts one image per call; use the first representative frame
    # and ground it with dual-view + item-level context.
    response = _call_vlm_client(vlm_client, asset_path=str(keyframes[0]), prompt=prompt, model_version=model_version)
    description = _first_text(response.get("description"), response.get("merged_scene_understanding"), fallback.get("merged_scene_understanding"))
    activities = _unique_strings([*_ensure_list(response.get("detected_activities")), response.get("possible_lab_action")])
    objects = _unique_strings([*_ensure_list(response.get("object_labels")), *_ensure_list(fallback.get("searchable_keywords"))])
    confidence = _bounded((_float(response.get("confidence"), fallback.get("confidence")) or 0.0) * 0.9 + float(bundle.get("confidence") or 0.0) * 0.1)
    fallback.update(
        {
            "model_version": model_version,
            "merged_scene_understanding": description,
            "merged_action_understanding": activities[0] if activities else fallback.get("merged_action_understanding", ""),
            "strong_facts": _unique_strings([description, *_ensure_list(fallback.get("strong_facts"))]),
            "weak_inferences": _unique_strings([*_ensure_list(fallback.get("weak_inferences")), *activities]),
            "confidence": confidence,
            "searchable_keywords": _unique_strings([*_ensure_list(fallback.get("searchable_keywords")), *objects, *activities]),
            "vlm_source": "real_qwen_bundle",
            "raw_vlm_response": response,
        }
    )
    return fallback


def _call_vlm_client(vlm_client: Any, *, asset_path: str, prompt: str, model_version: str) -> dict[str, Any]:
    previous_model = getattr(vlm_client, "model", None)
    model_changed = previous_model is not None and model_version
    if model_changed:
        try:
            setattr(vlm_client, "model", model_version)
        except Exception:
            model_changed = False
    try:
        response = vlm_client.describe_scene(asset_path, prompt=prompt, temperature=0.0)
    finally:
        if model_changed:
            try:
                setattr(vlm_client, "model", previous_model)
            except Exception:
                pass
    if isinstance(response, Mapping):
        raw = dict(response)
    else:
        raw = {
            "description": getattr(response, "description", ""),
            "detected_activities": getattr(response, "detected_activities", []),
            "object_labels": getattr(response, "object_labels", []),
            "step_indicators": getattr(response, "step_indicators", []),
            "confidence": getattr(response, "confidence", 0.0),
            "model": getattr(response, "model", model_version),
            "raw_response": getattr(response, "raw_response", {}),
            "inference_time_ms": getattr(response, "inference_time_ms", None),
        }
    raw.setdefault("model", model_version)
    return raw


def _item_vlm_prompt(item: Mapping[str, Any]) -> str:
    return _join_text(
        [
            "请只基于可见画面和给定 YOLO 物理证据做实验室关键素材理解。",
            "输出 JSON，区分看得见的事实、合理推断、不确定内容；不要凭空写实验名、样品名、项目名或 SOP 名。",
            f"action_name={item.get('action_name')}",
            f"canonical_action_type={item.get('canonical_action_type')}",
            f"primary_object={item.get('primary_object')}",
            f"detected_objects={', '.join(_ensure_list(item.get('detected_objects')))}",
            f"physical_event_type={item.get('physical_event_type')}",
            f"timestamp={item.get('timestamp_display')}",
            f"view={item.get('view')}",
            f"yolo_evidence_count={item.get('yolo_evidence_count')}",
        ]
    )


def _bundle_vlm_prompt(bundle: Mapping[str, Any], item_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    item_summaries = [
        str((item_by_id.get(row_id) or {}).get("visual_scene_summary") or "")
        for row_id in _ensure_list(bundle.get("vlm_item_result_ids"))
        if item_by_id.get(row_id)
    ]
    return _join_text(
        [
            "请合并双视角/多素材证据，输出该 evidence bundle 的结构化理解。",
            "必须绑定时间戳、动作、对象和 YOLO 证据；只写证据支持的事实，弱推断放到 uncertainty/weak inference。",
            f"bundle_id={bundle.get('bundle_id')}",
            f"action_name={bundle.get('action_name')}",
            f"canonical_action_type={bundle.get('canonical_action_type')}",
            f"primary_object={bundle.get('primary_object')}",
            f"views={', '.join(_ensure_list(bundle.get('views')))}",
            f"time_range={bundle.get('time_range')}",
            f"yolo_summary={bundle.get('yolo_summary')}",
            f"item_summaries={' | '.join(item_summaries[:6])}",
        ]
    )


def _build_item_vlm_results(
    db_path: Path,
    evidence_items: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str,
    model_version: str,
    vlm_mode: str = VLM_MODE_OFFLINE,
    vlm_client: Any | None = None,
    max_real_vlm_items: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mode = _normalize_vlm_mode(vlm_mode)
    results, stats = enhance_items_sync(
        evidence_items,
        sqlite_path=db_path,
        prompt_version=prompt_version,
        offline_model_version=VLM_MODEL_VERSION if mode != VLM_MODE_REAL_QWEN_ASYNC or vlm_client is None else model_version,
        qwen_model=model_version,
        vlm_client=vlm_client,
        enable_qwen=mode == VLM_MODE_REAL_QWEN_ASYNC,
        reuse_qwen_cache=mode == VLM_MODE_REUSE_EXISTING,
        max_qwen_calls=max_real_vlm_items,
        fallback_factory=lambda item, prompt, model: _infer_item_understanding(
            item,
            prompt_version=prompt,
            model_version=model,
        ),
    )
    stats["reuse"] = stats.get("reused_existing", 0)
    stats["real"] = stats.get("qwen_calls", 0)
    stats["fallback"] = stats.get("offline_fallback", 0)
    stats["error"] = stats.get("errors", 0)
    return results, stats


def _infer_item_understanding(item: Mapping[str, Any], *, prompt_version: str, model_version: str) -> dict[str, Any]:
    objects = _unique_strings(item.get("detected_objects") or [])
    action = _first_text(item.get("canonical_action_type"), item.get("action_name"), "unknown_action")
    primary = _first_text(item.get("primary_object"), "unknown_object")
    hands_visible = any(token in {"hand", "hands", "gloved_hand", "glove", "gloves"} for token in objects)
    facts = [
        f"素材来自 {item.get('view') or 'unknown_view'} 视角。",
        f"预处理证据指向 {action}。",
    ]
    if primary and primary != "unknown_object":
        facts.append(f"主要对象为 {primary}。")
    if objects:
        facts.append(f"检测对象包括 {', '.join(objects[:8])}。")
    inferences = []
    if action != "unknown_action" and primary != "unknown_object":
        inferences.append(f"可能代表 {primary} 相关的 {action} 实验动作。")
    unresolved = []
    if not hands_visible and "hand" in action:
        unresolved.append("动作名涉及手部，但当前素材的对象列表未稳定包含手部。")
    unresolved.append("实验名、项目名、样品名、SOP 名需要人工上下文确认。")
    keywords = _unique_strings([action, primary, *objects, item.get("view"), item.get("asset_type")])
    confidence = _bounded(float(item.get("confidence") or 0.0) * 0.8 + (0.15 if objects else 0.0))
    return {
        "schema_version": SCHEMA_VERSION,
        "vlm_result_id": _stable_id("item-vlm", item.get("evidence_id")),
        "task_type": f"item_{item.get('asset_type') or 'material'}",
        "material_id": item.get("material_id") or "",
        "evidence_id": item.get("evidence_id") or "",
        "session_id": item.get("session_id") or "",
        "experiment_id": item.get("experiment_id") or "",
        "segment_id": item.get("segment_id") or "",
        "micro_segment_id": item.get("micro_segment_id") or "",
        "sha256": item.get("sha256") or "",
        "date": item.get("date") or "",
        "view": item.get("view") or "",
        "prompt_version": prompt_version,
        "model_version": model_version,
        "visual_scene_summary": "；".join(facts),
        "visible_objects": objects,
        "manipulated_objects": [primary] if primary and primary != "unknown_object" else [],
        "hands_visible": hands_visible,
        "operation_type": action,
        "possible_lab_action": inferences[0] if inferences else "",
        "possible_experiment_stage": "",
        "physical_change_observed": item.get("physical_event_type") or "",
        "instrument_or_container_state": _instrument_state(objects),
        "material_state": "",
        "safety_relevance": "",
        "evidence_for_action": facts,
        "evidence_against_action": [],
        "ambiguity_notes": unresolved,
        "confidence": confidence,
        "searchable_keywords": keywords,
        "memory_candidate_text": "",
        "strong_facts": facts if item.get("evidence_strength") == "strong" else facts[:2],
        "weak_inferences": inferences,
        "unresolved_questions": unresolved,
        "created_at": _now_iso(),
    }


def _build_evidence_bundles(
    evidence_items: Sequence[Mapping[str, Any]],
    item_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results_by_evidence = {row.get("evidence_id"): row for row in item_results}
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in evidence_items:
        key = _bundle_group_key(item)
        grouped.setdefault(key, []).append(item)

    bundles: list[dict[str, Any]] = []
    for group_key, items in sorted(grouped.items()):
        item_ids = [str(item.get("evidence_id") or "") for item in items]
        material_ids = _unique_strings(str(item.get("material_id") or "") for item in items)
        sha256s = _unique_strings(
            [
                *[item.get("sha256") for item in items],
                *[sha for item in items for sha in _ensure_list(item.get("sha256s"))],
            ]
        )
        bundle_sha256 = _aggregate_sha256(sha256s)
        item_vlm = [results_by_evidence.get(item.get("evidence_id")) for item in items if results_by_evidence.get(item.get("evidence_id"))]
        views = sorted({str(item.get("view") or "") for item in items if item.get("view")})
        keyframes = _unique_strings(
            [
                *[item.get("keyframe") for item in items],
                *[ref for item in items for ref in _ensure_list(item.get("keyframe_refs"))],
                *[item.get("keyframe_path") for item in items],
            ]
        )
        keyclips = _unique_strings(
            [
                *[item.get("keyclip") for item in items],
                *[ref for item in items for ref in _ensure_list(item.get("keyclip_refs"))],
                *[item.get("keyclip_path") for item in items],
            ]
        )
        starts = [_float(item.get("global_start_sec"), 0.0) or 0.0 for item in items]
        ends = [_float(item.get("global_end_sec"), 0.0) or 0.0 for item in items]
        timestamps = [item.get("time_range") for item in items if isinstance(item.get("time_range"), Mapping)]
        micro_segment_ids = _unique_strings(
            [
                *[item.get("micro_segment_id") for item in items],
                *[micro_id for item in items for micro_id in _ensure_list(item.get("micro_segment_ids"))],
            ]
        )
        yolo_count = sum(int(_float(item.get("yolo_evidence_count"), 0.0) or 0) for item in items)
        confidence = _bounded(
            _mean([float(item.get("confidence") or 0.0) for item in items])
            + (0.08 if len(views) >= 2 else 0.0)
            + min(yolo_count, 20) * 0.005
        )
        representative = items[0]
        bundle_id = _stable_id("bundle", {"group_key": group_key, "materials": sorted(material_ids)})
        strong_facts = _unique_strings([fact for row in item_vlm for fact in _ensure_list(row.get("strong_facts"))])
        weak_inferences = _unique_strings([fact for row in item_vlm for fact in _ensure_list(row.get("weak_inferences"))])
        unresolved = _unique_strings([fact for row in item_vlm for fact in _ensure_list(row.get("unresolved_questions"))])
        evidence_strength = _bundle_strength(items, views)
        bundles.append(
            {
                "schema_version": SCHEMA_VERSION,
                "bundle_id": bundle_id,
                "date": representative.get("date") or "",
                "session_id": representative.get("session_id") or "",
                "experiment_id": representative.get("experiment_id") or "",
                "segment_id": representative.get("segment_id") or "",
                "micro_segment_id": representative.get("micro_segment_id") or "",
                "micro_segment_ids": micro_segment_ids,
                "material_id": material_ids[0] if material_ids else "",
                "sha256": bundle_sha256,
                "sha256s": sha256s,
                "time_range": {"start_sec": min(starts) if starts else None, "end_sec": max(ends) if ends else None},
                "timestamp": {"start_sec": min(starts) if starts else None, "end_sec": max(ends) if ends else None},
                "timestamps": timestamps,
                "event_type": representative.get("physical_event_type") or representative.get("canonical_action_type") or "",
                "action_name": representative.get("action_name") or "",
                "canonical_action_type": representative.get("canonical_action_type") or "",
                "primary_object": representative.get("primary_object") or "",
                "evidence_item_ids": item_ids,
                "material_ids": material_ids,
                "material_sha256s": sha256s,
                "keyframe": keyframes[0] if keyframes else "",
                "keyframes": keyframes,
                "keyframe_refs": keyframes,
                "keyclip": keyclips[0] if keyclips else "",
                "keyclips": keyclips,
                "keyclip_refs": keyclips,
                "views": views,
                "yolo_summary": {
                    "yolo_evidence_count": yolo_count,
                    "detected_objects": _unique_strings([obj for item in items for obj in _ensure_list(item.get("detected_objects"))]),
                },
                "vlm_item_result_ids": [str(row.get("vlm_result_id") or "") for row in item_vlm],
                "strong_facts": strong_facts,
                "weak_inferences": weak_inferences,
                "unresolved_questions": unresolved,
                "confidence": confidence,
                "evidence_strength": evidence_strength,
                "evidence_trace": {
                    "bundle_id": bundle_id,
                    "evidence_item_ids": item_ids,
                    "material_ids": material_ids,
                    "sha256": bundle_sha256,
                    "sha256s": sha256s,
                    "micro_segment_id": representative.get("micro_segment_id") or "",
                    "micro_segment_ids": micro_segment_ids,
                    "keyframe": keyframes[0] if keyframes else "",
                    "keyframes": keyframes,
                    "keyframe_refs": keyframes,
                    "keyclip": keyclips[0] if keyclips else "",
                    "keyclips": keyclips,
                    "keyclip_refs": keyclips,
                    "timestamp": {"start_sec": min(starts) if starts else None, "end_sec": max(ends) if ends else None},
                    "timestamps": timestamps,
                    "session_id": representative.get("session_id") or "",
                    "session_ids": _unique_strings([item.get("session_id") for item in items]),
                    "experiment_id": representative.get("experiment_id") or "",
                    "experiment_ids": _unique_strings([item.get("experiment_id") for item in items]),
                    "trace_complete": bool(material_ids and (keyframes or keyclips) and timestamps),
                },
                "related_bundles": [],
                "created_at": _now_iso(),
            }
        )
    return bundles


def _build_bundle_vlm_results(
    db_path: Path,
    bundles: Sequence[Mapping[str, Any]],
    item_results: Sequence[Mapping[str, Any]],
    *,
    prompt_version: str,
    model_version: str,
    vlm_mode: str = VLM_MODE_OFFLINE,
    vlm_client: Any | None = None,
    max_real_vlm_bundles: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mode = _normalize_vlm_mode(vlm_mode)
    results, stats = enhance_bundles_sync(
        bundles,
        item_results,
        sqlite_path=db_path,
        prompt_version=prompt_version,
        offline_model_version=VLM_MODEL_VERSION if mode != VLM_MODE_REAL_QWEN_ASYNC or vlm_client is None else model_version,
        qwen_model=model_version,
        vlm_client=vlm_client,
        enable_qwen=mode == VLM_MODE_REAL_QWEN_ASYNC,
        reuse_qwen_cache=mode == VLM_MODE_REUSE_EXISTING,
        max_qwen_calls=max_real_vlm_bundles,
        fallback_factory=lambda bundle, item_by_id, prompt, model: _infer_bundle_understanding(
            bundle,
            item_by_id,
            prompt_version=prompt,
            model_version=model,
        ),
    )
    stats["reuse"] = stats.get("reused_existing", 0)
    stats["real"] = stats.get("qwen_calls", 0)
    stats["fallback"] = stats.get("offline_fallback", 0)
    stats["error"] = stats.get("errors", 0)
    return results, stats


def _infer_bundle_understanding(
    bundle: Mapping[str, Any],
    item_by_id: Mapping[str, Mapping[str, Any]],
    *,
    prompt_version: str,
    model_version: str,
) -> dict[str, Any]:
    item_rows = [item_by_id.get(row_id) for row_id in _ensure_list(bundle.get("vlm_item_result_ids")) if item_by_id.get(row_id)]
    facts = _unique_strings([fact for row in item_rows for fact in _ensure_list(row.get("strong_facts"))])
    weak = _unique_strings([fact for row in item_rows for fact in _ensure_list(row.get("weak_inferences"))])
    unresolved = _unique_strings([fact for row in item_rows for fact in _ensure_list(row.get("unresolved_questions"))])
    views = _ensure_list(bundle.get("views"))
    view_agreement = "dual_view_supported" if len(views) >= 2 else "single_view_only"
    if len(views) < 2:
        unresolved.append("当前 evidence bundle 只有单视角素材，双视角支持不足。")
    yolo_objects = _ensure_list((bundle.get("yolo_summary") or {}).get("detected_objects") if isinstance(bundle.get("yolo_summary"), Mapping) else [])
    merged = _join_text(
        [
            f"动作: {bundle.get('canonical_action_type') or bundle.get('action_name')}",
            f"对象: {bundle.get('primary_object')}",
            f"视角: {', '.join(views)}",
            f"检测对象: {', '.join(yolo_objects[:8])}",
        ]
    )
    confidence = _bounded(float(bundle.get("confidence") or 0.0) + (0.05 if view_agreement == "dual_view_supported" else -0.08))
    return {
        "schema_version": SCHEMA_VERSION,
        "vlm_result_id": _stable_id("bundle-vlm", bundle.get("bundle_id")),
        "task_type": "bundle_merge",
        "bundle_id": bundle.get("bundle_id") or "",
        "session_id": bundle.get("session_id") or "",
        "experiment_id": bundle.get("experiment_id") or "",
        "segment_id": bundle.get("segment_id") or "",
        "micro_segment_id": bundle.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(bundle.get("micro_segment_ids")),
        "material_id": bundle.get("material_id") or "",
        "material_ids": _ensure_list(bundle.get("material_ids")),
        "sha256": bundle.get("sha256") or "",
        "sha256s": _ensure_list(bundle.get("sha256s")),
        "keyframe": bundle.get("keyframe") or "",
        "keyframe_refs": _ensure_list(bundle.get("keyframe_refs") or bundle.get("keyframes")),
        "keyclip": bundle.get("keyclip") or "",
        "keyclip_refs": _ensure_list(bundle.get("keyclip_refs") or bundle.get("keyclips")),
        "timestamp": bundle.get("timestamp") or bundle.get("time_range") or {},
        "timestamps": _ensure_list(bundle.get("timestamps") or bundle.get("time_range")),
        "date": bundle.get("date") or "",
        "prompt_version": prompt_version,
        "model_version": model_version,
        "merged_scene_understanding": merged,
        "merged_action_understanding": weak[0] if weak else merged,
        "view_agreement": view_agreement,
        "view_conflict": "",
        "strong_facts": facts,
        "weak_inferences": weak,
        "unresolved_questions": unresolved,
        "confidence": confidence,
        "searchable_keywords": _unique_strings([bundle.get("canonical_action_type"), bundle.get("action_name"), bundle.get("primary_object"), *yolo_objects]),
        "created_at": _now_iso(),
    }


def _build_daily_event_ledgers(
    bundles: Sequence[Mapping[str, Any]],
    bundle_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result_by_bundle = {row.get("bundle_id"): row for row in bundle_results}
    ledgers: list[dict[str, Any]] = []
    for bundle in bundles:
        result = result_by_bundle.get(bundle.get("bundle_id"), {})
        date_value = str(bundle.get("date") or result.get("date") or _today_utc().isoformat())
        facts = _unique_strings([*_ensure_list(bundle.get("strong_facts")), *_ensure_list(result.get("strong_facts"))])
        weak = _unique_strings([*_ensure_list(bundle.get("weak_inferences")), *_ensure_list(result.get("weak_inferences"))])
        unresolved = _unique_strings([*_ensure_list(bundle.get("unresolved_questions")), *_ensure_list(result.get("unresolved_questions"))])
        ledger_id = _stable_id("ledger", {"bundle_id": bundle.get("bundle_id"), "date": date_value})
        material_ids = _ensure_list(bundle.get("material_ids"))
        sha256s = _ensure_list(bundle.get("sha256s"))
        keyframe_refs = _ensure_list(bundle.get("keyframe_refs") or bundle.get("keyframes"))
        keyclip_refs = _ensure_list(bundle.get("keyclip_refs") or bundle.get("keyclips"))
        timestamp = bundle.get("timestamp") or bundle.get("time_range") or {}
        timestamps = _ensure_list(bundle.get("timestamps") or timestamp)
        micro_segment_ids = _ensure_list(bundle.get("micro_segment_ids") or bundle.get("micro_segment_id"))
        ledgers.append(
            {
                "schema_version": SCHEMA_VERSION,
                "ledger_event_id": ledger_id,
                "date": date_value,
                "session_id": bundle.get("session_id") or "",
                "experiment_id": bundle.get("experiment_id") or "",
                "segment_id": bundle.get("segment_id") or "",
                "micro_segment_id": bundle.get("micro_segment_id") or "",
                "micro_segment_ids": micro_segment_ids,
                "material_id": material_ids[0] if material_ids else "",
                "material_ids": material_ids,
                "sha256": bundle.get("sha256") or (sha256s[0] if len(sha256s) == 1 else _aggregate_sha256(sha256s)),
                "sha256s": sha256s,
                "time_range": bundle.get("time_range") or {},
                "timestamp": timestamp,
                "timestamps": timestamps,
                "canonical_action_type": bundle.get("canonical_action_type") or "",
                "action_name": bundle.get("action_name") or "",
                "primary_object": bundle.get("primary_object") or "",
                "detected_objects": (bundle.get("yolo_summary") or {}).get("detected_objects", []) if isinstance(bundle.get("yolo_summary"), Mapping) else [],
                "evidence_bundle_ids": [bundle.get("bundle_id")],
                "keyframe": keyframe_refs[0] if keyframe_refs else "",
                "keyframe_refs": keyframe_refs,
                "keyclip": keyclip_refs[0] if keyclip_refs else "",
                "keyclip_refs": keyclip_refs,
                "views": bundle.get("views") or [],
                "strong_facts": facts,
                "weak_inferences": weak,
                "unresolved_questions": unresolved,
                "confidence": _bounded(_mean([float(bundle.get("confidence") or 0.0), float(result.get("confidence") or 0.0)])),
                "evidence_strength": bundle.get("evidence_strength") or "weak",
                "human_feedback_status": "unconfirmed",
                "evidence_trace": {
                    "ledger_event_id": ledger_id,
                    "evidence_bundle_ids": [bundle.get("bundle_id")],
                    "bundle_ids": [bundle.get("bundle_id")],
                    "material_id": material_ids[0] if material_ids else "",
                    "material_ids": material_ids,
                    "sha256": bundle.get("sha256") or (sha256s[0] if len(sha256s) == 1 else _aggregate_sha256(sha256s)),
                    "sha256s": sha256s,
                    "micro_segment_id": bundle.get("micro_segment_id") or "",
                    "micro_segment_ids": micro_segment_ids,
                    "keyframe": keyframe_refs[0] if keyframe_refs else "",
                    "keyframe_refs": keyframe_refs,
                    "keyclip": keyclip_refs[0] if keyclip_refs else "",
                    "keyclip_refs": keyclip_refs,
                    "timestamp": timestamp,
                    "timestamps": timestamps,
                    "trace_complete": bool(material_ids and (keyframe_refs or keyclip_refs) and timestamps),
                },
                "created_at": _now_iso(),
            }
        )
    return ledgers


def _build_memory_clusters(
    ledger_events: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
    feedback_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    bundle_by_id = {bundle.get("bundle_id"): bundle for bundle in bundles}
    clusters: list[dict[str, Any]] = []
    review_clusters: list[dict[str, Any]] = []
    for event in sorted(ledger_events, key=lambda row: (str(row.get("date") or ""), str(row.get("ledger_event_id") or ""))):
        signature = _ledger_signature(event)
        candidate = _new_cluster_from_event(event, signature, bundle_by_id)
        best_index = -1
        best_score: dict[str, Any] | None = None
        for index, cluster in enumerate(clusters):
            score = cluster_similarity(cluster.get("cluster_signature") or {}, signature)
            if best_score is None or score["score"] > best_score["score"]:
                best_score = score
                best_index = index
        if best_score and best_score["decision"] == "merge" and best_index >= 0:
            clusters[best_index] = _merge_cluster_event(clusters[best_index], event, signature, bundle_by_id, best_score)
        elif best_score and best_score["decision"] == "review":
            candidate["status"] = "needs_review"
            candidate["merge_review"] = best_score
            review_clusters.append(candidate)
        else:
            clusters.append(candidate)
    clusters.extend(review_clusters)
    clusters = _deduplicate_clusters_by_id(clusters)
    clusters = [_finalize_cluster(cluster) for cluster in clusters]
    return _apply_feedback_to_clusters(clusters, feedback_entries)


def _ledger_signature(event: Mapping[str, Any]) -> dict[str, Any]:
    action_tokens = _tokens(_join_text([event.get("canonical_action_type"), event.get("action_name")]))
    object_tokens = _tokens(_join_text([event.get("primary_object"), " ".join(_ensure_list(event.get("detected_objects")))]))
    instrument_tokens = sorted({token for token in object_tokens if token in INSTRUMENT_LABELS})
    visual_tokens = _tokens(_join_text([event.get("strong_facts"), event.get("weak_inferences")]))
    start = _float((event.get("time_range") or {}).get("start_sec") if isinstance(event.get("time_range"), Mapping) else None, 0.0) or 0.0
    end = _float((event.get("time_range") or {}).get("end_sec") if isinstance(event.get("time_range"), Mapping) else None, start) or start
    duration = max(0.0, end - start)
    return {
        "action_signature": sorted(set(action_tokens)),
        "object_signature": sorted(set(object_tokens)),
        "instrument_signature": sorted(set(instrument_tokens)),
        "visual_semantic_signature": sorted(set(visual_tokens[:20])),
        "temporal_signature": [_duration_bucket(duration)],
        "sequence_signature": sorted(set(action_tokens[:3])),
    }


def _new_cluster_from_event(
    event: Mapping[str, Any],
    signature: Mapping[str, Any],
    bundle_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cluster_id = _cluster_id(signature)
    bundle_ids = _ensure_list(event.get("evidence_bundle_ids"))
    bundles = [bundle_by_id.get(bundle_id) for bundle_id in bundle_ids if bundle_by_id.get(bundle_id)]
    material_ids = _unique_strings([material_id for bundle in bundles for material_id in _ensure_list(bundle.get("material_ids"))] + _ensure_list(event.get("material_ids")))
    sha256s = _unique_strings([sha for bundle in bundles for sha in _ensure_list(bundle.get("sha256s"))] + _ensure_list(event.get("sha256s")))
    keyframe_refs = _unique_strings([path for bundle in bundles for path in _ensure_list(bundle.get("keyframes") or bundle.get("keyframe_refs"))] + _ensure_list(event.get("keyframe_refs")))
    keyclip_refs = _unique_strings([path for bundle in bundles for path in _ensure_list(bundle.get("keyclips") or bundle.get("keyclip_refs"))] + _ensure_list(event.get("keyclip_refs")))
    micro_segment_ids = _unique_strings([event.get("micro_segment_id"), *_ensure_list(event.get("micro_segment_ids"))])
    time_ranges = _ensure_list(event.get("timestamps") or event.get("time_range"))
    return {
        "schema_version": SCHEMA_VERSION,
        "cluster_id": cluster_id,
        "window_start_date": "",
        "window_end_date": "",
        "cluster_type": _cluster_type(event),
        "cluster_title": _cluster_title(event),
        "cluster_summary": "",
        "cluster_signature": dict(signature),
        "canonical_actions": _ensure_list(signature.get("action_signature")),
        "key_objects": _ensure_list(signature.get("object_signature")),
        "key_instruments": _ensure_list(signature.get("instrument_signature")),
        "related_dates": _unique_strings([event.get("date")]),
        "related_sessions": _unique_strings([event.get("session_id")]),
        "related_experiments": _unique_strings([event.get("experiment_id")]),
        "ledger_event_ids": _unique_strings([event.get("ledger_event_id")]),
        "evidence_bundle_ids": _unique_strings(bundle_ids),
        "material_id": material_ids[0] if material_ids else "",
        "material_ids": material_ids,
        "sha256": _aggregate_sha256(sha256s),
        "sha256s": sha256s,
        "micro_segment_id": micro_segment_ids[0] if micro_segment_ids else "",
        "micro_segment_ids": micro_segment_ids,
        "keyframe": keyframe_refs[0] if keyframe_refs else "",
        "keyframe_refs": keyframe_refs,
        "keyclip": keyclip_refs[0] if keyclip_refs else "",
        "keyclip_refs": keyclip_refs,
        "timestamp": time_ranges[0] if time_ranges else {},
        "time_ranges": time_ranges,
        "timestamps": time_ranges,
        "occurrence_count": 1,
        "day_count": 1,
        "view_coverage": _unique_strings([view for bundle in bundles for view in _ensure_list(bundle.get("views"))]),
        "evidence_strength_distribution": _strength_distribution([event]),
        "strong_facts": _ensure_list(event.get("strong_facts")),
        "weak_inferences": _ensure_list(event.get("weak_inferences")),
        "unresolved_questions": _ensure_list(event.get("unresolved_questions")),
        "human_confirmed_fields": {},
        "confidence": float(event.get("confidence") or 0.0),
        "confirmation_status": "auto_inferred",
        "status": "candidate",
        "memory_write_status": "candidate",
        "evidence_trace": {},
        "query_ready_text": "",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def _merge_cluster_event(
    cluster: Mapping[str, Any],
    event: Mapping[str, Any],
    signature: Mapping[str, Any],
    bundle_by_id: Mapping[str, Mapping[str, Any]],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(cluster)
    bundle_ids = _ensure_list(event.get("evidence_bundle_ids"))
    bundles = [bundle_by_id.get(bundle_id) for bundle_id in bundle_ids if bundle_by_id.get(bundle_id)]
    merged["related_dates"] = _unique_strings([*_ensure_list(merged.get("related_dates")), event.get("date")])
    merged["related_sessions"] = _unique_strings([*_ensure_list(merged.get("related_sessions")), event.get("session_id")])
    merged["related_experiments"] = _unique_strings([*_ensure_list(merged.get("related_experiments")), event.get("experiment_id")])
    merged["ledger_event_ids"] = _unique_strings([*_ensure_list(merged.get("ledger_event_ids")), event.get("ledger_event_id")])
    merged["evidence_bundle_ids"] = _unique_strings([*_ensure_list(merged.get("evidence_bundle_ids")), *bundle_ids])
    merged["material_ids"] = _unique_strings([*_ensure_list(merged.get("material_ids")), *[material_id for bundle in bundles for material_id in _ensure_list(bundle.get("material_ids"))], *_ensure_list(event.get("material_ids"))])
    merged["material_id"] = (_ensure_list(merged.get("material_ids")) or [""])[0]
    merged["sha256s"] = _unique_strings([*_ensure_list(merged.get("sha256s")), *[sha for bundle in bundles for sha in _ensure_list(bundle.get("sha256s"))], *_ensure_list(event.get("sha256s"))])
    merged["sha256"] = _aggregate_sha256(_ensure_list(merged.get("sha256s")))
    merged["micro_segment_ids"] = _unique_strings([*_ensure_list(merged.get("micro_segment_ids")), event.get("micro_segment_id"), *_ensure_list(event.get("micro_segment_ids"))])
    merged["micro_segment_id"] = (_ensure_list(merged.get("micro_segment_ids")) or [""])[0]
    merged["keyframe_refs"] = _unique_strings([*_ensure_list(merged.get("keyframe_refs")), *[path for bundle in bundles for path in _ensure_list(bundle.get("keyframes") or bundle.get("keyframe_refs"))], *_ensure_list(event.get("keyframe_refs"))])
    merged["keyframe"] = (_ensure_list(merged.get("keyframe_refs")) or [""])[0]
    merged["keyclip_refs"] = _unique_strings([*_ensure_list(merged.get("keyclip_refs")), *[path for bundle in bundles for path in _ensure_list(bundle.get("keyclips") or bundle.get("keyclip_refs"))], *_ensure_list(event.get("keyclip_refs"))])
    merged["keyclip"] = (_ensure_list(merged.get("keyclip_refs")) or [""])[0]
    event_timestamps = _ensure_list(event.get("timestamps") or event.get("time_range"))
    merged["time_ranges"] = [*_ensure_list(merged.get("time_ranges")), *event_timestamps]
    merged["timestamps"] = _ensure_list(merged.get("time_ranges"))
    merged["timestamp"] = (_ensure_list(merged.get("timestamps")) or [{}])[0]
    merged["occurrence_count"] = len(_ensure_list(merged.get("ledger_event_ids")))
    merged["day_count"] = len(_ensure_list(merged.get("related_dates")))
    merged["view_coverage"] = _unique_strings([*_ensure_list(merged.get("view_coverage")), *[view for bundle in bundles for view in _ensure_list(bundle.get("views"))]])
    merged["strong_facts"] = _unique_strings([*_ensure_list(merged.get("strong_facts")), *_ensure_list(event.get("strong_facts"))])
    merged["weak_inferences"] = _unique_strings([*_ensure_list(merged.get("weak_inferences")), *_ensure_list(event.get("weak_inferences"))])
    merged["unresolved_questions"] = _unique_strings([*_ensure_list(merged.get("unresolved_questions")), *_ensure_list(event.get("unresolved_questions"))])
    merged["confidence"] = _bounded(_mean([float(merged.get("confidence") or 0.0), float(event.get("confidence") or 0.0), float(score.get("score") or 0.0)]))
    merged["last_merge_score"] = dict(score)
    merged["updated_at"] = _now_iso()
    return merged


def _deduplicate_clusters_by_id(clusters: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse same-signature clusters before snapshot/query publication."""
    merged_by_id: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id") or "")
        if not cluster_id:
            continue
        if cluster_id not in merged_by_id:
            merged_by_id[cluster_id] = dict(cluster)
            order.append(cluster_id)
            continue
        base = merged_by_id[cluster_id]
        for key in (
            "related_dates",
            "related_sessions",
            "related_experiments",
            "ledger_event_ids",
            "evidence_bundle_ids",
            "material_ids",
            "sha256s",
            "micro_segment_ids",
            "keyframe_refs",
            "keyclip_refs",
            "view_coverage",
            "strong_facts",
            "weak_inferences",
            "unresolved_questions",
        ):
            base[key] = _unique_strings([*_ensure_list(base.get(key)), *_ensure_list(cluster.get(key))])
        base["time_ranges"] = [*_ensure_list(base.get("time_ranges")), *_ensure_list(cluster.get("time_ranges"))]
        base["occurrence_count"] = len(_ensure_list(base.get("ledger_event_ids")))
        base["day_count"] = len(_ensure_list(base.get("related_dates")))
        base["confidence"] = _bounded(_mean([float(base.get("confidence") or 0.0), float(cluster.get("confidence") or 0.0)]))
        if cluster.get("status") == "needs_review" or base.get("status") == "needs_review":
                base["status"] = "needs_review"
        base["material_id"] = (_ensure_list(base.get("material_ids")) or [""])[0]
        base["sha256"] = _aggregate_sha256(_ensure_list(base.get("sha256s")))
        base["micro_segment_id"] = (_ensure_list(base.get("micro_segment_ids")) or [""])[0]
        base["keyframe"] = (_ensure_list(base.get("keyframe_refs")) or [""])[0]
        base["keyclip"] = (_ensure_list(base.get("keyclip_refs")) or [""])[0]
        base["timestamps"] = _ensure_list(base.get("time_ranges"))
        base["timestamp"] = (_ensure_list(base.get("timestamps")) or [{}])[0]
        base["updated_at"] = _now_iso()
    return [merged_by_id[cluster_id] for cluster_id in order]


def _finalize_cluster(cluster: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(cluster)
    count = int(item.get("occurrence_count") or 0)
    days = int(item.get("day_count") or 0)
    confidence = float(item.get("confidence") or 0.0)
    if item.get("status") != "needs_review":
        if count >= 3 or days >= 2:
            item["status"] = "primary" if confidence >= 0.68 else "active"
        elif confidence >= 0.60:
            item["status"] = "active"
        else:
            item["status"] = "candidate"
    item["cluster_summary"] = _cluster_summary(item)
    item["evidence_strength_distribution"] = _distribution_from_values(_ensure_list(item.get("evidence_strength_distribution")))
    item["material_ids"] = _ensure_list(item.get("material_ids"))
    item["material_id"] = item["material_ids"][0] if item["material_ids"] else ""
    item["sha256s"] = _ensure_list(item.get("sha256s"))
    item["sha256"] = _aggregate_sha256(item["sha256s"])
    item["micro_segment_ids"] = _ensure_list(item.get("micro_segment_ids"))
    item["micro_segment_id"] = item["micro_segment_ids"][0] if item["micro_segment_ids"] else ""
    item["keyframe_refs"] = _ensure_list(item.get("keyframe_refs"))
    item["keyframe"] = item["keyframe_refs"][0] if item["keyframe_refs"] else ""
    item["keyclip_refs"] = _ensure_list(item.get("keyclip_refs"))
    item["keyclip"] = item["keyclip_refs"][0] if item["keyclip_refs"] else ""
    item["timestamps"] = _ensure_list(item.get("timestamps") or item.get("time_ranges"))
    item["timestamp"] = item["timestamps"][0] if item["timestamps"] else {}
    item["evidence_trace"] = {
        "cluster_id": item.get("cluster_id"),
        "ledger_event_ids": item.get("ledger_event_ids") or [],
        "evidence_bundle_ids": item.get("evidence_bundle_ids") or [],
        "bundle_ids": item.get("evidence_bundle_ids") or [],
        "material_id": item.get("material_id") or "",
        "material_ids": item.get("material_ids") or [],
        "sha256": item.get("sha256") or "",
        "sha256s": item.get("sha256s") or [],
        "micro_segment_id": item.get("micro_segment_id") or "",
        "micro_segment_ids": item.get("micro_segment_ids") or [],
        "keyframe": item.get("keyframe") or "",
        "keyframes": item.get("keyframe_refs") or [],
        "keyframe_refs": item.get("keyframe_refs") or [],
        "keyclip": item.get("keyclip") or "",
        "keyclips": item.get("keyclip_refs") or [],
        "keyclip_refs": item.get("keyclip_refs") or [],
        "timestamp": item.get("timestamp") or {},
        "timestamps": item.get("timestamps") or [],
        "trace_complete": bool(item.get("material_ids") and item.get("evidence_bundle_ids") and (item.get("keyframe_refs") or item.get("keyclip_refs")) and item.get("timestamps")),
    }
    item["query_ready_text"] = _join_text(
        [
            item.get("cluster_title"),
            item.get("cluster_summary"),
            " ".join(_ensure_list(item.get("canonical_actions"))),
            " ".join(_ensure_list(item.get("key_objects"))),
            " ".join(_ensure_list(item.get("key_instruments"))),
            " ".join(_ensure_list(item.get("strong_facts"))),
        ]
    )
    item["memory_write_status"] = "included" if item["status"] in {"active", "primary", "human_confirmed"} else item["status"]
    item["updated_at"] = _now_iso()
    return item


def _apply_feedback_to_clusters(
    clusters: Sequence[Mapping[str, Any]],
    feedback_entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    feedback_by_target: dict[str, list[Mapping[str, Any]]] = {}
    for entry in feedback_entries:
        feedback_by_target.setdefault(str(entry.get("target_id") or ""), []).append(entry)
    updated: list[dict[str, Any]] = []
    for cluster in clusters:
        item = dict(cluster)
        entries = feedback_by_target.get(str(item.get("cluster_id") or ""), [])
        confirmed_fields: dict[str, Any] = dict(item.get("human_confirmed_fields") or {})
        for entry in entries:
            feedback_type = str(entry.get("feedback_type") or "")
            if feedback_type in {"confirm", "accepted", "accept"}:
                item["status"] = "human_confirmed"
                item["confirmation_status"] = "human_confirmed"
                item["memory_write_status"] = "included"
            elif feedback_type in {"reject", "rejected"}:
                item["status"] = "human_rejected"
                item["confirmation_status"] = "human_rejected"
                item["memory_write_status"] = "excluded"
            elif feedback_type in {"needs_review", "uncertain"} and item.get("status") not in {"human_confirmed", "human_rejected"}:
                item["status"] = "needs_review"
                item["confirmation_status"] = "needs_review"
            for key, value in dict(entry.get("context_fields") or {}).items():
                if value not in (None, ""):
                    confirmed_fields[str(key)] = value
        item["human_confirmed_fields"] = confirmed_fields
        updated.append(item)
    return updated


def _build_memory_snapshot(
    clusters: Sequence[Mapping[str, Any]],
    ledger_events: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    job_id: str,
    window_start: date,
    window_end: date,
    window_days: int,
    job_type: str,
    prompt_version: str,
    vlm_model_version: str,
    vlm_mode: str,
    item_vlm_model: str,
    bundle_vlm_model: str,
    cluster_algorithm_version: str,
    feedback_entries: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    available_dates = sorted({str(row.get("date") or "") for row in ledger_events if row.get("date")})
    days_available = len(available_dates)
    completeness = "complete" if days_available >= window_days else "partial" if days_available > 0 else "empty"
    clusters_by_status = {
        status: [cluster for cluster in clusters if cluster.get("status") == status]
        for status in ("primary", "human_confirmed", "active", "candidate", "needs_review")
    }
    primary_clusters = [*_ensure_list(clusters_by_status.get("human_confirmed")), *_ensure_list(clusters_by_status.get("primary"))]
    if completeness != "complete":
        primary_clusters = [cluster for cluster in primary_clusters if cluster.get("status") == "human_confirmed"]
    top_actions = _top_values([action for cluster in clusters for action in _ensure_list(cluster.get("canonical_actions"))])
    top_objects = _top_values([obj for cluster in clusters for obj in _ensure_list(cluster.get("key_objects"))])
    top_instruments = _top_values([obj for cluster in clusters for obj in _ensure_list(cluster.get("key_instruments"))])
    workflow_context = build_workflow_context_reasoning(
        clusters=clusters,
        ledger_events=ledger_events,
        bundles=bundles,
        human_feedback_entries=feedback_entries or [],
        instrument_labels=INSTRUMENT_LABELS,
    )
    workflow_context = _apply_partial_window_reasoning_guard(
        workflow_context,
        completeness=completeness,
        days_available=days_available,
        window_days=window_days,
    )
    unresolved = _unique_strings(
        [
            *[q for cluster in clusters for q in _ensure_list(cluster.get("unresolved_questions"))],
            *_ensure_list(workflow_context.get("unresolved_questions")),
        ]
    )
    legacy_confirmed_contexts = [
        {"cluster_id": cluster.get("cluster_id"), **dict(cluster.get("human_confirmed_fields") or {})}
        for cluster in clusters
        if cluster.get("human_confirmed_fields")
    ]
    confirmed_contexts = _ensure_list(workflow_context.get("human_confirmed_contexts")) or legacy_confirmed_contexts
    snapshot_content_hash = workflow_reasoning_fingerprint(
        {
            "clusters": [
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "status": cluster.get("status"),
                    "confirmation_status": cluster.get("confirmation_status"),
                    "ledger_event_ids": cluster.get("ledger_event_ids"),
                    "evidence_bundle_ids": cluster.get("evidence_bundle_ids"),
                    "material_ids": cluster.get("material_ids"),
                    "human_confirmed_fields": cluster.get("human_confirmed_fields"),
                }
                for cluster in clusters
            ],
            "ledger_event_ids": [event.get("ledger_event_id") for event in ledger_events],
            "bundle_ids": [bundle.get("bundle_id") for bundle in bundles],
            "workflow_context": workflow_context,
        }
    )
    snapshot_id = _stable_id(
        "snapshot",
        {
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
            "clusters": [cluster.get("cluster_id") for cluster in clusters],
            "job_type": job_type,
            "prompt_version": prompt_version,
            "model": vlm_model_version,
            "cluster": cluster_algorithm_version,
            "content_hash": snapshot_content_hash,
        },
    )
    overview_prefix = (
        f"基于最近 {days_available}/{window_days} 天可用数据"
        if completeness != "complete"
        else f"基于完整最近 {window_days} 天数据"
    )
    partial_window_notice = (
        ""
        if completeness == "complete"
        else (
            f"Partial window only: available material dates cover {days_available}/{window_days} days; "
            "this snapshot must not be presented as a complete 30-day memory."
        )
    )
    snapshot_material_ids = _unique_strings([material_id for bundle in bundles for material_id in _ensure_list(bundle.get("material_ids"))])
    snapshot_sha256s = _unique_strings([sha for bundle in bundles for sha in _ensure_list(bundle.get("sha256s"))])
    snapshot_micro_segment_ids = _unique_strings(
        [row.get("micro_segment_id") for row in ledger_events]
        + [micro_id for row in ledger_events for micro_id in _ensure_list(row.get("micro_segment_ids"))]
        + [micro_id for bundle in bundles for micro_id in _ensure_list(bundle.get("micro_segment_ids"))]
    )
    snapshot_keyframe_refs = _unique_strings(
        [path for bundle in bundles for path in _ensure_list(bundle.get("keyframe_refs") or bundle.get("keyframes"))]
        + [path for row in ledger_events for path in _ensure_list(row.get("keyframe_refs"))]
    )
    snapshot_keyclip_refs = _unique_strings(
        [path for bundle in bundles for path in _ensure_list(bundle.get("keyclip_refs") or bundle.get("keyclips"))]
        + [path for row in ledger_events for path in _ensure_list(row.get("keyclip_refs"))]
    )
    snapshot_timestamps = [row.get("timestamp") or row.get("time_range") for row in ledger_events if row.get("timestamp") or row.get("time_range")]
    snapshot_ledger_ids = _unique_strings(row.get("ledger_event_id") for row in ledger_events)
    snapshot_bundle_ids = _unique_strings(bundle.get("bundle_id") for bundle in bundles)
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "snapshot_type": "rolling_30_day" if completeness == "complete" else "partial_window",
        "snapshot_kind": "complete" if completeness == "complete" else "partial",
        "is_partial": completeness != "complete",
        "is_partial_window": completeness != "complete",
        "is_full_30_day_memory": completeness == "complete" and int(window_days) == WINDOW_DAYS,
        "partial_window_notice": partial_window_notice,
        "window_start_date": window_start.isoformat(),
        "window_end_date": window_end.isoformat(),
        "window_days_expected": int(window_days),
        "window_days_available": days_available,
        "window_completeness": completeness,
        "window_completeness_ratio": f"{days_available}/{window_days}",
        "partial_window_reason": "" if completeness == "complete" else "available_material_dates_less_than_expected_window",
        "window_scope": {
            "window_start_date": window_start.isoformat(),
            "window_end_date": window_end.isoformat(),
            "window_days_expected": int(window_days),
            "window_days_available": days_available,
            "window_completeness": completeness,
            "window_completeness_ratio": f"{days_available}/{window_days}",
            "available_dates": available_dates,
            "partial_window_reason": "" if completeness == "complete" else "available_material_dates_less_than_expected_window",
            "partial_window_notice": partial_window_notice,
        },
        "generated_at": _now_iso(),
        "generation_reason": job_type,
        "source_job_id": job_id,
        "source_ledger_event_count": len(ledger_events),
        "source_bundle_count": len(bundles),
        "source_session_count": len({row.get("session_id") for row in ledger_events if row.get("session_id")}),
        "source_experiment_count": len({row.get("experiment_id") for row in ledger_events if row.get("experiment_id")}),
        "source_material_count": len({material_id for bundle in bundles for material_id in _ensure_list(bundle.get("material_ids"))}),
        "cluster_ids": _unique_strings([cluster.get("cluster_id") for cluster in clusters]),
        "ledger_event_ids": snapshot_ledger_ids,
        "evidence_bundle_ids": snapshot_bundle_ids,
        "material_ids": snapshot_material_ids,
        "sha256": _aggregate_sha256(snapshot_sha256s),
        "sha256s": snapshot_sha256s,
        "micro_segment_ids": snapshot_micro_segment_ids,
        "keyframe_refs": snapshot_keyframe_refs,
        "keyclip_refs": snapshot_keyclip_refs,
        "timestamps": snapshot_timestamps,
        "primary_clusters": [cluster.get("cluster_id") for cluster in primary_clusters],
        "activity_overview": _activity_overview(overview_prefix, primary_clusters, clusters, completeness),
        "top_actions": top_actions,
        "top_objects": top_objects,
        "top_instruments": top_instruments,
        "recurring_workflow_patterns": [] if completeness != "complete" else _workflow_patterns(primary_clusters),
        "workflow_reasoning_schema_version": workflow_context.get("schema_version"),
        "context_reasoning_schema_version": workflow_context.get("context_schema_version"),
        "workflow_patterns": _ensure_list(workflow_context.get("workflow_patterns")),
        "instrument_usage_patterns": _ensure_list(workflow_context.get("instrument_usage_patterns")),
        "project_hints": _ensure_list(workflow_context.get("project_hints")),
        "project_or_context_hints": _ensure_list(workflow_context.get("project_or_context_hints")),
        "sop_context_candidates": _ensure_list(workflow_context.get("step_reasoning_candidates")),
        "step_reasoning_candidates": _ensure_list(workflow_context.get("step_reasoning_candidates")),
        "process_completion_candidates": _ensure_list(workflow_context.get("process_completion_candidates")),
        "rule_candidates": _ensure_list(workflow_context.get("rule_candidates")),
        "reminder_candidates": _ensure_list(workflow_context.get("reminder_candidates")),
        "reminder_rule_candidates": _ensure_list(workflow_context.get("rule_candidates")) + _ensure_list(workflow_context.get("reminder_candidates")),
        "unresolved_questions": unresolved[:50],
        "human_confirmed_contexts": confirmed_contexts,
        "snapshot_content_hash": snapshot_content_hash,
        "high_confidence_facts": _snapshot_high_confidence_facts(primary_clusters, completeness),
        "weak_inferences": _unique_strings([inf for cluster in clusters for inf in _ensure_list(cluster.get("weak_inferences"))])[:50],
        "representative_evidence": _representative_evidence(primary_clusters or clusters),
        "query_ready_text": _join_text([cluster.get("query_ready_text") for cluster in clusters]),
        "evidence_trace_index": {
            str(cluster.get("cluster_id")): cluster.get("evidence_trace") or {}
            for cluster in clusters
        },
        "evidence_trace": {
            "cluster_ids": _unique_strings([cluster.get("cluster_id") for cluster in clusters]),
            "ledger_event_ids": snapshot_ledger_ids,
            "evidence_bundle_ids": snapshot_bundle_ids,
            "bundle_ids": snapshot_bundle_ids,
            "material_ids": snapshot_material_ids,
            "sha256": _aggregate_sha256(snapshot_sha256s),
            "sha256s": snapshot_sha256s,
            "micro_segment_ids": snapshot_micro_segment_ids,
            "keyframe_refs": snapshot_keyframe_refs,
            "keyclip_refs": snapshot_keyclip_refs,
            "timestamps": snapshot_timestamps,
            "trace_complete": bool(snapshot_bundle_ids and snapshot_material_ids and (snapshot_keyframe_refs or snapshot_keyclip_refs) and snapshot_timestamps),
        },
        "model_versions": {
            "vlm_model_version": vlm_model_version,
            "vlm_mode": vlm_mode,
            "item_vlm_model": item_vlm_model,
            "bundle_vlm_model": bundle_vlm_model,
            "prompt_version": prompt_version,
            "cluster_algorithm_version": cluster_algorithm_version,
            "schema_version": SCHEMA_VERSION,
        },
        "quality_metrics": {
            "cluster_count": len(clusters),
            "primary_cluster_count": len(primary_clusters),
            "ledger_event_count": len(ledger_events),
            "dual_view_bundle_count": len([bundle for bundle in bundles if len(_ensure_list(bundle.get("views"))) >= 2]),
            "workflow_pattern_count": len(_ensure_list(workflow_context.get("workflow_patterns"))),
            "instrument_usage_pattern_count": len(_ensure_list(workflow_context.get("instrument_usage_patterns"))),
            "human_confirmed_context_count": len(confirmed_contexts),
            "rule_candidate_count": len(_ensure_list(workflow_context.get("rule_candidates"))),
            "reminder_candidate_count": len(_ensure_list(workflow_context.get("reminder_candidates"))),
            "partial_window": completeness != "complete",
        },
        "sqlite_path": str(video_memory_db_path(root)),
        "created_at": _now_iso(),
    }


def _cluster_to_answer_claim(
    cluster: Mapping[str, Any],
    *,
    ledgers: Mapping[str, Mapping[str, Any]],
    bundles: Mapping[str, Mapping[str, Any]],
    filters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ledger_ids = _ensure_list(cluster.get("ledger_event_ids"))
    bundle_ids = _ensure_list(cluster.get("evidence_bundle_ids"))
    related_ledgers = [ledgers.get(row_id) for row_id in ledger_ids if ledgers.get(row_id)]
    related_bundles = [bundles.get(row_id) for row_id in bundle_ids if bundles.get(row_id)]
    if filters:
        related_ledgers = [row for row in related_ledgers if _memory_payload_matches_filters(row, filters)]
        related_bundles = [row for row in related_bundles if _memory_payload_matches_filters(row, filters)]
        ledger_ids = _unique_strings([row.get("ledger_event_id") for row in related_ledgers])
        bundle_ids = _unique_strings([row.get("bundle_id") for row in related_bundles])
    all_material_ids = _unique_strings(
        [material_id for bundle in related_bundles for material_id in _ensure_list(bundle.get("material_ids"))]
        + _ensure_list(cluster.get("material_ids"))
    )
    all_sha256s = _unique_strings(
        [sha for bundle in related_bundles for sha in _ensure_list(bundle.get("sha256s"))]
        + _ensure_list(cluster.get("sha256s"))
    )
    all_keyframes = _unique_strings(
        [path for bundle in related_bundles for path in _ensure_list(bundle.get("keyframes") or bundle.get("keyframe_refs"))]
        + _ensure_list(cluster.get("keyframe_refs"))
    )
    all_keyclips = _unique_strings(
        [path for bundle in related_bundles for path in _ensure_list(bundle.get("keyclips") or bundle.get("keyclip_refs"))]
        + _ensure_list(cluster.get("keyclip_refs"))
    )
    time_ranges = [row.get("time_range") for row in related_ledgers if row.get("time_range")]
    if not time_ranges:
        time_ranges = [row for row in _ensure_list(cluster.get("time_ranges")) if row]
    first_ledger = related_ledgers[0] if related_ledgers else {}
    first_bundle = related_bundles[0] if related_bundles else {}
    first_time_range = time_ranges[0] if time_ranges else {}
    session_ids = _unique_strings(
        [row.get("session_id") for row in related_ledgers]
        + [row.get("session_id") for row in related_bundles]
        + _ensure_list(cluster.get("related_sessions"))
    )
    experiment_ids = _unique_strings(
        [row.get("experiment_id") for row in related_ledgers]
        + [row.get("experiment_id") for row in related_bundles]
        + _ensure_list(cluster.get("related_experiments"))
    )
    segment_ids = _unique_strings([row.get("segment_id") for row in related_ledgers] + [row.get("segment_id") for row in related_bundles])
    micro_segment_ids = _unique_strings(
        [row.get("micro_segment_id") for row in related_ledgers]
        + [micro_id for row in related_ledgers for micro_id in _ensure_list(row.get("micro_segment_ids"))]
        + [row.get("micro_segment_id") for row in related_bundles]
        + [micro_id for row in related_bundles for micro_id in _ensure_list(row.get("micro_segment_ids"))]
        + _ensure_list(cluster.get("micro_segment_ids"))
    )
    evidence_links = [
        {
            "evidence_bundle_id": bundle.get("bundle_id") or "",
            "material_ids": _ensure_list(bundle.get("material_ids")),
            "sha256": bundle.get("sha256") or "",
            "sha256s": _ensure_list(bundle.get("sha256s")),
            "session_id": bundle.get("session_id") or "",
            "experiment_id": bundle.get("experiment_id") or "",
            "segment_id": bundle.get("segment_id") or "",
            "micro_segment_id": bundle.get("micro_segment_id") or "",
            "micro_segment_ids": _ensure_list(bundle.get("micro_segment_ids") or bundle.get("micro_segment_id")),
            "timestamp": bundle.get("time_range") or {},
            "timestamps": _ensure_list(bundle.get("timestamps") or bundle.get("time_range")),
            "keyframes": _ensure_list(bundle.get("keyframes") or bundle.get("keyframe_refs")),
            "keyclips": _ensure_list(bundle.get("keyclips") or bundle.get("keyclip_refs")),
            "confidence": float(bundle.get("confidence") or 0.0),
            "human_confirmation_status": cluster.get("confirmation_status") or "unconfirmed",
        }
        for bundle in related_bundles
    ]
    has_evidence_bundle = bool(related_bundles)
    if cluster.get("status") == "human_confirmed" and has_evidence_bundle:
        claim_type = "human_confirmed_context"
    elif cluster.get("status") in {"primary", "active"} and has_evidence_bundle:
        claim_type = "fact"
    else:
        claim_type = "inference"
    confidence = float(cluster.get("confidence") or 0.0)
    if not has_evidence_bundle:
        confidence = min(confidence, 0.49)
    return {
        "claim_id": _stable_id("claim", {"cluster_id": cluster.get("cluster_id"), "ledgers": ledger_ids}),
        "claim_schema_version": "video_memory.claim.v1",
        "claim_text": cluster.get("cluster_summary") or cluster.get("cluster_title") or "",
        "claim_type": claim_type,
        "cluster_id": cluster.get("cluster_id") or "",
        "cluster_ids": _unique_strings([cluster.get("cluster_id")]),
        "ledger_event_id": ledger_ids[0] if ledger_ids else "",
        "ledger_event_ids": ledger_ids,
        "evidence_bundle_id": bundle_ids[0] if bundle_ids else "",
        "evidence_bundle_ids": bundle_ids,
        "session_id": first_ledger.get("session_id") or first_bundle.get("session_id") or (session_ids or [""])[0],
        "session_ids": session_ids,
        "experiment_id": first_ledger.get("experiment_id") or first_bundle.get("experiment_id") or (experiment_ids or [""])[0],
        "experiment_ids": experiment_ids,
        "segment_id": first_ledger.get("segment_id") or first_bundle.get("segment_id") or "",
        "segment_ids": segment_ids,
        "micro_segment_id": first_ledger.get("micro_segment_id") or first_bundle.get("micro_segment_id") or "",
        "micro_segment_ids": micro_segment_ids,
        "material_id": all_material_ids[0] if all_material_ids else "",
        "material_ids": all_material_ids,
        "sha256": all_sha256s[0] if len(all_sha256s) == 1 else _aggregate_sha256(all_sha256s),
        "sha256s": all_sha256s,
        "keyframe": all_keyframes[0] if all_keyframes else "",
        "keyframe_refs": all_keyframes,
        "keyclip": all_keyclips[0] if all_keyclips else "",
        "keyclip_refs": all_keyclips,
        "timestamp": first_time_range,
        "timestamps": time_ranges,
        "supporting_objects": _ensure_list(cluster.get("key_objects")),
        "supporting_actions": _ensure_list(cluster.get("canonical_actions")),
        "confidence": confidence,
        "support": "supported" if has_evidence_bundle else "unsupported_no_evidence_bundle",
        "has_evidence_bundle": has_evidence_bundle,
        "fact_status": "bundle_supported_fact" if has_evidence_bundle else "not_a_strong_fact_without_evidence_bundle",
        "evidence_strength": "strong" if has_evidence_bundle and cluster.get("status") in {"primary", "human_confirmed"} else "medium" if has_evidence_bundle else "weak",
        "human_confirmation_status": cluster.get("confirmation_status") or "unconfirmed",
        "evidence_links": evidence_links,
        "evidence_trace": {
            "cluster_id": cluster.get("cluster_id") or "",
            "ledger_event_ids": ledger_ids,
            "evidence_bundle_ids": bundle_ids,
            "bundle_ids": bundle_ids,
            "material_id": all_material_ids[0] if all_material_ids else "",
            "material_ids": all_material_ids,
            "sha256": all_sha256s[0] if len(all_sha256s) == 1 else _aggregate_sha256(all_sha256s),
            "sha256s": all_sha256s,
            "micro_segment_id": micro_segment_ids[0] if micro_segment_ids else "",
            "micro_segment_ids": micro_segment_ids,
            "keyframe": all_keyframes[0] if all_keyframes else "",
            "keyframes": all_keyframes,
            "keyframe_refs": all_keyframes,
            "keyclip": all_keyclips[0] if all_keyclips else "",
            "keyclips": all_keyclips,
            "keyclip_refs": all_keyclips,
            "timestamp": first_time_range,
            "timestamps": time_ranges,
            "session_ids": session_ids,
            "experiment_ids": experiment_ids,
            "trace_complete": bool(has_evidence_bundle and all_material_ids and (all_keyframes or all_keyclips) and time_ranges),
        },
        "why_this_supports_claim": "该结论来自 30-Day Memory Cluster，并绑定 Daily Event Ledger、Evidence Bundle 与真实关键素材引用。",
        "limitations": "未人工确认的实验名、项目名、样品名、SOP 名不作为事实返回。",
    }


def _cache_lookup(
    db_path: Path,
    *,
    asset_sha256: str,
    material_id: str,
    prompt_version: str,
    model_version: str,
    task_type: str,
    input_context_hash: str,
) -> dict[str, Any] | None:
    cache_key = _cache_key(asset_sha256, material_id, prompt_version, model_version, task_type, input_context_hash)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT payload_json FROM vlm_result_cache WHERE cache_id = ? AND status != 'invalidated'",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["hit_count"] = int(payload.get("hit_count") or 0) + 1
        payload["last_used_at"] = _now_iso()
        _upsert_rows(db_path, "vlm_result_cache", [payload], key="cache_id")
        return payload
    finally:
        conn.close()


def _cache_store(
    db_path: Path,
    *,
    asset_sha256: str,
    material_id: str,
    asset_type: str,
    asset_path: str,
    prompt_version: str,
    model_version: str,
    task_type: str,
    input_context_hash: str,
    result: Mapping[str, Any],
    confidence: float,
) -> dict[str, Any]:
    cache_id = _cache_key(asset_sha256, material_id, prompt_version, model_version, task_type, input_context_hash)
    row = {
        "cache_id": cache_id,
        "schema_version": SCHEMA_VERSION,
        "cache_key": cache_id,
        "asset_sha256": asset_sha256,
        "material_id": material_id,
        "asset_type": asset_type,
        "asset_path": asset_path,
        "prompt_version": prompt_version,
        "model_version": model_version,
        "vlm_task_type": task_type,
        "input_context_hash": input_context_hash,
        "result_json": dict(result),
        "confidence": confidence,
        "status": "active",
        "created_at": _now_iso(),
        "last_used_at": _now_iso(),
        "hit_count": 0,
        "invalidated_at": "",
        "invalidate_reason": "",
    }
    _upsert_rows(db_path, "vlm_result_cache", [row], key="cache_id")
    return row


def _cache_key(
    asset_sha256: str,
    material_id: str,
    prompt_version: str,
    model_version: str,
    task_type: str,
    input_context_hash: str,
) -> str:
    return _stable_id(
        "vlm-cache",
        {
            "asset_sha256": asset_sha256,
            "material_id": material_id,
            "prompt_version": prompt_version,
            "model_version": model_version,
            "task_type": task_type,
            "input_context_hash": input_context_hash,
        },
    )


def _safe_cache_filename(cache_key: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(cache_key))[:180]


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _strip_absolute_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and re.match(r"^[A-Za-z]:\\", item):
                cleaned[str(key)] = Path(item).name
            else:
                cleaned[str(key)] = _strip_absolute_paths(item)
        return cleaned
    if isinstance(value, list):
        return [_strip_absolute_paths(item) for item in value]
    return value


def _strip_unserializable_client(value: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(value)
    if cleaned.get("vlm_client") is not None:
        cleaned["vlm_client"] = f"<{type(cleaned['vlm_client']).__name__}>"
    return cleaned


def _feedback_operation(operation: str, target_id: str, feedback_id: str, now: str | datetime | None) -> dict[str, Any]:
    return {
        "operation": operation,
        "target_id": str(target_id or "unknown"),
        "audit": {
            "feedback_ids": [feedback_id],
            "created_at": str(now or _now_iso()),
        },
    }


def _cluster_for_evidence(snapshot: Mapping[str, Any], evidence_id: str) -> str:
    for item in _ensure_list(snapshot.get("evidence_items")):
        if isinstance(item, Mapping) and item.get("evidence_id") == evidence_id:
            return str(item.get("cluster_id") or "unknown")
    return "unknown"


def _replace_rows(path: Path, table: str, rows: Sequence[Mapping[str, Any]], *, key: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(f"DELETE FROM {table}")
        for row in rows:
            _upsert_row_conn(conn, table, row, key=key)
        conn.commit()
    finally:
        conn.close()


def _upsert_rows(path: Path, table: str, rows: Sequence[Mapping[str, Any]], *, key: str) -> None:
    conn = sqlite3.connect(str(path))
    try:
        for row in rows:
            _upsert_row_conn(conn, table, row, key=key)
        conn.commit()
    finally:
        conn.close()


def _upsert_row_conn(conn: sqlite3.Connection, table: str, row: Mapping[str, Any], *, key: str) -> None:
    payload = dict(row)
    now = _now_iso()
    conn.execute(
        f"""
        INSERT OR REPLACE INTO {table}
        ({key}, schema_version, material_id, session_id, experiment_id, segment_id, micro_segment_id,
         sha256, date, status, confidence, payload_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(payload.get(key) or ""),
            str(payload.get("schema_version") or SCHEMA_VERSION),
            str(payload.get("material_id") or ""),
            str(payload.get("session_id") or ""),
            str(payload.get("experiment_id") or ""),
            str(payload.get("segment_id") or ""),
            str(payload.get("micro_segment_id") or ""),
            str(payload.get("sha256") or payload.get("asset_sha256") or ""),
            str(payload.get("date") or payload.get("window_end_date") or ""),
            str(payload.get("status") or payload.get("job_status") or payload.get("memory_write_status") or ""),
            _float(payload.get("confidence")),
            _json_dumps(payload),
            str(payload.get("created_at") or now),
            now,
        ),
    )


def _load_table(path: Path, table: str) -> list[dict[str, Any]]:
    ensure_video_memory_schema(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(f"SELECT payload_json FROM {table}").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]
    finally:
        conn.close()


def _load_cache_rows(path: Path) -> list[dict[str, Any]]:
    return _load_table(path, "vlm_result_cache")


def _write_outputs(root: Path, payloads: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in payloads.items():
        filename = JSONL_FILES.get(name)
        if not filename:
            continue
        write_jsonl(root / filename, list(rows))


def _bundle_group_key(item: Mapping[str, Any]) -> str:
    if item.get("micro_segment_id"):
        return f"micro:{item.get('session_id')}:{item.get('micro_segment_id')}"
    start = int((_float(item.get("global_start_sec"), 0.0) or 0.0) // 2)
    return "|".join(
        [
            "fallback",
            str(item.get("session_id") or ""),
            str(item.get("segment_id") or ""),
            str(item.get("canonical_action_type") or item.get("action_name") or ""),
            str(item.get("primary_object") or ""),
            str(start),
        ]
    )


def _cluster_id(signature: Mapping[str, Any]) -> str:
    core = {
        "action": signature.get("action_signature"),
        "object": signature.get("object_signature"),
        "instrument": signature.get("instrument_signature"),
    }
    return _stable_id("cluster", core)


def _cluster_type(event: Mapping[str, Any]) -> str:
    instruments = {token for token in _tokens(_join_text([event.get("primary_object"), event.get("detected_objects")])) if token in INSTRUMENT_LABELS}
    if instruments:
        return "instrument_usage_pattern"
    if event.get("canonical_action_type"):
        return "recurring_action"
    return "unresolved_context"


def _cluster_title(event: Mapping[str, Any]) -> str:
    action = _first_text(event.get("canonical_action_type"), event.get("action_name"), "实验物理动作")
    primary = _first_text(event.get("primary_object"), "")
    if primary:
        return f"{primary} 相关 {action}"
    return action


def _cluster_summary(cluster: Mapping[str, Any]) -> str:
    count = int(cluster.get("occurrence_count") or 0)
    days = int(cluster.get("day_count") or 0)
    actions = ", ".join(_ensure_list(cluster.get("canonical_actions"))[:4])
    objects = ", ".join(_ensure_list(cluster.get("key_objects"))[:6])
    return f"{cluster.get('cluster_title')} 在当前窗口出现 {count} 次，覆盖 {days} 天。动作特征: {actions or '未归一'}；对象特征: {objects or '未归一'}。"


def _strength_distribution(events: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(event.get("evidence_strength") or "weak") for event in events]


def _distribution_from_values(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if isinstance(value, Mapping):
            for key, count in value.items():
                counts[str(key)] = counts.get(str(key), 0) + int(count or 0)
        else:
            key = str(value or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _bundle_strength(items: Sequence[Mapping[str, Any]], views: Sequence[str]) -> str:
    strengths = [str(item.get("evidence_strength") or "weak") for item in items]
    if "strong" in strengths and len(views) >= 2:
        return "strong"
    if "strong" in strengths or "medium" in strengths:
        return "medium"
    return "weak"


def _evidence_strength(row: Mapping[str, Any]) -> str:
    yolo = int(_float(row.get("yolo_evidence_count"), 0.0) or 0)
    quality = float(_float(row.get("quality_score"), 0.0) or 0.0)
    if yolo >= 5 and quality >= 0.65:
        return "strong"
    if yolo >= 1 or quality >= 0.45:
        return "medium"
    return "weak"


def _snapshot_high_confidence_facts(clusters: Sequence[Mapping[str, Any]], completeness: str) -> list[str]:
    facts: list[str] = []
    for cluster in clusters:
        if completeness == "complete" or cluster.get("status") == "human_confirmed":
            facts.extend(_ensure_list(cluster.get("strong_facts")))
    return _unique_strings(facts)[:50]


def _activity_overview(prefix: str, primary: Sequence[Mapping[str, Any]], clusters: Sequence[Mapping[str, Any]], completeness: str) -> str:
    focus = primary or [cluster for cluster in clusters if cluster.get("status") in {"active", "candidate"}][:3]
    if not focus:
        return f"{prefix}，尚未形成可证据追溯的 Video Memory 结论。"
    summaries = "；".join(str(cluster.get("cluster_summary") or cluster.get("cluster_title") or "") for cluster in focus[:3])
    suffix = "这些结论仍需结合窗口完整度和人工确认状态解读。" if completeness != "complete" else "这些结论均可下钻到证据素材。"
    return f"{prefix}，主要观察到：{summaries}。{suffix}"


def _workflow_patterns(primary_clusters: Sequence[Mapping[str, Any]], *, completeness: str = "complete") -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for cluster in primary_clusters:
        sequence = _ensure_list((cluster.get("cluster_signature") or {}).get("sequence_signature") if isinstance(cluster.get("cluster_signature"), Mapping) else [])
        if sequence:
            if completeness != "complete" and cluster.get("status") != "human_confirmed":
                continue
            patterns.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "sequence_signature": sequence,
                    "pattern_type": "recurring_action_sequence",
                    "confidence": cluster.get("confidence"),
                    "human_confirmation_status": cluster.get("confirmation_status"),
                    "evidence_trace": cluster.get("evidence_trace") or {},
                    "partial_window_notice": "" if completeness == "complete" else "基于不完整 30 天窗口，仅展示人工确认的 workflow 线索。",
                }
            )
    return patterns


def _apply_partial_window_reasoning_guard(
    workflow_context: Mapping[str, Any],
    *,
    completeness: str,
    days_available: int,
    window_days: int,
) -> dict[str, Any]:
    guarded = dict(workflow_context)
    if completeness == "complete":
        return guarded
    notice = f"基于 {days_available}/{window_days} 天数据的 partial window；未人工确认前不能作为完整 30-Day recurring workflow 结论。"
    for key in (
        "workflow_patterns",
        "instrument_usage_patterns",
        "project_hints",
        "project_or_context_hints",
        "step_reasoning_candidates",
        "process_completion_candidates",
        "rule_candidates",
        "reminder_candidates",
    ):
        rows: list[dict[str, Any]] = []
        for row in _ensure_list(guarded.get(key)):
            if not isinstance(row, Mapping):
                continue
            item = dict(row)
            if item.get("human_confirmation_status") != "human_confirmed" and item.get("candidate_status") != "human_confirmed":
                item["candidate_status"] = "partial_window_candidate"
                item["confidence"] = min(float(_float(item.get("confidence"), 0.0) or 0.0), 0.64)
                item["partial_window_notice"] = notice
            rows.append(item)
        guarded[key] = rows
    guarded["partial_window_reasoning_notice"] = notice
    return guarded


def _instrument_usage_patterns(clusters: Sequence[Mapping[str, Any]], *, completeness: str) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for cluster in clusters:
        instruments = _ensure_list(cluster.get("key_instruments"))
        if not instruments:
            continue
        if completeness != "complete" and cluster.get("status") not in {"human_confirmed", "primary"}:
            continue
        patterns.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "instruments": instruments,
                "actions": _ensure_list(cluster.get("canonical_actions")),
                "occurrence_count": cluster.get("occurrence_count") or 0,
                "day_count": cluster.get("day_count") or 0,
                "confidence": cluster.get("confidence"),
                "human_confirmation_status": cluster.get("confirmation_status") or "auto_inferred",
                "evidence_trace": cluster.get("evidence_trace") or {},
            }
        )
    return sorted(patterns, key=lambda row: (-(int(row.get("occurrence_count") or 0)), str(row.get("cluster_id") or "")))[:20]


def _project_context_hints(
    clusters: Sequence[Mapping[str, Any]],
    confirmed_contexts: Sequence[Mapping[str, Any]],
    *,
    completeness: str,
) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    confirmed_by_cluster = {
        str(item.get("cluster_id") or ""): dict(item)
        for item in confirmed_contexts
        if item.get("cluster_id")
    }
    for cluster in clusters:
        context = confirmed_by_cluster.get(str(cluster.get("cluster_id") or ""))
        if context:
            hints.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "hint_type": "human_confirmed_context",
                    "project_name": context.get("project_name") or "",
                    "sample_name": context.get("sample_name") or "",
                    "sop_name": context.get("sop_name") or "",
                    "confidence": 1.0,
                    "human_confirmation_status": "human_confirmed",
                    "evidence_trace": cluster.get("evidence_trace") or {},
                }
            )
        elif completeness == "complete" and cluster.get("status") in {"primary", "human_confirmed"}:
            hints.append(
                {
                    "cluster_id": cluster.get("cluster_id"),
                    "hint_type": "unresolved_project_context",
                    "project_name": "",
                    "sample_name": "",
                    "sop_name": "",
                    "confidence": 0.0,
                    "human_confirmation_status": "needs_review",
                    "question": "该重复实验活动可能属于同一项目或 SOP，但缺少人工提供的项目名、样品名或 SOP 名。",
                    "evidence_trace": cluster.get("evidence_trace") or {},
                }
            )
    return hints[:30]


def _sop_context_candidates(
    clusters: Sequence[Mapping[str, Any]],
    confirmed_contexts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    confirmed_by_cluster = {
        str(item.get("cluster_id") or ""): dict(item)
        for item in confirmed_contexts
        if item.get("cluster_id") and item.get("sop_name")
    }
    candidates: list[dict[str, Any]] = []
    for cluster in clusters:
        context = confirmed_by_cluster.get(str(cluster.get("cluster_id") or ""))
        if not context:
            continue
        candidates.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "sop_name": context.get("sop_name"),
                "supported_actions": _ensure_list(cluster.get("canonical_actions")),
                "candidate_status": "ready_for_step_reasoning",
                "confidence": 1.0,
                "evidence_trace": cluster.get("evidence_trace") or {},
            }
        )
    return candidates


def _process_completion_candidates(primary_clusters: Sequence[Mapping[str, Any]], *, completeness: str) -> list[dict[str, Any]]:
    if completeness != "complete":
        return []
    return [
        {
            "cluster_id": cluster.get("cluster_id"),
            "candidate_type": "visual_workflow_gap_review",
            "status": "needs_human_context",
            "reason": "已有跨天动作证据，但没有 SOP/项目上下文，不能自动补全步骤。",
            "evidence_trace": cluster.get("evidence_trace") or {},
        }
        for cluster in primary_clusters[:10]
        if cluster.get("status") == "human_confirmed"
    ]


def _reminder_rule_candidates(primary_clusters: Sequence[Mapping[str, Any]], *, completeness: str) -> list[dict[str, Any]]:
    if completeness != "complete":
        return []
    return [
        {
            "cluster_id": cluster.get("cluster_id"),
            "candidate_type": "future_real_time_rule",
            "status": "draft_only",
            "reason": "该动作反复出现且有证据链，可在后续接入 SOP 后生成提醒规则；当前不直接实时提醒。",
            "evidence_trace": cluster.get("evidence_trace") or {},
        }
        for cluster in primary_clusters[:10]
        if int(cluster.get("occurrence_count") or 0) >= 3
    ]


def _representative_evidence(clusters: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for cluster in clusters[:10]:
        evidence.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "keyframe_refs": _ensure_list(cluster.get("keyframe_refs"))[:2],
                "keyclip_refs": _ensure_list(cluster.get("keyclip_refs"))[:2],
                "evidence_bundle_ids": _ensure_list(cluster.get("evidence_bundle_ids"))[:3],
                "ledger_event_ids": _ensure_list(cluster.get("ledger_event_ids"))[:3],
            }
        )
    return evidence


def _top_values(values: Iterable[Any], limit: int = 10) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        counts[text] = counts.get(text, 0) + 1
    return [
        {"value": key, "count": value}
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _signature_set(signature: Mapping[str, Any], key: str) -> set[str]:
    return set(_ensure_list(signature.get(key)))


def _jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    left = {str(value).strip().lower() for value in a if str(value).strip()}
    right = {str(value).strip().lower() for value in b if str(value).strip()}
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _material_index_version(path: Path) -> str:
    return _stable_id("material-index", _file_fingerprint(path))


def _file_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    stat = path.stat()
    return _stable_hash({"path": str(path), "size": stat.st_size, "mtime": stat.st_mtime_ns})


def _aggregate_sha256(values: Iterable[Any]) -> str:
    sha256s = _unique_strings(values)
    if not sha256s:
        return ""
    if len(sha256s) == 1:
        return sha256s[0]
    return _stable_hash({"sha256s": sha256s})


def _cache_context(row: Mapping[str, Any]) -> str:
    return _stable_hash(row)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads_json(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(_json_dumps(value).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{_stable_hash(value)[:24]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _duration_ms(start_iso: str, end_iso: str) -> int:
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        return int((end - start).total_seconds() * 1000)
    except ValueError:
        return 0


def _coerce_date(value: str | date | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _material_date(row: Mapping[str, Any]) -> str:
    for value in (row.get("created_at"), row.get("date")):
        if value:
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date().isoformat()
            except ValueError:
                pass
    text = _join_text([row.get("package_name"), row.get("experiment_id"), row.get("stored_file"), row.get("absolute_path")])
    match = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    absolute = Path(str(row.get("absolute_path") or ""))
    if absolute.exists():
        return datetime.fromtimestamp(absolute.stat().st_mtime, timezone.utc).date().isoformat()
    return _today_utc().isoformat()


def _date_in_window(value: Any, start: date, end: date) -> bool:
    try:
        current = _coerce_date(str(value))
    except (TypeError, ValueError):
        return False
    return current is not None and start <= current <= end


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _mean(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return 0.0
    return round(sum(filtered) / len(filtered), 4)


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        loaded = _loads_json(value, default=None)
        if isinstance(loaded, list):
            return loaded
        return [value] if value else []
    return [value]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            nested = _unique_strings(value)
            for item in nested:
                if item not in seen:
                    seen.add(item)
                    results.append(item)
            continue
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _first_value(key: str, row: Mapping[str, Any], payload: Mapping[str, Any]) -> Any:
    value = payload.get(key) if isinstance(payload, Mapping) else None
    if value not in (None, ""):
        return value
    return row.get(key)


def _first_mapping(*values: Any) -> dict[str, Any]:
    for value in values:
        loaded = _loads_json(value, default=value)
        if isinstance(loaded, Mapping):
            return dict(loaded)
    return {}


def _join_text(values: Iterable[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            parts.append(_json_dumps(value))
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value if str(item or "").strip())
        else:
            text = str(value).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _tokens(value: Any) -> list[str]:
    text = _join_text([value]).lower()
    if not text:
        return []
    tokens = re.split(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", text)
    return [token for token in tokens if token]


def _duration_bucket(seconds: float) -> str:
    if seconds < 3:
        return "duration_lt_3s"
    if seconds < 15:
        return "duration_3_15s"
    if seconds < 60:
        return "duration_15_60s"
    if seconds < 180:
        return "duration_1_3min"
    return "duration_gt_3min"


def _timestamp_display(start: float | None, end: float | None) -> str:
    if start is None and end is None:
        return ""
    if end is None or end == start:
        return f"{start:.2f}s" if start is not None else ""
    return f"{start:.2f}-{end:.2f}s"


def _camera_role(view: Any) -> str:
    value = str(view or "").lower()
    if "first" in value or "operator" in value:
        return "operator_view"
    if "third" in value or "top" in value:
        return "top_view"
    return value


def _instrument_state(objects: Sequence[str]) -> str:
    instruments = sorted({obj for obj in objects if obj in INSTRUMENT_LABELS})
    return ", ".join(instruments)


__all__ = [
    "AUTO_MERGE_THRESHOLD",
    "CLUSTER_ALGORITHM_VERSION",
    "DEFAULT_BUNDLE_VLM_MODEL",
    "DEFAULT_ITEM_VLM_MODEL",
    "EVIDENCE_ITEM_SCHEMA_VERSION",
    "EvidenceItem",
    "MEMORY_DB_NAME",
    "PROMPT_VERSION",
    "REVIEW_MERGE_THRESHOLD",
    "SCHEMA_VERSION",
    "SIGNATURE_WEIGHTS",
    "VLMCache",
    "VLMResultCache",
    "VLM_MODE_OFFLINE",
    "VLM_MODE_REAL_QWEN_ASYNC",
    "VLM_MODE_REUSE_EXISTING",
    "VLM_MODEL_VERSION",
    "WINDOW_DAYS",
    "answer_video_memory_query",
    "build_vlm_result_cache_key",
    "build_video_memory",
    "build_workflow_context_reasoning",
    "build_partial_snapshot",
    "build_vlm_cache_key",
    "cluster_similarity",
    "ensure_video_memory_schema",
    "get_memory_snapshot",
    "get_video_memory_rebuild_background_status",
    "load_human_feedback",
    "memory_index_root",
    "query_video_memory",
    "record_human_feedback",
    "run_feedback_update_job",
    "score_evidence_clusters",
    "start_video_memory_rebuild_background",
    "update_cluster_lifecycle",
    "validate_evidence_item",
    "video_memory_db_path",
]
