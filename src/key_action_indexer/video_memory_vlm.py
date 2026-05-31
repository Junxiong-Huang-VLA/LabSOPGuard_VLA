from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SUPPORTED_QWEN_MODELS = ("qwen3.5-flash", "qwen3.5-plus")
VLM_CACHE_SCHEMA_VERSION = "video_memory.vlm_result_cache.v1"
ITEM_QWEN_TASK_TYPE = "video_memory_item_qwen_enhance"
BUNDLE_QWEN_TASK_TYPE = "video_memory_bundle_qwen_enhance"


class VLMResultCache:
    """SQLite-backed cache keyed by asset, material, prompt, model, task, and input context."""

    def __init__(self, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def cache_key(
        self,
        *,
        asset_sha256: str,
        material_id: str,
        prompt_version: str,
        model_version: str,
        task_type: str,
        input_context_hash: str,
    ) -> str:
        return build_vlm_result_cache_key(
            asset_sha256=asset_sha256,
            material_id=material_id,
            prompt_version=prompt_version,
            model_version=model_version,
            task_type=task_type,
            input_context_hash=input_context_hash,
        )

    def get(
        self,
        *,
        asset_sha256: str,
        material_id: str,
        prompt_version: str,
        model_version: str,
        task_type: str,
        input_context_hash: str,
    ) -> dict[str, Any] | None:
        cache_id = self.cache_key(
            asset_sha256=asset_sha256,
            material_id=material_id,
            prompt_version=prompt_version,
            model_version=model_version,
            task_type=task_type,
            input_context_hash=input_context_hash,
        )
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT payload_json FROM vlm_result_cache WHERE cache_id = ? AND status != 'invalidated'",
                (cache_id,),
            ).fetchone()
            if row is None:
                return None
            payload = json.loads(row["payload_json"])
            payload["hit_count"] = int(payload.get("hit_count") or 0) + 1
            payload["last_used_at"] = _now_iso()
            self.put_cache_row(payload)
            return payload
        finally:
            conn.close()

    def put(
        self,
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
        cache_id = self.cache_key(
            asset_sha256=asset_sha256,
            material_id=material_id,
            prompt_version=prompt_version,
            model_version=model_version,
            task_type=task_type,
            input_context_hash=input_context_hash,
        )
        cache_scope = "bundle" if asset_type == "evidence_bundle" or task_type == BUNDLE_QWEN_TASK_TYPE else "item"
        result_payload = dict(result)
        result_material_ids = _unique_strings([material_id, *_ensure_list(result_payload.get("material_ids"))])
        result_sha256s = _unique_strings([asset_sha256, *_ensure_list(result_payload.get("sha256s"))])
        row = {
            "cache_id": cache_id,
            "schema_version": VLM_CACHE_SCHEMA_VERSION,
            "cache_key": cache_id,
            "cache_scope": cache_scope,
            "asset_sha256": str(asset_sha256 or ""),
            "material_id": str(material_id or ""),
            "material_ids": result_material_ids,
            "sha256": str(asset_sha256 or ""),
            "sha256s": result_sha256s,
            "bundle_id": result_payload.get("bundle_id") or "",
            "evidence_id": result_payload.get("evidence_id") or "",
            "asset_type": str(asset_type or ""),
            "asset_path": str(asset_path or ""),
            "prompt_version": str(prompt_version or ""),
            "model_version": str(model_version or ""),
            "vlm_task_type": str(task_type or ""),
            "input_context_hash": str(input_context_hash or ""),
            "result_json": result_payload,
            "confidence": float(confidence or 0.0),
            "status": "active",
            "created_at": _now_iso(),
            "last_used_at": _now_iso(),
            "hit_count": 0,
            "invalidated_at": "",
            "invalidate_reason": "",
        }
        self.put_cache_row(row)
        return row

    def put_cache_row(self, row: Mapping[str, Any]) -> None:
        self._ensure_schema()
        payload = dict(row)
        now = _now_iso()
        conn = sqlite3.connect(str(self.sqlite_path))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO vlm_result_cache
                (cache_id, schema_version, material_id, session_id, experiment_id, segment_id,
                 micro_segment_id, sha256, date, status, confidence, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(payload.get("cache_id") or payload.get("cache_key") or ""),
                    str(payload.get("schema_version") or VLM_CACHE_SCHEMA_VERSION),
                    str(payload.get("material_id") or ""),
                    str(payload.get("session_id") or ""),
                    str(payload.get("experiment_id") or ""),
                    str(payload.get("segment_id") or ""),
                    str(payload.get("micro_segment_id") or ""),
                    str(payload.get("asset_sha256") or payload.get("sha256") or ""),
                    str(payload.get("date") or ""),
                    str(payload.get("status") or ""),
                    _float(payload.get("confidence"), 0.0) or 0.0,
                    _json_dumps(payload),
                    str(payload.get("created_at") or now),
                    now,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        conn = sqlite3.connect(str(self.sqlite_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vlm_result_cache (
                    cache_id TEXT PRIMARY KEY,
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vm_cache_sha ON vlm_result_cache(sha256)")
            conn.commit()
        finally:
            conn.close()


class QwenVideoMemoryAdapter:
    """Async adapter for optional Qwen Video Memory enhancement.

    The adapter never creates a network client on its own. A caller must inject
    one; otherwise enhancement degrades to offline/reuse-existing behavior.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "qwen3.5-flash",
        prompt_version: str,
    ) -> None:
        if model not in SUPPORTED_QWEN_MODELS:
            raise ValueError(f"Unsupported Qwen model {model!r}; expected one of {', '.join(SUPPORTED_QWEN_MODELS)}")
        self.client = client
        self.model = model
        self.prompt_version = prompt_version

    async def enhance_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        if self.client is None:
            return build_offline_no_client_item_result(item, prompt_version=self.prompt_version, model_version=self.model)
        payload = build_item_prompt_payload(item)
        response = await self._call_client(
            ("enhance_video_memory_item", "enhance_item", "describe_video_memory_item", "describe_scene"),
            payload=payload,
            asset_path=_item_asset_path(item),
            task_type=ITEM_QWEN_TASK_TYPE,
        )
        return normalize_qwen_item_result(
            item,
            response,
            prompt_version=self.prompt_version,
            model_version=self.model,
            task_type=ITEM_QWEN_TASK_TYPE,
        )

    async def enhance_bundle(self, bundle: Mapping[str, Any], item_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        if self.client is None:
            return build_offline_no_client_bundle_result(bundle, prompt_version=self.prompt_version, model_version=self.model)
        payload = build_bundle_prompt_payload(bundle, item_results)
        response = await self._call_client(
            ("enhance_video_memory_bundle", "enhance_bundle", "describe_video_memory_bundle"),
            payload=payload,
            asset_path="",
            task_type=BUNDLE_QWEN_TASK_TYPE,
        )
        return normalize_qwen_bundle_result(
            bundle,
            response,
            prompt_version=self.prompt_version,
            model_version=self.model,
            task_type=BUNDLE_QWEN_TASK_TYPE,
        )

    async def _call_client(
        self,
        method_names: Sequence[str],
        *,
        payload: Mapping[str, Any],
        asset_path: str,
        task_type: str,
    ) -> Any:
        if self.client is None:
            return {"status": "offline_no_client"}
        for name in method_names:
            method = getattr(self.client, name, None)
            if method is None:
                continue
            result = _invoke_client_method(
                method,
                payload=payload,
                model=self.model,
                prompt_version=self.prompt_version,
                task_type=task_type,
                asset_path=asset_path,
            )
            if inspect.isawaitable(result):
                return await result
            return result
        raise AttributeError(
            "Injected VLM client must expose enhance_video_memory_item/enhance_video_memory_bundle "
            "or a compatible describe_scene method."
        )


def build_vlm_result_cache_key(
    *,
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
            "asset_sha256": str(asset_sha256 or ""),
            "material_id": str(material_id or ""),
            "prompt_version": str(prompt_version or ""),
            "model_version": str(model_version or ""),
            "task_type": str(task_type or ""),
            "input_context_hash": str(input_context_hash or ""),
        },
    )


def item_input_context_hash(item: Mapping[str, Any]) -> str:
    return _stable_hash(
        {
            "schema_version": item.get("schema_version"),
            "evidence_id": item.get("evidence_id"),
            "material_id": item.get("material_id"),
            "material_ids": item.get("material_ids"),
            "asset_sha256": item.get("sha256"),
            "sha256s": item.get("sha256s"),
            "asset_type": item.get("asset_type"),
            "time_range": item.get("time_range"),
            "timestamps": item.get("timestamps"),
            "micro_segment_ids": item.get("micro_segment_ids"),
            "keyframe_refs": item.get("keyframe_refs"),
            "keyclip_refs": item.get("keyclip_refs"),
            "action": item.get("action"),
            "physical_evidence": item.get("physical_evidence"),
            "views": item.get("views"),
            "trace": item.get("trace"),
            "existing_vlm_sources": [
                {
                    "source_type": source.get("source_type"),
                    "source_id": source.get("source_id"),
                    "payload_hash": _stable_hash(source.get("payload") or {}),
                }
                for source in reusable_vlm_sources(item)
                if isinstance(source, Mapping)
            ],
        }
    )


def bundle_input_context_hash(bundle: Mapping[str, Any], item_results: Mapping[str, Mapping[str, Any]]) -> str:
    item_payloads = []
    for result_id in _ensure_list(bundle.get("vlm_item_result_ids")):
        result = item_results.get(str(result_id))
        if result:
            item_payloads.append(
                {
                    "vlm_result_id": result.get("vlm_result_id"),
                    "vlm_source": result.get("vlm_source"),
                    "status": result.get("status"),
                    "confidence": result.get("confidence"),
                    "strong_facts": result.get("strong_facts"),
                    "weak_inferences": result.get("weak_inferences"),
                }
            )
    return _stable_hash(
        {
            "bundle_id": bundle.get("bundle_id"),
            "material_ids": bundle.get("material_ids"),
            "sha256s": bundle.get("sha256s"),
            "evidence_item_ids": bundle.get("evidence_item_ids"),
            "micro_segment_ids": bundle.get("micro_segment_ids"),
            "time_range": bundle.get("time_range"),
            "timestamps": bundle.get("timestamps"),
            "keyframe_refs": bundle.get("keyframe_refs") or bundle.get("keyframes"),
            "keyclip_refs": bundle.get("keyclip_refs") or bundle.get("keyclips"),
            "views": bundle.get("views"),
            "yolo_summary": bundle.get("yolo_summary"),
            "item_results": item_payloads,
        }
    )


def enhance_items_sync(
    evidence_items: Sequence[Mapping[str, Any]],
    *,
    sqlite_path: str | Path,
    prompt_version: str,
    offline_model_version: str,
    qwen_model: str,
    vlm_client: Any | None = None,
    enable_qwen: bool = False,
    reuse_qwen_cache: bool = False,
    max_qwen_calls: int | None = None,
    fallback_factory: Callable[[Mapping[str, Any], str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    return _run_async(
        enhance_items(
            evidence_items,
            sqlite_path=sqlite_path,
            prompt_version=prompt_version,
            offline_model_version=offline_model_version,
            qwen_model=qwen_model,
            vlm_client=vlm_client,
            enable_qwen=enable_qwen,
            reuse_qwen_cache=reuse_qwen_cache,
            max_qwen_calls=max_qwen_calls,
            fallback_factory=fallback_factory,
        )
    )


async def enhance_items(
    evidence_items: Sequence[Mapping[str, Any]],
    *,
    sqlite_path: str | Path,
    prompt_version: str,
    offline_model_version: str,
    qwen_model: str,
    vlm_client: Any | None = None,
    enable_qwen: bool = False,
    reuse_qwen_cache: bool = False,
    max_qwen_calls: int | None = None,
    fallback_factory: Callable[[Mapping[str, Any], str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if enable_qwen or reuse_qwen_cache:
        _validate_qwen_model(qwen_model)
    adapter = QwenVideoMemoryAdapter(client=vlm_client, model=qwen_model, prompt_version=prompt_version) if enable_qwen else None
    cache = VLMResultCache(sqlite_path)
    effective_model = qwen_model if enable_qwen and vlm_client is not None else offline_model_version
    stats = {
        "hit": 0,
        "miss": 0,
        "reused_existing": 0,
        "qwen_cache_reused": 0,
        "qwen_calls": 0,
        "offline_fallback": 0,
        "skipped_by_budget": 0,
        "errors": 0,
    }
    qwen_budget = max_qwen_calls if max_qwen_calls is not None else None
    results: list[dict[str, Any]] = []

    for item in evidence_items:
        task_type = ITEM_QWEN_TASK_TYPE if enable_qwen and vlm_client is not None else f"item_{item.get('asset_type') or 'material'}"
        context_hash = item_input_context_hash(item)
        if reuse_qwen_cache and not (enable_qwen and vlm_client is not None):
            cached_qwen = cache.get(
                asset_sha256=str(item.get("sha256") or item.get("asset_sha256") or ""),
                material_id=str(item.get("material_id") or ""),
                prompt_version=prompt_version,
                model_version=qwen_model,
                task_type=ITEM_QWEN_TASK_TYPE,
                input_context_hash=context_hash,
            )
            if cached_qwen is not None:
                stats["hit"] += 1
                stats["qwen_cache_reused"] += 1
                result = dict(cached_qwen.get("result_json") or {})
                result["cache_status"] = "hit"
                result["cache_reuse_mode"] = "reuse_existing_vlm"
                results.append(result)
                continue
        cached = cache.get(
            asset_sha256=str(item.get("sha256") or item.get("asset_sha256") or ""),
            material_id=str(item.get("material_id") or ""),
            prompt_version=prompt_version,
            model_version=effective_model,
            task_type=task_type,
            input_context_hash=context_hash,
        )
        if cached is not None:
            stats["hit"] += 1
            result = dict(cached.get("result_json") or {})
            result["cache_status"] = "hit"
            results.append(result)
            continue

        stats["miss"] += 1
        try:
            reused = build_reused_item_result(item, prompt_version=prompt_version, model_version=effective_model, task_type=task_type)
            if reused is not None:
                stats["reused_existing"] += 1
                result = reused
            elif enable_qwen and adapter is not None and vlm_client is not None and (qwen_budget is None or stats["qwen_calls"] < qwen_budget):
                stats["qwen_calls"] += 1
                result = await adapter.enhance_item(item)
            else:
                if enable_qwen and adapter is not None and vlm_client is not None and qwen_budget is not None:
                    stats["skipped_by_budget"] += 1
                stats["offline_fallback"] += 1
                result = fallback_factory(item, prompt_version, offline_model_version)
                result["vlm_source"] = "offline_fallback"
                result["cache_status"] = "miss"
            cache.put(
                asset_sha256=str(item.get("sha256") or item.get("asset_sha256") or ""),
                material_id=str(item.get("material_id") or ""),
                asset_type=str(item.get("asset_type") or ""),
                asset_path=_item_asset_path(item),
                prompt_version=prompt_version,
                model_version=effective_model,
                task_type=task_type,
                input_context_hash=context_hash,
                result=result,
                confidence=float(result.get("confidence") or 0.0),
            )
            results.append(result)
        except Exception as exc:
            stats["errors"] += 1
            result = fallback_factory(item, prompt_version, offline_model_version)
            result["vlm_source"] = "offline_fallback_after_qwen_error"
            result["qwen_error"] = str(exc)
            results.append(result)
    return results, stats


def enhance_bundles_sync(
    bundles: Sequence[Mapping[str, Any]],
    item_results: Sequence[Mapping[str, Any]],
    *,
    sqlite_path: str | Path,
    prompt_version: str,
    offline_model_version: str,
    qwen_model: str,
    vlm_client: Any | None = None,
    enable_qwen: bool = False,
    reuse_qwen_cache: bool = False,
    max_qwen_calls: int | None = None,
    fallback_factory: Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]], str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    return _run_async(
        enhance_bundles(
            bundles,
            item_results,
            sqlite_path=sqlite_path,
            prompt_version=prompt_version,
            offline_model_version=offline_model_version,
            qwen_model=qwen_model,
            vlm_client=vlm_client,
            enable_qwen=enable_qwen,
            reuse_qwen_cache=reuse_qwen_cache,
            max_qwen_calls=max_qwen_calls,
            fallback_factory=fallback_factory,
        )
    )


async def enhance_bundles(
    bundles: Sequence[Mapping[str, Any]],
    item_results: Sequence[Mapping[str, Any]],
    *,
    sqlite_path: str | Path,
    prompt_version: str,
    offline_model_version: str,
    qwen_model: str,
    vlm_client: Any | None = None,
    enable_qwen: bool = False,
    reuse_qwen_cache: bool = False,
    max_qwen_calls: int | None = None,
    fallback_factory: Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]], str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if enable_qwen or reuse_qwen_cache:
        _validate_qwen_model(qwen_model)
    adapter = QwenVideoMemoryAdapter(client=vlm_client, model=qwen_model, prompt_version=prompt_version) if enable_qwen else None
    cache = VLMResultCache(sqlite_path)
    effective_model = qwen_model if enable_qwen and vlm_client is not None else offline_model_version
    item_by_id = {str(row.get("vlm_result_id") or ""): row for row in item_results if row.get("vlm_result_id")}
    stats = {
        "hit": 0,
        "miss": 0,
        "reused_existing": 0,
        "qwen_cache_reused": 0,
        "qwen_calls": 0,
        "offline_fallback": 0,
        "skipped_by_budget": 0,
        "errors": 0,
    }
    qwen_budget = max_qwen_calls if max_qwen_calls is not None else None
    results: list[dict[str, Any]] = []

    for bundle in bundles:
        task_type = BUNDLE_QWEN_TASK_TYPE if enable_qwen and vlm_client is not None else "bundle_merge"
        context_hash = bundle_input_context_hash(bundle, item_by_id)
        asset_sha = _bundle_cache_sha256(bundle)
        material_id = _bundle_cache_material_id(bundle)
        if reuse_qwen_cache and not (enable_qwen and vlm_client is not None):
            cached_qwen = cache.get(
                asset_sha256=asset_sha,
                material_id=material_id,
                prompt_version=prompt_version,
                model_version=qwen_model,
                task_type=BUNDLE_QWEN_TASK_TYPE,
                input_context_hash=context_hash,
            )
            if cached_qwen is not None:
                stats["hit"] += 1
                stats["qwen_cache_reused"] += 1
                result = dict(cached_qwen.get("result_json") or {})
                result["cache_status"] = "hit"
                result["cache_reuse_mode"] = "reuse_existing_vlm"
                results.append(result)
                continue
        cached = cache.get(
            asset_sha256=asset_sha,
            material_id=material_id,
            prompt_version=prompt_version,
            model_version=effective_model,
            task_type=task_type,
            input_context_hash=context_hash,
        )
        if cached is not None:
            stats["hit"] += 1
            result = dict(cached.get("result_json") or {})
            result["cache_status"] = "hit"
            results.append(result)
            continue

        stats["miss"] += 1
        try:
            reused = build_reused_bundle_result(bundle, item_by_id, prompt_version=prompt_version, model_version=effective_model, task_type=task_type)
            if reused is not None:
                stats["reused_existing"] += 1
                result = reused
            elif enable_qwen and adapter is not None and vlm_client is not None and (qwen_budget is None or stats["qwen_calls"] < qwen_budget):
                stats["qwen_calls"] += 1
                result = await adapter.enhance_bundle(bundle, item_by_id)
            else:
                if enable_qwen and adapter is not None and vlm_client is not None and qwen_budget is not None:
                    stats["skipped_by_budget"] += 1
                stats["offline_fallback"] += 1
                result = fallback_factory(bundle, item_by_id, prompt_version, offline_model_version)
                result["vlm_source"] = "offline_fallback"
                result["cache_status"] = "miss"
            cache.put(
                asset_sha256=asset_sha,
                material_id=material_id,
                asset_type="evidence_bundle",
                asset_path="",
                prompt_version=prompt_version,
                model_version=effective_model,
                task_type=task_type,
                input_context_hash=context_hash,
                result=result,
                confidence=float(result.get("confidence") or 0.0),
            )
            results.append(result)
        except Exception as exc:
            stats["errors"] += 1
            result = fallback_factory(bundle, item_by_id, prompt_version, offline_model_version)
            result["vlm_source"] = "offline_fallback_after_qwen_error"
            result["qwen_error"] = str(exc)
            results.append(result)
    return results, stats


def reusable_vlm_sources(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = payload.get("vlm_existing_sources") or payload.get("existing_vlm_sources") or []
    if isinstance(sources, list):
        return [dict(source) for source in sources if isinstance(source, Mapping)]
    if isinstance(sources, Mapping):
        return [dict(sources)]
    return []


def build_reused_item_result(
    item: Mapping[str, Any],
    *,
    prompt_version: str,
    model_version: str,
    task_type: str,
) -> dict[str, Any] | None:
    source = _best_reusable_source(reusable_vlm_sources(item))
    if source is None:
        return None
    payload = source.get("payload") if isinstance(source.get("payload"), Mapping) else {}
    summary = _first_text(
        payload.get("description"),
        payload.get("summary"),
        payload.get("visual_scene_summary"),
        payload.get("reason"),
        source.get("summary"),
        source.get("source_type"),
    )
    objects = _unique_strings(
        [
            *_ensure_list(payload.get("confirmed_objects")),
            *_ensure_list(payload.get("visible_objects")),
            *_ensure_list(payload.get("supporting_object_labels")),
            *_ensure_list(item.get("detected_objects")),
            item.get("primary_object"),
        ]
    )
    facts = _unique_strings([summary, payload.get("reason"), *_ensure_list(payload.get("strong_facts"))])
    weak = _unique_strings([payload.get("semantic_action"), payload.get("possible_lab_action"), *_ensure_list(payload.get("weak_inferences"))])
    unresolved = _unique_strings([*_ensure_list(payload.get("missing_evidence")), *_ensure_list(payload.get("contradictions")), *_ensure_list(payload.get("unresolved_questions"))])
    return {
        "schema_version": str(item.get("schema_version") or "video_memory.v1"),
        "vlm_result_id": _stable_id("item-vlm", item.get("evidence_id")),
        "task_type": task_type,
        "material_id": item.get("material_id") or "",
        "material_ids": _ensure_list(item.get("material_ids") or item.get("material_id")),
        "evidence_id": item.get("evidence_id") or "",
        "session_id": item.get("session_id") or "",
        "experiment_id": item.get("experiment_id") or "",
        "segment_id": item.get("segment_id") or "",
        "micro_segment_id": item.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(item.get("micro_segment_ids") or item.get("micro_segment_id")),
        "sha256": item.get("sha256") or "",
        "sha256s": _ensure_list(item.get("sha256s") or item.get("sha256")),
        "keyframe": item.get("keyframe") or "",
        "keyframe_refs": _ensure_list(item.get("keyframe_refs") or item.get("keyframe")),
        "keyclip": item.get("keyclip") or "",
        "keyclip_refs": _ensure_list(item.get("keyclip_refs") or item.get("keyclip")),
        "timestamp": item.get("timestamp") or item.get("time_range") or {},
        "timestamps": _ensure_list(item.get("timestamps") or item.get("time_range")),
        "date": item.get("date") or "",
        "view": item.get("view") or "",
        "prompt_version": prompt_version,
        "model_version": str(payload.get("model") or payload.get("model_version") or model_version),
        "vlm_source": "reuse_existing",
        "reused_source_type": source.get("source_type") or "",
        "reused_source_id": source.get("source_id") or "",
        "status": str(payload.get("status") or payload.get("decision") or "reused_existing"),
        "visual_scene_summary": summary,
        "visible_objects": objects,
        "manipulated_objects": _unique_strings([item.get("primary_object"), *_ensure_list(payload.get("manipulated_objects"))]),
        "hands_visible": _hands_visible(objects),
        "operation_type": _first_text(payload.get("event_type"), payload.get("semantic_action"), item.get("canonical_action_type"), item.get("action_name")),
        "possible_lab_action": _first_text(payload.get("semantic_action"), payload.get("possible_lab_action"), item.get("action_name")),
        "possible_experiment_stage": _first_text(payload.get("experiment_stage"), payload.get("possible_experiment_stage")),
        "physical_change_observed": _first_text(payload.get("event_type"), item.get("physical_event_type")),
        "instrument_or_container_state": _first_text(payload.get("instrument_or_container_state")),
        "material_state": _first_text(payload.get("material_state")),
        "safety_relevance": _first_text(payload.get("safety_relevance")),
        "evidence_for_action": facts or [summary],
        "evidence_against_action": _ensure_list(payload.get("contradictions")),
        "ambiguity_notes": unresolved,
        "confidence": _bounded(_float(payload.get("confidence"), _float(item.get("confidence"), 0.0)) or 0.0),
        "searchable_keywords": _unique_strings([item.get("canonical_action_type"), item.get("action_name"), item.get("primary_object"), *objects]),
        "memory_candidate_text": summary,
        "strong_facts": facts or ([summary] if summary else []),
        "weak_inferences": weak,
        "unresolved_questions": unresolved,
        "cache_status": "miss",
        "created_at": _now_iso(),
    }


def build_reused_bundle_result(
    bundle: Mapping[str, Any],
    item_by_id: Mapping[str, Mapping[str, Any]],
    *,
    prompt_version: str,
    model_version: str,
    task_type: str,
) -> dict[str, Any] | None:
    reused_items = [
        row
        for row_id in _ensure_list(bundle.get("vlm_item_result_ids"))
        for row in [item_by_id.get(str(row_id))]
        if isinstance(row, Mapping) and row.get("vlm_source") == "reuse_existing"
    ]
    if not reused_items:
        return None
    facts = _unique_strings([fact for row in reused_items for fact in _ensure_list(row.get("strong_facts"))])
    weak = _unique_strings([fact for row in reused_items for fact in _ensure_list(row.get("weak_inferences"))])
    unresolved = _unique_strings([fact for row in reused_items for fact in _ensure_list(row.get("unresolved_questions"))])
    views = _ensure_list(bundle.get("views"))
    return {
        "schema_version": str(bundle.get("schema_version") or "video_memory.v1"),
        "vlm_result_id": _stable_id("bundle-vlm", bundle.get("bundle_id")),
        "task_type": task_type,
        "bundle_id": bundle.get("bundle_id") or "",
        "session_id": bundle.get("session_id") or "",
        "experiment_id": bundle.get("experiment_id") or "",
        "segment_id": bundle.get("segment_id") or "",
        "micro_segment_id": bundle.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(bundle.get("micro_segment_ids") or bundle.get("micro_segment_id")),
        "material_id": bundle.get("material_id") or "",
        "material_ids": _ensure_list(bundle.get("material_ids") or bundle.get("material_id")),
        "sha256": bundle.get("sha256") or "",
        "sha256s": _ensure_list(bundle.get("sha256s") or bundle.get("sha256")),
        "keyframe": bundle.get("keyframe") or "",
        "keyframe_refs": _ensure_list(bundle.get("keyframe_refs") or bundle.get("keyframes")),
        "keyclip": bundle.get("keyclip") or "",
        "keyclip_refs": _ensure_list(bundle.get("keyclip_refs") or bundle.get("keyclips")),
        "timestamp": bundle.get("timestamp") or bundle.get("time_range") or {},
        "timestamps": _ensure_list(bundle.get("timestamps") or bundle.get("time_range")),
        "date": bundle.get("date") or "",
        "prompt_version": prompt_version,
        "model_version": model_version,
        "vlm_source": "reuse_existing",
        "status": "reused_existing",
        "merged_scene_understanding": " ".join(facts[:3]) if facts else _join_text([bundle.get("action_name"), bundle.get("primary_object")]),
        "merged_action_understanding": weak[0] if weak else "",
        "view_agreement": "dual_view_supported" if len(views) >= 2 else "single_view_only",
        "view_conflict": "",
        "strong_facts": facts,
        "weak_inferences": weak,
        "unresolved_questions": unresolved,
        "confidence": _bounded(_mean([float(row.get("confidence") or 0.0) for row in reused_items])),
        "searchable_keywords": _unique_strings([bundle.get("canonical_action_type"), bundle.get("action_name"), bundle.get("primary_object")]),
        "cache_status": "miss",
        "created_at": _now_iso(),
    }


def build_offline_no_client_item_result(item: Mapping[str, Any], *, prompt_version: str, model_version: str) -> dict[str, Any]:
    return {
        "schema_version": str(item.get("schema_version") or "video_memory.v1"),
        "vlm_result_id": _stable_id("item-vlm", item.get("evidence_id")),
        "task_type": ITEM_QWEN_TASK_TYPE,
        "material_id": item.get("material_id") or "",
        "material_ids": _ensure_list(item.get("material_ids") or item.get("material_id")),
        "evidence_id": item.get("evidence_id") or "",
        "session_id": item.get("session_id") or "",
        "experiment_id": item.get("experiment_id") or "",
        "segment_id": item.get("segment_id") or "",
        "micro_segment_id": item.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(item.get("micro_segment_ids") or item.get("micro_segment_id")),
        "sha256": item.get("sha256") or "",
        "sha256s": _ensure_list(item.get("sha256s") or item.get("sha256")),
        "keyframe": item.get("keyframe") or "",
        "keyframe_refs": _ensure_list(item.get("keyframe_refs") or item.get("keyframe")),
        "keyclip": item.get("keyclip") or "",
        "keyclip_refs": _ensure_list(item.get("keyclip_refs") or item.get("keyclip")),
        "timestamp": item.get("timestamp") or item.get("time_range") or {},
        "timestamps": _ensure_list(item.get("timestamps") or item.get("time_range")),
        "prompt_version": prompt_version,
        "model_version": model_version,
        "vlm_source": "offline_no_client",
        "status": "offline_no_client",
        "visual_scene_summary": "",
        "visible_objects": _ensure_list(item.get("detected_objects")),
        "confidence": 0.0,
        "strong_facts": [],
        "weak_inferences": [],
        "unresolved_questions": ["No injected VLM client; no network call was attempted."],
        "created_at": _now_iso(),
    }


def build_offline_no_client_bundle_result(bundle: Mapping[str, Any], *, prompt_version: str, model_version: str) -> dict[str, Any]:
    return {
        "schema_version": str(bundle.get("schema_version") or "video_memory.v1"),
        "vlm_result_id": _stable_id("bundle-vlm", bundle.get("bundle_id")),
        "task_type": BUNDLE_QWEN_TASK_TYPE,
        "bundle_id": bundle.get("bundle_id") or "",
        "session_id": bundle.get("session_id") or "",
        "experiment_id": bundle.get("experiment_id") or "",
        "segment_id": bundle.get("segment_id") or "",
        "micro_segment_id": bundle.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(bundle.get("micro_segment_ids") or bundle.get("micro_segment_id")),
        "material_id": bundle.get("material_id") or "",
        "material_ids": _ensure_list(bundle.get("material_ids") or bundle.get("material_id")),
        "sha256": bundle.get("sha256") or "",
        "sha256s": _ensure_list(bundle.get("sha256s") or bundle.get("sha256")),
        "keyframe": bundle.get("keyframe") or "",
        "keyframe_refs": _ensure_list(bundle.get("keyframe_refs") or bundle.get("keyframes")),
        "keyclip": bundle.get("keyclip") or "",
        "keyclip_refs": _ensure_list(bundle.get("keyclip_refs") or bundle.get("keyclips")),
        "timestamp": bundle.get("timestamp") or bundle.get("time_range") or {},
        "timestamps": _ensure_list(bundle.get("timestamps") or bundle.get("time_range")),
        "prompt_version": prompt_version,
        "model_version": model_version,
        "vlm_source": "offline_no_client",
        "status": "offline_no_client",
        "merged_scene_understanding": "",
        "merged_action_understanding": "",
        "view_agreement": "unknown",
        "confidence": 0.0,
        "strong_facts": [],
        "weak_inferences": [],
        "unresolved_questions": ["No injected VLM client; no network call was attempted."],
        "created_at": _now_iso(),
    }


def build_item_prompt_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "video_memory.qwen_item_prompt.v1",
        "task": "video_memory_item_enhancement",
        "prompt_contract": "Use only the evidence item fields and referenced image/clip context; do not invent lab intent.",
        "evidence_id": item.get("evidence_id"),
        "material_id": item.get("material_id"),
        "asset_type": item.get("asset_type"),
        "time_range": item.get("time_range"),
        "action": item.get("action"),
        "physical_evidence": item.get("physical_evidence"),
        "views": item.get("views"),
        "trace": item.get("trace"),
        "existing_vlm_sources": [
            {
                "source_type": source.get("source_type"),
                "source_id": source.get("source_id"),
                "confidence": source.get("confidence"),
                "status": source.get("status"),
            }
            for source in reusable_vlm_sources(item)
        ],
        "required_output": {
            "description": "short factual visual description",
            "confirmed_objects": ["object labels already supported by evidence"],
            "semantic_action": "physical action label if supported",
            "evidence_alignment": "aligned | partial | uncertain",
            "reason": "why the item supports or does not support the action",
            "missing_evidence": [],
            "contradictions": [],
            "confidence": 0.0,
        },
    }


def build_bundle_prompt_payload(bundle: Mapping[str, Any], item_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result_rows = [
        item_results.get(str(row_id))
        for row_id in _ensure_list(bundle.get("vlm_item_result_ids"))
        if item_results.get(str(row_id))
    ]
    return {
        "schema_version": "video_memory.qwen_bundle_prompt.v1",
        "task": "video_memory_bundle_enhancement",
        "prompt_contract": "Merge item-level evidence without adding unsupported objects, actions, or experiment intent.",
        "bundle_id": bundle.get("bundle_id"),
        "material_ids": bundle.get("material_ids"),
        "sha256s": bundle.get("sha256s"),
        "micro_segment_ids": bundle.get("micro_segment_ids"),
        "time_range": bundle.get("time_range"),
        "timestamps": bundle.get("timestamps"),
        "keyframe_refs": bundle.get("keyframe_refs") or bundle.get("keyframes"),
        "keyclip_refs": bundle.get("keyclip_refs") or bundle.get("keyclips"),
        "views": bundle.get("views"),
        "yolo_summary": bundle.get("yolo_summary"),
        "item_results": [
            {
                "vlm_result_id": row.get("vlm_result_id"),
                "evidence_id": row.get("evidence_id"),
                "vlm_source": row.get("vlm_source"),
                "status": row.get("status"),
                "strong_facts": row.get("strong_facts"),
                "weak_inferences": row.get("weak_inferences"),
                "unresolved_questions": row.get("unresolved_questions"),
                "confidence": row.get("confidence"),
            }
            for row in result_rows
        ],
        "required_output": {
            "merged_scene_understanding": "",
            "merged_action_understanding": "",
            "view_agreement": "dual_view_supported | single_view_only | conflicting",
            "view_conflict": "",
            "strong_facts": [],
            "weak_inferences": [],
            "unresolved_questions": [],
            "confidence": 0.0,
        },
    }


def normalize_qwen_item_result(
    item: Mapping[str, Any],
    response: Any,
    *,
    prompt_version: str,
    model_version: str,
    task_type: str,
) -> dict[str, Any]:
    payload = _extract_response_payload(response)
    summary = _first_text(payload.get("description"), payload.get("summary"), payload.get("visual_scene_summary"), payload.get("reason"))
    objects = _unique_strings([*_ensure_list(payload.get("confirmed_objects")), *_ensure_list(payload.get("visible_objects")), *_ensure_list(payload.get("supporting_object_labels")), *_ensure_list(item.get("detected_objects")), item.get("primary_object")])
    facts = _unique_strings([summary, payload.get("reason"), *_ensure_list(payload.get("strong_facts"))])
    weak = _unique_strings([payload.get("semantic_action"), payload.get("possible_lab_action"), *_ensure_list(payload.get("weak_inferences"))])
    unresolved = _unique_strings([*_ensure_list(payload.get("missing_evidence")), *_ensure_list(payload.get("contradictions")), *_ensure_list(payload.get("unresolved_questions"))])
    return {
        "schema_version": str(item.get("schema_version") or "video_memory.v1"),
        "vlm_result_id": _stable_id("item-vlm", item.get("evidence_id")),
        "task_type": task_type,
        "material_id": item.get("material_id") or "",
        "material_ids": _ensure_list(item.get("material_ids") or item.get("material_id")),
        "evidence_id": item.get("evidence_id") or "",
        "session_id": item.get("session_id") or "",
        "experiment_id": item.get("experiment_id") or "",
        "segment_id": item.get("segment_id") or "",
        "micro_segment_id": item.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(item.get("micro_segment_ids") or item.get("micro_segment_id")),
        "sha256": item.get("sha256") or "",
        "sha256s": _ensure_list(item.get("sha256s") or item.get("sha256")),
        "keyframe": item.get("keyframe") or "",
        "keyframe_refs": _ensure_list(item.get("keyframe_refs") or item.get("keyframe")),
        "keyclip": item.get("keyclip") or "",
        "keyclip_refs": _ensure_list(item.get("keyclip_refs") or item.get("keyclip")),
        "timestamp": item.get("timestamp") or item.get("time_range") or {},
        "timestamps": _ensure_list(item.get("timestamps") or item.get("time_range")),
        "date": item.get("date") or "",
        "view": item.get("view") or "",
        "prompt_version": prompt_version,
        "model_version": str(payload.get("model") or model_version),
        "vlm_source": "qwen_async",
        "status": _first_text(payload.get("status"), payload.get("decision"), payload.get("evidence_alignment"), "qwen_enhanced"),
        "visual_scene_summary": summary,
        "visible_objects": objects,
        "manipulated_objects": _unique_strings([item.get("primary_object"), *_ensure_list(payload.get("manipulated_objects"))]),
        "hands_visible": _hands_visible(objects),
        "operation_type": _first_text(payload.get("event_type"), payload.get("semantic_action"), item.get("canonical_action_type"), item.get("action_name")),
        "possible_lab_action": _first_text(payload.get("semantic_action"), payload.get("possible_lab_action"), item.get("action_name")),
        "possible_experiment_stage": _first_text(payload.get("experiment_stage"), payload.get("possible_experiment_stage")),
        "physical_change_observed": _first_text(payload.get("event_type"), item.get("physical_event_type")),
        "instrument_or_container_state": _first_text(payload.get("instrument_or_container_state")),
        "material_state": _first_text(payload.get("material_state")),
        "safety_relevance": _first_text(payload.get("safety_relevance")),
        "evidence_for_action": facts,
        "evidence_against_action": _ensure_list(payload.get("contradictions")),
        "ambiguity_notes": unresolved,
        "confidence": _bounded(_float(payload.get("confidence"), _float(item.get("confidence"), 0.0)) or 0.0),
        "searchable_keywords": _unique_strings([item.get("canonical_action_type"), item.get("action_name"), item.get("primary_object"), *objects]),
        "memory_candidate_text": summary,
        "strong_facts": facts,
        "weak_inferences": weak,
        "unresolved_questions": unresolved,
        "cache_status": "miss",
        "created_at": _now_iso(),
    }


def normalize_qwen_bundle_result(
    bundle: Mapping[str, Any],
    response: Any,
    *,
    prompt_version: str,
    model_version: str,
    task_type: str,
) -> dict[str, Any]:
    payload = _extract_response_payload(response)
    facts = _unique_strings([payload.get("merged_scene_understanding"), *_ensure_list(payload.get("strong_facts"))])
    weak = _unique_strings([payload.get("merged_action_understanding"), *_ensure_list(payload.get("weak_inferences"))])
    unresolved = _unique_strings([payload.get("view_conflict"), *_ensure_list(payload.get("unresolved_questions"))])
    return {
        "schema_version": str(bundle.get("schema_version") or "video_memory.v1"),
        "vlm_result_id": _stable_id("bundle-vlm", bundle.get("bundle_id")),
        "task_type": task_type,
        "bundle_id": bundle.get("bundle_id") or "",
        "session_id": bundle.get("session_id") or "",
        "experiment_id": bundle.get("experiment_id") or "",
        "segment_id": bundle.get("segment_id") or "",
        "micro_segment_id": bundle.get("micro_segment_id") or "",
        "micro_segment_ids": _ensure_list(bundle.get("micro_segment_ids") or bundle.get("micro_segment_id")),
        "material_id": bundle.get("material_id") or "",
        "material_ids": _ensure_list(bundle.get("material_ids") or bundle.get("material_id")),
        "sha256": bundle.get("sha256") or "",
        "sha256s": _ensure_list(bundle.get("sha256s") or bundle.get("sha256")),
        "keyframe": bundle.get("keyframe") or "",
        "keyframe_refs": _ensure_list(bundle.get("keyframe_refs") or bundle.get("keyframes")),
        "keyclip": bundle.get("keyclip") or "",
        "keyclip_refs": _ensure_list(bundle.get("keyclip_refs") or bundle.get("keyclips")),
        "timestamp": bundle.get("timestamp") or bundle.get("time_range") or {},
        "timestamps": _ensure_list(bundle.get("timestamps") or bundle.get("time_range")),
        "date": bundle.get("date") or "",
        "prompt_version": prompt_version,
        "model_version": str(payload.get("model") or model_version),
        "vlm_source": "qwen_async",
        "status": _first_text(payload.get("status"), payload.get("view_agreement"), "qwen_enhanced"),
        "merged_scene_understanding": _first_text(payload.get("merged_scene_understanding"), payload.get("description"), payload.get("summary")),
        "merged_action_understanding": _first_text(payload.get("merged_action_understanding"), payload.get("semantic_action")),
        "view_agreement": _first_text(payload.get("view_agreement"), "unknown"),
        "view_conflict": _first_text(payload.get("view_conflict")),
        "strong_facts": facts,
        "weak_inferences": weak,
        "unresolved_questions": unresolved,
        "confidence": _bounded(_float(payload.get("confidence"), _float(bundle.get("confidence"), 0.0)) or 0.0),
        "searchable_keywords": _unique_strings([bundle.get("canonical_action_type"), bundle.get("action_name"), bundle.get("primary_object")]),
        "cache_status": "miss",
        "created_at": _now_iso(),
    }


def _invoke_client_method(
    method: Any,
    *,
    payload: Mapping[str, Any],
    model: str,
    prompt_version: str,
    task_type: str,
    asset_path: str,
) -> Any:
    owner = getattr(method, "__self__", None)
    old_model = getattr(owner, "model", None) if owner is not None else None
    did_set_model = False
    if owner is not None and old_model is not None and model:
        try:
            setattr(owner, "model", model)
            did_set_model = True
        except Exception:
            did_set_model = False
    try:
        if getattr(method, "__name__", "") == "describe_scene":
            prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            try:
                return method(str(asset_path or ""), prompt=prompt, model=model, prompt_version=prompt_version, task_type=task_type, temperature=0.0)
            except TypeError:
                try:
                    return method(str(asset_path or ""), prompt=prompt, temperature=0.0)
                except TypeError:
                    return method(str(asset_path or ""), prompt=prompt)
        try:
            return method(payload, model=model, prompt_version=prompt_version, task_type=task_type)
        except TypeError:
            try:
                return method(payload, model=model)
            except TypeError:
                return method(payload)
    finally:
        if did_set_model:
            try:
                setattr(owner, "model", old_model)
            except Exception:
                pass


def _extract_response_payload(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return dict(response)
    raw_response = getattr(response, "raw_response", None)
    if isinstance(raw_response, Mapping):
        return dict(raw_response)
    for attr in ("payload", "data", "output"):
        value = getattr(response, attr, None)
        if isinstance(value, Mapping):
            return dict(value)
    for attr in ("text", "content", "description"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return _loads_json(value, default={"description": value})
    if isinstance(response, str):
        return _loads_json(response, default={"description": response})
    return {}


def _best_reusable_source(sources: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    usable = [dict(source) for source in sources if isinstance(source, Mapping) and isinstance(source.get("payload"), Mapping)]
    if not usable:
        return None
    priority = {"qwen_event_audits": 0, "vlm_semantics": 1, "advanced_vision_evidence": 2}
    usable.sort(key=lambda source: (priority.get(str(source.get("source_type") or ""), 9), -float(source.get("confidence") or 0.0)))
    return usable[0]


def _validate_qwen_model(model: str) -> None:
    if model not in SUPPORTED_QWEN_MODELS:
        raise ValueError(f"Unsupported Qwen model {model!r}; expected one of {', '.join(SUPPORTED_QWEN_MODELS)}")


def _item_asset_path(item: Mapping[str, Any]) -> str:
    return _first_text(item.get("keyframe_path"), item.get("keyclip_path"), item.get("asset_path"), item.get("package_uri"))


def _bundle_cache_material_id(bundle: Mapping[str, Any]) -> str:
    material_ids = _unique_strings(_ensure_list(bundle.get("material_ids")) or [bundle.get("material_id")])
    if material_ids:
        return "bundle-materials:" + "|".join(material_ids)
    return "bundle:" + str(bundle.get("bundle_id") or "")


def _bundle_cache_sha256(bundle: Mapping[str, Any]) -> str:
    sha256s = _unique_strings(_ensure_list(bundle.get("sha256s")) or [bundle.get("sha256")])
    if not sha256s:
        return _stable_hash({"bundle_id": bundle.get("bundle_id"), "material_ids": _ensure_list(bundle.get("material_ids"))})
    if len(sha256s) == 1:
        return sha256s[0]
    return _stable_hash({"sha256s": sha256s})


def _run_async(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("Synchronous Video Memory VLM helper cannot run inside an active event loop; use the async API instead.")


def _hands_visible(objects: Sequence[Any]) -> bool:
    return any(str(token).lower() in {"hand", "hands", "gloved_hand", "glove", "gloves"} for token in objects)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads_json(value: Any, *, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
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


def _unique_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in _unique_strings(list(value)):
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


def _join_text(values: Sequence[Any]) -> str:
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


__all__ = [
    "BUNDLE_QWEN_TASK_TYPE",
    "ITEM_QWEN_TASK_TYPE",
    "QwenVideoMemoryAdapter",
    "SUPPORTED_QWEN_MODELS",
    "VLMResultCache",
    "build_bundle_prompt_payload",
    "build_item_prompt_payload",
    "build_reused_bundle_result",
    "build_reused_item_result",
    "build_vlm_result_cache_key",
    "bundle_input_context_hash",
    "enhance_bundles",
    "enhance_bundles_sync",
    "enhance_items",
    "enhance_items_sync",
    "item_input_context_hash",
    "normalize_qwen_bundle_result",
    "normalize_qwen_item_result",
    "reusable_vlm_sources",
]
