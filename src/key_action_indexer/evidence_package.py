"""Portable read-only evidence package loading, retrieval, and judgement.

The package format is intentionally independent from the LabSOPGuard backend.
OpenClaw or another assistant can copy a package folder to a different machine,
load it from disk, and query evidence through relative package paths.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .material_reference_index import (
    DEFAULT_REFERENCES_NAME,
    DEFAULT_SQLITE_NAME,
    MATERIAL_INDEX_JSON,
    MATERIAL_INDEX_JSONL,
    build_key_material_reference_index,
)
from .schemas import write_jsonl


EVIDENCE_PACKAGE_MANIFEST = "evidence_package_manifest.json"
PHYSICAL_CHANGE_LOG_JSONL = "physical_change_log.jsonl"
TIME_ALIGNMENT_JSON = "time_alignment.json"
SCHEMA_VERSION = "evidence_package_manifest.v1"
QUERY_RESULT_SCHEMA_VERSION = "evidence_package_query_result.v1"
EVIDENCE_PACKAGE_EVAL_SCHEMA_VERSION = "evidence_package_eval.v1"

ACTIVE_CHANGE_TYPES = {
    "hand_object_interaction",
    "object_move",
    "liquid_transfer",
    "panel_operation",
    "container_state_change",
}

OBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "reagent_bottle": ("试剂瓶", "试剂", "瓶子", "bottle", "reagent_bottle", "sample_bottle", "container"),
    "sample_bottle": ("样品瓶", "样品瓶蓝", "sample_bottle", "sample_bottle_blue"),
    "container": ("容器", "烧杯", "量筒", "称量瓶", "beaker", "cylinder", "container"),
    "spatula": ("药匙", "勺", "刮勺", "spatula", "scoop"),
    "balance": ("天平", "电子天平", "balance", "scale"),
    "paper": ("称量纸", "纸", "weighing_paper", "paper"),
    "panel": ("面板", "按钮", "旋钮", "读数", "panel", "button", "knob", "display"),
    "liquid": ("液体", "溶液", "转移液体", "liquid", "solution", "pour"),
}

INTENT_ALIASES: dict[str, tuple[str, ...]] = {
    "return_position_check": ("归位", "放回", "原位", "复位", "return", "restore", "put_back"),
    "object_move_check": ("移动", "挪动", "位置变化", "位移", "move", "movement"),
    "hand_object_check": ("接触", "拿起", "取", "放", "手", "hand", "contact", "grasp"),
    "liquid_transfer_check": ("液体转移", "倒液", "加液", "transfer", "pour"),
    "panel_operation_check": ("面板", "按钮", "旋钮", "读数", "panel", "button", "display"),
    "container_state_check": ("打开", "关闭", "盖子", "瓶盖", "open", "close", "cap"),
}


@dataclass
class EvidencePackage:
    root: Path
    manifest: dict[str, Any]
    evidence_manifest: dict[str, Any]
    references: list[dict[str, Any]]
    material_index: list[dict[str, Any]]
    physical_changes: list[dict[str, Any]]
    time_alignment: dict[str, Any]

    @classmethod
    def load(cls, package_dir: str | Path) -> "EvidencePackage":
        root = Path(package_dir).resolve()
        evidence_manifest = _read_json(root / EVIDENCE_PACKAGE_MANIFEST)
        manifest = _read_json(root / "manifest.json")
        entrypoints = _merged_entrypoints(manifest, evidence_manifest)

        references = _read_jsonl(_entrypoint(root, entrypoints, "key_material_references_jsonl", (DEFAULT_REFERENCES_NAME,)))
        material_index = _read_material_index(root, entrypoints)
        physical_changes = _read_jsonl(_entrypoint(root, entrypoints, "physical_change_log_jsonl", (PHYSICAL_CHANGE_LOG_JSONL,)))
        time_alignment = _read_json(_entrypoint(root, entrypoints, "time_alignment_json", (TIME_ALIGNMENT_JSON,)))

        if not references:
            references = _references_from_material_index(material_index)

        return cls(
            root=root,
            manifest=manifest,
            evidence_manifest=evidence_manifest,
            references=references,
            material_index=material_index,
            physical_changes=physical_changes,
            time_alignment=time_alignment,
        )

    @property
    def package_id(self) -> str:
        return str(
            self.evidence_manifest.get("package_id")
            or self.manifest.get("package_id")
            or self.manifest.get("experiment_id")
            or self.manifest.get("experiment_label")
            or self.root.name
        )

    def query(
        self,
        query_text: str,
        *,
        message_sent_at: str | None = None,
        limit: int = 8,
        window_before_sec: float | None = None,
        window_after_sec: float | None = None,
    ) -> dict[str, Any]:
        return query_evidence_package(
            self.root,
            query_text=query_text,
            message_sent_at=message_sent_at,
            limit=limit,
            window_before_sec=window_before_sec,
            window_after_sec=window_after_sec,
            package=self,
        )


def build_evidence_package(
    package_dir: str | Path,
    *,
    source_manifest: str | Path | None = None,
    key_action_index_dir: str | Path | None = None,
    package_id: str | None = None,
    experiment_id: str | None = None,
    include_reports: bool = False,
    sqlite_path: str | Path | None = None,
    references_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build portable package sidecars from a material folder.

    The function never calls the LabSOPGuard backend. It reads existing package
    files, normalizes references, writes a physical change log, writes a time
    alignment file, and writes an evidence package manifest with relative paths.
    """

    root = Path(package_dir)
    root.mkdir(parents=True, exist_ok=True)
    index_summary = build_key_material_reference_index(
        root,
        sqlite_path=sqlite_path,
        references_path=references_path,
        include_reports=include_reports,
    )
    refs_path = Path(index_summary["references_path"])
    references = _read_jsonl(refs_path)
    key_action_enrichment = {"enabled": False, "matched_count": 0, "source_count": 0}
    if key_action_index_dir is not None:
        references, key_action_enrichment = enrich_references_with_key_action_index(
            references,
            package_root=root,
            key_action_index_dir=key_action_index_dir,
        )
        write_jsonl(refs_path, references)

    physical_change_path = root / PHYSICAL_CHANGE_LOG_JSONL
    physical_changes = build_physical_change_log_rows(references)
    write_jsonl(physical_change_path, physical_changes)

    time_alignment_path = root / TIME_ALIGNMENT_JSON
    resolved_source_manifest = source_manifest
    if resolved_source_manifest is None and key_action_index_dir is not None:
        candidate_manifest = Path(key_action_index_dir) / "manifest.json"
        if candidate_manifest.exists():
            resolved_source_manifest = candidate_manifest
    time_alignment = build_time_alignment_payload(root, source_manifest=resolved_source_manifest)
    _write_json(time_alignment_path, time_alignment)

    manifest = build_evidence_package_manifest(
        root,
        package_id=package_id,
        experiment_id=experiment_id,
        source_manifest=resolved_source_manifest,
        key_action_index_dir=key_action_index_dir,
        references_path=refs_path,
        sqlite_path=Path(index_summary["sqlite_path"]),
        physical_change_log_path=physical_change_path,
        time_alignment_path=time_alignment_path,
    )
    manifest_path = root / EVIDENCE_PACKAGE_MANIFEST
    _write_json(manifest_path, manifest)

    return {
        "schema_version": "evidence_package_build.v1",
        "package_root": str(root),
        "package_id": manifest["package_id"],
        "manifest_path": str(manifest_path),
        "references_path": str(refs_path),
        "sqlite_path": index_summary["sqlite_path"],
        "physical_change_log_path": str(physical_change_path),
        "time_alignment_path": str(time_alignment_path),
        "reference_count": len(references),
        "physical_change_count": len(physical_changes),
        "portable": True,
        "key_action_enrichment": key_action_enrichment,
        "index_summary": index_summary,
    }


def build_evidence_package_manifest(
    package_dir: str | Path,
    *,
    package_id: str | None = None,
    experiment_id: str | None = None,
    source_manifest: str | Path | None = None,
    key_action_index_dir: str | Path | None = None,
    references_path: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    physical_change_log_path: str | Path | None = None,
    time_alignment_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(package_dir)
    existing_manifest = _read_json(root / "manifest.json")
    resolved_package_id = str(
        package_id
        or existing_manifest.get("package_id")
        or existing_manifest.get("experiment_id")
        or existing_manifest.get("experiment_label")
        or root.name
    )
    resolved_experiment_id = str(experiment_id or existing_manifest.get("experiment_id") or resolved_package_id)
    entrypoints = _detect_entrypoints(
        root,
        references_path=references_path,
        sqlite_path=sqlite_path,
        physical_change_log_path=physical_change_log_path,
        time_alignment_path=time_alignment_path,
    )
    files = [_file_entry(root, rel_path) for rel_path in sorted(set(entrypoints.values())) if rel_path]
    return {
        "schema_version": SCHEMA_VERSION,
        "package_id": resolved_package_id,
        "experiment_id": resolved_experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "path_mode": "relative_to_package_root",
        "portable": True,
        "read_only_consumer": True,
        "backend_required": False,
        "entrypoints": entrypoints,
        "files": [item for item in files if item],
        "provenance": {
            "source_manifest": _path_to_relative(source_manifest, root) if source_manifest else "",
            "key_action_index_dir": _path_to_relative(key_action_index_dir, root) if key_action_index_dir else "",
            "key_action_index_dir_name": Path(key_action_index_dir).name if key_action_index_dir else "",
        },
        "notes": [
            "All dereferenceable package paths are relative to this manifest's folder.",
            "Consumers should prefer package_uri or relative_path over local absolute paths.",
        ],
    }


def validate_evidence_package(
    package_dir: str | Path,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate a portable evidence package without contacting any backend."""

    root = Path(package_dir)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    manifest_path = root / EVIDENCE_PACKAGE_MANIFEST
    manifest = _read_json(manifest_path)
    if not manifest_path.exists():
        errors.append({"code": "missing_manifest", "path": EVIDENCE_PACKAGE_MANIFEST})
    if manifest and manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append({"code": "unsupported_manifest_schema", "value": manifest.get("schema_version")})
    if manifest and manifest.get("path_mode") != "relative_to_package_root":
        errors.append({"code": "non_portable_path_mode", "value": manifest.get("path_mode")})
    if manifest and manifest.get("backend_required") is not False:
        errors.append({"code": "backend_required_not_false", "value": manifest.get("backend_required")})

    entrypoints = manifest.get("entrypoints") if isinstance(manifest.get("entrypoints"), Mapping) else {}
    required_entrypoints = {
        "key_material_references_jsonl",
        "physical_change_log_jsonl",
        "time_alignment_json",
    }
    for key in sorted(required_entrypoints):
        value = entrypoints.get(key)
        if not value:
            errors.append({"code": "missing_entrypoint", "entrypoint": key})
            continue
        if _is_absolute_path_text(value):
            errors.append({"code": "absolute_entrypoint_path", "entrypoint": key, "path": value})
            continue
        if not (root / str(value)).exists():
            errors.append({"code": "missing_entrypoint_file", "entrypoint": key, "path": value})

    references_path = _entrypoint(root, entrypoints, "key_material_references_jsonl", (DEFAULT_REFERENCES_NAME,))
    physical_change_path = _entrypoint(root, entrypoints, "physical_change_log_jsonl", (PHYSICAL_CHANGE_LOG_JSONL,))
    time_alignment_path = _entrypoint(root, entrypoints, "time_alignment_json", (TIME_ALIGNMENT_JSON,))
    references = _read_jsonl(references_path)
    physical_changes = _read_jsonl(physical_change_path)
    time_alignment = _read_json(time_alignment_path)

    for row_index, ref in enumerate(references, start=1):
        _validate_reference_paths(root, ref, row_index=row_index, errors=errors, warnings=warnings, strict=strict)
        if not ref.get("material_id"):
            errors.append({"code": "reference_missing_material_id", "row_index": row_index})
        if not (ref.get("stored_file") or ref.get("formal_clip_path") or ref.get("clip_path")):
            warnings.append({"code": "reference_missing_primary_asset", "row_index": row_index, "material_id": ref.get("material_id")})

    change_material_ids = {
        str(material_id)
        for change in physical_changes
        for material_id in (change.get("evidence_material_ids") or [])
    }
    reference_ids = {str(ref.get("material_id")) for ref in references if ref.get("material_id")}
    dangling = sorted(change_material_ids - reference_ids)
    if dangling:
        errors.append({"code": "dangling_physical_change_material_ids", "material_ids": dangling[:20], "count": len(dangling)})

    if time_alignment and time_alignment.get("path_mode") != "relative_to_package_root":
        errors.append({"code": "time_alignment_non_portable_path_mode", "value": time_alignment.get("path_mode")})

    status = "passed"
    if errors:
        status = "failed"
    elif warnings:
        status = "warning"
    return {
        "schema_version": "evidence_package_validation.v1",
        "package_root": str(root),
        "status": status,
        "ok": not errors,
        "strict": strict,
        "counts": {
            "references": len(references),
            "physical_changes": len(physical_changes),
            "errors": len(errors),
            "warnings": len(warnings),
        },
        "errors": errors,
        "warnings": warnings,
    }


def build_time_alignment_payload(
    package_dir: str | Path,
    *,
    source_manifest: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(package_dir)
    manifest_path = Path(source_manifest) if source_manifest is not None else root / "manifest.json"
    manifest = _read_json(manifest_path)
    session_start = (
        manifest.get("session_start_at")
        or manifest.get("session_start_time")
        or manifest.get("created_at")
        or manifest.get("experiment_start_time")
    )
    streams: list[dict[str, Any]] = []
    videos = manifest.get("videos")
    if isinstance(videos, Mapping):
        for view, item in videos.items():
            if isinstance(item, Mapping):
                streams.append(_video_stream_from_manifest(root, str(view), item))
    elif isinstance(manifest.get("video_streams"), list):
        for index, item in enumerate(manifest["video_streams"]):
            if isinstance(item, Mapping):
                streams.append(_video_stream_from_manifest(root, str(item.get("view") or f"stream_{index + 1}"), item))
    return {
        "schema_version": "time_alignment.v1",
        "path_mode": "relative_to_package_root",
        "session_start_at": session_start,
        "source_manifest": _path_to_relative(manifest_path, root) if manifest_path.exists() else "",
        "video_streams": streams,
        "message_alignment_policy": {
            "default_window_before_sec": 90.0,
            "default_window_after_sec": 180.0,
            "live_message_time_maps_to_video_sec": True,
        },
    }


def build_physical_change_log_rows(references: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, ref in enumerate(references, start=1):
        payload = _payload_dict(ref)
        merged = {**payload, **dict(ref)}
        change_type = _change_type(merged)
        if change_type not in ACTIVE_CHANGE_TYPES and not _has_state_pair(merged):
            continue
        before_state, after_state = _state_pair(merged)
        material_id = str(merged.get("material_id") or f"material_{index:06d}")
        change_id = str(merged.get("change_id") or "chg_" + hashlib.sha1(f"{material_id}|{index}".encode("utf-8")).hexdigest()[:12])
        objects = _combined_objects(merged)
        actions = _combined_actions(merged)
        window_audit = _window_audit(merged)
        rows.append(
            {
                "schema_version": "physical_change.v1",
                "change_id": change_id,
                "event_type": change_type,
                "start_sec": _float_or_none(merged.get("start_sec")),
                "end_sec": _float_or_none(merged.get("end_sec")),
                "objects": objects,
                "actions": actions,
                "secondary_objects": _list_strings(merged.get("secondary_objects")),
                "secondary_actions": _list_strings(merged.get("secondary_actions")),
                "window_audit": window_audit,
                "target_object_support": window_audit.get("target_object_support") if window_audit else {},
                "secondary_object_support": window_audit.get("secondary_object_support") if window_audit else [],
                "uncertainty_reasons": _list_strings(
                    merged.get("uncertainty_reasons")
                    or (window_audit.get("uncertainty_reasons") if window_audit else [])
                    or (window_audit.get("reasons") if window_audit else [])
                ),
                "before": before_state,
                "after": after_state,
                "confidence": _safe_float(merged.get("confidence") or merged.get("quality_score"), 0.0),
                "source_view": merged.get("view") or merged.get("source_view") or merged.get("camera_view"),
                "decision_path": merged.get("decision_path") or "",
                "decision_trace": _list_strings(merged.get("decision_trace")),
                "segment_id": merged.get("segment_id") or merged.get("parent_segment_id"),
                "micro_segment_id": merged.get("micro_segment_id"),
                "evidence_strength": _evidence_strength(merged),
                "source_kind": "key_action_reference",
                "evidence_material_ids": [material_id],
            }
        )
    return rows


def enrich_references_with_key_action_index(
    references: Sequence[Mapping[str, Any]],
    *,
    package_root: str | Path,
    key_action_index_dir: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge segment/micro decision metadata from an existing key_action_index folder."""

    root = Path(package_root)
    index_dir = Path(key_action_index_dir)
    metadata_rows = _load_key_action_metadata_rows(index_dir)
    by_micro: dict[str, dict[str, Any]] = {}
    by_segment: dict[str, dict[str, Any]] = {}
    for row in metadata_rows:
        micro_id = str(row.get("micro_segment_id") or "")
        segment_id = str(row.get("segment_id") or row.get("parent_segment_id") or "")
        if micro_id:
            by_micro[micro_id] = row
        if segment_id and segment_id not in by_segment:
            by_segment[segment_id] = row

    enriched: list[dict[str, Any]] = []
    matched_count = 0
    for ref_raw in references:
        ref = dict(ref_raw)
        micro_id = str(ref.get("micro_segment_id") or "")
        segment_id = str(ref.get("segment_id") or ref.get("parent_segment_id") or "")
        match = by_micro.get(micro_id) if micro_id else None
        if match is None and segment_id:
            match = by_segment.get(segment_id)
        if match is not None:
            matched_count += 1
            ref = _merge_key_action_metadata(ref, match, package_root=root, key_action_index_dir=index_dir)
        enriched.append(ref)

    return enriched, {
        "enabled": True,
        "source_dir": _path_to_relative(index_dir, root) or index_dir.name,
        "source_count": len(metadata_rows),
        "matched_count": matched_count,
        "unmatched_count": max(0, len(references) - matched_count),
    }


def query_evidence_package(
    package_dir: str | Path,
    *,
    query_text: str,
    message_sent_at: str | None = None,
    limit: int = 8,
    window_before_sec: float | None = None,
    window_after_sec: float | None = None,
    package: EvidencePackage | None = None,
) -> dict[str, Any]:
    package = package or EvidencePackage.load(package_dir)
    query_info = _parse_query(query_text)
    time_context = _build_time_context(
        package.time_alignment,
        message_sent_at=message_sent_at,
        window_before_sec=window_before_sec,
        window_after_sec=window_after_sec,
    )
    candidates = _rank_references(
        package.references,
        package.physical_changes,
        query_text=query_text,
        query_info=query_info,
        window=time_context.get("search_window"),
        limit=limit,
    )
    if not candidates and time_context.get("search_window"):
        time_context["fallback_reason"] = "time_window_no_match"
        candidates = _rank_references(
            package.references,
            package.physical_changes,
            query_text=query_text,
            query_info=query_info,
            window=None,
            limit=limit,
        )
    bundles = [_build_evidence_bundle(package, item, package.physical_changes) for item in candidates]
    judgement = _judge_bundles(query_info, bundles)
    return {
        "schema_version": QUERY_RESULT_SCHEMA_VERSION,
        "package_id": package.package_id,
        "package_root": str(package.root),
        "path_mode": "relative_to_package_root",
        "query_text": query_text,
        "intent": query_info["intent"],
        "target_objects": query_info["target_objects"],
        "time_context": time_context,
        "judgement": judgement,
        "evidence_bundles": bundles,
    }


def evaluate_evidence_package_queries(
    package_dir: str | Path,
    queries: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    output_path: str | Path | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Run a small query set against a read-only evidence package.

    Query records may include expected_intent, expected_status,
    expected_label, expected_material_id, and expected_event_type. Metrics are
    only computed for expectations that are present, so the same command works
    both for exploratory smoke tests and curated regression sets.
    """

    package = EvidencePackage.load(package_dir)
    query_rows = _load_eval_queries(queries)
    results: list[dict[str, Any]] = []
    check_names = (
        "intent_match",
        "status_match",
        "label_match",
        "material_top1_match",
        "material_topk_match",
        "event_type_match",
    )
    totals = {name: 0 for name in check_names}
    passed = {name: 0 for name in check_names}
    insufficient_count = 0

    for row_index, row in enumerate(query_rows, start=1):
        query_text = str(row.get("query") or row.get("query_text") or "").strip()
        if not query_text:
            results.append(
                {
                    "row_index": row_index,
                    "query_text": query_text,
                    "status": "skipped",
                    "error": "missing_query_text",
                    "checks": {},
                }
            )
            continue
        query_limit = int(row.get("limit") or limit)
        query_result = query_evidence_package(
            package.root,
            query_text=query_text,
            message_sent_at=row.get("message_sent_at") or row.get("message_time"),
            limit=query_limit,
            window_before_sec=_float_or_none(row.get("window_before_sec")),
            window_after_sec=_float_or_none(row.get("window_after_sec")),
            package=package,
        )
        judgement = query_result.get("judgement") if isinstance(query_result.get("judgement"), Mapping) else {}
        if judgement.get("status") == "insufficient":
            insufficient_count += 1
        bundles = query_result.get("evidence_bundles") if isinstance(query_result.get("evidence_bundles"), list) else []
        top_material_ids = [str(bundle.get("material_id")) for bundle in bundles if isinstance(bundle, Mapping) and bundle.get("material_id")]
        event_types = _result_event_types(bundles)
        expected_material_id = str(row.get("expected_material_id") or "").strip()
        checks: dict[str, bool] = {}
        if row.get("expected_intent"):
            checks["intent_match"] = query_result.get("intent") == row.get("expected_intent")
        if row.get("expected_status"):
            checks["status_match"] = judgement.get("status") == row.get("expected_status")
        if row.get("expected_label"):
            checks["label_match"] = judgement.get("label") == row.get("expected_label")
        if expected_material_id:
            checks["material_top1_match"] = bool(top_material_ids and top_material_ids[0] == expected_material_id)
            checks["material_topk_match"] = expected_material_id in top_material_ids
        if row.get("expected_event_type"):
            checks["event_type_match"] = str(row.get("expected_event_type")) in event_types
        for name, value in checks.items():
            totals[name] += 1
            passed[name] += 1 if value else 0
        results.append(
            {
                "row_index": row_index,
                "query_text": query_text,
                "message_sent_at": row.get("message_sent_at") or row.get("message_time"),
                "expected": {
                    key: row.get(key)
                    for key in (
                        "expected_intent",
                        "expected_status",
                        "expected_label",
                        "expected_material_id",
                        "expected_event_type",
                    )
                    if row.get(key) not in (None, "")
                },
                "actual": {
                    "intent": query_result.get("intent"),
                    "status": judgement.get("status"),
                    "label": judgement.get("label"),
                    "evidence_material_id": judgement.get("evidence_material_id"),
                    "top_material_ids": top_material_ids,
                    "event_types": sorted(event_types),
                    "bundle_count": len(bundles),
                },
                "checks": checks,
                "judgement": judgement,
            }
        )

    summary = {
        "query_count": len(query_rows),
        "evaluated_query_count": sum(1 for item in results if item.get("status") != "skipped"),
        "insufficient_count": insufficient_count,
        "insufficient_rate": round(insufficient_count / len(query_rows), 4) if query_rows else 0.0,
    }
    for name in check_names:
        summary[f"{name}_count"] = passed[name]
        summary[f"{name}_total"] = totals[name]
        summary[f"{name}_rate"] = round(passed[name] / totals[name], 4) if totals[name] else None

    payload = {
        "schema_version": EVIDENCE_PACKAGE_EVAL_SCHEMA_VERSION,
        "package_id": package.package_id,
        "package_root": str(package.root),
        "path_mode": "relative_to_package_root",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "summary": summary,
        "results": results,
    }
    if output_path is not None:
        _write_json(Path(output_path), payload)
    return payload


def map_message_time_to_video_sec(alignment: Mapping[str, Any], message_sent_at: str | None) -> float | None:
    if not message_sent_at:
        return None
    message_dt = _parse_datetime(message_sent_at)
    session_dt = _parse_datetime(alignment.get("session_start_at") or alignment.get("session_start_time"))
    if message_dt is None or session_dt is None:
        return None
    delta = (message_dt - session_dt).total_seconds()
    streams = alignment.get("video_streams") if isinstance(alignment.get("video_streams"), list) else []
    if streams and isinstance(streams[0], Mapping):
        stream = streams[0]
        offset = _safe_float(stream.get("offset_sec"), 0.0)
        clock_scale = _safe_float(stream.get("clock_scale"), 1.0) or 1.0
        return round(max(0.0, (delta - offset) / clock_scale), 3)
    return round(max(0.0, delta), 3)


def _load_eval_queries(queries: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(queries, (str, Path)):
        text = Path(queries).read_text(encoding="utf-8-sig")
        try:
            payload: Any = json.loads(text)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = queries
    if isinstance(payload, Mapping):
        rows = payload.get("queries") or payload.get("items") or []
    else:
        rows = payload
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("Evidence package eval queries must be a list or an object with a queries list.")
    normalized: list[dict[str, Any]] = []
    for item in rows:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
        elif isinstance(item, str):
            normalized.append({"query": item})
    return normalized


def _result_event_types(bundles: Sequence[Any]) -> set[str]:
    event_types: set[str] = set()
    for bundle in bundles:
        if not isinstance(bundle, Mapping):
            continue
        inferred = _change_type(bundle)
        if inferred:
            event_types.add(inferred)
        for key in ("event_type",):
            if bundle.get(key):
                event_types.add(str(bundle[key]))
        for action in _list_strings(bundle.get("actions")):
            if action in ACTIVE_CHANGE_TYPES:
                event_types.add(action)
        for change in bundle.get("physical_changes") or []:
            if isinstance(change, Mapping) and change.get("event_type"):
                event_types.add(str(change["event_type"]))
            if isinstance(change, Mapping):
                inferred_change = _change_type(change)
                if inferred_change:
                    event_types.add(inferred_change)
    return event_types


def _load_key_action_metadata_rows(index_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        index_dir / "index" / "vector_metadata.jsonl",
        index_dir / "index" / "docstore.jsonl",
        index_dir / "metadata" / "vector_metadata.jsonl",
        index_dir / "metadata" / "micro_segments.jsonl",
        index_dir / "metadata" / "segment_metadata.jsonl",
    ]
    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for path in candidates:
        for row in _read_jsonl(path):
            key = str(row.get("embedding_id") or row.get("micro_segment_id") or row.get("segment_id") or json.dumps(row, sort_keys=True, default=str))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            rows.append(row)
    return rows


def _merge_key_action_metadata(
    ref: dict[str, Any],
    metadata: Mapping[str, Any],
    *,
    package_root: Path,
    key_action_index_dir: Path,
) -> dict[str, Any]:
    copy_keys = (
        "decision_path",
        "decision_trace",
        "detector_backend",
        "detector_source_view",
        "source_view",
        "fallback_used",
        "fallback_reason",
        "reason_code",
        "raw_score",
        "final_score",
        "run_manifest_id",
        "evidence_link",
        "retrieval_boost_factors",
        "alignment_health",
        "alignment_report",
        "yolo_interaction_count",
        "yolo_interactions",
        "yolo_evidence",
        "interaction_events",
        "interaction_keyframes",
        "secondary_objects",
        "secondary_actions",
        "window_audit",
        "evidence_level",
        "evidence_reasons",
        "limitations",
    )
    for key in copy_keys:
        if key in metadata and metadata.get(key) not in (None, "", [], {}):
            ref[key] = metadata.get(key)
    ref.setdefault("segment_id", metadata.get("segment_id") or metadata.get("parent_segment_id"))
    if metadata.get("micro_segment_id"):
        ref.setdefault("micro_segment_id", metadata.get("micro_segment_id"))
    if metadata.get("action_type") and not ref.get("canonical_action_type"):
        ref["canonical_action_type"] = metadata.get("action_type")
    objects = _combined_objects({**dict(metadata), **ref})
    detected_objects = _list_strings(metadata.get("detected_objects")) or _list_strings(metadata.get("primary_object"))
    if objects or detected_objects:
        ref["objects"] = sorted(set(objects + detected_objects))
    actions = _combined_actions({**dict(metadata), **ref})
    if metadata.get("action_type"):
        actions.append(str(metadata.get("action_type")))
    if actions:
        ref["actions"] = sorted(set(action for action in actions if action))

    key_action_assets = []
    for key in ("third_person_clip", "first_person_clip"):
        rel = _path_to_relative(metadata.get(key), package_root)
        if rel:
            key_action_assets.append({"kind": key, "relative_path": rel})
    for binding in metadata.get("asset_bindings") or []:
        if not isinstance(binding, Mapping):
            continue
        item = dict(binding)
        for path_key in ("clip_path", "keyframe_path", "video_path"):
            if item.get(path_key):
                item[path_key] = _path_to_relative(item[path_key], package_root) or Path(str(item[path_key])).name
        if isinstance(item.get("keyframe_paths"), list):
            item["keyframe_paths"] = [
                _path_to_relative(value, package_root) or Path(str(value)).name
                for value in item["keyframe_paths"]
                if value
            ]
        key_action_assets.append(item)

    ref["key_action_index"] = {
        "source_dir": _path_to_relative(key_action_index_dir, package_root) or key_action_index_dir.name,
        "index_level": metadata.get("index_level"),
        "embedding_id": metadata.get("embedding_id"),
        "segment_id": metadata.get("segment_id"),
        "micro_segment_id": metadata.get("micro_segment_id"),
        "decision_path": metadata.get("decision_path") or "",
        "reason_code": metadata.get("reason_code") or "",
        "secondary_objects": _list_strings(metadata.get("secondary_objects")),
        "secondary_actions": _list_strings(metadata.get("secondary_actions")),
        "window_audit": metadata.get("window_audit") if isinstance(metadata.get("window_audit"), Mapping) else {},
        "assets": key_action_assets,
    }
    searchable = [str(ref.get("searchable_text") or ""), str(metadata.get("index_text") or "")]
    if metadata.get("decision_path"):
        searchable.append(str(metadata["decision_path"]))
    if metadata.get("reason_code"):
        searchable.append(str(metadata["reason_code"]))
    ref["searchable_text"] = re.sub(r"\s+", " ", " ".join(searchable)).strip()[:24000]
    ref["payload_json"] = _merge_payload_json(ref, metadata)
    return ref


def _merge_payload_json(ref: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    payload = _payload_dict(ref)
    payload.setdefault("key_action_index", {})
    if isinstance(payload["key_action_index"], dict):
        payload["key_action_index"].update(
            {
                "decision_path": metadata.get("decision_path") or "",
                "decision_trace": metadata.get("decision_trace") or [],
                "detector_backend": metadata.get("detector_backend") or "",
                "detector_source_view": metadata.get("detector_source_view") or "",
                "fallback_used": bool(metadata.get("fallback_used")),
                "reason_code": metadata.get("reason_code") or "",
                "raw_score": metadata.get("raw_score"),
                "final_score": metadata.get("final_score"),
                "secondary_objects": _list_strings(metadata.get("secondary_objects")),
                "secondary_actions": _list_strings(metadata.get("secondary_actions")),
                "window_audit": metadata.get("window_audit") if isinstance(metadata.get("window_audit"), Mapping) else {},
            }
        )
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _detect_entrypoints(
    root: Path,
    *,
    references_path: str | Path | None,
    sqlite_path: str | Path | None,
    physical_change_log_path: str | Path | None,
    time_alignment_path: str | Path | None,
) -> dict[str, str]:
    candidates: dict[str, str | Path | None] = {
        "material_index_jsonl": root / MATERIAL_INDEX_JSONL,
        "material_index_json": root / MATERIAL_INDEX_JSON,
        "key_material_references_jsonl": references_path or root / DEFAULT_REFERENCES_NAME,
        "sqlite_index": sqlite_path or root / DEFAULT_SQLITE_NAME,
        "physical_change_log_jsonl": physical_change_log_path or root / PHYSICAL_CHANGE_LOG_JSONL,
        "time_alignment_json": time_alignment_path or root / TIME_ALIGNMENT_JSON,
    }
    entrypoints: dict[str, str] = {}
    for key, value in candidates.items():
        if value is None:
            continue
        path = Path(value)
        if path.exists():
            entrypoints[key] = _path_to_relative(path, root) or path.name
    return entrypoints


def _file_entry(root: Path, rel_path: str) -> dict[str, Any] | None:
    path = root / rel_path
    if not path.exists() or not path.is_file():
        return None
    return {
        "relative_path": rel_path.replace("\\", "/"),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _video_stream_from_manifest(root: Path, view: str, item: Mapping[str, Any]) -> dict[str, Any]:
    path_value = item.get("path") or item.get("file") or item.get("video_path")
    return {
        "view": item.get("role") or view,
        "path": _path_to_relative(path_value, root) if path_value else "",
        "start_time": item.get("start_time") or item.get("session_start_time") or item.get("created_at"),
        "offset_sec": _safe_float(item.get("offset_sec"), 0.0),
        "clock_scale": _safe_float(item.get("clock_scale"), 1.0) or 1.0,
        "fps": _float_or_none(item.get("fps")),
        "duration_sec": _float_or_none(item.get("duration_sec")),
        "camera_id": item.get("camera_id") or "",
    }


def _build_time_context(
    alignment: Mapping[str, Any],
    *,
    message_sent_at: str | None,
    window_before_sec: float | None,
    window_after_sec: float | None,
) -> dict[str, Any]:
    policy = alignment.get("message_alignment_policy") if isinstance(alignment.get("message_alignment_policy"), Mapping) else {}
    before = float(window_before_sec if window_before_sec is not None else policy.get("default_window_before_sec", 90.0))
    after = float(window_after_sec if window_after_sec is not None else policy.get("default_window_after_sec", 180.0))
    video_time = map_message_time_to_video_sec(alignment, message_sent_at)
    mode = "posthoc"
    window = None
    if video_time is not None:
        max_duration = _max_video_duration(alignment)
        if max_duration <= 0 or video_time <= max_duration + after:
            mode = "live_aligned"
            window = {"start_sec": round(max(0.0, video_time - before), 3), "end_sec": round(video_time + after, 3)}
        else:
            mode = "posthoc_message_time_outside_video"
    return {
        "message_sent_at": message_sent_at,
        "message_video_time_sec": video_time,
        "mode": mode,
        "search_window": window,
        "window_before_sec": before,
        "window_after_sec": after,
    }


def _parse_query(query_text: str) -> dict[str, Any]:
    normalized = _norm(query_text)
    target_objects = [
        canonical
        for canonical, aliases in OBJECT_ALIASES.items()
        if any(_norm(alias) in normalized for alias in aliases)
    ]
    intent = "step_evidence_check"
    for candidate, aliases in INTENT_ALIASES.items():
        if any(_norm(alias) in normalized for alias in aliases):
            intent = candidate
            break
    if intent == "step_evidence_check" and ("hand_" in normalized or "hand-" in str(query_text or "").lower()):
        intent = "hand_object_check"
    tokens = set(_basic_tokens(query_text))
    for canonical in target_objects:
        tokens.add(canonical)
        tokens.update(_norm(alias) for alias in OBJECT_ALIASES.get(canonical, ()))
    tokens.update(_norm(alias) for alias in INTENT_ALIASES.get(intent, ()))
    return {"intent": intent, "target_objects": target_objects, "tokens": sorted(token for token in tokens if token)}


def _rank_references(
    references: Sequence[Mapping[str, Any]],
    physical_changes: Sequence[Mapping[str, Any]],
    *,
    query_text: str,
    query_info: Mapping[str, Any],
    window: Mapping[str, float] | None,
    limit: int,
) -> list[dict[str, Any]]:
    changes_by_material = _changes_by_material_id(physical_changes)
    scored: list[tuple[float, dict[str, Any]]] = []
    normalized_query = _norm(query_text)
    for ref_raw in references:
        ref = _with_payload(ref_raw)
        score = 0.0
        score_factors: dict[str, float] = {}
        text_blob = _norm(_reference_text_blob(ref))
        for token in query_info["tokens"]:
            if token and token in text_blob:
                score += 2.0
                score_factors["text"] = score_factors.get("text", 0.0) + 2.0
        if any(_object_matches(ref, target) for target in query_info["target_objects"]):
            score += 4.0
            score_factors["object"] = 4.0
        if _intent_matches(ref, str(query_info["intent"])):
            score += 3.0
            score_factors["intent"] = 3.0
        for change in changes_by_material.get(str(ref.get("material_id") or ""), []):
            if _intent_matches(change, str(query_info["intent"])):
                score += 2.0
                score_factors["physical_change_intent"] = score_factors.get("physical_change_intent", 0.0) + 2.0
            if any(_object_matches(change, target) for target in query_info["target_objects"]):
                score += 2.0
                score_factors["physical_change_object"] = score_factors.get("physical_change_object", 0.0) + 2.0
        if window is not None:
            time_score = _time_score(ref, window)
            score += time_score
            score_factors["time_window"] = round(time_score, 4)
        step_name = str(ref.get("step_name") or ref.get("display_name") or "")
        if step_name and _norm(step_name) in normalized_query:
            score += 3.0
            score_factors["step_name"] = 3.0
        confidence_score = min(1.0, _safe_float(ref.get("confidence") or ref.get("quality_score"), 0.0))
        score += confidence_score
        score_factors["confidence"] = round(confidence_score, 4)
        view_score = _dual_view_score(ref)
        score += view_score
        score_factors["dual_view"] = round(view_score, 4)
        evidence_score = _physical_evidence_score(ref, changes_by_material.get(str(ref.get("material_id") or ""), []))
        score += evidence_score
        score_factors["physical_evidence"] = round(evidence_score, 4)
        if score > 0:
            enriched = dict(ref)
            enriched["retrieval_score"] = round(score, 4)
            enriched["retrieval_score_factors"] = score_factors
            scored.append((score, enriched))
    scored.sort(key=lambda item: (item[0], -_safe_float(item[1].get("start_sec"), 0.0)), reverse=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, item in scored:
        key = str(item.get("micro_segment_id") or item.get("material_id") or item.get("segment_id") or "")
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(item)
        if len(deduped) >= max(1, min(int(limit), 50)):
            break
    return deduped


def _build_evidence_bundle(
    package: EvidencePackage,
    ref: Mapping[str, Any],
    physical_changes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    linked_changes = [
        dict(change)
        for change in physical_changes
        if str(ref.get("material_id") or "") in [str(value) for value in (change.get("evidence_material_ids") or [])]
        or (
            _safe_float(change.get("start_sec"), -1.0) <= _safe_float(ref.get("end_sec"), -2.0)
            and _safe_float(change.get("end_sec"), -1.0) >= _safe_float(ref.get("start_sec"), -2.0)
        )
    ]
    relative_clip = _portable_path(ref, package.root, ("formal_clip_path", "clip_path", "stored_file"))
    relative_preview = _portable_path(ref, package.root, ("formal_preview_path", "preview_path"))
    keyframes = []
    for value in ref.get("keyframe_paths") or []:
        path = _path_to_relative(value, package.root)
        if path:
            keyframes.append(_asset_ref(package, path))
    before_state, after_state = _state_pair(ref)
    window_audit = _window_audit(ref)
    return {
        "material_id": ref.get("material_id"),
        "event_id": ref.get("event_id"),
        "event_type": ref.get("event_type") or ref.get("change_type") or _change_type(ref),
        "asset_type": ref.get("asset_type"),
        "step_id": ref.get("step_id"),
        "step_name": ref.get("step_name") or ref.get("display_name"),
        "segment_id": ref.get("experiment_segment_id") or ref.get("segment_id"),
        "time_range": [ref.get("start_sec"), ref.get("end_sec")],
        "key_timestamps": ref.get("key_timestamps") or [],
        "objects": _combined_objects(ref),
        "actions": _combined_actions(ref),
        "secondary_objects": _list_strings(ref.get("secondary_objects")),
        "secondary_actions": _list_strings(ref.get("secondary_actions")),
        "window_audit": window_audit,
        "target_object_support": window_audit.get("target_object_support") if window_audit else {},
        "secondary_object_support": window_audit.get("secondary_object_support") if window_audit else [],
        "uncertainty_reasons": _list_strings(
            ref.get("uncertainty_reasons")
            or (window_audit.get("uncertainty_reasons") if window_audit else [])
            or (window_audit.get("reasons") if window_audit else [])
        ),
        "before_state": before_state,
        "after_state": after_state,
        "physical_changes": linked_changes,
        "clip": _asset_ref(package, relative_clip) if relative_clip else None,
        "preview": _asset_ref(package, relative_preview) if relative_preview else None,
        "key_frames": keyframes,
        "retrieval_score": ref.get("retrieval_score"),
        "retrieval_score_factors": ref.get("retrieval_score_factors") or {},
        "confidence": ref.get("confidence") or ref.get("quality_score"),
        "decision_path": ref.get("decision_path") or "",
        "decision_trace": _list_strings(ref.get("decision_trace")),
        "detector_backend": ref.get("detector_backend") or "",
        "source_view": ref.get("source_view") or ref.get("view") or "",
        "reason_code": ref.get("reason_code") or "",
        "key_action_index": ref.get("key_action_index") or {},
        "evidence_summary": ref.get("evidence_summary") or "",
    }


def _judge_bundles(query_info: Mapping[str, Any], bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not bundles:
        return {
            "status": "insufficient",
            "label": "evidence_insufficient",
            "confidence": 0.0,
            "reason": "No matching material evidence was found in the package.",
        }
    target_bundles = _bundles_matching_targets(bundles, query_info)
    if query_info.get("target_objects") and not target_bundles:
        return {
            "status": "insufficient",
            "label": "target_object_not_confirmed",
            "confidence": 0.25,
            "reason": "Related materials were found, but none matched the requested target object.",
            "evidence_material_id": bundles[0].get("material_id"),
            "evidence_time_range": bundles[0].get("time_range"),
        }
    judgement_bundles = target_bundles or list(bundles)
    if query_info["intent"] == "return_position_check":
        for bundle in judgement_bundles:
            violation = _position_violation(bundle)
            if violation:
                return {
                    "status": "incorrect",
                    "label": "requirement_not_met",
                    "confidence": max(0.55, _safe_float(bundle.get("confidence"), 0.55)),
                    "reason": violation,
                    "evidence_material_id": bundle.get("material_id"),
                    "evidence_time_range": bundle.get("time_range"),
                }
        has_position_evidence = any(
            _zone_from_state(bundle.get("before_state") or {}) or _zone_from_state(bundle.get("after_state") or {})
            for bundle in judgement_bundles
        )
        return {
            "status": "correct" if has_position_evidence else "insufficient",
            "label": "requirement_met" if has_position_evidence else "evidence_insufficient",
            "confidence": 0.62 if has_position_evidence else 0.35,
            "reason": (
                "Position-related evidence was retrieved and no rule-level return-position violation was detected."
                if has_position_evidence
                else "Related evidence was retrieved, but before/after position states are incomplete."
            ),
            "evidence_material_id": judgement_bundles[0].get("material_id"),
            "evidence_time_range": judgement_bundles[0].get("time_range"),
        }
    intent_labels = {
        "object_move_check": ("object_move", "object_movement_observed"),
        "hand_object_check": ("hand_object_interaction", "hand_object_interaction_observed"),
        "liquid_transfer_check": ("liquid_transfer", "liquid_transfer_observed"),
        "panel_operation_check": ("panel_operation", "panel_operation_observed"),
        "container_state_check": ("container_state_change", "container_state_change_observed"),
    }
    if query_info["intent"] in intent_labels:
        event_type, label = intent_labels[str(query_info["intent"])]
        for bundle in judgement_bundles:
            if _bundle_has_event(bundle, event_type):
                return {
                    "status": "correct",
                    "label": label,
                    "confidence": max(0.5, _safe_float(bundle.get("confidence"), 0.5)),
                    "reason": f"Retrieved evidence contains {event_type} evidence linked to the requested action.",
                    "evidence_material_id": bundle.get("material_id"),
                    "evidence_time_range": bundle.get("time_range"),
                }
        return {
            "status": "insufficient",
            "label": "event_not_confirmed",
            "confidence": 0.3,
            "reason": f"Related materials were found, but none carried confirmed {event_type} evidence.",
            "evidence_material_id": judgement_bundles[0].get("material_id"),
            "evidence_time_range": judgement_bundles[0].get("time_range"),
        }
    return {
        "status": "correct",
        "label": "related_evidence_found",
        "confidence": max(0.45, _safe_float(judgement_bundles[0].get("confidence"), 0.45)),
        "reason": "Related key material evidence was retrieved for review or rule-based judgement.",
        "evidence_material_id": judgement_bundles[0].get("material_id"),
        "evidence_time_range": judgement_bundles[0].get("time_range"),
    }


def _bundles_matching_targets(
    bundles: Sequence[Mapping[str, Any]],
    query_info: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    targets = _list_strings(query_info.get("target_objects"))
    if not targets:
        return list(bundles)
    return [
        bundle
        for bundle in bundles
        if any(_object_matches(bundle, target) for target in targets)
    ]


def _position_violation(bundle: Mapping[str, Any]) -> str | None:
    states = [(bundle.get("before_state") or {}, bundle.get("after_state") or {})]
    for change in bundle.get("physical_changes") or []:
        states.append((change.get("before") or {}, change.get("after") or {}))
    for before, after in states:
        before_zone = _zone_from_state(before)
        after_zone = _zone_from_state(after)
        if before_zone and after_zone and before_zone != after_zone:
            return f"Target object moved from zone {before_zone} to zone {after_zone}; return-position check failed."
        distance = _centroid_distance(before, after)
        if distance is not None:
            threshold = 0.08 if distance <= 2.0 else 80.0
            if distance > threshold:
                return f"Target object centroid moved about {distance:.3g}, above allowed return-position threshold."
    return None


def _bundle_has_event(bundle: Mapping[str, Any], event_type: str) -> bool:
    if str(bundle.get("event_type") or "") == event_type:
        return True
    if _change_type(bundle) == event_type:
        return True
    if event_type in _list_strings(bundle.get("actions")):
        return True
    for change in bundle.get("physical_changes") or []:
        if isinstance(change, Mapping) and str(change.get("event_type") or "") == event_type:
            return True
        if isinstance(change, Mapping) and _change_type(change) == event_type:
            return True
    return event_type in _norm(_reference_text_blob(bundle))


def _asset_ref(package: EvidencePackage, relative_path: str) -> dict[str, Any]:
    normalized = relative_path.replace("\\", "/")
    local_path = package.root / normalized
    return {
        "relative_path": normalized,
        "package_uri": f"package://{package.package_id}/{normalized}",
        "local_path": str(local_path.resolve()),
        "exists": local_path.exists(),
    }


def _portable_path(ref: Mapping[str, Any], root: Path, keys: Sequence[str]) -> str | None:
    for key in keys:
        value = ref.get(key)
        path = _path_to_relative(value, root)
        if path:
            return path
    return None


def _path_to_relative(value: Any, root: Path) -> str | None:
    if not value:
        return None
    text = str(value)
    if text.startswith("package://"):
        return text.split("/", 3)[-1] if text.count("/") >= 3 else Path(text).name
    path = Path(text)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            return path.name if path.name else None
    return text.replace("\\", "/")


def _read_material_index(root: Path, entrypoints: Mapping[str, Any]) -> list[dict[str, Any]]:
    jsonl_path = _entrypoint(root, entrypoints, "material_index_jsonl", (MATERIAL_INDEX_JSONL,))
    rows = _read_jsonl(jsonl_path)
    if rows:
        return rows
    payload = _read_json(_entrypoint(root, entrypoints, "material_index_json", (MATERIAL_INDEX_JSON,)))
    records = payload.get("records") if isinstance(payload.get("records"), list) else []
    return [item for item in records if isinstance(item, dict)]


def _references_from_material_index(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
        ref = dict(payload or row)
        ref.setdefault("material_id", row.get("material_id"))
        ref.setdefault("asset_type", row.get("asset_type") or row.get("asset_kind"))
        ref.setdefault("event_type", row.get("event_type") or row.get("action_name"))
        ref.setdefault("start_sec", row.get("start_sec"))
        ref.setdefault("end_sec", row.get("end_sec"))
        ref.setdefault("secondary_objects", _list_strings(row.get("secondary_objects")))
        ref.setdefault("secondary_actions", _list_strings(row.get("secondary_actions")))
        ref.setdefault("objects", _combined_objects({**dict(row), **ref}))
        ref.setdefault("actions", _combined_actions({**dict(row), **ref}))
        if isinstance(row.get("window_audit"), Mapping):
            ref.setdefault("window_audit", dict(row["window_audit"]))
        ref.setdefault("formal_clip_path", row.get("clip_path") or row.get("stored_file"))
        ref.setdefault("formal_preview_path", row.get("preview_path"))
        refs.append(ref)
    return refs


def _entrypoint(root: Path, entrypoints: Mapping[str, Any], key: str, fallback_names: Sequence[str]) -> Path:
    configured = entrypoints.get(key)
    if configured:
        path = root / str(configured)
        if path.exists():
            return path
    for name in fallback_names:
        path = root / name
        if path.exists():
            return path
    return root / (str(configured) if configured else fallback_names[0])


def _merged_entrypoints(*manifests: Mapping[str, Any]) -> dict[str, Any]:
    entrypoints: dict[str, Any] = {}
    for manifest in manifests:
        if isinstance(manifest.get("entrypoints"), Mapping):
            entrypoints.update(manifest["entrypoints"])
    return entrypoints


def _changes_by_material_id(changes: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for change in changes:
        for material_id in change.get("evidence_material_ids") or []:
            grouped.setdefault(str(material_id), []).append(change)
    return grouped


def _object_matches(item: Mapping[str, Any], target: str) -> bool:
    aliases = OBJECT_ALIASES.get(target, (target,))
    text = _norm(_reference_text_blob(item))
    return any(_norm(alias) in text for alias in aliases)


def _intent_matches(item: Mapping[str, Any], intent: str) -> bool:
    event_type = str(item.get("event_type") or item.get("change_type") or "")
    if intent == "return_position_check":
        return event_type in {"object_move", "hand_object_interaction"} or any(
            token in _norm(_reference_text_blob(item)) for token in INTENT_ALIASES[intent]
        )
    mapping = {
        "object_move_check": "object_move",
        "hand_object_check": "hand_object_interaction",
        "liquid_transfer_check": "liquid_transfer",
        "panel_operation_check": "panel_operation",
        "container_state_check": "container_state_change",
    }
    return event_type == mapping.get(intent) or any(
        token in _norm(_reference_text_blob(item)) for token in INTENT_ALIASES.get(intent, ())
    )


def _time_score(ref: Mapping[str, Any], window: Mapping[str, float]) -> float:
    start = _safe_float(ref.get("start_sec"), 0.0)
    end = _safe_float(ref.get("end_sec"), start)
    if end >= window["start_sec"] and start <= window["end_sec"]:
        return 4.0
    center = (start + end) / 2.0
    distance = min(abs(center - window["start_sec"]), abs(center - window["end_sec"]))
    return max(0.0, 1.5 - distance / 300.0)


def _dual_view_score(ref: Mapping[str, Any]) -> float:
    text = _reference_text_blob(ref)
    has_first = bool(ref.get("first_person_clip") or "first_person" in text)
    has_third = bool(ref.get("third_person_clip") or "third_person" in text)
    assets = ref.get("key_action_index", {}).get("assets") if isinstance(ref.get("key_action_index"), Mapping) else []
    if isinstance(assets, list):
        for item in assets:
            if isinstance(item, Mapping):
                view = str(item.get("view") or item.get("kind") or "")
                has_first = has_first or "first_person" in view
                has_third = has_third or "third_person" in view
    if has_first and has_third:
        return 1.0
    if has_first or has_third or ref.get("view"):
        return 0.4
    return 0.0


def _physical_evidence_score(ref: Mapping[str, Any], changes: Sequence[Mapping[str, Any]]) -> float:
    score = 0.0
    if changes:
        score += 1.5
    if _has_state_pair(ref):
        score += 1.0
    if ref.get("yolo_interactions") or ref.get("interaction_events") or ref.get("yolo_evidence_count"):
        score += 0.75
    audit = _window_audit(ref)
    if _safe_float(audit.get("interaction_frame_count"), 0.0) > 0:
        score += 0.5
    if audit.get("target_object_support"):
        score += 0.25
    if ref.get("decision_path"):
        score += 0.4
    return min(score, 3.0)


def _reference_text_blob(item: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "searchable_text",
        "display_name",
        "step_name",
        "event_type",
        "change_type",
        "asset_type",
        "action_name",
        "primary_object",
        "canonical_object",
        "secondary_objects",
        "secondary_actions",
        "window_audit",
        "target_object_support",
        "secondary_object_support",
        "uncertainty_reasons",
        "evidence_summary",
        "payload_json",
    ):
        if item.get(key):
            values.append(str(item[key]))
    for key in ("objects", "actions", "secondary_objects", "secondary_actions", "involved_objects", "related_detection_classes"):
        value = item.get(key)
        if isinstance(value, list):
            values.extend(str(part) for part in value)
    payload = _payload_dict(item)
    if payload:
        values.append(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return " ".join(values)


def _text_has_keywords(text: str, keywords: Sequence[str]) -> bool:
    normalized = _norm(text)
    lower_text = str(text or "").lower()
    word_tokens = set(re.findall(r"[a-z0-9_]+", normalized))
    for keyword in keywords:
        raw = str(keyword or "")
        token = _norm(raw)
        if not token:
            continue
        if re.search(r"[\u4e00-\u9fff]", raw):
            if raw in text or token in normalized:
                return True
            continue
        if token in word_tokens:
            return True
        phrase = raw.lower().replace("_", " ")
        if " " in phrase and phrase in lower_text:
            return True
    return False


def _explicit_action_text(item: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in (
        "event_type",
        "change_type",
        "action_type",
        "canonical_action_type",
        "action_name",
        "display_name",
        "step_name",
    ):
        if item.get(key):
            values.append(str(item[key]))
    for key in ("actions", "secondary_actions", "event_types"):
        values.extend(_list_strings(item.get(key)))
    payload = _payload_dict(item)
    if payload:
        for key in (
            "event_type",
            "change_type",
            "action_type",
            "canonical_action_type",
            "action_name",
            "display_name",
            "step_name",
        ):
            if payload.get(key):
                values.append(str(payload[key]))
        for key in ("actions", "secondary_actions", "event_types"):
            values.extend(_list_strings(payload.get(key)))
    return " ".join(values)


def _explicit_change_type(item: Mapping[str, Any]) -> str:
    raw = str(item.get("event_type") or item.get("change_type") or "").strip()
    if raw in ACTIVE_CHANGE_TYPES:
        return raw

    for action in _list_strings(item.get("actions")):
        if action in ACTIVE_CHANGE_TYPES:
            return action

    explicit = _explicit_action_text(item)
    explicit_norm = _norm(explicit)
    if not explicit_norm:
        return ""
    explicit_tokens = set(re.findall(r"[a-z0-9_]+", explicit_norm))
    text = explicit
    for event_type in ACTIVE_CHANGE_TYPES:
        if event_type in explicit_tokens:
            return event_type
    if "liquid_transfer" in explicit_norm or "liquid" in explicit_tokens or "pour" in explicit_tokens or "transfer" in explicit_tokens:
        return "liquid_transfer"
    if "panel_operation" in explicit_norm or "panel" in explicit_tokens or "button" in explicit_tokens or "display" in explicit_tokens:
        return "panel_operation"
    if _text_has_keywords(text, ("hand_object", "hand_object_interaction", "hand", "contact", "grasp", "鎵嬩笌", "鎺ヨЕ", "鎷胯捣")):
        return "hand_object_interaction"
    if "object_move" in explicit_norm or "movement" in explicit_tokens or "move" in explicit_tokens:
        return "object_move"
    if (
        "hand_object" in explicit_norm
        or "hand" in explicit_tokens
        or explicit_norm.startswith("hand_")
        or "\u624b\u4e0e" in explicit
    ):
        return "hand_object_interaction"
    if "container_state" in explicit_norm or "open" in explicit_tokens or "close" in explicit_tokens or "cap" in explicit_tokens or "lid" in explicit_tokens:
        return "container_state_change"
    return ""


def _change_type(item: Mapping[str, Any]) -> str:
    explicit = _explicit_change_type(item)
    if explicit:
        return explicit
    text = _reference_text_blob(item)
    if _text_has_keywords(text, ("hand_object", "hand_object_interaction", "hand", "contact", "grasp", "鎵嬩笌", "鎺ヨЕ", "鎷胯捣")):
        return "hand_object_interaction"
    if _text_has_keywords(text, ("liquid", "solution", "pour", "transfer", "液体", "溶液", "倒液", "加液", "液体转移")):
        return "liquid_transfer"
    if _text_has_keywords(text, ("panel", "button", "knob", "display", "面板", "按钮", "旋钮", "读数")):
        return "panel_operation"
    if _text_has_keywords(text, ("open", "close", "cap", "lid", "打开", "关闭", "盖子", "瓶盖")):
        return "container_state_change"
    if _text_has_keywords(text, ("hand_object", "hand_object_interaction", "hand", "contact", "grasp", "手与", "接触", "拿起")):
        return "hand_object_interaction"
    if _has_state_pair(item):
        return "object_move"
    return "hand_object_interaction"


def _evidence_strength(item: Mapping[str, Any]) -> str:
    if item.get("before_state") or item.get("after_state") or item.get("before") or item.get("after"):
        return "state_change"
    if item.get("yolo_interactions") or item.get("interaction_events") or item.get("yolo_evidence_count"):
        return "interaction_evidence"
    if item.get("decision_path") or item.get("reason_code"):
        return "detector_decision"
    return "material_reference"


def _validate_reference_paths(
    root: Path,
    ref: Mapping[str, Any],
    *,
    row_index: int,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    strict: bool,
) -> None:
    path_fields = ("stored_file", "source_file", "formal_clip_path", "clip_path", "formal_preview_path", "preview_path")
    for field in path_fields:
        value = ref.get(field)
        if not value:
            continue
        if _is_absolute_path_text(value):
            errors.append({"code": "absolute_reference_path", "row_index": row_index, "field": field, "path": str(value)})
            continue
        if field in {"stored_file", "formal_clip_path", "clip_path", "formal_preview_path", "preview_path"} and not str(value).startswith("package://"):
            exists = (root / str(value)).exists()
            if not exists:
                item = {
                    "code": "missing_reference_file",
                    "row_index": row_index,
                    "field": field,
                    "path": str(value),
                    "material_id": ref.get("material_id"),
                }
                if strict or field == "stored_file":
                    errors.append(item)
                else:
                    warnings.append(item)
    for field in ("keyframe_paths",):
        values = ref.get(field)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if _is_absolute_path_text(value):
                errors.append({"code": "absolute_reference_path", "row_index": row_index, "field": f"{field}[{index}]", "path": str(value)})
            elif strict and value and not (root / str(value)).exists():
                errors.append({"code": "missing_reference_file", "row_index": row_index, "field": f"{field}[{index}]", "path": str(value)})


def _is_absolute_path_text(value: Any) -> bool:
    if not value:
        return False
    text = str(value)
    if text.startswith("package://"):
        return False
    return bool(re.match(r"^[a-zA-Z]:[\\/]", text) or text.startswith("\\\\") or Path(text).is_absolute())


def _with_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = _payload_dict(item)
    return {**payload, **dict(item)}


def _payload_dict(item: Mapping[str, Any]) -> dict[str, Any]:
    payload = item.get("payload")
    if isinstance(payload, Mapping):
        return dict(payload)
    payload_json = item.get("payload_json")
    if isinstance(payload_json, str) and payload_json.strip():
        try:
            parsed = json.loads(payload_json)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _has_state_pair(item: Mapping[str, Any]) -> bool:
    before, after = _state_pair(item)
    return bool(before or after)


def _state_dict(item: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    explicit = _explicit_state_dict(item, prefix)
    if explicit:
        return explicit
    before, after = _derived_state_pair_from_evidence(item)
    return before if prefix == "before" else after


def _state_pair(item: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _explicit_state_dict(item, "before")
    after = _explicit_state_dict(item, "after")
    derived_before: dict[str, Any] = {}
    derived_after: dict[str, Any] = {}
    if not before or not after:
        derived_before, derived_after = _derived_state_pair_from_evidence(item)
    return before or derived_before, after or derived_after


def _explicit_state_dict(item: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    for key in (f"{prefix}_state", prefix):
        value = item.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    payload = _payload_dict(item)
    for key in (f"{prefix}_state", prefix):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _derived_state_pair_from_evidence(item: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    states = _detection_states_from_evidence(item)
    if len(states) < 2:
        return {}, {}
    before = dict(states[0])
    after = dict(states[-1])
    for state in (before, after):
        state.pop("_sort_key", None)
        state.pop("_order", None)
    return before, after


def _detection_states_from_evidence(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = _payload_dict(item)
    merged = {**payload, **dict(item)}
    target_aliases = _target_alias_tokens(merged)
    states: list[dict[str, Any]] = []
    evidence_keys = (
        "yolo_evidence",
        "yolo_interactions",
        "interaction_events",
        "interaction_keyframes",
        "detections",
        "object_detections",
        "object_tracks",
        "track_points",
    )
    for key in evidence_keys:
        if key in merged:
            states.extend(_collect_detection_states(merged[key], source=key, target_aliases=target_aliases))
    for state in states:
        state["_sort_key"] = (
            _safe_float(state.get("time_sec"), 1e12),
            _safe_float(state.get("frame_index"), 1e12),
            _safe_float(state.get("_order"), 1e12),
        )
    states.sort(key=lambda state: state["_sort_key"])
    return states


def _collect_detection_states(value: Any, *, source: str, target_aliases: set[str], depth: int = 0, order_start: int = 0) -> list[dict[str, Any]]:
    if depth > 5:
        return []
    states: list[dict[str, Any]] = []
    if isinstance(value, list):
        for offset, item in enumerate(value):
            states.extend(
                _collect_detection_states(
                    item,
                    source=source,
                    target_aliases=target_aliases,
                    depth=depth + 1,
                    order_start=order_start + offset,
                )
            )
        return states
    if not isinstance(value, Mapping):
        return states

    state = _state_from_detection(value, source=source, target_aliases=target_aliases, order=order_start)
    if state:
        states.append(state)

    nested_keys = (
        "object",
        "target",
        "detection",
        "object_detection",
        "detected_object",
        "primary_detection",
        "before_detection",
        "after_detection",
        "detections",
        "objects",
        "tracks",
        "track",
        "frames",
    )
    for key in nested_keys:
        nested = value.get(key)
        if nested is not None:
            states.extend(
                _collect_detection_states(
                    nested,
                    source=source,
                    target_aliases=target_aliases,
                    depth=depth + 1,
                    order_start=order_start,
                )
            )
    return states


def _state_from_detection(
    detection: Mapping[str, Any],
    *,
    source: str,
    target_aliases: set[str],
    order: int,
) -> dict[str, Any]:
    bbox = _bbox_from_detection(detection)
    if bbox is None:
        return {}
    label = _detection_label(detection)
    if target_aliases and label and _norm(label) not in target_aliases:
        text = _norm(_reference_text_blob(detection))
        if not any(alias in text for alias in target_aliases):
            return {}
    frame_width = _float_or_none(
        detection.get("frame_width")
        or detection.get("image_width")
        or detection.get("video_width")
        or detection.get("width")
    )
    frame_height = _float_or_none(
        detection.get("frame_height")
        or detection.get("image_height")
        or detection.get("video_height")
        or detection.get("height")
    )
    centroid = [round((bbox[0] + bbox[2]) / 2.0, 4), round((bbox[1] + bbox[3]) / 2.0, 4)]
    state: dict[str, Any] = {
        "source": source,
        "bbox": [round(value, 4) for value in bbox],
        "centroid": centroid,
        "zone": _zone_from_bbox_centroid(centroid, bbox, frame_width=frame_width, frame_height=frame_height),
        "_order": order,
    }
    if label:
        state["label"] = label
    confidence = _float_or_none(detection.get("confidence") or detection.get("conf") or detection.get("score"))
    if confidence is not None:
        state["confidence"] = confidence
    time_sec = _float_or_none(
        detection.get("time_sec")
        or detection.get("timestamp_sec")
        or detection.get("local_time_sec")
        or detection.get("frame_time_sec")
        or detection.get("t_sec")
        or detection.get("sec")
    )
    if time_sec is not None:
        state["time_sec"] = time_sec
    frame_index = _float_or_none(detection.get("frame_index") or detection.get("frame") or detection.get("frame_no"))
    if frame_index is not None:
        state["frame_index"] = frame_index
    return state


def _bbox_from_detection(detection: Mapping[str, Any]) -> list[float] | None:
    for key in ("bbox", "box", "xyxy", "bounding_box"):
        value = detection.get(key)
        bbox = _bbox_from_value(value)
        if bbox is not None:
            return bbox
    if all(key in detection for key in ("x1", "y1", "x2", "y2")):
        return [
            _safe_float(detection.get("x1")),
            _safe_float(detection.get("y1")),
            _safe_float(detection.get("x2")),
            _safe_float(detection.get("y2")),
        ]
    if all(key in detection for key in ("left", "top", "right", "bottom")):
        return [
            _safe_float(detection.get("left")),
            _safe_float(detection.get("top")),
            _safe_float(detection.get("right")),
            _safe_float(detection.get("bottom")),
        ]
    if all(key in detection for key in ("x", "y", "w", "h")):
        x = _safe_float(detection.get("x"))
        y = _safe_float(detection.get("y"))
        return [x, y, x + _safe_float(detection.get("w")), y + _safe_float(detection.get("h"))]
    if all(key in detection for key in ("cx", "cy", "w", "h")):
        cx = _safe_float(detection.get("cx"))
        cy = _safe_float(detection.get("cy"))
        half_w = _safe_float(detection.get("w")) / 2.0
        half_h = _safe_float(detection.get("h")) / 2.0
        return [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    return None


def _bbox_from_value(value: Any) -> list[float] | None:
    if isinstance(value, Mapping):
        return _bbox_from_detection(value)
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
        except (TypeError, ValueError):
            return None
    return None


def _detection_label(detection: Mapping[str, Any]) -> str:
    for key in ("label", "object_label", "class_name", "class", "name", "object_name", "canonical_object", "primary_object"):
        value = detection.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _target_alias_tokens(item: Mapping[str, Any]) -> set[str]:
    raw_targets = [
        *_list_strings(item.get("objects")),
        *_list_strings(item.get("secondary_objects")),
    ] or _list_strings(item.get("primary_object") or item.get("canonical_object") or item.get("object_label"))
    tokens = {_norm(value) for value in raw_targets if _norm(value)}
    for canonical, aliases in OBJECT_ALIASES.items():
        if canonical in tokens or any(_norm(alias) in tokens for alias in aliases):
            tokens.add(canonical)
            tokens.update(_norm(alias) for alias in aliases)
    tokens.discard("gloved_hand")
    tokens.discard("hand")
    tokens.discard("手")
    return tokens


def _zone_from_bbox_centroid(
    centroid: Sequence[float],
    bbox: Sequence[float],
    *,
    frame_width: float | None,
    frame_height: float | None,
) -> str:
    max_coord = max(abs(value) for value in bbox) if bbox else 1.0
    width = frame_width if frame_width and frame_width > 0 else (1.0 if max_coord <= 1.5 else 1920.0)
    height = frame_height if frame_height and frame_height > 0 else (1.0 if max_coord <= 1.5 else 1080.0)
    x_ratio = max(0.0, min(1.0, centroid[0] / width))
    y_ratio = max(0.0, min(1.0, centroid[1] / height))
    col = "left" if x_ratio < 1 / 3 else "center" if x_ratio < 2 / 3 else "right"
    row = "top" if y_ratio < 1 / 3 else "middle" if y_ratio < 2 / 3 else "bottom"
    return f"{row}_{col}"


def _zone_from_state(state: Mapping[str, Any]) -> str | None:
    if not isinstance(state, Mapping):
        return None
    for key in ("zone", "region", "area", "location"):
        if state.get(key):
            return str(state[key])
    position = state.get("position") if isinstance(state.get("position"), Mapping) else {}
    for key in ("zone", "region", "area"):
        if position.get(key):
            return str(position[key])
    return None


def _centroid_distance(before: Mapping[str, Any], after: Mapping[str, Any]) -> float | None:
    before_point = _centroid(before)
    after_point = _centroid(after)
    if before_point is None or after_point is None:
        return None
    return math.dist(before_point, after_point)


def _centroid(state: Mapping[str, Any]) -> list[float] | None:
    for key in ("centroid", "center", "xy"):
        value = state.get(key)
        if isinstance(value, list) and len(value) >= 2:
            return [_safe_float(value[0], 0.0), _safe_float(value[1], 0.0)]
    bbox = state.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 4:
        return [
            (_safe_float(bbox[0], 0.0) + _safe_float(bbox[2], 0.0)) / 2.0,
            (_safe_float(bbox[1], 0.0) + _safe_float(bbox[3], 0.0)) / 2.0,
        ]
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _max_video_duration(alignment: Mapping[str, Any]) -> float:
    streams = alignment.get("video_streams") if isinstance(alignment.get("video_streams"), list) else []
    durations = [_safe_float(stream.get("duration_sec"), 0.0) for stream in streams if isinstance(stream, Mapping)]
    return max(durations) if durations else 0.0


def _basic_tokens(value: str) -> list[str]:
    normalized = _norm(value)
    tokens = re.findall(r"[\w\u4e00-\u9fff]{2,}", normalized)
    for aliases in (*OBJECT_ALIASES.values(), *INTENT_ALIASES.values()):
        for alias in aliases:
            token = _norm(alias)
            if token and token in normalized:
                tokens.append(token)
    return tokens


def _list_strings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None and str(item).strip()]
    return [str(value)]


def _dedupe_strings(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = _norm(text)
        if text and key not in seen:
            seen.add(key)
            ordered.append(text)
    return ordered


def _combined_objects(item: Mapping[str, Any]) -> list[str]:
    values = [
        *_list_strings(item.get("primary_object")),
        *_list_strings(item.get("canonical_object")),
        *_list_strings(item.get("object_label")),
        *_list_strings(item.get("objects")),
        *_list_strings(item.get("secondary_objects")),
    ]
    return _dedupe_strings(values)


def _combined_actions(item: Mapping[str, Any]) -> list[str]:
    values = [
        *_list_strings(item.get("actions")),
        *_list_strings(item.get("secondary_actions")),
        *_list_strings(item.get("action_name")),
        *_list_strings(item.get("canonical_action_type")),
        *_list_strings(item.get("event_type")),
    ]
    return _dedupe_strings(values)


def _window_audit(item: Mapping[str, Any]) -> dict[str, Any]:
    value = item.get("window_audit")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "EVIDENCE_PACKAGE_MANIFEST",
    "EVIDENCE_PACKAGE_EVAL_SCHEMA_VERSION",
    "EvidencePackage",
    "PHYSICAL_CHANGE_LOG_JSONL",
    "QUERY_RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TIME_ALIGNMENT_JSON",
    "build_evidence_package",
    "build_evidence_package_manifest",
    "build_physical_change_log_rows",
    "build_time_alignment_payload",
    "enrich_references_with_key_action_index",
    "evaluate_evidence_package_queries",
    "map_message_time_to_video_sec",
    "query_evidence_package",
    "validate_evidence_package",
]
