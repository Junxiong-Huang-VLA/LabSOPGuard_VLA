from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .artifact_schema import ARTIFACT_SPECS, SCHEMA_VERSION, get_artifact_spec, validate_artifact_file, validate_session_artifacts
from .process_record import build_process_record
from .reviewed_dataset import (
    REVIEWED_EVIDENCE_FILENAME,
    REVIEWED_MANIFEST_FILENAME,
    REVIEWED_MICROS_FILENAME,
    REVIEWED_SEGMENTS_FILENAME,
    REVIEWED_VECTOR_METADATA_FILENAME,
    freeze_reviewed_dataset,
    reviewed_index_dir,
)
from .schemas import write_jsonl
from .vector_index import VectorIndex


EXPORT_SCHEMA_VERSION = "key_action_export_bundle.v1"
DB_WRITE_PACKAGE_SCHEMA_VERSION = "key_action_db_write_package.v1"
RETRIEVAL_INTERFACE_SCHEMA_VERSION = "key_action_retrieval_interface.v1"
REPORT_INTERFACE_SCHEMA_VERSION = "key_action_report_interface.v1"
EXPORT_MANIFEST_FILENAME = "artifact_export_manifest.json"
EXPORT_SUMMARY_FILENAME = "artifact_export_summary.json"
EXPORT_HASH_FILENAME = "artifact_export_hashes.json"
DB_WRITE_PACKAGE_FILENAME = "db_write_package.json"
RETRIEVAL_INTERFACE_FILENAME = "retrieval_interface.json"
REPORT_INTERFACE_FILENAME = "report_interface.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _privacy_level(path: Path, artifact_type: str = "") -> str:
    text = f"{path} {artifact_type}".lower()
    if any(token in text for token in ("raw", ".mp4", ".mov", ".avi", ".mkv", "video")):
        return "restricted"
    if any(token in text for token in ("transcript", "dialogue", "confirmation", "audit")):
        return "confidential"
    return "internal"


def _export_audit_event(
    event_type: str,
    *,
    created_at: str,
    source: Path,
    target: Path,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "created_at": created_at,
        "source_path": str(source),
        "target_path": str(target),
        "details": dict(details or {}),
    }


def _privacy_level(path: Path, artifact_type: str = "") -> str:
    text = f"{path} {artifact_type}".lower()
    if any(token in text for token in ("raw", ".mp4", ".mov", ".avi", ".mkv", "video")):
        return "restricted"
    if any(token in text for token in ("transcript", "dialogue", "confirmation", "audit")):
        return "confidential"
    return "internal"


def _audit_trail(
    *,
    event_type: str,
    created_at: str,
    source: Path,
    target: Path,
    details: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "event_type": event_type,
            "created_at": created_at,
            "source_path": str(source),
            "target_path": str(target),
            "details": dict(details or {}),
        }
    ]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if text:
                row = json.loads(text)
                if isinstance(row, dict):
                    yield row


def _counter_dict(values: Iterable[Any]) -> dict[str, int]:
    counts = Counter(str(value or "unknown") for value in values)
    return dict(sorted(counts.items()))


def _read_json_if_present(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return _read_json(path)


def _read_jsonl_if_present(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path)) if path.exists() else []


def _infer_source_session_id(session: Path, process: Mapping[str, Any] | None = None, manifest: Mapping[str, Any] | None = None) -> str:
    for source in (process, manifest):
        if isinstance(source, Mapping) and source.get("session_id"):
            return str(source["session_id"])
    return session.name


def _audit_entry(action: str, *, source_session_id: str, actor: str = "key_action_indexer", details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "actor": actor,
        "action": action,
        "source_session_id": source_session_id,
        "details": dict(details or {}),
    }


def _versioned_row(
    row: Mapping[str, Any],
    *,
    source_session_id: str,
    version: int,
    audit_trail: list[dict[str, Any]],
) -> dict[str, Any]:
    item = dict(row)
    item.setdefault("source_session_id", source_session_id)
    item.setdefault("version", version)
    item.setdefault("audit_trail", audit_trail)
    return item


def _status_from_quality(row: Mapping[str, Any]) -> str:
    quality = row.get("quality")
    if isinstance(quality, Mapping):
        return str(quality.get("status") or "unknown")
    return "unknown"


def _summarize_jsonl_artifact(artifact_type: str, path: Path) -> dict[str, Any]:
    rows = list(_iter_jsonl(path))
    summary: dict[str, Any] = {"record_count": len(rows)}
    if artifact_type == "video_understanding":
        summary["event_type_counts"] = _counter_dict(row.get("event_type") for row in rows)
        summary["low_confidence_count"] = sum(1 for row in rows if float(row.get("confidence") or 0.0) < 0.45)
    elif artifact_type == "model_observation_events":
        summary["source_type_counts"] = _counter_dict(row.get("source_type") for row in rows)
        summary["event_type_counts"] = _counter_dict(row.get("event_type") for row in rows)
        summary["confirmed_or_measured_count"] = sum(
            1
            for row in rows
            if str(row.get("confirmation_level") or "").lower() in {"confirmed", "measured"}
        )
    elif artifact_type == "asset_catalog":
        summary["asset_type_counts"] = _counter_dict(row.get("asset_type") for row in rows)
        summary["quality_status_counts"] = _counter_dict(_status_from_quality(row) for row in rows)
        summary["missing_count"] = sum(1 for row in rows if _status_from_quality(row) == "missing")
        summary["dry_run_count"] = sum(1 for row in rows if bool(row.get("dry_run_placeholder")))
    elif artifact_type == "confirmation_queue":
        summary["status_counts"] = _counter_dict(row.get("status") for row in rows)
        summary["pending_count"] = sum(1 for row in rows if row.get("status") == "pending")
    return summary


def _summarize_json_artifact(artifact_type: str, path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, Mapping):
        return {"record_count": 1}
    if artifact_type == "experiment_context":
        return {
            "record_count": 1,
            "session_id": data.get("session_id"),
            "purpose": data.get("purpose"),
            "procedure_candidate_count": len(data.get("procedure_candidates") or []),
            "material_count": len(data.get("materials") or []),
            "parameter_count": len(data.get("parameters") or []),
            "confidence": data.get("confidence"),
            "gap_count": len(data.get("gaps") or []),
        }
    if artifact_type == "experiment_process":
        steps = data.get("steps") or []
        return {
            "record_count": 1,
            "session_id": data.get("session_id"),
            "process_status": data.get("process_status"),
            "step_count": data.get("step_count", len(steps)),
            "status_counts": data.get("status_counts") or {},
            "pending_confirmation_count": sum(
                1
                for step in steps
                if isinstance(step, Mapping) and step.get("requires_human_confirmation")
            ),
        }
    if artifact_type == "process_record":
        summary = data.get("summary") if isinstance(data.get("summary"), Mapping) else {}
        return {
            "record_count": 1,
            "session_id": data.get("session_id"),
            "step_count": summary.get("step_count", len(data.get("steps") or [])),
            "inferred_step_count": summary.get("inferred_step_count", 0),
            "pending_confirmation_count": summary.get("pending_confirmation_count", 0),
            "weak_evidence_step_count": summary.get("weak_evidence_step_count", 0),
            "evidence_count": len(data.get("evidence") or []),
            "audit_report_path": data.get("audit_report_path"),
        }
    return {"record_count": 1}


def summarize_artifact_file(path: str | Path, artifact_type: str) -> dict[str, Any]:
    source = Path(path)
    spec = get_artifact_spec(artifact_type)
    if not source.exists():
        return {"record_count": 0, "missing": True}
    if spec.file_format == "jsonl":
        return _summarize_jsonl_artifact(artifact_type, source)
    return _summarize_json_artifact(artifact_type, source)


def _copy_if_needed(source: Path, target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return False
    shutil.copy2(source, target)
    return True


def _copy_tree_if_present(source: Path, target: Path) -> bool:
    if not source.exists() or not source.is_dir():
        return False
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        _copy_if_needed(item, target / relative)
    return True


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    created_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
        "created_at": created_at,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "privacy_level": _privacy_level(path),
        "audit_trail": [
            {
                "event_type": "export_hash_recorded",
                "created_at": created_at,
                "source_path": str(path),
                "details": {"relative_path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path)},
            }
        ],
    }


def _hash_entries(output: Path) -> list[dict[str, Any]]:
    excluded = {EXPORT_MANIFEST_FILENAME, EXPORT_HASH_FILENAME}
    return [
        _file_entry(path, output)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _write_hash_manifest(output: Path) -> dict[str, Any]:
    entries = _hash_entries(output)
    payload = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generated_at": _now(),
        "file_count": len(entries),
        "files": entries,
    }
    target = output / EXPORT_HASH_FILENAME
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["hashes_path"] = str(target)
    return payload


def _latest_report_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size) if path.exists() else 0,
        "sha256": _sha256(path) if path.exists() else None,
    }


def _export_optional_file(source: Path, target: Path) -> dict[str, Any]:
    created_at = _now()
    entry: dict[str, Any] = {
        "source_path": str(source),
        "target_path": str(target),
        "exists": source.exists(),
        "copied": False,
        "size_bytes": 0,
        "sha256": None,
        "created_at": created_at,
        "schema_version": EXPORT_SCHEMA_VERSION,
        "privacy_level": _privacy_level(source),
        "audit_trail": [
            _export_audit_event("export_optional_prepared", created_at=created_at, source=source, target=target)
        ],
    }
    if source.exists():
        entry["copied"] = _copy_if_needed(source, target)
        entry["size_bytes"] = int(source.stat().st_size)
        entry["sha256"] = _sha256(source)
        entry["audit_trail"].append(
            _export_audit_event(
                "export_optional_copied" if entry["copied"] else "export_optional_reused",
                created_at=created_at,
                source=source,
                target=target,
                details={"sha256": entry["sha256"], "size_bytes": entry["size_bytes"]},
            )
        )
    else:
        entry["audit_trail"].append(
            _export_audit_event("export_optional_missing_source", created_at=created_at, source=source, target=target)
        )
    return entry


def _timeline_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    anchored = sum(1 for row in rows if row.get("global_time") or row.get("global_start_time"))
    return {
        "record_count": len(rows),
        "event_type_counts": _counter_dict(row.get("event_type") for row in rows),
        "anchored_count": anchored,
        "anchor_coverage": round(anchored / len(rows), 6) if rows else 0.0,
    }


def _build_db_write_package(session: Path, output: Path, *, version: int = 1) -> dict[str, Any]:
    metadata = session / "metadata"
    process = _read_json_if_present(metadata / "experiment_process.json", {})
    manifest = _read_json_if_present(session / "manifest.json", {})
    source_session_id = _infer_source_session_id(session, process, manifest)
    audit = [
        _audit_entry(
            "db_write_package_created",
            source_session_id=source_session_id,
            details={"session_dir": str(session), "output_dir": str(output)},
        )
    ]
    steps = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in (process.get("steps") if isinstance(process.get("steps"), list) else [])
        if isinstance(row, Mapping)
    ]
    process_timeline = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in _read_jsonl_if_present(metadata / "experiment_process_timeline.jsonl")
    ]
    unified_timeline = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in _read_jsonl_if_present(metadata / "unified_multimodal_timeline.jsonl")
    ]
    video_events = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in _read_jsonl_if_present(metadata / "video_understanding.jsonl")
    ]
    model_events = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in _read_jsonl_if_present(metadata / "model_observation_events.jsonl")
    ]
    assets = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in _read_jsonl_if_present(metadata / "material_asset_catalog.jsonl")
    ]
    confirmations = [
        _versioned_row(row, source_session_id=source_session_id, version=version, audit_trail=audit)
        for row in _read_jsonl_if_present(metadata / "human_confirmation_queue.jsonl")
    ]
    package = {
        "schema_version": DB_WRITE_PACKAGE_SCHEMA_VERSION,
        "generated_at": _now(),
        "source_session_id": source_session_id,
        "version": version,
        "audit_trail": audit,
        "write_mode": "upsert",
        "idempotency_keys": {
            "session": ["source_session_id", "version"],
            "process_steps": ["source_session_id", "version", "step_id"],
            "process_timeline_events": ["source_session_id", "version", "timeline_event_id"],
            "evidence_events": ["source_session_id", "version", "video_event_id", "observation_id"],
        },
        "tables": {
            "experiment_sessions": [
                {
                    "source_session_id": source_session_id,
                    "version": version,
                    "session_dir": str(session),
                    "process_status": process.get("process_status") if isinstance(process, Mapping) else None,
                    "step_count": process.get("step_count") if isinstance(process, Mapping) else len(steps),
                    "audit_trail": audit,
                }
            ],
            "process_steps": steps,
            "process_timeline_events": process_timeline,
            "unified_timeline_events": unified_timeline,
            "evidence_events": [*video_events, *model_events],
            "material_assets": assets,
            "confirmation_items": confirmations,
        },
        "table_counts": {
            "experiment_sessions": 1,
            "process_steps": len(steps),
            "process_timeline_events": len(process_timeline),
            "unified_timeline_events": len(unified_timeline),
            "evidence_events": len(video_events) + len(model_events),
            "material_assets": len(assets),
            "confirmation_items": len(confirmations),
        },
    }
    target = output / DB_WRITE_PACKAGE_FILENAME
    target.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package["package_path"] = str(target)
    return package


def _load_vector_metadata(session: Path) -> list[dict[str, Any]]:
    metadata = session / "metadata"
    for candidate in (
        metadata / REVIEWED_VECTOR_METADATA_FILENAME,
        metadata / "vector_metadata.jsonl",
        metadata / "micro_vector_metadata.jsonl",
        metadata / "key_action_segments.jsonl",
    ):
        rows = _read_jsonl_if_present(candidate)
        if rows:
            return rows
    return []


def _build_or_copy_reusable_index(session: Path, output: Path) -> dict[str, Any]:
    source_index = reviewed_index_dir(session)
    target_index = output / "reusable_index"
    copied_existing = _copy_tree_if_present(source_index, target_index)
    if not (target_index / "fallback_index.pkl").exists():
        rows = _load_vector_metadata(session)
        texts = [str(row.get("index_text") or row.get("text") or row.get("summary") or "") for row in rows]
        index = VectorIndex()
        index.build(texts, rows)
        index.save(target_index)
        write_jsonl(target_index / "docstore.jsonl", rows)
        copied_existing = False
    files = [_file_entry(path, output) for path in sorted(target_index.rglob("*")) if path.is_file()]
    return {
        "index_dir": str(target_index),
        "copied_existing": copied_existing,
        "file_count": len(files),
        "metadata_count": len(_read_jsonl_if_present(target_index / "vector_metadata.jsonl")),
        "files": files,
    }


def _build_retrieval_interface(session: Path, output: Path, reusable_index: Mapping[str, Any], *, version: int = 1) -> dict[str, Any]:
    process = _read_json_if_present(session / "metadata" / "experiment_process.json", {})
    manifest = _read_json_if_present(session / "manifest.json", {})
    source_session_id = _infer_source_session_id(session, process, manifest)
    payload = {
        "schema_version": RETRIEVAL_INTERFACE_SCHEMA_VERSION,
        "generated_at": _now(),
        "source_session_id": source_session_id,
        "version": version,
        "reusable_index": dict(reusable_index),
        "query_contract": {
            "command": "python -m key_action_indexer.cli query --index-dir <reusable_index> --query <text> --top-k <n>",
            "filters": {
                "index_level": ["segment", "micro_segment", "all"],
                "primary_object": "optional object label",
                "interaction_type": "optional interaction type",
            },
            "result_fields": [
                "score",
                "vector_score",
                "rerank_score",
                "segment_id",
                "micro_segment_id",
                "global_start_time",
                "global_end_time",
                "evidence_level",
                "evidence_reasons",
                "limitations",
            ],
        },
        "metadata_sources": {
            "vector_metadata": str(session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME)
            if (session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME).exists()
            else str(session / "metadata" / "vector_metadata.jsonl"),
            "micro_vector_metadata": str(session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME)
            if (session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME).exists()
            else str(session / "metadata" / "micro_vector_metadata.jsonl"),
            "reviewed_dataset_manifest": str(session / "metadata" / REVIEWED_MANIFEST_FILENAME),
            "asset_catalog": str(session / "metadata" / "material_asset_catalog.jsonl"),
        },
    }
    target = output / RETRIEVAL_INTERFACE_FILENAME
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["interface_path"] = str(target)
    return payload


def _build_report_interface(session: Path, output: Path, *, version: int = 1) -> dict[str, Any]:
    process = _read_json_if_present(session / "metadata" / "experiment_process.json", {})
    manifest = _read_json_if_present(session / "manifest.json", {})
    source_session_id = _infer_source_session_id(session, process, manifest)
    reports = {
        "mvp_validation_report": _latest_report_status(session / "reports" / "mvp_validation_report.md"),
        "formal_validation_report": _latest_report_status(session / "reports" / "formal_validation_report.md"),
        "process_audit_report": _latest_report_status(session / "reports" / "process_audit_report.md"),
        "process_quality_report": _latest_report_status(session / "metadata" / "process_quality_report.json"),
        "time_calibration_report": _latest_report_status(session / "metadata" / "time_calibration_report.json"),
    }
    payload = {
        "schema_version": REPORT_INTERFACE_SCHEMA_VERSION,
        "generated_at": _now(),
        "source_session_id": source_session_id,
        "version": version,
        "reports": reports,
        "report_contract": {
            "markdown_reports": ["mvp_validation_report", "formal_validation_report", "process_audit_report"],
            "machine_readable_reports": ["process_quality_report", "time_calibration_report", "process_record"],
            "primary_join_keys": ["source_session_id", "segment_id", "micro_segment_id", "step_id", "timeline_event_id"],
        },
    }
    target = output / REPORT_INTERFACE_FILENAME
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["interface_path"] = str(target)
    return payload


def _artifact_entry(
    *,
    artifact_type: str,
    source: Path,
    target: Path,
    validate: bool,
) -> dict[str, Any]:
    spec = get_artifact_spec(artifact_type)
    created_at = _now()
    entry: dict[str, Any] = {
        "artifact_type": artifact_type,
        "source_path": str(source),
        "target_path": str(target),
        "file_format": spec.file_format,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "privacy_level": _privacy_level(source, artifact_type),
        "audit_trail": _audit_trail(
            event_type="export_prepared",
            created_at=created_at,
            source=source,
            target=target,
            details={"artifact_type": artifact_type, "relative_path": spec.relative_path},
        ),
        "exists": source.exists(),
        "copied": False,
        "size_bytes": 0,
        "sha256": None,
        "record_count": 0,
        "summary": {"record_count": 0, "missing": not source.exists()},
        "validation": None,
    }
    if not source.exists():
        if validate:
            entry["validation"] = validate_artifact_file(source, artifact_type)
        entry["audit_trail"].append(
            {
                "event_type": "export_missing_source",
                "created_at": created_at,
                "source_path": str(source),
                "target_path": str(target),
                "details": {"artifact_type": artifact_type},
            }
        )
        return entry

    copied = _copy_if_needed(source, target)
    summary = summarize_artifact_file(source, artifact_type)
    validation = validate_artifact_file(source, artifact_type) if validate else None
    entry.update(
        {
            "copied": copied,
            "size_bytes": int(source.stat().st_size),
            "sha256": _sha256(source),
            "record_count": int(summary.get("record_count") or 0),
            "summary": summary,
            "validation": validation,
        }
    )
    entry["audit_trail"].append(
        {
            "event_type": "export_copied" if copied else "export_reused",
            "created_at": created_at,
            "source_path": str(source),
            "target_path": str(target),
            "details": {
                "artifact_type": artifact_type,
                "sha256": entry["sha256"],
                "record_count": entry["record_count"],
            },
        }
    )
    return entry


def export_artifact_bundle(
    session_dir: str | Path,
    output_dir: str | Path,
    artifact_types: Iterable[str] | None = None,
    validate: bool = True,
    include_interfaces: bool = True,
    include_reusable_index: bool = True,
) -> dict[str, Any]:
    session = Path(session_dir)
    output = Path(output_dir)
    artifacts_dir = output / "artifacts"
    selected = list(artifact_types) if artifact_types is not None else list(ARTIFACT_SPECS)
    output.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    if "process_record" in selected and not (session / "exports" / "process_record.json").exists():
        try:
            build_process_record(session)
        except Exception:
            pass
    try:
        freeze_reviewed_dataset(session, create_release=False)
    except Exception:
        pass

    artifacts = []
    for artifact_type in selected:
        spec = get_artifact_spec(artifact_type)
        source = session / spec.relative_path
        target = artifacts_dir / Path(spec.relative_path).name
        artifacts.append(_artifact_entry(artifact_type=artifact_type, source=source, target=target, validate=validate))

    supplemental = {
        "experiment_process_timeline": _export_optional_file(
            session / "metadata" / "experiment_process_timeline.jsonl",
            artifacts_dir / "experiment_process_timeline.jsonl",
        ),
        "unified_multimodal_timeline": _export_optional_file(
            session / "metadata" / "unified_multimodal_timeline.jsonl",
            artifacts_dir / "unified_multimodal_timeline.jsonl",
        ),
        "time_calibration_report": _export_optional_file(
            session / "metadata" / "time_calibration_report.json",
            artifacts_dir / "time_calibration_report.json",
        ),
        "vector_metadata": _export_optional_file(
            session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME
            if (session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME).exists()
            else session / "metadata" / "vector_metadata.jsonl",
            artifacts_dir / REVIEWED_VECTOR_METADATA_FILENAME,
        ),
        "reviewed_dataset_manifest": _export_optional_file(
            session / "metadata" / REVIEWED_MANIFEST_FILENAME,
            artifacts_dir / REVIEWED_MANIFEST_FILENAME,
        ),
        "reviewed_segments": _export_optional_file(
            session / "metadata" / REVIEWED_SEGMENTS_FILENAME,
            artifacts_dir / REVIEWED_SEGMENTS_FILENAME,
        ),
        "reviewed_micro_segments": _export_optional_file(
            session / "metadata" / REVIEWED_MICROS_FILENAME,
            artifacts_dir / REVIEWED_MICROS_FILENAME,
        ),
        "reviewed_evidence": _export_optional_file(
            session / "metadata" / REVIEWED_EVIDENCE_FILENAME,
            artifacts_dir / REVIEWED_EVIDENCE_FILENAME,
        ),
        "pipeline_summary": _export_optional_file(
            session / "pipeline_summary.json",
            artifacts_dir / "pipeline_summary.json",
        ),
        "process_audit_report": _export_optional_file(
            session / "reports" / "process_audit_report.md",
            artifacts_dir / "process_audit_report.md",
        ),
        "session_manifest": _export_optional_file(
            session / "manifest.json",
            artifacts_dir / "manifest.json",
        ),
    }
    process_timeline_rows = _read_jsonl_if_present(session / "metadata" / "experiment_process_timeline.jsonl")
    unified_timeline_rows = _read_jsonl_if_present(session / "metadata" / "unified_multimodal_timeline.jsonl")

    reusable_index = _build_or_copy_reusable_index(session, output) if include_reusable_index else None
    generated_interfaces: dict[str, Any] = {}
    if include_interfaces:
        db_package = _build_db_write_package(session, output)
        report_interface = _build_report_interface(session, output)
        retrieval_interface = _build_retrieval_interface(session, output, reusable_index or {})
        generated_interfaces = {
            "db_write_package": {
                "path": db_package["package_path"],
                "schema_version": db_package["schema_version"],
                "table_counts": db_package["table_counts"],
                "source_session_id": db_package["source_session_id"],
                "version": db_package["version"],
            },
            "retrieval_interface": {
                "path": retrieval_interface["interface_path"],
                "schema_version": retrieval_interface["schema_version"],
                "source_session_id": retrieval_interface["source_session_id"],
                "version": retrieval_interface["version"],
            },
            "report_interface": {
                "path": report_interface["interface_path"],
                "schema_version": report_interface["schema_version"],
                "source_session_id": report_interface["source_session_id"],
                "version": report_interface["version"],
            },
        }

    validation_summary = validate_session_artifacts(session, artifact_types=selected) if validate else None
    copied_count = sum(1 for item in artifacts if item["copied"])
    available_count = sum(1 for item in artifacts if item["exists"])
    missing_count = sum(1 for item in artifacts if not item["exists"])
    total_records = sum(int(item.get("record_count") or 0) for item in artifacts)
    summary = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "artifact_schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "session_dir": str(session),
        "artifact_count": len(artifacts),
        "available_count": available_count,
        "copied_count": copied_count,
        "missing_count": missing_count,
        "record_count": total_records,
        "governance_fields": ["source_path", "sha256", "created_at", "schema_version", "privacy_level", "audit_trail"],
        "artifacts": {item["artifact_type"]: item["summary"] for item in artifacts},
        "supplemental_artifacts": {
            "experiment_process_timeline": _timeline_summary(process_timeline_rows),
            "unified_multimodal_timeline": _timeline_summary(unified_timeline_rows),
            "vector_metadata": {
                "record_count": len(
                    _read_jsonl_if_present(
                        session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME
                        if (session / "metadata" / REVIEWED_VECTOR_METADATA_FILENAME).exists()
                        else session / "metadata" / "vector_metadata.jsonl"
                    )
                ),
            },
        },
        "interfaces": generated_interfaces,
        "reusable_index": reusable_index,
    }

    summary_path = output / EXPORT_SUMMARY_FILENAME
    manifest_path = output / EXPORT_MANIFEST_FILENAME
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "artifact_schema_version": SCHEMA_VERSION,
        "generated_at": summary["generated_at"],
        "session_dir": str(session),
        "output_dir": str(output),
        "artifacts_dir": str(artifacts_dir),
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "artifact_count": len(artifacts),
        "available_count": available_count,
        "copied_count": copied_count,
        "missing_count": missing_count,
        "record_count": total_records,
        "governance_fields": ["source_path", "sha256", "created_at", "schema_version", "privacy_level", "audit_trail"],
        "valid": bool(validation_summary["valid"]) if validation_summary is not None else None,
        "validation": validation_summary,
        "artifacts": artifacts,
        "supplemental_artifacts": supplemental,
        "generated_interfaces": generated_interfaces,
        "reusable_index": reusable_index,
    }
    hashes = _write_hash_manifest(output)
    manifest["hashes_path"] = hashes["hashes_path"]
    manifest["hashes"] = {
        "file_count": hashes["file_count"],
        "manifest_algorithm": "sha256",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


__all__ = [
    "DB_WRITE_PACKAGE_FILENAME",
    "DB_WRITE_PACKAGE_SCHEMA_VERSION",
    "EXPORT_HASH_FILENAME",
    "EXPORT_MANIFEST_FILENAME",
    "EXPORT_SCHEMA_VERSION",
    "EXPORT_SUMMARY_FILENAME",
    "REPORT_INTERFACE_FILENAME",
    "REPORT_INTERFACE_SCHEMA_VERSION",
    "RETRIEVAL_INTERFACE_FILENAME",
    "RETRIEVAL_INTERFACE_SCHEMA_VERSION",
    "export_artifact_bundle",
    "summarize_artifact_file",
]
