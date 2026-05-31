from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .artifact_schema import ARTIFACT_SPECS, OPTIONAL_ARTIFACT_SPECS, SCHEMA_VERSION
from .schemas import read_jsonl


GOVERNANCE_SCHEMA_VERSION = "key_action_data_governance.v1"
_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    cache_key = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
    cached = _SHA256_CACHE.get(cache_key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


def _created_at(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return _now()


def _is_external(path: str) -> bool:
    return path.lower().startswith(("http://", "https://", "s3://", "gs://", "az://"))


def _resolve_path(session: Path, value: Any) -> Path | None:
    text = str(value or "")
    if not text or _is_external(text):
        return None
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return session / candidate


def _privacy_level(path: str, asset_type: str = "", default: str = "internal") -> str:
    lowered = f"{path} {asset_type}".lower()
    if lowered.startswith(("http://", "https://")):
        return "external_reference"
    if any(token in lowered for token in ("raw", "video", ".mp4", ".mov", ".avi", ".mkv")):
        return "restricted"
    if any(token in lowered for token in ("transcript", "dialogue", "confirmation", "audit")):
        return "confidential"
    return default


def _audit(event_type: str, *, source_path: str, created_at: str, details: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    return [
        {
            "event_type": event_type,
            "created_at": created_at,
            "source_path": source_path,
            "details": dict(details or {}),
        }
    ]


def _artifact_records(session: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact_type, spec in {**ARTIFACT_SPECS, **OPTIONAL_ARTIFACT_SPECS}.items():
        path = session / spec.relative_path
        source_path = str(path)
        created_at = _created_at(path) if path.exists() else _now()
        rows.append(
            {
                "record_type": "structured_artifact",
                "artifact_type": artifact_type,
                "source_path": source_path,
                "sha256": _sha256(path),
                "created_at": created_at,
                "schema_version": SCHEMA_VERSION,
                "privacy_level": _privacy_level(source_path),
                "exists": path.exists(),
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "audit_trail": _audit(
                    "governance_artifact_scanned",
                    source_path=source_path,
                    created_at=created_at,
                    details={"relative_path": spec.relative_path, "file_format": spec.file_format},
                ),
            }
        )
    return rows


def _asset_records(session: Path) -> list[dict[str, Any]]:
    catalog = session / "metadata" / "material_asset_catalog.jsonl"
    if not catalog.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        assets = read_jsonl(catalog)
    except Exception:
        return []
    for index, asset in enumerate(assets, start=1):
        source_path = str(asset.get("source_path") or asset.get("path") or "")
        resolved = _resolve_path(session, source_path)
        exists = bool(resolved and resolved.exists())
        created_at = str(asset.get("created_at") or (_created_at(resolved) if resolved and exists else _now()))
        rows.append(
            {
                "record_type": "material_asset",
                "asset_id": asset.get("asset_id"),
                "asset_type": asset.get("asset_type"),
                "source_path": source_path,
                "resolved_path": str(resolved) if resolved else None,
                "sha256": asset.get("sha256") or (_sha256(resolved) if resolved else None),
                "created_at": created_at,
                "schema_version": asset.get("schema_version") or "key_action_asset.v1",
                "privacy_level": asset.get("privacy_level") or _privacy_level(source_path, str(asset.get("asset_type") or "")),
                "exists": exists,
                "size_bytes": int(asset.get("size_bytes") or (resolved.stat().st_size if resolved and exists and resolved.is_file() else 0)),
                "source_type": asset.get("source_type"),
                "source_id": asset.get("source_id"),
                "audit_trail": asset.get("audit_trail")
                or _audit(
                    "governance_asset_scanned",
                    source_path=source_path,
                    created_at=created_at,
                    details={"catalog_row": index, "source_type": asset.get("source_type")},
                ),
            }
        )
    return rows


def build_data_governance_report(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    default_privacy_level: str = "internal",
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    target = Path(output_path) if output_path is not None else metadata / "data_governance_report.json"

    records = _artifact_records(session) + _asset_records(session)
    for record in records:
        record.setdefault("privacy_level", default_privacy_level)
        record.setdefault("audit_trail", [])

    missing_hash_count = sum(1 for item in records if item.get("exists") and not item.get("sha256"))
    missing_governance_fields = []
    for item in records:
        missing = [
            field
            for field in ("source_path", "created_at", "schema_version", "privacy_level", "audit_trail")
            if item.get(field) in (None, "", [])
        ]
        if missing:
            missing_governance_fields.append(
                {
                    "record_type": item.get("record_type"),
                    "asset_id": item.get("asset_id"),
                    "artifact_type": item.get("artifact_type"),
                    "missing": missing,
                }
            )
    privacy_counts: dict[str, int] = {}
    for item in records:
        level = str(item.get("privacy_level") or "unknown")
        privacy_counts[level] = privacy_counts.get(level, 0) + 1

    report = {
        "schema_version": GOVERNANCE_SCHEMA_VERSION,
        "created_at": _now(),
        "session_dir": str(session),
        "record_count": len(records),
        "privacy_level_counts": dict(sorted(privacy_counts.items())),
        "missing_hash_count": missing_hash_count,
        "missing_governance_field_count": len(missing_governance_fields),
        "missing_governance_fields": missing_governance_fields,
        "policy": {
            "default_privacy_level": default_privacy_level,
            "path_hashing": "sha256 for local files that exist",
            "audit_trail": "append-friendly list of governance scan/export events",
        },
        "records": records,
    }
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


__all__ = ["build_data_governance_report"]
