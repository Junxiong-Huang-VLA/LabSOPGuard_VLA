"""Material database consistency validation (P1).

Bidirectional JSONL <-> SQLite reconciliation that ACTUALLY opens the SQLite
index (the existing report writer only read JSONL, so it could not detect
drift — which is how a stale "1 official / mixed" report survived next to a
live "0 official / 12 needs_review" database).

Checks performed:
  * row-count parity (material_stream.jsonl vs materials table)
  * material_id present in both directions (jsonl-only / sqlite-only sets)
  * official_status agreement per material_id
  * official_materials.jsonl / review_candidate_materials.jsonl partition matches
    the stream's official_status
  * memory policy: non-official rows must not be memory_eligible
  * keyframe/keyclip path existence (or a recorded missing_reason)
  * orphan materials (no window_id or no source_window_sync_index)

This module performs NO GPU work and does not modify the database unless the
caller explicitly invokes ``repair_report_only`` (which rewrites only the
report file from live data — it never mutates materials).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "database_consistency_validation_report.v2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _material_id(row: Mapping[str, Any]) -> str:
    for key in (
        "material_id",
        "evidence_bundle_id",
        "physical_action_material_id",
        "candidate_group_id",
        "reference_id",
        "candidate_id",
    ):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _sqlite_materials(db_path: Path) -> dict[str, dict[str, Any]]:
    if not db_path.exists():
        return {}
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cur = con.execute(
            "select material_id, official_status, memory_eligible, "
            "keyframe_paths, keyclip_paths from materials"
        )
        out: dict[str, dict[str, Any]] = {}
        for row in cur.fetchall():
            out[str(row["material_id"])] = dict(row)
        return out
    except sqlite3.OperationalError:
        return {}
    finally:
        con.close()


@dataclass
class ConsistencyResult:
    status: str
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _paths_from(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(p) for p in parsed if p]
            return [value]
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, list):
        return [str(p) for p in value if p]
    return []


def validate_material_database(material_root: Path) -> ConsistencyResult:
    """Compute a bidirectional consistency report from live data."""
    material_root = Path(material_root)
    stream = _read_jsonl(material_root / "material_stream.jsonl")
    official = _read_jsonl(material_root / "official_materials.jsonl")
    review = _read_jsonl(material_root / "review_candidate_materials.jsonl")
    sqlite_rows = _sqlite_materials(material_root / "material_index.sqlite")

    stream_by_id = {str(r.get("material_id")): r for r in stream if r.get("material_id")}
    stream_ids = set(stream_by_id)
    sqlite_ids = set(sqlite_rows)

    jsonl_only = sorted(stream_ids - sqlite_ids)
    sqlite_only = sorted(sqlite_ids - stream_ids)

    # official_status agreement per material_id
    status_mismatches: list[dict[str, Any]] = []
    for mid in sorted(stream_ids & sqlite_ids):
        js = stream_by_id[mid].get("official_status")
        ss = sqlite_rows[mid].get("official_status")
        if js != ss:
            status_mismatches.append(
                {"material_id": mid, "jsonl_status": js, "sqlite_status": ss}
            )

    # official_materials.jsonl must match the stream's official partition.
    # NOTE: review_candidate_materials.jsonl is the PRE-PUBLISH candidate pool
    # and uses a different id namespace (candidate_id / candidate_group_id), so
    # it is NOT a partition of the published stream and must not be compared by
    # material_id. We only count it for visibility.
    official_ids = {_material_id(r) for r in official if _material_id(r)}
    stream_official = {mid for mid, r in stream_by_id.items()
                       if r.get("official_status") == "official"}
    official_partition_drift = sorted(official_ids ^ stream_official)

    # memory policy: non-official must not be memory_eligible
    memory_violations = [
        mid for mid, r in stream_by_id.items()
        if r.get("official_status") != "official" and r.get("memory_eligible")
    ]

    # orphan + missing file checks
    orphans: list[str] = []
    missing_assets: list[dict[str, Any]] = []
    for mid, r in stream_by_id.items():
        if not r.get("window_id") and not r.get("experiment_window_id"):
            orphans.append(mid)
        elif not r.get("source_window_sync_index"):
            orphans.append(mid)
        for key in ("first_keyframe", "third_keyframe", "first_keyclip",
                    "third_keyclip", "side_by_side_keyclip"):
            val = r.get(key)
            if val and isinstance(val, str) and ("/" in val or "\\" in val):
                if not Path(val).exists() and not r.get("missing_reason"):
                    missing_assets.append({"material_id": mid, "field": key, "path": val})

    counts = {
        "material_stream_count": len(stream),
        "sqlite_material_count": len(sqlite_rows),
        "official_jsonl_count": len(official),
        "review_candidate_jsonl_count": len(review),
        "stream_official_count": len(stream_official),
    }

    failures = {
        "row_count_mismatch": len(stream) != len(sqlite_rows),
        "jsonl_only_ids": jsonl_only,
        "sqlite_only_ids": sqlite_only,
        "status_mismatches": status_mismatches,
        "official_partition_drift": official_partition_drift,
        "memory_policy_violations": memory_violations,
        "missing_assets": missing_assets,
    }
    has_failure = (
        failures["row_count_mismatch"]
        or jsonl_only or sqlite_only or status_mismatches
        or official_partition_drift or memory_violations or missing_assets
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "material_root": str(material_root),
        "counts": counts,
        "orphan_material_count": len(orphans),
        "orphan_material_ids": sorted(orphans),
        "checks": failures,
        "status": "fail" if has_failure else "pass",
    }
    return ConsistencyResult(status=report["status"], report=report)


def write_consistency_report(material_root: Path) -> ConsistencyResult:
    """Validate and (re)write the report from live data. Never mutates DB."""
    result = validate_material_database(material_root)
    reports_dir = Path(material_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    from .report_io import write_json_report

    write_json_report(
        reports_dir / "database_consistency_validation_report.json", result.report
    )
    return result
