from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import SessionManifest, write_jsonl


TEXT_KEYS = ("text", "content", "message", "summary", "description", "name", "title")
ACTION_KEYS = ("action_type", "expected_action", "event_type", "operation", "action")
MATERIAL_KEYS = ("materials", "objects", "primary_object", "required_material", "required_materials")
PARAMETER_KEYS = ("parameters", "params", "measurement", "measurements")
SOP_EXTENSIONS = {".json", ".jsonl", ".md", ".txt"}
DATABASE_EXTENSIONS = {".json", ".jsonl"}


def ingest_sop_and_database_records(
    manifest: SessionManifest,
    output_dir: str | Path,
    *,
    sop_sources: Iterable[str | Path] | None = None,
    database_sources: Iterable[str | Path] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    sop_paths = list(sop_sources or _manifest_sources(manifest, "sop"))
    database_paths = list(database_sources or _manifest_sources(manifest, "database"))

    sop_rows = normalize_sop_records(sop_paths, session_id=manifest.session_id)
    database_rows = normalize_database_records(database_paths, session_id=manifest.session_id)
    if dry_run and not sop_rows:
        sop_rows = _dry_run_sop_records(manifest)
    if dry_run and not database_rows:
        database_rows = _dry_run_database_records(manifest)

    sop_path = target / "sop_records.jsonl"
    database_path = target / "database_records.jsonl"
    write_jsonl(sop_path, sop_rows)
    write_jsonl(database_path, database_rows)
    summary = {
        "session_id": manifest.session_id,
        "sop_records": str(sop_path),
        "database_records": str(database_path),
        "sop_source_count": len(sop_paths),
        "database_source_count": len(database_paths),
        "sop_record_count": len(sop_rows),
        "database_record_count": len(database_rows),
        "dry_run": dry_run,
        "search_fields": ["experiment_type", "materials", "action_type", "parameters", "text"],
    }
    (target / "record_ingestion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def normalize_sop_records(sources: Iterable[str | Path], *, session_id: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _expand_sources(sources, SOP_EXTENSIONS):
        rows.extend(_records_from_source(source, source_type="sop", session_id=session_id))
    for index, row in enumerate(rows, start=1):
        row.setdefault("record_id", f"sop_{index:06d}")
        row.setdefault("step_id", row["record_id"])
        row.setdefault("step_order", index)
        row.setdefault("event_type", "sop_record")
        row.setdefault("confidence", 1.0)
        row.setdefault("evidence_source", "sop")
        row["search_text"] = _search_text(row)
    return rows


def normalize_database_records(sources: Iterable[str | Path], *, session_id: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in _expand_sources(sources, DATABASE_EXTENSIONS):
        rows.extend(_records_from_source(source, source_type="database", session_id=session_id))
    for index, row in enumerate(rows, start=1):
        row.setdefault("record_id", f"db_{index:06d}")
        row.setdefault("event_type", "database_record")
        row.setdefault("confidence", _bounded_float(row.get("confidence"), default=0.75))
        row.setdefault("evidence_source", "historical_database")
        row["search_text"] = _search_text(row)
    return rows


def search_records(
    records: Iterable[Mapping[str, Any]],
    *,
    query: str = "",
    experiment_type: str | None = None,
    material: str | None = None,
    action: str | None = None,
    parameter: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    terms = [_norm(item) for item in (query, experiment_type, material, action, parameter) if item]
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in records:
        item = dict(row)
        haystack = _norm(_search_text(item))
        score = sum(1 for term in terms if term and term in haystack)
        if terms and score <= 0:
            continue
        scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("record_id") or "")))
    return [item for _score, item in scored[: max(1, int(limit))]]


def _manifest_sources(manifest: SessionManifest, source_type: str) -> list[str]:
    paths = []
    for source in manifest.input_sources.values():
        if str(source.source_type).lower() == source_type:
            paths.append(source.path)
    return paths


def _expand_sources(sources: Iterable[str | Path], extensions: set[str]) -> list[Path]:
    expanded: list[Path] = []
    for value in sources:
        path = Path(value)
        if path.is_dir():
            expanded.extend(
                sorted(
                    item
                    for item in path.iterdir()
                    if item.is_file() and item.suffix.lower() in extensions
                )
            )
        elif path.exists() and path.suffix.lower() in extensions:
            expanded.append(path)
    return expanded


def _records_from_source(path: Path, *, source_type: str, session_id: str) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        raw_rows = _read_jsonl(path)
    elif suffix == ".json":
        raw_rows = _json_rows(json.loads(path.read_text(encoding="utf-8-sig")))
    else:
        raw_rows = _text_rows(path)

    normalized = []
    for index, row in enumerate(raw_rows, start=1):
        normalized.append(_normalize_record(row, path=path, index=index, source_type=source_type, session_id=session_id))
    return normalized


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _json_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        for key in ("steps", "records", "events", "experiments", "items"):
            child = value.get(key)
            if isinstance(child, list):
                return [dict(item) for item in child if isinstance(item, Mapping)]
        return [dict(value)]
    return []


def _text_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        text = re.sub(r"^\s*[-*#\d.)]+\s*", "", line).strip()
        if text:
            rows.append({"step_order": index, "text": text, "name": text[:80]})
    return rows


def _normalize_record(row: Mapping[str, Any], *, path: Path, index: int, source_type: str, session_id: str) -> dict[str, Any]:
    text = _row_text(row)
    action = _first_text(row, ACTION_KEYS)
    materials = _list_values(row, MATERIAL_KEYS)
    parameters = _parameters(row)
    record = {
        "record_id": str(row.get("record_id") or row.get("id") or row.get("step_id") or f"{source_type}_{path.stem}_{index:04d}"),
        "session_id": str(row.get("session_id") or row.get("session") or session_id),
        "source_type": source_type,
        "source_path": str(path),
        "experiment_type": str(row.get("experiment_type") or row.get("assay_type") or row.get("project") or ""),
        "step_id": str(row.get("step_id") or row.get("id") or f"{source_type}_{index:03d}"),
        "step_order": int(float(row.get("step_order") or row.get("order") or index)),
        "name": str(row.get("name") or row.get("title") or text[:80] or action or ""),
        "action_type": str(action or ""),
        "expected_action": str(row.get("expected_action") or action or ""),
        "materials": materials,
        "parameters": parameters,
        "global_start_time": row.get("global_start_time") or row.get("global_time"),
        "global_end_time": row.get("global_end_time"),
        "duration_sec": _optional_float(row.get("duration_sec")),
        "text": text,
        "payload": dict(row),
    }
    if source_type == "sop":
        record["event_type"] = "sop_record"
        record["evidence_source"] = "sop_document"
    else:
        record["event_type"] = "database_record"
        record["evidence_source"] = "historical_database"
    return record


def _row_text(row: Mapping[str, Any]) -> str:
    for key in TEXT_KEYS:
        if row.get(key):
            value = row[key]
            if isinstance(value, list):
                return " ".join(str(item) for item in value if item)
            return str(value)
    return ""


def _first_text(row: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if row.get(key):
            return str(row[key])
    return ""


def _list_values(row: Mapping[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif isinstance(value, Mapping):
            values.extend(str(item) for item in value.values() if item)
        elif value:
            values.append(str(value))
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        token = value.strip()
        if token and token not in seen:
            seen.add(token)
            output.append(token)
    return output


def _parameters(row: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for key in PARAMETER_KEYS:
        value = row.get(key)
        if isinstance(value, Mapping):
            params.update({str(k): v for k, v in value.items()})
        elif isinstance(value, list):
            params[key] = value
        elif value:
            params[key] = value
    for key in ("volume", "mass", "temperature", "duration", "concentration"):
        if row.get(key) is not None:
            params[key] = row[key]
    return params


def _search_text(row: Mapping[str, Any]) -> str:
    parts = [
        row.get("experiment_type"),
        row.get("action_type"),
        row.get("expected_action"),
        row.get("name"),
        row.get("text"),
        " ".join(str(item) for item in row.get("materials", []) if item),
        json.dumps(row.get("parameters", {}), ensure_ascii=False, sort_keys=True),
    ]
    return " ".join(str(item) for item in parts if item)


def _norm(value: Any) -> str:
    return str(value or "").casefold().replace("-", "_").replace(" ", "_")


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_float(value: Any, *, default: float) -> float:
    parsed = _optional_float(value)
    if parsed is None:
        parsed = default
    return max(0.0, min(1.0, float(parsed)))


def _dry_run_sop_records(manifest: SessionManifest) -> list[dict[str, Any]]:
    actions = ["weighing", "pipetting", "recording"]
    return [
        {
            "record_id": f"sop_dry_{index:03d}",
            "session_id": manifest.session_id,
            "source_type": "sop",
            "source_path": "dry_run",
            "event_type": "sop_record",
            "step_id": f"dry_step_{index:03d}",
            "step_order": index,
            "name": action.replace("_", " ").title(),
            "action_type": action,
            "expected_action": action,
            "materials": ["sample"] if action != "recording" else [],
            "parameters": {},
            "text": f"Dry-run SOP placeholder for {action}.",
            "confidence": 0.5,
            "evidence_source": "dry_run_placeholder",
            "search_text": f"{action} sample dry-run SOP",
        }
        for index, action in enumerate(actions, start=1)
    ]


def _dry_run_database_records(manifest: SessionManifest) -> list[dict[str, Any]]:
    actions = ["weighing", "pipetting", "recording"]
    return [
        {
            "record_id": f"db_dry_{index:03d}",
            "session_id": manifest.session_id,
            "source_type": "database",
            "source_path": "dry_run",
            "event_type": "database_record",
            "experiment_type": "dry_run_key_action_demo",
            "action_type": action,
            "expected_action": action,
            "materials": ["balance"] if action == "weighing" else ["pipette"] if action == "pipetting" else [],
            "parameters": {},
            "duration_sec": 30.0,
            "text": f"Dry-run historical record for {action}.",
            "confidence": 0.5,
            "evidence_source": "dry_run_placeholder",
            "search_text": f"{action} historical database dry-run",
        }
        for index, action in enumerate(actions, start=1)
    ]


__all__ = [
    "ingest_sop_and_database_records",
    "normalize_database_records",
    "normalize_sop_records",
    "search_records",
]
