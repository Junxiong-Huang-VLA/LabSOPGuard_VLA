from __future__ import annotations

import json
import sqlite3
import base64
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from labsopguard.retrieval import MaterialRetrievalIndex


DEFAULT_SCORING_PROFILE: Dict[str, Any] = {
    "schema_version": "material_scoring_profile.v1",
    "weights": {
        "evidence_grade": {"strong": 0.40, "medium": 0.25, "weak": 0.05, "default": 0.10},
        "review_status": {"auto_confirmed": 0.25, "approved": 0.25, "candidate_review": 0.10, "low_confidence": -0.10, "default": 0.0},
        "official_linked": 0.25,
        "warning_penalty": 0.03,
        "relevance": 0.05,
        "click_count": 0.02,
        "review_count": 0.03,
        "official_usage_count": 0.08,
        "experiment_type_priority": 0.10,
    },
    "caps": {"click_count": 20, "review_count": 10, "official_usage_count": 10, "warning_count": 5},
    "experiment_type_priorities": {},
}


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def experiment_id_from_dir(experiment_dir: str | Path) -> str:
    path = Path(experiment_dir)
    exp = _load_json(path / "experiment.json", {})
    return str(exp.get("experiment_id") or path.name)


def rebuild_experiment_material_index(experiment_dir: str | Path, *, force: bool = True) -> Dict[str, Any]:
    exp_dir = Path(experiment_dir)
    experiment_id = experiment_id_from_dir(exp_dir)
    material_stream_path = exp_dir / "material_stream.json"
    preprocessing_path = exp_dir / "preprocessing.json"
    index_path = exp_dir / "material_index.sqlite"
    if not material_stream_path.exists():
        return {
            "experiment_id": experiment_id,
            "status": "skipped",
            "reason": "material_stream.json missing",
            "index_path": str(index_path),
        }
    if index_path.exists() and not force:
        index = MaterialRetrievalIndex(index_path)
        try:
            health = index.health_check()
        finally:
            index.close()
        return {"experiment_id": experiment_id, "status": "exists", "index_path": str(index_path), "health": health}

    material_stream = _load_json(material_stream_path, [])
    preprocessing = _load_json(preprocessing_path, {})
    index = MaterialRetrievalIndex(index_path)
    try:
        index.reset()
        index.index_payloads(material_stream, preprocessing=preprocessing, experiment_id=experiment_id)
        health = index.health_check()
    finally:
        index.close()
    return {
        "experiment_id": experiment_id,
        "status": "rebuilt",
        "index_path": str(index_path),
        "health": health,
    }


def rebuild_workspace_material_index(
    experiments_root: str | Path,
    workspace_index_path: str | Path,
    *,
    force_experiment_indexes: bool = False,
) -> Dict[str, Any]:
    root = Path(experiments_root)
    index_path = Path(workspace_index_path)
    index = MaterialRetrievalIndex(index_path)
    experiment_results: List[Dict[str, Any]] = []
    try:
        index.reset()
        for exp_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            result = rebuild_experiment_material_index(exp_dir, force=force_experiment_indexes)
            experiment_results.append(result)
            if result["status"] == "skipped":
                continue
            material_stream_path = exp_dir / "material_stream.json"
            preprocessing_path = exp_dir / "preprocessing.json"
            material_stream = _load_json(material_stream_path, [])
            preprocessing = _load_json(preprocessing_path, {})
            index.index_payloads(material_stream, preprocessing=preprocessing, experiment_id=result["experiment_id"])
        health = index.health_check()
    finally:
        index.close()
    return {
        "schema_version": "workspace_material_index.v1",
        "status": "rebuilt",
        "index_path": str(index_path),
        "experiment_count": len(experiment_results),
        "experiments": experiment_results,
        "health": health,
    }


def rebuild_workspace_published_materials_index(
    experiments_root: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    """Aggregate per-experiment published material records into a workspace SQLite/FTS index."""
    root = Path(experiments_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    items_by_key: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    experiments: List[Dict[str, Any]] = []
    for exp_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        experiment_id = experiment_id_from_dir(exp_dir)
        experiment_meta = _experiment_metadata(exp_dir)
        source_mtime = float(exp_dir.stat().st_mtime)
        published_path = exp_dir / "published_materials.json"
        if not published_path.exists():
            experiments.append({"experiment_id": experiment_id, "status": "skipped", "reason": "published_materials.json missing"})
            continue
        payload = _load_json(published_path, {})
        count = 0
        official_counts = _official_usage_counts(exp_dir)
        review_counts = _review_usage_counts(exp_dir)
        usage_metrics = _material_usage_metrics(exp_dir)
        for item in payload.get("items") or []:
            event_id = str(item.get("event_id") or "")
            material_id = str(item.get("material_id") or event_id)
            usage = usage_metrics.get(material_id) or usage_metrics.get(event_id) or {}
            enriched = {
                **item,
                "experiment_id": item.get("experiment_id") or experiment_id,
                "experiment_type": experiment_meta.get("experiment_type"),
                "official_linked": 1 if official_counts.get(event_id, 0) else 0,
                "official_usage_count": int(official_counts.get(event_id, 0)),
                "review_count": int(review_counts.get(event_id, 0)),
                "click_count": int(usage.get("click_count") or usage.get("clicks") or 0),
            }
            effective_experiment_id = str(enriched.get("experiment_id") or experiment_id)
            effective_material_id = str(enriched.get("material_id") or material_id)
            workspace_key = f"{effective_experiment_id}:{effective_material_id}" if effective_experiment_id else effective_material_id
            previous = items_by_key.get(workspace_key)
            if previous is None or source_mtime >= previous[0]:
                items_by_key[workspace_key] = (source_mtime, enriched)
            count += 1
        experiments.append({"experiment_id": experiment_id, "status": "indexed", "published_count": count})
    items = [entry[1] for entry in items_by_key.values()]
    conn = sqlite3.connect(str(output))
    try:
        _init_workspace_published_schema(conn)
        conn.execute("DELETE FROM published_materials")
        conn.execute("DELETE FROM published_materials_fts")
        for item in items:
            _insert_workspace_published(conn, item)
        conn.commit()
    finally:
        conn.close()
    return {
        "schema_version": "workspace_published_materials_index.v1",
        "storage": "sqlite_fts",
        "total": len(items),
        "experiments": experiments,
        "index_path": str(output),
    }


def _init_workspace_published_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS published_materials (
            material_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            event_id TEXT,
            event_type TEXT,
            display_name TEXT,
            stable_name TEXT,
            actor_name TEXT,
            time_start REAL,
            time_end REAL,
            evidence_grade TEXT,
            review_status TEXT,
            published_path TEXT,
            material_publish_path TEXT,
            clip_path TEXT,
            preview_path TEXT,
            payload_json TEXT,
            searchable_text TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS published_materials_fts
        USING fts5(material_id UNINDEXED, searchable_text)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS material_usage_counts (
            material_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            event_id TEXT,
            click_count INTEGER DEFAULT 0,
            updated_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS material_usage_events (
            usage_event_id TEXT PRIMARY KEY,
            material_id TEXT,
            experiment_id TEXT,
            event_id TEXT,
            event_type TEXT,
            operator TEXT,
            created_at TEXT,
            payload_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_type ON published_materials(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_experiment ON published_materials(experiment_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_actor ON published_materials(actor_name)")
    for statement in (
        "ALTER TABLE published_materials ADD COLUMN time_start REAL",
        "ALTER TABLE published_materials ADD COLUMN time_end REAL",
        "ALTER TABLE published_materials ADD COLUMN evidence_grade TEXT",
        "ALTER TABLE published_materials ADD COLUMN review_status TEXT",
        "ALTER TABLE published_materials ADD COLUMN official_linked INTEGER DEFAULT 0",
        "ALTER TABLE published_materials ADD COLUMN warning_count INTEGER DEFAULT 0",
        "ALTER TABLE published_materials ADD COLUMN click_count INTEGER DEFAULT 0",
        "ALTER TABLE published_materials ADD COLUMN review_count INTEGER DEFAULT 0",
        "ALTER TABLE published_materials ADD COLUMN official_usage_count INTEGER DEFAULT 0",
        "ALTER TABLE published_materials ADD COLUMN experiment_type TEXT",
    ):
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_official ON published_materials(official_linked)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_experiment_type ON published_materials(experiment_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_material ON material_usage_events(material_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_counts_experiment ON material_usage_counts(experiment_id)")


def _insert_workspace_published(conn: sqlite3.Connection, item: Dict[str, Any]) -> None:
    paths = item.get("published_paths") or {}
    source = item.get("source_container") or {}
    target = item.get("target_container") or {}
    text = " ".join(
        str(part)
        for part in [
            item.get("display_name"),
            item.get("stable_name"),
            item.get("event_type"),
            item.get("actor_name"),
            source,
            target,
            item.get("evidence_grade"),
            item.get("review_status"),
            " ".join(item.get("warnings") or []),
        ]
        if part
    )
    material_id = str(item.get("material_id") or item.get("event_id"))
    experiment_id = str(item.get("experiment_id") or "")
    workspace_material_id = f"{experiment_id}:{material_id}" if experiment_id else material_id
    existing_usage = conn.execute("SELECT click_count FROM material_usage_counts WHERE material_id = ?", (workspace_material_id,)).fetchone()
    click_count = max(int(item.get("click_count") or 0), int(existing_usage[0] or 0) if existing_usage else 0)
    row = {
        "material_id": workspace_material_id,
        "experiment_id": experiment_id or item.get("experiment_id"),
        "event_id": item.get("event_id"),
        "event_type": item.get("event_type"),
        "display_name": item.get("display_name"),
        "stable_name": item.get("stable_name"),
        "actor_name": item.get("actor_name"),
        "time_start": item.get("time_start"),
        "time_end": item.get("time_end"),
        "evidence_grade": item.get("evidence_grade"),
        "review_status": item.get("review_status"),
        "published_path": str(Path(paths.get("material_publish", "")).parent) if paths.get("material_publish") else None,
        "material_publish_path": paths.get("material_publish"),
        "clip_path": paths.get("clip"),
        "preview_path": paths.get("preview"),
        "payload_json": json.dumps(item, ensure_ascii=False),
        "searchable_text": text,
        "official_linked": int(item.get("official_linked") or 0),
        "warning_count": len(item.get("warnings") or []),
        "click_count": click_count,
        "review_count": int(item.get("review_count") or 0),
        "official_usage_count": int(item.get("official_usage_count") or 0),
        "experiment_type": item.get("experiment_type"),
    }
    conn.execute(
        """
        INSERT OR REPLACE INTO published_materials
        (material_id, experiment_id, event_id, event_type, display_name, stable_name, actor_name,
         time_start, time_end, evidence_grade, review_status,
         published_path, material_publish_path, clip_path, preview_path, payload_json, searchable_text,
         official_linked, warning_count, click_count, review_count, official_usage_count, experiment_type)
        VALUES (:material_id, :experiment_id, :event_id, :event_type, :display_name, :stable_name, :actor_name,
         :time_start, :time_end, :evidence_grade, :review_status,
         :published_path, :material_publish_path, :clip_path, :preview_path, :payload_json, :searchable_text,
         :official_linked, :warning_count, :click_count, :review_count, :official_usage_count, :experiment_type)
        """,
        row,
    )
    conn.execute("INSERT INTO published_materials_fts(material_id, searchable_text) VALUES (?, ?)", (workspace_material_id, text))


SORT_FIELDS = {
    "experiment_id": "COALESCE(p.experiment_id, '')",
    "event_type": "COALESCE(p.event_type, '')",
    "display_name": "COALESCE(p.display_name, '')",
    "actor_name": "COALESCE(p.actor_name, '')",
    "time_start": "COALESCE(p.time_start, 0)",
    "evidence_grade": "COALESCE(p.evidence_grade, '')",
    "review_status": "COALESCE(p.review_status, '')",
    "material_id": "p.material_id",
}
SORT_FIELDS["relevance"] = "relevance_score"
SORT_FIELDS["business_score"] = "business_score"


def _experiment_metadata(exp_dir: Path) -> Dict[str, Any]:
    payload = _load_json(exp_dir / "experiment.json", {})
    experiment_type = payload.get("experiment_type") or payload.get("type") or payload.get("category")
    return {"experiment_type": str(experiment_type) if experiment_type else None}


def _official_usage_counts(exp_dir: Path) -> Dict[str, int]:
    payload = _load_json(exp_dir / "official_steps.json", {})
    counts: Dict[str, int] = {}
    for step in payload.get("official_steps") or []:
        for event_id in step.get("linked_event_ids") or []:
            key = str(event_id)
            counts[key] = counts.get(key, 0) + 1
        bundle = step.get("evidence_bundle") or {}
        for ref in bundle.get("published_material_refs") or []:
            if isinstance(ref, dict) and ref.get("event_id"):
                key = str(ref["event_id"])
                counts[key] = counts.get(key, 0) + 1
    return counts


def _review_usage_counts(exp_dir: Path) -> Dict[str, int]:
    payload = _load_json(exp_dir / "step_review_log.json", {})
    counts: Dict[str, int] = {}
    for collection_name in ("review_decisions", "governance_decisions", "lifecycle_events", "revisions"):
        for item in payload.get(collection_name) or []:
            if not isinstance(item, dict):
                continue
            event_ids = item.get("linked_event_ids") or item.get("matched_event_ids") or []
            if item.get("event_id"):
                event_ids = [item["event_id"], *list(event_ids)]
            for event_id in event_ids:
                key = str(event_id)
                counts[key] = counts.get(key, 0) + 1
    return counts


def _material_usage_metrics(exp_dir: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(exp_dir / "material_usage.json", {})
    if isinstance(payload, dict) and "items" not in payload:
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    metrics: Dict[str, Dict[str, Any]] = {}
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        key = item.get("material_id") or item.get("event_id")
        if key:
            metrics[str(key)] = item
    return metrics


def load_material_scoring_profile(profile_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load a scoring profile from JSON/YAML and merge it onto stable defaults."""
    path_value = profile_path or os.environ.get("REALITYLOOP_MATERIAL_SCORING_PROFILE")
    profile = json.loads(json.dumps(DEFAULT_SCORING_PROFILE))
    if not path_value:
        default_yaml = Path("configs") / "material_scoring.yaml"
        if default_yaml.exists():
            path_value = default_yaml
    if path_value:
        path = Path(path_value)
        if path.exists():
            if path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml

                    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    loaded = {}
            else:
                loaded = _load_json(path, {})
            _deep_update(profile, loaded)
    profile["profile_hash"] = _profile_hash(profile)
    return profile


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _profile_hash(profile: Dict[str, Any]) -> str:
    clone = {key: value for key, value in profile.items() if key != "profile_hash"}
    raw = json.dumps(clone, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _filter_hash(
    *,
    text: Optional[str],
    event_type: Optional[str],
    actor_name: Optional[str],
    operator_role: str,
    allowed_experiment_ids: Optional[List[str]],
    scoring_profile_hash: str,
) -> str:
    payload = {
        "text": text or "",
        "event_type": event_type or "",
        "actor_name": actor_name or "",
        "operator_role": operator_role or "",
        "allowed_experiment_ids": sorted(allowed_experiment_ids or []),
        "scoring_profile_hash": scoring_profile_hash,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _encode_cursor(
    sort_by: str,
    sort_order: str,
    sort_value: Any,
    material_id: str,
    *,
    filters_hash: str,
) -> str:
    payload = json.dumps(
        {
            "v": 2,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "sort_value": sort_value,
            "material_id": material_id,
            "filters_hash": filters_hash,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(cursor: Optional[str]) -> Dict[str, Any]:
    if not cursor:
        return {}
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _cursor_clause(sort_column: str, direction: str, cursor_payload: Dict[str, Any]) -> tuple[str, List[Any]]:
    if not cursor_payload or cursor_payload.get("sort_value") is None or not cursor_payload.get("material_id"):
        return "", []
    op = ">" if direction == "ASC" else "<"
    return f"(({sort_column} {op} ?) OR ({sort_column} = ? AND p.material_id > ?))", [
        cursor_payload.get("sort_value"),
        cursor_payload.get("sort_value"),
        cursor_payload.get("material_id"),
    ]


def _sql_float(value: Any, default: float = 0.0) -> str:
    try:
        return f"{float(value):.10f}"
    except Exception:
        return f"{default:.10f}"


def _case_mapping_expr(column: str, mapping: Dict[str, Any], default: float = 0.0) -> str:
    parts = [f"CASE COALESCE({column}, '')"]
    for key, value in mapping.items():
        if key == "default":
            continue
        escaped = str(key).replace("'", "''")
        parts.append(f"WHEN '{escaped}' THEN {_sql_float(value)}")
    parts.append(f"ELSE {_sql_float(mapping.get('default', default))} END")
    return " ".join(parts)


def _experiment_priority_expr(priorities: Dict[str, Any]) -> str:
    if not priorities:
        return "0"
    parts = ["CASE COALESCE(p.experiment_type, '')"]
    for key, value in priorities.items():
        escaped = str(key).replace("'", "''")
        parts.append(f"WHEN '{escaped}' THEN {_sql_float(value)}")
    parts.append("ELSE 0 END")
    return " ".join(parts)


def _business_score_expression(relevance_expr: Optional[str], scoring_profile: Dict[str, Any]) -> str:
    weights = scoring_profile.get("weights") or {}
    caps = scoring_profile.get("caps") or {}
    evidence_expr = _case_mapping_expr("p.evidence_grade", weights.get("evidence_grade") or {})
    review_status_expr = _case_mapping_expr("p.review_status", weights.get("review_status") or {})
    warning_cap = int(caps.get("warning_count") or 5)
    click_cap = int(caps.get("click_count") or 20)
    review_cap = int(caps.get("review_count") or 10)
    official_usage_cap = int(caps.get("official_usage_count") or 10)
    relevance_boost = f" + ((-1 * {relevance_expr}) * {_sql_float(weights.get('relevance'), 0.05)})" if relevance_expr else ""
    return (
        "("
        f"{evidence_expr} "
        f"+ {review_status_expr} "
        f"+ CASE COALESCE(p.official_linked, 0) WHEN 1 THEN {_sql_float(weights.get('official_linked'), 0.25)} ELSE 0 END "
        f"- (MIN(COALESCE(p.warning_count, 0), {warning_cap}) * {_sql_float(weights.get('warning_penalty'), 0.03)}) "
        f"+ (MIN(COALESCE(p.click_count, 0), {click_cap}) * {_sql_float(weights.get('click_count'), 0.02)}) "
        f"+ (MIN(COALESCE(p.review_count, 0), {review_cap}) * {_sql_float(weights.get('review_count'), 0.03)}) "
        f"+ (MIN(COALESCE(p.official_usage_count, 0), {official_usage_cap}) * {_sql_float(weights.get('official_usage_count'), 0.08)}) "
        f"+ ({_experiment_priority_expr(scoring_profile.get('experiment_type_priorities') or {})} * {_sql_float(weights.get('experiment_type_priority'), 0.10)})"
        f"{relevance_boost}"
        ")"
    )


def query_workspace_published_materials(
    index_path: str | Path,
    *,
    text: Optional[str] = None,
    event_type: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None,
    sort_by: str = "time_start",
    sort_order: str = "asc",
    operator_role: str = "admin",
    allowed_experiment_ids: Optional[List[str]] = None,
    actor_name: Optional[str] = None,
    scoring_profile_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    path = Path(index_path)
    if not path.exists():
        return {"schema_version": "workspace_published_materials_query.v1", "storage": "sqlite_fts", "total": 0, "items": [], "index_path": str(path), "next_cursor": None}
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    results = []
    try:
        params: List[Any] = []
        clauses: List[str] = []
        permission_applied = operator_role != "admin"
        scoring_profile = load_material_scoring_profile(scoring_profile_path)
        filters_hash = _filter_hash(
            text=text,
            event_type=event_type,
            actor_name=actor_name,
            operator_role=operator_role,
            allowed_experiment_ids=allowed_experiment_ids,
            scoring_profile_hash=str(scoring_profile.get("profile_hash") or ""),
        )
        relevance_expr = "bm25(published_materials_fts)" if text else None
        business_expr = _business_score_expression(relevance_expr, scoring_profile)
        score_expr = f"{relevance_expr if relevance_expr else 'NULL'} AS relevance_score, {business_expr} AS business_score"
        if text:
            sql = (
                f"SELECT p.*, {score_expr} FROM published_materials p "
                "JOIN published_materials_fts f ON p.material_id = f.material_id "
            )
            clauses.append("published_materials_fts MATCH ?")
            params.append(text)
        else:
            sql = f"SELECT p.*, {score_expr} FROM published_materials p"
        if event_type:
            clauses.append("p.event_type = ?")
            params.append(event_type)
        if actor_name:
            clauses.append("p.actor_name = ?")
            params.append(actor_name)
        if operator_role != "admin":
            allowed = [item for item in (allowed_experiment_ids or []) if item]
            if allowed:
                placeholders = ",".join("?" for _ in allowed)
                clauses.append(f"p.experiment_id IN ({placeholders})")
                params.extend(allowed)
            elif not actor_name:
                clauses.append("1 = 0")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        normalized_sort_by = sort_by if sort_by in SORT_FIELDS else "time_start"
        if normalized_sort_by == "relevance" and relevance_expr:
            sort_column = relevance_expr
        elif normalized_sort_by == "business_score":
            sort_column = business_expr
        else:
            sort_column = SORT_FIELDS.get(normalized_sort_by, SORT_FIELDS["time_start"])
        direction = "DESC" if str(sort_order).lower() == "desc" else "ASC"
        page_size = min(max(1, int(limit)), 500)
        cursor_payload = _decode_cursor(cursor)
        cursor_matches = (
            cursor_payload
            and cursor_payload.get("sort_by") == normalized_sort_by
            and cursor_payload.get("sort_order") == direction.lower()
            and cursor_payload.get("filters_hash") == filters_hash
        )
        if cursor_matches:
            clause, values = _cursor_clause(sort_column, direction, cursor_payload)
            if clause:
                sql += (" WHERE " if " WHERE " not in sql else " AND ") + clause
                params.extend(values)
        if normalized_sort_by == "relevance" and relevance_expr:
            sql += f" ORDER BY {relevance_expr} ASC, p.material_id ASC LIMIT ?"
        else:
            sql += f" ORDER BY {sort_column} {direction}, p.material_id ASC LIMIT ?"
        params.append(page_size + 1)
        for row in conn.execute(sql, params).fetchall():
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload_json") or "{}")
            except Exception:
                item["payload"] = {}
            item.pop("payload_json", None)
            results.append(item)
        has_more = len(results) > page_size
        if has_more:
            results = results[:page_size]
        if has_more and results:
            last = results[-1]
            cursor_sort_by = "relevance" if normalized_sort_by == "relevance" and relevance_expr else normalized_sort_by
            cursor_sort_order = "asc" if cursor_sort_by == "relevance" else direction.lower()
            cursor_value = last.get("relevance_score") if cursor_sort_by == "relevance" else last.get(cursor_sort_by)
            next_cursor = _encode_cursor(
                cursor_sort_by,
                cursor_sort_order,
                cursor_value,
                last.get("material_id"),
                filters_hash=filters_hash,
            )
        else:
            next_cursor = None
    finally:
        conn.close()
    return {
        "schema_version": "workspace_published_materials_query.v1",
        "storage": "sqlite_fts",
        "total": len(results),
        "items": results,
        "index_path": str(path),
        "next_cursor": next_cursor,
        "sort": {
            "sort_by": normalized_sort_by,
            "sort_order": ("asc" if normalized_sort_by == "relevance" and text else direction.lower()),
        },
        "cursor": {"version": 2, "filters_hash": filters_hash, "scoring_profile_hash": scoring_profile.get("profile_hash")},
        "scoring_profile": {
            "schema_version": scoring_profile.get("schema_version"),
            "profile_hash": scoring_profile.get("profile_hash"),
        },
        "permission_filter": {
            "operator_role": operator_role,
            "allowed_experiment_ids": allowed_experiment_ids or [],
            "actor_name": actor_name,
            "applied": permission_applied,
        },
    }


def _write_experiment_material_usage_click(
    experiments_root: str | Path,
    *,
    experiment_id: str,
    event_id: Optional[str],
    material_id: str,
    click_count: int,
) -> Optional[str]:
    exp_dir = Path(experiments_root) / experiment_id
    if not exp_dir.exists():
        return None
    usage_path = exp_dir / "material_usage.json"
    payload = _load_json(usage_path, {})
    if isinstance(payload, dict) and "items" not in payload:
        items = []
        for key, value in payload.items():
            if isinstance(value, dict):
                items.append({"material_id": key, **value})
        payload = {"schema_version": "material_usage.v1", "experiment_id": experiment_id, "items": items}
    if not isinstance(payload, dict):
        payload = {"schema_version": "material_usage.v1", "experiment_id": experiment_id, "items": []}
    payload.setdefault("schema_version", "material_usage.v1")
    payload["experiment_id"] = experiment_id
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    items = payload.setdefault("items", [])
    target = None
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("material_id") == material_id or (event_id and item.get("event_id") == event_id):
            target = item
            break
    if target is None:
        target = {"material_id": material_id, "event_id": event_id}
        items.append(target)
    target["material_id"] = material_id
    if event_id:
        target["event_id"] = event_id
    target["click_count"] = click_count
    target["last_clicked_at"] = payload["updated_at"]
    usage_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(usage_path)


def record_workspace_published_material_click(
    index_path: str | Path,
    material_id: str,
    *,
    experiments_root: Optional[str | Path] = None,
    operator: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Increment local click count and optionally persist it back to experiment material_usage.json."""
    path = Path(index_path)
    if not path.exists():
        return {"status": "missing_index", "material_id": material_id, "updated": False}
    conn = sqlite3.connect(str(path))
    try:
        _init_workspace_published_schema(conn)
        row = conn.execute(
            "SELECT material_id, click_count, experiment_id, event_id, payload_json FROM published_materials WHERE material_id = ?",
            (material_id,),
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT material_id, click_count, experiment_id, event_id, payload_json FROM published_materials "
                "WHERE event_id = ? OR material_id LIKE ? ORDER BY material_id LIMIT 1",
                (material_id, f"%:{material_id}"),
            ).fetchone()
        if not row:
            return {"status": "not_found", "material_id": material_id, "updated": False}
        workspace_id = row[0]
        new_count = int(row[1] or 0) + 1
        now = datetime.now(timezone.utc).isoformat()
        experiment_id = str(row[2])
        event_id = str(row[3]) if row[3] else None
        conn.execute("UPDATE published_materials SET click_count = ? WHERE material_id = ?", (new_count, workspace_id))
        conn.execute(
            """
            INSERT INTO material_usage_counts(material_id, experiment_id, event_id, click_count, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(material_id) DO UPDATE SET
                click_count=excluded.click_count,
                updated_at=excluded.updated_at,
                experiment_id=excluded.experiment_id,
                event_id=excluded.event_id
            """,
            (workspace_id, experiment_id, event_id, new_count, now),
        )
        usage_event_id = "usage_" + hashlib.sha1(f"{workspace_id}:{now}:{new_count}".encode("utf-8")).hexdigest()[:16]
        conn.execute(
            """
            INSERT INTO material_usage_events
            (usage_event_id, material_id, experiment_id, event_id, event_type, operator, created_at, payload_json)
            SELECT ?, material_id, experiment_id, event_id, event_type, ?, ?, ?
            FROM published_materials WHERE material_id = ?
            """,
            (
                usage_event_id,
                operator or "anonymous",
                now,
                json.dumps(payload or {"usage_type": "click"}, ensure_ascii=False),
                workspace_id,
            ),
        )
        conn.commit()
        usage_path = None
        if experiments_root:
            original_material_id = workspace_id.split(":", 1)[-1]
            try:
                payload = json.loads(row[4] or "{}")
                original_material_id = str(payload.get("material_id") or original_material_id)
            except Exception:
                pass
            usage_path = _write_experiment_material_usage_click(
                experiments_root,
                experiment_id=experiment_id,
                event_id=event_id,
                material_id=original_material_id,
                click_count=new_count,
            )
        return {
            "status": "updated",
            "material_id": workspace_id,
            "click_count": new_count,
            "usage_event_id": usage_event_id,
            "material_usage_path": usage_path,
            "updated": True,
        }
    finally:
        conn.close()


def scan_experiment_material_health(experiments_root: str | Path) -> Dict[str, Any]:
    root = Path(experiments_root)
    results: List[Dict[str, Any]] = []
    total_broken = 0
    total_items = 0
    for exp_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        index_path = exp_dir / "material_index.sqlite"
        if not index_path.exists():
            result = rebuild_experiment_material_index(exp_dir, force=False)
            if result.get("status") == "skipped":
                results.append(result)
                continue
        experiment_id = experiment_id_from_dir(exp_dir)
        index = MaterialRetrievalIndex(index_path)
        try:
            health = index.health_check()
        finally:
            index.close()
        total_items += int(health.get("total_items") or 0)
        total_broken += int(health.get("broken_clip_reference_count") or 0)
        results.append({"experiment_id": experiment_id, "status": "checked", "index_path": str(index_path), "health": health})
    return {
        "schema_version": "material_health_scan.v1",
        "experiment_count": len(results),
        "total_items": total_items,
        "total_broken_clip_references": total_broken,
        "experiments": results,
    }


def write_health_scan_report(report: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
