from __future__ import annotations

import json
import sqlite3
import base64
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from labsopguard.material_best_score import enrich_material_best_score, enrich_material_best_scores
from labsopguard.material_taxonomy import enrich_material_taxonomy
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

KEYFRAME_KIND = "\u5173\u952e\u5e27"
KEY_CLIP_KIND = "\u5173\u952e\u7247\u6bb5"
REPORT_KIND = "\u4e13\u4e1a\u62a5\u544a"
MATERIAL_INDEX_FILENAME = "\u7d20\u6750\u7d22\u5f15.jsonl"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def experiment_id_from_dir(experiment_dir: str | Path) -> str:
    path = Path(experiment_dir)
    exp = _load_json(path / "experiment.json", {})
    return str(exp.get("experiment_id") or path.name)


def _material_delivery_safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value).strip("._") or "material"


def _material_delivery_date_label(value: str) -> str:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%d")
        except ValueError:
            pass
        match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", value)
        if match:
            return "".join(match.groups())
    return datetime.now().strftime("%Y%m%d")


def _formal_material_reference_root_for_exp(exp_dir: Path) -> Path:
    local_root = exp_dir / "material_references"
    for meta_path in (local_root / "manifest.json", local_root / "\u7d20\u6750\u7d22\u5f15.json"):
        payload = _load_json(meta_path, {})
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("formal_material_references") or payload.get("simplified_material_references")
        if candidate:
            return Path(str(candidate))

    exp = _load_json(exp_dir / "experiment.json", {})
    title = str(
        exp.get("title")
        or exp.get("experiment_title")
        or exp.get("experiment_name")
        or exp.get("name")
        or exp_dir.name
    )
    date = _material_delivery_date_label(str(exp.get("created_at") or exp.get("experiment_date") or exp.get("date") or exp_dir.name))
    outputs_dir = exp_dir.parent.parent if exp_dir.parent.name == "experiments" else exp_dir.parent
    return outputs_dir / "material_references" / _material_delivery_safe_name(f"{title}_{date}")


def _material_reference_root_candidates(exp_dir: Path) -> List[Path]:
    formal_root = _formal_material_reference_root_for_exp(exp_dir)
    local_root = exp_dir / "material_references"
    roots: List[Path] = []
    for root in (formal_root, local_root):
        if root not in roots:
            roots.append(root)
    return roots


def _material_reference_rows_from_root(ref_root: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(ref_root / MATERIAL_INDEX_FILENAME)
    if rows:
        return rows
    payload = _load_json(ref_root / "\u7d20\u6750\u7d22\u5f15.json", {})
    records = payload.get("records") if isinstance(payload, dict) else None
    return [row for row in (records or []) if isinstance(row, dict)]


def _material_reference_root_and_rows(exp_dir: Path) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    for ref_root in _material_reference_root_candidates(exp_dir):
        rows = _material_reference_rows_from_root(ref_root)
        if rows:
            return ref_root, rows
    return None, []


def _material_reference_row_path(row: Dict[str, Any], ref_root: Path) -> Optional[Path]:
    raw_path = row.get("stored_file") or row.get("stored_path") or row.get("file_path")
    if raw_path:
        path = Path(str(raw_path))
        return path if path.is_absolute() else ref_root / path
    filename = row.get("stored_filename") or row.get("file_name")
    asset_kind = row.get("asset_kind") or row.get("material_type")
    if filename and asset_kind:
        return ref_root / str(asset_kind) / str(filename)
    return None


def _flatten_search_terms(value: Any) -> List[str]:
    terms: List[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            terms.extend(_flatten_search_terms(nested))
    elif isinstance(value, list):
        for nested in value:
            terms.extend(_flatten_search_terms(nested))
    elif value is not None:
        text = str(value).strip()
        if text:
            terms.append(text)
    return terms


def _action_object_terms(action_name: str) -> List[str]:
    terms: List[str] = []
    match = re.search(r"手与(.+?)操作", action_name)
    if match:
        obj = match.group(1).strip()
        if obj:
            terms.extend([obj, f"{obj} 操作", f"{obj}操作"])
    return terms


def _semantic_material_terms(row: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    action_name = str(row.get("action_name") or row.get("event_type") or "")
    if action_name:
        terms.extend([action_name, *_action_object_terms(action_name)])
    primary_object = str(row.get("primary_object") or row.get("object_label") or "")
    if primary_object:
        terms.append(primary_object)
    terms.extend(_flatten_search_terms(row.get("object_labels")))
    terms.extend(_flatten_search_terms(row.get("actions")))

    vlm = row.get("vlm_semantics") if isinstance(row.get("vlm_semantics"), dict) else {}
    description = str(vlm.get("description") or "")
    physical_action = str(vlm.get("physical_action") or "")
    terms.extend([description, physical_action, physical_action.replace("_", " ")])
    terms.extend(_flatten_search_terms(vlm.get("confirmed_objects")))
    terms.extend(_flatten_search_terms(vlm.get("uncertain_objects")))
    if "手套" in description or "gloved_hand" in terms:
        terms.extend(["戴手套", "戴手套 操作", "戴手套操作", "gloved hand operation"])

    yolo = row.get("yolo_recheck") if isinstance(row.get("yolo_recheck"), dict) else {}
    terms.extend(
        [
            str(yolo.get("status") or ""),
            str(yolo.get("primary_object") or ""),
            str(yolo.get("valid_evidence_count") or ""),
        ]
    )
    packet = vlm.get("evidence_packet") if isinstance(vlm.get("evidence_packet"), dict) else {}
    terms.extend(_flatten_search_terms(packet.get("allowed_confirmed_objects")))
    terms.extend(_flatten_search_terms(packet.get("top_detections")))
    terms.extend(_flatten_search_terms(packet.get("hand_object_interactions")))
    return [term for term in terms if term]


def _formal_material_reference_items(exp_dir: Path, experiment_id: str) -> Dict[str, Any]:
    ref_root, rows = _material_reference_root_and_rows(exp_dir)
    if ref_root is None:
        return {"items": [], "source": None, "source_mtime": float(exp_dir.stat().st_mtime)}
    items: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
        if asset_kind not in {KEYFRAME_KIND, KEY_CLIP_KIND}:
            continue
        path = _material_reference_row_path(row, ref_root)
        if path is None or not path.is_file():
            continue
        start_sec = row.get("time_start", row.get("start_sec"))
        end_sec = row.get("time_end", row.get("end_sec", start_sec))
        item_id = str(row.get("material_id") or row.get("item_id") or row.get("candidate_id") or row.get("micro_segment_id") or f"material_reference_{index:04d}")
        event_id = str(row.get("event_id") or row.get("micro_segment_id") or item_id)
        display_name = str(row.get("display_name") or row.get("action_name") or path.stem)
        primary_object = str(row.get("primary_object") or row.get("object_label") or "")
        object_labels = row.get("object_labels") if isinstance(row.get("object_labels"), list) else []
        if primary_object and primary_object not in object_labels:
            object_labels = [primary_object, *object_labels]
        published_paths = {
            "preview": str(path) if asset_kind == KEYFRAME_KIND else "",
            "clip": str(path) if asset_kind == KEY_CLIP_KIND else "",
            "keyframe": str(path) if asset_kind == KEYFRAME_KIND else "",
        }
        item = enrich_material_taxonomy(
            {
                **row,
                "material_id": item_id,
                "event_id": event_id,
                "experiment_id": row.get("experiment_id") or experiment_id,
                "event_type": row.get("event_type") or row.get("action_name") or asset_kind,
                "display_name": display_name,
                "stable_name": row.get("stable_name") or path.stem,
                "time_start": start_sec,
                "time_end": end_sec,
                "evidence_grade": row.get("evidence_grade") or row.get("evidence_level") or "strong",
                "review_status": row.get("review_status") or "accepted",
                "published_paths": {**(row.get("published_paths") or {}), **published_paths},
                "source_container": row.get("source_container") or {"class_name": primary_object} if primary_object else row.get("source_container"),
                "object_labels": object_labels,
                "actions": row.get("actions") or [row.get("action_name") or asset_kind],
                "semantic_search_terms": _semantic_material_terms(row),
                "payload": row,
            }
        )
        items.append(item)
    index_path = ref_root / MATERIAL_INDEX_FILENAME
    source_mtime = float(index_path.stat().st_mtime) if index_path.exists() else float(ref_root.stat().st_mtime)
    return {"items": enrich_material_best_scores(items), "source": str(ref_root), "source_mtime": source_mtime}


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
        published_path = exp_dir / "published_materials.json"
        source = "published_materials.json"
        source_mtime = float(published_path.stat().st_mtime) if published_path.exists() else float(exp_dir.stat().st_mtime)
        if published_path.exists():
            payload = _load_json(published_path, {})
            raw_items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
        else:
            fallback = _formal_material_reference_items(exp_dir, experiment_id)
            raw_items = [item for item in (fallback.get("items") or []) if isinstance(item, dict)]
            source = "formal_material_references"
            source_mtime = float(fallback.get("source_mtime") or source_mtime)
        if not raw_items:
            if not published_path.exists():
                source = "formal_material_references"
                experiments.append({"experiment_id": experiment_id, "status": "skipped", "reason": "published_materials.json missing"})
                continue
        count = 0
        official_counts = _official_usage_counts(exp_dir)
        review_counts = _review_usage_counts(exp_dir)
        usage_metrics = _material_usage_metrics(exp_dir)
        for item in raw_items:
            event_id = str(item.get("event_id") or "")
            material_id = str(item.get("material_id") or item.get("item_id") or item.get("candidate_id") or event_id)
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
            effective_material_id = str(enriched.get("material_id") or enriched.get("item_id") or enriched.get("candidate_id") or material_id)
            workspace_key = f"{effective_experiment_id}:{effective_material_id}" if effective_experiment_id else effective_material_id
            previous = items_by_key.get(workspace_key)
            if previous is None or source_mtime >= previous[0]:
                items_by_key[workspace_key] = (source_mtime, enriched)
            count += 1
        experiments.append({"experiment_id": experiment_id, "status": "indexed", "source": source, "published_count": count})
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


def _iso_timestamp(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), timezone.utc).isoformat()


def _workspace_published_sqlite_count(index_path: Path) -> Optional[int]:
    if not index_path.exists():
        return None
    conn = sqlite3.connect(str(index_path))
    try:
        _init_workspace_published_schema(conn)
        row = conn.execute("SELECT COUNT(*) FROM published_materials").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def _formal_reference_counts(exp_dir: Path) -> Dict[str, Any]:
    ref_root, rows = _material_reference_root_and_rows(exp_dir)
    if ref_root is None:
        return {
            "root": None,
            "source_path": None,
            "source_mtime": None,
            "material_count": 0,
            "report_count": 0,
        }
    material_count = 0
    report_count = 0
    for row in rows:
        asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
        if asset_kind == REPORT_KIND:
            report_count += 1
        elif asset_kind in {KEYFRAME_KIND, KEY_CLIP_KIND}:
            material_count += 1
    jsonl_path = ref_root / MATERIAL_INDEX_FILENAME
    json_path = ref_root / "\u7d20\u6750\u7d22\u5f15.json"
    source_path = jsonl_path if jsonl_path.exists() else json_path if json_path.exists() else ref_root
    return {
        "root": str(ref_root),
        "source_path": str(source_path),
        "source_mtime": float(source_path.stat().st_mtime) if source_path.exists() else None,
        "material_count": material_count,
        "report_count": report_count,
    }


def _workspace_published_lifecycle_snapshot(experiments_root: Path, index_path: Path) -> Dict[str, Any]:
    expected_keys: set[str] = set()
    formal_jsonl_material_count = 0
    formal_report_count = 0
    latest_source_mtime: Optional[float] = None
    experiments: List[Dict[str, Any]] = []
    if not experiments_root.exists():
        experiments_root.mkdir(parents=True, exist_ok=True)

    for exp_dir in sorted(path for path in experiments_root.iterdir() if path.is_dir()):
        experiment_id = experiment_id_from_dir(exp_dir)
        published_path = exp_dir / "published_materials.json"
        formal_counts = _formal_reference_counts(exp_dir)
        formal_jsonl_material_count += int(formal_counts["material_count"])
        formal_report_count += int(formal_counts["report_count"])
        source = "none"
        source_path: Optional[str] = None
        source_mtime: Optional[float] = None
        expected_items: List[Dict[str, Any]] = []
        if published_path.exists():
            payload = _load_json(published_path, {})
            expected_items = [item for item in (payload.get("items") or []) if isinstance(item, dict)]
            source = "published_materials.json"
            source_path = str(published_path)
            source_mtime = float(published_path.stat().st_mtime)
        else:
            fallback = _formal_material_reference_items(exp_dir, experiment_id)
            fallback_items = [item for item in (fallback.get("items") or []) if isinstance(item, dict)]
            if fallback_items or formal_counts["material_count"] or formal_counts["report_count"]:
                expected_items = fallback_items
                source = "formal_material_references"
                source_path = str(fallback.get("source") or formal_counts.get("source_path") or "")
                source_mtime = float(fallback.get("source_mtime") or formal_counts.get("source_mtime") or exp_dir.stat().st_mtime)
        experiment_keys: set[str] = set()
        for item in expected_items:
            event_id = str(item.get("event_id") or "")
            material_id = str(item.get("material_id") or event_id)
            effective_experiment_id = str(item.get("experiment_id") or experiment_id)
            effective_material_id = str(item.get("material_id") or material_id)
            workspace_key = f"{effective_experiment_id}:{effective_material_id}" if effective_experiment_id else effective_material_id
            expected_keys.add(workspace_key)
            experiment_keys.add(workspace_key)
        expected_count = len(experiment_keys)
        if source_mtime is not None:
            latest_source_mtime = max(latest_source_mtime or 0.0, source_mtime)
        if expected_count or formal_counts["material_count"] or formal_counts["report_count"] or published_path.exists():
            experiments.append(
                {
                    "experiment_id": experiment_id,
                    "source": source,
                    "source_path": source_path,
                    "source_mtime": _iso_timestamp(source_mtime),
                    "expected_indexable_count": expected_count,
                    "formal_jsonl_material_count": formal_counts["material_count"],
                    "formal_report_count": formal_counts["report_count"],
                }
            )

    index_exists = index_path.exists()
    index_mtime = float(index_path.stat().st_mtime) if index_exists else None
    sqlite_count = _workspace_published_sqlite_count(index_path)
    expected_total = len(expected_keys)
    warnings: List[Dict[str, Any]] = []
    if not index_exists:
        warnings.append({"code": "missing_index", "message": "workspace published materials index is missing"})
    if sqlite_count is not None and sqlite_count != expected_total:
        warnings.append(
            {
                "code": "count_mismatch",
                "message": "workspace published materials index count differs from formal source count",
                "sqlite_count": sqlite_count,
                "expected_indexable_count": expected_total,
            }
        )
    if index_mtime is not None and latest_source_mtime is not None and index_mtime + 1e-6 < latest_source_mtime:
        warnings.append(
            {
                "code": "stale_index",
                "message": "workspace published materials index is older than a published-material source",
                "index_mtime": _iso_timestamp(index_mtime),
                "latest_source_mtime": _iso_timestamp(latest_source_mtime),
            }
        )

    return {
        "schema_version": "workspace_published_materials_lifecycle.v1",
        "status": "ok" if not warnings else ("missing" if not index_exists else "needs_rebuild"),
        "index_path": str(index_path),
        "index_exists": index_exists,
        "index_mtime": _iso_timestamp(index_mtime),
        "latest_source_mtime": _iso_timestamp(latest_source_mtime),
        "sqlite_count": sqlite_count,
        "expected_indexable_count": expected_total,
        "formal_jsonl_material_count": formal_jsonl_material_count,
        "formal_report_count": formal_report_count,
        "experiment_count": len(experiments),
        "experiments": experiments,
        "warnings": warnings,
    }


def check_workspace_published_materials_lifecycle(
    experiments_root: str | Path,
    index_path: str | Path,
    *,
    auto_rebuild: bool = False,
) -> Dict[str, Any]:
    """Report whether workspace published-materials SQLite is aligned with its formal sources."""
    root = Path(experiments_root)
    output = Path(index_path)
    report = _workspace_published_lifecycle_snapshot(root, output)
    if not auto_rebuild or report.get("status") == "ok":
        return report
    rebuild_result = rebuild_workspace_published_materials_index(root, output)
    refreshed = _workspace_published_lifecycle_snapshot(root, output)
    refreshed["status"] = "rebuilt" if refreshed.get("status") == "ok" else refreshed.get("status")
    refreshed["warnings_before_rebuild"] = report.get("warnings") or []
    refreshed["rebuild"] = rebuild_result
    return refreshed


def _init_workspace_published_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS published_materials (
            material_id TEXT PRIMARY KEY,
            experiment_id TEXT,
            event_id TEXT,
            event_type TEXT,
            canonical_action_type TEXT,
            canonical_object TEXT,
            sop_phase TEXT,
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
    _ensure_columns(
        conn,
        "published_materials",
        {
            "canonical_action_type": "TEXT",
            "canonical_object": "TEXT",
            "sop_phase": "TEXT",
            "stable_name": "TEXT",
            "actor_name": "TEXT",
            "time_start": "REAL",
            "time_end": "REAL",
            "evidence_grade": "TEXT",
            "review_status": "TEXT",
            "official_linked": "INTEGER DEFAULT 0",
            "warning_count": "INTEGER DEFAULT 0",
            "click_count": "INTEGER DEFAULT 0",
            "review_count": "INTEGER DEFAULT 0",
            "official_usage_count": "INTEGER DEFAULT 0",
            "experiment_type": "TEXT",
        },
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_type ON published_materials(event_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_experiment ON published_materials(experiment_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_actor ON published_materials(actor_name)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_canonical_action ON published_materials(canonical_action_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_official ON published_materials(official_linked)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_workspace_published_experiment_type ON published_materials(experiment_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_events_material ON material_usage_events(material_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_counts_experiment ON material_usage_counts(experiment_id)")


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Dict[str, str]) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _insert_workspace_published(conn: sqlite3.Connection, item: Dict[str, Any]) -> None:
    item = enrich_material_best_score(enrich_material_taxonomy(item))
    paths = item.get("published_paths") or {}
    source = item.get("source_container") or {}
    target = item.get("target_container") or {}
    vlm = item.get("vlm_semantics") if isinstance(item.get("vlm_semantics"), dict) else {}
    yolo = item.get("yolo_recheck") if isinstance(item.get("yolo_recheck"), dict) else {}
    text = " ".join(
        str(part)
        for part in [
            item.get("display_name"),
            item.get("stable_name"),
            item.get("event_type"),
            item.get("canonical_action_type"),
            item.get("canonical_object"),
            item.get("sop_phase"),
            item.get("interaction_family"),
            item.get("best_reason"),
            item.get("best_score"),
            item.get("actor_name"),
            source,
            target,
            item.get("primary_object"),
            " ".join(str(label) for label in (item.get("object_labels") or [])),
            " ".join(str(action) for action in (item.get("actions") or [])),
            " ".join(str(term) for term in (item.get("semantic_search_terms") or [])),
            " ".join(_semantic_material_terms(item)),
            vlm.get("description"),
            vlm.get("physical_action"),
            str(vlm.get("physical_action") or "").replace("_", " "),
            " ".join(_flatten_search_terms(vlm.get("confirmed_objects"))),
            " ".join(_flatten_search_terms(vlm.get("uncertain_objects"))),
            yolo.get("status"),
            yolo.get("primary_object"),
            yolo.get("valid_evidence_count"),
            item.get("evidence_grade"),
            item.get("review_status"),
            " ".join(item.get("warnings") or []),
        ]
        if part
    )
    material_id = str(item.get("material_id") or item.get("item_id") or item.get("candidate_id") or item.get("event_id"))
    experiment_id = str(item.get("experiment_id") or "")
    workspace_material_id = f"{experiment_id}:{material_id}" if experiment_id else material_id
    existing_usage = conn.execute("SELECT click_count FROM material_usage_counts WHERE material_id = ?", (workspace_material_id,)).fetchone()
    click_count = max(int(item.get("click_count") or 0), int(existing_usage[0] or 0) if existing_usage else 0)
    row = {
        "material_id": workspace_material_id,
        "experiment_id": experiment_id or item.get("experiment_id"),
        "event_id": item.get("event_id"),
        "event_type": item.get("event_type"),
        "canonical_action_type": item.get("canonical_action_type"),
        "canonical_object": item.get("canonical_object"),
        "sop_phase": item.get("sop_phase"),
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
        (material_id, experiment_id, event_id, event_type, canonical_action_type, canonical_object, sop_phase,
         display_name, stable_name, actor_name,
         time_start, time_end, evidence_grade, review_status,
         published_path, material_publish_path, clip_path, preview_path, payload_json, searchable_text,
         official_linked, warning_count, click_count, review_count, official_usage_count, experiment_type)
        VALUES (:material_id, :experiment_id, :event_id, :event_type, :canonical_action_type, :canonical_object, :sop_phase,
         :display_name, :stable_name, :actor_name,
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
    "canonical_action_type": "COALESCE(p.canonical_action_type, '')",
    "canonical_object": "COALESCE(p.canonical_object, '')",
    "sop_phase": "COALESCE(p.sop_phase, '')",
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
    canonical_action_type: Optional[str],
    canonical_object: Optional[str],
    sop_phase: Optional[str],
    actor_name: Optional[str],
    operator_role: str,
    allowed_experiment_ids: Optional[List[str]],
    scoring_profile_hash: str,
) -> str:
    payload = {
        "text": text or "",
        "event_type": event_type or "",
        "canonical_action_type": canonical_action_type or "",
        "canonical_object": canonical_object or "",
        "sop_phase": sop_phase or "",
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


_CANONICAL_QUERY_RULES = [
    {
        "needles": ("天平", "称量", "balance", "weigh"),
        "canonical_action_type": "hand-balance",
        "canonical_object": "balance",
        "sop_phase": "balance-weighing",
        "terms": ("天平", "称量", "balance", "weighing"),
    },
    {
        "needles": ("试剂瓶", "取试剂", "瓶", "bottle", "reagent"),
        "canonical_action_type": "hand-bottle",
        "canonical_object": "bottle",
        "sop_phase": "reagent-bottle-handling",
        "terms": ("试剂瓶", "瓶", "bottle", "reagent bottle"),
    },
    {
        "needles": ("药匙", "加样", "spatula", "scoop"),
        "canonical_action_type": "hand-spatula",
        "canonical_object": "spatula",
        "sop_phase": "solid-transfer",
        "terms": ("药匙", "加样", "spatula", "solid transfer"),
    },
    {
        "needles": ("称量纸", "纸", "paper", "weighing paper"),
        "canonical_action_type": "hand-paper",
        "canonical_object": "paper",
        "sop_phase": "weighing-paper-prep",
        "terms": ("称量纸", "纸", "paper", "weighing paper"),
    },
    {
        "needles": ("承接", "hand-container", "container-handling"),
        "canonical_action_type": "hand-container",
        "canonical_object": "container",
        "sop_phase": "container-handling",
        "terms": ("容器", "承接", "烧杯", "beaker", "container"),
    },
]


def _split_query_filter(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _canonical_query_intent(text: Optional[str]) -> Dict[str, List[str]]:
    normalized = str(text or "").strip().lower()
    intent: Dict[str, List[str]] = {
        "canonical_action_type": [],
        "canonical_object": [],
        "sop_phase": [],
        "terms": [],
    }
    if not normalized:
        return intent
    for rule in _CANONICAL_QUERY_RULES:
        if rule["canonical_action_type"] == "hand-balance" and any(token in normalized for token in ("称量纸", "weighing paper", "paper")):
            continue
        if any(needle.lower() in normalized for needle in rule["needles"]):
            for key in ("canonical_action_type", "canonical_object", "sop_phase"):
                value = str(rule[key])
                if value not in intent[key]:
                    intent[key].append(value)
            for term in rule["terms"]:
                if term not in intent["terms"]:
                    intent["terms"].append(term)
    return intent


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql_in_expr(column: str, values: List[str]) -> str:
    if not values:
        return "0"
    return f"{column} IN ({','.join(_sql_literal(value) for value in values)})"


def _canonical_boost_expression(intent: Dict[str, List[str]], filters: Dict[str, List[str]]) -> str:
    action_values = sorted(set(intent.get("canonical_action_type") or []) | set(filters.get("canonical_action_type") or []))
    object_values = sorted(set(intent.get("canonical_object") or []) | set(filters.get("canonical_object") or []))
    phase_values = sorted(set(intent.get("sop_phase") or []) | set(filters.get("sop_phase") or []))
    return (
        "("
        f"CASE WHEN {_sql_in_expr('p.canonical_action_type', action_values)} THEN 3.0 ELSE 0 END + "
        f"CASE WHEN {_sql_in_expr('p.canonical_object', object_values)} THEN 1.2 ELSE 0 END + "
        f"CASE WHEN {_sql_in_expr('p.sop_phase', phase_values)} THEN 0.8 ELSE 0 END"
        ")"
    )


def _append_in_filter(clauses: List[str], params: List[Any], column: str, values: List[str]) -> None:
    if not values:
        return
    placeholders = ",".join("?" for _ in values)
    clauses.append(f"{column} IN ({placeholders})")
    params.extend(values)


def _canonical_text_clause(intent: Dict[str, List[str]], params: List[Any]) -> Optional[str]:
    parts: List[str] = []
    for column, key in (
        ("p.canonical_action_type", "canonical_action_type"),
        ("p.canonical_object", "canonical_object"),
        ("p.sop_phase", "sop_phase"),
    ):
        values = intent.get(key) or []
        if values:
            placeholders = ",".join("?" for _ in values)
            parts.append(f"{column} IN ({placeholders})")
            params.extend(values)
    for term in intent.get("terms") or []:
        parts.append("p.searchable_text LIKE ?")
        params.append(f"%{term}%")
    return "(" + " OR ".join(parts) + ")" if parts else None


def _fts_query_text(text: str) -> str:
    cleaned = str(text or "").strip().replace('"', '""')
    return f'"{cleaned}"' if cleaned else ""


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
    canonical_action_type: Optional[str] = None,
    canonical_object: Optional[str] = None,
    sop_phase: Optional[str] = None,
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
        canonical_intent = _canonical_query_intent(text)
        canonical_filters = {
            "canonical_action_type": _split_query_filter(canonical_action_type),
            "canonical_object": _split_query_filter(canonical_object),
            "sop_phase": _split_query_filter(sop_phase),
        }
        filters_hash = _filter_hash(
            text=text,
            event_type=event_type,
            canonical_action_type=canonical_action_type,
            canonical_object=canonical_object,
            sop_phase=sop_phase,
            actor_name=actor_name,
            operator_role=operator_role,
            allowed_experiment_ids=allowed_experiment_ids,
            scoring_profile_hash=str(scoring_profile.get("profile_hash") or ""),
        )
        canonical_boost_expr = _canonical_boost_expression(canonical_intent, canonical_filters)
        use_fts_text = bool(text and not any(canonical_intent.values()))
        relevance_expr = "bm25(published_materials_fts)" if use_fts_text else None
        business_expr = _business_score_expression(relevance_expr, scoring_profile)
        score_expr = (
            f"{relevance_expr if relevance_expr else canonical_boost_expr} AS relevance_score, "
            f"{canonical_boost_expr} AS canonical_match_score, "
            f"({business_expr} + {canonical_boost_expr}) AS business_score"
        )
        if use_fts_text:
            sql = (
                f"SELECT p.*, {score_expr} FROM published_materials p "
                "JOIN published_materials_fts f ON p.material_id = f.material_id "
            )
            clauses.append("published_materials_fts MATCH ?")
            params.append(_fts_query_text(text))
        else:
            sql = f"SELECT p.*, {score_expr} FROM published_materials p"
            if text:
                text_clauses: List[str] = []
                canonical_clause = _canonical_text_clause(canonical_intent, params)
                if canonical_clause:
                    text_clauses.append(canonical_clause)
                text_clauses.append("p.searchable_text LIKE ?")
                params.append(f"%{text}%")
                clauses.append("(" + " OR ".join(text_clauses) + ")")
        if event_type:
            clauses.append("p.event_type = ?")
            params.append(event_type)
        _append_in_filter(clauses, params, "p.canonical_action_type", canonical_filters["canonical_action_type"])
        _append_in_filter(clauses, params, "p.canonical_object", canonical_filters["canonical_object"])
        _append_in_filter(clauses, params, "p.sop_phase", canonical_filters["sop_phase"])
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
        elif normalized_sort_by == "relevance":
            sort_column = canonical_boost_expr
        elif normalized_sort_by == "business_score":
            sort_column = f"({business_expr} + {canonical_boost_expr})"
        else:
            sort_column = SORT_FIELDS.get(normalized_sort_by, SORT_FIELDS["time_start"])
        direction = "DESC" if str(sort_order).lower() == "desc" else "ASC"
        if normalized_sort_by == "relevance" and text and not relevance_expr:
            direction = "DESC"
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
        elif normalized_sort_by == "relevance":
            sql += f" ORDER BY {canonical_boost_expr} DESC, COALESCE(p.time_start, 0) ASC, p.material_id ASC LIMIT ?"
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
            cursor_sort_order = "asc" if cursor_sort_by == "relevance" and relevance_expr else direction.lower()
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
            "sort_order": ("asc" if normalized_sort_by == "relevance" and relevance_expr else direction.lower()),
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
