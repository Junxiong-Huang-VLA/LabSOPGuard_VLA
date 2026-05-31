from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import write_jsonl


MATERIAL_INDEX_JSON = "\u7d20\u6750\u7d22\u5f15.json"
MATERIAL_INDEX_JSONL = "\u7d20\u6750\u7d22\u5f15.jsonl"
KEYFRAME_KIND = "\u5173\u952e\u5e27"
KEY_CLIP_KIND = "\u5173\u952e\u7247\u6bb5"
REPORT_KIND = "\u4e13\u4e1a\u62a5\u544a"
DEFAULT_SQLITE_NAME = "key_material_references.sqlite"
DEFAULT_REFERENCES_NAME = "key_material_references.jsonl"
SCHEMA_VERSION = "key_material_reference_index.v1"
_STORED_FILE_REAL_CACHE: dict[tuple[str, int, int], bool] = {}
_STORED_FILE_REAL_CACHE_LOCK = threading.Lock()
_STORED_FILE_SHA256_CACHE: dict[tuple[str, int, int], str] = {}
_STORED_FILE_SHA256_CACHE_LOCK = threading.Lock()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
REPORT_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}
NON_REAL_ASSET_MARKERS = ("placeholder", "poster", "synthetic", "dry_run", "dry-run")
NON_REAL_VISUAL_MARKERS = (
    "black_screen",
    "white_screen",
    "blank_screen",
    "黑屏",
    "白屏",
    "黑白屏",
)

HAND_LABELS = {"hand", "hands", "gloved_hand", "glove", "gloves"}
ACTION_OBJECT_BY_LABEL = {
    "balance": "balance",
    "scale": "balance",
    "paper": "paper",
    "weighing_paper": "paper",
    "spatula": "spatula",
    "pipette": "pipette",
    "pipette_tip": "pipette_tip",
    "reagent_bottle": "bottle",
    "sample_bottle": "bottle",
    "sample_bottle_blue": "bottle",
    "bottle": "bottle",
    "vial": "bottle",
    "beaker": "container",
    "container": "container",
    "tube": "container",
    "flask": "container",
}


def _file_cache_key(path: Path) -> tuple[str, int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    inode = int(getattr(stat, "st_ino", 0) or 0)
    identity = f"{int(getattr(stat, 'st_dev', 0) or 0)}:{inode}" if inode else str(path.resolve())
    return identity, int(stat.st_size), int(stat.st_mtime_ns)


def _stored_file_is_real(path: Path | None, *, exists: bool) -> bool:
    if not exists or path is None:
        return False
    cache_key = _file_cache_key(path)
    if cache_key is not None:
        with _STORED_FILE_REAL_CACHE_LOCK:
            cached = _STORED_FILE_REAL_CACHE.get(cache_key)
            if cached is not None:
                return cached
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            if cache_key is not None:
                with _STORED_FILE_REAL_CACHE_LOCK:
                    _STORED_FILE_REAL_CACHE[cache_key] = False
            return False
        path_text = path.name.lower()
        if any(marker in path_text for marker in (*NON_REAL_ASSET_MARKERS, *NON_REAL_VISUAL_MARKERS)):
            if cache_key is not None:
                with _STORED_FILE_REAL_CACHE_LOCK:
                    _STORED_FILE_REAL_CACHE[cache_key] = False
            return False
        header = path.read_bytes()[:128].upper()
    except OSError:
        return False
    if header.startswith(b"DRY RUN") or b"PLACEHOLDER" in header or b"SYNTHETIC" in header:
        result = False
    else:
        result = _stored_file_visual_content_is_real(path)
    if cache_key is not None:
        with _STORED_FILE_REAL_CACHE_LOCK:
            _STORED_FILE_REAL_CACHE[cache_key] = result
    return result


def _stored_file_visual_content_is_real(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return _image_visual_content_is_real(path)
    if suffix in VIDEO_EXTENSIONS:
        return _video_visual_content_is_real(path)
    return True


def _image_visual_content_is_real(path: Path) -> bool:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.thumbnail((128, 128))
        pixels = list(image.getdata())
    except Exception:
        return True
    return _rgb_pixels_are_real_material(pixels)


def _video_visual_content_is_real(path: Path) -> bool:
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return True
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        positions = [0]
        if frame_count > 4:
            positions.extend([max(0, frame_count // 2), max(0, frame_count - 2)])
        frames: list[Any] = []
        for position in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()
        if not frames:
            return True
        return any(_cv_frame_is_real_material(frame, np_module=np) for frame in frames)
    except Exception:
        return True


def _cv_frame_is_real_material(frame: Any, *, np_module: Any) -> bool:
    try:
        height, width = frame.shape[:2]
        step_y = max(1, height // 96)
        step_x = max(1, width // 128)
        sample = frame[::step_y, ::step_x, :3].astype("float32")
        channels = sample.reshape((-1, 3))
        luma = (0.114 * channels[:, 0]) + (0.587 * channels[:, 1]) + (0.299 * channels[:, 2])
        mean_luma = float(luma.mean())
        std_luma = float(luma.std())
        dark_ratio = float((luma < 8.0).mean())
        bright_ratio = float((luma > 247.0).mean())
        channel_range = channels.max(axis=1) - channels.min(axis=1)
        color_std = float(np_module.mean(channel_range))
    except Exception:
        return True
    return _visual_stats_are_real_material(mean_luma, std_luma, dark_ratio, bright_ratio, color_std)


def _rgb_pixels_are_real_material(pixels: list[tuple[int, int, int]]) -> bool:
    if not pixels:
        return False
    count = float(len(pixels))
    luma_values = [(0.299 * r) + (0.587 * g) + (0.114 * b) for r, g, b in pixels]
    mean_luma = sum(luma_values) / count
    variance = sum((value - mean_luma) ** 2 for value in luma_values) / count
    std_luma = variance ** 0.5
    dark_ratio = sum(1 for value in luma_values if value < 8.0) / count
    bright_ratio = sum(1 for value in luma_values if value > 247.0) / count
    color_std = sum(max(pixel) - min(pixel) for pixel in pixels) / count
    return _visual_stats_are_real_material(mean_luma, std_luma, dark_ratio, bright_ratio, color_std)


def _visual_stats_are_real_material(
    mean_luma: float,
    std_luma: float,
    dark_ratio: float,
    bright_ratio: float,
    color_std: float,
) -> bool:
    if dark_ratio >= 0.985 or bright_ratio >= 0.985:
        return False
    if mean_luma <= 4.0 or mean_luma >= 251.0:
        return False
    if std_luma < 1.2 and color_std < 1.2:
        return False
    return True


def _row_has_non_real_marker(row: Mapping[str, Any]) -> bool:
    if bool(row.get("placeholder") or row.get("dry_run") or row.get("dry_run_placeholder")):
        return True
    fields: list[Any] = [
        row.get("candidate_source"),
        row.get("source_type"),
        row.get("asset_source"),
        row.get("role"),
        row.get("reason"),
        row.get("missing_reason"),
        Path(str(row.get("stored_file") or "")).name,
        Path(str(row.get("source_file") or "")).name,
        row.get("file_name"),
        row.get("stored_filename"),
    ]
    for key in ("quality_reasons", "warnings", "review_reason_codes"):
        value = row.get(key)
        if isinstance(value, list):
            fields.extend(value)
        else:
            fields.append(value)
    text = " ".join(str(value or "").lower() for value in fields)
    return any(marker in text for marker in NON_REAL_ASSET_MARKERS)


def _row_source_real(row: Mapping[str, Any], stored_path: Path | None, exists: bool) -> bool:
    if row.get("source_real") is False:
        return False
    if _row_has_non_real_marker(row):
        return False
    return _stored_file_is_real(stored_path, exists=exists)


def load_material_reference_rows(material_root: str | Path) -> list[dict[str, Any]]:
    """Load formal material reference rows from a material handoff directory."""

    root = Path(material_root)
    jsonl_path = root / MATERIAL_INDEX_JSONL
    if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
        rows: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {jsonl_path}:{line_number}: {exc.msg}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    references_path = root / DEFAULT_REFERENCES_NAME
    if references_path.exists() and references_path.stat().st_size > 0:
        rows = []
        with references_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {references_path}:{line_number}: {exc.msg}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    json_path = root / MATERIAL_INDEX_JSON
    if not json_path.exists():
        raise FileNotFoundError(f"Material index not found under {root}: {MATERIAL_INDEX_JSONL} or {MATERIAL_INDEX_JSON}")
    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        records = payload.get("records") or []
        return [dict(row) for row in records if isinstance(row, dict)]
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    raise ValueError(f"Expected material index object/list in {json_path}")


def build_key_material_reference_index(
    material_root: str | Path,
    *,
    sqlite_path: str | Path | None = None,
    references_path: str | Path | None = None,
    include_reports: bool = False,
) -> dict[str, Any]:
    """Build a local SQLite/FTS index and normalized JSONL references for key materials."""

    root = Path(material_root)
    rows = load_material_reference_rows(root)
    normalized: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows, start=1):
        item = normalize_material_reference(row, root, row_index=row_index)
        if item["asset_type"] == "report" and not include_reports:
            skipped.append({"row_index": row_index, "reason": "report_not_indexed", "file_name": item.get("file_name")})
            continue
        if item.get("source_real") is not True or item.get("placeholder") is True or not item.get("exists"):
            skipped.append({"row_index": row_index, "reason": item.get("missing_reason") or "non_real_material_not_indexed", "file_name": item.get("file_name")})
            continue
        if item["asset_type"] not in {"keyframe", "video_clip", "report"}:
            skipped.append({"row_index": row_index, "reason": "unsupported_asset_type", "file_name": item.get("file_name")})
            continue
        normalized.append(item)

    db_path = Path(sqlite_path) if sqlite_path is not None else root / DEFAULT_SQLITE_NAME
    refs_path = Path(references_path) if references_path is not None else root / DEFAULT_REFERENCES_NAME
    fts_enabled = _write_sqlite_index(db_path, normalized, material_root=root)
    write_jsonl(refs_path, normalized)

    counts: dict[str, int] = {}
    missing_count = 0
    for item in normalized:
        asset_type = str(item.get("asset_type") or "unknown")
        counts[asset_type] = counts.get(asset_type, 0) + 1
        if not item.get("exists"):
            missing_count += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "material_root": str(root),
        "source_rows": len(rows),
        "indexed_count": len(normalized),
        "skipped_count": len(skipped),
        "missing_count": missing_count,
        "asset_type_counts": dict(sorted(counts.items())),
        "sqlite_path": str(db_path),
        "references_path": str(refs_path),
        "fts_enabled": fts_enabled,
        "skipped": skipped,
    }


def normalize_material_reference(row: Mapping[str, Any], material_root: Path, *, row_index: int = 0) -> dict[str, Any]:
    asset_type = _asset_type(row)
    stored_path = _resolve_stored_path(row, material_root)
    exists = bool(stored_path and stored_path.exists())
    file_name = str(row.get("file_name") or row.get("stored_filename") or (stored_path.name if stored_path else ""))
    material_id = _material_id(row, stored_path=stored_path, row_index=row_index)
    start_sec = _float_or_none(row.get("time_start", row.get("start_sec")))
    end_sec = _float_or_none(row.get("time_end", row.get("end_sec", start_sec)))
    size_bytes = int(stored_path.stat().st_size) if exists and stored_path is not None else int(row.get("size_bytes") or 0)
    secondary_objects = _secondary_objects(row)
    secondary_actions = _secondary_actions(row, secondary_objects)
    objects = _objects(row, secondary_objects)
    actions = _actions(row, secondary_actions)
    window_audit = _mapping_or_empty(row.get("window_audit"))
    target_object_support = _mapping_or_empty(row.get("target_object_support") or window_audit.get("target_object_support"))
    secondary_object_support = _list_mappings(row.get("secondary_object_support") or window_audit.get("secondary_object_support"))
    uncertainty_reasons = _list_strings(row.get("uncertainty_reasons") or window_audit.get("uncertainty_reasons") or window_audit.get("reasons"))
    evidence_chain = _mapping_or_empty(row.get("evidence_chain"))
    dual_event_id = _string(
        row.get("dual_event_id")
        or row.get("dual_view_event_id")
        or row.get("dual_view_action_event_id")
        or evidence_chain.get("dual_event_id")
        or evidence_chain.get("dual_view_action_event_id")
    )
    physical_evidence_diagnostics = _mapping_or_empty(
        row.get("physical_evidence_diagnostics") or evidence_chain.get("physical_evidence_diagnostics")
    )
    formal_publish_gate = _mapping_or_empty(row.get("formal_publish_gate") or evidence_chain.get("formal_publish_gate"))
    source_yolo_evidence = _list_mappings(
        row.get("source_yolo_evidence")
        or evidence_chain.get("source_yolo_evidence")
        or row.get("yolo_evidence")
        or evidence_chain.get("yolo_evidence")
    )
    searchable_text = _searchable_text(
        {
            **dict(row),
            "secondary_objects": secondary_objects,
            "secondary_actions": secondary_actions,
            "objects": objects,
            "actions": actions,
            "window_audit": window_audit,
            "target_object_support": target_object_support,
            "secondary_object_support": secondary_object_support,
            "uncertainty_reasons": uncertainty_reasons,
            "dual_event_id": dual_event_id,
            "evidence_chain": evidence_chain,
            "physical_evidence_diagnostics": physical_evidence_diagnostics,
            "formal_publish_gate": formal_publish_gate,
        },
        stored_path=stored_path,
        asset_type=asset_type,
    )
    created_at = (
        datetime.fromtimestamp(stored_path.stat().st_mtime, timezone.utc).isoformat()
        if exists and stored_path is not None
        else datetime.now(timezone.utc).isoformat()
    )
    stored_file = _path_relative_to_root(stored_path, material_root) if stored_path is not None else ""
    source_file = _portable_path_value(
        row.get("source_file") or row.get("source_clip") or row.get("source_clip_path"),
        material_root,
    )
    source_real = _row_source_real(row, stored_path, exists)
    placeholder = not source_real
    session_id = _string(row.get("session_id") or row.get("run_id") or row.get("session") or row.get("package_session_id"))
    material_date = _material_reference_date(row, stored_path)

    return {
        "schema_version": SCHEMA_VERSION,
        "material_id": material_id,
        "experiment_id": _string(row.get("experiment_id")),
        "session_id": session_id,
        "date": material_date,
        "asset_type": asset_type,
        "asset_kind": _string(row.get("asset_kind") or row.get("material_type")),
        "action_name": _string(row.get("action_name") or row.get("event_type")),
        "display_name": _string(row.get("display_name") or row.get("action_name") or Path(file_name).stem),
        "segment_id": _string(row.get("segment_id") or row.get("parent_segment_id")),
        "micro_segment_id": _string(row.get("micro_segment_id")),
        "evidence_group_id": _string(row.get("evidence_group_id")),
        "material_group_id": _string(row.get("material_group_id")),
        "physical_action_material_id": _string(row.get("physical_action_material_id")),
        "evidence_window_id": _string(row.get("evidence_window_id")),
        "dual_event_id": dual_event_id,
        "dual_view_action_event_id": _string(row.get("dual_view_action_event_id") or dual_event_id),
        "dual_event_binding_source": _string(
            row.get("dual_event_binding_source") or evidence_chain.get("dual_event_binding_source")
        ),
        "formal_dual_view_action": (
            bool(row.get("formal_dual_view_action") or evidence_chain.get("formal_dual_view_action") or dual_event_id)
            if (row.get("formal_dual_view_action") is not None or evidence_chain.get("formal_dual_view_action") is not None or dual_event_id)
            else None
        ),
        "single_view_candidate": (
            bool(row.get("single_view_candidate"))
            if row.get("single_view_candidate") is not None
            else None
        ),
        "view": _string(row.get("view") or row.get("camera_view")),
        "frame_type": _string(row.get("frame_type") or row.get("frame_role") or row.get("role")),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "primary_object": _string(row.get("primary_object") or row.get("object_label") or row.get("canonical_object")),
        "primary_object_family": _string(row.get("primary_object_family") or row.get("object_family")),
        "object_family": _string(row.get("object_family") or row.get("primary_object_family")),
        "canonical_action_type": _string(row.get("canonical_action_type")),
        "canonical_object": _string(row.get("canonical_object")),
        "secondary_objects": secondary_objects,
        "secondary_actions": secondary_actions,
        "objects": objects,
        "actions": actions,
        "window_audit": window_audit,
        "target_object_support": target_object_support,
        "secondary_object_support": secondary_object_support,
        "uncertainty_reasons": uncertainty_reasons,
        "review_status": _string(row.get("review_status")),
        "candidate_status": _string(row.get("candidate_status")),
        "formal_material_reference": bool(row.get("formal_material_reference")) if row.get("formal_material_reference") is not None else None,
        "approved_by": _string(row.get("approved_by")),
        "approved_at": _string(row.get("approved_at")),
        "quality_score": _float_or_none(row.get("quality_score")),
        "yolo_evidence_count": _int_or_none(row.get("yolo_evidence_count")),
        "valid_yolo_evidence_count": _int_or_none(row.get("valid_yolo_evidence_count")),
        "usable_yolo_evidence_count": _int_or_none(row.get("usable_yolo_evidence_count")),
        "physical_evidence_mode": _string(row.get("physical_evidence_mode") or evidence_chain.get("physical_evidence_mode")),
        "candidate_source": _string(row.get("candidate_source") or evidence_chain.get("candidate_source")),
        "source_yolo_evidence": source_yolo_evidence,
        "physical_evidence_diagnostics": physical_evidence_diagnostics,
        "formal_publish_gate": formal_publish_gate,
        "evidence_chain": evidence_chain,
        "stored_file": stored_file,
        "source_file": source_file,
        "path_mode": "relative_to_material_root",
        "package_uri": f"package://material-root/{stored_file}" if stored_file else "",
        "file_name": file_name,
        "exists": exists,
        "source_real": source_real,
        "placeholder": placeholder,
        "publishable_material": bool(source_real and exists),
        "missing_reason": None if source_real else _string(row.get("missing_reason") or "non_real_or_missing_material_file"),
        "size_bytes": size_bytes,
        "sha256": _string(row.get("sha256")) or (_sha256(stored_path) if exists and stored_path is not None else None),
        "searchable_text": searchable_text,
        "created_at": created_at,
        "payload_json": json.dumps(dict(row), ensure_ascii=False, sort_keys=True),
    }


def query_key_material_reference_index(
    sqlite_path: str | Path,
    *,
    text: str = "",
    asset_type: str | None = None,
    primary_object: str | None = None,
    action: str | None = None,
    session_id: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query a SQLite key-material reference index."""

    db_path = Path(sqlite_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Material reference SQLite index not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        has_fts = _table_exists(conn, "key_material_refs_fts")
        clauses: list[str] = []
        params: list[Any] = []
        if asset_type:
            clauses.append("r.asset_type = ?")
            params.append(asset_type)
        if session_id:
            clauses.append("r.session_id = ?")
            params.append(session_id)
        if date:
            clauses.append("r.date = ?")
            params.append(date)
        if start_date:
            clauses.append("r.date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("r.date <= ?")
            params.append(end_date)
        if primary_object:
            clauses.append(
                "(LOWER(COALESCE(r.primary_object, '')) LIKE ? OR LOWER(COALESCE(r.canonical_object, '')) LIKE ? "
                "OR LOWER(COALESCE(r.secondary_objects, '')) LIKE ? OR LOWER(COALESCE(r.objects, '')) LIKE ?)"
            )
            object_param = f"%{primary_object.lower()}%"
            params.extend([object_param, object_param, object_param, object_param])
        if action:
            clauses.append(
                "(LOWER(COALESCE(r.action_name, '')) LIKE ? OR LOWER(COALESCE(r.canonical_action_type, '')) LIKE ? "
                "OR LOWER(COALESCE(r.secondary_actions, '')) LIKE ? OR LOWER(COALESCE(r.actions, '')) LIKE ? "
                "OR LOWER(COALESCE(r.searchable_text, '')) LIKE ?)"
            )
            action_param = f"%{action.lower()}%"
            params.extend([action_param, action_param, action_param, action_param, action_param])

        if text and has_fts:
            fts_query = _fts_query(text)
            sql = (
                "SELECT r.*, bm25(key_material_refs_fts) AS rank FROM key_material_refs r "
                "JOIN key_material_refs_fts f ON r.material_id = f.material_id "
                "WHERE key_material_refs_fts MATCH ?"
            )
            query_params: list[Any] = [fts_query]
            if clauses:
                sql += " AND " + " AND ".join(clauses)
                query_params.extend(params)
            sql += " ORDER BY rank ASC, COALESCE(r.start_sec, 0) ASC LIMIT ?"
            query_params.append(max(1, int(limit)))
        else:
            sql = "SELECT r.*, 0.0 AS rank FROM key_material_refs r"
            if text:
                clauses.append("LOWER(COALESCE(r.searchable_text, '')) LIKE ?")
                params.append(f"%{text.lower()}%")
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY COALESCE(r.start_sec, 0) ASC LIMIT ?"
            query_params = [*params, max(1, int(limit))]
        return [_row_to_dict(row) for row in conn.execute(sql, query_params).fetchall()]
    finally:
        conn.close()


def _write_sqlite_index(path: Path, rows: list[dict[str, Any]], *, material_root: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("DROP TABLE IF EXISTS key_material_refs_fts")
        conn.execute("DROP TABLE IF EXISTS key_material_refs")
        conn.execute("DROP TABLE IF EXISTS key_material_index_metadata")
        conn.execute(
            """
            CREATE TABLE key_material_refs (
                material_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                session_id TEXT,
                date TEXT,
                asset_type TEXT,
                asset_kind TEXT,
                action_name TEXT,
                display_name TEXT,
                segment_id TEXT,
                micro_segment_id TEXT,
                evidence_group_id TEXT,
                material_group_id TEXT,
                physical_action_material_id TEXT,
                evidence_window_id TEXT,
                dual_event_id TEXT,
                dual_view_action_event_id TEXT,
                dual_event_binding_source TEXT,
                formal_dual_view_action INTEGER,
                single_view_candidate INTEGER,
                view TEXT,
                frame_type TEXT,
                start_sec REAL,
                end_sec REAL,
                primary_object TEXT,
                primary_object_family TEXT,
                object_family TEXT,
                canonical_action_type TEXT,
                canonical_object TEXT,
                secondary_objects TEXT,
                secondary_actions TEXT,
                objects TEXT,
                actions TEXT,
                window_audit TEXT,
                target_object_support TEXT,
                secondary_object_support TEXT,
                uncertainty_reasons TEXT,
                review_status TEXT,
                candidate_status TEXT,
                formal_material_reference INTEGER,
                physical_evidence_mode TEXT,
                candidate_source TEXT,
                valid_yolo_evidence_count INTEGER,
                usable_yolo_evidence_count INTEGER,
                quality_score REAL,
                yolo_evidence_count INTEGER,
                physical_evidence_diagnostics TEXT,
                formal_publish_gate TEXT,
                evidence_chain TEXT,
                source_yolo_evidence TEXT,
                stored_file TEXT,
                source_file TEXT,
                path_mode TEXT,
                package_uri TEXT,
                file_name TEXT,
                "exists" INTEGER,
                size_bytes INTEGER,
                sha256 TEXT,
                searchable_text TEXT,
                created_at TEXT,
                payload_json TEXT
            )
            """
        )
        fts_enabled = True
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE key_material_refs_fts
                USING fts5(material_id UNINDEXED, searchable_text)
                """
            )
        except sqlite3.OperationalError:
            fts_enabled = False

        for row in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO key_material_refs
                (material_id, experiment_id, session_id, date, asset_type, asset_kind, action_name, display_name,
                 segment_id, micro_segment_id, evidence_group_id, material_group_id, physical_action_material_id,
                 evidence_window_id, dual_event_id, dual_view_action_event_id, dual_event_binding_source,
                 formal_dual_view_action, single_view_candidate, view, frame_type, start_sec, end_sec,
                 primary_object, primary_object_family, object_family, canonical_action_type, canonical_object, secondary_objects, secondary_actions,
                 objects, actions, window_audit, target_object_support, secondary_object_support,
                 uncertainty_reasons, review_status, candidate_status,
                 formal_material_reference, physical_evidence_mode, candidate_source, valid_yolo_evidence_count,
                 usable_yolo_evidence_count, quality_score, yolo_evidence_count, physical_evidence_diagnostics,
                 formal_publish_gate, evidence_chain, source_yolo_evidence, stored_file, source_file, path_mode, package_uri, file_name, "exists",
                 size_bytes, sha256, searchable_text, created_at, payload_json)
                VALUES
                (:material_id, :experiment_id, :session_id, :date, :asset_type, :asset_kind, :action_name, :display_name,
                 :segment_id, :micro_segment_id, :evidence_group_id, :material_group_id, :physical_action_material_id,
                 :evidence_window_id, :dual_event_id, :dual_view_action_event_id, :dual_event_binding_source,
                 :formal_dual_view_action_int, :single_view_candidate_int, :view, :frame_type, :start_sec, :end_sec,
                 :primary_object, :primary_object_family, :object_family, :canonical_action_type, :canonical_object, :secondary_objects_json, :secondary_actions_json,
                 :objects_json, :actions_json, :window_audit_json, :target_object_support_json, :secondary_object_support_json,
                 :uncertainty_reasons_json, :review_status, :candidate_status,
                 :formal_material_reference_int, :physical_evidence_mode, :candidate_source, :valid_yolo_evidence_count,
                 :usable_yolo_evidence_count, :quality_score, :yolo_evidence_count, :physical_evidence_diagnostics_json,
                 :formal_publish_gate_json, :evidence_chain_json, :source_yolo_evidence_json, :stored_file, :source_file, :path_mode, :package_uri, :file_name, :exists,
                 :size_bytes, :sha256, :searchable_text, :created_at, :payload_json)
                """,
                _sqlite_row(row),
            )
            if fts_enabled:
                conn.execute(
                    "INSERT INTO key_material_refs_fts(material_id, searchable_text) VALUES (?, ?)",
                    (row["material_id"], row.get("searchable_text") or ""),
                )

        conn.execute(
            """
            CREATE TABLE key_material_index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "material_root": str(material_root),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "row_count": str(len(rows)),
            "fts_enabled": str(bool(fts_enabled)).lower(),
        }
        for key, value in metadata.items():
            conn.execute("INSERT INTO key_material_index_metadata(key, value) VALUES (?, ?)", (key, value))
        conn.execute("CREATE INDEX idx_key_material_refs_asset_type ON key_material_refs(asset_type)")
        conn.execute("CREATE INDEX idx_key_material_refs_segment ON key_material_refs(segment_id, micro_segment_id)")
        conn.execute("CREATE INDEX idx_key_material_refs_evidence_group ON key_material_refs(evidence_group_id, material_group_id)")
        conn.execute("CREATE INDEX idx_key_material_refs_dual_event ON key_material_refs(dual_event_id)")
        conn.execute("CREATE INDEX idx_key_material_refs_object ON key_material_refs(primary_object)")
        conn.execute("CREATE INDEX idx_key_material_refs_session_date ON key_material_refs(session_id, date)")
        conn.execute("CREATE INDEX idx_key_material_refs_time ON key_material_refs(start_sec, end_sec)")
        conn.commit()
        return fts_enabled
    finally:
        conn.close()


def _asset_type(row: Mapping[str, Any]) -> str:
    kind = _string(row.get("asset_kind") or row.get("material_type")).strip()
    suffix = Path(_string(row.get("file_name") or row.get("stored_filename") or row.get("stored_file"))).suffix.lower()
    role_text = " ".join(
        _string(value)
        for value in (
            kind,
            row.get("role"),
            row.get("frame_type"),
            row.get("frame_role"),
            row.get("asset_type"),
        )
        if value
    ).lower()
    if kind == KEYFRAME_KIND or suffix in IMAGE_EXTENSIONS or "keyframe" in role_text:
        return "keyframe"
    if kind == KEY_CLIP_KIND or suffix in VIDEO_EXTENSIONS or "clip" in role_text:
        return "video_clip"
    if kind == REPORT_KIND or suffix in REPORT_EXTENSIONS or "report" in role_text:
        return "report"
    return _string(row.get("asset_type") or "unknown") or "unknown"


def _resolve_stored_path(row: Mapping[str, Any], root: Path) -> Path | None:
    raw = _string(row.get("stored_file") or row.get("stored_path") or row.get("file_path") or row.get("path"))
    candidates: list[Path] = []
    if raw:
        if raw.startswith("package://"):
            raw = raw.split("/", 3)[-1] if raw.count("/") >= 3 else Path(raw).name
        raw_path = Path(raw)
        candidates.append(raw_path if raw_path.is_absolute() else root / raw_path)
    file_name = _string(row.get("file_name") or row.get("stored_filename") or (Path(raw).name if raw else ""))
    kind = _string(row.get("asset_kind") or row.get("material_type"))
    if file_name:
        if kind:
            candidates.append(root / kind / file_name)
        candidates.append(root / file_name)
        candidates.extend(path for path in root.rglob(file_name) if path.is_file())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _path_relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        if path.is_absolute():
            return path.name
        return str(path).replace("\\", "/")


def _portable_path_value(value: Any, root: Path) -> str:
    text = _string(value)
    if not text:
        return ""
    if text.startswith("package://"):
        return text.split("/", 3)[-1] if text.count("/") >= 3 else Path(text).name
    return _path_relative_to_root(Path(text), root)


def _material_id(row: Mapping[str, Any], *, stored_path: Path | None, row_index: int) -> str:
    for key in ("material_id", "item_id", "candidate_id", "event_id"):
        value = _string(row.get(key))
        if value:
            return value
    raw = "|".join(
        [
            _string(row.get("experiment_id")),
            _string(row.get("segment_id") or row.get("parent_segment_id")),
            _string(row.get("micro_segment_id")),
            _string(stored_path),
            str(row_index),
        ]
    )
    return "material_ref_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _secondary_objects(row: Mapping[str, Any]) -> list[str]:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    primary = _norm_object(row.get("primary_object") or row.get("canonical_object") or row.get("object_label"))
    values = [
        *_list_strings(row.get("secondary_objects")),
        *_list_strings(interaction.get("secondary_objects")),
    ]
    excluded = {primary, "", *HAND_LABELS, "lab_coat", "ppe_storage"}
    return [item for item in _ordered_unique(_norm_object(value) for value in values) if item not in excluded]


def _secondary_actions(row: Mapping[str, Any], secondary_objects: list[str]) -> list[str]:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    primary = row.get("primary_object") or row.get("canonical_object") or row.get("object_label")
    raw = [
        *_list_strings(row.get("secondary_actions")),
        *_list_strings(interaction.get("secondary_actions")),
    ]
    actions = [_normalize_action(value) for value in raw]
    base_action = _normalize_action(row.get("canonical_action_type") or _canonical_hand_action(primary))
    secondary_action_objects: list[str] = []
    for obj in secondary_objects:
        actions.append(_canonical_hand_action(obj))
        secondary_action_objects.append(_canonical_action_object(obj).replace("_", "-"))
    if base_action and secondary_action_objects:
        actions.append(f"{base_action}+{'+'.join(secondary_action_objects)}")
    return _ordered_unique(action for action in actions if action)


def _objects(row: Mapping[str, Any], secondary_objects: list[str]) -> list[str]:
    values = [
        *_list_strings(row.get("objects")),
        row.get("primary_object"),
        row.get("canonical_object"),
        row.get("object_label"),
        *secondary_objects,
    ]
    return [item for item in _ordered_unique(_norm_object(value) for value in values) if item and item not in HAND_LABELS]


def _actions(row: Mapping[str, Any], secondary_actions: list[str]) -> list[str]:
    values = [
        *_list_strings(row.get("actions")),
        row.get("canonical_action_type"),
        row.get("action_name"),
        row.get("event_type"),
        *secondary_actions,
    ]
    return _ordered_unique(_string(value) for value in values if _string(value))


def _canonical_action_object(value: Any) -> str:
    label = _norm_object(value)
    return ACTION_OBJECT_BY_LABEL.get(label, label)


def _canonical_hand_action(value: Any) -> str:
    canonical = _canonical_action_object(value).replace("_", "-")
    return f"hand-{canonical}" if canonical else ""


def _normalize_action(value: Any) -> str:
    text = _string(value).lower().replace(" ", "_")
    if not text:
        return ""
    parts = [part.strip().replace("_", "-") for part in text.split("+") if part.strip()]
    return "+".join(parts)


def _norm_object(value: Any) -> str:
    return _string(value).lower().replace("-", "_").replace(" ", "_")


def _ordered_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = _string(value)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            ordered.append(text)
    return ordered


def _list_strings(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [_string(item) for item in value if _string(item)]
    return [_string(value)]


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sqlite_row(row: dict[str, Any]) -> dict[str, Any]:
    def nullable_bool_int(value: Any) -> int | None:
        if value is None:
            return None
        return int(bool(value))

    return {
        **row,
        "evidence_group_id": _string(row.get("evidence_group_id")),
        "material_group_id": _string(row.get("material_group_id")),
        "physical_action_material_id": _string(row.get("physical_action_material_id")),
        "evidence_window_id": _string(row.get("evidence_window_id")),
        "formal_dual_view_action_int": nullable_bool_int(row.get("formal_dual_view_action")),
        "single_view_candidate_int": nullable_bool_int(row.get("single_view_candidate")),
        "formal_material_reference_int": nullable_bool_int(row.get("formal_material_reference")),
        "exists": int(bool(row.get("exists"))),
        "secondary_objects_json": _json_text(row.get("secondary_objects") or []),
        "secondary_actions_json": _json_text(row.get("secondary_actions") or []),
        "objects_json": _json_text(row.get("objects") or []),
        "actions_json": _json_text(row.get("actions") or []),
        "window_audit_json": _json_text(row.get("window_audit") or {}),
        "target_object_support_json": _json_text(row.get("target_object_support") or {}),
        "secondary_object_support_json": _json_text(row.get("secondary_object_support") or []),
        "uncertainty_reasons_json": _json_text(row.get("uncertainty_reasons") or []),
        "physical_evidence_diagnostics_json": _json_text(row.get("physical_evidence_diagnostics") or {}),
        "formal_publish_gate_json": _json_text(row.get("formal_publish_gate") or {}),
        "evidence_chain_json": _json_text(row.get("evidence_chain") or {}),
        "source_yolo_evidence_json": _json_text(row.get("source_yolo_evidence") or []),
    }


def _searchable_text(row: Mapping[str, Any], *, stored_path: Path | None, asset_type: str) -> str:
    selected = [
        asset_type,
        stored_path.name if stored_path is not None else "",
        row.get("action_name"),
        row.get("display_name"),
        row.get("event_type"),
        row.get("primary_object"),
        row.get("object_label"),
        row.get("canonical_action_type"),
        row.get("canonical_object"),
        row.get("secondary_objects"),
        row.get("secondary_actions"),
        row.get("objects"),
        row.get("actions"),
        row.get("window_audit"),
        row.get("evidence_group_id"),
        row.get("material_group_id"),
        row.get("physical_action_material_id"),
        row.get("evidence_window_id"),
        row.get("target_object_support"),
        row.get("secondary_object_support"),
        row.get("uncertainty_reasons"),
        row.get("view"),
        row.get("frame_type"),
        row.get("role"),
        row.get("review_status"),
        row.get("candidate_status"),
        row.get("quality_reasons"),
        row.get("object_labels"),
        row.get("actions"),
        row.get("vlm_semantics"),
        row.get("yolo_recheck"),
        row.get("evidence_chain"),
    ]
    text = " ".join(_flatten_strings(selected))
    return re.sub(r"\s+", " ", text).strip()[:24000]


def _flatten_strings(values: Iterable[Any]) -> list[str]:
    parts: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, Mapping):
            for nested in value.values():
                add(nested)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                add(nested)
            return
        text = str(value).strip()
        if text:
            parts.append(text)

    for value in values:
        add(value)
    return parts


def _sha256(path: Path) -> str:
    cache_key = _file_cache_key(path)
    if cache_key is not None:
        with _STORED_FILE_SHA256_CACHE_LOCK:
            cached = _STORED_FILE_SHA256_CACHE.get(cache_key)
            if cached is not None:
                return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache_key is not None:
        with _STORED_FILE_SHA256_CACHE_LOCK:
            _STORED_FILE_SHA256_CACHE[cache_key] = value
    return value


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _material_reference_date(row: Mapping[str, Any], stored_path: Path | None) -> str:
    for key in ("date", "material_date", "capture_date", "session_date", "created_date"):
        parsed = _date_string(row.get(key))
        if parsed:
            return parsed
    for key in ("observed_at", "created_at", "approved_at", "session_start_time", "start_time"):
        parsed = _date_string(row.get(key))
        if parsed:
            return parsed
    text = " ".join(
        _string(value)
        for value in (
            row.get("file_name"),
            row.get("stored_file"),
            row.get("stored_filename"),
            row.get("source_file"),
            row.get("experiment_id"),
            stored_path,
        )
    )
    match = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    if stored_path is not None and stored_path.exists():
        return datetime.fromtimestamp(stored_path.stat().st_mtime, timezone.utc).date().isoformat()
    return ""


def _date_string(value: Any) -> str:
    text = _string(value)
    if not text:
        return ""
    match = re.match(r"^(20\d{2})[-_]?(\d{2})[-_]?(\d{2})$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ""


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _fts_query(text: str) -> str:
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", text, flags=re.UNICODE)
    return " ".join(tokens) if tokens else text.replace('"', '""')


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["exists"] = bool(item.get("exists"))
    for key in ("start_sec", "end_sec", "quality_score"):
        if item.get(key) is not None:
            item[key] = float(item[key])
    if item.get("yolo_evidence_count") is not None:
        item["yolo_evidence_count"] = int(item["yolo_evidence_count"])
    for key, default in (
        ("secondary_objects", []),
        ("secondary_actions", []),
        ("objects", []),
        ("actions", []),
        ("window_audit", {}),
        ("target_object_support", {}),
        ("secondary_object_support", []),
        ("uncertainty_reasons", []),
    ):
        item[key] = _json_value(item.get(key), default)
    return item


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return default
    return parsed


__all__ = [
    "DEFAULT_REFERENCES_NAME",
    "DEFAULT_SQLITE_NAME",
    "MATERIAL_INDEX_JSON",
    "MATERIAL_INDEX_JSONL",
    "SCHEMA_VERSION",
    "build_key_material_reference_index",
    "load_material_reference_rows",
    "normalize_material_reference",
    "query_key_material_reference_index",
]
