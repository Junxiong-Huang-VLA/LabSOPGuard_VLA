from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .material_reference_index import (
    DEFAULT_REFERENCES_NAME,
    DEFAULT_SQLITE_NAME,
    IMAGE_EXTENSIONS,
    MATERIAL_INDEX_JSON,
    MATERIAL_INDEX_JSONL,
    SCHEMA_VERSION as PACKAGE_SCHEMA_VERSION,
    VIDEO_EXTENSIONS,
    build_key_material_reference_index,
    load_material_reference_rows,
    normalize_material_reference,
)


GLOBAL_DB_NAME = "material_library.sqlite"
GLOBAL_SCHEMA_VERSION = "key_material_global_library.v1"
DEFAULT_LIBRARY_ROOT = Path("D:/LabMaterialLibrary")
_GLOBAL_FILE_REAL_CACHE: dict[tuple[str, int, int, str], bool] = {}
_GLOBAL_FILE_REAL_CACHE_LOCK = threading.Lock()
NON_REAL_ASSET_MARKERS = (
    "placeholder",
    "poster",
    "synthetic",
    "dry_run",
    "dry-run",
    "black_screen",
    "white_screen",
    "blank_screen",
    "黑屏",
    "白屏",
    "黑白屏",
)


def default_material_library_root() -> Path:
    configured = (
        os.environ.get("LAB_MATERIAL_LIBRARY_ROOT")
        or os.environ.get("KEY_ACTION_MATERIAL_LIBRARY_ROOT")
        or str(DEFAULT_LIBRARY_ROOT)
    )
    return Path(configured)


def material_references_root(library_root: str | Path | None = None) -> Path:
    root = Path(library_root) if library_root is not None else default_material_library_root()
    return root if root.name.lower() == "material_references" else root / "material_references"


def _global_material_library_worker_count() -> int:
    raw = os.environ.get("KEY_ACTION_MATERIAL_LIBRARY_WORKERS", "8")
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 8


def global_material_library_db_path(
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> Path:
    if sqlite_path is not None:
        return Path(sqlite_path)
    root = Path(library_root) if library_root is not None else default_material_library_root()
    if root.name.lower() == "material_references":
        root = root.parent
    return root / GLOBAL_DB_NAME


def iter_material_reference_packages(library_root: str | Path | None = None) -> Iterable[Path]:
    root = material_references_root(library_root)
    if _looks_like_package(root):
        yield root
        return
    if not root.exists():
        return
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_dir() and _looks_like_package(child):
            yield child


def sync_material_library(
    library_root: str | Path | None = None,
    *,
    sqlite_path: str | Path | None = None,
    rebuild: bool = False,
    include_reports: bool = False,
) -> dict[str, Any]:
    """Build or refresh the global material-reference catalog.

    The global catalog is a read-only sidecar over the published material
    packages. It does not mutate detection outputs or source videos.
    """

    root = Path(library_root) if library_root is not None else default_material_library_root()
    db_path = global_material_library_db_path(root, sqlite_path)
    packages = list(iter_material_reference_packages(root))
    rows: list[dict[str, Any]] = []
    package_summaries: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for package_root in packages:
        try:
            package_rows, package_summary = _load_package_rows(package_root, include_reports=include_reports)
        except Exception as exc:
            skipped.append({"package_root": str(package_root), "reason": str(exc)})
            continue
        for item in package_summary.get("skipped") or []:
            if isinstance(item, Mapping):
                skipped.append({"package_root": str(package_root), **dict(item)})
        normalized_rows: list[dict[str, Any]] = []
        for row_index, row in enumerate(package_rows, start=1):
            if not include_reports and row.get("asset_type") == "report":
                continue
            globalized = _globalize_material_row(row, package_root=package_root, library_root=root)
            if not _global_material_row_publishable(globalized):
                skipped.append(
                    {
                        "package_root": str(package_root),
                        "row_index": row_index,
                        "reason": globalized.get("missing_reason") or "non_real_material_not_indexed",
                        "stored_file": globalized.get("stored_file"),
                    }
                )
                continue
            normalized_rows.append(globalized)
        rows.extend(normalized_rows)
        package_summaries.append(
            {
                "package_name": package_root.name,
                "package_root": str(package_root),
                "source_rows": int(package_summary.get("source_rows") or len(package_rows)),
                "indexed_count": len(normalized_rows),
                "source": package_summary.get("source") or "package_references",
            }
        )

    fts_enabled = _write_global_index(db_path, rows, library_root=root, rebuild=rebuild)
    counts: dict[str, int] = {}
    missing_count = 0
    for row in rows:
        asset_type = str(row.get("asset_type") or "unknown")
        counts[asset_type] = counts.get(asset_type, 0) + 1
        if not row.get("exists"):
            missing_count += 1

    return {
        "schema_version": GLOBAL_SCHEMA_VERSION,
        "library_root": str(root),
        "material_references_root": str(material_references_root(root)),
        "sqlite_path": str(db_path),
        "package_count": len(packages),
        "indexed_count": len(rows),
        "missing_count": missing_count,
        "asset_type_counts": dict(sorted(counts.items())),
        "fts_enabled": fts_enabled,
        "packages": package_summaries,
        "skipped": skipped,
    }


def sync_material_library_package(
    package_root: str | Path,
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    include_reports: bool = False,
) -> dict[str, Any]:
    """Upsert one published material package into the global catalog."""

    package_path = Path(package_root)
    if library_root is None:
        inferred_root = _infer_library_root(package_path)
        root = inferred_root if inferred_root is not None else default_material_library_root()
    else:
        root = Path(library_root)
    package_rows, package_summary = _load_package_rows(package_path, include_reports=include_reports)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = [
        {"package_root": str(package_path), **dict(item)}
        for item in package_summary.get("skipped") or []
        if isinstance(item, Mapping)
    ]
    work_items = [(row_index, row) for row_index, row in enumerate(package_rows, start=1)]

    def _normalize_package_row(work_item: tuple[int, Mapping[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        row_index, row = work_item
        if not include_reports and row.get("asset_type") == "report":
            return None, None
        globalized = _globalize_material_row(row, package_root=package_path, library_root=root)
        if not _global_material_row_publishable(globalized):
            return None, {
                "package_root": str(package_path),
                "row_index": row_index,
                "reason": globalized.get("missing_reason") or "non_real_material_not_indexed",
                "stored_file": globalized.get("stored_file"),
            }
        return globalized, None

    workers = min(_global_material_library_worker_count(), max(1, len(work_items)))
    if workers <= 1 or len(work_items) <= 1:
        normalized_results = [_normalize_package_row(item) for item in work_items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            normalized_results = list(executor.map(_normalize_package_row, work_items))
    for globalized, skipped_item in normalized_results:
        if skipped_item is not None:
            skipped.append(skipped_item)
        if globalized is not None:
            rows.append(globalized)
    db_path = global_material_library_db_path(root, sqlite_path)
    fts_enabled = _write_global_index(
        db_path,
        rows,
        library_root=root,
        rebuild=False,
        replace_package_names=[package_path.name],
    )
    counts: dict[str, int] = {}
    missing_count = 0
    for row in rows:
        asset_type = str(row.get("asset_type") or "unknown")
        counts[asset_type] = counts.get(asset_type, 0) + 1
        if not row.get("exists"):
            missing_count += 1
    return {
        "schema_version": GLOBAL_SCHEMA_VERSION,
        "library_root": str(root),
        "material_references_root": str(material_references_root(root)),
        "sqlite_path": str(db_path),
        "package_name": package_path.name,
        "package_root": str(package_path),
        "source_rows": int(package_summary.get("source_rows") or len(package_rows)),
        "indexed_count": len(rows),
        "missing_count": missing_count,
        "asset_type_counts": dict(sorted(counts.items())),
        "fts_enabled": fts_enabled,
        "source": package_summary.get("source") or "package_references",
        "skipped": skipped,
    }


def query_material_library(
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
    text: str = "",
    asset_type: str | None = None,
    primary_object: str | None = None,
    action: str | None = None,
    view: str | None = None,
    session_id: str | None = None,
    experiment_id: str | None = None,
    package_name: str | None = None,
    date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    db_path = global_material_library_db_path(library_root, sqlite_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Global material library index not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        if _table_exists(conn, "material_refs"):
            _ensure_columns(conn, "material_refs", {"session_id": "TEXT", "date": "TEXT"})
        clauses: list[str] = []
        params: list[Any] = []
        if asset_type:
            clauses.append("asset_type = ?")
            params.append(asset_type)
        if view:
            clauses.append("view = ?")
            params.append(view)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if experiment_id:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if package_name:
            clauses.append("package_name = ?")
            params.append(package_name)
        if date:
            clauses.append("date = ?")
            params.append(date)
        if start_date:
            clauses.append("date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("date <= ?")
            params.append(end_date)
        if primary_object:
            object_param = f"%{primary_object.lower()}%"
            clauses.append(
                "(LOWER(COALESCE(primary_object, '')) LIKE ? OR LOWER(COALESCE(canonical_object, '')) LIKE ? "
                "OR LOWER(COALESCE(secondary_objects, '')) LIKE ? OR LOWER(COALESCE(objects, '')) LIKE ?)"
            )
            params.extend([object_param, object_param, object_param, object_param])
        if action:
            action_param = f"%{action.lower()}%"
            clauses.append(
                "(LOWER(COALESCE(action_name, '')) LIKE ? OR LOWER(COALESCE(canonical_action_type, '')) LIKE ? "
                "OR LOWER(COALESCE(secondary_actions, '')) LIKE ? OR LOWER(COALESCE(actions, '')) LIKE ? "
                "OR LOWER(COALESCE(searchable_text, '')) LIKE ?)"
            )
            params.extend([action_param, action_param, action_param, action_param, action_param])
        for token in _search_tokens(text):
            text_param = f"%{token.lower()}%"
            clauses.append(
                "(LOWER(COALESCE(searchable_text, '')) LIKE ? OR LOWER(COALESCE(display_name, '')) LIKE ? "
                "OR LOWER(COALESCE(experiment_title, '')) LIKE ? OR LOWER(COALESCE(primary_object, '')) LIKE ? "
                "OR LOWER(COALESCE(action_name, '')) LIKE ?)"
            )
            params.extend([text_param, text_param, text_param, text_param, text_param])

        sql = "SELECT *, 0.0 AS rank FROM material_refs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY COALESCE(start_sec, 0) ASC, package_name ASC LIMIT ?"
        params.append(max(1, int(limit)))
        return [_row_to_dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def get_material_reference(
    material_id: str,
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> dict[str, Any] | None:
    db_path = global_material_library_db_path(library_root, sqlite_path)
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM material_refs WHERE material_id = ?", (material_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None
    finally:
        conn.close()


def resolve_material_file(
    material_id: str,
    *,
    library_root: str | Path | None = None,
    sqlite_path: str | Path | None = None,
) -> Path:
    row = get_material_reference(material_id, library_root=library_root, sqlite_path=sqlite_path)
    if row is None:
        raise FileNotFoundError(f"Material reference not found: {material_id}")
    absolute_path = Path(str(row.get("absolute_path") or ""))
    root = Path(library_root) if library_root is not None else default_material_library_root()
    root = root if root.name.lower() != "material_references" else root.parent
    if not absolute_path.exists() or not absolute_path.is_file():
        raise FileNotFoundError(f"Material file not found: {absolute_path}")
    try:
        absolute_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise PermissionError(f"Material file is outside library root: {absolute_path}") from exc
    return absolute_path


def _looks_like_package(path: Path) -> bool:
    return any((path / name).exists() for name in (DEFAULT_REFERENCES_NAME, MATERIAL_INDEX_JSONL, MATERIAL_INDEX_JSON, DEFAULT_SQLITE_NAME, "manifest.json"))


def _global_material_file_is_real(path: Path | None, row: Mapping[str, Any]) -> bool:
    if path is None:
        return False
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        stat = path.stat()
        cache_key = (
            str(path.resolve()).lower(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            str(row.get("missing_reason") or row.get("candidate_source") or row.get("source_type") or ""),
        )
        with _GLOBAL_FILE_REAL_CACHE_LOCK:
            cached = _GLOBAL_FILE_REAL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        marker_text = " ".join(
            str(value or "").lower()
            for value in (
                path.name,
                Path(str(row.get("stored_file") or "")).name,
                row.get("file_name"),
                row.get("stored_filename"),
                row.get("candidate_source"),
                row.get("source_type"),
                row.get("missing_reason"),
            )
        )
        if any(marker in marker_text for marker in NON_REAL_ASSET_MARKERS):
            return False
        header = path.read_bytes()[:128].upper()
    except OSError:
        return False
    if header.startswith(b"DRY RUN") or b"PLACEHOLDER" in header or b"SYNTHETIC" in header:
        result = False
    else:
        result = _global_material_visual_content_is_real(path)
    with _GLOBAL_FILE_REAL_CACHE_LOCK:
        _GLOBAL_FILE_REAL_CACHE[cache_key] = result
    return result


def _global_material_visual_content_is_real(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return _global_image_visual_content_is_real(path)
    if suffix in VIDEO_EXTENSIONS:
        return _global_video_visual_content_is_real(path)
    return True


def _global_image_visual_content_is_real(path: Path) -> bool:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        image.thumbnail((128, 128))
        pixels = list(image.getdata())
    except Exception:
        return True
    return _rgb_pixels_are_real_material(pixels)


def _global_video_visual_content_is_real(path: Path) -> bool:
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


def _load_package_rows(package_root: Path, *, include_reports: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    references_path = package_root / DEFAULT_REFERENCES_NAME
    if references_path.exists() and references_path.stat().st_size > 0:
        rows = _read_jsonl(references_path)
        return rows, {"source": DEFAULT_REFERENCES_NAME, "source_rows": len(rows)}

    summary = build_key_material_reference_index(package_root, include_reports=include_reports)
    rows = _read_jsonl(package_root / DEFAULT_REFERENCES_NAME)
    return rows, {**summary, "source": "rebuilt_package_index"}


def _globalize_material_row(row: Mapping[str, Any], *, package_root: Path, library_root: Path) -> dict[str, Any]:
    item = dict(row)
    if item.get("schema_version") != PACKAGE_SCHEMA_VERSION or not item.get("stored_file"):
        item = normalize_material_reference(item, package_root)

    source_material_id = str(item.get("material_id") or "")
    global_material_id = _global_material_id(package_root.name, source_material_id)
    stored_file = str(item.get("stored_file") or "")
    absolute_path = _absolute_material_path(package_root, stored_file)
    manifest = _load_manifest(package_root)
    experiment_id = str(item.get("experiment_id") or manifest.get("experiment_id") or manifest.get("id") or "")
    session_id = str(
        item.get("session_id")
        or manifest.get("session_id")
        or manifest.get("run_id")
        or manifest.get("session")
        or ""
    )
    material_date = _material_date(item, package_root=package_root, absolute_path=absolute_path, manifest=manifest)
    experiment_title = str(
        manifest.get("experiment_title")
        or manifest.get("title")
        or manifest.get("experiment_name")
        or package_root.name
    )
    exists = absolute_path.exists() and absolute_path.is_file()
    source_real = bool(
        item.get("source_real") is not False
        and not item.get("placeholder")
        and _global_material_file_is_real(absolute_path, item)
    )
    return {
        **item,
        "material_id": global_material_id,
        "source_material_id": source_material_id,
        "experiment_id": experiment_id,
        "session_id": session_id,
        "date": material_date,
        "experiment_title": experiment_title,
        "package_name": package_root.name,
        "package_root": str(package_root),
        "evidence_group_id": str(item.get("evidence_group_id") or ""),
        "material_group_id": str(item.get("material_group_id") or ""),
        "physical_action_material_id": str(item.get("physical_action_material_id") or ""),
        "evidence_window_id": str(item.get("evidence_window_id") or ""),
        "absolute_path": str(absolute_path),
        "stored_file": stored_file,
        "package_uri": f"package://material-library/{package_root.name}/{stored_file}" if stored_file else "",
        "exists": bool(exists),
        "source_real": source_real,
        "placeholder": not source_real,
        "publishable_material": source_real,
        "missing_reason": None if source_real else str(item.get("missing_reason") or "non_real_or_missing_material_file"),
        "size_bytes": int(absolute_path.stat().st_size) if exists else int(item.get("size_bytes") or 0),
        "created_at": _global_material_created_at(item, material_date),
        "payload_json": item.get("payload_json") or json.dumps(dict(row), ensure_ascii=False, sort_keys=True),
        "searchable_text": _global_searchable_text(item, package_root.name, experiment_title),
    }


def _global_material_row_publishable(row: Mapping[str, Any]) -> bool:
    if row.get("source_real") is not True or row.get("placeholder") is True:
        return False
    if not row.get("exists"):
        return False
    text = " ".join(
        str(value or "").lower()
        for value in (
            Path(str(row.get("stored_file") or "")).name,
            row.get("file_name"),
            Path(str(row.get("source_file") or "")).name,
            row.get("candidate_source"),
            row.get("source_type"),
            row.get("missing_reason"),
        )
    )
    return not any(marker in text for marker in NON_REAL_ASSET_MARKERS)


def _global_material_id(package_name: str, source_material_id: str) -> str:
    raw = f"{package_name}|{source_material_id}"
    return "global_material_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _infer_library_root(package_root: Path) -> Path | None:
    for parent in package_root.resolve().parents:
        if parent.name.lower() == "material_references":
            return parent.parent
    return None


def _search_tokens(text: str) -> list[str]:
    value = str(text or "").strip()
    if not value:
        return []
    tokens = [token for token in re.split(r"\s+", value) if token]
    return tokens or [value]


def _absolute_material_path(package_root: Path, stored_file: str) -> Path:
    raw = stored_file
    if raw.startswith("package://"):
        raw = raw.split("/", 3)[-1] if raw.count("/") >= 3 else Path(raw).name
    path = Path(raw)
    return path if path.is_absolute() else package_root / path


def _global_searchable_text(item: Mapping[str, Any], package_name: str, experiment_title: str) -> str:
    values = [
        package_name,
        experiment_title,
        item.get("session_id"),
        item.get("date"),
        item.get("evidence_group_id"),
        item.get("material_group_id"),
        item.get("physical_action_material_id"),
        item.get("evidence_window_id"),
        item.get("searchable_text"),
        item.get("display_name"),
        item.get("action_name"),
        item.get("asset_type"),
        item.get("asset_kind"),
        item.get("primary_object"),
        item.get("canonical_action_type"),
        item.get("canonical_object"),
        item.get("secondary_objects"),
        item.get("secondary_actions"),
        item.get("objects"),
        item.get("actions"),
        item.get("view"),
        item.get("file_name"),
    ]
    return " ".join(_flatten_strings(values))[:24000]


def _material_date(
    item: Mapping[str, Any],
    *,
    package_root: Path,
    absolute_path: Path,
    manifest: Mapping[str, Any],
) -> str:
    for source in (item, manifest):
        for key in ("date", "material_date", "capture_date", "session_date", "created_date"):
            parsed = _date_string(source.get(key))
            if parsed:
                return parsed
        for key in ("observed_at", "created_at", "approved_at", "session_start_time", "start_time"):
            parsed = _date_string(source.get(key))
            if parsed:
                return parsed
    text = " ".join(
        str(value or "")
        for value in (
            item.get("file_name"),
            item.get("stored_file"),
            item.get("source_file"),
            item.get("experiment_id"),
            package_root.name,
            absolute_path,
        )
    )
    match = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    if absolute_path.exists():
        return datetime.fromtimestamp(absolute_path.stat().st_mtime, timezone.utc).date().isoformat()
    return ""


def _global_material_created_at(item: Mapping[str, Any], material_date: str) -> str:
    if material_date:
        return f"{material_date}T00:00:00+00:00"
    return str(item.get("created_at") or datetime.now(timezone.utc).isoformat())


def _date_string(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.match(r"^(20\d{2})[-_]?(\d{2})[-_]?(\d{2})$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return ""


def _ensure_columns(conn: sqlite3.Connection, table: str, columns: Mapping[str, str]) -> None:
    existing = {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for name, column_type in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {column_type}")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _write_global_index(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    library_root: Path,
    rebuild: bool,
    replace_package_names: Iterable[str] | None = None,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        if rebuild:
            conn.execute("DROP TABLE IF EXISTS material_refs_fts")
            conn.execute("DROP TABLE IF EXISTS material_refs")
            conn.execute("DROP TABLE IF EXISTS material_library_metadata")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS material_refs (
                material_id TEXT PRIMARY KEY,
                source_material_id TEXT,
                experiment_id TEXT,
                session_id TEXT,
                date TEXT,
                experiment_title TEXT,
                package_name TEXT,
                package_root TEXT,
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
                view TEXT,
                frame_type TEXT,
                start_sec REAL,
                end_sec REAL,
                primary_object TEXT,
                canonical_action_type TEXT,
                canonical_object TEXT,
                secondary_objects TEXT,
                secondary_actions TEXT,
                objects TEXT,
                actions TEXT,
                review_status TEXT,
                candidate_status TEXT,
                quality_score REAL,
                yolo_evidence_count INTEGER,
                stored_file TEXT,
                source_file TEXT,
                package_uri TEXT,
                file_name TEXT,
                absolute_path TEXT,
                "exists" INTEGER,
                size_bytes INTEGER,
                sha256 TEXT,
                searchable_text TEXT,
                created_at TEXT,
                payload_json TEXT
            )
            """
        )
        _ensure_columns(
            conn,
            "material_refs",
            {
                "session_id": "TEXT",
                "date": "TEXT",
                "evidence_group_id": "TEXT",
                "material_group_id": "TEXT",
                "physical_action_material_id": "TEXT",
                "evidence_window_id": "TEXT",
            },
        )
        fts_enabled = True
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS material_refs_fts
                USING fts5(material_id UNINDEXED, searchable_text, display_name, experiment_title, primary_object, action_name)
                """
            )
        except sqlite3.OperationalError:
            fts_enabled = False

        if rebuild:
            conn.execute("DELETE FROM material_refs")
            if fts_enabled:
                conn.execute("DELETE FROM material_refs_fts")
        elif replace_package_names:
            for package_name in replace_package_names:
                ids = [
                    str(row[0])
                    for row in conn.execute(
                        "SELECT material_id FROM material_refs WHERE package_name = ?",
                        (str(package_name),),
                    ).fetchall()
                ]
                conn.execute("DELETE FROM material_refs WHERE package_name = ?", (str(package_name),))
                if fts_enabled:
                    conn.executemany(
                        "DELETE FROM material_refs_fts WHERE material_id = ?",
                        [(material_id,) for material_id in ids],
                    )

        sqlite_rows = [_sqlite_row(row) for row in rows]
        conn.executemany(
            """
            INSERT OR REPLACE INTO material_refs
            (material_id, source_material_id, experiment_id, session_id, date, experiment_title, package_name, package_root,
             asset_type, asset_kind, action_name, display_name, segment_id, micro_segment_id,
             evidence_group_id, material_group_id, physical_action_material_id, evidence_window_id, view, frame_type,
             start_sec, end_sec, primary_object, canonical_action_type, canonical_object,
             secondary_objects, secondary_actions, objects, actions, review_status, candidate_status,
             quality_score, yolo_evidence_count, stored_file, source_file, package_uri, file_name, absolute_path,
             "exists", size_bytes, sha256, searchable_text, created_at, payload_json)
            VALUES
            (:material_id, :source_material_id, :experiment_id, :session_id, :date, :experiment_title, :package_name, :package_root,
             :asset_type, :asset_kind, :action_name, :display_name, :segment_id, :micro_segment_id,
             :evidence_group_id, :material_group_id, :physical_action_material_id, :evidence_window_id, :view, :frame_type,
             :start_sec, :end_sec, :primary_object, :canonical_action_type, :canonical_object,
             :secondary_objects_json, :secondary_actions_json, :objects_json, :actions_json, :review_status, :candidate_status,
             :quality_score, :yolo_evidence_count, :stored_file, :source_file, :package_uri, :file_name, :absolute_path,
             :exists, :size_bytes, :sha256, :searchable_text, :created_at, :payload_json)
            """,
            sqlite_rows,
        )
        if fts_enabled and rows:
            conn.executemany(
                "DELETE FROM material_refs_fts WHERE material_id = ?",
                [(row["material_id"],) for row in rows],
            )
            conn.executemany(
                """
                INSERT INTO material_refs_fts(material_id, searchable_text, display_name, experiment_title, primary_object, action_name)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        row.get("material_id"),
                        row.get("searchable_text") or "",
                        row.get("display_name") or "",
                        row.get("experiment_title") or "",
                        row.get("primary_object") or "",
                        row.get("action_name") or "",
                    )
                    for row in rows
                ],
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS material_library_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        metadata = {
            "schema_version": GLOBAL_SCHEMA_VERSION,
            "library_root": str(library_root),
            "material_references_root": str(material_references_root(library_root)),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "row_count": str(len(rows)),
            "fts_enabled": str(bool(fts_enabled)).lower(),
        }
        for key, value in metadata.items():
            conn.execute("INSERT OR REPLACE INTO material_library_metadata(key, value) VALUES (?, ?)", (key, value))
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_asset_type ON material_refs(asset_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_evidence_group ON material_refs(evidence_group_id, material_group_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_object ON material_refs(primary_object)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_action ON material_refs(action_name, canonical_action_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_view ON material_refs(view)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_session_date ON material_refs(session_id, date)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_time ON material_refs(start_sec, end_sec)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_material_refs_package ON material_refs(package_name)")
        conn.commit()
        return fts_enabled
    finally:
        conn.close()


def _sqlite_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **dict(row),
        "evidence_group_id": str(row.get("evidence_group_id") or ""),
        "material_group_id": str(row.get("material_group_id") or ""),
        "physical_action_material_id": str(row.get("physical_action_material_id") or ""),
        "evidence_window_id": str(row.get("evidence_window_id") or ""),
        "exists": int(bool(row.get("exists"))),
        "secondary_objects_json": _json_text(row.get("secondary_objects") or []),
        "secondary_actions_json": _json_text(row.get("secondary_actions") or []),
        "objects_json": _json_text(row.get("objects") or []),
        "actions_json": _json_text(row.get("actions") or []),
    }


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["exists"] = bool(item.get("exists"))
    for key in ("start_sec", "end_sec", "quality_score", "rank"):
        if item.get(key) is not None:
            item[key] = float(item[key])
    for key in ("yolo_evidence_count", "size_bytes"):
        if item.get(key) is not None:
            item[key] = int(item[key])
    for key in ("secondary_objects", "secondary_actions", "objects", "actions"):
        item[key] = _json_value(item.get(key), [])
    return item


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _load_manifest(package_root: Path) -> dict[str, Any]:
    for path in (package_root / "manifest.json", package_root / "evidence_package_manifest.json"):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


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


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_value(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


__all__ = [
    "GLOBAL_DB_NAME",
    "GLOBAL_SCHEMA_VERSION",
    "default_material_library_root",
    "get_material_reference",
    "global_material_library_db_path",
    "iter_material_reference_packages",
    "material_references_root",
    "query_material_library",
    "resolve_material_file",
    "sync_material_library",
    "sync_material_library_package",
]
