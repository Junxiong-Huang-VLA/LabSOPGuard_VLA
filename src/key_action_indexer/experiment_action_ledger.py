from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MATERIAL_ROOT = Path(r"D:\LabMaterialLibrary")
DEFAULT_LABVIDEO_ROOT = Path(r"D:\LabVideo")
LOCK_SUFFIX = ".lock"

ACTION_DISPLAY_NAMES = {
    "hand_object_contact": "手部与物体接触",
    "hand_object_interaction": "手部与物体接触",
    "object_move": "物体移动",
    "liquid_transfer": "液体转移",
    "liquid_movement": "液体移动",
    "device_panel_interaction": "设备面板操作",
    "equipment_panel_operation": "设备面板操作",
    "panel_operation": "设备面板操作",
    "balance_operation": "天平面板操作",
    "weighing_paper_operation": "称量纸操作",
    "pipette_operation": "移液枪操作",
    "container_state_change": "容器状态变化",
    "container_operation": "容器操作",
    "reagent_bottle_operation": "试剂瓶操作",
}

OBJECT_DISPLAY_NAMES = {
    "balance": "天平",
    "beaker": "烧杯",
    "bottle": "试剂瓶",
    "container": "容器",
    "paper": "称量纸",
    "pipette": "移液枪",
    "reagent_bottle": "试剂瓶",
    "spatula": "药匙",
    "tube": "试管",
    "weighing_paper": "称量纸",
}


def _utc_now() -> str:
    return datetime.now().astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                rows.append(data)
    except OSError:
        return []
    return rows


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        Path(tmp_name).replace(path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
    )


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _with_lock(lock_path: Path, fn: Any) -> Any:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{os.getpid()} {time.time()}\n")
            break
        except FileExistsError:
            if time.monotonic() - started > 30:
                try:
                    lock_path.unlink()
                except OSError:
                    pass
            time.sleep(0.05)
    try:
        return fn()
    finally:
        try:
            lock_path.unlink()
        except OSError:
            pass


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
    payload = json.dumps(dict(row), ensure_ascii=False, sort_keys=True, default=str)
    return "material_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _evidence_bundle_id(row: Mapping[str, Any]) -> str:
    return str(row.get("evidence_bundle_id") or row.get("dual_event_id") or _material_id(row))


def _first_text(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            text = str(value).strip()
            if text:
                return text
    return ""


def _action_type(row: Mapping[str, Any]) -> str:
    return str(
        row.get("action_type")
        or row.get("physical_action_type")
        or row.get("canonical_action_type")
        or row.get("action_name")
        or "unknown"
    )


def _window_id(row: Mapping[str, Any]) -> str:
    return _first_text(row.get("window_id"), row.get("experiment_window_id"), row.get("segment_id"), row.get("parent_segment_id"))


def _source_window_sync_index(row: Mapping[str, Any]) -> str:
    return _first_text(row.get("source_window_sync_index"), row.get("window_sync_index"))


def _is_orphan_material(row: Mapping[str, Any]) -> bool:
    if row.get("orphan_material") is True:
        return True
    if _official_status(row) == "official":
        return False
    return not _window_id(row) or not _source_window_sync_index(row)


def _timestamp_label(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if numeric <= 0:
        return ""
    # Large values are epoch microseconds; small values are session seconds.
    if numeric > 1_000_000_000_000:
        try:
            return datetime.fromtimestamp(numeric / 1_000_000).astimezone().strftime("%H%M%S")
        except (OSError, OverflowError, ValueError):
            return ""
    return f"{int(numeric):06d}"


def _display_name(row: Mapping[str, Any]) -> str:
    explicit = _first_text(row.get("display_name"), row.get("display_title"), row.get("action_name"))
    if explicit and not explicit.startswith(("event_candidate_group_", "view_action_review_bundle_")):
        return explicit
    action = ACTION_DISPLAY_NAMES.get(_action_type(row), _action_type(row))
    objects = _object_refs(row)
    object_name = OBJECT_DISPLAY_NAMES.get(objects[0], objects[0]) if objects else ""
    ts = _timestamp_label(_timestamp_value(row))
    parts = [part for part in (action, object_name, ts) if part]
    return "_".join(parts) if parts else _material_id(row)


def _official_status(row: Mapping[str, Any]) -> str:
    raw = str(row.get("official_status") or row.get("candidate_status") or row.get("review_status") or "").lower()
    if row.get("official_material") is True or raw in {"official", "approved", "accepted"}:
        return "official"
    if raw in {"rejected", "false_positive", "misclassified"}:
        return "rejected"
    return "needs_review"


def _active_review_status(row: Mapping[str, Any]) -> bool:
    raw = str(row.get("review_status") or row.get("candidate_status") or row.get("official_status") or "").lower()
    if raw in {"upgraded", "upgraded_to_official", "confirmed_official", "official", "approved", "accepted"}:
        return False
    return _official_status(row) == "needs_review"


def _object_refs(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    raw = row.get("object_refs")
    if isinstance(raw, list):
        values.extend(str(item) for item in raw if item)
    for key in ("primary_object", "object_label", "target_label", "raw_yolo_label"):
        value = row.get(key)
        if value:
            values.append(str(value))
    return sorted(set(values))


def _instrument_refs(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("instrument_refs")
    values: list[str] = []
    if isinstance(raw, list):
        values.extend(str(item) for item in raw if item)
    for value in _object_refs(row):
        if value in {"balance", "scale", "device_panel", "panel", "pipette"}:
            values.append(value)
    return sorted(set(values))


def _timestamp_value(row: Mapping[str, Any]) -> Any:
    for key in ("global_timestamp_us", "start_global_timestamp_us", "timestamp", "start_sec"):
        if row.get(key) not in (None, ""):
            return row.get(key)
    return None


def _keyframe_paths(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("first_keyframe", "third_keyframe", "keyframe", "stored_file"):
        value = row.get(key)
        if value and str(value).lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
            values.append(str(value))
    return sorted(set(values))


def _keyclip_paths(row: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("first_keyclip", "third_keyclip", "side_by_side_keyclip", "keyclip", "stored_file"):
        value = row.get(key)
        if value and str(value).lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
            values.append(str(value))
    return sorted(set(values))


def _load_experiment_material_rows(experiment_root: Path) -> dict[str, list[dict[str, Any]]]:
    stream = _read_jsonl(experiment_root / "material_stream.jsonl")
    official = _read_jsonl(experiment_root / "official_materials.jsonl")
    review = _read_jsonl(experiment_root / "review_candidate_materials.jsonl")
    legacy_rows = [
        *_read_jsonl(experiment_root / "key_material_references.jsonl"),
        *_read_jsonl(experiment_root / "素材索引.jsonl"),
    ]
    legacy_review_rows = [
        {
            **row,
            "official_status": "needs_review",
            "candidate_status": "legacy_unvalidated",
            "review_status": "legacy_unvalidated",
            "official_material": False,
            "memory_eligible": False,
            "memory_write_allowed": False,
            "review_reason": row.get("review_reason") or "legacy_unvalidated_no_window_visual_validation",
        }
        for row in legacy_rows
    ]
    if not stream and legacy_review_rows:
        stream = legacy_review_rows
    if not review and legacy_review_rows:
        review = legacy_review_rows
    return {"stream": stream, "official": official, "review": review}


def _stream_rows_from_asset_rows(rows: Sequence[Mapping[str, Any]], *, status: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_material_id(row)].append(row)
    stream: list[dict[str, Any]] = []
    for material_id, group in grouped.items():
        representative = group[0]
        keyframes: list[str] = []
        keyclips: list[str] = []
        views = sorted({str(row.get("view") or "") for row in group if row.get("view")})
        for row in group:
            keyframes.extend(_keyframe_paths(row))
            keyclips.extend(_keyclip_paths(row))
        stream.append(
            {
                "schema_version": "time_anchored_material_stream.v1",
                "material_id": material_id,
                "evidence_bundle_id": _evidence_bundle_id(representative),
                "action_event_id": representative.get("action_event_id") or representative.get("source_event_id"),
                "action_type": _action_type(representative),
                "official_status": status,
                "experiment_id": representative.get("experiment_id"),
                "experiment_window_id": representative.get("experiment_window_id") or representative.get("segment_id"),
                "global_timestamp_us": _timestamp_value(representative),
                "first_keyframe": next((path for path in keyframes if "first" in path.lower()), None),
                "third_keyframe": next((path for path in keyframes if "third" in path.lower()), None),
                "first_keyclip": next((path for path in keyclips if "first" in path.lower()), None),
                "third_keyclip": next((path for path in keyclips if "third" in path.lower()), None),
                "keyframe_paths": sorted(set(keyframes)),
                "keyclip_paths": sorted(set(keyclips)),
                "object_refs": _object_refs(representative),
                "instrument_refs": _instrument_refs(representative),
                "views": views,
                "confidence": representative.get("confidence") or representative.get("quality_score"),
                "memory_eligible": status == "official",
                "cli_ready_folder": representative.get("cli_ready_folder"),
                "frontend_item_id": representative.get("frontend_item_id") or representative.get("candidate_id"),
                "review_status": representative.get("review_status"),
            }
        )
    return stream


def normalize_material_stream(experiment_root: Path) -> list[dict[str, Any]]:
    rows = _load_experiment_material_rows(experiment_root)
    stream = list(rows["stream"])
    if not stream:
        stream.extend(_stream_rows_from_asset_rows(rows["official"], status="official"))
        stream.extend(_stream_rows_from_asset_rows(rows["review"], status="needs_review"))
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in stream:
        material_id = _material_id(row)
        if material_id in seen:
            continue
        seen.add(material_id)
        status = _official_status(row)
        window_id = _window_id(row)
        source_window_sync_index = _source_window_sync_index(row)
        orphan_material = _is_orphan_material(row)
        normalized.append(
            {
                **row,
                "material_id": material_id,
                "evidence_bundle_id": _evidence_bundle_id(row),
                "action_type": _action_type(row),
                "display_name": _display_name(row),
                "official_status": status,
                "window_id": window_id or row.get("window_id"),
                "experiment_window_id": _first_text(row.get("experiment_window_id"), window_id) or row.get("experiment_window_id"),
                "source_window_sync_index": source_window_sync_index or row.get("source_window_sync_index"),
                "orphan_material": orphan_material,
                "diagnostic_status": "orphan_material" if orphan_material else row.get("diagnostic_status"),
                "memory_eligible": bool(row.get("memory_eligible") or row.get("memory_write_allowed"))
                and status == "official",
                "object_refs": _object_refs(row),
                "instrument_refs": _instrument_refs(row),
                "keyframe_paths": _keyframe_paths(row) or row.get("keyframe_paths") or [],
                "keyclip_paths": _keyclip_paths(row) or row.get("keyclip_paths") or [],
                "timestamp": _timestamp_value(row),
            }
        )
    return normalized


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    return Counter(_official_status(row) for row in rows)


def _summarize_by_key(rows: Sequence[Mapping[str, Any]], key_fn: Any) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for row in rows:
        values = key_fn(row)
        if isinstance(values, list):
            for value in values:
                if value:
                    counter[str(value)] += 1
        elif values:
            counter[str(values)] += 1
    return [{"name": key, "count": count} for key, count in counter.most_common()]


def _representative_evidence(rows: Sequence[Mapping[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in rows[:limit]:
        evidence.append(
            {
                "material_id": row.get("material_id"),
                "evidence_bundle_id": row.get("evidence_bundle_id"),
                "action_type": row.get("action_type"),
                "official_status": row.get("official_status"),
                "display_name": row.get("display_name") or _display_name(row),
                "timestamp": row.get("timestamp") or _timestamp_value(row),
                "keyframe_refs": row.get("keyframe_paths") or _keyframe_paths(row),
                "keyclip_refs": row.get("keyclip_paths") or _keyclip_paths(row),
                "source_window_sync_index": row.get("source_window_sync_index"),
                "orphan_material": bool(row.get("orphan_material")),
                "cli_ready_folder": row.get("cli_ready_folder"),
            }
        )
    return evidence


def build_experiment_action_ledger(
    material_root: str | Path,
    experiment_id: str,
    *,
    source_labvideo_path: str | Path | None = None,
) -> dict[str, Any]:
    material_root = Path(material_root)
    experiment_root = material_root / experiment_id
    if not experiment_root.exists() and (material_root / "material_references" / experiment_id).exists():
        experiment_root = material_root / "material_references" / experiment_id
    raw_material_rows = _load_experiment_material_rows(experiment_root)
    rows = normalize_material_stream(experiment_root)
    status_counts = _status_counts(rows)
    official_rows = [row for row in rows if _official_status(row) == "official"]
    review_rows = [row for row in rows if _official_status(row) == "needs_review"]
    rejected_rows = [row for row in rows if _official_status(row) == "rejected"]
    official_asset_count = len(raw_material_rows["official"]) or len(official_rows)
    active_raw_review_rows = [row for row in raw_material_rows["review"] if _active_review_status(row)]
    review_asset_count = len(active_raw_review_rows) if raw_material_rows["review"] else len(review_rows)
    if official_rows and review_rows:
        ledger_status = "mixed"
    elif official_rows:
        ledger_status = "official_ready"
    elif review_rows:
        ledger_status = "needs_review_only"
    elif rejected_rows:
        ledger_status = "insufficient_evidence"
    else:
        ledger_status = "insufficient_evidence"
    memory_eligible = bool(official_rows)
    manifest = _read_json(experiment_root / "experiment_manifest.json")
    experiment_name = str(manifest.get("experiment_name") or manifest.get("experiment_title") or experiment_id)
    experiment_time = str(manifest.get("experiment_time") or manifest.get("date") or "")
    ledger = {
        "schema_version": "experiment_action_ledger.v1",
        "ledger_id": f"experiment_ledger_{experiment_id}",
        "ledger_type": "experiment_action_ledger",
        "ledger_status": ledger_status,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "experiment_time": experiment_time,
        "source_labvideo_path": str(source_labvideo_path) if source_labvideo_path else manifest.get("source_labvideo_path"),
        "source_material_library_path": str(experiment_root),
        "source_material_stream": str(experiment_root / "material_stream.jsonl"),
        "official_material_count": official_asset_count,
        "official_evidence_bundle_count": len(official_rows),
        "review_candidate_count": review_asset_count,
        "review_candidate_bundle_count": len(review_rows),
        "rejected_count": len(rejected_rows),
        "action_type_summary": _summarize_by_key(rows, _action_type),
        "object_summary": _summarize_by_key(rows, _object_refs),
        "instrument_summary": _summarize_by_key(rows, _instrument_refs),
        "confirmed_action_summary": _summarize_by_key(official_rows, _action_type),
        "needs_review_summary": _summarize_by_key(review_rows, _action_type),
        "rejected_summary": _summarize_by_key(rejected_rows, _action_type),
        "representative_evidence": _representative_evidence([*official_rows, *review_rows]),
        "unresolved_questions": [
            {
                "question": "这些候选素材需要人工确认后才能作为长期记忆事实。",
                "review_candidate_count": len(review_rows),
            }
        ]
        if review_rows
        else [],
        "quality_summary": {
            "status_counts": dict(status_counts),
            "memory_policy": "only_official_materials_are_factual_memory_inputs",
        },
        "memory_eligible": memory_eligible,
        "limitations": []
        if memory_eligible
        else ["当前实验没有 official 素材，不能作为 30-Day Memory 的事实输入。"],
        "last_updated_at": _utc_now(),
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_root / "experiment_action_ledger.json", ledger)
    _write_ledger_markdown(experiment_root / "experiment_action_ledger.md", ledger)
    _write_material_index_sqlite(experiment_root / "material_index.sqlite", rows, ledger)
    _write_evidence_trace_index(experiment_root / "evidence_trace_index.json", rows, ledger)
    _ensure_material_workspace_contract(experiment_root, experiment_id, ledger)
    _refresh_global_indexes_for_experiment(material_root, experiment_id, ledger, rows)
    return ledger


def _write_ledger_markdown(path: Path, ledger: Mapping[str, Any]) -> None:
    lines = [
        f"# {ledger.get('experiment_name') or ledger.get('experiment_id')} 实验动作账本",
        "",
        f"- 状态：`{ledger.get('ledger_status')}`",
        f"- official 素材：{ledger.get('official_material_count', 0)}",
        f"- 待确认候选：{ledger.get('review_candidate_count', 0)}",
        f"- 记忆可用：{ledger.get('memory_eligible')}",
        "",
        "## 已确认动作",
    ]
    for item in ledger.get("confirmed_action_summary") or []:
        lines.append(f"- {item.get('name')}: {item.get('count')}")
    lines.append("")
    lines.append("## 待确认动作")
    for item in ledger.get("needs_review_summary") or []:
        lines.append(f"- {item.get('name')}: {item.get('count')}")
    lines.append("")
    if ledger.get("limitations"):
        lines.append("## 限制")
        for item in ledger.get("limitations") or []:
            lines.append(f"- {item}")
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _ensure_material_workspace_contract(experiment_root: Path, experiment_id: str, ledger: Mapping[str, Any]) -> None:
    """Create stable sidecar files/directories without touching media assets."""
    experiment_root.mkdir(parents=True, exist_ok=True)
    for dirname in ("windows", "materials", "reports"):
        (experiment_root / dirname).mkdir(parents=True, exist_ok=True)
    for filename in (
        "material_stream.jsonl",
        "review_candidate_materials.jsonl",
        "official_materials.jsonl",
        "human_feedback.jsonl",
        "corrected_material_stream.jsonl",
    ):
        path = experiment_root / filename
        if not path.exists():
            _atomic_write_text(path, "")
    manifest_path = experiment_root / "experiment_manifest.json"
    if not manifest_path.exists():
        _write_json(
            manifest_path,
            {
                "schema_version": "lab_material_library_experiment_manifest.v1",
                "experiment_id": experiment_id,
                "experiment_name": ledger.get("experiment_name") or experiment_id,
                "material_root": str(experiment_root),
                "source_material_stream": str(experiment_root / "material_stream.jsonl"),
                "updated_at": _utc_now(),
            },
        )
    context_path = experiment_root / "context_bundle.json"
    if not context_path.exists():
        _write_json(
            context_path,
            {
                "schema_version": "lab_material_context_bundle.v1",
                "experiment_id": experiment_id,
                "material_root": str(experiment_root),
                "source_material_stream": str(experiment_root / "material_stream.jsonl"),
                "experiment_action_ledger": str(experiment_root / "experiment_action_ledger.json"),
                "memory_policy": "only_official_materials_are_factual_memory_inputs",
                "created_at": _utc_now(),
            },
        )


def _write_material_index_sqlite(path: Path, rows: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS materials (
                material_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                evidence_bundle_id TEXT,
                action_type TEXT,
                display_name TEXT,
                official_status TEXT,
                timestamp TEXT,
                window_id TEXT,
                source_window_sync_index TEXT,
                orphan_material INTEGER,
                keyframe_paths TEXT,
                keyclip_paths TEXT,
                cli_ready_folder TEXT,
                memory_eligible INTEGER,
                ledger_id TEXT
            )
            """
        )
        existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(materials)").fetchall()}
        for column, definition in {
            "display_name": "TEXT",
            "window_id": "TEXT",
            "source_window_sync_index": "TEXT",
            "orphan_material": "INTEGER",
        }.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE materials ADD COLUMN {column} {definition}")
        conn.execute("DELETE FROM materials")
        for row in rows:
            conn.execute(
                """
                INSERT OR REPLACE INTO materials (
                    material_id, experiment_id, evidence_bundle_id, action_type, display_name,
                    official_status, timestamp, window_id, source_window_sync_index, orphan_material,
                    keyframe_paths, keyclip_paths, cli_ready_folder, memory_eligible, ledger_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("material_id") or _material_id(row)),
                    str(ledger.get("experiment_id") or ""),
                    str(row.get("evidence_bundle_id") or _evidence_bundle_id(row)),
                    str(row.get("action_type") or _action_type(row)),
                    str(row.get("display_name") or _display_name(row)),
                    _official_status(row),
                    str(row.get("timestamp") or _timestamp_value(row) or ""),
                    str(_window_id(row) or ""),
                    str(_source_window_sync_index(row) or ""),
                    1 if _is_orphan_material(row) else 0,
                    json.dumps(row.get("keyframe_paths") or _keyframe_paths(row), ensure_ascii=False),
                    json.dumps(row.get("keyclip_paths") or _keyclip_paths(row), ensure_ascii=False),
                    str(row.get("cli_ready_folder") or ""),
                    1 if bool(row.get("memory_eligible")) and _official_status(row) == "official" else 0,
                    str(ledger.get("ledger_id") or ""),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _write_evidence_trace_index(path: Path, rows: Sequence[Mapping[str, Any]], ledger: Mapping[str, Any]) -> dict[str, Any]:
    trace = {
        "schema_version": "evidence_trace_index.v1",
        "experiment_id": ledger.get("experiment_id"),
        "ledger_id": ledger.get("ledger_id"),
        "created_at": _utc_now(),
        "claims": [
            {
                "claim_id": f"claim_{row.get('material_id') or _material_id(row)}",
                "claim_type": "material_action_evidence",
                "action_type": row.get("action_type") or _action_type(row),
                "official_status": _official_status(row),
                "supporting_evidence_bundle_ids": [row.get("evidence_bundle_id") or _evidence_bundle_id(row)],
                "keyframe_refs": row.get("keyframe_paths") or _keyframe_paths(row),
                "keyclip_refs": row.get("keyclip_paths") or _keyclip_paths(row),
                "timestamp_refs": [row.get("timestamp") or _timestamp_value(row)] if (row.get("timestamp") or _timestamp_value(row)) else [],
                "confidence": row.get("confidence"),
                "human_confirmation_status": _official_status(row),
            }
            for row in rows
        ],
    }
    _write_json(path, trace)
    return trace


def _experiment_id_from_session(session_root: Path) -> str:
    return session_root.parent.name if session_root.name == "key_action_index" else session_root.name


def sync_candidate_review_outputs(
    session_root: str | Path,
    candidate_root: str | Path,
    updated_rows: Sequence[Mapping[str, Any]],
    review_log_entry: Mapping[str, Any],
    *,
    material_root: str | Path | None = None,
    sync_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session_root = Path(session_root)
    candidate_root = Path(candidate_root)
    material_root = Path(material_root) if material_root else DEFAULT_MATERIAL_ROOT
    experiment_id = _experiment_id_from_session(session_root)
    target_root = material_root / experiment_id

    def _sync() -> dict[str, Any]:
        target_root.mkdir(parents=True, exist_ok=True)
        existing_by_id = {str(row.get("candidate_id") or ""): row for row in _read_jsonl(target_root / "review_candidate_materials.jsonl")}
        review_rows: list[dict[str, Any]] = []
        official_rows: list[dict[str, Any]] = []
        copy_specs: list[tuple[Path, Path]] = []
        for source_row in updated_rows:
            row = dict(source_row)
            candidate_id = str(row.get("candidate_id") or "")
            previous = existing_by_id.get(candidate_id, {})
            stored_file = row.get("stored_file") or previous.get("stored_file")
            if stored_file:
                source = Path(str(stored_file))
                if str(source).lower().startswith(str(target_root).lower()):
                    target = source
                else:
                    try:
                        rel = source.relative_to(candidate_root)
                    except ValueError:
                        rel = Path(str(row.get("asset_kind") or "assets")) / source.name
                    target = target_root / "review_candidates" / rel
                    if source.is_file():
                        copy_specs.append((source, target))
                row["stored_file"] = str(target)
            row["cli_ready_folder"] = str(target_root)
            row["experiment_id"] = str(row.get("experiment_id") or experiment_id)
            status = _official_status(row)
            if status == "official":
                official = {
                    **row,
                    "official_material": True,
                    "official_status": "official",
                    "memory_eligible": True,
                    "memory_write_allowed": True,
                    "review_status": row.get("review_status") or "accepted",
                }
                official_rows.append(official)
                review_rows.append({**row, "candidate_status": row.get("candidate_status") or "approved", "review_status": row.get("review_status") or "accepted"})
            else:
                review_rows.append(row)

        for source, target in copy_specs:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    import shutil

                    shutil.copy2(source, target)

        _write_jsonl(target_root / "review_candidate_materials.jsonl", review_rows)
        _write_jsonl(target_root / "official_materials.jsonl", official_rows)
        stream_rows = _merge_review_and_official_stream_rows(target_root, review_rows, official_rows)
        _write_jsonl(target_root / "material_stream.jsonl", stream_rows)
        _write_jsonl(target_root / "corrected_material_stream.jsonl", stream_rows)
        feedback_id = str(review_log_entry.get("feedback_id") or f"feedback_{uuid.uuid4().hex[:12]}")
        feedback = {
            "schema_version": "human_feedback_entry.v1",
            "feedback_id": feedback_id,
            "experiment_id": experiment_id,
            "created_at": review_log_entry.get("reviewed_at") or _utc_now(),
            "decision": review_log_entry.get("decision"),
            "candidate_group_id": review_log_entry.get("candidate_group_id"),
            "candidate_ids": review_log_entry.get("candidate_ids"),
            "reviewer": review_log_entry.get("reviewer"),
            "notes": review_log_entry.get("notes"),
            "reason_code": review_log_entry.get("reason_code"),
            "reason": review_log_entry.get("reason"),
        }
        _append_jsonl(target_root / "human_feedback.jsonl", feedback)
        job = {
            "schema_version": "feedback_update_job.v1",
            "job_id": f"feedback_job_{uuid.uuid4().hex[:12]}",
            "experiment_id": experiment_id,
            "feedback_id": feedback_id,
            "status": "completed",
            "created_at": _utc_now(),
            "outputs": {
                "review_candidate_materials": str(target_root / "review_candidate_materials.jsonl"),
                "official_materials": str(target_root / "official_materials.jsonl"),
                "material_stream": str(target_root / "material_stream.jsonl"),
            },
        }
        _append_jsonl(target_root / "feedback_update_jobs.jsonl", job)
        _write_json(
            target_root / "experiment_manifest.json",
            {
                **_read_json(target_root / "experiment_manifest.json"),
                "schema_version": "lab_material_library_experiment_manifest.v1",
                "experiment_id": experiment_id,
                "session_dir": str(session_root),
                "candidate_source_root": str(candidate_root),
                "material_root": str(target_root),
                "review_candidate_count": len([row for row in review_rows if _official_status(row) != "rejected"]),
                "official_material_count": len(official_rows),
                "updated_at": _utc_now(),
            },
        )
        ledger = build_experiment_action_ledger(material_root, experiment_id)
        corpus = refresh_labvideo_memory_corpus(material_root)
        return {
            "enabled": True,
            "material_root": str(target_root),
            "review_candidate_count": len(review_rows),
            "official_material_count": len(official_rows),
            "material_stream_count": len(stream_rows),
            "human_feedback": str(target_root / "human_feedback.jsonl"),
            "feedback_update_jobs": str(target_root / "feedback_update_jobs.jsonl"),
            "ledger": str(target_root / "experiment_action_ledger.json"),
            "corpus": str(material_root / "labvideo_memory_corpus.json"),
            "sync_summary": dict(sync_summary or {}),
            "ledger_status": ledger.get("ledger_status"),
            "corpus_experiment_count": corpus.get("experiment_count"),
        }

    return _with_lock(target_root / ("experiment_action_ledger" + LOCK_SUFFIX), _sync)


def apply_material_candidate_feedback(
    material_root: str | Path,
    experiment_id: str,
    *,
    candidate_group_id: str | None = None,
    material_id: str | None = None,
    evidence_bundle_id: str | None = None,
    candidate_ids: Sequence[str] | None = None,
    action: str = "upgrade_to_official",
    reviewer: str | None = None,
    notes: str | None = None,
    reason_code: str | None = None,
    reason: str | None = None,
    refresh_corpus: bool = False,
) -> dict[str, Any]:
    """Apply a human review decision to the local material database.

    This is the material-database counterpart to the frontend needs_review
    actions. It preserves the original review candidates while updating the
    canonical material stream, SQLite index, feedback log, and experiment
    ledger for one selected candidate group/material.
    """

    material_root = Path(material_root)
    target_root = material_root / experiment_id
    action_normalized = str(action or "").strip().lower()
    if action_normalized in {"approve", "approved"}:
        action_normalized = "upgrade_to_official"
    if action_normalized in {"confirmed"}:
        action_normalized = "confirm"
    if action_normalized in {"false_positive", "evidence_mismatch", "mark_evidence_mismatch"}:
        action_normalized = "reject"
    if action_normalized not in {"confirm", "upgrade_to_official", "reject", "rename"}:
        raise ValueError("action must be confirm, upgrade_to_official, reject, or rename")

    def _apply() -> dict[str, Any]:
        if not target_root.exists():
            raise FileNotFoundError(f"material experiment root not found: {target_root}")
        target_root.mkdir(parents=True, exist_ok=True)
        reports_root = target_root / "reports"
        reports_root.mkdir(parents=True, exist_ok=True)

        stream_rows = normalize_material_stream(target_root)
        review_rows = _read_jsonl(target_root / "review_candidate_materials.jsonl")
        official_rows = _read_jsonl(target_root / "official_materials.jsonl")
        before_ledger = _read_json(target_root / "experiment_action_ledger.json")
        before = {
            "official_material_count": len(official_rows),
            "stream_official_count": sum(1 for row in stream_rows if _official_status(row) == "official"),
            "review_candidate_count": len([row for row in review_rows if _active_review_status(row)]),
            "ledger_status": before_ledger.get("ledger_status"),
        }

        selected = _select_material_feedback_targets(
            stream_rows,
            review_rows,
            candidate_group_id=candidate_group_id,
            material_id=material_id,
            evidence_bundle_id=evidence_bundle_id,
            candidate_ids=candidate_ids,
        )
        if not selected["stream_rows"] and selected["review_rows"]:
            selected["stream_rows"] = _stream_rows_from_asset_rows(selected["review_rows"], status="needs_review")
        if not selected["stream_rows"]:
            raise ValueError("No material stream rows matched the feedback request")

        selected_material_ids = {str(row.get("material_id") or _material_id(row)) for row in selected["stream_rows"]}
        selected_bundle_ids = {str(row.get("evidence_bundle_id") or _evidence_bundle_id(row)) for row in selected["stream_rows"]}
        selected_group_ids = {
            str(row.get("candidate_group_id") or "")
            for row in selected["review_rows"]
            if row.get("candidate_group_id")
        }
        if candidate_group_id:
            selected_group_ids.add(str(candidate_group_id))

        feedback_id = f"feedback_{uuid.uuid4().hex[:12]}"
        reviewed_at = _utc_now()
        promoted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        ineligible: list[dict[str, Any]] = []
        updated_stream: list[dict[str, Any]] = []
        official_by_id = {str(row.get("material_id") or _material_id(row)): dict(row) for row in official_rows}

        for row in stream_rows:
            row_id = str(row.get("material_id") or _material_id(row))
            bundle_id = str(row.get("evidence_bundle_id") or _evidence_bundle_id(row))
            if row_id not in selected_material_ids and bundle_id not in selected_bundle_ids:
                updated_stream.append(dict(row))
                continue

            if action_normalized in {"confirm", "upgrade_to_official"}:
                prepared = _prepare_official_material_row(target_root, row)
                eligibility = _official_promotion_eligibility(prepared, target_root)
                if eligibility["eligible"]:
                    official_row = {
                        **prepared,
                        "official_status": "official",
                        "review_status": "confirmed",
                        "human_confirmation_status": "human_confirmed",
                        "candidate_status": "upgraded",
                        "official_material": True,
                        "memory_eligible": True,
                        "memory_write_allowed": True,
                        "confirmed_at": reviewed_at,
                        "confirmed_by": reviewer or "operator",
                        "confirmation_feedback_id": feedback_id,
                        "promotion_action": action_normalized,
                    }
                    official_by_id[row_id] = official_row
                    updated_stream.append(official_row)
                    promoted.append(
                        {
                            "material_id": row_id,
                            "evidence_bundle_id": bundle_id,
                            "eligibility": eligibility,
                        }
                    )
                else:
                    needs_review_row = {
                        **row,
                        "official_status": "needs_review",
                        "review_status": "needs_review",
                        "memory_eligible": False,
                        "memory_write_allowed": False,
                        "last_promotion_attempt_at": reviewed_at,
                        "last_promotion_feedback_id": feedback_id,
                        "promotion_blockers": eligibility["missing"],
                    }
                    updated_stream.append(needs_review_row)
                    ineligible.append(
                        {
                            "material_id": row_id,
                            "evidence_bundle_id": bundle_id,
                            "eligibility": eligibility,
                        }
                    )
            elif action_normalized == "reject":
                rejected_row = {
                    **row,
                    "official_status": "rejected",
                    "review_status": "rejected",
                    "candidate_status": "rejected",
                    "memory_eligible": False,
                    "memory_write_allowed": False,
                    "rejected_at": reviewed_at,
                    "rejected_by": reviewer or "operator",
                    "rejection_reason_code": reason_code,
                    "rejection_reason": reason or notes,
                    "confirmation_feedback_id": feedback_id,
                }
                updated_stream.append(rejected_row)
                rejected.append({"material_id": row_id, "evidence_bundle_id": bundle_id})
            else:
                updated_stream.append(dict(row))

        updated_review_rows: list[dict[str, Any]] = []
        selected_candidate_ids = {str(item) for item in (candidate_ids or []) if str(item).strip()}
        for row in review_rows:
            row_id = str(row.get("material_id") or _material_id(row))
            bundle_id = str(row.get("evidence_bundle_id") or _evidence_bundle_id(row))
            group_id = str(row.get("candidate_group_id") or "")
            candidate_id = str(row.get("candidate_id") or "")
            is_selected = (
                row_id in selected_material_ids
                or bundle_id in selected_bundle_ids
                or (group_id and group_id in selected_group_ids)
                or (candidate_id and candidate_id in selected_candidate_ids)
            )
            updated = dict(row)
            if is_selected and promoted and action_normalized in {"confirm", "upgrade_to_official"}:
                updated.update(
                    {
                        "candidate_status": "upgraded",
                        "review_status": "upgraded",
                        "upgraded_to_official": True,
                        "upgraded_at": reviewed_at,
                        "upgraded_by": reviewer or "operator",
                        "confirmation_feedback_id": feedback_id,
                        "memory_eligible": False,
                        "memory_write_allowed": False,
                    }
                )
            elif is_selected and action_normalized == "reject":
                updated.update(
                    {
                        "candidate_status": "rejected",
                        "review_status": "rejected",
                        "rejected_at": reviewed_at,
                        "rejected_by": reviewer or "operator",
                        "rejection_reason_code": reason_code,
                        "rejection_reason": reason or notes,
                        "memory_eligible": False,
                        "memory_write_allowed": False,
                    }
                )
            updated_review_rows.append(updated)

        official_rows_out = sorted(official_by_id.values(), key=lambda row: str(row.get("material_id") or _material_id(row)))
        updated_stream = _dedupe_material_rows(updated_stream)
        _write_jsonl(target_root / "review_candidate_materials.jsonl", updated_review_rows)
        _write_jsonl(target_root / "official_materials.jsonl", official_rows_out)
        _write_jsonl(target_root / "material_stream.jsonl", updated_stream)
        _write_jsonl(target_root / "corrected_material_stream.jsonl", updated_stream)

        feedback = {
            "schema_version": "human_feedback_entry.v1",
            "feedback_id": feedback_id,
            "experiment_id": experiment_id,
            "created_at": reviewed_at,
            "decision": action_normalized,
            "candidate_group_id": candidate_group_id,
            "candidate_ids": list(candidate_ids or []),
            "material_ids": sorted(selected_material_ids),
            "evidence_bundle_ids": sorted(selected_bundle_ids),
            "reviewer": reviewer or "operator",
            "notes": notes,
            "reason_code": reason_code,
            "reason": reason,
            "promoted_count": len(promoted),
            "rejected_count": len(rejected),
            "ineligible_count": len(ineligible),
        }
        _append_jsonl(target_root / "human_feedback.jsonl", feedback)
        feedback_job = {
            "schema_version": "feedback_update_job.v1",
            "job_id": f"feedback_job_{uuid.uuid4().hex[:12]}",
            "experiment_id": experiment_id,
            "feedback_id": feedback_id,
            "status": "completed",
            "created_at": _utc_now(),
            "action": action_normalized,
            "outputs": {
                "official_materials": str(target_root / "official_materials.jsonl"),
                "review_candidate_materials": str(target_root / "review_candidate_materials.jsonl"),
                "material_stream": str(target_root / "material_stream.jsonl"),
                "corrected_material_stream": str(target_root / "corrected_material_stream.jsonl"),
            },
        }
        _append_jsonl(target_root / "feedback_update_jobs.jsonl", feedback_job)

        ledger = build_experiment_action_ledger(material_root, experiment_id)
        _refresh_global_indexes_for_experiment(material_root, experiment_id, ledger, normalize_material_stream(target_root))
        corpus = refresh_labvideo_memory_corpus(material_root) if refresh_corpus else None
        after = {
            "official_material_count": len(_read_jsonl(target_root / "official_materials.jsonl")),
            "stream_official_count": sum(1 for row in normalize_material_stream(target_root) if _official_status(row) == "official"),
            "review_candidate_count": len([row for row in _read_jsonl(target_root / "review_candidate_materials.jsonl") if _active_review_status(row)]),
            "ledger_status": ledger.get("ledger_status"),
            "confirmed_action_summary_count": len(ledger.get("confirmed_action_summary") or []),
        }
        report = {
            "schema_version": "candidate_promotion_report.v1",
            "experiment_id": experiment_id,
            "action": action_normalized,
            "candidate_group_id": candidate_group_id,
            "material_id": material_id,
            "evidence_bundle_id": evidence_bundle_id,
            "candidate_ids": list(candidate_ids or []),
            "feedback_id": feedback_id,
            "before": before,
            "after": after,
            "promoted": promoted,
            "rejected": rejected,
            "ineligible": ineligible,
            "human_feedback": str(target_root / "human_feedback.jsonl"),
            "corrected_material_stream": str(target_root / "corrected_material_stream.jsonl"),
            "official_materials": str(target_root / "official_materials.jsonl"),
            "ledger": str(target_root / "experiment_action_ledger.json"),
            "corpus_refreshed": bool(corpus),
            "corpus_is_real_30_day_memory": corpus.get("is_real_30_day_memory") if corpus else False,
        }
        _write_json(reports_root / "candidate_promotion_report.json", report)
        _write_json(
            target_root / "ledger_update_report.json",
            {
                "schema_version": "ledger_update_report.v1",
                "experiment_id": experiment_id,
                "feedback_id": feedback_id,
                "before": before,
                "after": after,
                "ledger": str(target_root / "experiment_action_ledger.json"),
                "ledger_status": ledger.get("ledger_status"),
                "memory_eligible": ledger.get("memory_eligible"),
            },
        )
        _write_json(
            reports_root / "frontend_candidate_confirm_e2e_report.json",
            {
                "schema_version": "frontend_candidate_confirm_e2e_report.v1",
                "validation_mode": "backend_api_equivalent",
                "experiment_id": experiment_id,
                "candidate_group_id": candidate_group_id,
                "action": action_normalized,
                "promoted_count": len(promoted),
                "official_material_count_after": after["official_material_count"],
                "frontend_expected_behavior": "needs_review item moves to official tab after cache refresh; original review candidate remains audit-visible.",
                "backend_mutation": "apply_material_candidate_feedback",
            },
        )
        return report

    return _with_lock(target_root / ("experiment_action_ledger" + LOCK_SUFFIX), _apply)


def _select_material_feedback_targets(
    stream_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_group_id: str | None,
    material_id: str | None,
    evidence_bundle_id: str | None,
    candidate_ids: Sequence[str] | None,
) -> dict[str, list[dict[str, Any]]]:
    explicit_candidate_ids = {str(item) for item in (candidate_ids or []) if str(item).strip()}
    selected_review: list[dict[str, Any]] = []
    for row in review_rows:
        candidate_id = str(row.get("candidate_id") or "")
        group_id = str(row.get("candidate_group_id") or "")
        row_material = str(row.get("material_id") or _material_id(row))
        row_bundle = str(row.get("evidence_bundle_id") or _evidence_bundle_id(row))
        if explicit_candidate_ids and candidate_id in explicit_candidate_ids:
            selected_review.append(dict(row))
        elif not explicit_candidate_ids and candidate_group_id and group_id == str(candidate_group_id):
            selected_review.append(dict(row))
        elif material_id and row_material == str(material_id):
            selected_review.append(dict(row))
        elif evidence_bundle_id and row_bundle == str(evidence_bundle_id):
            selected_review.append(dict(row))
    bundle_ids = {str(row.get("evidence_bundle_id") or _evidence_bundle_id(row)) for row in selected_review}
    material_ids = {str(row.get("material_id") or _material_id(row)) for row in selected_review}
    selected_stream: list[dict[str, Any]] = []
    for row in stream_rows:
        row_material = str(row.get("material_id") or _material_id(row))
        row_bundle = str(row.get("evidence_bundle_id") or _evidence_bundle_id(row))
        if material_id and row_material == str(material_id):
            selected_stream.append(dict(row))
        elif evidence_bundle_id and row_bundle == str(evidence_bundle_id):
            selected_stream.append(dict(row))
        elif row_bundle in bundle_ids or row_material in material_ids:
            selected_stream.append(dict(row))
    return {"stream_rows": selected_stream, "review_rows": selected_review}


def _dedupe_material_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        material_id = str(row.get("material_id") or _material_id(row))
        existing = by_id.get(material_id, {})
        merged = {**existing, **dict(row)}
        by_id[material_id] = merged
    return sorted(by_id.values(), key=lambda row: str(row.get("material_id") or _material_id(row)))


def _prepare_official_material_row(target_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(row)
    material_id = str(prepared.get("material_id") or _material_id(prepared))
    material_folder = target_root / "materials" / material_id
    material_folder.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("first_keyframe", "first_keyframe.jpg"),
        ("third_keyframe", "third_keyframe.jpg"),
        ("first_keyclip", "first_keyclip.mp4"),
        ("third_keyclip", "third_keyclip.mp4"),
        ("side_by_side_keyclip", "side_by_side_keyclip.mp4"),
    ):
        source = prepared.get(key)
        if not source:
            continue
        target = material_folder / filename
        copied = _link_or_copy_file(Path(str(source)), target)
        if copied:
            prepared[key] = str(target)
    if prepared.get("keyframe_paths"):
        prepared["keyframe_paths"] = _keyframe_paths(prepared) or prepared.get("keyframe_paths")
    if prepared.get("keyclip_paths"):
        prepared["keyclip_paths"] = _keyclip_paths(prepared) or prepared.get("keyclip_paths")
    lineage = prepared.get("lineage") if isinstance(prepared.get("lineage"), Mapping) else {}
    lineage_path = material_folder / "lineage.json"
    _write_json(lineage_path, {"schema_version": "material_lineage.v1", **dict(lineage), "material_id": material_id})
    quality = {
        "schema_version": "material_quality_report.v1",
        "material_id": material_id,
        "keyframe_quality_score": prepared.get("keyframe_quality_score") or prepared.get("selected_keyframe_score"),
        "selected_keyframe_reason": prepared.get("selected_keyframe_reason"),
        "confidence": prepared.get("confidence"),
        "quality_flags": prepared.get("quality_flags") or [],
        "promotion_validation": "human_confirmed_official_candidate",
    }
    quality_path = material_folder / "quality_report.json"
    _write_json(quality_path, quality)
    evidence_path = material_folder / "evidence_bundle.json"
    _write_json(
        evidence_path,
        {
            "schema_version": "evidence_bundle.v1",
            "evidence_bundle_id": prepared.get("evidence_bundle_id") or _evidence_bundle_id(prepared),
            "material_id": material_id,
            "action_type": prepared.get("action_type") or _action_type(prepared),
            "keyframe_refs": _keyframe_paths(prepared),
            "keyclip_refs": _keyclip_paths(prepared),
            "source_window_sync_index": prepared.get("source_window_sync_index"),
        },
    )
    material_json_path = material_folder / "material.json"
    prepared.update(
        {
            "material_id": material_id,
            "material_folder": str(material_folder),
            "cli_ready_folder": str(material_folder),
            "lineage_path": str(lineage_path),
            "quality_report": str(quality_path),
            "evidence_bundle_path": str(evidence_path),
            "material_json": str(material_json_path),
        }
    )
    _write_json(material_json_path, prepared)
    return prepared


def _link_or_copy_file(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return True
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    return True


def _official_promotion_eligibility(row: Mapping[str, Any], target_root: Path) -> dict[str, Any]:
    missing: list[str] = []
    for key in ("material_id", "evidence_bundle_id", "window_id", "source_window_sync_index"):
        if not row.get(key):
            missing.append(key)
    if _official_status(row) == "official":
        pass
    elif _official_status(row) != "needs_review":
        missing.append("official_status_needs_review")
    phase_status = _material_phase_status(row, target_root)
    if phase_status not in {"dual_view_valid", "first_dominant_with_reason"}:
        missing.append("dual_view_action_phase_status")
    if not (row.get("keyframe_quality_score") or row.get("selected_keyframe_score")):
        missing.append("keyframe_quality_score")
    for key in ("first_keyframe", "third_keyframe", "first_keyclip", "third_keyclip"):
        if not row.get(key) or not Path(str(row.get(key))).is_file():
            missing.append(key)
    if not row.get("cli_ready_folder") or not Path(str(row.get("cli_ready_folder"))).exists():
        missing.append("cli_ready_folder")
    if not (row.get("lineage") or (row.get("lineage_path") and Path(str(row.get("lineage_path"))).is_file())):
        missing.append("lineage")
    if not row.get("quality_report") or not Path(str(row.get("quality_report"))).is_file():
        missing.append("quality_report")
    return {
        "eligible": not missing,
        "missing": sorted(set(missing)),
        "dual_view_action_phase_status": phase_status,
    }


def _material_phase_status(row: Mapping[str, Any], target_root: Path) -> str:
    for key in ("dual_view_action_phase_status", "action_phase_status", "cross_view_consistency"):
        value = str(row.get(key) or "").strip()
        if value in {"dual_view_valid", "first_dominant_with_reason"}:
            return value
    report_paths = []
    lineage = row.get("lineage") if isinstance(row.get("lineage"), Mapping) else {}
    session_dir = lineage.get("session_dir")
    if session_dir:
        report_paths.append(Path(str(session_dir)) / "metadata" / "dual_view_action_phase_report.json")
    report_paths.append(target_root / "reports" / "dual_view_action_phase_report.json")
    bundle_id = str(row.get("evidence_bundle_id") or _evidence_bundle_id(row))
    for path in report_paths:
        report = _read_json(path)
        for event in report.get("events") or []:
            if not isinstance(event, Mapping):
                continue
            if str(event.get("evidence_bundle_id") or event.get("dual_event_id") or "") == bundle_id:
                return str(event.get("status") or "")
    return str(row.get("cross_view_consistency") or "")


def _refresh_global_indexes_for_experiment(
    material_root: Path,
    experiment_id: str,
    ledger: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    material_root.mkdir(parents=True, exist_ok=True)
    ledger_index_path = material_root / "global_experiment_ledger_index.jsonl"
    ledger_rows = [
        row
        for row in _read_jsonl(ledger_index_path)
        if str(row.get("experiment_id") or "") != str(experiment_id)
    ]
    ledger_rows.append(
        {
            "experiment_id": experiment_id,
            "ledger_id": ledger.get("ledger_id"),
            "ledger_status": ledger.get("ledger_status"),
            "official_material_count": ledger.get("official_material_count"),
            "review_candidate_count": ledger.get("review_candidate_count"),
            "memory_eligible": ledger.get("memory_eligible"),
            "ledger_path": str(Path(str(ledger.get("source_material_library_path") or material_root / experiment_id)) / "experiment_action_ledger.json"),
        }
    )
    _write_jsonl(ledger_index_path, sorted(ledger_rows, key=lambda row: str(row.get("experiment_id") or "")))

    material_index_path = material_root / "global_material_search_index.jsonl"
    material_rows = [
        row
        for row in _read_jsonl(material_index_path)
        if str(row.get("experiment_id") or "") != str(experiment_id)
    ]
    trace_index_path = material_root / "global_evidence_trace_index.jsonl"
    trace_rows = [
        row
        for row in _read_jsonl(trace_index_path)
        if str(row.get("experiment_id") or "") != str(experiment_id)
    ]
    for row in rows:
        material = {
            "experiment_id": experiment_id,
            "material_id": row.get("material_id") or _material_id(row),
            "action_type": row.get("action_type") or _action_type(row),
            "display_name": row.get("display_name") or _display_name(row),
            "official_status": _official_status(row),
            "timestamp": row.get("timestamp") or _timestamp_value(row),
            "keyframe_paths": row.get("keyframe_paths") or _keyframe_paths(row),
            "keyclip_paths": row.get("keyclip_paths") or _keyclip_paths(row),
            "first_keyframe": row.get("first_keyframe"),
            "third_keyframe": row.get("third_keyframe"),
            "first_keyclip": row.get("first_keyclip"),
            "third_keyclip": row.get("third_keyclip"),
            "side_by_side_keyclip": row.get("side_by_side_keyclip"),
            "evidence_bundle_id": row.get("evidence_bundle_id") or _evidence_bundle_id(row),
            "window_id": _window_id(row) or None,
            "experiment_window_id": _first_text(row.get("experiment_window_id"), _window_id(row)) or None,
            "source_window_sync_index": _source_window_sync_index(row) or None,
            "orphan_material": _is_orphan_material(row),
            "diagnostic_status": "orphan_material" if _is_orphan_material(row) else row.get("diagnostic_status"),
            "cli_ready_folder": row.get("cli_ready_folder"),
            "memory_eligible": bool(row.get("memory_eligible")) and _official_status(row) == "official",
            "review_status": row.get("review_status") or row.get("official_status"),
            "ledger_id": ledger.get("ledger_id"),
        }
        material_rows.append(material)
        trace_rows.append(
            {
                "experiment_id": experiment_id,
                "material_id": material["material_id"],
                "evidence_bundle_id": material["evidence_bundle_id"],
                "action_type": material["action_type"],
                "evidence_trace": {
                    "keyframe_refs": material["keyframe_paths"],
                    "keyclip_refs": material["keyclip_paths"],
                    "timestamp_refs": [material["timestamp"]] if material["timestamp"] else [],
                },
                "official_status": material["official_status"],
            }
        )
    _write_jsonl(material_index_path, sorted(material_rows, key=lambda row: (str(row.get("experiment_id") or ""), str(row.get("material_id") or ""))))
    _write_jsonl(trace_index_path, sorted(trace_rows, key=lambda row: (str(row.get("experiment_id") or ""), str(row.get("material_id") or ""))))


def _merge_review_and_official_stream_rows(
    target_root: Path,
    review_rows: Sequence[Mapping[str, Any]],
    official_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    existing_stream = normalize_material_stream(target_root)
    by_material: dict[str, dict[str, Any]] = {str(row.get("material_id") or _material_id(row)): dict(row) for row in existing_stream}
    for row in _stream_rows_from_asset_rows(review_rows, status="needs_review"):
        if _official_status(row) == "rejected":
            continue
        by_material[str(row.get("material_id"))] = {**by_material.get(str(row.get("material_id")), {}), **row}
    for row in _stream_rows_from_asset_rows(official_rows, status="official"):
        row["memory_eligible"] = True
        by_material[str(row.get("material_id"))] = {**by_material.get(str(row.get("material_id")), {}), **row}
    return sorted(by_material.values(), key=lambda row: str(row.get("material_id") or ""))


def iter_material_experiment_roots(material_root: str | Path) -> list[Path]:
    material_root = Path(material_root)
    roots: list[Path] = []
    if not material_root.exists():
        return roots
    for child in material_root.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if child.name in {"memory_index", "material_references"}:
            continue
        if any((child / name).exists() for name in ("material_stream.jsonl", "official_materials.jsonl", "review_candidate_materials.jsonl", "key_material_references.jsonl")):
            roots.append(child)
    legacy_root = material_root / "material_references"
    if legacy_root.is_dir():
        for child in legacy_root.iterdir():
            if child.is_dir() and any((child / name).exists() for name in ("key_material_references.jsonl", "素材索引.jsonl", "physical_change_log.jsonl")):
                roots.append(child)
    return sorted(roots, key=lambda path: str(path).lower())


def refresh_labvideo_memory_corpus(
    material_root: str | Path = DEFAULT_MATERIAL_ROOT,
    *,
    labvideo_root: str | Path = DEFAULT_LABVIDEO_ROOT,
) -> dict[str, Any]:
    material_root = Path(material_root)
    labvideo_root = Path(labvideo_root)
    ledgers: list[dict[str, Any]] = []
    for root in iter_material_experiment_roots(material_root):
        experiment_id = root.name
        try:
            ledger = build_experiment_action_ledger(material_root if root.parent != material_root / "material_references" else root.parent, experiment_id)
        except Exception as exc:
            ledger = {
                "ledger_id": f"experiment_ledger_{experiment_id}",
                "ledger_status": "failed",
                "experiment_id": experiment_id,
                "source_material_library_path": str(root),
                "official_material_count": 0,
                "review_candidate_count": 0,
                "memory_eligible": False,
                "limitations": [str(exc)],
            }
        ledgers.append(ledger)
    status_counter = Counter(str(ledger.get("ledger_status") or "unknown") for ledger in ledgers)
    all_stream_rows: list[dict[str, Any]] = []
    for root in iter_material_experiment_roots(material_root):
        all_stream_rows.extend(normalize_material_stream(root))
    corpus = {
        "schema_version": "labvideo_memory_corpus.v1",
        "corpus_id": "labvideo_memory_corpus",
        "source_mode": "existing_labvideo_backfill",
        "is_real_30_day_memory": False,
        "created_at": _utc_now(),
        "labvideo_root": str(labvideo_root),
        "material_root": str(material_root),
        "experiment_count": len(ledgers),
        "ledger_count": len(ledgers),
        "official_ready_ledger_count": status_counter.get("official_ready", 0),
        "needs_review_only_ledger_count": status_counter.get("needs_review_only", 0),
        "mixed_ledger_count": status_counter.get("mixed", 0),
        "insufficient_evidence_ledger_count": status_counter.get("insufficient_evidence", 0),
        "failed_ledger_count": status_counter.get("failed", 0),
        "total_official_materials": sum(int(ledger.get("official_material_count") or 0) for ledger in ledgers),
        "total_review_candidate_materials": sum(int(ledger.get("review_candidate_count") or 0) for ledger in ledgers),
        "top_action_types": _summarize_by_key(all_stream_rows, _action_type)[:20],
        "top_objects": _summarize_by_key(all_stream_rows, _object_refs)[:20],
        "top_instruments": _summarize_by_key(all_stream_rows, _instrument_refs)[:20],
        "experiment_ledgers": [
            {
                "experiment_id": ledger.get("experiment_id"),
                "ledger_id": ledger.get("ledger_id"),
                "ledger_status": ledger.get("ledger_status"),
                "official_material_count": ledger.get("official_material_count"),
                "review_candidate_count": ledger.get("review_candidate_count"),
                "memory_eligible": ledger.get("memory_eligible"),
                "path": str(Path(str(ledger.get("source_material_library_path") or material_root / str(ledger.get("experiment_id")))) / "experiment_action_ledger.json"),
            }
            for ledger in ledgers
        ],
        "unresolved_questions_summary": [
            {
                "experiment_id": ledger.get("experiment_id"),
                "questions": ledger.get("unresolved_questions") or [],
            }
            for ledger in ledgers
            if ledger.get("unresolved_questions")
        ],
        "memory_readiness": {
            "can_build_30_day_memory": False,
            "reason": "not_enough_real_continuous_30_day_data",
            "next_step": "confirm_candidates_and_accumulate_real_daily_ledgers",
        },
    }
    material_root.mkdir(parents=True, exist_ok=True)
    _write_json(material_root / "labvideo_memory_corpus.json", corpus)
    _write_corpus_markdown(material_root / "labvideo_memory_corpus.md", corpus)
    _write_global_indexes(material_root, ledgers, all_stream_rows)
    return corpus


def _write_corpus_markdown(path: Path, corpus: Mapping[str, Any]) -> None:
    lines = [
        "# LabVideo Memory Corpus",
        "",
        f"- source_mode: `{corpus.get('source_mode')}`",
        f"- is_real_30_day_memory: `{corpus.get('is_real_30_day_memory')}`",
        f"- 实验账本数：{corpus.get('ledger_count', 0)}",
        f"- official 素材：{corpus.get('total_official_materials', 0)}",
        f"- 待确认候选：{corpus.get('total_review_candidate_materials', 0)}",
        "",
        "## 账本状态",
        f"- official_ready: {corpus.get('official_ready_ledger_count', 0)}",
        f"- mixed: {corpus.get('mixed_ledger_count', 0)}",
        f"- needs_review_only: {corpus.get('needs_review_only_ledger_count', 0)}",
        f"- insufficient_evidence: {corpus.get('insufficient_evidence_ledger_count', 0)}",
        f"- failed: {corpus.get('failed_ledger_count', 0)}",
        "",
        "## 记忆就绪度",
        f"- can_build_30_day_memory: {corpus.get('memory_readiness', {}).get('can_build_30_day_memory')}",
        f"- reason: {corpus.get('memory_readiness', {}).get('reason')}",
    ]
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _write_global_indexes(material_root: Path, ledgers: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    _write_jsonl(
        material_root / "global_experiment_ledger_index.jsonl",
        [
            {
                "experiment_id": ledger.get("experiment_id"),
                "ledger_id": ledger.get("ledger_id"),
                "ledger_status": ledger.get("ledger_status"),
                "official_material_count": ledger.get("official_material_count"),
                "review_candidate_count": ledger.get("review_candidate_count"),
                "memory_eligible": ledger.get("memory_eligible"),
                "ledger_path": str(Path(str(ledger.get("source_material_library_path") or material_root / str(ledger.get("experiment_id")))) / "experiment_action_ledger.json"),
            }
            for ledger in ledgers
        ],
    )
    material_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for row in rows:
        experiment_id = str(row.get("experiment_id") or "")
        if not experiment_id:
            cli_root = row.get("cli_ready_folder")
            if cli_root:
                experiment_id = Path(str(cli_root)).name
        material = {
            "experiment_id": experiment_id,
            "material_id": row.get("material_id") or _material_id(row),
            "action_type": row.get("action_type") or _action_type(row),
            "display_name": row.get("display_name") or _display_name(row),
            "official_status": _official_status(row),
            "timestamp": row.get("timestamp") or _timestamp_value(row),
            "keyframe_paths": row.get("keyframe_paths") or _keyframe_paths(row),
            "keyclip_paths": row.get("keyclip_paths") or _keyclip_paths(row),
            "first_keyframe": row.get("first_keyframe"),
            "third_keyframe": row.get("third_keyframe"),
            "first_keyclip": row.get("first_keyclip"),
            "third_keyclip": row.get("third_keyclip"),
            "side_by_side_keyclip": row.get("side_by_side_keyclip"),
            "evidence_bundle_id": row.get("evidence_bundle_id") or _evidence_bundle_id(row),
            "window_id": _window_id(row) or None,
            "experiment_window_id": _first_text(row.get("experiment_window_id"), _window_id(row)) or None,
            "source_window_sync_index": _source_window_sync_index(row) or None,
            "orphan_material": _is_orphan_material(row),
            "diagnostic_status": "orphan_material" if _is_orphan_material(row) else row.get("diagnostic_status"),
            "cli_ready_folder": row.get("cli_ready_folder"),
            "memory_eligible": bool(row.get("memory_eligible")) and _official_status(row) == "official",
            "review_status": row.get("review_status") or row.get("official_status"),
            "ledger_id": f"experiment_ledger_{experiment_id}" if experiment_id else None,
        }
        material_rows.append(material)
        trace_rows.append(
            {
                "experiment_id": material["experiment_id"],
                "material_id": material["material_id"],
                "evidence_bundle_id": material["evidence_bundle_id"],
                "action_type": material["action_type"],
                "evidence_trace": {
                    "keyframe_refs": material["keyframe_paths"],
                    "keyclip_refs": material["keyclip_paths"],
                    "timestamp_refs": [material["timestamp"]] if material["timestamp"] else [],
                },
                "official_status": material["official_status"],
            }
        )
    _write_jsonl(material_root / "global_material_search_index.jsonl", material_rows)
    _write_jsonl(material_root / "global_evidence_trace_index.jsonl", trace_rows)


def scan_labvideo_backfill(
    labvideo_root: str | Path = DEFAULT_LABVIDEO_ROOT,
    material_root: str | Path = DEFAULT_MATERIAL_ROOT,
) -> dict[str, Any]:
    labvideo_root = Path(labvideo_root)
    material_root = Path(material_root)
    imports_root = labvideo_root / "raw_uploads" / "by_import"
    entries: list[dict[str, Any]] = []
    material_roots = iter_material_experiment_roots(material_root)
    source_text_by_root: dict[Path, str] = {}
    for root in material_roots:
        texts: list[str] = []
        for name in ("material_stream.jsonl", "review_candidate_materials.jsonl", "official_materials.jsonl", "experiment_manifest.json"):
            path = root / name
            if path.exists():
                try:
                    texts.append(path.read_text(encoding="utf-8", errors="ignore")[:200000])
                except OSError:
                    pass
        source_text_by_root[root] = "\n".join(texts)
    import_dirs = [path for path in imports_root.iterdir() if path.is_dir()] if imports_root.is_dir() else []
    for directory in sorted(import_dirs, key=lambda path: path.name.lower()):
        video_files = list(directory.rglob("*.mp4"))
        frames = list(directory.rglob("frames.csv"))
        associated = [
            root.name
            for root, text in source_text_by_root.items()
            if str(directory) in text or directory.name in text
        ]
        if not associated:
            associated = _heuristic_material_experiment_matches(directory.name, material_roots)
        if associated:
            ledger_exists = any((root / "experiment_action_ledger.json").exists() for root in material_roots if root.name in associated)
            mode = "reuse" if ledger_exists else "incremental_rebuild"
        elif "virtual_camera_cache" in directory.name.lower():
            mode = "skipped"
        elif len(video_files) >= 2 and len(frames) >= 2:
            mode = "full_analysis"
        else:
            mode = "skipped"
        entries.append(
            {
                "source_path": str(directory),
                "source_name": directory.name,
                "video_file_count": len(video_files),
                "frames_csv_count": len(frames),
                "associated_material_experiments": associated,
                "mode": mode,
                "skip_reason": None if mode != "skipped" else "missing_dual_view_video_or_frames_csv",
            }
        )
    report = {
        "schema_version": "labvideo_scan_report.v1",
        "created_at": _utc_now(),
        "labvideo_root": str(labvideo_root),
        "material_root": str(material_root),
        "entries": entries,
        "mode_counts": dict(Counter(entry["mode"] for entry in entries)),
    }
    material_root.mkdir(parents=True, exist_ok=True)
    _write_json(material_root / "labvideo_scan_report.json", report)
    _write_json(
        material_root / "batch_video_asset_cache_report.json",
        {
            "schema_version": "batch_video_asset_cache_report.v1",
            "created_at": _utc_now(),
            "labvideo_root": str(labvideo_root),
            "asset_count": len(entries),
            "cache_policy": "reuse_existing_labvideo_assets_by_path_and_sha_when_registered",
            "entries": entries,
        },
    )
    _write_json(
        material_root / "batch_backfill_plan.json",
        {
            "schema_version": "batch_backfill_plan.v1",
            "created_at": _utc_now(),
            "entries": entries,
            "full_analysis_policy": "record_required_full_analysis_without_blocking_reuse_and_incremental_ledger_refresh",
        },
    )
    return report


def _heuristic_material_experiment_matches(source_name: str, material_roots: Sequence[Path]) -> list[str]:
    lowered_source = source_name.lower()
    date_tokens = set(re.findall(r"20\d{6}", source_name.replace("-", "")))
    if not date_tokens:
        return []
    source_is_latest = "latest_continuous" in lowered_source or "continuous_stream" in lowered_source
    source_is_desktop = "desktop_import" in lowered_source
    matches: list[str] = []
    for root in material_roots:
        lowered = root.name.lower()
        root_dates = set(re.findall(r"20\d{6}", root.name.replace("-", "")))
        if not (date_tokens & root_dates):
            continue
        if source_is_latest and any(marker in lowered for marker in ("latest", "最新连续", "健康时间轴")):
            matches.append(root.name)
        elif source_is_desktop and "desktop-import" in lowered:
            matches.append(root.name)
    return sorted(set(matches))


def run_labvideo_backfill(
    labvideo_root: str | Path = DEFAULT_LABVIDEO_ROOT,
    material_root: str | Path = DEFAULT_MATERIAL_ROOT,
) -> dict[str, Any]:
    scan = scan_labvideo_backfill(labvideo_root, material_root)
    material_root = Path(material_root)
    ledgers: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for root in iter_material_experiment_roots(material_root):
        try:
            ledger = build_experiment_action_ledger(material_root if root.parent != material_root / "material_references" else root.parent, root.name)
            ledgers.append(ledger)
        except Exception as exc:
            failures.append({"experiment_id": root.name, "error": str(exc)})
    corpus = refresh_labvideo_memory_corpus(material_root, labvideo_root=labvideo_root)
    report = {
        "schema_version": "batch_backfill_run_report.v1",
        "created_at": _utc_now(),
        "labvideo_root": str(labvideo_root),
        "material_root": str(material_root),
        "scanned_experiment_count": len(scan.get("entries") or []),
        "generated_ledger_count": len(ledgers),
        "failure_count": len(failures),
        "failures": failures,
        "corpus": str(material_root / "labvideo_memory_corpus.json"),
    }
    _write_json(material_root / "batch_backfill_run_report.json", report)
    _write_json(
        material_root / "batch_backfill_quality_report.json",
        {
            "schema_version": "batch_backfill_quality_report.v1",
            "created_at": _utc_now(),
            "ledger_status_counts": {
                "official_ready": corpus.get("official_ready_ledger_count", 0),
                "mixed": corpus.get("mixed_ledger_count", 0),
                "needs_review_only": corpus.get("needs_review_only_ledger_count", 0),
                "insufficient_evidence": corpus.get("insufficient_evidence_ledger_count", 0),
                "failed": corpus.get("failed_ledger_count", 0),
            },
            "is_real_30_day_memory": False,
            "memory_readiness": corpus.get("memory_readiness"),
        },
    )
    return report


def query_materials(
    material_root: str | Path = DEFAULT_MATERIAL_ROOT,
    *,
    action_type: str | None = None,
    experiment_id: str | None = None,
    official_status: str | None = None,
    material_id: str | None = None,
    evidence_bundle_id: str | None = None,
    has_keyframe: bool | None = None,
    has_keyclip: bool | None = None,
    source_window_sync_index: str | None = None,
    include_orphans: bool = False,
) -> list[dict[str, Any]]:
    material_root = Path(material_root)
    index = material_root / "global_material_search_index.jsonl"
    if not index.exists():
        refresh_labvideo_memory_corpus(material_root)
    rows = _read_jsonl(index)
    result: list[dict[str, Any]] = []
    for row in rows:
        if not include_orphans and row.get("orphan_material") is True and str(row.get("official_status") or "") != "official":
            continue
        if action_type and str(row.get("action_type") or "") != action_type:
            continue
        if experiment_id and str(row.get("experiment_id") or "") != experiment_id:
            continue
        if official_status and str(row.get("official_status") or "") != official_status:
            continue
        if material_id and str(row.get("material_id") or "") != material_id:
            continue
        if evidence_bundle_id and str(row.get("evidence_bundle_id") or "") != evidence_bundle_id:
            continue
        if source_window_sync_index and str(row.get("source_window_sync_index") or "") != source_window_sync_index:
            continue
        keyframe_available = bool(row.get("first_keyframe") or row.get("third_keyframe") or row.get("keyframe_paths"))
        keyclip_available = bool(row.get("first_keyclip") or row.get("third_keyclip") or row.get("side_by_side_keyclip") or row.get("keyclip_paths"))
        if has_keyframe is not None and keyframe_available is not has_keyframe:
            continue
        if has_keyclip is not None and keyclip_available is not has_keyclip:
            continue
        result.append(row)
    return result


def main_refresh_ledger(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default=str(DEFAULT_MATERIAL_ROOT))
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args(argv)
    ledger = build_experiment_action_ledger(args.material_root, args.experiment_id)
    print(json.dumps({"ledger": ledger}, ensure_ascii=False, indent=2))
    return 0


def main_refresh_corpus(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default=str(DEFAULT_MATERIAL_ROOT))
    parser.add_argument("--labvideo-root", default=str(DEFAULT_LABVIDEO_ROOT))
    args = parser.parse_args(argv)
    run_report = run_labvideo_backfill(args.labvideo_root, args.material_root)
    print(json.dumps(run_report, ensure_ascii=False, indent=2))
    return 0


def main_list_ledgers(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default=str(DEFAULT_MATERIAL_ROOT))
    args = parser.parse_args(argv)
    corpus = refresh_labvideo_memory_corpus(args.material_root)
    print(json.dumps(corpus.get("experiment_ledgers", []), ensure_ascii=False, indent=2))
    return 0


def main_query_materials(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--material-root", default=str(DEFAULT_MATERIAL_ROOT))
    parser.add_argument("--action-type")
    parser.add_argument("--experiment-id")
    parser.add_argument("--official-status")
    parser.add_argument("--material-id")
    parser.add_argument("--evidence-bundle-id")
    parser.add_argument("--source-window-sync-index")
    parser.add_argument("--needs-review", action="store_true")
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--keyframe", dest="has_keyframe", action="store_true")
    parser.add_argument("--no-keyframe", dest="has_keyframe", action="store_false")
    parser.set_defaults(has_keyframe=None)
    parser.add_argument("--keyclip", dest="has_keyclip", action="store_true")
    parser.add_argument("--no-keyclip", dest="has_keyclip", action="store_false")
    parser.set_defaults(has_keyclip=None)
    parser.add_argument("--include-orphans", action="store_true")
    args = parser.parse_args(argv)
    official_status = args.official_status
    if args.official and args.needs_review:
        parser.error("--official and --needs-review are mutually exclusive")
    if args.official:
        official_status = "official"
    elif args.needs_review:
        official_status = "needs_review"
    rows = query_materials(
        args.material_root,
        action_type=args.action_type,
        experiment_id=args.experiment_id,
        official_status=official_status,
        material_id=args.material_id,
        evidence_bundle_id=args.evidence_bundle_id,
        source_window_sync_index=args.source_window_sync_index,
        has_keyframe=args.has_keyframe,
        has_keyclip=args.has_keyclip,
        include_orphans=args.include_orphans,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0
