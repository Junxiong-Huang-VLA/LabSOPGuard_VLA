from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl, write_jsonl
from .sop_state_machine import build_sop_state_machine


SCHEMA_VERSION = "lightweight_context_import/v1"


def import_lightweight_context(
    session_dir: str | Path,
    *,
    sop_text: str | None = None,
    note_text: str | None = None,
    record_text: str | None = None,
    output_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    sop_rows = _sop_rows_from_text(sop_text or "", session_id=_session_id(session), source_kind="sop_text")
    user_rows = [
        *_user_text_rows(note_text or "", session_id=_session_id(session), source_kind="note_text", event_type="manual_note"),
        *_user_text_rows(record_text or "", session_id=_session_id(session), source_kind="record_text", event_type="record_text"),
    ]

    sop_path = metadata / "sop_records.jsonl"
    user_text_path = metadata / "user_text_events.jsonl"
    if sop_rows:
        write_jsonl(sop_path, _upsert_rows(_read_jsonl_if_exists(sop_path), sop_rows, key="record_id"))
    elif not sop_path.exists():
        write_jsonl(sop_path, [])
    if user_rows:
        write_jsonl(user_text_path, _upsert_rows(_read_jsonl_if_exists(user_text_path), user_rows, key="event_id"))
    elif not user_text_path.exists():
        write_jsonl(user_text_path, [])

    summary_path = Path(output_summary_path) if output_summary_path is not None else metadata / "lightweight_context_import_summary.json"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "session_id": _session_id(session),
        "sop_records": str(sop_path),
        "user_text_events": str(user_text_path),
        "imported_sop_record_count": len(sop_rows),
        "imported_user_text_event_count": len(user_rows),
        "imported_sources": {
            "sop_text": bool((sop_text or "").strip()),
            "note_text": bool((note_text or "").strip()),
            "record_text": bool((record_text or "").strip()),
        },
        "evidence_policy": "text anchors only; not visual evidence and not strong process evidence",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def _sop_rows_from_text(text: str, *, session_id: str, source_kind: str) -> list[dict[str, Any]]:
    clean = text.strip()
    if not clean:
        return []
    source_hash = _source_hash(source_kind, clean)
    machine = build_sop_state_machine({"sop_text": clean})
    rows = []
    for index, step in enumerate(machine.get("steps") or [], start=1):
        if not isinstance(step, Mapping):
            continue
        body = str(step.get("description") or step.get("name") or "").strip()
        expected = _refine_expected_action(step.get("expected_action"), body)
        row = {
            "record_id": f"sop_text_{source_hash}_{index:03d}",
            "session_id": session_id,
            "source_type": "lightweight_context",
            "source_kind": source_kind,
            "event_type": "sop_record",
            "step_id": f"sop_ctx_{source_hash}_{index:03d}",
            "step_order": index,
            "name": str(step.get("name") or expected or f"step_{index:03d}"),
            "action_type": expected,
            "expected_action": expected,
            "materials": [],
            "parameters": step.get("parameters") or [],
            "completion_conditions": step.get("completion_conditions") or [{"any_actions": [expected]}],
            "evidence_requirements": step.get("evidence_requirements") or [{"type": "video_action", "action": expected}],
            "text": body,
            "confidence": 1.0,
            "evidence_source": "user_supplied_sop_text",
            "evidence_level": "text_support",
            "not_visual_evidence": True,
            "process_eligible": False,
            "payload": {
                "schema_version": SCHEMA_VERSION,
                "source_hash": source_hash,
                "source_kind": source_kind,
                "raw_step": dict(step),
            },
        }
        row["search_text"] = _search_text(row)
        rows.append(row)
    return rows


def _user_text_rows(text: str, *, session_id: str, source_kind: str, event_type: str) -> list[dict[str, Any]]:
    clean = text.strip()
    if not clean:
        return []
    source_hash = _source_hash(source_kind, clean)
    row = {
        "event_id": f"user_text_{source_kind}_{source_hash}",
        "timeline_event_id": f"user_text_{source_kind}_{source_hash}",
        "session_id": session_id,
        "source_type": "lightweight_context",
        "source_kind": source_kind,
        "event_type": event_type,
        "text": clean,
        "content": clean,
        "confidence": 0.65,
        "evidence_source": "user_supplied_text",
        "evidence_level": "text_support",
        "not_visual_evidence": True,
        "process_eligible": False,
        "payload": {
            "schema_version": SCHEMA_VERSION,
            "source_hash": source_hash,
            "source_kind": source_kind,
        },
    }
    row["search_text"] = _search_text(row)
    return [row]


def _upsert_rows(existing: list[dict[str, Any]], incoming: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    by_key = {str(row.get(key) or ""): dict(row) for row in existing if row.get(key)}
    passthrough = [dict(row) for row in existing if not row.get(key)]
    for row in incoming:
        by_key[str(row.get(key) or "")] = dict(row)
    return [*passthrough, *[by_key[item] for item in sorted(by_key)]]


def _refine_expected_action(value: Any, text: str) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ("record", "readout", "note", "记录", "读数")):
        return "recording"
    if any(token in lowered for token in ("pipette", "transfer", "liquid transfer", "200 ul", "200ul", "移液", "加样")):
        return "pipetting"
    if any(token in lowered for token in ("weigh", "balance", "mass", "称量", "天平")):
        return "weighing"
    return str(value or "")


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _source_hash(kind: str, text: str) -> str:
    return hashlib.sha1(f"{kind}\n{text}".encode("utf-8")).hexdigest()[:12]


def _search_text(row: Mapping[str, Any]) -> str:
    parts = [
        row.get("source_kind"),
        row.get("event_type"),
        row.get("action_type"),
        row.get("expected_action"),
        row.get("name"),
        row.get("text"),
        json.dumps(row.get("parameters", []), ensure_ascii=False, sort_keys=True),
    ]
    return " ".join(str(item) for item in parts if item)


def _session_id(session: Path) -> str:
    manifest = session / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            if data.get("session_id"):
                return str(data["session_id"])
        except (OSError, json.JSONDecodeError):
            pass
    return session.name


__all__ = ["SCHEMA_VERSION", "import_lightweight_context"]
