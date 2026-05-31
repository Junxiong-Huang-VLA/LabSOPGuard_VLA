"""Helpers for writing final physical event artifacts through the v2 gate.

The backend still has several legacy producers that create PhysicalEvent-like
objects.  This module keeps those records available for diagnostics while
ensuring the final ``physical_events.json`` artifact only contains gated v2
events.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PHYSICAL_EVENTS_SCHEMA = "physical_events.v4"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    return {"value": repr(value)}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def validate_gated_physical_events_payload(payload: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return whether a payload is allowed to become final physical_events.json."""

    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return False, ["payload_not_mapping"]

    schema = payload.get("schema") or payload.get("schema_version")
    if schema != PHYSICAL_EVENTS_SCHEMA:
        errors.append("unsupported_schema")

    events = payload.get("events")
    if events is None:
        errors.append("missing_events")
        events = []
    if not isinstance(events, list):
        errors.append("events_not_list")
        events = []

    for idx, event in enumerate(events):
        if not isinstance(event, Mapping):
            errors.append(f"event_{idx}_not_mapping")
            continue
        status = event.get("status")
        hard_gate = event.get("hard_gate")
        for field in ("status", "hard_gate", "evidence_detail", "reject_reasons", "limitations"):
            if field not in event:
                errors.append(f"event_{idx}_missing_{field}")
        if status == "confirmed":
            if not isinstance(hard_gate, Mapping):
                errors.append(f"event_{idx}_confirmed_missing_hard_gate")
            elif hard_gate.get("passed") is not True:
                errors.append(f"event_{idx}_confirmed_gate_not_passed")

    return not errors, errors


def empty_gated_physical_events_payload(
    *,
    experiment_id: str | None = None,
    video_id: str | None = None,
    status: str = "failed",
    failure_reason: str = "unknown",
    source: str = "formal_v2",
) -> dict[str, Any]:
    return {
        "schema": PHYSICAL_EVENTS_SCHEMA,
        "schema_version": PHYSICAL_EVENTS_SCHEMA,
        "status": status,
        "failure_reason": failure_reason,
        "events": [],
        "event_count": 0,
        "gate_required": True,
        "source": source,
        "experiment_id": experiment_id,
        "video_id": video_id,
        "created_at": _utc_now(),
    }


def physical_event_gate_summary(
    *,
    status: str = "ok",
    failure_reason: str | None = None,
    qwen_audit_enabled: bool = False,
    qwen_audit_count: int = 0,
    config_path_used: str | None = None,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "confirmed": 0,
        "candidate": 0,
        "rejected": 0,
        "uncertain": 0,
        "qwen_audit_enabled": qwen_audit_enabled,
        "qwen_audit_count": qwen_audit_count,
        "config_path_used": config_path_used,
        "created_at": _utc_now(),
    }
    if failure_reason:
        summary["failure_reason"] = failure_reason
    return summary


def legacy_physical_event_candidate_rows(
    events: Iterable[Any],
    *,
    source: str = "legacy_pipeline",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, event in enumerate(events):
        data = _as_dict(event)
        metadata = data.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        row = {
            **data,
            "candidate_id": data.get("event_id") or f"legacy_{idx:05d}",
            "source": source,
            "status": metadata.get("status") or "ungated_legacy",
            "hard_gate": metadata.get(
                "hard_gate",
                {
                    "passed": False,
                    "gate_name": "legacy_ungated",
                    "required_evidence": ["physical_event_gate"],
                    "passed_evidence": [],
                    "failed_evidence": ["missing_hard_gate"],
                },
            ),
            "evidence_detail": metadata.get("evidence_detail") or metadata,
            "reject_reasons": list(metadata.get("reject_reasons") or ["legacy_ungated_event"]),
            "limitations": list(metadata.get("limitations") or ["not eligible for final physical_events.json"]),
            "legacy_ungated": True,
            "created_at": _utc_now(),
        }
        if row["status"] == "confirmed":
            row["status"] = "ungated_legacy"
        rows.append(row)
    return rows


def write_gated_physical_events(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    source: str = "formal_v2",
) -> bool:
    """Write final physical_events.json only if the payload passes gate checks."""

    out = Path(output_dir)
    valid, errors = validate_gated_physical_events_payload(payload)
    if not valid:
        diagnostic = {
            "source": source,
            "status": "rejected",
            "errors": errors,
            "created_at": _utc_now(),
        }
        _write_json(out / "physical_events_write_rejected.json", diagnostic)
        rejected_rows = legacy_physical_event_candidate_rows(payload.get("events") or [], source=source)
        if rejected_rows:
            _write_jsonl(out / "physical_events_rejected_by_writer.jsonl", rejected_rows)
        return False

    safe_payload = dict(payload)
    safe_payload.setdefault("schema", PHYSICAL_EVENTS_SCHEMA)
    safe_payload.setdefault("schema_version", PHYSICAL_EVENTS_SCHEMA)
    safe_payload["gate_required"] = True
    safe_payload["event_count"] = len(safe_payload.get("events") or [])
    safe_payload.setdefault("source", source)
    _write_json(out / "physical_events.json", safe_payload)
    return True

