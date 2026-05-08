from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl, write_jsonl


QUEUE_FILENAME = "human_confirmation_queue.jsonl"
DECISIONS_FILENAME = "human_confirmation_decisions.jsonl"
REVIEW_BUNDLE_FILENAME = "human_confirmation_review_bundle.json"
AUDIT_TRAIL_FILENAME = "human_confirmation_audit_trail.jsonl"
REVIEW_SUMMARY_FILENAME = "human_confirmation_review_summary.json"
BATCH_RESULT_FILENAME = "human_confirmation_batch_result.json"
MACHINE_BACKLOG_FILENAME = "human_confirmation_machine_backlog.jsonl"
LOW_SIGNAL_VIDEO_CANDIDATE_TYPES = {
    "container_state_change_candidate",
    "equipment_panel_operation_candidate",
    "liquid_flow_candidate_visual",
    "liquid_transfer_candidate",
    "object_movement_candidate",
}


def build_confirmation_queue(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    process_path = metadata / "experiment_process.json"
    process = _read_json(process_path)
    session_id = str(process.get("session_id") or session.name)
    existing_decisions = _decisions_by_id(metadata / DECISIONS_FILENAME)
    replay_result = _apply_decisions_to_process_dict(process, existing_decisions.values())
    if replay_result["changed"]:
        process_path.write_text(json.dumps(process, ensure_ascii=False, indent=2), encoding="utf-8")
    context = _load_review_context(session)
    created_at = _now()
    rows = []
    backlog_rows = []
    bundle_items = []
    step_video_event_ids = _step_video_event_ids(process)
    for step in process.get("steps", []):
        if not isinstance(step, Mapping) or not step.get("requires_human_confirmation"):
            continue
        confirmation_id = _confirmation_id(session_id, step)
        decision = existing_decisions.get(confirmation_id, {})
        review_item = _build_review_item(session, session_id, step, confirmation_id, decision, context, created_at)
        rows.append(_queue_row_from_review_item(review_item, step, decision, created_at))
        bundle_items.append(review_item)
    for event in _video_events_requiring_confirmation(context):
        event_id = str(event.get("video_event_id") or "")
        confirmation_id = f"{session_id}:video_event:{event_id}"
        if _video_event_aliases(event) & step_video_event_ids:
            backlog_rows.append(_backlog_row_from_video_event(session_id, event, created_at, reason="covered_by_step_review"))
            continue
        decision = existing_decisions.get(confirmation_id, {})
        review_item = _build_video_event_review_item(session, session_id, event, confirmation_id, decision, context, created_at)
        rows.append(_queue_row_from_video_review_item(review_item, event, decision, created_at))
        bundle_items.append(review_item)
    target = Path(output_path) if output_path is not None else metadata / QUEUE_FILENAME
    write_jsonl(target, rows)
    backlog_path = metadata / MACHINE_BACKLOG_FILENAME
    write_jsonl(backlog_path, backlog_rows)
    review_bundle_path = metadata / REVIEW_BUNDLE_FILENAME
    _write_json(
        review_bundle_path,
        _review_bundle_payload(
            session_id=session_id,
            process_path=process_path,
            queue_path=target,
            items=bundle_items,
            created_at=created_at,
        ),
    )
    review_summary = build_confirmation_review_summary(
        session,
        output_path=metadata / REVIEW_SUMMARY_FILENAME,
        process=process,
        context=context,
        generated_at=created_at,
    )
    return {
        "session_id": session_id,
        "queue_path": str(target),
        "review_bundle_path": str(review_bundle_path),
        "audit_trail_path": str(metadata / AUDIT_TRAIL_FILENAME),
        "review_summary_path": str(metadata / REVIEW_SUMMARY_FILENAME),
        "machine_backlog_path": str(backlog_path),
        "pending_count": sum(1 for row in rows if row["status"] == "pending"),
        "resolved_count": sum(1 for row in rows if row["status"] != "pending"),
        "item_count": len(rows),
        "step_item_count": sum(1 for row in rows if row.get("item_type") == "experiment_step"),
        "standalone_video_item_count": sum(1 for row in rows if row.get("item_type") == "video_event"),
        "machine_backlog_count": len(backlog_rows),
        "audit_event_count": review_summary.get("audit_event_count", 0),
    }


def list_confirmation_queue(session_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(session_dir) / "metadata" / QUEUE_FILENAME
    return read_jsonl(path) if path.exists() else []


def apply_confirmation_decision(
    session_dir: str | Path,
    confirmation_id: str,
    decision: str,
    reviewer: str = "system",
    note: str = "",
) -> dict[str, Any]:
    normalized = str(decision or "").strip().lower()
    if normalized not in {"approved", "rejected", "needs_review"}:
        raise ValueError("decision must be one of: approved, rejected, needs_review")
    session = Path(session_dir)
    metadata = session / "metadata"
    decisions_path = metadata / DECISIONS_FILENAME
    process_path = metadata / "experiment_process.json"
    process_before = _read_json(process_path)
    session_id = str(process_before.get("session_id") or session.name)
    step_before = _find_step(process_before, confirmation_id)
    context = _load_review_context(session)
    previous_decisions = read_jsonl(decisions_path) if decisions_path.exists() else []
    previous_decision = next(
        (row for row in reversed(previous_decisions) if row.get("confirmation_id") == confirmation_id),
        None,
    )
    decided_at = _now()
    audit_id = _audit_id(confirmation_id, decided_at)
    pre_decision_review = (
        _build_review_item(session, session_id, step_before, confirmation_id, previous_decision or {}, context, decided_at)
        if step_before is not None
        else None
    )
    decisions = read_jsonl(decisions_path) if decisions_path.exists() else []
    record = {
        "audit_id": audit_id,
        "confirmation_id": confirmation_id,
        "decision": normalized,
        "reviewer": reviewer,
        "note": note,
        "decided_at": decided_at,
    }
    decisions = [row for row in decisions if row.get("confirmation_id") != confirmation_id]
    decisions.append(record)
    write_jsonl(decisions_path, decisions)
    apply_result = _apply_decision_to_process(process_path, record)
    audit_record = _audit_record(
        audit_id=audit_id,
        session_id=session_id,
        confirmation_id=confirmation_id,
        decision=record,
        previous_decision=previous_decision,
        before_state=apply_result.get("before_state") or _step_state(step_before),
        after_state=apply_result.get("after_state"),
        changed=bool(apply_result.get("changed")),
        review_item=pre_decision_review,
    )
    _append_audit_record(metadata / AUDIT_TRAIL_FILENAME, audit_record)
    queue_summary = build_confirmation_queue(session)
    return {"decision": record, "audit": audit_record, "queue_summary": queue_summary}


def apply_confirmation_batch_decisions(
    session_dir: str | Path,
    decisions_path: str | Path,
    output_path: str | Path | None = None,
    *,
    reviewer: str = "system",
    note: str = "",
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    process_path = metadata / "experiment_process.json"
    process = _read_json(process_path)
    session_id = str(process.get("session_id") or session.name)
    source = Path(decisions_path)
    raw_rows, batch_defaults = _load_batch_decision_rows(source)
    rows = [
        _normalize_batch_decision_row(row, session_id, reviewer=reviewer, note=note, defaults=batch_defaults)
        for row in raw_rows
    ]
    valid_confirmation_ids = {
        _confirmation_id(session_id, step)
        for step in process.get("steps", [])
        if isinstance(step, Mapping) and step.get("step_id") is not None
    }
    enforce_confirmation_ids = process_path.exists()

    started_at = _now()
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        confirmation_id = str(row.get("confirmation_id") or "")
        decision = str(row.get("decision") or "")
        item_result = {
            "index": index,
            "confirmation_id": confirmation_id,
            "decision": decision,
            "reviewer": row.get("reviewer"),
            "note": row.get("note"),
            "applied": False,
            "changed": False,
            "audit_id": None,
            "before_state": None,
            "after_state": None,
            "error": None,
        }
        if row.get("error"):
            item_result["error"] = row["error"]
            results.append(item_result)
            continue
        if enforce_confirmation_ids and confirmation_id not in valid_confirmation_ids:
            item_result["error"] = "confirmation_id_not_found"
            results.append(item_result)
            continue
        try:
            applied = apply_confirmation_decision(
                session,
                confirmation_id=confirmation_id,
                decision=decision,
                reviewer=str(row.get("reviewer") or reviewer or "system"),
                note=str(row.get("note") or ""),
            )
        except ValueError as exc:
            item_result["error"] = str(exc)
            results.append(item_result)
            continue
        audit = _as_dict(applied.get("audit"))
        item_result.update(
            {
                "applied": True,
                "changed": bool(audit.get("changed")),
                "audit_id": audit.get("audit_id"),
                "before_state": audit.get("before_state"),
                "after_state": audit.get("after_state"),
                "previous_decision": audit.get("previous_decision"),
            }
        )
        results.append(item_result)

    queue_summary = build_confirmation_queue(session)
    review_summary = build_confirmation_review_summary(session)
    applied_results = [row for row in results if row.get("applied")]
    failed_results = [row for row in results if not row.get("applied")]
    result = {
        "schema_version": "human_confirmation_batch_result/v1",
        "session_id": session_id,
        "source_decisions_path": str(source),
        "generated_at": _now(),
        "started_at": started_at,
        "input_count": len(rows),
        "applied_count": len(applied_results),
        "failed_count": len(failed_results),
        "decision_counts": dict(sorted(Counter(str(row.get("decision") or "unknown") for row in applied_results).items())),
        "error_counts": dict(sorted(Counter(str(row.get("error") or "unknown") for row in failed_results).items())),
        "results": results,
        "updated_review_state": {
            "queue_summary": queue_summary,
            "review_summary_path": str(metadata / REVIEW_SUMMARY_FILENAME),
            "latest_decision_counts": review_summary.get("latest_decision_counts", {}),
            "confirmation_item_count": review_summary.get("confirmation_item_count", 0),
            "audit_event_count": review_summary.get("audit_event_count", 0),
        },
        "paths": {
            "batch_result": str(Path(output_path) if output_path is not None else metadata / BATCH_RESULT_FILENAME),
            "decisions": str(metadata / DECISIONS_FILENAME),
            "audit_trail": str(metadata / AUDIT_TRAIL_FILENAME),
            "queue": str(metadata / QUEUE_FILENAME),
            "review_bundle": str(metadata / REVIEW_BUNDLE_FILENAME),
            "review_summary": str(metadata / REVIEW_SUMMARY_FILENAME),
        },
    }
    target = Path(output_path) if output_path is not None else metadata / BATCH_RESULT_FILENAME
    _write_json(target, result)
    return result


def build_confirmation_review_bundle(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    process_path = metadata / "experiment_process.json"
    process = _read_json(process_path)
    session_id = str(process.get("session_id") or session.name)
    existing_decisions = _decisions_by_id(metadata / DECISIONS_FILENAME)
    context = _load_review_context(session)
    created_at = _now()
    items = []
    for step in process.get("steps", []):
        if not isinstance(step, Mapping) or not step.get("requires_human_confirmation"):
            continue
        confirmation_id = _confirmation_id(session_id, step)
        items.append(
            _build_review_item(
                session,
                session_id,
                step,
                confirmation_id,
                existing_decisions.get(confirmation_id, {}),
                context,
                created_at,
            )
        )
    target = Path(output_path) if output_path is not None else metadata / REVIEW_BUNDLE_FILENAME
    bundle = _review_bundle_payload(
        session_id=session_id,
        process_path=process_path,
        queue_path=metadata / QUEUE_FILENAME,
        items=items,
        created_at=created_at,
    )
    _write_json(target, bundle)
    return bundle


def build_confirmation_review_summary(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    process: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    process_data = dict(process) if isinstance(process, Mapping) else _read_json(metadata / "experiment_process.json")
    session_id = str(process_data.get("session_id") or session.name)
    review_context = dict(context) if isinstance(context, Mapping) else _load_review_context(session)
    decisions = _decisions_by_id(metadata / DECISIONS_FILENAME)
    audit_rows = _read_jsonl_if_exists(metadata / AUDIT_TRAIL_FILENAME)
    queue_rows = _read_jsonl_if_exists(metadata / QUEUE_FILENAME)
    backlog_rows = _read_jsonl_if_exists(metadata / MACHINE_BACKLOG_FILENAME)
    audits_by_confirmation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        confirmation_key = str(row.get("confirmation_id") or "")
        if confirmation_key:
            audits_by_confirmation[confirmation_key].append(row)

    timestamp = generated_at or _now()
    step_entries: list[dict[str, Any]] = []
    evidence_entries: dict[str, dict[str, Any]] = {}
    latest_decision_counts: Counter[str] = Counter()
    for step in process_data.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        confirmation_id = _confirmation_id(session_id, step)
        latest_decision = decisions.get(confirmation_id, {})
        latest_decision_value = str(latest_decision.get("decision") or step.get("confirmation_status") or "unreviewed")
        latest_decision_counts[latest_decision_value] += 1
        review_item = _build_review_item(
            session,
            session_id,
            step,
            confirmation_id,
            latest_decision,
            review_context,
            timestamp,
        )
        audit_history = audits_by_confirmation.get(confirmation_id, [])
        step_entry = {
            "confirmation_id": confirmation_id,
            "step_id": step.get("step_id"),
            "name": step.get("name"),
            "expected_action": step.get("expected_action"),
            "status": step.get("status"),
            "confirmation_status": step.get("confirmation_status"),
            "requires_human_confirmation": bool(step.get("requires_human_confirmation")),
            "completed": step.get("completed"),
            "skipped": step.get("skipped"),
            "inferred": step.get("inferred"),
            "abnormal": step.get("abnormal"),
            "confidence": step.get("confidence"),
            "latest_decision": _public_decision(latest_decision),
            "decision_count": len(audit_history),
            "last_audit": _public_audit(audit_history[-1]) if audit_history else {},
            "evidence_summary": review_item["evidence_summary"],
            "keyframe_refs": review_item["keyframe_refs"],
            "clip_refs": review_item["clip_refs"],
            "segment_refs": review_item["segment_refs"],
            "suggested_action": review_item["suggested_action"],
        }
        step_entries.append(step_entry)
        for evidence_ref in review_item["resolved_evidence_refs"]:
            key = str(evidence_ref.get("evidence_key") or "")
            if not key:
                continue
            entry = evidence_entries.setdefault(
                key,
                {
                    "evidence_key": key,
                    "type": evidence_ref.get("type"),
                    "resolved": bool(evidence_ref.get("resolved")),
                    "event_type": evidence_ref.get("event_type"),
                    "state_type": evidence_ref.get("state_type"),
                    "asset_type": evidence_ref.get("asset_type"),
                    "path": evidence_ref.get("path"),
                    "segment_id": evidence_ref.get("segment_id"),
                    "micro_segment_id": evidence_ref.get("micro_segment_id"),
                    "confidence": evidence_ref.get("confidence"),
                    "step_ids": [],
                    "confirmation_ids": [],
                    "latest_decisions": [],
                    "keyframe_refs": [],
                    "clip_refs": [],
                },
            )
            _append_unique(entry["step_ids"], str(step.get("step_id") or ""))
            _append_unique(entry["confirmation_ids"], confirmation_id)
            if latest_decision:
                _append_unique(entry["latest_decisions"], str(latest_decision.get("decision") or ""))
            entry["keyframe_refs"].extend(_as_list(evidence_ref.get("keyframe_refs")))
            entry["clip_refs"].extend(_as_list(evidence_ref.get("clip_refs")))
            for keyframe_ref in review_item["keyframe_refs"]:
                if _ref_matches_evidence(keyframe_ref, evidence_ref):
                    entry["keyframe_refs"].append(keyframe_ref)
            for clip_ref in review_item["clip_refs"]:
                if _ref_matches_evidence(clip_ref, evidence_ref):
                    entry["clip_refs"].append(clip_ref)
            entry["keyframe_refs"] = _dedupe_media_refs(entry["keyframe_refs"])
            entry["clip_refs"] = _dedupe_media_refs(entry["clip_refs"])

    summary = {
        "schema_version": "confirmation_review_summary/v1",
        "session_id": session_id,
        "generated_at": timestamp,
        "step_count": len(step_entries),
        "confirmation_item_count": sum(1 for row in step_entries if row["requires_human_confirmation"]),
        "queue_item_count": len(queue_rows),
        "queue_step_item_count": sum(1 for row in queue_rows if row.get("item_type") == "experiment_step"),
        "queue_video_item_count": sum(1 for row in queue_rows if row.get("item_type") == "video_event"),
        "machine_backlog_count": len(backlog_rows),
        "audit_event_count": len(audit_rows),
        "latest_decision_counts": dict(sorted(latest_decision_counts.items())),
        "steps": step_entries,
        "evidence": sorted(evidence_entries.values(), key=lambda row: str(row.get("evidence_key") or "")),
        "paths": {
            "queue": str(metadata / QUEUE_FILENAME),
            "decisions": str(metadata / DECISIONS_FILENAME),
            "audit_trail": str(metadata / AUDIT_TRAIL_FILENAME),
            "review_bundle": str(metadata / REVIEW_BUNDLE_FILENAME),
            "machine_backlog": str(metadata / MACHINE_BACKLOG_FILENAME),
        },
    }
    target = Path(output_path) if output_path is not None else metadata / REVIEW_SUMMARY_FILENAME
    _write_json(target, summary)
    return summary


def resolve_review_evidence_for_step(
    session_dir: str | Path,
    step: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    review_context = context if isinstance(context, Mapping) else _load_review_context(session)
    return _resolve_step_evidence(step, review_context)


def _apply_decision_to_process(process_path: Path, decision: Mapping[str, Any]) -> dict[str, Any]:
    process = _read_json(process_path)
    result = _apply_decision_to_process_dict(process, decision)
    if result["changed"]:
        _refresh_process_rollup(process)
        process_path.write_text(json.dumps(process, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _apply_decisions_to_process_dict(process: dict[str, Any], decisions: Any) -> dict[str, Any]:
    changed = False
    before_states: dict[str, Any] = {}
    after_states: dict[str, Any] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            continue
        result = _apply_decision_to_process_dict(process, decision)
        if result["changed"]:
            changed = True
            confirmation_id = str(decision.get("confirmation_id") or "")
            before_states[confirmation_id] = result.get("before_state")
            after_states[confirmation_id] = result.get("after_state")
    if changed:
        _refresh_process_rollup(process)
    return {"changed": changed, "before_states": before_states, "after_states": after_states}


def _apply_decision_to_process_dict(process: dict[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    confirmation_id = str(decision.get("confirmation_id") or "")
    step_id = confirmation_id.rsplit(":", 1)[-1]
    normalized = str(decision.get("decision") or "").strip().lower()
    changed = False
    before_state: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    for step in process.get("steps", []):
        if not isinstance(step, dict) or str(step.get("step_id")) != step_id:
            continue
        before_state = _step_state(step)
        step["confirmation_status"] = normalized
        step["confirmation_decision"] = dict(decision)
        if normalized == "approved":
            step["requires_human_confirmation"] = False
            if step.get("inferred") or step.get("status") in {"inferred_missing", "skipped_or_unobserved", "not_observed"}:
                step["completed"] = True
                step["status"] = "human_confirmed"
        elif normalized == "rejected":
            step["completed"] = False
            step["status"] = "human_rejected"
            step["requires_human_confirmation"] = False
        elif normalized == "needs_review":
            step["requires_human_confirmation"] = True
        changed = _step_state(step) != before_state
        after_state = _step_state(step)
    return {"changed": changed, "before_state": before_state, "after_state": after_state}


def _load_batch_decision_rows(path: Path) -> tuple[list[Any], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"batch decisions file not found: {path}")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return read_jsonl(path), {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, Mapping):
        return [payload], {}

    defaults = {
        "reviewer": payload.get("reviewer") or payload.get("decider") or payload.get("approved_by"),
        "note": payload.get("note") or payload.get("comment") or payload.get("reason"),
    }
    if isinstance(payload.get("decisions"), list):
        return list(payload.get("decisions") or []), defaults
    if _looks_like_decision_row(payload):
        return [payload], defaults

    rows: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"reviewer", "decider", "approved_by", "note", "comment", "reason"}:
            continue
        if isinstance(value, Mapping):
            rows.append({"confirmation_id": key, **dict(value)})
        else:
            rows.append({"confirmation_id": key, "decision": value})
    return rows, defaults


def _normalize_batch_decision_row(
    row: Any,
    session_id: str,
    *,
    reviewer: str,
    note: str,
    defaults: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        return {"source": row, "error": "decision_row_must_be_object"}
    source = dict(row)
    confirmation_id = str(source.get("confirmation_id") or source.get("id") or "").strip()
    step_id = str(source.get("step_id") or source.get("item_id") or "").strip()
    if not confirmation_id and step_id:
        confirmation_id = step_id if ":" in step_id else f"{session_id}:{step_id}"
    if confirmation_id and ":" not in confirmation_id and step_id == "":
        confirmation_id = f"{session_id}:{confirmation_id}"
    decision = str(source.get("decision") or source.get("status") or source.get("action") or "").strip().lower()
    normalized = {
        "confirmation_id": confirmation_id,
        "decision": decision,
        "reviewer": source.get("reviewer")
        or source.get("decider")
        or source.get("approved_by")
        or defaults.get("reviewer")
        or reviewer
        or "system",
        "note": source.get("note") or source.get("comment") or source.get("reason") or defaults.get("note") or note or "",
        "source": source,
    }
    if not confirmation_id:
        normalized["error"] = "missing_confirmation_id"
    elif decision not in {"approved", "rejected", "needs_review"}:
        normalized["error"] = "invalid_decision"
    return normalized


def _looks_like_decision_row(row: Mapping[str, Any]) -> bool:
    return any(key in row for key in ("confirmation_id", "step_id", "item_id", "decision", "status", "action"))


def _refresh_process_rollup(process: dict[str, Any]) -> None:
    steps = [step for step in process.get("steps", []) if isinstance(step, Mapping)]
    process["step_count"] = len(steps)
    process["status_counts"] = dict(sorted(Counter(str(step.get("status") or "unknown") for step in steps).items()))
    process["completed_step_ids"] = [str(step.get("step_id")) for step in steps if step.get("completed") and step.get("step_id") is not None]
    process["pending_confirmation_step_ids"] = [
        str(step.get("step_id"))
        for step in steps
        if step.get("requires_human_confirmation") and step.get("step_id") is not None
    ]
    process["human_rejected_step_ids"] = [
        str(step.get("step_id"))
        for step in steps
        if str(step.get("status") or "") == "human_rejected" and step.get("step_id") is not None
    ]
    process["process_status"] = (
        "completed"
        if steps and all(step.get("completed") or step.get("skipped") for step in steps)
        else "in_progress"
    )
    current = next((step for step in steps if not step.get("completed") and not step.get("skipped")), None)
    if current is None and steps:
        process["current_step_id"] = steps[-1].get("step_id")
        process["next_step_id"] = None
    else:
        process["current_step_id"] = current.get("step_id") if current else None
        process["next_step_id"] = current.get("step_id") if current else None


def _queue_row_from_review_item(
    review_item: Mapping[str, Any],
    step: Mapping[str, Any],
    decision: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    return {
        "confirmation_id": review_item.get("confirmation_id"),
        "session_id": review_item.get("session_id"),
        "item_type": "experiment_step",
        "item_id": step.get("step_id"),
        "status": decision.get("decision") or "pending",
        "reason": review_item.get("reason"),
        "confidence": step.get("confidence"),
        "summary": step.get("name"),
        "proposed_update": {
            "completed": step.get("completed"),
            "skipped": step.get("skipped"),
            "inferred": step.get("inferred"),
            "abnormal": step.get("abnormal"),
        },
        "evidence_refs": step.get("evidence_refs") or [],
        "evidence_summary": review_item.get("evidence_summary"),
        "keyframe_refs": review_item.get("keyframe_refs"),
        "clip_refs": review_item.get("clip_refs"),
        "segment_refs": review_item.get("segment_refs"),
        "suggested_action": review_item.get("suggested_action"),
        "review_bundle_item_id": review_item.get("review_item_id"),
        "created_at": created_at,
        "decision": dict(decision),
    }


def _queue_row_from_video_review_item(
    review_item: Mapping[str, Any],
    event: Mapping[str, Any],
    decision: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    return {
        "confirmation_id": review_item.get("confirmation_id"),
        "session_id": review_item.get("session_id"),
        "item_type": "video_event",
        "item_id": event.get("video_event_id"),
        "status": decision.get("decision") or "pending",
        "reason": review_item.get("reason"),
        "confidence": event.get("confidence"),
        "summary": event.get("semantic_description") or event.get("text"),
        "proposed_update": {
            "conclusion_status": event.get("conclusion_status"),
            "event_type": event.get("event_type"),
            "anomaly_flags": event.get("anomaly_flags") or [],
        },
        "evidence_refs": event.get("evidence_refs") or [],
        "evidence_summary": review_item.get("evidence_summary"),
        "keyframe_refs": review_item.get("keyframe_refs"),
        "clip_refs": review_item.get("clip_refs"),
        "segment_refs": review_item.get("segment_refs"),
        "suggested_action": review_item.get("suggested_action"),
        "review_bundle_item_id": review_item.get("review_item_id"),
        "created_at": created_at,
        "decision": dict(decision),
    }


def _review_bundle_payload(
    *,
    session_id: str,
    process_path: Path,
    queue_path: Path,
    items: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    status_counts = Counter(str(item.get("status") or "pending") for item in items)
    return {
        "schema_version": "confirmation_review_bundle/v1",
        "session_id": session_id,
        "created_at": created_at,
        "source_process_path": str(process_path),
        "queue_path": str(queue_path),
        "item_count": len(items),
        "status_counts": dict(sorted(status_counts.items())),
        "items": items,
    }


def _build_review_item(
    session: Path,
    session_id: str,
    step: Mapping[str, Any],
    confirmation_id: str,
    decision: Mapping[str, Any],
    context: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    evidence_payload = _resolve_step_evidence(step, context)
    evidence_summary = _summarize_evidence(step, evidence_payload)
    suggested_action = _suggested_action(step, evidence_summary)
    return {
        "schema_version": "confirmation_review_item/v1",
        "review_item_id": confirmation_id,
        "confirmation_id": confirmation_id,
        "session_id": session_id,
        "session_dir": str(session),
        "item_type": "experiment_step",
        "item_id": step.get("step_id"),
        "status": decision.get("decision") or "pending",
        "reason": _confirmation_reason(step),
        "created_at": created_at,
        "step": _step_review_payload(step),
        "decision": dict(decision),
        "evidence_summary": evidence_summary,
        "resolved_evidence_refs": evidence_payload["resolved_evidence_refs"],
        "keyframe_refs": evidence_payload["keyframe_refs"],
        "clip_refs": evidence_payload["clip_refs"],
        "segment_refs": evidence_payload["segment_refs"],
        "suggested_action": suggested_action,
    }


def _build_video_event_review_item(
    session: Path,
    session_id: str,
    event: Mapping[str, Any],
    confirmation_id: str,
    decision: Mapping[str, Any],
    context: Mapping[str, Any],
    created_at: str,
) -> dict[str, Any]:
    source_ref = {"type": "video_event", "video_event_id": event.get("video_event_id"), "confidence": event.get("confidence")}
    evidence_entry = _video_event_evidence_entry(str(event.get("video_event_id") or ""), source_ref, event)
    media_refs = _media_refs_from_video_event(event, context)
    keyframe_refs = [ref for ref in media_refs if ref.get("ref_type") == "keyframe"]
    clip_refs = [ref for ref in media_refs if ref.get("ref_type") == "clip"]
    segment_refs = _segment_refs_from_video_event(event, context)
    evidence_summary = {
        "text": event.get("semantic_description") or event.get("text"),
        "evidence_ref_count": len(event.get("evidence_refs") or []),
        "resolved_ref_count": 1,
        "video_event_type_counts": {str(event.get("event_type") or "unknown"): 1},
        "objects": [str(event.get("primary_object") or "")],
        "action_types": [str(event.get("action_type") or "")],
        "confidence": {"max": event.get("confidence"), "avg": event.get("confidence")},
        "limitations": event.get("anomaly_flags") or [],
    }
    return {
        "schema_version": "confirmation_review_item/v1",
        "review_item_id": confirmation_id,
        "confirmation_id": confirmation_id,
        "session_id": session_id,
        "session_dir": str(session),
        "item_type": "video_event",
        "item_id": event.get("video_event_id"),
        "status": decision.get("decision") or "pending",
        "reason": _video_event_confirmation_reason(event),
        "created_at": created_at,
        "video_event": dict(event),
        "decision": dict(decision),
        "evidence_summary": evidence_summary,
        "resolved_evidence_refs": [evidence_entry],
        "keyframe_refs": _dedupe_media_refs(keyframe_refs),
        "clip_refs": _dedupe_media_refs(clip_refs),
        "segment_refs": _dedupe_segment_refs(segment_refs),
        "suggested_action": {"decision": "needs_review", "reason": "candidate_or_anomalous_video_event"},
    }


def _backlog_row_from_video_event(session_id: str, event: Mapping[str, Any], created_at: str, *, reason: str) -> dict[str, Any]:
    event_id = str(event.get("video_event_id") or "")
    flags = [str(flag) for flag in _as_list(event.get("anomaly_flags")) if flag]
    return {
        "schema_version": "confirmation_machine_backlog/v1",
        "session_id": session_id,
        "item_type": "video_event",
        "item_id": event_id,
        "reason": reason,
        "created_at": created_at,
        "status": "machine_backlog",
        "event_type": event.get("event_type"),
        "conclusion_status": event.get("conclusion_status"),
        "confidence": event.get("confidence"),
        "primary_object": event.get("primary_object"),
        "action_type": event.get("action_type"),
        "anomaly_flags": flags,
        "summary": event.get("semantic_description") or event.get("text"),
    }


def _resolve_step_evidence(step: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    resolved: list[dict[str, Any]] = []
    keyframe_refs: list[dict[str, Any]] = []
    clip_refs: list[dict[str, Any]] = []
    segment_refs: list[dict[str, Any]] = []
    for source_ref in _normalize_evidence_refs(step.get("evidence_refs")):
        ref_type = _evidence_ref_type(source_ref)
        if ref_type == "video_event":
            event_id = _first_ref_id(source_ref, "video_event_id", "event_id", "evidence_id", "id")
            event = context.get("video_events_by_id", {}).get(event_id)
            entry = _video_event_evidence_entry(event_id, source_ref, event)
            resolved.append(entry)
            if isinstance(event, Mapping):
                media_refs = _media_refs_from_video_event(event, context)
                keyframe_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "keyframe")
                clip_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "clip")
                segment_refs.extend(_segment_refs_from_video_event(event, context))
        elif ref_type == "state_change":
            state_id = _first_ref_id(source_ref, "state_change_id", "event_id", "evidence_id", "id")
            state = context.get("states_by_id", {}).get(state_id)
            entry = _state_evidence_entry(state_id, source_ref, state)
            resolved.append(entry)
            if isinstance(state, Mapping):
                media_refs = _media_refs_from_asset_refs(
                    _as_list(state.get("asset_refs")),
                    source="state_change",
                    segment_id=state.get("segment_id"),
                    micro_segment_id=state.get("micro_segment_id"),
                    evidence_key=entry.get("evidence_key"),
                )
                media_refs.extend(_media_refs_from_related_segment(state, context, source="state_change", evidence_key=entry.get("evidence_key")))
                keyframe_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "keyframe")
                clip_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "clip")
                segment_refs.extend(_segment_refs_from_state(state, context))
        elif ref_type == "asset":
            asset_id = _first_ref_id(source_ref, "asset_id", "event_id", "evidence_id", "id")
            asset = context.get("assets_by_id", {}).get(asset_id)
            if not asset and source_ref.get("path"):
                asset = context.get("assets_by_path", {}).get(str(source_ref.get("path")))
            entry = _asset_evidence_entry(asset_id, source_ref, asset)
            resolved.append(entry)
            media_refs = _media_refs_from_asset_refs([asset or source_ref], source="asset_ref", evidence_key=entry.get("evidence_key"))
            if isinstance(asset, Mapping):
                media_refs.extend(_media_refs_from_related_segment(asset, context, source="asset_ref", evidence_key=entry.get("evidence_key")))
            keyframe_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "keyframe")
            clip_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "clip")
            if isinstance(asset, Mapping):
                segment_refs.extend(_segment_refs_from_asset(asset))
        elif ref_type == "model_observation_event":
            observation_id = _first_ref_id(source_ref, "observation_id", "event_id", "evidence_id", "id")
            observation = context.get("model_observations_by_id", {}).get(observation_id)
            entry = _model_observation_evidence_entry(observation_id, source_ref, observation)
            resolved.append(entry)
            if isinstance(observation, Mapping):
                media_refs = _media_refs_from_model_observation(observation, context, evidence_key=entry.get("evidence_key"))
                keyframe_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "keyframe")
                clip_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "clip")
                segment_refs.extend(_segment_refs_from_model_observation(observation, context))
        elif ref_type == "labsopguard_physical_event":
            entry, media_refs, target_segment_refs = _resolve_labsopguard_physical_event(source_ref, context)
            resolved.append(entry)
            keyframe_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "keyframe")
            clip_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "clip")
            segment_refs.extend(target_segment_refs)
        else:
            entry, media_refs, target_segment_refs = _resolve_generic_event_ref(source_ref, context)
            resolved.append(entry)
            keyframe_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "keyframe")
            clip_refs.extend(ref for ref in media_refs if ref.get("ref_type") == "clip")
            segment_refs.extend(target_segment_refs)
    return {
        "resolved_evidence_refs": resolved,
        "keyframe_refs": _dedupe_media_refs(keyframe_refs),
        "clip_refs": _dedupe_media_refs(clip_refs),
        "segment_refs": _dedupe_segment_refs(segment_refs),
    }


def _video_event_evidence_entry(event_id: str, source_ref: Mapping[str, Any], event: Any) -> dict[str, Any]:
    entry = {
        "evidence_key": f"video_event:{event_id}",
        "type": "video_event",
        "video_event_id": event_id,
        "resolved": isinstance(event, Mapping),
        "source_ref": dict(source_ref),
    }
    if isinstance(event, Mapping):
        entry.update(
            {
                "event_type": event.get("event_type"),
                "segment_id": event.get("segment_id"),
                "micro_segment_id": event.get("micro_segment_id"),
                "global_start_time": event.get("global_start_time"),
                "global_end_time": event.get("global_end_time"),
                "primary_object": event.get("primary_object"),
                "action_type": event.get("action_type"),
                "confidence": event.get("confidence"),
                "confidence_reasons": event.get("confidence_reasons") or [],
                "anomaly_flags": event.get("anomaly_flags") or [],
                "text": _short_text(event.get("text")),
            }
        )
    return entry


def _state_evidence_entry(state_id: str, source_ref: Mapping[str, Any], state: Any) -> dict[str, Any]:
    entry = {
        "evidence_key": f"state_change:{state_id}",
        "type": "state_change",
        "state_change_id": state_id,
        "resolved": isinstance(state, Mapping),
        "source_ref": dict(source_ref),
    }
    if isinstance(state, Mapping):
        entry.update(
            {
                "state_type": state.get("state_type"),
                "segment_id": state.get("segment_id"),
                "micro_segment_id": state.get("micro_segment_id"),
                "global_time": state.get("global_time"),
                "global_start_time": state.get("global_start_time"),
                "global_end_time": state.get("global_end_time"),
                "confidence": state.get("confidence"),
                "text": _short_text(state.get("text") or state.get("description")),
            }
        )
    return entry


def _asset_evidence_entry(asset_id: str, source_ref: Mapping[str, Any], asset: Any) -> dict[str, Any]:
    entry = {
        "evidence_key": f"asset:{asset_id or source_ref.get('path') or 'unknown'}",
        "type": "asset",
        "asset_id": asset_id or source_ref.get("asset_id"),
        "resolved": isinstance(asset, Mapping) or bool(source_ref.get("path")),
        "source_ref": dict(source_ref),
        "path": source_ref.get("path"),
    }
    if isinstance(asset, Mapping):
        entry.update(
            {
                "asset_type": asset.get("asset_type"),
                "source_type": asset.get("source_type"),
                "source_id": asset.get("source_id"),
                "segment_id": asset.get("segment_id"),
                "micro_segment_id": asset.get("micro_segment_id"),
                "global_start_time": asset.get("global_start_time"),
                "global_end_time": asset.get("global_end_time"),
                "path": asset.get("path") or source_ref.get("path"),
                "quality": asset.get("quality"),
                "evidence_level": asset.get("evidence_level"),
            }
        )
    return entry


def _model_observation_evidence_entry(observation_id: str, source_ref: Mapping[str, Any], observation: Any) -> dict[str, Any]:
    entry = {
        "evidence_key": f"model_observation_event:{observation_id or source_ref.get('event_id') or 'unknown'}",
        "type": "model_observation_event",
        "observation_id": observation_id or source_ref.get("observation_id") or source_ref.get("event_id"),
        "resolved": isinstance(observation, Mapping),
        "source_ref": dict(source_ref),
    }
    if isinstance(observation, Mapping):
        asset_refs = _as_list(observation.get("asset_refs"))
        first_asset = next((ref for ref in asset_refs if isinstance(ref, Mapping)), {})
        entry.update(
            {
                "event_type": observation.get("event_type") or observation.get("observation_type"),
                "source_type": observation.get("source_type"),
                "observation_type": observation.get("observation_type"),
                "segment_id": observation.get("segment_id"),
                "micro_segment_id": observation.get("micro_segment_id"),
                "global_start_time": observation.get("global_start_time"),
                "global_end_time": observation.get("global_end_time"),
                "primary_object": observation.get("object_label"),
                "object_label": observation.get("object_label"),
                "action_type": observation.get("action_type"),
                "confidence": observation.get("confidence"),
                "confidence_reasons": observation.get("confidence_reasons") or observation.get("evidence_reasons") or [],
                "confirmation_level": observation.get("confirmation_level"),
                "asset_type": _mapping_get(first_asset, "asset_type"),
                "path": _mapping_get(first_asset, "path"),
                "text": _short_text(_model_observation_text(observation)),
            }
        )
    return entry


def _resolve_labsopguard_physical_event(
    source_ref: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    event_id = _first_ref_id(source_ref, "event_id", "labsopguard_event_id", "observation_id", "video_event_id", "state_change_id", "asset_id", "evidence_id", "id")
    target_type, target = _lookup_physical_event_target(event_id, source_ref, context)
    if target_type == "video_event":
        entry = _video_event_evidence_entry(str(_mapping_get(target, "video_event_id") or event_id), source_ref, target)
        entry["source_evidence_type"] = source_ref.get("evidence_type") or source_ref.get("type")
        media_refs = _media_refs_from_video_event(target, context) if isinstance(target, Mapping) else []
        segment_refs = _segment_refs_from_video_event(target, context) if isinstance(target, Mapping) else []
    elif target_type == "model_observation_event":
        entry = _model_observation_evidence_entry(str(_mapping_get(target, "observation_id") or event_id), source_ref, target)
        entry["source_evidence_type"] = source_ref.get("evidence_type") or source_ref.get("type")
        media_refs = _media_refs_from_model_observation(target, context, evidence_key=entry.get("evidence_key")) if isinstance(target, Mapping) else []
        segment_refs = _segment_refs_from_model_observation(target, context) if isinstance(target, Mapping) else []
    elif target_type == "state_change":
        entry = _state_evidence_entry(str(_mapping_get(target, "state_change_id") or event_id), source_ref, target)
        entry["source_evidence_type"] = source_ref.get("evidence_type") or source_ref.get("type")
        if isinstance(target, Mapping):
            media_refs = _media_refs_from_asset_refs(
                _as_list(target.get("asset_refs")),
                source="state_change",
                segment_id=target.get("segment_id"),
                micro_segment_id=target.get("micro_segment_id"),
                evidence_key=entry.get("evidence_key"),
            )
            media_refs.extend(_media_refs_from_related_segment(target, context, source="state_change", evidence_key=entry.get("evidence_key")))
            segment_refs = _segment_refs_from_state(target, context)
        else:
            media_refs = []
            segment_refs = []
    elif target_type == "asset":
        entry = _asset_evidence_entry(str(_mapping_get(target, "asset_id") or event_id), source_ref, target)
        entry["source_evidence_type"] = source_ref.get("evidence_type") or source_ref.get("type")
        media_refs = _media_refs_from_asset_refs([target or source_ref], source="asset_ref", evidence_key=entry.get("evidence_key"))
        if isinstance(target, Mapping):
            media_refs.extend(_media_refs_from_related_segment(target, context, source="asset_ref", evidence_key=entry.get("evidence_key")))
            segment_refs = _segment_refs_from_asset(target)
        else:
            segment_refs = []
    else:
        entry = {
            "evidence_key": _evidence_key(source_ref),
            "type": "labsopguard_physical_event",
            "event_id": event_id,
            "resolved": False,
            "source_ref": dict(source_ref),
        }
        media_refs = []
        segment_refs = []
    _attach_media_refs(entry, media_refs)
    return entry, media_refs, segment_refs


def _resolve_generic_event_ref(
    source_ref: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    event_id = _first_ref_id(source_ref, "event_id", "observation_id", "video_event_id", "state_change_id", "asset_id", "evidence_id", "id")
    target_type, target = _lookup_physical_event_target(event_id, source_ref, context)
    if target_type:
        entry, media_refs, segment_refs = _resolve_labsopguard_physical_event(
            {**dict(source_ref), "event_id": event_id, "evidence_type": source_ref.get("evidence_type") or source_ref.get("type") or "generic_event_ref"},
            context,
        )
        return entry, media_refs, segment_refs
    return (
        {
            "evidence_key": _evidence_key(source_ref),
            "type": _evidence_ref_type(source_ref),
            "resolved": False,
            "source_ref": dict(source_ref),
        },
        [],
        [],
    )


def _attach_media_refs(entry: dict[str, Any], media_refs: list[dict[str, Any]]) -> None:
    entry["keyframe_refs"] = _dedupe_media_refs([ref for ref in media_refs if ref.get("ref_type") == "keyframe"])
    entry["clip_refs"] = _dedupe_media_refs([ref for ref in media_refs if ref.get("ref_type") == "clip"])


def _lookup_physical_event_target(
    event_id: str,
    source_ref: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None]:
    if not event_id and source_ref.get("path"):
        asset = context.get("assets_by_path", {}).get(str(source_ref.get("path")))
        return ("asset", asset) if isinstance(asset, Mapping) else ("", None)
    lookups = (
        ("model_observation_event", context.get("model_observations_by_id", {})),
        ("video_event", context.get("video_events_by_id", {})),
        ("state_change", context.get("states_by_id", {})),
        ("asset", context.get("assets_by_id", {})),
        ("asset", context.get("assets_by_path", {})),
    )
    for target_type, index in lookups:
        target = index.get(event_id) if isinstance(index, Mapping) else None
        if isinstance(target, Mapping):
            return target_type, target
    return "", None


def _summarize_evidence(step: Mapping[str, Any], evidence_payload: Mapping[str, Any]) -> dict[str, Any]:
    refs = evidence_payload.get("resolved_evidence_refs") or []
    ref_count = len(_normalize_evidence_refs(step.get("evidence_refs")))
    resolved_count = sum(1 for ref in refs if isinstance(ref, Mapping) and ref.get("resolved"))
    type_counts = Counter(str(ref.get("type") or "unknown") for ref in refs if isinstance(ref, Mapping))
    event_counts = Counter(
        str(ref.get("event_type") or "unknown")
        for ref in refs
        if isinstance(ref, Mapping) and ref.get("event_type") and ref.get("type") in {"video_event", "model_observation_event"}
    )
    objects = _ordered_unique(str(ref.get("primary_object") or "") for ref in refs if isinstance(ref, Mapping))
    action_types = _ordered_unique(str(ref.get("action_type") or "") for ref in refs if isinstance(ref, Mapping))
    confidences = [_as_float(ref.get("confidence")) for ref in refs if isinstance(ref, Mapping)]
    confidence_values = [value for value in confidences if value is not None]
    limitations = _ordered_unique(
        str(flag)
        for ref in refs
        if isinstance(ref, Mapping)
        for flag in _as_list(ref.get("anomaly_flags"))
    )
    keyframe_count = len(_as_list(evidence_payload.get("keyframe_refs")))
    clip_count = len(_as_list(evidence_payload.get("clip_refs")))
    event_count = sum(event_counts.values())
    state_count = type_counts.get("state_change", 0)
    asset_count = type_counts.get("asset", 0)
    hand_object_count = sum(count for event_type, count in event_counts.items() if "hand_object" in event_type or "contact" in event_type)
    if ref_count == 0:
        text = "No direct evidence refs are attached to this step."
    else:
        bits = [
            f"refs={ref_count}",
            f"resolved={resolved_count}",
            f"events={event_count}",
            f"states={state_count}",
            f"assets={asset_count}",
            f"keyframes={keyframe_count}",
            f"clips={clip_count}",
        ]
        if confidence_values:
            bits.append(f"max_confidence={max(confidence_values):.3f}")
        if objects:
            bits.append(f"objects={','.join(objects[:4])}")
        text = "; ".join(bits)
    return {
        "text": text,
        "evidence_ref_count": ref_count,
        "resolved_ref_count": resolved_count,
        "evidence_type_counts": dict(sorted(type_counts.items())),
        "video_event_type_counts": dict(sorted(event_counts.items())),
        "event_count": event_count,
        "state_change_count": state_count,
        "asset_count": asset_count,
        "hand_object_interaction_count": hand_object_count,
        "keyframe_count": keyframe_count,
        "clip_count": clip_count,
        "objects": objects,
        "action_types": action_types,
        "confidence": {
            "max": max(confidence_values) if confidence_values else None,
            "avg": round(sum(confidence_values) / len(confidence_values), 4) if confidence_values else None,
        },
        "limitations": limitations,
    }


def _suggested_action(step: Mapping[str, Any], evidence_summary: Mapping[str, Any]) -> dict[str, Any]:
    conflict_flags = _as_list(step.get("conflict_flags"))
    status = str(step.get("status") or "")
    confidence = _as_float(step.get("confidence")) or 0.0
    resolved_refs = int(evidence_summary.get("resolved_ref_count") or 0)
    max_evidence_confidence = _as_float(_as_dict(evidence_summary.get("confidence")).get("max"))
    effective_confidence = max(confidence, max_evidence_confidence or 0.0)
    if conflict_flags or step.get("abnormal"):
        decision = "needs_review"
        reason = "Resolve conflict or abnormal evidence before changing process state."
    elif status in {"not_observed", "skipped_or_unobserved"} and resolved_refs == 0:
        decision = "rejected"
        reason = "No direct evidence is attached; reject completion unless an external record confirms it."
    elif status == "inferred_missing" and resolved_refs > 0 and effective_confidence >= 0.5:
        decision = "approved"
        reason = "The step was inferred and has supporting evidence refs for reviewer confirmation."
    elif effective_confidence >= 0.75 and int(evidence_summary.get("event_count") or 0) > 0:
        decision = "approved"
        reason = "Direct video evidence has high confidence."
    else:
        decision = "needs_review"
        reason = "Evidence is limited or ambiguous; keep the item in manual review."
    return {
        "decision": decision,
        "reason": reason,
        "reviewer_checks": _reviewer_checks(step, evidence_summary),
        "applied_effect": _decision_effect(decision),
    }


def _reviewer_checks(step: Mapping[str, Any], evidence_summary: Mapping[str, Any]) -> list[str]:
    checks = [
        "Inspect referenced keyframes and clips before applying the decision.",
        "Compare the evidence time window with the expected SOP step.",
    ]
    if step.get("inferred"):
        checks.append("Confirm whether the inferred completion is supported by neighboring observed steps.")
    if step.get("abnormal") or step.get("conflict_flags"):
        checks.append("Resolve abnormal or conflict flags explicitly in the reviewer note.")
    if int(evidence_summary.get("keyframe_count") or 0) == 0:
        checks.append("No keyframe refs are available; rely on clips or external records.")
    return checks


def _decision_effect(decision: str) -> str:
    if decision == "approved":
        return "clear confirmation requirement; inferred or missing steps may become human_confirmed"
    if decision == "rejected":
        return "mark step incomplete with status human_rejected; clear confirmation requirement"
    return "record reviewer note and keep confirmation requirement for another pass"


def _audit_record(
    *,
    audit_id: str,
    session_id: str,
    confirmation_id: str,
    decision: Mapping[str, Any],
    previous_decision: Mapping[str, Any] | None,
    before_state: Mapping[str, Any] | None,
    after_state: Mapping[str, Any] | None,
    changed: bool,
    review_item: Mapping[str, Any] | None,
) -> dict[str, Any]:
    before_status = before_state.get("status") if isinstance(before_state, Mapping) else None
    after_status = after_state.get("status") if isinstance(after_state, Mapping) else None
    return {
        "schema_version": "confirmation_audit_trail/v1",
        "audit_id": audit_id,
        "session_id": session_id,
        "confirmation_id": confirmation_id,
        "step_id": confirmation_id.rsplit(":", 1)[-1],
        "decision": decision.get("decision"),
        "reviewer": decision.get("reviewer"),
        "note": decision.get("note"),
        "decided_at": decision.get("decided_at"),
        "previous_decision": _public_decision(previous_decision or {}),
        "before_state": dict(before_state) if isinstance(before_state, Mapping) else None,
        "after_state": dict(after_state) if isinstance(after_state, Mapping) else None,
        "changed": changed,
        "status_changed": before_status != after_status,
        "evidence_summary": dict(review_item.get("evidence_summary")) if isinstance(review_item, Mapping) else {},
        "keyframe_refs": list(review_item.get("keyframe_refs") or []) if isinstance(review_item, Mapping) else [],
        "clip_refs": list(review_item.get("clip_refs") or []) if isinstance(review_item, Mapping) else [],
        "suggested_action_at_decision": dict(review_item.get("suggested_action")) if isinstance(review_item, Mapping) else {},
    }


def _load_review_context(session: Path) -> dict[str, Any]:
    metadata = session / "metadata"
    micro_path = metadata / "micro_segments_corrected.jsonl"
    if not micro_path.exists():
        micro_path = metadata / "micro_segments.jsonl"
    process = _read_json(metadata / "experiment_process.json")
    video_events = _read_jsonl_if_exists(metadata / "video_understanding.jsonl")
    model_observation_rows = _read_jsonl_if_exists(metadata / "model_observation_events.jsonl")
    state_rows = _read_jsonl_if_exists(metadata / "state_change_index.jsonl")
    asset_rows = _read_jsonl_if_exists(metadata / "material_asset_catalog.jsonl")
    queue_rows = _read_jsonl_if_exists(metadata / QUEUE_FILENAME)
    segment_rows = _read_jsonl_if_exists(metadata / "key_action_segments.jsonl")
    micro_rows = _read_jsonl_if_exists(micro_path)
    video_events_by_id = _index_rows_by_alias(video_events)
    model_observations_by_id = _index_rows_by_alias(model_observation_rows)
    states_by_id = _index_rows_by_alias(state_rows)
    assets_by_id = _index_rows_by_alias(asset_rows)
    assets_by_path = {
        str(row.get("path")): row for row in asset_rows if isinstance(row, Mapping) and row.get("path")
    }
    return {
        "video_events": video_events,
        "model_observation_rows": model_observation_rows,
        "state_rows": state_rows,
        "asset_rows": asset_rows,
        "queue_rows": queue_rows,
        "process": process,
        "segment_rows": segment_rows,
        "micro_rows": micro_rows,
        "video_events_by_id": video_events_by_id,
        "video_events_by_related_id": _group_rows_by_alias(video_events),
        "model_observations_by_id": model_observations_by_id,
        "states_by_id": states_by_id,
        "assets_by_id": assets_by_id,
        "assets_by_path": assets_by_path,
        "segments_by_id": {
            str(row.get("segment_id")): row for row in segment_rows if row.get("segment_id")
        },
        "micros_by_id": {
            str(row.get("micro_segment_id")): row for row in micro_rows if row.get("micro_segment_id")
        },
    }


def _video_events_requiring_confirmation(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for event in context.get("video_events") or []:
        if not isinstance(event, Mapping):
            continue
        flags = {str(flag) for flag in _as_list(event.get("anomaly_flags"))}
        status = str(event.get("conclusion_status") or "")
        confidence = _as_float(event.get("confidence"))
        if _should_queue_video_event(event, flags=flags, status=status, confidence=confidence):
            rows.append(dict(event))
    return rows


def _step_video_event_ids(process: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for step in process.get("steps") or []:
        if not isinstance(step, Mapping):
            continue
        for ref in _normalize_evidence_refs(step.get("evidence_refs")):
            ref_type = _evidence_ref_type(ref)
            if ref_type == "video_event":
                event_id = _first_ref_id(ref, "video_event_id", "event_id", "evidence_id", "id")
                if event_id:
                    ids.add(event_id)
            for key in ("micro_segment_id", "segment_id"):
                value = str(ref.get(key) or "").strip()
                if value:
                    ids.add(value)
    return ids


def _video_event_aliases(event: Mapping[str, Any]) -> set[str]:
    aliases = set(_row_id_aliases(event))
    for key in ("micro_segment_id", "segment_id"):
        value = str(event.get(key) or "").strip()
        if value:
            aliases.add(value)
    return aliases


def _should_queue_video_event(
    event: Mapping[str, Any],
    *,
    flags: set[str],
    status: str,
    confidence: float | None,
) -> bool:
    if "conflicting_physical_event_evidence" in flags:
        return True
    if _is_low_signal_capability_candidate(event, flags=flags, status=status):
        return False
    if confidence is not None and confidence < 0.5:
        return True
    return status == "candidate" or "requires_human_confirmation" in flags


def _is_low_signal_capability_candidate(event: Mapping[str, Any], *, flags: set[str], status: str) -> bool:
    event_type = str(event.get("event_type") or "")
    if event_type not in LOW_SIGNAL_VIDEO_CANDIDATE_TYPES:
        return False
    if status != "candidate" and "heuristic_candidate" not in flags and "advanced_evidence_candidate" not in flags:
        return False
    capability_gap_flags = {
        "heuristic_candidate",
        "advanced_evidence_candidate",
        "low_confidence_candidate_event",
        "not_container_open_close_confirmed",
        "not_panel_ocr_confirmed",
        "not_visual_liquid_flow_confirmed",
        "visual_confirmation_limited",
    }
    if flags & capability_gap_flags:
        return True
    payload = _as_dict(event.get("payload"))
    source = str(payload.get("source") or "")
    return source in {"micro_segment", "advanced_vision_evidence"}


def _index_rows_by_alias(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        for alias in _row_id_aliases(row):
            index.setdefault(alias, dict(row))
    return index


def _group_rows_by_alias(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_dict = dict(row)
        for alias in _row_id_aliases(row):
            grouped[alias].append(row_dict)
    return grouped


def _row_id_aliases(row: Mapping[str, Any]) -> list[str]:
    aliases: list[str] = []
    for key in (
        "video_event_id",
        "observation_id",
        "state_change_id",
        "asset_id",
        "event_id",
        "evidence_id",
        "id",
        "source_id",
        "source_event_id",
        "labsopguard_event_id",
        "timeline_event_id",
    ):
        _append_unique(aliases, str(row.get(key) or ""))
    payload = _as_dict(row.get("payload"))
    for key in ("model_observation", "advanced_evidence", "source_row", "labsopguard_event", "physical_event"):
        nested = _as_dict(payload.get(key))
        if nested:
            for alias in _row_id_aliases(nested):
                _append_unique(aliases, alias)
    for ref_key in ("evidence_refs", "asset_refs"):
        for ref in _normalize_evidence_refs(row.get(ref_key)):
            for key in (
                "video_event_id",
                "observation_id",
                "state_change_id",
                "asset_id",
                "event_id",
                "evidence_id",
                "id",
                "source_id",
                "labsopguard_event_id",
            ):
                _append_unique(aliases, str(ref.get(key) or ""))
            if ref.get("path"):
                _append_unique(aliases, str(ref.get("path") or ""))
    return aliases


def _video_event_confirmation_reason(event: Mapping[str, Any]) -> str:
    flags = [str(flag) for flag in _as_list(event.get("anomaly_flags")) if flag]
    if flags:
        return "video_event_requires_review:" + ",".join(flags[:6])
    status = str(event.get("conclusion_status") or "candidate")
    return f"video_event_conclusion_status={status}"


def _media_refs_from_video_event(event: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    segment_id = event.get("segment_id")
    micro_id = event.get("micro_segment_id")
    refs.extend(
        _media_refs_from_asset_refs(
            _as_list(event.get("asset_refs")),
            source="video_event",
            segment_id=segment_id,
            micro_segment_id=micro_id,
            evidence_key=f"video_event:{event.get('video_event_id')}",
        )
    )
    payload = _as_dict(event.get("payload"))
    micro = _as_dict(payload.get("micro_segment")) or context.get("micros_by_id", {}).get(str(micro_id or ""), {})
    if isinstance(micro, Mapping) and micro:
        refs.extend(_media_refs_from_micro(micro, source="micro_segment", evidence_key=f"video_event:{event.get('video_event_id')}"))
    segment = context.get("segments_by_id", {}).get(str(segment_id or ""))
    if isinstance(segment, Mapping) and segment:
        refs.extend(_media_refs_from_segment(segment, source="key_action_segment", evidence_key=f"video_event:{event.get('video_event_id')}"))
    return _dedupe_media_refs(refs)


def _media_refs_from_micro(micro: Mapping[str, Any], *, source: str, evidence_key: str | None = None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    segment_id = micro.get("parent_segment_id") or micro.get("segment_id")
    micro_id = micro.get("micro_segment_id")
    keyframes = _as_dict(micro.get("keyframes"))
    ordered_keys = ["contact_frame", "peak_frame", "release_frame"]
    ordered_keys.extend(key for key in keyframes if key not in ordered_keys)
    for key in ordered_keys:
        path = keyframes.get(key)
        if path:
            refs.append(
                _media_ref(
                    "keyframe",
                    path,
                    source=source,
                    role=f"keyframes.{key}",
                    segment_id=segment_id,
                    micro_segment_id=micro_id,
                    evidence_key=evidence_key,
                )
            )
    for view_key in ("first_person", "third_person"):
        view = _as_dict(micro.get(view_key))
        for field in ("clip_path", "annotated_clip_path"):
            path = view.get(field)
            if path:
                refs.append(
                    _media_ref(
                        "clip",
                        path,
                        source=source,
                        role=f"{view_key}.{field}",
                        segment_id=segment_id,
                        micro_segment_id=micro_id,
                        view=view_key,
                        evidence_key=evidence_key,
                    )
                )
    return refs


def _media_refs_from_segment(segment: Mapping[str, Any], *, source: str, evidence_key: str | None = None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    segment_id = segment.get("segment_id")
    for view_key in ("first_person", "third_person"):
        view = _as_dict(segment.get(view_key))
        path = view.get("clip_path")
        if path:
            refs.append(
                _media_ref(
                    "clip",
                    path,
                    source=source,
                    role=f"{view_key}.clip_path",
                    segment_id=segment_id,
                    view=view_key,
                    evidence_key=evidence_key,
                )
            )
    for index, frame in enumerate(_as_list(segment.get("interaction_keyframes"))[:5], start=1):
        if not isinstance(frame, Mapping):
            continue
        path = frame.get("path") or frame.get("keyframe_path")
        if path:
            refs.append(
                _media_ref(
                    "keyframe",
                    path,
                    source=source,
                    role=f"interaction_keyframes[{index}]",
                    segment_id=segment_id,
                    view=frame.get("view"),
                    event_id=frame.get("event_id"),
                    evidence_key=evidence_key,
                )
            )
    for index, event in enumerate(_as_list(segment.get("interaction_events"))[:5], start=1):
        if not isinstance(event, Mapping):
            continue
        path = event.get("keyframe_path")
        if path:
            refs.append(
                _media_ref(
                    "keyframe",
                    path,
                    source=source,
                    role=f"interaction_events[{index}].keyframe_path",
                    segment_id=segment_id,
                    view=event.get("view"),
                    event_id=event.get("event_id"),
                    evidence_key=evidence_key,
                )
            )
    return refs


def _media_refs_from_asset_refs(
    asset_refs: list[Any],
    *,
    source: str,
    segment_id: Any = None,
    micro_segment_id: Any = None,
    evidence_key: str | None = None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for asset in asset_refs:
        if not isinstance(asset, Mapping):
            continue
        path = asset.get("path") or asset.get("media_path") or asset.get("file_path") or asset.get("image_path")
        if not path:
            continue
        asset_type = str(asset.get("asset_type") or asset.get("type") or "")
        rel = str(asset.get("rel") or asset.get("role") or asset.get("path_field") or asset_type or "asset")
        ref_type = "clip" if _looks_like_clip(path, asset_type, rel) else "keyframe"
        refs.append(
            _media_ref(
                ref_type,
                path,
                source=source,
                role=rel,
                segment_id=asset.get("segment_id") or segment_id,
                micro_segment_id=asset.get("micro_segment_id") or micro_segment_id,
                asset_id=asset.get("asset_id"),
                asset_type=asset_type or None,
                evidence_key=evidence_key,
            )
        )
    return refs


def _media_refs_from_model_observation(
    observation: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    evidence_key: str | None = None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    observation_id = str(observation.get("observation_id") or "")
    refs.extend(
        _media_refs_from_asset_refs(
            _as_list(observation.get("asset_refs")),
            source="model_observation_event",
            segment_id=observation.get("segment_id"),
            micro_segment_id=observation.get("micro_segment_id"),
            evidence_key=evidence_key,
        )
    )
    refs.extend(_media_refs_from_related_segment(observation, context, source="model_observation_event", evidence_key=evidence_key))
    linked_events = context.get("video_events_by_related_id", {}).get(observation_id, []) if observation_id else []
    for event in linked_events:
        if isinstance(event, Mapping):
            refs.extend(_media_refs_from_video_event(event, context))
    return _dedupe_media_refs(refs)


def _media_refs_from_related_segment(
    row: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    source: str,
    evidence_key: str | None = None,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    micro_id = row.get("micro_segment_id") or row.get("source_id")
    segment_id = row.get("segment_id")
    micro = context.get("micros_by_id", {}).get(str(micro_id or ""))
    if isinstance(micro, Mapping) and micro:
        refs.extend(_media_refs_from_micro(micro, source=source, evidence_key=evidence_key))
        segment_id = segment_id or micro.get("parent_segment_id") or micro.get("segment_id")
    segment = context.get("segments_by_id", {}).get(str(segment_id or ""))
    if isinstance(segment, Mapping) and segment:
        refs.extend(_media_refs_from_segment(segment, source=source, evidence_key=evidence_key))
    return _dedupe_media_refs(refs)


def _segment_refs_from_video_event(event: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    segment_id = event.get("segment_id")
    micro_id = event.get("micro_segment_id")
    segment = context.get("segments_by_id", {}).get(str(segment_id or ""))
    micro = context.get("micros_by_id", {}).get(str(micro_id or ""))
    return [
        {
            "source": "video_event",
            "video_event_id": event.get("video_event_id"),
            "segment_id": segment_id,
            "micro_segment_id": micro_id,
            "global_start_time": event.get("global_start_time") or _mapping_get(micro, "global_start_time") or _mapping_get(segment, "global_start_time"),
            "global_end_time": event.get("global_end_time") or _mapping_get(micro, "global_end_time") or _mapping_get(segment, "global_end_time"),
            "event_type": event.get("event_type"),
            "action_type": event.get("action_type"),
            "primary_object": event.get("primary_object"),
            "confidence": event.get("confidence"),
        }
    ]


def _segment_refs_from_state(state: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    micro_id = state.get("micro_segment_id")
    micro = context.get("micros_by_id", {}).get(str(micro_id or ""))
    return [
        {
            "source": "state_change",
            "state_change_id": state.get("state_change_id"),
            "segment_id": state.get("segment_id") or _mapping_get(micro, "parent_segment_id"),
            "micro_segment_id": micro_id,
            "global_start_time": state.get("global_start_time") or state.get("global_time") or _mapping_get(micro, "global_start_time"),
            "global_end_time": state.get("global_end_time") or state.get("global_time") or _mapping_get(micro, "global_end_time"),
            "state_type": state.get("state_type"),
            "confidence": state.get("confidence"),
        }
    ]


def _segment_refs_from_model_observation(observation: Mapping[str, Any], context: Mapping[str, Any]) -> list[dict[str, Any]]:
    micro_id = observation.get("micro_segment_id")
    micro = context.get("micros_by_id", {}).get(str(micro_id or ""))
    return [
        {
            "source": "model_observation_event",
            "observation_id": observation.get("observation_id"),
            "segment_id": observation.get("segment_id") or _mapping_get(micro, "parent_segment_id"),
            "micro_segment_id": micro_id,
            "global_start_time": observation.get("global_start_time") or _mapping_get(micro, "global_start_time"),
            "global_end_time": observation.get("global_end_time") or _mapping_get(micro, "global_end_time"),
            "event_type": observation.get("event_type") or observation.get("observation_type"),
            "source_type": observation.get("source_type"),
            "primary_object": observation.get("object_label"),
            "confidence": observation.get("confidence"),
        }
    ]


def _segment_refs_from_asset(asset: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not any(asset.get(key) for key in ("segment_id", "micro_segment_id", "asset_id", "path")):
        return []
    return [
        {
            "source": "asset",
            "asset_id": asset.get("asset_id"),
            "segment_id": asset.get("segment_id"),
            "micro_segment_id": asset.get("micro_segment_id") or asset.get("source_id"),
            "asset_type": asset.get("asset_type"),
            "path": asset.get("path"),
        }
    ]


def _media_ref(ref_type: str, path: Any, *, source: str, role: str, **extra: Any) -> dict[str, Any]:
    row = {
        "ref_type": ref_type,
        "source": source,
        "role": role,
        "path": str(path),
    }
    for key, value in extra.items():
        if value is not None and value != "":
            row[key] = value
    return row


def _dedupe_media_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for ref in refs:
        key = (
            str(ref.get("ref_type") or ""),
            str(ref.get("path") or ""),
            str(ref.get("role") or ""),
            str(ref.get("source") or ""),
            str(ref.get("segment_id") or ""),
            str(ref.get("micro_segment_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(ref)
    return output


def _dedupe_segment_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for ref in refs:
        key = (
            str(ref.get("source") or ""),
            str(ref.get("segment_id") or ""),
            str(ref.get("micro_segment_id") or ""),
            str(ref.get("video_event_id") or ref.get("state_change_id") or ref.get("observation_id") or ref.get("asset_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(ref)
    return output


def _step_review_payload(step: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "step_id",
        "name",
        "expected_action",
        "status",
        "observed",
        "inferred",
        "completed",
        "skipped",
        "repeated",
        "abnormal",
        "confidence",
        "confidence_reasons",
        "global_start_time",
        "global_end_time",
        "missing_completion_reason",
        "next_step_hint",
        "requires_human_confirmation",
        "repeat_count",
        "conflict_flags",
        "reasoning",
        "condition_results",
        "history_prior",
        "history_deviation",
        "branch_enabled",
        "confirmation_status",
    )
    return {field: step.get(field) for field in fields}


def _step_state(step: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(step, Mapping):
        return None
    fields = (
        "step_id",
        "name",
        "status",
        "observed",
        "inferred",
        "completed",
        "skipped",
        "abnormal",
        "confidence",
        "requires_human_confirmation",
        "confirmation_status",
        "missing_completion_reason",
        "conflict_flags",
        "evidence_refs",
        "reasoning",
        "condition_results",
        "history_prior",
        "history_deviation",
        "confirmation_decision",
    )
    return {field: step.get(field) for field in fields}


def _find_step(process: Mapping[str, Any], confirmation_id: str) -> Mapping[str, Any] | None:
    step_id = str(confirmation_id or "").rsplit(":", 1)[-1]
    for step in process.get("steps", []):
        if isinstance(step, Mapping) and str(step.get("step_id")) == step_id:
            return step
    return None


def _confirmation_id(session_id: str, step: Mapping[str, Any]) -> str:
    return f"{session_id}:{step.get('step_id')}"


def _confirmation_reason(step: Mapping[str, Any]) -> str:
    return (
        str(step.get("missing_completion_reason") or "")
        or "; ".join(str(flag) for flag in _as_list(step.get("conflict_flags")) if flag)
        or "requires human confirmation"
    )


def _evidence_key(ref: Mapping[str, Any]) -> str:
    ref_type = _evidence_ref_type(ref)
    if ref_type == "video_event":
        return f"video_event:{ref.get('video_event_id') or ref.get('event_id') or 'unknown'}"
    if ref_type == "state_change":
        return f"state_change:{ref.get('state_change_id') or ref.get('event_id') or 'unknown'}"
    if ref_type == "asset":
        return f"asset:{ref.get('asset_id') or ref.get('event_id') or ref.get('path') or 'unknown'}"
    if ref_type == "model_observation_event":
        return f"model_observation_event:{ref.get('observation_id') or ref.get('event_id') or 'unknown'}"
    if ref_type == "labsopguard_physical_event":
        return f"labsopguard_physical_event:{ref.get('event_id') or ref.get('id') or 'unknown'}"
    try:
        payload = json.dumps(dict(ref), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        payload = str(dict(ref))
    return f"{ref_type}:{ref.get('id') or ref.get('event_id') or ref.get('path') or payload}"


def _ref_matches_evidence(media_ref: Mapping[str, Any], evidence_ref: Mapping[str, Any]) -> bool:
    evidence_key = evidence_ref.get("evidence_key")
    if media_ref.get("evidence_key") and evidence_key:
        return media_ref.get("evidence_key") == evidence_key
    for key in ("asset_id", "video_event_id", "state_change_id", "observation_id", "event_id", "segment_id", "micro_segment_id"):
        if media_ref.get(key) and evidence_ref.get(key) and media_ref.get(key) == evidence_ref.get(key):
            return True
    return False


def _public_decision(decision: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(decision, Mapping) or not decision:
        return {}
    return {
        "audit_id": decision.get("audit_id"),
        "confirmation_id": decision.get("confirmation_id"),
        "decision": decision.get("decision"),
        "reviewer": decision.get("reviewer"),
        "note": decision.get("note"),
        "decided_at": decision.get("decided_at"),
    }


def _public_audit(audit: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit, Mapping) or not audit:
        return {}
    return {
        "audit_id": audit.get("audit_id"),
        "decision": audit.get("decision"),
        "reviewer": audit.get("reviewer"),
        "note": audit.get("note"),
        "decided_at": audit.get("decided_at"),
        "status_changed": audit.get("status_changed"),
        "before_status": _as_dict(audit.get("before_state")).get("status"),
        "after_status": _as_dict(audit.get("after_state")).get("status"),
    }


def _append_audit_record(path: Path, record: Mapping[str, Any]) -> None:
    rows = read_jsonl(path) if path.exists() else []
    rows.append(dict(record))
    write_jsonl(path, rows)


def _decisions_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row.get("confirmation_id")): row for row in read_jsonl(path) if row.get("confirmation_id")}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except (json.JSONDecodeError, OSError):
        return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_id(confirmation_id: str, decided_at: str) -> str:
    return f"{confirmation_id}:{decided_at}"


def _looks_like_clip(path: Any, asset_type: str, rel: str) -> bool:
    text = f"{asset_type} {rel} {path}".lower()
    return "clip" in text or str(path).lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))


def _normalize_evidence_refs(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, Mapping):
            refs.append(dict(item))
            return
        if isinstance(item, list):
            for child in item:
                add(child)
            return
        if isinstance(item, str):
            text = item.strip()
            if not text:
                return
            decoded = _safe_json_value(text)
            if isinstance(decoded, (Mapping, list)):
                add(decoded)
                return
            refs.append({"type": "unknown", "event_id": text, "raw_ref": text})

    add(value)
    return refs


def _safe_json_value(text: str) -> Any:
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _evidence_ref_type(ref: Mapping[str, Any]) -> str:
    ref_type = str(ref.get("type") or "").strip()
    evidence_type = str(ref.get("evidence_type") or "").strip()
    if ref_type:
        return ref_type
    if evidence_type == "labsopguard_physical_event":
        return "labsopguard_physical_event"
    if ref.get("video_event_id"):
        return "video_event"
    if ref.get("state_change_id"):
        return "state_change"
    if ref.get("asset_id") or ref.get("path"):
        return "asset"
    if ref.get("observation_id"):
        return "model_observation_event"
    return evidence_type or "unknown"


def _first_ref_id(ref: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = ref.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _model_observation_text(observation: Mapping[str, Any]) -> str:
    pieces = [
        observation.get("source_type"),
        observation.get("observation_type"),
        observation.get("event_type"),
        observation.get("object_label"),
        observation.get("state"),
    ]
    measurement = _as_dict(observation.get("measurement"))
    if measurement:
        pieces.append(", ".join(f"{key}={value}" for key, value in sorted(measurement.items())[:6]))
    return " ".join(str(piece) for piece in pieces if piece)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def _mapping_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


__all__ = [
    "build_confirmation_queue",
    "list_confirmation_queue",
    "apply_confirmation_decision",
    "apply_confirmation_batch_decisions",
    "build_confirmation_review_bundle",
    "build_confirmation_review_summary",
    "resolve_review_evidence_for_step",
]
