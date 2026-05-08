from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import read_jsonl


PROCESS_RECORD_SCHEMA_VERSION = "process_record/v1"
PROCESS_RECORD_FILENAME = "process_record.json"
PROCESS_AUDIT_REPORT_FILENAME = "process_audit_report.md"
LOW_CONFIDENCE_THRESHOLD = 0.5

EVIDENCE_LEVEL_SCORES = {
    "direct_visual": 1.0,
    "visual_confirmed": 0.95,
    "trajectory_confirmed": 0.9,
    "model_confirmed": 0.85,
    "state_change": 0.75,
    "visual_asset": 0.7,
    "text_support": 0.55,
    "database_record": 0.5,
    "model_candidate": 0.45,
    "visual_candidate": 0.4,
    "inferred_support": 0.35,
    "weak": 0.2,
    "none": 0.0,
}


def build_process_record(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    audit_report_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    exports = session / "exports"
    reports = session / "reports"
    target = Path(output_path) if output_path is not None else exports / PROCESS_RECORD_FILENAME
    audit_target = Path(audit_report_path) if audit_report_path is not None else reports / PROCESS_AUDIT_REPORT_FILENAME

    process = _read_json(metadata / "experiment_process.json")
    context = _read_json(metadata / "experiment_context.json")
    quality = _read_json(metadata / "process_quality_report.json")
    confirmation_summary = _read_json(metadata / "human_confirmation_review_summary.json")
    confirmation_queue = _read_jsonl(metadata / "human_confirmation_queue.jsonl")
    sources = _load_sources(session)
    session_id = str(process.get("session_id") or context.get("session_id") or session.name)
    generated_at = datetime.now(timezone.utc).isoformat()

    evidence_entries: dict[str, dict[str, Any]] = {}
    step_records: list[dict[str, Any]] = []
    steps = [step for step in process.get("steps", []) if isinstance(step, Mapping)]
    for order, step in enumerate(steps, start=1):
        refs = _complete_step_refs(step, order)
        resolved_refs = []
        for ref in refs:
            evidence = _resolve_evidence(ref, sources)
            _attach_step(evidence_entries, evidence, str(step.get("step_id") or ""))
            resolved_refs.append(_step_evidence_ref(evidence, ref))
        evidence_summary = _evidence_summary(resolved_refs)
        step_records.append(
            {
                "step_id": str(step.get("step_id") or f"step_{order:03d}"),
                "order": order,
                "name": step.get("name"),
                "expected_action": step.get("expected_action"),
                "status": step.get("status"),
                "observed": bool(step.get("observed")),
                "inferred": bool(step.get("inferred")),
                "completed": bool(step.get("completed")),
                "skipped": bool(step.get("skipped")),
                "abnormal": bool(step.get("abnormal")),
                "global_start_time": step.get("global_start_time"),
                "global_end_time": step.get("global_end_time"),
                "confidence": _bounded_float(step.get("confidence")),
                "confidence_reasons": _strings(step.get("confidence_reasons")),
                "evidence_refs": resolved_refs,
                "evidence_summary": evidence_summary,
                "inference": _inference_payload(step, refs),
                "reasoning": {
                    "missing_completion_reason": step.get("missing_completion_reason") or "",
                    "condition_results": step.get("condition_results") or {},
                    "history_prior": step.get("history_prior") or {},
                    "history_deviation": step.get("history_deviation") or {},
                    "history_basis": step.get("history_basis") or [],
                    "next_step_hint": step.get("next_step_hint"),
                },
                "confirmation": _confirmation_payload(step, confirmation_queue, session_id),
                "audit_flags": _audit_flags(step, evidence_summary),
            }
        )

    evidence = sorted(evidence_entries.values(), key=lambda row: str(row.get("evidence_id") or ""))
    evidence_index = {row["evidence_id"]: sorted(row.get("step_ids") or []) for row in evidence}
    step_index = {step["step_id"]: [ref["evidence_id"] for ref in step["evidence_refs"]] for step in step_records}
    status_counts = Counter(str(step.get("status") or "unknown") for step in step_records)
    inferred_steps = [step for step in step_records if step["inferred"]]
    pending_steps = [step for step in step_records if step["confirmation"]["requires_human_confirmation"]]
    weak_steps = [step for step in step_records if step["evidence_summary"]["strongest_score"] < LOW_CONFIDENCE_THRESHOLD]
    record = {
        "schema_version": PROCESS_RECORD_SCHEMA_VERSION,
        "session_id": session_id,
        "generated_at": generated_at,
        "source_paths": {
            "experiment_process": str(metadata / "experiment_process.json"),
            "experiment_context": str(metadata / "experiment_context.json"),
            "quality_report": str(metadata / "process_quality_report.json"),
            "confirmation_review_summary": str(metadata / "human_confirmation_review_summary.json"),
        },
        "summary": {
            "step_count": len(step_records),
            "status_counts": dict(sorted(status_counts.items())),
            "inferred_step_count": len(inferred_steps),
            "pending_confirmation_count": len(pending_steps),
            "weak_evidence_step_count": len(weak_steps),
            "process_status": process.get("process_status"),
            "quality_status": quality.get("overall_status"),
            "quality_score": quality.get("overall_score"),
        },
        "steps": step_records,
        "evidence": evidence,
        "evidence_index": evidence_index,
        "step_index": step_index,
        "confirmation_review": {
            "queue_count": len(confirmation_queue),
            "review_summary_path": str(metadata / "human_confirmation_review_summary.json"),
            "latest_decision_counts": confirmation_summary.get("latest_decision_counts", {}),
        },
        "audit_report_path": str(audit_target),
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_audit_report(audit_target, record)
    return record


def _load_sources(session: Path) -> dict[str, Any]:
    metadata = session / "metadata"
    micro_path = metadata / "micro_segments_corrected.jsonl"
    if not micro_path.exists():
        micro_path = metadata / "micro_segments.jsonl"
    video_events = _read_jsonl(metadata / "video_understanding.jsonl")
    state_rows = _read_jsonl(metadata / "state_change_index.jsonl")
    assets = _read_jsonl(metadata / "material_asset_catalog.jsonl")
    segments = _read_jsonl(metadata / "key_action_segments.jsonl")
    micros = _read_jsonl(micro_path)
    model_events = _read_jsonl(metadata / "model_observation_events.jsonl")
    advanced_evidence = _read_jsonl(metadata / "advanced_vision_evidence.jsonl")
    return {
        "video_events_by_id": {str(row.get("video_event_id")): row for row in video_events if row.get("video_event_id")},
        "states_by_id": {str(row.get("state_change_id")): row for row in state_rows if row.get("state_change_id")},
        "assets_by_id": {str(row.get("asset_id")): row for row in assets if row.get("asset_id")},
        "assets_by_path": {str(row.get("path")): row for row in assets if row.get("path")},
        "segments_by_id": {str(row.get("segment_id")): row for row in segments if row.get("segment_id")},
        "micros_by_id": {str(row.get("micro_segment_id")): row for row in micros if row.get("micro_segment_id")},
        "model_events_by_id": {str(row.get("observation_id")): row for row in model_events if row.get("observation_id")},
        "advanced_evidence_by_id": {str(row.get("evidence_id")): row for row in advanced_evidence if row.get("evidence_id")},
    }


def _complete_step_refs(step: Mapping[str, Any], order: int) -> list[dict[str, Any]]:
    refs = [dict(ref) for ref in step.get("evidence_refs") or [] if isinstance(ref, Mapping)]
    step_id = str(step.get("step_id") or f"step_{order:03d}")
    if not any(str(ref.get("type") or "") == "sop_step" for ref in refs):
        refs.append(
            {
                "type": "sop_step",
                "evidence_id": f"sop:{step_id}",
                "sop_step_id": step_id,
                "name": step.get("name"),
                "expected_action": step.get("expected_action"),
                "confidence": 1.0,
                "evidence_level": "text_support",
            }
        )
    if (step.get("inferred") or step.get("status") in {"inferred_missing", "skipped_or_unobserved"}) and not any(
        str(ref.get("type") or "") == "inference" for ref in refs
    ):
        refs.append(
            {
                "type": "inference",
                "evidence_id": f"inference:{step_id}",
                "inference_id": f"inference:{step_id}",
                "source_step_ids": step.get("inferred_from_step_ids") or [],
                "reason": step.get("inference_reason") or step.get("missing_completion_reason"),
                "confidence": step.get("inference_confidence") or step.get("confidence"),
                "evidence_level": "inferred_support",
            }
        )
    return _dedupe_refs(refs)


def _resolve_evidence(ref: Mapping[str, Any], sources: Mapping[str, Any]) -> dict[str, Any]:
    ref_type = str(ref.get("type") or "unknown")
    if ref_type == "video_event":
        event_id = str(ref.get("video_event_id") or ref.get("evidence_id") or "")
        event = sources.get("video_events_by_id", {}).get(event_id)
        return _video_event_entry(event_id, ref, event, sources)
    if ref_type == "state_change":
        state_id = str(ref.get("state_change_id") or ref.get("evidence_id") or "")
        state = sources.get("states_by_id", {}).get(state_id)
        return _state_entry(state_id, ref, state, sources)
    if ref_type == "asset":
        asset_id = str(ref.get("asset_id") or ref.get("evidence_id") or "")
        asset = sources.get("assets_by_id", {}).get(asset_id) or sources.get("assets_by_path", {}).get(str(ref.get("path") or ""))
        return _asset_entry(asset_id, ref, asset)
    if ref_type == "database_record":
        return _generic_entry(ref, evidence_type="database_record", resolved=True)
    if ref_type == "inference":
        return _generic_entry(ref, evidence_type="inference", resolved=True)
    if ref_type == "sop_step":
        return _generic_entry(ref, evidence_type="sop_step", resolved=True)
    if ref_type == "model_event":
        model_id = str(ref.get("model_event_id") or ref.get("observation_id") or ref.get("evidence_id") or "")
        model_event = sources.get("model_events_by_id", {}).get(model_id)
        return _model_event_entry(model_id, ref, model_event)
    return _generic_entry(ref, evidence_type=ref_type, resolved=False)


def _video_event_entry(event_id: str, ref: Mapping[str, Any], event: Any, sources: Mapping[str, Any]) -> dict[str, Any]:
    event_data = event if isinstance(event, Mapping) else {}
    evidence_level = _classify_level(ref, event_data, default="visual_candidate")
    segment_id = event_data.get("segment_id") or ref.get("segment_id")
    micro_id = event_data.get("micro_segment_id") or ref.get("micro_segment_id")
    return {
        "evidence_id": event_id or str(ref.get("evidence_id") or "video_event:unknown"),
        "type": "video_event",
        "resolved": isinstance(event, Mapping),
        "source_id": event_id,
        "event_type": event_data.get("event_type"),
        "segment_id": segment_id,
        "micro_segment_id": micro_id,
        "global_start_time": event_data.get("global_start_time") or ref.get("global_start_time"),
        "global_end_time": event_data.get("global_end_time") or ref.get("global_end_time"),
        "confidence": _bounded_float(event_data.get("confidence", ref.get("confidence"))),
        "evidence_level": evidence_level,
        "strength_score": _level_score(evidence_level),
        "text": event_data.get("text"),
        "trace": {
            "video_event_id": event_id,
            "segment_id": segment_id,
            "micro_segment_id": micro_id,
            "keyframe_refs": _keyframe_refs(event_data, sources, segment_id=segment_id, micro_segment_id=micro_id),
            "clip_refs": _clip_refs(sources, segment_id=segment_id, micro_segment_id=micro_id),
            "model_event_id": _model_event_id(event_data),
        },
    }


def _state_entry(state_id: str, ref: Mapping[str, Any], state: Any, sources: Mapping[str, Any]) -> dict[str, Any]:
    state_data = state if isinstance(state, Mapping) else {}
    evidence_level = _classify_level(ref, state_data, default="state_change")
    segment_id = state_data.get("segment_id") or ref.get("segment_id")
    micro_id = state_data.get("micro_segment_id") or ref.get("micro_segment_id")
    return {
        "evidence_id": state_id or str(ref.get("evidence_id") or "state_change:unknown"),
        "type": "state_change",
        "resolved": isinstance(state, Mapping),
        "source_id": state_id,
        "state_type": state_data.get("state_type") or ref.get("state_type"),
        "segment_id": segment_id,
        "micro_segment_id": micro_id,
        "global_start_time": state_data.get("global_start_time") or state_data.get("global_time") or ref.get("global_start_time"),
        "global_end_time": state_data.get("global_end_time") or state_data.get("global_time") or ref.get("global_end_time"),
        "confidence": _bounded_float(state_data.get("confidence", ref.get("confidence"))),
        "evidence_level": evidence_level,
        "strength_score": _level_score(evidence_level),
        "text": state_data.get("text") or state_data.get("description"),
        "trace": {
            "state_change_id": state_id,
            "segment_id": segment_id,
            "micro_segment_id": micro_id,
            "keyframe_refs": _keyframe_refs(state_data, sources, segment_id=segment_id, micro_segment_id=micro_id),
            "clip_refs": _clip_refs(sources, segment_id=segment_id, micro_segment_id=micro_id),
        },
    }


def _asset_entry(asset_id: str, ref: Mapping[str, Any], asset: Any) -> dict[str, Any]:
    asset_data = asset if isinstance(asset, Mapping) else {}
    evidence_level = _classify_level(ref, asset_data, default="visual_asset")
    path = asset_data.get("path") or ref.get("path")
    return {
        "evidence_id": asset_id or str(path or ref.get("evidence_id") or "asset:unknown"),
        "type": "asset",
        "resolved": isinstance(asset, Mapping) or bool(path),
        "source_id": asset_id,
        "asset_type": asset_data.get("asset_type") or ref.get("asset_type"),
        "segment_id": asset_data.get("segment_id") or ref.get("segment_id"),
        "micro_segment_id": asset_data.get("micro_segment_id") or ref.get("micro_segment_id"),
        "global_start_time": asset_data.get("global_start_time") or ref.get("global_start_time"),
        "global_end_time": asset_data.get("global_end_time") or ref.get("global_end_time"),
        "confidence": _bounded_float(asset_data.get("confidence", ref.get("confidence"))),
        "evidence_level": evidence_level,
        "strength_score": _level_score(evidence_level),
        "path": path,
        "text": asset_data.get("search_text"),
        "trace": {
            "asset_id": asset_id,
            "path": path,
            "keyframe_refs": [_media_ref("keyframe", path, source="asset")] if _looks_like_keyframe(path, asset_data.get("asset_type")) else [],
            "clip_refs": [_media_ref("clip", path, source="asset")] if _looks_like_clip(path, asset_data.get("asset_type")) else [],
        },
    }


def _model_event_entry(model_id: str, ref: Mapping[str, Any], model_event: Any) -> dict[str, Any]:
    data = model_event if isinstance(model_event, Mapping) else {}
    evidence_level = _classify_level(ref, data, default="model_confirmed")
    return {
        "evidence_id": model_id or str(ref.get("evidence_id") or "model_event:unknown"),
        "type": "model_event",
        "resolved": isinstance(model_event, Mapping),
        "source_id": model_id,
        "event_type": data.get("event_type"),
        "segment_id": data.get("segment_id") or ref.get("segment_id"),
        "micro_segment_id": data.get("micro_segment_id") or ref.get("micro_segment_id"),
        "global_start_time": data.get("global_start_time") or ref.get("global_start_time"),
        "global_end_time": data.get("global_end_time") or ref.get("global_end_time"),
        "confidence": _bounded_float(data.get("confidence", ref.get("confidence"))),
        "evidence_level": evidence_level,
        "strength_score": _level_score(evidence_level),
        "text": data.get("text") or data.get("state"),
        "trace": {"model_event_id": model_id, "source_file": data.get("source_file")},
    }


def _generic_entry(ref: Mapping[str, Any], *, evidence_type: str, resolved: bool) -> dict[str, Any]:
    evidence_id = str(
        ref.get("evidence_id")
        or ref.get("inference_id")
        or ref.get("history_id")
        or ref.get("sop_step_id")
        or ref.get("database_record_id")
        or f"{evidence_type}:unknown"
    )
    evidence_level = _classify_level(ref, ref, default=_default_level(evidence_type))
    return {
        "evidence_id": evidence_id,
        "type": evidence_type,
        "resolved": resolved,
        "source_id": evidence_id,
        "confidence": _bounded_float(ref.get("confidence")),
        "evidence_level": evidence_level,
        "strength_score": _level_score(evidence_level),
        "text": ref.get("reason") or ref.get("name") or ref.get("expected_action"),
        "trace": {
            "source": ref.get("source"),
            "source_step_ids": ref.get("source_step_ids") or [],
            "history_basis": ref.get("history_basis") or {},
            "sop_step_id": ref.get("sop_step_id"),
        },
    }


def _attach_step(entries: dict[str, dict[str, Any]], evidence: Mapping[str, Any], step_id: str) -> None:
    evidence_id = str(evidence.get("evidence_id") or "")
    if not evidence_id:
        return
    existing = entries.setdefault(evidence_id, dict(evidence, step_ids=[]))
    if step_id and step_id not in existing["step_ids"]:
        existing["step_ids"].append(step_id)
    if float(evidence.get("strength_score") or 0.0) > float(existing.get("strength_score") or 0.0):
        existing.update({key: value for key, value in evidence.items() if key != "step_ids"})


def _step_evidence_ref(evidence: Mapping[str, Any], source_ref: Mapping[str, Any]) -> dict[str, Any]:
    trace = evidence.get("trace") if isinstance(evidence.get("trace"), Mapping) else {}
    return {
        "evidence_id": evidence.get("evidence_id"),
        "type": evidence.get("type"),
        "resolved": bool(evidence.get("resolved")),
        "evidence_level": evidence.get("evidence_level"),
        "strength_score": evidence.get("strength_score"),
        "confidence": evidence.get("confidence"),
        "segment_id": evidence.get("segment_id"),
        "micro_segment_id": evidence.get("micro_segment_id"),
        "path": evidence.get("path"),
        "text": evidence.get("text"),
        "trace": trace,
        "source_ref": dict(source_ref),
    }


def _inference_payload(step: Mapping[str, Any], refs: list[Mapping[str, Any]]) -> dict[str, Any]:
    inference_refs = [ref for ref in refs if str(ref.get("type") or "") in {"inference", "database_record"}]
    return {
        "inferred": bool(step.get("inferred")),
        "source": _strings(step.get("inference_source")) or (["status:" + str(step.get("status"))] if step.get("inferred") else []),
        "reason": step.get("inference_reason") or step.get("missing_completion_reason") or "",
        "confidence": _bounded_float(step.get("inference_confidence", step.get("confidence"))),
        "inferred_from_step_ids": _strings(step.get("inferred_from_step_ids")),
        "evidence_ids": [str(ref.get("evidence_id") or "") for ref in inference_refs if ref.get("evidence_id")],
    }


def _confirmation_payload(
    step: Mapping[str, Any],
    confirmation_queue: list[Mapping[str, Any]],
    session_id: str,
) -> dict[str, Any]:
    step_id = str(step.get("step_id") or "")
    confirmation_id = f"{session_id}:{step_id}" if step_id else ""
    queue_row = next((row for row in confirmation_queue if row.get("confirmation_id") == confirmation_id), {})
    confidence = _bounded_float(step.get("confidence")) or 0.0
    triggers = []
    if step.get("requires_human_confirmation"):
        triggers.append("process_requires_confirmation")
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        triggers.append("low_confidence")
    if step.get("inferred"):
        triggers.append("inferred_step")
    if step.get("abnormal") or step.get("conflict_flags"):
        triggers.append("abnormal_or_conflict")
    return {
        "confirmation_id": confirmation_id,
        "requires_human_confirmation": bool(step.get("requires_human_confirmation")),
        "status": step.get("confirmation_status") or queue_row.get("status") or ("pending" if step.get("requires_human_confirmation") else "auto_confirmed"),
        "policy": "manual_review" if triggers else "auto_confirm",
        "threshold": LOW_CONFIDENCE_THRESHOLD,
        "triggers": triggers,
        "queue_status": queue_row.get("status"),
        "decision": step.get("confirmation_decision") or queue_row.get("decision") or {},
    }


def _evidence_summary(refs: list[Mapping[str, Any]]) -> dict[str, Any]:
    levels = [str(ref.get("evidence_level") or "none") for ref in refs]
    scores = [float(ref.get("strength_score") or 0.0) for ref in refs]
    type_counts = Counter(str(ref.get("type") or "unknown") for ref in refs)
    return {
        "evidence_count": len(refs),
        "resolved_count": sum(1 for ref in refs if ref.get("resolved")),
        "type_counts": dict(sorted(type_counts.items())),
        "level_counts": dict(sorted(Counter(levels).items())),
        "strongest_level": levels[scores.index(max(scores))] if scores else "none",
        "strongest_score": max(scores) if scores else 0.0,
        "has_video_trace": any(ref.get("type") == "video_event" or _as_dict(ref.get("trace")).get("clip_refs") for ref in refs),
        "has_keyframe_trace": any(_as_dict(ref.get("trace")).get("keyframe_refs") for ref in refs),
        "has_text_or_database_trace": any(ref.get("type") in {"sop_step", "database_record", "inference"} for ref in refs),
    }


def _audit_flags(step: Mapping[str, Any], evidence_summary: Mapping[str, Any]) -> list[str]:
    flags = _strings(step.get("conflict_flags"))
    if int(evidence_summary.get("evidence_count") or 0) == 0:
        flags.append("missing_evidence_refs")
    if float(evidence_summary.get("strongest_score") or 0.0) < LOW_CONFIDENCE_THRESHOLD:
        flags.append("weak_evidence")
    if step.get("inferred") and not step.get("inference_reason") and not step.get("missing_completion_reason"):
        flags.append("missing_inference_reason")
    if step.get("requires_human_confirmation"):
        flags.append("pending_confirmation")
    return _dedupe_strings(flags)


def _write_audit_report(path: Path, record: Mapping[str, Any]) -> None:
    lines = [
        "# Process Audit Report",
        "",
        f"- Session: `{record.get('session_id')}`",
        f"- Generated: `{record.get('generated_at')}`",
        f"- Steps: {record.get('summary', {}).get('step_count', 0)}",
        f"- Pending confirmation: {record.get('summary', {}).get('pending_confirmation_count', 0)}",
        f"- Inferred steps: {record.get('summary', {}).get('inferred_step_count', 0)}",
        "",
        "## Step Findings",
        "",
        "| Step | Status | Confidence | Evidence | Confirmation | Flags |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for step in record.get("steps", []):
        evidence = step.get("evidence_summary") if isinstance(step, Mapping) else {}
        confirmation = step.get("confirmation") if isinstance(step, Mapping) else {}
        flags = ", ".join(_strings(step.get("audit_flags"))) or "none"
        lines.append(
            "| {step} | {status} | {confidence:.3f} | {level} ({count}) | {confirmation} | {flags} |".format(
                step=_escape_md(str(step.get("step_id") or "")),
                status=_escape_md(str(step.get("status") or "")),
                confidence=float(step.get("confidence") or 0.0),
                level=_escape_md(str(evidence.get("strongest_level") or "none")),
                count=int(evidence.get("evidence_count") or 0),
                confirmation=_escape_md(str(confirmation.get("status") or "")),
                flags=_escape_md(flags),
            )
        )
    lines.extend(["", "## Evidence Index", ""])
    for evidence_id, step_ids in sorted((record.get("evidence_index") or {}).items()):
        lines.append(f"- `{evidence_id}` -> {', '.join(f'`{step_id}`' for step_id in step_ids)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _keyframe_refs(event: Mapping[str, Any], sources: Mapping[str, Any], *, segment_id: Any, micro_segment_id: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for asset_ref in event.get("asset_refs") or []:
        if isinstance(asset_ref, Mapping):
            asset = sources.get("assets_by_id", {}).get(str(asset_ref.get("asset_id") or ""))
            path = asset_ref.get("path") or (asset or {}).get("path")
            if _looks_like_keyframe(path, asset_ref.get("asset_type") or (asset or {}).get("asset_type")):
                refs.append(_media_ref("keyframe", path, source="asset_ref", asset_id=asset_ref.get("asset_id")))
    micro = sources.get("micros_by_id", {}).get(str(micro_segment_id or ""))
    if isinstance(micro, Mapping):
        keyframes = micro.get("keyframes") if isinstance(micro.get("keyframes"), Mapping) else {}
        for role, path in keyframes.items():
            if path:
                refs.append(_media_ref("keyframe", path, source="micro_segment", role=role, micro_segment_id=micro_segment_id))
    segment = sources.get("segments_by_id", {}).get(str(segment_id or ""))
    if isinstance(segment, Mapping):
        for item in segment.get("interaction_keyframes") or []:
            if isinstance(item, Mapping) and item.get("path"):
                refs.append(_media_ref("keyframe", item.get("path"), source="segment", role=item.get("event_id"), segment_id=segment_id))
    return _dedupe_media(refs)


def _clip_refs(sources: Mapping[str, Any], *, segment_id: Any, micro_segment_id: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    micro = sources.get("micros_by_id", {}).get(str(micro_segment_id or ""))
    if isinstance(micro, Mapping):
        for view in ("first_person", "third_person"):
            view_data = micro.get(view) if isinstance(micro.get(view), Mapping) else {}
            if view_data.get("clip_path"):
                refs.append(_media_ref("clip", view_data.get("clip_path"), source="micro_segment", role=view, micro_segment_id=micro_segment_id))
    segment = sources.get("segments_by_id", {}).get(str(segment_id or ""))
    if isinstance(segment, Mapping):
        for view in ("first_person", "third_person"):
            view_data = segment.get(view) if isinstance(segment.get(view), Mapping) else {}
            if view_data.get("clip_path"):
                refs.append(_media_ref("clip", view_data.get("clip_path"), source="segment", role=view, segment_id=segment_id))
    return _dedupe_media(refs)


def _classify_level(ref: Mapping[str, Any], resolved: Mapping[str, Any], *, default: str) -> str:
    explicit = str(ref.get("evidence_level") or resolved.get("evidence_level") or "").strip()
    if explicit:
        return explicit
    confirmation_level = str(resolved.get("confirmation_level") or "").lower()
    confidence = _bounded_float(resolved.get("confidence", ref.get("confidence"))) or 0.0
    if confirmation_level in {"confirmed", "measured"}:
        return "model_confirmed"
    if default == "visual_candidate" and confidence >= 0.75:
        return "direct_visual"
    return default


def _default_level(evidence_type: str) -> str:
    return {
        "database_record": "database_record",
        "inference": "inferred_support",
        "sop_step": "text_support",
        "text": "text_support",
        "model_event": "model_confirmed",
    }.get(evidence_type, "weak")


def _level_score(level: Any) -> float:
    return float(EVIDENCE_LEVEL_SCORES.get(str(level or "weak"), 0.3))


def _model_event_id(event: Mapping[str, Any]) -> str | None:
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    for key in ("observation_id", "model_event_id", "source_observation_id"):
        if payload.get(key):
            return str(payload[key])
    return None


def _media_ref(ref_type: str, path: Any, *, source: str, **extra: Any) -> dict[str, Any]:
    return {"ref_type": ref_type, "path": str(path), "source": source, **{key: value for key, value in extra.items() if value}}


def _dedupe_media(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for ref in refs:
        key = (str(ref.get("ref_type") or ""), str(ref.get("path") or ""), str(ref.get("role") or ""))
        if key not in seen:
            seen.add(key)
            output.append(dict(ref))
    return output


def _dedupe_refs(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    output = []
    for ref in refs:
        key = (
            str(ref.get("type") or ""),
            str(
                ref.get("evidence_id")
                or ref.get("video_event_id")
                or ref.get("state_change_id")
                or ref.get("asset_id")
                or ref.get("inference_id")
                or ref.get("history_id")
                or ref.get("sop_step_id")
                or ref.get("path")
                or ""
            ),
        )
        if key not in seen:
            seen.add(key)
            output.append(dict(ref))
    return output


def _looks_like_keyframe(path: Any, asset_type: Any = None) -> bool:
    text = str(path or "").lower()
    return str(asset_type or "").lower() == "keyframe" or text.endswith((".jpg", ".jpeg", ".png", ".webp"))


def _looks_like_clip(path: Any, asset_type: Any = None) -> bool:
    text = str(path or "").lower()
    return str(asset_type or "").lower() == "clip" or text.endswith((".mp4", ".mov", ".avi", ".mkv"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _bounded_float(value: Any) -> float | None:
    try:
        return round(max(0.0, min(1.0, float(value))), 4)
    except (TypeError, ValueError):
        return None


def _strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item or "")]
    return [str(value)] if str(value or "") else []


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        text = str(value or "")
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = [
    "PROCESS_AUDIT_REPORT_FILENAME",
    "PROCESS_RECORD_FILENAME",
    "PROCESS_RECORD_SCHEMA_VERSION",
    "build_process_record",
]
