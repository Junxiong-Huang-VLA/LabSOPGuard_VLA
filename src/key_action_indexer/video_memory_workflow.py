from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence


WORKFLOW_REASONING_SCHEMA_VERSION = "video_memory.workflow_reasoning.v2"
CONTEXT_REASONING_SCHEMA_VERSION = "video_memory.context_reasoning.v3"

DEFAULT_INSTRUMENT_LABELS = {
    "balance",
    "scale",
    "panel",
    "display",
    "pipette",
    "beaker",
    "tube",
    "tube_rack",
    "rack",
    "bottle",
    "reagent_bottle",
    "weighing_paper",
    "paper",
    "spatula",
    "container",
    "flask",
    "vial",
}

CONTEXT_FIELD_ALIASES = {
    "sop_name": ("sop_name", "sop", "sop_title", "protocol_name", "protocol"),
    "sample_name": ("sample_name", "sample", "sample_id", "specimen_name"),
    "project_name": ("project_name", "project", "project_id", "study_name"),
}


def build_workflow_context_reasoning(
    *,
    clusters: Sequence[Mapping[str, Any]],
    ledger_events: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
    human_feedback_entries: Sequence[Mapping[str, Any]] | None = None,
    instrument_labels: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build V2 workflow reasoning and V3 context candidates from metadata.

    This function is intentionally pure: callers pass JSON-compatible rows in
    and receive JSON-compatible rows out. It does not decode video, call VLMs,
    mutate caches, or inspect workstation paths.
    """

    cluster_rows = [dict(row) for row in clusters if isinstance(row, Mapping)]
    ledger_rows = [dict(row) for row in ledger_events if isinstance(row, Mapping)]
    bundle_rows = [dict(row) for row in bundles if isinstance(row, Mapping)]
    feedback_rows = [dict(row) for row in human_feedback_entries or [] if isinstance(row, Mapping)]
    labels = {str(label).strip().lower() for label in (instrument_labels or DEFAULT_INSTRUMENT_LABELS) if str(label).strip()}
    indexes = _build_source_indexes(cluster_rows, ledger_rows, bundle_rows)

    workflow_patterns = _build_workflow_patterns(cluster_rows, ledger_rows, indexes)
    instrument_usage_patterns = _build_instrument_usage_patterns(
        cluster_rows,
        ledger_rows,
        indexes,
        instrument_labels=labels,
    )
    project_hints = _build_project_hints(workflow_patterns)
    human_contexts = _build_human_confirmed_contexts(cluster_rows, feedback_rows, indexes)
    context_candidates = _build_context_candidates(
        human_contexts,
        workflow_patterns,
        instrument_usage_patterns,
    )
    unresolved_questions = _build_unresolved_questions(
        cluster_rows,
        project_hints,
        human_contexts,
    )

    return {
        "schema_version": WORKFLOW_REASONING_SCHEMA_VERSION,
        "context_schema_version": CONTEXT_REASONING_SCHEMA_VERSION,
        "workflow_patterns": workflow_patterns,
        "instrument_usage_patterns": instrument_usage_patterns,
        "project_hints": project_hints,
        "project_or_context_hints": project_hints,
        "human_confirmed_contexts": human_contexts,
        "step_reasoning_candidates": context_candidates["step_reasoning_candidates"],
        "process_completion_candidates": context_candidates["process_completion_candidates"],
        "rule_candidates": context_candidates["rule_candidates"],
        "reminder_candidates": context_candidates["reminder_candidates"],
        "unresolved_questions": unresolved_questions,
    }


def workflow_reasoning_fingerprint(value: Mapping[str, Any] | Sequence[Any]) -> str:
    """Return a stable hash for cache-safe snapshot rebuild decisions."""

    return _stable_hash(_strip_volatile(value))


def _build_source_indexes(
    clusters: Sequence[Mapping[str, Any]],
    ledger_events: Sequence[Mapping[str, Any]],
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cluster_by_id = {str(row.get("cluster_id") or ""): dict(row) for row in clusters if row.get("cluster_id")}
    ledger_by_id = {str(row.get("ledger_event_id") or ""): dict(row) for row in ledger_events if row.get("ledger_event_id")}
    bundle_by_id = {str(row.get("bundle_id") or ""): dict(row) for row in bundles if row.get("bundle_id")}
    cluster_ids_by_ledger: dict[str, list[str]] = {}
    cluster_ids_by_bundle: dict[str, list[str]] = {}
    cluster_ids_by_material: dict[str, list[str]] = {}
    bundle_ids_by_material: dict[str, list[str]] = {}

    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id") or "")
        if not cluster_id:
            continue
        for ledger_id in _ensure_list(cluster.get("ledger_event_ids")):
            cluster_ids_by_ledger.setdefault(str(ledger_id), []).append(cluster_id)
        for bundle_id in _ensure_list(cluster.get("evidence_bundle_ids")):
            cluster_ids_by_bundle.setdefault(str(bundle_id), []).append(cluster_id)
        for material_id in _ensure_list(cluster.get("material_ids")):
            cluster_ids_by_material.setdefault(str(material_id), []).append(cluster_id)

    for ledger in ledger_events:
        ledger_id = str(ledger.get("ledger_event_id") or "")
        for bundle_id in _ensure_list(ledger.get("evidence_bundle_ids")):
            for cluster_id in cluster_ids_by_ledger.get(ledger_id, []):
                cluster_ids_by_bundle.setdefault(str(bundle_id), []).append(cluster_id)

    for bundle in bundles:
        bundle_id = str(bundle.get("bundle_id") or "")
        for material_id in _ensure_list(bundle.get("material_ids")):
            material_key = str(material_id)
            bundle_ids_by_material.setdefault(material_key, []).append(bundle_id)
            for cluster_id in cluster_ids_by_bundle.get(bundle_id, []):
                cluster_ids_by_material.setdefault(material_key, []).append(cluster_id)

    return {
        "cluster_by_id": cluster_by_id,
        "ledger_by_id": ledger_by_id,
        "bundle_by_id": bundle_by_id,
        "cluster_ids_by_ledger": {key: _unique_strings(value) for key, value in cluster_ids_by_ledger.items()},
        "cluster_ids_by_bundle": {key: _unique_strings(value) for key, value in cluster_ids_by_bundle.items()},
        "cluster_ids_by_material": {key: _unique_strings(value) for key, value in cluster_ids_by_material.items()},
        "bundle_ids_by_material": {key: _unique_strings(value) for key, value in bundle_ids_by_material.items()},
    }


def _build_workflow_patterns(
    clusters: Sequence[Mapping[str, Any]],
    ledger_events: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    for cluster in sorted(clusters, key=lambda row: str(row.get("cluster_id") or "")):
        if str(cluster.get("status") or "") in {"human_rejected", "suppressed", "archived", "expired_from_current_window"}:
            continue
        occurrence_count = int(_float(cluster.get("occurrence_count"), 0.0) or len(_ensure_list(cluster.get("ledger_event_ids"))))
        day_count = int(_float(cluster.get("day_count"), 0.0) or len(_ensure_list(cluster.get("related_dates"))))
        if occurrence_count < 2 and day_count < 2:
            continue
        trace = _trace_from_ids(
            cluster_ids=_ensure_list(cluster.get("cluster_id")),
            ledger_event_ids=_ensure_list(cluster.get("ledger_event_ids")),
            bundle_ids=_ensure_list(cluster.get("evidence_bundle_ids")),
            material_ids=_ensure_list(cluster.get("material_ids")),
            indexes=indexes,
        )
        if not _trace_is_complete(trace):
            continue
        actions = _unique_strings(cluster.get("canonical_actions") or [])
        objects = _unique_strings(cluster.get("key_objects") or [])
        instruments = _unique_strings(cluster.get("key_instruments") or [])
        sequence_signature = _ensure_list(
            (cluster.get("cluster_signature") or {}).get("sequence_signature")
            if isinstance(cluster.get("cluster_signature"), Mapping)
            else []
        )
        confidence = _pattern_confidence(cluster, occurrence_count, day_count)
        patterns.append(
            {
                "schema_version": WORKFLOW_REASONING_SCHEMA_VERSION,
                "pattern_id": _stable_id(
                    "workflow-pattern",
                    {
                        "cluster_id": cluster.get("cluster_id"),
                        "ledger_event_ids": trace["ledger_event_ids"],
                        "kind": "repeated_cluster",
                    },
                ),
                "reasoning_kind": "workflow_pattern",
                "pattern_type": "repeated_cluster",
                "candidate_status": "candidate",
                "summary": _workflow_summary(actions, objects, occurrence_count, day_count),
                "canonical_actions": actions,
                "key_objects": objects,
                "key_instruments": instruments,
                "sequence_signature": _unique_strings(sequence_signature or actions),
                "occurrence_count": occurrence_count,
                "day_count": day_count,
                "confidence": confidence,
                "evidence_basis": ["daily_event_ledger", "memory_cluster"],
                "evidence_trace": trace,
                "limitations": ["No experiment, project, sample, or SOP name is inferred without human confirmation."],
            }
        )

    patterns.extend(_build_repeated_sequence_patterns(ledger_events, indexes))
    return sorted(patterns, key=lambda row: (-float(row.get("confidence") or 0.0), str(row.get("pattern_id") or "")))


def _build_repeated_sequence_patterns(
    ledger_events: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for ledger in ledger_events:
        date_key = str(ledger.get("date") or "")
        session_key = str(ledger.get("session_id") or "")
        grouped.setdefault((date_key, session_key), []).append(ledger)

    by_sequence: dict[tuple[str, ...], list[list[Mapping[str, Any]]]] = {}
    for rows in grouped.values():
        ordered = sorted(rows, key=_ledger_sort_key)
        steps = [_ledger_step(row) for row in ordered]
        steps = [step for step in steps if step]
        if len(steps) < 2:
            continue
        by_sequence.setdefault(tuple(steps[:6]), []).append(ordered)

    patterns: list[dict[str, Any]] = []
    for sequence, grouped_rows in sorted(by_sequence.items(), key=lambda item: item[0]):
        dates = _unique_strings(row.get("date") for rows in grouped_rows for row in rows)
        if len(grouped_rows) < 2 and len(dates) < 2:
            continue
        ledger_ids = _unique_strings(row.get("ledger_event_id") for rows in grouped_rows for row in rows)
        trace = _trace_from_ids(ledger_event_ids=ledger_ids, indexes=indexes)
        if not _trace_is_complete(trace):
            continue
        confidences = [float(_float(row.get("confidence"), 0.0) or 0.0) for rows in grouped_rows for row in rows]
        confidence = _bounded(_mean(confidences) * 0.82 + min(len(dates), 5) * 0.03 + min(len(grouped_rows), 5) * 0.02)
        patterns.append(
            {
                "schema_version": WORKFLOW_REASONING_SCHEMA_VERSION,
                "pattern_id": _stable_id("workflow-pattern", {"sequence": sequence, "ledger_event_ids": ledger_ids}),
                "reasoning_kind": "workflow_pattern",
                "pattern_type": "repeated_sequence",
                "candidate_status": "candidate",
                "summary": f"Repeated ledger sequence observed across {len(dates)} day(s).",
                "canonical_actions": _unique_strings(step.split(":", 1)[0] for step in sequence),
                "key_objects": _unique_strings(step.split(":", 1)[1] for step in sequence if ":" in step),
                "key_instruments": [],
                "sequence_signature": list(sequence),
                "occurrence_count": len(grouped_rows),
                "day_count": len(dates),
                "confidence": confidence,
                "evidence_basis": ["daily_event_ledger"],
                "evidence_trace": trace,
                "limitations": ["Repeated order is a candidate workflow pattern, not a named experiment or SOP fact."],
            }
        )
    return patterns


def _build_instrument_usage_patterns(
    clusters: Sequence[Mapping[str, Any]],
    ledger_events: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Any],
    *,
    instrument_labels: set[str],
) -> list[dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for cluster in clusters:
        instruments = {
            token
            for token in _tokens(_join_text([cluster.get("key_instruments"), cluster.get("key_objects")]))
            if token in instrument_labels
        }
        for instrument in instruments:
            entry = sources.setdefault(instrument, {"clusters": [], "ledgers": [], "bundles": [], "materials": [], "confidences": []})
            entry["clusters"].append(cluster.get("cluster_id"))
            entry["ledgers"].extend(_ensure_list(cluster.get("ledger_event_ids")))
            entry["bundles"].extend(_ensure_list(cluster.get("evidence_bundle_ids")))
            entry["materials"].extend(_ensure_list(cluster.get("material_ids")))
            entry["confidences"].append(_float(cluster.get("confidence"), 0.0) or 0.0)

    for ledger in ledger_events:
        instruments = {
            token
            for token in _tokens(_join_text([ledger.get("primary_object"), ledger.get("detected_objects")]))
            if token in instrument_labels
        }
        for instrument in instruments:
            entry = sources.setdefault(instrument, {"clusters": [], "ledgers": [], "bundles": [], "materials": [], "confidences": []})
            entry["ledgers"].append(ledger.get("ledger_event_id"))
            entry["bundles"].extend(_ensure_list(ledger.get("evidence_bundle_ids")))
            entry["confidences"].append(_float(ledger.get("confidence"), 0.0) or 0.0)

    patterns: list[dict[str, Any]] = []
    ledger_by_id = indexes.get("ledger_by_id") or {}
    for instrument, source in sorted(sources.items()):
        trace = _trace_from_ids(
            cluster_ids=source["clusters"],
            ledger_event_ids=source["ledgers"],
            bundle_ids=source["bundles"],
            material_ids=source["materials"],
            indexes=indexes,
        )
        if not _trace_is_complete(trace):
            continue
        related_ledgers = [ledger_by_id.get(row_id) for row_id in trace["ledger_event_ids"] if ledger_by_id.get(row_id)]
        dates = _unique_strings(row.get("date") for row in related_ledgers)
        occurrence_count = len(trace["ledger_event_ids"])
        if occurrence_count < 2 and len(dates) < 2:
            continue
        actions = _unique_strings(_first_text(row.get("canonical_action_type"), row.get("action_name")) for row in related_ledgers)
        co_objects = _unique_strings(
            token
            for row in related_ledgers
            for token in _tokens(_join_text([row.get("primary_object"), row.get("detected_objects")]))
            if token and token != instrument
        )
        confidence = _bounded(_mean([float(value or 0.0) for value in source["confidences"]]) * 0.82 + min(occurrence_count, 6) * 0.025)
        patterns.append(
            {
                "schema_version": WORKFLOW_REASONING_SCHEMA_VERSION,
                "pattern_id": _stable_id("instrument-pattern", {"instrument": instrument, "trace": trace}),
                "reasoning_kind": "instrument_usage_pattern",
                "pattern_type": "instrument_usage_habit",
                "candidate_status": "candidate",
                "instrument_name": instrument,
                "usage_actions": actions,
                "co_observed_objects": co_objects,
                "occurrence_count": occurrence_count,
                "day_count": len(dates),
                "confidence": confidence,
                "evidence_basis": ["daily_event_ledger", "memory_cluster"],
                "evidence_trace": trace,
                "limitations": ["Instrument usage habit is a candidate pattern and needs review before operational use."],
            }
        )
    return sorted(patterns, key=lambda row: (-float(row.get("confidence") or 0.0), str(row.get("instrument_name") or "")))


def _build_project_hints(workflow_patterns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for pattern in workflow_patterns:
        occurrence_count = int(_float(pattern.get("occurrence_count"), 0.0) or 0)
        day_count = int(_float(pattern.get("day_count"), 0.0) or 0)
        if occurrence_count < 2 and day_count < 2:
            continue
        trace = dict(pattern.get("evidence_trace") or {})
        if not _trace_is_complete(trace):
            continue
        confidence = _bounded(min(float(pattern.get("confidence") or 0.0), 0.62))
        hints.append(
            {
                "schema_version": WORKFLOW_REASONING_SCHEMA_VERSION,
                "hint_id": _stable_id("project-hint", {"pattern_id": pattern.get("pattern_id"), "trace": trace}),
                "reasoning_kind": "project_hint",
                "hint_type": "possible_continuing_project_line",
                "candidate_status": "unresolved",
                "hint_text": "Repeated evidence suggests a continuing work line; project, sample, experiment, and SOP names require human confirmation.",
                "requires_human_confirmation": True,
                "proposed_project_name": "",
                "proposed_experiment_name": "",
                "suggested_context_fields": ["project_name", "sample_name", "sop_name"],
                "basis_pattern_id": pattern.get("pattern_id"),
                "confidence": confidence,
                "evidence_trace": trace,
                "limitations": ["No named project or experiment is written from evidence-only metadata."],
            }
        )
    return hints


def _build_human_confirmed_contexts(
    clusters: Sequence[Mapping[str, Any]],
    feedback_entries: Sequence[Mapping[str, Any]],
    indexes: Mapping[str, Any],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in sorted(feedback_entries, key=lambda row: (str(row.get("created_at") or ""), str(row.get("feedback_id") or ""))):
        fields = _normalize_context_fields(entry.get("context_fields") if isinstance(entry.get("context_fields"), Mapping) else {})
        if not fields:
            continue
        target_type = str(entry.get("target_type") or "")
        target_id = str(entry.get("target_id") or "")
        trace = _trace_for_target(target_type, target_id, indexes)
        context = _context_payload(
            fields,
            target_type=target_type,
            target_id=target_id,
            evidence_trace=trace,
            source_feedback_id=str(entry.get("feedback_id") or ""),
            confirmed_at=str(entry.get("created_at") or ""),
            user_id=str(entry.get("user_id") or ""),
            note=str(entry.get("note") or ""),
        )
        dedupe_key = _stable_hash({"target_type": target_type, "target_id": target_id, "fields": fields})
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        contexts.append(context)

    for cluster in clusters:
        fields = _normalize_context_fields(cluster.get("human_confirmed_fields") if isinstance(cluster.get("human_confirmed_fields"), Mapping) else {})
        if not fields:
            continue
        target_id = str(cluster.get("cluster_id") or "")
        trace = _trace_for_target("cluster", target_id, indexes)
        context = _context_payload(
            fields,
            target_type="cluster",
            target_id=target_id,
            evidence_trace=trace,
            source_feedback_id="",
            confirmed_at=str(cluster.get("updated_at") or ""),
            user_id="",
            note="",
        )
        dedupe_key = _stable_hash({"target_type": "cluster", "target_id": target_id, "fields": fields})
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        contexts.append(context)

    return sorted(contexts, key=lambda row: str(row.get("context_id") or ""))


def _build_context_candidates(
    human_contexts: Sequence[Mapping[str, Any]],
    workflow_patterns: Sequence[Mapping[str, Any]],
    instrument_usage_patterns: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    step_candidates: list[dict[str, Any]] = []
    completion_candidates: list[dict[str, Any]] = []
    rule_candidates: list[dict[str, Any]] = []
    reminder_candidates: list[dict[str, Any]] = []
    all_patterns = [*workflow_patterns, *instrument_usage_patterns]

    for context in human_contexts:
        context_trace = dict(context.get("evidence_trace") or {})
        if not _trace_is_complete(context_trace):
            continue
        matched_patterns = [pattern for pattern in all_patterns if _traces_overlap(context_trace, pattern.get("evidence_trace") or {})]
        if not matched_patterns:
            matched_patterns = [{"pattern_id": "", "confidence": 0.55, "evidence_trace": context_trace, "summary": ""}]
        for pattern in matched_patterns[:5]:
            trace = _combine_traces(context_trace, dict(pattern.get("evidence_trace") or {}))
            if not _trace_is_complete(trace):
                continue
            confidence = _bounded(min(0.92, max(float(pattern.get("confidence") or 0.55), 0.55) + 0.08))
            if context.get("sop_name"):
                step_candidates.append(_step_reasoning_candidate(context, pattern, trace, confidence))
                rule_candidates.append(_rule_candidate(context, pattern, trace, confidence))
                reminder_candidates.append(_reminder_candidate(context, pattern, trace, confidence))
            if context.get("project_name") or context.get("sample_name"):
                completion_candidates.append(_process_completion_candidate(context, pattern, trace, confidence))

    return {
        "step_reasoning_candidates": _dedupe_by_id(step_candidates, "candidate_id"),
        "process_completion_candidates": _dedupe_by_id(completion_candidates, "candidate_id"),
        "rule_candidates": _dedupe_by_id(rule_candidates, "candidate_id"),
        "reminder_candidates": _dedupe_by_id(reminder_candidates, "candidate_id"),
    }


def _step_reasoning_candidate(
    context: Mapping[str, Any],
    pattern: Mapping[str, Any],
    trace: Mapping[str, Any],
    confidence: float,
) -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_REASONING_SCHEMA_VERSION,
        "candidate_id": _stable_id("step-candidate", {"context": context.get("context_id"), "pattern": pattern.get("pattern_id")}),
        "reasoning_kind": "step_reasoning_candidate",
        "candidate_type": "sop_step_reasoning_candidate",
        "candidate_status": "candidate",
        "context_id": context.get("context_id"),
        "sop_name": context.get("sop_name") or "",
        "project_name": context.get("project_name") or "",
        "sample_name": context.get("sample_name") or "",
        "basis_pattern_id": pattern.get("pattern_id") or "",
        "candidate_text": "Use the human-confirmed SOP context to review possible next-step alignment for this observed workflow.",
        "compliance_fact_status": "not_evaluated",
        "confidence": confidence,
        "evidence_trace": dict(trace),
        "limitations": ["This is a candidate for reviewer use, not a SOP compliance fact."],
    }


def _process_completion_candidate(
    context: Mapping[str, Any],
    pattern: Mapping[str, Any],
    trace: Mapping[str, Any],
    confidence: float,
) -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_REASONING_SCHEMA_VERSION,
        "candidate_id": _stable_id("completion-candidate", {"context": context.get("context_id"), "pattern": pattern.get("pattern_id")}),
        "reasoning_kind": "process_completion_candidate",
        "candidate_type": "project_process_completion_candidate",
        "candidate_status": "candidate",
        "context_id": context.get("context_id"),
        "sop_name": context.get("sop_name") or "",
        "project_name": context.get("project_name") or "",
        "sample_name": context.get("sample_name") or "",
        "basis_pattern_id": pattern.get("pattern_id") or "",
        "candidate_text": "Use the confirmed project or sample context to review whether the observed workflow completes or continues a process segment.",
        "completion_fact_status": "not_asserted",
        "confidence": confidence,
        "evidence_trace": dict(trace),
        "limitations": ["Completion is a candidate interpretation and must remain reviewable."],
    }


def _rule_candidate(
    context: Mapping[str, Any],
    pattern: Mapping[str, Any],
    trace: Mapping[str, Any],
    confidence: float,
) -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_REASONING_SCHEMA_VERSION,
        "candidate_id": _stable_id("rule-candidate", {"context": context.get("context_id"), "pattern": pattern.get("pattern_id")}),
        "reasoning_kind": "rule_candidate",
        "candidate_type": "context_scoped_observation_rule",
        "candidate_status": "candidate",
        "context_id": context.get("context_id"),
        "sop_name": context.get("sop_name") or "",
        "project_name": context.get("project_name") or "",
        "sample_name": context.get("sample_name") or "",
        "basis_pattern_id": pattern.get("pattern_id") or "",
        "rule_text": "When this observed workflow recurs under the confirmed SOP context, surface it for human review.",
        "enforcement_status": "not_enforced",
        "compliance_fact_status": "not_evaluated",
        "confidence": confidence,
        "evidence_trace": dict(trace),
    }


def _reminder_candidate(
    context: Mapping[str, Any],
    pattern: Mapping[str, Any],
    trace: Mapping[str, Any],
    confidence: float,
) -> dict[str, Any]:
    return {
        "schema_version": CONTEXT_REASONING_SCHEMA_VERSION,
        "candidate_id": _stable_id("reminder-candidate", {"context": context.get("context_id"), "pattern": pattern.get("pattern_id")}),
        "reasoning_kind": "reminder_candidate",
        "candidate_type": "candidate_only_real_time_prompt",
        "candidate_status": "candidate",
        "context_id": context.get("context_id"),
        "sop_name": context.get("sop_name") or "",
        "project_name": context.get("project_name") or "",
        "sample_name": context.get("sample_name") or "",
        "basis_pattern_id": pattern.get("pattern_id") or "",
        "reminder_text": "Candidate reminder for a reviewer to check the relevant SOP next-step context.",
        "delivery_status": "not_scheduled",
        "force_real_time_alert": False,
        "compliance_fact_status": "not_evaluated",
        "confidence": confidence,
        "evidence_trace": dict(trace),
    }


def _build_unresolved_questions(
    clusters: Sequence[Mapping[str, Any]],
    project_hints: Sequence[Mapping[str, Any]],
    human_contexts: Sequence[Mapping[str, Any]],
) -> list[str]:
    questions = _unique_strings(question for cluster in clusters for question in _ensure_list(cluster.get("unresolved_questions")))
    if project_hints and not human_contexts:
        questions.append("Which SOP, project, and sample context should be attached to the repeated workflow evidence?")
    if project_hints:
        questions.append("Should any repeated workflow candidate be promoted after human confirmation?")
    return _unique_strings(questions)


def _context_payload(
    fields: Mapping[str, str],
    *,
    target_type: str,
    target_id: str,
    evidence_trace: Mapping[str, Any],
    source_feedback_id: str,
    confirmed_at: str,
    user_id: str,
    note: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": CONTEXT_REASONING_SCHEMA_VERSION,
        "context_id": _stable_id(
            "human-context",
            {
                "target_type": target_type,
                "target_id": target_id,
                "fields": dict(fields),
            },
        ),
        "context_type": "human_confirmed_sop_project_sample_context",
        "confirmation_status": "human_confirmed",
        "target_type": target_type,
        "target_id": target_id,
        "sop_name": fields.get("sop_name", ""),
        "project_name": fields.get("project_name", ""),
        "sample_name": fields.get("sample_name", ""),
        "source_feedback_id": source_feedback_id,
        "confirmed_at": confirmed_at,
        "user_id": user_id,
        "note": note,
        "confidence": 1.0,
        "evidence_trace": dict(evidence_trace),
        "limitations": ["Human context names are confirmed labels; SOP compliance still requires separate review."],
    }
    payload["applies_to_cluster_ids"] = _ensure_list(payload["evidence_trace"].get("cluster_ids"))
    payload["applies_to_ledger_event_ids"] = _ensure_list(payload["evidence_trace"].get("ledger_event_ids"))
    return payload


def _normalize_context_fields(fields: Mapping[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for canonical, aliases in CONTEXT_FIELD_ALIASES.items():
        for alias in aliases:
            value = fields.get(alias)
            if value not in (None, ""):
                normalized[canonical] = str(value).strip()
                break
    return {key: value for key, value in normalized.items() if value}


def _trace_for_target(target_type: str, target_id: str, indexes: Mapping[str, Any]) -> dict[str, Any]:
    normalized_type = str(target_type or "").strip().lower()
    target = str(target_id or "").strip()
    if normalized_type == "cluster":
        return _trace_from_ids(cluster_ids=[target], indexes=indexes)
    if normalized_type in {"ledger", "ledger_event", "daily_event_ledger"}:
        return _trace_from_ids(ledger_event_ids=[target], indexes=indexes)
    if normalized_type in {"bundle", "evidence_bundle"}:
        return _trace_from_ids(bundle_ids=[target], indexes=indexes)
    if normalized_type in {"material", "material_id"}:
        return _trace_from_ids(material_ids=[target], indexes=indexes)
    trace = _trace_from_ids(indexes=indexes)
    trace["target_type"] = normalized_type
    trace["target_id"] = target
    return trace


def _trace_from_ids(
    *,
    cluster_ids: Sequence[Any] | None = None,
    ledger_event_ids: Sequence[Any] | None = None,
    bundle_ids: Sequence[Any] | None = None,
    material_ids: Sequence[Any] | None = None,
    indexes: Mapping[str, Any],
) -> dict[str, Any]:
    cluster_ids_out = _unique_strings(cluster_ids or [])
    ledger_ids_out = _unique_strings(ledger_event_ids or [])
    bundle_ids_out = _unique_strings(bundle_ids or [])
    material_ids_out = _unique_strings(material_ids or [])

    cluster_by_id = indexes.get("cluster_by_id") or {}
    ledger_by_id = indexes.get("ledger_by_id") or {}
    bundle_by_id = indexes.get("bundle_by_id") or {}
    cluster_ids_by_ledger = indexes.get("cluster_ids_by_ledger") or {}
    cluster_ids_by_bundle = indexes.get("cluster_ids_by_bundle") or {}
    cluster_ids_by_material = indexes.get("cluster_ids_by_material") or {}
    bundle_ids_by_material = indexes.get("bundle_ids_by_material") or {}

    for _ in range(3):
        for cluster_id in list(cluster_ids_out):
            cluster = cluster_by_id.get(cluster_id) or {}
            ledger_ids_out = _unique_strings([*ledger_ids_out, *_ensure_list(cluster.get("ledger_event_ids"))])
            bundle_ids_out = _unique_strings([*bundle_ids_out, *_ensure_list(cluster.get("evidence_bundle_ids"))])
            material_ids_out = _unique_strings([*material_ids_out, *_ensure_list(cluster.get("material_ids"))])
        for ledger_id in list(ledger_ids_out):
            ledger = ledger_by_id.get(ledger_id) or {}
            cluster_ids_out = _unique_strings([*cluster_ids_out, *_ensure_list(cluster_ids_by_ledger.get(ledger_id))])
            bundle_ids_out = _unique_strings([*bundle_ids_out, *_ensure_list(ledger.get("evidence_bundle_ids"))])
        for bundle_id in list(bundle_ids_out):
            bundle = bundle_by_id.get(bundle_id) or {}
            cluster_ids_out = _unique_strings([*cluster_ids_out, *_ensure_list(cluster_ids_by_bundle.get(bundle_id))])
            material_ids_out = _unique_strings([*material_ids_out, *_ensure_list(bundle.get("material_ids"))])
        for material_id in list(material_ids_out):
            cluster_ids_out = _unique_strings([*cluster_ids_out, *_ensure_list(cluster_ids_by_material.get(material_id))])
            bundle_ids_out = _unique_strings([*bundle_ids_out, *_ensure_list(bundle_ids_by_material.get(material_id))])

    related_clusters = [cluster_by_id.get(row_id) or {} for row_id in cluster_ids_out]
    related_ledgers = [ledger_by_id.get(row_id) or {} for row_id in ledger_ids_out]
    related_bundles = [bundle_by_id.get(row_id) or {} for row_id in bundle_ids_out]
    sha256s = _unique_strings(
        [sha for row in related_clusters for sha in _ensure_list(row.get("sha256s"))]
        + [sha for row in related_ledgers for sha in _ensure_list(row.get("sha256s"))]
        + [sha for row in related_bundles for sha in _ensure_list(row.get("sha256s"))]
    )
    micro_segment_ids = _unique_strings(
        [row.get("micro_segment_id") for row in related_ledgers]
        + [micro_id for row in related_ledgers for micro_id in _ensure_list(row.get("micro_segment_ids"))]
        + [row.get("micro_segment_id") for row in related_bundles]
        + [micro_id for row in related_bundles for micro_id in _ensure_list(row.get("micro_segment_ids"))]
        + [micro_id for row in related_clusters for micro_id in _ensure_list(row.get("micro_segment_ids"))]
    )
    keyframe_refs = _unique_strings(
        [path for row in related_bundles for path in _ensure_list(row.get("keyframe_refs") or row.get("keyframes"))]
        + [path for row in related_ledgers for path in _ensure_list(row.get("keyframe_refs"))]
        + [path for row in related_clusters for path in _ensure_list(row.get("keyframe_refs"))]
    )
    keyclip_refs = _unique_strings(
        [path for row in related_bundles for path in _ensure_list(row.get("keyclip_refs") or row.get("keyclips"))]
        + [path for row in related_ledgers for path in _ensure_list(row.get("keyclip_refs"))]
        + [path for row in related_clusters for path in _ensure_list(row.get("keyclip_refs"))]
    )
    timestamps = [
        row.get("timestamp") or row.get("time_range")
        for row in [*related_ledgers, *related_bundles, *related_clusters]
        if row.get("timestamp") or row.get("time_range")
    ]
    trace = {
        "cluster_id": cluster_ids_out[0] if cluster_ids_out else "",
        "cluster_ids": cluster_ids_out,
        "ledger_event_id": ledger_ids_out[0] if ledger_ids_out else "",
        "ledger_event_ids": ledger_ids_out,
        "bundle_id": bundle_ids_out[0] if bundle_ids_out else "",
        "bundle_ids": bundle_ids_out,
        "evidence_bundle_ids": bundle_ids_out,
        "material_id": material_ids_out[0] if material_ids_out else "",
        "material_ids": material_ids_out,
        "sha256": sha256s[0] if len(sha256s) == 1 else _stable_hash({"sha256s": sha256s}) if sha256s else "",
        "sha256s": sha256s,
        "micro_segment_id": micro_segment_ids[0] if micro_segment_ids else "",
        "micro_segment_ids": micro_segment_ids,
        "keyframe": keyframe_refs[0] if keyframe_refs else "",
        "keyframe_refs": keyframe_refs,
        "keyclip": keyclip_refs[0] if keyclip_refs else "",
        "keyclip_refs": keyclip_refs,
        "timestamp": timestamps[0] if timestamps else {},
        "timestamps": timestamps,
        "trace_schema_version": "video_memory.reasoning_trace.v1",
    }
    trace["trace_complete"] = _trace_is_complete(trace)
    return trace


def _combine_traces(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    trace = {
        "cluster_ids": _unique_strings([*_ensure_list(left.get("cluster_ids")), *_ensure_list(right.get("cluster_ids"))]),
        "ledger_event_ids": _unique_strings([*_ensure_list(left.get("ledger_event_ids")), *_ensure_list(right.get("ledger_event_ids"))]),
        "bundle_ids": _unique_strings([*_ensure_list(left.get("bundle_ids")), *_ensure_list(right.get("bundle_ids"))]),
        "material_ids": _unique_strings([*_ensure_list(left.get("material_ids")), *_ensure_list(right.get("material_ids"))]),
        "sha256s": _unique_strings([*_ensure_list(left.get("sha256s")), *_ensure_list(right.get("sha256s"))]),
        "micro_segment_ids": _unique_strings([*_ensure_list(left.get("micro_segment_ids")), *_ensure_list(right.get("micro_segment_ids"))]),
        "keyframe_refs": _unique_strings([*_ensure_list(left.get("keyframe_refs")), *_ensure_list(right.get("keyframe_refs"))]),
        "keyclip_refs": _unique_strings([*_ensure_list(left.get("keyclip_refs")), *_ensure_list(right.get("keyclip_refs"))]),
        "timestamps": [*_ensure_list(left.get("timestamps")), *_ensure_list(right.get("timestamps"))],
        "trace_schema_version": "video_memory.reasoning_trace.v1",
    }
    trace["evidence_bundle_ids"] = trace["bundle_ids"]
    trace["cluster_id"] = trace["cluster_ids"][0] if trace["cluster_ids"] else ""
    trace["ledger_event_id"] = trace["ledger_event_ids"][0] if trace["ledger_event_ids"] else ""
    trace["bundle_id"] = trace["bundle_ids"][0] if trace["bundle_ids"] else ""
    trace["material_id"] = trace["material_ids"][0] if trace["material_ids"] else ""
    trace["sha256"] = trace["sha256s"][0] if len(trace["sha256s"]) == 1 else _stable_hash({"sha256s": trace["sha256s"]}) if trace["sha256s"] else ""
    trace["micro_segment_id"] = trace["micro_segment_ids"][0] if trace["micro_segment_ids"] else ""
    trace["keyframe"] = trace["keyframe_refs"][0] if trace["keyframe_refs"] else ""
    trace["keyclip"] = trace["keyclip_refs"][0] if trace["keyclip_refs"] else ""
    trace["timestamp"] = trace["timestamps"][0] if trace["timestamps"] else {}
    trace["trace_complete"] = _trace_is_complete(trace)
    return trace


def _trace_is_complete(trace: Mapping[str, Any]) -> bool:
    return bool(trace.get("cluster_id") and trace.get("ledger_event_id") and trace.get("bundle_id") and trace.get("material_id"))


def _traces_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in ("cluster_ids", "ledger_event_ids", "bundle_ids", "material_ids"):
        if set(_ensure_list(left.get(key))) & set(_ensure_list(right.get(key))):
            return True
    return False


def _workflow_summary(actions: Sequence[str], objects: Sequence[str], occurrence_count: int, day_count: int) -> str:
    action_text = ", ".join(actions[:3]) if actions else "observed actions"
    object_text = ", ".join(objects[:4]) if objects else "tracked objects"
    return f"Repeated workflow candidate: {action_text} involving {object_text}, observed {occurrence_count} time(s) across {day_count} day(s)."


def _pattern_confidence(cluster: Mapping[str, Any], occurrence_count: int, day_count: int) -> float:
    base = float(_float(cluster.get("confidence"), 0.0) or 0.0)
    view_bonus = 0.05 if len(_ensure_list(cluster.get("view_coverage"))) >= 2 else 0.0
    repeat_bonus = min(0.18, occurrence_count * 0.025 + day_count * 0.035)
    return _bounded(base * 0.82 + repeat_bonus + view_bonus)


def _ledger_step(row: Mapping[str, Any]) -> str:
    action = _first_text(row.get("canonical_action_type"), row.get("action_name"))
    primary = _first_text(row.get("primary_object"))
    if not action and not primary:
        return ""
    return f"{action or 'unknown_action'}:{primary or 'unknown_object'}"


def _ledger_sort_key(row: Mapping[str, Any]) -> tuple[str, str, float, str]:
    time_range = row.get("time_range") if isinstance(row.get("time_range"), Mapping) else {}
    return (
        str(row.get("date") or ""),
        str(row.get("session_id") or ""),
        float(_float(time_range.get("start_sec"), 0.0) or 0.0),
        str(row.get("ledger_event_id") or ""),
    )


def _dedupe_by_id(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = str(row.get(key) or "")
        if not row_id or row_id in seen:
            continue
        seen.add(row_id)
        output.append(dict(row))
    return output


def _strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        omitted = {"generated_at", "created_at", "updated_at", "confirmed_at", "source_feedback_id", "note"}
        return {str(key): _strip_volatile(item) for key, item in sorted(value.items()) if str(key) not in omitted}
    if isinstance(value, (list, tuple)):
        return [_strip_volatile(item) for item in value]
    return value


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, str):
        if not value:
            return []
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return loaded
        except json.JSONDecodeError:
            pass
        return [value]
    return [value]


def _unique_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            for item in _unique_strings(value):
                if item not in seen:
                    seen.add(item)
                    results.append(item)
            continue
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        results.append(text)
    return results


def _join_text(values: Iterable[Any]) -> str:
    parts: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, Mapping):
            parts.append(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
        elif isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value if str(item or "").strip())
        else:
            text = str(value).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _tokens(value: Any) -> list[str]:
    text = _join_text([value]).lower()
    if not text:
        return []
    return [token for token in re.split(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", text) if token]


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _mean(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return 0.0
    return round(sum(filtered) / len(filtered), 4)


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{_stable_hash(value)[:24]}"


__all__ = [
    "CONTEXT_REASONING_SCHEMA_VERSION",
    "WORKFLOW_REASONING_SCHEMA_VERSION",
    "build_workflow_context_reasoning",
    "workflow_reasoning_fingerprint",
]
