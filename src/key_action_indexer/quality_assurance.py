from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl
from .scope_config import load_stage_scope, split_scope_values


CHECK_FIELDS = (
    "check_id",
    "name",
    "status",
    "score",
    "details",
    "recommendations",
    "blocking_tasks",
    "suggested_commands",
    "required_inputs",
)

TASK_TEMPLATES: dict[str, dict[str, Any]] = {
    "time_alignment": {
        "task_id": "P-03/T-TIME-ALIGNMENT-ANCHORS",
        "title": "Backfill multimodal timeline anchors",
        "required_inputs": ["manifest.json", "metadata/unified_multimodal_timeline.jsonl", "metadata/time_calibration_report.json"],
        "suggested_commands": ["timeline", "quality-report"],
    },
    "segment_completeness": {
        "task_id": "P-01/T-YOLO-MICROSEGMENT-REFINE",
        "title": "Regenerate YOLO-backed key action and micro-segments",
        "required_inputs": ["manifest.json", "metadata/key_action_segments.jsonl", "metadata/micro_segments.jsonl", "cv_outputs/yolo_frame_rows.jsonl"],
        "suggested_commands": ["run-dry", "quality-report"],
    },
    "keyframe_representativeness": {
        "task_id": "P-01/T-KEYFRAME-BACKFILL",
        "title": "Backfill contact, peak, and release keyframes",
        "required_inputs": ["metadata/micro_segments.jsonl", "metadata/material_asset_catalog.jsonl", "keyframes/"],
        "suggested_commands": ["assets", "quality-report"],
    },
    "action_recognition": {
        "task_id": "P-01/T-ACTION-CLASSIFICATION-REVIEW",
        "title": "Review low-confidence action classifications",
        "required_inputs": ["metadata/video_understanding.jsonl", "metadata/micro_segments.jsonl", "metadata/material_asset_catalog.jsonl"],
        "suggested_commands": ["understand-video", "quality-report"],
    },
    "state_change_detection": {
        "task_id": "P-01/T-STANDARD-STATE-EVIDENCE",
        "title": "Attach model-backed state-change evidence",
        "required_inputs": [
            "metadata/model_observation_events.jsonl",
            "metadata/liquid_segmentation.jsonl",
            "metadata/equipment_panel_states.jsonl",
            "metadata/container_state_events.jsonl",
            "metadata/object_tracks.jsonl",
            "metadata/advanced_vision_evidence_summary.json",
            "metadata/capability_gap_report.json",
        ],
        "suggested_commands": ["advanced-vision", "understand-video", "quality-report"],
    },
    "step_reasoning": {
        "task_id": "P-04/T-HUMAN-CONFIRMATION-BATCH",
        "title": "Resolve steps requiring human confirmation",
        "required_inputs": ["metadata/experiment_process.json", "metadata/human_confirmation_queue.jsonl", "metadata/human_confirmation_review_bundle.json"],
        "suggested_commands": ["confirmation-queue", "quality-report"],
    },
    "next_step_reasoning": {
        "task_id": "P-05/T-MISSING-STEP-RECOVERY",
        "title": "Recover current and next step evidence",
        "required_inputs": ["metadata/experiment_process.json", "metadata/video_understanding.jsonl", "transcript/aligned_transcript.jsonl"],
        "suggested_commands": ["process", "quality-report"],
    },
    "process_completion": {
        "task_id": "P-05/T-MISSING-STEP-RECOVERY",
        "title": "Recover unobserved or inferred process steps",
        "required_inputs": ["metadata/experiment_process.json", "metadata/video_understanding.jsonl", "transcript/aligned_transcript.jsonl", "metadata/material_asset_catalog.jsonl"],
        "suggested_commands": ["process", "confirmation-queue", "quality-report"],
    },
    "evidence_chain": {
        "task_id": "P-03/T-EVIDENCE-CHAIN-BACKFILL",
        "title": "Backfill process evidence references and reverse index",
        "required_inputs": ["metadata/experiment_process.json", "metadata/video_understanding.jsonl", "metadata/state_change_index.jsonl", "metadata/material_asset_catalog.jsonl"],
        "suggested_commands": ["process", "quality-report"],
    },
    "json_artifacts": {
        "task_id": "P-03/T-ARTIFACT-REBUILD",
        "title": "Rebuild missing QA input artifacts",
        "required_inputs": ["manifest.json", "metadata/"],
        "suggested_commands": ["run-dry", "quality-report"],
    },
    "searchability": {
        "task_id": "P-03/T-REINDEX-RETRIEVAL-METADATA",
        "title": "Rebuild searchable asset and vector metadata",
        "required_inputs": ["metadata/vector_metadata.jsonl", "metadata/material_asset_catalog.jsonl", "index/"],
        "suggested_commands": ["assets", "quality-report"],
    },
    "history_reuse": {
        "task_id": "P-03/T-HISTORY-MODEL-SEED",
        "title": "Seed or rebuild local history model",
        "required_inputs": ["metadata/history_model.json", "metadata/experiment_process.json"],
        "suggested_commands": ["history-model", "quality-report"],
    },
    "human_confirmation": {
        "task_id": "P-04/T-HUMAN-CONFIRMATION-BATCH",
        "title": "Resolve pending human confirmation queue items",
        "required_inputs": ["metadata/human_confirmation_queue.jsonl", "metadata/human_confirmation_review_bundle.json", "metadata/human_confirmation_review_summary.json"],
        "suggested_commands": ["confirmation-queue", "quality-report"],
    },
    "model_coverage": {
        "task_id": "P-02/T-CAPABILITY-GAP-AUDIT",
        "title": "Audit labels and model capabilities blocking strong confirmation",
        "required_inputs": ["metadata/model_inventory.json", "metadata/capability_gap_report.json", "LabSOPGuard/data/dataset", "LabSOPGuard/configs/data/class_schema.yaml"],
        "suggested_commands": ["model-inventory", "quality-report"],
    },
    "context_fusion": {
        "task_id": "P-03/T-CONTEXT-FUSION-INPUTS",
        "title": "Backfill procedure, material, and parameter context",
        "required_inputs": ["metadata/experiment_context.json", "transcript/aligned_transcript.jsonl", "metadata/material_asset_catalog.jsonl"],
        "suggested_commands": ["context", "quality-report"],
    },
}


def build_quality_assurance_report(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    target = Path(output_path) if output_path else metadata / "process_quality_report.json"
    timeline = _read_jsonl(metadata / "unified_multimodal_timeline.jsonl")
    segments = _read_jsonl(metadata / "key_action_segments.jsonl")
    micro_segments = _read_jsonl(metadata / "micro_segments.jsonl")
    video_events = _read_jsonl(metadata / "video_understanding.jsonl")
    assets = _read_jsonl(metadata / "material_asset_catalog.jsonl")
    vector_metadata = _read_jsonl(metadata / "vector_metadata.jsonl")
    confirmation_queue = _read_jsonl(metadata / "human_confirmation_queue.jsonl")
    time_anchors = _read_jsonl(metadata / "time_anchors.jsonl")
    stage_scope = load_stage_scope(session)
    capability_gap_path = metadata / "capability_gap_report.json"
    capability_gap_report = _read_json(capability_gap_path)
    capability_gap_summary = _capability_gap_summary(
        capability_gap_report,
        present=capability_gap_path.exists(),
        stage_scope=stage_scope,
    )
    calibration = _read_json(metadata / "time_calibration_report.json")
    process = _read_json(metadata / "experiment_process.json")
    context = _read_json(metadata / "experiment_context.json")
    model_inventory = _read_json(metadata / "model_inventory.json")
    history_model = _read_json(metadata / "history_model.json")
    artifact_validation = _read_json(metadata / "artifact_validation_report.json")

    checks = [
        _time_alignment_check(calibration, timeline, time_anchors),
        _segment_completeness_check(segments, micro_segments),
        _keyframe_representativeness_check(micro_segments, assets),
        _action_recognition_check(video_events),
        _state_change_check(video_events, metadata),
        _step_reasoning_check(process),
        _next_step_check(process),
        _completion_check(process, history_model),
        _evidence_chain_check(process),
        _json_artifact_check(metadata, artifact_validation),
        _searchability_check(vector_metadata, assets),
        _history_reuse_check(history_model),
        _human_confirmation_check(confirmation_queue, process),
        _model_coverage_check(model_inventory, capability_gap_summary, stage_scope),
        _context_fusion_check(context),
    ]
    checks = _attach_task_mappings(
        checks,
        session=session,
        metadata=metadata,
        process=process,
        confirmation_queue=confirmation_queue,
        capability_gap_summary=capability_gap_summary,
    )
    status_counts = Counter(str(check["status"]) for check in checks)
    score = round(sum(float(check.get("score") or 0.0) for check in checks) / max(len(checks), 1), 4)
    scorecard = _scorecard(checks)
    result = {
        "metadata_version": "key_action_quality_assurance.v1",
        "session_id": _session_id(session, process, context, timeline),
        "overall_status": _overall_status(status_counts),
        "overall_score": score,
        "scorecard": scorecard,
        "check_count": len(checks),
        "status_counts": dict(sorted(status_counts.items())),
        "checks": checks,
        "artifact_counts": {
            "timeline_events": len(timeline),
            "segments": len(segments),
            "micro_segments": len(micro_segments),
            "video_events": len(video_events),
            "assets": len(assets),
            "vector_metadata": len(vector_metadata),
            "confirmation_items": len(confirmation_queue),
            "capability_gap_reports": 1 if capability_gap_summary["present"] else 0,
            "capability_gap_items": capability_gap_summary["gap_count"],
            "out_of_scope_capability_gap_items": len(capability_gap_summary.get("out_of_scope_capabilities") or []),
        },
        "diagnostics": _diagnostics(checks, metadata, artifact_validation),
        "next_round_scheduler": _next_round_scheduler(checks),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _check(
    check_id: str,
    name: str,
    status: str,
    score: float,
    details: Mapping[str, Any] | None = None,
    recommendations: list[str] | None = None,
) -> dict[str, Any]:
    row = {
        "check_id": check_id,
        "name": name,
        "status": status,
        "score": round(float(score), 4),
        "details": dict(details or {}),
        "recommendations": recommendations or [],
        "blocking_tasks": [],
        "suggested_commands": [],
        "required_inputs": [],
    }
    return {field: row.get(field) for field in CHECK_FIELDS}


def _time_alignment_check(
    calibration: Mapping[str, Any],
    timeline: list[Mapping[str, Any]],
    time_anchors: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    sources = calibration.get("sources") if isinstance(calibration.get("sources"), Mapping) else {}
    artifact_anchored_sources = _artifact_anchored_sources(time_anchors or [])
    residuals = []
    unanchored = []
    for name, info in sources.items():
        if not isinstance(info, Mapping):
            continue
        residuals.append(float(info.get("residual_max_abs_sec") or 0.0))
        if int(info.get("anchor_count") or 0) == 0 and int(info.get("input_event_count") or 0) > 0:
            if str(name) not in artifact_anchored_sources:
                unanchored.append(str(name))
    max_residual = max(residuals, default=0.0)
    if not timeline:
        return _check("time_alignment", "Time alignment accuracy", "fail", 0.0, {"timeline_events": 0}, ["build unified timeline before QA"])
    if unanchored:
        status, score = "needs_review", 0.65
    elif max_residual > 1.0:
        status, score = "needs_review", 0.7
    else:
        status, score = "pass", 0.95
    return _check(
        "time_alignment",
        "Time alignment accuracy",
        status,
        score,
        {
            "timeline_events": len(timeline),
            "max_residual_sec": max_residual,
            "unanchored_sources": unanchored,
            "artifact_anchored_sources": sorted(artifact_anchored_sources),
        },
        ["add calibration anchors for user/AI/upload streams"] if unanchored else [],
    )


def _artifact_anchored_sources(time_anchors: list[Mapping[str, Any]]) -> set[str]:
    anchored: set[str] = set()
    for anchor in time_anchors:
        if not isinstance(anchor, Mapping):
            continue
        source_event = anchor.get("source_event") if isinstance(anchor.get("source_event"), Mapping) else {}
        target_event = anchor.get("target_event") if isinstance(anchor.get("target_event"), Mapping) else {}
        source = str(source_event.get("source") or "")
        reason = str(anchor.get("reason") or "")
        confidence = float(anchor.get("confidence") or 0.0)
        if source and target_event.get("global_time") and confidence > 0.0 and reason in {"absolute_time", "existing_global_time", "fallback_session_time"}:
            anchored.add(source)
    return anchored


def _segment_completeness_check(segments: list[Mapping[str, Any]], micro_segments: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not segments:
        return _check("segment_completeness", "Key segment completeness", "fail", 0.0, {"segments": 0}, ["run key action detection"])
    micro_ratio = len(micro_segments) / max(len(segments), 1)
    parent_ids = {str(row.get("segment_id") or "") for row in segments if row.get("segment_id")}
    parent_with_micro = {str(row.get("parent_segment_id") or "") for row in micro_segments if row.get("parent_segment_id")}
    missing_parent_ids = sorted(parent_id for parent_id in parent_ids if parent_id not in parent_with_micro)
    parent_coverage = len(parent_with_micro & parent_ids) / max(len(parent_ids), 1)
    status = "pass" if micro_ratio >= 1.0 and not missing_parent_ids else "needs_review"
    score = min(1.0, 0.55 + 0.20 * min(micro_ratio, 1.0) + 0.15 * parent_coverage)
    return _check(
        "segment_completeness",
        "Key segment completeness",
        status,
        score,
        {
            "segments": len(segments),
            "micro_segments": len(micro_segments),
            "micro_per_segment": round(micro_ratio, 3),
            "parent_micro_coverage_ratio": round(parent_coverage, 4),
            "parent_without_micro_segment_ids": missing_parent_ids,
        },
    )


def _keyframe_representativeness_check(micro_segments: list[Mapping[str, Any]], assets: list[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("contact_frame", "peak_frame", "release_frame")
    total = len(micro_segments) * len(required)
    present = 0
    for row in micro_segments:
        keyframes = row.get("keyframes") if isinstance(row.get("keyframes"), Mapping) else {}
        present += sum(1 for key in required if keyframes.get(key))
    ratio = present / max(total, 1)
    keyframe_assets = sum(1 for item in assets if str(item.get("asset_type") or "") == "keyframe")
    status = "pass" if ratio >= 0.8 else ("needs_review" if ratio > 0 else "fail")
    return _check(
        "keyframe_representativeness",
        "Keyframe representativeness",
        status,
        ratio,
        {"required_keyframes": total, "present_keyframes": present, "keyframe_assets": keyframe_assets},
        ["extract or backfill contact/peak/release keyframes"] if ratio < 0.8 else [],
    )


def _action_recognition_check(video_events: list[Mapping[str, Any]]) -> dict[str, Any]:
    action_rows = [row for row in video_events if str(row.get("event_type") or "") == "experiment_action_classification"]
    low_conf = [row for row in action_rows if float(row.get("confidence") or 0.0) < 0.6]
    status = "pass" if action_rows and not low_conf else ("needs_review" if action_rows else "fail")
    score = 0.0 if not action_rows else max(0.4, 1.0 - len(low_conf) / max(len(action_rows), 1))
    return _check("action_recognition", "Action recognition quality", status, score, {"action_events": len(action_rows), "low_confidence_events": len(low_conf)})


def _state_change_check(video_events: list[Mapping[str, Any]], metadata: Path) -> dict[str, Any]:
    state_types = {
        "object_state_change",
        "object_movement_detected",
        "object_track_observed",
        "container_state_change_candidate",
        "container_state_change_detected",
        "container_color_change_detected",
        "liquid_transfer_candidate",
        "liquid_flow_candidate_visual",
        "liquid_flow_detected",
        "liquid_level_change_detected",
        "equipment_panel_operation_candidate",
        "equipment_panel_operation_detected",
    }
    rows = [row for row in video_events if str(row.get("event_type") or "") in state_types]
    advanced_summary = _read_json(metadata / "advanced_vision_evidence_summary.json")
    confirmed_levels = [
        name
        for name, count in dict(advanced_summary.get("visual_confirmation_level_counts") or {}).items()
        if int(count or 0) > 0 and "confirmed" in str(name)
    ]
    status = "pass" if rows and confirmed_levels else ("needs_review" if rows else "fail")
    score = 0.8 if rows and confirmed_levels else (0.55 if rows else 0.0)
    return _check("state_change_detection", "State change detection reliability", status, score, {"state_events": len(rows), "confirmed_visual_levels": confirmed_levels}, ["train/add state-specific visual models"] if rows and not confirmed_levels else [])


def _step_reasoning_check(process: Mapping[str, Any]) -> dict[str, Any]:
    steps = _steps(process)
    if not steps:
        return _check("step_reasoning", "Step recognition correctness", "fail", 0.0, {"steps": 0}, ["build experiment process"])
    completed = sum(1 for step in steps if step.get("completed"))
    rejected = sum(1 for step in steps if _step_human_rejected(step))
    skipped = sum(1 for step in steps if step.get("skipped"))
    needs_review = sum(1 for step in steps if step.get("requires_human_confirmation"))
    resolved = sum(1 for step in steps if step.get("completed") or _step_human_rejected(step) or step.get("skipped"))
    score = resolved / max(len(steps), 1)
    status = "pass" if score >= 0.8 and needs_review == 0 else ("needs_review" if resolved else "fail")
    return _check(
        "step_reasoning",
        "Step recognition correctness",
        status,
        score,
        {
            "steps": len(steps),
            "completed": completed,
            "human_rejected": rejected,
            "skipped": skipped,
            "resolved": resolved,
            "requires_human_confirmation": needs_review,
        },
    )


def _next_step_check(process: Mapping[str, Any]) -> dict[str, Any]:
    current_id = process.get("current_step_id")
    next_id = process.get("next_step_id")
    status = "pass" if current_id or next_id else "needs_review"
    return _check("next_step_reasoning", "Next step reasoning", status, 0.85 if status == "pass" else 0.4, {"current_step_id": current_id, "next_step_id": next_id})


def _completion_check(process: Mapping[str, Any], history_model: Mapping[str, Any]) -> dict[str, Any]:
    steps = _steps(process)
    inferred = sum(1 for step in steps if step.get("inferred") and not _step_human_rejected(step))
    not_observed = sum(1 for step in steps if str(step.get("status") or "") == "not_observed" and not _step_human_rejected(step))
    history_available = bool(history_model.get("event_count") or history_model.get("transition_probabilities"))
    if inferred and history_available:
        status, score = "pass", 0.8
    elif inferred or not_observed:
        status, score = "needs_review", 0.55
    else:
        status, score = "pass", 0.9
    return _check("process_completion", "Process completion trust", status, score, {"inferred_steps": inferred, "not_observed_steps": not_observed, "history_available": history_available}, ["feed history_model into process reasoning"] if (inferred or not_observed) and not history_available else [])


def _step_human_rejected(step: Mapping[str, Any]) -> bool:
    return str(step.get("status") or "") == "human_rejected" or str(step.get("confirmation_status") or "") == "rejected"


def _evidence_chain_check(process: Mapping[str, Any]) -> dict[str, Any]:
    steps = _steps(process)
    with_evidence = [step for step in steps if step.get("evidence_refs")]
    observed = [step for step in steps if step.get("observed")]
    observed_with_evidence = [step for step in observed if step.get("evidence_refs")]
    reverse_index = process.get("evidence_index") if isinstance(process.get("evidence_index"), Mapping) else {}
    ratio = len(with_evidence) / max(len(steps), 1)
    observed_ratio = len(observed_with_evidence) / max(len(observed), 1)
    status = "pass" if ratio >= 0.8 and reverse_index else ("needs_review" if observed else "fail")
    return _check(
        "evidence_chain",
        "Evidence chain completeness",
        status,
        ratio,
        {
            "steps": len(steps),
            "steps_with_evidence": len(with_evidence),
            "observed_steps": len(observed),
            "observed_steps_with_evidence": len(observed_with_evidence),
            "observed_only_score": round(observed_ratio, 4),
            "reverse_index_entries": len(reverse_index),
        },
    )


def _json_artifact_check(metadata: Path, validation_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
    required = [
        "unified_multimodal_timeline.jsonl",
        "video_understanding.jsonl",
        "experiment_context.json",
        "experiment_process.json",
        "human_confirmation_queue.jsonl",
        "material_asset_catalog.jsonl",
    ]
    missing = [name for name in required if not (metadata / name).exists()]
    validation = validation_report if isinstance(validation_report, Mapping) else {}
    error_count = int(validation.get("error_count") or 0)
    valid = validation.get("valid")
    if missing:
        status = "fail"
    elif error_count:
        status = "fail"
    elif valid is False:
        status = "needs_review"
    else:
        status = "pass"
    score = max(0.0, 1.0 - len(missing) / len(required) - min(0.5, error_count * 0.05))
    return _check(
        "json_artifacts",
        "JSON artifact and schema validity",
        status,
        score,
        {
            "required": required,
            "missing": missing,
            "schema_validation_present": bool(validation),
            "schema_valid": validation.get("valid"),
            "schema_error_count": error_count,
            "validated_artifacts": validation.get("artifact_count"),
        },
        ["run validate-artifacts and fix schema errors"] if error_count else [],
    )


def _searchability_check(vector_metadata: list[Mapping[str, Any]], assets: list[Mapping[str, Any]]) -> dict[str, Any]:
    searchable_assets = [item for item in assets if item.get("search_text") or item.get("index_text") or item.get("asset_id")]
    status = "pass" if vector_metadata and searchable_assets else ("needs_review" if vector_metadata or searchable_assets else "fail")
    score = 0.9 if vector_metadata and searchable_assets else (0.5 if vector_metadata or searchable_assets else 0.0)
    return _check("searchability", "Data searchability", status, score, {"vector_metadata": len(vector_metadata), "searchable_assets": len(searchable_assets)})


def _history_reuse_check(history_model: Mapping[str, Any]) -> dict[str, Any]:
    event_count = int(history_model.get("event_count") or 0)
    transition_count = sum(int(value or 0) for value in dict(history_model.get("transition_counts") or {}).values())
    status = "pass" if event_count > 0 and transition_count > 0 else "needs_review"
    return _check("history_reuse", "Historical process reuse", status, 0.8 if status == "pass" else 0.35, {"event_count": event_count, "transition_count": transition_count}, ["build or import history_model.json"] if status != "pass" else [])


def _human_confirmation_check(queue: list[Mapping[str, Any]], process: Mapping[str, Any]) -> dict[str, Any]:
    pending = sum(1 for item in queue if str(item.get("status") or "") == "pending")
    approved = sum(1 for item in queue if str(item.get("status") or "") in {"approved", "auto_confirmed"})
    rejected = sum(1 for item in queue if str(item.get("status") or "") == "rejected")
    step_rows = _steps(process)
    required_steps = [
        step
        for step in step_rows
        if step.get("requires_human_confirmation")
        or str(step.get("confirmation_status") or "") in {"pending", "needs_review", "observed_candidate"}
    ]
    unresolved_steps = [
        step
        for step in required_steps
        if str(step.get("confirmation_status") or "") not in {"approved", "rejected", "auto_confirmed", "completed_confirmed"}
    ]
    status = "pass" if pending == 0 and not unresolved_steps else "needs_review"
    if status == "pass":
        score = 0.9
    elif queue:
        score = 0.55
    else:
        score = 0.35
    return _check(
        "human_confirmation",
        "Human confirmation coverage",
        status,
        score,
        {
            "pending_items": pending,
            "approved_items": approved,
            "rejected_items": rejected,
            "required_steps": len(required_steps),
            "unresolved_step_ids": [str(step.get("step_id") or "") for step in unresolved_steps],
        },
    )


def _model_coverage_check(
    inventory: Mapping[str, Any],
    capability_gap_summary: Mapping[str, Any],
    stage_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    capabilities = inventory.get("capabilities") if isinstance(inventory.get("capabilities"), Mapping) else {}
    raw_unavailable = [name for name, item in capabilities.items() if isinstance(item, Mapping) and not item.get("available")]
    reported_missing = [str(item) for item in capability_gap_summary.get("missing_capabilities") or []]
    unavailable_split = split_scope_values(raw_unavailable, stage_scope)
    unavailable = _ordered_unique([*unavailable_split["in_scope"], *reported_missing])
    out_of_scope = _ordered_unique(
        [
            *unavailable_split["out_of_scope"],
            *[str(item) for item in capability_gap_summary.get("out_of_scope_capabilities") or []],
        ]
    )
    model_count = int(inventory.get("model_count") or 0)
    status = "pass" if model_count and not unavailable else ("needs_review" if model_count else "fail")
    score = 0.9 if status == "pass" else (0.65 if model_count else 0.0)
    if status == "pass" and out_of_scope:
        score = 0.88
    return _check(
        "model_coverage",
        "Model and label coverage",
        status,
        score,
        {
            "model_count": model_count,
            "dataset_count": inventory.get("dataset_count", 0),
            "unavailable_capabilities": unavailable,
            "out_of_scope_unavailable_capabilities": out_of_scope,
            "capability_gap_report_present": bool(capability_gap_summary.get("present")),
            "reported_missing_capabilities": reported_missing,
            "recommended_labels": list(capability_gap_summary.get("recommended_labels") or []),
            "out_of_scope_recommended_labels": list(capability_gap_summary.get("out_of_scope_recommended_labels") or []),
            "stage_scope": _stage_scope_summary(stage_scope),
        },
    )


def _context_fusion_check(context: Mapping[str, Any]) -> dict[str, Any]:
    procedure_count = len(context.get("procedure_candidates") or [])
    material_count = len(context.get("materials") or [])
    parameter_count = len(context.get("parameters") or [])
    status = "pass" if procedure_count and material_count else "needs_review"
    score = 0.85 if status == "pass" else 0.45
    return _check("context_fusion", "Context fusion completeness", status, score, {"procedure_count": procedure_count, "material_count": material_count, "parameter_count": parameter_count})


def _scorecard(checks: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {str(check.get("check_id") or ""): check for check in checks}

    def item(category: str, check_id: str, label: str | None = None) -> dict[str, Any]:
        check = by_id.get(check_id, {})
        return {
            "category": category,
            "check_id": check_id,
            "label": label or str(check.get("name") or check_id),
            "status": check.get("status"),
            "score": check.get("score", 0.0),
            "details": check.get("details", {}),
        }

    return {
        "time_alignment": item("time_alignment", "time_alignment", "Time alignment"),
        "segment_extraction": item("segment_extraction", "segment_completeness", "Key segment extraction"),
        "keyframes": item("keyframes", "keyframe_representativeness", "Keyframe coverage"),
        "action_recognition": item("action_recognition", "action_recognition", "Action recognition"),
        "state_detection": item("state_detection", "state_change_detection", "State detection"),
        "step_reasoning": item("step_reasoning", "step_reasoning", "Step reasoning"),
        "completion": item("completion", "process_completion", "Missing-step completion"),
        "evidence_chain": item("evidence_chain", "evidence_chain", "Evidence chain"),
        "json": item("json", "json_artifacts", "JSON/schema validity"),
        "retrieval": item("retrieval", "searchability", "Retrieval readiness"),
        "history_reuse": item("history_reuse", "history_reuse", "Historical reuse"),
        "human_confirmation": item("human_confirmation", "human_confirmation", "Human confirmation"),
        "model_coverage": item("model_coverage", "model_coverage", "Model/label coverage"),
        "context_fusion": item("context_fusion", "context_fusion", "Context fusion"),
    }


def _diagnostics(checks: list[Mapping[str, Any]], metadata: Path, validation_report: Mapping[str, Any] | None) -> dict[str, Any]:
    failed = [check for check in checks if str(check.get("status") or "") == "fail"]
    needs_review = [check for check in checks if str(check.get("status") or "") == "needs_review"]
    missing_artifacts = []
    for name in (
        "unified_multimodal_timeline.jsonl",
        "key_action_segments.jsonl",
        "micro_segments.jsonl",
        "video_understanding.jsonl",
        "experiment_process.json",
        "vector_metadata.jsonl",
    ):
        if not (metadata / name).exists():
            missing_artifacts.append(str(metadata / name))
    validation = validation_report if isinstance(validation_report, Mapping) else {}
    return {
        "failed_check_ids": [check.get("check_id") for check in failed],
        "needs_review_check_ids": [check.get("check_id") for check in needs_review],
        "missing_artifacts": missing_artifacts,
        "schema_error_count": int(validation.get("error_count") or 0),
        "schema_issue_count": int(validation.get("issue_count") or 0),
        "top_recommendations": _ordered_unique(
            recommendation
            for check in [*failed, *needs_review]
            for recommendation in check.get("recommendations") or []
        )[:20],
    }


def _attach_task_mappings(
    checks: list[dict[str, Any]],
    *,
    session: Path,
    metadata: Path,
    process: Mapping[str, Any],
    confirmation_queue: list[Mapping[str, Any]],
    capability_gap_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    command_context = _command_context(session, metadata, confirmation_queue)
    pending_confirmation_ids = _pending_confirmation_ids(confirmation_queue)
    unobserved_step_ids = _unobserved_step_ids(process)
    unavailable_capabilities = _ordered_unique(
        capability_gap_summary.get("missing_capabilities") or []
    )
    enriched = []
    for check in checks:
        row = dict(check)
        row["details"] = dict(row.get("details") or {})
        check_id = str(row.get("check_id") or "")
        status = str(row.get("status") or "")
        if check_id in {"step_reasoning", "human_confirmation"}:
            row["details"]["pending_confirmation_ids"] = pending_confirmation_ids
        if check_id == "process_completion":
            row["details"]["unobserved_step_ids"] = unobserved_step_ids
        if check_id in {"state_change_detection", "model_coverage"}:
            row["details"]["capability_gap_summary"] = dict(capability_gap_summary)
        if status != "pass":
            tasks, inputs, commands = _mapping_for_check(
                row,
                command_context=command_context,
                pending_confirmation_ids=pending_confirmation_ids,
                unobserved_step_ids=unobserved_step_ids,
                unavailable_capabilities=unavailable_capabilities,
                capability_gap_summary=capability_gap_summary,
            )
            row["blocking_tasks"] = tasks
            row["required_inputs"] = inputs
            row["suggested_commands"] = commands
        else:
            row["blocking_tasks"] = []
            row["required_inputs"] = []
            row["suggested_commands"] = []
        enriched.append(row)
    return enriched


def _mapping_for_check(
    check: Mapping[str, Any],
    *,
    command_context: Mapping[str, str],
    pending_confirmation_ids: list[str],
    unobserved_step_ids: list[str],
    unavailable_capabilities: list[str],
    capability_gap_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    check_id = str(check.get("check_id") or "")
    template = TASK_TEMPLATES.get(check_id) or {
        "task_id": f"P-03/T-QA-{check_id.upper() or 'FOLLOWUP'}",
        "title": f"Resolve QA check {check_id or 'unknown'}",
        "required_inputs": ["metadata/process_quality_report.json"],
        "suggested_commands": ["quality-report"],
    }
    required_inputs = list(template.get("required_inputs") or [])
    suggested_commands = _commands_for_template(template, command_context)
    tasks = [_task_from_template(template, check, required_inputs)]

    if check_id == "state_change_detection" and (
        not capability_gap_summary.get("present") or unavailable_capabilities
    ):
        gap_template = TASK_TEMPLATES["model_coverage"]
        gap_inputs = list(gap_template.get("required_inputs") or [])
        tasks.append(_task_from_template(gap_template, check, gap_inputs))
        required_inputs.extend(gap_inputs)
        suggested_commands.extend(_commands_for_template(gap_template, command_context))
    if check_id in {"step_reasoning", "human_confirmation"}:
        required_inputs.extend(f"confirmation_id:{item}" for item in pending_confirmation_ids)
    if check_id in {"process_completion", "next_step_reasoning"}:
        required_inputs.extend(f"step_id:{item}" for item in unobserved_step_ids)
    if check_id == "model_coverage":
        required_inputs.extend(f"capability:{item}" for item in unavailable_capabilities)
        required_inputs.extend(f"recommended_label:{item}" for item in capability_gap_summary.get("recommended_labels") or [])

    return _dedupe_tasks(tasks), _ordered_unique(required_inputs), _ordered_unique(suggested_commands)


def _task_from_template(template: Mapping[str, Any], check: Mapping[str, Any], required_inputs: list[str]) -> dict[str, Any]:
    return {
        "task_id": str(template.get("task_id") or "P-03/T-QA-FOLLOWUP"),
        "title": str(template.get("title") or "Resolve QA check"),
        "source_check_id": check.get("check_id"),
        "source_status": check.get("status"),
        "reason": _task_reason(check),
        "required_inputs": _ordered_unique(required_inputs),
    }


def _task_reason(check: Mapping[str, Any]) -> str:
    details = check.get("details") if isinstance(check.get("details"), Mapping) else {}
    recommendations = [str(item) for item in check.get("recommendations") or [] if item]
    if recommendations:
        return "; ".join(recommendations)
    if details:
        compact = {key: value for key, value in details.items() if value not in (None, "", [], {})}
        if compact:
            return json.dumps(compact, ensure_ascii=False, sort_keys=True)
    return f"{check.get('name') or check.get('check_id')} is {check.get('status')}"


def _commands_for_template(template: Mapping[str, Any], context: Mapping[str, str]) -> list[str]:
    return [_render_command(name, context) for name in template.get("suggested_commands") or []]


def _render_command(name: Any, context: Mapping[str, str]) -> str:
    command = str(name or "")
    if command == "quality-report":
        return f"python -m key_action_indexer.cli quality-report --session-dir {context['session']}"
    if command == "timeline":
        return f"python -m key_action_indexer.cli timeline --manifest {context['manifest']} --output {context['metadata']} --dry-run"
    if command == "run-dry":
        return f"python -m key_action_indexer.cli run --manifest {context['manifest']} --dry-run"
    if command == "assets":
        return f"python -m key_action_indexer.cli assets --session-dir {context['session']}"
    if command == "advanced-vision":
        return f"python -m key_action_indexer.cli advanced-vision --session-dir {context['session']}"
    if command == "understand-video":
        return f"python -m key_action_indexer.cli understand-video --session-dir {context['session']}"
    if command == "process":
        return f"python -m key_action_indexer.cli process --session-dir {context['session']}"
    if command == "confirmation-queue":
        return f"python -m key_action_indexer.cli confirmation-queue --session-dir {context['session']} --list"
    if command == "model-inventory":
        return f"python -m key_action_indexer.cli model-inventory --project-root . --output {context['model_inventory']}"
    if command == "history-model":
        return f"python -m key_action_indexer.cli history-model --source {context['experiment_process']} --output {context['history_model']}"
    if command == "context":
        return f"python -m key_action_indexer.cli context --session-dir {context['session']}"
    return command.format(**context)


def _command_context(session: Path, metadata: Path, confirmation_queue: list[Mapping[str, Any]]) -> dict[str, str]:
    pending_ids = _pending_confirmation_ids(confirmation_queue)
    return {
        "session": _quote_path(session),
        "metadata": _quote_path(metadata),
        "manifest": _quote_path(session / "manifest.json"),
        "model_inventory": _quote_path(metadata / "model_inventory.json"),
        "history_model": _quote_path(metadata / "history_model.json"),
        "experiment_process": _quote_path(metadata / "experiment_process.json"),
        "first_confirmation_id": _quote_path(pending_ids[0] if pending_ids else "<confirmation_id>"),
    }


def _next_round_scheduler(checks: list[Mapping[str, Any]]) -> dict[str, Any]:
    tasks = []
    for check in checks:
        if str(check.get("status") or "") == "pass":
            continue
        for task in check.get("blocking_tasks") or []:
            if not isinstance(task, Mapping):
                continue
            tasks.append(
                {
                    "task_id": task.get("task_id"),
                    "title": task.get("title"),
                    "source_check_id": check.get("check_id"),
                    "source_status": check.get("status"),
                    "required_inputs": _ordered_unique(
                        list(task.get("required_inputs") or []) + list(check.get("required_inputs") or [])
                    ),
                    "suggested_commands": list(check.get("suggested_commands") or []),
                    "reason": task.get("reason"),
                }
            )
    return {
        "metadata_version": "key_action_qa_scheduler_input.v1",
        "blocking_check_count": sum(1 for check in checks if str(check.get("status") or "") != "pass"),
        "needs_review_check_ids": [check.get("check_id") for check in checks if str(check.get("status") or "") == "needs_review"],
        "task_count": len(tasks),
        "tasks": tasks,
    }


def _capability_gap_summary(
    report: Mapping[str, Any],
    *,
    present: bool,
    stage_scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_missing_capabilities = _collect_missing_capabilities(report)
    raw_recommended_labels = _collect_recommended_labels(report)
    split_missing = split_scope_values(raw_missing_capabilities, stage_scope)
    split_labels = split_scope_values(raw_recommended_labels, stage_scope)
    missing_capabilities = split_missing["in_scope"]
    recommended_labels = split_labels["in_scope"]
    explicit_gap_count = _explicit_gap_count(report)
    return {
        "present": bool(present),
        "gap_count": max(len(missing_capabilities), explicit_gap_count),
        "blocking_gap_count": len(missing_capabilities),
        "raw_gap_count": max(len(raw_missing_capabilities), explicit_gap_count),
        "missing_capabilities": missing_capabilities,
        "out_of_scope_capabilities": split_missing["out_of_scope"],
        "raw_missing_capabilities": raw_missing_capabilities,
        "recommended_labels": recommended_labels,
        "out_of_scope_recommended_labels": split_labels["out_of_scope"],
        "raw_recommended_labels": raw_recommended_labels,
        "stage_scope": _stage_scope_summary(stage_scope),
    }


def _stage_scope_summary(stage_scope: Mapping[str, Any] | None) -> dict[str, Any]:
    if not stage_scope:
        return {}
    return {
        "scope_name": stage_scope.get("scope_name"),
        "stage": stage_scope.get("stage"),
        "status": stage_scope.get("status"),
        "qa_policy": dict(stage_scope.get("qa_policy") or {}) if isinstance(stage_scope.get("qa_policy"), Mapping) else {},
    }


def _collect_missing_capabilities(report: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    for key in ("missing_capabilities", "unavailable_capabilities", "blocking_capabilities"):
        missing.extend(_strings_from_value(report.get(key)))
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        for key in (
            "capabilities_missing_label_foundation",
            "unavailable_inventory_capabilities",
            "missing_capabilities",
            "unavailable_capabilities",
        ):
            missing.extend(_strings_from_value(summary.get(key)))
    capabilities = report.get("capabilities")
    if isinstance(capabilities, Mapping):
        for name, item in capabilities.items():
            if isinstance(item, Mapping) and item.get("available") is False:
                missing.append(str(name))
    for key in ("gaps", "capability_gaps", "items", "checks"):
        rows = report.get(key)
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            status = str(item.get("status") or item.get("state") or "").strip().lower()
            is_gap = item.get("available") is False or item.get("present") is False or status in {
                "missing",
                "unavailable",
                "gap",
                "needs_review",
                "fail",
                "failed",
            }
            if is_gap:
                missing.extend(
                    _strings_from_value(
                        item.get("capability")
                        or item.get("capability_id")
                        or item.get("name")
                        or item.get("id")
                    )
                )
    return _ordered_unique(missing)


def _collect_recommended_labels(report: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    label_keys = {
        "recommended_labels",
        "suggested_labels",
        "recommended_new_labels",
        "missing_labels",
        "missing_classes",
        "suggested_classes",
        "recommended_classes",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key) in label_keys:
                    labels.extend(_strings_from_value(child))
                elif isinstance(child, (Mapping, list)):
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return _ordered_unique(labels)


def _explicit_gap_count(report: Mapping[str, Any]) -> int:
    counts = []
    for key in ("gap_count", "missing_count", "blocking_gap_count", "required_capability_count"):
        try:
            counts.append(int(report.get(key) or 0))
        except (TypeError, ValueError):
            pass
    summary = report.get("summary")
    if isinstance(summary, Mapping):
        for key in ("required_capability_count", "capabilities_missing_label_foundation"):
            value = summary.get(key)
            try:
                counts.append(len(value) if isinstance(value, list) else int(value or 0))
            except (TypeError, ValueError):
                pass
    for key in ("gaps", "capability_gaps"):
        value = report.get(key)
        if isinstance(value, list):
            counts.append(len(value))
    return max(counts, default=0)


def _strings_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Mapping):
        output = []
        for key in ("capability", "capability_id", "name", "label", "class_name", "id"):
            if value.get(key):
                output.append(str(value[key]))
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(_strings_from_value(item))
        return output
    return [str(value)]


def _pending_confirmation_ids(queue: list[Mapping[str, Any]]) -> list[str]:
    return _ordered_unique(
        str(item.get("confirmation_id") or "")
        for item in queue
        if str(item.get("status") or "") == "pending" and item.get("confirmation_id")
    )


def _unobserved_step_ids(process: Mapping[str, Any]) -> list[str]:
    return _ordered_unique(
        str(step.get("step_id") or "")
        for step in _steps(process)
        if str(step.get("status") or "") in {"not_observed", "inferred_missing", "skipped_or_unobserved"} or step.get("inferred")
    )


def _dedupe_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        output.append(task)
    return output


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _quote_path(value: Any) -> str:
    text = str(value)
    return '"' + text.replace('"', '\\"') + '"'


def _overall_status(status_counts: Counter[str]) -> str:
    if status_counts.get("fail"):
        return "fail"
    if status_counts.get("needs_review"):
        return "needs_review"
    return "pass"


def _steps(process: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = process.get("steps")
    return [step for step in steps if isinstance(step, Mapping)] if isinstance(steps, list) else []


def _session_id(session: Path, process: Mapping[str, Any], context: Mapping[str, Any], timeline: list[Mapping[str, Any]]) -> str:
    for source in (process, context):
        if source.get("session_id"):
            return str(source["session_id"])
    for row in timeline:
        if row.get("session_id"):
            return str(row["session_id"])
    return session.name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except Exception:
        return []
