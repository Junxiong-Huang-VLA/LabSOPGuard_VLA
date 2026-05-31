from __future__ import annotations

import json
import hashlib
import os
import shutil
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field as dataclass_field, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

from .action_detector import detect_key_action_segments
from .advanced_vision_evidence import build_advanced_vision_evidence
from .analysis_proxy import analysis_proxy_cache_payload, analysis_proxy_enabled, build_analysis_proxies
from .asset_library import build_material_asset_catalog
from .capability_gap_report import build_capability_gap_report
from .chinese_index import refresh_micro_row_chinese_index, refresh_segment_chinese_index
from .confirmation_loop import build_confirmation_queue
from .clip_extractor import extract_multiview_clips
from .config import DetectorConfig
from .context_fusion import build_experiment_context
from .debug_viz import save_frame_score_plot, save_roi_preview, save_segment_contact_sheet
from .data_governance import build_data_governance_report
from .description_builder import build_segment_description
from .dual_view_action_alignment import build_dual_view_action_alignment
from .dual_view_frame_alignment import analyze_dual_view_frame_alignment, run_dual_view_alignment_pipeline
from .evidence import apply_segment_evidence
from .event_candidate_trace import build_event_candidate_trace
from .episode_segmenter import (
    _lifecycle_window_for_action,
    _lifecycle_windows_from_rows,
    rebuild_episode_segments_from_micro_evidence,
)
from .history_learning import build_history_model
from .input_ingestion import ingest_manifest_inputs, write_video_source_metadata
from .evaluation import build_pipeline_evaluation_report, compute_micro_quality_stats
from .experiment_focus import extract_experiment_focus_clips
from .experiment_window_state import (
    build_experiment_state_artifacts,
    write_chunk_manifest,
    write_speed_and_quality_reports,
)
from .frame_time_map import capture_sec_to_video_sec, frame_time_map_summary, video_sec_to_capture_sec
from .artifact_schema import validate_session_artifacts
from .micro_postprocess import merge_same_object_adjacent_micro_segments
from .micro_segmenter import generate_micro_segments, micro_row_to_vector_metadata
from .material_references import (
    approve_material_candidates,
    apply_formal_dual_view_material_publish_gate,
    build_yolo_material_candidates,
    build_yolo_material_references,
    filter_aligned_dual_view_material_rows,
    filter_complete_dual_view_material_rows,
    paired_view_context_scene_gate_passed,
)
from .model_inventory import discover_lab_assets
from .model_observations import build_model_observation_events
from .lab_model_signal_inputs import build_lab_model_signal_inputs
from .performance import build_long_video_processing_plan
from .process_record import build_process_record
from .process_reasoner import build_experiment_process
from .quality_assurance import build_quality_assurance_report
from .record_ingestion import ingest_sop_and_database_records
from .report import DEFAULT_QUERY, generate_formal_validation_report, generate_report
from .session_layout import initialize_session_dir
from .session_context_seed import seed_session_context
from .schemas import (
    DetectedSegment,
    KeyActionSegment,
    SessionManifest,
    TranscriptUtterance,
    VideoSource,
    VectorMetadata,
    read_jsonl,
    to_json_dict,
    write_jsonl,
)
from .state_index import build_state_change_index
from .time_alignment import apply_alignment_correction, estimate_sliding_window_drift, evaluate_time_alignment, find_dialogue_for_segment, generate_multimodal_alignment, global_time_to_local_sec, local_sec_to_global_time, parse_time, strict_common_overlap_from_view_intervals
from .time_axis_health import STATUS_UNRELIABLE, analyze_dual_view_time_axis
from .transcript import load_aligned_transcript
from .unified_timeline import generate_unified_timeline
from .validation import validate_manifest
from .vector_index import VectorIndex
from .video_utils import get_video_duration_sec
from .video_understanding import build_video_understanding
from .yolo_observation_inputs import build_yolo_observation_inputs

DETECTION_DECISION_MOTION_BASELINE = "motion_baseline"
DETECTION_DECISION_YOLO_INTERACTION = "yolo.interaction"
DETECTION_DECISION_YOLO_FALLBACK_MOTION = "yolo_fallback.motion"
DECISION_REASON_YOLO_INTERACTION_DETECTED = "yolo_interaction_detected"
DECISION_REASON_MOTION_FALLBACK_AFTER_YOLO_FAILURE = "motion_fallback_after_yolo_failure"
DECISION_REASON_MOTION_BASELINE = "motion_baseline"

_YOLO_PARENT_ACTIVITY_VERSION = "hand_object_copresence_v2"
_YOLO_HAND_LABELS = {"gloved_hand", "hand"}
_YOLO_OPERABLE_OBJECT_LABELS = {
    "balance",
    "beaker",
    "container",
    "paper",
    "pipette",
    "pipette_tip",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
    "tube",
}

PIPELINE_STAGES = ("YOLO_PRIMARY", "RULE_FILTER", "BACKUP_BASELINE", "MICRO_SEGMENT")
ARTIFACT_VALIDATION_TYPES = [
    "model_observation_events",
    "video_understanding",
    "experiment_context",
    "sop_state_machine",
    "experiment_process",
    "process_record",
    "asset_catalog",
    "confirmation_queue",
    "process_quality_report",
]

import logging as _logging

_logger = _logging.getLogger("key_action_indexer")


def _emit_pipeline_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    stage: str,
    progress: float,
    message: str,
    **extra: Any,
) -> None:
    if callback is None:
        return
    payload = {
        "stage": stage,
        "progress": max(0.0, min(1.0, float(progress))),
        "message": message,
        **extra,
    }
    try:
        callback(payload)
    except Exception:
        _logger.debug("Pipeline progress callback failed", exc_info=True)


@dataclass
class RunContext:
    run_id: str = dataclass_field(default_factory=lambda: str(uuid.uuid4()))
    stages: list[dict[str, Any]] = dataclass_field(default_factory=list)
    _current_stage: str | None = dataclass_field(default=None, repr=False)
    _stage_start: float = dataclass_field(default=0.0, repr=False)

    def begin_stage(self, name: str) -> None:
        self._current_stage = name
        self._stage_start = time.time()

    def end_stage(self, *, inputs: int = 0, outputs: int = 0, errors: int = 0) -> None:
        if self._current_stage is None:
            return
        self.stages.append({
            "stage": self._current_stage,
            "duration_sec": round(time.time() - self._stage_start, 3),
            "inputs": inputs,
            "outputs": outputs,
            "errors": errors,
        })
        self._current_stage = None

    def stage_stats(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage_count": len(self.stages),
            "total_duration_sec": round(sum(s["duration_sec"] for s in self.stages), 3),
            "stages": self.stages,
        }


@dataclass
class PipelineBootstrap:
    manifest: SessionManifest
    output_dir: Path
    paths: dict[str, Path]
    run_ctx: RunContext
    file_handler: _logging.Handler
    active_detector_config: DetectorConfig
    video_source_rows: list[dict[str, Any]]
    long_video_plan: Mapping[str, Any]
    model_inventory: Mapping[str, Any]
    capability_gap_report: dict[str, Any]


@dataclass
class PipelineEvidenceOutputs:
    yolo_rows_path: str | None
    micro_source_path: str | None
    key_segment_rows_for_index: list[dict[str, Any]]
    episode_rebuild_summary: dict[str, Any]
    micro_refine_summary: dict[str, Any]
    raw_micro_rows: list[dict[str, Any]]
    micro_rows: list[dict[str, Any]]
    micro_dedup_log: list[dict[str, Any]]
    micro_merge_stats: dict[str, Any]
    micro_quality_stats: dict[str, Any]
    dual_view_action_summary: dict[str, Any]
    experiment_focus_summary: dict[str, Any]
    index: VectorIndex


@dataclass
class PipelineContextInputs:
    utterances: list[TranscriptUtterance]
    input_ingestion_summary: dict[str, Any]
    session_context_summary: dict[str, Any]
    record_ingestion_summary: dict[str, Any]
    history_model: Mapping[str, Any]


@dataclass
class PipelineDetectionOutputs:
    detected_segments: list[Any]
    generated_yolo_rows: list[dict[str, Any]]
    detector_summary: dict[str, Any]
    experiment_episode_rows: list[dict[str, Any]]
    alignment_health: dict[str, Any]
    drift_result: dict[str, Any]
    alignment_degradation: float


@dataclass
class PipelineTimelineOutputs:
    unified_timeline_summary: dict[str, Any]
    material_library_summary: dict[str, Any]
    state_change_summary: dict[str, Any]
    yolo_observation_input_summary: dict[str, Any]
    lab_model_signal_input_summary: dict[str, Any]
    advanced_vision_summary: dict[str, Any]
    model_observation_summary: dict[str, Any]


@dataclass
class PipelineProcessOutputs:
    video_understanding_summary: dict[str, Any]
    experiment_context_summary: dict[str, Any]
    experiment_process_summary: dict[str, Any]
    confirmation_queue_summary: dict[str, Any]
    quality_assurance_summary: dict[str, Any]
    process_record_summary: dict[str, Any]


@dataclass
class PipelineValidationOutputs:
    artifact_validation_summary: dict[str, Any]
    quality_assurance_summary: dict[str, Any]
    pipeline_evaluation_summary: dict[str, Any]
    data_governance_summary: dict[str, Any]


def _mkdirs(output_dir: Path) -> dict[str, Path]:
    return initialize_session_dir(output_dir)


def _copy_manifest(manifest_path: str | Path, manifest: SessionManifest, output_dir: Path) -> None:
    source = Path(manifest_path)
    target = output_dir / "manifest.json"
    if source.exists():
        try:
            if source.resolve() == target.resolve():
                return
        except Exception:
            pass
        shutil.copy2(source, target)
    else:
        target.write_text(json.dumps(manifest.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _vector_metadata_from_segment(segment: KeyActionSegment) -> VectorMetadata:
    def interaction_text(event: Any) -> str:
        if isinstance(event, dict):
            return str(event.get("interaction") or "")
        return str(getattr(event, "interaction", "") or "")

    visual_keywords = {
        interaction_text(event)
        for event in segment.interaction_events
        if interaction_text(event)
    }
    for event in segment.interaction_events:
        for key in ("hand_label", "object_label", "object_name", "interaction"):
            value = event.get(key) if isinstance(event, dict) else getattr(event, key, None)
            if value:
                visual_keywords.add(str(value))
    for item in segment.yolo_interactions[:20]:
        for value in (item.hand_label, item.object_label, item.object_name, item.interaction):
            if value:
                visual_keywords.add(str(value))
        for detection in item.detections[:8]:
            label = str(detection.get("label") or "")
            if label:
                visual_keywords.add(label)
    compact_yolo_interactions = [
        {
            "view": item.view,
            "local_time_sec": item.local_time_sec,
            "global_time": item.global_time,
            "interaction": item.interaction,
            "hand_label": item.hand_label,
            "object_label": item.object_label,
            "object_name": item.object_name,
            "confidence": item.confidence,
            "source": item.source,
            "detections": item.detections[:4],
        }
        for item in segment.yolo_interactions[:10]
    ]
    yolo_evidence = []
    for item in segment.yolo_interactions[:12]:
        labels = []
        for detection in item.detections[:6]:
            label = str(detection.get("label") or "")
            if label:
                labels.append(label)
                visual_keywords.add(label)
        visual_keywords.add(item.object_label)
        visual_keywords.add(item.object_name)
        yolo_evidence.append(
            {
                "view": item.view,
                "local_time_sec": item.local_time_sec,
                "global_time": item.global_time,
                "object_label": item.object_label,
                "confidence": item.confidence,
                "source": item.source,
                "labels": labels,
                "detections": item.detections[:6],
            }
        )
    detected_objects = sorted(
        {
            *[str(item) for item in segment.text_description.objects if str(item)],
            *[str(item) for item in segment.text_description.tools if str(item)],
            *[str(key) for key in segment.yolo_label_counts if str(key)],
            *[
                str(item.get("primary_object"))
                for item in segment.micro_segments
                if isinstance(item, dict) and item.get("primary_object")
            ],
        }
    )
    primary_object = next(
        (
            str(item.get("primary_object"))
            for item in segment.micro_segments
            if isinstance(item, dict) and item.get("primary_object")
        ),
        detected_objects[0] if detected_objects else None,
    )
    interaction_type = next(
        (
            str(item.get("interaction_type"))
            for item in segment.micro_segments
            if isinstance(item, dict) and item.get("interaction_type")
        ),
        None,
    )
    return VectorMetadata(
        embedding_id=segment.index.embedding_id,
        segment_id=segment.segment_id,
        session_id=segment.session_id,
        index_text=segment.index.index_text,
        global_start_time=segment.global_start_time,
        global_end_time=segment.global_end_time,
        third_person_clip=segment.third_person.clip_path,
        first_person_clip=segment.first_person.clip_path if segment.first_person else None,
        related_dialogue=segment.dialogue_context,
        action_type=segment.text_description.action_type,
        detector_backend=str(getattr(segment, "detector_backend", "motion")),
        detector_source_view=str(getattr(segment, "detector_source_view", "third_person")),
        decision_path=str(getattr(segment, "decision_path", "")),
        decision_trace=list(getattr(segment, "decision_trace", [])),
        fallback_used=bool(getattr(segment, "fallback_used", False)),
        fallback_reason=str(getattr(segment, "fallback_reason", "")),
        reason_code=str(getattr(segment, "reason_code", "")),
        raw_score=float(getattr(segment, "raw_score", 0.0) or 0.0),
        final_score=float(getattr(segment, "final_score", 0.0) or 0.0),
        retrieval_boost_factors=dict(getattr(segment, "retrieval_boost_factors", {}) or {}),
        interaction_keyframes=to_json_dict(segment.interaction_keyframes),
        interaction_events=to_json_dict(segment.interaction_events),
        yolo_interactions=compact_yolo_interactions,
        yolo_evidence=yolo_evidence,
        asset_bindings=segment.asset_bindings,
        visual_keywords=sorted(label for label in visual_keywords if label),
        index_level="segment",
        primary_object=primary_object,
        interaction_type=interaction_type,
        detected_objects=detected_objects,
        evidence=segment.evidence,
        evidence_level=segment.evidence.get("evidence_level"),
        evidence_reasons=segment.evidence.get("evidence_reasons", []),
        limitations=segment.evidence.get("limitations", []),
        dialogue_context_available=bool(segment.dialogue_context),
        dialogue_match_window_sec=segment.dialogue_match_window_sec,
        dialogue_keywords=segment.dialogue_keywords,
    )


def _write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _optional_int_value(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _formal_action_alignment_gate_from_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = summary if isinstance(summary, Mapping) else {}
    diagnostics = payload.get("view_alignment_diagnostics") if isinstance(payload.get("view_alignment_diagnostics"), Mapping) else {}
    formal_event_count = None
    formal_event_count_source = ""
    for source, value in (
        ("summary.formal_event_count", payload.get("formal_event_count")),
        ("summary.view_alignment_diagnostics.formal_event_count", diagnostics.get("formal_event_count")),
        ("summary.dual_view_action_event_count", payload.get("dual_view_action_event_count")),
    ):
        parsed = _optional_int_value(value)
        if parsed is not None:
            formal_event_count = parsed
            formal_event_count_source = source
            break
    if formal_event_count is None:
        formal_event_count = 0
        formal_event_count_source = "missing"
    explicit_allowed = payload.get("formal_results_allowed") if "formal_results_allowed" in payload else None
    formal_results_allowed = bool(explicit_allowed) if explicit_allowed is not None else formal_event_count > 0
    blocked_reason = ""
    if explicit_allowed is False:
        blocked_reason = "formal_results_not_allowed"
    elif formal_event_count <= 0:
        blocked_reason = "no_confirmed_dual_view_action_events"
    elif not formal_results_allowed:
        blocked_reason = "formal_results_not_allowed"
    return {
        "schema_version": "formal_action_alignment_gate.v1",
        "status": "blocked" if blocked_reason else "passed",
        "allowed": not bool(blocked_reason),
        "blocked_reason": blocked_reason or None,
        "formal_results_allowed": bool(formal_results_allowed),
        "formal_results_allowed_explicit": explicit_allowed if explicit_allowed is not None else None,
        "formal_event_count": int(formal_event_count),
        "formal_event_count_source": formal_event_count_source,
        "dual_view_action_event_count": _optional_int_value(payload.get("dual_view_action_event_count")),
        "decision": payload.get("decision"),
    }


def _formal_window_visual_review_gate_status(paths: dict[str, Path]) -> dict[str, Any]:
    manifest_path = paths["metadata"] / "formal_window_human_review_manifest.json"
    formal_windows_path = paths["metadata"] / "formal_experiment_windows.json"
    formal_windows = _read_json_if_exists(formal_windows_path)
    window_count = 0
    if isinstance(formal_windows.get("windows"), list):
        window_count = len(formal_windows["windows"])
    elif formal_windows.get("window_count") is not None:
        parsed = _optional_int_value(formal_windows.get("window_count"))
        window_count = int(parsed or 0)

    payload = _read_json_if_exists(manifest_path)
    if not payload:
        if window_count <= 0:
            return {
                "schema_version": "formal_window_visual_review_gate.v1",
                "status": "not_applicable",
                "allowed": True,
                "blocked_reason": None,
                "formal_window_count": 0,
            }
        return {
            "schema_version": "formal_window_visual_review_gate.v1",
            "status": "blocked",
            "allowed": False,
            "blocked_reason": "formal_window_visual_review_missing",
            "formal_window_count": window_count,
            "source_path": str(manifest_path),
            "policy": "formal windows must be validated by per-window side-by-side visual review before formal publication",
        }

    total = _optional_int_value(payload.get("total_formal_windows"))
    total = int(total if total is not None else window_count)
    passed = int(_optional_int_value(payload.get("passed_visual_review_count")) or 0)
    rejected = [
        str(item)
        for item in (payload.get("recommended_reject_window_ids") or [])
        if str(item).strip()
    ] if isinstance(payload.get("recommended_reject_window_ids"), list) else []
    suspicious = [
        str(item)
        for item in (payload.get("suspicious_window_ids") or [])
        if str(item).strip()
    ] if isinstance(payload.get("suspicious_window_ids"), list) else []
    pending = [
        str(item)
        for item in (payload.get("pending_visual_review_window_ids") or [])
        if str(item).strip()
    ] if isinstance(payload.get("pending_visual_review_window_ids"), list) else []

    blocked_reason = None
    if rejected:
        blocked_reason = "formal_window_visual_review_failed"
    elif suspicious or pending or passed < total:
        blocked_reason = "formal_window_needs_human_review"

    return {
        "schema_version": "formal_window_visual_review_gate.v1",
        "status": "blocked" if blocked_reason else "passed",
        "allowed": not bool(blocked_reason),
        "blocked_reason": blocked_reason,
        "formal_window_count": total,
        "passed_visual_review_count": passed,
        "recommended_reject_window_ids": rejected,
        "suspicious_window_ids": suspicious,
        "pending_visual_review_window_ids": pending,
        "source_path": str(manifest_path),
        "policy": payload.get("policy"),
    }


def _formal_time_axis_gate_status(paths: dict[str, Path]) -> dict[str, Any]:
    for path in (
        paths["metadata"] / "time_axis_health.json",
        paths["metadata"] / "pre_coarse_timeline_alignment.json",
        paths["metadata"] / "view_alignment_from_yolo.json",
    ):
        payload = _read_json_if_exists(path)
        if not payload:
            continue
        status = str(payload.get("status") or payload.get("alignment_status") or "").strip().lower()
        blocked_reason = ""
        if bool(payload.get("time_axis_unreliable")) or status == STATUS_UNRELIABLE:
            blocked_reason = "time_axis_unreliable"
        elif payload.get("formal_results_allowed") is False or payload.get("can_publish_formal_materials") is False:
            blocked_reason = "time_axis_not_formal_publishable"
        if blocked_reason:
            return {
                "schema_version": "formal_time_axis_gate.v1",
                "status": "blocked",
                "allowed": False,
                "blocked_reason": blocked_reason,
                "source_path": str(path),
                "source_status": status,
                "formal_results_allowed": bool(payload.get("formal_results_allowed", False)),
                "video_memory_allowed": bool(payload.get("video_memory_allowed", payload.get("can_write_video_memory", False))),
            }
    return {
        "schema_version": "formal_time_axis_gate.v1",
        "status": "passed",
        "allowed": True,
        "blocked_reason": None,
        "formal_results_allowed": True,
        "video_memory_allowed": True,
    }


def _formal_output_gate_status(
    paths: dict[str, Path],
    *,
    action_summary: Mapping[str, Any] | None = None,
    require_action_alignment: bool,
) -> dict[str, Any]:
    time_gate = _formal_time_axis_gate_status(paths)
    window_gate = (
        _formal_window_visual_review_gate_status(paths)
        if require_action_alignment
        else {
            "schema_version": "formal_window_visual_review_gate.v1",
            "status": "not_required",
            "allowed": True,
            "blocked_reason": None,
            "formal_window_count": None,
        }
    )
    action_gate = (
        _formal_action_alignment_gate_from_summary(action_summary)
        if require_action_alignment
        else {
            "schema_version": "formal_action_alignment_gate.v1",
            "status": "not_required",
            "allowed": True,
            "blocked_reason": None,
            "formal_results_allowed": True,
            "formal_event_count": None,
        }
    )
    blocked_reason = None
    if not time_gate.get("allowed"):
        blocked_reason = str(time_gate.get("blocked_reason") or "time_axis_not_formal_publishable")
    elif not window_gate.get("allowed"):
        blocked_reason = str(window_gate.get("blocked_reason") or "formal_window_visual_review_failed")
    elif not action_gate.get("allowed"):
        blocked_reason = str(action_gate.get("blocked_reason") or "formal_results_not_allowed")
    status = "blocked" if blocked_reason else "passed"
    return {
        "schema_version": "formal_output_gate.v1",
        "status": status,
        "formal_results_allowed": not bool(blocked_reason),
        "video_memory_allowed": bool(not blocked_reason and time_gate.get("video_memory_allowed", True)),
        "blocked_reason": blocked_reason,
        "time_axis_gate": time_gate,
        "formal_window_visual_review_gate": window_gate,
        "dual_view_action_gate": action_gate,
    }


def _write_formal_output_gate(paths: dict[str, Path], gate: Mapping[str, Any]) -> None:
    _write_json(paths["metadata"] / "formal_output_gate.json", dict(gate))


def _write_phase_consistency_from_formal_gate(paths: dict[str, Path], gate: Mapping[str, Any]) -> None:
    dual_gate = gate.get("dual_view_action_gate") if isinstance(gate.get("dual_view_action_gate"), Mapping) else {}
    time_gate = gate.get("time_axis_gate") if isinstance(gate.get("time_axis_gate"), Mapping) else {}
    window_gate = gate.get("formal_window_visual_review_gate") if isinstance(gate.get("formal_window_visual_review_gate"), Mapping) else {}
    formal_event_count = dual_gate.get("formal_event_count")
    allowed = bool(gate.get("formal_results_allowed"))
    if allowed:
        status = "action_phase_verified"
        issue_type = None
        warnings: list[str] = []
        score = 1.0
    elif not bool(time_gate.get("allowed", True)):
        status = "timestamp_alignment_rejected"
        issue_type = str(gate.get("blocked_reason") or time_gate.get("blocked_reason") or "time_axis_not_formal_publishable")
        warnings = ["time-axis reliability is required before visual/action phase verification"]
        score = 0.0
    elif not bool(window_gate.get("allowed", True)):
        blocked = str(gate.get("blocked_reason") or window_gate.get("blocked_reason") or "formal_window_needs_human_review")
        status = "action_phase_rejected" if blocked == "formal_window_visual_review_failed" else "needs_human_review"
        issue_type = blocked
        warnings = ["timestamp delta is not sufficient; every formal window needs side-by-side visual phase review"]
        score = 0.0
    else:
        status = "action_phase_rejected"
        issue_type = str(gate.get("blocked_reason") or dual_gate.get("blocked_reason") or "no_confirmed_dual_view_action_events")
        warnings = ["single-view or phase-mismatched evidence is not allowed to enter formal analysis"]
        score = 0.0
    report = {
        "schema_version": "phase_consistency_report.v1",
        "status": status,
        "consistency_score": score,
        "timestamp_alignment_ready": bool(time_gate.get("allowed", True)),
        "visual_alignment_verified": bool(window_gate.get("allowed", False)),
        "action_phase_verified": bool(allowed),
        "formal_event_count": formal_event_count,
        "issue_type": issue_type,
        "third_phase": "derived_from_dual_view_action_gate",
        "first_phase": "derived_from_dual_view_action_gate",
        "inconsistent_ranges": [],
        "warnings": warnings,
        "formal_output_gate": dict(gate),
    }
    _write_json(paths["metadata"] / "phase_consistency_report.json", report)
    alignment_dir = paths["metadata"] / "dual_view_alignment"
    if alignment_dir.exists():
        _write_json(alignment_dir / "phase_consistency_report.json", report)


def _coarse_yolo_sample_fps(config: DetectorConfig) -> float:
    if _fast_locate_runtime_enabled():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_STAGE1_SAMPLE_FPS") or os.environ.get("KEY_ACTION_FAST_LOCATE_COARSE_SAMPLE_FPS")
        if raw is not None:
            try:
                return max(0.001, float(raw))
            except (TypeError, ValueError):
                pass
    if bool(getattr(config, "long_video_two_stage_sampling", True)) and getattr(config, "long_video_stage1_sample_fps", None):
        return max(0.001, float(config.long_video_stage1_sample_fps or 0.0))
    return max(0.001, float(config.parent_sample_fps or config.sample_fps))


def _refined_yolo_sample_fps(config: DetectorConfig) -> float:
    if _fast_locate_runtime_enabled():
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_STAGE2_SAMPLE_FPS") or os.environ.get("KEY_ACTION_FAST_LOCATE_FINE_SAMPLE_FPS")
        if raw is not None:
            try:
                return max(0.001, float(raw))
            except (TypeError, ValueError):
                pass
    if bool(getattr(config, "long_video_two_stage_sampling", True)) and getattr(config, "long_video_stage2_sample_fps", None):
        return max(0.001, float(config.long_video_stage2_sample_fps or 0.0))
    return max(0.001, float(config.micro_refine_sample_fps))


def _timing_device_is_gpu(value: Any) -> bool:
    device = str(value or "").strip().lower()
    if not device:
        return False
    if device == "mps" or device.startswith(("cuda", "gpu")):
        return True
    parts = [part.strip() for part in device.split(",") if part.strip()]
    return bool(parts) and all(part.isdigit() for part in parts)


def _append_yolo_timing_rows(paths: dict[str, Path], rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _read_json_if_exists(paths["metadata"] / "yolo_timing_summary.json")
    timing_path = paths["metadata"] / "yolo_timing_rows.jsonl"
    existing = read_jsonl(timing_path) if timing_path.exists() else []
    normalized_rows = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.setdefault("timing_row_id", f"timing_{int(time.time() * 1000)}_{index}")
        item.setdefault("recorded_at", time.time())
        normalized_rows.append(item)
    all_rows = [*existing, *normalized_rows]
    write_jsonl(timing_path, all_rows)

    by_stage: dict[str, dict[str, Any]] = {}
    worker_total_wall_sec = 0.0
    total_sampled_frames = 0
    for row in all_rows:
        stage = str(row.get("pipeline_stage") or row.get("stage") or "unknown")
        bucket = by_stage.setdefault(
            stage,
            {
                "rows": 0,
                "wall_sec": 0.0,
                "decode_sec": 0.0,
                "inference_sec": 0.0,
                "postprocess_sec": 0.0,
                "sampled_frames": 0,
                "read_frames": 0,
                "grab_frames": 0,
                "scan_duration_sec": 0.0,
                "views": [],
                "scan_backends": [],
                "planned_scan_backends": [],
                "requested_devices": [],
                "actual_devices": [],
                "batch_sizes": [],
                "batch_size_sources": [],
                "gpu_batch_default_applied": False,
                "device_fallbacks": [],
                "streaming_fine_scan": False,
                "batch_count": 0,
                "yolo_predict_call_count": 0,
                "yolo_batch_predict_attempts": 0,
                "yolo_batch_predict_calls": 0,
                "yolo_frame_predict_calls": 0,
                "yolo_custom_detector_calls": 0,
                "yolo_batch_fallback_count": 0,
                "yolo_batch_fallback_errors": [],
                "_actual_batch_size_values": [],
                "_parallel_elapsed_sec": None,
                "_parallel_workers": None,
                "_parallel_scan_task_count": None,
            },
        )
        bucket["rows"] += 1
        for key in ("wall_sec", "decode_sec", "inference_sec", "postprocess_sec", "scan_duration_sec"):
            try:
                bucket[key] += float(row.get(key) or 0.0)
            except (TypeError, ValueError):
                pass
        for key in ("sampled_frames", "read_frames", "grab_frames"):
            try:
                bucket[key] += int(row.get(key) or 0)
            except (TypeError, ValueError):
                pass
        view = str(row.get("source_view") or row.get("view") or "")
        if view and view not in bucket["views"]:
            bucket["views"].append(view)
        for source_key, bucket_key in (
            ("scan_backend", "scan_backends"),
            ("planned_scan_backend", "planned_scan_backends"),
            ("requested_device", "requested_devices"),
            ("actual_device", "actual_devices"),
            ("device_fallback", "device_fallbacks"),
        ):
            value = str(row.get(source_key) or "").strip()
            if value and value not in bucket[bucket_key]:
                bucket[bucket_key].append(value)
        try:
            batch_size = int(float(row.get("batch_size") or 0))
        except (TypeError, ValueError):
            batch_size = 0
        if batch_size > 0 and batch_size not in bucket["batch_sizes"]:
            bucket["batch_sizes"].append(batch_size)
        batch_source = str(row.get("batch_size_source") or "").strip()
        if batch_source and batch_source not in bucket["batch_size_sources"]:
            bucket["batch_size_sources"].append(batch_source)
        bucket["gpu_batch_default_applied"] = bool(bucket.get("gpu_batch_default_applied")) or bool(
            row.get("gpu_batch_default_applied")
        )
        bucket["streaming_fine_scan"] = bool(bucket.get("streaming_fine_scan")) or bool(row.get("streaming_fine_scan"))
        for key in (
            "batch_count",
            "yolo_predict_call_count",
            "yolo_batch_predict_attempts",
            "yolo_batch_predict_calls",
            "yolo_frame_predict_calls",
            "yolo_custom_detector_calls",
            "yolo_batch_fallback_count",
        ):
            try:
                bucket[key] += int(row.get(key) or 0)
            except (TypeError, ValueError):
                pass
        actual_sizes = row.get("actual_batch_sizes")
        if isinstance(actual_sizes, list):
            for value in actual_sizes:
                try:
                    size = int(float(value))
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    bucket["_actual_batch_size_values"].append(size)
        fallback_errors = row.get("yolo_batch_fallback_errors")
        if isinstance(fallback_errors, list):
            for error in fallback_errors:
                text = str(error)
                if text and text not in bucket["yolo_batch_fallback_errors"] and len(bucket["yolo_batch_fallback_errors"]) < 10:
                    bucket["yolo_batch_fallback_errors"].append(text)
        if row.get("stage_parallel_elapsed_sec") is not None:
            try:
                elapsed = float(row.get("stage_parallel_elapsed_sec") or 0.0)
            except (TypeError, ValueError):
                elapsed = 0.0
            if elapsed > 0.0:
                current = bucket.get("_parallel_elapsed_sec")
                bucket["_parallel_elapsed_sec"] = max(float(current or 0.0), elapsed)
        if row.get("stage_parallel_workers") is not None:
            try:
                workers = int(float(row.get("stage_parallel_workers") or 0))
            except (TypeError, ValueError):
                workers = 0
            if workers > 0:
                current = bucket.get("_parallel_workers")
                bucket["_parallel_workers"] = max(int(current or 0), workers)
        if row.get("stage_scan_task_count") is not None:
            try:
                task_count = int(float(row.get("stage_scan_task_count") or 0))
            except (TypeError, ValueError):
                task_count = 0
            if task_count > 0:
                current = bucket.get("_parallel_scan_task_count")
                bucket["_parallel_scan_task_count"] = max(int(current or 0), task_count)
        try:
            worker_total_wall_sec += float(row.get("wall_sec") or 0.0)
            total_sampled_frames += int(row.get("sampled_frames") or 0)
        except (TypeError, ValueError):
            pass
    effective_total_wall_sec = 0.0
    for bucket in by_stage.values():
        worker_wall = float(bucket.get("wall_sec") or 0.0)
        wall = worker_wall
        parallel_elapsed = bucket.pop("_parallel_elapsed_sec", None)
        parallel_workers = bucket.pop("_parallel_workers", None)
        parallel_scan_task_count = bucket.pop("_parallel_scan_task_count", None)
        if parallel_elapsed is not None and float(parallel_elapsed or 0.0) > 0.0:
            wall = float(parallel_elapsed)
            bucket["worker_wall_sec"] = round(worker_wall, 6)
            bucket["parallel_elapsed_sec"] = round(wall, 6)
            if parallel_workers is not None:
                bucket["parallel_workers"] = int(parallel_workers)
                bucket["parallel_enabled"] = int(parallel_workers) > 1
            if parallel_scan_task_count is not None:
                bucket["scan_task_count"] = int(parallel_scan_task_count)
        sampled = int(bucket.get("sampled_frames") or 0)
        bucket["effective_sampled_fps"] = round(sampled / wall, 6) if wall > 0 else 0.0
        bucket["batch_sizes"] = sorted(int(value) for value in bucket.get("batch_sizes", []))
        bucket["batch_enabled"] = any(int(value) > 1 for value in bucket["batch_sizes"])
        actual_batch_sizes = [int(value) for value in bucket.pop("_actual_batch_size_values", [])]
        actual_batch_counts = Counter(actual_batch_sizes)
        bucket["actual_batch_sizes"] = sorted(actual_batch_counts)
        bucket["actual_batch_size_counts"] = {str(size): int(count) for size, count in sorted(actual_batch_counts.items())}
        bucket["max_actual_batch_size"] = max(actual_batch_sizes, default=0)
        bucket["avg_actual_batch_size"] = round(sum(actual_batch_sizes) / len(actual_batch_sizes), 6) if actual_batch_sizes else 0.0
        bucket["underfilled_batch_count"] = sum(
            count
            for size, count in actual_batch_counts.items()
            if bucket["batch_sizes"] and int(size) < max(int(value) for value in bucket["batch_sizes"])
        )
        bucket["gpu_device_observed"] = any(_timing_device_is_gpu(value) for value in bucket.get("actual_devices", []))
        effective_total_wall_sec += wall
        for key in ("wall_sec", "decode_sec", "inference_sec", "postprocess_sec", "scan_duration_sec"):
            bucket[key] = round(float(bucket.get(key) or 0.0), 6)
    summary = {
        "schema_version": "key_action_yolo_timing.v1",
        "timing_rows_path": str(timing_path),
        "row_count": len(all_rows),
        "last_append_count": len(normalized_rows),
        "total_wall_sec": round(effective_total_wall_sec, 6),
        "worker_total_wall_sec": round(worker_total_wall_sec, 6),
        "effective_total_wall_sec": round(effective_total_wall_sec, 6),
        "total_sampled_frames": int(total_sampled_frames),
        "effective_sampled_fps": round(total_sampled_frames / effective_total_wall_sec, 6) if effective_total_wall_sec > 0 else 0.0,
        "by_stage": by_stage,
        "updated_at": time.time(),
    }
    _write_json(paths["metadata"] / "yolo_timing_summary.json", summary)
    return summary


def _compact_yolo_executor_stage(stage: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(stage, Mapping):
        return {}
    keys = (
        "sampled_frames",
        "wall_sec",
        "worker_wall_sec",
        "parallel_elapsed_sec",
        "parallel_workers",
        "scan_task_count",
        "decode_sec",
        "inference_sec",
        "postprocess_sec",
        "effective_sampled_fps",
        "scan_backends",
        "planned_scan_backends",
        "requested_devices",
        "actual_devices",
        "device_fallbacks",
        "gpu_device_observed",
        "batch_sizes",
        "batch_size_sources",
        "gpu_batch_default_applied",
        "batch_enabled",
        "streaming_fine_scan",
        "actual_batch_sizes",
        "actual_batch_size_counts",
        "max_actual_batch_size",
        "avg_actual_batch_size",
        "underfilled_batch_count",
        "batch_count",
        "yolo_predict_call_count",
        "yolo_batch_predict_attempts",
        "yolo_batch_predict_calls",
        "yolo_frame_predict_calls",
        "yolo_custom_detector_calls",
        "yolo_batch_fallback_count",
        "yolo_batch_fallback_errors",
    )
    return {key: stage.get(key) for key in keys if key in stage}


def _yolo_executor_diagnostics(
    timing_summary: Mapping[str, Any] | None,
    *,
    coarse_yolo_row_count: int | None = None,
    fine_yolo_row_count: int | None = None,
) -> dict[str, Any]:
    by_stage = timing_summary.get("by_stage") if isinstance(timing_summary, Mapping) else {}
    by_stage = by_stage if isinstance(by_stage, Mapping) else {}
    diagnostics = {
        "schema_version": "key_action_yolo_executor_diagnostics.v1",
        "coarse_yolo_row_count": int(coarse_yolo_row_count or 0),
        "fine_yolo_row_count": int(fine_yolo_row_count or 0),
        "total_sampled_frames": int((timing_summary or {}).get("total_sampled_frames") or 0)
        if isinstance(timing_summary, Mapping)
        else 0,
        "total_wall_sec": float((timing_summary or {}).get("total_wall_sec") or 0.0)
        if isinstance(timing_summary, Mapping)
        else 0.0,
        "worker_total_wall_sec": float((timing_summary or {}).get("worker_total_wall_sec") or 0.0)
        if isinstance(timing_summary, Mapping)
        else 0.0,
        "coarse_segment_scan": _compact_yolo_executor_stage(by_stage.get("coarse_segment_scan")),
        "micro_refine_window_scan": _compact_yolo_executor_stage(by_stage.get("micro_refine_window_scan")),
        "paired_micro_refine_window_scan": _compact_yolo_executor_stage(by_stage.get("paired_micro_refine_window_scan")),
    }
    return diagnostics


def _key_segment_scan_windows(
    key_segments: list[KeyActionSegment],
    view: str,
    *,
    padding_sec: float,
    merge_gap_sec: float,
    max_window_sec: float | None = None,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for segment in key_segments:
        ref = getattr(segment, view, None)
        if ref is None:
            continue
        try:
            start = max(0.0, float(ref.local_start_sec) - padding_sec)
            end = max(start, float(ref.local_end_sec) + padding_sec)
        except (TypeError, ValueError):
            continue
        windows.append(
            {
                "start_sec": start,
                "end_sec": end,
                "segment_ids": [str(segment.segment_id)],
            }
        )
    if not windows:
        return []
    merged: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: float(item["start_sec"])):
        would_merge = bool(merged and float(window["start_sec"]) <= float(merged[-1]["end_sec"]) + merge_gap_sec)
        if would_merge and max_window_sec is not None:
            candidate_end = max(float(merged[-1]["end_sec"]), float(window["end_sec"]))
            would_merge = candidate_end - float(merged[-1]["start_sec"]) <= float(max_window_sec)
        if not would_merge:
            merged.append(dict(window))
            continue
        merged[-1]["end_sec"] = max(float(merged[-1]["end_sec"]), float(window["end_sec"]))
        merged[-1]["segment_ids"] = sorted(set([*merged[-1].get("segment_ids", []), *window.get("segment_ids", [])]))
    for window in merged:
        window["duration_sec"] = round(max(0.0, float(window["end_sec"]) - float(window["start_sec"])), 6)
    return _coalesce_scan_windows(
        merged,
        merge_gap_sec=_float_env_value("KEY_ACTION_FAST_LOCATE_FINE_WINDOW_COALESCE_GAP_SEC", 6.0),
        max_window_sec=_float_env_value("KEY_ACTION_FAST_LOCATE_MAX_COALESCED_FINE_WINDOW_SEC", 90.0),
    )


def _coalesce_scan_windows(
    windows: list[dict[str, Any]],
    *,
    merge_gap_sec: float,
    max_window_sec: float | None = None,
) -> list[dict[str, Any]]:
    if not windows:
        return []
    merged: list[dict[str, Any]] = []
    for window in sorted(windows, key=lambda item: float(item.get("start_sec") or 0.0)):
        start = float(window.get("start_sec") or 0.0)
        end = max(start + 0.1, float(window.get("end_sec") or start))
        if not merged:
            merged.append({**dict(window), "start_sec": start, "end_sec": end})
            continue
        previous = merged[-1]
        candidate_start = float(previous.get("start_sec") or 0.0)
        candidate_end = max(float(previous.get("end_sec") or candidate_start), end)
        would_merge = start <= float(previous.get("end_sec") or 0.0) + merge_gap_sec
        if would_merge and max_window_sec is not None:
            would_merge = candidate_end - candidate_start <= float(max_window_sec)
        if not would_merge:
            merged.append({**dict(window), "start_sec": start, "end_sec": end})
            continue
        previous["end_sec"] = candidate_end
        previous["duration_sec"] = round(max(0.0, candidate_end - candidate_start), 6)
        previous["segment_ids"] = sorted(set([*previous.get("segment_ids", []), *window.get("segment_ids", [])]))
        if window.get("seed_window_id"):
            seed_ids = list(previous.get("seed_window_ids") or [])
            if previous.get("seed_window_id"):
                seed_ids.append(str(previous.get("seed_window_id")))
                previous.pop("seed_window_id", None)
            seed_ids.append(str(window.get("seed_window_id")))
            previous["seed_window_ids"] = sorted(set(seed_ids))
    for window in merged:
        start = float(window.get("start_sec") or 0.0)
        end = max(start + 0.1, float(window.get("end_sec") or start))
        window["start_sec"] = round(start, 6)
        window["end_sec"] = round(end, 6)
        window["duration_sec"] = round(max(0.0, end - start), 6)
    return merged


def _session_sec_to_view_local_sec(manifest: SessionManifest, view: str, session_sec: float) -> float:
    source = _source_for_view(manifest, view)
    global_time = _global_time_from_session_sec(manifest, session_sec)
    capture_sec = max(0.0, global_time_to_local_sec(source, global_time))
    return _capture_sec_to_clamped_video_sec(source, capture_sec)


def _source_video_duration_limit_sec(source: VideoSource | None, duration_sec: float | None = None) -> float | None:
    if duration_sec is not None:
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        if duration > 0:
            return duration
    if source is None:
        return None
    try:
        duration = float(getattr(source, "duration_sec", 0.0) or 0.0)
    except (TypeError, ValueError, OverflowError):
        duration = 0.0
    if duration > 0:
        return duration
    try:
        duration = float(get_video_duration_sec(source.path))
    except Exception:
        return None
    return duration if duration > 0 else None


def _clamp_source_video_sec(
    source: VideoSource | None,
    video_sec: float,
    *,
    duration_sec: float | None = None,
) -> float:
    try:
        value = max(0.0, float(video_sec))
    except (TypeError, ValueError, OverflowError):
        value = 0.0
    limit = _source_video_duration_limit_sec(source, duration_sec)
    if limit is not None:
        value = min(value, float(limit))
    return round(value, 6)


def _capture_sec_to_clamped_video_sec(
    source: VideoSource | None,
    capture_sec: float,
    *,
    duration_sec: float | None = None,
) -> float:
    if source is None:
        return _clamp_source_video_sec(source, capture_sec, duration_sec=duration_sec)
    video_sec = capture_sec_to_video_sec(source, capture_sec, use_frame_time_map="auto")
    return _clamp_source_video_sec(source, video_sec, duration_sec=duration_sec)


def _micro_pair_scan_windows_from_yolo_rows(
    manifest: SessionManifest,
    primary_rows: list[dict[str, Any]],
    paired_view: str,
    config: DetectorConfig,
) -> list[dict[str, Any]]:
    interaction_rows = [
        row
        for row in primary_rows
        if isinstance(row, dict) and _row_has_hand_object_interaction(row)
    ]
    if not interaction_rows:
        interaction_rows = [
            row
            for row in primary_rows
            if isinstance(row, dict)
            and bool(row.get("is_experiment_active") or row.get("is_active"))
            and float(row.get("active_score") or 0.0) >= max(0.05, float(config.end_threshold or 0.0))
        ]
    if not interaction_rows:
        return []

    gap_sec = max(
        0.25,
        _float_env_value(
            "KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN_GAP_SEC",
            max(1.0, min(3.0, float(config.merge_gap_sec or 2.0))),
        ),
    )
    pad_sec = max(
        0.0,
        _float_env_value(
            "KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN_PAD_SEC",
            max(0.75, min(2.0, float(config.buffer_sec or 1.0))),
        ),
    )
    min_window_sec = max(0.5, _float_env_value("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_MIN_WINDOW_SEC", 2.0))
    max_window_sec = max(min_window_sec, _float_env_value("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_MAX_WINDOW_SEC", 8.0))

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_time: float | None = None
    for row in sorted(interaction_rows, key=_row_alignment_sec):
        row_time = _row_alignment_sec(row)
        if current and last_time is not None and row_time > last_time + gap_sec:
            clusters.append(current)
            current = []
        current.append(row)
        last_time = row_time
    if current:
        clusters.append(current)

    windows: list[dict[str, Any]] = []
    for index, cluster in enumerate(clusters, start=1):
        times = [_row_alignment_sec(row) for row in cluster]
        start_session_sec = max(0.0, min(times) - pad_sec)
        end_session_sec = max(times) + pad_sec
        if end_session_sec - start_session_sec < min_window_sec:
            center = (start_session_sec + end_session_sec) / 2.0
            start_session_sec = max(0.0, center - min_window_sec / 2.0)
            end_session_sec = start_session_sec + min_window_sec
        slice_start = start_session_sec
        while slice_start < end_session_sec - 0.001:
            slice_end = min(end_session_sec, slice_start + max_window_sec)
            local_start = _session_sec_to_view_local_sec(manifest, paired_view, slice_start)
            local_end = max(local_start + 0.1, _session_sec_to_view_local_sec(manifest, paired_view, slice_end))
            windows.append(
                {
                    "start_sec": round(local_start, 6),
                    "end_sec": round(local_end, 6),
                    "segment_ids": sorted(
                        {
                            str(row.get("segment_id") or row.get("parent_segment_id") or "")
                            for row in cluster
                            if row.get("segment_id") or row.get("parent_segment_id")
                        }
                    ),
                    "paired_from_alignment_start_sec": round(slice_start, 6),
                    "paired_from_alignment_end_sec": round(slice_end, 6),
                    "paired_cluster_index": int(index),
                    "source_role": "paired_micro_support",
                }
            )
            slice_start = slice_end

    if not windows:
        return []
    merged: list[dict[str, Any]] = []
    merge_gap_sec = max(0.1, min(1.0, gap_sec / 2.0))
    for window in sorted(windows, key=lambda item: float(item["start_sec"])):
        if not merged or float(window["start_sec"]) > float(merged[-1]["end_sec"]) + merge_gap_sec:
            merged.append(dict(window))
            continue
        merged[-1]["end_sec"] = max(float(merged[-1]["end_sec"]), float(window["end_sec"]))
        merged[-1]["segment_ids"] = sorted(set([*merged[-1].get("segment_ids", []), *window.get("segment_ids", [])]))
        merged[-1]["paired_from_alignment_end_sec"] = max(
            float(merged[-1].get("paired_from_alignment_end_sec") or 0.0),
            float(window.get("paired_from_alignment_end_sec") or 0.0),
        )
    for window in merged:
        window["duration_sec"] = round(max(0.0, float(window["end_sec"]) - float(window["start_sec"])), 6)
    return _coalesce_scan_windows(
        merged,
        merge_gap_sec=_float_env_value("KEY_ACTION_FAST_LOCATE_FINE_WINDOW_COALESCE_GAP_SEC", 6.0),
        max_window_sec=_float_env_value("KEY_ACTION_FAST_LOCATE_MAX_COALESCED_FINE_WINDOW_SEC", 90.0),
    )


def _micro_refine_use_coarse_seed_windows(config: DetectorConfig) -> bool:
    default = bool(
        getattr(config, "long_video_two_stage_sampling", True)
        or _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False)
        or _bool_env("KEY_ACTION_DEFER_SEGMENT_ASSETS", False)
    )
    return _bool_env("KEY_ACTION_MICRO_REFINE_USE_COARSE_SEED_WINDOWS", default)


def _micro_refine_windows_from_coarse_rows(
    manifest: SessionManifest,
    key_segments: list[KeyActionSegment],
    coarse_yolo_rows: list[dict[str, Any]],
    config: DetectorConfig,
    view: str,
) -> list[dict[str, Any]]:
    if not coarse_yolo_rows or not _micro_refine_use_coarse_seed_windows(config):
        return []
    segment_windows: list[DetectedSegment] = []
    for index, segment in enumerate(key_segments, start=1):
        try:
            session_start = max(0.0, _session_time_sec(manifest, str(segment.global_start_time)))
            session_end = max(session_start + 0.1, _session_time_sec(manifest, str(segment.global_end_time)))
        except Exception:
            continue
        segment_windows.append(
            DetectedSegment(
                segment_id=str(getattr(segment, "segment_id", "") or f"segment_{index:06d}"),
                start_sec=session_start,
                end_sec=session_end,
                duration_sec=session_end - session_start,
                global_start_time=str(segment.global_start_time),
                global_end_time=str(segment.global_end_time),
                avg_motion_score=float((segment.cv_detection.avg_motion_score if segment.cv_detection else 0.0) or 0.0),
                avg_active_score=float((segment.cv_detection.avg_active_score if segment.cv_detection else 0.0) or 0.0),
                start_reason="key_segment_start",
                end_reason="key_segment_end",
                detector_backend=str(segment.detector_backend or ""),
                detector_source_view=str(segment.detector_source_view or ""),
                yolo_label_counts=dict(segment.yolo_label_counts or {}),
                yolo_interaction_count=int(segment.yolo_interaction_count or 0),
                boundary_confidence=float(segment.final_score or segment.raw_score or 0.0),
                boundary_support_count=int(segment.yolo_interaction_count or 0),
                boundary_source="key_segment_window",
                decision_path=str(segment.decision_path or ""),
                decision_trace=list(segment.decision_trace or []),
                reason_code=str(segment.reason_code or ""),
                raw_score=float(segment.raw_score or 0.0),
                final_score=float(segment.final_score or 0.0),
            )
        )
    if not segment_windows:
        return []
    filtered_rows = [
        row
        for row in coarse_yolo_rows
        if isinstance(row, dict) and _row_inside_detected_segments(_row_alignment_sec(row), segment_windows)
    ]
    if not filtered_rows:
        return []
    seed_segments = _fast_locate_fine_window_segments_from_yolo_rows(
        manifest,
        segment_windows,
        filtered_rows,
        config,
    )
    windows: list[dict[str, Any]] = []
    for index, segment in enumerate(seed_segments, start=1):
        try:
            session_start = max(0.0, float(segment.start_sec))
            session_end = max(session_start + 0.1, float(segment.end_sec))
        except (TypeError, ValueError):
            continue
        local_start = _session_sec_to_view_local_sec(manifest, view, session_start)
        local_end = max(local_start + 0.1, _session_sec_to_view_local_sec(manifest, view, session_end))
        windows.append(
            {
                "start_sec": round(local_start, 6),
                "end_sec": round(local_end, 6),
                "duration_sec": round(max(0.0, local_end - local_start), 6),
                "segment_ids": [str(getattr(segment, "segment_id", f"seed_window_{index:03d}"))],
                "seed_window_id": str(getattr(segment, "segment_id", f"seed_window_{index:03d}")),
                "seed_window_start_sec": round(session_start, 6),
                "seed_window_end_sec": round(session_end, 6),
                "source_role": "coarse_yolo_seed_window",
            }
        )
    return _coalesce_scan_windows(
        windows,
        merge_gap_sec=_float_env_value("KEY_ACTION_FAST_LOCATE_FINE_WINDOW_COALESCE_GAP_SEC", 6.0),
        max_window_sec=_float_env_value("KEY_ACTION_FAST_LOCATE_MAX_COALESCED_FINE_WINDOW_SEC", 90.0),
    )


def _pipeline_config(manifest: SessionManifest) -> dict[str, Any]:
    return dict(manifest.config or {})


def _discover_lab_assets_for_pipeline(
    manifest: SessionManifest,
    output_dir: Path,
    output_path: Path,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    config = _pipeline_config(manifest)
    project_root = config.get("model_inventory_project_root")
    if dry_run and not project_root:
        project_root = output_dir
    return discover_lab_assets(project_root=project_root, output_path=output_path)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _int_env_value(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return int(default)


def _float_env_value(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)


def _float_env_any_value(names: tuple[str, ...], default: float, *, minimum: float | None = None) -> float:
    value = float(default)
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        try:
            value = float(str(raw).strip())
            break
        except (TypeError, ValueError):
            value = float(default)
            break
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _first_configured_env(names: tuple[str, ...] | list[str]) -> tuple[str, str] | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None:
            return name, raw
    return None


def _format_env_float(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


@contextmanager
def _temporary_env(overrides: Mapping[str, Any]):
    if not overrides:
        yield
        return
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[str(key)] = str(value)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _coarse_scan_task_duration_sec(task: Mapping[str, Any]) -> float:
    for start_key, end_key in (("scan_start_sec", "scan_end_sec"), ("chunk_start_sec", "chunk_end_sec")):
        try:
            start = float(task.get(start_key) or 0.0)
            end_value = task.get(end_key)
            if end_value is None:
                continue
            return max(0.0, float(end_value) - start)
        except (TypeError, ValueError):
            continue
    try:
        return max(0.0, float(task.get("source_duration_sec") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _max_coarse_scan_task_duration_sec(scan_tasks: list[dict[str, Any]]) -> float:
    return max((_coarse_scan_task_duration_sec(task) for task in scan_tasks), default=0.0)


def _coarse_scan_role_env_names(*, fast_locate: bool, suffix: str) -> tuple[str, ...]:
    if fast_locate:
        return (f"KEY_ACTION_FAST_LOCATE_COARSE_{suffix}", f"KEY_ACTION_YOLO_COARSE_{suffix}")
    return (f"KEY_ACTION_YOLO_COARSE_{suffix}",)


def _coarse_scan_runtime_env_overrides(
    config: DetectorConfig,
    sample_fps: float,
    scan_tasks: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> tuple[dict[str, str], dict[str, Any]]:
    fast_locate = _fast_locate_runtime_enabled()
    enabled = _bool_env("KEY_ACTION_YOLO_AUTO_COARSE_SCAN_RUNTIME", True)
    max_task_duration = _max_coarse_scan_task_duration_sec(scan_tasks)
    overrides: dict[str, str] = {}
    reasons: list[str] = []
    sparse_names = _coarse_scan_role_env_names(fast_locate=fast_locate, suffix="FFMPEG_SPARSE_MODE")
    chunk_names = _coarse_scan_role_env_names(fast_locate=fast_locate, suffix="FFMPEG_CHUNK_SEC")
    if enabled:
        chunk_max_fps = _float_env_value("KEY_ACTION_YOLO_AUTO_COARSE_CHUNK_MAX_FPS", 2.5)
        chunk_min_duration = _float_env_value("KEY_ACTION_YOLO_AUTO_COARSE_CHUNK_MIN_SEC", 30.0)
        if _first_configured_env(sparse_names) is None and sample_fps <= chunk_max_fps and (
            dry_run or max_task_duration >= chunk_min_duration or fast_locate
        ):
            overrides[sparse_names[0]] = "chunks"
            reasons.append(
                "auto_selected_ffmpeg_chunks_to_avoid_per_frame_seek"
            )
        if _first_configured_env(chunk_names) is None:
            planned_chunk_sec = _coarse_scan_chunk_sec(config, sample_fps)
            if planned_chunk_sec <= 0.0:
                planned_chunk_sec = _float_env_value("KEY_ACTION_YOLO_AUTO_COARSE_FFMPEG_CHUNK_SEC", 900.0)
            max_ffmpeg_chunk_sec = _float_env_value("KEY_ACTION_YOLO_AUTO_COARSE_MAX_FFMPEG_CHUNK_SEC", 900.0)
            ffmpeg_chunk_sec = max(5.0, min(float(planned_chunk_sec), float(max_ffmpeg_chunk_sec)))
            if max_task_duration > 0.0:
                ffmpeg_chunk_sec = max(5.0, min(ffmpeg_chunk_sec, max_task_duration))
            overrides[chunk_names[0]] = _format_env_float(ffmpeg_chunk_sec)
            reasons.append("aligned_ffmpeg_chunk_sec_with_pipeline_coarse_chunks")
    plan = {
        "auto_runtime_enabled": bool(enabled),
        "fast_locate_runtime": bool(fast_locate),
        "dry_run": bool(dry_run),
        "sample_fps": float(sample_fps),
        "max_task_duration_sec": round(max_task_duration, 6),
        "env_overrides": dict(overrides),
        "reasons": reasons,
    }
    return overrides, plan


def _resolve_yolo_coarse_scan_io_limited_workers(
    scan_task_count: int,
    requested_workers: int,
    *,
    sample_fps: float,
    scan_tasks: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    if scan_task_count <= 0:
        return 0, {"enabled": False, "requested_workers": int(requested_workers), "resolved_workers": 0}
    requested = max(1, min(int(scan_task_count), int(requested_workers or 1)))
    fast_locate = _fast_locate_runtime_enabled()
    enabled_name = (
        "KEY_ACTION_FAST_LOCATE_COARSE_SCAN_IO_LIMIT"
        if fast_locate and os.environ.get("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_IO_LIMIT") is not None
        else "KEY_ACTION_YOLO_COARSE_SCAN_IO_LIMIT"
    )
    limit_enabled = _bool_env(enabled_name, True)
    max_io_default = max(1, min(8, int(os.cpu_count() or 8)))
    max_io_workers = max(1, _int_env_value("KEY_ACTION_YOLO_SCAN_IO_MAX_WORKERS", max_io_default))
    for name in (
        "KEY_ACTION_FAST_LOCATE_COARSE_SCAN_IO_MAX_WORKERS" if fast_locate else "",
        "KEY_ACTION_YOLO_COARSE_SCAN_IO_MAX_WORKERS",
    ):
        if name and os.environ.get(name) is not None:
            max_io_workers = max(1, _int_env_value(name, max_io_workers))
            break

    model_ref = {"scan_role": "long_video_coarse"}
    max_task_duration = _max_coarse_scan_task_duration_sec(scan_tasks)
    sparse_mode = "unknown"
    ffmpeg_workers_per_task = 1
    try:
        from .yolo_detector import _ffmpeg_worker_count_for_scan, _resolve_ffmpeg_sparse_mode_for_scan

        sparse_mode = _resolve_ffmpeg_sparse_mode_for_scan(sample_fps, max_task_duration, model_ref)
        if sparse_mode in {"seek", "chunks"}:
            ffmpeg_workers_per_task = max(1, int(_ffmpeg_worker_count_for_scan(model_ref)))
    except Exception:
        sparse_mode = "unknown"
        ffmpeg_workers_per_task = 1

    resolved = requested
    if limit_enabled:
        if sparse_mode in {"seek", "chunks"}:
            worker_cap = max(1, max_io_workers // max(1, ffmpeg_workers_per_task))
        else:
            worker_cap = max_io_workers
        resolved = max(1, min(requested, worker_cap))
    ffmpeg_available = bool(shutil.which("ffmpeg"))
    if sparse_mode == "chunks":
        expected_backend = "ffmpeg_sparse_pipe_chunks_or_chunks" if ffmpeg_available else "opencv_sparse_seek_fallback"
    elif sparse_mode == "seek":
        expected_backend = "ffmpeg_sparse_seek" if ffmpeg_available else "opencv_sparse_seek_fallback"
    elif sparse_mode == "opencv_seek":
        expected_backend = "opencv_sparse_seek"
    else:
        expected_backend = "opencv_frame_skip_or_detector_default"
    plan = {
        "enabled": bool(limit_enabled),
        "requested_workers": int(requested),
        "resolved_workers": int(resolved),
        "task_count": int(scan_task_count),
        "max_io_workers": int(max_io_workers),
        "sparse_mode": sparse_mode,
        "expected_scan_backend": expected_backend,
        "ffmpeg_available": ffmpeg_available,
        "ffmpeg_workers_per_task": int(ffmpeg_workers_per_task),
        "expected_concurrent_extractors": int(resolved * max(1, ffmpeg_workers_per_task)),
        "max_task_duration_sec": round(max_task_duration, 6),
    }
    if resolved < requested:
        plan["cap_reason"] = "coarse_scan_io_limit"
    return resolved, plan


def _resolve_yolo_coarse_scan_workers(scan_task_count: int, *, default_workers: int = 1) -> int:
    workers = max(1, int(default_workers or 1))
    for env_name in ("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_WORKERS", "KEY_ACTION_YOLO_COARSE_SCAN_WORKERS"):
        if os.environ.get(env_name) is not None:
            workers = _int_env_value(env_name, workers)
            break
    if scan_task_count <= 0:
        return 0
    return max(1, min(int(scan_task_count), int(workers)))


def _coarse_scan_chunking_enabled(config: DetectorConfig) -> bool:
    if _fast_locate_runtime_enabled():
        return _bool_env("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_CHUNKED", True)
    return _bool_env("KEY_ACTION_YOLO_COARSE_SCAN_CHUNKED", True)


def _coarse_scan_chunk_sec(config: DetectorConfig, sample_fps: float) -> float:
    default = float(getattr(config, "long_video_chunk_sec", 0.0) or 0.0)
    if _fast_locate_runtime_enabled():
        default = _float_env_value("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_CHUNK_SEC", default)
    else:
        default = _float_env_value("KEY_ACTION_YOLO_COARSE_SCAN_CHUNK_SEC", default)
    if default <= 0.0:
        return 0.0
    sample_period = 1.0 / max(0.001, float(sample_fps))
    return max(sample_period, float(default))


def _coarse_scan_overlap_sec(config: DetectorConfig, sample_fps: float) -> float:
    sample_period = 1.0 / max(0.001, float(sample_fps))
    default = min(90.0, sample_period * max(1, int(getattr(config, "yolo_continuity_frames", 3) or 1) - 1))
    if _fast_locate_runtime_enabled():
        return max(0.0, _float_env_value("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_OVERLAP_SEC", default))
    return max(0.0, _float_env_value("KEY_ACTION_YOLO_COARSE_SCAN_OVERLAP_SEC", default))


def _coarse_scan_source_duration_sec(
    source: VideoSource | None,
    *,
    dry_run: bool,
) -> float | None:
    if dry_run:
        return max(1.0, _float_env_value("KEY_ACTION_DRY_RUN_DURATION_SEC", 960.0))
    if source is None:
        return None
    try:
        return max(0.0, float(get_video_duration_sec(source.path)))
    except Exception:
        return None


def _source_capture_duration_sec(
    source: VideoSource | None,
    *,
    dry_run: bool,
) -> float | None:
    duration_sec = _coarse_scan_source_duration_sec(source, dry_run=dry_run)
    if dry_run or source is None:
        return duration_sec
    try:
        frame_summary = frame_time_map_summary(source)
    except Exception:
        frame_summary = None
    if isinstance(frame_summary, Mapping) and frame_summary.get("auto_frame_time_map_applied"):
        try:
            capture_span = float(frame_summary.get("capture_span_sec") or 0.0)
        except (TypeError, ValueError):
            capture_span = 0.0
        if capture_span > 0:
            return capture_span
    return duration_sec


def _source_session_start_sec(manifest: SessionManifest, source: VideoSource | None) -> float:
    if source is None:
        return 0.0
    try:
        source_start = parse_time(source.start_time) + timedelta(seconds=float(getattr(source, "offset_sec", 0.0) or 0.0))
        session_start = parse_time(manifest.session_start_time)
        return round((source_start - session_start).total_seconds(), 6)
    except Exception:
        return 0.0


def _source_has_reliable_capture_anchor(source: VideoSource | None) -> bool:
    if source is None:
        return False
    status = str(getattr(source, "capture_start_status", "") or "").strip().lower()
    source_name = str(getattr(source, "capture_start_source", "") or "").strip().lower()
    frames_csv = str(getattr(source, "frames_csv_path", "") or "").strip()
    if status == "inferred_from_capture_metadata":
        return True
    if source_name.startswith(("frames_csv_", "meta_json_", "video_store_frames_csv_", "video_store_meta_json_")):
        return True
    return bool(frames_csv)


def _dual_view_common_overlap_sec(
    manifest: SessionManifest,
    *,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    if not _bool_env("KEY_ACTION_REQUIRE_COMMON_DUAL_VIEW_OVERLAP", True):
        return None
    timeline_config = manifest.config.get("timeline_alignment") if isinstance(manifest.config, dict) else None
    first_source = manifest.videos.first_person
    third_source = manifest.videos.third_person
    if first_source is None or third_source is None:
        return None

    spans: dict[str, dict[str, Any]] = {}
    for view, source in (("first_person", first_source), ("third_person", third_source)):
        duration_sec = _source_capture_duration_sec(source, dry_run=dry_run)
        if duration_sec is None or duration_sec <= 0:
            return None
        global_start = _source_session_start_sec(manifest, source)
        global_end = global_start + float(duration_sec)
        spans[view] = {
            "global_start_sec": round(global_start, 6),
            "global_end_sec": round(global_end, 6),
            "global_start": round(global_start, 6),
            "global_end": round(global_end, 6),
            "duration_sec": round(float(duration_sec), 6),
        }

    has_requested_overlap = isinstance(timeline_config, Mapping) and (
        timeline_config.get("common_overlap") is not None
        or timeline_config.get("common_overlap_start_sec") is not None
        or timeline_config.get("common_overlap_end_sec") is not None
    )
    return strict_common_overlap_from_view_intervals(
        spans,
        source="manifest.timeline_alignment" if has_requested_overlap else "video_source_metadata",
        requested_overlap=timeline_config if isinstance(timeline_config, Mapping) else None,
    )


def _common_overlap_from_alignment_payload(
    payload: Mapping[str, Any],
    view_intervals: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if view_intervals:
        return strict_common_overlap_from_view_intervals(
            view_intervals,
            source="existing_timeline_alignment",
            requested_overlap=payload,
        )
    common = payload.get("common_overlap") if isinstance(payload.get("common_overlap"), Mapping) else {}
    common_views = common.get("views") if isinstance(common.get("views"), Mapping) else None
    if common_views:
        return strict_common_overlap_from_view_intervals(
            common_views,
            source="existing_timeline_alignment",
            requested_overlap=payload,
        )
    try:
        start = float(common.get("global_start_sec", payload.get("common_overlap_start_sec")))
        end = float(common.get("global_end_sec", payload.get("common_overlap_end_sec")))
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return {
        "available": True,
        "source": "existing_timeline_alignment",
        "global_start_sec": round(start, 6),
        "global_end_sec": round(end, 6),
        "global_start": round(start, 6),
        "global_end": round(end, 6),
        "duration_sec": round(end - start, 6),
        "views": dict(common.get("views") or {}),
    }


def _existing_timeline_alignment_payload(manifest: SessionManifest, paths: dict[str, Path]) -> tuple[dict[str, Any], str | None]:
    candidates = [
        paths["metadata"] / "timeline_alignment.json",
        Path(manifest.output_dir).parent / "timeline_alignment.json",
    ]
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        payload = _read_json_if_exists(candidate)
        if payload:
            return payload, str(candidate)
    return {}, None


def _source_frames_csv_path(source: VideoSource | None) -> str | None:
    if source is None:
        return None
    value = getattr(source, "frames_csv_path", None)
    if value:
        return str(value)
    path = getattr(source, "path", None)
    if path:
        candidate = Path(str(path)).parent / "frames.csv"
        if candidate.exists():
            return str(candidate)
    return None


def _apply_aligned_videos_to_manifest(
    manifest: SessionManifest,
    alignment_summary: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(alignment_summary, Mapping):
        return False
    outputs = alignment_summary.get("aligned_video_outputs")
    if not isinstance(outputs, Mapping):
        return False
    third_path = outputs.get("aligned_third_video")
    first_path = outputs.get("aligned_first_video")
    if not third_path or not first_path:
        return False
    third = Path(str(third_path))
    first = Path(str(first_path))
    if not (third.exists() and first.exists()):
        return False
    third_source = manifest.videos.third_person
    first_source = manifest.videos.first_person
    if first_source is None:
        return False
    raw_sources = {
        "third_person": {
            "raw_video_path": third_source.path,
            "raw_frames_csv_path": getattr(third_source, "frames_csv_path", None),
        },
        "first_person": {
            "raw_video_path": first_source.path,
            "raw_frames_csv_path": getattr(first_source, "frames_csv_path", None),
        },
    }
    third_source.path = str(third)
    first_source.path = str(first)
    third_source.frames_csv_path = None
    first_source.frames_csv_path = None
    try:
        third_source.duration_sec = float(get_video_duration_sec(str(third)))
    except Exception:
        third_source.duration_sec = None
    try:
        first_source.duration_sec = float(get_video_duration_sec(str(first)))
    except Exception:
        first_source.duration_sec = None
    manifest.config = dict(getattr(manifest, "config", {}) or {})
    manifest.config["aligned_video_analysis"] = {
        "enabled": True,
        "source": "dual_view_alignment_pipeline",
        "raw_sources": raw_sources,
        "aligned_third_video": str(third),
        "aligned_first_video": str(first),
        "aligned_side_by_side_video": str(outputs.get("aligned_side_by_side_video") or ""),
        "sync_index_csv": (
            (alignment_summary.get("artifacts") or {}).get("sync_index_csv")
            if isinstance(alignment_summary.get("artifacts"), Mapping)
            else None
        ),
        "alignment_quality_report": (
            (alignment_summary.get("artifacts") or {}).get("alignment_quality_report")
            if isinstance(alignment_summary.get("artifacts"), Mapping)
            else None
        ),
        "phase_consistency_report": (
            (alignment_summary.get("artifacts") or {}).get("phase_consistency_report")
            if isinstance(alignment_summary.get("artifacts"), Mapping)
            else None
        ),
    }
    return True


def _ensure_pre_coarse_timeline_alignment(
    manifest: SessionManifest,
    paths: dict[str, Path],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    existing_payload, existing_path = _existing_timeline_alignment_payload(manifest, paths)
    sources = manifest.videos.all_sources()
    views: dict[str, dict[str, Any]] = {}
    for view in ("first_person", "third_person"):
        source = sources.get(view)
        if source is None:
            continue
        duration_sec = _source_capture_duration_sec(source, dry_run=dry_run)
        global_start = _source_session_start_sec(manifest, source)
        global_end = global_start + float(duration_sec or 0.0) if duration_sec is not None else None
        views[view] = {
            "video_path": str(source.path),
            "start_time": source.start_time,
            "offset_sec": float(getattr(source, "offset_sec", 0.0) or 0.0),
            "source_duration_sec": round(float(duration_sec or 0.0), 6),
            "global_start_sec": round(global_start, 6),
            "global_end_sec": round(global_end, 6) if global_end is not None else None,
            "global_start": round(global_start, 6),
            "global_end": round(global_end, 6) if global_end is not None else None,
            "fps": float(getattr(source, "fps", 0.0) or 0.0),
            "role": getattr(source, "role", None) or view,
            "camera_id": getattr(source, "camera_id", None),
            "frames_csv_path": getattr(source, "frames_csv_path", None),
            "capture_start_source": getattr(source, "capture_start_source", None),
            "capture_start_status": getattr(source, "capture_start_status", None),
            "capture_anchor_reliable": _source_has_reliable_capture_anchor(source),
        }

    timeline_config = manifest.config.get("timeline_alignment") if isinstance(manifest.config, dict) else None
    existing_common_overlap = _common_overlap_from_alignment_payload(existing_payload, views) if existing_payload else None
    manifest_common_overlap = strict_common_overlap_from_view_intervals(
        views,
        source="manifest.timeline_alignment" if isinstance(timeline_config, Mapping) else "video_source_metadata",
        requested_overlap=timeline_config if isinstance(timeline_config, Mapping) else None,
    )
    common_overlap = existing_common_overlap or manifest_common_overlap

    reliable_statuses = {
        "aligned",
        "explicit",
        "shared_recording",
        "capture_start_common_timeline",
        "dual_view_capture_start_common_timeline",
        "calibrated",
        "calibrated_zero_offset",
        "manual_offset",
    }
    config_status = ""
    config_reliable = False
    if isinstance(timeline_config, Mapping):
        config_status = str(
            timeline_config.get("alignment_status")
            or timeline_config.get("status")
            or timeline_config.get("timeline_alignment_status")
            or ""
        ).strip().lower()
        explicit_config_reliable = timeline_config.get("alignment_reliable_for_dual_view_pairing")
        config_reliable = (
            config_status in reliable_statuses
            if explicit_config_reliable is None
            else bool(explicit_config_reliable)
        )

    existing_status = str(
        existing_payload.get("alignment_status")
        or existing_payload.get("status")
        or existing_payload.get("timeline_alignment_status")
        or ""
    ).strip().lower()
    explicit_existing_reliable = existing_payload.get("alignment_reliable_for_dual_view_pairing")
    existing_reliable = (
        existing_status in reliable_statuses
        if explicit_existing_reliable is None
        else bool(explicit_existing_reliable)
    )
    reliable = bool(existing_reliable or config_reliable)
    has_dual_view = {"first_person", "third_person"}.issubset(views)
    if has_dual_view and not reliable:
        reliable = bool(
            views["first_person"].get("capture_anchor_reliable")
            and views["third_person"].get("capture_anchor_reliable")
            and common_overlap
            and common_overlap.get("available")
            and float(common_overlap.get("duration_sec") or 0.0) > 0.05
        )
    if dry_run and has_dual_view:
        reliable = True
    status = existing_status or config_status or ("aligned" if reliable else "unreliable")
    time_axis_health: dict[str, Any] | None = None
    time_axis_unreliable = False
    if has_dual_view and not dry_run:
        try:
            time_axis_health = analyze_dual_view_time_axis(
                sources.get("third_person"),
                sources.get("first_person"),
            )
            time_axis_unreliable = time_axis_health.get("status") == STATUS_UNRELIABLE
            if time_axis_health:
                _write_json(paths["metadata"] / "time_axis_health.json", time_axis_health)
        except Exception as exc:
            time_axis_health = {
                "status": "warning",
                "reasons": [f"time_axis_health_check_exception:{exc.__class__.__name__}"],
                "can_publish_formal_materials": False,
                "can_write_video_memory": False,
            }
            _write_json(paths["metadata"] / "time_axis_health.json", time_axis_health)
    if time_axis_unreliable:
        reliable = False
        status = STATUS_UNRELIABLE
    frame_alignment_health: dict[str, Any] | None = None
    dual_view_alignment_pipeline_summary: dict[str, Any] | None = None
    frame_alignment_unreliable = False
    if has_dual_view and not dry_run and _bool_env("KEY_ACTION_REQUIRE_DUAL_VIEW_FRAME_ALIGNMENT", True):
        try:
            third_source = sources.get("third_person")
            first_source = sources.get("first_person")
            # The sync_index is the canonical alignment artifact. Full-run
            # aligned mp4 rebuild is useful for audit, but it is too expensive
            # to block first-pass analysis on long videos. Downstream stages
            # must keep sync_index/global timestamps as facts and only build
            # video previews/clips on demand.
            use_aligned_videos_for_analysis = _bool_env("KEY_ACTION_USE_ALIGNED_VIDEOS_FOR_ANALYSIS", False)
            build_full_aligned_videos = _bool_env("KEY_ACTION_BUILD_ALIGNED_VIDEOS", False)
            sync_output_fps = _float_env_value("KEY_ACTION_SYNC_OUTPUT_FPS", 5.0)
            dual_view_alignment_pipeline_summary = run_dual_view_alignment_pipeline(
                manifest,
                paths["metadata"],
                timestamp_field=os.environ.get("KEY_ACTION_SYNC_TIMESTAMP_FIELD") or None,
                max_delta_ms=_float_env_value("KEY_ACTION_SYNC_MAX_DELTA_MS", 300.0),
                median_gate_ms=_float_env_value("KEY_ACTION_SYNC_MEDIAN_GATE_MS", 50.0),
                p90_gate_ms=_float_env_value("KEY_ACTION_SYNC_P90_GATE_MS", 150.0),
                make_aligned_videos=bool(build_full_aligned_videos),
                target_fps=sync_output_fps,
            )
            aligned_manifest_applied = False
            if use_aligned_videos_for_analysis and str(dual_view_alignment_pipeline_summary.get("status") or "") == "alignment_ready_pending_yolo_phase":
                aligned_manifest_applied = _apply_aligned_videos_to_manifest(
                    manifest,
                    dual_view_alignment_pipeline_summary,
                )
                dual_view_alignment_pipeline_summary = {
                    **dual_view_alignment_pipeline_summary,
                    "aligned_videos_applied_to_manifest": bool(aligned_manifest_applied),
                    "downstream_analysis_input": "aligned_videos" if aligned_manifest_applied else "raw_videos_blocked",
                }
                if not aligned_manifest_applied:
                    dual_view_alignment_pipeline_summary = {
                        **dual_view_alignment_pipeline_summary,
                        "status": "frame_alignment_unreliable",
                            "blocked_reason": "aligned_video_outputs_missing_or_unreadable",
                        }
            elif str(dual_view_alignment_pipeline_summary.get("status") or "") == "alignment_ready_pending_yolo_phase":
                dual_view_alignment_pipeline_summary = {
                    **dual_view_alignment_pipeline_summary,
                    "aligned_videos_applied_to_manifest": False,
                    "downstream_analysis_input": "virtual_sync_index",
                    "skip_full_aligned_rebuild_for_analysis": True,
                    "blocking_aligned_video_rebuild": False,
                    "analysis_time_basis": "sync_index_global_timestamp",
                    "preview_video_policy": "build_window_or_material_clips_on_demand",
                }
            frame_alignment_health = analyze_dual_view_frame_alignment(
                third_video=str(getattr(third_source, "path", "")),
                first_video=str(getattr(first_source, "path", "")),
                third_frames_csv=_source_frames_csv_path(third_source),
                first_frames_csv=_source_frames_csv_path(first_source),
                target_fps=_float_env_value("KEY_ACTION_FRAME_ALIGNMENT_PREFLIGHT_FPS", 2.0),
            )
            frame_alignment_unreliable = (
                str(frame_alignment_health.get("status") or "") == "frame_time_alignment_unreliable"
                or frame_alignment_health.get("formal_results_allowed") is False
                or (
                    isinstance(dual_view_alignment_pipeline_summary, dict)
                    and str(dual_view_alignment_pipeline_summary.get("status") or "") == "frame_alignment_unreliable"
                )
            )
            _write_json(paths["metadata"] / "dual_view_frame_alignment_preflight.json", frame_alignment_health)
            if isinstance(dual_view_alignment_pipeline_summary, dict):
                _write_json(paths["metadata"] / "dual_view_alignment_pipeline_summary.json", dual_view_alignment_pipeline_summary)
        except Exception as exc:
            frame_alignment_health = {
                "schema_version": "dual_view_nearest_neighbor_alignment.v1",
                "status": "frame_time_alignment_unreliable",
                "formal_results_allowed": False,
                "video_memory_allowed": False,
                "reasons": [f"frame_alignment_preflight_exception:{exc.__class__.__name__}"],
                "error": str(exc),
            }
            dual_view_alignment_pipeline_summary = {
                "schema_version": "dual_view_alignment_pipeline.v1",
                "status": "frame_time_alignment_unreliable",
                "formal_results_allowed": False,
                "video_memory_allowed": False,
                "blocked_reason": "dual_view_alignment_pipeline_exception",
                "error": str(exc),
            }
            frame_alignment_unreliable = True
            _write_json(paths["metadata"] / "dual_view_frame_alignment_preflight.json", frame_alignment_health)
            _write_json(paths["metadata"] / "dual_view_alignment_pipeline_summary.json", dual_view_alignment_pipeline_summary)
    if frame_alignment_unreliable:
        reliable = False
        status = "frame_time_alignment_unreliable"
    gate_required = bool(
        has_dual_view
        and not dry_run
        and _bool_env("KEY_ACTION_REQUIRE_RELIABLE_DUAL_VIEW_ALIGNMENT", True)
    )
    gate_passed = bool((not gate_required) or reliable)
    payload = {
        "schema_version": "key_action_pre_coarse_timeline_alignment.v1",
        "stage": "time_alignment_preflight",
        "execution_order": [
            "time_alignment_preflight",
            "coarse_seek_scan",
            "coarse_dedupe_and_merge",
            "fine_parallel_segment_scan",
            "physical_action_extraction",
            "key_material_generation",
            "formal_publish_gate",
            "material_library_and_30_day_memory_write",
        ],
        "dual_view_alignment_execution_order": [
            "video_registration",
            "time_alignment_preflight",
            "single_view_time_axis_health",
            "dual_view_state_scan",
            "trim_irrelevant_frames",
            "alignment_units",
            "local_offset_estimation",
            "nearest_neighbor_sync_index",
            "aligned_video_rebuild",
            "alignment_quality_report",
            "action_phase_consistency",
            "formal_experiment_windows",
            "coarse_seek_scan",
            "coarse_dedupe_and_merge",
            "fine_parallel_segment_scan",
            "physical_action_extraction",
            "key_material_generation",
            "formal_publish_gate",
            "material_library_and_30_day_memory_write",
        ],
        "status": status,
        "alignment_status": status,
        "alignment_reliable_for_dual_view_pairing": bool(reliable),
        "time_axis_health": time_axis_health,
        "frame_alignment_health": frame_alignment_health,
        "dual_view_alignment_pipeline": dual_view_alignment_pipeline_summary,
        "time_axis_unreliable": bool(time_axis_unreliable),
        "frame_alignment_unreliable": bool(frame_alignment_unreliable),
        "formal_results_allowed": bool(not time_axis_unreliable and not frame_alignment_unreliable and reliable),
        "video_memory_allowed": bool(
            not time_axis_unreliable
            and not frame_alignment_unreliable
            and (
                not isinstance(time_axis_health, dict)
                or bool(time_axis_health.get("can_write_video_memory", True))
            )
        ),
        "gate_required": bool(gate_required),
        "gate_passed": bool(gate_passed),
        "gate_failure_reason": (
            None
            if gate_passed
            else "time_axis_unreliable"
            if time_axis_unreliable
            else "frame_time_alignment_unreliable"
            if frame_alignment_unreliable
            else "dual_view_timeline_alignment_not_reliable"
        ),
        "alignment_method": existing_payload.get("alignment_method") or ((timeline_config or {}).get("alignment_method") if isinstance(timeline_config, Mapping) else None),
        "reliability_reasons": existing_payload.get("reliability_reasons") or ((timeline_config or {}).get("reliability_reasons") if isinstance(timeline_config, Mapping) else None) or ([] if reliable else list((frame_alignment_health or time_axis_health or {}).get("reasons") or ["dual_view_timeline_alignment_unreliable"])),
        "source": "existing_timeline_alignment" if existing_payload else "video_source_metadata",
        "existing_timeline_alignment_path": existing_path,
        "dry_run": bool(dry_run),
        "common_overlap": common_overlap,
        "common_overlap_start_sec": common_overlap.get("global_start_sec") if isinstance(common_overlap, dict) else None,
        "common_overlap_end_sec": common_overlap.get("global_end_sec") if isinstance(common_overlap, dict) else None,
        "common_overlap_sec": common_overlap.get("duration_sec") if isinstance(common_overlap, dict) else None,
        "views": views,
    }
    if isinstance(common_overlap, dict) and common_overlap.get("available"):
        manifest.config = dict(getattr(manifest, "config", {}) or {})
        manifest.config["timeline_alignment"] = {
            "alignment_status": status,
            "alignment_reliable_for_dual_view_pairing": bool(reliable),
            "time_axis_health_status": (time_axis_health or {}).get("status") if isinstance(time_axis_health, dict) else None,
            "time_axis_unreliable": bool(time_axis_unreliable),
            "frame_alignment_health_status": (frame_alignment_health or {}).get("status") if isinstance(frame_alignment_health, dict) else None,
            "frame_alignment_unreliable": bool(frame_alignment_unreliable),
            "gate_required": bool(gate_required),
            "gate_passed": bool(gate_passed),
            "alignment_method": payload.get("alignment_method"),
            "reliability_reasons": payload.get("reliability_reasons"),
            "common_overlap_start_sec": common_overlap.get("global_start_sec"),
            "common_overlap_end_sec": common_overlap.get("global_end_sec"),
            "common_overlap_sec": common_overlap.get("duration_sec"),
            "global_start": common_overlap.get("global_start"),
            "global_end": common_overlap.get("global_end"),
            "views": common_overlap.get("views"),
            "source": payload["source"],
        }
    _write_json(paths["metadata"] / "pre_coarse_timeline_alignment.json", payload)
    if gate_required and not gate_passed:
        raise RuntimeError(
            "Dual-view timeline alignment is not reliable before coarse scan. "
            "Refusing to select experiment segments until first/third timestamps and frame mappings are aligned."
        )
    return payload


def _local_scan_bounds_for_common_overlap(
    manifest: SessionManifest,
    source: VideoSource | None,
    duration_sec: float | None,
    common_overlap: dict[str, Any] | None,
) -> tuple[float, float | None, bool]:
    if source is None or duration_sec is None:
        return 0.0, duration_sec, False
    if not common_overlap or not common_overlap.get("available"):
        return 0.0, float(duration_sec), False
    source_global_start = _source_session_start_sec(manifest, source)
    overlap_start = float(common_overlap.get("global_start_sec") or 0.0)
    overlap_end = float(common_overlap.get("global_end_sec") or 0.0)
    capture_start = max(0.0, overlap_start - source_global_start)
    capture_end = max(capture_start, overlap_end - source_global_start)
    local_start = _capture_sec_to_clamped_video_sec(source, capture_start, duration_sec=duration_sec)
    local_end = _capture_sec_to_clamped_video_sec(source, capture_end, duration_sec=duration_sec)
    local_end = min(float(duration_sec), max(local_start, local_end))
    if local_end <= local_start:
        return 0.0, 0.0, True
    return round(local_start, 6), round(local_end, 6), True


def _coarse_scan_tasks(
    manifest: SessionManifest,
    views: list[str],
    config: DetectorConfig,
    sample_fps: float,
    *,
    dry_run: bool,
) -> list[dict[str, Any]]:
    sources = manifest.videos.all_sources()
    chunking_enabled = _coarse_scan_chunking_enabled(config)
    chunk_sec = _coarse_scan_chunk_sec(config, sample_fps) if chunking_enabled else 0.0
    overlap_sec = _coarse_scan_overlap_sec(config, sample_fps) if chunk_sec > 0.0 else 0.0
    common_overlap = _dual_view_common_overlap_sec(manifest, dry_run=dry_run)
    tasks: list[dict[str, Any]] = []
    for view_index, view in enumerate(views):
        source = sources.get(view)
        duration_sec = _coarse_scan_source_duration_sec(source, dry_run=dry_run)
        local_start, local_end, common_overlap_applied = _local_scan_bounds_for_common_overlap(
            manifest,
            source,
            duration_sec,
            common_overlap,
        )
        scan_duration_sec = (
            max(0.0, float(local_end) - float(local_start))
            if local_end is not None
            else None
        )
        if not chunking_enabled or chunk_sec <= 0.0 or scan_duration_sec is None or scan_duration_sec <= chunk_sec:
            tasks.append(
                {
                    "view_index": int(view_index),
                    "view": view,
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "scan_start_sec": float(local_start) if common_overlap_applied else None,
                    "scan_end_sec": float(local_end) if common_overlap_applied and local_end is not None else None,
                    "chunk_start_sec": float(local_start),
                    "chunk_end_sec": float(local_end or 0.0),
                    "source_duration_sec": float(duration_sec or 0.0),
                    "common_overlap_applied": bool(common_overlap_applied),
                    "common_overlap_global_start_sec": common_overlap.get("global_start_sec") if common_overlap else None,
                    "common_overlap_global_end_sec": common_overlap.get("global_end_sec") if common_overlap else None,
                    "chunked": False,
                    "chunk_sec": float(chunk_sec),
                    "overlap_sec": float(overlap_sec),
                }
            )
            continue
        chunk_ranges: list[tuple[float, float]] = []
        cursor = float(local_start)
        local_scan_end = float(local_end)
        while cursor < local_scan_end - 1e-6:
            end = min(local_scan_end, cursor + chunk_sec)
            chunk_ranges.append((cursor, end))
            cursor = end
        chunk_count = len(chunk_ranges)
        for chunk_index, (chunk_start, chunk_end) in enumerate(chunk_ranges):
            scan_start = max(float(local_start), chunk_start - (overlap_sec if chunk_index > 0 else 0.0))
            scan_end = min(local_scan_end, chunk_end + (overlap_sec if chunk_index < chunk_count - 1 else 0.0))
            tasks.append(
                {
                    "view_index": int(view_index),
                    "view": view,
                    "chunk_index": int(chunk_index),
                    "chunk_count": int(chunk_count),
                    "scan_start_sec": float(scan_start),
                    "scan_end_sec": float(scan_end),
                    "chunk_start_sec": float(chunk_start),
                    "chunk_end_sec": float(chunk_end),
                    "source_duration_sec": float(duration_sec),
                    "common_overlap_applied": bool(common_overlap_applied),
                    "common_overlap_global_start_sec": common_overlap.get("global_start_sec") if common_overlap else None,
                    "common_overlap_global_end_sec": common_overlap.get("global_end_sec") if common_overlap else None,
                    "chunked": True,
                    "chunk_sec": float(chunk_sec),
                    "overlap_sec": float(overlap_sec),
                }
            )
    task_count = len(tasks)
    for task_index, task in enumerate(tasks):
        task["task_index"] = int(task_index)
        task["task_count"] = int(task_count)
    return tasks


def _dedupe_coarse_scan_tasks(tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for task in tasks:
        key = (
            str(task.get("view") or ""),
            round(float(task.get("scan_start_sec") or 0.0), 3),
            round(float(task.get("scan_end_sec") or task.get("chunk_end_sec") or 0.0), 3),
            round(float(task.get("chunk_start_sec") or 0.0), 3),
            round(float(task.get("chunk_end_sec") or 0.0), 3),
        )
        if key in seen:
            duplicates.append(
                {
                    "view": task.get("view"),
                    "chunk_index": task.get("chunk_index"),
                    "scan_start_sec": task.get("scan_start_sec"),
                    "scan_end_sec": task.get("scan_end_sec"),
                }
            )
            continue
        seen.add(key)
        deduped.append(dict(task))
    task_count = len(deduped)
    for task_index, task in enumerate(deduped):
        task["task_index"] = int(task_index)
        task["task_count"] = int(task_count)
    return deduped, {
        "input_task_count": len(tasks),
        "output_task_count": len(deduped),
        "duplicate_task_count": len(duplicates),
        "duplicates": duplicates[:20],
    }


def _dedupe_yolo_rows_by_view_time(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}

    def row_score(row: Mapping[str, Any]) -> tuple[float, float, int]:
        detections = row.get("detections") if isinstance(row.get("detections"), list) else []
        try:
            interaction = float(row.get("interaction_score") or 0.0)
        except (TypeError, ValueError):
            interaction = 0.0
        try:
            active = float(row.get("active_score") or row.get("motion_score") or 0.0)
        except (TypeError, ValueError):
            active = 0.0
        return interaction, active, len(detections)

    for row in rows:
        try:
            time_sec = float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0)
        except (TypeError, ValueError):
            time_sec = 0.0
        key = (
            str(row.get("source_view") or row.get("view") or ""),
            str(row.get("video_path") or ""),
            round(time_sec, 3),
        )
        current = best_by_key.get(key)
        if current is None or row_score(row) > row_score(current):
            best_by_key[key] = row
    return _sort_yolo_rows_by_alignment_time(list(best_by_key.values()))


def _resolve_yolo_fine_scan_workers(
    window_count: int,
    *,
    default_workers: int = 1,
    fast_locate_refined: bool = False,
) -> int:
    env_names = ["KEY_ACTION_YOLO_FINE_SCAN_WORKERS"]
    if fast_locate_refined:
        env_names.insert(0, "KEY_ACTION_FAST_LOCATE_FINE_SCAN_WORKERS")
    workers = max(1, int(default_workers or 1))
    env_configured = False
    for env_name in env_names:
        if os.environ.get(env_name) is not None:
            workers = _int_env_value(env_name, workers)
            env_configured = True
            break
    if fast_locate_refined and not env_configured:
        cpu_count = max(1, int(os.cpu_count() or 1))
        adaptive_workers = min(8, max(2, cpu_count // 2))
        workers = max(workers, adaptive_workers)
    if window_count <= 0:
        return 0
    return max(1, min(int(window_count), int(workers)))


def _resolve_yolo_fine_scan_model_mode(*, fast_locate_refined: bool = False) -> str:
    env_names = ["KEY_ACTION_YOLO_FINE_SCAN_MODEL_MODE"]
    if fast_locate_refined:
        env_names.insert(0, "KEY_ACTION_FAST_LOCATE_FINE_SCAN_MODEL_MODE")
    raw = None
    for env_name in env_names:
        if os.environ.get(env_name) is not None:
            raw = os.environ.get(env_name)
            break
    mode = str(raw or ("shared" if fast_locate_refined else "per_worker")).strip().lower().replace("-", "_")
    if mode in {"shared", "shared_model", "single_model"}:
        return "shared"
    if mode in {"serial", "single", "single_worker"}:
        return "serial"
    return "per_worker"


def _coarse_scan_view_worker_enabled(scan_tasks: list[dict[str, Any]], worker_count: int) -> bool:
    if not scan_tasks:
        return False
    raw_override = os.environ.get("KEY_ACTION_FAST_LOCATE_COARSE_VIEW_WORKERS")
    if raw_override is None:
        raw_override = os.environ.get("KEY_ACTION_YOLO_COARSE_VIEW_WORKERS")
    chunked = any(bool(task.get("chunked")) for task in scan_tasks)
    enabled = (
        str(raw_override).strip().lower() not in {"", "0", "false", "no", "off"}
        if raw_override is not None
        else not chunked
    )
    if not enabled:
        return False
    views = {str(task.get("view") or "") for task in scan_tasks if task.get("view")}
    return bool(len(views) > 0 and (worker_count > 1 or len(scan_tasks) > len(views)))


def _coarse_scan_grouped_by_view(scan_tasks: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    by_view: dict[str, list[dict[str, Any]]] = {}
    for task in scan_tasks:
        view = str(task.get("view") or "")
        by_view.setdefault(view, []).append(task)
    return [
        (
            view,
            sorted(
                tasks,
                key=lambda item: (
                    int(item.get("chunk_index") or 0),
                    float(item.get("scan_start_sec") or item.get("chunk_start_sec") or 0.0),
                ),
            ),
        )
        for view, tasks in sorted(
            by_view.items(),
            key=lambda item: (
                int(item[1][0].get("view_index") or 0) if item[1] else 0,
                item[0],
            ),
        )
    ]


def _yolo_row_alignment_sort_key(row: Mapping[str, Any]) -> tuple[float, float, str, int]:
    def _float_value(*values: Any, default: float = 0.0) -> float:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return default

    return (
        _float_value(row.get("alignment_time_sec"), row.get("session_time_sec"), row.get("local_time_sec"), row.get("time_sec")),
        _float_value(row.get("local_time_sec"), row.get("time_sec")),
        str(row.get("source_view") or row.get("view") or ""),
        int(_float_value(row.get("frame_index"), default=0.0)),
    )


def _sort_yolo_rows_by_alignment_time(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=_yolo_row_alignment_sort_key)


def _file_sha256_if_exists(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists() or not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _yolo_stage_cache_file_hashes(yolo_rows_path: Path, frame_rows_path: Path) -> dict[str, Any]:
    return {
        "yolo_rows_sha256": _file_sha256_if_exists(yolo_rows_path),
        "frame_rows_sha256": _file_sha256_if_exists(frame_rows_path),
    }


def _validate_yolo_stage_cache_files(
    cached_meta: Mapping[str, Any],
    *,
    yolo_rows_path: Path,
    frame_rows_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "valid": True,
        "hash_checked": False,
        "legacy_hash_missing": False,
        "files": {},
    }
    for label, path, hash_key in (
        ("yolo_rows", yolo_rows_path, "yolo_rows_sha256"),
        ("frame_rows", frame_rows_path, "frame_rows_sha256"),
    ):
        exists = path.exists() and path.stat().st_size > 0
        actual_hash = _file_sha256_if_exists(path) if exists else None
        expected_hash = str(cached_meta.get(hash_key) or "")
        file_result = {
            "path": str(path),
            "exists": bool(exists),
            "sha256": actual_hash,
            "expected_sha256": expected_hash or None,
        }
        if not exists:
            result["valid"] = False
            file_result["status"] = "missing_or_empty"
        elif expected_hash:
            result["hash_checked"] = True
            if actual_hash != expected_hash:
                result["valid"] = False
                file_result["status"] = "hash_mismatch"
            else:
                file_result["status"] = "hash_match"
        else:
            result["legacy_hash_missing"] = True
            file_result["status"] = "legacy_hash_missing"
        result["files"][label] = file_result
    return result


def _coarse_scan_backend_cache_config(sample_fps: float, configured_imgsz: int | None) -> dict[str, Any]:
    model_ref = {"scan_role": "long_video_coarse"}
    payload: dict[str, Any] = {
        "sample_fps": float(sample_fps),
        "ffmpeg_pipe_scan": _bool_env("KEY_ACTION_YOLO_FFMPEG_PIPE_SCAN", True),
        "ffmpeg_sparse_max_fps": _float_env_value("KEY_ACTION_YOLO_FFMPEG_SPARSE_MAX_FPS", 1.25),
        "batch_size": _yolo_planned_batch_size("long_video_coarse"),
        "coarse_scan_runtime_version": "coarse_scan_runtime.v2",
    }
    try:
        from .yolo_detector import (
            _ffmpeg_chunk_sec_for_scan,
            _ffmpeg_scale_width_for_scan,
            _ffmpeg_worker_count_for_scan,
            _resolve_ffmpeg_sparse_mode_for_scan,
        )

        payload.update(
            {
                "sparse_mode": _resolve_ffmpeg_sparse_mode_for_scan(sample_fps, None, model_ref),
                "ffmpeg_chunk_sec": float(_ffmpeg_chunk_sec_for_scan(model_ref)),
                "ffmpeg_scale_width": int(_ffmpeg_scale_width_for_scan(model_ref, int(configured_imgsz or 640))),
                "ffmpeg_workers": int(_ffmpeg_worker_count_for_scan(model_ref)),
            }
        )
    except Exception as exc:
        payload["backend_config_error"] = str(exc)
    return payload


def _yolo_planned_batch_size(scan_role: str) -> int:
    try:
        from .yolo_detector import _default_yolo_batch_size

        return int(_default_yolo_batch_size({"scan_role": scan_role}))
    except Exception:
        return max(1, _int_env_value("KEY_ACTION_YOLO_BATCH_SIZE", 16))


def _file_fingerprint(path_value: str | Path | None, *, deep_hash: bool = False) -> dict[str, Any]:
    if not path_value:
        return {"path": None, "exists": False}
    path = Path(path_value)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    payload: dict[str, Any] = {
        "path": str(path.resolve()),
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    should_hash = deep_hash or stat.st_size <= 64 * 1024 * 1024 or path.suffix.lower() in {".pt", ".onnx", ".engine", ".yaml", ".json"}
    if should_hash:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload["sha256"] = digest.hexdigest()
    else:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            head = handle.read(1024 * 1024)
            if stat.st_size > 1024 * 1024:
                handle.seek(max(0, stat.st_size - 1024 * 1024))
                tail = handle.read(1024 * 1024)
            else:
                tail = b""
        digest.update(head)
        digest.update(tail)
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        payload["fast_sha256"] = digest.hexdigest()
    return payload


def _yolo_detection_cache_signature(
    manifest: SessionManifest,
    config: DetectorConfig,
    *,
    views: list[str],
    model_paths_by_view: dict[str, str | None],
    analysis_proxy_summary: dict[str, Any] | None = None,
    coarse_sample_fps_override: float | None = None,
) -> dict[str, Any]:
    deep_video_hash = _bool_env("KEY_ACTION_STAGE_CACHE_DEEP_VIDEO_HASH", False)
    videos = {
        "third_person": _file_fingerprint(manifest.videos.third_person.path, deep_hash=deep_video_hash),
    }
    if manifest.videos.first_person is not None:
        videos["first_person"] = _file_fingerprint(manifest.videos.first_person.path, deep_hash=deep_video_hash)
    models = {
        view: _file_fingerprint(path, deep_hash=True)
        for view, path in sorted(model_paths_by_view.items())
    }
    coarse_sample_fps = (
        max(0.001, float(coarse_sample_fps_override))
        if coarse_sample_fps_override is not None
        else _coarse_yolo_sample_fps(config)
    )
    adaptive_imgsz = bool(getattr(config, "yolo_adaptive_imgsz", True))
    configured_imgsz = int(config.yolo_imgsz) if getattr(config, "yolo_imgsz", None) else None
    configured_min_imgsz = int(getattr(config, "yolo_min_imgsz", 960))
    configured_max_imgsz = int(getattr(config, "yolo_max_imgsz", 1280))
    if not adaptive_imgsz and configured_imgsz:
        configured_min_imgsz = configured_imgsz
        configured_max_imgsz = configured_imgsz
    config_payload = {
        "detector_backend": config.detector_backend,
        "yolo_parent_activity_version": _YOLO_PARENT_ACTIVITY_VERSION,
        "frame_time_map_version": "v4_true_n_view_worker_coarse_scan",
        "sample_fps": float(coarse_sample_fps),
        "long_video_two_stage_sampling": bool(getattr(config, "long_video_two_stage_sampling", True)),
        "long_video_stage1_sample_fps": float(getattr(config, "long_video_stage1_sample_fps", 0.0) or 0.0),
        "long_video_stage2_sample_fps": float(getattr(config, "long_video_stage2_sample_fps", 0.0) or 0.0),
        "long_video_chunk_sec": float(getattr(config, "long_video_chunk_sec", 0.0) or 0.0),
        "coarse_scan_chunked": bool(_coarse_scan_chunking_enabled(config)),
        "coarse_scan_chunk_sec": float(_coarse_scan_chunk_sec(config, coarse_sample_fps)),
        "coarse_scan_overlap_sec": float(_coarse_scan_overlap_sec(config, coarse_sample_fps)),
        "coarse_scan_backend": _coarse_scan_backend_cache_config(coarse_sample_fps, configured_imgsz or configured_min_imgsz),
        "yolo_conf": float(config.yolo_conf),
        "yolo_iou": float(config.yolo_iou),
        "yolo_device": str(config.yolo_device),
        "yolo_imgsz": configured_imgsz,
        "yolo_imgsz_by_view": {
            view: _yolo_imgsz_for_view(config, view)
            for view in views
        },
        "yolo_first_person_imgsz": int(getattr(config, "yolo_first_person_imgsz", 0) or 0) or None,
        "yolo_third_person_imgsz": int(getattr(config, "yolo_third_person_imgsz", 0) or 0) or None,
        "yolo_adaptive_imgsz": adaptive_imgsz,
        "yolo_min_imgsz": configured_min_imgsz,
        "yolo_max_imgsz": configured_max_imgsz,
        "yolo_scan_both_views": bool(_bool_from_config(config.yolo_scan_both_views)),
        "yolo_preferred_view": str(config.yolo_preferred_view or ""),
        "yolo_continuity_frames": int(config.yolo_continuity_frames),
        "expected_experiment_count": int(config.expected_experiment_count)
        if getattr(config, "expected_experiment_count", None) is not None
        else None,
        "expected_experiment_count_env_allowed": bool(_bool_env("KEY_ACTION_ALLOW_EXPECTED_EXPERIMENT_COUNT_ENV", False)),
        "coarse_view_worker_enabled": bool(_bool_env("KEY_ACTION_FAST_LOCATE_COARSE_VIEW_WORKERS", False)),
        "yolo_predict_lock_scope": str(os.environ.get("KEY_ACTION_YOLO_PREDICT_LOCK_SCOPE", "model")),
        "start_threshold": float(config.start_threshold),
        "end_threshold": float(config.end_threshold),
        "merge_gap_sec": float(config.merge_gap_sec),
        "min_segment_duration_sec": float(config.min_segment_duration_sec),
        "buffer_sec": float(config.buffer_sec),
        "class_thresholds": config.yolo_class_thresholds or {},
    }
    proxy_cache_payload = analysis_proxy_cache_payload(analysis_proxy_summary)
    proxy_views = proxy_cache_payload.get("views") if isinstance(proxy_cache_payload, dict) else {}
    if not any(bool(meta.get("proxy_used")) for meta in (proxy_views or {}).values() if isinstance(meta, dict)):
        proxy_cache_payload = {
            "schema_version": proxy_cache_payload.get("schema_version", "key_action_analysis_proxy.v1")
            if isinstance(proxy_cache_payload, dict)
            else "key_action_analysis_proxy.v1",
            "enabled": bool(proxy_cache_payload.get("enabled")) if isinstance(proxy_cache_payload, dict) else False,
            "proxy_used": False,
        }
    signature_payload = {
        "schema_version": "key_action_yolo_detection_cache.v1",
        "views": views,
        "videos": videos,
        "models": models,
        "config": config_payload,
        "analysis_proxy": proxy_cache_payload,
        "coarse_sample_fps_effective": float(coarse_sample_fps),
    }
    signature_payload["signature"] = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return signature_payload


def _global_yolo_stage_cache_root() -> Path:
    configured = os.environ.get("KEY_ACTION_YOLO_STAGE_CACHE_ROOT") or os.environ.get("KEY_ACTION_GLOBAL_YOLO_CACHE_ROOT")
    if configured:
        return Path(configured)
    lab_video_root = Path("D:/LabVideo")
    if lab_video_root.exists():
        return lab_video_root / "analysis_cache" / "yolo_detection"
    return Path("D:/LabMaterialLibrary") / "stage_cache" / "yolo_detection"


def _global_yolo_stage_cache_paths(cache_signature: dict[str, Any]) -> dict[str, Path]:
    signature = str(cache_signature.get("signature") or "")
    safe_signature = "".join(ch for ch in signature if ch.isalnum()) or "unknown"
    root = _global_yolo_stage_cache_root() / safe_signature[:2] / safe_signature
    return {
        "root": root,
        "meta": root / "yolo_detection_cache.json",
        "yolo_rows": root / "yolo_frame_rows.jsonl",
        "frame_rows": root / "frame_scores.jsonl",
    }


def _try_restore_global_yolo_stage_cache(
    *,
    cache_signature: dict[str, Any],
    yolo_rows_path: Path,
    frame_rows_path: Path,
    cache_meta_path: Path,
) -> dict[str, Any] | None:
    if not _bool_env("KEY_ACTION_YOLO_GLOBAL_STAGE_CACHE", True):
        return None
    cache_paths = _global_yolo_stage_cache_paths(cache_signature)
    meta_path = cache_paths["meta"]
    global_yolo_rows = cache_paths["yolo_rows"]
    global_frame_rows = cache_paths["frame_rows"]
    if not (meta_path.exists() and global_yolo_rows.exists() and global_frame_rows.exists()):
        return None
    meta = _read_json_if_exists(meta_path)
    if not isinstance(meta, dict) or meta.get("signature") != cache_signature.get("signature"):
        return None
    if global_yolo_rows.stat().st_size <= 0 or global_frame_rows.stat().st_size <= 0:
        return None
    yolo_rows_path.parent.mkdir(parents=True, exist_ok=True)
    frame_rows_path.parent.mkdir(parents=True, exist_ok=True)
    cache_meta_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(global_yolo_rows, yolo_rows_path)
    shutil.copy2(global_frame_rows, frame_rows_path)
    restored_meta = {
        **meta,
        "cache_hit": True,
        "cache_scope": "global",
        "global_cache_root": str(cache_paths["root"]),
        "restored_at": time.time(),
    }
    _write_json(cache_meta_path, restored_meta)
    return restored_meta


def _publish_global_yolo_stage_cache(
    *,
    cache_signature: dict[str, Any],
    yolo_rows_path: Path,
    frame_rows_path: Path,
    cache_meta_path: Path,
) -> None:
    if not _bool_env("KEY_ACTION_YOLO_GLOBAL_STAGE_CACHE", True):
        return
    if not (yolo_rows_path.exists() and frame_rows_path.exists() and cache_meta_path.exists()):
        return
    cache_paths = _global_yolo_stage_cache_paths(cache_signature)
    root = cache_paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(yolo_rows_path, cache_paths["yolo_rows"])
    shutil.copy2(frame_rows_path, cache_paths["frame_rows"])
    meta = _read_json_if_exists(cache_meta_path)
    if isinstance(meta, dict):
        meta = {
            **meta,
            "cache_scope": "global",
            "global_cache_root": str(root),
            "published_at": time.time(),
        }
        _write_json(cache_paths["meta"], meta)


def _resolution_profiles_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for row in rows:
        view = str(row.get("source_view") or row.get("view") or "unknown")
        profile = row.get("resolution_profile") if isinstance(row.get("resolution_profile"), dict) else {}
        width = profile.get("frame_width") or row.get("frame_width")
        height = profile.get("frame_height") or row.get("frame_height")
        imgsz = profile.get("yolo_imgsz") or row.get("yolo_imgsz")
        current = profiles.setdefault(
            view,
            {
                "frame_width": width,
                "frame_height": height,
                "yolo_imgsz": imgsz,
                "adaptive_imgsz": profile.get("adaptive_imgsz"),
                "low_resolution_input": bool(profile.get("low_resolution_input")),
                "high_resolution_input": bool(profile.get("high_resolution_input")),
                "row_count": 0,
            },
        )
        current["row_count"] = int(current.get("row_count") or 0) + 1
        if width:
            current["frame_width"] = width
        if height:
            current["frame_height"] = height
        if imgsz:
            current["yolo_imgsz"] = imgsz
        current["low_resolution_input"] = bool(current.get("low_resolution_input")) or bool(profile.get("low_resolution_input"))
        current["high_resolution_input"] = bool(current.get("high_resolution_input")) or bool(profile.get("high_resolution_input"))
        if profile.get("adaptive_imgsz") is not None:
            current["adaptive_imgsz"] = bool(profile.get("adaptive_imgsz"))
    return profiles


def _fast_locate_runtime_enabled() -> bool:
    return _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False) or _bool_env(
        "KEY_ACTION_DEFER_SEGMENT_ASSETS",
        False,
    )


def _resolve_yolo_coarse_scan_views(manifest: SessionManifest, config: DetectorConfig) -> tuple[list[str], bool, str]:
    configured_scan_both = _bool_from_config(config.yolo_scan_both_views) and manifest.videos.first_person is not None
    if _fast_locate_runtime_enabled():
        scan_both = _bool_env("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_BOTH_VIEWS", configured_scan_both)
    else:
        scan_both = _bool_env("KEY_ACTION_YOLO_COARSE_SCAN_BOTH_VIEWS", configured_scan_both)
    scan_both = bool(scan_both and manifest.videos.first_person is not None)
    fallback_view = str(config.yolo_preferred_view or "first_person")
    views = ["first_person", "third_person"] if scan_both else [fallback_view]
    if views == ["first_person"] and manifest.videos.first_person is None:
        views = ["third_person"]
    return views, scan_both, fallback_view


def _has_explicit_labsopguard_context(manifest: SessionManifest) -> bool:
    config = _pipeline_config(manifest)
    return any(
        bool(config.get(key))
        for key in (
            "labsopguard_root",
            "labsopguard_experiment_id",
            "labsopguard_experiments",
        )
    )


def _dry_run_labsopguard_summary(output_dir: Path) -> dict[str, Any]:
    summary = {
        "schema_version": "labsopguard_ingest_summary.v1",
        "session_dir": str(output_dir.resolve()),
        "dry_run": True,
        "labsopguard_root": None,
        "requested_experiment_ids": [],
        "processed_experiment_ids": [],
        "ingest_results": [],
        "evidence_tightening": None,
        "skipped": [],
        "reason": "dry_run_no_explicit_labsopguard_context",
    }
    metadata_dir = output_dir / "metadata"
    _write_json(metadata_dir / "labsopguard_ingest_summary.json", summary)
    write_jsonl(metadata_dir / "labsopguard_ingest_skipped.jsonl", [])
    return summary


def _ingest_labsopguard_for_pipeline(
    output_dir: Path,
    manifest: SessionManifest,
    *,
    dry_run: bool,
) -> dict[str, Any] | None:
    if dry_run and not _has_explicit_labsopguard_context(manifest):
        return _dry_run_labsopguard_summary(output_dir)
    return ingest_labsopguard_experiments(output_dir, manifest)


def _ensure_dry_run_confirmation_queue_compat() -> None:
    try:
        from . import confirmation_loop
    except Exception:
        return
    if hasattr(confirmation_loop, "_normalize_evidence_refs"):
        return

    def _normalize_evidence_refs(value: Any) -> list[dict[str, Any]]:
        if not value:
            return []
        if isinstance(value, dict):
            return [dict(value)]
        if isinstance(value, (list, tuple)):
            return [dict(item) for item in value if isinstance(item, dict)]
        return []

    confirmation_loop._normalize_evidence_refs = _normalize_evidence_refs  # type: ignore[attr-defined]


def _attach_micro_refs_to_parents(key_segments: list[KeyActionSegment], micro_rows: list[dict[str, Any]]) -> None:
    refs_by_parent: dict[str, list[dict[str, Any]]] = {}
    for micro in sorted(micro_rows, key=lambda item: (str(item.get("parent_segment_id") or ""), float(item.get("start_sec", 0.0) or 0.0))):
        interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
        keyframes = micro.get("keyframes") if isinstance(micro.get("keyframes"), dict) else {}
        first_person = micro.get("first_person") if isinstance(micro.get("first_person"), dict) else {}
        third_person = micro.get("third_person") if isinstance(micro.get("third_person"), dict) else {}
        quality = micro.get("quality") if isinstance(micro.get("quality"), dict) else {}
        evidence = micro.get("evidence") if isinstance(micro.get("evidence"), dict) else {}
        refs_by_parent.setdefault(str(micro.get("parent_segment_id") or ""), []).append(
            {
                "micro_segment_id": micro.get("micro_segment_id"),
                "display_order": micro.get("display_order"),
                "display_id": micro.get("display_id"),
                "primary_object": interaction.get("primary_object"),
                "interaction_type": interaction.get("interaction_type"),
                "global_start_time": micro.get("global_start_time"),
                "global_end_time": micro.get("global_end_time"),
                "duration_sec": micro.get("duration_sec"),
                "max_interaction_score": interaction.get("max_interaction_score"),
                "confidence": quality.get("confidence"),
                "peak_keyframe": keyframes.get("peak_frame"),
                "first_person_clip": first_person.get("clip_path"),
                "third_person_clip": third_person.get("clip_path"),
                "manual_corrected": micro.get("manual_corrected", False),
                "dialogue_context_available": micro.get("dialogue_context_available", False),
                "dialogue_match_window_sec": micro.get("dialogue_match_window_sec"),
                "dialogue_keywords": micro.get("dialogue_keywords", []),
                "evidence_level": evidence.get("evidence_level") or micro.get("evidence_level"),
                "evidence": evidence,
                "asset_bindings": micro.get("asset_bindings", []),
                "yolo_evidence": micro.get("yolo_evidence", []),
                "class_threshold": micro.get("class_threshold", {}),
                "merged_from_micro_segment_ids": micro.get("merged_from_micro_segment_ids", []),
                "merge_reason": micro.get("merge_reason"),
            }
        )
    for segment in key_segments:
        segment.micro_segments = refs_by_parent.get(segment.segment_id, [])


def _validation_errors(validation_result: dict[str, Any]) -> list[str]:
    return [str(item.get("message")) for item in validation_result.get("issues", []) if item.get("severity") == "error"]


def _bool_from_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _source_for_view(manifest: SessionManifest, view: str | None) -> VideoSource:
    normalized = str(view or "").strip().lower()
    return manifest.videos.get(normalized) or manifest.videos.third_person


def _global_relative_sec(manifest: SessionManifest, global_time: str) -> float:
    return (parse_time(global_time) - parse_time(manifest.session_start_time)).total_seconds()


def _session_time_sec(manifest: SessionManifest, global_time: str) -> float:
    return _global_relative_sec(manifest, global_time)


def _global_time_from_session_sec(manifest: SessionManifest, session_sec: float) -> Any:
    return parse_time(manifest.session_start_time) + timedelta(seconds=max(0.0, float(session_sec)))


def _source_video_sec_to_session_sec(manifest: SessionManifest, source: VideoSource, video_sec: float) -> tuple[float, float]:
    capture_sec = video_sec_to_capture_sec(source, float(video_sec), use_frame_time_map="auto")
    global_time = local_sec_to_global_time(source, capture_sec)
    return _session_time_sec(manifest, global_time.isoformat()), capture_sec


def _apply_view_alignment_from_yolo(
    manifest: SessionManifest,
    rows: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    stats_by_view: dict[str, dict[str, Any]] = {}
    for row in rows:
        view = str(row.get("source_view") or row.get("view") or "unknown")
        stats = stats_by_view.setdefault(
            view,
            {
                "row_count": 0,
                "active_row_count": 0,
                "local_start_sec": None,
                "local_end_sec": None,
                "global_start_time": None,
                "global_end_time": None,
            },
        )
        stats["row_count"] += 1
        if bool(row.get("is_experiment_active")):
            stats["active_row_count"] += 1
        try:
            local_time = float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0)
        except (TypeError, ValueError):
            local_time = 0.0
        if stats["local_start_sec"] is None or local_time < float(stats["local_start_sec"]):
            stats["local_start_sec"] = local_time
        if stats["local_end_sec"] is None or local_time > float(stats["local_end_sec"]):
            stats["local_end_sec"] = local_time

    for view, stats in stats_by_view.items():
        source = _source_for_view(manifest, view)
        if stats["local_start_sec"] is not None:
            stats["local_start_sec"] = round(float(stats["local_start_sec"]), 6)
            capture_start = video_sec_to_capture_sec(source, float(stats["local_start_sec"]), use_frame_time_map="auto")
            stats["capture_start_sec"] = round(float(capture_start), 6)
            stats["global_start_time"] = local_sec_to_global_time(source, capture_start).isoformat()
        if stats["local_end_sec"] is not None:
            stats["local_end_sec"] = round(float(stats["local_end_sec"]), 6)
            capture_end = video_sec_to_capture_sec(source, float(stats["local_end_sec"]), use_frame_time_map="auto")
            stats["capture_end_sec"] = round(float(capture_end), 6)
            stats["global_end_time"] = local_sec_to_global_time(source, capture_end).isoformat()
        stats["offset_sec"] = float(getattr(source, "offset_sec", 0.0) or 0.0)
        stats["video_start_time"] = getattr(source, "start_time", None)
        stats["video_path"] = getattr(source, "path", None)
        stats["camera_id"] = getattr(source, "camera_id", None)
        stats["role"] = getattr(source, "role", None) or view
        stats["frame_time_map"] = frame_time_map_summary(source)

    offsets_sec: dict[str, float | None] = {"first_person": None, "third_person": None}
    if manifest.videos.first_person is not None:
        offsets_sec["first_person"] = float(getattr(manifest.videos.first_person, "offset_sec", 0.0) or 0.0)
    if manifest.videos.third_person is not None:
        offsets_sec["third_person"] = float(getattr(manifest.videos.third_person, "offset_sec", 0.0) or 0.0)
    for view, source in manifest.videos.extra_views.items():
        offsets_sec[str(view)] = float(getattr(source, "offset_sec", 0.0) or 0.0)

    alignment_payload, alignment_payload_path = _existing_timeline_alignment_payload(manifest, paths)
    timeline_status = str(
        alignment_payload.get("alignment_status")
        or alignment_payload.get("status")
        or ""
    ).strip().lower()
    reliable_statuses = {
        "aligned",
        "explicit",
        "shared_recording",
        "capture_start_common_timeline",
        "dual_view_capture_start_common_timeline",
        "calibrated",
        "calibrated_zero_offset",
        "manual_offset",
    }
    explicit_alignment_reliable = alignment_payload.get("alignment_reliable_for_dual_view_pairing")
    alignment_reliable = (
        bool(timeline_status in reliable_statuses)
        if explicit_alignment_reliable is None
        else bool(explicit_alignment_reliable)
    )
    if alignment_reliable:
        for stream in alignment_payload.get("streams") or []:
            if not isinstance(stream, dict):
                continue
            role = str(stream.get("role") or stream.get("view_type") or "").strip()
            if role not in {"first_person", "third_person"}:
                continue
            stream_status = str(stream.get("alignment_status") or stream.get("status") or "").strip().lower()
            if stream_status not in reliable_statuses:
                alignment_reliable = False
                break
    manifest_alignment_reliable = False
    if not alignment_reliable and _bool_env("KEY_ACTION_TRUST_MANIFEST_DUAL_VIEW_ALIGNMENT", True):
        first_source = manifest.videos.first_person
        third_source = manifest.videos.third_person
        if first_source is not None and third_source is not None and {"first_person", "third_person"}.issubset(stats_by_view):
            try:
                first_offset = float(getattr(first_source, "offset_sec", 0.0) or 0.0)
                third_offset = float(getattr(third_source, "offset_sec", 0.0) or 0.0)
                max_delta = _float_env_value("KEY_ACTION_MANIFEST_ALIGNMENT_MAX_OFFSET_DELTA_SEC", 0.05)
                offsets_match = abs(first_offset - third_offset) <= max_delta
            except (TypeError, ValueError):
                offsets_match = False
            first_start = str(getattr(first_source, "start_time", "") or "").strip()
            third_start = str(getattr(third_source, "start_time", "") or "").strip()
            start_times_match = bool(first_start and third_start and first_start == third_start)
            manifest_alignment_reliable = bool(offsets_match and start_times_match)
    if manifest_alignment_reliable:
        alignment_reliable = True
        timeline_status = "manifest_shared_recording"
    summary_status = "empty"
    if stats_by_view:
        summary_status = "aligned" if alignment_reliable else "pending"
    summary = {
        "schema_version": "view_alignment.yolo.v1",
        "method": "manifest_offsets",
        "status": summary_status,
        "alignment_status": summary_status,
        "alignment_reliable_for_dual_view_pairing": bool(alignment_reliable),
        "manifest_alignment_reliable": bool(manifest_alignment_reliable),
        "timeline_alignment_status": timeline_status or None,
        "timeline_alignment_path": alignment_payload_path,
        "unsafe_pairing_reason": None if alignment_reliable else "timeline_alignment_not_reliable",
        "views": sorted(stats_by_view),
        "row_counts_by_view": {view: int(stats["row_count"]) for view, stats in sorted(stats_by_view.items())},
        "offsets_sec": offsets_sec,
        "time_ranges_by_view": {view: stats for view, stats in sorted(stats_by_view.items())},
        "common_overlap": _dual_view_common_overlap_sec(manifest),
    }
    _write_json(paths["metadata"] / "view_alignment_from_yolo.json", summary)
    return summary


def _detect_segments(
    manifest: SessionManifest,
    config: DetectorConfig,
    paths: dict[str, Path],
    *,
    dry_run: bool,
) -> tuple[list[Any], list[Any], list[dict[str, Any]], dict[str, Any]]:
    return _detect_with_config(manifest, paths, config, dry_run=dry_run)


def _load_yolo_frame_rows(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], str | None]:
    candidates: list[Path] = []
    env_path = os.environ.get("KEY_ACTION_YOLO_FRAME_ROWS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            paths["cv_outputs"] / "yolo_frame_rows.jsonl",
            paths["metadata"] / "yolo_frame_rows.jsonl",
            paths["cv_outputs"] / "yolo_detections.jsonl",
            paths["metadata"] / "yolo_detections.jsonl",
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            rows = read_jsonl(candidate)
        except Exception as exc:
            print(f"[pipeline] Skipping unreadable YOLO frame rows {candidate}: {exc}")
            continue
        if rows:
            return rows, str(candidate)
    return [], None


def _resolve_yolo_frame_rows(
    paths: dict[str, Path],
    generated_yolo_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    yolo_frame_rows, yolo_rows_path = _load_yolo_frame_rows(paths)
    if generated_yolo_rows and not yolo_frame_rows:
        return generated_yolo_rows, str(paths["cv_outputs"] / "yolo_frame_rows.jsonl")
    return yolo_frame_rows, yolo_rows_path


def _resolve_micro_source_rows(
    paths: dict[str, Path],
    *,
    refined_yolo_rows: list[dict[str, Any]],
    yolo_frame_rows: list[dict[str, Any]],
    yolo_rows_path: str | None,
    key_segments: list[KeyActionSegment],
    dry_run: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    micro_source_rows = refined_yolo_rows or yolo_frame_rows
    if dry_run and not micro_source_rows:
        micro_source_rows = _mock_yolo_micro_rows_for_dry_run(key_segments)
    micro_source_path = (
        str(paths["cv_outputs"] / "yolo_micro_frame_rows.jsonl")
        if refined_yolo_rows
        else yolo_rows_path
    )
    return micro_source_rows, micro_source_path


def _label_count_sum(label_counts: Mapping[str, Any], labels: set[str]) -> int:
    total = 0
    for label in labels:
        try:
            total += int(label_counts.get(label, 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _yolo_parent_activity_score(row: Mapping[str, Any], *, interaction_score: float) -> tuple[float, str]:
    interactions = list(row.get("hand_object_interactions") or [])
    if interactions:
        return max(1.0, interaction_score), "hand_object_interaction"
    if interaction_score > 0.0:
        return interaction_score, "interaction_score"

    label_counts = row.get("label_counts") or {}
    if not isinstance(label_counts, Mapping):
        label_counts = {}
    hand_count = _label_count_sum(label_counts, _YOLO_HAND_LABELS)
    object_count = _label_count_sum(label_counts, _YOLO_OPERABLE_OBJECT_LABELS)
    if hand_count <= 0 or object_count <= 0:
        return 0.0, "no_hand_object_copresence"

    try:
        detector_active_score = float(row.get("active_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        detector_active_score = 0.0
    try:
        presence_score = float(row.get("presence_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        presence_score = 0.0

    copresence_score = 0.28 + 0.07 * min(hand_count, 2) + 0.06 * min(object_count, 3)
    detector_support = max(detector_active_score, presence_score)
    if detector_support > 0.0:
        copresence_score = max(copresence_score, min(0.75, detector_support))
    return min(0.82, copresence_score), "hand_object_copresence"


def _normalize_yolo_rows_for_pipeline(
    manifest: SessionManifest,
    rows: list[dict[str, Any]],
    config: DetectorConfig,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        view = str(item.get("source_view") or item.get("view") or "third_person")
        source = _source_for_view(manifest, view)
        local_time = float(item.get("local_time_sec", item.get("time_sec", 0.0)) or 0.0)
        alignment_time_sec, capture_time = _source_video_sec_to_session_sec(manifest, source, local_time)
        global_time = _global_time_from_session_sec(manifest, alignment_time_sec).isoformat()
        interaction_score = float(item.get("interaction_score", 0.0) or 0.0)
        raw_yolo_active_score = float(item.get("active_score", 0.0) or 0.0)
        evidence_score, evidence_source = _yolo_parent_activity_score(item, interaction_score=interaction_score)
        item["source_view"] = view
        item["view"] = view
        item["local_time_sec"] = local_time
        item["video_time_sec"] = local_time
        item["capture_time_sec"] = round(float(capture_time), 6)
        item["global_time"] = global_time
        item["alignment_time_sec"] = round(float(alignment_time_sec), 6)
        item["frame_time_map_applied"] = bool(abs(float(capture_time) - local_time) > 0.001)
        item["raw_yolo_active_score"] = raw_yolo_active_score
        item["motion_score"] = evidence_score
        item["active_score"] = evidence_score
        item["interaction_score"] = interaction_score
        item["is_active"] = bool(evidence_score >= config.start_threshold)
        item["is_experiment_active"] = bool(evidence_score >= config.start_threshold)
        item["active_score_source"] = evidence_source
        item["parent_activity_version"] = _YOLO_PARENT_ACTIVITY_VERSION
        normalized.append(item)
    return _sort_yolo_rows_by_alignment_time(normalized)


def _filter_rows_to_common_overlap(
    rows: list[dict[str, Any]],
    common_overlap: dict[str, Any] | None,
    *,
    row_kind: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not common_overlap or not common_overlap.get("available"):
        return rows, {
            "row_kind": row_kind,
            "applied": False,
            "input_rows": len(rows),
            "output_rows": len(rows),
            "reason": common_overlap.get("reason") if isinstance(common_overlap, dict) else "common_overlap_not_available",
        }
    start_sec = float(common_overlap.get("global_start_sec") or 0.0)
    end_sec = float(common_overlap.get("global_end_sec") or 0.0)
    if end_sec <= start_sec:
        return rows, {
            "row_kind": row_kind,
            "applied": False,
            "input_rows": len(rows),
            "output_rows": len(rows),
            "reason": "invalid_common_overlap",
        }
    filtered: list[dict[str, Any]] = []
    for row in rows:
        try:
            alignment_time = float(row.get("alignment_time_sec", row.get("local_time_sec", row.get("time_sec", 0.0))) or 0.0)
        except (TypeError, ValueError):
            alignment_time = 0.0
        if start_sec - 1e-6 <= alignment_time <= end_sec + 1e-6:
            filtered.append(row)
    return filtered, {
        "row_kind": row_kind,
        "applied": True,
        "input_rows": len(rows),
        "output_rows": len(filtered),
        "removed_rows": max(0, len(rows) - len(filtered)),
        "global_start_sec": round(start_sec, 6),
        "global_end_sec": round(end_sec, 6),
        "duration_sec": round(end_sec - start_sec, 6),
    }


def _frame_score_rows_from_yolo(
    manifest: SessionManifest,
    yolo_rows: list[dict[str, Any]],
    config: DetectorConfig,
) -> list[dict[str, Any]]:
    by_time: dict[float, dict[str, Any]] = {}
    for row in _normalize_yolo_rows_for_pipeline(manifest, yolo_rows, config):
        t = round(float(row.get("alignment_time_sec", 0.0)), 3)
        current = by_time.get(t)
        if current is not None and float(current.get("active_score", 0.0)) >= float(row.get("active_score", 0.0)):
            continue
        by_time[t] = {
            "frame_index": int(row.get("frame_index", 0) or 0),
            "local_time_sec": t,
            "time_sec": t,
            "global_time": row.get("global_time"),
            "motion_score": float(row.get("motion_score", 0.0)),
            "active_score": float(row.get("active_score", 0.0)),
            "roi": None,
            "is_active": bool(row.get("is_active")),
            "source_view": row.get("source_view"),
            "label_counts": row.get("label_counts") or {},
            "hand_object_interactions": row.get("hand_object_interactions") or [],
            "interaction_score": float(row.get("interaction_score", 0.0) or 0.0),
            "active_score_source": row.get("active_score_source"),
            "raw_yolo_active_score": float(row.get("raw_yolo_active_score", 0.0) or 0.0),
            "parent_activity_version": row.get("parent_activity_version") or _YOLO_PARENT_ACTIVITY_VERSION,
        }
    return [by_time[key] for key in sorted(by_time)]


def _segment_yolo_stats(segment: Any, frame_rows: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    label_counts: Counter[str] = Counter()
    interaction_count = 0
    for row in frame_rows:
        t = float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0)
        if not (float(segment.start_sec) <= t <= float(segment.end_sec)):
            continue
        label_counts.update({str(key): int(value) for key, value in (row.get("label_counts") or {}).items()})
        interaction_count += len(row.get("hand_object_interactions") or [])
    return dict(label_counts), interaction_count


def _segment_avg_score(
    segment: Any,
    frame_rows: list[dict[str, Any]],
    field: str,
) -> float:
    values: list[float] = []
    for row in frame_rows:
        t = float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0)
        if not (float(segment.start_sec) <= t <= float(segment.end_sec)):
            continue
        try:
            values.append(float(row.get(field, 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return sum(values) / len(values) if values else 0.0


_YOLO_CONTEXT_ONLY_LABELS = {"paper", "lab_coat", "ppe_storage"}
_YOLO_BOUNDARY_OBJECT_LABELS = {
    "balance",
    "beaker",
    "container",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
    "tube",
    "tube_cap",
}
_YOLO_WEIGHING_ANCHOR_LABELS = {"balance", "sample_bottle", "sample_bottle_blue", "spatula"}
_YOLO_TRANSFER_ONLY_LABELS = {"pipette", "pipette_tip", "spearhead"}


def _norm_yolo_label(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _row_time_sec(row: Mapping[str, Any]) -> float:
    for key in ("alignment_time_sec", "session_time_sec", "local_time_sec", "time_sec"):
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _row_labels(row: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    counts = row.get("label_counts") if isinstance(row.get("label_counts"), Mapping) else {}
    for label, count in dict(counts).items():
        try:
            if int(count) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        normalized = _norm_yolo_label(label)
        if normalized:
            labels.add(normalized)
    for item in row.get("detections") or []:
        if isinstance(item, Mapping):
            normalized = _norm_yolo_label(item.get("label") or item.get("raw_label"))
            if normalized:
                labels.add(normalized)
    return labels


def _interaction_object_label(interaction: Mapping[str, Any]) -> str:
    return _norm_yolo_label(interaction.get("object_label") or interaction.get("object_name") or interaction.get("label"))


def _is_weighing_session(manifest: SessionManifest, rows: list[dict[str, Any]] | None = None) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            manifest.session_id,
            manifest.config.get("experiment_title") if isinstance(manifest.config, dict) else "",
            manifest.config.get("experiment_name") if isinstance(manifest.config, dict) else "",
        )
    ).casefold()
    if any(token in text for token in ("weigh", "balance", "称量", "天平", "绉伴噺")):
        return True
    label_counts: Counter[str] = Counter()
    for row in rows or []:
        label_counts.update(_row_labels(row))
    return label_counts["balance"] >= 3 and any(label_counts[label] > 0 for label in _YOLO_WEIGHING_ANCHOR_LABELS - {"balance"})


def _boundary_support_score(row: Mapping[str, Any], *, weighing_like: bool) -> float:
    labels = _row_labels(row)
    if labels and labels <= _YOLO_CONTEXT_ONLY_LABELS:
        return 0.0

    best = 0.0
    has_weighing_anchor = bool(labels & _YOLO_WEIGHING_ANCHOR_LABELS)
    for interaction in row.get("hand_object_interactions") or []:
        if not isinstance(interaction, Mapping):
            continue
        obj = _interaction_object_label(interaction)
        if not obj or obj in _YOLO_CONTEXT_ONLY_LABELS:
            continue
        try:
            score = float(interaction.get("score", interaction.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if score < 0.5:
            continue
        if weighing_like and obj not in _YOLO_WEIGHING_ANCHOR_LABELS and not has_weighing_anchor:
            continue
        if weighing_like and obj in _YOLO_TRANSFER_ONLY_LABELS and not has_weighing_anchor:
            continue
        if obj in _YOLO_BOUNDARY_OBJECT_LABELS or obj in _YOLO_TRANSFER_ONLY_LABELS:
            best = max(best, min(1.0, 0.45 + 0.55 * score))

    if weighing_like:
        if "balance" in labels and (labels & (_YOLO_WEIGHING_ANCHOR_LABELS - {"balance"})):
            best = max(best, 0.62)
        if "spatula" in labels and (labels & {"balance", "sample_bottle", "sample_bottle_blue", "reagent_bottle", "beaker"}):
            best = max(best, 0.58)
        if {"gloved_hand", "hand"} & labels and labels & _YOLO_WEIGHING_ANCHOR_LABELS:
            best = max(best, 0.5)
        return best

    if {"gloved_hand", "hand"} & labels and labels & (_YOLO_BOUNDARY_OBJECT_LABELS | _YOLO_TRANSFER_ONLY_LABELS):
        best = max(best, 0.52)
    if len(labels & _YOLO_BOUNDARY_OBJECT_LABELS) >= 3:
        best = max(best, 0.45)
    return best


def _sample_period_sec(rows: list[dict[str, Any]], fallback_fps: float) -> float:
    times = sorted({_row_time_sec(row) for row in rows})
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return 1.0 / max(float(fallback_fps or 1.0), 0.001)
    return float(min(deltas))


def _avg_row_score(rows: list[dict[str, Any]], start_sec: float, end_sec: float, field: str) -> float:
    values: list[float] = []
    for row in rows:
        t = _row_time_sec(row)
        if not (start_sec <= t <= end_sec):
            continue
        try:
            values.append(float(row.get(field, 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    return float(sum(values) / len(values)) if values else 0.0


def _cluster_boundary_support_rows(
    support_rows: list[tuple[dict[str, Any], float]],
    *,
    max_gap_sec: float,
) -> list[list[tuple[dict[str, Any], float]]]:
    clusters: list[list[tuple[dict[str, Any], float]]] = []
    current: list[tuple[dict[str, Any], float]] = []
    last_time: float | None = None
    for row, score in sorted(support_rows, key=lambda item: _row_time_sec(item[0])):
        current_time = _row_time_sec(row)
        if last_time is not None and current_time - last_time > max_gap_sec and current:
            clusters.append(current)
            current = []
        current.append((row, score))
        last_time = current_time
    if current:
        clusters.append(current)
    return clusters


def _apply_refined_boundary(
    segment: Any,
    *,
    start_sec: float,
    end_sec: float,
    frame_rows: list[dict[str, Any]],
    support_count: int,
    max_support: float,
    pseudo_source: VideoSource,
    duration_sec: float,
    boundary_source: str,
) -> Any:
    segment.start_sec = round(float(start_sec), 6)
    segment.end_sec = round(float(min(end_sec, duration_sec)), 6)
    segment.duration_sec = round(float(segment.end_sec - segment.start_sec), 6)
    segment.global_start_time = local_sec_to_global_time(pseudo_source, segment.start_sec).isoformat()
    segment.global_end_time = local_sec_to_global_time(pseudo_source, segment.end_sec).isoformat()
    segment.avg_motion_score = _avg_row_score(frame_rows, segment.start_sec, segment.end_sec, "motion_score")
    segment.avg_active_score = _avg_row_score(frame_rows, segment.start_sec, segment.end_sec, "active_score")
    segment.boundary_support_count = support_count
    segment.boundary_confidence = round(min(1.0, max_support + min(0.2, support_count * 0.01)), 6)
    segment.boundary_source = boundary_source
    segment.start_reason = "yolo_physical_evidence_window_start"
    segment.end_reason = "yolo_physical_evidence_window_end"
    return segment


def _refine_yolo_detected_segments(
    manifest: SessionManifest,
    detected_segments: list[Any],
    frame_rows: list[dict[str, Any]],
    config: DetectorConfig,
    duration_sec: float,
) -> list[Any]:
    if not detected_segments or not frame_rows:
        return detected_segments
    weighing_like = _is_weighing_session(manifest, frame_rows)
    sample_period = _sample_period_sec(frame_rows, float(config.parent_sample_fps or config.sample_fps))
    pre_roll = max(sample_period, min(float(config.buffer_sec or 0.0), sample_period))
    post_roll = max(sample_period, float(config.buffer_sec or 0.0))
    tail_grace_sec = max(6.0, 3.0 * post_roll)
    support_gap_sec = max(2.0 * sample_period, min(float(config.merge_gap_sec or 5.0), 5.0))
    pseudo_source = VideoSource(
        name="global_multiview",
        path="global_multiview",
        start_time=manifest.session_start_time,
        fps=float(config.parent_sample_fps or config.sample_fps),
        offset_sec=0.0,
    )
    refined_segments: list[Any] = []
    for segment in detected_segments:
        segment_rows = [
            row
            for row in frame_rows
            if float(segment.start_sec) <= _row_time_sec(row) <= float(segment.end_sec)
        ]
        support_rows = [
            (row, _boundary_support_score(row, weighing_like=weighing_like))
            for row in segment_rows
        ]
        support_rows = [(row, score) for row, score in support_rows if score >= 0.45]
        max_support = max((score for _row, score in support_rows), default=0.0)
        segment.boundary_support_count = len(support_rows)
        segment.boundary_confidence = round(min(1.0, max_support + min(0.2, len(support_rows) * 0.01)), 6)
        if not support_rows:
            segment.boundary_source = "yolo_active_score_window"
            refined_segments.append(segment)
            continue

        clusters = _cluster_boundary_support_rows(support_rows, max_gap_sec=support_gap_sec)
        valid_clusters: list[list[tuple[dict[str, Any], float]]] = []
        for cluster in clusters:
            first_support = _row_time_sec(cluster[0][0])
            last_support = _row_time_sec(cluster[-1][0])
            candidate_start = max(float(segment.start_sec), first_support - pre_roll)
            candidate_end = min(float(segment.end_sec), last_support + post_roll)
            if len(clusters) == 1 and float(segment.end_sec) - last_support <= tail_grace_sec:
                candidate_end = float(segment.end_sec)
            if candidate_end - candidate_start >= float(config.min_segment_duration_sec or 0.0):
                valid_clusters.append(cluster)

        if len(valid_clusters) > 1:
            for cluster in valid_clusters:
                first_support = _row_time_sec(cluster[0][0])
                last_support = _row_time_sec(cluster[-1][0])
                split_segment = replace(segment)
                refined_segments.append(
                    _apply_refined_boundary(
                        split_segment,
                        start_sec=max(float(segment.start_sec), first_support - pre_roll),
                        end_sec=min(float(segment.end_sec), last_support + post_roll),
                        frame_rows=frame_rows,
                        support_count=len(cluster),
                        max_support=max(score for _row, score in cluster),
                        pseudo_source=pseudo_source,
                        duration_sec=duration_sec,
                        boundary_source="yolo_physical_evidence_cluster",
                    )
                )
            continue

        active_support = valid_clusters[0] if valid_clusters else support_rows
        first_support = _row_time_sec(active_support[0][0])
        last_support = _row_time_sec(active_support[-1][0])
        refined_start = max(float(segment.start_sec), first_support - pre_roll)
        refined_end = min(float(segment.end_sec), last_support + post_roll)
        if float(segment.end_sec) - last_support <= tail_grace_sec:
            refined_end = float(segment.end_sec)
        if refined_end - refined_start < float(config.min_segment_duration_sec or 0.0):
            segment.boundary_source = "yolo_active_score_window"
            refined_segments.append(segment)
            continue

        trimmed = abs(refined_start - float(segment.start_sec)) > 0.25 or abs(refined_end - float(segment.end_sec)) > 0.25
        if trimmed:
            segment = _apply_refined_boundary(
                segment,
                start_sec=refined_start,
                end_sec=refined_end,
                frame_rows=frame_rows,
                support_count=len(active_support),
                max_support=max(score for _row, score in active_support),
                pseudo_source=pseudo_source,
                duration_sec=duration_sec,
                boundary_source="yolo_physical_evidence_window",
            )
        else:
            segment.boundary_source = "yolo_active_score_window"
        refined_segments.append(segment)

    for index, segment in enumerate(refined_segments, start=1):
        segment.segment_id = f"seg_{index:06d}"
    return refined_segments


def _experiment_episode_rows(
    manifest: SessionManifest,
    segments: list[Any],
    *,
    detector_summary: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        rows.append(
            {
                "schema_version": "key_action_experiment_episode.v1",
                "session_id": manifest.session_id,
                "episode_id": f"episode_{index:06d}",
                "segment_id": getattr(segment, "segment_id", None) or str(index),
                "global_start_time": getattr(segment, "global_start_time", ""),
                "global_end_time": getattr(segment, "global_end_time", ""),
                "session_start_sec": float(getattr(segment, "start_sec", 0.0) or 0.0),
                "session_end_sec": float(getattr(segment, "end_sec", 0.0) or 0.0),
                "duration_sec": float(getattr(segment, "duration_sec", 0.0) or 0.0),
                "detector_backend": getattr(segment, "detector_backend", "unknown"),
                "detector_source_view": getattr(segment, "detector_source_view", "unknown"),
                "decision_path": getattr(segment, "decision_path", ""),
                "decision_trace": list(getattr(segment, "decision_trace", [])),
                "fallback_used": bool(getattr(segment, "fallback_used", False)),
                "fallback_reason": getattr(segment, "fallback_reason", ""),
                "reason_code": getattr(segment, "reason_code", ""),
                "raw_score": float(getattr(segment, "raw_score", 0.0) or 0.0),
                "final_score": float(getattr(segment, "final_score", 0.0) or 0.0),
                "avg_active_score": float(getattr(segment, "avg_active_score", 0.0) or 0.0),
                "avg_motion_score": float(getattr(segment, "avg_motion_score", 0.0) or 0.0),
                "start_reason": getattr(segment, "start_reason", ""),
                "end_reason": getattr(segment, "end_reason", ""),
                "view_alignment": dict(detector_summary.get("view_alignment") or {}) if detector_summary else {},
                "source_layer": "episode_activity",
                "episode_status": "official",
                "candidate_status": "official_episode",
                "official_episode": True,
                "formal_results_allowed": True,
                "single_view_candidate": False,
                "candidate_reasons": [],
                "episode_window_expansion": (
                    dict((getattr(segment, "retrieval_boost_factors", {}) or {}).get("experiment_window_expansion") or {})
                    if isinstance(getattr(segment, "retrieval_boost_factors", {}), Mapping)
                    else {}
                ),
                "interpretation": "official_continuous_experiment_episode",
            }
        )
    return rows


def _yolo_model_path_for_view(config: DetectorConfig, view: str) -> Path | None:
    from .yolo_analysis import resolve_default_yolo_model

    view_name = str(view or "").strip().lower()
    explicit = None
    if view_name == "first_person":
        explicit = getattr(config, "yolo_first_person_model_path", None)
    elif view_name == "third_person":
        explicit = getattr(config, "yolo_third_person_model_path", None)
    return resolve_default_yolo_model(explicit or config.yolo_model_path)


def _yolo_imgsz_for_view(config: DetectorConfig, view: str) -> int | None:
    view_name = str(view or "").strip().lower()
    explicit = None
    if view_name == "first_person":
        explicit = getattr(config, "yolo_first_person_imgsz", None)
    elif view_name == "third_person":
        explicit = getattr(config, "yolo_third_person_imgsz", None)
    if explicit is not None:
        return int(explicit)
    return int(config.yolo_imgsz) if getattr(config, "yolo_imgsz", None) else None


def _yolo_inventory_refs(paths: dict[str, Path]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    inventory_path = paths["metadata"] / "model_inventory.json"
    if not inventory_path.exists():
        return None, []
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None, []
    class_schema = dict(inventory.get("class_schema") or {})
    if inventory.get("class_schema_path"):
        class_schema["path"] = inventory.get("class_schema_path")
        class_schema["available"] = True
    schema_classes = class_schema.get("classes")
    if not schema_classes and inventory.get("classes"):
        schema_classes = [{"name": str(label)} for label in inventory.get("classes") or []]
        class_schema["classes"] = schema_classes
        class_schema["class_count"] = len(schema_classes)
    capability_roles: dict[str, set[str]] = {}
    capabilities = inventory.get("capabilities") if isinstance(inventory.get("capabilities"), dict) else {}
    for role, info in capabilities.items():
        if not isinstance(info, dict):
            continue
        for label in info.get("classes") or []:
            normalized = str(label).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized:
                capability_roles.setdefault(normalized, set()).add(str(role))
    for item in class_schema.get("classes") or []:
        if not isinstance(item, dict):
            continue
        normalized = str(item.get("name") or item.get("label") or "").strip().lower().replace("-", "_").replace(" ", "_")
        roles = set(str(value) for value in item.get("roles", []) or [])
        roles.update(capability_roles.get(normalized, set()))
        if roles:
            item["roles"] = sorted(roles)
    annotation_refs = []
    for dataset in inventory.get("datasets") or []:
        if not isinstance(dataset, dict):
            continue
        label_count = int(dataset.get("total_label_count") or 0)
        image_count = int(dataset.get("total_image_count") or 0)
        if not label_count and not image_count:
            continue
        annotation_refs.append(
            {
                "asset_type": "annotation_dataset",
                "path": dataset.get("path"),
                "dataset_root": dataset.get("dataset_root"),
                "label_count": label_count,
                "image_count": image_count,
                "classes": list(dataset.get("class_names") or []),
            }
        )
    return (class_schema if class_schema.get("classes") or class_schema.get("path") else None), annotation_refs


def _run_yolo_segment_detection(
    manifest: SessionManifest,
    paths: dict[str, Path],
    config: DetectorConfig,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .yolo_detector import _load_yolo_model, build_segments_from_yolo_frame_rows, scan_yolo_video

    views, scan_both, preferred = _resolve_yolo_coarse_scan_views(manifest, config)
    coarse_sample_fps = _coarse_yolo_sample_fps(config)

    errors: list[str] = []
    timing_rows: list[dict[str, Any]] = []
    coarse_timing_summary: dict[str, Any] = {}
    model_paths_by_view: dict[str, str | None] = {}
    class_schema, annotation_asset_refs = _yolo_inventory_refs(paths)
    for view in views:
        model_path = _yolo_model_path_for_view(config, view)
        model_paths_by_view[view] = str(model_path) if model_path else None

    coarse_sources_by_view = manifest.videos.all_sources()
    proxy_default_enabled = _bool_env("KEY_ACTION_FAST_LOCATE_USE_ANALYSIS_PROXY_BY_DEFAULT", False)
    proxy_enabled = analysis_proxy_enabled(default=proxy_default_enabled)
    analysis_proxy_summary: dict[str, Any] = {
        "schema_version": "key_action_analysis_proxy.v1",
        "enabled": False,
        "reason": "disabled",
        "views": {},
        "timing_rows": [],
    }
    if proxy_enabled or os.environ.get("KEY_ACTION_ANALYSIS_PROXY_ENABLED") is not None:
        _emit_pipeline_progress(
            progress_callback,
            stage="analysis_proxy",
            progress=0.30,
            message="Preparing coarse-scan analysis proxy",
            views=views,
        )
        coarse_sources_by_view, analysis_proxy_summary = build_analysis_proxies(
            manifest,
            proxy_root=Path(manifest.output_dir).parent / "analysis_proxy",
            views=views,
            enabled=proxy_enabled,
            dry_run=dry_run,
            existing_only=(
                _bool_env("KEY_ACTION_FAST_LOCATE_ANALYSIS_PROXY_EXISTING_ONLY", True)
                if _fast_locate_runtime_enabled()
                else None
            ),
        )
        _write_json(paths["metadata"] / "analysis_proxy_summary.json", analysis_proxy_summary)
        proxy_timing_rows = [
            row for row in analysis_proxy_summary.get("timing_rows", []) if isinstance(row, dict)
        ]
        if proxy_timing_rows:
            _append_yolo_timing_rows(paths, proxy_timing_rows)
        _emit_pipeline_progress(
            progress_callback,
            stage="analysis_proxy",
            progress=0.32,
            message="Coarse-scan analysis proxy ready",
            views=views,
            proxy_used=bool(analysis_proxy_summary.get("proxy_used")),
        )
        if analysis_proxy_summary.get("proxy_used"):
            proxy_sample_fps = _float_env_value(
                "KEY_ACTION_FAST_LOCATE_PROXY_STAGE1_SAMPLE_FPS",
                coarse_sample_fps,
            )
            coarse_sample_fps = max(coarse_sample_fps, proxy_sample_fps)

    cache_meta_path = paths["metadata"] / "yolo_detection_cache.json"
    cached_meta = _read_json_if_exists(cache_meta_path)
    yolo_rows_path = paths["cv_outputs"] / "yolo_frame_rows.jsonl"
    frame_rows_path = paths["cv_outputs"] / "frame_scores.jsonl"
    for stale_streaming_path in (
        paths["cv_outputs"] / "yolo_streaming_micro_frame_rows.jsonl",
        paths["metadata"] / "streaming_fine_scan_summary.json",
    ):
        try:
            stale_streaming_path.unlink()
        except FileNotFoundError:
            pass
    cache_enabled = _bool_env("KEY_ACTION_YOLO_STAGE_CACHE", True) and not dry_run
    common_overlap = _dual_view_common_overlap_sec(manifest, dry_run=dry_run)
    common_overlap_filter_summary: dict[str, Any] = {
        "common_overlap": common_overlap,
        "yolo_rows": {"applied": False, "input_rows": 0, "output_rows": 0},
        "frame_rows": {"applied": False, "input_rows": 0, "output_rows": 0},
    }
    planned_coarse_scan_tasks = _coarse_scan_tasks(
        manifest,
        views,
        config,
        coarse_sample_fps,
        dry_run=dry_run,
    )
    scan_tasks, coarse_task_dedupe = _dedupe_coarse_scan_tasks(planned_coarse_scan_tasks)
    try:
        write_chunk_manifest(
            paths["metadata"],
            manifest,
            scan_tasks,
            sample_fps=coarse_sample_fps,
        )
    except Exception as exc:
        _write_json(
            paths["metadata"] / "chunk_manifest_error.json",
            {
                "schema_version": "chunk_manifest_error.v1",
                "error": str(exc),
                "task_count": len(scan_tasks),
            },
        )
    coarse_runtime_env, coarse_runtime_plan = _coarse_scan_runtime_env_overrides(
        config,
        coarse_sample_fps,
        scan_tasks,
        dry_run=dry_run,
    )
    with _temporary_env(coarse_runtime_env):
        cache_signature = _yolo_detection_cache_signature(
            manifest,
            config,
            views=views,
            model_paths_by_view=model_paths_by_view,
            analysis_proxy_summary=analysis_proxy_summary,
            coarse_sample_fps_override=coarse_sample_fps,
        )
        planned_task_total = max(1, len(scan_tasks))
        default_coarse_workers = min(planned_task_total, 8 if scan_both else 4)
        requested_coarse_workers = _resolve_yolo_coarse_scan_workers(
            planned_task_total,
            default_workers=default_coarse_workers,
        )
        coarse_worker_count, coarse_io_plan = _resolve_yolo_coarse_scan_io_limited_workers(
            planned_task_total,
            requested_coarse_workers,
            sample_fps=coarse_sample_fps,
            scan_tasks=scan_tasks,
        )
        coarse_view_worker_enabled = _coarse_scan_view_worker_enabled(scan_tasks, coarse_worker_count)
    coarse_scan_plan = {
        "schema_version": "key_action_coarse_scan_plan.v2",
        "sample_fps": float(coarse_sample_fps),
        "views": views,
        "scan_both_views": bool(scan_both),
        "task_count": len(scan_tasks),
        "planned_task_count_before_dedupe": len(planned_coarse_scan_tasks),
        "requested_parallel_workers": int(requested_coarse_workers),
        "resolved_parallel_workers": int(coarse_worker_count),
        "parallel_enabled": bool(coarse_worker_count > 1),
        "execution_mode": "view_serial_queues" if coarse_view_worker_enabled else "chunk_futures",
        "task_dedupe": coarse_task_dedupe,
        "runtime_env": coarse_runtime_plan,
        "io_limit": coarse_io_plan,
        "view_worker_enabled": bool(coarse_view_worker_enabled),
        "view_worker_groups": [
            {"view": view, "task_count": len(tasks)}
            for view, tasks in _coarse_scan_grouped_by_view(scan_tasks)
        ],
        "chunked": bool(_coarse_scan_chunking_enabled(config)),
        "chunk_sec": float(_coarse_scan_chunk_sec(config, coarse_sample_fps)),
        "overlap_sec": float(_coarse_scan_overlap_sec(config, coarse_sample_fps)),
    }
    cache_hit = (
        cache_enabled
        and cached_meta.get("signature") == cache_signature.get("signature")
        and yolo_rows_path.exists()
        and yolo_rows_path.stat().st_size > 0
        and frame_rows_path.exists()
        and frame_rows_path.stat().st_size > 0
    )
    cache_validation: dict[str, Any] = {"valid": False, "reason": "cache_disabled_or_signature_miss"}
    if cache_hit:
        cache_validation = _validate_yolo_stage_cache_files(
            cached_meta,
            yolo_rows_path=yolo_rows_path,
            frame_rows_path=frame_rows_path,
        )
        cache_hit = bool(cache_validation.get("valid"))
    if not cache_hit and cache_enabled:
        restored_meta = _try_restore_global_yolo_stage_cache(
            cache_signature=cache_signature,
            yolo_rows_path=yolo_rows_path,
            frame_rows_path=frame_rows_path,
            cache_meta_path=cache_meta_path,
        )
        if restored_meta is not None:
            cached_meta = restored_meta
            cache_validation = _validate_yolo_stage_cache_files(
                cached_meta,
                yolo_rows_path=yolo_rows_path,
                frame_rows_path=frame_rows_path,
            )
            cache_hit = bool(cache_validation.get("valid"))
    streaming_refined_rows: list[dict[str, Any]] = []
    streaming_fine_summary: dict[str, Any] = {
        "available": False,
        "enabled": False,
        "reason": "cache_hit" if cache_hit else "not_started",
    }
    if cache_hit:
        cache_load_started = time.perf_counter()
        _emit_pipeline_progress(
            progress_callback,
            stage="yolo_detection",
            progress=0.56,
            message="YOLO detection cache hit; loading frame rows",
            cache_hit=True,
            cache_scope=cached_meta.get("cache_scope") or "local",
            views=views,
        )
        yolo_rows = read_jsonl(yolo_rows_path)
        frame_rows = read_jsonl(frame_rows_path)
        cache_load_wall_sec = time.perf_counter() - cache_load_started
        view_alignment = _apply_view_alignment_from_yolo(manifest, yolo_rows, paths)
        yolo_rows, yolo_filter = _filter_rows_to_common_overlap(yolo_rows, common_overlap, row_kind="yolo_rows")
        frame_rows, frame_filter = _filter_rows_to_common_overlap(frame_rows, common_overlap, row_kind="frame_rows")
        common_overlap_filter_summary = {
            "common_overlap": common_overlap,
            "yolo_rows": yolo_filter,
            "frame_rows": frame_filter,
        }
        view_alignment["common_overlap_filter"] = common_overlap_filter_summary
        timing_rows = [
            {
                "stage": "yolo_scan",
                "pipeline_stage": "coarse_segment_scan",
                "scan_role": "long_video_coarse",
                "source_view": "multiview" if scan_both else preferred,
                "sample_fps": float(coarse_sample_fps),
                "sampled_frames": len(yolo_rows),
                "read_frames": 0,
                "grab_frames": 0,
                "decode_sec": 0.0,
                "inference_sec": 0.0,
                "postprocess_sec": 0.0,
                "wall_sec": round(cache_load_wall_sec, 6),
                "stage_parallel_elapsed_sec": round(cache_load_wall_sec, 6),
                "stage_parallel_workers": 0,
                "stage_scan_task_count": len(scan_tasks),
                "effective_sampled_fps": round(len(yolo_rows) / cache_load_wall_sec, 6) if cache_load_wall_sec > 0 else 0.0,
                "scan_backend": f"yolo_stage_cache_{cached_meta.get('cache_scope') or 'local'}",
                "planned_scan_backend": coarse_io_plan.get("expected_scan_backend"),
                "planned_sparse_mode": coarse_io_plan.get("sparse_mode"),
                "planned_batch_size": _yolo_planned_batch_size("long_video_coarse"),
                "cache_hit": True,
                "cache_scope": cached_meta.get("cache_scope") or "local",
                "cache_hash_checked": bool(cache_validation.get("hash_checked")),
            }
        ]
        coarse_timing_summary = _append_yolo_timing_rows(paths, timing_rows)
    else:
        raw_rows: list[dict[str, Any]] = []
        task_total = max(1, len(scan_tasks))
        shared_models_by_view: dict[str, Any] = {}
        coarse_model_load_mode = "model_path_cache"
        if (
            _bool_env("KEY_ACTION_FAST_LOCATE_COARSE_PRELOAD_VIEW_MODELS", True)
            and not dry_run
            and not os.environ.get("KEY_ACTION_DISABLE_YOLO_MODEL_PRELOAD")
        ):
            preload_started = time.perf_counter()
            for view in views:
                model_path_value = model_paths_by_view.get(view)
                if not model_path_value:
                    continue
                try:
                    model_load_started = time.perf_counter()
                    shared_models_by_view[view] = _load_yolo_model(None, Path(model_path_value))
                    timing_rows.append(
                        {
                            "stage": "yolo_model_load",
                            "pipeline_stage": "coarse_segment_scan",
                            "scan_role": "long_video_coarse",
                            "source_view": view,
                            "model_path": str(model_path_value),
                            "wall_sec": round(time.perf_counter() - model_load_started, 6),
                            "model_load_mode": "coarse_view_worker_preload",
                        }
                    )
                except Exception as exc:
                    errors.append(f"{view} model preload failed: {exc}")
            if shared_models_by_view:
                coarse_model_load_mode = "coarse_view_worker_preload"
            coarse_scan_plan["model_preload"] = {
                "enabled": True,
                "loaded_views": sorted(shared_models_by_view),
                "wall_sec": round(time.perf_counter() - preload_started, 6),
                "mode": coarse_model_load_mode,
            }

        streaming_fine_enabled = bool(
            _fast_locate_streaming_fine_scan_enabled()
            and bool(getattr(config, "long_video_two_stage_sampling", True))
        )
        streaming_fine_rows: list[dict[str, Any]] = []
        streaming_fine_timing_rows: list[dict[str, Any]] = []
        streaming_fine_errors: list[str] = []
        streaming_fine_futures: list[Any] = []
        streaming_fine_windows_by_view: dict[str, list[dict[str, Any]]] = {}
        streaming_seen_windows: set[tuple[str, float, float]] = set()
        streaming_lock = threading.Lock()
        streaming_first_submit_sec: float | None = None
        streaming_submit_count = 0
        streaming_fine_worker_count = 0
        streaming_fine_started = time.perf_counter()
        streaming_fine_executor: ThreadPoolExecutor | None = None
        streaming_fine_sample_fps = _refined_yolo_sample_fps(config)
        streaming_fine_scan_both_views = _bool_env(
            "KEY_ACTION_FAST_LOCATE_FINE_SCAN_BOTH_VIEWS",
            bool(config.yolo_scan_both_views) and manifest.videos.first_person is not None,
        )
        streaming_fine_preferred = os.environ.get("KEY_ACTION_FAST_LOCATE_FINE_PREFERRED_VIEW") or config.yolo_preferred_view
        if streaming_fine_preferred == "first_person" and manifest.videos.first_person is None:
            streaming_fine_preferred = "third_person"
        streaming_fine_views = (
            ["first_person", "third_person"]
            if streaming_fine_scan_both_views and manifest.videos.first_person is not None
            else [str(streaming_fine_preferred or preferred)]
        )
        if streaming_fine_enabled:
            streaming_fine_worker_count = _resolve_yolo_fine_scan_workers(
                max(1, len(scan_tasks)),
                default_workers=4,
                fast_locate_refined=True,
            )
            if streaming_fine_worker_count > 0:
                streaming_fine_executor = ThreadPoolExecutor(
                    max_workers=streaming_fine_worker_count,
                    thread_name_prefix="yolo_streaming_fine_scan",
                )

        def _run_streaming_fine_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
            view = str(task["view"])
            window = dict(task["window"])
            task_timing_rows: list[dict[str, Any]] = []
            model_path = task.get("model_path")
            shared_model = shared_models_by_view.get(view)

            def _scan_timing(event: dict[str, Any]) -> None:
                task_timing_rows.append(
                    {
                        **event,
                        "pipeline_stage": "micro_refine_window_scan",
                        "source_view": view,
                        "scan_role": "streaming_micro_refine",
                        "streaming_fine_scan": True,
                        "coarse_task_index": int(task.get("coarse_task_index") or 0),
                        "coarse_chunk_index": int(task.get("coarse_chunk_index") or 0),
                        "window_index": int(task.get("window_index") or 0),
                        "window_start_sec": float(window["start_sec"]),
                        "window_end_sec": float(window["end_sec"]),
                        "window_duration_sec": float(window.get("duration_sec") or 0.0),
                        "segment_ids": list(window.get("segment_ids") or []),
                    }
                )

            try:
                rows = scan_yolo_video(
                    first_person_path=manifest.videos.first_person,
                    third_person_path=manifest.videos.third_person,
                    preferred_view=view,
                    source_view=view,
                    model=shared_model,
                    model_path=None if shared_model is not None else model_path,
                    model_ref={
                        "path": str(model_path) if model_path else None,
                        "backend": "ultralytics_yolo",
                        "view": view,
                        "scan_role": "micro_refine",
                    },
                    class_schema=class_schema,
                    annotation_asset_refs=annotation_asset_refs,
                    sample_fps=streaming_fine_sample_fps,
                    conf=config.yolo_conf,
                    iou=config.yolo_iou,
                    device=config.yolo_device,
                    imgsz=task.get("imgsz"),
                    adaptive_imgsz=config.yolo_adaptive_imgsz,
                    min_imgsz=config.yolo_min_imgsz,
                    max_imgsz=config.yolo_max_imgsz,
                    class_thresholds=config.yolo_class_thresholds,
                    active_threshold=config.start_threshold,
                    continuity_frames=config.yolo_continuity_frames,
                    dry_run=dry_run,
                    scan_start_sec=float(window["start_sec"]),
                    scan_end_sec=float(window["end_sec"]),
                    timing_callback=_scan_timing,
                )
                return rows, task_timing_rows, None
            except Exception as exc:
                return [], task_timing_rows, f"{view} streaming fine window {task.get('window_index')}: {exc}"

        def _submit_streaming_fine_for_rows(task_rows: list[dict[str, Any]], coarse_task: dict[str, Any]) -> None:
            nonlocal streaming_first_submit_sec, streaming_submit_count
            if streaming_fine_executor is None or not task_rows:
                return
            if _bool_env("KEY_ACTION_FAST_LOCATE_FINE_SEED_REQUIRE_COARSE_SEGMENT", False):
                return
            try:
                normalized_rows = _normalize_yolo_rows_for_pipeline(manifest, task_rows, config)
                normalized_rows, _filter_summary = _filter_rows_to_common_overlap(
                    normalized_rows,
                    common_overlap,
                    row_kind="streaming_fine_seed_rows",
                )
                fine_segments = _fast_locate_fine_window_segments_from_yolo_rows(
                    manifest,
                    [],
                    normalized_rows,
                    config,
                )
            except Exception as exc:
                streaming_fine_errors.append(f"coarse task {coarse_task.get('task_index')} fine seed failed: {exc}")
                return
            if not fine_segments:
                return
            for segment in fine_segments:
                try:
                    session_start = max(0.0, float(segment.start_sec))
                    session_end = max(session_start + 0.1, float(segment.end_sec))
                except (TypeError, ValueError):
                    continue
                for fine_view in streaming_fine_views:
                    local_start = _session_sec_to_view_local_sec(manifest, fine_view, session_start)
                    local_end = max(local_start + 0.1, _session_sec_to_view_local_sec(manifest, fine_view, session_end))
                    key = (fine_view, round(local_start, 2), round(local_end, 2))
                    with streaming_lock:
                        if key in streaming_seen_windows:
                            continue
                        streaming_seen_windows.add(key)
                        window_index = len(streaming_seen_windows)
                    window = {
                        "start_sec": round(local_start, 6),
                        "end_sec": round(local_end, 6),
                        "duration_sec": round(max(0.0, local_end - local_start), 6),
                        "segment_ids": [str(getattr(segment, "segment_id", f"stream_seed_{window_index:06d}"))],
                        "seed_window_start_sec": round(session_start, 6),
                        "seed_window_end_sec": round(session_end, 6),
                        "source_role": "streaming_coarse_yolo_seed_window",
                    }
                    streaming_fine_windows_by_view.setdefault(fine_view, []).append(window)
                    fine_task = {
                        "view": fine_view,
                        "window": window,
                        "window_index": int(window_index),
                        "coarse_task_index": int(coarse_task.get("task_index") or 0),
                        "coarse_chunk_index": int(coarse_task.get("chunk_index") or 0),
                        "model_path": _yolo_model_path_for_view(config, fine_view),
                        "imgsz": _yolo_imgsz_for_view(config, fine_view),
                    }
                    with streaming_lock:
                        if streaming_first_submit_sec is None:
                            streaming_first_submit_sec = time.perf_counter()
                        streaming_submit_count += 1
                        streaming_fine_futures.append(streaming_fine_executor.submit(_run_streaming_fine_task, fine_task))

        def _finish_streaming_fine_scan(coarse_dispatch_wall_sec: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            if streaming_fine_executor is None:
                return [], {
                    "available": False,
                    "enabled": bool(streaming_fine_enabled),
                    "reason": "disabled" if not streaming_fine_enabled else "no_executor",
                }
            for future in as_completed(list(streaming_fine_futures)):
                try:
                    task_rows, task_timings, error = future.result()
                except Exception as exc:
                    streaming_fine_errors.append(f"streaming fine future failed: {exc}")
                    continue
                streaming_fine_rows.extend(task_rows)
                streaming_fine_timing_rows.extend(task_timings)
                if error:
                    streaming_fine_errors.append(error)
            streaming_fine_executor.shutdown(wait=True)
            streaming_wall_sec = time.perf_counter() - streaming_fine_started
            for row in streaming_fine_timing_rows:
                row["stage_parallel_elapsed_sec"] = round(streaming_wall_sec, 6)
                row["stage_parallel_workers"] = int(streaming_fine_worker_count)
                row["stage_scan_task_count"] = int(streaming_submit_count)
                row["streaming_fine_scan"] = True
            refined_streaming_rows = _sort_yolo_rows_by_alignment_time(
                _dedupe_yolo_rows_by_view_time(
                    _normalize_yolo_rows_for_pipeline(manifest, streaming_fine_rows, config)
                )
            )
            if refined_streaming_rows:
                write_jsonl(paths["cv_outputs"] / "yolo_streaming_micro_frame_rows.jsonl", refined_streaming_rows)
            timing_summary = _append_yolo_timing_rows(paths, streaming_fine_timing_rows) if streaming_fine_timing_rows else _read_json_if_exists(paths["metadata"] / "yolo_timing_summary.json")
            first_submit_delay_sec = (
                max(0.0, float(streaming_first_submit_sec - coarse_dispatch_started))
                if streaming_first_submit_sec is not None
                else None
            )
            summary = {
                "schema_version": "key_action_streaming_fine_scan.v1",
                "available": bool(refined_streaming_rows),
                "enabled": bool(streaming_fine_enabled),
                "source": "coarse_chunk_streaming",
                "sample_fps": float(streaming_fine_sample_fps),
                "rows": len(refined_streaming_rows),
                "raw_rows": len(streaming_fine_rows),
                "window_count": int(streaming_submit_count),
                "parallel_workers": int(streaming_fine_worker_count),
                "parallel_enabled": bool(streaming_fine_worker_count > 1),
                "dispatch_wall_sec": round(streaming_wall_sec, 6),
                "coarse_dispatch_wall_sec": round(coarse_dispatch_wall_sec, 6),
                "first_submit_delay_sec": round(first_submit_delay_sec, 6) if first_submit_delay_sec is not None else None,
                "overlapped_with_coarse_scan": bool(
                    first_submit_delay_sec is not None and first_submit_delay_sec < float(coarse_dispatch_wall_sec)
                ),
                "views": list(streaming_fine_views),
                "windows_by_view": streaming_fine_windows_by_view,
                "timing_summary": timing_summary,
                "errors": streaming_fine_errors[:20],
            }
            _write_json(paths["metadata"] / "streaming_fine_scan_summary.json", summary)
            return refined_streaming_rows, summary

        def _run_view_scan(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
            task_index = int(task.get("task_index") or 0)
            view_index = int(task.get("view_index") or 0)
            view = str(task.get("view") or preferred)
            chunk_index = int(task.get("chunk_index") or 0)
            chunk_count = int(task.get("chunk_count") or 1)
            scan_start_sec = task.get("scan_start_sec")
            scan_end_sec = task.get("scan_end_sec")
            task_timing_rows: list[dict[str, Any]] = []
            model_path = Path(model_paths_by_view[view]) if model_paths_by_view.get(view) else None
            shared_model = shared_models_by_view.get(view)
            view_imgsz = _yolo_imgsz_for_view(config, view)
            view_start = 0.32 + (float(task_index) / float(task_total)) * 0.24
            view_end = 0.32 + (float(task_index + 1) / float(task_total)) * 0.24
            _emit_pipeline_progress(
                progress_callback,
                stage="yolo_detection",
                progress=view_start,
                message=f"Scanning {view} video chunk {chunk_index + 1}/{chunk_count} with YOLO",
                source_view=view,
                cache_hit=False,
                coarse_chunk={
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "scan_start_sec": scan_start_sec,
                    "scan_end_sec": scan_end_sec,
                },
            )

            def _scan_progress(event: dict[str, Any], *, _view: str = view, _start: float = view_start, _end: float = view_end) -> None:
                local_progress = float(event.get("progress") or 0.0)
                mapped = _start + max(0.0, min(1.0, local_progress)) * (_end - _start)
                _emit_pipeline_progress(
                    progress_callback,
                    stage="yolo_detection",
                    progress=mapped,
                    message=f"Scanning {_view} video with YOLO",
                    source_view=_view,
                    scan_progress=event,
                )

            def _scan_timing(event: dict[str, Any], *, _view: str = view) -> None:
                task_timing_rows.append(
                    {
                        **event,
                        "pipeline_stage": "coarse_segment_scan",
                        "source_view": _view,
                        "scan_role": "long_video_coarse",
                        "long_video_chunk_sec": float(getattr(config, "long_video_chunk_sec", 0.0) or 0.0),
                        "coarse_scan_chunked": bool(task.get("chunked")),
                        "coarse_scan_chunk_index": chunk_index,
                        "coarse_scan_chunk_count": chunk_count,
                        "coarse_scan_chunk_start_sec": float(task.get("chunk_start_sec") or 0.0),
                        "coarse_scan_chunk_end_sec": float(task.get("chunk_end_sec") or 0.0),
                        "coarse_scan_chunk_sec": float(task.get("chunk_sec") or 0.0),
                        "coarse_scan_overlap_sec": float(task.get("overlap_sec") or 0.0),
                        "coarse_scan_source_duration_sec": float(task.get("source_duration_sec") or 0.0),
                        "planned_scan_backend": coarse_io_plan.get("expected_scan_backend"),
                        "planned_sparse_mode": coarse_io_plan.get("sparse_mode"),
                        "planned_ffmpeg_workers_per_task": coarse_io_plan.get("ffmpeg_workers_per_task"),
                        "planned_batch_size": _yolo_planned_batch_size("long_video_coarse"),
                        "model_load_mode": coarse_model_load_mode if shared_model is not None else "model_path_cache",
                        "coarse_view_worker_mode": bool(_coarse_scan_view_worker_enabled(scan_tasks, coarse_worker_count)),
                    }
                )

            try:
                view_rows = scan_yolo_video(
                    first_person_path=coarse_sources_by_view.get("first_person"),
                    third_person_path=coarse_sources_by_view.get("third_person") or manifest.videos.third_person,
                    preferred_view=view,
                    source_view=view,
                    model=shared_model,
                    model_path=None if shared_model is not None else model_path,
                    model_ref={
                        "path": str(model_path) if model_path else None,
                        "backend": "ultralytics_yolo",
                        "view": view,
                        "scan_role": "long_video_coarse",
                    },
                    class_schema=class_schema,
                    annotation_asset_refs=annotation_asset_refs,
                    sample_fps=coarse_sample_fps,
                    conf=config.yolo_conf,
                    iou=config.yolo_iou,
                    device=config.yolo_device,
                    imgsz=view_imgsz,
                    adaptive_imgsz=config.yolo_adaptive_imgsz,
                    min_imgsz=config.yolo_min_imgsz,
                    max_imgsz=config.yolo_max_imgsz,
                    class_thresholds=config.yolo_class_thresholds,
                    active_threshold=config.start_threshold,
                    continuity_frames=config.yolo_continuity_frames,
                    dry_run=dry_run,
                    progress_callback=_scan_progress,
                    timing_callback=_scan_timing,
                    scan_start_sec=float(scan_start_sec) if scan_start_sec is not None else None,
                    scan_end_sec=float(scan_end_sec) if scan_end_sec is not None else None,
                )
                _emit_pipeline_progress(
                    progress_callback,
                    stage="yolo_detection",
                    progress=view_end,
                    message=f"Completed YOLO scan for {view} chunk {chunk_index + 1}/{chunk_count}",
                    source_view=view,
                    cache_hit=False,
                )
                return view_rows, task_timing_rows, None
            except Exception as exc:
                return [], task_timing_rows, f"{view} chunk {chunk_index + 1}/{chunk_count}: {exc}"

        coarse_dispatch_started = time.perf_counter()
        with _temporary_env(coarse_runtime_env):
            if coarse_worker_count <= 1:
                for task in scan_tasks:
                    task_rows, task_timings, error = _run_view_scan(task)
                    raw_rows.extend(task_rows)
                    timing_rows.extend(task_timings)
                    _submit_streaming_fine_for_rows(task_rows, task)
                    if error:
                        errors.append(error)
            elif coarse_view_worker_enabled:
                view_groups = _coarse_scan_grouped_by_view(scan_tasks)

                def _run_view_group(view: str, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
                    group_rows: list[dict[str, Any]] = []
                    group_timings: list[dict[str, Any]] = []
                    group_errors: list[str] = []
                    for task in tasks:
                        task_rows, task_timings, error = _run_view_scan(task)
                        group_rows.extend(task_rows)
                        group_timings.extend(task_timings)
                        if error:
                            group_errors.append(error)
                    return group_rows, group_timings, group_errors

                with ThreadPoolExecutor(
                    max_workers=min(len(view_groups), int(coarse_worker_count)),
                    thread_name_prefix="yolo_coarse_view_worker",
                ) as executor:
                    futures = [
                        executor.submit(_run_view_group, view, tasks)
                        for view, tasks in view_groups
                    ]
                    for future in as_completed(futures):
                        task_rows, task_timings, task_errors = future.result()
                        raw_rows.extend(task_rows)
                        timing_rows.extend(task_timings)
                        _submit_streaming_fine_for_rows(
                            task_rows,
                            {"task_index": 0, "chunk_index": 0, "view": "view_group"},
                        )
                        errors.extend(task_errors)
            else:
                with ThreadPoolExecutor(max_workers=coarse_worker_count, thread_name_prefix="yolo_coarse_scan") as executor:
                    futures = [executor.submit(_run_view_scan, task) for task in scan_tasks]
                    future_tasks = {future: task for future, task in zip(futures, scan_tasks)}
                    for future in as_completed(futures):
                        task_rows, task_timings, error = future.result()
                        raw_rows.extend(task_rows)
                        timing_rows.extend(task_timings)
                        _submit_streaming_fine_for_rows(task_rows, future_tasks.get(future, {}))
                        if error:
                            errors.append(error)
        coarse_dispatch_wall_sec = time.perf_counter() - coarse_dispatch_started
        streaming_refined_rows, streaming_fine_summary = _finish_streaming_fine_scan(coarse_dispatch_wall_sec)
        for row in timing_rows:
            row["stage_parallel_elapsed_sec"] = round(coarse_dispatch_wall_sec, 6)
            row["stage_parallel_workers"] = int(coarse_worker_count)
            row["stage_scan_task_count"] = int(len(scan_tasks))

        if not raw_rows:
            raise RuntimeError("YOLO scan produced no frame rows" + (": " + "; ".join(errors) if errors else ""))

        raw_rows = _dedupe_yolo_rows_by_view_time(raw_rows)
        view_alignment = _apply_view_alignment_from_yolo(manifest, raw_rows, paths)
        raw_rows = _sort_yolo_rows_by_alignment_time(raw_rows)
        timing_rows = sorted(
            timing_rows,
            key=lambda item: (
                str(item.get("source_view") or ""),
                float(item.get("scan_start_sec") or 0.0),
                float(item.get("scan_end_sec") or 0.0),
            ),
        )
        yolo_rows = _normalize_yolo_rows_for_pipeline(manifest, raw_rows, config)
        yolo_rows, yolo_filter = _filter_rows_to_common_overlap(yolo_rows, common_overlap, row_kind="yolo_rows")
        frame_rows = _frame_score_rows_from_yolo(manifest, yolo_rows, config)
        frame_rows, frame_filter = _filter_rows_to_common_overlap(frame_rows, common_overlap, row_kind="frame_rows")
        common_overlap_filter_summary = {
            "common_overlap": common_overlap,
            "yolo_rows": yolo_filter,
            "frame_rows": frame_filter,
        }
        view_alignment["common_overlap_filter"] = common_overlap_filter_summary
        coarse_timing_summary = _append_yolo_timing_rows(paths, timing_rows)
        _emit_pipeline_progress(
            progress_callback,
            stage="yolo_detection",
            progress=0.58,
            message="YOLO frame rows normalized into parent action scores",
            yolo_row_count=len(yolo_rows),
            frame_score_count=len(frame_rows),
        )
    pseudo_source = VideoSource(
        name="global_multiview",
        path="global_multiview",
        start_time=manifest.session_start_time,
        fps=coarse_sample_fps,
        offset_sec=0.0,
    )
    segment_rows = [
        {
            **row,
            "time_sec": _row_alignment_sec(row),
            "video_start_time": manifest.session_start_time,
            "source_view": "global_multiview",
            "video_path": "global_multiview",
        }
        for row in frame_rows
    ]
    parent_sample_fps = coarse_sample_fps
    duration_sec = max([float(row.get("time_sec", 0.0)) for row in segment_rows], default=0.0) + (1.0 / max(parent_sample_fps, 0.001))
    detected_segments = build_segments_from_yolo_frame_rows(
        segment_rows,
        video_source=pseudo_source,
        duration_sec=duration_sec,
        start_threshold=config.start_threshold,
        end_threshold=config.end_threshold,
        start_min_duration_sec=config.start_min_duration_sec,
        end_min_duration_sec=config.end_min_duration_sec,
        merge_gap_sec=config.merge_gap_sec,
        min_segment_duration_sec=config.min_segment_duration_sec,
        buffer_sec=config.buffer_sec,
    )
    detected_segments = _refine_yolo_detected_segments(manifest, detected_segments, frame_rows, config, duration_sec)
    yolo_source_view = "multiview" if scan_both else str(frame_rows[0].get("source_view") or preferred)
    for segment in detected_segments:
        labels, interaction_count = _segment_yolo_stats(segment, frame_rows)
        raw_score = _segment_avg_score(segment, frame_rows, "motion_score")
        final_score = _segment_avg_score(segment, frame_rows, "interaction_score")
        segment.detector_backend = "yolo_interaction"
        segment.detector_source_view = yolo_source_view
        if not str(segment.start_reason).startswith("yolo_physical_evidence"):
            segment.start_reason = "yolo_active_score_above_threshold"
        if not str(segment.end_reason).startswith("yolo_physical_evidence"):
            segment.end_reason = "yolo_active_score_below_threshold"
        segment.yolo_label_counts = labels
        segment.yolo_interaction_count = interaction_count
        segment.decision_path = DETECTION_DECISION_YOLO_INTERACTION
        segment.decision_trace = [
            f"backend=yolo_interaction",
            f"views={','.join(sorted(set(views)))}",
            f"scan_both_views={scan_both}",
            f"preferred_view={preferred}",
            f"coarse_sample_fps={coarse_sample_fps}",
            f"long_video_two_stage_sampling={bool(getattr(config, 'long_video_two_stage_sampling', True))}",
            f"long_video_stage2_sample_fps={_refined_yolo_sample_fps(config)}",
            f"conf={config.yolo_conf}",
            f"iou={config.yolo_iou}",
            f"device={config.yolo_device}",
            f"imgsz={config.yolo_imgsz or 'adaptive'}",
            f"imgsz_by_view=first_person:{_yolo_imgsz_for_view(config, 'first_person') or 'adaptive'},third_person:{_yolo_imgsz_for_view(config, 'third_person') or 'adaptive'}",
            f"adaptive_imgsz={config.yolo_adaptive_imgsz}",
            f"imgsz_range={config.yolo_min_imgsz}-{config.yolo_max_imgsz}",
            f"continuity_frames={config.yolo_continuity_frames}",
            f"frames={len(frame_rows)}",
            f"fallback=none",
        ]
        segment.fallback_used = False
        segment.fallback_reason = ""
        segment.reason_code = DECISION_REASON_YOLO_INTERACTION_DETECTED
        segment.raw_score = raw_score
        segment.final_score = final_score

    min_segment_interactions = int(
        float(
            os.environ.get(
                "KEY_ACTION_YOLO_MIN_SEGMENT_INTERACTIONS",
                "1" if bool(getattr(config, "long_video_two_stage_sampling", True)) else "0",
            )
        )
    )
    pre_filter_segment_count = len(detected_segments)
    if min_segment_interactions > 0:
        detected_segments = [
            segment
            for segment in detected_segments
            if int(getattr(segment, "yolo_interaction_count", 0) or 0) >= min_segment_interactions
        ]

    label_counts: Counter[str] = Counter()
    interaction_count = 0
    for row in frame_rows:
        label_counts.update({str(key): int(value) for key, value in (row.get("label_counts") or {}).items()})
        interaction_count += len(row.get("hand_object_interactions") or [])
    actual_yolo_devices = sorted(
        {
            str(row.get("actual_yolo_device") or row.get("actual_device") or "").strip()
            for row in yolo_rows
            if str(row.get("actual_yolo_device") or row.get("actual_device") or "").strip()
        }
    )
    requested_yolo_devices = sorted(
        {
            str(row.get("requested_yolo_device") or row.get("requested_device") or "").strip()
            for row in yolo_rows
            if str(row.get("requested_yolo_device") or row.get("requested_device") or "").strip()
        }
    )
    scan_backends = sorted(
        {
            str(row.get("scan_backend") or "").strip()
            for row in timing_rows
            if str(row.get("scan_backend") or "").strip()
        }
    )
    if not cache_hit:
        cache_validation = {
            "valid": True,
            "reason": "fresh_scan" if cache_enabled else "cache_disabled",
            "hash_checked": False,
            "legacy_hash_missing": False,
        }
    summary = {
        "available": True,
        "detector_backend": "yolo_interaction",
        "reason_code": DECISION_REASON_YOLO_INTERACTION_DETECTED,
        "model_path": model_paths_by_view.get(preferred),
        "model_paths_by_view": model_paths_by_view,
        "class_schema_path": class_schema.get("path") if isinstance(class_schema, dict) else None,
        "annotation_asset_count": len(annotation_asset_refs),
        "scan_both_views": scan_both,
        "views": views,
        "parallel_workers": int(coarse_worker_count),
        "parallel_enabled": bool(coarse_worker_count > 1),
        "scan_backends": scan_backends,
        "coarse_scan_plan": coarse_scan_plan,
        "coarse_sample_fps": coarse_sample_fps,
        "long_video_two_stage_sampling": bool(getattr(config, "long_video_two_stage_sampling", True)),
        "long_video_stage2_sample_fps": _refined_yolo_sample_fps(config),
        "long_video_chunk_sec": float(getattr(config, "long_video_chunk_sec", 0.0) or 0.0),
        "frame_rows": len(yolo_rows),
        "frame_score_rows": len(frame_rows),
        "common_overlap_filter": common_overlap_filter_summary,
        "segment_count": len(detected_segments),
        "pre_filter_segment_count": pre_filter_segment_count,
        "min_segment_interactions": min_segment_interactions,
        "filtered_segment_count": max(0, pre_filter_segment_count - len(detected_segments)),
        "boundary_refined_segment_count": sum(
            1 for segment in detected_segments if str(getattr(segment, "boundary_source", "")).startswith("yolo_physical_evidence")
        ),
        "view_alignment": view_alignment,
        "label_counts": dict(label_counts),
        "interaction_count": interaction_count,
        "yolo_imgsz": int(config.yolo_imgsz) if config.yolo_imgsz else None,
        "yolo_imgsz_by_view": {
            view: _yolo_imgsz_for_view(config, view)
            for view in views
        },
        "requested_yolo_devices": requested_yolo_devices or [str(config.yolo_device)],
        "actual_yolo_devices": actual_yolo_devices,
        "yolo_adaptive_imgsz": bool(config.yolo_adaptive_imgsz),
        "yolo_min_imgsz": int(config.yolo_min_imgsz),
        "yolo_max_imgsz": int(config.yolo_max_imgsz),
        "resolution_profiles": _resolution_profiles_from_rows(yolo_rows),
        "analysis_proxy": analysis_proxy_summary,
        "stage_cache": {
            "enabled": bool(cache_enabled),
            "hit": bool(cache_hit),
            "scope": cached_meta.get("cache_scope") if cache_hit and isinstance(cached_meta, dict) else "local",
            "validation": cache_validation,
        },
        "streaming_fine_scan": {
            **streaming_fine_summary,
            "row_count": len(streaming_refined_rows),
            "reuse_enabled": bool(_fast_locate_reuse_streaming_fine_scan_enabled()),
        },
        "timing_summary": coarse_timing_summary,
        "executor_diagnostics": _yolo_executor_diagnostics(
            coarse_timing_summary,
            coarse_yolo_row_count=len(yolo_rows),
        ),
        "errors": errors,
    }
    write_jsonl(paths["cv_outputs"] / "yolo_frame_rows.jsonl", yolo_rows)
    write_jsonl(paths["cv_outputs"] / "frame_scores.jsonl", frame_rows)
    cache_file_hashes = _yolo_stage_cache_file_hashes(yolo_rows_path, frame_rows_path)
    summary["stage_cache"].update(cache_file_hashes)
    _write_json(
        cache_meta_path,
        {
            **cache_signature,
            **cache_file_hashes,
            "cache_enabled": cache_enabled,
            "cache_hit": cache_hit,
            "cache_scope": cached_meta.get("cache_scope") if cache_hit and isinstance(cached_meta, dict) else "local",
            "global_cache_root": cached_meta.get("global_cache_root") if cache_hit and isinstance(cached_meta, dict) else None,
            "updated_at": time.time(),
            "view_alignment": view_alignment,
            "yolo_frame_rows": str(yolo_rows_path),
            "frame_scores": str(frame_rows_path),
            "row_count": len(yolo_rows),
            "frame_score_row_count": len(frame_rows),
            "common_overlap_filter": common_overlap_filter_summary,
            "coarse_scan_plan": coarse_scan_plan,
            "cache_validation": cache_validation,
        },
    )
    if cache_enabled and not cache_hit:
        _publish_global_yolo_stage_cache(
            cache_signature=cache_signature,
            yolo_rows_path=yolo_rows_path,
            frame_rows_path=frame_rows_path,
            cache_meta_path=cache_meta_path,
        )
    _write_json(paths["metadata"] / "yolo_scan_summary.json", summary)
    _write_json(
        paths["metadata"] / "yolo_frame_scan.json",
        {
            "available": True,
            "metadata_version": "key_action_yolo_scan.v1",
            "detection_frame_count": len(yolo_rows),
            "sampled_frame_count": len(yolo_rows),
            "physical_event_count": interaction_count,
            "event_types": ["hand_object_interaction"] if interaction_count else [],
            "frames": yolo_rows[:50],
            "sample_frames": yolo_rows[:50],
            "summary": summary,
        },
    )
    return detected_segments, frame_rows, yolo_rows, summary


def _run_yolo_micro_refine_rescan(
    manifest: SessionManifest,
    paths: dict[str, Path],
    config: DetectorConfig,
    key_segments: list[KeyActionSegment],
    *,
    dry_run: bool,
    default_fine_scan_workers: int = 1,
    fast_locate_refined: bool = False,
    coarse_yolo_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary: dict[str, Any] = {
        "available": False,
        "enabled": bool(config.enable_micro_refine_rescan),
        "source": "coarse_yolo_frame_rows",
    }
    if not config.enable_micro_refine_rescan or not key_segments:
        return [], summary
    backend = str(config.detector_backend or "").lower()
    if backend not in {"yolo", "yolo_interaction", "multiview_yolo"}:
        summary["reason"] = "detector_backend_not_yolo"
        return [], summary
    try:
        from .yolo_detector import _load_yolo_model, scan_yolo_video

        preferred = str(config.yolo_preferred_view or "first_person")
        if preferred == "first_person" and manifest.videos.first_person is None:
            preferred = "third_person"
        scan_both = _bool_from_config(config.yolo_scan_both_views) and manifest.videos.first_person is not None
        requested_views = ["first_person", "third_person"] if scan_both else [preferred]
        full_dual_view_micro_scan = bool(
            fast_locate_refined
            and scan_both
            and _bool_env("KEY_ACTION_FAST_LOCATE_FULL_DUAL_VIEW_MICRO_SCAN", True)
        )
        paired_micro_scan_enabled = bool(
            fast_locate_refined
            and scan_both
            and not full_dual_view_micro_scan
            and _bool_env("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN", False)
        )
        initial_views = [preferred] if paired_micro_scan_enabled else requested_views
        paired_views = [view for view in requested_views if view not in initial_views]
        raw_rows: list[dict[str, Any]] = []
        timing_rows: list[dict[str, Any]] = []
        model_paths_by_view: dict[str, str | None] = {}
        class_schema, annotation_asset_refs = _yolo_inventory_refs(paths)
        refine_sample_fps = _refined_yolo_sample_fps(config)
        window_padding_sec = max(0.5, float(config.buffer_sec or 0.0))
        window_merge_gap_sec = max(0.25, min(2.0, float(config.merge_gap_sec or 0.0)))
        windows_by_view: dict[str, list[dict[str, Any]]] = {}
        scan_tasks: list[dict[str, Any]] = []
        for view in requested_views:
            model_path = _yolo_model_path_for_view(config, view)
            model_paths_by_view[view] = str(model_path) if model_path else None
            if view not in initial_views:
                windows_by_view[view] = []
                continue
            scan_windows = _micro_refine_windows_from_coarse_rows(
                manifest,
                key_segments,
                list(coarse_yolo_rows or []),
                config,
                view,
            )
            if not scan_windows:
                scan_windows = _key_segment_scan_windows(
                    key_segments,
                    view,
                    padding_sec=window_padding_sec,
                    merge_gap_sec=window_merge_gap_sec,
                )
            windows_by_view[view] = scan_windows
            if not scan_windows:
                continue
            for window_index, window in enumerate(scan_windows):
                scan_tasks.append(
                    {
                        "view": view,
                        "window": dict(window),
                        "window_index": int(window_index),
                        "model_path": model_path,
                        "imgsz": _yolo_imgsz_for_view(config, view),
                        "scan_role": "micro_refine",
                        "pipeline_stage": "micro_refine_window_scan",
                        "sample_fps": refine_sample_fps,
                    }
                )

        initial_window_count = len(scan_tasks)
        model_load_mode = _resolve_yolo_fine_scan_model_mode(fast_locate_refined=fast_locate_refined)
        worker_count = _resolve_yolo_fine_scan_workers(
            initial_window_count,
            default_workers=default_fine_scan_workers,
            fast_locate_refined=fast_locate_refined,
        )
        if model_load_mode == "serial":
            worker_count = 1

        shared_models_by_view: dict[str, Any] = {}
        if not dry_run and model_load_mode == "shared":
            for task in scan_tasks:
                view = str(task["view"])
                if view not in shared_models_by_view:
                    shared_models_by_view[view] = _load_yolo_model(None, task.get("model_path"))

        thread_state = threading.local()

        def _model_for_task(task: dict[str, Any]) -> Any | None:
            if dry_run:
                return None
            view = str(task["view"])
            model_path = task.get("model_path")
            if model_load_mode == "shared":
                if view not in shared_models_by_view:
                    shared_models_by_view[view] = _load_yolo_model(None, model_path)
                return shared_models_by_view[view]
            cache = getattr(thread_state, "models_by_key", None)
            if cache is None:
                cache = {}
                thread_state.models_by_key = cache
            key = (view, str(model_path) if model_path else "")
            if key not in cache:
                cache[key] = _load_yolo_model(None, model_path)
            return cache[key]

        def _run_scan_task(task: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            view = str(task["view"])
            window = dict(task["window"])
            window_index = int(task["window_index"])
            model_path = task.get("model_path")
            view_imgsz = task.get("imgsz")
            scan_role = str(task.get("scan_role") or "micro_refine")
            pipeline_stage = str(task.get("pipeline_stage") or "micro_refine_window_scan")
            sample_fps = max(0.001, float(task.get("sample_fps") or refine_sample_fps))
            task_timing_rows: list[dict[str, Any]] = []

            def _scan_timing(event: dict[str, Any]) -> None:
                task_timing_rows.append(
                    {
                        **event,
                        "pipeline_stage": pipeline_stage,
                        "source_view": view,
                        "scan_role": scan_role,
                        "window_index": window_index,
                        "window_start_sec": float(window["start_sec"]),
                        "window_end_sec": float(window["end_sec"]),
                        "window_duration_sec": float(window.get("duration_sec") or 0.0),
                        "segment_ids": list(window.get("segment_ids") or []),
                    }
                )

            rows = scan_yolo_video(
                first_person_path=manifest.videos.first_person,
                third_person_path=manifest.videos.third_person,
                preferred_view=view,
                source_view=view,
                model=_model_for_task(task),
                model_path=model_path,
                model_ref={
                    "path": str(model_path) if model_path else None,
                    "backend": "ultralytics_yolo",
                    "view": view,
                    "scan_role": scan_role,
                },
                class_schema=class_schema,
                annotation_asset_refs=annotation_asset_refs,
                sample_fps=sample_fps,
                conf=config.yolo_conf,
                iou=config.yolo_iou,
                device=config.yolo_device,
                imgsz=int(view_imgsz) if view_imgsz is not None else None,
                adaptive_imgsz=config.yolo_adaptive_imgsz,
                min_imgsz=config.yolo_min_imgsz,
                max_imgsz=config.yolo_max_imgsz,
                class_thresholds=config.yolo_class_thresholds,
                active_threshold=config.start_threshold,
                continuity_frames=config.yolo_continuity_frames,
                dry_run=dry_run,
                scan_start_sec=float(window["start_sec"]),
                scan_end_sec=float(window["end_sec"]),
                timing_callback=_scan_timing,
            )
            return rows, task_timing_rows

        dispatch_started = time.perf_counter()
        if worker_count <= 1:
            for task in scan_tasks:
                task_rows, task_timings = _run_scan_task(task)
                raw_rows.extend(task_rows)
                timing_rows.extend(task_timings)
        elif scan_tasks:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="yolo_fine_scan") as executor:
                futures = [executor.submit(_run_scan_task, task) for task in scan_tasks]
                for future in as_completed(futures):
                    task_rows, task_timings = future.result()
                    raw_rows.extend(task_rows)
                    timing_rows.extend(task_timings)
        initial_dispatch_wall_sec = time.perf_counter() - dispatch_started
        for row in timing_rows:
            row["stage_parallel_elapsed_sec"] = round(initial_dispatch_wall_sec, 6)
            row["stage_parallel_workers"] = int(worker_count)
            row["stage_scan_task_count"] = int(initial_window_count)

        paired_timing_rows: list[dict[str, Any]] = []
        paired_dispatch_wall_sec = 0.0
        paired_scan_tasks: list[dict[str, Any]] = []
        pair_worker_count = 0
        paired_sample_fps = max(
            refine_sample_fps,
            _float_env_value("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN_FPS", 0.5),
        )
        if paired_micro_scan_enabled and paired_views and raw_rows:
            normalized_primary_rows = _normalize_yolo_rows_for_pipeline(manifest, raw_rows, config)
            for paired_view in paired_views:
                paired_model_path = _yolo_model_path_for_view(config, paired_view)
                pair_windows = _micro_pair_scan_windows_from_yolo_rows(
                    manifest,
                    normalized_primary_rows,
                    paired_view,
                    config,
                )
                windows_by_view[paired_view] = pair_windows
                for window_index, window in enumerate(pair_windows):
                    paired_scan_tasks.append(
                        {
                            "view": paired_view,
                            "window": dict(window),
                            "window_index": int(window_index),
                            "model_path": paired_model_path,
                            "imgsz": _yolo_imgsz_for_view(config, paired_view),
                            "scan_role": "paired_micro_refine",
                            "pipeline_stage": "paired_micro_refine_window_scan",
                            "sample_fps": paired_sample_fps,
                        }
                    )
            if paired_scan_tasks:
                pair_worker_count = _resolve_yolo_fine_scan_workers(
                    len(paired_scan_tasks),
                    default_workers=default_fine_scan_workers,
                    fast_locate_refined=fast_locate_refined,
                )
                if model_load_mode == "serial":
                    pair_worker_count = 1
                pair_started = time.perf_counter()
                if pair_worker_count <= 1:
                    for task in paired_scan_tasks:
                        task_rows, task_timings = _run_scan_task(task)
                        raw_rows.extend(task_rows)
                        paired_timing_rows.extend(task_timings)
                else:
                    with ThreadPoolExecutor(max_workers=pair_worker_count, thread_name_prefix="yolo_pair_micro_scan") as executor:
                        futures = [executor.submit(_run_scan_task, task) for task in paired_scan_tasks]
                        for future in as_completed(futures):
                            task_rows, task_timings = future.result()
                            raw_rows.extend(task_rows)
                            paired_timing_rows.extend(task_timings)
                paired_dispatch_wall_sec = time.perf_counter() - pair_started
                for row in paired_timing_rows:
                    row["stage_parallel_elapsed_sec"] = round(paired_dispatch_wall_sec, 6)
                    row["stage_parallel_workers"] = int(pair_worker_count)
                    row["stage_scan_task_count"] = int(len(paired_scan_tasks))
                timing_rows.extend(paired_timing_rows)
        dispatch_wall_sec = initial_dispatch_wall_sec + paired_dispatch_wall_sec
        window_count = initial_window_count + len(paired_scan_tasks)

        raw_rows = _sort_yolo_rows_by_alignment_time(raw_rows)
        timing_rows = sorted(
            timing_rows,
            key=lambda item: (
                float(item.get("window_start_sec", item.get("scan_start_sec", 0.0)) or 0.0),
                str(item.get("source_view") or ""),
                int(item.get("window_index", 0) or 0),
            ),
        )
        for row in timing_rows:
            row.setdefault("stage_parallel_elapsed_sec", round(dispatch_wall_sec, 6))
            row.setdefault("stage_parallel_workers", int(worker_count))
            row.setdefault("stage_scan_task_count", int(window_count))
        refined = _sort_yolo_rows_by_alignment_time(_normalize_yolo_rows_for_pipeline(manifest, raw_rows, config))
        write_jsonl(paths["cv_outputs"] / "yolo_micro_frame_rows.jsonl", refined)
        timing_summary = _append_yolo_timing_rows(paths, timing_rows)
        if timing_summary and timing_rows:
            by_stage = timing_summary.get("by_stage") if isinstance(timing_summary.get("by_stage"), dict) else {}
            fine_stage = by_stage.get("micro_refine_window_scan") if isinstance(by_stage, dict) else None
            if isinstance(fine_stage, dict):
                worker_wall_sec = float(fine_stage.get("worker_wall_sec") or fine_stage.get("wall_sec") or 0.0)
                fine_stage["parallel_enabled"] = bool(worker_count > 1)
                fine_stage["parallel_workers"] = int(worker_count)
                fine_stage["scan_task_count"] = int(initial_window_count)
                fine_stage["model_load_mode"] = model_load_mode
                fine_stage["worker_wall_sec"] = round(worker_wall_sec, 6)
                fine_stage["parallel_elapsed_sec"] = round(initial_dispatch_wall_sec, 6)
                fine_stage["wall_sec"] = round(initial_dispatch_wall_sec, 6)
                old_total = float(timing_summary.get("worker_total_wall_sec") or timing_summary.get("total_wall_sec") or 0.0)
                effective_total = float(timing_summary.get("effective_total_wall_sec") or timing_summary.get("total_wall_sec") or 0.0)
                sampled_total = int(timing_summary.get("total_sampled_frames") or 0)
                timing_summary["worker_total_wall_sec"] = round(old_total, 6)
                timing_summary["total_wall_sec"] = round(effective_total, 6)
                timing_summary["effective_total_wall_sec"] = round(effective_total, 6)
                timing_summary["effective_sampled_fps"] = round(sampled_total / effective_total, 6) if effective_total > 0 else 0.0
                timing_summary["parallel_fine_scan"] = {
                    "enabled": bool(worker_count > 1),
                    "workers": int(worker_count),
                    "scan_task_count": int(initial_window_count),
                    "model_load_mode": model_load_mode,
                    "elapsed_sec": round(initial_dispatch_wall_sec, 6),
                    "worker_wall_sec": round(worker_wall_sec, 6),
                }
            pair_stage = by_stage.get("paired_micro_refine_window_scan") if isinstance(by_stage, dict) else None
            if isinstance(pair_stage, dict):
                pair_worker_wall_sec = float(pair_stage.get("worker_wall_sec") or pair_stage.get("wall_sec") or 0.0)
                pair_stage["parallel_enabled"] = bool(pair_worker_count > 1)
                pair_stage["parallel_workers"] = int(pair_worker_count)
                pair_stage["scan_task_count"] = int(len(paired_scan_tasks))
                pair_stage["model_load_mode"] = model_load_mode
                pair_stage["sample_fps"] = paired_sample_fps
                pair_stage["worker_wall_sec"] = round(pair_worker_wall_sec, 6)
                pair_stage["parallel_elapsed_sec"] = round(paired_dispatch_wall_sec, 6)
                pair_stage["wall_sec"] = round(paired_dispatch_wall_sec, 6)
                timing_summary["parallel_paired_micro_scan"] = {
                    "enabled": bool(pair_worker_count > 1),
                    "workers": int(pair_worker_count),
                    "scan_task_count": int(len(paired_scan_tasks)),
                    "elapsed_sec": round(paired_dispatch_wall_sec, 6),
                    "worker_wall_sec": round(pair_worker_wall_sec, 6),
                    "sample_fps": paired_sample_fps,
                }
            if isinstance(fine_stage, dict) or isinstance(pair_stage, dict):
                _write_json(paths["metadata"] / "yolo_timing_summary.json", timing_summary)
        summary = {
            "available": True,
            "enabled": True,
            "source": "refined_yolo_micro_frame_rows",
            "sample_fps": refine_sample_fps,
            "rows": len(refined),
            "raw_rows": len(raw_rows),
            "model_path": model_paths_by_view.get(preferred),
            "model_paths_by_view": model_paths_by_view,
            "class_schema_path": class_schema.get("path") if isinstance(class_schema, dict) else None,
            "annotation_asset_count": len(annotation_asset_refs),
            "preferred_view": preferred,
            "views": requested_views,
            "initial_views": initial_views,
            "full_dual_view_micro_scan": {
                "enabled": full_dual_view_micro_scan,
                "views": requested_views if full_dual_view_micro_scan else [],
            },
            "paired_micro_scan": {
                "enabled": paired_micro_scan_enabled,
                "paired_views": paired_views,
                "sample_fps": paired_sample_fps,
                "scan_task_count": len(paired_scan_tasks),
                "dispatch_wall_sec": round(paired_dispatch_wall_sec, 6),
            },
            "window_padding_sec": window_padding_sec,
            "window_merge_gap_sec": window_merge_gap_sec,
            "window_count": window_count,
            "initial_window_count": initial_window_count,
            "paired_window_count": len(paired_scan_tasks),
            "parallel_enabled": worker_count > 1,
            "parallel_workers": worker_count,
            "default_parallel_workers": max(1, int(default_fine_scan_workers or 1)),
            "model_load_mode": model_load_mode,
            "dispatch_wall_sec": round(dispatch_wall_sec, 6),
            "fast_locate_refined": bool(fast_locate_refined),
            "window_scanned_duration_sec": round(
                sum(float(window.get("duration_sec") or 0.0) for windows in windows_by_view.values() for window in windows),
                6,
            ),
            "windows_by_view": windows_by_view,
            "timing_summary": timing_summary,
            "yolo_imgsz": int(config.yolo_imgsz) if config.yolo_imgsz else None,
            "yolo_imgsz_by_view": {
                view: _yolo_imgsz_for_view(config, view)
                for view in requested_views
            },
            "yolo_adaptive_imgsz": bool(config.yolo_adaptive_imgsz),
            "yolo_min_imgsz": int(config.yolo_min_imgsz),
            "yolo_max_imgsz": int(config.yolo_max_imgsz),
            "resolution_profiles": _resolution_profiles_from_rows(refined),
        }
        _write_json(paths["metadata"] / "yolo_micro_scan_summary.json", summary)
        return refined, summary
    except Exception as exc:
        summary["error"] = str(exc)
        summary["fallback"] = "coarse_yolo_frame_rows"
        _write_json(paths["metadata"] / "yolo_micro_scan_summary.json", summary)
        print(f"[pipeline] YOLO micro refine rescan failed; using coarse rows: {exc}")
        return [], summary


def _detect_with_config(
    manifest: SessionManifest,
    paths: dict[str, Path],
    config: DetectorConfig,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[Any], list[Any], list[dict[str, Any]], dict[str, Any]]:
    backend = str(config.detector_backend or "motion").lower()
    yolo_error: str | None = None
    did_fallback = False
    if backend in {"yolo", "yolo_interaction", "multiview_yolo"}:
        try:
            return _run_yolo_segment_detection(
                manifest,
                paths,
                config,
                dry_run=dry_run,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            yolo_error = str(exc)
            fast_locate_runtime = _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False) or _bool_env(
                "KEY_ACTION_DEFER_SEGMENT_ASSETS",
                False,
            )
            allow_motion_fallback = _bool_from_config(config.yolo_fallback_to_motion) and not fast_locate_runtime
            _write_json(
                paths["metadata"] / "yolo_scan_summary.json",
                {
                    "available": False,
                    "error": str(exc),
                    "fallback": "motion" if allow_motion_fallback else "disabled_fast_locate",
                    "reason_code": DECISION_REASON_MOTION_FALLBACK_AFTER_YOLO_FAILURE,
                },
            )
            _write_json(paths["metadata"] / "yolo_frame_scan.json", {"available": False, "error": str(exc), "frames": [], "sample_frames": []})
            if not allow_motion_fallback:
                raise
            print(f"[pipeline] YOLO detector failed; falling back to motion baseline: {exc}")

    _emit_pipeline_progress(
        progress_callback,
        stage="motion_detection",
        progress=0.42,
        message="Running motion fallback detector",
        fallback_reason=yolo_error or "",
    )
    detected_segments, scores = detect_key_action_segments(
        manifest.videos.third_person,
        roi=manifest.workbench_roi,
        config=config,
        dry_run=dry_run,
        frame_scores_output_path=paths["cv_outputs"] / "frame_scores.jsonl",
    )
    for segment in detected_segments:
        segment.detector_backend = "motion"
        segment.detector_source_view = "third_person"
        segment.fallback_used = False
        segment.fallback_reason = ""
        if yolo_error:
            did_fallback = True
            segment.decision_path = DETECTION_DECISION_YOLO_FALLBACK_MOTION
            segment.decision_trace = [
                f"detector_requested={backend}",
                "fallback_path=motion_baseline",
                f"yolo_fallback_to_motion={_bool_from_config(config.yolo_fallback_to_motion)}",
                f"yolo_error={yolo_error}",
            ]
            segment.fallback_reason = yolo_error
            segment.reason_code = DECISION_REASON_MOTION_FALLBACK_AFTER_YOLO_FAILURE
            segment.fallback_used = True
        else:
            segment.decision_path = DETECTION_DECISION_MOTION_BASELINE
            segment.decision_trace = [
                f"detector_requested={backend}",
                "fallback_not_triggered=True",
            ]
            segment.reason_code = DECISION_REASON_MOTION_BASELINE
        if segment.raw_score <= 0.0:
            segment.raw_score = float(segment.avg_motion_score)
        if segment.final_score <= 0.0:
            segment.final_score = float(segment.avg_active_score)
    return (
        detected_segments,
        scores,
        [],
        {
            "available": True,
            "detector_backend": "motion",
            "segment_count": len(detected_segments),
            "fallback_used": bool(did_fallback),
            "fallback_reason": yolo_error or "",
            "reason_code": DECISION_REASON_MOTION_FALLBACK_AFTER_YOLO_FAILURE if did_fallback else DECISION_REASON_MOTION_BASELINE,
        },
    )


def _mock_yolo_micro_rows_for_dry_run(key_segments: list[KeyActionSegment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in key_segments:
        action_type = segment.text_description.action_type
        if action_type == "weighing":
            obj = "balance"
        elif action_type == "pipetting":
            obj = "pipette"
        elif "spatula" in action_type:
            obj = "spatula"
        else:
            obj = "sample_bottle"
        view = "first_person" if segment.first_person else "third_person"
        ref = segment.first_person or segment.third_person
        start = float(ref.local_start_sec) + 0.5
        end = min(float(ref.local_end_sec), start + 1.2)
        for local_time_sec in (start, end):
            hand_bbox = [10, 10, 60, 60]
            object_bbox = [40, 40, 100, 100]
            rows.append(
                {
                    "source_view": view,
                    "frame_index": int(local_time_sec * 30),
                    "local_time_sec": local_time_sec,
                    "detections": [
                        {"label": "hand", "confidence": 0.9, "bbox": hand_bbox},
                        {"label": obj, "confidence": 0.86, "bbox": object_bbox},
                    ],
                    "hand_object_interactions": [
                        {
                            "hand_label": "hand",
                            "object_label": obj,
                            "score": 0.86,
                            "distance_px": 8,
                            "iou": 0.2,
                            "hand_bbox": hand_bbox,
                            "object_bbox": object_bbox,
                        }
                    ],
                }
            )
    return rows


def _build_error_diagnostics(
    summary: dict[str, Any],
    quality_assurance: dict[str, Any],
    artifact_validation: dict[str, Any],
) -> dict[str, Any]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    missing_artifacts = []
    for name, value in artifacts.items():
        if not value or name.endswith("_dir") or name == "index_dir":
            continue
        path = Path(str(value))
        if not path.exists():
            missing_artifacts.append({"artifact": name, "path": str(path)})
    qa_diagnostics = quality_assurance.get("diagnostics") if isinstance(quality_assurance.get("diagnostics"), dict) else {}
    detector_errors = []
    detector_summary = summary.get("detector_summary") if isinstance(summary.get("detector_summary"), dict) else {}
    if detector_summary.get("errors"):
        detector_errors.extend(str(item) for item in detector_summary.get("errors") or [])
    if detector_summary.get("error"):
        detector_errors.append(str(detector_summary["error"]))
    return {
        "missing_artifacts": missing_artifacts,
        "detector_errors": detector_errors,
        "qa_failed_check_ids": qa_diagnostics.get("failed_check_ids", []),
        "qa_needs_review_check_ids": qa_diagnostics.get("needs_review_check_ids", []),
        "schema_valid": artifact_validation.get("valid"),
        "schema_error_count": artifact_validation.get("error_count", 0),
        "top_recommendations": qa_diagnostics.get("top_recommendations", []),
    }


def _module_runs_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    diagnostics = summary.get("error_diagnostics") if isinstance(summary.get("error_diagnostics"), dict) else {}
    missing_by_path = {str(item.get("path")) for item in diagnostics.get("missing_artifacts", []) if isinstance(item, dict)}
    modules = {
        "manifest_validation": ["video_info"],
        "long_video_planning": ["long_video_processing_plan", "long_video_checkpoint"],
        "detection": ["detected_segments", "frame_scores"],
        "clip_and_keyframe_extraction": ["key_action_segments", "micro_segments"],
        "vector_index": ["vector_metadata", "index_dir"],
        "timeline": ["unified_multimodal_timeline", "time_calibration_report"],
        "assets": ["material_asset_catalog", "material_library_summary"],
        "model_observations": ["model_observation_events", "advanced_vision_evidence"],
        "understanding": ["video_understanding", "experiment_context", "sop_state_machine", "experiment_process"],
        "quality": ["process_quality_report", "artifact_validation_report", "pipeline_evaluation_report"],
        "governance": ["data_governance_report"],
    }
    rows: list[dict[str, Any]] = []
    for module_name, artifact_keys in modules.items():
        output_paths = [str(artifacts.get(key)) for key in artifact_keys if artifacts.get(key)]
        missing = [path for path in output_paths if path in missing_by_path]
        rows.append(
            {
                "module": module_name,
                "status": "fail" if missing else "pass",
                "duration_sec": 0.0,
                "inputs": [],
                "outputs": output_paths,
                "errors": [f"missing output: {path}" for path in missing],
                "rerunnable": module_name in {"timeline", "assets", "model_observations", "understanding", "quality", "governance"},
            }
        )
    return rows


def _build_pipeline_artifacts(
    paths: dict[str, Path],
    output_dir: Path,
    *,
    input_ingestion_summary: Mapping[str, Any],
    report_path: Path,
    formal_report_path: Path,
    yolo_rows_path: str | None,
    micro_source_path: str | None,
) -> dict[str, Any]:
    input_artifacts = input_ingestion_summary["artifacts"]
    return {
        "detected_segments": str(paths["cv_outputs"] / "detected_segments.jsonl"),
        "experiment_episodes": str(paths["metadata"] / "experiment_episodes.jsonl"),
        "frame_scores": str(paths["cv_outputs"] / "frame_scores.jsonl"),
        "video_sources": str(paths["metadata"] / "video_sources.jsonl"),
        "user_text_events": input_artifacts["user_text"],
        "ai_reply_events": input_artifacts["ai_reply"],
        "upload_events": input_artifacts["upload"],
        "input_ingestion_summary": str(paths["metadata"] / "input_ingestion_summary.json"),
        "session_context_events": str(paths["metadata"] / "session_context_events.jsonl"),
        "session_context_seed_summary": str(paths["metadata"] / "session_context_seed_summary.json"),
        "sop_records": str(paths["metadata"] / "sop_records.jsonl"),
        "database_records": str(paths["metadata"] / "database_records.jsonl"),
        "record_ingestion_summary": str(paths["metadata"] / "record_ingestion_summary.json"),
        "history_model": str(paths["metadata"] / "history_model.json"),
        "long_video_processing_plan": str(paths["metadata"] / "long_video_processing_plan.json"),
        "long_video_checkpoint": str(paths["metadata"] / "long_video_checkpoint.json"),
        "multimodal_alignment": str(paths["metadata"] / "multimodal_alignment.jsonl"),
        "key_action_segments": str(paths["metadata"] / "key_action_segments.jsonl"),
        "vector_metadata": str(paths["metadata"] / "vector_metadata.jsonl"),
        "micro_segments": str(paths["metadata"] / "micro_segments.jsonl"),
        "micro_vector_metadata": str(paths["metadata"] / "micro_vector_metadata.jsonl"),
        "view_action_evidence": str(paths["metadata"] / "view_action_evidence.jsonl"),
        "dual_view_action_events": str(paths["metadata"] / "dual_view_action_events.jsonl"),
        "unmatched_view_evidence": str(paths["metadata"] / "unmatched_view_evidence.jsonl"),
        "dual_view_action_alignment_summary": str(paths["metadata"] / "dual_view_action_alignment_summary.json"),
        "experiment_focus_window": str(paths["metadata"] / "experiment_focus_window.json"),
        "experiment_focus_clips": str(paths["metadata"] / "experiment_focus_clips.json"),
        "unified_multimodal_timeline": str(paths["metadata"] / "unified_multimodal_timeline.jsonl"),
        "time_calibration_report": str(paths["metadata"] / "time_calibration_report.json"),
        "time_anchors": str(paths["metadata"] / "time_anchors.jsonl"),
        "state_change_index": str(paths["metadata"] / "state_change_index.jsonl"),
        "material_asset_catalog": str(paths["metadata"] / "material_asset_catalog.jsonl"),
        "material_library_summary": str(paths["metadata"] / "material_library_summary.json"),
        "advanced_vision_evidence": str(paths["metadata"] / "advanced_vision_evidence.jsonl"),
        "advanced_vision_evidence_summary": str(paths["metadata"] / "advanced_vision_evidence_summary.json"),
        "object_tracks": str(paths["metadata"] / "object_tracks.jsonl"),
        "yolo_observation_inputs_summary": str(paths["metadata"] / "yolo_observation_inputs_summary.json"),
        "lab_model_signal_inputs_summary": str(paths["metadata"] / "lab_model_signal_inputs_summary.json"),
        "model_observation_events": str(paths["metadata"] / "model_observation_events.jsonl"),
        "model_observation_events_summary": str(paths["metadata"] / "model_observation_events_summary.json"),
        "model_inventory": str(paths["metadata"] / "model_inventory.json"),
        "capability_gap_report": str(paths["metadata"] / "capability_gap_report.json"),
        "video_understanding": str(paths["metadata"] / "video_understanding.jsonl"),
        "video_understanding_summary": str(paths["metadata"] / "video_understanding_summary.json"),
        "experiment_context": str(paths["metadata"] / "experiment_context.json"),
        "sop_state_machine": str(paths["metadata"] / "sop_state_machine.json"),
        "experiment_process": str(paths["metadata"] / "experiment_process.json"),
        "experiment_process_timeline": str(paths["metadata"] / "experiment_process_timeline.jsonl"),
        "human_confirmation_queue": str(paths["metadata"] / "human_confirmation_queue.jsonl"),
        "process_quality_report": str(paths["metadata"] / "process_quality_report.json"),
        "process_record": str(paths["exports"] / "process_record.json"),
        "process_audit_report": str(paths["reports"] / "process_audit_report.md"),
        "artifact_validation_report": str(paths["metadata"] / "artifact_validation_report.json"),
        "pipeline_evaluation_report": str(paths["evaluation"] / "pipeline_evaluation_report.json"),
        "data_governance_report": str(paths["metadata"] / "data_governance_report.json"),
        "index_dir": str(paths["index"]),
        "video_info": str(output_dir / "video_info.json"),
        "debug_report": str(report_path),
        "formal_report": str(formal_report_path),
        "roi_preview": str(paths["debug"] / "roi_preview.jpg"),
        "frame_score_plot": str(paths["debug"] / "frame_scores.png"),
        "segments_contact_sheet": str(paths["debug"] / "segments_contact_sheet.jpg"),
        "yolo_frame_rows": yolo_rows_path,
        "yolo_micro_frame_rows": micro_source_path,
    }


def _build_run_manifest(
    manifest: SessionManifest,
    active_detector_config: DetectorConfig,
    run_ctx: RunContext,
    *,
    dry_run: bool,
    alignment_health: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "run_manifest.v1",
        "run_id": run_ctx.run_id,
        "session_id": manifest.session_id,
        "dry_run": dry_run,
        "model_versions": {
            "detector_backend": str(getattr(active_detector_config, "detector_backend", "motion")),
            "yolo_model_path": str(getattr(active_detector_config, "yolo_model_path", None) or ""),
            "yolo_conf": float(getattr(active_detector_config, "yolo_conf", 0.25)),
            "yolo_iou": float(getattr(active_detector_config, "yolo_iou", 0.45)),
        },
        "parameters": {
            "sample_fps": float(getattr(active_detector_config, "sample_fps", 2.0)),
            "start_threshold": float(getattr(active_detector_config, "start_threshold", 0.6)),
            "end_threshold": float(getattr(active_detector_config, "end_threshold", 0.3)),
            "merge_gap_sec": float(getattr(active_detector_config, "merge_gap_sec", 5.0)),
            "min_segment_duration_sec": float(getattr(active_detector_config, "min_segment_duration_sec", 5.0)),
            "buffer_sec": float(getattr(active_detector_config, "buffer_sec", 2.0)),
        },
        "timing": run_ctx.stage_stats(),
        "alignment_health": alignment_health,
        "failure_nodes": [s for s in run_ctx.stages if s.get("errors", 0) > 0],
    }


def _build_long_video_plan_summary(long_video_plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "chunk_count": long_video_plan.get("chunk_count"),
        "pending_chunk_count": long_video_plan.get("pending_chunk_count"),
        "completed_chunk_count": long_video_plan.get("completed_chunk_count"),
        "cache_enabled": long_video_plan.get("cache_enabled"),
        "resume_enabled": long_video_plan.get("resume_enabled"),
        "two_stage_sampling": long_video_plan.get("two_stage_sampling"),
        "chunk_sec": long_video_plan.get("chunk_sec"),
    }


def _build_history_model_summary(history_model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "session_count": history_model.get("session_count", 0),
        "event_count": history_model.get("event_count", 0),
        "action_counts": history_model.get("action_counts", {}),
    }


def _build_model_inventory_summary(model_inventory: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary_model": model_inventory.get("primary_model", {}),
        "model_count": model_inventory.get("model_count", 0),
        "dataset_count": model_inventory.get("dataset_count", 0),
        "capabilities": model_inventory.get("capabilities", {}),
    }


def _build_experiment_context_summary(experiment_context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "confidence": experiment_context.get("confidence"),
        "purpose": experiment_context.get("purpose"),
        "procedure_count": len(experiment_context.get("procedure_candidates") or []),
        "material_count": len(experiment_context.get("materials") or []),
        "parameter_count": len(experiment_context.get("parameters") or []),
        "gaps": experiment_context.get("gaps", []),
    }


def _build_experiment_process_summary(experiment_process: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "process_status": experiment_process.get("process_status"),
        "step_count": experiment_process.get("step_count"),
        "status_counts": experiment_process.get("status_counts", {}),
        "current_step_id": experiment_process.get("current_step_id"),
        "next_step_id": experiment_process.get("next_step_id"),
    }


def _build_quality_assurance_summary(quality_assurance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall_status": quality_assurance.get("overall_status"),
        "overall_score": quality_assurance.get("overall_score"),
        "status_counts": quality_assurance.get("status_counts", {}),
        "scorecard": quality_assurance.get("scorecard", {}),
        "diagnostics": quality_assurance.get("diagnostics", {}),
    }


def _build_artifact_validation_summary(artifact_validation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "valid": artifact_validation.get("valid"),
        "artifact_count": artifact_validation.get("artifact_count"),
        "error_count": artifact_validation.get("error_count"),
        "issue_count": artifact_validation.get("issue_count"),
    }


def _build_pipeline_evaluation_summary(pipeline_evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall_score": pipeline_evaluation.get("overall_score"),
        "scores": pipeline_evaluation.get("scores", {}),
    }


def _build_data_governance_summary(data_governance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "record_count": data_governance.get("record_count"),
        "missing_hash_count": data_governance.get("missing_hash_count"),
        "missing_governance_field_count": data_governance.get("missing_governance_field_count"),
        "privacy_level_counts": data_governance.get("privacy_level_counts", {}),
    }


def _compute_alignment_health(
    manifest: SessionManifest,
    paths: dict[str, Path],
    alignment_options: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    alignment_opts = alignment_options or {}
    manifest_alignment_cfg = getattr(manifest, "config", {}) or {}
    time_alignment_cfg = manifest_alignment_cfg.get("time_alignment") if isinstance(manifest_alignment_cfg, dict) else {}
    time_alignment_cfg = time_alignment_cfg if isinstance(time_alignment_cfg, dict) else {}
    align_window = int(alignment_opts.get("alignment_window", time_alignment_cfg.get("drift_window_size", 5)))
    align_threshold = float(alignment_opts.get("alignment_alert_threshold", time_alignment_cfg.get("drift_alert_threshold_sec", 1.5)))
    align_alpha = float(alignment_opts.get("alignment_smoothing", time_alignment_cfg.get("drift_smoothing_alpha", 0.3)))
    _align_degradation = float(time_alignment_cfg.get("alignment_degradation_factor", 0.85))
    time_anchors_path = paths["metadata"] / "time_anchors.jsonl"
    if time_anchors_path.exists():
        alignment_eval = evaluate_time_alignment(
            time_anchors_path,
            output_path=alignment_opts.get("alignment_report_path") or paths["metadata"] / "alignment_report.json",
        )
        offset_history = alignment_eval.get("offset_history", [])
        drift_result = estimate_sliding_window_drift(
            offset_history,
            window_size=align_window,
            alert_threshold_sec=align_threshold,
            smoothing_alpha=align_alpha,
        )
        metrics = alignment_eval.get("metrics", {}) if isinstance(alignment_eval, dict) else {}
        alert_reasons = alignment_eval.get("alignment_alert_reason", []) if isinstance(alignment_eval, dict) else []
        drift_status = drift_result["summary"]["status"]
        alignment_quality = {
            "schema_version": "key_action_time_alignment_quality.v1",
            "status": "warning" if alignment_eval.get("alignment_alert") or drift_status == "drift_alert" else "pass",
            "anchor_coverage_rate": metrics.get("anchor_coverage_rate"),
            "mae_sec": metrics.get("mae_sec"),
            "max_residual_sec": metrics.get("max_residual_sec"),
            "jitter_sec": metrics.get("jitter_sec"),
            "drift_error_per_min": metrics.get("drift_error_per_min"),
            "drift_status": drift_status,
            "alert_reasons": alert_reasons,
        }
        alignment_health = {
            "mean_offset_ms": drift_result["summary"]["mean_offset_ms"],
            "jitter_ms": drift_result["summary"]["jitter_ms"],
            "drift_events": drift_result["summary"]["drift_events"],
            "max_drift_sec": drift_result["summary"]["max_drift_sec"],
            "status": drift_status,
            "time_alignment_quality": alignment_quality,
        }
        _write_json(paths["metadata"] / "alignment_health.json", drift_result)
        return alignment_health, drift_result, _align_degradation
    return {
        "status": "no_anchors",
        "mean_offset_ms": 0.0,
        "jitter_ms": 0.0,
        "drift_events": 0,
        "time_alignment_quality": {
            "schema_version": "key_action_time_alignment_quality.v1",
            "status": "unknown",
            "reason": "no_time_anchors",
        },
    }, {}, _align_degradation


def _finalize_rebuilt_episode_outputs(
    paths: dict[str, Path],
    *,
    run_id: str,
    detector_summary: dict[str, Any],
    alignment_health: dict[str, Any],
    drift_result: dict[str, Any],
    alignment_degradation: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decision_reason = detector_summary.get("reason_code", "") if isinstance(detector_summary, dict) else ""
    alignment_payload = alignment_health if alignment_health.get("status") != "no_anchors" else None

    key_segment_rows_for_index = read_jsonl(paths["metadata"] / "key_action_segments.jsonl")
    for row in key_segment_rows_for_index:
        if isinstance(row, dict):
            if not row.get("run_manifest_id"):
                row["run_manifest_id"] = run_id
            if not row.get("decision_path"):
                row["decision_path"] = decision_reason
            if not row.get("decision_trace"):
                row["decision_trace"] = []
            if not row.get("reason_code"):
                row["reason_code"] = decision_reason
            if not row.get("alignment_health"):
                row["alignment_health"] = alignment_payload
    if alignment_health.get("status") == "drift_alert":
        key_segment_rows_for_index = apply_alignment_correction(
            key_segment_rows_for_index,
            drift_result,
            degradation_factor=alignment_degradation,
        )
    write_jsonl(paths["metadata"] / "key_action_segments.jsonl", key_segment_rows_for_index)

    micro_rows = read_jsonl(paths["metadata"] / "micro_segments.jsonl")
    for row in micro_rows:
        if isinstance(row, dict) and not row.get("run_manifest_id"):
            row["run_manifest_id"] = run_id
    write_jsonl(paths["metadata"] / "micro_segments.jsonl", micro_rows)

    micro_vector_metadata = read_jsonl(paths["metadata"] / "micro_vector_metadata.jsonl")
    for row in micro_vector_metadata:
        if isinstance(row, dict) and not row.get("run_manifest_id"):
            row["run_manifest_id"] = run_id
    write_jsonl(paths["metadata"] / "micro_vector_metadata.jsonl", micro_vector_metadata)

    vector_metadata = [
        row
        for row in read_jsonl(paths["metadata"] / "vector_metadata.jsonl")
        if str(row.get("index_level") or "segment") == "segment"
    ]
    for row in vector_metadata:
        if isinstance(row, dict) and not row.get("run_manifest_id"):
            row["run_manifest_id"] = run_id
    return key_segment_rows_for_index, micro_rows, micro_vector_metadata, vector_metadata


def _build_key_segments_and_vector_metadata(
    manifest: SessionManifest,
    paths: dict[str, Path],
    detected_segments: list[Any],
    utterances: list[dict[str, Any]],
    yolo_frame_rows: list[dict[str, Any]],
    alignment_health: dict[str, Any],
    run_id: str,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[KeyActionSegment], list[VectorMetadata]]:
    key_segments: list[KeyActionSegment] = []
    vector_metadata: list[VectorMetadata] = []
    print("[pipeline] Extracting multiview clips and keyframes")
    _emit_pipeline_progress(progress_callback, stage="clips", progress=0.62, message="Extracting multiview clips and keyframes")
    for segment in detected_segments:
        key_segment = extract_multiview_clips(
            manifest=manifest,
            segment=segment,
            clips_dir=paths["clips"],
            keyframes_dir=paths["keyframes"],
            yolo_frame_rows=yolo_frame_rows,
            dry_run=dry_run,
        )
        dialogue = find_dialogue_for_segment(segment.global_start_time, segment.global_end_time, utterances)
        key_segment = build_segment_description(key_segment, dialogue)
        key_segment = apply_segment_evidence(key_segment)
        key_segment.detector_backend = str(getattr(segment, "detector_backend", "motion"))
        key_segment.detector_source_view = str(getattr(segment, "detector_source_view", "third_person"))
        key_segment.decision_path = str(getattr(segment, "decision_path", ""))
        key_segment.decision_trace = list(getattr(segment, "decision_trace", []))
        key_segment.fallback_used = bool(getattr(segment, "fallback_used", False))
        key_segment.fallback_reason = str(getattr(segment, "fallback_reason", ""))
        key_segment.reason_code = str(getattr(segment, "reason_code", ""))
        key_segment.raw_score = float(getattr(segment, "raw_score", 0.0) or 0.0)
        key_segment.final_score = float(getattr(segment, "final_score", 0.0) or 0.0)
        key_segment.run_manifest_id = run_id
        key_segment.alignment_health = alignment_health if alignment_health.get("status") != "no_anchors" else None
        key_segments.append(key_segment)
        vector_metadata.append(_vector_metadata_from_segment(key_segment))
    return key_segments, vector_metadata


def _run_detection_alignment_stage(
    manifest: SessionManifest,
    paths: dict[str, Path],
    detector_config: DetectorConfig,
    utterances: list[TranscriptUtterance],
    run_ctx: RunContext,
    *,
    dry_run: bool,
    alignment_options: Mapping[str, Any] | None,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> PipelineDetectionOutputs:
    print("[pipeline] Preparing pre-coarse timeline alignment")
    _emit_pipeline_progress(progress_callback, stage="time_alignment_preflight", progress=0.24, message="Aligning dual-view timeline before coarse scan")
    run_ctx.begin_stage("time_alignment_preflight")
    pre_coarse_alignment = _ensure_pre_coarse_timeline_alignment(manifest, paths, dry_run=dry_run)
    run_ctx.end_stage(inputs=len(manifest.videos.all_sources()), outputs=1)

    print("[pipeline] Detecting key action segments and writing frame_scores.jsonl")
    _emit_pipeline_progress(progress_callback, stage="detection", progress=0.30, message="Detecting key action segments")
    run_ctx.begin_stage("detection")
    detected_segments, _scores, generated_yolo_rows, detector_summary = _detect_with_config(
        manifest,
        paths,
        detector_config,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )
    if isinstance(detector_summary, dict):
        detector_summary["pre_coarse_timeline_alignment"] = pre_coarse_alignment
    write_jsonl(paths["cv_outputs"] / "detected_segments.jsonl", detected_segments)
    run_ctx.end_stage(inputs=1, outputs=len(detected_segments))
    _emit_pipeline_progress(
        progress_callback,
        stage="detection",
        progress=0.60,
        message="Key-action detection finished",
        segment_count=len(detected_segments),
    )
    experiment_episode_rows = _experiment_episode_rows(
        manifest,
        detected_segments,
        detector_summary=detector_summary if isinstance(detector_summary, dict) else None,
    )
    write_jsonl(paths["metadata"] / "experiment_episodes.jsonl", experiment_episode_rows)

    print("[pipeline] Plotting frame score debug chart")
    save_frame_score_plot(
        paths["cv_outputs"] / "frame_scores.jsonl",
        paths["cv_outputs"] / "detected_segments.jsonl",
        paths["debug"] / "frame_scores.png",
    )

    print("[pipeline] Generating multimodal alignment")
    generate_multimodal_alignment(
        manifest=manifest,
        segments=detected_segments,
        utterances=utterances,
        output_path=paths["metadata"] / "multimodal_alignment.jsonl",
    )

    print("[pipeline] Computing alignment health")
    alignment_health, drift_result, alignment_degradation = _compute_alignment_health(manifest, paths, alignment_options)
    return PipelineDetectionOutputs(
        detected_segments=detected_segments,
        generated_yolo_rows=generated_yolo_rows,
        detector_summary=detector_summary,
        experiment_episode_rows=experiment_episode_rows,
        alignment_health=alignment_health,
        drift_result=drift_result,
        alignment_degradation=alignment_degradation,
    )


def _generate_and_write_micro_segments(
    manifest: SessionManifest,
    paths: dict[str, Path],
    key_segments: list[KeyActionSegment],
    micro_source_rows: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    run_ctx: RunContext,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[VectorMetadata]]:
    print("[pipeline] Generating YOLO interaction micro-segments")
    _emit_pipeline_progress(progress_callback, stage="micro_segments", progress=0.68, message="Generating YOLO-backed micro evidence")
    run_ctx.begin_stage("micro_segmentation")
    raw_micro_segments, micro_dedup_log = generate_micro_segments(
        manifest=manifest,
        key_segments=key_segments,
        yolo_frame_rows=micro_source_rows,
        utterances=utterances,
        clips_dir=paths["clips"],
        keyframes_dir=paths["keyframes"],
        config=manifest.micro_segment_config,
        dry_run=dry_run,
    )
    write_jsonl(paths["metadata"] / "micro_dedup_log.jsonl", micro_dedup_log)
    raw_micro_rows = [to_json_dict(item) for item in raw_micro_segments]
    micro_rows, micro_merge_stats = merge_same_object_adjacent_micro_segments(
        raw_micro_rows,
        config=manifest.micro_segment_config,
    )
    _attach_micro_refs_to_parents(key_segments, micro_rows)
    for key_segment in key_segments:
        refresh_segment_chinese_index(key_segment)
    micro_rows = [refresh_micro_row_chinese_index(item) for item in micro_rows]
    vector_metadata = [_vector_metadata_from_segment(item) for item in key_segments]
    micro_vector_metadata = [micro_row_to_vector_metadata(item) for item in micro_rows]

    print("[pipeline] Writing segment metadata")
    write_jsonl(paths["metadata"] / "key_action_segments.jsonl", key_segments)
    run_ctx.end_stage(inputs=len(key_segments), outputs=len(micro_rows))
    write_jsonl(paths["metadata"] / "micro_segments_raw.jsonl", raw_micro_rows)
    write_jsonl(paths["metadata"] / "micro_segments.jsonl", micro_rows)
    write_jsonl(paths["metadata"] / "micro_vector_metadata.jsonl", micro_vector_metadata)
    return raw_micro_rows, micro_rows, micro_dedup_log, micro_merge_stats, micro_vector_metadata, vector_metadata


def _build_vector_indexes(
    paths: dict[str, Path],
    *,
    combined_vector_metadata: list[dict[str, Any]],
    segment_vector_rows: list[dict[str, Any]],
    micro_vector_metadata: list[dict[str, Any]],
) -> VectorIndex:
    index = VectorIndex()
    index.build([str(item.get("index_text") or "") for item in combined_vector_metadata], combined_vector_metadata)
    index.save(paths["index"])
    write_jsonl(paths["index"] / "docstore.jsonl", combined_vector_metadata)

    segment_index = VectorIndex()
    segment_index.build([str(item.get("index_text") or "") for item in segment_vector_rows], segment_vector_rows)
    segment_index.save(paths["index"] / "segments")

    micro_index = VectorIndex()
    micro_index.build([str(item.get("index_text") or "") for item in micro_vector_metadata], micro_vector_metadata)
    micro_index.save(paths["index"] / "micro_segments")
    return index


def _run_optional_pipeline_stage(
    stage_name: str,
    failed_stages: list[str],
    action: Callable[[], Any],
    default: Any,
) -> Any:
    try:
        return action()
    except Exception as exc:
        failed_stages.append(stage_name)
        print(f"[pipeline] WARNING: {stage_name} failed: {exc}")
        return default


def _write_query_smoke_test(index: VectorIndex, report_path: Path) -> None:
    fusion_weights = {
        "text_similarity": 0.50,
        "time_proximity": 0.20,
        "dual_view_consistency": 0.15,
        "evidence_strength": 0.15,
    }
    try:
        smoke_results = index.query(DEFAULT_QUERY, top_k=3, fusion_weights=fusion_weights)
    except Exception as exc:
        smoke_results = [{"error": str(exc)}]
    _write_json(report_path, {"query": DEFAULT_QUERY, "results": smoke_results})


def _build_process_understanding_outputs(
    output_dir: Path,
    failed_stages: list[str],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> PipelineProcessOutputs:
    print("[pipeline] Building structured video understanding")
    video_understanding_summary = _run_optional_pipeline_stage(
        "video_understanding",
        failed_stages,
        lambda: build_video_understanding(output_dir),
        {},
    )

    print("[pipeline] Building experiment context")
    experiment_context_summary = _run_optional_pipeline_stage(
        "experiment_context",
        failed_stages,
        lambda: build_experiment_context(output_dir),
        {},
    )

    print("[pipeline] Building experiment process reasoning")
    _emit_pipeline_progress(progress_callback, stage="process_reasoning", progress=0.89, message="Building process reasoning and confirmation queue")
    experiment_process_summary = _run_optional_pipeline_stage(
        "experiment_process",
        failed_stages,
        lambda: build_experiment_process(output_dir),
        {},
    )

    print("[pipeline] Building human confirmation queue")
    confirmation_queue_summary = _run_optional_pipeline_stage(
        "confirmation_queue",
        failed_stages,
        lambda: build_confirmation_queue(output_dir),
        {},
    )

    print("[pipeline] Building process quality assurance report")
    quality_assurance_summary = _run_optional_pipeline_stage(
        "quality_assurance",
        failed_stages,
        lambda: build_quality_assurance_report(output_dir),
        {},
    )

    print("[pipeline] Building final process record and audit report")
    process_record_summary = _run_optional_pipeline_stage(
        "process_record",
        failed_stages,
        lambda: build_process_record(output_dir),
        {},
    )
    return PipelineProcessOutputs(
        video_understanding_summary=video_understanding_summary,
        experiment_context_summary=experiment_context_summary,
        experiment_process_summary=experiment_process_summary,
        confirmation_queue_summary=confirmation_queue_summary,
        quality_assurance_summary=quality_assurance_summary,
        process_record_summary=process_record_summary,
    )


def _build_validation_evaluation_and_governance_reports(
    output_dir: Path,
    paths: dict[str, Path],
    index: VectorIndex,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> PipelineValidationOutputs:
    print("[pipeline] Validating JSON/JSONL artifact schemas")
    artifact_validation_summary = validate_session_artifacts(
        output_dir,
        artifact_types=ARTIFACT_VALIDATION_TYPES,
        output_path=paths["metadata"] / "artifact_validation_report.json",
    )
    quality_assurance_summary = build_quality_assurance_report(output_dir)

    print("[pipeline] Building pipeline evaluation report")
    _emit_pipeline_progress(progress_callback, stage="evaluation", progress=0.92, message="Building evaluation and governance reports")
    pipeline_evaluation_summary = build_pipeline_evaluation_report(output_dir)

    print("[pipeline] Building data governance report")
    data_governance_summary = build_data_governance_report(output_dir)

    print("[pipeline] Running query smoke test")
    _write_query_smoke_test(index, paths["reports"] / "query_smoke_test.json")
    return PipelineValidationOutputs(
        artifact_validation_summary=artifact_validation_summary,
        quality_assurance_summary=quality_assurance_summary,
        pipeline_evaluation_summary=pipeline_evaluation_summary,
        data_governance_summary=data_governance_summary,
    )


def _final_segment_rows_and_duration(
    paths: dict[str, Path],
    fallback_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    segment_path = paths["metadata"] / "key_action_segments.jsonl"
    if segment_path.exists():
        stored_rows = read_jsonl(segment_path)
        final_segment_rows = stored_rows if stored_rows or not fallback_rows else fallback_rows
    else:
        final_segment_rows = fallback_rows
    total_action_duration = float(
        sum(float(row.get("duration_sec") or 0.0) for row in final_segment_rows if isinstance(row, dict))
    )
    return final_segment_rows, total_action_duration


def _rebuild_episode_outputs(
    manifest: SessionManifest,
    output_dir: Path,
    paths: dict[str, Path],
    key_segments: list[KeyActionSegment],
    micro_rows: list[dict[str, Any]],
    raw_micro_rows: list[dict[str, Any]],
    micro_merge_stats: dict[str, Any],
    micro_vector_metadata: list[dict[str, Any]],
    vector_metadata: list[VectorMetadata],
    micro_source_rows: list[Mapping[str, Any]],
    utterances: list[TranscriptUtterance],
    detector_summary: dict[str, Any],
    alignment_health: dict[str, Any],
    drift_result: dict[str, Any],
    alignment_degradation: dict[str, Any],
    run_id: str,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[VectorMetadata],
    dict[str, Any],
    dict[str, Any],
]:
    print("[pipeline] Rebuilding true experiment episodes from YOLO micro evidence")
    _emit_pipeline_progress(progress_callback, stage="episodes", progress=0.71, message="Rebuilding experiment episodes from micro evidence")
    try:
        episode_rebuild_summary = rebuild_episode_segments_from_micro_evidence(
            manifest=manifest,
            session_dir=output_dir,
            key_segments=key_segments,
            micro_rows=micro_rows,
            yolo_frame_rows=micro_source_rows,
            utterances=utterances,
            detector_summary=detector_summary if isinstance(detector_summary, dict) else None,
            dry_run=dry_run,
        )
        if episode_rebuild_summary.get("rebuilt"):
            key_segment_rows_for_index, micro_rows, micro_vector_metadata, vector_metadata = _finalize_rebuilt_episode_outputs(
                paths,
                run_id=run_id,
                detector_summary=detector_summary if isinstance(detector_summary, dict) else {},
                alignment_health=alignment_health,
                drift_result=drift_result,
                alignment_degradation=alignment_degradation,
            )
        else:
            key_segment_rows_for_index = [to_json_dict(item) for item in key_segments]
    except Exception as exc:
        episode_rebuild_summary = {"rebuilt": False, "error": str(exc)}
        _write_json(paths["metadata"] / "episode_segmentation_summary.json", episode_rebuild_summary)
        key_segment_rows_for_index = [to_json_dict(item) for item in key_segments]

    micro_quality_stats = compute_micro_quality_stats(
        paths["metadata"] / "micro_segments.jsonl",
        paths["evaluation"] / "micro_quality_stats.json",
    )
    parent_duration_sec = float(micro_quality_stats.get("parent_duration_sec") or 0.0)
    if parent_duration_sec > 0:
        micro_merge_stats["micro_per_minute_before"] = len(raw_micro_rows) / parent_duration_sec * 60.0
        micro_merge_stats["micro_per_minute_after"] = len(micro_rows) / parent_duration_sec * 60.0
        micro_merge_stats["parent_duration_sec"] = parent_duration_sec
    _write_json(paths["evaluation"] / "micro_merge_stats.json", micro_merge_stats)
    return (
        episode_rebuild_summary,
        key_segment_rows_for_index,
        micro_rows,
        micro_vector_metadata,
        vector_metadata,
        micro_quality_stats,
        micro_merge_stats,
    )


def _extract_experiment_focus_summary(
    output_dir: Path,
    paths: dict[str, Path],
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    print("[pipeline] Extracting experiment focus dual-view clips")
    _emit_pipeline_progress(progress_callback, stage="focus_clips", progress=0.74, message="Extracting dual-view experiment focus clips")
    try:
        return extract_experiment_focus_clips(output_dir, dry_run=dry_run)
    except Exception as exc:
        experiment_focus_summary = {"available": False, "error": str(exc)}
        _write_json(paths["metadata"] / "experiment_focus_clips.json", experiment_focus_summary)
        return experiment_focus_summary


def _build_dual_view_action_alignment_summary(
    manifest: SessionManifest,
    output_dir: Path,
    paths: dict[str, Path],
    *,
    yolo_frame_rows: list[dict[str, Any]],
    refined_yolo_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    print("[pipeline] Building dual-view action alignment events")
    try:
        from .dual_view_action_alignment import build_dual_view_action_events

        return build_dual_view_action_events(
            output_dir,
            yolo_frame_rows=yolo_frame_rows,
            yolo_micro_frame_rows=refined_yolo_rows if refined_yolo_rows else None,
            micro_segments=micro_rows,
            manifest=manifest,
        )
    except Exception as exc:
        summary = {
            "schema_version": "key_action_dual_view_action_alignment_summary.v1",
            "available": False,
            "error": str(exc),
            "artifacts": {
                "view_action_evidence": str(paths["metadata"] / "view_action_evidence.jsonl"),
                "dual_view_action_events": str(paths["metadata"] / "dual_view_action_events.jsonl"),
                "unmatched_view_evidence": str(paths["metadata"] / "unmatched_view_evidence.jsonl"),
            },
        }
        write_jsonl(paths["metadata"] / "view_action_evidence.jsonl", [])
        write_jsonl(paths["metadata"] / "dual_view_action_events.jsonl", [])
        write_jsonl(paths["metadata"] / "unmatched_view_evidence.jsonl", [])
        _write_json(paths["metadata"] / "dual_view_action_alignment_summary.json", summary)
        return summary


def _write_vector_metadata_and_build_index(
    paths: dict[str, Path],
    vector_metadata: list[VectorMetadata],
    micro_vector_metadata: list[dict[str, Any]],
    run_ctx: RunContext,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> tuple[VectorIndex, list[dict[str, Any]], list[dict[str, Any]]]:
    segment_vector_rows = [to_json_dict(item) for item in vector_metadata]
    combined_vector_metadata = segment_vector_rows + micro_vector_metadata
    write_jsonl(paths["metadata"] / "vector_metadata.jsonl", combined_vector_metadata)

    print("[pipeline] Building vector index")
    _emit_pipeline_progress(progress_callback, stage="vector_index", progress=0.78, message="Building segment and micro-evidence vector indexes")
    run_ctx.begin_stage("vector_index")
    index = _build_vector_indexes(
        paths,
        combined_vector_metadata=combined_vector_metadata,
        segment_vector_rows=segment_vector_rows,
        micro_vector_metadata=micro_vector_metadata,
    )
    run_ctx.end_stage(inputs=len(combined_vector_metadata), outputs=len(combined_vector_metadata))
    return index, segment_vector_rows, combined_vector_metadata


def _build_evidence_package_outputs(
    manifest: SessionManifest,
    output_dir: Path,
    paths: dict[str, Path],
    active_detector_config: DetectorConfig,
    detected_segments: list[Any],
    generated_yolo_rows: list[dict[str, Any]],
    utterances: list[TranscriptUtterance],
    detector_summary: dict[str, Any],
    alignment_health: dict[str, Any],
    drift_result: dict[str, Any],
    alignment_degradation: float,
    run_ctx: RunContext,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> PipelineEvidenceOutputs:
    yolo_frame_rows, yolo_rows_path = _resolve_yolo_frame_rows(paths, generated_yolo_rows)
    if yolo_rows_path:
        print(f"[pipeline] Loaded YOLO frame rows: {yolo_rows_path} ({len(yolo_frame_rows)} rows)")
    else:
        print("[pipeline] No YOLO frame rows found; using start/middle/end keyframes only")
    if (
        detected_segments
        and _bool_env("KEY_ACTION_FAST_LOCATE_PRECOALESCE_PARENT_SEGMENTS", True)
        and bool(getattr(active_detector_config, "long_video_two_stage_sampling", True))
    ):
        typed_segments = [segment for segment in detected_segments if isinstance(segment, DetectedSegment)]
        if len(typed_segments) == len(detected_segments):
            macro_segments = _coalesce_detected_segments_to_macro_episodes(
                manifest,
                typed_segments,
                active_detector_config,
            )
            if macro_segments and len(macro_segments) < len(detected_segments):
                _write_json(
                    paths["metadata"] / "preclip_macro_episode_coalesce.json",
                    {
                        "schema_version": "key_action_preclip_macro_episode_coalesce.v1",
                        "input_segment_count": len(detected_segments),
                        "output_segment_count": len(macro_segments),
                        "strategy": "density_gap_macro_episode",
                        "macro_merge_gap_sec": _fast_locate_experiment_macro_merge_gap_sec(),
                        "reason": "avoid clipping fine-grained YOLO fragments before experiment-level refinement",
                    },
                )
                detected_segments = macro_segments
                write_jsonl(paths["cv_outputs"] / "detected_segments.jsonl", detected_segments)
                write_jsonl(
                    paths["metadata"] / "experiment_episodes.jsonl",
                    _experiment_episode_rows(
                        manifest,
                        detected_segments,
                        detector_summary=detector_summary if isinstance(detector_summary, dict) else None,
                    ),
                )
    key_segments, vector_metadata = _build_key_segments_and_vector_metadata(
        manifest,
        paths,
        detected_segments,
        utterances,
        yolo_frame_rows,
        alignment_health,
        run_ctx.run_id,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )

    print("[pipeline] Generating segment contact sheet")
    save_segment_contact_sheet(
        paths["keyframes"],
        paths["cv_outputs"] / "detected_segments.jsonl",
        paths["debug"] / "segments_contact_sheet.jpg",
    )

    print("[pipeline] Running optional YOLO micro refine rescan")
    refined_yolo_rows, micro_refine_summary = _run_yolo_micro_refine_rescan(
        manifest,
        paths,
        active_detector_config,
        key_segments,
        dry_run=dry_run,
        coarse_yolo_rows=yolo_frame_rows,
    )
    micro_source_rows, micro_source_path = _resolve_micro_source_rows(
        paths,
        refined_yolo_rows=refined_yolo_rows,
        yolo_frame_rows=yolo_frame_rows,
        yolo_rows_path=yolo_rows_path,
        key_segments=key_segments,
        dry_run=dry_run,
    )

    (
        raw_micro_rows,
        micro_rows,
        micro_dedup_log,
        micro_merge_stats,
        micro_vector_metadata,
        vector_metadata,
    ) = _generate_and_write_micro_segments(
        manifest,
        paths,
        key_segments,
        micro_source_rows,
        utterances,
        run_ctx,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )

    (
        episode_rebuild_summary,
        key_segment_rows_for_index,
        micro_rows,
        micro_vector_metadata,
        vector_metadata,
        micro_quality_stats,
        micro_merge_stats,
    ) = _rebuild_episode_outputs(
        manifest,
        output_dir,
        paths,
        key_segments,
        micro_rows,
        raw_micro_rows,
        micro_merge_stats,
        micro_vector_metadata,
        vector_metadata,
        micro_source_rows,
        utterances,
        detector_summary,
        alignment_health,
        drift_result,
        alignment_degradation,
        run_ctx.run_id,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )

    experiment_focus_summary = _extract_experiment_focus_summary(
        output_dir,
        paths,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )
    dual_view_action_summary = _build_dual_view_action_alignment_summary(
        manifest,
        output_dir,
        paths,
        yolo_frame_rows=yolo_frame_rows,
        refined_yolo_rows=refined_yolo_rows,
        micro_rows=micro_rows,
    )
    formal_output_gate = _formal_output_gate_status(
        paths,
        action_summary=dual_view_action_summary,
        require_action_alignment=bool(not dry_run and manifest.videos.first_person is not None),
    )
    if not formal_output_gate.get("formal_results_allowed"):
        blocked_reason = str(formal_output_gate.get("blocked_reason") or "formal_output_gate_blocked")
        candidate_action_windows = _candidate_action_windows_from_blocked_key_segments(
            manifest,
            key_segments,
            blocked_reason=blocked_reason,
        )
        write_jsonl(paths["metadata"] / "candidate_action_windows.jsonl", candidate_action_windows)
        write_jsonl(paths["metadata"] / "experiment_episode_candidates.jsonl", candidate_action_windows)
        if dry_run:
            formal_output_gate["dry_run_preserved_segments"] = True
            formal_output_gate["dry_run_preserved_segment_count"] = len(key_segments)
        else:
            write_jsonl(paths["metadata"] / "key_action_segments.jsonl", [])
            key_segments = []
            key_segment_rows_for_index = []
            vector_metadata = []
    _write_formal_output_gate(paths, formal_output_gate)
    _write_phase_consistency_from_formal_gate(paths, formal_output_gate)
    index, _segment_vector_rows, _combined_vector_metadata = _write_vector_metadata_and_build_index(
        paths,
        vector_metadata,
        micro_vector_metadata,
        run_ctx,
        progress_callback,
    )
    return PipelineEvidenceOutputs(
        yolo_rows_path=yolo_rows_path,
        micro_source_path=micro_source_path,
        key_segment_rows_for_index=key_segment_rows_for_index,
        episode_rebuild_summary=episode_rebuild_summary,
        micro_refine_summary=micro_refine_summary,
        raw_micro_rows=raw_micro_rows,
        micro_rows=micro_rows,
        micro_dedup_log=micro_dedup_log,
        micro_merge_stats=micro_merge_stats,
        micro_quality_stats=micro_quality_stats,
        dual_view_action_summary=dual_view_action_summary,
        experiment_focus_summary=experiment_focus_summary,
        index=index,
    )


def _build_timeline_asset_observation_outputs(
    manifest_path: str | Path,
    output_dir: Path,
    paths: dict[str, Path],
    input_ingestion_summary: dict[str, Any],
    failed_stages: list[str],
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> PipelineTimelineOutputs:
    print("[pipeline] Generating unified multimodal timeline")
    _emit_pipeline_progress(progress_callback, stage="timeline", progress=0.82, message="Building timeline, asset, and state-change indexes")
    unified_timeline_summary = _run_optional_pipeline_stage(
        "unified_timeline",
        failed_stages,
        lambda: generate_unified_timeline(
            manifest_path=manifest_path,
            output_dir=paths["metadata"],
            user_events_path=input_ingestion_summary["artifacts"]["user_text"],
            ai_events_path=input_ingestion_summary["artifacts"]["ai_reply"],
            uploads_path=input_ingestion_summary["artifacts"]["upload"],
            dry_run=dry_run,
        ),
        {},
    )

    print("[pipeline] Building material asset catalog")
    material_library_summary = _run_optional_pipeline_stage(
        "material_asset_catalog",
        failed_stages,
        lambda: build_material_asset_catalog(output_dir),
        {},
    )

    print("[pipeline] Building state-change index")
    state_change_summary = _run_optional_pipeline_stage(
        "state_change_index",
        failed_stages,
        lambda: build_state_change_index(output_dir),
        {},
    )

    print("[pipeline] Converting YOLO detections into model observation inputs")
    yolo_observation_input_summary = _run_optional_pipeline_stage(
        "yolo_observation_inputs",
        failed_stages,
        lambda: build_yolo_observation_inputs(output_dir),
        {},
    )

    print("[pipeline] Bridging LabSOPGuard model signals into state observation inputs")
    lab_model_signal_input_summary = _run_optional_pipeline_stage(
        "lab_model_signal_inputs",
        failed_stages,
        lambda: build_lab_model_signal_inputs(output_dir),
        {},
    )

    print("[pipeline] Normalizing external model observation outputs")
    model_observation_summary = _run_optional_pipeline_stage(
        "model_observation_events",
        failed_stages,
        lambda: build_model_observation_events(
            output_dir,
            output_path=paths["metadata"] / "model_observation_events.jsonl",
        ),
        {},
    )

    print("[pipeline] Building advanced vision evidence")
    _emit_pipeline_progress(progress_callback, stage="advanced_evidence", progress=0.86, message="Building advanced vision evidence adapters")
    advanced_vision_summary: dict[str, Any] = {}
    advanced_result = _run_optional_pipeline_stage(
        "advanced_vision_evidence",
        failed_stages,
        lambda: build_advanced_vision_evidence(output_dir),
        None,
    )
    if advanced_result is not None:
        advanced_vision_summary = advanced_result
        print("[pipeline] Refreshing material asset catalog with advanced evidence refs")
        material_refresh_result = _run_optional_pipeline_stage(
            "advanced_vision_evidence",
            failed_stages,
            lambda: build_material_asset_catalog(output_dir),
            None,
        )
        if material_refresh_result is not None:
            material_library_summary = material_refresh_result

    print("[pipeline] Writing v2.2 event candidate trace artifacts")
    _run_optional_pipeline_stage(
        "event_candidate_trace_v2_2",
        failed_stages,
        lambda: build_event_candidate_trace(output_dir),
        {},
    )

    return PipelineTimelineOutputs(
        unified_timeline_summary=unified_timeline_summary,
        material_library_summary=material_library_summary,
        state_change_summary=state_change_summary,
        yolo_observation_input_summary=yolo_observation_input_summary,
        lab_model_signal_input_summary=lab_model_signal_input_summary,
        advanced_vision_summary=advanced_vision_summary,
        model_observation_summary=model_observation_summary,
    )


def _build_pipeline_summary(
    *,
    manifest: SessionManifest,
    output_dir: Path,
    dry_run: bool,
    run_ctx: RunContext,
    alignment_health: dict[str, Any],
    failed_stages: list[str],
    final_segment_rows: list[dict[str, Any]],
    total_action_duration: float,
    episode_rebuild_summary: dict[str, Any],
    paths: dict[str, Path],
    input_ingestion_summary: dict[str, Any],
    report_path: Path,
    formal_report_path: Path,
    yolo_rows_path: str | None,
    micro_source_path: str | None,
    detector_summary: dict[str, Any],
    experiment_episode_rows: list[dict[str, Any]],
    video_source_rows: list[dict[str, Any]],
    session_context_summary: dict[str, Any],
    record_ingestion_summary: dict[str, Any],
    long_video_plan: Mapping[str, Any],
    history_model: Mapping[str, Any],
    model_inventory: Mapping[str, Any],
    capability_gap_report: dict[str, Any],
    micro_refine_summary: dict[str, Any],
    micro_rows: list[dict[str, Any]],
    raw_micro_rows: list[dict[str, Any]],
    micro_dedup_log: list[dict[str, Any]],
    micro_merge_stats: dict[str, Any],
    micro_quality_stats: dict[str, Any],
    dual_view_action_summary: dict[str, Any],
    experiment_focus_summary: dict[str, Any],
    unified_timeline_summary: dict[str, Any],
    state_change_summary: dict[str, Any],
    material_library_summary: dict[str, Any],
    yolo_observation_input_summary: dict[str, Any],
    lab_model_signal_input_summary: dict[str, Any],
    advanced_vision_summary: dict[str, Any],
    model_observation_summary: dict[str, Any],
    video_understanding_summary: dict[str, Any],
    experiment_context_summary: dict[str, Any],
    experiment_process_summary: dict[str, Any],
    confirmation_queue_summary: dict[str, Any],
    quality_assurance_summary: dict[str, Any],
    process_record_summary: dict[str, Any],
    artifact_validation_summary: dict[str, Any],
    pipeline_evaluation_summary: dict[str, Any],
    data_governance_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": manifest.session_id,
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "run_id": run_ctx.run_id,
        "run_manifest_id": run_ctx.run_id,
        "stage_stats": run_ctx.stage_stats(),
        "alignment_health": alignment_health,
        "failed_stages": failed_stages,
        "segment_count": len(final_segment_rows),
        "total_action_duration_sec": total_action_duration,
        "episode_segmentation": episode_rebuild_summary,
        "artifacts": _build_pipeline_artifacts(
            paths,
            output_dir,
            input_ingestion_summary=input_ingestion_summary,
            report_path=report_path,
            formal_report_path=formal_report_path,
            yolo_rows_path=yolo_rows_path,
            micro_source_path=micro_source_path,
        ),
        "detector_summary": detector_summary,
        "experiment_episode_count": len(experiment_episode_rows),
        "video_source_count": len(video_source_rows),
        "input_ingestion_summary": input_ingestion_summary,
        "session_context_summary": session_context_summary,
        "record_ingestion_summary": record_ingestion_summary,
        "long_video_plan_summary": _build_long_video_plan_summary(long_video_plan),
        "history_model_summary": _build_history_model_summary(history_model),
        "model_inventory_summary": _build_model_inventory_summary(model_inventory),
        "capability_gap_summary": capability_gap_report.get("summary", {}),
        "micro_refine_summary": micro_refine_summary,
        "micro_segment_count": len(micro_rows),
        "raw_micro_segment_count": len(raw_micro_rows),
        "micro_dedup_count": len(micro_dedup_log),
        "micro_merge_stats": micro_merge_stats,
        "micro_quality_stats": micro_quality_stats,
        "dual_view_action_summary": dual_view_action_summary,
        "formal_output_gate": _read_json_if_exists(paths["metadata"] / "formal_output_gate.json"),
        "experiment_focus_summary": experiment_focus_summary,
        "unified_timeline_summary": unified_timeline_summary,
        "state_change_summary": state_change_summary,
        "material_library_summary": material_library_summary,
        "yolo_observation_input_summary": yolo_observation_input_summary,
        "lab_model_signal_input_summary": lab_model_signal_input_summary,
        "advanced_vision_summary": advanced_vision_summary,
        "model_observation_summary": model_observation_summary,
        "video_understanding_summary": video_understanding_summary,
        "experiment_context_summary": _build_experiment_context_summary(experiment_context_summary),
        "experiment_process_summary": _build_experiment_process_summary(experiment_process_summary),
        "confirmation_queue_summary": confirmation_queue_summary,
        "quality_assurance_summary": _build_quality_assurance_summary(quality_assurance_summary),
        "process_record_summary": process_record_summary.get("summary", {}),
        "artifact_validation_summary": _build_artifact_validation_summary(artifact_validation_summary),
        "pipeline_evaluation_summary": _build_pipeline_evaluation_summary(pipeline_evaluation_summary),
        "data_governance_summary": _build_data_governance_summary(data_governance_summary),
    }


def _finalize_pipeline_run(
    summary: dict[str, Any],
    *,
    manifest: SessionManifest,
    detector_config: DetectorConfig,
    run_ctx: RunContext,
    output_dir: Path,
    report_path: Path,
    formal_report_path: Path,
    quality_assurance_summary: dict[str, Any],
    artifact_validation_summary: dict[str, Any],
    failed_stages: list[str],
    file_handler: _logging.Handler,
    dry_run: bool,
    alignment_health: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    run_manifest = _build_run_manifest(
        manifest,
        detector_config,
        run_ctx,
        dry_run=dry_run,
        alignment_health=alignment_health,
    )
    _write_json(output_dir / "run_manifest.json", run_manifest)

    summary["error_diagnostics"] = _build_error_diagnostics(summary, quality_assurance_summary, artifact_validation_summary)
    summary["module_runs"] = _module_runs_from_summary(summary)
    _write_json(output_dir / "pipeline_summary.json", summary)
    print("[pipeline] Generating validation report")
    _emit_pipeline_progress(progress_callback, stage="reports", progress=0.93, message="Generating validation reports")
    generate_report(output_dir, output_path=report_path)
    generate_formal_validation_report(output_dir, output_path=formal_report_path)
    _emit_pipeline_progress(progress_callback, stage="pipeline_complete", progress=0.935, message="Key-action pipeline artifacts are ready")
    _logger.info("Pipeline completed: segments=%d micros=%d failed_stages=%s", summary["segment_count"], summary.get("micro_segment_count", 0), failed_stages)
    _logger.removeHandler(file_handler)
    file_handler.close()
    print("[pipeline] Completed")
    return summary


def _build_detection_only_summary(
    manifest: SessionManifest,
    paths: dict[str, Path],
    detected_segments: list[Any],
    detector_summary: dict[str, Any],
    model_inventory: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": manifest.session_id,
        "segment_count": len(detected_segments),
        "detected_segments": str(paths["cv_outputs"] / "detected_segments.jsonl"),
        "frame_scores": str(paths["cv_outputs"] / "frame_scores.jsonl"),
        "frame_score_plot": str(paths["debug"] / "frame_scores.png"),
        "detector_summary": detector_summary,
        "model_inventory_summary": {
            "primary_model": model_inventory.get("primary_model", {}),
            "model_count": model_inventory.get("model_count", 0),
            "dataset_count": model_inventory.get("dataset_count", 0),
        },
    }


def _validate_pipeline_inputs(
    manifest_path: str | Path,
    output_dir: Path,
    paths: dict[str, Path],
    run_ctx: RunContext,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    print("[pipeline] Validating manifest and inputs")
    _emit_pipeline_progress(progress_callback, stage="validation", progress=0.05, message="Validating input videos and manifest")
    run_ctx.begin_stage("validation")
    validation_result = validate_manifest(manifest_path, validate_video=not dry_run)
    _write_json(paths["metadata"] / "validation.json", validation_result)
    video_info = {
        "video_sources": validation_result.get("video_sources", {}),
        "transcript": validation_result.get("transcript"),
        "input_sources": validation_result.get("input_sources", {}),
        "can_run_real_pipeline": validation_result.get("can_run_real_pipeline"),
        "issues": validation_result.get("issues", []),
    }
    _write_json(output_dir / "video_info.json", video_info)
    errors = _validation_errors(validation_result)
    if errors and not dry_run:
        raise RuntimeError("Manifest validation failed:\n- " + "\n- ".join(errors))

    run_ctx.end_stage(inputs=1, outputs=1)
    return validation_result


def _build_model_inventory_and_capability_gap(
    manifest: SessionManifest,
    output_dir: Path,
    paths: dict[str, Path],
    config: Mapping[str, Any],
    *,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    print("[pipeline] Discovering local trained models and labeled datasets")
    if _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False):
        model_inventory = {
            "skipped": True,
            "reason": "fast_locate_only",
            "primary_model": {},
            "model_count": 0,
            "dataset_count": 0,
        }
        _write_json(paths["metadata"] / "model_inventory.json", model_inventory)
    else:
        model_inventory = _discover_lab_assets_for_pipeline(
            manifest,
            output_dir,
            paths["metadata"] / "model_inventory.json",
            dry_run=dry_run,
        )
    print("[pipeline] Auditing label and capability gaps")
    capability_gap_project_root = config.get("capability_gap_project_root")
    if dry_run and not capability_gap_project_root:
        capability_gap_project_root = model_inventory.get("project_root") or str(output_dir)
    capability_gap_report = build_capability_gap_report(
        project_root=capability_gap_project_root,
        model_inventory=model_inventory,
        output_path=paths["metadata"] / "capability_gap_report.json",
    )
    return model_inventory, capability_gap_report


def _prepare_pipeline_bootstrap(
    manifest_path: str | Path,
    detector_config: DetectorConfig | None,
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> PipelineBootstrap:
    print(f"[pipeline] Loading manifest: {manifest_path}")
    _emit_pipeline_progress(progress_callback, stage="bootstrap", progress=0.02, message="Loading key-action manifest")
    manifest = SessionManifest.load(manifest_path)
    output_dir = Path(manifest.output_dir)
    paths = _mkdirs(output_dir)
    run_ctx = RunContext()

    log_path = output_dir / "pipeline.log"
    file_handler = _logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    file_handler.setFormatter(_logging.Formatter(f"%(asctime)s [%(levelname)s] [run={run_ctx.run_id[:8]}] %(message)s"))
    _logger.addHandler(file_handler)
    _logger.setLevel(_logging.DEBUG)
    _logger.info("Pipeline started: session=%s manifest=%s dry_run=%s", manifest.session_id, manifest_path, dry_run)

    validation_result = _validate_pipeline_inputs(
        manifest_path,
        output_dir,
        paths,
        run_ctx,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )
    _copy_manifest(manifest_path, manifest, output_dir)
    print("[pipeline] Writing normalized video source metadata")
    _emit_pipeline_progress(progress_callback, stage="preprocess", progress=0.10, message="Preparing normalized input metadata")
    video_source_rows = write_video_source_metadata(manifest, output_dir, paths["metadata"] / "video_sources.jsonl")
    active_detector_config = detector_config or manifest.detection_config
    config = _pipeline_config(manifest)
    print("[pipeline] Detection config:")
    print(json.dumps(to_json_dict(active_detector_config), ensure_ascii=False, indent=2))
    _write_json(paths["metadata"] / "detector_config.json", to_json_dict(active_detector_config))
    print("[pipeline] Planning long-video chunking, cache, resume, and two-stage sampling")
    long_video_plan = build_long_video_processing_plan(
        manifest,
        validation_result,
        active_detector_config,
        output_dir,
        dry_run=dry_run,
    )
    model_inventory, capability_gap_report = _build_model_inventory_and_capability_gap(
        manifest,
        output_dir,
        paths,
        config,
        dry_run=dry_run,
    )

    print("[pipeline] Generating ROI preview")
    save_roi_preview(
        manifest.videos.third_person.path,
        manifest.workbench_roi,
        paths["debug"] / "roi_preview.jpg",
        dry_run=dry_run,
    )
    return PipelineBootstrap(
        manifest=manifest,
        output_dir=output_dir,
        paths=paths,
        run_ctx=run_ctx,
        file_handler=file_handler,
        active_detector_config=active_detector_config,
        video_source_rows=video_source_rows,
        long_video_plan=long_video_plan,
        model_inventory=model_inventory,
        capability_gap_report=capability_gap_report,
    )


def _build_context_inputs_for_pipeline(
    manifest: SessionManifest,
    output_dir: Path,
    paths: dict[str, Path],
    *,
    dry_run: bool,
) -> PipelineContextInputs:
    print("[pipeline] Aligning transcript")
    utterances = load_aligned_transcript(manifest.transcript)
    write_jsonl(paths["transcript"] / "aligned_transcript.jsonl", utterances)

    print("[pipeline] Normalizing user, AI, and upload input events")
    input_ingestion_summary = ingest_manifest_inputs(manifest, output_dir, dry_run=dry_run)
    print("[pipeline] Seeding non-label operational session context")
    session_context_summary = seed_session_context(output_dir)

    print("[pipeline] Normalizing SOP and historical database records")
    record_ingestion_summary = ingest_sop_and_database_records(
        manifest,
        paths["metadata"],
        dry_run=dry_run,
    )
    history_model = build_history_model(
        [paths["metadata"] / "database_records.jsonl"],
        output_path=paths["metadata"] / "history_model.json",
    )
    return PipelineContextInputs(
        utterances=utterances,
        input_ingestion_summary=input_ingestion_summary,
        session_context_summary=session_context_summary,
        record_ingestion_summary=record_ingestion_summary,
        history_model=history_model,
    )


def _summarize_and_finalize_pipeline_run(
    bootstrap: PipelineBootstrap,
    context_inputs: PipelineContextInputs,
    detection_outputs: PipelineDetectionOutputs,
    evidence_outputs: PipelineEvidenceOutputs,
    timeline_outputs: PipelineTimelineOutputs,
    process_outputs: PipelineProcessOutputs,
    validation_outputs: PipelineValidationOutputs,
    failed_stages: list[str],
    *,
    dry_run: bool,
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    manifest = bootstrap.manifest
    output_dir = bootstrap.output_dir
    paths = bootstrap.paths
    report_path = paths["reports"] / "mvp_validation_report.md"
    formal_report_path = paths["reports"] / "formal_validation_report.md"
    final_segment_rows, total_action_duration = _final_segment_rows_and_duration(paths, evidence_outputs.key_segment_rows_for_index)
    summary = _build_pipeline_summary(
        manifest=manifest,
        output_dir=output_dir,
        dry_run=dry_run,
        run_ctx=bootstrap.run_ctx,
        alignment_health=detection_outputs.alignment_health,
        failed_stages=failed_stages,
        final_segment_rows=final_segment_rows,
        total_action_duration=total_action_duration,
        episode_rebuild_summary=evidence_outputs.episode_rebuild_summary,
        paths=paths,
        input_ingestion_summary=context_inputs.input_ingestion_summary,
        report_path=report_path,
        formal_report_path=formal_report_path,
        yolo_rows_path=evidence_outputs.yolo_rows_path,
        micro_source_path=evidence_outputs.micro_source_path,
        detector_summary=detection_outputs.detector_summary,
        experiment_episode_rows=detection_outputs.experiment_episode_rows,
        video_source_rows=bootstrap.video_source_rows,
        session_context_summary=context_inputs.session_context_summary,
        record_ingestion_summary=context_inputs.record_ingestion_summary,
        long_video_plan=bootstrap.long_video_plan,
        history_model=context_inputs.history_model,
        model_inventory=bootstrap.model_inventory,
        capability_gap_report=bootstrap.capability_gap_report,
        micro_refine_summary=evidence_outputs.micro_refine_summary,
        micro_rows=evidence_outputs.micro_rows,
        raw_micro_rows=evidence_outputs.raw_micro_rows,
        micro_dedup_log=evidence_outputs.micro_dedup_log,
        micro_merge_stats=evidence_outputs.micro_merge_stats,
        micro_quality_stats=evidence_outputs.micro_quality_stats,
        dual_view_action_summary=evidence_outputs.dual_view_action_summary,
        experiment_focus_summary=evidence_outputs.experiment_focus_summary,
        unified_timeline_summary=timeline_outputs.unified_timeline_summary,
        state_change_summary=timeline_outputs.state_change_summary,
        material_library_summary=timeline_outputs.material_library_summary,
        yolo_observation_input_summary=timeline_outputs.yolo_observation_input_summary,
        lab_model_signal_input_summary=timeline_outputs.lab_model_signal_input_summary,
        advanced_vision_summary=timeline_outputs.advanced_vision_summary,
        model_observation_summary=timeline_outputs.model_observation_summary,
        video_understanding_summary=process_outputs.video_understanding_summary,
        experiment_context_summary=process_outputs.experiment_context_summary,
        experiment_process_summary=process_outputs.experiment_process_summary,
        confirmation_queue_summary=process_outputs.confirmation_queue_summary,
        quality_assurance_summary=validation_outputs.quality_assurance_summary,
        process_record_summary=process_outputs.process_record_summary,
        artifact_validation_summary=validation_outputs.artifact_validation_summary,
        pipeline_evaluation_summary=validation_outputs.pipeline_evaluation_summary,
        data_governance_summary=validation_outputs.data_governance_summary,
    )
    return _finalize_pipeline_run(
        summary,
        manifest=manifest,
        detector_config=bootstrap.active_detector_config,
        run_ctx=bootstrap.run_ctx,
        output_dir=output_dir,
        report_path=report_path,
        formal_report_path=formal_report_path,
        quality_assurance_summary=validation_outputs.quality_assurance_summary,
        artifact_validation_summary=validation_outputs.artifact_validation_summary,
        failed_stages=failed_stages,
        file_handler=bootstrap.file_handler,
        dry_run=dry_run,
        alignment_health=detection_outputs.alignment_health,
        progress_callback=progress_callback,
    )


def run_pipeline(
    manifest_path: str | Path,
    dry_run: bool = False,
    detector_config: DetectorConfig | None = None,
    alignment_options: dict[str, Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    bootstrap = _prepare_pipeline_bootstrap(
        manifest_path,
        detector_config,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )
    manifest = bootstrap.manifest
    output_dir = bootstrap.output_dir
    paths = bootstrap.paths
    run_ctx = bootstrap.run_ctx
    active_detector_config = bootstrap.active_detector_config

    context_inputs = _build_context_inputs_for_pipeline(
        manifest,
        output_dir,
        paths,
        dry_run=dry_run,
    )

    detection_outputs = _run_detection_alignment_stage(
        manifest,
        paths,
        active_detector_config,
        context_inputs.utterances,
        run_ctx,
        dry_run=dry_run,
        alignment_options=alignment_options,
        progress_callback=progress_callback,
    )

    evidence_outputs = _build_evidence_package_outputs(
        manifest,
        output_dir,
        paths,
        active_detector_config,
        detection_outputs.detected_segments,
        detection_outputs.generated_yolo_rows,
        context_inputs.utterances,
        detection_outputs.detector_summary,
        detection_outputs.alignment_health,
        detection_outputs.drift_result,
        detection_outputs.alignment_degradation,
        run_ctx,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )

    failed_stages: list[str] = []

    timeline_outputs = _build_timeline_asset_observation_outputs(
        manifest_path,
        output_dir,
        paths,
        context_inputs.input_ingestion_summary,
        failed_stages,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )

    process_outputs = _build_process_understanding_outputs(output_dir, failed_stages, progress_callback)

    validation_outputs = _build_validation_evaluation_and_governance_reports(
        output_dir,
        paths,
        evidence_outputs.index,
        progress_callback,
    )

    return _summarize_and_finalize_pipeline_run(
        bootstrap,
        context_inputs,
        detection_outputs,
        evidence_outputs,
        timeline_outputs,
        process_outputs,
        validation_outputs,
        failed_stages=failed_stages,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )


def _fast_locate_experiment_id(manifest: SessionManifest) -> str:
    output_parent = Path(manifest.output_dir).parent
    return output_parent.name if output_parent.name else str(manifest.session_id)


def _fast_locate_source_url(experiment_id: str, view: str) -> str:
    return f"/api/v1/experiments/{experiment_id}/key-actions/source-video/{view}"


def _fast_locate_source_ref(
    source: VideoSource | None,
    *,
    experiment_id: str,
    view: str,
    label_counts: Mapping[str, Any],
    interaction_count: int,
) -> dict[str, Any] | None:
    if source is None:
        return None
    return {
        "video_path": str(getattr(source, "path", "") or ""),
        "clip_path": str(getattr(source, "path", "") or ""),
        "clip_url": _fast_locate_source_url(experiment_id, view),
        "local_start_sec": 0.0,
        "local_end_sec": float(getattr(source, "duration_sec", 0.0) or 0.0),
        "source_time_basis": "source_video_local_time",
        "fast_locate_source_reference": True,
        "yolo_label_counts": dict(label_counts),
        "yolo_detection_count": int(interaction_count),
    }


def _write_detection_only_frontend_projection(
    manifest: SessionManifest,
    paths: dict[str, Path],
    detected_segments: list[Any],
) -> None:
    segment_rows: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    material_candidate_rows: list[dict[str, Any]] = []
    experiment_id = _fast_locate_experiment_id(manifest)
    third_source = manifest.videos.third_person
    first_source = manifest.videos.first_person
    for index, segment in enumerate(detected_segments, start=1):
        row = to_json_dict(segment)
        segment_id = str(row.get("segment_id") or f"fast_locate_segment_{index:03d}")
        start_sec = float(row.get("start_sec") or row.get("start_time_sec") or 0.0)
        end_sec = float(row.get("end_sec") or row.get("end_time_sec") or start_sec)
        duration_sec = max(0.0, float(row.get("duration_sec") or (end_sec - start_sec)))
        interaction_count = int(row.get("yolo_interaction_count") or 0)
        label_counts = row.get("yolo_label_counts") if isinstance(row.get("yolo_label_counts"), dict) else {}
        confidence = float(row.get("boundary_confidence") or row.get("avg_active_score") or row.get("final_score") or 0.0)
        third_ref = _fast_locate_source_ref(
            third_source,
            experiment_id=experiment_id,
            view="third_person",
            label_counts=label_counts,
            interaction_count=interaction_count,
        )
        first_ref = _fast_locate_source_ref(
            first_source,
            experiment_id=experiment_id,
            view="first_person",
            label_counts=label_counts,
            interaction_count=interaction_count,
        )
        source_view = str(row.get("detector_source_view") or row.get("source_view") or "third_person")
        material_clip_url = _fast_locate_source_url(
            experiment_id,
            "third_person" if source_view in {"multiview", "global_multiview"} else source_view,
        )
        segment_rows.append(
            {
                **row,
                "segment_id": segment_id,
                "start_time_sec": start_sec,
                "end_time_sec": end_sec,
                "duration_sec": duration_sec,
                "status": "needs_review",
                "segment_type": "experiment_candidate",
                "fast_locate_only": True,
                "metadata_version": "key_action_fast_locate_segment.v1",
                "third_person": third_ref,
                "first_person": first_ref,
            }
        )
        timeline_rows.append(
            {
                "timeline_event_id": f"fast_locate_{segment_id}",
                "event_type": "experiment_step",
                "step_id": f"fast_locate_step_{index:03d}",
                "order": index,
                "text": f"Candidate experiment segment {index}",
                "name": f"Candidate experiment segment {index}",
                "status": "needs_review",
                "start_time_sec": start_sec,
                "end_time_sec": end_sec,
                "duration_sec": duration_sec,
                "confidence": confidence,
                "reasoning": (
                    f"YOLO fast-locate candidate; hand-object interactions={interaction_count}; "
                    f"labels={','.join(sorted(str(label) for label in label_counts.keys()))}"
                ),
                "evidence_refs": [
                    {
                        "evidence_id": f"{segment_id}:yolo_fast_locate",
                        "kind": "yolo_fast_locate",
                        "segment_id": segment_id,
                        "start_time_sec": start_sec,
                        "end_time_sec": end_sec,
                        "interaction_count": interaction_count,
                    }
                ],
                "metadata_version": "key_action_fast_locate_step_projection.v1",
            }
        )
        material_candidate_rows.append(
            {
                "schema_version": "fast_locate_material_candidate.v1",
                "candidate_id": f"fast_locate_material_{index:03d}",
                "candidate_group_id": f"fast_locate_group_{index:03d}",
                "candidate_status": "pending",
                "review_status": "pending",
                "review_required": True,
                "recommended": True,
                "material_type": "\u5173\u952e\u7247\u6bb5",
                "asset_kind": "\u5173\u952e\u7247\u6bb5",
                "role": "fast_locate_source_clip",
                "display_title": f"Fast locate candidate {index}: {start_sec:.1f}-{end_sec:.1f}s",
                "action_name": f"Fast locate candidate {index}",
                "event_type": "experiment_candidate",
                "physical_action_type": "experiment_candidate",
                "canonical_action_type": "experiment_candidate",
                "canonical_object": "experiment_segment",
                "interaction_family": "hand-object",
                "primary_object": "experiment_segment",
                "view": source_view,
                "camera_view": source_view,
                "segment_id": segment_id,
                "parent_segment_id": segment_id,
                "source_event_id": segment_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "time_start": start_sec,
                "time_end": end_sec,
                "timestamp_sec": start_sec,
                "duration_sec": duration_sec,
                "clip_url": material_clip_url,
                "source_video_url": material_clip_url,
                "source_video_path": (
                    str(getattr(third_source, "path", "") or "")
                    if source_view in {"third_person", "multiview", "global_multiview"}
                    else str(getattr(first_source, "path", "") or "")
                ),
                "exists": True,
                "generated": True,
                "fast_locate_only": True,
                "yolo_annotation_rendered": False,
                "yolo_annotated_required": False,
                "yolo_evidence_count": interaction_count,
                "valid_yolo_evidence_count": interaction_count,
                "quality_score": max(0.0, min(1.0, confidence)),
                "quality_bucket": "review_candidate",
                "pipeline_stage": "fast_locate_candidate_generation",
                "pipeline_status": "pending_frontend_review",
                "candidate_source": "yolo_fast_locate",
                "review_route": "frontend_review",
                "review_gate_policy": "Fast-locate candidates must be reviewed before publishing as formal key materials.",
                "labels": sorted(str(label) for label in label_counts.keys()),
                "label_counts": label_counts,
                "evidence_chain": {
                    "schema_version": "fast_locate_evidence_trace.v1",
                    "source": "yolo_fast_locate",
                    "segment_id": segment_id,
                    "time_start": start_sec,
                    "time_end": end_sec,
                    "interaction_count": interaction_count,
                    "label_counts": label_counts,
                },
            }
        )
    write_jsonl(paths["metadata"] / "key_action_segments.jsonl", segment_rows)
    write_jsonl(paths["metadata"] / "experiment_process_timeline.jsonl", timeline_rows)
    candidate_root = Path(manifest.output_dir).parent / "_material_review_queue"
    candidate_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(candidate_root / "\u7d20\u6750\u5019\u9009\u7d22\u5f15.jsonl", material_candidate_rows)
    _write_json(
        candidate_root / "manifest.json",
        {
            "schema_version": "fast_locate_material_candidates.manifest.v1",
            "experiment_id": experiment_id,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "candidate_group_count": len(material_candidate_rows),
            "source": "yolo_fast_locate_detection_only",
            "fast_locate_only": True,
        },
    )


def _fast_locate_refine_enabled() -> bool:
    return _bool_env("KEY_ACTION_FAST_LOCATE_REFINE", True)


def _fast_locate_streaming_fine_scan_enabled() -> bool:
    return bool(
        _fast_locate_runtime_enabled()
        and _fast_locate_refine_enabled()
        and _bool_env("KEY_ACTION_FAST_LOCATE_STREAMING_FINE_SCAN", False)
    )


def _fast_locate_reuse_streaming_fine_scan_enabled() -> bool:
    return _bool_env("KEY_ACTION_FAST_LOCATE_REUSE_STREAMING_FINE_SCAN", False)


def _fast_locate_refined_step_rows(
    key_segments: list[KeyActionSegment],
    micro_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if micro_rows:
        for index, micro in enumerate(micro_rows, start=1):
            interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
            primary = str(interaction.get("primary_object") or micro.get("primary_object") or "object")
            start_sec = float(micro.get("start_sec") or 0.0)
            end_sec = float(micro.get("end_sec") or start_sec)
            rows.append(
                {
                    "timeline_event_id": f"micro_{micro.get('micro_segment_id') or index}",
                    "event_type": "physical_interaction",
                    "step_id": str(micro.get("micro_segment_id") or f"micro_step_{index:03d}"),
                    "order": index,
                    "text": str(micro.get("display_id") or f"Physical interaction {index}"),
                    "name": str(micro.get("display_id") or f"Physical interaction {index}: hand-{primary}"),
                    "status": "needs_review",
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "duration_sec": max(0.0, end_sec - start_sec),
                    "confidence": (micro.get("quality") or {}).get("confidence") if isinstance(micro.get("quality"), dict) else None,
                    "reasoning": f"YOLO fine-scan physical interaction; primary_object={primary}",
                    "evidence_refs": [
                        {
                            "kind": "micro_segment",
                            "segment_id": micro.get("parent_segment_id"),
                            "micro_segment_id": micro.get("micro_segment_id"),
                            "start_time_sec": start_sec,
                            "end_time_sec": end_sec,
                            "primary_object": primary,
                            "keyframes": micro.get("keyframes"),
                            "third_person_clip": (micro.get("third_person") or {}).get("clip_path")
                            if isinstance(micro.get("third_person"), dict)
                            else None,
                            "first_person_clip": (micro.get("first_person") or {}).get("clip_path")
                            if isinstance(micro.get("first_person"), dict)
                            else None,
                        }
                    ],
                    "metadata_version": "key_action_fast_locate_refined_step.v1",
                }
            )
        return rows

    for index, segment in enumerate(key_segments, start=1):
        start_sec = float(segment.cv_detection.start_sec or segment.third_person.local_start_sec or 0.0)
        end_sec = float(segment.cv_detection.end_sec or segment.third_person.local_end_sec or start_sec)
        rows.append(
            {
                "timeline_event_id": f"fast_locate_{segment.segment_id}",
                "event_type": "experiment_segment",
                "step_id": f"fast_locate_step_{index:03d}",
                "order": index,
                "text": f"Candidate experiment segment {index}",
                "name": f"Candidate experiment segment {index}",
                "status": "needs_review",
                "start_time_sec": start_sec,
                "end_time_sec": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
                "confidence": segment.cv_detection.confidence,
                "reasoning": "YOLO coarse-locate candidate; no micro interaction survived fine scan.",
                "evidence_refs": [{"kind": "key_action_segment", "segment_id": segment.segment_id}],
                "metadata_version": "key_action_fast_locate_refined_step.v1",
            }
        )
    return rows


def _fine_merge_gap_sec(config: DetectorConfig) -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_EXPERIMENT_MERGE_GAP_SEC")
    if raw is None:
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_FINE_MERGE_GAP_SEC")
    if raw is not None:
        try:
            return max(0.25, float(raw))
        except (TypeError, ValueError):
            pass
    return max(5.0, float(os.environ.get("KEY_ACTION_FAST_LOCATE_DEFAULT_EXPERIMENT_MERGE_GAP_SEC", "90.0")))


def _fast_locate_interaction_cluster_gap_sec() -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_INTERACTION_CLUSTER_GAP_SEC")
    if raw is None:
        raw = os.environ.get("KEY_ACTION_FAST_LOCATE_EXPERIMENT_CLUSTER_GAP_SEC", "30.0")
    try:
        return max(0.5, float(raw))
    except (TypeError, ValueError):
        return 30.0


def _fast_locate_min_cluster_interaction_rows() -> int:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_MIN_CLUSTER_INTERACTION_ROWS", "3")
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 3


def _fast_locate_episode_buffer_sec() -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_EPISODE_BUFFER_SEC", "0.0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _fast_locate_experiment_macro_merge_gap_sec() -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_EXPERIMENT_MACRO_MERGE_GAP_SEC", "120.0")
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 120.0


def _fast_locate_orphan_experiment_merge_gap_sec() -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_ORPHAN_EXPERIMENT_MERGE_GAP_SEC")
    if raw is None:
        raw = "180.0"
    try:
        return max(1.0, float(raw))
    except (TypeError, ValueError):
        return 180.0


def _fast_locate_experiment_macro_attach_gap_sec() -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_EXPERIMENT_MACRO_ATTACH_GAP_SEC", "45.0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 45.0


def _fast_locate_semantic_macro_split_gap_sec() -> float:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_SEMANTIC_MACRO_SPLIT_GAP_SEC", "90.0")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 90.0


def _fast_locate_orphan_experiment_min_interactions() -> int:
    raw = os.environ.get("KEY_ACTION_FAST_LOCATE_ORPHAN_EXPERIMENT_MIN_INTERACTIONS", "3")
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 3


def _fast_locate_expected_experiment_count(config: DetectorConfig) -> int | None:
    raw = getattr(config, "expected_experiment_count", None)
    if raw is None and _bool_env("KEY_ACTION_ALLOW_EXPECTED_EXPERIMENT_COUNT_ENV", False):
        raw = (
            os.environ.get("KEY_ACTION_EXPECTED_EXPERIMENT_COUNT")
            or os.environ.get("KEY_ACTION_FAST_LOCATE_EXPECTED_EPISODE_COUNT")
        )
    try:
        count = int(float(raw)) if raw is not None and str(raw).strip() != "" else 0
    except (TypeError, ValueError):
        return None
    return count if count > 0 else None


def _iso_time(value: Any) -> str:
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _detected_segment_start(segment: DetectedSegment) -> float:
    try:
        return float(segment.start_sec)
    except (TypeError, ValueError):
        return 0.0


def _detected_segment_end(segment: DetectedSegment) -> float:
    try:
        return float(segment.end_sec)
    except (TypeError, ValueError):
        return _detected_segment_start(segment)


def _detected_segment_source_views(segment: DetectedSegment) -> set[str]:
    views: set[str] = set()
    raw_source = str(getattr(segment, "detector_source_view", "") or "")
    for part in raw_source.replace(";", ",").replace("|", ",").split(","):
        value = part.strip()
        if value:
            views.add(value)
    for trace in getattr(segment, "decision_trace", []) or []:
        text = str(trace or "")
        if text.startswith("source_views="):
            for part in text.split("=", 1)[1].split(","):
                value = part.strip()
                if value:
                    views.add(value)
        elif text.startswith("source_view="):
            value = text.split("=", 1)[1].strip()
            if value:
                views.add(value)
    return views


def _detected_segment_has_third_or_multiview_support(segment: DetectedSegment) -> bool:
    views = _detected_segment_source_views(segment)
    if not views:
        return False
    if "third_person" in views or "top_view" in views:
        return True
    if "global_multiview" in views or "multiview" in views:
        return True
    return len(views) > 1


def _detected_segment_has_formal_dual_view_support(segment: DetectedSegment) -> bool:
    views = _detected_segment_source_views(segment)
    if "dual_view" in views:
        return True
    return {"first_person", "third_person"}.issubset(views)


def _candidate_action_window_row_from_segment(
    manifest: SessionManifest,
    segment: DetectedSegment,
    *,
    index: int,
    diagnostics: Mapping[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    start_sec = _detected_segment_start(segment)
    end_sec = max(start_sec + 0.1, _detected_segment_end(segment))
    return {
        "schema_version": "key_action_candidate_action_window.v1",
        "session_id": manifest.session_id,
        "action_window_id": f"action_window_{index:06d}",
        "candidate_id": f"candidate_action_window_{index:06d}",
        "candidate_status": "candidate_action_window",
        "official_episode": False,
        "formal_results_allowed": False,
        "single_view_candidate": "single_view_candidate" in reasons,
        "candidate_reasons": list(reasons),
        "segment_id": str(getattr(segment, "segment_id", "")),
        "session_start_sec": round(start_sec, 6),
        "session_end_sec": round(end_sec, 6),
        "duration_sec": round(max(0.0, end_sec - start_sec), 6),
        "global_start_time": str(getattr(segment, "global_start_time", "") or _global_time_from_session_sec(manifest, start_sec).isoformat()),
        "global_end_time": str(getattr(segment, "global_end_time", "") or _global_time_from_session_sec(manifest, end_sec).isoformat()),
        "detector_backend": str(getattr(segment, "detector_backend", "")),
        "detector_source_view": str(getattr(segment, "detector_source_view", "")),
        "source_views": sorted(_detected_segment_source_views(segment)),
        "decision_path": str(getattr(segment, "decision_path", "")),
        "decision_trace": list(getattr(segment, "decision_trace", []) or []),
        "reason_code": str(getattr(segment, "reason_code", "")),
        "boundary_source": str(getattr(segment, "boundary_source", "")),
        "boundary_support_count": int(getattr(segment, "boundary_support_count", 0) or 0),
        "boundary_confidence": _detected_segment_confidence(segment),
        "episode_window_expansion": (
            dict((getattr(segment, "retrieval_boost_factors", {}) or {}).get("experiment_window_expansion") or {})
            if isinstance(getattr(segment, "retrieval_boost_factors", {}), Mapping)
            else {}
        ),
        "yolo_interaction_count": int(getattr(segment, "yolo_interaction_count", 0) or 0),
        "yolo_label_counts": dict(getattr(segment, "yolo_label_counts", {}) or {}),
        "diagnostics": dict(diagnostics),
        "interpretation": "candidate_action_window_not_official_experiment_episode",
    }


def _candidate_action_windows_from_blocked_official_segments(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    *,
    blocked_reason: str,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, segment in enumerate(segments, start=start_index):
        rows.append(
            _candidate_action_window_row_from_segment(
                manifest,
                segment,
                index=offset,
                diagnostics={"formal_output_gate_blocked": True, "blocked_reason": blocked_reason},
                reasons=[blocked_reason],
            )
        )
    return rows


def _candidate_action_windows_from_blocked_key_segments(
    manifest: SessionManifest,
    key_segments: list[KeyActionSegment],
    *,
    blocked_reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(key_segments, start=1):
        try:
            start_sec = _session_time_sec(manifest, str(getattr(segment, "global_start_time", "")))
        except Exception:
            start_sec = 0.0
        try:
            end_sec = _session_time_sec(manifest, str(getattr(segment, "global_end_time", "")))
        except Exception:
            end_sec = start_sec + float(getattr(segment, "duration_sec", 0.0) or 0.0)
        rows.append(
            {
                "schema_version": "key_action_candidate_action_window.v1",
                "session_id": manifest.session_id,
                "action_window_id": f"action_window_{index:06d}",
                "candidate_id": f"candidate_action_window_{index:06d}",
                "candidate_status": "candidate_action_window",
                "official_episode": False,
                "formal_results_allowed": False,
                "blocked_reason": blocked_reason,
                "candidate_reasons": [blocked_reason],
                "segment_id": str(getattr(segment, "segment_id", "") or ""),
                "session_start_sec": round(float(start_sec or 0.0), 6),
                "session_end_sec": round(float(end_sec or start_sec or 0.0), 6),
                "duration_sec": round(float(getattr(segment, "duration_sec", 0.0) or 0.0), 6),
                "global_start_time": str(getattr(segment, "global_start_time", "") or ""),
                "global_end_time": str(getattr(segment, "global_end_time", "") or ""),
                "detector_backend": str(getattr(segment, "detector_backend", "")),
                "detector_source_view": str(getattr(segment, "detector_source_view", "")),
                "decision_path": str(getattr(segment, "decision_path", "")),
                "decision_trace": list(getattr(segment, "decision_trace", []) or []),
                "diagnostics": {"formal_output_gate_blocked": True, "blocked_reason": blocked_reason},
                "interpretation": "candidate_action_window_not_official_experiment_episode",
            }
        )
    return rows


def _detected_segment_gap_sec(left: list[DetectedSegment], right: list[DetectedSegment]) -> float:
    left_end = max(_detected_segment_end(segment) for segment in left)
    right_start = min(_detected_segment_start(segment) for segment in right)
    return float(right_start - left_end)


def _detected_segment_group_label_counts(group: list[DetectedSegment]) -> Counter[str]:
    labels: Counter[str] = Counter()
    for segment in group:
        retrieval_boost_factors = getattr(segment, "retrieval_boost_factors", {}) or {}
        interaction_counts = retrieval_boost_factors.get("interaction_label_counts") if isinstance(retrieval_boost_factors, Mapping) else None
        if isinstance(interaction_counts, Mapping) and interaction_counts:
            labels.update(interaction_counts)
            continue
        labels.update(getattr(segment, "yolo_label_counts", {}) or {})
    return labels


def _detected_segment_group_stirrer_dominant(group: list[DetectedSegment]) -> bool:
    labels = _detected_segment_group_label_counts(group)
    if not labels:
        return False
    stirrer_count = int(labels.get("magnetic_stirrer", 0) or 0)
    if stirrer_count <= 0:
        return False
    paper_balance_count = int(labels.get("paper", 0) or 0) + int(labels.get("balance", 0) or 0)
    return stirrer_count >= max(2, paper_balance_count)


def _detected_segment_groups_semantically_continuous(
    left: list[DetectedSegment],
    right: list[DetectedSegment],
    *,
    gap_sec: float,
) -> bool:
    if gap_sec < _fast_locate_semantic_macro_split_gap_sec():
        return True
    if not _bool_env("KEY_ACTION_FAST_LOCATE_SPLIT_STIRRER_TO_NONSTIRRER_ORPHANS", True):
        return True
    return _detected_segment_group_stirrer_dominant(left) == _detected_segment_group_stirrer_dominant(right)


def _coalesce_refined_with_coarse_macro_episodes(
    manifest: SessionManifest,
    coarse_segments: list[DetectedSegment],
    refined_segments: list[DetectedSegment],
    config: DetectorConfig,
    expected_count: int | None,
) -> list[DetectedSegment]:
    if not coarse_segments:
        return _coalesce_detected_segments_to_expected_count(manifest, refined_segments, expected_count)

    macro_merge_gap = _fast_locate_experiment_macro_merge_gap_sec()
    orphan_merge_gap = _fast_locate_orphan_experiment_merge_gap_sec()
    attach_gap = _fast_locate_experiment_macro_attach_gap_sec()
    min_orphan_interactions = _fast_locate_orphan_experiment_min_interactions()
    require_third_or_multiview = _bool_env("KEY_ACTION_FAST_LOCATE_ORPHAN_EXPERIMENT_REQUIRE_THIRD_VIEW", True)

    groups: list[tuple[list[DetectedSegment], bool]] = []
    for segment in sorted(coarse_segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item))):
        if groups and _detected_segment_gap_sec(groups[-1][0], [segment]) <= macro_merge_gap:
            groups[-1][0].append(segment)
        else:
            groups.append(([segment], True))

    for segment in sorted(refined_segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item))):
        best_index: int | None = None
        best_gap: float | None = None
        for index, (group, _has_coarse) in enumerate(groups):
            gap = max(
                _detected_segment_start(segment) - max(_detected_segment_end(item) for item in group),
                min(_detected_segment_start(item) for item in group) - _detected_segment_end(segment),
                0.0,
            )
            overlaps = _detected_segment_start(segment) <= max(_detected_segment_end(item) for item in group) and _detected_segment_end(segment) >= min(_detected_segment_start(item) for item in group)
            if overlaps or gap <= attach_gap:
                if best_gap is None or gap < best_gap:
                    best_gap = gap
                    best_index = index
        if best_index is not None:
            groups[best_index][0].append(segment)
            continue

        if require_third_or_multiview and not _detected_segment_has_third_or_multiview_support(segment):
            continue
        try:
            interaction_count = int(getattr(segment, "yolo_interaction_count", 0) or 0)
        except (TypeError, ValueError):
            interaction_count = 0
        if interaction_count < min_orphan_interactions:
            continue
        groups.append(([segment], False))

    groups.sort(key=lambda item: (min(_detected_segment_start(segment) for segment in item[0]), max(_detected_segment_end(segment) for segment in item[0])))
    merged: list[tuple[list[DetectedSegment], bool]] = []
    for group, has_coarse in groups:
        if not merged:
            merged.append((list(group), bool(has_coarse)))
            continue
        previous_group, previous_has_coarse = merged[-1]
        gap = _detected_segment_gap_sec(previous_group, group)
        if (previous_has_coarse and has_coarse and gap <= macro_merge_gap) or (
            not previous_has_coarse
            and not has_coarse
            and gap <= orphan_merge_gap
            and _detected_segment_groups_semantically_continuous(previous_group, group, gap_sec=gap)
        ) or gap <= attach_gap:
            previous_group.extend(group)
            merged[-1] = (previous_group, previous_has_coarse or has_coarse)
        else:
            merged.append((list(group), bool(has_coarse)))

    episodes = [
        _coalesced_detected_segment(
            manifest,
            group,
            index,
            strategy="coarse_episode_plus_refined_macro",
            expected_count=expected_count,
        )
        for index, (group, _has_coarse) in enumerate(merged, start=1)
    ]
    if expected_count is not None and len(episodes) != expected_count:
        return _coalesce_detected_segments_to_expected_count(manifest, episodes, expected_count)
    return episodes


def _coalesced_detected_segment(
    manifest: SessionManifest,
    group: list[DetectedSegment],
    index: int,
    *,
    strategy: str,
    expected_count: int | None,
) -> DetectedSegment:
    start_sec = max(0.0, min(_detected_segment_start(segment) for segment in group))
    end_sec = max(start_sec + 0.1, max(_detected_segment_end(segment) for segment in group))
    duration_sec = end_sec - start_sec
    labels: Counter[str] = Counter()
    source_views: set[str] = set()
    interaction_count = 0
    support_count = 0
    active_scores: list[float] = []
    raw_scores: list[float] = []
    boundary_confidences: list[float] = []
    merged_ids: list[str] = []
    interaction_label_counts: Counter[str] = Counter()
    child_expansions: list[dict[str, Any]] = []
    for segment in group:
        merged_ids.append(str(segment.segment_id))
        labels.update(getattr(segment, "yolo_label_counts", {}) or {})
        retrieval_boost_factors = getattr(segment, "retrieval_boost_factors", {}) or {}
        expansion = (
            retrieval_boost_factors.get("experiment_window_expansion")
            if isinstance(retrieval_boost_factors, Mapping)
            else None
        )
        if isinstance(expansion, Mapping) and expansion:
            child_expansions.append(dict(expansion))
        raw_interaction_counts = (
            retrieval_boost_factors.get("interaction_label_counts")
            if isinstance(retrieval_boost_factors, Mapping)
            else None
        )
        if isinstance(raw_interaction_counts, Mapping):
            interaction_label_counts.update(raw_interaction_counts)
        interaction_count += int(getattr(segment, "yolo_interaction_count", 0) or 0)
        support_count += int(getattr(segment, "boundary_support_count", 0) or 0)
        source_views.update(_detected_segment_source_views(segment))
        for value in (getattr(segment, "avg_active_score", None), getattr(segment, "final_score", None)):
            try:
                active_scores.append(float(value or 0.0))
            except (TypeError, ValueError):
                pass
        for value in (getattr(segment, "avg_motion_score", None), getattr(segment, "raw_score", None)):
            try:
                raw_scores.append(float(value or 0.0))
            except (TypeError, ValueError):
                pass
        try:
            boundary_confidences.append(float(getattr(segment, "boundary_confidence", 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    avg_active = sum(active_scores) / len(active_scores) if active_scores else 0.0
    avg_raw = sum(raw_scores) / len(raw_scores) if raw_scores else avg_active
    confidence = max(boundary_confidences) if boundary_confidences else min(1.0, 0.55 + min(0.35, 0.01 * interaction_count))
    return DetectedSegment(
        segment_id=f"seg_{index:06d}",
        start_sec=start_sec,
        end_sec=end_sec,
        duration_sec=duration_sec,
        global_start_time=_iso_time(_global_time_from_session_sec(manifest, start_sec)),
        global_end_time=_iso_time(_global_time_from_session_sec(manifest, end_sec)),
        avg_motion_score=avg_raw,
        avg_active_score=avg_active,
        start_reason="yolo_episode_cluster_start",
        end_reason="yolo_episode_cluster_end",
        review_required=False,
        detector_backend="yolo_interaction",
        detector_source_view="global_multiview" if len(source_views) > 1 else (next(iter(source_views)) if source_views else "third_person"),
        yolo_label_counts=dict(labels),
        yolo_interaction_count=interaction_count,
        boundary_confidence=min(1.0, confidence),
        boundary_support_count=support_count,
        boundary_source="yolo_episode_coalesced_from_action_clusters",
        decision_path=DETECTION_DECISION_YOLO_INTERACTION,
        decision_trace=[
            "backend=yolo_interaction",
            "scan_role=experiment_episode",
            f"merge_strategy={strategy}",
            f"expected_experiment_count={expected_count or ''}",
            f"merged_cluster_count={len(group)}",
            f"merged_from_segment_ids={','.join(merged_ids)}",
            f"source_views={','.join(sorted(source_views)) if source_views else 'unknown'}",
        ],
        reason_code=DECISION_REASON_YOLO_INTERACTION_DETECTED,
        raw_score=avg_raw,
        final_score=avg_active,
        source="experiment_episode",
        retrieval_boost_factors={
            "episode_layer": True,
            "merged_from_segment_ids": merged_ids,
            "action_cluster_count": len(group),
            "interaction_label_counts": dict(interaction_label_counts),
            "experiment_window_expansion": (
                {
                    "schema_version": "official_experiment_window_expansion.v1",
                    "expanded": any(bool(item.get("expanded")) for item in child_expansions),
                    "source": "merged_child_experiment_window_expansions",
                    "segment_id": f"seg_{index:06d}",
                    "original_start_sec": round(min(float(item.get("original_start_sec") or start_sec) for item in child_expansions), 6),
                    "original_end_sec": round(max(float(item.get("original_end_sec") or end_sec) for item in child_expansions), 6),
                    "expanded_start_sec": round(start_sec, 6),
                    "expanded_end_sec": round(end_sec, 6),
                    "child_expansion_count": len(child_expansions),
                    "child_expansions": child_expansions[:20],
                }
                if child_expansions
                else {}
            ),
        },
    )


def _coalesce_detected_segments_to_expected_count(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    expected_count: int | None,
) -> list[DetectedSegment]:
    ordered = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    if not ordered:
        return []
    if expected_count is None or expected_count <= 0 or len(ordered) <= expected_count:
        return [
            _coalesced_detected_segment(
                manifest,
                [segment],
                index,
                strategy="preserve_action_cluster_as_episode",
                expected_count=expected_count,
            )
            for index, segment in enumerate(ordered, start=1)
        ]
    boundary_count = max(0, expected_count - 1)
    gaps: list[tuple[float, int]] = []
    for index in range(1, len(ordered)):
        gap = _detected_segment_start(ordered[index]) - _detected_segment_end(ordered[index - 1])
        gaps.append((float(gap), index))
    boundary_indexes = {
        index
        for _gap, index in sorted(gaps, key=lambda item: (item[0], item[1]), reverse=True)[:boundary_count]
    }
    groups: list[list[DetectedSegment]] = []
    current: list[DetectedSegment] = []
    for index, segment in enumerate(ordered):
        if current and index in boundary_indexes:
            groups.append(current)
            current = []
        current.append(segment)
    if current:
        groups.append(current)
    return [
        _coalesced_detected_segment(
            manifest,
            group,
            index,
            strategy="expected_count_largest_gap_partition",
            expected_count=expected_count,
        )
        for index, group in enumerate(groups, start=1)
    ]


def _coalesce_detected_segments_to_macro_episodes(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    config: DetectorConfig,
) -> list[DetectedSegment]:
    """Collapse dense YOLO fragments before expensive parent clip extraction."""

    expected_count = _fast_locate_expected_experiment_count(config)
    if expected_count is not None:
        return _coalesce_detected_segments_to_expected_count(manifest, segments, expected_count)
    ordered = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    if not ordered:
        return []
    min_fragments = int(_float_env_value("KEY_ACTION_FAST_LOCATE_MACRO_COALESCE_MIN_FRAGMENTS", 6.0))
    if len(ordered) < max(2, min_fragments):
        return _coalesce_detected_segments_to_expected_count(manifest, ordered, None)

    macro_merge_gap = _fast_locate_experiment_macro_merge_gap_sec()
    groups: list[list[DetectedSegment]] = []
    current: list[DetectedSegment] = []
    for segment in ordered:
        if not current:
            current = [segment]
            continue
        gap = _detected_segment_gap_sec(current, [segment])
        if gap <= macro_merge_gap and _detected_segment_groups_semantically_continuous(current, [segment], gap_sec=gap):
            current.append(segment)
            continue
        groups.append(current)
        current = [segment]
    if current:
        groups.append(current)

    return [
        _coalesced_detected_segment(
            manifest,
            group,
            index,
            strategy="density_gap_macro_episode",
            expected_count=None,
        )
        for index, group in enumerate(groups, start=1)
    ]


def _experiment_window_min_duration_sec(default: float = 120.0) -> float:
    return _float_env_any_value(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_MIN_SEC",
            "KEY_ACTION_FAST_LOCATE_MIN_EXPERIMENT_EPISODE_SEC",
            "KEY_ACTION_MIN_OFFICIAL_EXPERIMENT_DURATION_SEC",
            "KEY_ACTION_EPISODE_MIN_OFFICIAL_DURATION_SEC",
        ),
        default,
        minimum=0.0,
    )


def _experiment_window_silence_gap_sec() -> float:
    return _float_env_any_value(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_SILENCE_GAP_SEC",
            "KEY_ACTION_ACTIVITY_VALLEY_GAP_SEC",
            "KEY_ACTION_EPISODE_ACTIVITY_VALLEY_GAP_SEC",
        ),
        30.0,
        minimum=0.5,
    )


def _experiment_window_attach_gap_sec() -> float:
    return _float_env_any_value(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_ATTACH_GAP_SEC",
            "KEY_ACTION_ACTIVITY_WINDOW_ATTACH_GAP_SEC",
        ),
        10.0,
        minimum=0.0,
    )


def _experiment_window_activity_min_score() -> float:
    return _float_env_any_value(
        (
            "KEY_ACTION_EXPERIMENT_WINDOW_ACTIVITY_MIN_SCORE",
            "KEY_ACTION_ACTIVITY_WINDOW_MIN_SCORE",
        ),
        0.05,
        minimum=0.0,
    )


def _fast_locate_min_episode_duration_sec() -> float:
    return _experiment_window_min_duration_sec()


def _fast_locate_min_short_episode_interactions() -> int:
    return max(0, _int_env_value("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_INTERACTIONS", 8))


def _fast_locate_min_short_episode_support_count() -> int:
    return max(0, _int_env_value("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_SUPPORT_COUNT", 3))


def _fast_locate_min_short_episode_confidence() -> float:
    return max(0.0, min(1.0, _float_env_value("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_CONFIDENCE", 0.65)))


def _fast_locate_short_weak_merge_gap_sec() -> float:
    return max(0.0, _float_env_value("KEY_ACTION_FAST_LOCATE_SHORT_WEAK_MERGE_GAP_SEC", 45.0))


def _detected_segment_support_count(segment: DetectedSegment) -> int:
    counts = []
    for value in (
        getattr(segment, "boundary_support_count", 0),
        getattr(segment, "yolo_interaction_count", 0),
    ):
        try:
            counts.append(int(value or 0))
        except (TypeError, ValueError):
            continue
    return max(counts, default=0)


def _detected_segment_confidence(segment: DetectedSegment) -> float:
    values = []
    for value in (
        getattr(segment, "boundary_confidence", 0.0),
        getattr(segment, "final_score", 0.0),
        getattr(segment, "avg_active_score", 0.0),
    ):
        try:
            values.append(float(value or 0.0))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def _short_weak_episode_diagnostics(
    segment: DetectedSegment,
    *,
    min_duration: float,
    min_interactions: int,
    min_support_count: int,
    min_confidence: float,
) -> dict[str, Any]:
    duration = float(getattr(segment, "duration_sec", 0.0) or 0.0)
    interactions = int(getattr(segment, "yolo_interaction_count", 0) or 0)
    support_count = _detected_segment_support_count(segment)
    confidence = _detected_segment_confidence(segment)
    return {
        "segment_id": str(getattr(segment, "segment_id", "")),
        "start_sec": float(getattr(segment, "start_sec", 0.0) or 0.0),
        "end_sec": float(getattr(segment, "end_sec", 0.0) or 0.0),
        "duration_sec": duration,
        "yolo_interaction_count": interactions,
        "boundary_support_count": support_count,
        "confidence": round(confidence, 6),
        "short": bool(duration < min_duration),
        "weak_interactions": bool(interactions < min_interactions),
        "weak_support": bool(support_count < min_support_count),
        "weak_confidence": bool(confidence < min_confidence),
    }


def _is_short_weak_experiment_episode(
    segment: DetectedSegment,
    *,
    min_duration: float,
    min_interactions: int,
    min_support_count: int,
    min_confidence: float,
) -> bool:
    details = _short_weak_episode_diagnostics(
        segment,
        min_duration=min_duration,
        min_interactions=min_interactions,
        min_support_count=min_support_count,
        min_confidence=min_confidence,
    )
    return bool(
        details["short"]
        and details["weak_interactions"]
        and details["weak_support"]
        and details["weak_confidence"]
    )


def _merge_short_weak_experiment_fragments(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    *,
    min_duration: float,
    min_interactions: int,
    min_support_count: int,
    min_confidence: float,
    merge_gap_sec: float,
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if len(segments) <= 1 or merge_gap_sec <= 0.0:
        return segments, {"enabled": False, "merged_count": 0}
    ordered = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    groups: list[list[DetectedSegment]] = []
    merge_events: list[dict[str, Any]] = []
    weak_ids: set[str] = set()

    def is_weak(segment: DetectedSegment) -> bool:
        weak = _is_short_weak_experiment_episode(
            segment,
            min_duration=min_duration,
            min_interactions=min_interactions,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
        )
        if weak:
            weak_ids.add(str(getattr(segment, "segment_id", "")))
        return weak

    for segment in ordered:
        segment_is_weak = is_weak(segment)
        if not groups:
            groups.append([segment])
            continue
        previous_group = groups[-1]
        previous_tail_is_weak = is_weak(previous_group[-1])
        gap = _detected_segment_gap_sec(previous_group, [segment])
        if gap <= merge_gap_sec and (segment_is_weak or previous_tail_is_weak):
            merge_events.append(
                {
                    "decision": "merge",
                    "reason": "short_weak_episode_fragment_adjacent_to_activity_evidence",
                    "gap_sec": round(gap, 6),
                    "threshold_sec": round(merge_gap_sec, 6),
                    "left_segment_id": str(getattr(previous_group[-1], "segment_id", "")),
                    "right_segment_id": str(getattr(segment, "segment_id", "")),
                }
            )
            previous_group.append(segment)
            continue
        groups.append([segment])

    merged: list[DetectedSegment] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged.append(
            _coalesced_detected_segment(
                manifest,
                group,
                len(merged) + 1,
                strategy="short_weak_oversegmentation_merge",
                expected_count=None,
            )
        )
    return merged, {
        "enabled": True,
        "input_count": len(segments),
        "output_count": len(merged),
        "merged_count": len(segments) - len(merged),
        "weak_fragment_segment_ids": sorted(value for value in weak_ids if value),
        "merge_gap_sec": merge_gap_sec,
        "events": merge_events[:50],
    }


def _merge_adjacent_action_window_fragments(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    *,
    min_duration: float,
    merge_gap_sec: float,
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if len(segments) <= 1 or merge_gap_sec <= 0.0:
        return segments, {"enabled": False, "merged_count": 0}
    ordered = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    groups: list[list[DetectedSegment]] = []
    events: list[dict[str, Any]] = []
    for segment in ordered:
        if not groups:
            groups.append([segment])
            continue
        previous_group = groups[-1]
        gap = _detected_segment_gap_sec(previous_group, [segment])
        group_has_short = any(_detected_segment_end(item) - _detected_segment_start(item) < min_duration for item in previous_group)
        segment_is_short = _detected_segment_end(segment) - _detected_segment_start(segment) < min_duration
        if gap <= merge_gap_sec and (group_has_short or segment_is_short):
            events.append(
                {
                    "decision": "merge",
                    "reason": "adjacent_action_fragments_within_episode_gap",
                    "gap_sec": round(gap, 6),
                    "threshold_sec": round(merge_gap_sec, 6),
                    "left_segment_id": str(getattr(previous_group[-1], "segment_id", "")),
                    "right_segment_id": str(getattr(segment, "segment_id", "")),
                }
            )
            previous_group.append(segment)
            continue
        groups.append([segment])

    merged: list[DetectedSegment] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged.append(
            _coalesced_detected_segment(
                manifest,
                group,
                len(merged) + 1,
                strategy="adjacent_action_window_gap_merge",
                expected_count=None,
            )
        )
    return merged, {
        "enabled": True,
        "input_count": len(segments),
        "output_count": len(merged),
        "merged_count": len(segments) - len(merged),
        "merge_gap_sec": merge_gap_sec,
        "events": events[:50],
    }


def _merge_overlapping_experiment_windows(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if len(segments) <= 1:
        return segments, {"enabled": True, "input_count": len(segments), "output_count": len(segments), "merged_count": 0}
    ordered = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    groups: list[list[DetectedSegment]] = []
    events: list[dict[str, Any]] = []
    for segment in ordered:
        if not groups:
            groups.append([segment])
            continue
        previous_group = groups[-1]
        gap = _detected_segment_gap_sec(previous_group, [segment])
        if gap < -0.001:
            events.append(
                {
                    "decision": "merge",
                    "reason": "overlapping_expanded_experiment_windows",
                    "gap_sec": round(gap, 6),
                    "left_segment_id": str(getattr(previous_group[-1], "segment_id", "")),
                    "right_segment_id": str(getattr(segment, "segment_id", "")),
                }
            )
            previous_group.append(segment)
            continue
        groups.append([segment])

    merged: list[DetectedSegment] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged.append(
            _coalesced_detected_segment(
                manifest,
                group,
                len(merged) + 1,
                strategy="overlapping_experiment_window_merge",
                expected_count=None,
            )
        )
    return merged, {
        "enabled": True,
        "input_count": len(segments),
        "output_count": len(merged),
        "merged_count": len(segments) - len(merged),
        "events": events[:50],
    }


def _merge_same_lifecycle_expanded_experiment_windows(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    expansions: list[Mapping[str, Any]],
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if len(segments) <= 1:
        return segments, {
            "enabled": True,
            "input_count": len(segments),
            "output_count": len(segments),
            "merged_count": 0,
        }
    if not _bool_env("KEY_ACTION_MERGE_SAME_LIFECYCLE_EXPERIMENT_WINDOWS", True):
        return segments, {
            "enabled": False,
            "input_count": len(segments),
            "output_count": len(segments),
            "merged_count": 0,
            "reason": "disabled_by_env",
        }

    expansion_by_segment_id = {
        str(item.get("segment_id") or ""): item
        for item in expansions
        if isinstance(item, Mapping)
    }

    def lifecycle_id_for(segment: DetectedSegment) -> str:
        expansion = expansion_by_segment_id.get(str(getattr(segment, "segment_id", "") or "")) or {}
        lifecycle = expansion.get("lifecycle_window") if isinstance(expansion, Mapping) else None
        if not isinstance(lifecycle, Mapping):
            return ""
        return str(lifecycle.get("window_id") or "").strip()

    max_gap_sec = _float_env_any_value(
        ("KEY_ACTION_SAME_LIFECYCLE_MERGE_MAX_GAP_SEC",),
        1.0,
        minimum=0.0,
    )
    ordered = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    groups: list[list[DetectedSegment]] = []
    events: list[dict[str, Any]] = []
    for segment in ordered:
        if not groups:
            groups.append([segment])
            continue
        previous_group = groups[-1]
        left = previous_group[-1]
        left_lifecycle = lifecycle_id_for(left)
        right_lifecycle = lifecycle_id_for(segment)
        gap = _detected_segment_gap_sec(previous_group, [segment])
        if left_lifecycle and left_lifecycle == right_lifecycle and gap <= max_gap_sec:
            events.append(
                {
                    "decision": "merge",
                    "reason": "same_experiment_lifecycle_without_exit_boundary",
                    "lifecycle_window_id": left_lifecycle,
                    "gap_sec": round(gap, 6),
                    "threshold_sec": round(max_gap_sec, 6),
                    "left_segment_id": str(getattr(left, "segment_id", "")),
                    "right_segment_id": str(getattr(segment, "segment_id", "")),
                }
            )
            previous_group.append(segment)
            continue
        groups.append([segment])

    merged: list[DetectedSegment] = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue
        merged.append(
            _coalesced_detected_segment(
                manifest,
                group,
                len(merged) + 1,
                strategy="same_lifecycle_experiment_window_merge",
                expected_count=None,
            )
        )
    return merged, {
        "enabled": True,
        "input_count": len(segments),
        "output_count": len(merged),
        "merged_count": len(segments) - len(merged),
        "max_gap_sec": round(max_gap_sec, 6),
        "events": events[:50],
    }


def _filter_short_weak_experiment_episodes(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    *,
    expected_count: int | None,
    dry_run: bool = False,
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if dry_run:
        return segments, {"enabled": False, "removed_count": 0, "reason": "dry_run"}
    if not _bool_env("KEY_ACTION_FAST_LOCATE_FILTER_SHORT_WEAK_EPISODES", True):
        return segments, {"enabled": False, "removed_count": 0}
    min_duration = _fast_locate_min_episode_duration_sec()
    min_interactions = _fast_locate_min_short_episode_interactions()
    min_support_count = _fast_locate_min_short_episode_support_count()
    min_confidence = _fast_locate_min_short_episode_confidence()
    merge_gap_sec = _fast_locate_short_weak_merge_gap_sec()
    segments, merge_summary = _merge_short_weak_experiment_fragments(
        manifest,
        segments,
        min_duration=min_duration,
        min_interactions=min_interactions,
        min_support_count=min_support_count,
        min_confidence=min_confidence,
        merge_gap_sec=merge_gap_sec,
    )
    segments, action_window_merge_summary = _merge_adjacent_action_window_fragments(
        manifest,
        segments,
        min_duration=min_duration,
        merge_gap_sec=merge_gap_sec,
    )
    segments, overlap_merge_summary = _merge_overlapping_experiment_windows(manifest, segments)
    kept: list[DetectedSegment] = []
    removed: list[dict[str, Any]] = []
    candidate_action_windows: list[dict[str, Any]] = []
    for segment in segments:
        diagnostics = _short_weak_episode_diagnostics(
            segment,
            min_duration=min_duration,
            min_interactions=min_interactions,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
        )
        candidate_reasons: list[str] = []
        if bool(diagnostics["short"]):
            candidate_reasons.append("too_short_action_window")
        if not _detected_segment_has_formal_dual_view_support(segment):
            candidate_reasons.append("single_view_candidate")
        if _is_short_weak_experiment_episode(
            segment,
            min_duration=min_duration,
            min_interactions=min_interactions,
            min_support_count=min_support_count,
            min_confidence=min_confidence,
        ):
            candidate_reasons.append("short_weak_episode_without_enough_physical_interaction_evidence")
        candidate_reasons = _ordered_unique(candidate_reasons)
        if candidate_reasons:
            removed_row = {
                **diagnostics,
                "reason": candidate_reasons[0],
                "candidate_reasons": candidate_reasons,
            }
            removed.append(removed_row)
            candidate_action_windows.append(
                _candidate_action_window_row_from_segment(
                    manifest,
                    segment,
                    index=len(candidate_action_windows) + 1,
                    diagnostics=diagnostics,
                    reasons=candidate_reasons,
                )
            )
            continue
        kept.append(segment)
    for index, segment in enumerate(kept, start=1):
        segment.segment_id = f"seg_{index:06d}"
    return kept, {
        "enabled": True,
        "input_count": len(segments),
        "output_count": len(kept),
        "removed_count": len(removed),
        "min_duration_sec": min_duration,
        "min_short_episode_interactions": min_interactions,
        "min_short_episode_support_count": min_support_count,
        "min_short_episode_confidence": min_confidence,
        "short_weak_merge": merge_summary,
        "action_window_gap_merge": action_window_merge_summary,
        "overlap_window_merge": overlap_merge_summary,
        "removed": removed,
        "candidate_action_window_count": len(candidate_action_windows),
        "candidate_action_windows": candidate_action_windows,
        "official_episode_count": len(kept),
        "formal_results_allowed": bool(kept),
    }


def _row_alignment_sec(row: dict[str, Any]) -> float:
    for key in ("alignment_time_sec", "session_time_sec", "local_time_sec", "time_sec"):
        try:
            return float(row.get(key))
        except (TypeError, ValueError):
            continue
    return 0.0


def _row_has_hand_object_interaction(row: dict[str, Any]) -> bool:
    interactions = row.get("hand_object_interactions")
    if isinstance(interactions, list) and interactions:
        return True
    try:
        return float(row.get("interaction_score") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


def _row_detection_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    detections = row.get("detections")
    if not isinstance(detections, list):
        return labels
    for detection in detections:
        if not isinstance(detection, dict):
            continue
        label = str(detection.get("label") or detection.get("class_name") or detection.get("raw_label") or "").strip()
        if label:
            labels.append(label)
    return labels


def _row_interaction_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    interactions = row.get("hand_object_interactions")
    if isinstance(interactions, list):
        for interaction in interactions:
            if not isinstance(interaction, Mapping):
                continue
            label = str(
                interaction.get("object_label")
                or interaction.get("label")
                or interaction.get("object_name")
                or ""
            ).strip()
            if label:
                labels.append(label)
    if labels:
        return labels
    return _row_detection_labels(row) if _row_has_hand_object_interaction(row) else []


def _row_interaction_count(row: dict[str, Any]) -> int:
    interactions = row.get("hand_object_interactions")
    if isinstance(interactions, list) and interactions:
        return len(interactions)
    return 1 if _row_has_hand_object_interaction(row) else 0


def _expected_count_segments_from_yolo_rows(
    manifest: SessionManifest,
    rows: list[dict[str, Any]],
    config: DetectorConfig,
    expected_count: int,
) -> list[DetectedSegment]:
    if expected_count <= 0 or len(rows) < expected_count:
        return []
    ordered = sorted(rows, key=_row_alignment_sec)
    gaps: list[tuple[float, int]] = []
    previous_time = _row_alignment_sec(ordered[0])
    for index, row in enumerate(ordered[1:], start=1):
        current_time = _row_alignment_sec(row)
        gaps.append((float(current_time - previous_time), index))
        previous_time = current_time
    boundary_indexes = {
        index
        for _gap, index in sorted(gaps, key=lambda item: (item[0], item[1]), reverse=True)[: max(0, expected_count - 1)]
    }
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        if current and index in boundary_indexes:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    if len(groups) != expected_count:
        return []

    episode_buffer_sec = _fast_locate_episode_buffer_sec()
    sample_period = 1.0 / max(_refined_yolo_sample_fps(config), 0.001)
    segments: list[DetectedSegment] = []
    for group in groups:
        times = [_row_alignment_sec(row) for row in group]
        start_sec = max(0.0, min(times) - episode_buffer_sec)
        end_sec = max(start_sec + sample_period, max(times) + sample_period + episode_buffer_sec)
        labels = Counter(label for row in group for label in _row_detection_labels(row))
        interaction_count = sum(_row_interaction_count(row) for row in group)
        active_scores = []
        raw_scores = []
        for row in group:
            try:
                active_scores.append(float(row.get("active_score") or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                raw_scores.append(float(row.get("raw_score") or row.get("probability") or 0.0))
            except (TypeError, ValueError):
                pass
        avg_active = sum(active_scores) / len(active_scores) if active_scores else 0.0
        avg_raw = sum(raw_scores) / len(raw_scores) if raw_scores else avg_active
        source_views = sorted({str(row.get("source_view") or row.get("view") or "") for row in group if row.get("source_view") or row.get("view")})
        segment_index = len(segments) + 1
        segments.append(
            DetectedSegment(
                segment_id=f"seg_{segment_index:06d}",
                start_sec=start_sec,
                end_sec=end_sec,
                duration_sec=end_sec - start_sec,
                global_start_time=_global_time_from_session_sec(manifest, start_sec),
                global_end_time=_global_time_from_session_sec(manifest, end_sec),
                avg_motion_score=avg_raw,
                avg_active_score=avg_active,
                start_reason="expected_count_gap_partition_start",
                end_reason="expected_count_gap_partition_end",
                review_required=False,
                detector_backend="yolo_interaction",
                detector_source_view="global_multiview" if len(source_views) > 1 else (source_views[0] if source_views else "third_person"),
                yolo_label_counts=dict(labels),
                yolo_interaction_count=interaction_count,
                boundary_confidence=min(1.0, 0.52 + min(0.35, 0.01 * interaction_count)),
                boundary_support_count=len(group),
                boundary_source="expected_count_row_gap_partition",
                decision_path=DETECTION_DECISION_YOLO_INTERACTION,
                decision_trace=[
                    "backend=yolo_interaction",
                    "scan_role=experiment_episode_refine",
                    "merge_strategy=expected_count_row_gap_partition",
                    f"expected_experiment_count={expected_count}",
                    f"interaction_rows={len(group)}",
                    f"interaction_count={interaction_count}",
                    f"source_views={','.join(source_views) if source_views else 'unknown'}",
                ],
                reason_code=DECISION_REASON_YOLO_INTERACTION_DETECTED,
                raw_score=avg_raw,
                final_score=avg_active,
            )
        )
    return segments


def _row_detection_count(row: dict[str, Any]) -> int:
    detections = row.get("detections")
    if isinstance(detections, list):
        return len([item for item in detections if isinstance(item, dict)])
    counts = row.get("label_counts")
    if isinstance(counts, dict):
        total = 0
        for value in counts.values():
            try:
                total += int(value or 0)
            except (TypeError, ValueError):
                continue
        return total
    return 0


def _fast_locate_row_seed_score(row: dict[str, Any]) -> float:
    score = 0.0
    for key in ("active_score", "interaction_score", "motion_score", "raw_yolo_active_score", "probability"):
        try:
            score = max(score, float(row.get(key) or 0.0))
        except (TypeError, ValueError):
            continue
    interaction_count = _row_interaction_count(row)
    detection_count = _row_detection_count(row)
    labels = Counter(_row_detection_labels(row))
    label_diversity = len([label for label, count in labels.items() if count > 0 and label not in _YOLO_HAND_LABELS])
    hand_count = sum(int(labels.get(label, 0) or 0) for label in _YOLO_HAND_LABELS)
    pipette_count = int(labels.get("pipette", 0) or 0)
    if interaction_count:
        score += min(2.0, 0.75 + 0.25 * interaction_count)
    if bool(row.get("is_active") or row.get("is_experiment_active")):
        score += 0.35
    if detection_count:
        score += min(0.4, 0.04 * detection_count)
    if label_diversity >= 4:
        score += min(0.6, 0.12 * label_diversity)
    if pipette_count >= 5 and not interaction_count and not hand_count and label_diversity <= 3:
        score -= min(0.45, 0.05 * pipette_count)
    return max(0.0, float(score))


def _row_has_experiment_window_activity(row: Mapping[str, Any]) -> bool:
    if _row_has_hand_object_interaction(dict(row)):
        return True
    if bool(row.get("is_active") or row.get("is_experiment_active")):
        return True
    labels = set(_row_detection_labels(dict(row)))
    if labels & _YOLO_HAND_LABELS and labels & _YOLO_OPERABLE_OBJECT_LABELS:
        return True
    return _fast_locate_row_seed_score(dict(row)) >= _experiment_window_activity_min_score()


def _experiment_window_sample_period(times: list[float]) -> float:
    ordered = sorted(set(round(value, 6) for value in times))
    gaps = [
        ordered[index + 1] - ordered[index]
        for index in range(len(ordered) - 1)
        if ordered[index + 1] > ordered[index]
    ]
    if not gaps:
        return 1.0
    return max(0.1, min(10.0, sorted(gaps)[len(gaps) // 2]))


def _session_duration_limit_for_experiment_windows(
    manifest: SessionManifest,
    yolo_rows: list[Mapping[str, Any]],
    segments: list[DetectedSegment],
    *,
    dry_run: bool,
) -> float | None:
    if dry_run:
        return max(1.0, _float_env_value("KEY_ACTION_DRY_RUN_DURATION_SEC", 960.0))
    durations = [
        float(duration)
        for duration in (
            _source_video_duration_limit_sec(source)
            for source in manifest.videos.all_sources().values()
        )
        if duration is not None and duration > 0
    ]
    if durations:
        return max(durations)
    ends: list[float] = []
    for segment in segments:
        ends.append(_detected_segment_end(segment))
    for row in yolo_rows:
        if isinstance(row, Mapping):
            ends.append(_row_alignment_sec(dict(row)))
    return max(ends) if ends else None


def _activity_windows_from_yolo_activity_rows(
    yolo_rows: list[Mapping[str, Any]],
    *,
    duration_limit_sec: float | None,
    allowed_intervals: list[tuple[float, float]] | None = None,
    allowed_attach_gap_sec: float = 0.0,
) -> list[dict[str, Any]]:
    timed_rows: list[tuple[float, Mapping[str, Any]]] = []
    for row in yolo_rows:
        if not isinstance(row, Mapping) or not _row_has_experiment_window_activity(row):
            continue
        time_sec = _row_alignment_sec(dict(row))
        if allowed_intervals:
            in_allowed_interval = any(
                float(start) - float(allowed_attach_gap_sec)
                <= float(time_sec)
                <= float(end) + float(allowed_attach_gap_sec)
                for start, end in allowed_intervals
            )
            if not in_allowed_interval:
                continue
        timed_rows.append((max(0.0, float(time_sec)), row))
    if not timed_rows:
        return []
    timed_rows.sort(key=lambda item: item[0])
    sample_period = _experiment_window_sample_period([time_sec for time_sec, _row in timed_rows])
    silence_gap_sec = _experiment_window_silence_gap_sec()
    groups: list[list[tuple[float, Mapping[str, Any]]]] = []
    current: list[tuple[float, Mapping[str, Any]]] = []
    last_time: float | None = None
    for item in timed_rows:
        time_sec = item[0]
        if current and last_time is not None and time_sec - last_time > silence_gap_sec:
            groups.append(current)
            current = []
        current.append(item)
        last_time = time_sec
    if current:
        groups.append(current)

    windows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start_sec = max(0.0, min(time_sec for time_sec, _row in group) - sample_period)
        end_sec = max(time_sec for time_sec, _row in group) + sample_period
        if duration_limit_sec is not None and duration_limit_sec > 0:
            end_sec = min(end_sec, float(duration_limit_sec))
        windows.append(
            {
                "window_id": f"activity_window_{index:06d}",
                "source": "yolo_activity_rows",
                "start_sec": round(start_sec, 6),
                "end_sec": round(max(end_sec, start_sec + 0.1), 6),
                "row_count": len(group),
                "sample_period_sec": round(sample_period, 6),
                "silence_gap_sec": round(silence_gap_sec, 6),
            }
        )
    return windows


def _dual_view_activity_aligned_segments_from_yolo_rows(
    manifest: SessionManifest,
    yolo_rows: list[Mapping[str, Any]],
    *,
    duration_limit_sec: float | None,
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if manifest.videos.first_person is None:
        return [], {
            "schema_version": "dual_view_activity_alignment_summary.v1",
            "enabled": False,
            "reason": "single_view_manifest",
        }
    if not _bool_env("KEY_ACTION_ALLOW_DUAL_VIEW_ACTIVITY_EPISODES", True):
        return [], {
            "schema_version": "dual_view_activity_alignment_summary.v1",
            "enabled": False,
            "reason": "disabled_by_env",
        }
    rows_by_view: dict[str, list[Mapping[str, Any]]] = {"first_person": [], "third_person": []}
    for row in yolo_rows:
        if not isinstance(row, Mapping):
            continue
        view = str(row.get("source_view") or row.get("view") or "").strip()
        if view in rows_by_view:
            rows_by_view[view].append(row)
    first_lifecycle_windows = _lifecycle_windows_from_rows(
        manifest,
        rows_by_view["first_person"],
        duration_limit_sec,
    )
    third_lifecycle_windows = _lifecycle_windows_from_rows(
        manifest,
        rows_by_view["third_person"],
        duration_limit_sec,
    )
    first_windows = first_lifecycle_windows or _activity_windows_from_yolo_activity_rows(
        rows_by_view["first_person"],
        duration_limit_sec=duration_limit_sec,
    )
    third_windows = third_lifecycle_windows or _activity_windows_from_yolo_activity_rows(
        rows_by_view["third_person"],
        duration_limit_sec=duration_limit_sec,
    )
    window_source = (
        "dual_view_lifecycle_overlap"
        if first_lifecycle_windows and third_lifecycle_windows
        else "dual_view_activity_overlap"
    )
    overlap_min_sec = _float_env_any_value(
        ("KEY_ACTION_DUAL_VIEW_ACTIVITY_MIN_OVERLAP_SEC",),
        3.0,
        minimum=0.0,
    )
    min_segment_sec = _float_env_any_value(
        ("KEY_ACTION_DUAL_VIEW_ACTIVITY_MIN_SEGMENT_SEC",),
        _experiment_window_min_duration_sec(),
        minimum=0.0,
    )
    max_gap_sec = _float_env_any_value(
        ("KEY_ACTION_DUAL_VIEW_ACTIVITY_MAX_GAP_SEC",),
        2.0,
        minimum=0.0,
    )
    candidates: list[dict[str, Any]] = []
    for first in first_windows:
        first_start = float(first.get("start_sec") or 0.0)
        first_end = float(first.get("end_sec") or first_start)
        for third in third_windows:
            third_start = float(third.get("start_sec") or 0.0)
            third_end = float(third.get("end_sec") or third_start)
            overlap = min(first_end, third_end) - max(first_start, third_start)
            gap = max(first_start, third_start) - min(first_end, third_end)
            if overlap < overlap_min_sec and gap > max_gap_sec:
                continue
            if overlap >= overlap_min_sec:
                start_sec = max(first_start, third_start)
                end_sec = min(first_end, third_end)
            else:
                start_sec = max(0.0, min(first_start, third_start))
                end_sec = max(first_end, third_end)
            candidates.append(
                {
                    "start_sec": start_sec,
                    "end_sec": max(start_sec + 0.1, end_sec),
                    "overlap_sec": max(0.0, overlap),
                    "gap_sec": max(0.0, gap),
                    "first_window_id": first.get("window_id"),
                    "third_window_id": third.get("window_id"),
                    "first_row_count": int(first.get("row_count") or 0),
                    "third_row_count": int(third.get("row_count") or 0),
                    "first_source": first.get("source"),
                    "third_source": third.get("source"),
                }
            )
    if not candidates:
        return [], {
            "schema_version": "dual_view_activity_alignment_summary.v1",
            "enabled": True,
            "decision": "no_dual_view_activity_overlap",
            "first_window_count": len(first_windows),
            "third_window_count": len(third_windows),
            "overlap_min_sec": overlap_min_sec,
            "max_gap_sec": max_gap_sec,
        }
    candidates.sort(key=lambda item: (float(item["start_sec"]), float(item["end_sec"])))
    merged: list[dict[str, Any]] = []
    merge_gap_sec = _float_env_any_value(
        ("KEY_ACTION_DUAL_VIEW_ACTIVITY_MERGE_GAP_SEC",),
        max(60.0, _experiment_window_silence_gap_sec()),
        minimum=0.0,
    )
    for item in candidates:
        if not merged or float(item["start_sec"]) - float(merged[-1]["end_sec"]) > merge_gap_sec:
            merged.append(dict(item))
            continue
        merged[-1]["end_sec"] = max(float(merged[-1]["end_sec"]), float(item["end_sec"]))
        merged[-1]["overlap_sec"] = float(merged[-1].get("overlap_sec") or 0.0) + float(item.get("overlap_sec") or 0.0)
        merged[-1]["first_row_count"] = int(merged[-1].get("first_row_count") or 0) + int(item.get("first_row_count") or 0)
        merged[-1]["third_row_count"] = int(merged[-1].get("third_row_count") or 0) + int(item.get("third_row_count") or 0)

    segments: list[DetectedSegment] = []
    dropped_short: list[dict[str, Any]] = []
    for index, item in enumerate(merged, start=1):
        start_sec = max(0.0, float(item["start_sec"]))
        end_sec = max(start_sec + 0.1, float(item["end_sec"]))
        if duration_limit_sec is not None and duration_limit_sec > 0:
            end_sec = min(end_sec, float(duration_limit_sec))
        duration_sec = max(0.0, end_sec - start_sec)
        if duration_sec < min_segment_sec:
            dropped_short.append(
                {
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "duration_sec": round(duration_sec, 6),
                    "reason": "dual_view_activity_overlap_shorter_than_min_segment",
                    "min_segment_sec": round(min_segment_sec, 6),
                    "first_row_count": int(item.get("first_row_count") or 0),
                    "third_row_count": int(item.get("third_row_count") or 0),
                }
            )
            continue
        row_count = int(item.get("first_row_count") or 0) + int(item.get("third_row_count") or 0)
        segments.append(
            DetectedSegment(
                segment_id=f"dual_activity_{len(segments) + 1:06d}",
                start_sec=round(start_sec, 6),
                end_sec=round(max(start_sec + 0.1, end_sec), 6),
                duration_sec=round(max(0.1, end_sec - start_sec), 6),
                global_start_time=_iso_time(_global_time_from_session_sec(manifest, start_sec)),
                global_end_time=_iso_time(_global_time_from_session_sec(manifest, end_sec)),
                avg_motion_score=0.72,
                avg_active_score=0.72,
                start_reason="dual_view_activity_window_start",
                end_reason="dual_view_activity_window_end",
                review_required=False,
                detector_backend="dual_view_activity_alignment",
                detector_source_view="first_person,third_person",
                yolo_label_counts={},
                yolo_interaction_count=row_count,
                boundary_confidence=0.72,
                boundary_support_count=row_count,
                boundary_source=window_source,
                decision_path="dual_view_activity_alignment",
                decision_trace=[
                    "formal_episode_requires_first_and_third_lifecycle_or_activity_overlap",
                    f"overlap_sec={float(item.get('overlap_sec') or 0.0):.6f}",
                    f"first_row_count={int(item.get('first_row_count') or 0)}",
                    f"third_row_count={int(item.get('third_row_count') or 0)}",
                    "source_views=first_person,third_person",
                    f"window_source={window_source}",
                ],
                fallback_used=False,
                fallback_reason="",
                reason_code="dual_view_activity_aligned",
                raw_score=0.72,
                score=0.72,
                source=window_source,
                source_view="first_person,third_person",
                detector_version="dual_view_activity_alignment.v1",
                final_score=0.72,
                retrieval_boost_factors={
                    "dual_view_activity_alignment": 1.0,
                    "first_row_count": int(item.get("first_row_count") or 0),
                    "third_row_count": int(item.get("third_row_count") or 0),
                },
            )
        )
    summary = {
        "schema_version": "dual_view_activity_alignment_summary.v1",
        "enabled": True,
        "decision": "dual_view_activity_aligned" if segments else "no_dual_view_activity_overlap",
        "first_window_count": len(first_windows),
        "third_window_count": len(third_windows),
        "first_lifecycle_window_count": len(first_lifecycle_windows),
        "third_lifecycle_window_count": len(third_lifecycle_windows),
        "window_source": window_source,
        "candidate_overlap_count": len(candidates),
        "merged_candidate_count": len(merged),
        "segment_count": len(segments),
        "overlap_min_sec": overlap_min_sec,
        "min_segment_sec": min_segment_sec,
        "max_gap_sec": max_gap_sec,
        "merge_gap_sec": merge_gap_sec,
        "dropped_short_count": len(dropped_short),
        "dropped_short_candidates": dropped_short[:50],
        "segments": [
            {
                "segment_id": item.segment_id,
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "duration_sec": item.duration_sec,
            }
            for item in segments
        ],
    }
    return segments, summary


def _coarse_windows_from_detected_segments(segments: list[DetectedSegment]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        start_sec = _detected_segment_start(segment)
        end_sec = max(start_sec + 0.1, _detected_segment_end(segment))
        windows.append(
            {
                "window_id": f"coarse_segment_{index:06d}",
                "source": "coarse_detected_segment",
                "segment_id": str(getattr(segment, "segment_id", "") or ""),
                "start_sec": round(start_sec, 6),
                "end_sec": round(end_sec, 6),
            }
        )
    return windows


def _window_touches_interval(
    window: Mapping[str, Any],
    start_sec: float,
    end_sec: float,
    *,
    attach_gap_sec: float,
) -> bool:
    window_start = float(window.get("start_sec") or 0.0)
    window_end = float(window.get("end_sec") or window_start)
    return start_sec <= window_end + attach_gap_sec and end_sec >= window_start - attach_gap_sec


def _expand_interval_to_min_experiment_window(
    start_sec: float,
    end_sec: float,
    *,
    min_window_sec: float,
    duration_limit_sec: float | None,
) -> tuple[float, float]:
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 0.1, float(end_sec))
    current_duration = end_sec - start_sec
    if current_duration >= min_window_sec:
        return start_sec, end_sec
    target_duration = float(min_window_sec)
    if duration_limit_sec is not None and duration_limit_sec > 0:
        target_duration = min(target_duration, float(duration_limit_sec))
    if target_duration <= current_duration:
        return start_sec, end_sec
    missing = target_duration - current_duration
    expanded_start = max(0.0, start_sec - missing / 2.0)
    expanded_end = end_sec + missing / 2.0
    if duration_limit_sec is not None and duration_limit_sec > 0 and expanded_end > duration_limit_sec:
        overflow = expanded_end - float(duration_limit_sec)
        expanded_end = float(duration_limit_sec)
        expanded_start = max(0.0, expanded_start - overflow)
    if expanded_start <= 0.0:
        expanded_end = max(expanded_end, min(target_duration, float(duration_limit_sec or target_duration)))
    return expanded_start, max(expanded_start + 0.1, expanded_end)


def _compact_experiment_window_refs(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "window_id": item.get("window_id"),
            "source": item.get("source"),
            "segment_id": item.get("segment_id"),
            "start_sec": item.get("start_sec"),
            "end_sec": item.get("end_sec"),
            "row_count": item.get("row_count"),
        }
        for item in windows[:10]
    ]


def _expand_detected_segment_to_activity_window(
    manifest: SessionManifest,
    segment: DetectedSegment,
    *,
    activity_windows: list[dict[str, Any]],
    lifecycle_windows: list[dict[str, Any]],
    coarse_windows: list[dict[str, Any]],
    duration_limit_sec: float | None,
    min_window_sec: float,
    attach_gap_sec: float,
    left_guard_sec: float | None = None,
    right_guard_sec: float | None = None,
    allow_lifecycle_start: bool = True,
    allow_lifecycle_end: bool = True,
) -> tuple[DetectedSegment, dict[str, Any]]:
    original_start = _detected_segment_start(segment)
    original_end = max(original_start + 0.1, _detected_segment_end(segment))
    expanded_start = original_start
    expanded_end = original_end
    matched_activity = [
        window
        for window in activity_windows
        if _window_touches_interval(window, original_start, original_end, attach_gap_sec=attach_gap_sec)
    ]
    max_coarse_fallback_sec = _float_env_any_value(
        ("KEY_ACTION_EXPERIMENT_WINDOW_MAX_COARSE_FALLBACK_SEC",),
        max(float(min_window_sec) * 2.0, 180.0),
        minimum=max(float(min_window_sec), 1.0),
    )
    matched_coarse = [
        window
        for window in coarse_windows
        if _window_touches_interval(window, original_start, original_end, attach_gap_sec=attach_gap_sec)
        and float(window.get("end_sec") or 0.0) - float(window.get("start_sec") or 0.0) <= max_coarse_fallback_sec
    ]
    for window in [*matched_activity, *matched_coarse]:
        expanded_start = min(expanded_start, float(window.get("start_sec") or expanded_start))
        expanded_end = max(expanded_end, float(window.get("end_sec") or expanded_end))
    matched_lifecycle = _lifecycle_window_for_action(
        lifecycle_windows,
        action_start=original_start,
        action_end=original_end,
    )
    lifecycle_start_applied = False
    lifecycle_end_applied = False
    lifecycle_suppressed_reasons: list[str] = []
    if matched_lifecycle:
        lifecycle_start = float(matched_lifecycle.get("selected_start_sec") or expanded_start)
        lifecycle_end = float(matched_lifecycle.get("selected_end_sec") or expanded_end)
        max_lifecycle_prep_sec = _float_env_any_value(
            ("KEY_ACTION_EXPERIMENT_LIFECYCLE_MAX_PREP_SEC",),
            360.0,
            minimum=0.0,
        )
        if allow_lifecycle_start and original_start - lifecycle_start <= max_lifecycle_prep_sec:
            expanded_start = min(expanded_start, lifecycle_start)
            lifecycle_start_applied = True
        else:
            lifecycle_suppressed_reasons.append("lifecycle_start_too_far_or_not_first_action_cluster")
        end_confirmed = bool(matched_lifecycle.get("end_boundary_confirmed"))
        if allow_lifecycle_end and end_confirmed:
            expanded_end = max(expanded_end, lifecycle_end)
            lifecycle_end_applied = True
        elif lifecycle_end > expanded_end:
            lifecycle_suppressed_reasons.append("unconfirmed_lifecycle_end_not_used_for_formal_episode")
    if left_guard_sec is not None:
        expanded_start = max(expanded_start, float(left_guard_sec))
    if right_guard_sec is not None:
        expanded_end = min(expanded_end, float(right_guard_sec))
    expanded_end = max(expanded_end, expanded_start + 0.1)
    expanded_start, expanded_end = _expand_interval_to_min_experiment_window(
        expanded_start,
        expanded_end,
        min_window_sec=min_window_sec,
        duration_limit_sec=duration_limit_sec,
    )
    if left_guard_sec is not None:
        expanded_start = max(expanded_start, float(left_guard_sec))
    if right_guard_sec is not None:
        expanded_end = min(expanded_end, float(right_guard_sec))
    if duration_limit_sec is not None and duration_limit_sec > 0:
        expanded_end = min(expanded_end, float(duration_limit_sec))
    expanded_end = max(expanded_end, expanded_start + 0.1)
    if expanded_end - expanded_start < float(min_window_sec):
        missing = float(min_window_sec) - (expanded_end - expanded_start)
        right_limit = float(right_guard_sec) if right_guard_sec is not None else None
        if duration_limit_sec is not None and duration_limit_sec > 0:
            right_limit = min(right_limit, float(duration_limit_sec)) if right_limit is not None else float(duration_limit_sec)
        if right_limit is None or expanded_end + missing <= right_limit + 1e-6:
            expanded_end += missing
        elif left_guard_sec is None:
            expanded_start = max(0.0, expanded_start - missing)
        expanded_end = max(expanded_end, expanded_start + 0.1)
    expanded = abs(expanded_start - original_start) > 1e-6 or abs(expanded_end - original_end) > 1e-6
    expansion = {
        "schema_version": "official_experiment_window_expansion.v1",
        "expanded": bool(expanded),
        "source": (
            "experiment_lifecycle_state_and_action_evidence"
            if matched_lifecycle
            else "coarse_activity_evidence_and_min_duration"
        ),
        "segment_id": str(getattr(segment, "segment_id", "") or ""),
        "original_start_sec": round(original_start, 6),
        "original_end_sec": round(original_end, 6),
        "expanded_start_sec": round(expanded_start, 6),
        "expanded_end_sec": round(expanded_end, 6),
        "min_window_sec": round(float(min_window_sec), 6),
        "silence_gap_sec": round(_experiment_window_silence_gap_sec(), 6),
        "attach_gap_sec": round(attach_gap_sec, 6),
        "activity_window_count": len(matched_activity),
        "coarse_window_count": len(matched_coarse),
        "lifecycle_window_count": 1 if matched_lifecycle else 0,
        "activity_windows": _compact_experiment_window_refs(matched_activity),
        "coarse_windows": _compact_experiment_window_refs(matched_coarse),
        "lifecycle_window": (
            {
                "window_id": matched_lifecycle.get("window_id"),
                "start_sec": matched_lifecycle.get("selected_start_sec"),
                "end_sec": matched_lifecycle.get("selected_end_sec"),
                "start_reason": matched_lifecycle.get("start_reason"),
                "end_reason": matched_lifecycle.get("end_reason"),
                "start_boundary_confirmed": matched_lifecycle.get("start_boundary_confirmed"),
                "end_boundary_confirmed": matched_lifecycle.get("end_boundary_confirmed"),
                "start_applied": lifecycle_start_applied,
                "end_applied": lifecycle_end_applied,
                "suppressed_reasons": lifecycle_suppressed_reasons,
                "source_views": matched_lifecycle.get("source_views"),
            }
            if matched_lifecycle
            else None
        ),
        "window_guards": {
            "left_guard_sec": round(float(left_guard_sec), 6) if left_guard_sec is not None else None,
            "right_guard_sec": round(float(right_guard_sec), 6) if right_guard_sec is not None else None,
            "reason": "prevent_lifecycle_or_activity_expansion_from_crossing_adjacent_formal_action_cluster",
        },
    }
    if not expanded:
        return segment, expansion

    expanded_segment = replace(
        segment,
        start_sec=round(expanded_start, 6),
        end_sec=round(max(expanded_end, expanded_start + 0.1), 6),
        duration_sec=round(max(0.1, expanded_end - expanded_start), 6),
        global_start_time=_iso_time(_global_time_from_session_sec(manifest, expanded_start)),
        global_end_time=_iso_time(_global_time_from_session_sec(manifest, expanded_end)),
        start_reason="coarse_activity_experiment_window_start",
        end_reason="coarse_activity_experiment_window_end",
        boundary_source="coarse_activity_experiment_window",
    )
    expanded_segment.decision_trace = [
        *list(getattr(segment, "decision_trace", []) or []),
        "official_window_expanded=True",
        f"official_window_min_sec={float(min_window_sec):.6f}",
        (
            "official_window_source=experiment_lifecycle_state_and_action_evidence"
            if matched_lifecycle
            else "official_window_source=coarse_activity_evidence_and_min_duration"
        ),
    ]
    retrieval_boost_factors = dict(getattr(segment, "retrieval_boost_factors", {}) or {})
    retrieval_boost_factors["experiment_window_expansion"] = expansion
    expanded_segment.retrieval_boost_factors = retrieval_boost_factors
    return expanded_segment, expansion


def _expand_official_experiment_segments_to_activity_windows(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    *,
    yolo_rows: list[Mapping[str, Any]],
    coarse_segments: list[DetectedSegment],
    dry_run: bool,
) -> tuple[list[DetectedSegment], dict[str, Any]]:
    if not segments:
        return [], {
            "schema_version": "official_experiment_window_expansion_summary.v1",
            "enabled": True,
            "input_count": 0,
            "expanded_count": 0,
        }
    if (
        not _bool_env("KEY_ACTION_EXPAND_DUAL_VIEW_ACTIVITY_EPISODES", False)
        and all(_detected_segment_is_dual_view_activity_episode(segment) for segment in segments)
    ):
        return segments, {
            "schema_version": "official_experiment_window_expansion_summary.v1",
            "enabled": False,
            "input_count": len(segments),
            "expanded_count": 0,
            "output_count": len(segments),
            "reason": "dual_view_activity_episode_boundaries_are_already_aligned",
        }
    if not _bool_env("KEY_ACTION_EXPERIMENT_WINDOW_EXPANSION", True):
        return segments, {
            "schema_version": "official_experiment_window_expansion_summary.v1",
            "enabled": False,
            "input_count": len(segments),
            "expanded_count": 0,
        }
    all_segments_for_limit = [*segments, *coarse_segments]
    duration_limit = _session_duration_limit_for_experiment_windows(
        manifest,
        yolo_rows,
        all_segments_for_limit,
        dry_run=dry_run,
    )
    attach_gap_sec = _experiment_window_attach_gap_sec()
    activity_attach_gap_sec = max(
        attach_gap_sec,
        _float_env_value("KEY_ACTION_EXPERIMENT_WINDOW_ACTIVITY_CONTEXT_SEC", 45.0),
    )
    segment_intervals = [
        (_detected_segment_start(segment), _detected_segment_end(segment))
        for segment in segments
    ]
    activity_windows = _activity_windows_from_yolo_activity_rows(
        yolo_rows,
        duration_limit_sec=duration_limit,
        allowed_intervals=segment_intervals,
        allowed_attach_gap_sec=activity_attach_gap_sec,
    )
    lifecycle_windows = _lifecycle_windows_from_rows(manifest, yolo_rows, duration_limit)
    if duration_limit is None and lifecycle_windows:
        duration_limit = max(float(item.get("end_sec") or 0.0) for item in lifecycle_windows)
    coarse_windows = _coarse_windows_from_detected_segments(coarse_segments)
    min_window_sec = _experiment_window_min_duration_sec()
    ordered_segments = sorted(segments, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
    lifecycle_first_segment_ids: set[str] = set()
    lifecycle_last_segment_ids: set[str] = set()
    lifecycle_segment_members: dict[str, list[DetectedSegment]] = {}
    for segment in ordered_segments:
        matched_lifecycle = _lifecycle_window_for_action(
            lifecycle_windows,
            action_start=_detected_segment_start(segment),
            action_end=max(_detected_segment_start(segment) + 0.1, _detected_segment_end(segment)),
        )
        lifecycle_id = str((matched_lifecycle or {}).get("window_id") or "")
        if lifecycle_id:
            lifecycle_segment_members.setdefault(lifecycle_id, []).append(segment)
    for members in lifecycle_segment_members.values():
        ordered_members = sorted(members, key=lambda item: (_detected_segment_start(item), _detected_segment_end(item)))
        if ordered_members:
            lifecycle_first_segment_ids.add(str(getattr(ordered_members[0], "segment_id", "")))
            lifecycle_last_segment_ids.add(str(getattr(ordered_members[-1], "segment_id", "")))

    expanded_segments: list[DetectedSegment] = []
    expansions: list[dict[str, Any]] = []
    for index, segment in enumerate(ordered_segments):
        previous_segment = ordered_segments[index - 1] if index > 0 else None
        next_segment = ordered_segments[index + 1] if index + 1 < len(ordered_segments) else None
        left_guard_sec = None
        right_guard_sec = None
        if previous_segment is not None:
            previous_end = max(_detected_segment_start(previous_segment) + 0.1, _detected_segment_end(previous_segment))
            current_start = _detected_segment_start(segment)
            if current_start > previous_end:
                left_guard_sec = previous_end + (current_start - previous_end) / 2.0
        if next_segment is not None:
            current_end = max(_detected_segment_start(segment) + 0.1, _detected_segment_end(segment))
            next_start = _detected_segment_start(next_segment)
            if next_start > current_end:
                right_guard_sec = current_end + (next_start - current_end) / 2.0
        segment_id = str(getattr(segment, "segment_id", ""))
        expanded_segment, expansion = _expand_detected_segment_to_activity_window(
            manifest,
            segment,
            activity_windows=activity_windows,
            lifecycle_windows=lifecycle_windows,
            coarse_windows=coarse_windows,
            duration_limit_sec=duration_limit,
            min_window_sec=min_window_sec,
            attach_gap_sec=attach_gap_sec,
            left_guard_sec=left_guard_sec,
            right_guard_sec=right_guard_sec,
            allow_lifecycle_start=segment_id in lifecycle_first_segment_ids or not lifecycle_segment_members,
            allow_lifecycle_end=segment_id in lifecycle_last_segment_ids or not lifecycle_segment_members,
        )
        expanded_segments.append(expanded_segment)
        expansions.append(expansion)
    merged_segments, lifecycle_merge_summary = _merge_same_lifecycle_expanded_experiment_windows(
        manifest,
        expanded_segments,
        expansions,
    )
    return merged_segments, {
        "schema_version": "official_experiment_window_expansion_summary.v1",
        "enabled": True,
        "input_count": len(segments),
        "expanded_count": sum(1 for item in expansions if item.get("expanded")),
        "output_count": len(merged_segments),
        "min_window_sec": round(float(min_window_sec), 6),
        "silence_gap_sec": round(_experiment_window_silence_gap_sec(), 6),
        "attach_gap_sec": round(attach_gap_sec, 6),
        "activity_context_sec": round(activity_attach_gap_sec, 6),
        "activity_window_count": len(activity_windows),
        "lifecycle_window_count": len(lifecycle_windows),
        "coarse_window_count": len(coarse_windows),
        "duration_limit_sec": round(float(duration_limit), 6) if duration_limit is not None else None,
        "expanded_segments": expansions[:50],
        "same_lifecycle_merge": lifecycle_merge_summary,
    }


def _detected_segment_is_dual_view_activity_episode(segment: DetectedSegment) -> bool:
    fields = (
        getattr(segment, "detector_backend", ""),
        getattr(segment, "decision_path", ""),
        getattr(segment, "reason_code", ""),
        getattr(segment, "boundary_source", ""),
        getattr(segment, "detector_version", ""),
        getattr(segment, "source", ""),
    )
    text = " ".join(str(value or "").lower() for value in fields)
    if "dual_view_activity" in text:
        return True
    if "dual_view_lifecycle_overlap" in text:
        return True
    return False


def _row_inside_detected_segments(row_time: float, detected_segments: list[Any]) -> bool:
    if not detected_segments:
        return True
    for segment in detected_segments:
        try:
            if float(segment.start_sec) - 0.001 <= row_time <= float(segment.end_sec) + 0.001:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _fast_locate_fine_seed_cluster_gap_sec(config: DetectorConfig) -> float:
    coarse_period = 1.0 / max(_coarse_yolo_sample_fps(config), 0.001)
    default_gap = min(120.0, max(30.0, coarse_period * 1.5))
    return max(0.5, _float_env_value("KEY_ACTION_FAST_LOCATE_FINE_SEED_CLUSTER_GAP_SEC", default_gap))


def _fast_locate_proxy_fine_seed_cluster_gap_sec(config: DetectorConfig) -> float:
    base_gap = _fast_locate_fine_seed_cluster_gap_sec(config)
    return max(0.5, _float_env_value("KEY_ACTION_FAST_LOCATE_PROXY_FINE_SEED_CLUSTER_GAP_SEC", min(base_gap, 45.0)))


def _fast_locate_fine_window_pad_sec(config: DetectorConfig) -> float:
    coarse_period = 1.0 / max(_coarse_yolo_sample_fps(config), 0.001)
    default_pad = min(60.0, max(12.0, coarse_period * 0.75))
    return max(0.0, _float_env_value("KEY_ACTION_FAST_LOCATE_FINE_WINDOW_PAD_SEC", default_pad))


def _fast_locate_row_uses_analysis_proxy(row: dict[str, Any]) -> bool:
    video_path = str(row.get("video_path") or row.get("source_path") or "").replace("\\", "/").lower()
    if "analysis_proxy" in video_path:
        return True
    for key in (
        "analysis_proxy",
        "analysis_proxy_used",
        "proxy_used",
        "source_is_analysis_proxy",
        "from_analysis_proxy",
    ):
        value = row.get(key)
        if isinstance(value, str):
            if value.strip().lower() in {"1", "true", "yes", "on"}:
                return True
        elif bool(value):
            return True
    source_kind = str(row.get("source_kind") or row.get("scan_source") or row.get("source_role") or "").lower()
    return "analysis_proxy" in source_kind or source_kind == "proxy"


def _fast_locate_proxy_fine_window_pads(base_pad: float) -> tuple[float, float]:
    pre_pad = max(
        float(base_pad),
        _float_env_value("KEY_ACTION_FAST_LOCATE_PROXY_FINE_WINDOW_PRE_PAD_SEC", 120.0),
    )
    post_pad = max(
        float(base_pad),
        _float_env_value("KEY_ACTION_FAST_LOCATE_PROXY_FINE_WINDOW_POST_PAD_SEC", 45.0),
    )
    return pre_pad, post_pad


def _fast_locate_max_fine_window_sec() -> float:
    return max(10.0, _float_env_value("KEY_ACTION_FAST_LOCATE_MAX_FINE_WINDOW_SEC", 120.0))


def _fast_locate_max_total_fine_scan_sec(window_count: int | None = None) -> float:
    configured = max(0.0, _float_env_value("KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC", 420.0))
    adaptive_env = os.environ.get("KEY_ACTION_FAST_LOCATE_ADAPTIVE_TOTAL_FINE_SCAN_BUDGET")
    max_total_configured = "KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC" in os.environ
    adaptive_enabled = _bool_env("KEY_ACTION_FAST_LOCATE_ADAPTIVE_TOTAL_FINE_SCAN_BUDGET", True)
    if max_total_configured and adaptive_env is None:
        return max(10.0, configured)
    if max_total_configured and not adaptive_enabled:
        return max(10.0, configured)
    if not adaptive_enabled:
        return max(10.0, configured)
    if window_count is None or window_count <= 0:
        return max(10.0, configured)
    max_window_sec = _fast_locate_max_fine_window_sec()
    per_window_sec = max(
        max_window_sec,
        _float_env_value("KEY_ACTION_FAST_LOCATE_FINE_SCAN_BUDGET_PER_WINDOW_SEC", max_window_sec),
    )
    min_budget_sec = max(
        configured,
        _float_env_value("KEY_ACTION_FAST_LOCATE_MIN_TOTAL_FINE_SCAN_SEC", 450.0),
    )
    hard_cap_sec = max(
        min_budget_sec,
        _float_env_value("KEY_ACTION_FAST_LOCATE_MAX_ADAPTIVE_TOTAL_FINE_SCAN_SEC", 21600.0),
    )
    adaptive_budget = min(hard_cap_sec, max(min_budget_sec, float(window_count) * per_window_sec))
    return max(10.0, adaptive_budget)


def _fast_locate_keep_all_fine_windows() -> bool:
    return _bool_env("KEY_ACTION_FAST_LOCATE_KEEP_ALL_FINE_WINDOWS", False)


def _split_and_limit_fast_locate_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_window_sec = _fast_locate_max_fine_window_sec()
    split_windows: list[dict[str, Any]] = []
    for window in windows:
        start = float(window["start_sec"])
        end = max(start, float(window["end_sec"]))
        duration = end - start
        if duration <= max_window_sec:
            split_windows.append(dict(window))
            continue
        cursor = start
        split_index = 0
        while cursor < end - 0.001:
            chunk_end = min(end, cursor + max_window_sec)
            item = dict(window)
            item["start_sec"] = cursor
            item["end_sec"] = chunk_end
            item["split_index"] = split_index
            split_windows.append(item)
            cursor = chunk_end
            split_index += 1

    for window in split_windows:
        window["duration_sec"] = round(max(0.0, float(window["end_sec"]) - float(window["start_sec"])), 6)
    if _fast_locate_keep_all_fine_windows():
        return sorted(split_windows, key=lambda item: float(item["start_sec"]))

    total_budget = _fast_locate_max_total_fine_scan_sec(len(split_windows))
    if sum(float(item.get("duration_sec") or 0.0) for item in split_windows) <= total_budget:
        return sorted(split_windows, key=lambda item: float(item["start_sec"]))

    selected: list[dict[str, Any]] = []
    used = 0.0
    if _bool_env("KEY_ACTION_FAST_LOCATE_BALANCED_FINE_WINDOW_SELECTION", True):
        bucket_sec = max(max_window_sec, _float_env_value("KEY_ACTION_FAST_LOCATE_FINE_WINDOW_SELECTION_BUCKET_SEC", 600.0))
        near_best_ratio = max(
            0.0,
            min(1.0, _float_env_value("KEY_ACTION_FAST_LOCATE_FINE_WINDOW_SELECTION_NEAR_BEST_RATIO", 0.75)),
        )
        buckets: dict[int, list[dict[str, Any]]] = {}
        for window in sorted(split_windows, key=lambda item: (-float(item.get("score") or 0.0), float(item["start_sec"]))):
            center = (float(window["start_sec"]) + float(window["end_sec"])) / 2.0
            bucket = int(center // bucket_sec)
            buckets.setdefault(bucket, []).append(window)
        ranked_buckets = sorted(
            buckets.items(),
            key=lambda item: (-max(float(window.get("score") or 0.0) for window in item[1]), item[0]),
        )
        for _bucket, bucket_windows in ranked_buckets:
            if used >= total_budget:
                break
            best_score = max(float(window.get("score") or 0.0) for window in bucket_windows)
            preferred_windows = sorted(
                [
                    window
                    for window in bucket_windows
                    if best_score <= 0.0 or float(window.get("score") or 0.0) >= best_score * near_best_ratio
                ],
                key=lambda item: float(item["start_sec"]),
            )
            for window in preferred_windows or bucket_windows:
                duration = float(window.get("duration_sec") or 0.0)
                if used + duration > total_budget and selected:
                    continue
                selected.append(window)
                used += duration
                break
        selected_keys = {(float(item["start_sec"]), float(item["end_sec"])) for item in selected}
        for window in sorted(split_windows, key=lambda item: (-float(item.get("score") or 0.0), float(item["start_sec"]))):
            if used >= total_budget:
                break
            key = (float(window["start_sec"]), float(window["end_sec"]))
            if key in selected_keys:
                continue
            duration = float(window.get("duration_sec") or 0.0)
            if used + duration > total_budget and selected:
                continue
            selected.append(window)
            selected_keys.add(key)
            used += duration
        return sorted(selected, key=lambda item: float(item["start_sec"]))

    for window in sorted(split_windows, key=lambda item: (-float(item.get("score") or 0.0), float(item["start_sec"]))):
        duration = float(window.get("duration_sec") or 0.0)
        if used + duration > total_budget and selected:
            continue
        selected.append(window)
        used += duration
        if used >= total_budget:
            break
    return sorted(selected, key=lambda item: float(item["start_sec"]))


def _fast_locate_fine_window_segments_from_yolo_rows(
    manifest: SessionManifest,
    detected_segments: list[Any],
    yolo_rows: list[dict[str, Any]],
    config: DetectorConfig,
) -> list[DetectedSegment]:
    proxy_source_mode = any(_fast_locate_row_uses_analysis_proxy(row) for row in yolo_rows if isinstance(row, dict))
    min_seed_score = _float_env_value(
        "KEY_ACTION_FAST_LOCATE_FINE_SEED_MIN_SCORE",
        max(0.05, float(config.end_threshold or 0.0)),
    )
    require_interaction_seed = _bool_env("KEY_ACTION_FAST_LOCATE_FINE_SEED_REQUIRE_INTERACTION", False)
    if proxy_source_mode and _bool_env("KEY_ACTION_FAST_LOCATE_PROXY_FINE_SEED_REQUIRE_INTERACTION", True):
        require_interaction_seed = True
    require_coarse_segment_seed = _bool_env("KEY_ACTION_FAST_LOCATE_FINE_SEED_REQUIRE_COARSE_SEGMENT", False)
    seed_rows: list[dict[str, Any]] = []
    for row in yolo_rows:
        if not isinstance(row, dict):
            continue
        row_time = _row_alignment_sec(row)
        if require_coarse_segment_seed and not _row_inside_detected_segments(row_time, detected_segments):
            continue
        has_interaction = _row_has_hand_object_interaction(row)
        if require_interaction_seed and not has_interaction:
            continue
        score = _fast_locate_row_seed_score(row)
        if score < min_seed_score and not has_interaction:
            continue
        seed = dict(row)
        seed["_seed_time_sec"] = row_time
        seed["_seed_score"] = score
        seed_rows.append(seed)

    if not seed_rows:
        fallback_windows = []
        pad = _fast_locate_fine_window_pad_sec(config)
        pre_pad = post_pad = pad
        if proxy_source_mode:
            pre_pad, post_pad = _fast_locate_proxy_fine_window_pads(pad)
        max_window_sec = _fast_locate_max_fine_window_sec()
        for segment in detected_segments:
            try:
                start = float(segment.start_sec)
                end = float(segment.end_sec)
            except (TypeError, ValueError):
                continue
            if end <= start:
                continue
            if end - start > max_window_sec:
                center = start + (end - start) / 2.0
                start = max(0.0, center - max_window_sec / 2.0)
                end = start + max_window_sec
            fallback_windows.append(
                {
                    "start_sec": max(0.0, start - pre_pad),
                    "end_sec": end + post_pad,
                    "score": float(getattr(segment, "final_score", 0.0) or getattr(segment, "avg_active_score", 0.0) or 0.0),
                    "row_count": 0,
                    "segment_ids": [str(getattr(segment, "segment_id", ""))],
                    "source_views": [str(getattr(segment, "detector_source_view", "third_person"))],
                    "proxy_lookback_applied": bool(pre_pad != pad or post_pad != pad),
                }
            )
        windows = _split_and_limit_fast_locate_windows(fallback_windows)
    else:
        pad = _fast_locate_fine_window_pad_sec(config)
        proxy_seed_mode = any(_fast_locate_row_uses_analysis_proxy(row) for row in seed_rows)
        cluster_gap_sec = (
            _fast_locate_proxy_fine_seed_cluster_gap_sec(config)
            if proxy_seed_mode
            else _fast_locate_fine_seed_cluster_gap_sec(config)
        )
        pre_pad, post_pad = _fast_locate_proxy_fine_window_pads(pad) if proxy_seed_mode else (pad, pad)
        clusters: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        last_time: float | None = None
        for row in sorted(seed_rows, key=lambda item: float(item["_seed_time_sec"])):
            row_time = float(row["_seed_time_sec"])
            if current and last_time is not None and row_time > last_time + cluster_gap_sec:
                clusters.append(current)
                current = []
            current.append(row)
            last_time = row_time
        if current:
            clusters.append(current)

        windows = []
        for cluster in clusters:
            times = [float(row["_seed_time_sec"]) for row in cluster]
            cluster_start = max(0.0, min(times) - pre_pad)
            cluster_end = max(times) + post_pad
            cluster_source_views = sorted(
                {
                    str(row.get("source_view") or row.get("view") or "")
                    for row in cluster
                    if row.get("source_view") or row.get("view")
                }
            )
            cluster_score = sum(float(row.get("_seed_score") or 0.0) for row in cluster)
            max_window_sec = _fast_locate_max_fine_window_sec()
            slice_start = cluster_start
            while slice_start < cluster_end - 0.001:
                slice_end = min(cluster_end, slice_start + max_window_sec)
                slice_rows = [
                    row
                    for row in cluster
                    if slice_start - 0.001 <= float(row["_seed_time_sec"]) <= slice_end + 0.001
                ]
                if not slice_rows:
                    if proxy_seed_mode and slice_end <= min(times) + 0.001:
                        segment_ids = []
                        for segment in detected_segments:
                            try:
                                if float(segment.end_sec) < slice_start or float(segment.start_sec) > slice_end:
                                    continue
                            except (TypeError, ValueError):
                                continue
                            segment_ids.append(str(getattr(segment, "segment_id", "")))
                        windows.append(
                            {
                                "start_sec": max(0.0, slice_start),
                                "end_sec": slice_end,
                                "score": max(0.01, cluster_score),
                                "row_count": 0,
                                "segment_ids": sorted(set(segment_ids)),
                                "source_views": cluster_source_views or ["third_person"],
                                "proxy_lookback_applied": True,
                                "proxy_window_role": "pre_action_lookback",
                            }
                        )
                    slice_start = slice_end
                    continue
                source_views = sorted(
                    {
                        str(row.get("source_view") or row.get("view") or "")
                        for row in slice_rows
                        if row.get("source_view") or row.get("view")
                    }
                )
                slice_times = [float(row["_seed_time_sec"]) for row in slice_rows]
                segment_ids = []
                for segment in detected_segments:
                    try:
                        if float(segment.end_sec) < min(slice_times) or float(segment.start_sec) > max(slice_times):
                            continue
                    except (TypeError, ValueError):
                        continue
                    segment_ids.append(str(getattr(segment, "segment_id", "")))
                windows.append(
                    {
                        "start_sec": max(0.0, slice_start),
                        "end_sec": slice_end,
                        "score": sum(float(row.get("_seed_score") or 0.0) for row in slice_rows),
                        "row_count": len(slice_rows),
                        "segment_ids": sorted(set(segment_ids)),
                        "source_views": source_views or ["third_person"],
                        "proxy_lookback_applied": bool(proxy_seed_mode and (pre_pad != pad or post_pad != pad)),
                    }
                )
                slice_start = slice_end
        windows = _split_and_limit_fast_locate_windows(windows)

    fine_segments: list[DetectedSegment] = []
    for index, window in enumerate(windows, start=1):
        start_sec = max(0.0, float(window["start_sec"]))
        end_sec = max(start_sec + 0.1, float(window["end_sec"]))
        rows_in_window = [
            row
            for row in seed_rows
            if start_sec - 0.001 <= float(row.get("_seed_time_sec") or 0.0) <= end_sec + 0.001
        ]
        labels = Counter(label for row in rows_in_window for label in _row_detection_labels(row))
        interaction_count = sum(_row_interaction_count(row) for row in rows_in_window)
        active_values = []
        for row in rows_in_window:
            try:
                active_values.append(float(row.get("active_score") or row.get("_seed_score") or 0.0))
            except (TypeError, ValueError):
                continue
        avg_active = sum(active_values) / len(active_values) if active_values else float(window.get("score") or 0.0)
        source_views = list(window.get("source_views") or [])
        segment = DetectedSegment(
            segment_id=f"fine_seed_{index:06d}",
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=end_sec - start_sec,
            global_start_time=_global_time_from_session_sec(manifest, start_sec).isoformat(),
            global_end_time=_global_time_from_session_sec(manifest, end_sec).isoformat(),
            avg_motion_score=avg_active,
            avg_active_score=avg_active,
            start_reason="yolo_coarse_seed_window_start",
            end_reason="yolo_coarse_seed_window_end",
            review_required=False,
            detector_backend="yolo_interaction",
            detector_source_view="global_multiview" if len(source_views) > 1 else (source_views[0] if source_views else "third_person"),
            yolo_label_counts=dict(labels),
            yolo_interaction_count=interaction_count,
            boundary_confidence=min(1.0, 0.45 + min(0.4, 0.02 * len(rows_in_window))),
            boundary_support_count=len(rows_in_window),
            boundary_source="yolo_coarse_seed_fine_window",
            decision_path=DETECTION_DECISION_YOLO_INTERACTION,
            decision_trace=[
                "backend=yolo_interaction",
                "scan_role=fast_locate_fine_window_seed",
                f"row_count={int(window.get('row_count') or len(rows_in_window))}",
                f"score={float(window.get('score') or 0.0):.4f}",
                f"source_views={','.join(source_views) if source_views else 'unknown'}",
                f"source_segments={','.join(str(item) for item in (window.get('segment_ids') or []) if item)}",
                f"proxy_lookback_applied={bool(window.get('proxy_lookback_applied'))}",
                f"proxy_window_role={str(window.get('proxy_window_role') or '')}",
            ],
            reason_code=DECISION_REASON_YOLO_INTERACTION_DETECTED,
            raw_score=avg_active,
            final_score=avg_active,
        )
        fine_segments.append(segment)
    return fine_segments


def _interaction_cluster_segments_from_yolo_rows(
    manifest: SessionManifest,
    yolo_rows: list[dict[str, Any]],
    config: DetectorConfig,
) -> list[DetectedSegment]:
    interaction_rows = [row for row in yolo_rows if isinstance(row, dict) and _row_has_hand_object_interaction(row)]
    if not interaction_rows:
        return []
    cluster_gap_sec = _fast_locate_interaction_cluster_gap_sec()
    min_interaction_rows = _fast_locate_min_cluster_interaction_rows()
    episode_buffer_sec = _fast_locate_episode_buffer_sec()
    sample_period = 1.0 / max(_refined_yolo_sample_fps(config), 0.001)

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    last_time: float | None = None
    for row in sorted(interaction_rows, key=_row_alignment_sec):
        row_time = _row_alignment_sec(row)
        if current and last_time is not None and row_time > last_time + cluster_gap_sec:
            clusters.append(current)
            current = []
        current.append(row)
        last_time = row_time
    if current:
        clusters.append(current)

    segments: list[DetectedSegment] = []
    expected_count = _fast_locate_expected_experiment_count(config)
    for cluster in clusters:
        interaction_count = sum(_row_interaction_count(row) for row in cluster)
        if interaction_count < min_interaction_rows:
            continue
        times = [_row_alignment_sec(row) for row in cluster]
        start_sec = max(0.0, min(times) - episode_buffer_sec)
        end_sec = max(start_sec + sample_period, max(times) + sample_period + episode_buffer_sec)
        duration_sec = max(0.0, end_sec - start_sec)
        if duration_sec < max(0.5, float(config.min_segment_duration_sec or 0.0)):
            continue
        labels = Counter(label for row in cluster for label in _row_detection_labels(row))
        interaction_labels = Counter(label for row in cluster for label in _row_interaction_labels(row))
        active_scores = []
        raw_scores = []
        for row in cluster:
            try:
                active_scores.append(float(row.get("active_score") or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                raw_scores.append(float(row.get("raw_score") or row.get("probability") or 0.0))
            except (TypeError, ValueError):
                pass
        avg_active = sum(active_scores) / len(active_scores) if active_scores else 0.0
        avg_raw = sum(raw_scores) / len(raw_scores) if raw_scores else avg_active
        source_views = sorted({str(row.get("source_view") or row.get("view") or "") for row in cluster if row.get("source_view") or row.get("view")})
        segment = DetectedSegment(
            segment_id=f"seg_{len(segments) + 1:06d}",
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=duration_sec,
            global_start_time=_global_time_from_session_sec(manifest, start_sec),
            global_end_time=_global_time_from_session_sec(manifest, end_sec),
            avg_motion_score=avg_raw,
            avg_active_score=avg_active,
            start_reason="yolo_hand_object_cluster_start",
            end_reason="yolo_hand_object_cluster_gap",
            review_required=False,
            detector_backend="yolo_interaction",
            detector_source_view="global_multiview" if len(source_views) > 1 else (source_views[0] if source_views else "third_person"),
            yolo_label_counts=dict(labels),
            yolo_interaction_count=interaction_count,
            boundary_confidence=min(1.0, 0.55 + min(0.35, 0.01 * interaction_count)),
            boundary_support_count=len(cluster),
            boundary_source="yolo_hand_object_interaction_cluster",
            decision_path=DETECTION_DECISION_YOLO_INTERACTION,
            decision_trace=[
                "backend=yolo_interaction",
                "scan_role=experiment_episode_refine",
                f"cluster_gap_sec={cluster_gap_sec}",
                f"episode_buffer_sec={episode_buffer_sec}",
                f"interaction_rows={len(cluster)}",
                f"interaction_count={interaction_count}",
                f"source_views={','.join(source_views) if source_views else 'unknown'}",
            ],
            reason_code=DECISION_REASON_YOLO_INTERACTION_DETECTED,
            raw_score=avg_raw,
            final_score=avg_active,
            retrieval_boost_factors={
                "interaction_label_counts": dict(interaction_labels),
            },
        )
        segments.append(segment)
    if expected_count is not None and len(segments) < expected_count:
        expected_segments = _expected_count_segments_from_yolo_rows(
            manifest,
            interaction_rows,
            config,
            expected_count,
        )
        if len(expected_segments) == expected_count:
            return expected_segments
    return _coalesce_detected_segments_to_expected_count(
        manifest,
        segments,
        expected_count,
    )


def _refined_experiment_segments_from_yolo_rows(
    manifest: SessionManifest,
    yolo_rows: list[dict[str, Any]],
    config: DetectorConfig,
    *,
    base_segments: list[DetectedSegment] | None = None,
) -> list[Any]:
    if not yolo_rows:
        return []
    if _bool_env("KEY_ACTION_FAST_LOCATE_CLUSTER_INTERACTIONS", True):
        clustered = _interaction_cluster_segments_from_yolo_rows(manifest, yolo_rows, config)
        if clustered:
            if base_segments:
                return _coalesce_refined_with_coarse_macro_episodes(
                    manifest,
                    base_segments,
                    clustered,
                    config,
                    _fast_locate_expected_experiment_count(config),
                )
            return clustered
    from .yolo_detector import build_segments_from_yolo_frame_rows

    sample_fps = _refined_yolo_sample_fps(config)
    sample_period = 1.0 / max(sample_fps, 0.001)
    segment_rows = [
        {
            **row,
            "time_sec": _row_alignment_sec(row),
            "video_start_time": manifest.session_start_time,
            "source_view": "global_multiview",
            "video_path": "global_multiview",
        }
        for row in yolo_rows
        if isinstance(row, dict)
    ]
    if not segment_rows:
        return []
    pseudo_source = VideoSource(
        name="global_multiview",
        path="global_multiview",
        start_time=manifest.session_start_time,
        fps=sample_fps,
        offset_sec=0.0,
    )
    duration_sec = max(float(row.get("time_sec", 0.0) or 0.0) for row in segment_rows) + sample_period
    fine_merge_gap = _fine_merge_gap_sec(config)
    refined_segments = build_segments_from_yolo_frame_rows(
        segment_rows,
        video_source=pseudo_source,
        duration_sec=duration_sec,
        start_threshold=config.start_threshold,
        end_threshold=config.end_threshold,
        start_min_duration_sec=config.start_min_duration_sec,
        end_min_duration_sec=config.end_min_duration_sec,
        merge_gap_sec=fine_merge_gap,
        min_segment_duration_sec=config.min_segment_duration_sec,
        buffer_sec=config.buffer_sec,
    )
    refined_segments = _refine_yolo_detected_segments(manifest, refined_segments, segment_rows, config, duration_sec)
    yolo_source_view = str(yolo_rows[0].get("source_view") or yolo_rows[0].get("view") or "third_person")
    for segment in refined_segments:
        labels, interaction_count = _segment_yolo_stats(segment, segment_rows)
        segment.detector_backend = "yolo_interaction"
        segment.detector_source_view = yolo_source_view
        segment.yolo_label_counts = labels
        segment.yolo_interaction_count = interaction_count
        segment.decision_path = DETECTION_DECISION_YOLO_INTERACTION
        segment.decision_trace = [
            "backend=yolo_interaction",
            "scan_role=micro_refine",
            f"source_view={yolo_source_view}",
            f"sample_fps={sample_fps}",
            f"fine_merge_gap_sec={fine_merge_gap}",
            f"frames={len(segment_rows)}",
        ]
        segment.fallback_used = False
        segment.fallback_reason = ""
        segment.reason_code = DECISION_REASON_YOLO_INTERACTION_DETECTED
        segment.raw_score = _segment_avg_score(segment, segment_rows, "motion_score")
        segment.final_score = _segment_avg_score(segment, segment_rows, "interaction_score")
        if not str(segment.start_reason).startswith("yolo_physical_evidence"):
            segment.start_reason = "yolo_refined_active_score_above_threshold"
        if not str(segment.end_reason).startswith("yolo_physical_evidence"):
            segment.end_reason = "yolo_refined_active_score_below_threshold"
    min_segment_interactions = int(
        float(
            os.environ.get(
                "KEY_ACTION_YOLO_MIN_SEGMENT_INTERACTIONS",
                "1" if bool(getattr(config, "long_video_two_stage_sampling", True)) else "0",
            )
        )
    )
    if min_segment_interactions > 0:
        refined_segments = [
            segment
            for segment in refined_segments
            if int(getattr(segment, "yolo_interaction_count", 0) or 0) >= min_segment_interactions
        ]
    for index, segment in enumerate(refined_segments, start=1):
        segment.segment_id = f"seg_{index:06d}"
    if base_segments:
        return _coalesce_refined_with_coarse_macro_episodes(
            manifest,
            base_segments,
            refined_segments,
            config,
            _fast_locate_expected_experiment_count(config),
        )
    return refined_segments


def _fast_locate_key_segments_from_detected(
    manifest: SessionManifest,
    paths: dict[str, Path],
    detected_segments: list[Any],
    yolo_frame_rows: list[dict[str, Any]],
    run_ctx: RunContext,
    *,
    dry_run: bool,
    include_first_person: bool | None = None,
) -> list[KeyActionSegment]:
    key_segments: list[KeyActionSegment] = []
    for segment in detected_segments:
        key_segment = extract_multiview_clips(
            manifest=manifest,
            segment=segment,
            clips_dir=paths["clips"],
            keyframes_dir=paths["keyframes"],
            yolo_frame_rows=yolo_frame_rows,
            dry_run=dry_run,
            fast_locate_include_first_person=include_first_person,
        )
        key_segment = build_segment_description(key_segment, [])
        key_segment = apply_segment_evidence(key_segment)
        key_segment.detector_backend = str(getattr(segment, "detector_backend", "yolo_interaction"))
        key_segment.detector_source_view = str(getattr(segment, "detector_source_view", "third_person"))
        key_segment.yolo_label_counts = dict(getattr(segment, "yolo_label_counts", {}) or {})
        key_segment.yolo_interaction_count = int(getattr(segment, "yolo_interaction_count", 0) or 0)
        key_segment.decision_path = str(getattr(segment, "decision_path", "yolo.interaction"))
        key_segment.decision_trace = list(getattr(segment, "decision_trace", []))
        key_segment.fallback_used = bool(getattr(segment, "fallback_used", False))
        key_segment.fallback_reason = str(getattr(segment, "fallback_reason", ""))
        key_segment.reason_code = str(getattr(segment, "reason_code", DECISION_REASON_YOLO_INTERACTION_DETECTED))
        key_segment.raw_score = float(getattr(segment, "raw_score", 0.0) or 0.0)
        key_segment.final_score = float(getattr(segment, "final_score", 0.0) or 0.0)
        key_segment.retrieval_boost_factors = dict(getattr(segment, "retrieval_boost_factors", {}) or {})
        key_segment.run_manifest_id = run_ctx.run_id
        key_segments.append(key_segment)
    return key_segments


def _write_fast_locate_frontend_segment_projection(
    paths: dict[str, Path],
    key_segments: list[KeyActionSegment],
    *,
    stage: str,
    micro_rows_ready: bool = False,
) -> list[VectorMetadata]:
    for key_segment in key_segments:
        refresh_segment_chinese_index(key_segment)
    segment_vector_metadata = [_vector_metadata_from_segment(segment) for segment in key_segments]
    write_jsonl(paths["metadata"] / "key_action_segments.jsonl", key_segments)
    write_jsonl(paths["metadata"] / "vector_metadata.jsonl", segment_vector_metadata)
    write_jsonl(paths["metadata"] / "experiment_process_timeline.jsonl", _fast_locate_refined_step_rows(key_segments, []))
    _write_json(
        paths["metadata"] / "fast_locate_frontend_projection_status.json",
        {
            "schema_version": "fast_locate_frontend_projection_status.v1",
            "stage": stage,
            "segment_count": len(key_segments),
            "micro_rows_ready": bool(micro_rows_ready),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return segment_vector_metadata


def _ensure_fast_material_preprocess_defaults() -> None:
    defaults = {
        "KEY_ACTION_PHYSICAL_ACTION_SCOPE": "core_v1",
        "KEY_ACTION_DEFER_SEGMENT_ASSETS": "1",
        "KEY_ACTION_FAST_LOCATE_SKIP_MODEL_INVENTORY": "1",
        "KEY_ACTION_FAST_LOCATE_SEGMENT_INTERACTIONS": "0",
        "KEY_ACTION_FAST_LOCATE_BUILD_MATERIAL_CANDIDATES": "1",
        "KEY_ACTION_FAST_LOCATE_COARSE_SCAN_BOTH_VIEWS": "1",
        "KEY_ACTION_FAST_LOCATE_FINE_SCAN_BOTH_VIEWS": "1",
        "KEY_ACTION_FAST_LOCATE_FULL_DUAL_VIEW_MICRO_SCAN": "0",
        "KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN": "1",
        "KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN_FPS": "3.0",
        "KEY_ACTION_FAST_LOCATE_FINE_SEED_REQUIRE_INTERACTION": "1",
        "KEY_ACTION_FAST_LOCATE_EPISODE_BUFFER_SEC": "18",
        "KEY_ACTION_FAST_LOCATE_EXPERIMENT_MACRO_MERGE_GAP_SEC": "120",
        "KEY_ACTION_FAST_LOCATE_ORPHAN_EXPERIMENT_MERGE_GAP_SEC": "180",
        "KEY_ACTION_FAST_LOCATE_EXPERIMENT_MACRO_ATTACH_GAP_SEC": "45",
        "KEY_ACTION_EXPERIMENT_LIFECYCLE_MAX_PREP_SEC": "360",
        "KEY_ACTION_MERGE_SAME_LIFECYCLE_EXPERIMENT_WINDOWS": "1",
        "KEY_ACTION_SAME_LIFECYCLE_MERGE_MAX_GAP_SEC": "1",
        "KEY_ACTION_FAST_LOCATE_SEMANTIC_MACRO_SPLIT_GAP_SEC": "90",
        "KEY_ACTION_FAST_LOCATE_SPLIT_STIRRER_TO_NONSTIRRER_ORPHANS": "1",
        "KEY_ACTION_FAST_LOCATE_ORPHAN_EXPERIMENT_REQUIRE_THIRD_VIEW": "1",
        "KEY_ACTION_FAST_LOCATE_ORPHAN_EXPERIMENT_MIN_INTERACTIONS": "3",
        "KEY_ACTION_FAST_LOCATE_COARSE_SCAN_WORKERS": "8",
        "KEY_ACTION_FAST_LOCATE_COARSE_SCAN_CHUNKED": "1",
        "KEY_ACTION_FAST_LOCATE_COARSE_SCAN_CHUNK_SEC": "900",
        "KEY_ACTION_FAST_LOCATE_STAGE1_SAMPLE_FPS": "0.2",
        "KEY_ACTION_FAST_LOCATE_COARSE_SEEK_SCAN": "1",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SPARSE_MODE": "chunks",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_WORKERS": "4",
        "KEY_ACTION_FAST_LOCATE_STAGE2_SAMPLE_FPS": "1.5",
        "KEY_ACTION_FAST_LOCATE_FINE_SCAN_WORKERS": "8",
        "KEY_ACTION_FAST_LOCATE_FINE_SCAN_MODEL_MODE": "shared",
        "KEY_ACTION_FAST_LOCATE_STREAMING_FINE_SCAN": "0",
        "KEY_ACTION_FAST_LOCATE_REUSE_STREAMING_FINE_SCAN": "0",
        "KEY_ACTION_YOLO_FINE_SCAN_WORKERS": "8",
        "KEY_ACTION_YOLO_FINE_SCAN_MODEL_MODE": "shared",
        "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_WORKERS": "4",
        "KEY_ACTION_FAST_LOCATE_FINE_WINDOW_PAD_SEC": "18",
        "KEY_ACTION_FAST_LOCATE_MAX_FINE_WINDOW_SEC": "45",
        "KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC": "360",
        "KEY_ACTION_FAST_LOCATE_ADAPTIVE_TOTAL_FINE_SCAN_BUDGET": "1",
        "KEY_ACTION_FAST_LOCATE_FINE_SCAN_BUDGET_PER_WINDOW_SEC": "45",
        "KEY_ACTION_FAST_LOCATE_MAX_ADAPTIVE_TOTAL_FINE_SCAN_SEC": "900",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SCALE_WIDTH": "640",
        "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_SPARSE_MODE": "chunks",
        "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_CHUNK_SEC": "45",
        "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_SCALE_WIDTH": "640",
        "KEY_ACTION_YOLO_FFMPEG_SPARSE_MAX_FPS": "1.25",
        "KEY_ACTION_FAST_LOCATE_FINE_WINDOW_COALESCE_GAP_SEC": "6",
        "KEY_ACTION_FAST_LOCATE_MAX_COALESCED_FINE_WINDOW_SEC": "90",
        "KEY_ACTION_YOLO_FFMPEG_PIPE_SCAN": "1",
        "KEY_ACTION_YOLO_BATCH_SIZE": "16",
        "KEY_ACTION_YOLO_MODEL_CACHE": "1",
        "KEY_ACTION_YOLO_PREDICT_LOCK_SCOPE": "model",
        "KEY_ACTION_ALLOW_DUAL_VIEW_SPARSE_PAIRING": "0",
        "KEY_ACTION_REQUIRE_DUAL_VIEW_TEMPORAL_OVERLAP": "1",
        "KEY_ACTION_DUAL_VIEW_MIN_INTERACTION_ROWS": "1",
        "KEY_ACTION_DUAL_VIEW_MAX_PEAK_DELTA_SEC": "0.75",
        "KEY_ACTION_DUAL_VIEW_ALLOW_OPERATION_CONTEXT_ACTION_MATCH": "0",
        "KEY_ACTION_DUAL_VIEW_ALLOW_OPERATION_CONTEXT_OBJECT_MATCH": "0",
        "KEY_ACTION_FAST_LOCATE_COARSE_VIEW_WORKERS": "0",
        "KEY_ACTION_YOLO_GLOBAL_STAGE_CACHE": "1",
        "KEY_ACTION_FAST_LOCATE_USE_ANALYSIS_PROXY_BY_DEFAULT": "0",
        "KEY_ACTION_FAST_LOCATE_ANALYSIS_PROXY_EXISTING_ONLY": "1",
        "KEY_ACTION_ANALYSIS_PROXY_WIDTH": "640",
        "KEY_ACTION_ANALYSIS_PROXY_FPS": "0.1",
        "KEY_ACTION_ANALYSIS_PROXY_GOP": "1",
        "KEY_ACTION_ANALYSIS_PROXY_CRF": "20",
        "KEY_ACTION_FAST_LOCATE_PROXY_STAGE1_SAMPLE_FPS": "0.0333333",
        "KEY_ACTION_FAST_LOCATE_PROXY_FINE_SEED_REQUIRE_INTERACTION": "1",
        "KEY_ACTION_FAST_LOCATE_PROXY_FINE_SEED_CLUSTER_GAP_SEC": "45",
        "KEY_ACTION_FAST_LOCATE_PROXY_FINE_WINDOW_PRE_PAD_SEC": "45",
        "KEY_ACTION_FAST_LOCATE_PROXY_FINE_WINDOW_POST_PAD_SEC": "20",
        "KEY_ACTION_FAST_LOCATE_MICRO_ASSET_WORKERS": "12",
        "KEY_ACTION_FAST_LOCATE_MICRO_PARENT_WORKERS": "4",
        "KEY_ACTION_FAST_LOCATE_MICRO_CLIP_VIEWS": "all",
        "KEY_ACTION_FAST_LOCATE_MICRO_CLIP_STREAM_COPY": "1",
        "KEY_ACTION_FAST_LOCATE_MAX_MICROS_PER_SEGMENT": "6",
        "KEY_ACTION_FAST_LOCATE_KEYFRAME_ROLES": "peak",
        "KEY_ACTION_FAST_LOCATE_KEYFRAME_BOX_MODE": "strict",
        "KEY_ACTION_FAST_LOCATE_KEYFRAME_DRAW_BOXES": "1",
        "KEY_ACTION_TRUST_MANIFEST_DUAL_VIEW_ALIGNMENT": "0",
        "KEY_ACTION_ALLOW_PAIRED_VIEW_CONTEXT_MATERIAL": "0",
        "KEY_ACTION_AUTO_PUBLISH_PAIRED_VIEW_CONTEXT_MATERIAL": "0",
        "KEY_ACTION_PAIRED_VIEW_CONTEXT_SCENE_GATE": "1",
        "KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS": "1",
        "KEY_ACTION_REQUIRE_RELIABLE_DUAL_VIEW_ALIGNMENT": "1",
        "KEY_ACTION_FAST_LOCATE_MATERIAL_CANDIDATE_RERENDER_BOXES": "0",
        "KEY_ACTION_FAST_LOCATE_MATERIAL_CANDIDATE_WORKERS": "12",
        "KEY_ACTION_MATERIAL_CANDIDATE_WORKERS": "12",
        "KEY_ACTION_FORMAL_MATERIAL_REFERENCES_DUAL_EVENTS_ONLY": "1",
        "KEY_ACTION_MATERIAL_REFERENCE_WORKERS": "8",
        "KEY_ACTION_GENERATE_SEGMENT_PREVIEW_CLIPS": "0",
        "KEY_ACTION_GENERATE_SEGMENT_POSTERS": "0",
        "KEY_ACTION_MATERIAL_HARDLINK_ENABLED": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
    if not _bool_env("KEY_ACTION_PERMIT_RELAXED_FORMAL_DUAL_VIEW_MATCHING", False):
        for key, value in {
            "KEY_ACTION_ALLOW_DUAL_VIEW_SPARSE_PAIRING": "0",
            "KEY_ACTION_REQUIRE_DUAL_VIEW_TEMPORAL_OVERLAP": "1",
            "KEY_ACTION_DUAL_VIEW_MIN_INTERACTION_ROWS": "1",
            "KEY_ACTION_DUAL_VIEW_ALLOW_OPERATION_CONTEXT_ACTION_MATCH": "0",
            "KEY_ACTION_DUAL_VIEW_ALLOW_OPERATION_CONTEXT_OBJECT_MATCH": "0",
            "KEY_ACTION_ALLOW_PAIRED_VIEW_CONTEXT_MATERIAL": "0",
            "KEY_ACTION_AUTO_PUBLISH_PAIRED_VIEW_CONTEXT_MATERIAL": "0",
            "KEY_ACTION_FAST_LOCATE_STAGE2_SAMPLE_FPS": "1.5",
            "KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN_FPS": "3.0",
        }.items():
            os.environ[key] = value


_FAST_MATERIAL_CORE_ACTIONS = {"hand_object_contact", "object_movement", "equipment_panel_operation"}
_FAST_MATERIAL_EXCLUDED_ACTIONS = {"", "experiment_candidate", "experiment_segment", "none", "unknown"}
_FAST_MATERIAL_PANEL_LABELS = {"balance", "scale", "panel", "display", "weigh", "weighing", "天平", "称量"}


def _fast_material_action_texts(row: Mapping[str, Any]) -> list[str]:
    values: list[Any] = [
        row.get("physical_action_type"),
        row.get("canonical_action_type"),
        row.get("canonical_object"),
        row.get("primary_object"),
        row.get("manipulated_object"),
        row.get("action_name"),
        row.get("display_title"),
        row.get("sop_phase"),
        row.get("interaction_family"),
    ]
    for key in ("secondary_objects", "secondary_actions", "objects", "actions"):
        value = row.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)
    texts: list[str] = []
    for value in values:
        text = str(value or "").strip().lower().replace("_", "-")
        if text:
            texts.append(text)
    return texts


def _fast_material_physical_action_type(row: Mapping[str, Any]) -> str:
    raw_action = str(row.get("physical_action_type") or "").strip()
    if raw_action in _FAST_MATERIAL_CORE_ACTIONS:
        return raw_action
    canonical = str(row.get("canonical_action_type") or "").strip().lower().replace("_", "-")
    if canonical.replace("-", "_") in _FAST_MATERIAL_CORE_ACTIONS:
        return canonical.replace("-", "_")
    if canonical in _FAST_MATERIAL_EXCLUDED_ACTIONS:
        return ""
    texts = _fast_material_action_texts(row)
    if any(any(label in text for label in _FAST_MATERIAL_PANEL_LABELS) for text in texts):
        return "equipment_panel_operation"
    if any("object-movement" in text or "movement" in text or "move" in text for text in texts):
        return "object_movement"
    interaction_family = str(row.get("interaction_family") or "").strip().lower()
    if canonical.startswith("hand-") or interaction_family == "hand-object":
        return "hand_object_contact"
    return ""


def _fast_material_is_paired_view_context(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("physical_evidence_mode") or "") == "paired_view_time_alignment"
        or str(row.get("candidate_source") or "")
        in {"paired_view_micro_segment_key_asset_reference", "paired_view_time_alignment"}
        or str(row.get("box_filter") or "") == "paired_view_time_alignment_asset_reference"
    )


def _fast_material_is_strict_yolo_evidence(row: Mapping[str, Any]) -> bool:
    return str(row.get("physical_evidence_mode") or "") == "strict_yolo_physical_evidence"


def _fast_material_is_explicit_dual_event_sparse_evidence(row: Mapping[str, Any]) -> bool:
    if str(row.get("physical_evidence_mode") or "") != "sparse_yolo_interaction_review_required":
        return False
    return bool(str(row.get("dual_event_id") or row.get("dual_view_action_event_id") or "").strip())


def _auto_publish_recommended_material_candidates(session_dir: str | Path, candidate_summary: Mapping[str, Any]) -> dict[str, Any]:
    enabled = _bool_env("KEY_ACTION_FAST_LOCATE_AUTO_PUBLISH_MATERIAL_CANDIDATES", True)
    result: dict[str, Any] = {
        "schema_version": "fast_locate_material_auto_publish.v1",
        "enabled": enabled,
        "status": "skipped",
        "eligible_count": 0,
        "approved_count": 0,
        "quality_policy": "Only recommended candidates that passed YOLO evidence and material quality gates are auto-published.",
    }
    if not enabled:
        result["reason"] = "disabled_by_env"
        return result
    formal_output_gate = _read_json_if_exists(Path(session_dir) / "metadata" / "formal_output_gate.json")
    if str(formal_output_gate.get("status") or "").strip().lower() == "blocked":
        result.update(
            {
                "status": "skipped",
                "reason": "formal_output_gate_blocked",
                "blocked_reason": formal_output_gate.get("blocked_reason"),
                "formal_output_gate": formal_output_gate,
                "policy": "formal materials require passed time-axis and dual-view action gates before publication.",
            }
        )
        return result
    if not isinstance(candidate_summary, Mapping) or candidate_summary.get("status") == "failed":
        result["reason"] = "candidate_generation_unavailable"
        return result
    metadata_dir = Path(session_dir) / "metadata"
    dual_summary_path = metadata_dir / "dual_view_action_alignment_summary.json"
    if dual_summary_path.exists():
        try:
            dual_summary = json.loads(dual_summary_path.read_text(encoding="utf-8"))
        except Exception:
            dual_summary = {}
        dual_event_count = int(float((dual_summary or {}).get("dual_view_action_event_count") or 0))
        formal_allowed = bool((dual_summary or {}).get("formal_results_allowed"))
        if dual_event_count <= 0 or not formal_allowed:
            result.update(
                {
                    "status": "skipped",
                    "reason": "no_confirmed_dual_view_action_events",
                    "dual_view_action_event_count": dual_event_count,
                    "formal_results_allowed": formal_allowed,
                    "decision": (dual_summary or {}).get("decision"),
                    "policy": "formal materials require confirmed first/third action-aligned evidence before publication.",
                }
            )
            return result
    records = candidate_summary.get("records")
    if not isinstance(records, list):
        result["reason"] = "candidate_records_unavailable"
        return result
    eligible_ids: list[str] = []
    eligible_group_ids: set[str] = set()
    for row in records:
        if not isinstance(row, Mapping):
            continue
        if row.get("recommended") is not True or row.get("exists") is False:
            continue
        if str(row.get("review_route") or "") in {"vlm_review", "human_review"}:
            continue
        if not _fast_material_is_strict_yolo_evidence(row):
            continue
        if str(row.get("quality_bucket") or "") == "low_quality":
            continue
        action_type = _fast_material_physical_action_type(row)
        if action_type not in _FAST_MATERIAL_CORE_ACTIONS:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id:
            eligible_ids.append(candidate_id)
            group_id = str(row.get("candidate_group_id") or "").strip()
            if group_id:
                eligible_group_ids.add(group_id)
    companion_count = 0
    allow_paired_context_companions = _bool_env("KEY_ACTION_AUTO_PUBLISH_PAIRED_VIEW_CONTEXT_MATERIAL", True)
    if eligible_group_ids:
        for row in records:
            if not isinstance(row, Mapping):
                continue
            if str(row.get("candidate_group_id") or "").strip() not in eligible_group_ids:
                continue
            if row.get("exists") is False:
                continue
            if str(row.get("quality_bucket") or "") == "low_quality":
                continue
            action_type = _fast_material_physical_action_type(row)
            if action_type not in _FAST_MATERIAL_CORE_ACTIONS:
                continue
            is_paired_context = _fast_material_is_paired_view_context(row)
            if is_paired_context:
                if not allow_paired_context_companions:
                    continue
                if not paired_view_context_scene_gate_passed(row):
                    continue
            elif not (
                _fast_material_is_strict_yolo_evidence(row)
                or _fast_material_is_explicit_dual_event_sparse_evidence(row)
            ):
                continue
            candidate_id = str(row.get("candidate_id") or "").strip()
            if candidate_id and candidate_id not in eligible_ids:
                eligible_ids.append(candidate_id)
                companion_count += 1
    eligible_ids = sorted(set(eligible_ids))
    eligible_rows = [
        dict(row)
        for row in records
        if isinstance(row, Mapping) and str(row.get("candidate_id") or "").strip() in eligible_ids
    ]
    complete_rows = filter_aligned_dual_view_material_rows(
        session_dir,
        filter_complete_dual_view_material_rows(eligible_rows),
    )
    complete_rows, formal_gate_suppressed = apply_formal_dual_view_material_publish_gate(
        session_dir,
        complete_rows,
    )
    if formal_gate_suppressed:
        result["formal_dual_view_gate_filtered_count"] = len(formal_gate_suppressed)
    complete_ids = sorted({str(row.get("candidate_id") or "").strip() for row in complete_rows if str(row.get("candidate_id") or "").strip()})
    if complete_ids != eligible_ids:
        result["dual_view_incomplete_filtered_count"] = len(eligible_ids) - len(complete_ids)
    if eligible_ids and not complete_ids:
        result["dual_view_alignment_filtered_count"] = len(eligible_ids)
    eligible_ids = complete_ids
    result["eligible_count"] = len(eligible_ids)
    result["eligible_group_count"] = len(eligible_group_ids)
    result["dual_view_complete_group_count"] = len({
        str(row.get("candidate_group_id") or "").strip()
        for row in complete_rows
        if str(row.get("candidate_group_id") or "").strip()
    })
    result["paired_companion_count"] = companion_count
    if not eligible_ids:
        result["reason"] = "no_recommended_quality_passed_candidates"
        return result
    try:
        approval = approve_material_candidates(
            session_dir,
            candidate_ids=eligible_ids,
            reviewer="key_action_fast_pipeline",
            notes="Auto-published recommended YOLO quality-passed core-v1 physical-action materials during fast preprocessing.",
            reason_code="fast_core_v1_yolo_quality_passed",
            reason="Recommended candidate passed YOLO evidence and material quality gates.",
        )
    except Exception as exc:
        return {**result, "status": "failed", "error": str(exc)}
    approved_count = int(approval.get("approved_count") or 0) if isinstance(approval, Mapping) else 0
    return {**result, "status": "completed", "approved_count": approved_count, "approval": approval}


def _run_fast_locate_refined_outputs(
    manifest: SessionManifest,
    paths: dict[str, Path],
    config: DetectorConfig,
    detected_segments: list[Any],
    yolo_frame_rows: list[dict[str, Any]],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    _ensure_fast_material_preprocess_defaults()
    run_ctx = RunContext()
    timings: dict[str, float] = {}

    fine_seed_started = time.perf_counter()
    fine_window_segments = _fast_locate_fine_window_segments_from_yolo_rows(
        manifest,
        detected_segments,
        yolo_frame_rows,
        config,
    )
    experiment_window_state: dict[str, Any] = {}
    try:
        experiment_window_state = build_experiment_state_artifacts(
            paths["metadata"],
            manifest,
            yolo_frame_rows,
            fine_window_segments or detected_segments,
            # This is the minute-level experiment-window layer, not the
            # second-level micro/action layer.  Using the detector/micro
            # min_segment_duration_sec here lets a few-second hand-object
            # snippet masquerade as an experiment window.
            min_duration_sec=_experiment_window_min_duration_sec(default=120.0),
        )
    except Exception as exc:
        experiment_window_state = {
            "summary": {
                "schema_version": "experiment_window_state_summary.v1",
                "available": False,
                "error": str(exc),
            },
            "formal_segments": [],
        }
        _write_json(paths["metadata"] / "experiment_window_state_error.json", experiment_window_state["summary"])
    state_formal_segments = [
        segment
        for segment in experiment_window_state.get("formal_segments", [])
        if isinstance(segment, DetectedSegment)
    ]
    fine_scan_seed_segments = state_formal_segments or fine_window_segments or detected_segments
    write_jsonl(paths["metadata"] / "fast_locate_fine_scan_windows.jsonl", fine_window_segments)
    timings["fine_window_seed_sec"] = time.perf_counter() - fine_seed_started

    coarse_projection_started = time.perf_counter()
    coarse_key_segments = _fast_locate_key_segments_from_detected(
        manifest,
        paths,
        fine_scan_seed_segments,
        yolo_frame_rows,
        run_ctx,
        dry_run=dry_run,
        include_first_person=bool(config.yolo_scan_both_views and manifest.videos.first_person is not None),
    )
    timings["coarse_window_projection_sec"] = time.perf_counter() - coarse_projection_started

    fine_scan_both_views = _bool_env(
        "KEY_ACTION_FAST_LOCATE_FINE_SCAN_BOTH_VIEWS",
        bool(config.yolo_scan_both_views) and manifest.videos.first_person is not None,
    )
    fine_preferred_view = os.environ.get("KEY_ACTION_FAST_LOCATE_FINE_PREFERRED_VIEW") or config.yolo_preferred_view
    refine_config = replace(
        config,
        enable_micro_refine_rescan=True,
        yolo_preferred_view=str(fine_preferred_view or config.yolo_preferred_view),
        yolo_scan_both_views=bool(fine_scan_both_views and manifest.videos.first_person is not None),
    )
    refine_started = time.perf_counter()
    streaming_rows_path = paths["cv_outputs"] / "yolo_streaming_micro_frame_rows.jsonl"
    streaming_fine_summary = _read_json_if_exists(paths["metadata"] / "streaming_fine_scan_summary.json")
    if (
        _fast_locate_reuse_streaming_fine_scan_enabled()
        and bool(streaming_fine_summary.get("available"))
        and streaming_rows_path.exists()
    ):
        refined_yolo_rows = _sort_yolo_rows_by_alignment_time(
            _dedupe_yolo_rows_by_view_time(
                _normalize_yolo_rows_for_pipeline(manifest, read_jsonl(streaming_rows_path), refine_config)
            )
        )
        micro_refine_summary = {
            **streaming_fine_summary,
            "available": bool(refined_yolo_rows),
            "enabled": True,
            "reused_streaming_fine_scan": True,
            "source": "streaming_refined_yolo_micro_frame_rows",
            "rows": len(refined_yolo_rows),
            "model_load_mode": "streaming_reuse",
        }
        _write_json(paths["metadata"] / "yolo_micro_scan_summary.json", micro_refine_summary)
        write_jsonl(paths["cv_outputs"] / "yolo_micro_frame_rows.jsonl", refined_yolo_rows)
    else:
        refined_yolo_rows, micro_refine_summary = _run_yolo_micro_refine_rescan(
            manifest,
            paths,
            refine_config,
            coarse_key_segments,
            dry_run=dry_run,
            default_fine_scan_workers=4,
            fast_locate_refined=True,
            coarse_yolo_rows=yolo_frame_rows,
        )
    timings["fine_scan_dispatch_sec"] = time.perf_counter() - refine_started
    micro_source_rows = refined_yolo_rows or yolo_frame_rows

    dual_alignment_started = time.perf_counter()
    dual_view_action_alignment: dict[str, Any] = {}
    dual_aligned_segments: list[DetectedSegment] = []
    dual_action_alignment_required = bool(
        not dry_run
        and manifest.videos.first_person is not None
        and _bool_env("KEY_ACTION_REQUIRE_DUAL_VIEW_ACTION_ALIGNMENT", True)
    )
    if dual_action_alignment_required:
        try:
            dual_view_action_alignment = build_dual_view_action_alignment(
                manifest,
                micro_source_rows,
                output_dir=paths["metadata"],
                strict=True,
            )
            dual_aligned_segments = [
                segment
                for segment in dual_view_action_alignment.get("episodes", [])
                if isinstance(segment, DetectedSegment)
            ]
        except Exception as exc:
            dual_view_action_alignment = {
                "summary": {
                    "schema_version": "dual_view_action_alignment.v1",
                    "available": False,
                    "strict": True,
                    "formal_results_allowed": False,
                    "decision": "dual_view_action_alignment_failed",
                    "error": str(exc),
                },
                "episodes": [],
            }
            write_jsonl(paths["metadata"] / "view_action_evidence.jsonl", [])
            write_jsonl(paths["metadata"] / "dual_view_action_events.jsonl", [])
            write_jsonl(paths["metadata"] / "unmatched_view_evidence.jsonl", [])
            _write_json(
                paths["metadata"] / "dual_view_action_alignment_summary.json",
                dual_view_action_alignment["summary"],
            )
    timings["dual_view_action_alignment_sec"] = time.perf_counter() - dual_alignment_started

    build_started = time.perf_counter()
    refined_detected_segments = _refined_experiment_segments_from_yolo_rows(
        manifest,
        refined_yolo_rows,
        refine_config,
        base_segments=detected_segments,
    )
    expected_count = _fast_locate_expected_experiment_count(refine_config)
    window_expansion_summary: dict[str, Any] = {}
    if dual_action_alignment_required:
        duration_limit = _session_duration_limit_for_experiment_windows(
            manifest,
            [*list(yolo_frame_rows or []), *list(micro_source_rows or [])],
            list(detected_segments),
            dry_run=dry_run,
        )
        dual_activity_segments, dual_activity_summary = _dual_view_activity_aligned_segments_from_yolo_rows(
            manifest,
            [*list(yolo_frame_rows or []), *list(micro_source_rows or [])],
            duration_limit_sec=duration_limit,
        )
        write_jsonl(paths["metadata"] / "dual_view_activity_aligned_experiment_episodes.jsonl", dual_activity_segments)
        _write_json(paths["metadata"] / "dual_view_activity_alignment_summary.json", dual_activity_summary)
        # Experiment overview windows are lifecycle/activity intervals. Action
        # events are intentionally kept for key-material publishing, otherwise
        # a brief hand-object event can collapse a real multi-minute experiment.
        refined_detected_segments = list(dual_activity_segments or dual_aligned_segments)
        official_source_segments = list(dual_activity_segments or dual_aligned_segments)
        expanded_dual_segments, window_expansion_summary = _expand_official_experiment_segments_to_activity_windows(
            manifest,
            official_source_segments,
            yolo_rows=[*list(yolo_frame_rows or []), *list(micro_source_rows or [])],
            coarse_segments=detected_segments,
            dry_run=dry_run,
        )
        final_detected_segments, dual_episode_filter_summary = _filter_short_weak_experiment_episodes(
            manifest,
            expanded_dual_segments,
            expected_count=expected_count,
            dry_run=dry_run,
        )
        dual_episode_filter_summary = dict(dual_episode_filter_summary)
        dual_episode_filter_summary["window_expansion"] = window_expansion_summary
        episode_filter_summary = {
            "schema_version": "key_action_episode_filter_summary.v1",
            "strategy": "strict_dual_view_action_alignment",
            "input_count": len(dual_view_action_alignment.get("view_action_evidence") or []),
            "output_count": len(final_detected_segments),
            "expected_count": expected_count,
            "activity_aligned_episode_count": len(dual_activity_segments),
            "activity_alignment": dual_activity_summary,
            "dual_view_action_event_count": int(
                ((dual_view_action_alignment.get("summary") or {}).get("dual_view_action_event_count") or 0)
                if isinstance(dual_view_action_alignment.get("summary"), Mapping)
                else 0
            ),
            "unmatched_view_evidence_count": int(
                ((dual_view_action_alignment.get("summary") or {}).get("unmatched_view_evidence_count") or 0)
                if isinstance(dual_view_action_alignment.get("summary"), Mapping)
                else 0
            ),
            "formal_results_allowed": bool(final_detected_segments),
            "decision": (
                "use_dual_view_activity_aligned_episodes"
                if dual_activity_segments and final_detected_segments
                else "use_dual_view_action_aligned_episodes"
                if dual_aligned_segments and final_detected_segments
                else "no_formal_dual_view_aligned_episodes"
            ),
            "reason": (
                "formal_episode_uses_first_and_third_lifecycle_or_activity_overlap"
                if dual_activity_segments
                else "formal_episode_uses_confirmed_first_and_third_action_evidence"
                if dual_aligned_segments
                else "only_single_view_or_unmatched_action_evidence_detected"
            ),
            "candidate_action_windows": (
                dual_episode_filter_summary.get("candidate_action_windows", [])
                if isinstance(dual_episode_filter_summary, Mapping)
                else []
            ),
            "candidate_action_window_count": (
                dual_episode_filter_summary.get("candidate_action_window_count", 0)
                if isinstance(dual_episode_filter_summary, Mapping)
                else 0
            ),
            "window_expansion": window_expansion_summary,
            "official_episode_filter": dual_episode_filter_summary,
        }
    else:
        raw_final_detected_segments = refined_detected_segments or detected_segments
        expanded_final_detected_segments, window_expansion_summary = _expand_official_experiment_segments_to_activity_windows(
            manifest,
            list(raw_final_detected_segments),
            yolo_rows=[*list(yolo_frame_rows or []), *list(micro_source_rows or [])],
            coarse_segments=detected_segments,
            dry_run=dry_run,
        )
        final_detected_segments, episode_filter_summary = _filter_short_weak_experiment_episodes(
            manifest,
            list(expanded_final_detected_segments),
            expected_count=expected_count,
            dry_run=dry_run,
        )
        episode_filter_summary = dict(episode_filter_summary)
        episode_filter_summary["window_expansion"] = window_expansion_summary
    preserve_coarse_episode_count = _bool_env("KEY_ACTION_FAST_LOCATE_PRESERVE_COARSE_EPISODE_COUNT", True)
    max_preserved_coarse_count = max(
        1,
        _int_env_value("KEY_ACTION_FAST_LOCATE_MAX_PRESERVED_COARSE_EPISODE_COUNT", 12),
    )
    if (
        preserve_coarse_episode_count
        and not dual_action_alignment_required
        and detected_segments
        and refined_detected_segments
        and len(detected_segments) <= max_preserved_coarse_count
        and len(final_detected_segments) != len(detected_segments)
    ):
        _write_json(
            paths["metadata"] / "fast_locate_refine_segment_count_guard.json",
            {
                "schema_version": "fast_locate_refine_segment_count_guard.v1",
                "expected_experiment_count": expected_count,
                "refined_segment_count": len(refined_detected_segments),
                "coarse_segment_count": len(detected_segments),
                "max_preserved_coarse_episode_count": max_preserved_coarse_count,
                "decision": "preserve_coarse_experiment_episodes_for_segment_projection",
                "reason": "micro refine evidence is used for key physical materials, but it must not split or drop the long-video experiment episodes shown in the overview.",
            },
        )
        final_detected_segments, window_expansion_summary = _expand_official_experiment_segments_to_activity_windows(
            manifest,
            list(detected_segments),
            yolo_rows=[*list(yolo_frame_rows or []), *list(micro_source_rows or [])],
            coarse_segments=detected_segments,
            dry_run=dry_run,
        )
        final_detected_segments, episode_filter_summary = _filter_short_weak_experiment_episodes(
            manifest,
            list(final_detected_segments),
            expected_count=expected_count,
            dry_run=dry_run,
        )
        episode_filter_summary = dict(episode_filter_summary)
        episode_filter_summary["window_expansion"] = window_expansion_summary
    action_gate_summary = (
        dual_view_action_alignment.get("summary")
        if isinstance(dual_view_action_alignment.get("summary"), Mapping)
        else {}
    )
    formal_output_gate = _formal_output_gate_status(
        paths,
        action_summary=action_gate_summary,
        require_action_alignment=dual_action_alignment_required,
    )
    if not formal_output_gate.get("formal_results_allowed"):
        blocked_reason = str(formal_output_gate.get("blocked_reason") or "formal_output_gate_blocked")
        blocked_segments = [segment for segment in final_detected_segments if isinstance(segment, DetectedSegment)]
        blocked_candidate_windows = _candidate_action_windows_from_blocked_official_segments(
            manifest,
            blocked_segments,
            blocked_reason=blocked_reason,
            start_index=1,
        )
        existing_candidate_value = (
            episode_filter_summary.get("candidate_action_windows", [])
            if isinstance(episode_filter_summary, Mapping)
            else []
        )
        existing_candidates = [item for item in existing_candidate_value if isinstance(item, dict)] if isinstance(existing_candidate_value, list) else []
        episode_filter_summary = dict(episode_filter_summary)
        episode_filter_summary.update(
            {
                "status": "blocked",
                "blocked_reason": blocked_reason,
                "formal_results_allowed": False,
                "video_memory_allowed": False,
                "official_episode_count": 0,
                "output_count": 0,
                "candidate_action_windows": [*existing_candidates, *blocked_candidate_windows],
                "candidate_action_window_count": len(existing_candidates) + len(blocked_candidate_windows),
                "formal_output_gate": formal_output_gate,
            }
        )
        final_detected_segments = []
        refined_detected_segments = []
    _write_formal_output_gate(paths, formal_output_gate)
    _write_phase_consistency_from_formal_gate(paths, formal_output_gate)
    candidate_action_windows = (
        episode_filter_summary.get("candidate_action_windows", [])
        if isinstance(episode_filter_summary, Mapping)
        else []
    )
    write_jsonl(paths["metadata"] / "candidate_action_windows.jsonl", candidate_action_windows)
    write_jsonl(paths["metadata"] / "experiment_episode_candidates.jsonl", candidate_action_windows)
    _write_json(paths["metadata"] / "fast_locate_episode_filter_summary.json", episode_filter_summary)
    key_segments = _fast_locate_key_segments_from_detected(
        manifest,
        paths,
        final_detected_segments,
        micro_source_rows,
        run_ctx,
        dry_run=dry_run,
        include_first_person=bool(refine_config.yolo_scan_both_views and manifest.videos.first_person is not None),
    )
    write_jsonl(paths["cv_outputs"] / "detected_segments.jsonl", final_detected_segments)
    write_jsonl(
        paths["metadata"] / "experiment_episodes.jsonl",
        _experiment_episode_rows(manifest, final_detected_segments),
    )
    _write_fast_locate_frontend_segment_projection(
        paths,
        key_segments,
        stage="segments_ready_before_micro_assets",
        micro_rows_ready=False,
    )
    timings["segment_projection_sec"] = time.perf_counter() - build_started

    micro_started = time.perf_counter()
    (
        raw_micro_rows,
        micro_rows,
        micro_dedup_log,
        micro_merge_stats,
        micro_vector_metadata,
        _vector_metadata,
    ) = _generate_and_write_micro_segments(
        manifest,
        paths,
        key_segments,
        micro_source_rows,
        [],
        run_ctx,
        dry_run=dry_run,
        progress_callback=None,
    )
    timings["micro_assets_sec"] = time.perf_counter() - micro_started

    dual_event_bind_started = time.perf_counter()
    dual_view_action_summary = _build_dual_view_action_alignment_summary(
        manifest,
        Path(manifest.output_dir),
        paths,
        yolo_frame_rows=yolo_frame_rows,
        refined_yolo_rows=refined_yolo_rows,
        micro_rows=micro_rows,
    )
    timings["dual_view_action_event_bind_sec"] = time.perf_counter() - dual_event_bind_started

    candidate_summary: dict[str, Any] = {}
    material_auto_publish: dict[str, Any] = {}
    candidate_started = time.perf_counter()
    if _bool_env("KEY_ACTION_FAST_LOCATE_BUILD_MATERIAL_CANDIDATES", True):
        try:
            candidate_build_started = time.perf_counter()
            candidate_summary = build_yolo_material_candidates(
                manifest.output_dir,
                dry_run=dry_run,
                archive_existing=True,
                rebuild_source=True,
                enable_vlm=False,
            )
            timings["material_candidate_build_sec"] = time.perf_counter() - candidate_build_started
            publish_started = time.perf_counter()
            material_auto_publish = _auto_publish_recommended_material_candidates(manifest.output_dir, candidate_summary)
            timings["material_auto_publish_sec"] = time.perf_counter() - publish_started
            candidate_summary["auto_publish"] = material_auto_publish
        except Exception as exc:
            candidate_summary = {"available": False, "error": str(exc)}
    timings["material_candidates_sec"] = time.perf_counter() - candidate_started

    reference_started = time.perf_counter()
    material_reference_summary: dict[str, Any] = {}
    if _bool_env("KEY_ACTION_FAST_LOCATE_BUILD_FORMAL_MATERIAL_REFERENCES", True):
        try:
            material_reference_summary = build_yolo_material_references(
                manifest.output_dir,
                dry_run=dry_run,
                archive_existing=True,
            )
        except Exception as exc:
            material_reference_summary = {"available": False, "error": str(exc)}
    timings["formal_material_references_sec"] = time.perf_counter() - reference_started

    timeline_rows = _fast_locate_refined_step_rows(key_segments, micro_rows)
    write_jsonl(paths["metadata"] / "experiment_process_timeline.jsonl", timeline_rows)
    segment_vector_metadata = [_vector_metadata_from_segment(segment) for segment in key_segments]
    write_jsonl(paths["metadata"] / "vector_metadata.jsonl", [*segment_vector_metadata, *micro_vector_metadata])
    write_jsonl(paths["index"] / "docstore.jsonl", [*segment_vector_metadata, *micro_vector_metadata])
    _write_json(
        paths["metadata"] / "fast_locate_frontend_projection_status.json",
        {
            "schema_version": "fast_locate_frontend_projection_status.v1",
            "stage": "micro_assets_ready",
            "segment_count": len(key_segments),
            "micro_rows_ready": True,
            "micro_segment_count": len(micro_rows),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )

    fast_refine_stage_timings = {key: round(value, 6) for key, value in timings.items()}
    timing_summary = _read_json_if_exists(paths["metadata"] / "yolo_timing_summary.json")
    yolo_executor_diagnostics = _yolo_executor_diagnostics(
        timing_summary,
        coarse_yolo_row_count=len(yolo_frame_rows),
        fine_yolo_row_count=len(refined_yolo_rows),
    )
    report_summary: dict[str, Any] = {}
    try:
        report_summary = write_speed_and_quality_reports(
            paths["metadata"],
            manifest,
            stage_timings=fast_refine_stage_timings,
            yolo_timing_summary=timing_summary if isinstance(timing_summary, Mapping) else {},
            coarse_yolo_row_count=len(yolo_frame_rows),
            fine_yolo_row_count=len(refined_yolo_rows),
            fine_window_count=len(fine_scan_seed_segments),
            fine_window_duration_sec=round(sum(float(item.duration_sec) for item in fine_scan_seed_segments), 6),
            candidate_summary=candidate_summary,
            material_reference_summary=material_reference_summary,
            formal_output_gate=formal_output_gate,
            episode_filter_summary=episode_filter_summary,
            micro_segment_count=len(micro_rows),
        )
    except Exception as exc:
        report_summary = {
            "speed_report": {"schema_version": "speed_report.v1", "error": str(exc)},
            "quality_report": {"schema_version": "quality_report.v1", "error": str(exc)},
        }
        _write_json(paths["metadata"] / "speed_quality_report_error.json", report_summary)
    _write_json(
        paths["metadata"] / "fast_locate_stage_timings.json",
        {
            "schema_version": "fast_locate_stage_timings.v1",
            "coarse_segment_count": len(detected_segments),
            "coarse_yolo_row_count": len(yolo_frame_rows),
            "fine_scan_window_count": len(fine_window_segments),
            "fine_scan_window_duration_sec": round(sum(float(item.duration_sec) for item in fine_window_segments), 6),
            "state_machine_fine_scan_window_count": len(fine_scan_seed_segments),
            "experiment_window_state_summary": (
                experiment_window_state.get("summary")
                if isinstance(experiment_window_state, Mapping)
                else {}
            ),
            "refined_segment_count": len(refined_detected_segments),
            "key_segment_count": len(key_segments),
            "fine_yolo_row_count": len(refined_yolo_rows),
            "micro_segment_count": len(micro_rows),
            "raw_micro_segment_count": len(raw_micro_rows),
            "candidate_count": candidate_summary.get("candidate_count") if isinstance(candidate_summary, dict) else None,
            "candidate_group_count": candidate_summary.get("candidate_group_count") if isinstance(candidate_summary, dict) else None,
            "published_count": (
                material_auto_publish.get("approved_count")
                if isinstance(material_auto_publish, dict)
                else None
            ),
            "episode_filter_summary": episode_filter_summary,
            "formal_output_gate": formal_output_gate,
            "dual_view_action_alignment_summary": (
                dual_view_action_summary
                if isinstance(dual_view_action_summary, dict)
                else (
                    dual_view_action_alignment.get("summary")
                    if isinstance(dual_view_action_alignment, dict)
                    else {}
                )
            ),
            "stage_timings": fast_refine_stage_timings,
            "material_candidate_pipeline": (
                {
                    key: (candidate_summary.get("pipeline_summary") or {}).get(key)
                    for key in (
                        "parallel_workers",
                        "candidate_count",
                        "group_count",
                        "recommended_total",
                        "timing_sec",
                    )
                    if isinstance(candidate_summary.get("pipeline_summary"), dict)
                    and key in candidate_summary.get("pipeline_summary")
                }
                if isinstance(candidate_summary, dict)
                else {}
            ),
            "material_reference_summary": {
                key: material_reference_summary.get(key)
                for key in (
                    "file_count",
                    "published_real_file_count",
                    "formal_published_file_count",
                    "planned_file_count",
                    "skipped_count",
                    "parallel_workers",
                    "material_generation_task_count",
                )
                if isinstance(material_reference_summary, dict) and key in material_reference_summary
            },
            "yolo_executor_diagnostics": yolo_executor_diagnostics,
            "speed_report": report_summary.get("speed_report") if isinstance(report_summary, Mapping) else {},
            "quality_report": report_summary.get("quality_report") if isinstance(report_summary, Mapping) else {},
            "micro_refine_summary": {
                key: micro_refine_summary.get(key)
                for key in (
                    "rows",
                    "window_count",
                    "window_scanned_duration_sec",
                    "parallel_workers",
                    "dispatch_wall_sec",
                    "model_load_mode",
                )
                if isinstance(micro_refine_summary, dict) and key in micro_refine_summary
            },
        },
    )
    _write_json(
        paths["metadata"] / "fast_locate_refined_summary.json",
        {
            "schema_version": "fast_locate_refined_outputs.v1",
            "coarse_segment_count": len(detected_segments),
            "coarse_yolo_row_count": len(yolo_frame_rows),
            "fine_scan_window_count": len(fine_window_segments),
            "fine_scan_window_duration_sec": round(sum(float(item.duration_sec) for item in fine_window_segments), 6),
            "state_machine_fine_scan_window_count": len(fine_scan_seed_segments),
            "experiment_window_state_summary": (
                experiment_window_state.get("summary")
                if isinstance(experiment_window_state, Mapping)
                else {}
            ),
            "refined_segment_count": len(refined_detected_segments),
            "key_segment_count": len(key_segments),
            "fine_yolo_row_count": len(refined_yolo_rows),
            "micro_segment_count": len(micro_rows),
            "raw_micro_segment_count": len(raw_micro_rows),
            "material_candidate_summary": candidate_summary,
            "material_auto_publish": material_auto_publish,
            "material_reference_summary": material_reference_summary,
            "episode_filter_summary": episode_filter_summary,
            "formal_output_gate": formal_output_gate,
            "stage_timings": fast_refine_stage_timings,
            "speed_report": report_summary.get("speed_report") if isinstance(report_summary, Mapping) else {},
            "quality_report": report_summary.get("quality_report") if isinstance(report_summary, Mapping) else {},
            "micro_refine_summary": micro_refine_summary,
            "yolo_executor_diagnostics": yolo_executor_diagnostics,
            "micro_merge_stats": micro_merge_stats,
            "micro_dedup_count": len(micro_dedup_log),
        },
    )
    return {
        "fast_locate_only": True,
        "mode": "yolo_fast_locate_refined",
        "coarse_segment_count": len(detected_segments),
        "coarse_yolo_row_count": len(yolo_frame_rows),
        "fine_scan_window_count": len(fine_window_segments),
        "fine_scan_window_duration_sec": round(sum(float(item.duration_sec) for item in fine_window_segments), 6),
        "state_machine_fine_scan_window_count": len(fine_scan_seed_segments),
        "experiment_window_state_summary": (
            experiment_window_state.get("summary")
            if isinstance(experiment_window_state, Mapping)
            else {}
        ),
        "refined_segment_count": len(refined_detected_segments),
        "segment_count": len(key_segments),
        "fine_yolo_row_count": len(refined_yolo_rows),
        "micro_segment_count": len(micro_rows),
        "raw_micro_segment_count": len(raw_micro_rows),
        "material_candidate_summary": candidate_summary,
        "material_auto_publish": material_auto_publish,
        "material_reference_summary": material_reference_summary,
        "episode_filter_summary": episode_filter_summary,
        "formal_output_gate": formal_output_gate,
        "micro_refine_summary": micro_refine_summary,
        "micro_merge_stats": micro_merge_stats,
        "stage_stats": run_ctx.stage_stats(),
        "fast_refine_stage_timings": fast_refine_stage_timings,
        "yolo_executor_diagnostics": yolo_executor_diagnostics,
        "speed_report": report_summary.get("speed_report") if isinstance(report_summary, Mapping) else {},
        "quality_report": report_summary.get("quality_report") if isinstance(report_summary, Mapping) else {},
    }


def run_detection_only(
    manifest_path: str | Path,
    dry_run: bool = False,
    detector_config: DetectorConfig | None = None,
) -> dict[str, Any]:
    manifest = SessionManifest.load(manifest_path)
    output_dir = Path(manifest.output_dir)
    paths = _mkdirs(output_dir)
    active_detector_config = detector_config or manifest.detection_config
    _write_json(paths["metadata"] / "detector_config.json", to_json_dict(active_detector_config))
    validation_result = validate_manifest(manifest_path, validate_video=not dry_run)
    build_long_video_processing_plan(
        manifest,
        validation_result,
        active_detector_config,
        output_dir,
        dry_run=dry_run,
    )
    write_video_source_metadata(manifest, output_dir, paths["metadata"] / "video_sources.jsonl")
    requested_fast_locate = _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False) or _bool_env("KEY_ACTION_DEFER_SEGMENT_ASSETS", False)
    if requested_fast_locate:
        _ensure_fast_material_preprocess_defaults()
    fast_locate_runtime = _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False) or _bool_env("KEY_ACTION_DEFER_SEGMENT_ASSETS", False)
    skip_model_inventory = fast_locate_runtime and _bool_env("KEY_ACTION_FAST_LOCATE_SKIP_MODEL_INVENTORY", True)
    if skip_model_inventory:
        model_inventory = {
            "skipped": True,
            "reason": "fast_locate_runtime",
            "primary_model": {},
            "model_count": 0,
            "dataset_count": 0,
        }
        _write_json(paths["metadata"] / "model_inventory.json", model_inventory)
    else:
        model_inventory = _discover_lab_assets_for_pipeline(
            manifest,
            output_dir,
            paths["metadata"] / "model_inventory.json",
            dry_run=dry_run,
        )
    for timing_path in (
        paths["metadata"] / "yolo_timing_rows.jsonl",
        paths["metadata"] / "yolo_timing_summary.json",
    ):
        try:
            timing_path.unlink()
        except FileNotFoundError:
            pass
    pre_coarse_alignment = _ensure_pre_coarse_timeline_alignment(manifest, paths, dry_run=dry_run)
    detected_segments, _scores, yolo_rows, detector_summary = _detect_with_config(
        manifest,
        paths,
        active_detector_config,
        dry_run=dry_run,
    )
    if isinstance(detector_summary, dict):
        detector_summary["pre_coarse_timeline_alignment"] = pre_coarse_alignment
    write_jsonl(paths["cv_outputs"] / "detected_segments.jsonl", detected_segments)
    refined_summary: dict[str, Any] = {}
    if _fast_locate_refine_enabled():
        refined_summary = _run_fast_locate_refined_outputs(
            manifest,
            paths,
            active_detector_config,
            detected_segments,
            yolo_rows,
            dry_run=dry_run,
        )
    else:
        _write_detection_only_frontend_projection(manifest, paths, detected_segments)
    if (not fast_locate_runtime and not _bool_env("KEY_ACTION_FAST_LOCATE_ONLY", False)) or _bool_env(
        "KEY_ACTION_FAST_LOCATE_SAVE_SCORE_PLOT",
        False,
    ):
        save_frame_score_plot(
            paths["cv_outputs"] / "frame_scores.jsonl",
            paths["cv_outputs"] / "detected_segments.jsonl",
            paths["debug"] / "frame_scores.png",
        )
    summary = _build_detection_only_summary(manifest, paths, detected_segments, detector_summary, model_inventory)
    if refined_summary:
        summary.update(refined_summary)
    timing_summary = _read_json_if_exists(paths["metadata"] / "yolo_timing_summary.json")
    if isinstance(timing_summary, dict):
        summary["yolo_timing_summary"] = timing_summary
    return summary
