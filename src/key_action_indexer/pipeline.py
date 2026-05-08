from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .action_detector import detect_key_action_segments
from .advanced_vision_evidence import build_advanced_vision_evidence
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
from .evidence import apply_segment_evidence
from .history_learning import build_history_model
from .input_ingestion import ingest_manifest_inputs, write_video_source_metadata
from .evaluation import build_pipeline_evaluation_report, compute_micro_quality_stats
from .experiment_focus import extract_experiment_focus_clips
from .artifact_schema import validate_session_artifacts
from .micro_postprocess import merge_same_object_adjacent_micro_segments
from .micro_segmenter import generate_micro_segments, micro_row_to_vector_metadata
from .material_references import build_yolo_material_references
from .model_inventory import discover_lab_assets
from .model_observations import build_model_observation_events
from .lab_model_signal_inputs import build_lab_model_signal_inputs
from .performance import build_long_video_processing_plan
from .process_record import build_process_record
from .process_reasoner import build_experiment_process
from .process_record import build_process_record
from .performance import build_long_video_processing_plan
from .quality_assurance import build_quality_assurance_report
from .record_ingestion import ingest_sop_and_database_records
from .record_ingestion import ingest_sop_and_database_records
from .report import DEFAULT_QUERY, generate_formal_validation_report, generate_report
from .session_layout import initialize_session_dir
from .session_context_seed import seed_session_context
from .schemas import (
    KeyActionSegment,
    SessionManifest,
    VideoSource,
    VectorMetadata,
    read_jsonl,
    to_json_dict,
    write_jsonl,
)
from .state_index import build_state_change_index
from .time_alignment import find_dialogue_for_segment, generate_multimodal_alignment, local_sec_to_global_time, parse_time
from .transcript import load_aligned_transcript
from .unified_timeline import generate_unified_timeline
from .validation import validate_manifest
from .vector_index import VectorIndex
from .video_understanding import build_video_understanding
from .yolo_observation_inputs import build_yolo_observation_inputs


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


def _boolish(value: Any) -> bool:
    return _bool_from_config(value)


def _source_for_view(manifest: SessionManifest, view: str | None) -> VideoSource:
    normalized = str(view or "").strip().lower()
    return manifest.videos.get(normalized) or manifest.videos.third_person


def _global_relative_sec(manifest: SessionManifest, global_time: str) -> float:
    return (parse_time(global_time) - parse_time(manifest.session_start_time)).total_seconds()


def _session_time_sec(manifest: SessionManifest, global_time: str) -> float:
    return _global_relative_sec(manifest, global_time)


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
            stats["global_start_time"] = local_sec_to_global_time(source, float(stats["local_start_sec"])).isoformat()
        if stats["local_end_sec"] is not None:
            stats["local_end_sec"] = round(float(stats["local_end_sec"]), 6)
            stats["global_end_time"] = local_sec_to_global_time(source, float(stats["local_end_sec"])).isoformat()
        stats["offset_sec"] = float(getattr(source, "offset_sec", 0.0) or 0.0)
        stats["video_start_time"] = getattr(source, "start_time", None)
        stats["video_path"] = getattr(source, "path", None)
        stats["camera_id"] = getattr(source, "camera_id", None)
        stats["role"] = getattr(source, "role", None) or view

    offsets_sec: dict[str, float | None] = {"first_person": None, "third_person": None}
    if manifest.videos.first_person is not None:
        offsets_sec["first_person"] = float(getattr(manifest.videos.first_person, "offset_sec", 0.0) or 0.0)
    if manifest.videos.third_person is not None:
        offsets_sec["third_person"] = float(getattr(manifest.videos.third_person, "offset_sec", 0.0) or 0.0)
    for view, source in manifest.videos.extra_views.items():
        offsets_sec[str(view)] = float(getattr(source, "offset_sec", 0.0) or 0.0)

    summary = {
        "schema_version": "view_alignment.yolo.v1",
        "method": "manifest_offsets",
        "status": "aligned" if stats_by_view else "empty",
        "views": sorted(stats_by_view),
        "row_counts_by_view": {view: int(stats["row_count"]) for view, stats in sorted(stats_by_view.items())},
        "offsets_sec": offsets_sec,
        "time_ranges_by_view": {view: stats for view, stats in sorted(stats_by_view.items())},
    }
    _write_json(paths["metadata"] / "view_alignment_from_yolo.json", summary)
    return summary


def _normalize_yolo_rows_for_alignment(manifest: SessionManifest, rows: list[dict[str, Any]], config: DetectorConfig) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        view = str(item.get("source_view") or item.get("view") or "third_person")
        source = _source_for_view(manifest, view)
        local_time = float(item.get("local_time_sec", item.get("time_sec", 0.0)) or 0.0)
        global_time = item.get("global_time") or local_sec_to_global_time(source, local_time).isoformat()
        item["view"] = view
        item["source_view"] = view
        item["local_time_sec"] = local_time
        item["global_time"] = global_time
        item["motion_score"] = float(item.get("interaction_score", item.get("active_score", 0.0)) or 0.0)
        item["active_score"] = float(item.get("active_score", 0.0) or 0.0)
        item["is_active"] = bool(item.get("is_experiment_active", item["active_score"] >= config.start_threshold))
        item["alignment_time_sec"] = _global_relative_sec(manifest, str(global_time))
        normalized.append(item)
    return normalized


def _frame_score_rows_from_yolo(manifest: SessionManifest, yolo_rows: list[dict[str, Any]], config: DetectorConfig) -> list[dict[str, Any]]:
    by_time: dict[float, dict[str, Any]] = {}
    for row in _normalize_yolo_rows_for_alignment(manifest, yolo_rows, config):
        t = round(float(row.get("alignment_time_sec", 0.0)), 3)
        existing = by_time.get(t)
        if existing is None or float(row.get("active_score", 0.0)) > float(existing.get("active_score", 0.0)):
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
            }
    return [by_time[key] for key in sorted(by_time)]


def _yolo_segment_rows_from_frame_scores(frame_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "time_sec": float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0),
            "source_view": "global_multiview",
            "video_path": "global_multiview",
            "video_start_time": row.get("global_time"),
        }
        for row in frame_score_rows
    ]


def _run_yolo_detection(
    manifest: SessionManifest,
    config: DetectorConfig,
    paths: dict[str, Path],
    *,
    dry_run: bool,
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .yolo_analysis import resolve_default_yolo_model
    from .yolo_detector import build_segments_from_yolo_frame_rows, scan_yolo_video

    model_path = resolve_default_yolo_model(config.yolo_model_path)
    scan_both = _bool_from_config(config.yolo_scan_both_views) and manifest.videos.first_person is not None
    views = ["first_person", "third_person"] if scan_both else [config.yolo_preferred_view]
    if "third_person" not in views and manifest.videos.first_person is None:
        views = ["third_person"]
    yolo_rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for view in views:
        try:
            rows = scan_yolo_video(
                first_person_path=manifest.videos.first_person,
                third_person_path=manifest.videos.third_person,
                preferred_view=view,
                source_view=view,
                model_path=model_path,
                sample_fps=config.sample_fps,
                conf=config.yolo_conf,
                iou=config.yolo_iou,
                device=config.yolo_device,
                class_thresholds=config.yolo_class_thresholds,
                active_threshold=config.start_threshold,
                continuity_frames=config.yolo_continuity_frames,
                dry_run=dry_run,
            )
            yolo_rows.extend(rows)
        except Exception as exc:
            errors.append(f"{view}: {exc}")

    if not yolo_rows:
        raise RuntimeError("YOLO scan produced no frame rows" + (": " + "; ".join(errors) if errors else ""))

    yolo_rows = _normalize_yolo_rows_for_alignment(manifest, yolo_rows, config)
    frame_rows = _frame_score_rows_from_yolo(manifest, yolo_rows, config)
    segment_rows = _yolo_segment_rows_from_frame_scores(frame_rows)
    pseudo_source = VideoSource(
        name="global_multiview",
        path="global_multiview",
        start_time=manifest.session_start_time,
        fps=config.sample_fps,
        offset_sec=0.0,
    )
    duration_sec = max([float(row.get("time_sec", 0.0)) for row in segment_rows], default=0.0) + (1.0 / max(config.sample_fps, 0.001))
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
    labels = Counter()
    interaction_count = 0
    for row in yolo_rows:
        labels.update({str(k): int(v) for k, v in (row.get("label_counts") or {}).items()})
        interaction_count += len(row.get("hand_object_interactions") or [])
    for segment in detected_segments:
        segment.detector_backend = "yolo_interaction"
        segment.detector_source_view = "multiview" if scan_both else str(yolo_rows[0].get("source_view") or config.yolo_preferred_view)
        if not str(segment.start_reason).startswith("yolo_physical_evidence"):
            segment.start_reason = "yolo_active_score_above_threshold"
        if not str(segment.end_reason).startswith("yolo_physical_evidence"):
            segment.end_reason = "yolo_active_score_below_threshold"
        segment.yolo_label_counts = dict(labels)
        segment.yolo_interaction_count = interaction_count

    summary = {
        "available": True,
        "detector_backend": "yolo_interaction",
        "model_path": str(model_path) if model_path else None,
        "scan_both_views": scan_both,
        "views": views,
        "frame_rows": len(yolo_rows),
        "frame_score_rows": len(frame_rows),
        "segment_count": len(detected_segments),
        "boundary_refined_segment_count": sum(
            1 for segment in detected_segments if str(getattr(segment, "boundary_source", "")).startswith("yolo_physical_evidence")
        ),
        "label_counts": dict(labels),
        "interaction_count": interaction_count,
        "errors": errors,
    }
    _write_json(paths["metadata"] / "yolo_frame_scan.json", {
        "available": True,
        "metadata_version": "key_action_yolo_scan.v1",
        "detection_frame_count": len(yolo_rows),
        "sampled_frame_count": len(yolo_rows),
        "physical_event_count": interaction_count,
        "event_types": ["hand_object_interaction"] if interaction_count else [],
        "sample_frames": yolo_rows[:50],
        "frames": yolo_rows[:50],
        "summary": summary,
    })
    _write_json(paths["metadata"] / "yolo_scan_summary.json", summary)
    write_jsonl(paths["cv_outputs"] / "yolo_frame_scores.jsonl", yolo_rows)
    write_jsonl(paths["cv_outputs"] / "frame_scores.jsonl", frame_rows)
    return detected_segments, frame_rows, yolo_rows, summary


def _detect_segments(
    manifest: SessionManifest,
    config: DetectorConfig,
    paths: dict[str, Path],
    *,
    dry_run: bool,
) -> tuple[list[Any], list[Any], list[dict[str, Any]], dict[str, Any]]:
    backend = str(config.detector_backend or "motion").lower()
    if backend in {"yolo", "yolo_interaction", "multiview_yolo"}:
        try:
            detected, frame_rows, yolo_rows, summary = _run_yolo_detection(manifest, config, paths, dry_run=dry_run)
            return detected, frame_rows, yolo_rows, summary
        except Exception as exc:
            if not _bool_from_config(config.yolo_fallback_to_motion):
                raise
            fallback_summary = {
                "available": False,
                "detector_backend": "yolo_interaction",
                "error": str(exc),
                "fallback": "motion",
            }
            _write_json(paths["metadata"] / "yolo_scan_summary.json", fallback_summary)
            _write_json(paths["metadata"] / "yolo_frame_scan.json", {"available": False, "error": str(exc), "frames": [], "sample_frames": []})
            print(f"[pipeline] YOLO detector failed; falling back to motion baseline: {exc}")

    detected_segments, scores = detect_key_action_segments(
        manifest.videos.third_person,
        roi=manifest.workbench_roi,
        config=config,
        dry_run=dry_run,
        frame_scores_output_path=paths["cv_outputs"] / "frame_scores.jsonl",
    )
    summary = {"available": True, "detector_backend": "motion", "segment_count": len(detected_segments)}
    return detected_segments, scores, [], summary


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
        global_time = str(item.get("global_time") or local_sec_to_global_time(source, local_time).isoformat())
        active_score = float(item.get("active_score", 0.0) or 0.0)
        interaction_score = float(item.get("interaction_score", active_score) or 0.0)
        item["source_view"] = view
        item["view"] = view
        item["local_time_sec"] = local_time
        item["global_time"] = global_time
        item["alignment_time_sec"] = _session_time_sec(manifest, global_time)
        item["motion_score"] = interaction_score
        item["active_score"] = active_score
        item["is_active"] = bool(item.get("is_experiment_active", active_score >= config.start_threshold))
        normalized.append(item)
    return sorted(normalized, key=lambda item: (float(item.get("alignment_time_sec", 0.0)), str(item.get("source_view", ""))))


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
    try:
        return float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0)
    except (TypeError, ValueError):
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
                "avg_active_score": float(getattr(segment, "avg_active_score", 0.0) or 0.0),
                "avg_motion_score": float(getattr(segment, "avg_motion_score", 0.0) or 0.0),
                "start_reason": getattr(segment, "start_reason", ""),
                "end_reason": getattr(segment, "end_reason", ""),
                "view_alignment": dict(detector_summary.get("view_alignment") or {}) if detector_summary else {},
                "interpretation": "continuous_experiment_episode",
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
) -> tuple[list[Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from .yolo_detector import build_segments_from_yolo_frame_rows, scan_yolo_video

    scan_both = _boolish(config.yolo_scan_both_views) and manifest.videos.first_person is not None
    preferred = str(config.yolo_preferred_view or "first_person")
    views = ["first_person", "third_person"] if scan_both else [preferred]
    if views == ["first_person"] and manifest.videos.first_person is None:
        views = ["third_person"]

    raw_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    model_paths_by_view: dict[str, str | None] = {}
    class_schema, annotation_asset_refs = _yolo_inventory_refs(paths)
    for view in views:
        model_path = _yolo_model_path_for_view(config, view)
        model_paths_by_view[view] = str(model_path) if model_path else None
        try:
            raw_rows.extend(
                scan_yolo_video(
                    first_person_path=manifest.videos.first_person,
                    third_person_path=manifest.videos.third_person,
                    preferred_view=view,
                    source_view=view,
                    model_path=model_path,
                    model_ref={
                        "path": str(model_path) if model_path else None,
                        "backend": "ultralytics_yolo",
                        "view": view,
                    },
                    class_schema=class_schema,
                    annotation_asset_refs=annotation_asset_refs,
                    sample_fps=float(config.parent_sample_fps or config.sample_fps),
                    conf=config.yolo_conf,
                    iou=config.yolo_iou,
                    device=config.yolo_device,
                    class_thresholds=config.yolo_class_thresholds,
                    active_threshold=config.start_threshold,
                    continuity_frames=config.yolo_continuity_frames,
                    dry_run=dry_run,
                )
            )
        except Exception as exc:
            errors.append(f"{view}: {exc}")

    if not raw_rows:
        raise RuntimeError("YOLO scan produced no frame rows" + (": " + "; ".join(errors) if errors else ""))

    view_alignment = _apply_view_alignment_from_yolo(manifest, raw_rows, paths)
    yolo_rows = _normalize_yolo_rows_for_pipeline(manifest, raw_rows, config)
    frame_rows = _frame_score_rows_from_yolo(manifest, yolo_rows, config)
    pseudo_source = VideoSource(
        name="global_multiview",
        path="global_multiview",
        start_time=manifest.session_start_time,
        fps=float(config.parent_sample_fps or config.sample_fps),
        offset_sec=0.0,
    )
    segment_rows = [
        {
            **row,
            "time_sec": float(row.get("local_time_sec", row.get("time_sec", 0.0)) or 0.0),
            "video_start_time": manifest.session_start_time,
            "source_view": "global_multiview",
            "video_path": "global_multiview",
        }
        for row in frame_rows
    ]
    parent_sample_fps = float(config.parent_sample_fps or config.sample_fps)
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
    for segment in detected_segments:
        labels, interaction_count = _segment_yolo_stats(segment, frame_rows)
        segment.detector_backend = "yolo_interaction"
        segment.detector_source_view = "multiview" if scan_both else str(frame_rows[0].get("source_view") or preferred)
        if not str(segment.start_reason).startswith("yolo_physical_evidence"):
            segment.start_reason = "yolo_active_score_above_threshold"
        if not str(segment.end_reason).startswith("yolo_physical_evidence"):
            segment.end_reason = "yolo_active_score_below_threshold"
        segment.yolo_label_counts = labels
        segment.yolo_interaction_count = interaction_count

    label_counts: Counter[str] = Counter()
    interaction_count = 0
    for row in frame_rows:
        label_counts.update({str(key): int(value) for key, value in (row.get("label_counts") or {}).items()})
        interaction_count += len(row.get("hand_object_interactions") or [])
    summary = {
        "available": True,
        "detector_backend": "yolo_interaction",
        "model_path": model_paths_by_view.get(preferred),
        "model_paths_by_view": model_paths_by_view,
        "class_schema_path": class_schema.get("path") if isinstance(class_schema, dict) else None,
        "annotation_asset_count": len(annotation_asset_refs),
        "scan_both_views": scan_both,
        "views": views,
        "frame_rows": len(yolo_rows),
        "frame_score_rows": len(frame_rows),
        "segment_count": len(detected_segments),
        "boundary_refined_segment_count": sum(
            1 for segment in detected_segments if str(getattr(segment, "boundary_source", "")).startswith("yolo_physical_evidence")
        ),
        "view_alignment": view_alignment,
        "label_counts": dict(label_counts),
        "interaction_count": interaction_count,
        "errors": errors,
    }
    write_jsonl(paths["cv_outputs"] / "yolo_frame_rows.jsonl", yolo_rows)
    write_jsonl(paths["cv_outputs"] / "frame_scores.jsonl", frame_rows)
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
        from .yolo_detector import scan_yolo_video

        preferred = str(config.yolo_preferred_view or "first_person")
        if preferred == "first_person" and manifest.videos.first_person is None:
            preferred = "third_person"
        scan_both = _boolish(config.yolo_scan_both_views) and manifest.videos.first_person is not None
        views = ["first_person", "third_person"] if scan_both else [preferred]
        raw_rows: list[dict[str, Any]] = []
        model_paths_by_view: dict[str, str | None] = {}
        class_schema, annotation_asset_refs = _yolo_inventory_refs(paths)
        for view in views:
            model_path = _yolo_model_path_for_view(config, view)
            model_paths_by_view[view] = str(model_path) if model_path else None
            raw_rows.extend(
                scan_yolo_video(
                    first_person_path=manifest.videos.first_person,
                    third_person_path=manifest.videos.third_person,
                    preferred_view=view,
                    source_view=view,
                    model_path=model_path,
                    model_ref={
                        "path": str(model_path) if model_path else None,
                        "backend": "ultralytics_yolo",
                        "view": view,
                        "scan_role": "micro_refine",
                    },
                    class_schema=class_schema,
                    annotation_asset_refs=annotation_asset_refs,
                    sample_fps=float(config.micro_refine_sample_fps),
                    conf=config.yolo_conf,
                    iou=config.yolo_iou,
                    device=config.yolo_device,
                    class_thresholds=config.yolo_class_thresholds,
                    active_threshold=config.start_threshold,
                    continuity_frames=config.yolo_continuity_frames,
                    dry_run=dry_run,
                )
            )
        normalized = _normalize_yolo_rows_for_pipeline(manifest, raw_rows, config)
        segment_windows = [
            (parse_time(segment.global_start_time), parse_time(segment.global_end_time))
            for segment in key_segments
        ]
        refined = [
            row
            for row in normalized
            if any(start <= parse_time(str(row.get("global_time"))) <= end for start, end in segment_windows)
        ]
        write_jsonl(paths["cv_outputs"] / "yolo_micro_frame_rows.jsonl", refined)
        summary = {
            "available": True,
            "enabled": True,
            "source": "refined_yolo_micro_frame_rows",
            "sample_fps": float(config.micro_refine_sample_fps),
            "rows": len(refined),
            "model_path": model_paths_by_view.get(preferred),
            "model_paths_by_view": model_paths_by_view,
            "class_schema_path": class_schema.get("path") if isinstance(class_schema, dict) else None,
            "annotation_asset_count": len(annotation_asset_refs),
            "preferred_view": preferred,
            "views": views,
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
) -> tuple[list[Any], list[Any], list[dict[str, Any]], dict[str, Any]]:
    backend = str(config.detector_backend or "motion").lower()
    if backend in {"yolo", "yolo_interaction", "multiview_yolo"}:
        try:
            return _run_yolo_segment_detection(manifest, paths, config, dry_run=dry_run)
        except Exception as exc:
            _write_json(paths["metadata"] / "yolo_scan_summary.json", {"available": False, "error": str(exc), "fallback": "motion"})
            _write_json(paths["metadata"] / "yolo_frame_scan.json", {"available": False, "error": str(exc), "frames": [], "sample_frames": []})
            if not _boolish(config.yolo_fallback_to_motion):
                raise
            print(f"[pipeline] YOLO detector failed; falling back to motion baseline: {exc}")

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
    return detected_segments, scores, [], {"available": True, "detector_backend": "motion", "segment_count": len(detected_segments)}


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


def run_pipeline(
    manifest_path: str | Path,
    dry_run: bool = False,
    detector_config: DetectorConfig | None = None,
) -> dict[str, Any]:
    print(f"[pipeline] Loading manifest: {manifest_path}")
    manifest = SessionManifest.load(manifest_path)
    output_dir = Path(manifest.output_dir)
    paths = _mkdirs(output_dir)

    print("[pipeline] Validating manifest and inputs")
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

    _copy_manifest(manifest_path, manifest, output_dir)
    print("[pipeline] Writing normalized video source metadata")
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
    print("[pipeline] Discovering local trained models and labeled datasets")
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

    print("[pipeline] Generating ROI preview")
    save_roi_preview(
        manifest.videos.third_person.path,
        manifest.workbench_roi,
        paths["debug"] / "roi_preview.jpg",
        dry_run=dry_run,
    )

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

    print("[pipeline] Detecting key action segments and writing frame_scores.jsonl")
    detected_segments, _scores, generated_yolo_rows, detector_summary = _detect_with_config(
        manifest,
        paths,
        active_detector_config,
        dry_run=dry_run,
    )
    write_jsonl(paths["cv_outputs"] / "detected_segments.jsonl", detected_segments)
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

    key_segments: list[KeyActionSegment] = []
    vector_metadata: list[VectorMetadata] = []
    yolo_frame_rows, yolo_rows_path = _load_yolo_frame_rows(paths)
    if generated_yolo_rows and not yolo_frame_rows:
        yolo_frame_rows = generated_yolo_rows
        yolo_rows_path = str(paths["cv_outputs"] / "yolo_frame_rows.jsonl")
    if yolo_rows_path:
        print(f"[pipeline] Loaded YOLO frame rows: {yolo_rows_path} ({len(yolo_frame_rows)} rows)")
    else:
        print("[pipeline] No YOLO frame rows found; using start/middle/end keyframes only")
    print("[pipeline] Extracting multiview clips and keyframes")
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
        key_segments.append(key_segment)
        vector_metadata.append(_vector_metadata_from_segment(key_segment))

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
    )
    micro_source_rows = refined_yolo_rows or yolo_frame_rows
    if dry_run and not micro_source_rows:
        micro_source_rows = _mock_yolo_micro_rows_for_dry_run(key_segments)
    micro_source_path = (
        str(paths["cv_outputs"] / "yolo_micro_frame_rows.jsonl")
        if refined_yolo_rows
        else yolo_rows_path
    )

    print("[pipeline] Generating YOLO interaction micro-segments")
    raw_micro_segments = generate_micro_segments(
        manifest=manifest,
        key_segments=key_segments,
        yolo_frame_rows=micro_source_rows,
        utterances=utterances,
        clips_dir=paths["clips"],
        keyframes_dir=paths["keyframes"],
        config=manifest.micro_segment_config,
        dry_run=dry_run,
    )
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
    write_jsonl(paths["metadata"] / "micro_segments_raw.jsonl", raw_micro_rows)
    write_jsonl(paths["metadata"] / "micro_segments.jsonl", micro_rows)
    write_jsonl(paths["metadata"] / "micro_vector_metadata.jsonl", micro_vector_metadata)
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
    print("[pipeline] Extracting experiment focus dual-view clips")
    try:
        experiment_focus_summary = extract_experiment_focus_clips(output_dir, dry_run=dry_run)
    except Exception as exc:
        experiment_focus_summary = {"available": False, "error": str(exc)}
        _write_json(paths["metadata"] / "experiment_focus_clips.json", experiment_focus_summary)
    combined_vector_metadata = [to_json_dict(item) for item in vector_metadata] + micro_vector_metadata
    write_jsonl(paths["metadata"] / "vector_metadata.jsonl", combined_vector_metadata)

    print("[pipeline] Building vector index")
    index = VectorIndex()
    index.build([str(item.get("index_text") or "") for item in combined_vector_metadata], combined_vector_metadata)
    index.save(paths["index"])
    write_jsonl(paths["index"] / "docstore.jsonl", combined_vector_metadata)

    segment_index = VectorIndex()
    segment_index.build([item.index_text for item in vector_metadata], vector_metadata)
    segment_index.save(paths["index"] / "segments")
    micro_index = VectorIndex()
    micro_index.build([str(item.get("index_text") or "") for item in micro_vector_metadata], micro_vector_metadata)
    micro_index.save(paths["index"] / "micro_segments")

    print("[pipeline] Generating unified multimodal timeline")
    unified_timeline_summary = generate_unified_timeline(
        manifest_path=manifest_path,
        output_dir=paths["metadata"],
        user_events_path=input_ingestion_summary["artifacts"]["user_text"],
        ai_events_path=input_ingestion_summary["artifacts"]["ai_reply"],
        uploads_path=input_ingestion_summary["artifacts"]["upload"],
        dry_run=dry_run,
    )

    print("[pipeline] Building material asset catalog")
    material_library_summary = build_material_asset_catalog(output_dir)

    print("[pipeline] Building state-change index")
    state_change_summary = build_state_change_index(output_dir)

    print("[pipeline] Converting YOLO detections into model observation inputs")
    yolo_observation_input_summary = build_yolo_observation_inputs(output_dir)

    print("[pipeline] Bridging LabSOPGuard model signals into state observation inputs")
    lab_model_signal_input_summary = build_lab_model_signal_inputs(output_dir)

    print("[pipeline] Normalizing external model observation outputs")
    model_observation_summary = build_model_observation_events(
        output_dir,
        output_path=paths["metadata"] / "model_observation_events.jsonl",
    )

    print("[pipeline] Building advanced vision evidence")
    advanced_vision_summary = build_advanced_vision_evidence(output_dir)
    print("[pipeline] Refreshing material asset catalog with advanced evidence refs")
    material_library_summary = build_material_asset_catalog(output_dir)

    print("[pipeline] Building structured video understanding")
    video_understanding_summary = build_video_understanding(output_dir)

    print("[pipeline] Building experiment context")
    experiment_context_summary = build_experiment_context(output_dir)

    print("[pipeline] Building experiment process reasoning")
    experiment_process_summary = build_experiment_process(output_dir)

    print("[pipeline] Building human confirmation queue")
    confirmation_queue_summary = build_confirmation_queue(output_dir)

    print("[pipeline] Building process quality assurance report")
    quality_assurance_summary = build_quality_assurance_report(output_dir)

    print("[pipeline] Building final process record and audit report")
    process_record_summary = build_process_record(output_dir)

    print("[pipeline] Validating JSON/JSONL artifact schemas")
    artifact_validation_summary = validate_session_artifacts(
        output_dir,
        artifact_types=[
            "model_observation_events",
            "video_understanding",
            "experiment_context",
            "sop_state_machine",
            "experiment_process",
            "process_record",
            "asset_catalog",
            "confirmation_queue",
            "process_quality_report",
        ],
        output_path=paths["metadata"] / "artifact_validation_report.json",
    )
    quality_assurance_summary = build_quality_assurance_report(output_dir)

    print("[pipeline] Building pipeline evaluation report")
    pipeline_evaluation_summary = build_pipeline_evaluation_report(output_dir)

    print("[pipeline] Building data governance report")
    data_governance_summary = build_data_governance_report(output_dir)

    print("[pipeline] Running query smoke test")
    try:
        smoke_results = index.query(DEFAULT_QUERY, top_k=3)
    except Exception as exc:
        smoke_results = [{"error": str(exc)}]
    _write_json(paths["reports"] / "query_smoke_test.json", {"query": DEFAULT_QUERY, "results": smoke_results})

    report_path = paths["reports"] / "mvp_validation_report.md"
    formal_report_path = paths["reports"] / "formal_validation_report.md"
    formal_report_path = paths["reports"] / "formal_validation_report.md"
    total_action_duration = float(sum(segment.duration_sec for segment in detected_segments))
    summary = {
        "session_id": manifest.session_id,
        "output_dir": str(output_dir),
        "dry_run": dry_run,
        "segment_count": len(key_segments),
        "total_action_duration_sec": total_action_duration,
        "artifacts": {
            "detected_segments": str(paths["cv_outputs"] / "detected_segments.jsonl"),
            "experiment_episodes": str(paths["metadata"] / "experiment_episodes.jsonl"),
            "frame_scores": str(paths["cv_outputs"] / "frame_scores.jsonl"),
            "video_sources": str(paths["metadata"] / "video_sources.jsonl"),
            "user_text_events": input_ingestion_summary["artifacts"]["user_text"],
            "ai_reply_events": input_ingestion_summary["artifacts"]["ai_reply"],
            "upload_events": input_ingestion_summary["artifacts"]["upload"],
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
            "formal_report": str(formal_report_path),
            "roi_preview": str(paths["debug"] / "roi_preview.jpg"),
            "frame_score_plot": str(paths["debug"] / "frame_scores.png"),
            "segments_contact_sheet": str(paths["debug"] / "segments_contact_sheet.jpg"),
            "yolo_frame_rows": yolo_rows_path,
            "yolo_micro_frame_rows": micro_source_path,
        },
        "detector_summary": detector_summary,
        "experiment_episode_count": len(experiment_episode_rows),
        "video_source_count": len(video_source_rows),
        "input_ingestion_summary": input_ingestion_summary,
        "session_context_summary": session_context_summary,
        "record_ingestion_summary": record_ingestion_summary,
        "long_video_plan_summary": {
            "chunk_count": long_video_plan.get("chunk_count"),
            "pending_chunk_count": long_video_plan.get("pending_chunk_count"),
            "completed_chunk_count": long_video_plan.get("completed_chunk_count"),
            "cache_enabled": long_video_plan.get("cache_enabled"),
            "resume_enabled": long_video_plan.get("resume_enabled"),
            "two_stage_sampling": long_video_plan.get("two_stage_sampling"),
            "chunk_sec": long_video_plan.get("chunk_sec"),
        },
        "history_model_summary": {
            "session_count": history_model.get("session_count", 0),
            "event_count": history_model.get("event_count", 0),
            "action_counts": history_model.get("action_counts", {}),
        },
        "model_inventory_summary": {
            "primary_model": model_inventory.get("primary_model", {}),
            "model_count": model_inventory.get("model_count", 0),
            "dataset_count": model_inventory.get("dataset_count", 0),
            "capabilities": model_inventory.get("capabilities", {}),
        },
        "capability_gap_summary": capability_gap_report.get("summary", {}),
        "micro_refine_summary": micro_refine_summary,
        "micro_segment_count": len(micro_rows),
        "raw_micro_segment_count": len(raw_micro_rows),
        "micro_merge_stats": micro_merge_stats,
        "micro_quality_stats": micro_quality_stats,
        "experiment_focus_summary": experiment_focus_summary,
        "unified_timeline_summary": unified_timeline_summary,
        "state_change_summary": state_change_summary,
        "material_library_summary": material_library_summary,
        "yolo_observation_input_summary": yolo_observation_input_summary,
        "lab_model_signal_input_summary": lab_model_signal_input_summary,
        "advanced_vision_summary": advanced_vision_summary,
        "model_observation_summary": model_observation_summary,
        "video_understanding_summary": video_understanding_summary,
        "experiment_context_summary": {
            "confidence": experiment_context_summary.get("confidence"),
            "purpose": experiment_context_summary.get("purpose"),
            "procedure_count": len(experiment_context_summary.get("procedure_candidates") or []),
            "material_count": len(experiment_context_summary.get("materials") or []),
            "parameter_count": len(experiment_context_summary.get("parameters") or []),
            "gaps": experiment_context_summary.get("gaps", []),
        },
        "experiment_process_summary": {
            "process_status": experiment_process_summary.get("process_status"),
            "step_count": experiment_process_summary.get("step_count"),
            "status_counts": experiment_process_summary.get("status_counts", {}),
            "current_step_id": experiment_process_summary.get("current_step_id"),
            "next_step_id": experiment_process_summary.get("next_step_id"),
        },
        "confirmation_queue_summary": confirmation_queue_summary,
        "quality_assurance_summary": {
            "overall_status": quality_assurance_summary.get("overall_status"),
            "overall_score": quality_assurance_summary.get("overall_score"),
            "status_counts": quality_assurance_summary.get("status_counts", {}),
            "scorecard": quality_assurance_summary.get("scorecard", {}),
            "diagnostics": quality_assurance_summary.get("diagnostics", {}),
        },
        "process_record_summary": process_record_summary.get("summary", {}),
        "artifact_validation_summary": {
            "valid": artifact_validation_summary.get("valid"),
            "artifact_count": artifact_validation_summary.get("artifact_count"),
            "error_count": artifact_validation_summary.get("error_count"),
            "issue_count": artifact_validation_summary.get("issue_count"),
        },
        "pipeline_evaluation_summary": {
            "overall_score": pipeline_evaluation_summary.get("overall_score"),
            "scores": pipeline_evaluation_summary.get("scores", {}),
        },
        "data_governance_summary": {
            "record_count": data_governance_summary.get("record_count"),
            "missing_hash_count": data_governance_summary.get("missing_hash_count"),
            "missing_governance_field_count": data_governance_summary.get("missing_governance_field_count"),
            "privacy_level_counts": data_governance_summary.get("privacy_level_counts", {}),
        },
    }
    summary["error_diagnostics"] = _build_error_diagnostics(summary, quality_assurance_summary, artifact_validation_summary)
    summary["module_runs"] = _module_runs_from_summary(summary)
    _write_json(output_dir / "pipeline_summary.json", summary)
    print("[pipeline] Generating validation report")
    generate_report(output_dir, output_path=report_path)
    generate_formal_validation_report(output_dir, output_path=formal_report_path)
    print("[pipeline] Completed")
    return summary


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
    model_inventory = _discover_lab_assets_for_pipeline(
        manifest,
        output_dir,
        paths["metadata"] / "model_inventory.json",
        dry_run=dry_run,
    )
    detected_segments, _scores, _yolo_rows, detector_summary = _detect_with_config(
        manifest,
        paths,
        active_detector_config,
        dry_run=dry_run,
    )
    write_jsonl(paths["cv_outputs"] / "detected_segments.jsonl", detected_segments)
    save_frame_score_plot(
        paths["cv_outputs"] / "frame_scores.jsonl",
        paths["cv_outputs"] / "detected_segments.jsonl",
        paths["debug"] / "frame_scores.png",
    )
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
