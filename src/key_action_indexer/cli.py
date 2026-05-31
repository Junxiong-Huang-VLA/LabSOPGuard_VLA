"""CLI entrypoint for key_action_indexer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import sys
from pathlib import Path
from typing import Any

from .acceptance_pipeline import add_acceptance_pipeline_parser, add_validate_artifacts_parser
from .advanced_vision_evidence import build_advanced_vision_evidence
from .asset_library import build_material_asset_catalog
from .batch_refresh import batch_refresh_sessions
from .boss_report import generate_boss_acceptance_report
from .confirmation_loop import apply_confirmation_batch_decisions, build_confirmation_queue
from .context_fusion import build_experiment_context
from .derived_refresh import refresh_derived_artifacts
from .evidence_adapter_validation import validate_evidence_adapters
from .evidence_package import (
    build_evidence_package,
    evaluate_evidence_package_queries,
    query_evidence_package,
    validate_evidence_package,
)
from .evaluation_manifest import build_micro_gt_template_manifest
from .export_interfaces import export_artifact_bundle
from .frontend_sync import sync_frontend_artifacts, validate_frontend_artifact_sync
from .history_learning import build_history_model
from .health_report import build_run_health_report
from .lab_model_signal_inputs import build_lab_model_signal_inputs
from .lightweight_context_import import import_lightweight_context
from .material_search import search_material_assets
from .material_references import build_yolo_material_candidates
from .material_reference_index import build_key_material_reference_index, query_key_material_reference_index
from .material_library_store import query_material_library, sync_material_library
from .missing_step_recovery import build_missing_step_recovery_plan
from .micro_coverage_backfill import backfill_micro_coverage
from .micro_quality_enrichment import enrich_micro_quality
from .model_inventory import discover_lab_assets
from .model_observations import build_model_observation_events
from .pipeline import run_detection_only, run_pipeline
from .process_reasoner import build_experiment_process
from .promotion_readiness import build_promotion_readiness_report
from .quality_gate import build_quality_gate
from .query_validation import query_index, query_session_index, validate_queries
from .report import generate_report
from .retrieval_eval import build_default_chinese_query_eval_config, build_gold_query_benchmark, confirm_gold_query_benchmark, run_default_chinese_query_eval
from .review_bundle import apply_review_to_process, export_review_bundle
from .review_packet import build_recovery_review_packet
from .reviewed_dataset import freeze_reviewed_dataset, promote_reviewed_release, rollback_reviewed_release
from .schemas import DetectionConfig
from .schemas import SessionManifest
from .session_context_seed import seed_session_context
from .session_audit import build_session_audit_report
from .scope_config import build_stage_scope
from .sop_compliance import build_sop_compliance_report
from .state_index import build_state_change_index
from .transcript_convert import convert_transcript_to_jsonl
from .tuning import parse_float_list, tune_detector
from .video_understanding import build_video_understanding
from .video_memory import (
    DEFAULT_BUNDLE_VLM_MODEL,
    DEFAULT_ITEM_VLM_MODEL,
    VLM_MODE_OFFLINE,
    VLM_MODE_REAL_QWEN_ASYNC,
    VLM_MODE_REUSE_EXISTING,
    build_video_memory,
    get_memory_snapshot,
    get_video_memory_rebuild_background_status,
    query_video_memory,
    record_human_feedback,
    start_video_memory_rebuild_background,
)
from .yolo_observation_inputs import build_yolo_observation_inputs


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _float_list(value: str | None) -> list[float] | None:
    if value is None:
        return None
    return parse_float_list(value, 0.0)


def _combined_cli_text(inline_values: list[str] | None, file_values: list[str] | None) -> str:
    chunks = [value for value in inline_values or [] if value]
    for file_value in file_values or []:
        chunks.append(Path(file_value).read_text(encoding="utf-8-sig"))
    return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())


def _compact_material_library_cli_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "material_id": row.get("material_id"),
        "experiment_id": row.get("experiment_id"),
        "session_id": row.get("session_id"),
        "date": row.get("date"),
        "experiment_title": row.get("experiment_title"),
        "package_name": row.get("package_name"),
        "asset_type": row.get("asset_type"),
        "asset_kind": row.get("asset_kind"),
        "display_name": row.get("display_name"),
        "action_name": row.get("action_name"),
        "segment_id": row.get("segment_id"),
        "micro_segment_id": row.get("micro_segment_id"),
        "view": row.get("view"),
        "time_start": row.get("start_sec"),
        "time_end": row.get("end_sec"),
        "primary_object": row.get("primary_object"),
        "canonical_action_type": row.get("canonical_action_type"),
        "secondary_objects": row.get("secondary_objects") or [],
        "secondary_actions": row.get("secondary_actions") or [],
        "stored_file": row.get("stored_file"),
        "absolute_path": row.get("absolute_path"),
        "keyframe_path": row.get("absolute_path") if row.get("asset_type") == "keyframe" else "",
        "keyclip_path": row.get("absolute_path") if row.get("asset_type") == "video_clip" else "",
        "package_uri": row.get("package_uri"),
        "exists": row.get("exists"),
        "size_bytes": row.get("size_bytes"),
        "sha256": row.get("sha256"),
    }


def _add_detector_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detector-backend",
        choices=["motion", "yolo", "yolo_interaction", "multiview_yolo"],
        default=None,
        help="Select detector backend: motion (default), yolo, yolo_interaction, or multiview_yolo",
    )
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--parent-sample-fps", type=float, default=None)
    parser.add_argument("--micro-refine-sample-fps", type=float, default=None)
    parser.add_argument(
        "--enable-micro-refine-rescan",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run a denser YOLO rescan inside detected key-action windows for micro-segment recall.",
    )
    parser.add_argument("--start-threshold", type=float, default=None)
    parser.add_argument("--end-threshold", type=float, default=None)
    parser.add_argument("--start-min-duration-sec", type=float, default=None)
    parser.add_argument("--end-min-duration-sec", type=float, default=None)
    parser.add_argument("--merge-gap-sec", type=float, default=None)
    parser.add_argument("--min-segment-duration-sec", type=float, default=None)
    parser.add_argument("--buffer-sec", type=float, default=None)
    parser.add_argument("--motion-normalization", default=None)
    parser.add_argument("--roi-mode", default=None)
    parser.add_argument("--yolo-preferred-view", choices=["first_person", "third_person"], default=None)
    parser.add_argument("--yolo-scan-both-views", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yolo-model-path", default=None)
    parser.add_argument("--yolo-first-person-model-path", default=None)
    parser.add_argument("--yolo-third-person-model-path", default=None)
    parser.add_argument("--yolo-conf", type=float, default=None)
    parser.add_argument("--yolo-iou", type=float, default=None)
    parser.add_argument("--yolo-device", default=None)
    parser.add_argument("--yolo-imgsz", type=int, default=None)
    parser.add_argument("--yolo-adaptive-imgsz", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--yolo-min-imgsz", type=int, default=None)
    parser.add_argument("--yolo-max-imgsz", type=int, default=None)
    parser.add_argument("--yolo-continuity-frames", type=int, default=None)
    parser.add_argument(
        "--yolo-fallback-to-motion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fallback to motion baseline when YOLO fails or is unavailable",
    )
    parser.add_argument("--long-video-chunk-sec", type=float, default=None)
    parser.add_argument("--long-video-stage1-sample-fps", type=float, default=None)
    parser.add_argument("--long-video-stage2-sample-fps", type=float, default=None)
    parser.add_argument(
        "--long-video-two-stage-sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a coarse YOLO pass over the long video, then rescan only detected experiment windows.",
    )


def _build_detector_config(manifest_path: str | Path, args: argparse.Namespace) -> DetectionConfig:
    manifest = SessionManifest.load(manifest_path)
    base = manifest.detection_config
    values = asdict(base)
    overrides = {
        "detector_backend": args.detector_backend,
        "sample_fps": args.sample_fps,
        "parent_sample_fps": args.parent_sample_fps,
        "micro_refine_sample_fps": args.micro_refine_sample_fps,
        "enable_micro_refine_rescan": args.enable_micro_refine_rescan,
        "start_threshold": args.start_threshold,
        "end_threshold": args.end_threshold,
        "start_min_duration_sec": args.start_min_duration_sec,
        "end_min_duration_sec": args.end_min_duration_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "min_segment_duration_sec": args.min_segment_duration_sec,
        "buffer_sec": args.buffer_sec,
        "motion_normalization": args.motion_normalization,
        "roi_mode": args.roi_mode,
        "yolo_preferred_view": args.yolo_preferred_view,
        "yolo_scan_both_views": args.yolo_scan_both_views,
        "yolo_model_path": args.yolo_model_path,
        "yolo_first_person_model_path": args.yolo_first_person_model_path,
        "yolo_third_person_model_path": args.yolo_third_person_model_path,
        "yolo_conf": args.yolo_conf,
        "yolo_iou": args.yolo_iou,
        "yolo_device": args.yolo_device,
        "yolo_imgsz": args.yolo_imgsz,
        "yolo_adaptive_imgsz": args.yolo_adaptive_imgsz,
        "yolo_min_imgsz": args.yolo_min_imgsz,
        "yolo_max_imgsz": args.yolo_max_imgsz,
        "yolo_continuity_frames": args.yolo_continuity_frames,
        "yolo_fallback_to_motion": args.yolo_fallback_to_motion,
        "long_video_chunk_sec": args.long_video_chunk_sec,
        "long_video_stage1_sample_fps": args.long_video_stage1_sample_fps,
        "long_video_stage2_sample_fps": args.long_video_stage2_sample_fps,
        "long_video_two_stage_sampling": args.long_video_two_stage_sampling,
    }
    filtered = {key: value for key, value in overrides.items() if value is not None}
    values.update(filtered)
    return DetectionConfig.from_dict(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="key-action-indexer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run full key action indexing pipeline")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--alignment-report", default=None, help="Output path for alignment health JSON report")
    run_parser.add_argument("--alignment-window", type=int, default=None, help="Sliding window size for drift estimation")
    run_parser.add_argument("--alignment-alert-threshold", type=float, default=None, help="Drift alert threshold in seconds")
    run_parser.add_argument("--alignment-smoothing", type=float, default=None, help="EMA smoothing alpha for alignment offsets")
    run_parser.add_argument("--print-manifest", action="store_true", help="Print run manifest to console")
    run_parser.add_argument("--manifest-output", default=None, help="Output path for run manifest JSON")
    _add_detector_override_args(run_parser)

    detect_parser = subparsers.add_parser("detect", help="Run key action detection only")
    detect_parser.add_argument("--manifest", required=True)
    detect_parser.add_argument("--dry-run", action="store_true")
    _add_detector_override_args(detect_parser)

    advanced_parser = subparsers.add_parser("advanced-vision", help="Build advanced visual evidence")
    advanced_parser.add_argument("--session-dir", required=True)
    advanced_parser.add_argument("--output")

    yolo_obs_parser = subparsers.add_parser("yolo-observation-inputs", help="Build object-track observation inputs from YOLO rows")
    yolo_obs_parser.add_argument("--session-dir", required=True)
    yolo_obs_parser.add_argument("--output")
    yolo_obs_parser.add_argument("--min-points", type=int, default=2)
    yolo_obs_parser.add_argument("--min-motion-px", type=float, default=5.0)
    yolo_obs_parser.add_argument("--max-points-per-track", type=int, default=160)

    lab_signal_parser = subparsers.add_parser("lab-model-signal-inputs", help="Bridge YOLO/model signals into observation inputs")
    lab_signal_parser.add_argument("--session-dir", required=True)
    lab_signal_parser.add_argument("--min-confidence", type=float, default=0.25)
    lab_signal_parser.add_argument("--output-summary")

    model_obs_parser = subparsers.add_parser("model-observations", help="Normalize external model outputs")
    model_obs_parser.add_argument("--session-dir", required=True)
    model_obs_parser.add_argument("--output")

    understand_parser = subparsers.add_parser("understand-video", help="Build structured video understanding events")
    understand_parser.add_argument("--session-dir", required=True)
    understand_parser.add_argument("--output")

    context_parser = subparsers.add_parser("context", help="Build fused experiment context")
    context_parser.add_argument("--session-dir", required=True)
    context_parser.add_argument("--output")
    context_parser.add_argument("--database", action="append", default=[])

    context_text_parser = subparsers.add_parser("context-text", help="Import lightweight SOP, note, or record text anchors")
    context_text_parser.add_argument("--session-dir", required=True)
    context_text_parser.add_argument("--sop-text", action="append", default=[])
    context_text_parser.add_argument("--sop-file", action="append", default=[])
    context_text_parser.add_argument("--note-text", action="append", default=[])
    context_text_parser.add_argument("--note-file", action="append", default=[])
    context_text_parser.add_argument("--record-text", action="append", default=[])
    context_text_parser.add_argument("--record-file", action="append", default=[])
    context_text_parser.add_argument("--output-summary")

    process_parser = subparsers.add_parser("process", help="Build experiment process reasoning")
    process_parser.add_argument("--session-dir", required=True)
    process_parser.add_argument("--sop")
    process_parser.add_argument("--output")
    process_parser.add_argument("--timeline-output")

    history_parser = subparsers.add_parser("history-model", help="Build a local historical experiment model")
    history_parser.add_argument("--source", action="append", required=True)
    history_parser.add_argument("--output", required=True)

    inventory_parser = subparsers.add_parser("model-inventory", help="Discover available model and dataset assets")
    inventory_parser.add_argument("--project-root")
    inventory_parser.add_argument("--output")

    stage_scope_parser = subparsers.add_parser("stage-scope", help="Write the current key-action stage scope")
    stage_scope_parser.add_argument("--session-dir", required=True)
    stage_scope_parser.add_argument("--output")
    stage_scope_parser.add_argument("--scope-name")

    confirmation_parser = subparsers.add_parser("confirmation-queue", help="Build human confirmation queue")
    confirmation_parser.add_argument("--session-dir", required=True)
    confirmation_parser.add_argument("--output")

    confirmation_batch_parser = subparsers.add_parser("confirmation-batch", help="Apply batch human confirmation decisions")
    confirmation_batch_parser.add_argument("--session-dir", required=True)
    confirmation_batch_parser.add_argument("--decisions", required=True)
    confirmation_batch_parser.add_argument("--output")
    confirmation_batch_parser.add_argument("--reviewer", default="system")
    confirmation_batch_parser.add_argument("--note", default="")

    context_seed_parser = subparsers.add_parser("context-seed", help="Seed non-label operational session context")
    context_seed_parser.add_argument("--session-dir", required=True)
    context_seed_parser.add_argument("--force", action="store_true")
    context_seed_parser.add_argument("--output-summary")

    refresh_parser = subparsers.add_parser("refresh-derived", help="Refresh no-label derived artifacts after metadata changes")
    refresh_parser.add_argument("--session-dir", required=True)
    refresh_parser.add_argument("--query", action="append", default=[])
    refresh_parser.add_argument("--output-summary")

    micro_coverage_parser = subparsers.add_parser("micro-coverage", help="Backfill retrieval-only micro coverage for parent segments without micro rows")
    micro_coverage_parser.add_argument("--session-dir", required=True)
    micro_coverage_parser.add_argument("--output-summary")

    batch_refresh_parser = subparsers.add_parser("batch-refresh", help="Refresh derived artifacts for multiple key-action sessions")
    batch_refresh_parser.add_argument("--source", action="append", required=True)
    batch_refresh_parser.add_argument("--query", action="append", default=[])
    batch_refresh_parser.add_argument("--output-summary")
    batch_refresh_parser.add_argument("--stop-on-error", action="store_true")

    audit_parser = subparsers.add_parser("audit-sessions", help="Build a cross-session key-action audit report")
    audit_parser.add_argument("--source", action="append", required=True)
    audit_parser.add_argument("--query", action="append", default=[])
    audit_parser.add_argument("--output-json")
    audit_parser.add_argument("--output-md")

    promotion_audit_parser = subparsers.add_parser("promotion-audit", help="Audit reviewed-release promotion readiness across sessions")
    promotion_audit_parser.add_argument("--source", action="append", required=True)
    promotion_audit_parser.add_argument("--query-count", type=int, default=50)
    promotion_audit_parser.add_argument("--output-json")
    promotion_audit_parser.add_argument("--output-md")

    health_parser = subparsers.add_parser("health", help="Build a no-label key-action run health report")
    health_parser.add_argument("--session-dir", required=True)
    health_parser.add_argument("--output-json")
    health_parser.add_argument("--output-md")
    health_parser.add_argument("--query", action="append", default=[])
    health_parser.add_argument("--max-total-coverage-ratio", type=float, default=0.65)
    health_parser.add_argument("--max-longest-segment-ratio", type=float, default=0.5)
    health_parser.add_argument("--min-boundary-confidence", type=float, default=0.01)
    health_parser.add_argument("--max-human-queue-count", type=int, default=100)
    health_parser.add_argument("--fail-on", choices=["never", "error", "warning"], default="error")

    review_parser = subparsers.add_parser("review-bundle", help="Export a human review bundle")
    review_parser.add_argument("--session-dir", required=True)
    review_parser.add_argument("--output", required=True)
    review_parser.add_argument("--format", choices=["json", "md", "html"], default="json")

    recovery_packet_parser = subparsers.add_parser("recovery-review-packet", help="Build a readable packet from missing-step recovery JSON")
    recovery_packet_parser.add_argument("--plan", required=True)
    recovery_packet_parser.add_argument("--output-md", required=True)
    recovery_packet_parser.add_argument("--output-decisions")
    recovery_packet_parser.add_argument("--max-candidates", type=int, default=5)

    reviewed_dataset_parser = subparsers.add_parser("freeze-reviewed-dataset", help="Freeze review decisions into reviewed delivery artifacts")
    reviewed_dataset_parser.add_argument("--session-dir", required=True)

    rollback_reviewed_parser = subparsers.add_parser("rollback-reviewed-release", help="Rollback reviewed dataset to a previous reviewed release")
    rollback_reviewed_parser.add_argument("--session-dir", required=True)
    rollback_reviewed_parser.add_argument("--version")

    promote_reviewed_parser = subparsers.add_parser("promote-reviewed-release", help="Promote a reviewed release for default retrieval/export")
    promote_reviewed_parser.add_argument("--session-dir", required=True)
    promote_reviewed_parser.add_argument("--version")
    promote_reviewed_parser.add_argument("--reviewer", required=True)
    promote_reviewed_parser.add_argument("--note", default="")
    promote_reviewed_parser.add_argument("--query-count", type=int, default=50)

    quality_gate_parser = subparsers.add_parser("quality-gate", help="Build the key-action completion quality gate")
    quality_gate_parser.add_argument("--session-dir", required=True)
    quality_gate_parser.add_argument("--output")
    quality_gate_parser.add_argument("--strict", action="store_true")

    adapter_validation_parser = subparsers.add_parser("validate-evidence-adapters", help="Validate advanced evidence adapter JSONL inputs")
    adapter_validation_parser.add_argument("--session-dir", required=True)
    adapter_validation_parser.add_argument("--output")
    adapter_validation_parser.add_argument("--strict", action="store_true")

    review_apply_parser = subparsers.add_parser("apply-review", help="Apply a human review decision to experiment process artifacts")
    review_apply_parser.add_argument("--session-dir", required=True)
    review_apply_parser.add_argument("--confirmation-id", required=True)
    review_apply_parser.add_argument("--decision", required=True)
    review_apply_parser.add_argument("--operator", required=True)
    review_apply_parser.add_argument("--rationale", default="")

    missing_step_parser = subparsers.add_parser("missing-step-recovery", help="Plan recovery evidence for missing or low-confidence process steps")
    missing_step_parser.add_argument("--session-dir", required=True)
    missing_step_parser.add_argument("--output")
    missing_step_parser.add_argument("--confidence-threshold", type=float, default=0.5)
    missing_step_parser.add_argument("--window-padding-sec", type=float, default=5.0)

    export_parser = subparsers.add_parser("export-artifacts", help="Export validated artifacts for downstream systems")
    export_parser.add_argument("--session-dir", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--strict", action="store_true")

    boss_parser = subparsers.add_parser("boss-report", help="Generate a compact acceptance report")
    boss_parser.add_argument("--session-dir", required=True)
    boss_parser.add_argument("--output")

    assets_parser = subparsers.add_parser("assets", help="Build material asset catalog")
    assets_parser.add_argument("--session-dir", required=True)
    assets_parser.add_argument("--output")
    assets_parser.add_argument("--summary")

    material_candidates_parser = subparsers.add_parser("material-candidates", help="Rebuild YOLO-backed material review candidates")
    material_candidates_parser.add_argument("--session-dir", required=True)
    material_candidates_parser.add_argument("--dry-run", action="store_true")
    material_candidates_parser.add_argument("--ffmpeg-path", default="ffmpeg")
    material_candidates_parser.add_argument("--archive-existing", action=argparse.BooleanOptionalAction, default=True)
    material_candidates_parser.add_argument("--rebuild-source", action=argparse.BooleanOptionalAction, default=True)
    material_candidates_parser.add_argument("--enable-vlm", action=argparse.BooleanOptionalAction, default=False)
    material_candidates_parser.add_argument("--max-vlm-groups", type=int, default=8)
    material_candidates_parser.add_argument("--vlm-model-name")

    search_assets_parser = subparsers.add_parser("search-assets", help="Search material assets")
    search_assets_parser.add_argument("--session-dir", required=True)
    search_assets_parser.add_argument("--query", default="")
    search_assets_parser.add_argument("--asset-type")
    search_assets_parser.add_argument("--objects")
    search_assets_parser.add_argument("--actions")
    search_assets_parser.add_argument("--state-tags")
    search_assets_parser.add_argument("--start-time")
    search_assets_parser.add_argument("--end-time")
    search_assets_parser.add_argument("--limit", type=int, default=20)

    material_ref_index_parser = subparsers.add_parser("material-reference-index", help="Build a SQLite/JSONL index from a formal material reference folder")
    material_ref_index_parser.add_argument("--material-root", required=True)
    material_ref_index_parser.add_argument("--sqlite-output")
    material_ref_index_parser.add_argument("--references-output")
    material_ref_index_parser.add_argument("--include-reports", action="store_true")
    material_ref_index_parser.add_argument("--query", default="")
    material_ref_index_parser.add_argument("--asset-type")
    material_ref_index_parser.add_argument("--object")
    material_ref_index_parser.add_argument("--action")
    material_ref_index_parser.add_argument("--limit", type=int, default=20)

    material_library_sync_parser = subparsers.add_parser(
        "material-library-sync",
        help="Build the global SQLite index over D:/LabMaterialLibrary material packages",
    )
    material_library_sync_parser.add_argument("--library-root", default=None)
    material_library_sync_parser.add_argument("--sqlite-output")
    material_library_sync_parser.add_argument("--rebuild", action="store_true")
    material_library_sync_parser.add_argument("--include-reports", action="store_true")

    material_library_query_parser = subparsers.add_parser(
        "material-library-query",
        help="Query the global material library index",
    )
    material_library_query_parser.add_argument("--library-root", default=None)
    material_library_query_parser.add_argument("--sqlite-path")
    material_library_query_parser.add_argument("--query", default="")
    material_library_query_parser.add_argument("--asset-type")
    material_library_query_parser.add_argument("--object")
    material_library_query_parser.add_argument("--action")
    material_library_query_parser.add_argument("--view")
    material_library_query_parser.add_argument("--session-id")
    material_library_query_parser.add_argument("--experiment-id")
    material_library_query_parser.add_argument("--package-name")
    material_library_query_parser.add_argument("--date")
    material_library_query_parser.add_argument("--start-date")
    material_library_query_parser.add_argument("--end-date")
    material_library_query_parser.add_argument("--limit", type=int, default=20)
    material_library_query_parser.add_argument("--full", action="store_true", help="Print full indexed rows including raw payload text")

    video_memory_build_parser = subparsers.add_parser(
        "video-memory-build",
        help="Build the rolling 30-Day Video Memory snapshot from the material library",
    )
    video_memory_build_parser.add_argument("--library-root", default=None)
    video_memory_build_parser.add_argument("--sqlite-path")
    video_memory_build_parser.add_argument("--window-end-date")
    video_memory_build_parser.add_argument("--window-days", type=int, default=30)
    video_memory_build_parser.add_argument(
        "--job-type",
        choices=["backfill", "incremental", "rebuild", "feedback_update"],
        default="incremental",
    )
    video_memory_build_parser.add_argument("--force-material-sync", action="store_true")
    video_memory_build_parser.add_argument(
        "--vlm-mode",
        choices=[VLM_MODE_OFFLINE, VLM_MODE_REUSE_EXISTING, VLM_MODE_REAL_QWEN_ASYNC],
        default=VLM_MODE_OFFLINE,
        help="Video Memory VLM mode. Real Qwen requires an injected client in API integrations; CLI defaults to no-network fallback.",
    )
    video_memory_build_parser.add_argument("--item-vlm-model", default=DEFAULT_ITEM_VLM_MODEL, choices=["qwen3.5-flash", "qwen3.5-plus"])
    video_memory_build_parser.add_argument("--bundle-vlm-model", default=DEFAULT_BUNDLE_VLM_MODEL, choices=["qwen3.5-flash", "qwen3.5-plus"])
    video_memory_build_parser.add_argument("--max-real-vlm-items", type=int)
    video_memory_build_parser.add_argument("--max-real-vlm-bundles", type=int)
    video_memory_build_parser.add_argument(
        "--background",
        action="store_true",
        help="Start the rebuild through the in-process background hook for local API integrations.",
    )

    video_memory_rebuild_status_parser = subparsers.add_parser(
        "video-memory-rebuild-status",
        help="Inspect an in-process Video Memory background rebuild job",
    )
    video_memory_rebuild_status_parser.add_argument("--job-id", required=True)

    video_memory_snapshot_parser = subparsers.add_parser(
        "video-memory-snapshot",
        help="Print the latest or selected 30-Day Video Memory snapshot",
    )
    video_memory_snapshot_parser.add_argument("--library-root", default=None)
    video_memory_snapshot_parser.add_argument("--sqlite-path")
    video_memory_snapshot_parser.add_argument("--snapshot-id")

    video_memory_query_parser = subparsers.add_parser(
        "video-memory-query",
        help="Query 30-Day Video Memory with evidence-linked claims",
    )
    video_memory_query_parser.add_argument("--library-root", default=None)
    video_memory_query_parser.add_argument("--sqlite-path")
    video_memory_query_parser.add_argument("--snapshot-id")
    video_memory_query_parser.add_argument("--query", required=True)
    video_memory_query_parser.add_argument("--limit", type=int, default=5)

    video_memory_feedback_parser = subparsers.add_parser(
        "video-memory-feedback",
        help="Record lightweight MVP feedback for a cluster, ledger, bundle, or material",
    )
    video_memory_feedback_parser.add_argument("--library-root", default=None)
    video_memory_feedback_parser.add_argument("--sqlite-path")
    video_memory_feedback_parser.add_argument("--target-type", required=True)
    video_memory_feedback_parser.add_argument("--target-id", required=True)
    video_memory_feedback_parser.add_argument("--feedback-type", required=True)
    video_memory_feedback_parser.add_argument("--context-json", default="{}")
    video_memory_feedback_parser.add_argument("--sop-name")
    video_memory_feedback_parser.add_argument("--sample-name")
    video_memory_feedback_parser.add_argument("--project-name")
    video_memory_feedback_parser.add_argument("--note", default="")
    video_memory_feedback_parser.add_argument("--user-id", default="local_user")

    evidence_package_build_parser = subparsers.add_parser(
        "evidence-package-build",
        help="Build portable read-only evidence package sidecars from a material folder",
    )
    evidence_package_build_parser.add_argument("--package-root", required=True)
    evidence_package_build_parser.add_argument("--source-manifest")
    evidence_package_build_parser.add_argument("--key-action-index-dir")
    evidence_package_build_parser.add_argument("--package-id")
    evidence_package_build_parser.add_argument("--experiment-id")
    evidence_package_build_parser.add_argument("--sqlite-output")
    evidence_package_build_parser.add_argument("--references-output")
    evidence_package_build_parser.add_argument("--include-reports", action="store_true")

    evidence_package_validate_parser = subparsers.add_parser(
        "evidence-package-validate",
        help="Validate a portable read-only evidence package",
    )
    evidence_package_validate_parser.add_argument("--package-root", required=True)
    evidence_package_validate_parser.add_argument("--strict", action="store_true")

    evidence_package_query_parser = subparsers.add_parser(
        "evidence-package-query",
        help="Query a portable read-only evidence package",
    )
    evidence_package_query_parser.add_argument("--package-root", required=True)
    evidence_package_query_parser.add_argument("--query", required=True)
    evidence_package_query_parser.add_argument("--message-time")
    evidence_package_query_parser.add_argument("--limit", type=int, default=8)
    evidence_package_query_parser.add_argument("--window-before-sec", type=float)
    evidence_package_query_parser.add_argument("--window-after-sec", type=float)

    evidence_package_eval_parser = subparsers.add_parser(
        "evidence-package-eval",
        help="Evaluate a portable read-only evidence package with a query set",
    )
    evidence_package_eval_parser.add_argument("--package-root", required=True)
    evidence_package_eval_parser.add_argument("--queries", required=True)
    evidence_package_eval_parser.add_argument("--output")
    evidence_package_eval_parser.add_argument("--limit", type=int, default=8)

    sop_compliance_parser = subparsers.add_parser(
        "sop-compliance",
        help="Map local key actions, evidence refs, and physical_change_log rows to SOP compliance events",
    )
    sop_compliance_parser.add_argument("--sop", required=True)
    sop_compliance_parser.add_argument("--package-root")
    sop_compliance_parser.add_argument("--key-actions", action="append", default=[])
    sop_compliance_parser.add_argument("--evidence", action="append", default=[])
    sop_compliance_parser.add_argument("--physical-change-log", action="append", default=[])
    sop_compliance_parser.add_argument("--min-confidence", type=float, default=0.5)
    sop_compliance_parser.add_argument("--include-unmapped", action=argparse.BooleanOptionalAction, default=True)
    sop_compliance_parser.add_argument("--output")
    sop_compliance_parser.add_argument("--events-output")

    frontend_sync_parser = subparsers.add_parser(
        "frontend-sync",
        help="Synchronize and validate an offline key-action evidence run for a frontend experiment folder",
    )
    frontend_sync_parser.add_argument("--target-experiment-dir", required=True)
    frontend_sync_parser.add_argument("--source-session-dir")
    frontend_sync_parser.add_argument("--source-key-action-index-dir")
    frontend_sync_parser.add_argument("--source-material-root")
    frontend_sync_parser.add_argument("--experiment-id")
    frontend_sync_parser.add_argument("--experiment-title")
    frontend_sync_parser.add_argument("--third-person-video")
    frontend_sync_parser.add_argument("--first-person-video")
    frontend_sync_parser.add_argument("--archive-existing", action=argparse.BooleanOptionalAction, default=True)
    frontend_sync_parser.add_argument("--hardlink-media", action=argparse.BooleanOptionalAction, default=True)
    frontend_sync_parser.add_argument("--approve-materials", action=argparse.BooleanOptionalAction, default=True)
    frontend_sync_parser.add_argument("--refresh-focus", action=argparse.BooleanOptionalAction, default=True)
    frontend_sync_parser.add_argument("--force-refresh-focus", action="store_true")
    frontend_sync_parser.add_argument("--run-yolo-overlay", action="store_true")
    frontend_sync_parser.add_argument("--require-yolo-overlay", action=argparse.BooleanOptionalAction, default=True)
    frontend_sync_parser.add_argument("--min-focus-duration-sec", type=float, default=30.0)
    frontend_sync_parser.add_argument("--yolo-model-path")
    frontend_sync_parser.add_argument("--yolo-first-person-model-path")
    frontend_sync_parser.add_argument("--yolo-third-person-model-path")
    frontend_sync_parser.add_argument("--yolo-project-root")
    frontend_sync_parser.add_argument("--yolo-device", default="auto")
    frontend_sync_parser.add_argument("--yolo-conf", type=float, default=0.25)
    frontend_sync_parser.add_argument("--yolo-iou", type=float, default=0.45)
    frontend_sync_parser.add_argument("--yolo-detect-fps", type=float, default=5.0)
    frontend_sync_parser.add_argument("--dry-run", action="store_true")
    frontend_sync_parser.add_argument("--validate-only", action="store_true")
    frontend_sync_parser.add_argument("--output-summary")
    frontend_sync_parser.add_argument("--fail-on-error", action="store_true")

    tune_parser = subparsers.add_parser("tune", help="Tune detector thresholds against manual labels")
    tune_parser.add_argument("--manifest", required=True)
    tune_parser.add_argument("--ground-truth", required=True)
    tune_parser.add_argument("--start-threshold")
    tune_parser.add_argument("--end-threshold")
    tune_parser.add_argument("--merge-gap-sec")
    tune_parser.add_argument("--min-segment-duration-sec")
    tune_parser.add_argument("--iou-threshold", type=float, default=0.3)
    tune_parser.add_argument("--dry-run", action="store_true")
    tune_parser.add_argument("--run-detection", action="store_true")
    tune_parser.add_argument("--output")

    micro_gt_parser = subparsers.add_parser("micro-gt-template", help="Build a manual micro-GT labeling template")
    micro_gt_parser.add_argument("--session-dir")
    micro_gt_parser.add_argument("--micro-segments")
    micro_gt_parser.add_argument("--key-action-segments")
    micro_gt_parser.add_argument("--output-dir")
    micro_gt_parser.add_argument("--gt-completeness", default="unknown")

    micro_quality_parser = subparsers.add_parser("micro-quality", help="Refresh micro-segment continuity and low-signal evidence metadata")
    micro_quality_parser.add_argument("--session-dir", required=True)
    micro_quality_parser.add_argument("--output-report")
    micro_quality_parser.add_argument("--object", dest="objects", action="append")

    timeline_parser = subparsers.add_parser("timeline", help="Generate unified multimodal timeline artifacts")
    timeline_parser.add_argument("--manifest", required=True)
    timeline_parser.add_argument("--output")
    timeline_parser.add_argument("--user-events")
    timeline_parser.add_argument("--ai-events")
    timeline_parser.add_argument("--uploads")
    timeline_parser.add_argument("--calibration")
    timeline_parser.add_argument("--dry-run", action="store_true")

    transcript_parser = subparsers.add_parser("transcript-convert", help="Convert transcript files to aligned JSONL")
    transcript_parser.add_argument("--input", required=True)
    transcript_parser.add_argument("--output", required=True)
    transcript_parser.add_argument("--duration-sec", type=float)
    transcript_parser.add_argument("--summary-output")

    query_parser = subparsers.add_parser("query", help="Query a built vector index, or validate a query config")
    query_parser.add_argument("--session-dir")
    query_parser.add_argument("--index-dir")
    query_parser.add_argument("--query", action="append", default=[])
    query_parser.add_argument("--config", help="JSON query-validation config path")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument("--fusion-weights", default=None, help="JSON fusion weights for reranking, e.g. '{\"text_similarity\":0.6}'")
    query_parser.add_argument("--object", dest="objects", action="append")
    query_parser.add_argument("--action", dest="actions", action="append")
    query_parser.add_argument("--asset-type", dest="asset_types", action="append")
    query_parser.add_argument("--start-time")
    query_parser.add_argument("--end-time")
    query_parser.add_argument("--output")

    default_query_eval_parser = subparsers.add_parser("default-query-eval", help="Build and run the default Chinese key-action retrieval evaluation")
    default_query_eval_parser.add_argument("--session-dir", required=True)
    default_query_eval_parser.add_argument("--config-output")
    default_query_eval_parser.add_argument("--output")
    default_query_eval_parser.add_argument("--query-count", type=int, default=50)

    gold_query_parser = subparsers.add_parser("gold-query-benchmark", help="Build the fixed Chinese gold query benchmark scaffold")
    gold_query_parser.add_argument("--session-dir", required=True)
    gold_query_parser.add_argument("--output")
    gold_query_parser.add_argument("--query-count", type=int, default=50)
    gold_query_parser.add_argument("--overwrite", action="store_true")

    confirm_gold_parser = subparsers.add_parser("confirm-gold-query-benchmark", help="Mark the fixed Chinese gold benchmark as manually verified against reviewed release")
    confirm_gold_parser.add_argument("--session-dir", required=True)
    confirm_gold_parser.add_argument("--decisions", required=True, help="Human decision JSON/JSONL with query_id decisions and expected ids")
    confirm_gold_parser.add_argument("--output")
    confirm_gold_parser.add_argument("--query-count", type=int, default=50)
    confirm_gold_parser.add_argument("--reviewer", default="manual_reviewer")
    confirm_gold_parser.add_argument("--note", default="Human-verified against the current reviewed release.")

    report_parser = subparsers.add_parser("report", help="Generate validation report")
    report_parser.add_argument("--session-dir", required=True)

    add_acceptance_pipeline_parser(subparsers)
    add_validate_artifacts_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return int(args.func(args))

    if args.command == "run":
        detector_config = _build_detector_config(args.manifest, args)
        alignment_opts: dict[str, Any] = {}
        if getattr(args, "alignment_window", None) is not None:
            alignment_opts["alignment_window"] = args.alignment_window
        if getattr(args, "alignment_alert_threshold", None) is not None:
            alignment_opts["alignment_alert_threshold"] = args.alignment_alert_threshold
        if getattr(args, "alignment_smoothing", None) is not None:
            alignment_opts["alignment_smoothing"] = args.alignment_smoothing
        if getattr(args, "alignment_report", None) is not None:
            alignment_opts["alignment_report_path"] = args.alignment_report
        result = run_pipeline(
            args.manifest,
            dry_run=args.dry_run,
            detector_config=detector_config,
            alignment_options=alignment_opts or None,
        )
        if getattr(args, "manifest_output", None):
            import json as _json
            from pathlib import Path as _Path
            _Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
            _Path(args.manifest_output).write_text(_json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        if getattr(args, "print_manifest", False):
            _print_json(result)
        else:
            _print_json(result)
        return 0
    if args.command == "detect":
        detector_config = _build_detector_config(args.manifest, args)
        _print_json(run_detection_only(args.manifest, dry_run=args.dry_run, detector_config=detector_config))
        return 0
    if args.command == "advanced-vision":
        _print_json(build_advanced_vision_evidence(args.session_dir, output_path=args.output))
        return 0
    if args.command == "yolo-observation-inputs":
        _print_json(
            build_yolo_observation_inputs(
                args.session_dir,
                output_path=args.output,
                min_points=args.min_points,
                min_motion_px=args.min_motion_px,
                max_points_per_track=args.max_points_per_track,
            )
        )
        return 0
    if args.command == "lab-model-signal-inputs":
        _print_json(build_lab_model_signal_inputs(args.session_dir, min_confidence=args.min_confidence, output_summary_path=args.output_summary))
        return 0
    if args.command == "model-observations":
        _print_json(build_model_observation_events(args.session_dir, output_path=args.output))
        return 0
    if args.command == "understand-video":
        _print_json(build_video_understanding(args.session_dir, output_path=args.output))
        return 0
    if args.command == "context":
        _print_json(build_experiment_context(args.session_dir, output_path=args.output, database_paths=args.database))
        return 0
    if args.command == "context-text":
        _print_json(
            import_lightweight_context(
                args.session_dir,
                sop_text=_combined_cli_text(args.sop_text, args.sop_file),
                note_text=_combined_cli_text(args.note_text, args.note_file),
                record_text=_combined_cli_text(args.record_text, args.record_file),
                output_summary_path=args.output_summary,
            )
        )
        return 0
    if args.command == "process":
        _print_json(build_experiment_process(args.session_dir, sop_path=args.sop, output_path=args.output, timeline_output_path=args.timeline_output))
        return 0
    if args.command == "history-model":
        _print_json(build_history_model(args.source, output_path=args.output))
        return 0
    if args.command == "model-inventory":
        _print_json(discover_lab_assets(project_root=args.project_root, output_path=args.output))
        return 0
    if args.command == "stage-scope":
        overrides = {"scope_name": args.scope_name} if args.scope_name else None
        _print_json(build_stage_scope(args.session_dir, output_path=args.output, overrides=overrides))
        return 0
    if args.command == "confirmation-queue":
        _print_json(build_confirmation_queue(args.session_dir, output_path=args.output))
        return 0
    if args.command == "confirmation-batch":
        _print_json(
            apply_confirmation_batch_decisions(
                args.session_dir,
                args.decisions,
                output_path=args.output,
                reviewer=args.reviewer,
                note=args.note,
            )
        )
        return 0
    if args.command == "context-seed":
        _print_json(seed_session_context(args.session_dir, force=args.force, output_summary_path=args.output_summary))
        return 0
    if args.command == "refresh-derived":
        _print_json(refresh_derived_artifacts(args.session_dir, query_texts=args.query, output_summary_path=args.output_summary))
        return 0
    if args.command == "micro-coverage":
        _print_json(backfill_micro_coverage(args.session_dir, output_report=args.output_summary))
        return 0
    if args.command == "batch-refresh":
        _print_json(batch_refresh_sessions(args.source, query_texts=args.query, output_summary_path=args.output_summary, stop_on_error=args.stop_on_error))
        return 0
    if args.command == "audit-sessions":
        _print_json(build_session_audit_report(args.source, query_texts=args.query, output_json=args.output_json, output_md=args.output_md))
        return 0
    if args.command == "promotion-audit":
        _print_json(
            build_promotion_readiness_report(
                args.source,
                query_count=args.query_count,
                output_json=args.output_json,
                output_md=args.output_md,
            )
        )
        return 0
    if args.command == "health":
        report = build_run_health_report(
            args.session_dir,
            query_texts=args.query,
            output_json=args.output_json,
            output_md=args.output_md,
            max_total_coverage_ratio=args.max_total_coverage_ratio,
            max_longest_segment_ratio=args.max_longest_segment_ratio,
            min_boundary_confidence=args.min_boundary_confidence,
            max_human_queue_count=args.max_human_queue_count,
        )
        _print_json(report)
        if args.fail_on == "warning" and report.get("status") in {"warn", "fail"}:
            return 1
        if args.fail_on == "error" and report.get("gate_status") == "fail":
            return 1
        return 0
    if args.command == "review-bundle":
        _print_json(export_review_bundle(args.session_dir, args.output, format=args.format))
        return 0
    if args.command == "recovery-review-packet":
        _print_json(
            build_recovery_review_packet(
                args.plan,
                args.output_md,
                output_decisions=args.output_decisions,
                max_candidates=args.max_candidates,
            )
        )
        return 0
    if args.command == "freeze-reviewed-dataset":
        _print_json(freeze_reviewed_dataset(args.session_dir))
        return 0
    if args.command == "rollback-reviewed-release":
        _print_json(rollback_reviewed_release(args.session_dir, version=args.version))
        return 0
    if args.command == "promote-reviewed-release":
        _print_json(
            promote_reviewed_release(
                args.session_dir,
                version=args.version,
                reviewer=args.reviewer,
                note=args.note,
                query_count=args.query_count,
            )
        )
        return 0
    if args.command == "quality-gate":
        result = build_quality_gate(args.session_dir, output_path=args.output)
        _print_json(result)
        return 1 if args.strict and not result.get("can_mark_complete") else 0
    if args.command == "validate-evidence-adapters":
        result = validate_evidence_adapters(args.session_dir, output_path=args.output)
        _print_json(result)
        return 1 if args.strict and result.get("status") == "fail" else 0
    if args.command == "apply-review":
        _print_json(
            apply_review_to_process(
                args.session_dir,
                args.confirmation_id,
                args.decision,
                args.operator,
                rationale=args.rationale,
            )
        )
        return 0
    if args.command == "missing-step-recovery":
        _print_json(
            build_missing_step_recovery_plan(
                args.session_dir,
                output_path=args.output,
                confidence_threshold=args.confidence_threshold,
                window_padding_sec=args.window_padding_sec,
            )
        )
        return 0
    if args.command == "export-artifacts":
        summary = export_artifact_bundle(args.session_dir, args.output_dir, validate=True)
        _print_json(summary)
        return 1 if args.strict and not summary.get("valid", True) else 0
    if args.command == "boss-report":
        _print_json(generate_boss_acceptance_report(args.session_dir, output_path=args.output))
        return 0
    if args.command == "assets":
        state_summary = build_state_change_index(args.session_dir)
        summary = build_material_asset_catalog(args.session_dir, output_path=args.output, summary_path=args.summary)
        summary["state_change_index"] = state_summary.get("state_change_index")
        _print_json(summary)
        return 0
    if args.command == "material-candidates":
        _print_json(
            build_yolo_material_candidates(
                args.session_dir,
                dry_run=args.dry_run,
                ffmpeg_path=args.ffmpeg_path,
                archive_existing=args.archive_existing,
                rebuild_source=args.rebuild_source,
                enable_vlm=args.enable_vlm,
                max_vlm_groups=args.max_vlm_groups,
                vlm_model_name=args.vlm_model_name,
            )
        )
        return 0
    if args.command == "search-assets":
        _print_json(
            search_material_assets(
                args.session_dir,
                query=args.query,
                asset_type=args.asset_type,
                objects=args.objects,
                actions=args.actions,
                state_tags=args.state_tags,
                start_time=args.start_time,
                end_time=args.end_time,
                limit=args.limit,
            )
        )
        return 0
    if args.command == "material-reference-index":
        summary = build_key_material_reference_index(
            args.material_root,
            sqlite_path=args.sqlite_output,
            references_path=args.references_output,
            include_reports=args.include_reports,
        )
        if args.query or args.asset_type or args.object or args.action:
            summary["query_results"] = query_key_material_reference_index(
                summary["sqlite_path"],
                text=args.query,
                asset_type=args.asset_type,
                primary_object=args.object,
                action=args.action,
                limit=args.limit,
            )
        _print_json(summary)
        return 0
    if args.command == "material-library-sync":
        _print_json(
            sync_material_library(
                args.library_root,
                sqlite_path=args.sqlite_output,
                rebuild=args.rebuild,
                include_reports=args.include_reports,
            )
        )
        return 0
    if args.command == "material-library-query":
        rows = query_material_library(
            library_root=args.library_root,
            sqlite_path=args.sqlite_path,
            text=args.query,
            asset_type=args.asset_type,
            primary_object=args.object,
            action=args.action,
            view=args.view,
            session_id=args.session_id,
            experiment_id=args.experiment_id,
            package_name=args.package_name,
            date=args.date,
            start_date=args.start_date,
            end_date=args.end_date,
            limit=args.limit,
        )
        _print_json(
            {
                "total": len(rows),
                "results": rows if args.full else [_compact_material_library_cli_row(row) for row in rows],
            }
        )
        return 0
    if args.command == "video-memory-build":
        build_kwargs = {
            "library_root": args.library_root,
            "sqlite_path": args.sqlite_path,
            "window_end_date": args.window_end_date,
            "window_days": args.window_days,
            "job_type": args.job_type,
            "force_material_sync": args.force_material_sync,
            "vlm_mode": args.vlm_mode,
            "item_vlm_model": args.item_vlm_model,
            "bundle_vlm_model": args.bundle_vlm_model,
            "max_real_vlm_items": args.max_real_vlm_items,
            "max_real_vlm_bundles": args.max_real_vlm_bundles,
        }
        if args.background:
            _print_json(start_video_memory_rebuild_background(**build_kwargs))
        else:
            _print_json(build_video_memory(**build_kwargs))
        return 0
    if args.command == "video-memory-rebuild-status":
        _print_json(get_video_memory_rebuild_background_status(args.job_id))
        return 0
    if args.command == "video-memory-snapshot":
        snapshot = get_memory_snapshot(
            library_root=args.library_root,
            sqlite_path=args.sqlite_path,
            snapshot_id=args.snapshot_id,
        )
        _print_json(snapshot or {"status": "missing", "message": "No 30-Day Video Memory snapshot found"})
        return 0
    if args.command == "video-memory-query":
        _print_json(
            query_video_memory(
                args.query,
                library_root=args.library_root,
                sqlite_path=args.sqlite_path,
                snapshot_id=args.snapshot_id,
                limit=args.limit,
            )
        )
        return 0
    if args.command == "video-memory-feedback":
        context_fields = json.loads(args.context_json or "{}")
        for key, value in {
            "sop_name": args.sop_name,
            "sample_name": args.sample_name,
            "project_name": args.project_name,
        }.items():
            if value:
                context_fields[key] = value
        _print_json(
            record_human_feedback(
                target_type=args.target_type,
                target_id=args.target_id,
                feedback_type=args.feedback_type,
                context_fields=context_fields,
                note=args.note,
                user_id=args.user_id,
                library_root=args.library_root,
                sqlite_path=args.sqlite_path,
            )
        )
        return 0
    if args.command == "evidence-package-build":
        _print_json(
            build_evidence_package(
                args.package_root,
                source_manifest=args.source_manifest,
                key_action_index_dir=args.key_action_index_dir,
                package_id=args.package_id,
                experiment_id=args.experiment_id,
                sqlite_path=args.sqlite_output,
                references_path=args.references_output,
                include_reports=args.include_reports,
            )
        )
        return 0
    if args.command == "evidence-package-validate":
        _print_json(validate_evidence_package(args.package_root, strict=args.strict))
        return 0
    if args.command == "evidence-package-query":
        _print_json(
            query_evidence_package(
                args.package_root,
                query_text=args.query,
                message_sent_at=args.message_time,
                limit=args.limit,
                window_before_sec=args.window_before_sec,
                window_after_sec=args.window_after_sec,
            )
        )
        return 0
    if args.command == "evidence-package-eval":
        _print_json(
            evaluate_evidence_package_queries(
                args.package_root,
                args.queries,
                output_path=args.output,
                limit=args.limit,
            )
        )
        return 0
    if args.command == "sop-compliance":
        _print_json(
            build_sop_compliance_report(
                args.sop,
                key_actions=args.key_actions,
                evidence_refs=args.evidence,
                physical_changes=args.physical_change_log,
                package_dir=args.package_root,
                min_confidence=args.min_confidence,
                include_unmapped_observations=args.include_unmapped,
                output_path=args.output,
                events_output_path=args.events_output,
            )
        )
        return 0
    if args.command == "frontend-sync":
        if args.validate_only:
            result = validate_frontend_artifact_sync(
                args.target_experiment_dir,
                require_yolo_overlay=args.require_yolo_overlay,
                min_focus_duration_sec=args.min_focus_duration_sec,
            )
            if args.output_summary:
                Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output_summary).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            result = sync_frontend_artifacts(
                target_experiment_dir=args.target_experiment_dir,
                source_session_dir=args.source_session_dir,
                source_key_action_index_dir=args.source_key_action_index_dir,
                source_material_root=args.source_material_root,
                experiment_id=args.experiment_id,
                experiment_title=args.experiment_title,
                third_person_video=args.third_person_video,
                first_person_video=args.first_person_video,
                archive_existing=args.archive_existing,
                hardlink_media=args.hardlink_media,
                approve_materials=args.approve_materials,
                refresh_focus=args.refresh_focus,
                force_refresh_focus=args.force_refresh_focus,
                run_yolo_overlay=args.run_yolo_overlay,
                require_yolo_overlay=args.require_yolo_overlay,
                min_focus_duration_sec=args.min_focus_duration_sec,
                yolo_model_path=args.yolo_model_path,
                yolo_first_person_model_path=args.yolo_first_person_model_path,
                yolo_third_person_model_path=args.yolo_third_person_model_path,
                yolo_project_root=args.yolo_project_root,
                yolo_device=args.yolo_device,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_detect_fps=args.yolo_detect_fps,
                dry_run=args.dry_run,
                output_summary_path=args.output_summary,
            )
        _print_json(result)
        return 1 if args.fail_on_error and result.get("status") == "failed" else 0
    if args.command == "tune":
        _print_json(
            tune_detector(
                args.manifest,
                args.ground_truth,
                start_thresholds=_float_list(args.start_threshold),
                end_thresholds=_float_list(args.end_threshold),
                merge_gap_secs=_float_list(args.merge_gap_sec),
                min_segment_duration_secs=_float_list(args.min_segment_duration_sec),
                iou_threshold=args.iou_threshold,
                dry_run=args.dry_run,
                run_detection=args.run_detection,
                output_path=args.output,
            )
        )
        return 0
    if args.command == "micro-gt-template":
        _print_json(
            build_micro_gt_template_manifest(
                args.session_dir,
                micro_segments_path=args.micro_segments,
                key_action_segments_path=args.key_action_segments,
                output_dir=args.output_dir,
                gt_completeness=args.gt_completeness,
            )
        )
        return 0
    if args.command == "micro-quality":
        _print_json(enrich_micro_quality(args.session_dir, output_report=args.output_report, target_objects=args.objects))
        return 0
    if args.command == "timeline":
        module = importlib.import_module("key_action_indexer.unified_timeline")
        _print_json(
            module.generate_unified_timeline(
                args.manifest,
                args.output,
                user_events_path=args.user_events,
                ai_events_path=args.ai_events,
                uploads_path=args.uploads,
                calibration_path=args.calibration,
                dry_run=args.dry_run,
            )
        )
        return 0
    if args.command == "transcript-convert":
        _print_json(
            convert_transcript_to_jsonl(
                args.input,
                args.output,
                duration_sec=args.duration_sec,
                summary_output_path=args.summary_output,
            )
        )
        return 0
    if args.command == "query":
        if args.config:
            if not args.session_dir:
                parser.error("query --config requires --session-dir")
            _print_json(validate_queries(args.session_dir, args.config, output_path=args.output))
            return 0
        if not args.query:
            parser.error("query requires at least one --query unless --config is supplied")
        filters: dict[str, Any] = {}
        if args.objects:
            filters["objects"] = args.objects
        if args.actions:
            filters["actions"] = args.actions
        if args.asset_types:
            filters["asset_type"] = args.asset_types
        if args.start_time:
            filters["start_time"] = args.start_time
        if args.end_time:
            filters["end_time"] = args.end_time
        fusion_weights = json.loads(args.fusion_weights) if getattr(args, "fusion_weights", None) else None
        if args.session_dir:
            _print_json(
                query_session_index(
                    args.session_dir,
                    args.query,
                    top_k=args.top_k,
                    filters=filters,
                    fusion_weights=fusion_weights,
                    output_path=args.output,
                )
            )
            return 0
        if args.index_dir:
            _print_json(
                query_index(
                    args.index_dir,
                    args.query,
                    top_k=args.top_k,
                    filters=filters,
                    fusion_weights=fusion_weights,
                    output_path=args.output,
                )
            )
            return 0
        parser.error("query requires --session-dir or --index-dir")
        return 0
    if args.command == "default-query-eval":
        if args.config_output:
            build_default_chinese_query_eval_config(args.session_dir, output_path=args.config_output, query_count=args.query_count)
        _print_json(
            run_default_chinese_query_eval(
                args.session_dir,
                config_path=args.config_output,
                output_path=args.output,
                query_count=args.query_count,
            )
        )
        return 0
    if args.command == "gold-query-benchmark":
        _print_json(
            build_gold_query_benchmark(
                args.session_dir,
                output_path=args.output,
                query_count=args.query_count,
                overwrite=args.overwrite,
            )
        )
        return 0
    if args.command == "confirm-gold-query-benchmark":
        _print_json(
            confirm_gold_query_benchmark(
                args.session_dir,
                decisions_path=args.decisions,
                output_path=args.output,
                query_count=args.query_count,
                reviewer=args.reviewer,
                note=args.note,
            )
        )
        return 0
    if args.command == "report":
        print(str(generate_report(args.session_dir)))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
