"""CLI entrypoint for key_action_indexer."""

from __future__ import annotations

import argparse
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
from .evaluation_manifest import build_micro_gt_template_manifest
from .export_interfaces import export_artifact_bundle
from .history_learning import build_history_model
from .health_report import build_run_health_report
from .lab_model_signal_inputs import build_lab_model_signal_inputs
from .lightweight_context_import import import_lightweight_context
from .material_search import search_material_assets
from .missing_step_recovery import build_missing_step_recovery_plan
from .micro_coverage_backfill import backfill_micro_coverage
from .micro_quality_enrichment import enrich_micro_quality
from .model_inventory import discover_lab_assets
from .model_observations import build_model_observation_events
from .pipeline import run_detection_only, run_pipeline
from .process_reasoner import build_experiment_process
from .quality_gate import build_quality_gate
from .query_validation import query_index, query_session_index, validate_queries
from .report import generate_report
from .retrieval_eval import build_default_chinese_query_eval_config, build_gold_query_benchmark, run_default_chinese_query_eval
from .review_bundle import apply_review_to_process, export_review_bundle
from .review_packet import build_recovery_review_packet
from .reviewed_dataset import freeze_reviewed_dataset, rollback_reviewed_release
from .schemas import SessionManifest
from .session_context_seed import seed_session_context
from .session_audit import build_session_audit_report
from .scope_config import build_stage_scope
from .state_index import build_state_change_index
from .transcript_convert import convert_transcript_to_jsonl
from .tuning import parse_float_list, tune_detector
from .video_understanding import build_video_understanding
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="key-action-indexer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run full key action indexing pipeline")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    detect_parser = subparsers.add_parser("detect", help="Run key action detection only")
    detect_parser.add_argument("--manifest", required=True)
    detect_parser.add_argument("--dry-run", action="store_true")

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
        _print_json(run_pipeline(args.manifest, dry_run=args.dry_run))
        return 0
    if args.command == "detect":
        _print_json(run_detection_only(args.manifest, dry_run=args.dry_run))
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
        if args.session_dir:
            _print_json(
                query_session_index(
                    args.session_dir,
                    args.query,
                    top_k=args.top_k,
                    filters=filters,
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
    if args.command == "report":
        print(str(generate_report(args.session_dir)))
        return 0

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
