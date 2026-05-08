from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifact_schema import validate_session_artifacts
from .confirmation_loop import build_confirmation_queue
from .context_fusion import build_experiment_context
from .evaluation import build_pipeline_evaluation_report
from .health_report import build_run_health_report
from .process_reasoner import build_experiment_process
from .process_record import build_process_record
from .quality_assurance import build_quality_assurance_report
from .session_context_seed import seed_session_context
from .unified_timeline import generate_unified_timeline


def refresh_derived_artifacts(
    session_dir: str | Path,
    *,
    query_texts: list[str] | None = None,
    output_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    reports = session / "reports"
    evaluation = session / "evaluation"
    manifest_path = session / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest for derived refresh: {manifest_path}")

    metadata.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)

    context_seed = seed_session_context(session)
    timeline = generate_unified_timeline(manifest_path=manifest_path, output_dir=metadata)
    experiment_context = build_experiment_context(session)
    experiment_process = build_experiment_process(session)
    confirmation = build_confirmation_queue(session)
    process_path = metadata / "experiment_process.json"
    if process_path.exists():
        experiment_process = json.loads(process_path.read_text(encoding="utf-8"))
    quality = build_quality_assurance_report(session)
    process_record = build_process_record(session)
    pipeline_evaluation = build_pipeline_evaluation_report(session)
    artifact_validation = validate_session_artifacts(
        session,
        artifact_types=[
            "video_understanding",
            "experiment_context",
            "experiment_process",
            "process_record",
            "asset_catalog",
            "confirmation_queue",
            "process_quality_report",
            "pipeline_evaluation_report",
        ],
        output_path=metadata / "artifact_validation_report.json",
    )
    health = build_run_health_report(
        session,
        query_texts=query_texts or ["balance weighing"],
        output_json=reports / "run_health_report.json",
        output_md=reports / "run_health_report.md",
    )

    summary = {
        "schema_version": "key_action_derived_refresh.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "steps": {
            "context_seed": _compact_context_seed(context_seed),
            "timeline": {
                "event_count": timeline.get("event_count"),
                "time_anchor_count": timeline.get("time_anchor_count"),
                "sources": sorted((timeline.get("sources") or {}).keys()),
            },
            "experiment_context": {
                "confidence": experiment_context.get("confidence"),
                "source_counts": experiment_context.get("source_counts"),
                "gaps": experiment_context.get("gaps"),
            },
            "experiment_process": {
                "process_status": experiment_process.get("process_status"),
                "step_count": experiment_process.get("step_count"),
                "pending_confirmation_step_ids": experiment_process.get("pending_confirmation_step_ids"),
                "status_counts": experiment_process.get("status_counts"),
            },
            "confirmation_queue": confirmation,
            "artifact_validation": {
                "valid": artifact_validation.get("valid"),
                "issue_count": artifact_validation.get("issue_count"),
                "artifact_count": artifact_validation.get("artifact_count"),
            },
            "quality": {
                "overall_status": quality.get("overall_status"),
                "overall_score": quality.get("overall_score"),
                "status_counts": quality.get("status_counts"),
            },
            "process_record": process_record.get("summary", {}),
            "pipeline_evaluation": {
                "overall_score": pipeline_evaluation.get("overall_score"),
                "scores": pipeline_evaluation.get("scores"),
            },
            "health": {
                "status": health.get("status"),
                "gate_status": health.get("gate_status"),
                "warning_count": health.get("warning_count"),
                "note_count": health.get("note_count"),
            },
        },
        "paths": {
            "summary": str(Path(output_summary_path) if output_summary_path else reports / "derived_refresh_summary.json"),
            "health_json": str(reports / "run_health_report.json"),
            "health_md": str(reports / "run_health_report.md"),
            "quality": str(metadata / "process_quality_report.json"),
            "pipeline_evaluation": str(evaluation / "pipeline_evaluation_report.json"),
            "confirmation_queue": str(metadata / "human_confirmation_queue.jsonl"),
            "machine_backlog": str(metadata / "human_confirmation_machine_backlog.jsonl"),
        },
    }
    target = Path(output_summary_path) if output_summary_path else reports / "derived_refresh_summary.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _compact_context_seed(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_count": summary.get("row_count"),
        "written": summary.get("written"),
        "skipped": summary.get("skipped"),
        "artifact": summary.get("artifact"),
        "non_label_context": summary.get("non_label_context"),
    }


__all__ = ["refresh_derived_artifacts"]
