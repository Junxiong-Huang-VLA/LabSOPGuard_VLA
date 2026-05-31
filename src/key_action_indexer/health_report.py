from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .query_validation import query_session_index


REQUIRED_ARTIFACTS = {
    "detected_segments": Path("cv_outputs/detected_segments.jsonl"),
    "key_action_segments": Path("metadata/key_action_segments.jsonl"),
    "micro_segments": Path("metadata/micro_segments.jsonl"),
    "vector_metadata": Path("metadata/vector_metadata.jsonl"),
    "index": Path("index/fallback_index.pkl"),
}

CONTEXT_INPUT_FILES = {
    "session_context_events": Path("metadata/session_context_events.jsonl"),
    "user_text_events": Path("metadata/user_text_events.jsonl"),
    "ai_reply_events": Path("metadata/ai_reply_events.jsonl"),
    "upload_events": Path("metadata/upload_events.jsonl"),
    "database_records": Path("metadata/database_records.jsonl"),
    "sop_records": Path("metadata/sop_records.jsonl"),
}

PATH_FIELD_NAMES = {
    "path",
    "clip",
    "clip_path",
    "keyframe",
    "keyframe_path",
    "peak_keyframe",
    "contact_frame",
    "peak_frame",
    "release_frame",
    "first_person_clip",
    "third_person_clip",
    "video_path",
}

OPTIONAL_PATH_MARKERS = {
    "model_path",
    "model_paths_by_view",
    "class_schema_path",
}

FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".pkl",
    ".sqlite",
    ".yaml",
    ".yml",
    ".txt",
}


def build_run_health_report(
    session_dir: str | Path,
    *,
    query_texts: list[str] | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    max_total_coverage_ratio: float = 0.65,
    max_longest_segment_ratio: float = 0.5,
    min_boundary_confidence: float = 0.01,
    max_human_queue_count: int = 100,
    max_path_checks: int = 4000,
) -> dict[str, Any]:
    session = Path(session_dir)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []

    if not session.exists():
        errors.append(_issue("missing_session", "Session directory does not exist.", path=str(session)))
        report = _report(session, errors, warnings, notes, metrics={})
        _write_outputs(report, output_json=output_json, output_md=output_md)
        return report

    artifact_metrics = _artifact_metrics(session, errors)
    detected_segments = _read_jsonl(session / "cv_outputs" / "detected_segments.jsonl", errors=errors)
    key_segments = _read_jsonl(session / "metadata" / "key_action_segments.jsonl", errors=errors)
    micro_segments = _read_jsonl(session / "metadata" / "micro_segments.jsonl", errors=errors)
    vector_metadata = _read_jsonl(session / "metadata" / "vector_metadata.jsonl", errors=errors)
    micro_vector_metadata = _read_jsonl(session / "metadata" / "micro_vector_metadata.jsonl", errors=errors)
    human_queue = _read_jsonl(session / "metadata" / "human_confirmation_queue.jsonl", errors=errors)

    video_duration = _video_duration(session / "video_info.json")
    segment_metrics = _segment_metrics(
        detected_segments,
        video_duration_sec=video_duration,
        min_boundary_confidence=min_boundary_confidence,
    )
    _add_segment_issues(
        segment_metrics,
        errors=errors,
        warnings=warnings,
        max_total_coverage_ratio=max_total_coverage_ratio,
        max_longest_segment_ratio=max_longest_segment_ratio,
    )

    if micro_segments and len(micro_vector_metadata) == 0:
        errors.append(_issue("missing_micro_vector_metadata", "Micro segments exist but micro vector metadata is empty."))
    if len(vector_metadata) == 0:
        errors.append(_issue("missing_vector_metadata", "Segment vector metadata is empty."))
    if len(human_queue) > max_human_queue_count:
        warnings.append(
            _issue(
                "large_confirmation_queue",
                "Human confirmation queue is above the no-label smoke threshold.",
                count=len(human_queue),
                threshold=max_human_queue_count,
            )
        )

    context_metrics = _context_metrics(session)
    if context_metrics["known_input_file_count"] and context_metrics["total_context_rows"] == 0:
        warnings.append(_issue("thin_multimodal_context", "All known session/text/SOP/database context inputs are empty."))

    video_understanding_metrics = _video_understanding_metrics(session)
    if video_understanding_metrics.get("candidate_ratio", 0.0) > 0.6:
        warnings.append(
            _issue(
                "candidate_heavy_video_understanding",
                "Video understanding is dominated by candidate events.",
                candidate_ratio=video_understanding_metrics["candidate_ratio"],
            )
        )

    evaluation_metrics = _evaluation_metrics(session)
    if evaluation_metrics.get("overall_score") is not None and evaluation_metrics["overall_score"] < 0.75:
        warnings.append(
            _issue(
                "low_pipeline_evaluation_score",
                "Pipeline evaluation is below the no-label regression threshold.",
                overall_score=evaluation_metrics["overall_score"],
            )
        )
    if evaluation_metrics.get("evidence_chain") is not None and evaluation_metrics["evidence_chain"] < 0.6:
        notes.append(
            _issue(
                "weak_evidence_chain_score",
                "Evidence-chain score is still limited without richer context or GT.",
                evidence_chain=evaluation_metrics["evidence_chain"],
            )
        )

    query_validation_metrics = _query_validation_metrics(session)
    if query_validation_metrics.get("failed_artifact_count", 0):
        errors.append(
            _issue(
                "query_validation_failed",
                "One or more fixed query validation artifacts failed acceptance thresholds.",
                failed_artifact_count=query_validation_metrics["failed_artifact_count"],
                failed_artifacts=query_validation_metrics["failed_artifacts"],
            )
        )
    if query_validation_metrics.get("bootstrap_failed_artifact_count", 0):
        warnings.append(
            _issue(
                "bootstrap_query_validation_failed",
                "Bootstrap query validation missed thresholds; manual gold verification is still pending.",
                failed_artifact_count=query_validation_metrics["bootstrap_failed_artifact_count"],
                failed_artifacts=query_validation_metrics["bootstrap_failed_artifacts"],
            )
        )

    path_metrics = _path_metrics(
        session,
        rows=[*detected_segments, *key_segments, *micro_segments, *vector_metadata, *micro_vector_metadata],
        max_path_checks=max_path_checks,
    )
    if path_metrics["missing_path_count"]:
        errors.append(
            _issue(
                "missing_asset_paths",
                "One or more referenced clip/keyframe assets are missing.",
                missing_path_count=path_metrics["missing_path_count"],
                examples=path_metrics["missing_path_examples"],
            )
        )

    query_metrics = _query_metrics(session, query_texts or [], errors=errors)
    for item in query_metrics.get("queries", []):
        if item.get("result_count", 0) == 0:
            errors.append(
                _issue(
                    "query_smoke_no_results",
                    "A configured smoke query returned no results.",
                    query=item.get("query"),
                )
            )

    metrics = {
        **artifact_metrics,
        **segment_metrics,
        "key_action_segment_count": len(key_segments),
        "micro_segment_count": len(micro_segments),
        "vector_metadata_count": len(vector_metadata),
        "micro_vector_metadata_count": len(micro_vector_metadata),
        "human_confirmation_queue_count": len(human_queue),
        "context": context_metrics,
        "video_understanding": video_understanding_metrics,
        "evaluation": evaluation_metrics,
        "query_validation": query_validation_metrics,
        "paths": path_metrics,
        "query_smoke": query_metrics,
    }
    report = _report(session, errors, warnings, notes, metrics=metrics)
    _write_outputs(report, output_json=output_json, output_md=output_md)
    return report


def _report(
    session: Path,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    *,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    status = "fail" if errors else "warn" if warnings else "pass"
    return {
        "schema_version": "key_action_run_health.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_dir": str(session),
        "status": status,
        "gate_status": "fail" if errors else "pass",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "note_count": len(notes),
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
        "metrics": metrics,
    }


def _artifact_metrics(session: Path, errors: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, rel_path in REQUIRED_ARTIFACTS.items():
        path = session / rel_path
        exists = path.exists()
        artifacts[name] = {
            "path": str(path),
            "exists": exists,
            "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
        }
        if not exists:
            errors.append(_issue("missing_required_artifact", "Required key-action artifact is missing.", artifact=name, path=str(path)))
    return {"artifacts": artifacts}


def _segment_metrics(
    segments: list[dict[str, Any]],
    *,
    video_duration_sec: float | None,
    min_boundary_confidence: float,
) -> dict[str, Any]:
    durations = [_float(row.get("duration_sec")) for row in segments]
    durations = [value for value in durations if value is not None and value >= 0]
    total_duration = round(sum(durations), 6)
    longest = max(durations) if durations else 0.0
    confidence_values = []
    for row in segments:
        confidence = _float(row.get("boundary_confidence"))
        if confidence is None:
            confidence = _float((row.get("cv_detection") or {}).get("confidence")) if isinstance(row.get("cv_detection"), dict) else None
        confidence_values.append(confidence or 0.0)
    zero_confidence_count = sum(1 for value in confidence_values if value <= min_boundary_confidence)
    refined_count = sum(1 for row in segments if str(row.get("boundary_source") or "").startswith("yolo_physical_evidence"))
    support_counts = [_float(row.get("boundary_support_count")) or 0.0 for row in segments]
    coverage = (total_duration / video_duration_sec) if video_duration_sec else None
    longest_ratio = (longest / video_duration_sec) if video_duration_sec else None
    return {
        "video_duration_sec": video_duration_sec,
        "segment_count": len(segments),
        "total_action_duration_sec": total_duration,
        "total_action_coverage_ratio": round(coverage, 6) if coverage is not None else None,
        "longest_segment_sec": round(longest, 6),
        "longest_segment_ratio": round(longest_ratio, 6) if longest_ratio is not None else None,
        "boundary_zero_confidence_count": zero_confidence_count,
        "boundary_refined_segment_count": refined_count,
        "boundary_support_total": round(sum(support_counts), 6),
        "boundary_confidence_min": round(min(confidence_values), 6) if confidence_values else None,
        "boundary_confidence_avg": round(sum(confidence_values) / len(confidence_values), 6) if confidence_values else None,
    }


def _add_segment_issues(
    metrics: dict[str, Any],
    *,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    max_total_coverage_ratio: float,
    max_longest_segment_ratio: float,
) -> None:
    if metrics["segment_count"] == 0:
        errors.append(_issue("no_detected_segments", "No detected segments were produced."))
    if metrics["boundary_zero_confidence_count"]:
        errors.append(
            _issue(
                "zero_boundary_confidence",
                "One or more segment boundaries have zero or near-zero confidence.",
                count=metrics["boundary_zero_confidence_count"],
            )
        )
    if metrics["segment_count"] and metrics["boundary_refined_segment_count"] == 0:
        warnings.append(_issue("unrefined_boundaries", "Segments do not advertise YOLO physical boundary refinement."))
    coverage = metrics.get("total_action_coverage_ratio")
    if coverage is not None and coverage > max_total_coverage_ratio:
        warnings.append(
            _issue(
                "high_total_action_coverage",
                "Detected action duration covers too much of the video for a key-action index.",
                coverage=coverage,
                threshold=max_total_coverage_ratio,
            )
        )
    longest = metrics.get("longest_segment_ratio")
    if longest is not None and longest > max_longest_segment_ratio:
        warnings.append(
            _issue(
                "coarse_longest_segment",
                "The longest segment is still too large relative to the source video.",
                longest_segment_ratio=longest,
                threshold=max_longest_segment_ratio,
            )
        )


def _context_metrics(session: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for name, rel_path in CONTEXT_INPUT_FILES.items():
        path = session / rel_path
        if path.exists():
            counts[name] = _count_jsonl(path)
    return {
        "known_input_file_count": len(counts),
        "counts": counts,
        "total_context_rows": sum(counts.values()),
    }


def _video_understanding_metrics(session: Path) -> dict[str, Any]:
    summary = _read_json(session / "metadata" / "video_understanding_summary.json")
    if not summary:
        return {"available": False}
    conclusion_counts = summary.get("conclusion_status_counts") if isinstance(summary.get("conclusion_status_counts"), dict) else {}
    total = sum(int(value or 0) for value in conclusion_counts.values())
    candidate = int(conclusion_counts.get("candidate") or 0)
    return {
        "available": True,
        "video_event_count": summary.get("video_event_count"),
        "conclusion_status_counts": conclusion_counts,
        "candidate_ratio": round(candidate / total, 6) if total else 0.0,
        "human_review_candidate_count": summary.get("human_review_candidate_count"),
    }


def _evaluation_metrics(session: Path) -> dict[str, Any]:
    data = _read_json(session / "evaluation" / "pipeline_evaluation_report.json")
    if not data:
        return {"available": False}
    scores = data.get("scores") if isinstance(data.get("scores"), dict) else {}
    return {
        "available": True,
        "overall_score": _float(data.get("overall_score")),
        "segments": _float(scores.get("segments")),
        "actions_and_states": _float(scores.get("actions_and_states")),
        "step_reasoning": _float(scores.get("step_reasoning")),
        "evidence_chain": _float(scores.get("evidence_chain")),
        "json_schema": _float(scores.get("json_schema")),
        "metric_mode": (data.get("segment_keyframe_eval") or {}).get("segment_metrics", {}).get("metric_mode")
        if isinstance(data.get("segment_keyframe_eval"), dict)
        else None,
    }


def _query_validation_metrics(session: Path) -> dict[str, Any]:
    evaluation = session / "evaluation"
    if not evaluation.exists():
        return {"available": False, "artifact_count": 0, "artifacts": []}
    artifacts = []
    for path in sorted(evaluation.glob("*query_validation*.json")):
        if ".candidate." in path.name:
            continue
        data = _read_json(path)
        if not data:
            continue
        status = str(data.get("status") or "unknown")
        threshold_failures = data.get("threshold_failures") if isinstance(data.get("threshold_failures"), list) else []
        failed_query_count = int(data.get("failed_query_count") or 0)
        query_count = _int_or_default(data.get("query_count"), 0)
        artifact = {
            "path": str(path),
            "name": path.name,
            "status": status,
            "benchmark_binding_mode": data.get("benchmark_binding_mode"),
            "human_verified_query_count": _int_or_default(data.get("human_verified_query_count"), 0),
            "human_reviewed_query_count": _int_or_default(data.get("human_reviewed_query_count"), _int_or_default(data.get("human_verified_query_count"), 0)),
            "query_count": query_count,
            "total_query_count": _int_or_default(data.get("total_query_count"), query_count),
            "applicable_query_count": _int_or_default(data.get("applicable_query_count"), query_count),
            "excluded_query_count": _int_or_default(data.get("excluded_query_count"), 0),
            "acceptance_hit_rate": _float(data.get("acceptance_hit_rate")),
            "query_hit_rate": _float(data.get("query_hit_rate")),
            "top1_hit_rate": _float(data.get("top1_hit_rate")),
            "traceability_hit_rate": _float(data.get("traceability_hit_rate")),
            "quality_hit_rate": _float(data.get("quality_hit_rate")),
            "failed_query_count": failed_query_count,
            "threshold_failure_count": len(threshold_failures),
        }
        artifacts.append(artifact)
    failed_all = [item for item in artifacts if item.get("status") == "fail" or item.get("threshold_failure_count", 0) > 0]
    bootstrap_failed = [
        item
        for item in failed_all
        if str(item.get("benchmark_binding_mode") or "") == "bootstrap_pending_human_verification"
    ]
    failed = [item for item in failed_all if item not in bootstrap_failed]
    return {
        "available": bool(artifacts),
        "artifact_count": len(artifacts),
        "failed_artifact_count": len(failed),
        "failed_artifacts": [item["name"] for item in failed],
        "bootstrap_failed_artifact_count": len(bootstrap_failed),
        "bootstrap_failed_artifacts": [item["name"] for item in bootstrap_failed],
        "artifacts": artifacts,
    }


def _path_metrics(session: Path, *, rows: list[dict[str, Any]], max_path_checks: int) -> dict[str, Any]:
    refs = []
    seen: set[str] = set()
    for row in rows:
        for key_path, value in _iter_path_values(row):
            cleaned = _clean_path_value(value)
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            refs.append({"field": key_path, "path": cleaned})
            if len(refs) >= max_path_checks:
                break
        if len(refs) >= max_path_checks:
            break
    missing = []
    absolute_count = 0
    for ref in refs:
        path_value = ref["path"]
        if _is_absolute_path_string(path_value):
            absolute_count += 1
        resolved = _resolve_file_path(session, path_value)
        if resolved is not None and not resolved.exists():
            missing.append({**ref, "resolved_path": str(resolved)})
    return {
        "checked_path_count": len(refs),
        "absolute_path_reference_count": absolute_count,
        "missing_path_count": len(missing),
        "missing_path_examples": missing[:10],
        "path_check_truncated": len(refs) >= max_path_checks,
    }


def _query_metrics(session: Path, query_texts: list[str], *, errors: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [str(item).strip() for item in query_texts if str(item).strip()]
    if not queries:
        return {"configured": False, "queries": []}
    try:
        payload = query_session_index(session, queries, top_k=3)
    except Exception as exc:
        errors.append(_issue("query_smoke_failed", "Smoke query execution failed.", error=str(exc)))
        return {"configured": True, "queries": [], "error": str(exc)}
    rows = []
    for item in payload.get("queries", []):
        rows.append(
            {
                "query": item.get("query"),
                "top_k": item.get("top_k"),
                "result_count": item.get("result_count"),
                "top_result_id": _top_result_id(item.get("results") or []),
            }
        )
    return {
        "configured": True,
        "index_dir": payload.get("index_dir"),
        "queries": rows,
    }


def _top_result_id(results: list[dict[str, Any]]) -> str | None:
    if not results:
        return None
    first = results[0]
    return first.get("micro_segment_id") or first.get("segment_id")


def _video_duration(path: Path) -> float | None:
    data = _read_json(path)
    sources = data.get("video_sources") if isinstance(data.get("video_sources"), dict) else {}
    durations = []
    for source in sources.values():
        if isinstance(source, dict):
            value = _float(source.get("duration_sec"))
            if value is not None:
                durations.append(value)
    return max(durations) if durations else None


def _iter_path_values(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            key_name = str(key)
            key_path = f"{prefix}.{key_name}" if prefix else key_name
            lower_key = key_name.lower()
            if lower_key in OPTIONAL_PATH_MARKERS:
                continue
            if isinstance(item, str) and _is_path_field(lower_key) and _looks_like_file_path(item):
                yield key_path, item
            else:
                yield from _iter_path_values(item, key_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            key_path = f"{prefix}[{index}]"
            if isinstance(item, str) and _looks_like_file_path(item):
                yield key_path, item
            else:
                yield from _iter_path_values(item, key_path)


def _is_path_field(key: str) -> bool:
    return (
        key in PATH_FIELD_NAMES
        or key.endswith("_path")
        or key.endswith("_paths")
        or key.endswith("_clip")
        or key.endswith("_frame")
        or key.endswith("_keyframe")
    ) and "url" not in key


def _looks_like_file_path(value: str) -> bool:
    if "\n" in value or "\r" in value:
        return False
    if value.startswith(("http://", "https://", "/api/")):
        return False
    cleaned = _clean_path_value(value)
    if not cleaned:
        return False
    suffix = Path(cleaned).suffix.lower()
    return suffix in FILE_EXTENSIONS and ("/" in cleaned or "\\" in cleaned or _is_absolute_path_string(cleaned))


def _clean_path_value(value: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        return ""
    return cleaned.split("?", 1)[0]


def _resolve_file_path(session: Path, value: str) -> Path | None:
    if not value:
        return None
    if _is_absolute_path_string(value):
        return Path(value)
    if value.startswith(("http://", "https://", "/api/")):
        return None
    return session / value


def _is_absolute_path_string(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", value)) or value.startswith(("/", "\\\\"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _read_jsonl(path: Path, *, errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(
                _issue(
                    "malformed_jsonl",
                    "A JSONL artifact contains a malformed row.",
                    path=str(path),
                    line=line_number,
                    error=str(exc),
                )
            )
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _write_outputs(report: dict[str, Any], *, output_json: str | Path | None, output_md: str | Path | None) -> None:
    if output_json:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_md:
        target = Path(output_md)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_run_health_markdown(report), encoding="utf-8")


def render_run_health_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    lines = [
        "# Key Action Run Health",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Gate: `{report.get('gate_status')}`",
        f"- Session: `{report.get('session_dir')}`",
        f"- Generated: `{report.get('generated_at')}`",
        "",
        "## Core Metrics",
        "",
        f"- Segments: `{metrics.get('segment_count', 0)}`",
        f"- Micro segments: `{metrics.get('micro_segment_count', 0)}`",
        f"- Vector metadata: `{metrics.get('vector_metadata_count', 0)}`",
        f"- Total action duration: `{metrics.get('total_action_duration_sec')}` sec",
        f"- Total coverage ratio: `{metrics.get('total_action_coverage_ratio')}`",
        f"- Longest segment ratio: `{metrics.get('longest_segment_ratio')}`",
        f"- Boundary zero confidence: `{metrics.get('boundary_zero_confidence_count', 0)}`",
        f"- Human queue: `{metrics.get('human_confirmation_queue_count', 0)}`",
        "",
    ]
    lines.extend(_markdown_issue_section("Errors", report.get("errors") or []))
    lines.extend(_markdown_issue_section("Warnings", report.get("warnings") or []))
    lines.extend(_markdown_issue_section("Notes", report.get("notes") or []))
    query = metrics.get("query_smoke") if isinstance(metrics.get("query_smoke"), dict) else {}
    if query.get("configured"):
        lines.extend(["## Query Smoke", ""])
        for item in query.get("queries", []):
            lines.append(f"- `{item.get('query')}` -> `{item.get('result_count')}` results, top `{item.get('top_result_id')}`")
        lines.append("")
    query_validation = metrics.get("query_validation") if isinstance(metrics.get("query_validation"), dict) else {}
    if query_validation.get("available"):
        lines.extend(["## Query Validation", ""])
        for item in query_validation.get("artifacts", []):
            lines.append(
                "- "
                f"`{item.get('name')}` status `{item.get('status')}`, "
                f"acceptance `{item.get('acceptance_hit_rate')}`, "
                f"query `{item.get('query_hit_rate')}`, "
                f"quality `{item.get('quality_hit_rate')}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_issue_section(title: str, issues: list[dict[str, Any]]) -> list[str]:
    lines = [f"## {title}", ""]
    if not issues:
        lines.extend(["- None", ""])
        return lines
    for issue in issues:
        detail = {key: value for key, value in issue.items() if key not in {"code", "message"}}
        suffix = f" `{json.dumps(detail, ensure_ascii=False, default=str)}`" if detail else ""
        lines.append(f"- `{issue.get('code')}`: {issue.get('message')}{suffix}")
    lines.append("")
    return lines
