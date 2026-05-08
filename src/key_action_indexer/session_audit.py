from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .query_validation import query_session_index
from .schemas import read_jsonl


AUDIT_SCHEMA_VERSION = "key_action_session_audit.v1"


def build_session_audit_report(
    sources: Iterable[str | Path],
    *,
    query_texts: list[str] | None = None,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
) -> dict[str, Any]:
    sessions = [_resolve_session_dir(source) for source in sources]
    rows = [_audit_session(session, query_texts=query_texts or []) for session in sessions]
    summary = _summary(rows)
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_count": len(rows),
        "summary": summary,
        "sessions": rows,
    }
    if output_json:
        target = Path(output_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_md:
        target = Path(output_md)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_session_audit_markdown(report), encoding="utf-8")
    return report


def render_session_audit_markdown(report: Mapping[str, Any]) -> str:
    summary = _as_dict(report.get("summary"))
    lines = [
        "# P4 Session Audit Summary",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Sessions audited: `{report.get('session_count')}`",
        f"- Health pass: `{summary.get('health_pass_count')}/{report.get('session_count')}`",
        f"- QA pass: `{summary.get('qa_pass_count')}/{report.get('session_count')}`",
        f"- Average candidate ratio: `{summary.get('avg_candidate_ratio')}`",
        f"- Total strong micro evidence: `{summary.get('strong_process_micro_count')}`",
        f"- Total retrieval-only micro evidence: `{summary.get('retrieval_candidate_micro_count')}`",
        "",
        "## Session Metrics",
        "",
        "| Session | Segments | Micro | Strong | Retrieval-only | Video events | Candidate ratio | Rollup | QA | Health | Queue | Risks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in report.get("sessions") or []:
        if not isinstance(row, Mapping):
            continue
        metrics = _as_dict(row.get("metrics"))
        micro = _as_dict(metrics.get("micro_evidence"))
        video = _as_dict(metrics.get("video_understanding"))
        rollup = _as_dict(video.get("candidate_rollup"))
        qa = _as_dict(metrics.get("quality"))
        health = _as_dict(metrics.get("health"))
        confirmation = _as_dict(metrics.get("confirmation"))
        risk_count = len(row.get("risks") or [])
        lines.append(
            "| "
            f"`{row.get('session_id')}` | "
            f"{metrics.get('key_action_segment_count', 0)} | "
            f"{metrics.get('micro_segment_count', 0)} | "
            f"{micro.get('strong_process_evidence', 0)} | "
            f"{micro.get('retrieval_candidate', 0)} | "
            f"{video.get('video_event_count', 0)} | "
            f"{video.get('candidate_ratio', 0.0)} | "
            f"{rollup.get('removed_candidate_event_count', 0)} | "
            f"`{qa.get('overall_status', 'missing')}` | "
            f"`{health.get('gate_status', 'missing')}` | "
            f"{confirmation.get('pending_count', 0)} | "
            f"{risk_count} |"
        )
    lines.extend(["", "## Risks", ""])
    for row in report.get("sessions") or []:
        if not isinstance(row, Mapping):
            continue
        risks = row.get("risks") or []
        lines.append(f"### {row.get('session_id')}")
        if not risks:
            lines.append("- None")
        else:
            for risk in risks:
                lines.append(f"- `{risk.get('code')}`: {risk.get('message')}")
        lines.append("")
    lines.extend(
        [
            "",
            "## Retrieval Acceptance",
            "",
            "| Session | Query | Results | Top Result | Traceable | Dual-view Clips | Keyframes |",
            "|---|---|---:|---|---|---|---:|",
        ]
    )
    for row in report.get("sessions") or []:
        if not isinstance(row, Mapping):
            continue
        query = _as_dict(_as_dict(row.get("metrics")).get("query_smoke"))
        if not query.get("queries"):
            continue
        for item in query.get("queries") or []:
            lines.append(
                "| "
                f"`{row.get('session_id')}` | "
                f"`{item.get('query')}` | "
                f"{item.get('result_count')} | "
                f"`{item.get('top_result_id')}` | "
                f"`{item.get('top_traceable')}` | "
                f"`{item.get('top_dual_view_clip_traceable')}` | "
                f"{item.get('top_keyframe_count', 0)} |"
            )
    lines.extend(
        [
            "",
            "## Backfill Evidence Guard",
            "",
            "| Session | Retrieval-only | Strong | Segment backfill | Promoted segment backfill |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report.get("sessions") or []:
        if not isinstance(row, Mapping):
            continue
        metrics = _as_dict(row.get("metrics"))
        micro = _as_dict(metrics.get("micro_evidence"))
        lines.append(
            "| "
            f"`{row.get('session_id')}` | "
            f"{micro.get('retrieval_candidate', 0)} | "
            f"{micro.get('strong_process_evidence', 0)} | "
            f"{micro.get('segment_level_backfill_count', 0)} | "
            f"{micro.get('segment_level_backfill_promoted_count', 0)} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _audit_session(session: Path, *, query_texts: list[str]) -> dict[str, Any]:
    metadata = session / "metadata"
    key_segments = _read_jsonl(metadata / "key_action_segments.jsonl")
    micro_segments = _read_jsonl(metadata / "micro_segments.jsonl")
    vector_metadata = _read_jsonl(metadata / "vector_metadata.jsonl")
    micro_vector_metadata = _read_jsonl(metadata / "micro_vector_metadata.jsonl")
    video_summary = _read_json(metadata / "video_understanding_summary.json")
    quality = _read_json(metadata / "process_quality_report.json")
    process = _read_json(metadata / "experiment_process.json")
    history = _read_json(metadata / "history_model.json")
    health = _read_json(session / "reports" / "run_health_report.json")
    confirmation_queue = _read_jsonl(metadata / "human_confirmation_queue.jsonl")
    machine_backlog = _read_jsonl(metadata / "human_confirmation_machine_backlog.jsonl")
    micro_roles = _micro_role_counts(micro_segments)
    video_metrics = _video_metrics(video_summary)
    metrics = {
        "key_action_segment_count": len(key_segments),
        "micro_segment_count": len(micro_segments),
        "vector_metadata_count": len(vector_metadata),
        "micro_vector_metadata_count": len(micro_vector_metadata),
        "micro_evidence": micro_roles,
        "video_understanding": video_metrics,
        "quality": {
            "overall_status": quality.get("overall_status"),
            "overall_score": quality.get("overall_score"),
            "status_counts": quality.get("status_counts") or {},
        },
        "health": {
            "status": health.get("status"),
            "gate_status": health.get("gate_status"),
            "error_count": health.get("error_count"),
            "warning_count": health.get("warning_count"),
        },
        "confirmation": {
            "pending_count": sum(1 for row in confirmation_queue if str(row.get("status") or "pending") == "pending"),
            "item_count": len(confirmation_queue),
            "machine_backlog_count": len(machine_backlog),
        },
        "process": {
            "process_status": process.get("process_status"),
            "step_count": process.get("step_count"),
            "status_counts": process.get("status_counts") or {},
            "pending_confirmation_step_ids": process.get("pending_confirmation_step_ids") or [],
        },
        "history": {
            "session_count": history.get("session_count"),
            "event_count": history.get("event_count"),
            "source_session_ids": history.get("source_session_ids") or [],
        },
        "query_smoke": _query_smoke(session, query_texts),
    }
    row = {
        "session_dir": str(session),
        "session_id": _session_id(session, video_summary, process, history),
        "metrics": metrics,
    }
    row["risks"] = _risks(row)
    return row


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_ratios = []
    strong_count = 0
    retrieval_count = 0
    health_pass = 0
    qa_pass = 0
    for row in rows:
        metrics = _as_dict(row.get("metrics"))
        micro = _as_dict(metrics.get("micro_evidence"))
        video = _as_dict(metrics.get("video_understanding"))
        health = _as_dict(metrics.get("health"))
        quality = _as_dict(metrics.get("quality"))
        if video.get("candidate_ratio") is not None:
            candidate_ratios.append(float(video.get("candidate_ratio") or 0.0))
        strong_count += int(micro.get("strong_process_evidence") or 0)
        retrieval_count += int(micro.get("retrieval_candidate") or 0)
        health_pass += int(str(health.get("gate_status") or "") == "pass")
        qa_pass += int(str(quality.get("overall_status") or "") == "pass")
    return {
        "health_pass_count": health_pass,
        "qa_pass_count": qa_pass,
        "avg_candidate_ratio": round(sum(candidate_ratios) / len(candidate_ratios), 6) if candidate_ratios else None,
        "max_candidate_ratio": round(max(candidate_ratios), 6) if candidate_ratios else None,
        "strong_process_micro_count": strong_count,
        "retrieval_candidate_micro_count": retrieval_count,
        "risk_count": sum(len(row.get("risks") or []) for row in rows),
    }


def _micro_role_counts(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    grades: Counter[str] = Counter()
    strong = 0
    retrieval = 0
    segment_level_backfill = 0
    promoted_segment_level_backfill = 0
    for row in rows:
        evidence = _as_dict(row.get("evidence"))
        role = str(evidence.get("process_evidence_role") or "unknown")
        roles[role] += 1
        if role == "strong_process_evidence" or evidence.get("process_eligible"):
            strong += 1
        if role == "retrieval_candidate" or evidence.get("retrieval_candidate_only"):
            retrieval += 1
        is_segment_backfill = _is_segment_level_backfill(evidence)
        if is_segment_backfill:
            segment_level_backfill += 1
            if role == "strong_process_evidence" or evidence.get("process_eligible"):
                promoted_segment_level_backfill += 1
        if evidence.get("retrieval_priority_bucket"):
            buckets[str(evidence["retrieval_priority_bucket"])] += 1
        if evidence.get("coverage_signal_grade"):
            grades[str(evidence["coverage_signal_grade"])] += 1
    return {
        "role_counts": dict(sorted(roles.items())),
        "retrieval_priority_bucket_counts": dict(sorted(buckets.items())),
        "coverage_signal_grade_counts": dict(sorted(grades.items())),
        "strong_process_evidence": strong,
        "retrieval_candidate": retrieval,
        "segment_level_backfill_count": segment_level_backfill,
        "segment_level_backfill_promoted_count": promoted_segment_level_backfill,
    }


def _video_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    status_counts = _as_dict(summary.get("conclusion_status_counts"))
    total = sum(int(value or 0) for value in status_counts.values())
    candidate = int(status_counts.get("candidate") or 0)
    return {
        "video_event_count": summary.get("video_event_count"),
        "event_type_counts": summary.get("event_type_counts") or {},
        "conclusion_status_counts": status_counts,
        "human_review_candidate_count": summary.get("human_review_candidate_count"),
        "candidate_ratio": round(candidate / total, 6) if total else None,
        "candidate_rollup": summary.get("candidate_rollup") or {},
    }


def _query_smoke(session: Path, query_texts: list[str]) -> dict[str, Any]:
    queries = [query for query in query_texts if str(query).strip()]
    if not queries:
        return {"configured": False, "queries": []}
    try:
        payload = query_session_index(session, queries, top_k=3)
    except Exception as exc:
        return {"configured": True, "error": str(exc), "queries": []}
    rows = []
    for item in payload.get("queries") or []:
        results = item.get("results") if isinstance(item.get("results"), list) else []
        top = results[0] if results else {}
        keyframes = top.get("keyframes")
        if isinstance(keyframes, Mapping):
            keyframe_count = sum(1 for value in keyframes.values() if value)
        elif isinstance(keyframes, list):
            keyframe_count = len(keyframes)
        else:
            keyframe_count = 0
        rows.append(
            {
                "query": item.get("query"),
                "result_count": item.get("result_count"),
                "top_result_id": top.get("micro_segment_id") or top.get("segment_id"),
                "top_traceable": _traceable(top),
                "top_has_first_person_clip": bool(top.get("first_person_clip")),
                "top_has_third_person_clip": bool(top.get("third_person_clip")),
                "top_dual_view_clip_traceable": bool(top.get("first_person_clip") and top.get("third_person_clip")),
                "top_keyframe_count": keyframe_count,
            }
        )
    return {"configured": True, "queries": rows}


def _risks(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = _as_dict(row.get("metrics"))
    micro = _as_dict(metrics.get("micro_evidence"))
    video = _as_dict(metrics.get("video_understanding"))
    quality = _as_dict(metrics.get("quality"))
    health = _as_dict(metrics.get("health"))
    history = _as_dict(metrics.get("history"))
    query = _as_dict(metrics.get("query_smoke"))
    risks = []
    if str(health.get("gate_status") or "") != "pass":
        risks.append({"code": "health_not_pass", "message": "Run health gate is not passing."})
    if str(quality.get("overall_status") or "") != "pass":
        risks.append({"code": "qa_not_pass", "message": "Quality assurance is not passing."})
    ratio = video.get("candidate_ratio")
    if ratio is not None and float(ratio) > 0.30:
        risks.append({"code": "candidate_ratio_above_target", "message": f"Candidate ratio {ratio} is above the 0.30 target."})
    if int(micro.get("strong_process_evidence") or 0) == 0:
        risks.append({"code": "no_strong_micro_evidence", "message": "No micro-segments are eligible as strong process evidence."})
    if int(history.get("session_count") or 0) < 6:
        risks.append({"code": "history_under_sampled", "message": "History model has fewer than 6 sessions."})
    if int(micro.get("segment_level_backfill_promoted_count") or 0) > 0:
        risks.append(
            {
                "code": "segment_backfill_promoted_to_strong",
                "message": "One or more segment-level retrieval backfills were promoted to strong process evidence.",
            }
        )
    for item in query.get("queries") or []:
        if int(item.get("result_count") or 0) == 0:
            risks.append({"code": "query_no_results", "message": f"Query returned no results: {item.get('query')}"})
        elif not item.get("top_traceable"):
            risks.append({"code": "query_top_not_traceable", "message": f"Top query result lacks clip/keyframe traceability: {item.get('query')}"})
    if query.get("error"):
        risks.append({"code": "query_smoke_error", "message": str(query.get("error"))})
    return risks


def _traceable(result: Mapping[str, Any]) -> bool:
    if not result:
        return False
    has_anchor = bool(result.get("micro_segment_id") or result.get("segment_id"))
    has_media = bool(result.get("third_person_clip") or result.get("first_person_clip") or result.get("keyframes") or result.get("asset_bindings"))
    return has_anchor and has_media


def _is_segment_level_backfill(evidence: Mapping[str, Any]) -> bool:
    if str(evidence.get("retrieval_priority_bucket") or "") == "segment_level_backfill":
        return True
    warnings = evidence.get("warnings")
    if isinstance(warnings, list) and "segment_level_retrieval_backfill" in {str(item) for item in warnings}:
        return True
    return bool(evidence.get("segment_level_retrieval_backfill"))


def _resolve_session_dir(source: str | Path) -> Path:
    path = Path(source)
    if (path / "metadata").exists():
        return path
    if (path / "key_action_index" / "metadata").exists():
        return path / "key_action_index"
    return path


def _session_id(session: Path, *payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        value = payload.get("session_id")
        if value:
            return str(value)
    manifest = session / "manifest.json"
    if manifest.exists():
        data = _read_json(manifest)
        if data.get("session_id"):
            return str(data["session_id"])
    return session.name


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


__all__ = ["AUDIT_SCHEMA_VERSION", "build_session_audit_report", "render_session_audit_markdown"]
