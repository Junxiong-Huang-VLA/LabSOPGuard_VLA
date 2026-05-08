"""Generate a compact acceptance report for key-action evidence sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .schemas import read_jsonl


def generate_boss_acceptance_report(
    session_dir: str | Path,
    output_path: str | Path | None = None,
) -> Dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    exports = session / "exports"
    out = Path(output_path) if output_path else session / "reports" / "boss_acceptance_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    process = _json(metadata / "experiment_process.json")
    quality = _json(metadata / "process_quality_report.json")
    context = _json(metadata / "experiment_context.json")
    gap = _json(metadata / "capability_gap_report.json")
    history = _json(metadata / "history_model.json")
    manifest = _json(session / "manifest.json")
    record = _json(exports / "process_record.json")
    segments = _jsonl(metadata / "key_action_segments.jsonl")
    micros = _jsonl(metadata / "micro_segments.jsonl")
    timeline = _jsonl(metadata / "unified_multimodal_timeline.jsonl")
    queue = _jsonl(metadata / "human_confirmation_queue.jsonl")

    steps = [step for step in process.get("steps", []) if isinstance(step, Mapping)]
    scorecard = quality.get("scorecard") or {}
    session_id = str(process.get("session_id") or manifest.get("session_id") or session.name)
    sections = [
        _section_1_meta(session_id, manifest, process, context, record),
        _section_2_timeline(timeline, steps),
        _section_3_segments(segments, micros),
        _section_4_step_table(steps),
        _section_5_observation_split(steps),
        _section_6_capability_gap(gap),
        _section_7_history(history, scorecard),
        _section_8_metrics(scorecard, quality),
        _section_9_action_items(scorecard, queue),
    ]
    out.write_text("\n\n".join(sections), encoding="utf-8")
    return {
        "output_path": str(out),
        "session_id": session_id,
        "section_count": len(sections),
        "step_count": len(steps),
        "overall_score": quality.get("overall_score"),
        "overall_status": quality.get("overall_status"),
    }


def _section_1_meta(session_id: str, manifest: Dict[str, Any], process: Dict[str, Any], context: Dict[str, Any], record: Dict[str, Any]) -> str:
    videos = manifest.get("videos") or {}
    third = videos.get("third_person") or {}
    first = videos.get("first_person") or {}
    config = manifest.get("config") or {}
    lines = [
        "# Experiment Acceptance Report",
        "",
        "## 1. Experiment Metadata",
        "",
        f"- Session ID: `{session_id}`",
        f"- Third-person video: `{third.get('path', 'N/A')}`",
        f"- First-person video: `{first.get('path', 'N/A')}`",
        f"- LabSOPGuard experiment IDs: `{', '.join(config.get('labsopguard_experiments', []))}`",
        f"- Session start: {manifest.get('session_start_time', 'N/A')}",
        f"- Process status: {process.get('process_status', 'N/A')}",
        f"- Step count: {process.get('step_count', 0)}",
        f"- Context purpose: {context.get('purpose', 'N/A')}",
    ]
    reviewers = record.get("reviewers") or []
    if reviewers:
        lines.append(f"- Reviewer count: {len(reviewers)}")
    return "\n".join(lines)


def _section_2_timeline(timeline: List[dict[str, Any]], steps: List[Mapping[str, Any]]) -> str:
    lines = ["## 2. Unified Timeline", "", f"- Timeline events: {len(timeline)}", ""]
    if steps:
        lines.append("Step status overview:")
        for step in steps:
            status = str(step.get("status") or "unknown")
            marker = {"completed": "ok", "observed": "seen", "inferred": "inferred", "not_observed": "missing"}.get(status, "review")
            lines.append(f"- {marker}: {step.get('name', step.get('step_id', ''))} [{status}]")
    return "\n".join(lines)


def _section_3_segments(segments: List[dict[str, Any]], micros: List[dict[str, Any]]) -> str:
    lines = ["## 3. Key Segments", "", f"- Segment count: {len(segments)}", f"- Micro-segment count: {len(micros)}"]
    for segment in segments[:10]:
        sid = segment.get("segment_id", "")
        duration = float(segment.get("duration_sec", 0.0) or 0.0)
        lines.append(f"- `{sid}` ({duration:.1f}s)")
    return "\n".join(lines)


def _section_4_step_table(steps: List[Mapping[str, Any]]) -> str:
    lines = [
        "## 4. Step State Table",
        "",
        "| step_id | name | status | confidence | confirmation |",
        "|---------|------|--------|------------|--------------|",
    ]
    for step in steps:
        lines.append(
            f"| {step.get('step_id', '')} | {step.get('name', '')} | {step.get('status', '')} | "
            f"{float(step.get('confidence', 0.0) or 0.0):.2f} | {step.get('confirmation_status', '')} |"
        )
    return "\n".join(lines)


def _section_5_observation_split(steps: List[Mapping[str, Any]]) -> str:
    observed = [step for step in steps if step.get("observed") and not step.get("inferred")]
    inferred = [step for step in steps if step.get("inferred")]
    pending = [step for step in steps if step.get("requires_human_confirmation")]
    return "\n".join(
        [
            "## 5. Observed vs Inferred vs Pending",
            "",
            "| category | count | steps |",
            "|----------|-------|-------|",
            f"| observed | {len(observed)} | {', '.join(str(s.get('step_id', '')) for s in observed)} |",
            f"| inferred | {len(inferred)} | {', '.join(str(s.get('step_id', '')) for s in inferred)} |",
            f"| pending human review | {len(pending)} | {', '.join(str(s.get('step_id', '')) for s in pending)} |",
        ]
    )


def _section_6_capability_gap(gap: Dict[str, Any]) -> str:
    remaining = gap.get("remaining_gaps") or []
    covered = gap.get("covered_via_labsopguard_outputs") or []
    lines = ["## 6. Capability Gaps", "", f"- Covered capabilities: {len(covered)}", f"- Remaining gaps: {len(remaining)}"]
    for item in remaining:
        if isinstance(item, Mapping):
            classes = ", ".join(str(value) for value in item.get("recommended_new_classes", []))
            lines.append(f"- {item.get('capability', '')}: annotate {classes}")
    return "\n".join(lines)


def _section_7_history(history: Dict[str, Any], scorecard: Dict[str, Any]) -> str:
    history_reuse = scorecard.get("history_reuse") or {}
    return "\n".join(
        [
            "## 7. History Reuse",
            "",
            f"- history_reuse score: {history_reuse.get('score', 'N/A')}",
            f"- history record count: {history.get('history_record_count', 0)}",
            f"- session count: {history.get('session_count', 0)}",
            f"- recommended SOP step count: {len(history.get('recommended_sop', []))}",
        ]
    )


def _section_8_metrics(scorecard: Dict[str, Any], quality: Dict[str, Any]) -> str:
    lines = [
        "## 8. Metric Scorecard",
        "",
        f"Overall: {quality.get('overall_status', '')} (score={quality.get('overall_score', '')})",
        "",
        "| metric | status | score |",
        "|--------|--------|-------|",
    ]
    for key, item in scorecard.items():
        if isinstance(item, Mapping):
            lines.append(f"| {key} | {item.get('status', '')} | {item.get('score', '')} |")
    return "\n".join(lines)


def _section_9_action_items(scorecard: Dict[str, Any], queue: List[dict[str, Any]]) -> str:
    pending = [row for row in queue if str(row.get("status") or "") == "pending"]
    needs_review = [
        (key, item)
        for key, item in scorecard.items()
        if isinstance(item, Mapping) and item.get("status") in {"needs_review", "fail"}
    ]
    lines = ["## 9. Open Items And Next Steps", "", f"- Pending review queue: {len(pending)}"]
    if needs_review:
        for key, item in needs_review:
            recommendations = item.get("recommendations") or ["manual review required"]
            lines.append(f"- {key} ({item.get('status')}, score={item.get('score')}): {'; '.join(map(str, recommendations[:3]))}")
    else:
        lines.append("- All metrics passed.")
    return "\n".join(lines)


def _json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}


def _jsonl(path: Path) -> List[dict[str, Any]]:
    if not path.exists():
        return []
    return [dict(row) for row in read_jsonl(path) if isinstance(row, Mapping)]


__all__ = ["generate_boss_acceptance_report"]
