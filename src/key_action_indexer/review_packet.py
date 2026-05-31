from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MAX_CANDIDATES = 5


def build_recovery_review_packet(
    recovery_plan_path: str | Path,
    output_md: str | Path,
    *,
    output_decisions: str | Path | None = None,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Build a human-readable packet from a missing-step recovery plan.

    The packet is intentionally evidence-forward: it keeps approval out of the
    automated path and gives the reviewer direct clips/keyframes, confidence,
    time windows, and a decision template that can later feed confirmation-batch.
    """

    plan_path = Path(recovery_plan_path)
    plan = _read_json(plan_path)
    if not isinstance(plan, Mapping):
        raise ValueError(f"recovery plan is not a JSON object: {plan_path}")
    steps = [step for step in plan.get("steps") or [] if isinstance(step, Mapping)]
    session_id = str(plan.get("session_id") or "unknown_session")
    session_dir = _infer_session_dir(plan)
    generated_at = _now()
    output = Path(output_md)
    output.parent.mkdir(parents=True, exist_ok=True)
    decision_rows = [_decision_template_row(session_id, step) for step in steps]
    lines = _packet_markdown(
        plan=plan,
        plan_path=plan_path,
        session_id=session_id,
        session_dir=session_dir,
        generated_at=generated_at,
        steps=steps,
        max_candidates=max(1, max_candidates),
    )
    output.write_text("\n".join(lines), encoding="utf-8")

    decision_path = Path(output_decisions) if output_decisions is not None else None
    if decision_path is not None:
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_payload = {
            "schema_version": "human_confirmation_decision_template.v1",
            "session_id": session_id,
            "session_dir": session_dir,
            "source_recovery_plan": str(plan_path),
            "source_review_packet": str(output),
            "generated_at": generated_at,
            "instructions": [
                "Inspect candidate clips and keyframes before editing decisions.",
                "Set decision to approve, reject, or needs_more_review.",
                "Run confirmation-batch only after a human reviewer has filled reviewer and note fields.",
            ],
            "decision_allowed_values": ["approve", "reject", "needs_more_review"],
            "decision_normalization": {
                "approve": "approved",
                "reject": "rejected",
                "needs_more_review": "needs_review",
            },
            "decisions": decision_rows,
        }
        decision_path.write_text(json.dumps(decision_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "schema_version": "recovery_review_packet.v1",
        "session_id": session_id,
        "step_count": len(steps),
        "output_md": str(output),
        "output_decisions": str(decision_path) if decision_path is not None else None,
        "source_recovery_plan": str(plan_path),
        "candidate_summary": _candidate_summary(steps),
    }


def _packet_markdown(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    session_id: str,
    session_dir: str | None,
    generated_at: str,
    steps: list[Mapping[str, Any]],
    max_candidates: int,
) -> list[str]:
    summary = _candidate_summary(steps)
    lines = [
        f"# Missing-Step Review Packet: {session_id}",
        "",
        f"- Generated: `{generated_at}`",
        f"- Recovery plan: `{plan_path}`",
        f"- Session dir: `{session_dir or ''}`",
        f"- Target steps: `{len(steps)}`",
        f"- Candidate totals: video `{summary['video_events']}`, transcript `{summary['transcript_utterances']}`, assets `{summary['assets']}`",
        "",
        "## Review Rules",
        "",
        "- Do not auto-approve this packet.",
        "- Approve only when the listed clip/keyframe evidence visually supports the SOP step.",
        "- Use transcript or text-only evidence as support, not as strong visual confirmation.",
        "- Keep segment-level retrieval backfill as retrieval-only unless a reviewer confirms real visual process evidence.",
        "",
        "## Step Summary",
        "",
        "| # | Step | Action | Status | Confidence | Window | Best Video | Best Asset | Suggested |",
        "|---|---|---|---|---:|---|---:|---:|---|",
    ]
    for index, step in enumerate(steps, start=1):
        suggestion = _as_mapping(step.get("human_confirmation_suggestion"))
        strength = _as_mapping(suggestion.get("candidate_strength"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _cell(str(step.get("name") or step.get("step_id") or "")),
                    _cell(str(step.get("expected_action") or "")),
                    _cell(str(step.get("status") or "")),
                    _fmt_float(step.get("confidence")),
                    _cell(_window_text(_as_mapping(step.get("recovery_window")))),
                    _fmt_float(strength.get("best_video_match_score")),
                    _fmt_float(strength.get("best_asset_match_score")),
                    _cell(str(suggestion.get("decision_hint") or "")),
                ]
            )
            + " |"
        )
    lines.append("")

    for index, step in enumerate(steps, start=1):
        evidence = _as_mapping(step.get("candidate_evidence"))
        counts = _as_mapping(evidence.get("counts"))
        suggestion = _as_mapping(step.get("human_confirmation_suggestion"))
        lines.extend(
            [
                f"## {index}. {_heading(str(step.get('name') or step.get('step_id') or 'Step'))}",
                "",
                f"- Step ID: `{step.get('step_id') or ''}`",
                f"- Expected action: `{step.get('expected_action') or ''}`",
                f"- Current status: `{step.get('status') or ''}`",
                f"- Confidence: `{_fmt_float(step.get('confidence'))}`",
                f"- Recovery reason: {_plain(step.get('recovery_reason'))}",
                f"- Window: `{_window_text(_as_mapping(step.get('recovery_window')))}`",
                f"- Candidate counts: video `{counts.get('video_events') or 0}`, transcript `{counts.get('transcript_utterances') or 0}`, assets `{counts.get('assets') or 0}`",
                f"- Reviewer hint: `{suggestion.get('decision_hint') or ''}`; {_plain(suggestion.get('rationale'))}",
                "",
                "### Candidate Video Events",
                "",
            ]
        )
        lines.extend(_video_table(evidence.get("video_events") or [], max_candidates=max_candidates))
        lines.extend(["", "### Candidate Assets", ""])
        lines.extend(_asset_table(evidence.get("assets") or [], max_candidates=max_candidates))
        transcript_rows = evidence.get("transcript_utterances") or []
        if transcript_rows:
            lines.extend(["", "### Transcript Support", ""])
            lines.extend(_transcript_table(transcript_rows, max_candidates=max_candidates))
        lines.extend(["", "### Decision Template", "", "```json"])
        lines.append(json.dumps(_decision_template_row(session_id, step), ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
    return lines


def _video_table(rows: Any, *, max_candidates: int) -> list[str]:
    candidates = [row for row in rows if isinstance(row, Mapping)][:max_candidates]
    if not candidates:
        return ["No video candidates found."]
    lines = [
        "| # | Score | Conf | Event | Object | Segment | Micro | Time | Key Clips/Frames |",
        "|---|---:|---:|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(candidates, start=1):
        paths = _top_asset_paths(row.get("asset_refs") or [], limit=4)
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _fmt_float(row.get("match_score")),
                    _fmt_float(row.get("confidence")),
                    _cell(str(row.get("event_type") or "")),
                    _cell(str(row.get("primary_object") or "")),
                    _cell(str(row.get("segment_id") or "")),
                    _cell(str(row.get("micro_segment_id") or "")),
                    _cell(_range_text(row)),
                    _cell("<br>".join(f"`{path}`" for path in paths)),
                ]
            )
            + " |"
        )
    return lines


def _asset_table(rows: Any, *, max_candidates: int) -> list[str]:
    candidates = [row for row in rows if isinstance(row, Mapping)][:max_candidates]
    if not candidates:
        return ["No asset candidates found."]
    lines = [
        "| # | Score | Type | Source | Segment | Micro | Time | Path |",
        "|---|---:|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(candidates, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _fmt_float(row.get("match_score")),
                    _cell(str(row.get("asset_type") or "")),
                    _cell(str(row.get("source_type") or "")),
                    _cell(str(row.get("segment_id") or "")),
                    _cell(str(row.get("micro_segment_id") or "")),
                    _cell(_range_text(row)),
                    _cell(f"`{row.get('path') or ''}`"),
                ]
            )
            + " |"
        )
    return lines


def _transcript_table(rows: Any, *, max_candidates: int) -> list[str]:
    candidates = [row for row in rows if isinstance(row, Mapping)][:max_candidates]
    lines = ["| # | Score | Time | Text |", "|---|---:|---|---|"]
    for index, row in enumerate(candidates, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    _fmt_float(row.get("match_score")),
                    _cell(_range_text(row)),
                    _cell(str(row.get("text") or row.get("transcript") or "")),
                ]
            )
            + " |"
        )
    return lines


def _decision_template_row(session_id: str, step: Mapping[str, Any]) -> dict[str, Any]:
    step_id = str(step.get("step_id") or "")
    suggestion = _as_mapping(step.get("human_confirmation_suggestion"))
    strength = _as_mapping(suggestion.get("candidate_strength"))
    return {
        "confirmation_id": f"{session_id}:{step_id}" if step_id else "",
        "step_id": step_id,
        "step_name": step.get("name"),
        "decision": "",
        "reviewer": "",
        "note": suggestion.get("note_template") or f"{step_id}: visual_match=; transcript_support=; decision=",
        "visual_match": "",
        "transcript_support": "",
        "chosen_evidence_ids": [],
        "best_video_match_score": strength.get("best_video_match_score"),
        "best_asset_match_score": strength.get("best_asset_match_score"),
        "transcript_candidate_count": strength.get("transcript_candidate_count"),
    }


def _candidate_summary(steps: list[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for step in steps:
        evidence = _as_mapping(step.get("candidate_evidence"))
        counts = _as_mapping(evidence.get("counts"))
        totals["video_events"] += int(counts.get("video_events") or 0)
        totals["transcript_utterances"] += int(counts.get("transcript_utterances") or 0)
        totals["assets"] += int(counts.get("assets") or 0)
    return dict(totals)


def _top_asset_paths(rows: Any, *, limit: int) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        path = str(row.get("path") or "")
        if not path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def _range_text(row: Mapping[str, Any]) -> str:
    start = row.get("global_start_time") or row.get("start_time") or row.get("global_time") or ""
    end = row.get("global_end_time") or row.get("end_time") or row.get("global_time") or ""
    if start and end and start != end:
        return f"{start} to {end}"
    return str(start or end or "")


def _window_text(window: Mapping[str, Any]) -> str:
    start = window.get("global_start_time") or ""
    end = window.get("global_end_time") or ""
    return f"{start} to {end}" if start or end else ""


def _infer_session_dir(plan: Mapping[str, Any]) -> str | None:
    source_paths = _as_mapping(plan.get("source_paths"))
    process = _as_mapping(source_paths.get("experiment_process"))
    path = process.get("path")
    if not path:
        return None
    process_path = Path(str(path))
    if process_path.name == "experiment_process.json" and process_path.parent.name == "metadata":
        return str(process_path.parent.parent)
    return str(process_path.parent)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _fmt_float(value: Any) -> str:
    try:
        if value in (None, ""):
            return ""
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value or "")


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>")


def _plain(value: Any) -> str:
    return str(value or "").replace("\n", " ")


def _heading(value: str) -> str:
    return value.replace("#", "").strip() or "Step"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["build_recovery_review_packet"]
