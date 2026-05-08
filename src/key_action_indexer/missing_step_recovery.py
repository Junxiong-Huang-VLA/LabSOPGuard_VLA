from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from .schemas import read_jsonl
from .time_alignment import parse_time


PLAN_FILENAME = "missing_step_recovery_plan.json"
LOW_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_WINDOW_PADDING_SEC = 5.0
DEFAULT_OPEN_WINDOW_SEC = 20.0
RECOVERY_STATUSES = {"not_observed", "skipped_or_unobserved", "inferred_missing"}
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
STOP_TERMS = {
    "and",
    "but",
    "direct",
    "from",
    "missing",
    "not",
    "observed",
    "observation",
    "step",
    "the",
    "this",
    "video",
}
ACTION_ALIASES = {
    "recording": ["record", "readout", "display", "notebook", "measurement", "log", "balance"],
    "weighing": ["weigh", "balance", "mass", "scale", "measurement"],
    "pipetting": ["pipette", "transfer", "liquid", "tip", "sample"],
    "sample_adding_candidate": ["sample", "adding", "transfer", "liquid", "pipette"],
    "sample_handling": ["sample", "bottle", "tube", "container", "handling"],
}


def build_missing_step_recovery_plan(
    session_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    window_padding_sec: float = DEFAULT_WINDOW_PADDING_SEC,
    max_candidates: int = 8,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    process_path = metadata / "experiment_process.json"
    video_path = metadata / "video_understanding.jsonl"
    asset_path = metadata / "material_asset_catalog.jsonl"
    transcript_path = _first_existing(
        [
            session / "transcript" / "aligned_transcript.jsonl",
            metadata / "transcript" / "aligned_transcript.jsonl",
            metadata / "aligned_transcript.jsonl",
        ]
    )

    process = _read_json_if_exists(process_path)
    video_rows = _read_jsonl_if_exists(video_path)
    transcript_rows = _read_jsonl_if_exists(transcript_path) if transcript_path is not None else []
    asset_rows = _read_jsonl_if_exists(asset_path)
    session_id = str(process.get("session_id") or _session_id(video_rows, transcript_rows, asset_rows, session=session))
    steps = [step for step in process.get("steps", []) if isinstance(step, Mapping)]

    generated_at = datetime.now().astimezone().isoformat()
    step_plans: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        if not _needs_recovery(step, confidence_threshold):
            continue
        step_plans.append(
            _build_step_recovery(
                session=session,
                step=step,
                steps=steps,
                step_index=index,
                video_rows=video_rows,
                transcript_rows=transcript_rows,
                asset_rows=asset_rows,
                confidence_threshold=confidence_threshold,
                window_padding_sec=window_padding_sec,
                max_candidates=max_candidates,
            )
        )

    status_counts = Counter(str(plan.get("status") or "unknown") for plan in step_plans)
    result = {
        "schema_version": "missing_step_recovery_plan/v1",
        "session_id": session_id,
        "generated_at": generated_at,
        "confidence_threshold": confidence_threshold,
        "window_padding_sec": window_padding_sec,
        "target_step_count": len(step_plans),
        "target_status_counts": dict(sorted(status_counts.items())),
        "source_paths": {
            "experiment_process": _path_status(process_path),
            "video_understanding": _path_status(video_path),
            "aligned_transcript": _path_status(transcript_path),
            "material_asset_catalog": _path_status(asset_path),
        },
        "steps": step_plans,
        "summary": {
            "video_event_count": len(video_rows),
            "transcript_utterance_count": len(transcript_rows),
            "asset_count": len(asset_rows),
            "recovery_plan_path": str(Path(output_path) if output_path is not None else metadata / PLAN_FILENAME),
        },
    }
    target = Path(output_path) if output_path is not None else metadata / PLAN_FILENAME
    _write_json(target, result)
    return result


def _build_step_recovery(
    *,
    session: Path,
    step: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
    step_index: int,
    video_rows: list[Mapping[str, Any]],
    transcript_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
    confidence_threshold: float,
    window_padding_sec: float,
    max_candidates: int,
) -> dict[str, Any]:
    window = _recovery_window(step, steps, step_index, window_padding_sec)
    terms = _search_terms(step)
    video_candidates = _video_candidates(video_rows, step, terms, window, max_candidates)
    transcript_candidates = _transcript_candidates(transcript_rows, terms, window, max_candidates)
    asset_candidates = _asset_candidates(asset_rows, step, terms, window, max_candidates)
    candidate_counts = {
        "video_events": len(video_candidates),
        "transcript_utterances": len(transcript_candidates),
        "assets": len(asset_candidates),
    }
    return {
        "step_id": step.get("step_id"),
        "name": step.get("name"),
        "expected_action": step.get("expected_action"),
        "status": step.get("status"),
        "observed": step.get("observed"),
        "inferred": step.get("inferred"),
        "completed": step.get("completed"),
        "confidence": step.get("confidence"),
        "recovery_reason": _recovery_reason(step, confidence_threshold),
        "recovery_window": window,
        "candidate_evidence": {
            "counts": candidate_counts,
            "video_events": video_candidates,
            "transcript_utterances": transcript_candidates,
            "assets": asset_candidates,
        },
        "search_conditions": _search_conditions(session, step, terms, window),
        "human_confirmation_suggestion": _human_confirmation_suggestion(
            step,
            video_candidates=video_candidates,
            transcript_candidates=transcript_candidates,
            asset_candidates=asset_candidates,
        ),
    }


def _needs_recovery(step: Mapping[str, Any], threshold: float) -> bool:
    status = str(step.get("status") or "")
    confidence = _as_float(step.get("confidence")) or 0.0
    if status == "not_observed":
        return True
    if status in {"skipped_or_unobserved", "inferred_missing"} and confidence < threshold:
        return True
    if step.get("inferred") and confidence < threshold:
        return True
    return bool(step.get("requires_human_confirmation") and not step.get("observed") and confidence < threshold)


def _recovery_window(
    step: Mapping[str, Any],
    steps: list[Mapping[str, Any]],
    index: int,
    padding_sec: float,
) -> dict[str, Any]:
    own_start = _row_start(step)
    own_end = _row_end(step)
    previous_step, previous_time = _neighbor_time(steps[:index], reverse=True, end=True)
    next_step, next_time = _neighbor_time(steps[index + 1 :], reverse=False, end=False)

    source = "no_temporal_anchor"
    start = own_start or own_end
    end = own_end or own_start
    if start or end:
        source = "step_time"
    elif previous_time and next_time:
        start = previous_time
        end = next_time
        source = "between_neighbor_steps"
    elif previous_time:
        start = previous_time
        end = previous_time + timedelta(seconds=DEFAULT_OPEN_WINDOW_SEC)
        source = "after_previous_step"
    elif next_time:
        start = next_time - timedelta(seconds=DEFAULT_OPEN_WINDOW_SEC)
        end = next_time
        source = "before_next_step"

    if start and end and start > end:
        start, end = end, start
    padded_start = start - timedelta(seconds=padding_sec) if start else None
    padded_end = end + timedelta(seconds=padding_sec) if end else None
    return {
        "global_start_time": _iso(padded_start),
        "global_end_time": _iso(padded_end),
        "padding_sec": padding_sec,
        "source": source,
        "previous_step_id": previous_step.get("step_id") if previous_step else None,
        "next_step_id": next_step.get("step_id") if next_step else None,
    }


def _neighbor_time(
    candidates: list[Mapping[str, Any]],
    *,
    reverse: bool,
    end: bool,
) -> tuple[Mapping[str, Any] | None, datetime | None]:
    iterable = reversed(candidates) if reverse else candidates
    for row in iterable:
        timestamp = _row_end(row) if end else _row_start(row)
        timestamp = timestamp or _row_start(row) or _row_end(row)
        if timestamp is not None:
            return row, timestamp
    return None, None


def _video_candidates(
    rows: list[Mapping[str, Any]],
    step: Mapping[str, Any],
    terms: list[str],
    window: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        score, reasons = _candidate_score(row, step, terms, window)
        if score <= 0 and "in_recovery_window" not in reasons:
            continue
        candidates.append(
            {
                "source": "video_understanding",
                "video_event_id": row.get("video_event_id"),
                "event_type": row.get("event_type"),
                "action_type": row.get("action_type"),
                "primary_object": row.get("primary_object"),
                "confidence": row.get("confidence"),
                "global_start_time": row.get("global_start_time") or row.get("global_time"),
                "global_end_time": row.get("global_end_time") or row.get("global_time"),
                "segment_id": row.get("segment_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "asset_refs": row.get("asset_refs") or [],
                "text": _short_text(row.get("text")),
                "match_score": round(score, 4),
                "match_reasons": reasons,
            }
        )
    return _top_candidates(candidates, limit)


def _transcript_candidates(
    rows: list[Mapping[str, Any]],
    terms: list[str],
    window: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        text = _row_text(row)
        match_terms = [term for term in terms if term and term in text]
        in_window = _in_window(row, window)
        if not match_terms and not in_window:
            continue
        score = min(1.0, 0.15 * len(match_terms) + (0.35 if in_window else 0.0))
        candidates.append(
            {
                "source": "aligned_transcript",
                "utterance_id": row.get("utterance_id") or row.get("id"),
                "global_start_time": row.get("global_start_time") or row.get("global_time"),
                "global_end_time": row.get("global_end_time") or row.get("global_time"),
                "text": _short_text(row.get("text")),
                "match_score": round(score, 4),
                "match_reasons": _dedupe(["keyword_match" if match_terms else "", "in_recovery_window" if in_window else ""]),
                "matched_terms": match_terms[:8],
            }
        )
    return _top_candidates(candidates, limit)


def _asset_candidates(
    rows: list[Mapping[str, Any]],
    step: Mapping[str, Any],
    terms: list[str],
    window: Mapping[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        score, reasons = _candidate_score(row, step, terms, window)
        if score <= 0 and "in_recovery_window" not in reasons:
            continue
        candidates.append(
            {
                "source": "material_asset_catalog",
                "asset_id": row.get("asset_id"),
                "asset_type": row.get("asset_type"),
                "path": row.get("path"),
                "source_type": row.get("source_type"),
                "segment_id": row.get("segment_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "global_start_time": row.get("global_start_time") or row.get("global_time"),
                "global_end_time": row.get("global_end_time") or row.get("global_time"),
                "objects": row.get("objects") or [],
                "actions": row.get("actions") or [],
                "state_tags": row.get("state_tags") or [],
                "evidence_level": row.get("evidence_level"),
                "quality": row.get("quality") or {},
                "search_text": _short_text(row.get("search_text")),
                "match_score": round(score, 4),
                "match_reasons": reasons,
            }
        )
    return _top_candidates(candidates, limit)


def _candidate_score(
    row: Mapping[str, Any],
    step: Mapping[str, Any],
    terms: list[str],
    window: Mapping[str, Any],
) -> tuple[float, list[str]]:
    text = _row_text(row)
    expected = _normalize(step.get("expected_action"))
    score = 0.0
    reasons: list[str] = []
    if expected and expected in text:
        score += 0.35
        reasons.append("expected_action_match")
    matched_terms = [term for term in terms if term and term in text]
    if matched_terms:
        score += min(0.3, 0.06 * len(matched_terms))
        reasons.append("keyword_match")
    if _in_window(row, window):
        score += 0.25
        reasons.append("in_recovery_window")
    confidence = _as_float(row.get("confidence"))
    if confidence is not None:
        score += min(0.2, confidence * 0.2)
        if confidence >= 0.65:
            reasons.append("moderate_or_high_confidence")
    return min(1.0, score), _dedupe(reasons)


def _search_conditions(
    session: Path,
    step: Mapping[str, Any],
    terms: list[str],
    window: Mapping[str, Any],
) -> dict[str, Any]:
    expected = str(step.get("expected_action") or "")
    query_text = " ".join(_dedupe([expected, str(step.get("name") or ""), *terms[:6]]))
    command = f'python -m key_action_indexer.cli search-assets --session-dir "{session}" --query "{query_text}"'
    return {
        "query_text": query_text,
        "expected_action": expected,
        "action_aliases": ACTION_ALIASES.get(_normalize(expected), []),
        "time_window": {
            "global_start_time": window.get("global_start_time"),
            "global_end_time": window.get("global_end_time"),
        },
        "video_event_types": [
            "experiment_action_classification",
            "hand_object_contact",
            "object_state_change",
            "equipment_panel_operation_candidate",
            "liquid_transfer_candidate",
        ],
        "asset_filters": {
            "actions": _dedupe([expected, *ACTION_ALIASES.get(_normalize(expected), [])]),
            "state_tags": ["interaction_keyframe", "peak_interaction", "visual_confirmed"],
        },
        "transcript_keywords": terms[:12],
        "suggested_commands": [command],
    }


def _human_confirmation_suggestion(
    step: Mapping[str, Any],
    *,
    video_candidates: list[Mapping[str, Any]],
    transcript_candidates: list[Mapping[str, Any]],
    asset_candidates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    best_video = max((_as_float(row.get("match_score")) or 0.0 for row in video_candidates), default=0.0)
    best_asset = max((_as_float(row.get("match_score")) or 0.0 for row in asset_candidates), default=0.0)
    if best_video >= 0.55 and asset_candidates:
        hint = "review_for_approval"
        rationale = "video and asset candidates overlap the recovery window or expected action"
    elif video_candidates or transcript_candidates or asset_candidates:
        hint = "needs_review_with_candidates"
        rationale = "candidate evidence exists but should not be auto-approved"
    else:
        hint = "recover_window_scan_required"
        rationale = "no candidate evidence was found in current artifacts"
    return {
        "decision_hint": hint,
        "rationale": rationale,
        "reviewer_actions": [
            "Inspect candidate clips and keyframes before changing confirmation status.",
            "Compare candidate time ranges with neighboring SOP steps.",
            "Use transcript evidence only as support unless paired with visual evidence.",
        ],
        "note_template": f"{step.get('step_id')}: reviewed recovery candidates; visual_match=; transcript_support=; decision=",
        "candidate_strength": {
            "best_video_match_score": round(best_video, 4),
            "best_asset_match_score": round(best_asset, 4),
            "transcript_candidate_count": len(transcript_candidates),
        },
    }


def _search_terms(step: Mapping[str, Any]) -> list[str]:
    values: list[Any] = [
        step.get("expected_action"),
        step.get("name"),
        step.get("missing_completion_reason"),
        *_as_list(step.get("confidence_reasons")),
        *_as_list(step.get("conflict_flags")),
    ]
    terms: list[str] = []
    for value in values:
        normalized = _normalize(value)
        if normalized and normalized not in STOP_TERMS:
            terms.append(normalized)
        for token in WORD_RE.findall(str(value or "").lower()):
            token = token.strip("_")
            if len(token) >= 3 and token not in STOP_TERMS:
                terms.append(token)
    expected = _normalize(step.get("expected_action"))
    terms.extend(ACTION_ALIASES.get(expected, []))
    return _dedupe(terms)


def _recovery_reason(step: Mapping[str, Any], threshold: float) -> str:
    status = str(step.get("status") or "")
    confidence = _as_float(step.get("confidence")) or 0.0
    if status == "not_observed":
        return "step is not observed in the current process artifact"
    if step.get("inferred") and confidence < threshold:
        return f"inferred step confidence {confidence:.3f} is below threshold {threshold:.3f}"
    if confidence < threshold:
        return f"step confidence {confidence:.3f} is below threshold {threshold:.3f}"
    return "step requires recovery review"


def _in_window(row: Mapping[str, Any], window: Mapping[str, Any]) -> bool:
    window_start = _parse_optional_time(window.get("global_start_time"))
    window_end = _parse_optional_time(window.get("global_end_time"))
    if window_start is None or window_end is None:
        return False
    row_start = _row_start(row) or _row_end(row)
    row_end = _row_end(row) or row_start
    if row_start is None or row_end is None:
        return False
    return row_start <= window_end and row_end >= window_start


def _row_start(row: Mapping[str, Any]) -> datetime | None:
    return _parse_optional_time(row.get("global_start_time") or row.get("global_time") or row.get("start_time"))


def _row_end(row: Mapping[str, Any]) -> datetime | None:
    return _parse_optional_time(row.get("global_end_time") or row.get("global_time") or row.get("end_time"))


def _parse_optional_time(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        return parse_time(str(value))
    except (TypeError, ValueError):
        return None


def _row_text(row: Mapping[str, Any]) -> str:
    values: list[Any] = []
    for key in (
        "text",
        "summary",
        "search_text",
        "event_type",
        "action_type",
        "primary_object",
        "asset_type",
        "source_type",
        "evidence_level",
    ):
        values.append(row.get(key))
    for key in ("objects", "actions", "state_tags", "confidence_reasons", "anomaly_flags"):
        values.extend(_as_list(row.get(key)))
    return _normalize(" ".join(str(value or "") for value in values))


def _session_id(*row_groups: list[Mapping[str, Any]], session: Path) -> str:
    for rows in row_groups:
        for row in rows:
            if row.get("session_id"):
                return str(row.get("session_id"))
    return session.name


def _top_candidates(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (_as_float(row.get("match_score")) or 0.0, _as_float(row.get("confidence")) or 0.0), reverse=True)
    return rows[: max(0, limit)]


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return paths[0] if paths else None


def _path_status(path: Path | None) -> dict[str, Any]:
    return {"path": str(path) if path is not None else None, "exists": bool(path and path.exists())}


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("-", "_").split())


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _short_text(value: Any, limit: int = 240) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


__all__ = ["build_missing_step_recovery_plan"]
