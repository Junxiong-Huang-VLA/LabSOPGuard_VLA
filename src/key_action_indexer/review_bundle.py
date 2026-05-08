"""Export lightweight human review bundles for process steps."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .confirmation_loop import resolve_review_evidence_for_step
from .schemas import read_jsonl, write_jsonl


def export_review_bundle(
    session_dir: str | Path,
    output_path: str | Path,
    *,
    format: str = "json",
) -> Dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    queue_path = metadata / "human_confirmation_queue.jsonl"
    queue = read_jsonl(queue_path) if queue_path.exists() else []
    process = _read_json(metadata / "experiment_process.json")
    steps_by_id = {str(step.get("step_id") or ""): step for step in process.get("steps", []) if isinstance(step, Mapping)}

    items: List[Dict[str, Any]] = []
    for row in queue:
        if not isinstance(row, Mapping):
            continue
        step_id = str(row.get("item_id") or "")
        step = steps_by_id.get(step_id, {})
        evidence_payload = (
            resolve_review_evidence_for_step(session, step)
            if step
            else {"resolved_evidence_refs": [], "keyframe_refs": [], "clip_refs": []}
        )
        keyframe_paths = _ordered_paths(
            [
                *_all_refs(row, "keyframe"),
                *[ref.get("path") for ref in evidence_payload.get("keyframe_refs") or [] if isinstance(ref, Mapping)],
            ]
        )
        clip_paths = _ordered_paths(
            [
                _first_ref(row, "clip"),
                *[ref.get("path") for ref in evidence_payload.get("clip_refs") or [] if isinstance(ref, Mapping)],
            ]
        )
        items.append(
            {
                "confirmation_id": row.get("confirmation_id"),
                "step_id": step_id,
                "step_name": step.get("name") or row.get("summary"),
                "micro_segment_id": row.get("micro_segment_id"),
                "candidate_event_types": row.get("candidate_event_types") or [],
                "clip_path": clip_paths[0] if clip_paths else "",
                "keyframe_paths": keyframe_paths,
                "evidence_refs": row.get("evidence_refs") or [],
                "resolved_evidence_refs": evidence_payload.get("resolved_evidence_refs") or [],
                "clip_refs": evidence_payload.get("clip_refs") or [],
                "keyframe_refs": evidence_payload.get("keyframe_refs") or [],
                "current_confidence": row.get("confidence"),
                "status": row.get("status"),
                "suggested_decision": "approve" if float(row.get("confidence") or 0.0) >= 0.8 else "review",
                "reviewer_prompt": _reviewer_prompt(row, step),
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if format == "md":
        _write_md(out, items)
    elif format == "html":
        _write_html(out, items)
    else:
        out.write_text(json.dumps({"items": items, "count": len(items)}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"format": format, "item_count": len(items), "output_path": str(out)}


def apply_review_to_process(
    session_dir: str | Path,
    confirmation_id: str,
    decision: str,
    operator: str,
    rationale: str = "",
) -> Dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    exports = session / "exports"
    exports.mkdir(parents=True, exist_ok=True)

    process_path = metadata / "experiment_process.json"
    process = _read_json(process_path)
    review_entry = {
        "confirmation_id": confirmation_id,
        "decision": decision,
        "operator": operator,
        "rationale": rationale,
        "timestamp": _now(),
    }

    for step in process.get("steps", []):
        if not isinstance(step, Mapping):
            continue
        candidate_id = f"{process.get('session_id', '')}:{step.get('step_id', '')}"
        if candidate_id == confirmation_id or str(step.get("step_id") or "") == confirmation_id:
            step["confirmation_status"] = decision
            if decision in {"approved", "approve"}:
                step["status"] = "completed"
                step["completed"] = True
                step["requires_human_confirmation"] = False
            break

    process_path.write_text(json.dumps(process, ensure_ascii=False, indent=2), encoding="utf-8")

    record_path = exports / "process_record.json"
    record = _read_json(record_path)
    reviewers = record.get("reviewers") or []
    reviewers.append(
        {
            "operator": operator,
            "role": "reviewer",
            "decision": decision,
            "decision_time": _now(),
            "note": rationale,
            "confirmation_id": confirmation_id,
        }
    )
    record["reviewers"] = reviewers
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    review_path = metadata / "step_review_history.jsonl"
    existing = read_jsonl(review_path) if review_path.exists() else []
    existing.append(review_entry)
    write_jsonl(review_path, existing)
    return {"confirmation_id": confirmation_id, "decision": decision, "operator": operator}


def _reviewer_prompt(row: Mapping[str, Any], step: Mapping[str, Any]) -> str:
    name = step.get("name") or row.get("summary") or "unknown step"
    confidence = float(row.get("confidence") or 0.0)
    status = row.get("status") or "pending"
    return f"Review '{name}' (confidence={confidence:.2f}, status={status}). Approve, reject, or defer?"


def _first_ref(row: Mapping[str, Any], asset_type: str) -> str:
    for collection_name in ("evidence_refs", "clip_refs"):
        for ref in row.get(collection_name) or []:
            if not isinstance(ref, Mapping):
                continue
            ref_type = str(ref.get("asset_type") or ref.get("type") or "")
            if collection_name == "clip_refs" or ref_type.startswith(asset_type):
                return str(ref.get("path") or "")
    return ""


def _all_refs(row: Mapping[str, Any], asset_type: str) -> List[str]:
    paths: list[str] = []
    for collection_name in ("evidence_refs", "keyframe_refs"):
        for ref in row.get(collection_name) or []:
            if not isinstance(ref, Mapping):
                continue
            ref_type = str(ref.get("asset_type") or ref.get("type") or "")
            if collection_name == "keyframe_refs" or ref_type.startswith(asset_type):
                paths.append(str(ref.get("path") or ""))
    return [path for path in paths if path]


def _ordered_paths(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "")
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _write_md(path: Path, items: List[Dict[str, Any]]) -> None:
    lines = ["# Review Bundle", "", f"Total items: {len(items)}", ""]
    lines.append("| # | Step | Confidence | Status | Suggested |")
    lines.append("|---|------|------------|--------|-----------|")
    for idx, item in enumerate(items, 1):
        confidence = float(item.get("current_confidence") or 0.0)
        lines.append(
            f"| {idx} | {item.get('step_name', '')} | {confidence:.2f} | "
            f"{item.get('status', '')} | {item.get('suggested_decision', '')} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, items: List[Dict[str, Any]]) -> None:
    json_content = json.dumps({"items": items}, ensure_ascii=False, indent=2)
    path.write_text(f"<html><body><h1>Review Bundle</h1><pre>{json_content}</pre></body></html>", encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, OSError):
        return {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["apply_review_to_process", "export_review_bundle"]
