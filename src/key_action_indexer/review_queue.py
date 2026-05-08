from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .health_report import build_run_health_report
from .schemas import read_jsonl, write_jsonl


REVIEW_STATE_FILENAME = "key_action_review_state.json"
REVIEW_EXPORT_FILENAME = "key_action_review_export.json"


def build_quality_convergence(session_dir: str | Path) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    health = build_run_health_report(session)
    segments = _read_jsonl(metadata / "key_action_segments.jsonl")
    micros = _read_jsonl(metadata / "micro_segments.jsonl")
    vector_rows = _read_jsonl(metadata / "vector_metadata.jsonl")
    micro_vector_rows = _read_jsonl(metadata / "micro_vector_metadata.jsonl")
    state = load_review_state(session)
    decisions = _latest_decisions(state)
    metrics = health.get("metrics") if isinstance(health.get("metrics"), Mapping) else {}
    review_counts = Counter(str(row.get("decision") or "pending") for row in decisions.values())
    unreviewed = _unreviewed_evidence_count(segments, micros, decisions)
    score = _health_score(health, metrics, unreviewed)
    split_candidates = _long_segment_split_candidates(
        segments,
        micros,
        video_duration_sec=_float(metrics.get("video_duration_sec")),
        max_longest_segment_ratio=0.5,
        target_chunk_sec=30.0,
    )
    boundary_candidates = _boundary_refinement_candidates(segments)
    convergence = {
        "schema_version": "key_action_quality_convergence.v1",
        "generated_at": _now(),
        "session_dir": str(session),
        "status": "pass" if score >= 82 and not health.get("error_count") else "needs_review",
        "health_score": score,
        "health": health,
        "core_metrics": {
            "segment_count": len(segments),
            "micro_segment_count": len(micros),
            "longest_segment_sec": metrics.get("longest_segment_sec"),
            "longest_segment_ratio": metrics.get("longest_segment_ratio"),
            "total_action_coverage_ratio": metrics.get("total_action_coverage_ratio"),
            "vector_count": len(vector_rows) + len(micro_vector_rows),
            "unreviewed_count": unreviewed,
            "review_decision_counts": dict(sorted(review_counts.items())),
        },
        "boundary_policy": {
            "min_boundary_confidence": 0.01,
            "max_total_coverage_ratio": 0.65,
            "max_longest_segment_ratio": 0.5,
            "target_split_chunk_sec": 30.0,
            "strategy": [
                "Prefer YOLO physical-evidence boundary support when present.",
                "Split coarse segments by existing micro-segment windows first.",
                "Fallback to equal time windows for review-only boundary proposals.",
                "Do not mutate raw detection artifacts until a reviewer approves the proposal.",
            ],
        },
        "coverage_check": {
            "status": "warning"
            if _float(metrics.get("total_action_coverage_ratio"), 0.0) > 0.65
            else "pass",
            "coverage_ratio": metrics.get("total_action_coverage_ratio"),
            "threshold": 0.65,
        },
        "boundary_refinement_candidates": boundary_candidates,
        "long_segment_split_candidates": split_candidates,
        "recommendations": _quality_recommendations(health, metrics, split_candidates, boundary_candidates, unreviewed),
    }
    try:
        from .quality_gate import build_quality_gate

        gate = build_quality_gate(session, quality_convergence=convergence)
        convergence["quality_gate"] = gate
        gate_summary = gate.get("summary") if isinstance(gate.get("summary"), Mapping) else {}
        if gate_summary.get("metric_source") == "reviewed_dataset":
            gate_metrics = gate.get("metrics") if isinstance(gate.get("metrics"), Mapping) else {}
            convergence["core_metrics"].update(
                {
                    "segment_count": gate_metrics.get("segment_count", convergence["core_metrics"].get("segment_count")),
                    "micro_segment_count": gate_metrics.get("micro_segment_count", convergence["core_metrics"].get("micro_segment_count")),
                    "longest_segment_sec": gate_metrics.get("longest_segment_sec"),
                    "longest_segment_ratio": gate_metrics.get("longest_segment_ratio"),
                    "total_action_coverage_ratio": gate_metrics.get("total_action_coverage_ratio"),
                    "vector_count": gate_summary.get("vector_count", convergence["core_metrics"].get("vector_count")),
                    "unreviewed_count": gate_summary.get("unreviewed_count", convergence["core_metrics"].get("unreviewed_count")),
                }
            )
            score_check = next((item for item in gate.get("checks") or [] if isinstance(item, Mapping) and item.get("name") == "health_score"), None)
            if isinstance(score_check, Mapping) and score_check.get("actual") is not None:
                convergence["health_score"] = score_check.get("actual")
        convergence["status"] = "pass" if gate.get("can_mark_complete") else "blocked"
    except Exception as exc:
        convergence["quality_gate"] = {
            "schema_version": "key_action_quality_gate.error",
            "status": "fail",
            "can_mark_complete": False,
            "error": str(exc),
        }
    target = metadata / "quality_convergence.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(convergence, ensure_ascii=False, indent=2), encoding="utf-8")
    return convergence


def build_review_queue(
    session_dir: str | Path,
    *,
    material_candidates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    segments = _read_jsonl(metadata / "key_action_segments.jsonl")
    micros = _read_jsonl(metadata / "micro_segments.jsonl")
    convergence = build_quality_convergence(session)
    state = load_review_state(session)
    decisions = _latest_decisions(state)
    items: list[dict[str, Any]] = []

    for issue in (convergence.get("health") or {}).get("errors") or []:
        if isinstance(issue, Mapping):
            items.append(_issue_item(issue, severity="error", decisions=decisions))
    for issue in (convergence.get("health") or {}).get("warnings") or []:
        if isinstance(issue, Mapping):
            items.append(_issue_item(issue, severity="warning", decisions=decisions))
    gate = convergence.get("quality_gate") if isinstance(convergence.get("quality_gate"), Mapping) else {}
    adapter_validation = gate.get("adapter_validation") if isinstance(gate.get("adapter_validation"), Mapping) else {}
    for item in _semantic_adapter_issue_items(adapter_validation, decisions=decisions):
        items.append(item)

    split_by_segment = {
        str(item.get("segment_id") or ""): item
        for item in convergence.get("long_segment_split_candidates") or []
        if isinstance(item, Mapping)
    }
    boundary_by_segment = {
        str(item.get("segment_id") or ""): item
        for item in convergence.get("boundary_refinement_candidates") or []
        if isinstance(item, Mapping)
    }
    for index, segment in enumerate(segments, start=1):
        segment_id = str(segment.get("segment_id") or f"segment_{index:06d}")
        confidence = _segment_confidence(segment)
        reasons = []
        if confidence is None or confidence < 0.55:
            reasons.append("low_boundary_or_activity_confidence")
        if segment_id in split_by_segment:
            reasons.append("coarse_long_segment")
        if segment_id in boundary_by_segment:
            reasons.append("needs_boundary_refinement")
        if not reasons:
            continue
        item = _segment_item(segment, index, reasons, split_by_segment.get(segment_id), boundary_by_segment.get(segment_id))
        item.update(_decision_public(decisions.get(item["item_id"])))
        items.append(item)

    for index, micro in enumerate(micros, start=1):
        reasons = _micro_review_reasons(micro)
        if not reasons:
            continue
        item = _micro_item(micro, index, reasons)
        item.update(_decision_public(decisions.get(item["item_id"])))
        items.append(item)

    for group in _material_groups(material_candidates):
        item = _material_item(group)
        item.update(_decision_public(decisions.get(item["item_id"])))
        items.append(item)

    items.sort(key=lambda row: (_severity_rank(row.get("severity")), _type_rank(row.get("item_type")), str(row.get("item_id"))))
    counts = Counter(str(item.get("review_status") or "pending") for item in items)
    payload = {
        "schema_version": "key_action_review_queue.v1",
        "generated_at": _now(),
        "session_dir": str(session),
        "summary": {
            "total": len(items),
            "pending": counts.get("pending", 0),
            "approved": counts.get("approved", 0),
            "rejected": counts.get("rejected", 0),
            "needs_review": counts.get("needs_review", 0),
            "quality_score": convergence.get("health_score"),
            "segment_count": (convergence.get("core_metrics") or {}).get("segment_count", len(segments)),
            "micro_segment_count": (convergence.get("core_metrics") or {}).get("micro_segment_count", len(micros)),
            "long_segment_candidate_count": len(convergence.get("long_segment_split_candidates") or []),
            "boundary_refinement_candidate_count": len(convergence.get("boundary_refinement_candidates") or []),
        },
        "quality": convergence,
        "items": items,
        "state_path": str(_state_path(session)),
    }
    (metadata / "review_queue.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(metadata / "review_queue.jsonl", items)
    return payload


def apply_review_decision(
    session_dir: str | Path,
    *,
    item_id: str,
    decision: str,
    reviewer: str = "system",
    note: str = "",
    boundary_start_sec: float | None = None,
    boundary_end_sec: float | None = None,
) -> dict[str, Any]:
    normalized = str(decision or "").strip().lower()
    if normalized not in {"approved", "rejected", "needs_review", "pending"}:
        raise ValueError("decision must be one of: approved, rejected, needs_review, pending")
    session = Path(session_dir)
    state = load_review_state(session)
    decisions = state.setdefault("decisions", {})
    audit = state.setdefault("audit", [])
    now = _now()
    record = {
        "item_id": item_id,
        "decision": normalized,
        "reviewer": reviewer or "system",
        "note": note or "",
        "updated_at": now,
    }
    if boundary_start_sec is not None:
        record["boundary_start_sec"] = float(boundary_start_sec)
    if boundary_end_sec is not None:
        record["boundary_end_sec"] = float(boundary_end_sec)
    previous = decisions.get(item_id)
    decisions[item_id] = record
    audit.append({"audit_id": f"{item_id}:{now}", "previous": previous, "current": record})
    state["updated_at"] = now
    save_review_state(session, state)
    try:
        from .reviewed_dataset import freeze_reviewed_dataset

        freeze_reviewed_dataset(session, create_release=False)
    except Exception:
        # A review decision must remain durable even if downstream artifacts need
        # a later rebuild.
        pass
    return record


def export_review_queue(session_dir: str | Path, queue: Mapping[str, Any] | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    reviewed_dataset = None
    reviewed_export = None
    try:
        from .reviewed_dataset import freeze_reviewed_dataset, load_reviewed_export

        reviewed_dataset = freeze_reviewed_dataset(session)
        reviewed_export = load_reviewed_export(session)
    except Exception as exc:
        reviewed_dataset = {"schema_version": "key_action_reviewed_dataset.error", "error": str(exc)}
    payload = {
        "schema_version": "key_action_review_export.v1",
        "generated_at": _now(),
        "session_dir": str(session),
        "state": load_review_state(session),
        "queue": dict(queue or build_review_queue(session)),
        "reviewed_dataset": reviewed_dataset,
        "reviewed_export": reviewed_export,
    }
    target = session / "metadata" / REVIEW_EXPORT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    payload["export_path"] = str(target)
    return payload


def load_review_state(session_dir: str | Path) -> dict[str, Any]:
    path = _state_path(Path(session_dir))
    if not path.exists():
        return {"schema_version": "key_action_review_state.v1", "decisions": {}, "audit": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("schema_version", "key_action_review_state.v1")
    data.setdefault("decisions", {})
    data.setdefault("audit", [])
    return data


def save_review_state(session_dir: str | Path, state: Mapping[str, Any]) -> None:
    path = _state_path(Path(session_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(state), ensure_ascii=False, indent=2), encoding="utf-8")


def _issue_item(issue: Mapping[str, Any], *, severity: str, decisions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    code = str(issue.get("code") or "quality_issue")
    item_id = f"qa:{code}"
    item = {
        "item_id": item_id,
        "item_type": "qa_warning",
        "source_id": code,
        "title": code.replace("_", " "),
        "summary": issue.get("message") or code,
        "severity": severity,
        "review_status": "pending",
        "reasons": [code],
        "payload": dict(issue),
    }
    item.update(_decision_public(decisions.get(item_id)))
    return item


def _semantic_adapter_issue_items(
    adapter_validation: Mapping[str, Any],
    *,
    decisions: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    adapters = adapter_validation.get("adapters") if isinstance(adapter_validation.get("adapters"), Mapping) else {}
    items: list[dict[str, Any]] = []
    for adapter_name, adapter in adapters.items():
        if not isinstance(adapter, Mapping):
            continue
        for index, issue in enumerate(adapter.get("issues") or [], start=1):
            if not isinstance(issue, Mapping):
                continue
            code = str(issue.get("code") or "")
            if not code.startswith("semantic_"):
                continue
            row = issue.get("row")
            item_id = f"evidence_semantic:{adapter_name}:{code}:{row or index}"
            start = None
            end = None
            coverage = adapter.get("coverage") if isinstance(adapter.get("coverage"), Mapping) else {}
            if row is None:
                start = _float(coverage.get("start_sec"))
                end = _float(coverage.get("end_sec"))
            item = {
                "item_id": item_id,
                "item_type": "evidence_semantic",
                "source_id": f"{adapter_name}:{row or index}",
                "title": f"{adapter_name} semantic evidence",
                "summary": issue.get("message") or code,
                "severity": issue.get("severity") or "warning",
                "review_status": "pending",
                "start_sec": start,
                "end_sec": end,
                "duration_sec": (end - start) if start is not None and end is not None and end >= start else None,
                "reasons": [code],
                "payload": {
                    "adapter": adapter_name,
                    "issue": dict(issue),
                    "adapter_status": adapter.get("status"),
                    "coverage": dict(coverage),
                },
            }
            item.update(_decision_public(decisions.get(item_id)))
            items.append(item)
    return items


def _segment_item(
    segment: Mapping[str, Any],
    index: int,
    reasons: list[str],
    split_candidate: Mapping[str, Any] | None,
    boundary_candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    segment_id = str(segment.get("segment_id") or f"segment_{index:06d}")
    start, end = _segment_seconds(segment)
    item_id = f"segment:{segment_id}"
    return {
        "item_id": item_id,
        "item_type": "segment",
        "source_id": segment_id,
        "segment_id": segment_id,
        "title": _segment_title(segment, index),
        "summary": _index_text(segment),
        "severity": "warning" if "coarse_long_segment" in reasons else "info",
        "review_status": "pending",
        "confidence": _segment_confidence(segment),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": _float(segment.get("duration_sec")),
        "reasons": reasons,
        "boundary": {
            "original_start_sec": start,
            "original_end_sec": end,
            "proposed_start_sec": start,
            "proposed_end_sec": end,
            "split_candidate": split_candidate,
            "refinement_candidate": boundary_candidate,
        },
        "clip_paths": _clip_paths(segment),
        "preview_paths": _preview_paths(segment),
        "payload": {"segment": dict(segment)},
    }


def _micro_item(micro: Mapping[str, Any], index: int, reasons: list[str]) -> dict[str, Any]:
    micro_id = str(micro.get("micro_segment_id") or f"micro_{index:06d}")
    start = _float(micro.get("start_sec"))
    end = _float(micro.get("end_sec"))
    item_id = f"micro:{micro_id}"
    return {
        "item_id": item_id,
        "item_type": "micro_segment",
        "source_id": micro_id,
        "segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "micro_segment_id": micro_id,
        "title": _micro_title(micro, index),
        "summary": _index_text(micro),
        "severity": "warning" if any("insufficient" in reason or "low" in reason for reason in reasons) else "info",
        "review_status": "pending",
        "confidence": _micro_confidence(micro),
        "start_sec": start,
        "end_sec": end,
        "duration_sec": _float(micro.get("duration_sec")),
        "reasons": reasons,
        "boundary": {
            "original_start_sec": start,
            "original_end_sec": end,
            "proposed_start_sec": start,
            "proposed_end_sec": end,
        },
        "clip_paths": _clip_paths(micro),
        "preview_paths": _preview_paths(micro),
        "payload": {"micro_segment": dict(micro)},
    }


def _material_item(group: Mapping[str, Any]) -> dict[str, Any]:
    group_id = str(group.get("candidate_group_id") or "material_candidate")
    files = [item for item in group.get("files") or [] if isinstance(item, Mapping)]
    item_id = f"material:{group_id}"
    return {
        "item_id": item_id,
        "item_type": "material_candidate",
        "source_id": group_id,
        "segment_id": group.get("parent_segment_id"),
        "micro_segment_id": group.get("micro_segment_id"),
        "title": "material candidate " + group_id,
        "summary": " / ".join(str(value) for value in [group.get("action_name"), group.get("primary_object")] if value),
        "severity": "info",
        "review_status": "pending",
        "confidence": _float(group.get("quality_score")),
        "reasons": ["unconfirmed_material_candidate"],
        "preview_paths": [row.get("path") or row.get("frame_path") for row in files if str(row.get("asset_kind") or "") == "关键帧"],
        "clip_paths": [row.get("path") or row.get("clip_file_path") for row in files if str(row.get("asset_kind") or "") == "关键片段"],
        "payload": {"candidate_group": dict(group)},
    }


def _material_groups(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    groups = []
    for group in payload.get("items") or []:
        if not isinstance(group, Mapping):
            continue
        status = str(group.get("status") or group.get("review_status") or "pending").lower()
        if status in {"approved", "accepted", "locked"}:
            continue
        groups.append(group)
    return groups


def _boundary_refinement_candidates(segments: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        boundary_source = str(segment.get("boundary_source") or "")
        support_count = _float(segment.get("boundary_support_count"), 0.0)
        confidence = _segment_confidence(segment)
        if boundary_source.startswith("yolo_physical_evidence") and support_count > 0 and (confidence is None or confidence >= 0.55):
            continue
        start, end = _segment_seconds(segment)
        output.append(
            {
                "segment_id": segment_id,
                "reason": "unrefined_or_low_confidence_boundary",
                "boundary_source": boundary_source or None,
                "boundary_support_count": support_count,
                "confidence": confidence,
                "current_start_sec": start,
                "current_end_sec": end,
                "proposal": "review yolo_frame_rows around start/end and tighten to first/last physical interaction support",
            }
        )
    return output


def _long_segment_split_candidates(
    segments: list[Mapping[str, Any]],
    micros: list[Mapping[str, Any]],
    *,
    video_duration_sec: float | None,
    max_longest_segment_ratio: float,
    target_chunk_sec: float,
) -> list[dict[str, Any]]:
    micros_by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for micro in micros:
        parent = str(micro.get("parent_segment_id") or micro.get("segment_id") or "")
        if parent:
            micros_by_parent.setdefault(parent, []).append(micro)
    output = []
    absolute_threshold = max(target_chunk_sec * 1.5, 45.0)
    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        duration = _float(segment.get("duration_sec"), 0.0)
        ratio = (duration / video_duration_sec) if video_duration_sec else None
        if duration <= absolute_threshold and not (ratio is not None and ratio > max_longest_segment_ratio):
            continue
        start, end = _segment_seconds(segment)
        children = sorted(micros_by_parent.get(segment_id, []), key=lambda row: _float(row.get("start_sec"), 0.0))
        if children:
            proposed = [
                {
                    "micro_segment_id": child.get("micro_segment_id"),
                    "start_sec": _float(child.get("start_sec")),
                    "end_sec": _float(child.get("end_sec")),
                    "primary_object": child.get("primary_object"),
                    "interaction_type": child.get("interaction_type"),
                }
                for child in children
            ]
            source = "micro_segments"
        else:
            proposed = _equal_chunks(start, end, target_chunk_sec)
            source = "equal_time_chunks"
        output.append(
            {
                "segment_id": segment_id,
                "duration_sec": duration,
                "longest_segment_ratio": round(ratio, 6) if ratio is not None else None,
                "proposal_source": source,
                "proposed_splits": proposed,
            }
        )
    return output


def _equal_chunks(start: float | None, end: float | None, target_chunk_sec: float) -> list[dict[str, Any]]:
    if start is None or end is None or end <= start:
        return []
    duration = end - start
    count = max(2, int(math.ceil(duration / max(target_chunk_sec, 1.0))))
    chunk = duration / count
    return [
        {"start_sec": round(start + chunk * index, 4), "end_sec": round(start + chunk * (index + 1), 4)}
        for index in range(count)
    ]


def _quality_recommendations(
    health: Mapping[str, Any],
    metrics: Mapping[str, Any],
    split_candidates: list[Mapping[str, Any]],
    boundary_candidates: list[Mapping[str, Any]],
    unreviewed: int,
) -> list[dict[str, Any]]:
    recs = []
    codes = {str(item.get("code") or "") for item in [*(health.get("warnings") or []), *(health.get("errors") or [])] if isinstance(item, Mapping)}
    if "unrefined_boundaries" in codes or boundary_candidates:
        recs.append({"priority": "P0", "action": "review_boundary_candidates", "reason": "segments need YOLO-backed physical boundary support"})
    if "high_total_action_coverage" in codes:
        recs.append({"priority": "P0", "action": "tighten_coverage", "reason": f"coverage ratio {metrics.get('total_action_coverage_ratio')} exceeds 0.65"})
    if "coarse_longest_segment" in codes or split_candidates:
        recs.append({"priority": "P1", "action": "split_long_segments", "reason": "longest segment is too coarse for retrieval and review"})
    if unreviewed:
        recs.append({"priority": "P1", "action": "clear_review_queue", "reason": f"{unreviewed} evidence items still need human review"})
    return recs


def _health_score(health: Mapping[str, Any], metrics: Mapping[str, Any], unreviewed: int) -> int:
    score = 100.0
    score -= int(health.get("error_count") or 0) * 22
    score -= int(health.get("warning_count") or 0) * 7
    coverage = _float(metrics.get("total_action_coverage_ratio"))
    if coverage is not None and coverage > 0.65:
        score -= min(18.0, (coverage - 0.65) * 40)
    longest = _float(metrics.get("longest_segment_ratio"))
    if longest is not None and longest > 0.5:
        score -= min(15.0, (longest - 0.5) * 38)
    score -= min(14.0, unreviewed * 0.8)
    return int(max(0, min(100, round(score))))


def _unreviewed_evidence_count(
    segments: list[Mapping[str, Any]],
    micros: list[Mapping[str, Any]],
    decisions: Mapping[str, Mapping[str, Any]],
) -> int:
    count = 0
    for index, segment in enumerate(segments, start=1):
        item_id = f"segment:{segment.get('segment_id') or f'segment_{index:06d}'}"
        if item_id not in decisions and (_segment_confidence(segment) or 0.0) < 0.55:
            count += 1
    for index, micro in enumerate(micros, start=1):
        item_id = f"micro:{micro.get('micro_segment_id') or f'micro_{index:06d}'}"
        if item_id not in decisions and _micro_review_reasons(micro):
            count += 1
    return count


def _micro_review_reasons(micro: Mapping[str, Any]) -> list[str]:
    reasons = []
    level = str(micro.get("evidence_level") or micro.get("confirmation_level") or "").lower()
    confidence = _micro_confidence(micro)
    review_status = str(micro.get("review_status") or "").lower()
    quality = micro.get("quality") if isinstance(micro.get("quality"), Mapping) else {}
    warnings = [str(item) for item in quality.get("warnings") or micro.get("quality_warnings") or []]
    if any(token in level for token in ("candidate", "insufficient", "weak", "review")):
        reasons.append(f"evidence_level:{level}")
    if confidence is not None and confidence < 0.65:
        reasons.append("low_micro_confidence")
    if review_status in {"pending", "needs_review"}:
        reasons.append(f"review_status:{review_status}")
    if warnings:
        reasons.extend([f"quality_warning:{item}" for item in warnings[:4]])
    return _dedupe(reasons)


def _decision_public(decision: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(decision, Mapping):
        return {"review_status": "pending"}
    output = {
        "review_status": decision.get("decision") or "pending",
        "reviewer": decision.get("reviewer"),
        "review_note": decision.get("note"),
        "reviewed_at": decision.get("updated_at"),
    }
    if decision.get("boundary_start_sec") is not None:
        output["adjusted_start_sec"] = decision.get("boundary_start_sec")
    if decision.get("boundary_end_sec") is not None:
        output["adjusted_end_sec"] = decision.get("boundary_end_sec")
    return output


def _latest_decisions(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = state.get("decisions") if isinstance(state.get("decisions"), Mapping) else {}
    return {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}


def _state_path(session: Path) -> Path:
    return session / "metadata" / REVIEW_STATE_FILENAME


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _segment_seconds(segment: Mapping[str, Any]) -> tuple[float | None, float | None]:
    for key in ("third_person", "first_person"):
        ref = segment.get(key)
        if isinstance(ref, Mapping):
            start = _float(ref.get("local_start_sec"))
            end = _float(ref.get("local_end_sec"))
            if start is not None or end is not None:
                return start, end
    return _float(segment.get("start_sec")), _float(segment.get("end_sec"))


def _segment_confidence(segment: Mapping[str, Any]) -> float | None:
    for key in ("boundary_confidence", "confidence", "score"):
        value = _float(segment.get(key))
        if value is not None:
            return value
    cv = segment.get("cv_detection") if isinstance(segment.get("cv_detection"), Mapping) else {}
    values = [_float(cv.get("avg_active_score")), _float(cv.get("avg_motion_score")), _float(cv.get("confidence"))]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _micro_confidence(micro: Mapping[str, Any]) -> float | None:
    for key in ("confidence_score", "max_interaction_score", "score"):
        value = _float(micro.get(key))
        if value is not None:
            return value
    interaction = micro.get("interaction") if isinstance(micro.get("interaction"), Mapping) else {}
    for key in ("max_interaction_score", "avg_interaction_score"):
        value = _float(interaction.get(key))
        if value is not None:
            return value
    text = str(micro.get("confidence") or "").lower()
    if "high" in text:
        return 0.85
    if "medium" in text:
        return 0.6
    if "low" in text:
        return 0.35
    return None


def _segment_title(segment: Mapping[str, Any], index: int) -> str:
    text_description = segment.get("text_description") if isinstance(segment.get("text_description"), Mapping) else {}
    return str(text_description.get("summary") or text_description.get("action_type") or segment.get("segment_id") or f"Segment {index}")


def _micro_title(micro: Mapping[str, Any], index: int) -> str:
    text_description = micro.get("text_description") if isinstance(micro.get("text_description"), Mapping) else {}
    return str(
        text_description.get("summary")
        or micro.get("display_id")
        or micro.get("interaction_type")
        or micro.get("primary_object")
        or micro.get("micro_segment_id")
        or f"Micro {index}"
    )


def _index_text(row: Mapping[str, Any]) -> str:
    index = row.get("index") if isinstance(row.get("index"), Mapping) else {}
    text_description = row.get("text_description") if isinstance(row.get("text_description"), Mapping) else {}
    return str(index.get("index_text") or text_description.get("index_text") or text_description.get("summary") or "")


def _clip_paths(row: Mapping[str, Any]) -> list[Any]:
    paths = []
    for key in ("first_person", "third_person"):
        ref = row.get(key)
        if isinstance(ref, Mapping):
            paths.extend([ref.get("annotated_clip_path"), ref.get("clip_path")])
    paths.extend([row.get("first_person_clip"), row.get("third_person_clip"), row.get("clip_path")])
    return [path for path in _dedupe(paths) if path]


def _preview_paths(row: Mapping[str, Any]) -> list[Any]:
    paths = []
    for key in ("keyframes",):
        ref = row.get(key)
        if isinstance(ref, Mapping):
            paths.extend(ref.get(name) for name in ("contact_frame", "peak_frame", "release_frame", "start", "middle", "end"))
    paths.extend([row.get("peak_keyframe"), row.get("keyframe_path"), row.get("preview_path")])
    return [path for path in _dedupe(paths) if path]


def _type_rank(value: Any) -> int:
    return {"qa_warning": 0, "evidence_semantic": 1, "segment": 2, "micro_segment": 3, "material_candidate": 4}.get(str(value), 9)


def _severity_rank(value: Any) -> int:
    return {"error": 0, "warning": 1, "info": 2}.get(str(value), 3)


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe(values: Any) -> list[Any]:
    output = []
    seen = set()
    for value in values or []:
        if value is None or value == "":
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "REVIEW_EXPORT_FILENAME",
    "REVIEW_STATE_FILENAME",
    "apply_review_decision",
    "build_quality_convergence",
    "build_review_queue",
    "export_review_queue",
    "load_review_state",
    "save_review_state",
]
