from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .physical_event_gate import gate_hand_object_contact, summarize_gate_decisions
from .schemas import read_jsonl, write_jsonl


HAND_LABELS = {"hand", "gloved_hand", "glove"}
OBJECT_LABELS = {
    "balance",
    "beaker",
    "container",
    "paper",
    "pipette",
    "pipette_tip",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
    "tube",
    "tube_cap",
    "tube-cap",
    "vial",
}
CONTAINER_LABELS = {"beaker", "container", "reagent_bottle", "sample_bottle", "sample_bottle_blue", "tube", "tube_cap", "tube-cap", "vial"}
DEVICE_LABELS = {"balance", "scale", "device", "panel"}
LIQUID_RELATED_LABELS = CONTAINER_LABELS | {"pipette", "pipette_tip", "spearhead"}


def build_event_candidate_trace(output_dir: str | Path) -> dict[str, Any]:
    """Write v2.2 candidate-recall diagnostics for the existing key-action pipeline.

    The key-action pipeline is the canonical dual-view YOLO evidence path in
    this repository.  This function only adds trace/evaluation artifacts; it
    does not loosen the physical gate or write final confirmed physical events.
    """
    root = Path(output_dir)
    cv_dir = root / "cv_outputs"
    metadata_dir = root / "metadata"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    yolo_rows = _read_first_existing(
        cv_dir / "yolo_frame_rows.jsonl",
        metadata_dir / "yolo_frame_rows.jsonl",
        cv_dir / "yolo_micro_frame_rows.jsonl",
    )
    detected_segments = _read_first_existing(cv_dir / "detected_segments.jsonl")
    micro_rows = _read_first_existing(metadata_dir / "micro_segments.jsonl")
    raw_micro_rows = _read_first_existing(metadata_dir / "micro_segments_raw.jsonl")
    advanced_rows = _read_first_existing(metadata_dir / "advanced_vision_evidence.jsonl")
    model_observation_rows = _read_first_existing(metadata_dir / "model_observation_events.jsonl")
    pipeline_summary = _read_json(root / "pipeline_summary.json")

    raw_proposals = _build_raw_proposals(yolo_rows, micro_rows, advanced_rows, model_observation_rows)
    gate_decisions = _gate_raw_proposals(raw_proposals)
    rejected = [_rejected_row(row, decision) for row, decision in zip(raw_proposals, gate_decisions) if decision.get("status") == "rejected"]
    gate_summary = summarize_gate_decisions(gate_decisions)
    gate_summary.setdefault("qwen_audit_enabled", False)
    gate_summary.setdefault("qwen_audit_count", 0)
    gate_summary.setdefault("pipeline", "key_action_indexer")

    trace_rows, trace_summary = _build_trace_rows(
        output_dir=root,
        yolo_rows=yolo_rows,
        detected_segments=detected_segments,
        raw_micro_rows=raw_micro_rows,
        micro_rows=micro_rows,
        raw_proposals=raw_proposals,
        gate_decisions=gate_decisions,
        pipeline_summary=pipeline_summary,
    )

    write_jsonl(root / "event_candidate_trace.jsonl", trace_rows)
    write_jsonl(root / "raw_event_proposals.jsonl", raw_proposals)
    write_jsonl(root / "physical_event_gate_decisions.jsonl", gate_decisions)
    write_jsonl(root / "rejected_physical_event_candidates.jsonl", rejected)
    write_jsonl(root / "qwen_event_audits.jsonl", [])
    _write_json(root / "physical_event_gate_summary.json", gate_summary)
    _write_json(root / "event_candidate_trace_summary.json", trace_summary)
    return {
        "schema": "event_candidate_trace_build.v2.2",
        "output_dir": str(root),
        "raw_proposal_count": len(raw_proposals),
        "gate_summary": gate_summary,
        "trace_summary": trace_summary,
        "artifacts": {
            "event_candidate_trace": str(root / "event_candidate_trace.jsonl"),
            "event_candidate_trace_summary": str(root / "event_candidate_trace_summary.json"),
            "raw_event_proposals": str(root / "raw_event_proposals.jsonl"),
            "physical_event_gate_decisions": str(root / "physical_event_gate_decisions.jsonl"),
            "rejected_physical_event_candidates": str(root / "rejected_physical_event_candidates.jsonl"),
            "qwen_event_audits": str(root / "qwen_event_audits.jsonl"),
            "physical_event_gate_summary": str(root / "physical_event_gate_summary.json"),
        },
    }


def _build_raw_proposals(
    yolo_rows: Sequence[Mapping[str, Any]],
    micro_rows: Sequence[Mapping[str, Any]],
    advanced_rows: Sequence[Mapping[str, Any]],
    model_observation_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in yolo_rows:
        interactions = [item for item in row.get("hand_object_interactions") or [] if isinstance(item, Mapping)]
        for idx, interaction in enumerate(interactions):
            proposal = _proposal_from_yolo_interaction(row, interaction, idx)
            if proposal["proposal_id"] not in seen:
                proposals.append(proposal)
                seen.add(proposal["proposal_id"])

    for row in micro_rows:
        proposal = _proposal_from_micro(row)
        if proposal and proposal["proposal_id"] not in seen:
            proposals.append(proposal)
            seen.add(proposal["proposal_id"])

    for row in advanced_rows:
        proposal = _proposal_from_advanced(row)
        if proposal and proposal["proposal_id"] not in seen:
            proposals.append(proposal)
            seen.add(proposal["proposal_id"])

    for row in model_observation_rows:
        proposal = _proposal_from_model_observation(row)
        if proposal and proposal["proposal_id"] not in seen:
            proposals.append(proposal)
            seen.add(proposal["proposal_id"])

    return sorted(proposals, key=lambda item: (float(item.get("time_start") or 0.0), str(item.get("event_type") or "")))


def _proposal_from_yolo_interaction(row: Mapping[str, Any], interaction: Mapping[str, Any], index: int) -> dict[str, Any]:
    time_sec = _float(row.get("alignment_time_sec", row.get("local_time_sec", row.get("time_sec"))))
    obj = _norm(interaction.get("object_label") or interaction.get("object_name") or "")
    frame_index = int(row.get("frame_index") or 0)
    proposal_id = f"yolo_{_norm(row.get('source_view') or row.get('view') or 'view')}_{frame_index}_{index}_{obj or 'object'}"
    return {
        "proposal_id": proposal_id,
        "event_type": "hand_object_interaction",
        "time_start": round(max(0.0, time_sec - 0.5), 3),
        "time_end": round(time_sec + 0.5, 3),
        "source": "key_action_yolo_frame_interaction",
        "actor_track_id": None,
        "object_track_ids": [],
        "evidence": {
            "frame_index": frame_index,
            "source_view": row.get("source_view") or row.get("view"),
            "detections": row.get("detections") or [],
            "hand_object_interactions": [dict(interaction)],
            "object_label": obj,
            "score": interaction.get("score") or interaction.get("confidence"),
        },
        "proposal_reason": "YOLO hand-object interaction row",
        "will_enter_gate": True,
        "blocked_before_gate": False,
        "block_reason": None,
    }


def _proposal_from_micro(row: Mapping[str, Any]) -> dict[str, Any] | None:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    event_type = _event_type_from_micro(row)
    start = _float(row.get("start_sec"))
    end = _float(row.get("end_sec"), start + 0.8)
    proposal_id = str(row.get("micro_segment_id") or row.get("display_id") or f"micro_{start:.3f}_{event_type}")
    return {
        "proposal_id": proposal_id,
        "event_type": event_type,
        "time_start": round(start, 3),
        "time_end": round(max(end, start + 0.1), 3),
        "source": "key_action_micro_segment",
        "actor_track_id": None,
        "object_track_ids": [],
        "evidence": {
            "interaction": interaction,
            "yolo_evidence": row.get("yolo_evidence") or [],
            "evidence_frame_indices": interaction.get("evidence_frame_indices") or [],
            "quality": row.get("quality") or {},
        },
        "proposal_reason": str(interaction.get("interaction_type") or row.get("evidence_level") or "micro segment candidate"),
        "will_enter_gate": True,
        "blocked_before_gate": False,
        "block_reason": None,
    }


def _proposal_from_advanced(row: Mapping[str, Any]) -> dict[str, Any] | None:
    evidence_type = str(row.get("evidence_type") or "")
    event_type = {
        "equipment_control_change": "panel_operation",
        "liquid_transfer": "liquid_transfer",
        "container_state_change": "container_state_change",
        "object_trajectory_movement": "object_move",
    }.get(evidence_type)
    if not event_type:
        return None
    start = _time_from_iso_or_fallback(row.get("global_start_time"), row.get("time_start"))
    end = _time_from_iso_or_fallback(row.get("global_end_time"), row.get("time_end"))
    if end <= start:
        end = start + 0.8
    evidence_id = str(row.get("evidence_id") or f"advanced_{event_type}_{start:.3f}")
    return {
        "proposal_id": evidence_id,
        "event_type": event_type,
        "time_start": round(start, 3),
        "time_end": round(end, 3),
        "source": "advanced_vision_evidence",
        "actor_track_id": None,
        "object_track_ids": [],
        "evidence": dict(row),
        "proposal_reason": evidence_type,
        "will_enter_gate": True,
        "blocked_before_gate": False,
        "block_reason": None,
    }


def _proposal_from_model_observation(row: Mapping[str, Any]) -> dict[str, Any] | None:
    event_type = str(row.get("event_type") or row.get("physical_action_type") or "")
    if event_type not in {"hand_object_interaction", "hand_object_contact", "object_move", "liquid_transfer", "panel_operation", "container_state_change"}:
        return None
    start = _float(row.get("time_start", row.get("start_sec", row.get("timestamp_sec"))))
    end = _float(row.get("time_end", row.get("end_sec", start + 0.8)))
    return {
        "proposal_id": str(row.get("event_id") or row.get("observation_id") or f"model_obs_{event_type}_{start:.3f}"),
        "event_type": "hand_object_interaction" if event_type == "hand_object_contact" else event_type,
        "time_start": round(start, 3),
        "time_end": round(max(end, start + 0.1), 3),
        "source": "model_observation_events",
        "actor_track_id": row.get("actor_track_id"),
        "object_track_ids": row.get("object_track_ids") or [],
        "evidence": dict(row),
        "proposal_reason": "model observation candidate",
        "will_enter_gate": True,
        "blocked_before_gate": False,
        "block_reason": None,
    }


def _gate_raw_proposals(raw_proposals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for proposal in raw_proposals:
        event_type = str(proposal.get("event_type") or "")
        if event_type == "hand_object_interaction":
            decision = gate_hand_object_contact(
                event_candidate=proposal,
                frame_evidence_list=_contact_frame_rows(proposal),
                external_observation=_contact_external_observation(proposal),
            )
        else:
            decision = _candidate_decision(proposal, reason="non_hand_object_candidate_audit_pending")
        decision.setdefault("candidate_id", proposal.get("proposal_id"))
        decision.setdefault("time_start", proposal.get("time_start"))
        decision.setdefault("time_end", proposal.get("time_end"))
        decision.setdefault("object_labels", _proposal_object_labels(proposal))
        decision.setdefault("source", proposal.get("source"))
        decisions.append(decision)
    return decisions


def _contact_frame_rows(proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), Mapping) else {}
    if evidence.get("detections") or evidence.get("hand_object_interactions"):
        return [
            {
                "frame_index": evidence.get("frame_index"),
                "detections": evidence.get("detections") or [],
                "hand_object_interactions": evidence.get("hand_object_interactions") or [],
            }
        ]
    rows: list[dict[str, Any]] = []
    for item in evidence.get("yolo_evidence") or []:
        if isinstance(item, Mapping):
            rows.append(
                {
                    "frame_index": item.get("frame_index"),
                    "detections": item.get("detections") or [],
                    "hand_object_interactions": item.get("hand_object_interactions") or [],
                }
            )
    return rows


def _contact_external_observation(proposal: Mapping[str, Any]) -> dict[str, Any]:
    rows = _contact_frame_rows(proposal)
    interactions = [item for row in rows for item in row.get("hand_object_interactions") or [] if isinstance(item, Mapping)]
    has_overlap = any((_float(item.get("iou")) > 0.0 or _float(item.get("bbox_overlap")) > 0.0) for item in interactions)
    return {
        "has_hand": bool(interactions) or any(_norm(det.get("label")) in HAND_LABELS for row in rows for det in row.get("detections") or [] if isinstance(det, Mapping)),
        "has_object": bool(_proposal_object_labels(proposal)) or any(_norm(det.get("label")) not in HAND_LABELS for row in rows for det in row.get("detections") or [] if isinstance(det, Mapping)),
        "near_only": bool(interactions) and not has_overlap,
    }


def _candidate_decision(proposal: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    event_type = str(proposal.get("event_type") or "unknown")
    return {
        "status": "candidate",
        "event_type": event_type,
        "confidence": 0.35,
        "hard_gate": {
            "passed": False,
            "status": "candidate",
            "gate_name": f"gate_{event_type}",
            "required_evidence": [],
            "passed_evidence": [],
            "failed_evidence": ["hard_gate_not_run_for_trace_only_candidate"],
        },
        "evidence": dict(proposal.get("evidence") or {}),
        "reject_reasons": [reason],
        "limitations": ["key-action trace candidate; not a final confirmed physical_event"],
        "audit": {},
    }


def _rejected_row(proposal: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": proposal.get("proposal_id"),
        "event_type": decision.get("event_type") or proposal.get("event_type"),
        "status": decision.get("status"),
        "time_start": proposal.get("time_start"),
        "time_end": proposal.get("time_end"),
        "source_view": (proposal.get("evidence") or {}).get("source_view") if isinstance(proposal.get("evidence"), Mapping) else None,
        "actor_track_id": proposal.get("actor_track_id"),
        "object_track_ids": proposal.get("object_track_ids") or [],
        "object_labels": _proposal_object_labels(proposal),
        "reject_reasons": decision.get("reject_reasons") or [],
        "evidence_detail": decision.get("evidence") or {},
        "limitations": decision.get("limitations") or [],
    }


def _build_trace_rows(
    *,
    output_dir: Path,
    yolo_rows: Sequence[Mapping[str, Any]],
    detected_segments: Sequence[Mapping[str, Any]],
    raw_micro_rows: Sequence[Mapping[str, Any]],
    micro_rows: Sequence[Mapping[str, Any]],
    raw_proposals: Sequence[Mapping[str, Any]],
    gate_decisions: Sequence[Mapping[str, Any]],
    pipeline_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = Counter()
    frames_with_detections = 0
    interactions = 0
    views = Counter()
    model_paths_by_view: dict[str, Any] = {}
    resolution_by_view: dict[str, dict[str, Any]] = {}
    resolution_profile_by_view: dict[str, dict[str, Any]] = {}
    yolo_imgsz_by_view: dict[str, Any] = {}
    sample_fps_by_view: dict[str, Any] = {}
    for row in yolo_rows:
        view = str(row.get("source_view") or row.get("view") or "unknown")
        views[view] += 1
        detections = [item for item in row.get("detections") or [] if isinstance(item, Mapping)]
        if detections:
            frames_with_detections += 1
        for det in detections:
            labels[_norm(det.get("label") or det.get("raw_label"))] += 1
        interactions += len(row.get("hand_object_interactions") or [])
        model_ref = row.get("model_ref")
        if isinstance(model_ref, Mapping):
            view = str(model_ref.get("view") or row.get("source_view") or row.get("view") or "unknown")
            model_paths_by_view[view] = model_ref.get("path")
        if row.get("frame_width") or row.get("frame_height"):
            resolution_by_view[view] = {
                "frame_width": row.get("frame_width"),
                "frame_height": row.get("frame_height"),
            }
        if isinstance(row.get("resolution_profile"), Mapping):
            resolution_profile_by_view[view] = dict(row.get("resolution_profile") or {})
        if row.get("yolo_imgsz") is not None:
            yolo_imgsz_by_view[view] = row.get("yolo_imgsz")
        if row.get("sample_fps") is not None:
            sample_fps_by_view[view] = row.get("sample_fps")
    proposal_counts = Counter(str(item.get("event_type") or "unknown") for item in raw_proposals)
    gate_counts_by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in gate_decisions:
        gate_counts_by_type[str(decision.get("event_type") or "unknown")][str(decision.get("status") or "uncertain")] += 1
    first_zero = _first_zero_stage(
        frames_sampled=len(yolo_rows),
        detections=sum(labels.values()),
        tracklets=len(raw_micro_rows),
        relations=interactions,
        raw_proposals=len(raw_proposals),
        gate_decisions=len(gate_decisions),
    )
    diagnosis = _diagnosis(first_zero, labels, yolo_rows, raw_micro_rows, raw_proposals)
    summary = {
        "schema": "event_candidate_trace_summary.v2.2",
        "pipeline": "key_action_indexer",
        "output_dir": str(output_dir),
        "video_id": str(pipeline_summary.get("session_id") or ""),
        "view": "dual_view" if len(views) > 1 else (next(iter(views)) if views else "unknown"),
        "frames_total": None,
        "frames_sampled": len(yolo_rows),
        "yolo_frames_with_detections": frames_with_detections,
        "detections_by_label": dict(labels.most_common(50)),
        "tracklets_by_label": {},
        "relations_by_type": {"hand_object_interaction_rows": interactions},
        "raw_proposals_by_type": dict(proposal_counts),
        "gate_decisions_by_type": {key: dict(value) for key, value in sorted(gate_counts_by_type.items())},
        "micro_segments_raw": len(raw_micro_rows),
        "micro_segments": len(micro_rows),
        "detected_segments": len(detected_segments),
        "first_zero_stage": first_zero,
        "zero_candidate_diagnosis": diagnosis,
        "runtime": {
            "detector_backend": (pipeline_summary.get("detector_summary") or {}).get("detector_backend") if isinstance(pipeline_summary.get("detector_summary"), Mapping) else None,
            "model_paths_by_view": model_paths_by_view or ((pipeline_summary.get("detector_summary") or {}).get("model_paths_by_view") if isinstance(pipeline_summary.get("detector_summary"), Mapping) else {}),
            "scan_both_views": (pipeline_summary.get("detector_summary") or {}).get("scan_both_views") if isinstance(pipeline_summary.get("detector_summary"), Mapping) else None,
            "resolution_by_view": resolution_by_view,
            "resolution_profile_by_view": resolution_profile_by_view,
            "yolo_imgsz_by_view": yolo_imgsz_by_view,
            "sample_fps_by_view": sample_fps_by_view,
            "small_object_resolution_warning": any(
                int((item or {}).get("frame_width") or 0) < 960 or int((item or {}).get("frame_height") or 0) < 540
                for item in resolution_by_view.values()
            ),
        },
    }
    rows = [
        _trace_row(summary, "frame_sampling", {"frames_sampled": len(yolo_rows)}, {}),
        _trace_row(summary, "yolo_detection", _detection_counts(labels), dict(labels.most_common(20))),
        _trace_row(summary, "tracking", {"tracklets": len(raw_micro_rows), "hand_tracklets": 0, "object_tracklets": len(raw_micro_rows)}, {}),
        _trace_row(summary, "relation", {"relations": interactions}, {"hand_object_interactions": interactions}),
        _trace_row(summary, "proposal", {"raw_proposals": len(raw_proposals)}, dict(proposal_counts)),
        _trace_row(summary, "gate", _gate_counts(gate_decisions), {}),
    ]
    return rows, summary


def _trace_row(summary: Mapping[str, Any], stage: str, counts: Mapping[str, Any], top_labels: Mapping[str, Any]) -> dict[str, Any]:
    base_counts = {
        "frames_sampled": 0,
        "detections": 0,
        "hand_detections": 0,
        "object_detections": 0,
        "container_detections": 0,
        "device_detections": 0,
        "liquid_related_detections": 0,
        "tracklets": 0,
        "hand_tracklets": 0,
        "object_tracklets": 0,
        "relations": 0,
        "raw_proposals": 0,
        "gate_confirmed": 0,
        "gate_candidate": 0,
        "gate_rejected": 0,
        "gate_uncertain": 0,
    }
    for key, value in counts.items():
        if key in base_counts:
            base_counts[key] = int(value or 0)
    return {
        "video_id": summary.get("video_id"),
        "view": summary.get("view"),
        "window_id": "all",
        "time_start": 0.0,
        "time_end": None,
        "stage": stage,
        "counts": base_counts,
        "top_labels": dict(top_labels),
        "drop_reasons": summary.get("zero_candidate_diagnosis") if summary.get("first_zero_stage") == stage else [],
        "notes": "key_action_indexer dual-view trace",
    }


def _event_type_from_micro(row: Mapping[str, Any]) -> str:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), Mapping) else {}
    action_type = _norm((row.get("text_description") or {}).get("action_type") if isinstance(row.get("text_description"), Mapping) else "")
    interaction_type = _norm(interaction.get("interaction_type") or "")
    primary = _norm(interaction.get("primary_object") or "")
    if "liquid" in action_type or "pipette" in primary:
        return "liquid_transfer"
    if primary in DEVICE_LABELS or "panel" in interaction_type:
        return "hand_object_interaction"
    if primary in CONTAINER_LABELS and ("state" in action_type or "open" in action_type or "close" in action_type):
        return "container_state_change"
    return "hand_object_interaction"


def _proposal_object_labels(proposal: Mapping[str, Any]) -> list[str]:
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), Mapping) else {}
    labels = []
    if evidence.get("object_label"):
        labels.append(str(evidence.get("object_label")))
    interaction = evidence.get("interaction") if isinstance(evidence.get("interaction"), Mapping) else {}
    if interaction.get("primary_object"):
        labels.append(str(interaction.get("primary_object")))
    for det in evidence.get("detections") or []:
        if isinstance(det, Mapping):
            label = _norm(det.get("label") or det.get("raw_label"))
            if label and label not in HAND_LABELS:
                labels.append(label)
    return list(dict.fromkeys(label for label in labels if label))


def _detection_counts(labels: Counter[str]) -> dict[str, int]:
    return {
        "detections": sum(labels.values()),
        "hand_detections": sum(labels[label] for label in HAND_LABELS),
        "object_detections": sum(labels[label] for label in OBJECT_LABELS),
        "container_detections": sum(labels[label] for label in CONTAINER_LABELS),
        "device_detections": sum(labels[label] for label in DEVICE_LABELS),
        "liquid_related_detections": sum(labels[label] for label in LIQUID_RELATED_LABELS),
    }


def _gate_counts(gate_decisions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("status") or "uncertain") for item in gate_decisions)
    return {
        "gate_confirmed": counts.get("confirmed", 0),
        "gate_candidate": counts.get("candidate", 0),
        "gate_rejected": counts.get("rejected", 0),
        "gate_uncertain": counts.get("uncertain", 0),
    }


def _first_zero_stage(*, frames_sampled: int, detections: int, tracklets: int, relations: int, raw_proposals: int, gate_decisions: int) -> str | None:
    if frames_sampled <= 0:
        return "frame_sampling"
    if detections <= 0:
        return "yolo_detection"
    if tracklets <= 0:
        return "tracking"
    if relations <= 0:
        return "relation"
    if raw_proposals <= 0:
        return "proposal"
    if gate_decisions <= 0:
        return "gate"
    return None


def _diagnosis(first_zero: str | None, labels: Counter[str], yolo_rows: Sequence[Mapping[str, Any]], raw_micro_rows: Sequence[Mapping[str, Any]], raw_proposals: Sequence[Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if first_zero:
        reasons.append(f"first_zero_stage:{first_zero}")
    if not yolo_rows:
        reasons.append("no_frame_sampling_or_yolo_rows")
    if not labels:
        reasons.append("no_yolo_detections")
    if sum(labels[label] for label in HAND_LABELS) <= 0:
        reasons.append("no_hand_detections")
    if sum(labels[label] for label in OBJECT_LABELS) <= 0:
        reasons.append("no_interaction_object_detections")
    if not raw_micro_rows:
        reasons.append("micro_segments_raw_empty")
    if not raw_proposals:
        reasons.append("no_raw_proposals")
    model_paths = [
        ((row.get("model_ref") or {}).get("path") if isinstance(row.get("model_ref"), Mapping) else None)
        for row in yolo_rows[:10]
    ]
    if not any(model_paths):
        reasons.append("missing_yolo_model_ref_in_rows")
    return sorted(set(reasons))


def _read_first_existing(*paths: Path) -> list[dict[str, Any]]:
    for path in paths:
        if path.exists():
            try:
                return read_jsonl(path)
            except Exception:
                return []
    return []


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _time_from_iso_or_fallback(_iso_value: Any, fallback: Any = None) -> float:
    # Existing key-action metadata carries local seconds elsewhere; for ISO-only
    # advanced evidence we keep a zero-based placeholder rather than fabricating
    # absolute experiment time.
    return _float(fallback, 0.0)
