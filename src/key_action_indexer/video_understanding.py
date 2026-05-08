from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .model_observations import load_or_build_model_observation_events
from .schemas import read_jsonl, write_jsonl


EVENT_FIELDS = (
    "video_event_id",
    "session_id",
    "segment_id",
    "micro_segment_id",
    "event_type",
    "global_start_time",
    "global_end_time",
    "primary_object",
    "action_type",
    "state_change_types",
    "confidence",
    "confidence_reasons",
    "anomaly_flags",
    "asset_refs",
    "evidence_refs",
    "conclusion_status",
    "normalized_object",
    "object_category",
    "action_classification",
    "semantic_description",
    "extracted_entities",
    "text",
    "payload",
)

EVIDENCE_SCORES = {
    "visual_and_transcript_confirmed": 0.9,
    "visual_confirmed": 0.78,
    "transcript_supported": 0.55,
    "weak_visual_evidence": 0.38,
    "insufficient_evidence": 0.18,
}

OBJECT_CATEGORY_KEYWORDS = {
    "hand": ("hand", "gloved_hand"),
    "container": ("tube", "vial", "bottle", "beaker", "flask", "container", "jar", "sample_bottle", "reagent_bottle"),
    "closure": ("cap", "lid", "stopper", "tube_cap", "bottle_cap"),
    "liquid_or_reagent": ("liquid", "reagent", "solution", "buffer", "water", "sample"),
    "tool": ("pipette", "tip", "spatula", "forceps", "dropper"),
    "equipment": ("balance", "scale", "centrifuge", "vortex", "incubator", "heater", "plate", "mixer"),
    "equipment_control": ("panel", "display", "screen", "button", "knob", "switch", "readout"),
}

ACTION_FAMILIES = {
    "liquid_transfer": ("pipetting", "transfer", "liquid_flow", "liquid_level", "sample_adding", "add"),
    "weighing_or_readout": ("weigh", "balance", "readout", "recording", "panel", "equipment_panel"),
    "container_state": ("container", "open", "close", "cap", "lid", "color_change", "state_change"),
    "object_tracking": ("movement", "trajectory", "track", "contact"),
    "sample_handling": ("sample", "handling", "mix", "shake", "vortex"),
}

CONFIRMED_EVENT_TYPES = {
    "container_color_change_detected",
    "container_state_change_detected",
    "equipment_panel_operation_detected",
    "liquid_flow_detected",
    "liquid_level_change_detected",
    "object_movement_detected",
    "object_track_observed",
}

PHYSICAL_EVENT_REVIEW_THRESHOLD = 0.65
CANDIDATE_ROLLUP_MAX_GAP_SEC = 4.0
CROSS_FAMILY_ROLLUP_EVENT_FAMILIES = {
    "container_color",
    "container_state",
    "equipment_panel",
    "liquid_flow",
    "liquid_level",
    "object_movement",
}


def build_video_understanding(session_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    micro_rows = _read_jsonl_if_exists(metadata / "micro_segments.jsonl")
    state_rows = _read_jsonl_if_exists(metadata / "state_change_index.jsonl")
    asset_rows = _read_jsonl_if_exists(metadata / "material_asset_catalog.jsonl")
    advanced_rows = _read_jsonl_if_exists(metadata / "advanced_vision_evidence.jsonl")
    object_track_rows = _read_jsonl_if_exists(metadata / "object_tracks.jsonl")
    model_observation_rows, model_observation_summary = load_or_build_model_observation_events(session)
    target = Path(output_path) if output_path is not None else metadata / "video_understanding.jsonl"
    summary_path = target.with_name("video_understanding_summary.json")

    states_by_micro: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        micro_id = str(row.get("micro_segment_id") or "")
        if micro_id:
            states_by_micro[micro_id].append(row)

    assets_by_micro: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in asset_rows:
        micro_id = str(row.get("micro_segment_id") or row.get("source_id") or "")
        if micro_id:
            assets_by_micro[micro_id].append(row)

    events: list[dict[str, Any]] = []
    for index, micro in enumerate(micro_rows, start=1):
        if not isinstance(micro, Mapping):
            continue
        events.extend(_events_for_micro(micro, states_by_micro.get(str(micro.get("micro_segment_id") or ""), []), assets_by_micro, index))
    events.extend(_events_from_advanced_evidence(advanced_rows))
    events.extend(_events_from_model_observations(model_observation_rows, skip_observation_ids=_advanced_model_observation_ids(advanced_rows)))
    events, candidate_rollup_summary = _finalize_events(events)

    events.sort(key=lambda item: (str(item.get("global_start_time") or ""), str(item.get("video_event_id") or "")))
    write_jsonl(target, [_stable_event(row) for row in events])

    counts = Counter(str(row.get("event_type") or "unknown") for row in events)
    anomaly_counts = Counter(flag for row in events for flag in row.get("anomaly_flags", []))
    status_counts = Counter(str(row.get("conclusion_status") or "unknown") for row in events)
    object_counts = Counter(str(_as_dict(row.get("normalized_object")).get("canonical_label") or row.get("primary_object") or "unknown") for row in events)
    session_id = _first_text(
        [
            *(row.get("session_id") for row in events if isinstance(row, Mapping)),
            *(row.get("session_id") for row in model_observation_rows if isinstance(row, Mapping)),
            *(row.get("session_id") for row in micro_rows if isinstance(row, Mapping)),
            _manifest_session_id(session),
        ]
    )
    summary = {
        "session_id": session_id,
        "video_event_count": len(events),
        "event_type_counts": dict(sorted(counts.items())),
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "input_counts": {
            "micro_segments": len(micro_rows),
            "state_changes": len(state_rows),
            "material_assets": len(asset_rows),
            "advanced_vision_evidence": len(advanced_rows),
            "object_tracks": len(object_track_rows),
            "model_observation_events": len(model_observation_rows),
            "model_observation_inputs": model_observation_summary.get("input_counts", {}),
        },
        "conclusion_status_counts": dict(sorted(status_counts.items())),
        "normalized_object_counts": dict(sorted(object_counts.items())),
        "human_review_candidate_count": sum(1 for row in events if _requires_review(row)),
        "candidate_rollup": candidate_rollup_summary,
        "video_understanding": str(target),
        "summary_path": str(summary_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_video_understanding(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def _events_for_micro(
    micro: Mapping[str, Any],
    state_rows: list[dict[str, Any]],
    assets_by_micro: Mapping[str, list[dict[str, Any]]],
    row_index: int,
) -> list[dict[str, Any]]:
    interaction = _as_dict(micro.get("interaction"))
    text_description = _as_dict(micro.get("text_description"))
    evidence = _as_dict(micro.get("evidence"))
    micro_id = str(micro.get("micro_segment_id") or f"micro_{row_index:06d}")
    state_types = _dedupe([str(row.get("state_type")) for row in state_rows if row.get("state_type")])
    asset_refs = _asset_refs(micro, state_rows, assets_by_micro.get(micro_id, []))
    primary_object = str(interaction.get("primary_object") or "")
    action_type = str(text_description.get("action_type") or micro.get("action_type") or "")
    base = {
        "session_id": micro.get("session_id"),
        "segment_id": micro.get("parent_segment_id") or micro.get("segment_id"),
        "micro_segment_id": micro_id,
        "global_start_time": micro.get("global_start_time"),
        "global_end_time": micro.get("global_end_time"),
        "primary_object": primary_object,
        "action_type": action_type,
        "state_change_types": state_types,
        "asset_refs": asset_refs,
        "payload": {"source": "micro_segment", "micro_segment": dict(micro)},
    }
    confidence, confidence_reasons = _confidence(micro, state_rows, asset_refs)
    anomaly_flags = _anomalies(micro, state_rows, asset_refs, confidence)
    text = _join_text(text_description.get("summary"), text_description.get("index_text"))
    if evidence.get("segment_level_coverage_backfill") or evidence.get("force_retrieval_candidate"):
        event_type = "segment_level_retrieval_candidate"
        return [
            {
                **base,
                "video_event_id": f"{micro_id}:{event_type}",
                "event_type": event_type,
                "confidence": round(max(0.0, min(0.45, confidence)), 4),
                "confidence_reasons": _dedupe([*confidence_reasons, "segment-level retrieval backfill compressed to one candidate event"]),
                "anomaly_flags": _dedupe([*anomaly_flags, "segment_level_retrieval_backfill", "requires_human_confirmation"]),
                "text": _event_text(event_type, primary_object, action_type, text),
                "payload": {**base["payload"], "event_type": event_type, "evidence": evidence},
            }
        ]

    event_types = ["hand_object_contact", "object_state_change", "experiment_action_classification"]
    if _movement_candidate(interaction, state_types):
        event_types.append("object_movement_candidate")
    if _liquid_transfer_candidate(primary_object, action_type, text):
        event_types.append("liquid_transfer_candidate")
    if _equipment_panel_candidate(primary_object, action_type, text):
        event_types.append("equipment_panel_operation_candidate")
    if _container_state_candidate(primary_object, state_types, text):
        event_types.append("container_state_change_candidate")

    events = []
    for event_type in event_types:
        event_confidence = confidence
        flags = list(anomaly_flags)
        reasons = list(confidence_reasons)
        if event_type.endswith("_candidate"):
            event_confidence = min(event_confidence, 0.62)
            flags.append("heuristic_candidate")
            reasons.append("candidate generated from object/action/state heuristics")
        if event_type == "liquid_transfer_candidate":
            flags.append("not_visual_liquid_flow_confirmed")
        if event_type == "equipment_panel_operation_candidate":
            flags.append("not_panel_ocr_confirmed")
        if event_type == "container_state_change_candidate":
            flags.append("not_container_open_close_confirmed")
        events.append(
            {
                **base,
                "video_event_id": f"{micro_id}:{event_type}",
                "event_type": event_type,
                "confidence": round(max(0.0, min(1.0, event_confidence)), 4),
                "confidence_reasons": _dedupe(reasons),
                "anomaly_flags": _dedupe(flags),
                "text": _event_text(event_type, primary_object, action_type, text),
                "payload": {**base["payload"], "event_type": event_type, "evidence": evidence},
            }
        )
    return events


def _events_from_advanced_evidence(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for row in rows:
        evidence_type = str(row.get("evidence_type") or "")
        level = str(row.get("visual_confirmation_level") or "")
        event_type = _advanced_event_type(evidence_type, level)
        if not event_type:
            continue
        confidence = float(row.get("confidence") or 0.0)
        limitations = _strings(row.get("limitations"))
        flags = []
        if level.startswith("candidate"):
            flags.append("advanced_evidence_candidate")
        if row.get("requires_human_confirmation"):
            flags.append("requires_human_confirmation")
        if any("not confirm" in item.lower() or "requires" in item.lower() for item in limitations):
            flags.append("visual_confirmation_limited")
        events.append(
            {
                "video_event_id": f"{row.get('evidence_id')}:{event_type}",
                "session_id": row.get("session_id"),
                "segment_id": row.get("segment_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "event_type": event_type,
                "global_start_time": row.get("global_start_time"),
                "global_end_time": row.get("global_end_time"),
                "primary_object": row.get("object_label"),
                "action_type": row.get("action_type"),
                "state_change_types": [],
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "confidence_reasons": _dedupe(_strings(row.get("evidence_reasons")) + [f"advanced_vision_level={level}"]),
                "anomaly_flags": _dedupe(flags),
                "asset_refs": row.get("asset_refs") or [],
                "text": _advanced_text(event_type, row),
                "payload": {"source": "advanced_vision_evidence", "advanced_evidence": dict(row)},
            }
        )
    return events


def _events_from_model_observations(rows: list[Mapping[str, Any]], skip_observation_ids: set[str] | None = None) -> list[dict[str, Any]]:
    skipped = skip_observation_ids or set()
    events = []
    for row in rows:
        observation_id = str(row.get("observation_id") or "")
        if observation_id in skipped:
            continue
        event_type = _model_observation_event_type(row)
        if not event_type:
            continue
        confidence = _as_float(row.get("confidence"))
        if confidence is None:
            confidence = 0.75
        confirmation_level = str(row.get("confirmation_level") or "")
        flags = []
        if "candidate" in confirmation_level.lower():
            flags.append("model_observation_candidate")
        if confidence < 0.5:
            flags.append("low_confidence_model_observation")
        events.append(
            {
                "video_event_id": f"{observation_id}:{event_type}",
                "session_id": row.get("session_id"),
                "segment_id": row.get("segment_id"),
                "micro_segment_id": row.get("micro_segment_id"),
                "event_type": event_type,
                "global_start_time": row.get("global_start_time"),
                "global_end_time": row.get("global_end_time"),
                "primary_object": row.get("object_label"),
                "action_type": _model_observation_action_type(row),
                "state_change_types": _model_state_change_types(row),
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "confidence_reasons": _dedupe(
                    [
                        *_strings(row.get("evidence_reasons")),
                        f"model_observation_level={confirmation_level or 'unknown'}",
                        f"model_observation_event={observation_id}",
                    ]
                ),
                "anomaly_flags": _dedupe(flags),
                "asset_refs": row.get("asset_refs") or [],
                "text": _model_observation_text(event_type, row),
                "payload": {"source": "model_observation_events", "model_observation": dict(row)},
            }
        )
    return events


def _finalize_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finalized = [_gate_and_enrich_event(dict(row)) for row in events]
    finalized, rollup_summary = _rollup_candidate_events(finalized)
    _annotate_conflicts(finalized)
    return finalized, rollup_summary


def _gate_and_enrich_event(row: dict[str, Any]) -> dict[str, Any]:
    flags = _strings(row.get("anomaly_flags"))
    reasons = _strings(row.get("confidence_reasons"))
    confidence = _as_float(row.get("confidence"))
    confidence = 0.0 if confidence is None else max(0.0, min(1.0, confidence))
    status = _conclusion_status(row)

    if status in {"confirmed", "measured"} and not _has_model_or_visual_support(row):
        flags.extend(["confirmed_without_model_or_visual_evidence", "requires_human_confirmation"])
        reasons.append("downgraded_to_candidate_without_model_or_visual_evidence")
        status = "candidate"
        row["event_type"] = _candidate_event_type(str(row.get("event_type") or ""))
        confidence = min(confidence, 0.62)

    if status in {"confirmed", "measured"} and confidence < PHYSICAL_EVENT_REVIEW_THRESHOLD:
        flags.extend(["low_confidence_confirmed_or_measured_event", "requires_human_confirmation"])
    elif status == "candidate":
        flags.append("requires_human_confirmation")
        if confidence < 0.5:
            flags.append("low_confidence_candidate_event")

    normalized = _normalized_object(row)
    action = _action_classification(row, normalized)
    row["confidence"] = round(confidence, 4)
    row["confidence_reasons"] = _dedupe(reasons)
    row["anomaly_flags"] = _dedupe(flags)
    row["conclusion_status"] = status
    row["normalized_object"] = normalized
    row["object_category"] = normalized.get("category")
    row["action_classification"] = action
    row["semantic_description"] = _semantic_description(row, normalized, action)
    row["extracted_entities"] = _event_entities(row, normalized, action)
    row["evidence_refs"] = _event_evidence_refs(row)
    return row


def _annotate_conflicts(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        family = _event_family(str(row.get("event_type") or ""))
        if family not in {"container_state", "container_color", "liquid_level", "equipment_panel"}:
            continue
        state = _state_signature(row)
        if not state:
            continue
        normalized = _as_dict(row.get("normalized_object"))
        key = (
            str(row.get("micro_segment_id") or row.get("segment_id") or ""),
            str(normalized.get("canonical_label") or row.get("primary_object") or ""),
            family,
        )
        grouped[key][state].append(row)
    for states in grouped.values():
        if len(states) <= 1:
            continue
        state_names = sorted(states)
        for rows_for_state in states.values():
            for row in rows_for_state:
                flags = _strings(row.get("anomaly_flags"))
                flags.extend(["conflicting_physical_event_evidence", "requires_human_confirmation"])
                row["anomaly_flags"] = _dedupe(flags)
                payload = _as_dict(row.get("payload"))
                payload["conflicting_state_signatures"] = state_names[:8]
                row["payload"] = payload


def _rollup_candidate_events(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_rows = [row for row in rows if _is_rollup_candidate(row)]
    passthrough = [row for row in rows if not _is_rollup_candidate(row)]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        grouped[_candidate_rollup_key(row)].append(row)

    rolled: list[dict[str, Any]] = []
    primary_rollup_groups: list[dict[str, Any]] = []
    group_index = 0
    for key in sorted(grouped):
        chunks = _candidate_rollup_chunks(grouped[key])
        for chunk in chunks:
            if len(chunk) == 1:
                rolled.append(chunk[0])
                continue
            group_index += 1
            merged = _merge_candidate_chunk(chunk, group_index)
            rolled.append(merged)
            primary_rollup_groups.append(_candidate_rollup_group_summary(merged, chunk, "event_family"))

    rolled, cross_family_rollup_groups, group_index = _rollup_same_micro_cross_family_candidates(rolled, group_index)
    rolled, weak_bundle_rollup_groups, group_index = _rollup_same_micro_weak_candidate_bundles(rolled, group_index)

    output = [*passthrough, *rolled]
    input_candidate_count = len(candidate_rows)
    output_candidate_count = sum(1 for row in output if str(row.get("conclusion_status") or "") == "candidate")
    rollup_groups = [*primary_rollup_groups, *cross_family_rollup_groups, *weak_bundle_rollup_groups]
    summary = {
        "enabled": True,
        "max_gap_sec": CANDIDATE_ROLLUP_MAX_GAP_SEC,
        "input_event_count": len(rows),
        "output_event_count": len(output),
        "input_candidate_count": input_candidate_count,
        "output_candidate_count": output_candidate_count,
        "input_candidate_ratio": round(input_candidate_count / len(rows), 6) if rows else 0.0,
        "output_candidate_ratio": round(output_candidate_count / len(output), 6) if output else 0.0,
        "rollup_group_count": len(rollup_groups),
        "primary_group_count": len(primary_rollup_groups),
        "cross_family_group_count": len(cross_family_rollup_groups),
        "weak_bundle_group_count": len(weak_bundle_rollup_groups),
        "rolled_source_event_count": sum(group["source_event_count"] for group in rollup_groups),
        "removed_candidate_event_count": max(0, input_candidate_count - output_candidate_count),
        "groups": rollup_groups[:50],
        "primary_groups": primary_rollup_groups[:50],
        "cross_family_groups": cross_family_rollup_groups[:50],
        "weak_bundle_groups": weak_bundle_rollup_groups[:50],
    }
    return output, summary


def _is_rollup_candidate(row: Mapping[str, Any]) -> bool:
    if str(row.get("conclusion_status") or "") != "candidate":
        return False
    event_type = str(row.get("event_type") or "")
    if not event_type:
        return False
    return bool(row.get("video_event_id") or row.get("evidence_refs") or row.get("asset_refs"))


def _candidate_rollup_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    normalized = _as_dict(row.get("normalized_object"))
    action = _as_dict(row.get("action_classification"))
    return (
        str(row.get("segment_id") or row.get("micro_segment_id") or ""),
        str(normalized.get("canonical_label") or row.get("primary_object") or "unknown_object"),
        str(action.get("family") or _event_family(str(row.get("event_type") or "")) or "unknown_action"),
        _event_family(str(row.get("event_type") or "")),
    )


def _rollup_same_micro_cross_family_candidates(
    rows: list[dict[str, Any]],
    group_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        key = _same_micro_cross_family_rollup_key(row)
        if key is None:
            passthrough.append(row)
            continue
        grouped[key].append(row)

    rolled: list[dict[str, Any]] = []
    rollup_groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        chunks = _candidate_rollup_chunks(grouped[key])
        for chunk in chunks:
            event_families = {_event_family(str(row.get("event_type") or "")) for row in chunk}
            if len(chunk) == 1 or len(event_families) < 2:
                rolled.extend(chunk)
                continue
            group_index += 1
            merged = _merge_candidate_chunk(chunk, group_index, rollup_mode="same_micro_cross_family")
            rolled.append(merged)
            rollup_groups.append(_candidate_rollup_group_summary(merged, chunk, "same_micro_cross_family"))

    return [*passthrough, *rolled], rollup_groups, group_index


def _same_micro_cross_family_rollup_key(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if not _is_rollup_candidate(row):
        return None
    if _event_family(str(row.get("event_type") or "")) not in CROSS_FAMILY_ROLLUP_EVENT_FAMILIES:
        return None
    micro_id = _single_micro_rollup_scope(row)
    if not micro_id:
        return None
    normalized = _as_dict(row.get("normalized_object"))
    action = _as_dict(row.get("action_classification"))
    action_family = str(action.get("family") or "")
    if not action_family or action_family == "other_experiment_action":
        return None
    canonical = str(normalized.get("canonical_label") or row.get("primary_object") or "")
    if not canonical or canonical == "unknown_object":
        return None
    return (micro_id, canonical, action_family)


def _rollup_same_micro_weak_candidate_bundles(
    rows: list[dict[str, Any]],
    group_index: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    passthrough: list[dict[str, Any]] = []
    for row in rows:
        key = _same_micro_weak_bundle_rollup_key(row)
        if key is None:
            passthrough.append(row)
            continue
        grouped[key].append(row)

    rolled: list[dict[str, Any]] = []
    rollup_groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        chunks = _candidate_rollup_chunks(grouped[key])
        for chunk in chunks:
            action_families = {
                str(_as_dict(row.get("action_classification")).get("family") or "")
                for row in chunk
            }
            event_families = {_event_family(str(row.get("event_type") or "")) for row in chunk}
            if len(chunk) == 1 or len(action_families) < 2 or len(event_families) < 2 or not _weak_candidate_bundle(chunk):
                rolled.extend(chunk)
                continue
            group_index += 1
            merged = _merge_candidate_chunk(chunk, group_index, rollup_mode="same_micro_weak_candidate_bundle")
            rolled.append(merged)
            rollup_groups.append(_candidate_rollup_group_summary(merged, chunk, "same_micro_weak_candidate_bundle"))

    return [*passthrough, *rolled], rollup_groups, group_index


def _same_micro_weak_bundle_rollup_key(row: Mapping[str, Any]) -> tuple[str, str] | None:
    if not _is_rollup_candidate(row):
        return None
    if _event_family(str(row.get("event_type") or "")) not in CROSS_FAMILY_ROLLUP_EVENT_FAMILIES:
        return None
    micro_id = _single_micro_rollup_scope(row)
    if not micro_id:
        return None
    normalized = _as_dict(row.get("normalized_object"))
    canonical = str(normalized.get("canonical_label") or row.get("primary_object") or "")
    if not canonical or canonical == "unknown_object":
        return None
    return (micro_id, canonical)


def _weak_candidate_bundle(rows: list[Mapping[str, Any]]) -> bool:
    for row in rows:
        flags = set(_strings(row.get("anomaly_flags")))
        if "requires_human_confirmation" not in flags:
            return False
        if not flags.intersection(
            {
                "advanced_evidence_candidate",
                "candidate_cross_family_rollup",
                "candidate_rollup",
                "evidence_limitation_missing_visual_or_transcript",
                "heuristic_candidate",
                "low_confidence_candidate_event",
                "visual_confirmation_limited",
            }
        ):
            return False
    return True


def _single_micro_rollup_scope(row: Mapping[str, Any]) -> str:
    micro_id = str(row.get("micro_segment_id") or "")
    if not micro_id:
        return ""
    payload = _as_dict(row.get("payload"))
    rollup = _as_dict(payload.get("rollup"))
    source_micro_ids = _dedupe(_strings(rollup.get("source_micro_segment_ids")))
    if source_micro_ids and source_micro_ids != [micro_id]:
        return ""
    return micro_id


def _candidate_rollup_chunks(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: (_event_start_sec(row), str(row.get("video_event_id") or "")))
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_end: float | None = None
    for row in ordered:
        start = _event_start_sec(row)
        end = _event_end_sec(row)
        if not current:
            current = [row]
            current_end = end if end is not None else start
            continue
        if _candidate_adjacent(start, current_end):
            current.append(row)
            current_end = _max_optional(current_end, end if end is not None else start)
            continue
        chunks.append(current)
        current = [row]
        current_end = end if end is not None else start
    if current:
        chunks.append(current)
    return chunks


def _merge_candidate_chunk(rows: list[dict[str, Any]], group_index: int, rollup_mode: str = "event_family") -> dict[str, Any]:
    representative = max(rows, key=lambda row: (float(row.get("confidence") or 0.0), -len(str(row.get("video_event_id") or ""))))
    source_event_ids = _dedupe(str(row.get("video_event_id") or "") for row in rows if row.get("video_event_id"))
    source_micro_ids = _dedupe(str(row.get("micro_segment_id") or "") for row in rows if row.get("micro_segment_id"))
    source_segment_ids = _dedupe(str(row.get("segment_id") or "") for row in rows if row.get("segment_id"))
    asset_refs = _dedupe_refs(ref for row in rows for ref in _mapping_refs(row.get("asset_refs")))
    source_evidence_refs = _dedupe_refs(ref for row in rows for ref in _mapping_refs(row.get("evidence_refs")))
    event_refs = [
        {
            "type": "video_event",
            "video_event_id": event_id,
            "source_event_id": event_id,
            "evidence_level": "candidate_rollup_source",
        }
        for event_id in source_event_ids
    ]
    confidence_values = [float(row.get("confidence") or 0.0) for row in rows]
    start = _min_text(row.get("global_start_time") for row in rows)
    end = _max_text(row.get("global_end_time") or row.get("global_start_time") for row in rows)
    normalized = _as_dict(representative.get("normalized_object"))
    action = _as_dict(representative.get("action_classification"))
    scope = str(representative.get("segment_id") or representative.get("micro_segment_id") or "session")
    event_family = _event_family(str(representative.get("event_type") or "candidate"))
    canonical = str(normalized.get("canonical_label") or representative.get("primary_object") or "object")
    action_family = str(action.get("family") or "experiment_action")
    rollup_id = f"candidate_rollup_{group_index:04d}:{_safe_id(scope)}:{_safe_id(canonical)}:{_safe_id(action_family)}:{_safe_id(event_family)}"
    flags = [*(flag for row in rows for flag in _strings(row.get("anomaly_flags"))), "candidate_rollup"]
    if rollup_mode == "same_micro_cross_family":
        flags.append("candidate_cross_family_rollup")
    if rollup_mode == "same_micro_weak_candidate_bundle":
        flags.append("candidate_weak_bundle_rollup")
    row = {
        "video_event_id": rollup_id,
        "session_id": _first_text(row.get("session_id") for row in rows),
        "segment_id": source_segment_ids[0] if len(source_segment_ids) == 1 else representative.get("segment_id"),
        "micro_segment_id": source_micro_ids[0] if len(source_micro_ids) == 1 else representative.get("micro_segment_id"),
        "event_type": representative.get("event_type"),
        "global_start_time": start,
        "global_end_time": end,
        "primary_object": representative.get("primary_object"),
        "action_type": representative.get("action_type"),
        "state_change_types": _dedupe(value for row in rows for value in _strings(row.get("state_change_types"))),
        "confidence": round(max(confidence_values) if confidence_values else 0.0, 4),
        "confidence_reasons": _dedupe(
            [
                *(reason for row in rows for reason in _strings(row.get("confidence_reasons"))),
                f"candidate_rollup_sources={len(source_event_ids)}",
            ]
        ),
        "anomaly_flags": _dedupe(flags),
        "asset_refs": asset_refs,
        "evidence_refs": _dedupe_refs([*source_evidence_refs, *event_refs]),
        "conclusion_status": "candidate",
        "normalized_object": normalized,
        "object_category": representative.get("object_category"),
        "action_classification": action,
        "extracted_entities": _merge_entities(rows),
        "text": _candidate_rollup_text(representative, rows, rollup_mode),
        "payload": {
            "source": "candidate_rollup",
            "rollup": {
                "rollup_mode": rollup_mode,
                "source_event_ids": source_event_ids,
                "source_event_types": dict(sorted(Counter(str(row.get("event_type") or "unknown") for row in rows).items())),
                "source_event_families": dict(sorted(Counter(_event_family(str(row.get("event_type") or "")) for row in rows).items())),
                "source_action_families": dict(
                    sorted(
                        Counter(
                            str(_as_dict(row.get("action_classification")).get("family") or "unknown_action")
                            for row in rows
                        ).items()
                    )
                ),
                "source_micro_segment_ids": source_micro_ids,
                "source_segment_ids": source_segment_ids,
                "source_count": len(rows),
                "max_gap_sec": CANDIDATE_ROLLUP_MAX_GAP_SEC,
            },
        },
    }
    row["semantic_description"] = _semantic_description(row, normalized, action)
    return row


def _candidate_rollup_group_summary(
    merged: Mapping[str, Any],
    chunk: list[Mapping[str, Any]],
    rollup_mode: str,
) -> dict[str, Any]:
    return {
        "rollup_event_id": merged.get("video_event_id"),
        "rollup_mode": rollup_mode,
        "source_event_count": len(chunk),
        "source_event_ids": [str(row.get("video_event_id") or "") for row in chunk if row.get("video_event_id")],
        "source_event_families": dict(sorted(Counter(_event_family(str(row.get("event_type") or "")) for row in chunk).items())),
        "source_action_families": dict(
            sorted(Counter(str(_as_dict(row.get("action_classification")).get("family") or "unknown_action") for row in chunk).items())
        ),
        "normalized_object": _as_dict(merged.get("normalized_object")).get("canonical_label"),
        "action_family": _as_dict(merged.get("action_classification")).get("family"),
        "event_family": _event_family(str(merged.get("event_type") or "")),
        "segment_id": merged.get("segment_id"),
        "micro_segment_id": merged.get("micro_segment_id"),
        "global_start_time": merged.get("global_start_time"),
        "global_end_time": merged.get("global_end_time"),
    }


def _candidate_adjacent(start: float | None, current_end: float | None) -> bool:
    if start is None or current_end is None:
        return True
    return start - current_end <= CANDIDATE_ROLLUP_MAX_GAP_SEC


def _candidate_rollup_text(representative: Mapping[str, Any], rows: list[Mapping[str, Any]], rollup_mode: str = "event_family") -> str:
    obj = str(_as_dict(representative.get("normalized_object")).get("canonical_label") or representative.get("primary_object") or "object")
    event_type = str(representative.get("event_type") or "candidate_event")
    if rollup_mode == "same_micro_cross_family":
        families = ", ".join(sorted({_event_family(str(row.get("event_type") or "")) for row in rows}))
        return f"rolled-up same-micro candidate evidence for {obj} across {families}; {len(rows)} source events require confirmation"
    if rollup_mode == "same_micro_weak_candidate_bundle":
        families = ", ".join(sorted(str(_as_dict(row.get("action_classification")).get("family") or "candidate") for row in rows))
        return f"rolled-up weak same-micro candidate bundle for {obj} across {families}; {len(rows)} source hypotheses require confirmation"
    return f"rolled-up candidate {event_type} for {obj} from {len(rows)} adjacent source events; requires confirmation"


def _merge_entities(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"materials": [], "reagents": [], "equipment": [], "parameters": [], "action_family": None}
    for key in ("materials", "reagents", "equipment"):
        output[key] = _dedupe(
            value
            for row in rows
            for value in _strings(_as_dict(row.get("extracted_entities")).get(key))
        )
    parameter_refs = []
    for row in rows:
        for item in _as_list(_as_dict(row.get("extracted_entities")).get("parameters")):
            if isinstance(item, Mapping):
                parameter_refs.append(dict(item))
    output["parameters"] = _dedupe_param_refs(parameter_refs)
    action_families = _dedupe(
        str(_as_dict(row.get("extracted_entities")).get("action_family") or _as_dict(row.get("action_classification")).get("family") or "")
        for row in rows
    )
    output["action_family"] = action_families[0] if action_families else None
    return output


def _event_start_sec(row: Mapping[str, Any]) -> float | None:
    return _event_time_sec(row.get("global_start_time"))


def _event_end_sec(row: Mapping[str, Any]) -> float | None:
    return _event_time_sec(row.get("global_end_time") or row.get("global_start_time"))


def _event_time_sec(value: Any) -> float | None:
    if value is None or value == "":
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return None


def _mapping_refs(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in _as_list(value) if isinstance(item, Mapping)]


def _dedupe_param_refs(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for ref in refs:
        key = (str(ref.get("name") or ""), str(ref.get("value") or ""), str(ref.get("unit") or ""))
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(ref))
    return output


def _safe_id(value: Any) -> str:
    text = str(value or "unknown").lower().replace(" ", "_").replace("-", "_")
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "unknown"


def _min_text(values: Iterable[Any]) -> Any:
    strings = [str(value) for value in values if value]
    return min(strings) if strings else None


def _max_text(values: Iterable[Any]) -> Any:
    strings = [str(value) for value in values if value]
    return max(strings) if strings else None


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _advanced_model_observation_ids(rows: list[Mapping[str, Any]]) -> set[str]:
    ids = set()
    for row in rows:
        payload = _as_dict(row.get("payload"))
        model_observation = _as_dict(payload.get("model_observation"))
        observation_id = model_observation.get("observation_id")
        if observation_id:
            ids.add(str(observation_id))
    return ids


def _advanced_event_type(evidence_type: str, visual_level: str = "") -> str:
    confirmed_or_measured = _confirmed_or_measured_level(visual_level)
    if evidence_type == "liquid_flow_candidate_visual":
        return "liquid_flow_detected" if confirmed_or_measured else "liquid_flow_candidate_visual"
    if evidence_type == "equipment_control_change":
        return "equipment_panel_operation_detected" if confirmed_or_measured else "equipment_panel_operation_candidate"
    if evidence_type == "container_open_close":
        return "container_state_change_detected" if confirmed_or_measured else "container_state_change_candidate"
    return {
        "object_trajectory_movement": "object_movement_detected",
        "object_track_observation": "object_track_observed",
        "liquid_level_change": "liquid_level_change_detected",
        "equipment_panel_ocr": "equipment_panel_operation_detected",
        "container_color_change": "container_color_change_detected",
    }.get(evidence_type, "")


def _model_observation_event_type(row: Mapping[str, Any]) -> str:
    event_type = str(row.get("event_type") or "")
    observation_type = str(row.get("observation_type") or row.get("source_type") or "")
    confirmation_level = str(row.get("confirmation_level") or "").lower()
    is_candidate = "candidate" in confirmation_level or event_type.endswith("_candidate")
    if event_type == "liquid_level_measured":
        return "liquid_level_change_detected"
    if event_type.startswith("container_color_change"):
        return "container_color_change_detected" if not is_candidate else "container_state_change_candidate"
    if is_candidate and observation_type == "liquid_segmentation":
        return "liquid_transfer_candidate"
    if event_type == "liquid_flow_observed" or observation_type == "liquid_segmentation":
        return "liquid_flow_detected"
    if is_candidate and observation_type == "equipment_panel_state":
        return "equipment_panel_operation_candidate"
    if event_type.startswith("equipment_panel_state") or event_type.startswith("equipment_control_state") or observation_type == "equipment_panel_state":
        return "equipment_panel_operation_detected"
    if is_candidate and observation_type == "container_state":
        return "container_state_change_candidate"
    if event_type == "container_state_confirmed" or event_type.startswith("container_open") or event_type.startswith("container_close") or observation_type == "container_state":
        return "container_state_change_detected"
    if event_type in {"object_movement_measured", "object_track_measured"}:
        return "object_movement_detected"
    if event_type == "object_track_observed" or observation_type == "object_track":
        return "object_track_observed"
    return ""


def _advanced_text(event_type: str, row: Mapping[str, Any]) -> str:
    obj = str(row.get("object_label") or "object")
    if event_type == "object_movement_detected":
        return f"trajectory-confirmed object movement for {obj}"
    if event_type == "object_track_observed":
        return f"model-confirmed object track for {obj}"
    if event_type == "liquid_flow_detected":
        return f"model-confirmed liquid flow evidence for {obj}"
    if event_type == "liquid_level_change_detected":
        return f"classical visual liquid-level change evidence for {obj}"
    if event_type == "container_color_change_detected":
        return f"classical keyframe color-change evidence for {obj}"
    if event_type == "equipment_panel_operation_detected":
        return f"OCR-supported equipment panel evidence for {obj}"
    if event_type == "container_state_change_detected":
        return f"model-confirmed container state evidence for {obj}"
    return f"advanced visual evidence for {obj}: {row.get('evidence_type')}"


def _model_observation_action_type(row: Mapping[str, Any]) -> Any:
    payload = _as_dict(row.get("payload"))
    source_row = _as_dict(payload.get("source_row"))
    return _first_non_empty(row.get("action_type"), source_row.get("action_type"), source_row.get("action"))


def _model_state_change_types(row: Mapping[str, Any]) -> list[str]:
    values = [str(row.get("event_type") or ""), str(row.get("state") or "")]
    source_type = str(row.get("source_type") or "")
    if source_type:
        values.append(source_type)
    return _dedupe(values)


def _model_observation_text(event_type: str, row: Mapping[str, Any]) -> str:
    obj = str(row.get("object_label") or "object")
    state = str(row.get("state") or "").strip()
    measurement = _as_dict(row.get("measurement"))
    confirmation_level = str(row.get("confirmation_level") or "").lower()
    prefix = "model-candidate" if "candidate" in confirmation_level or str(row.get("event_type") or "").endswith("_candidate") else "model-confirmed"
    if event_type == "liquid_transfer_candidate":
        return f"{prefix} liquid-transfer signal for {obj}"
    if event_type == "liquid_flow_detected":
        return f"{prefix} liquid flow for {obj}"
    if event_type == "liquid_level_change_detected":
        keys = ", ".join(sorted(str(key) for key in measurement)[:4])
        return f"model-measured liquid level for {obj}" + (f" ({keys})" if keys else "")
    if event_type == "equipment_panel_operation_detected":
        return f"{prefix} equipment panel state for {obj}" + (f": {state}" if state else "")
    if event_type == "equipment_panel_operation_candidate":
        return f"{prefix} equipment panel operation signal for {obj}" + (f": {state}" if state else "")
    if event_type == "container_state_change_detected":
        return f"{prefix} container state for {obj}" + (f": {state}" if state else "")
    if event_type == "container_state_change_candidate":
        return f"{prefix} container state-change signal for {obj}" + (f": {state}" if state else "")
    if event_type == "object_movement_detected":
        return f"model-measured object trajectory for {obj}"
    if event_type == "object_track_observed":
        return f"model-confirmed object track for {obj}" + (f": {state}" if state else "")
    return f"model observation for {obj}: {row.get('event_type')}"


def _confidence(micro: Mapping[str, Any], state_rows: list[Mapping[str, Any]], asset_refs: list[dict[str, Any]]) -> tuple[float, list[str]]:
    evidence = _as_dict(micro.get("evidence"))
    quality = _as_dict(micro.get("quality"))
    interaction = _as_dict(micro.get("interaction"))
    level = str(evidence.get("evidence_level") or micro.get("evidence_level") or "")
    score = EVIDENCE_SCORES.get(level, 0.45)
    reasons = [f"evidence_level={level or 'unknown'}"]
    if str(quality.get("confidence") or "").lower() == "high":
        score += 0.06
        reasons.append("micro_quality=high")
    max_interaction = _as_float(interaction.get("max_interaction_score"))
    if max_interaction is not None:
        score += min(max_interaction, 1.0) * 0.08
        reasons.append(f"max_interaction_score={max_interaction:.3f}")
    if state_rows:
        score += 0.04
        reasons.append(f"state_changes={len(state_rows)}")
    if any(ref.get("asset_id") for ref in asset_refs):
        score += 0.04
        reasons.append("asset_refs_with_asset_id")
    return max(0.0, min(1.0, score)), reasons


def _anomalies(micro: Mapping[str, Any], state_rows: list[Mapping[str, Any]], asset_refs: list[dict[str, Any]], confidence: float) -> list[str]:
    evidence = _as_dict(micro.get("evidence"))
    level = str(evidence.get("evidence_level") or micro.get("evidence_level") or "")
    limitations = _strings(evidence.get("limitations")) + _strings(micro.get("limitations"))
    flags: list[str] = []
    if level in {"weak_visual_evidence", "insufficient_evidence"} or confidence < 0.45:
        flags.append("low_confidence_evidence")
    if level == "transcript_supported":
        flags.append("transcript_supported_without_strong_visual")
    if any("missing" in item.lower() for item in limitations):
        flags.append("evidence_limitation_missing_visual_or_transcript")
    keyframes = _as_dict(micro.get("keyframes"))
    if not any(keyframes.get(key) for key in ("contact_frame", "peak_frame", "release_frame")):
        flags.append("missing_keyframe")
    if not state_rows:
        flags.append("missing_state_change")
    if any(_as_dict(ref.get("quality")).get("status") == "missing" for ref in asset_refs):
        flags.append("missing_asset")
    return _dedupe(flags)


def _asset_refs(micro: Mapping[str, Any], state_rows: list[Mapping[str, Any]], asset_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for state in state_rows:
        for ref in _as_list(state.get("asset_refs")):
            if isinstance(ref, Mapping):
                refs.append(dict(ref))
    for asset in asset_rows:
        refs.append(
            {
                "asset_id": asset.get("asset_id"),
                "asset_type": asset.get("asset_type"),
                "source_type": asset.get("source_type"),
                "path": asset.get("path"),
                "quality": asset.get("quality"),
            }
        )
    if not refs:
        keyframes = _as_dict(micro.get("keyframes"))
        for key, path in keyframes.items():
            if path:
                refs.append({"asset_type": "keyframe", "rel": key, "path": path})
    return _dedupe_refs(refs)


def _conclusion_status(row: Mapping[str, Any]) -> str:
    event_type = str(row.get("event_type") or "").lower()
    level_text = _source_level_text(row)
    flags = " ".join(_strings(row.get("anomaly_flags"))).lower()
    if "candidate" in event_type or "candidate" in level_text or "candidate" in flags:
        return "candidate"
    if "measured" in level_text or "measurement" in level_text:
        return "measured"
    if event_type in CONFIRMED_EVENT_TYPES or event_type.endswith("_detected") or event_type.endswith("_observed"):
        return "confirmed"
    if any(token in level_text for token in ("confirmed", "ocr_text_detected", "trajectory", "classical_image_change_detected")):
        return "confirmed"
    micro = _as_dict(_as_dict(row.get("payload")).get("micro_segment"))
    evidence = _as_dict(micro.get("evidence"))
    evidence_level = str(evidence.get("evidence_level") or micro.get("evidence_level") or "").lower()
    if evidence_level in {"visual_confirmed", "visual_and_transcript_confirmed"}:
        return "confirmed"
    return "candidate"


def _source_level_text(row: Mapping[str, Any]) -> str:
    payload = _as_dict(row.get("payload"))
    parts = [
        str(row.get("event_type") or ""),
        str(row.get("action_type") or ""),
    ]
    advanced = _as_dict(payload.get("advanced_evidence"))
    if advanced:
        parts.extend(
            [
                str(advanced.get("visual_confirmation_level") or ""),
                str(advanced.get("evidence_type") or ""),
            ]
        )
    observation = _as_dict(payload.get("model_observation"))
    if observation:
        parts.extend(
            [
                str(observation.get("confirmation_level") or ""),
                str(observation.get("event_type") or ""),
                str(observation.get("observation_type") or ""),
            ]
        )
    micro = _as_dict(payload.get("micro_segment"))
    if micro:
        evidence = _as_dict(micro.get("evidence"))
        parts.append(str(evidence.get("evidence_level") or micro.get("evidence_level") or ""))
    return " ".join(parts).lower()


def _has_model_or_visual_support(row: Mapping[str, Any]) -> bool:
    payload = _as_dict(row.get("payload"))
    source = str(payload.get("source") or "").lower()
    if source == "model_observation_events":
        observation = _as_dict(payload.get("model_observation"))
        confirmation_level = str(observation.get("confirmation_level") or "").lower()
        if "candidate" in confirmation_level:
            return False
        return bool(observation.get("evidence_reasons") or observation.get("measurement") or observation.get("asset_refs") or observation.get("metrics"))
    if source == "advanced_vision_evidence":
        advanced = _as_dict(payload.get("advanced_evidence"))
        level = str(advanced.get("visual_confirmation_level") or "").lower()
        if "candidate" in level:
            return False
        return bool(advanced.get("evidence_reasons") or advanced.get("metrics") or advanced.get("asset_refs") or row.get("asset_refs"))
    if source == "micro_segment":
        micro = _as_dict(payload.get("micro_segment"))
        evidence = _as_dict(micro.get("evidence"))
        level = str(evidence.get("evidence_level") or micro.get("evidence_level") or "").lower()
        keyframes = _as_dict(micro.get("keyframes"))
        return level in {"visual_confirmed", "visual_and_transcript_confirmed"} and any(keyframes.values())
    reasons = " ".join(_strings(row.get("confidence_reasons"))).lower()
    return bool(row.get("asset_refs")) or "visual" in reasons or "model" in reasons or "ocr" in reasons


def _candidate_event_type(event_type: str) -> str:
    if event_type.endswith("_candidate"):
        return event_type
    return {
        "container_color_change_detected": "container_color_change_candidate",
        "container_state_change_detected": "container_state_change_candidate",
        "equipment_panel_operation_detected": "equipment_panel_operation_candidate",
        "liquid_flow_detected": "liquid_flow_candidate_visual",
        "liquid_level_change_detected": "liquid_level_change_candidate",
        "object_movement_detected": "object_movement_candidate",
        "object_track_observed": "object_track_candidate",
    }.get(event_type, f"{event_type}_candidate" if event_type else "physical_event_candidate")


def _normalized_object(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = str(row.get("primary_object") or "").strip()
    event_type = str(row.get("event_type") or "")
    if not raw:
        raw = "unknown_object"
    canonical = _canonical_label(raw)
    category = _object_category(canonical, event_type)
    return {
        "raw_label": raw,
        "canonical_label": canonical,
        "category": category,
        "aliases": _object_aliases(canonical),
    }


def _canonical_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    if "sample" in text and any(token in text for token in ("bottle", "vial", "container")):
        return "sample_bottle"
    if "reagent" in text and "bottle" in text:
        return "reagent_bottle"
    if any(keyword in text for keyword in ("panel", "display", "screen", "readout")):
        return "equipment_panel"
    for canonical, keywords in {
        "pipette": ("pipette", "pipettor"),
        "pipette_tip": ("pipette_tip", "tip"),
        "tube": ("tube", "test_tube"),
        "vial": ("vial",),
        "beaker": ("beaker",),
        "balance": ("balance", "scale"),
        "button": ("button",),
        "knob": ("knob",),
        "switch": ("switch", "toggle"),
        "cap": ("cap", "lid", "stopper"),
        "liquid": ("liquid", "solution", "buffer", "reagent"),
    }.items():
        if any(keyword in text for keyword in keywords):
            return canonical
    return text or "unknown_object"


def _object_category(canonical: str, event_type: str) -> str:
    haystack = f"{canonical} {event_type}".lower()
    if "equipment_panel" in event_type or any(token in haystack for token in OBJECT_CATEGORY_KEYWORDS["equipment_control"]):
        return "equipment_control"
    for category, keywords in OBJECT_CATEGORY_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            return category
    return "object"


def _object_aliases(canonical: str) -> list[str]:
    aliases = {
        "sample_bottle": ["sample bottle", "vial", "bottle"],
        "reagent_bottle": ["reagent bottle", "bottle"],
        "equipment_panel": ["panel", "display", "readout"],
        "pipette": ["pipettor", "liquid transfer tool"],
    }
    return aliases.get(canonical, [])


def _action_classification(row: Mapping[str, Any], normalized: Mapping[str, Any]) -> dict[str, Any]:
    haystack = " ".join(
        [
            str(row.get("event_type") or ""),
            str(row.get("action_type") or ""),
            str(row.get("text") or ""),
            str(normalized.get("canonical_label") or ""),
            str(normalized.get("category") or ""),
        ]
    ).lower()
    family = "other_experiment_action"
    for candidate, keywords in ACTION_FAMILIES.items():
        if candidate in haystack or any(keyword in haystack for keyword in keywords):
            family = candidate
            break
    return {
        "family": family,
        "raw_action_type": row.get("action_type"),
        "event_type": row.get("event_type"),
        "confidence": row.get("confidence"),
    }


def _semantic_description(row: Mapping[str, Any], normalized: Mapping[str, Any], action: Mapping[str, Any]) -> str:
    status = str(row.get("conclusion_status") or "candidate")
    obj = str(normalized.get("canonical_label") or row.get("primary_object") or "object")
    family = str(action.get("family") or "experiment_action")
    event_type = str(row.get("event_type") or "physical_event")
    confidence = _as_float(row.get("confidence"))
    source = str(_as_dict(row.get("payload")).get("source") or "unknown_source")
    prefix = "candidate" if status == "candidate" else status
    score = f"{confidence:.2f}" if confidence is not None else "unknown"
    return f"{prefix} {family} event for {obj}: {event_type} (confidence={score}, evidence={source})"


def _event_entities(row: Mapping[str, Any], normalized: Mapping[str, Any], action: Mapping[str, Any]) -> dict[str, Any]:
    category = str(normalized.get("category") or "")
    canonical = str(normalized.get("canonical_label") or "")
    measurement = _source_measurement(row)
    entities = {
        "materials": [],
        "reagents": [],
        "equipment": [],
        "parameters": _measurement_parameters(measurement),
        "action_family": action.get("family"),
    }
    if category in {"container", "tool", "closure"} and canonical:
        entities["materials"].append(canonical)
    if category == "liquid_or_reagent" and canonical:
        entities["reagents"].append(canonical)
    if category in {"equipment", "equipment_control"} and canonical:
        entities["equipment"].append(canonical)
    return entities


def _source_measurement(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = _as_dict(row.get("payload"))
    observation = _as_dict(payload.get("model_observation"))
    measurement = _as_dict(observation.get("measurement"))
    if measurement:
        return measurement
    advanced = _as_dict(payload.get("advanced_evidence"))
    metrics = _as_dict(advanced.get("metrics"))
    return metrics


def _measurement_parameters(measurement: Mapping[str, Any]) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for key in ("value", "reading", "readout", "display_text", "volume_ml", "volume_ul", "knob_angle_deg", "liquid_level_y_norm", "liquid_level_before", "liquid_level_after"):
        if key not in measurement or measurement.get(key) is None:
            continue
        unit = measurement.get("unit")
        if key == "volume_ml":
            unit = "ml"
        elif key == "volume_ul":
            unit = "ul"
        elif key == "knob_angle_deg":
            unit = "deg"
        elif key == "liquid_level_y_norm":
            unit = "normalized_y"
        elif key in {"liquid_level_before", "liquid_level_after"}:
            unit = "normalized_y"
        params.append({"name": key, "value": measurement.get(key), "unit": unit, "source": "video_understanding"})
    return params


def _event_evidence_refs(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for asset in _as_list(row.get("asset_refs")):
        if isinstance(asset, Mapping):
            refs.append({"type": "asset", **dict(asset)})
    payload = _as_dict(row.get("payload"))
    advanced = _as_dict(payload.get("advanced_evidence"))
    if advanced.get("evidence_id"):
        refs.append({"type": "advanced_vision_evidence", "evidence_id": advanced.get("evidence_id")})
    observation = _as_dict(payload.get("model_observation"))
    if observation.get("observation_id"):
        refs.append({"type": "model_observation_event", "observation_id": observation.get("observation_id"), "source_type": observation.get("source_type")})
    micro = _as_dict(payload.get("micro_segment"))
    if micro.get("micro_segment_id"):
        refs.append({"type": "micro_segment", "micro_segment_id": micro.get("micro_segment_id")})
    return _dedupe_refs(refs)


def _event_family(event_type: str) -> str:
    text = str(event_type or "")
    if "liquid_level" in text:
        return "liquid_level"
    if "liquid_flow" in text or "liquid_transfer" in text:
        return "liquid_flow"
    if "container_color" in text:
        return "container_color"
    if "container_state" in text:
        return "container_state"
    if "equipment_panel" in text:
        return "equipment_panel"
    if "movement" in text or "track" in text or "trajectory" in text:
        return "object_movement"
    return text


def _state_signature(row: Mapping[str, Any]) -> str:
    payload = _as_dict(row.get("payload"))
    observation = _as_dict(payload.get("model_observation"))
    measurement = _as_dict(observation.get("measurement"))
    state = _first_non_empty(observation.get("state"), measurement.get("state"), measurement.get("after_state"), measurement.get("open_closed_state"))
    if state:
        return str(state).strip().lower()
    advanced = _as_dict(payload.get("advanced_evidence"))
    metrics = _as_dict(advanced.get("metrics"))
    for key in ("color_change_indicator", "liquid_level_delta", "level_values"):
        value = metrics.get(key)
        if value:
            return f"{key}:{value}"
    return ""


def _requires_review(row: Mapping[str, Any]) -> bool:
    flags = set(_strings(row.get("anomaly_flags")))
    return str(row.get("conclusion_status") or "") == "candidate" or "requires_human_confirmation" in flags


def _stable_event(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in EVENT_FIELDS}


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _strings(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if item is not None and str(item)]


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _dedupe_refs(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str, str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for ref in refs:
        key = (
            str(ref.get("type") or ""),
            str(ref.get("asset_id") or ""),
            str(ref.get("asset_type") or ""),
            str(ref.get("path") or ""),
            str(ref.get("evidence_id") or ref.get("observation_id") or ""),
            str(ref.get("video_event_id") or ref.get("source_event_id") or ""),
            str(ref.get("micro_segment_id") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(ref))
    return output


def _first_text(values: Iterable[Any]) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def _manifest_session_id(session: Path) -> str:
    manifest = session / "manifest.json"
    if not manifest.exists():
        return ""
    try:
        data = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("session_id") or "")


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value) != "":
            return value
    return None


def _join_text(*values: Any) -> str:
    return " ".join(str(value) for value in values if value)


def _confirmed_or_measured_level(value: str) -> bool:
    text = str(value or "").lower()
    return "confirmed" in text or "measured" in text


def _movement_candidate(interaction: Mapping[str, Any], state_types: list[str]) -> bool:
    return bool(interaction.get("primary_object") and {"contact_started", "contact_released"} & set(state_types))


def _liquid_transfer_candidate(primary_object: str, action_type: str, text: str) -> bool:
    haystack = f"{primary_object} {action_type} {text}".lower()
    return any(token in haystack for token in ("pipette", "tube", "liquid", "transfer", "pipetting", "sample_adding", "移液", "加样", "微升"))


def _equipment_panel_candidate(primary_object: str, action_type: str, text: str) -> bool:
    haystack = f"{primary_object} {action_type} {text}".lower()
    return any(token in haystack for token in ("balance", "panel", "button", "display", "readout", "recording", "天平", "读数", "记录", "面板"))


def _container_state_candidate(primary_object: str, state_types: list[str], text: str) -> bool:
    haystack = f"{primary_object} {text}".lower()
    return bool({"contact_started", "contact_released"} & set(state_types)) and any(
        token in haystack for token in ("bottle", "tube", "vial", "container", "flask", "样品瓶", "瓶", "试管", "容器")
    )


def _event_text(event_type: str, primary_object: str, action_type: str, text: str) -> str:
    subject = primary_object or "unknown_object"
    action = action_type or "unknown_action"
    if event_type == "hand_object_contact":
        return f"hand-object contact with {subject}"
    if event_type == "object_state_change":
        return f"state changes observed around {subject}"
    if event_type == "experiment_action_classification":
        return f"classified action as {action}"
    if event_type == "liquid_transfer_candidate":
        return f"candidate liquid transfer related to {subject}; requires visual confirmation"
    if event_type == "equipment_panel_operation_candidate":
        return f"candidate equipment panel/readout operation for {subject}; OCR not confirmed"
    if event_type == "container_state_change_candidate":
        return f"candidate container state change for {subject}; open/close/liquid level not confirmed"
    if event_type == "object_movement_candidate":
        return f"candidate object movement inferred from contact state changes for {subject}"
    return text[:200]


__all__ = ["build_video_understanding", "load_video_understanding"]
