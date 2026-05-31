from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .jitter_profile import jitter_profile_for_label
from .physical_event_types import EventType, GateDecision, GateStatus, HardGate, JitterProfile, SceneMotionEvidence, TrackEvidence
from .scene_stabilizer import no_scene_stabilization
from .track_normalizer import normalize_track_evidence


DEFAULT_CONFIG = {
    "object_move": {
        "min_track_points": 4,
        "min_identity_confidence": 0.75,
        "max_id_switch_risk": 0.30,
        "base_motion_threshold_px": 10.0,
        "fallback_motion_threshold_px": 12.0,
        "bbox_size_motion_ratio": 0.04,
        "fallback_bbox_size_motion_ratio": 0.05,
        "jitter_sigma_multiplier": 3.0,
        "no_scene_stabilization_requires_hand_contact": True,
        "no_scene_stabilization_large_motion_multiplier": 2.5,
    },
    "hand_object_contact": {
        "min_contact_frames": 2,
        "min_overlap_ratio": 0.03,
        "min_object_coverage_by_hand": 0.08,
        "near_only_is_candidate": True,
    },
    "liquid_transfer": {"require_liquid_state_change_for_confirmed": True},
    "panel_operation": {"require_control_contact_or_state_change": True},
    "container_state_change": {"require_pre_post_state_diff": True},
    "qwen_audit": {"enable": True, "allow_qwen_upgrade": False, "allow_qwen_downgrade": True, "require_vlm_audit": False},
}


def gate_object_move(
    *,
    event_candidate: Mapping[str, Any] | None = None,
    track: Mapping[str, Any] | TrackEvidence | None = None,
    scene_motion: Mapping[str, Any] | SceneMotionEvidence | None = None,
    jitter_profile: Mapping[str, Any] | JitterProfile | None = None,
    hand_contact: Mapping[str, Any] | None = None,
    frame_evidence_list: Sequence[Mapping[str, Any]] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG["object_move"], **dict(thresholds or {})}
    track_ev = normalize_track_evidence(track)
    scene = _scene(scene_motion)
    required = [
        "stable_instance_track",
        "identity_confidence",
        "low_id_switch_risk",
        "stabilized_displacement",
        "anti_jitter_threshold",
        "persistent_motion",
        "no_camera_motion",
    ]
    failed: list[str] = []
    passed: list[str] = []
    reject: list[str] = []
    limitations: list[str] = []
    evidence: dict[str, Any] = {}
    if track_ev is None:
        return _decision(EventType.OBJECT_MOVE.value, GateStatus.REJECTED.value, 0.0, "gate_object_move", required, [], required, {}, ["missing_track"], [])

    jitter = _jitter(jitter_profile, track_ev.object_label)
    threshold = _motion_threshold(track_ev, jitter, cfg)
    stabilized = track_ev.stabilized_displacement_px
    if stabilized is None:
        stabilized = max(0.0, track_ev.raw_displacement_px - scene.background_shift_px)
    evidence = {
        "raw_displacement_px": round(track_ev.raw_displacement_px, 3),
        "path_length_px": round(track_ev.path_length_px, 3),
        "background_shift_px": round(scene.background_shift_px, 3),
        "stabilized_displacement_px": round(stabilized, 3),
        "jitter_sigma_px": round(jitter.sigma_px, 3),
        "motion_threshold_px": round(threshold, 3),
        "median_bbox_size": round(track_ev.median_bbox_size, 3),
        "point_count": track_ev.point_count,
        "identity_confidence": round(track_ev.identity_confidence, 4),
        "id_switch_risk": round(track_ev.id_switch_risk, 4),
        "track_type": track_ev.track_type,
        "can_confirm_motion": track_ev.can_confirm_motion,
        "motion_persistent": track_ev.motion_persistent,
        "hand_contact_before_or_during_motion": _confirmed(hand_contact),
        "scene_motion_method": scene.method,
        "change_score": _first_float((event_candidate or {}).get("change_score")),
    }
    limitations.extend(track_ev.limitations)
    limitations.extend(scene.limitations)

    if track_ev.track_type == "label_level_pseudo_track":
        failed.append("stable_instance_track")
        reject.append("label_level_pseudo_track")
    else:
        passed.append("stable_instance_track")
    if not track_ev.can_confirm_motion:
        failed.append("stable_instance_track")
        reject.append("track_cannot_confirm_motion")
    if track_ev.point_count < int(cfg["min_track_points"]):
        failed.append("stable_instance_track")
        reject.append("too_few_points")
    if track_ev.identity_confidence < float(cfg["min_identity_confidence"]):
        failed.append("identity_confidence")
        reject.append("low_identity_confidence")
    else:
        passed.append("identity_confidence")
    if track_ev.id_switch_risk > float(cfg["max_id_switch_risk"]):
        failed.append("low_id_switch_risk")
        reject.append("high_id_switch_risk")
    else:
        passed.append("low_id_switch_risk")
    if scene.is_camera_motion:
        failed.append("no_camera_motion")
        reject.append("camera_motion")
    else:
        passed.append("no_camera_motion")
    if scene.is_scene_cut:
        failed.append("no_scene_cut")
        reject.append("scene_cut")
    if stabilized < threshold:
        failed.append("stabilized_displacement")
        reject.append("displacement_below_threshold")
        reject.append("bbox_jitter_or_static_object")
    else:
        passed.append("stabilized_displacement")
        passed.append("anti_jitter_threshold")
    if not track_ev.motion_persistent:
        failed.append("persistent_motion")
        reject.append("non_persistent_motion")
    else:
        passed.append("persistent_motion")
    if (event_candidate or {}).get("change_score") and stabilized < threshold:
        reject.append("change_score_only")

    hard_failed = sorted(set(failed))
    if any(reason in reject for reason in ("label_level_pseudo_track", "track_cannot_confirm_motion", "too_few_points", "low_identity_confidence", "high_id_switch_risk", "camera_motion", "scene_cut", "displacement_below_threshold", "non_persistent_motion")):
        status = GateStatus.REJECTED.value
    else:
        status = GateStatus.CONFIRMED.value

    if status == GateStatus.CONFIRMED.value and scene.method == "none":
        large_motion = stabilized >= float(cfg["no_scene_stabilization_large_motion_multiplier"]) * threshold
        if bool(cfg["no_scene_stabilization_requires_hand_contact"]) and not (_confirmed(hand_contact) and large_motion):
            status = GateStatus.CANDIDATE.value
            hard_failed.append("scene_stabilization")
            limitations.append("no scene stabilization; motion kept as candidate")

    confidence = 0.0
    if status == GateStatus.CONFIRMED.value:
        confidence = min(0.95, 0.45 + min(stabilized / max(threshold, 1.0), 3.0) * 0.12 + track_ev.identity_confidence * 0.25)
    elif status == GateStatus.CANDIDATE.value:
        confidence = min(0.74, 0.35 + min(stabilized / max(threshold, 1.0), 2.0) * 0.12)
    else:
        confidence = min(0.45, max(0.05, stabilized / max(threshold * 3.0, 1.0)))
    return _decision(EventType.OBJECT_MOVE.value, status, confidence, "gate_object_move", required, passed, hard_failed, evidence, sorted(set(reject)), sorted(set(limitations)))


def gate_hand_object_contact(
    *,
    event_candidate: Mapping[str, Any] | None = None,
    hand_track: Mapping[str, Any] | None = None,
    object_track: Mapping[str, Any] | None = None,
    frame_evidence_list: Sequence[Mapping[str, Any]] | None = None,
    relations: Sequence[Mapping[str, Any]] | None = None,
    external_observation: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_CONFIG["hand_object_contact"], **dict(thresholds or {})}
    metrics = _contact_metrics(frame_evidence_list or [], external_observation or {})
    required = ["hand_detection", "object_detection", "continuous_contact", "overlap_or_occlusion_or_contact_observation"]
    passed: list[str] = []
    failed: list[str] = []
    reject: list[str] = []
    if metrics["has_hand"]:
        passed.append("hand_detection")
    else:
        failed.append("hand_detection")
        reject.append("no_hand_detection")
    if metrics["has_object"]:
        passed.append("object_detection")
    else:
        failed.append("object_detection")
        reject.append("no_object_detection")
    contact_enough = metrics["continuous_contact_frames"] >= int(cfg["min_contact_frames"])
    overlap_enough = metrics["max_iou"] >= float(cfg["min_overlap_ratio"]) or metrics["max_object_coverage_by_hand"] >= float(cfg["min_object_coverage_by_hand"]) or bool(metrics["occlusion_score"])
    external_contact = str(metrics.get("contact_type") or "").lower() in {"grasp", "hold", "press", "touch", "occlude", "contact"}
    if contact_enough:
        passed.append("continuous_contact")
    else:
        failed.append("continuous_contact")
    if overlap_enough or external_contact:
        passed.append("overlap_or_occlusion_or_contact_observation")
    else:
        failed.append("overlap_or_occlusion_or_contact_observation")
    if contact_enough and (overlap_enough or external_contact):
        status = GateStatus.CONFIRMED.value
    elif metrics["near_only"] or metrics["contact_frames"] == 1 or metrics["min_distance_px"] is not None:
        status = GateStatus.CANDIDATE.value if bool(cfg.get("near_only_is_candidate", True)) else GateStatus.REJECTED.value
        reject.extend(["near_only"] if metrics["near_only"] else ["single_frame_near"])
    else:
        status = GateStatus.REJECTED.value
    confidence = 0.82 if status == GateStatus.CONFIRMED.value else (0.48 if status == GateStatus.CANDIDATE.value else 0.12)
    return _decision(EventType.HAND_OBJECT_INTERACTION.value, status, confidence, "gate_hand_object_contact", required, passed, failed, metrics, sorted(set(reject)), [])


def gate_liquid_transfer(
    *,
    event_candidate: Mapping[str, Any] | None = None,
    source_container_track: Mapping[str, Any] | None = None,
    target_container_track: Mapping[str, Any] | None = None,
    tool_track: Mapping[str, Any] | None = None,
    liquid_observation: Mapping[str, Any] | None = None,
    frame_pair_evidence: Mapping[str, Any] | None = None,
    external_observation: Mapping[str, Any] | None = None,
    qwen_semantics: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    obs = {**dict(frame_pair_evidence or {}), **dict(liquid_observation or {}), **dict(external_observation or {})}
    level_delta = abs(_first_float(obs.get("liquid_level_delta")) or 0.0)
    source_down = bool(obs.get("source_level_down"))
    target_up = bool(obs.get("target_level_up"))
    droplet = bool(obs.get("droplet_detected"))
    stream = bool(obs.get("stream_detected") or obs.get("liquid_flow"))
    region_change = bool(obs.get("liquid_region_changed")) or abs(_first_float(obs.get("liquid_region_area_delta")) or 0.0) > 0.05
    visual_change = _first_float(obs.get("visual_change_score")) or 0.0
    evidence = {
        "has_liquid_region": bool(obs.get("has_liquid_region")),
        "liquid_region_area_before": obs.get("liquid_region_area_before"),
        "liquid_region_area_after": obs.get("liquid_region_area_after"),
        "liquid_level_before": obs.get("liquid_level_before"),
        "liquid_level_after": obs.get("liquid_level_after"),
        "liquid_level_delta": level_delta,
        "droplet_detected": droplet,
        "stream_detected": stream,
        "source_container_id": _track_id(source_container_track),
        "target_container_id": _track_id(target_container_track),
        "tool_track_id": _track_id(tool_track),
        "source_level_down": source_down,
        "target_level_up": target_up,
        "visual_change_score": visual_change,
        "lighting_change_risk": bool(obs.get("lighting_change_risk")),
    }
    required = ["liquid_state_change_or_flow"]
    if level_delta >= 0.08 or (source_down and target_up) or droplet or stream or region_change or str(obs.get("status")) == "confirmed":
        return _decision(EventType.LIQUID_TRANSFER.value, GateStatus.CONFIRMED.value, 0.82, "gate_liquid_transfer", required, required, [], evidence, [], [])
    if evidence["has_liquid_region"] and not (_track_id(tool_track) or visual_change > 0.08 or qwen_semantics):
        return _decision(EventType.LIQUID_TRANSFER.value, GateStatus.REJECTED.value, 0.12, "gate_liquid_transfer", required, [], required, evidence, ["only_liquid_present", "no_liquid_region_change", "no_level_change"], [])
    if evidence["has_liquid_region"] or _track_id(tool_track) or visual_change > 0.08 or qwen_semantics:
        reasons = ["no_liquid_region_change", "no_level_change"]
        if qwen_semantics and not (level_delta or droplet or stream):
            reasons.append("qwen_semantic_only")
        return _decision(EventType.LIQUID_TRANSFER.value, GateStatus.CANDIDATE.value, 0.42, "gate_liquid_transfer", required, [], required, evidence, reasons, ["liquid transfer cue without hard liquid motion evidence"])
    return _decision(EventType.LIQUID_TRANSFER.value, GateStatus.REJECTED.value, 0.1, "gate_liquid_transfer", required, [], required, evidence, ["only_container_present", "no_liquid_region_change", "no_level_change"], [])


def gate_panel_operation(
    *,
    event_candidate: Mapping[str, Any] | None = None,
    hand_track: Mapping[str, Any] | None = None,
    device_track: Mapping[str, Any] | None = None,
    panel_roi: Any = None,
    control_roi: Any = None,
    ocr_before_after: Mapping[str, Any] | None = None,
    display_state: Mapping[str, Any] | None = None,
    external_observation: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    obs = {**dict(ocr_before_after or {}), **dict(display_state or {}), **dict(external_observation or {})}
    hand_frames = int(obs.get("hand_in_control_roi_frames") or obs.get("contact_frames") or 0)
    state_changed = any(bool(obs.get(key)) for key in ("display_changed", "button_state_changed", "switch_state_changed", "external_state_change")) or abs(_first_float(obs.get("knob_angle_delta")) or 0.0) >= 5.0
    evidence = {
        "device_track_id": _track_id(device_track),
        "hand_track_id": _track_id(hand_track),
        "panel_roi": panel_roi,
        "control_roi": control_roi,
        "hand_in_control_roi_frames": hand_frames,
        "contact_frames": int(obs.get("contact_frames") or hand_frames),
        "display_text_before": obs.get("display_text_before"),
        "display_text_after": obs.get("display_text_after"),
        "display_changed": bool(obs.get("display_changed")),
        "button_state_changed": bool(obs.get("button_state_changed")),
        "knob_angle_delta": _first_float(obs.get("knob_angle_delta")),
        "switch_state_changed": bool(obs.get("switch_state_changed")),
        "external_state_change": bool(obs.get("external_state_change")),
    }
    required = ["hand_control_contact_or_panel_state_change"]
    external_confirmed = str(obs.get("status") or obs.get("external_status") or "").lower() == "confirmed"
    if state_changed or external_confirmed:
        return _decision(EventType.PANEL_OPERATION.value, GateStatus.CONFIRMED.value, 0.82, "gate_panel_operation", required, required, [], evidence, [], [])
    if hand_frames > 0:
        return _decision(EventType.PANEL_OPERATION.value, GateStatus.CANDIDATE.value, 0.38, "gate_panel_operation", required, [], required, evidence, ["no_panel_state_change", "no_control_roi" if control_roi is None else "no_hand_control_contact"], [])
    return _decision(EventType.PANEL_OPERATION.value, GateStatus.REJECTED.value, 0.08, "gate_panel_operation", required, [], required, evidence, ["device_presence_only", "no_hand_control_contact", "no_panel_state_change"], [])


def gate_container_state_change(
    *,
    event_candidate: Mapping[str, Any] | None = None,
    container_track: Mapping[str, Any] | None = None,
    pre_state: Mapping[str, Any] | str | None = None,
    post_state: Mapping[str, Any] | str | None = None,
    frame_pair_evidence: Mapping[str, Any] | None = None,
    external_observation: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pre = _state_dict(pre_state, "before")
    post = _state_dict(post_state, "after")
    obs = {**dict(frame_pair_evidence or {}), **dict(external_observation or {})}
    changed = list(obs.get("changed_fields") or [])
    observable = {"cap_state", "lid_state", "mouth_state", "liquid_level", "has_content", "content_color", "turbidity", "solid_presence", "content_amount"}
    for key in observable:
        if key in pre or key in post:
            if pre.get(key) != post.get(key):
                changed.append(key)
    changed = sorted(set(str(item) for item in changed if str(item) in observable))
    track_id = _track_id(container_track)
    track_type = str((container_track or {}).get("track_type") or "") if isinstance(container_track, Mapping) else ""
    pre_instance = pre.get("container_id") or pre.get("track_id")
    post_instance = post.get("container_id") or post.get("track_id")
    external_same = bool(obs.get("same_container_instance") or obs.get("same_container_id"))
    explicit_same = bool(pre_instance and post_instance and str(pre_instance) == str(post_instance))
    explicit_conflict = bool(pre_instance and post_instance and str(pre_instance) != str(post_instance))
    track_can_prove_same = bool(track_id and track_type != "label_level_pseudo_track")
    same_container_instance = bool(
        track_can_prove_same
        and not explicit_conflict
        and (external_same or explicit_same or (not pre_instance and not post_instance))
    )
    evidence = {
        "container_track_id": track_id,
        "container_track_type": track_type,
        "same_container_instance": same_container_instance,
        "pre_state": pre,
        "post_state": post,
        "changed_fields": changed,
        "cap_state_before": pre.get("cap_state"),
        "cap_state_after": post.get("cap_state"),
        "mouth_state_before": pre.get("mouth_state"),
        "mouth_state_after": post.get("mouth_state"),
        "liquid_level_before": pre.get("liquid_level"),
        "liquid_level_after": post.get("liquid_level"),
        "content_color_before": pre.get("content_color"),
        "content_color_after": post.get("content_color"),
        "turbidity_before": pre.get("turbidity"),
        "turbidity_after": post.get("turbidity"),
        "lighting_change_risk": bool(obs.get("lighting_change_risk")),
    }
    required = ["same_container_instance", "pre_post_state_diff", "observable_changed_fields"]
    if same_container_instance:
        passed = ["same_container_instance"]
    else:
        passed = []
    if explicit_conflict:
        return _decision(EventType.CONTAINER_STATE_CHANGE.value, GateStatus.REJECTED.value, 0.12, "gate_container_state_change", required, passed, [item for item in required if item not in passed], evidence, ["different_container_instances"], [])
    if evidence["lighting_change_risk"]:
        return _decision(EventType.CONTAINER_STATE_CHANGE.value, GateStatus.REJECTED.value, 0.12, "gate_container_state_change", required, passed, [item for item in required if item not in passed], evidence, ["lighting_change_only"], [])
    if changed and same_container_instance and (pre or post):
        return _decision(EventType.CONTAINER_STATE_CHANGE.value, GateStatus.CONFIRMED.value, 0.8, "gate_container_state_change", required, [*passed, "pre_post_state_diff", "observable_changed_fields"], [], evidence, [], [])
    if changed or pre or post:
        reason = "missing_same_container_track" if not same_container_instance else "no_pre_post_state"
        return _decision(EventType.CONTAINER_STATE_CHANGE.value, GateStatus.CANDIDATE.value, 0.4, "gate_container_state_change", required, passed, [item for item in required if item not in passed], evidence, [reason], [])
    return _decision(EventType.CONTAINER_STATE_CHANGE.value, GateStatus.REJECTED.value, 0.08, "gate_container_state_change", required, passed, [item for item in required if item not in passed], evidence, ["container_presence_only", "no_pre_post_state", "no_changed_fields"], [])


def normalize_gate_decision(decision: Mapping[str, Any] | GateDecision) -> dict[str, Any]:
    if isinstance(decision, GateDecision):
        return decision.to_dict()
    data = dict(decision)
    data.setdefault("status", GateStatus.UNCERTAIN.value)
    data.setdefault("confidence", 0.0)
    data.setdefault("hard_gate", {"passed": data.get("status") == GateStatus.CONFIRMED.value, "gate_name": "unknown", "required_evidence": [], "passed_evidence": [], "failed_evidence": []})
    data.setdefault("evidence", {})
    data.setdefault("reject_reasons", [])
    data.setdefault("limitations", [])
    data.setdefault("audit", {})
    return data


def merge_gate_with_qwen_audit(gate_decision: Mapping[str, Any] | GateDecision, qwen_audit: Mapping[str, Any] | str | None, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    gate = normalize_gate_decision(gate_decision)
    cfg = {**DEFAULT_CONFIG["qwen_audit"], **dict((config or {}).get("qwen_audit") or config or {})}
    audit = parse_qwen_audit(qwen_audit, event_type=gate.get("event_type"))
    final_status = gate["status"]
    should_write = False
    if gate["status"] != GateStatus.CONFIRMED.value:
        should_write = False
    elif audit.get("status") == "parse_failed":
        should_write = not bool(cfg.get("require_vlm_audit"))
        final_status = GateStatus.UNCERTAIN.value if cfg.get("require_vlm_audit") else GateStatus.CONFIRMED.value
    elif audit.get("decision") == "accept":
        final_status = GateStatus.CONFIRMED.value
        should_write = True
    elif audit.get("decision") == "uncertain":
        final_status = GateStatus.UNCERTAIN.value if bool(cfg.get("allow_qwen_downgrade", True)) else GateStatus.CONFIRMED.value
    elif audit.get("decision") == "reject":
        final_status = "rejected_by_audit" if bool(cfg.get("allow_qwen_downgrade", True)) else GateStatus.CONFIRMED.value
    merged = {
        **gate,
        "final_status": final_status,
        "qwen_audit": audit,
        "should_write_confirmed_event": bool(should_write and final_status == GateStatus.CONFIRMED.value),
    }
    if gate["status"] != GateStatus.CONFIRMED.value and audit.get("decision") == "accept":
        merged.setdefault("audit", {})["qwen_upgrade_blocked"] = True
        merged["qwen_audit"]["should_write_confirmed_event"] = False
    return merged


def parse_qwen_audit(value: Mapping[str, Any] | str | None, *, event_type: str | None = None) -> dict[str, Any]:
    if value is None:
        return {"status": "not_run", "decision": "uncertain", "event_type": event_type, "should_write_confirmed_event": False}
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        text = str(value or "").strip()
        try:
            payload = json.loads(text)
        except Exception:
            return {"status": "parse_failed", "decision": "uncertain", "event_type": event_type, "should_write_confirmed_event": False, "raw_text": text}
    decision = str(payload.get("decision") or "uncertain").strip().lower()
    if decision not in {"accept", "reject", "uncertain"}:
        decision = "uncertain"
    payload["decision"] = decision
    payload.setdefault("event_type", event_type)
    payload["should_write_confirmed_event"] = bool(payload.get("should_write_confirmed_event")) and decision == "accept"
    payload.setdefault("status", "parsed")
    return payload


def write_rejected_candidate(path: str | Path, candidate: Mapping[str, Any], decision: Mapping[str, Any] | GateDecision) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    gate = normalize_gate_decision(decision)
    row = {
        "candidate_id": candidate.get("candidate_id") or candidate.get("proposal_id") or candidate.get("event_id"),
        "event_type": gate.get("event_type") or candidate.get("event_type"),
        "status": gate.get("status"),
        "time_start": candidate.get("time_start") or candidate.get("start_time_sec") or candidate.get("start_sec"),
        "time_end": candidate.get("time_end") or candidate.get("end_time_sec") or candidate.get("end_sec"),
        "source_view": candidate.get("source_view") or candidate.get("view"),
        "actor_track_id": candidate.get("actor_track_id"),
        "object_track_ids": candidate.get("object_track_ids") or candidate.get("involved_track_ids") or [],
        "object_labels": candidate.get("object_labels") or candidate.get("involved_objects") or [],
        "reject_reasons": gate.get("reject_reasons") or [],
        "evidence_detail": gate.get("evidence") or {},
        "limitations": gate.get("limitations") or [],
    }
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_gate_decisions(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(str(item.get("status") or "unknown") for item in decisions)
    by_type: dict[str, Counter[str]] = {}
    reason_counts: Counter[str] = Counter()
    for item in decisions:
        event_type = str(item.get("event_type") or "unknown")
        by_type.setdefault(event_type, Counter())[str(item.get("status") or "unknown")] += 1
        reason_counts.update(str(reason) for reason in item.get("reject_reasons") or [])
    return {
        "total_candidates": len(decisions),
        "confirmed": status_counts.get(GateStatus.CONFIRMED.value, 0),
        "candidate": status_counts.get(GateStatus.CANDIDATE.value, 0),
        "uncertain": status_counts.get(GateStatus.UNCERTAIN.value, 0),
        "rejected": status_counts.get(GateStatus.REJECTED.value, 0),
        "by_event_type": {key: dict(value) for key, value in sorted(by_type.items())},
        "top_reject_reasons": reason_counts.most_common(20),
    }


def _decision(event_type: str, status: str, confidence: float, gate_name: str, required: Sequence[str], passed: Sequence[str], failed: Sequence[str], evidence: Mapping[str, Any], reject: Sequence[str], limitations: Sequence[str]) -> dict[str, Any]:
    return GateDecision(
        status=status,
        event_type=event_type,
        confidence=round(max(0.0, min(1.0, float(confidence))), 4),
        hard_gate=HardGate(
            passed=status == GateStatus.CONFIRMED.value,
            gate_name=gate_name,
            required_evidence=list(dict.fromkeys(required)),
            passed_evidence=list(dict.fromkeys(passed)),
            failed_evidence=list(dict.fromkeys(failed)),
        ),
        evidence=dict(evidence),
        reject_reasons=list(dict.fromkeys(reject)),
        limitations=list(dict.fromkeys(limitations)),
        audit={},
    ).to_dict()


def _motion_threshold(track: TrackEvidence, jitter: JitterProfile, cfg: Mapping[str, Any]) -> float:
    if jitter.source == "fallback":
        return max(float(cfg["fallback_motion_threshold_px"]), float(cfg["fallback_bbox_size_motion_ratio"]) * track.median_bbox_size, float(cfg["jitter_sigma_multiplier"]) * jitter.sigma_px)
    return max(float(cfg["base_motion_threshold_px"]), float(cfg["bbox_size_motion_ratio"]) * track.median_bbox_size, float(cfg["jitter_sigma_multiplier"]) * jitter.sigma_px)


def _scene(value: Mapping[str, Any] | SceneMotionEvidence | None) -> SceneMotionEvidence:
    if isinstance(value, SceneMotionEvidence):
        return value
    if not value:
        return no_scene_stabilization()
    return SceneMotionEvidence(
        is_camera_motion=bool(value.get("is_camera_motion")),
        is_scene_cut=bool(value.get("is_scene_cut")),
        background_shift_px=_first_float(value.get("background_shift_px")) or 0.0,
        homography_confidence=_first_float(value.get("homography_confidence")) or 0.0,
        global_motion_ratio=_first_float(value.get("global_motion_ratio")) or 0.0,
        method=str(value.get("method") or "none"),
        limitations=list(value.get("limitations") or []),
    )


def _jitter(value: Mapping[str, Any] | JitterProfile | None, label: str) -> JitterProfile:
    if isinstance(value, JitterProfile):
        return value
    if isinstance(value, Mapping) and "sigma_px" in value:
        return JitterProfile(object_label=str(value.get("object_label") or label), sigma_px=_first_float(value.get("sigma_px")) or 3.0, source=str(value.get("source") or "fallback"), sample_count=int(value.get("sample_count") or 0))
    return jitter_profile_for_label(label)


def _contact_metrics(rows: Sequence[Mapping[str, Any]], external: Mapping[str, Any]) -> dict[str, Any]:
    contact_frames = 0
    overlap_frames = 0
    max_iou = 0.0
    max_coverage = 0.0
    min_distance: float | None = None
    has_hand = bool(external.get("has_hand"))
    has_object = bool(external.get("has_object"))
    near_only = bool(external.get("near_only"))
    for row in rows:
        detections = row.get("detections") or []
        interactions = row.get("hand_object_interactions") or []
        has_hand = has_hand or any(str(det.get("label") or det.get("class_name") or "").lower() in {"hand", "gloved_hand"} for det in detections if isinstance(det, Mapping))
        has_object = has_object or any(str(det.get("label") or det.get("class_name") or "").lower() not in {"hand", "gloved_hand"} for det in detections if isinstance(det, Mapping))
        for item in interactions:
            if not isinstance(item, Mapping):
                continue
            score = _first_float(item.get("score") or item.get("interaction_score")) or 0.0
            iou = _first_float(item.get("iou") or item.get("bbox_overlap")) or 0.0
            coverage = _first_float(item.get("object_coverage_by_hand")) or 0.0
            distance = _first_float(item.get("distance_px"))
            max_iou = max(max_iou, iou)
            max_coverage = max(max_coverage, coverage)
            if distance is not None:
                min_distance = distance if min_distance is None else min(min_distance, distance)
            if iou > 0.0 or coverage > 0.0 or score >= 0.65:
                contact_frames += 1
            if iou > 0.0 or coverage > 0.0:
                overlap_frames += 1
            elif distance is not None:
                near_only = True
    contact_frames = max(contact_frames, int(external.get("contact_frames") or 0))
    overlap_frames = max(overlap_frames, int(external.get("overlap_frames") or 0))
    return {
        "min_distance_px": min_distance if min_distance is not None else external.get("min_distance_px"),
        "overlap_frames": overlap_frames,
        "contact_frames": contact_frames,
        "continuous_contact_frames": int(external.get("continuous_contact_frames") or contact_frames),
        "max_iou": max(max_iou, _first_float(external.get("max_iou")) or 0.0),
        "max_object_coverage_by_hand": max(max_coverage, _first_float(external.get("max_object_coverage_by_hand")) or 0.0),
        "contact_type": external.get("contact_type"),
        "near_only": near_only and overlap_frames == 0 and contact_frames == 0,
        "hand_track_id": external.get("hand_track_id"),
        "object_track_id": external.get("object_track_id"),
        "has_hand": has_hand or bool(external.get("hand_track_id")),
        "has_object": has_object or bool(external.get("object_track_id")),
        "occlusion_score": _first_float(external.get("occlusion_score")) or 0.0,
    }


def _confirmed(value: Mapping[str, Any] | None) -> bool:
    return bool(value and (value.get("status") == GateStatus.CONFIRMED.value or value.get("event_status") == GateStatus.CONFIRMED.value or value.get("confirmed") is True))


def _track_id(value: Mapping[str, Any] | None) -> str | None:
    return str(value.get("track_id")) if isinstance(value, Mapping) and value.get("track_id") else None


def _state_dict(value: Mapping[str, Any] | str | None, key: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    text = str(value or "").strip()
    return {key: text} if text else {}


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None
