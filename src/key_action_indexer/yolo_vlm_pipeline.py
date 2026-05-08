from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .physical_evidence import (
    PHYSICAL_EVIDENCE_MIN_FRAMES,
    evidence_view,
    valid_yolo_physical_evidence,
    yolo_physical_evidence_diagnostics,
)
from .schemas import read_jsonl
from .yolo_detector import HAND_LABELS, canonical_yolo_label


PIPELINE_SCHEMA_VERSION = "yolo_vlm_physical_action_pipeline.v1"
PIPELINE_FLOW = [
    "yolo_continuous_detection",
    "physical_action_candidate",
    "vlm_constrained_semantic_review",
    "yolo_evidence_recheck",
    "frontend_review_gate",
]
FRONTEND_REVIEW_GATE_POLICY = (
    "Only frontend-approved candidates are synchronized into material_references "
    "and the frontend key material library."
)


def apply_yolo_vlm_review_pipeline(
    session_dir: str | Path,
    candidate_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]] | None = None,
    *,
    vlm_client: Any | None = None,
    enable_vlm: bool = False,
    max_vlm_groups: int = 8,
    vlm_model_name: str | None = None,
) -> dict[str, Any]:
    """Attach the durable YOLO -> VLM -> review-gate pipeline evidence to candidates.

    The VLM step is deliberately advisory. YOLO physical evidence remains the
    admission gate; VLM output is constrained to the object labels already found
    by YOLO and can never add a new confirmed object.
    """

    session_root = Path(session_dir)
    if micro_rows is None:
        micro_path = session_root / "metadata" / "micro_segments.jsonl"
        micro_rows = read_jsonl(micro_path) if micro_path.exists() else []

    grouped = _candidate_groups(candidate_rows)
    micro_by_id = {
        str(row.get("micro_segment_id") or ""): row
        for row in micro_rows
        if isinstance(row, dict) and row.get("micro_segment_id")
    }
    vlm_budget = max(0, int(max_vlm_groups or 0))
    groups_rechecked = 0
    groups_rejected = 0
    groups_vlm_reviewed = 0
    vlm_status_counts: Counter[str] = Counter()
    yolo_status_counts: Counter[str] = Counter()

    for group_id, rows in grouped.items():
        first = rows[0]
        primary = canonical_yolo_label(first.get("primary_object"))
        view = str(first.get("view") or "")
        micro_id = str(first.get("micro_segment_id") or "")
        micro = micro_by_id.get(micro_id, {})
        raw_evidence = [row for row in (micro.get("yolo_evidence") or []) if isinstance(row, dict)]
        scoped_evidence = [row for row in raw_evidence if not view or evidence_view(row) == view] or raw_evidence
        valid_evidence = valid_yolo_physical_evidence(scoped_evidence, primary)
        diagnostics = yolo_physical_evidence_diagnostics(scoped_evidence, primary)
        yolo_passed = bool(primary) and len(valid_evidence) >= PHYSICAL_EVIDENCE_MIN_FRAMES
        yolo_recheck = {
            "status": "passed" if yolo_passed else "failed",
            "primary_object": primary,
            "view": view or None,
            "micro_segment_id": micro_id or None,
            "valid_evidence_count": len(valid_evidence),
            "required_min_frames": PHYSICAL_EVIDENCE_MIN_FRAMES,
            "diagnostics": diagnostics,
        }
        yolo_status_counts[yolo_recheck["status"]] += 1
        groups_rechecked += 1

        evidence_packet = _build_yolo_evidence_packet(
            group_id=group_id,
            rows=rows,
            micro=micro,
            primary_object=primary,
            view=view,
            valid_evidence=valid_evidence,
            diagnostics=diagnostics,
        )
        should_call_vlm = bool(enable_vlm and vlm_client and yolo_passed and groups_vlm_reviewed < vlm_budget)
        vlm_semantics = _run_constrained_vlm_review(
            group_id=group_id,
            rows=rows,
            evidence_packet=evidence_packet,
            vlm_client=vlm_client if should_call_vlm else None,
            vlm_enabled=bool(enable_vlm),
            vlm_model_name=vlm_model_name,
        )
        if vlm_semantics.get("status") not in {"disabled", "not_configured", "skipped_no_keyframe", "skipped_budget_exhausted"}:
            groups_vlm_reviewed += 1
        elif enable_vlm and vlm_client and yolo_passed and groups_vlm_reviewed >= vlm_budget:
            vlm_semantics = {
                **vlm_semantics,
                "status": "skipped_budget_exhausted",
                "model": vlm_model_name,
                "reason": "max_vlm_groups_reached",
            }
        vlm_status_counts[str(vlm_semantics.get("status") or "unknown")] += 1

        pipeline_status = _pipeline_status(yolo_recheck, vlm_semantics)
        if not yolo_passed:
            groups_rejected += 1

        for row in rows:
            row.update(
                {
                    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                    "pipeline_flow": list(PIPELINE_FLOW),
                    "pipeline_stage": "frontend_review_gate" if yolo_passed else "blocked_by_yolo_recheck",
                    "pipeline_status": pipeline_status,
                    "review_gate_policy": FRONTEND_REVIEW_GATE_POLICY,
                    "yolo_recheck": yolo_recheck,
                    "vlm_semantics": vlm_semantics,
                }
            )
            row["quality_reasons"] = _merged_reasons(
                row.get("quality_reasons"),
                [
                    "yolo_recheck_passed" if yolo_passed else "yolo_recheck_failed",
                    f"vlm_{vlm_semantics.get('status') or 'unknown'}",
                    "frontend_review_gate_required",
                ],
            )
            if not yolo_passed:
                row["candidate_status"] = "rejected"
                row["review_status"] = "rejected"
                row["review_required"] = False

    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "pipeline_flow": list(PIPELINE_FLOW),
        "review_gate_policy": FRONTEND_REVIEW_GATE_POLICY,
        "candidate_count": len(candidate_rows),
        "group_count": len(grouped),
        "groups_rechecked": groups_rechecked,
        "groups_rejected_by_yolo_recheck": groups_rejected,
        "groups_waiting_frontend_review": sum(
            1 for rows in grouped.values() if any(str(row.get("candidate_status") or "pending") == "pending" for row in rows)
        ),
        "vlm_enabled": bool(enable_vlm),
        "vlm_configured": bool(vlm_client),
        "vlm_model": vlm_model_name,
        "max_vlm_groups": vlm_budget,
        "groups_vlm_reviewed": groups_vlm_reviewed,
        "yolo_status_counts": dict(yolo_status_counts),
        "vlm_status_counts": dict(vlm_status_counts),
    }


def _candidate_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("candidate_group_id") or "").strip()
        if not group_id:
            group_id = "|".join(
                str(row.get(key) or "")
                for key in ("action_name", "primary_object", "micro_segment_id", "view", "time_range_sec")
            )
        grouped.setdefault(group_id, []).append(row)
    return grouped


def _build_yolo_evidence_packet(
    *,
    group_id: str,
    rows: list[dict[str, Any]],
    micro: dict[str, Any],
    primary_object: str,
    view: str,
    valid_evidence: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    action_name = str(rows[0].get("action_name") or "")
    time_range = str(rows[0].get("time_range_sec") or "")
    allowed = _allowed_confirmed_objects(primary_object, valid_evidence)
    return {
        "candidate_group_id": group_id,
        "action_name": action_name,
        "primary_object": primary_object,
        "view": view or None,
        "micro_segment_id": str(micro.get("micro_segment_id") or rows[0].get("micro_segment_id") or ""),
        "time_range_sec": time_range,
        "yolo_recheck": {
            "valid_evidence_count": len(valid_evidence),
            "required_min_frames": PHYSICAL_EVIDENCE_MIN_FRAMES,
            "diagnostics": diagnostics,
        },
        "allowed_confirmed_objects": allowed,
        "top_detections": _top_detections(valid_evidence, allowed),
        "hand_object_interactions": _top_interactions(valid_evidence, primary_object),
        "instruction": "VLM may only confirm labels in allowed_confirmed_objects; unsupported visual guesses must be uncertain.",
    }


def _allowed_confirmed_objects(primary_object: str, evidence_rows: list[dict[str, Any]]) -> list[str]:
    labels = set(HAND_LABELS)
    if primary_object:
        labels.add(primary_object)
    for row in evidence_rows:
        for detection in row.get("detections") or []:
            if not isinstance(detection, dict):
                continue
            label = canonical_yolo_label(detection.get("label"))
            if label in HAND_LABELS or label == primary_object:
                labels.add(label)
    return sorted(label for label in labels if label)


def _top_detections(evidence_rows: list[dict[str, Any]], allowed_labels: list[str]) -> list[dict[str, Any]]:
    allowed = {canonical_yolo_label(label) for label in allowed_labels}
    detections: list[dict[str, Any]] = []
    for row in evidence_rows:
        for detection in row.get("detections") or []:
            if not isinstance(detection, dict):
                continue
            label = canonical_yolo_label(detection.get("label"))
            if label not in allowed:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": _safe_float(detection.get("confidence"), _safe_float(detection.get("score"), 0.0)),
                    "bbox": detection.get("bbox"),
                    "local_time_sec": row.get("local_time_sec"),
                    "view": evidence_view(row) or None,
                }
            )
    detections.sort(key=lambda item: (-_safe_float(item.get("confidence"), 0.0), str(item.get("label") or "")))
    return detections[:8]


def _top_interactions(evidence_rows: list[dict[str, Any]], primary_object: str) -> list[dict[str, Any]]:
    primary = canonical_yolo_label(primary_object)
    interactions: list[dict[str, Any]] = []
    for row in evidence_rows:
        for interaction in row.get("hand_object_interactions") or []:
            if not isinstance(interaction, dict):
                continue
            object_label = canonical_yolo_label(
                interaction.get("object_label") or interaction.get("target_label") or interaction.get("object")
            )
            if object_label != primary:
                continue
            interactions.append(
                {
                    "hand_label": canonical_yolo_label(interaction.get("hand_label") or "gloved_hand"),
                    "object_label": object_label,
                    "score": _safe_float(
                        interaction.get("score"),
                        _safe_float(interaction.get("interaction_score"), _safe_float(interaction.get("confidence"), 0.0)),
                    ),
                    "hand_bbox": interaction.get("hand_bbox"),
                    "object_bbox": interaction.get("object_bbox"),
                    "local_time_sec": row.get("local_time_sec"),
                    "view": evidence_view(row) or None,
                }
            )
    interactions.sort(key=lambda item: (-_safe_float(item.get("score"), 0.0), _safe_float(item.get("local_time_sec"), 0.0)))
    return interactions[:8]


def _run_constrained_vlm_review(
    *,
    group_id: str,
    rows: list[dict[str, Any]],
    evidence_packet: dict[str, Any],
    vlm_client: Any | None,
    vlm_enabled: bool,
    vlm_model_name: str | None,
) -> dict[str, Any]:
    representative = _representative_keyframe(rows)
    if not vlm_enabled:
        return {
            "status": "disabled",
            "model": vlm_model_name,
            "reason": "KEY_ACTION_ENABLE_VLM_ASSIST disabled",
            "evidence_packet": evidence_packet,
        }
    if vlm_client is None:
        return {
            "status": "not_configured",
            "model": vlm_model_name,
            "reason": "No VLM client configured; YOLO evidence still gates candidates.",
            "evidence_packet": evidence_packet,
        }
    if representative is None:
        return {
            "status": "skipped_no_keyframe",
            "model": vlm_model_name,
            "reason": "No representative keyframe available for VLM review.",
            "evidence_packet": evidence_packet,
        }

    prompt = _build_vlm_prompt(evidence_packet)
    try:
        try:
            description = vlm_client.describe_scene(str(representative), prompt=prompt, temperature=0.0)
        except TypeError:
            description = vlm_client.describe_scene(str(representative), prompt=prompt)
        payload = _extract_vlm_payload(description)
        normalized = _normalize_vlm_payload(payload, evidence_packet)
        return {
            **normalized,
            "model": getattr(description, "model", None) or vlm_model_name,
            "representative_keyframe": str(representative),
            "prompt_schema": "yolo_evidence_constrained_json.v1",
            "evidence_packet": evidence_packet,
        }
    except Exception as exc:
        return {
            "status": "error",
            "model": vlm_model_name,
            "representative_keyframe": str(representative),
            "error": str(exc),
            "evidence_packet": evidence_packet,
            "candidate_group_id": group_id,
        }


def _representative_keyframe(rows: list[dict[str, Any]]) -> Path | None:
    keyframes = [row for row in rows if str(row.get("asset_kind") or "") == "关键帧"]
    keyframes.sort(
        key=lambda row: (
            0 if row.get("recommended") else 1,
            {"peak": 0, "hold": 1, "contact": 2, "approach": 3, "release": 4}.get(str(row.get("frame_role") or ""), 9),
            _safe_float(row.get("frame_local_time_sec"), _safe_float(row.get("source_offset_sec"), 0.0)),
        )
    )
    for row in keyframes:
        path = Path(str(row.get("stored_file") or ""))
        if path.is_file():
            return path
    return None


def _build_vlm_prompt(evidence_packet: dict[str, Any]) -> str:
    packet = json.dumps(evidence_packet, ensure_ascii=False, indent=2)
    return (
        "You are assisting a YOLO-backed laboratory physical-action pipeline.\n"
        "YOLO is the source of truth for object labels and hand-object evidence.\n"
        "Return a JSON object only. Required schema:\n"
        "{\n"
        '  "description": "brief Chinese description of the physical action",\n'
        '  "physical_action": "same or more specific action name",\n'
        '  "confirmed_objects": ["labels copied only from allowed_confirmed_objects"],\n'
        '  "uncertain_objects": ["visible guesses not supported by YOLO evidence"],\n'
        '  "evidence_alignment": "aligned|partial|uncertain",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Rules:\n"
        "1. Do not invent confirmed objects. Confirm only labels listed in allowed_confirmed_objects.\n"
        "2. If a monitor, keyboard, cable, reflection, or background object looks like equipment, put it in uncertain_objects.\n"
        "3. Never relabel a background monitor as balance or another lab object.\n"
        "4. The final material still requires frontend human review before publishing.\n\n"
        f"YOLO_EVIDENCE_JSON:\n{packet}\n"
    )


def _extract_vlm_payload(description: Any) -> dict[str, Any]:
    raw_response = getattr(description, "raw_response", None)
    if isinstance(raw_response, dict):
        payload = dict(raw_response)
        nested = _json_payload_from_text(payload.get("description"))
        return nested or payload
    if isinstance(description, dict):
        payload = dict(description)
        nested = _json_payload_from_text(payload.get("description"))
        return nested or payload
    raw_text = getattr(description, "description", None)
    if raw_text is None:
        raw_text = str(description or "")
    if isinstance(raw_text, dict):
        return dict(raw_text)
    text = str(raw_text or "").strip()
    if not text:
        return {}
    parsed = _json_payload_from_text(text)
    if parsed:
        return parsed
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {"description": text}


def _json_payload_from_text(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    for candidate in (cleaned, text):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            continue
    return {}


def _normalize_vlm_payload(payload: dict[str, Any], evidence_packet: dict[str, Any]) -> dict[str, Any]:
    allowed = {canonical_yolo_label(label) for label in evidence_packet.get("allowed_confirmed_objects") or []}
    primary = canonical_yolo_label(evidence_packet.get("primary_object"))
    raw_objects = _string_list(
        payload.get("confirmed_objects")
        or payload.get("object_labels")
        or payload.get("detected_objects")
        or payload.get("objects")
    )
    confirmed: list[str] = []
    uncertain = _string_list(payload.get("uncertain_objects"))
    unsupported: list[str] = []
    for label in raw_objects:
        canonical = canonical_yolo_label(label)
        if canonical in allowed and canonical not in confirmed:
            confirmed.append(canonical)
        elif canonical and canonical not in unsupported:
            unsupported.append(canonical)
    for label in unsupported:
        if label not in uncertain:
            uncertain.append(label)

    has_primary = bool(primary and primary in confirmed)
    has_hand = any(label in HAND_LABELS for label in confirmed)
    if has_primary and has_hand:
        status = "aligned"
    elif has_primary or confirmed:
        status = "partial_yolo_alignment"
    else:
        status = "weak_but_yolo_preserved"

    alignment = str(payload.get("evidence_alignment") or "").strip().lower()
    if alignment == "uncertain":
        status = "uncertain_vlm_review"
    elif alignment == "partial" and status != "aligned":
        status = "partial_vlm_review"

    return {
        "status": status,
        "description": str(payload.get("description") or payload.get("scene_description") or "").strip(),
        "physical_action": str(payload.get("physical_action") or payload.get("action_name") or evidence_packet.get("action_name") or "").strip(),
        "confirmed_objects": confirmed,
        "uncertain_objects": uncertain,
        "unsupported_confirmed_objects": unsupported,
        "confidence": _safe_float(payload.get("confidence"), 0.0),
        "evidence_alignment": alignment or None,
    }


def _pipeline_status(yolo_recheck: dict[str, Any], vlm_semantics: dict[str, Any]) -> str:
    if yolo_recheck.get("status") != "passed":
        return "blocked_by_yolo_recheck"
    vlm_status = str(vlm_semantics.get("status") or "")
    if vlm_status in {"aligned", "partial_yolo_alignment", "aligned_vlm_review", "partial_vlm_review"}:
        return "vlm_assisted_yolo_recheck_passed"
    if vlm_status in {"disabled", "not_configured", "skipped_no_keyframe", "skipped_budget_exhausted"}:
        return "yolo_recheck_passed_frontend_review_required"
    return "yolo_recheck_passed_vlm_advisory_uncertain"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list | tuple | set):
        return [str(value)] if str(value).strip() else []
    return [str(item).strip() for item in value if str(item).strip()]


def _merged_reasons(current: Any, additions: list[str]) -> list[str]:
    values = _string_list(current)
    for item in additions:
        if item and item not in values:
            values.append(item)
    return values


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default
