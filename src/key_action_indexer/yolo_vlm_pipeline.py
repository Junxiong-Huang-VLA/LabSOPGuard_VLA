from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .physical_evidence import (
    PHYSICAL_EVIDENCE_MIN_FRAMES,
    evidence_view,
    valid_yolo_physical_evidence,
    yolo_physical_evidence_diagnostics,
)
from .schemas import read_jsonl, write_jsonl
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
    qwen_audit_rows: list[dict[str, Any]] = []

    for group_id, rows in grouped.items():
        first = rows[0]
        primary = canonical_yolo_label(first.get("primary_object"))
        view = str(first.get("view") or "")
        micro_id = str(first.get("micro_segment_id") or "")
        segment_level_candidate = str(first.get("candidate_source") or "") == "segment_level_key_action"
        micro = micro_by_id.get(micro_id, {})
        candidate_evidence = _candidate_source_yolo_evidence(rows)
        raw_evidence = candidate_evidence or [row for row in (micro.get("yolo_evidence") or []) if isinstance(row, dict)]
        scoped_evidence = [row for row in raw_evidence if not view or evidence_view(row) == view] or raw_evidence
        valid_evidence = valid_yolo_physical_evidence(scoped_evidence, primary)
        diagnostics = yolo_physical_evidence_diagnostics(scoped_evidence, primary)
        yolo_passed = bool(primary) and len(valid_evidence) >= PHYSICAL_EVIDENCE_MIN_FRAMES
        segment_review_required = bool(segment_level_candidate and not raw_evidence and first.get("segment_id"))
        sparse_review_required = bool(
            not yolo_passed
            and not segment_review_required
            and _group_has_sparse_yolo_evidence(rows)
        )
        yolo_recheck = {
            "status": (
                "segment_level_review_required"
                if segment_review_required
                else "passed"
                if yolo_passed
                else "sparse_yolo_review_required"
                if sparse_review_required
                else "failed"
            ),
            "primary_object": primary,
            "view": view or None,
            "micro_segment_id": micro_id or None,
            "segment_id": first.get("segment_id"),
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
        if str(vlm_semantics.get("status") or "") not in {
            "disabled",
            "not_configured",
            "skipped_no_keyframe",
            "skipped_budget_exhausted",
        }:
            qwen_audit_rows.append(_qwen_event_audit_row(group_id, rows, vlm_semantics))

        if segment_review_required:
            pipeline_status = "segment_level_key_action_review_required"
        elif sparse_review_required:
            pipeline_status = "sparse_yolo_evidence_frontend_review_required"
        else:
            pipeline_status = _pipeline_status(yolo_recheck, vlm_semantics)
        if not yolo_passed and not segment_review_required and not sparse_review_required:
            groups_rejected += 1

        for row in rows:
            row.update(
                {
                    "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                    "pipeline_flow": list(PIPELINE_FLOW),
                    "pipeline_stage": "frontend_review_gate" if yolo_passed or segment_review_required or sparse_review_required else "blocked_by_yolo_recheck",
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
            if segment_review_required:
                row["quality_reasons"] = _merged_reasons(
                    row.get("quality_reasons"),
                    [
                        "segment_level_key_action_candidate",
                        "micro_yolo_evidence_unavailable",
                        "manual_material_review_required",
                    ],
                )
            if sparse_review_required:
                row["quality_reasons"] = _merged_reasons(
                    row.get("quality_reasons"),
                    [
                        "sparse_yolo_evidence",
                        "valid_yolo_evidence_below_auto_threshold",
                        "manual_material_review_required",
                    ],
                )
                row["review_required"] = True
            if not yolo_passed and not segment_review_required and not sparse_review_required:
                row["candidate_status"] = "rejected"
                row["review_status"] = "rejected"
                row["review_required"] = False

    if enable_vlm:
        write_jsonl(session_root / "metadata" / "qwen_event_audits.jsonl", qwen_audit_rows)

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
        "qwen_audit_count": len(qwen_audit_rows),
        "yolo_status_counts": dict(yolo_status_counts),
        "vlm_status_counts": dict(vlm_status_counts),
    }


def _qwen_event_audit_row(
    group_id: str,
    rows: list[dict[str, Any]],
    vlm_semantics: dict[str, Any],
) -> dict[str, Any]:
    gates = [
        item
        for item in (vlm_semantics.get("evidence_packet") or {}).get("physical_event_gates") or []
        if isinstance(item, dict)
    ]
    statuses = [str(gate.get("status") or "") for gate in gates]
    passed_values = [
        (gate.get("hard_gate") or {}).get("passed")
        for gate in gates
        if isinstance(gate.get("hard_gate"), dict)
    ]
    times = [
        _safe_float(row.get("frame_local_time_sec"), _safe_float(row.get("source_offset_sec"), None))
        for row in rows
    ]
    times = [value for value in times if value is not None]
    return {
        "candidate_id": group_id,
        "event_type": str(rows[0].get("action_name") or rows[0].get("semantic_action") or "") if rows else "",
        "time_start": min(times) if times else 0.0,
        "time_end": max(times) if times else 0.0,
        "hard_gate_status": statuses[0] if len(statuses) == 1 else statuses,
        "hard_gate_passed": bool(gates) and all(value is True for value in passed_values) and len(passed_values) == len(gates),
        "qwen_decision": str(vlm_semantics.get("decision") or vlm_semantics.get("status") or "uncertain"),
        "should_write_confirmed_event": bool(vlm_semantics.get("should_write_confirmed_event") is True),
        "missing_evidence": list(vlm_semantics.get("missing_evidence") or []),
        "contradictions": list(vlm_semantics.get("contradictions") or []),
        "reason": str(vlm_semantics.get("reason") or "; ".join(vlm_semantics.get("semantic_review_notes") or []) or ""),
        "raw_response_excerpt": str(vlm_semantics.get("raw_response_excerpt") or "")[:500],
        "created_at": datetime.now(timezone.utc).isoformat(),
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


def _candidate_source_yolo_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        for item in row.get("source_yolo_evidence") or row.get("effective_yolo_evidence") or []:
            if not isinstance(item, dict):
                continue
            key = json.dumps(
                {
                    "view": item.get("view") or item.get("source_view"),
                    "time": item.get("time_sec") or item.get("local_time_sec"),
                    "frame": item.get("frame_index"),
                    "detections": [
                        detection.get("label")
                        for detection in item.get("detections") or []
                        if isinstance(detection, dict)
                    ],
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            if key in seen:
                continue
            seen.add(key)
            evidence.append(item)
    return evidence


def _group_has_sparse_yolo_evidence(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        if str(row.get("physical_evidence_mode") or "") != "sparse_yolo_interaction_review_required":
            continue
        try:
            usable_count = int(row.get("usable_yolo_evidence_count") or row.get("yolo_evidence_count") or 0)
        except Exception:
            usable_count = 0
        if usable_count > 0:
            return True
    return False


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
    gate_rows = [
        row.get("physical_event_gate")
        for row in rows
        if isinstance(row.get("physical_event_gate"), dict)
    ]
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
        "physical_event_gates": gate_rows,
        "allowed_confirmed_objects": allowed,
        "candidate_ids": _candidate_ids(rows),
        "yolo_evidence_refs": _yolo_evidence_refs(valid_evidence),
        "top_detections": _top_detections(valid_evidence, allowed),
        "hand_object_interactions": _top_interactions(valid_evidence, primary_object),
        "instruction": (
            "VLM may only audit YOLO/hard-gate evidence. Unsupported visual guesses must be uncertain. "
            "If any hard_gate.status is not confirmed, VLM must not accept or request confirmed event writeback."
        ),
    }


def _candidate_ids(rows: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if candidate_id and candidate_id not in values:
            values.append(candidate_id)
    return values


def _yolo_evidence_refs(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for row_index, row in enumerate(evidence_rows):
        if not isinstance(row, dict):
            continue
        frame_id = _evidence_frame_id(row)
        time_sec = _evidence_time_sec(row)
        view = evidence_view(row) or None
        for detection_index, detection in enumerate(row.get("detections") or []):
            if not isinstance(detection, dict):
                continue
            label = canonical_yolo_label(detection.get("label") or detection.get("name") or detection.get("raw_label"))
            if not label:
                continue
            confidence = _safe_float(
                detection.get("confidence"),
                _safe_float(detection.get("score"), _safe_float(detection.get("conf"), 0.0)),
            )
            ref_id = _yolo_ref_id(
                frame_id=frame_id,
                time_sec=time_sec,
                view=view,
                label=label,
                index=detection_index,
            )
            refs.append(
                {
                    "ref_id": ref_id or f"yolo_ref_{row_index}_{detection_index}",
                    "frame_id": frame_id,
                    "time_sec": time_sec,
                    "view": view,
                    "label": label,
                    "confidence": confidence,
                }
            )
    return refs[:32]


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
            "raw_response_excerpt": str(description)[:500],
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
        "You are a laboratory video physical-event auditor. You are not a free event generator.\n"
        "YOLO detections, hard_gate, track ids, timestamps, and evidence metrics are the source of truth.\n"
        "Return a JSON object only. Required schema:\n"
        "{\n"
        '  "decision": "accept|reject|uncertain",\n'
        '  "should_write_confirmed_event": false,\n'
        '  "description": "brief Chinese description of the physical action",\n'
        '  "physical_action": "same or more specific action name",\n'
        '  "semantic_action": "optional specific semantic action, backed by yolo_evidence_refs",\n'
        '  "corrected_primary_object": "optional YOLO-backed correction of primary_object",\n'
        '  "confirmed_objects": ["labels copied only from allowed_confirmed_objects"],\n'
        '  "uncertain_objects": ["visible guesses not supported by YOLO evidence"],\n'
        '  "yolo_evidence_refs": [{"ref_id": "copy from YOLO_EVIDENCE_JSON, or candidate_id/frame_id/time_sec/view/label/confidence"}],\n'
        '  "evidence_alignment": "aligned|partial|uncertain",\n'
        '  "confidence": 0.0\n'
        "}\n"
        "Hard rules:\n"
        "1. Do not invent confirmed objects. Confirm only labels listed in allowed_confirmed_objects.\n"
        "2. Do not treat object presence as object_move.\n"
        "3. Do not treat hand-object proximity as contact.\n"
        "4. Do not treat static liquid presence as liquid_transfer.\n"
        "5. Do not treat device presence as panel_operation.\n"
        "6. Do not treat container presence as container_state_change.\n"
        "7. If hard_gate.status is candidate, rejected, or uncertain, decision must be reject or uncertain and should_write_confirmed_event must be false.\n"
        "8. semantic_action and corrected_primary_object must cite yolo_evidence_refs from YOLO_EVIDENCE_JSON.\n"
        "9. Without valid YOLO evidence refs, corrected_primary_object is only a proposed/uncertain object and cannot overwrite primary_object.\n\n"
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
    audit_decision = str(payload.get("decision") or "").strip().lower()
    parse_failed = not bool(payload)
    if audit_decision not in {"accept", "reject", "uncertain"}:
        audit_decision = "uncertain" if payload else "reject"
    physical_event_gates = [
        item for item in evidence_packet.get("physical_event_gates") or [] if isinstance(item, dict)
    ]
    hard_gate_statuses = [str(item.get("status") or "") for item in physical_event_gates]
    hard_gate_allows_accept = (
        bool(physical_event_gates)
        and all(str(gate.get("status") or "") == "confirmed" for gate in physical_event_gates)
        and all((gate.get("hard_gate") or {}).get("passed") is True for gate in physical_event_gates)
    )
    if audit_decision == "accept" and not hard_gate_allows_accept:
        audit_decision = "uncertain"
    semantic_action = str(payload.get("semantic_action") or "").strip()
    corrected_primary = canonical_yolo_label(payload.get("corrected_primary_object"))
    yolo_refs, unsupported_refs = _normalize_vlm_yolo_refs(payload.get("yolo_evidence_refs"), evidence_packet)
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

    correction_supported = bool(corrected_primary and yolo_refs and _refs_support_label(yolo_refs, corrected_primary))
    review_notes: list[str] = []
    missing_evidence = _string_list(payload.get("missing_evidence"))
    if not physical_event_gates and "missing_hard_gate" not in missing_evidence:
        missing_evidence.append("missing_hard_gate")
    result = {
        "status": "parse_failed" if parse_failed else status,
        "decision": audit_decision,
        "should_write_confirmed_event": bool(
            audit_decision == "accept"
            and hard_gate_allows_accept
            and payload.get("should_write_confirmed_event") is True
        ),
        "description": str(payload.get("description") or payload.get("scene_description") or "").strip(),
        "physical_action": str(payload.get("physical_action") or payload.get("action_name") or evidence_packet.get("action_name") or "").strip(),
        "confirmed_objects": confirmed,
        "uncertain_objects": uncertain,
        "unsupported_confirmed_objects": unsupported,
        "confidence": _safe_float(payload.get("confidence"), 0.0),
        "evidence_alignment": alignment or None,
        "yolo_evidence_refs": yolo_refs,
        "unsupported_yolo_evidence_refs": unsupported_refs,
        "missing_evidence": missing_evidence,
        "contradictions": _string_list(payload.get("contradictions")),
        "hard_gate_statuses": hard_gate_statuses,
        "hard_gate_required": True,
        "hard_gate_allows_accept": hard_gate_allows_accept,
        "parse_failed": parse_failed,
        "primary_object_override_allowed": correction_supported,
    }
    if not hard_gate_allows_accept:
        review_notes.append("hard_gate_not_confirmed_qwen_upgrade_blocked")
    if parse_failed:
        review_notes.append("qwen_json_parse_failed")
    if semantic_action:
        if yolo_refs:
            result["semantic_action"] = semantic_action
        else:
            result["proposed_semantic_action"] = semantic_action
            review_notes.append("semantic_action_missing_yolo_evidence_refs")
    if corrected_primary:
        if correction_supported:
            result["corrected_primary_object"] = corrected_primary
        else:
            result["proposed_primary_object"] = corrected_primary
            if corrected_primary not in uncertain:
                uncertain.append(corrected_primary)
            review_notes.append("corrected_primary_object_missing_matching_yolo_evidence_refs")
    if unsupported_refs:
        review_notes.append("unsupported_yolo_evidence_refs")
    if review_notes:
        result["semantic_review_notes"] = review_notes
    return result


def _normalize_vlm_yolo_refs(value: Any, evidence_packet: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_refs = [item for item in evidence_packet.get("yolo_evidence_refs") or [] if isinstance(item, dict)]
    source_by_id = {str(item.get("ref_id") or ""): item for item in source_refs if item.get("ref_id")}
    candidate_ids = {str(item) for item in evidence_packet.get("candidate_ids") or [] if str(item)}
    accepted: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for raw_ref in _dict_list(value):
        normalized = _normalize_single_vlm_ref(raw_ref, source_by_id, source_refs, candidate_ids)
        if normalized:
            if normalized not in accepted:
                accepted.append(normalized)
        else:
            unsupported.append(dict(raw_ref))
    return accepted, unsupported


def _normalize_single_vlm_ref(
    raw_ref: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
    source_refs: list[dict[str, Any]],
    candidate_ids: set[str],
) -> dict[str, Any]:
    candidate_id = str(raw_ref.get("candidate_id") or raw_ref.get("candidate") or "").strip()
    if candidate_id and candidate_id in candidate_ids:
        return {"candidate_id": candidate_id, "source": "candidate_id"}

    ref_id = str(raw_ref.get("ref_id") or raw_ref.get("evidence_ref_id") or raw_ref.get("yolo_ref_id") or "").strip()
    if ref_id and ref_id in source_by_id:
        return dict(source_by_id[ref_id])

    label = canonical_yolo_label(raw_ref.get("label") or raw_ref.get("object_label") or raw_ref.get("primary_object"))
    if not label:
        return {}
    view = str(raw_ref.get("view") or raw_ref.get("source_view") or raw_ref.get("requested_view") or "").strip()
    frame_id = str(raw_ref.get("frame_id") or raw_ref.get("source_frame_id") or raw_ref.get("frame_index") or "").strip()
    time_sec = _evidence_time_sec(raw_ref)
    confidence = _optional_float(raw_ref.get("confidence") or raw_ref.get("score"))
    if not (view or frame_id or time_sec is not None or confidence is not None):
        return {}
    for source in source_refs:
        if canonical_yolo_label(source.get("label")) != label:
            continue
        if view and str(source.get("view") or "") != view:
            continue
        if frame_id and str(source.get("frame_id") or "") != frame_id:
            continue
        if time_sec is not None and source.get("time_sec") is not None and abs(float(source["time_sec"]) - time_sec) > 0.05:
            continue
        if confidence is not None and source.get("confidence") is not None and abs(float(source["confidence"]) - confidence) > 0.08:
            continue
        return dict(source)
    return {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        text = value.strip()
        return [{"candidate_id": text, "ref_id": text}] if text else []
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list | tuple | set):
        return []
    refs: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            refs.append(item)
            continue
        text = str(item or "").strip()
        if text:
            refs.append({"candidate_id": text, "ref_id": text})
    return refs


def _refs_support_label(refs: list[dict[str, Any]], label: str) -> bool:
    canonical = canonical_yolo_label(label)
    if not canonical:
        return False
    return any(canonical_yolo_label(ref.get("label")) == canonical or ref.get("candidate_id") for ref in refs)


def _evidence_frame_id(row: dict[str, Any]) -> str | None:
    for key in ("frame_id", "source_frame_id", "frame_index", "frame_number"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _evidence_time_sec(row: dict[str, Any]) -> float | None:
    for key in ("time_sec", "local_time_sec", "timestamp_sec", "source_offset_sec"):
        value = row.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _yolo_ref_id(*, frame_id: str | None, time_sec: float | None, view: str | None, label: str, index: int) -> str:
    pieces = [
        "yolo",
        str(view or "view"),
        str(frame_id or (f"{time_sec:.3f}" if time_sec is not None else "frame")),
        label,
        str(index),
    ]
    return ":".join(piece.replace(":", "_") for piece in pieces if piece)


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
