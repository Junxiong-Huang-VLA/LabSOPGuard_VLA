from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set


CANONICAL_ACTION_LABELS: Dict[str, str] = {
    "hand-bottle": "hand-bottle",
    "hand-balance": "hand-balance",
    "hand-spatula": "hand-spatula",
    "hand-paper": "hand-paper",
    "hand-container": "hand-container",
}


def enrich_material_best_scores(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = [dict(item) for item in items]
    return [enrich_material_best_score(item, peers=rows) for item in rows]


def enrich_material_best_score(item: Dict[str, Any], *, peers: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    scored = dict(item)
    breakdown = material_best_score_breakdown(scored, peers=peers)
    scored.update(
        {
            "best_score": breakdown["best_score"],
            "best_score_breakdown": breakdown,
            "best_score_formula": (
                "quality_score + yolo_evidence_weight + hand_object_interaction_weight "
                "+ multiview_support_weight + sop_phase_match_weight "
                "- duplicate_penalty - bad_visibility_penalty"
            ),
            "best_reason": material_best_reason(scored, breakdown),
        }
    )
    return scored


def material_best_score_breakdown(item: Dict[str, Any], *, peers: Optional[Iterable[Dict[str, Any]]] = None) -> Dict[str, Any]:
    quality = _quality_score(item)
    yolo_count = _yolo_evidence_count(item)
    interaction_score = _interaction_score(item)
    views = _views(item)
    sop_match = _sop_match(item)
    duplicate_penalty = _duplicate_penalty(item, peers or [])
    visibility_penalty = _bad_visibility_penalty(item)

    quality_component = round(quality * 0.45, 4)
    yolo_component = round(min(yolo_count, 6) / 6 * 0.18, 4)
    interaction_component = round(interaction_score * 0.17, 4)
    multiview_component = 0.10 if len(views) >= 2 else (0.05 if len(views) == 1 else 0.0)
    sop_component = 0.08 if sop_match == "full" else (0.04 if sop_match == "partial" else 0.0)
    raw_score = (
        quality_component
        + yolo_component
        + interaction_component
        + multiview_component
        + sop_component
        - duplicate_penalty
        - visibility_penalty
    )
    best_score = round(max(0.0, min(1.0, raw_score)), 4)
    return {
        "schema_version": "material_best_score.v2",
        "best_score": best_score,
        "raw_quality_score": item.get("quality_score"),
        "quality_score": quality,
        "quality_component": quality_component,
        "yolo_evidence_count": yolo_count,
        "yolo_evidence_weight": yolo_component,
        "hand_object_interaction_score": round(interaction_score, 4),
        "hand_object_interaction_weight": interaction_component,
        "view_count": len(views),
        "views": sorted(views),
        "multiview_support_weight": multiview_component,
        "sop_phase_match": sop_match,
        "sop_phase_match_weight": sop_component,
        "duplicate_penalty": duplicate_penalty,
        "bad_visibility_penalty": visibility_penalty,
    }


def material_best_reason(item: Dict[str, Any], breakdown: Dict[str, Any]) -> str:
    action = str(item.get("canonical_action_type") or item.get("event_type") or item.get("action_name") or "material")
    label = CANONICAL_ACTION_LABELS.get(action, action)
    obj = str(item.get("canonical_object") or item.get("primary_object") or "-")
    start = _safe_float(item.get("time_start", item.get("start_sec")), 0.0)
    end = _safe_float(item.get("time_end", item.get("end_sec")), start)
    view_text = ", ".join(str(view) for view in breakdown.get("views") or []) or "single evidence view"
    reasons = [
        f"{label} representative for {obj}",
        f"quality {breakdown.get('quality_score', 0):.2f}",
        f"{int(breakdown.get('yolo_evidence_count') or 0)} YOLO evidence frames",
        f"view support: {view_text}",
        f"time window {start:.2f}-{end:.2f}s",
        f"SOP phase: {item.get('sop_phase') or '-'}",
    ]
    penalties: List[str] = []
    if float(breakdown.get("duplicate_penalty") or 0) > 0:
        penalties.append("near-duplicate penalty applied")
    if float(breakdown.get("bad_visibility_penalty") or 0) > 0:
        penalties.append("visibility/access penalty applied")
    if penalties:
        reasons.append("; ".join(penalties))
    return "; ".join(reasons)


def _payload(item: Dict[str, Any]) -> Dict[str, Any]:
    payload = item.get("payload")
    return payload if isinstance(payload, dict) else {}


def _nested_dict(item: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = item.get(key)
    if isinstance(value, dict):
        return value
    payload_value = _payload(item).get(key)
    return payload_value if isinstance(payload_value, dict) else {}


def _quality_score(item: Dict[str, Any]) -> float:
    payload = _payload(item)
    extra = payload.get("extra") if isinstance(payload.get("extra"), dict) else {}
    score = _safe_float(item.get("quality_score", payload.get("quality_score", extra.get("quality_score"))), 0.0)
    if score > 1.5:
        score /= 100.0
    return round(max(0.0, min(1.0, score)), 4)


def _yolo_evidence_count(item: Dict[str, Any]) -> int:
    yolo = _nested_dict(item, "yolo_recheck")
    payload = _payload(item)
    values = [item.get("yolo_evidence_count"), yolo.get("valid_evidence_count"), payload.get("yolo_evidence_count")]
    diagnostics = yolo.get("diagnostics") if isinstance(yolo.get("diagnostics"), dict) else {}
    values.append(diagnostics.get("valid_evidence_count"))
    for value in values:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            continue
    return 0


def _interaction_score(item: Dict[str, Any]) -> float:
    yolo = _nested_dict(item, "yolo_recheck")
    vlm = _nested_dict(item, "vlm_semantics")
    candidates: List[float] = []
    for source in (yolo, vlm):
        interactions = source.get("hand_object_interactions")
        if not isinstance(interactions, list):
            packet = source.get("evidence_packet") if isinstance(source.get("evidence_packet"), dict) else {}
            interactions = packet.get("hand_object_interactions")
        for interaction in interactions or []:
            if isinstance(interaction, dict):
                candidates.append(_safe_float(interaction.get("score"), 0.0))
    if candidates:
        return max(0.0, min(1.0, max(candidates)))
    if str(yolo.get("status") or "").lower() == "passed" and _yolo_evidence_count(item) >= 2:
        return 0.70
    if str(vlm.get("status") or "").lower() == "aligned":
        return 0.65
    return 0.0


def _views(item: Dict[str, Any]) -> Set[str]:
    views: Set[str] = set()
    for value in (item.get("view"), item.get("camera_id"), item.get("stream_id")):
        if value:
            views.add(str(value))
    yolo = _nested_dict(item, "yolo_recheck")
    if yolo.get("view"):
        views.add(str(yolo["view"]))
    diagnostics = yolo.get("diagnostics") if isinstance(yolo.get("diagnostics"), dict) else {}
    for key in ("evidence_by_view", "valid_evidence_by_view"):
        by_view = diagnostics.get(key)
        if isinstance(by_view, dict):
            views.update(str(view) for view, count in by_view.items() if count)
    vlm = _nested_dict(item, "vlm_semantics")
    packet = vlm.get("evidence_packet") if isinstance(vlm.get("evidence_packet"), dict) else {}
    for detection in packet.get("top_detections") or []:
        if isinstance(detection, dict) and detection.get("view"):
            views.add(str(detection["view"]))
    return {view for view in views if view}


def _sop_match(item: Dict[str, Any]) -> str:
    if item.get("canonical_action_type") and item.get("canonical_object") and item.get("sop_phase"):
        return "full"
    if item.get("canonical_action_type") or item.get("canonical_object"):
        return "partial"
    return "none"


def _duplicate_penalty(item: Dict[str, Any], peers: Iterable[Dict[str, Any]]) -> float:
    canonical = str(item.get("canonical_action_type") or "")
    if not canonical:
        return 0.0
    item_id = str(item.get("item_id") or item.get("candidate_id") or item.get("material_id") or id(item))
    start = _safe_float(item.get("time_start", item.get("start_sec")), 0.0)
    asset_kind = str(item.get("asset_kind") or item.get("material_type") or _asset_kind(item) or "")
    quality = _quality_score(item)
    for peer in peers:
        peer_id = str(peer.get("item_id") or peer.get("candidate_id") or peer.get("material_id") or id(peer))
        if peer_id == item_id:
            continue
        if str(peer.get("canonical_action_type") or "") != canonical:
            continue
        peer_kind = str(peer.get("asset_kind") or peer.get("material_type") or _asset_kind(peer) or "")
        if peer_kind != asset_kind:
            continue
        peer_start = _safe_float(peer.get("time_start", peer.get("start_sec")), 0.0)
        if abs(peer_start - start) <= 2.5 and _quality_score(peer) >= quality:
            return 0.08
    return 0.0


def _asset_kind(item: Dict[str, Any]) -> str:
    if item.get("clip_url") or item.get("clip_file_path"):
        return "clip"
    if item.get("preview_url") or item.get("frame_path"):
        return "frame"
    return ""


def _bad_visibility_penalty(item: Dict[str, Any]) -> float:
    penalty = 0.0
    if item.get("exists") is False or item.get("material_exists") is False or item.get("clip_exists") is False:
        penalty += 0.12
    if _quality_score(item) < 0.55:
        penalty += 0.06
    yolo_status = str(_nested_dict(item, "yolo_recheck").get("status") or "").lower()
    vlm_status = str(_nested_dict(item, "vlm_semantics").get("status") or "").lower()
    if yolo_status in {"failed", "rejected"}:
        penalty += 0.08
    if vlm_status in {"error", "uncertain_vlm_review", "weak_but_yolo_preserved"}:
        penalty += 0.04
    return round(min(0.20, penalty), 4)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
