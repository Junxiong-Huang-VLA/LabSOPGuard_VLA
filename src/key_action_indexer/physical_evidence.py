from __future__ import annotations



from collections import Counter

from dataclasses import asdict, dataclass

from typing import Any



from .physical_event_gate import gate_hand_object_contact
from .yolo_detector import bbox_iou, canonical_yolo_label, filter_implausible_detections, find_hand_object_interactions





HAND_LABELS = frozenset({"gloved_hand", "hand"})

PHYSICAL_EVIDENCE_MIN_FRAMES = 2

EXPLICIT_OBJECT_DETECTION_REQUIRED = frozenset({"balance"})

INTERACTION_OBJECT_PROXY_MIN_SCORE = 0.75





@dataclass(frozen=True)

class PhysicalEvidenceRule:

    object_confidence: float = 0.50

    hand_confidence: float = 0.45

    interaction_score: float = 0.50

    max_object_area_ratio: float = 0.28

    max_hand_area_ratio: float = 0.35





DEFAULT_PHYSICAL_EVIDENCE_RULE = PhysicalEvidenceRule()



PHYSICAL_EVIDENCE_RULES: dict[str, PhysicalEvidenceRule] = {

    "balance": PhysicalEvidenceRule(

        object_confidence=0.55,

        hand_confidence=0.45,

        interaction_score=0.55,

        max_object_area_ratio=0.22,

    ),

    "spatula": PhysicalEvidenceRule(

        object_confidence=0.50,

        hand_confidence=0.45,

        interaction_score=0.50,

        max_object_area_ratio=0.12,

    ),

    "pipette": PhysicalEvidenceRule(

        object_confidence=0.50,

        hand_confidence=0.45,

        interaction_score=0.50,

        max_object_area_ratio=0.14,

    ),

    "pipette_tip": PhysicalEvidenceRule(

        object_confidence=0.50,

        hand_confidence=0.45,

        interaction_score=0.50,

        max_object_area_ratio=0.08,

    ),

    "paper": PhysicalEvidenceRule(

        object_confidence=0.50,

        hand_confidence=0.45,

        interaction_score=0.55,

        max_object_area_ratio=0.25,

    ),

    "sample_bottle": PhysicalEvidenceRule(object_confidence=0.50, interaction_score=0.50),

    "sample_bottle_blue": PhysicalEvidenceRule(object_confidence=0.50, interaction_score=0.50),

    "reagent_bottle": PhysicalEvidenceRule(object_confidence=0.50, interaction_score=0.50),

    "beaker": PhysicalEvidenceRule(object_confidence=0.45, hand_confidence=0.55, interaction_score=0.32),

    "container": PhysicalEvidenceRule(object_confidence=0.50, interaction_score=0.50),

    "tube": PhysicalEvidenceRule(object_confidence=0.50, interaction_score=0.50),

    "tube_cap": PhysicalEvidenceRule(object_confidence=0.50, interaction_score=0.50),

}





def physical_evidence_policy_summary() -> dict[str, Any]:

    return {

        "min_valid_frames_per_view": PHYSICAL_EVIDENCE_MIN_FRAMES,

        "default_rule": asdict(DEFAULT_PHYSICAL_EVIDENCE_RULE),

        "object_rules": {label: asdict(rule) for label, rule in sorted(PHYSICAL_EVIDENCE_RULES.items())},

    }





def evidence_view(evidence: dict[str, Any]) -> str:

    return str(evidence.get("view") or evidence.get("source_view") or evidence.get("requested_view") or "").strip()





def valid_yolo_physical_evidence(

    evidence_rows: list[dict[str, Any]],

    primary_object: str,

) -> list[dict[str, Any]]:

    primary = canonical_yolo_label(primary_object)

    valid: list[dict[str, Any]] = []

    for row in evidence_rows:

        if not isinstance(row, dict):

            continue

        ok, _reasons = validate_yolo_physical_evidence(row, primary)

        if ok:

            valid.append(row)

    return valid





def yolo_physical_evidence_diagnostics(

    evidence_rows: list[dict[str, Any]],

    primary_object: str,

) -> dict[str, Any]:

    primary = canonical_yolo_label(primary_object)

    reason_counts: Counter[str] = Counter()

    valid_count = 0

    views: Counter[str] = Counter()

    valid_views: Counter[str] = Counter()

    for row in evidence_rows:

        if not isinstance(row, dict):

            reason_counts["invalid_evidence_row"] += 1

            continue

        view = evidence_view(row) or "unknown"

        views[view] += 1

        ok, reasons = validate_yolo_physical_evidence(row, primary)

        if ok:

            valid_count += 1

            valid_views[view] += 1

        else:

            reason_counts.update(reasons or ["invalid_physical_evidence"])

    return {

        "primary_object": primary,

        "total_evidence_count": len([row for row in evidence_rows if isinstance(row, dict)]),

        "valid_evidence_count": valid_count,

        "evidence_by_view": dict(views),

        "valid_evidence_by_view": dict(valid_views),

        "invalid_reason_counts": dict(reason_counts),

        "rule": asdict(_rule_for_object(primary)),

    }





def validate_yolo_physical_evidence(evidence: dict[str, Any], primary_object: str) -> tuple[bool, list[str]]:

    primary = canonical_yolo_label(primary_object)

    if not primary:

        return False, ["missing_primary_object"]

    rule = _rule_for_object(primary)

    detections = _filtered_detections(evidence)

    interactions = _interactions(evidence, primary, detections)

    reasons: list[str] = []



    interaction = _best_interaction(interactions)

    object_detection = _best_detection(detections, {primary})

    if object_detection is None:

        proxy_ok, proxy_reasons = _interaction_object_proxy_reasons(interaction, evidence, primary, rule)

        if proxy_ok:

            object_detection = {

                "label": primary,

                "confidence": _confidence(interaction or {}, keys=("score", "interaction_score", "confidence")),

                "bbox": (interaction or {}).get("object_bbox"),

                "source": "hand_object_interaction_proxy",

            }

        else:

            reasons.extend(proxy_reasons or ["missing_primary_object_detection"])

    else:

        object_confidence = _confidence(object_detection)

        if object_confidence < rule.object_confidence:

            reasons.append("primary_object_confidence_below_threshold")

        bbox_reasons = _bbox_reasons(object_detection.get("bbox"), evidence, rule.max_object_area_ratio)

        reasons.extend(f"primary_object_{reason}" for reason in bbox_reasons)



    if interaction is None:

        reasons.append("missing_matching_hand_object_interaction")

    else:

        interaction_score = _confidence(interaction, keys=("score", "interaction_score", "confidence"))

        if interaction_score < rule.interaction_score:

            reasons.append("interaction_score_below_threshold")

        if not _valid_bbox(interaction.get("hand_bbox")):

            reasons.append("missing_hand_bbox")

        if not _valid_bbox(interaction.get("object_bbox")):

            reasons.append("missing_object_bbox")



    hand_detection = _best_detection(detections, HAND_LABELS)

    if hand_detection is not None:

        hand_confidence = _confidence(hand_detection)

        if hand_confidence < rule.hand_confidence:

            reasons.append("hand_confidence_below_threshold")

        bbox_reasons = _bbox_reasons(hand_detection.get("bbox"), evidence, rule.max_hand_area_ratio)

        reasons.extend(f"hand_{reason}" for reason in bbox_reasons)

    elif interaction is None or not _valid_bbox(interaction.get("hand_bbox")):

        reasons.append("missing_hand_detection_or_interaction_bbox")

    if not reasons:

        overlap = _confidence(interaction or {}, keys=("iou", "bbox_overlap"))
        coverage = _confidence(interaction or {}, keys=("object_coverage_by_hand",))

        if overlap <= 0 and interaction and _valid_bbox(interaction.get("hand_bbox")) and _valid_bbox(interaction.get("object_bbox")):

            overlap = bbox_iou(interaction["hand_bbox"], interaction["object_bbox"])
            coverage = max(coverage, _object_coverage(interaction["hand_bbox"], interaction["object_bbox"]))

        distance = (interaction or {}).get("distance_px")

        near_only = bool(distance is not None and overlap <= 0)

        gate = gate_hand_object_contact(

            event_candidate={"event_type": "hand_object_interaction", "object_labels": [primary]},

            frame_evidence_list=[evidence],

            external_observation={

                "has_hand": hand_detection is not None or bool(interaction and _valid_bbox(interaction.get("hand_bbox"))),

                "has_object": object_detection is not None or bool(interaction and _valid_bbox(interaction.get("object_bbox"))),

                "contact_frames": PHYSICAL_EVIDENCE_MIN_FRAMES if overlap > 0 else 0,

                "continuous_contact_frames": PHYSICAL_EVIDENCE_MIN_FRAMES if overlap > 0 else 0,

                "overlap_frames": PHYSICAL_EVIDENCE_MIN_FRAMES if overlap > 0 else 0,

                "max_iou": overlap,

                "max_object_coverage_by_hand": coverage,

                "min_distance_px": distance,

                "near_only": near_only,

                "contact_type": "overlap" if overlap > 0 else None,

            },

        )

        if gate.get("status") != "confirmed":

            reasons.extend(gate.get("reject_reasons") or ["physical_event_gate_not_confirmed"])



    return not reasons, sorted(set(reasons))





def _rule_for_object(primary_object: str) -> PhysicalEvidenceRule:

    return PHYSICAL_EVIDENCE_RULES.get(canonical_yolo_label(primary_object), DEFAULT_PHYSICAL_EVIDENCE_RULE)





def _detections(evidence: dict[str, Any]) -> list[dict[str, Any]]:

    return [item for item in (evidence.get("detections") or []) if isinstance(item, dict)]





def _filtered_detections(evidence: dict[str, Any]) -> list[dict[str, Any]]:

    width, height = _frame_size(evidence)

    detections, _ignored = filter_implausible_detections(
        _detections(evidence),
        frame_width=int(width) if width else None,
        frame_height=int(height) if height else None,
        source_view=evidence_view(evidence),
    )

    return detections





def _interactions(evidence: dict[str, Any], primary_object: str, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:

    primary = canonical_yolo_label(primary_object)

    rows = [item for item in (evidence.get("hand_object_interactions") or []) if isinstance(item, dict)]
    if not rows and _valid_bbox(evidence.get("hand_box")) and _valid_bbox(evidence.get("object_box")):
        rows = [
            {
                "hand_label": evidence.get("hand_label") or "gloved_hand",
                "object_label": evidence.get("object_label") or evidence.get("primary_object") or primary,
                "score": evidence.get("interaction_score") or evidence.get("score") or evidence.get("confidence"),
                "hand_bbox": evidence.get("hand_box"),
                "object_bbox": evidence.get("object_box"),
                "source": "legacy_evidence_box_fields",
            }
        ]

    width, height = _frame_size(evidence)
    recomputed = find_hand_object_interactions(
        detections,
        frame_width=int(width) if width else None,
        frame_height=int(height) if height else None,
        source_view=evidence_view(evidence),
        min_interaction_score=0.1,
    )
    kept: list[dict[str, Any]] = [
        item
        for item in recomputed
        if canonical_yolo_label(item.get("object_label") or item.get("target_label") or item.get("object")) == primary
    ]
    for row in rows:
        if canonical_yolo_label(row.get("object_label") or row.get("target_label") or row.get("object")) != primary:
            continue
        if not _interaction_matches_filtered_detections(row, primary, detections):
            continue
        kept.append(row)
    return kept





def _interaction_matches_filtered_detections(
    interaction: dict[str, Any],
    primary_object: str,
    detections: list[dict[str, Any]],
) -> bool:

    hand_bbox = interaction.get("hand_bbox")
    object_bbox = interaction.get("object_bbox")
    if not _valid_bbox(hand_bbox) or not _valid_bbox(object_bbox):
        return False

    if not _bbox_matches_any_detection(hand_bbox, detections, HAND_LABELS):
        return False

    primary = canonical_yolo_label(primary_object)
    object_detection = _best_detection(detections, {primary})
    if object_detection is None:
        return primary not in EXPLICIT_OBJECT_DETECTION_REQUIRED

    return _bbox_matches_any_detection(object_bbox, detections, {primary})





def _bbox_matches_any_detection(
    bbox: Any,
    detections: list[dict[str, Any]],
    labels: frozenset[str] | set[str],
    *,
    min_iou: float = 0.72,
) -> bool:

    if not _valid_bbox(bbox):
        return False
    normalized_labels = {canonical_yolo_label(label) for label in labels}
    for detection in detections:
        if canonical_yolo_label(detection.get("label")) not in normalized_labels:
            continue
        other = detection.get("bbox")
        if _valid_bbox(other) and bbox_iou(list(bbox[:4]), list(other[:4])) >= min_iou:
            return True
    return False


def _object_coverage(hand_bbox: Any, object_bbox: Any) -> float:

    try:

        hx1, hy1, hx2, hy2 = [float(v) for v in hand_bbox]

        ox1, oy1, ox2, oy2 = [float(v) for v in object_bbox]

    except Exception:

        return 0.0

    ix1, iy1 = max(hx1, ox1), max(hy1, oy1)

    ix2, iy2 = min(hx2, ox2), min(hy2, oy2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    object_area = max(1.0, max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1))

    return inter / object_area





def _best_detection(detections: list[dict[str, Any]], labels: frozenset[str] | set[str]) -> dict[str, Any] | None:

    matches = [item for item in detections if canonical_yolo_label(item.get("label")) in labels]

    if not matches:

        return None

    return max(matches, key=_confidence)





def _best_interaction(interactions: list[dict[str, Any]]) -> dict[str, Any] | None:

    if not interactions:

        return None

    return max(interactions, key=lambda item: _confidence(item, keys=("score", "interaction_score", "confidence")))





def _interaction_object_proxy_reasons(

    interaction: dict[str, Any] | None,

    evidence: dict[str, Any],

    primary_object: str,

    rule: PhysicalEvidenceRule,

) -> tuple[bool, list[str]]:

    if primary_object in EXPLICIT_OBJECT_DETECTION_REQUIRED:

        return False, ["missing_primary_object_detection"]

    if interaction is None:

        return False, ["missing_primary_object_detection"]

    score = _confidence(interaction, keys=("score", "interaction_score", "confidence"))

    if score < max(rule.interaction_score, INTERACTION_OBJECT_PROXY_MIN_SCORE):

        return False, ["interaction_object_proxy_score_below_threshold"]

    bbox_reasons = _bbox_reasons(interaction.get("object_bbox"), evidence, rule.max_object_area_ratio)

    if bbox_reasons:

        return False, [f"interaction_object_proxy_{reason}" for reason in bbox_reasons]

    return True, []





def _confidence(item: dict[str, Any], *, keys: tuple[str, ...] = ("confidence", "score")) -> float:

    for key in keys:

        value = item.get(key)

        if value is None:

            continue

        try:

            return max(0.0, min(1.0, float(value)))

        except Exception:

            continue

    return 0.0





def _bbox_reasons(bbox: Any, evidence: dict[str, Any], max_area_ratio: float) -> list[str]:

    if not _valid_bbox(bbox):

        return ["bbox_missing_or_invalid"]

    area_ratio = _bbox_area_ratio(bbox, evidence)

    if area_ratio is None:

        return []

    if area_ratio <= 0.0:

        return ["bbox_missing_or_invalid"]

    if area_ratio > max_area_ratio:

        return ["bbox_area_too_large"]

    return []





def _valid_bbox(bbox: Any) -> bool:

    if not isinstance(bbox, list | tuple) or len(bbox) < 4:

        return False

    try:

        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]

    except Exception:

        return False

    return x2 > x1 and y2 > y1





def _bbox_area_ratio(bbox: Any, evidence: dict[str, Any]) -> float | None:

    width, height = _frame_size(evidence)

    if not width or not height:

        return None

    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]

    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return area / max(float(width) * float(height), 1e-6)





def _frame_size(evidence: dict[str, Any]) -> tuple[float | None, float | None]:

    width = _numeric(evidence.get("frame_width") or evidence.get("source_width") or evidence.get("video_width"))

    height = _numeric(evidence.get("frame_height") or evidence.get("source_height") or evidence.get("video_height"))

    if width and height:

        return width, height

    max_x = 0.0

    max_y = 0.0

    for bbox in _all_bboxes(evidence):

        if not _valid_bbox(bbox):

            continue

        max_x = max(max_x, float(bbox[0]), float(bbox[2]))

        max_y = max(max_y, float(bbox[1]), float(bbox[3]))

    if max_x <= 0.0 or max_y <= 0.0:

        return None, None

    if max_x <= 960 and max_y <= 540:

        return 960.0, 540.0

    if max_x <= 1280 and max_y <= 720:

        return 1280.0, 720.0

    if max_x <= 1920 and max_y <= 1080:

        return 1920.0, 1080.0

    return max_x, max_y





def _all_bboxes(evidence: dict[str, Any]) -> list[Any]:

    boxes: list[Any] = []
    boxes.append(evidence.get("hand_box"))
    boxes.append(evidence.get("object_box"))

    for detection in _detections(evidence):

        boxes.append(detection.get("bbox"))

    for interaction in [item for item in (evidence.get("hand_object_interactions") or []) if isinstance(item, dict)]:

        boxes.append(interaction.get("hand_bbox"))

        boxes.append(interaction.get("object_bbox"))

    return boxes





def _numeric(value: Any) -> float | None:

    try:

        if value is None:

            return None

        number = float(value)

    except Exception:

        return None

    return number if number > 0.0 else None
