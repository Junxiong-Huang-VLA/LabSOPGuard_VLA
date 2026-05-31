from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any, Mapping, Sequence


SEMANTIC_SCHEMA_VERSION = "material_semantic_action.v1"
WEIGHING_OPERATION = "weighing_operation"

BALANCE_LABELS = {"balance", "scale"}
BALANCE_PANEL_LABELS = {"balance", "scale", "panel", "display"}
BOTTLE_OPERATION_LABELS = {
    "reagent_bottle",
    "reagent_bottle_open",
    "bottle_cap",
    "sample_bottle",
    "sample_bottle_blue",
    "bottle",
    "vial",
}
PAPER_OPERATION_LABELS = {"paper", "weighing_paper"}
REAGENT_BOTTLE_OPERATION_TITLE = "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c"
WEIGHING_PAPER_OPERATION_TITLE = "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c"
BALANCE_PANEL_OPERATION_TITLE = "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c"
WEIGHING_MANIPULATED_LABELS = {
    "paper",
    "weighing_paper",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
}

WEIGHING_TAXONOMY = {
    "canonical_action_type": "hand-balance",
    "canonical_object": "balance",
    "sop_phase": "balance-weighing",
    "interaction_family": "hand-object",
}

LABEL_ALIASES = {
    "glove": "gloved_hand",
    "gloves": "gloved_hand",
    "gloved_hands": "gloved_hand",
    "person_hand": "hand",
    "hands": "hand",
    "electronic_balance": "balance",
    "scale": "balance",
    "weighing_scale": "balance",
    "weighing_paper": "weighing_paper",
    "filter_paper": "paper",
    "spoon": "spatula",
    "scoop": "spatula",
    "sample_vial": "sample_bottle",
    "vial": "sample_bottle",
    "blue_sample_bottle": "sample_bottle_blue",
    "reagent": "reagent_bottle",
    "reagent_bottle_open": "reagent_bottle_open",
    "open_reagent_bottle": "reagent_bottle_open",
    "opened_reagent_bottle": "reagent_bottle_open",
    "bottle_cap": "bottle_cap",
    "cap": "bottle_cap",
    "panel": "panel",
    "display_panel": "panel",
    "equipment_panel": "panel",
    "test_tube": "tube",
}

OBJECT_DISPLAY_NAMES = {
    "reagent_bottle": "\u8bd5\u5242\u74f6",
    "reagent_bottle_open": "\u8bd5\u5242\u74f6",
    "bottle_cap": "\u8bd5\u5242\u74f6",
    "sample_bottle": "\u6837\u54c1\u74f6",
    "sample_bottle_blue": "\u6837\u54c1\u74f6",
    "bottle": "\u74f6\u5b50",
    "balance": "\u5929\u5e73",
    "panel": "\u8bbe\u5907\u9762\u677f",
    "display": "\u8bbe\u5907\u9762\u677f",
    "spatula": "\u836f\u5319",
    "paper": "\u79f0\u91cf\u7eb8",
    "weighing_paper": "\u79f0\u91cf\u7eb8",
    "pipette": "\u79fb\u6db2\u67aa",
    "pipette_tip": "\u79fb\u6db2\u67aa\u67aa\u5934",
    "tube": "\u8bd5\u7ba1",
    "beaker": "\u70e7\u676f",
    "container": "\u5bb9\u5668",
    "flask": "\u70e7\u74f6",
    "magnetic_stir_bar": "\u78c1\u529b\u6405\u62cc\u5b50",
    "magnetic_stirrer": "\u78c1\u529b\u6405\u62cc\u5668",
}


def enhance_material_semantics(
    candidate: Mapping[str, Any] | None,
    *,
    micro: Mapping[str, Any] | None = None,
    evidence_rows: Sequence[Mapping[str, Any]] | None = None,
    primary_object: Any = None,
    secondary_objects: Sequence[Any] | None = None,
    action_name: Any = None,
    vlm_semantics: Mapping[str, Any] | None = None,
    min_balance_frames: int = 2,
) -> dict[str, Any]:
    """Return semantic fields that separate manipulated object from lab context.

    The function is deliberately data-only and does not decode video.  It keeps
    the raw YOLO primary object intact while allowing a stable instrument
    context, such as a balance, to drive the user-facing action title and
    retrieval taxonomy.
    """

    candidate = candidate or {}
    micro = micro or {}
    interaction = _mapping(micro.get("interaction"))
    evidence = [
        dict(row)
        for row in (
            evidence_rows
            if evidence_rows is not None
            else candidate.get("yolo_evidence") or micro.get("yolo_evidence") or []
        )
        if isinstance(row, Mapping)
    ]
    raw_primary = _canon(
        primary_object
        or candidate.get("primary_object")
        or micro.get("primary_object")
        or interaction.get("primary_object")
        or candidate.get("canonical_object")
    )
    secondary = _unique(
        [
            *(_canon(item) for item in (secondary_objects or [])),
            *(_canon(item) for item in _as_list(candidate.get("secondary_objects"))),
            *(_canon(item) for item in _as_list(micro.get("secondary_objects"))),
            *(_canon(item) for item in _as_list(interaction.get("secondary_objects"))),
        ]
    )
    manipulated = _initial_manipulated(raw_primary, secondary)
    vlm_patch = _evidence_supported_vlm_patch(vlm_semantics or {})
    weighing = infer_weighing_context(
        evidence,
        primary_object=raw_primary,
        secondary_objects=secondary,
        min_balance_frames=min_balance_frames,
    )

    fields: dict[str, Any] = {
        "semantic_schema_version": SEMANTIC_SCHEMA_VERSION,
        "raw_primary_object": raw_primary or None,
        "manipulated_object": manipulated or None,
        "instrument_context": None,
        "semantic_action": vlm_patch.get("semantic_action"),
        "semantic_action_source": vlm_patch.get("semantic_action_source"),
        "corrected_primary_object": vlm_patch.get("corrected_primary_object"),
        "semantic_correction_status": vlm_patch.get("semantic_correction_status") or "not_corrected",
        "semantic_evidence_refs": vlm_patch.get("yolo_evidence_refs") or [],
        "display_title": str(action_name or "").strip() or _hand_object_display_title(raw_primary),
    }

    if weighing["matched"]:
        manipulated_objects = _unique(weighing["manipulated_objects"] or [manipulated])
        title_objects = "/".join(_unique_display_names(manipulated_objects)) or _display_name(manipulated)
        business_title = _business_display_title(manipulated_objects[0] if manipulated_objects else manipulated)
        fields.update(
            {
                **WEIGHING_TAXONOMY,
                "instrument_context": "balance",
                "semantic_action": "weighing_operation",
                "semantic_action_source": "weighing_priority_rule",
                "semantic_correction_status": "rule_supported_by_yolo",
                "corrected_primary_object": "balance",
                "display_title": business_title or f"天平称量-手与{title_objects}",
                "semantic_rule": "balance_stable_and_hand_manipulated_object_in_balance_region",
                "semantic_confidence": weighing["confidence"],
                "semantic_evidence_refs": weighing["evidence_refs"],
                "weighing_context": weighing,
            }
        )
    elif vlm_patch.get("corrected_primary_object"):
        corrected = str(vlm_patch["corrected_primary_object"])
        fields.update(
            {
                "display_title": _semantic_title(
                    semantic_action=vlm_patch.get("semantic_action"),
                    corrected_primary_object=corrected,
                    manipulated_object=manipulated,
                    fallback=str(action_name or ""),
                ),
            }
        )

    return {key: value for key, value in fields.items() if value not in (None, "", [])}


def enrich_semantic_action(
    candidate: Mapping[str, Any] | None,
    micro_segment: Mapping[str, Any] | None = None,
    yolo_window_evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    min_balance_frames: int = 2,
    min_balance_frame_ratio: float = 0.5,
) -> dict[str, Any]:
    """Return a candidate copy enriched with semantic action fields.

    This wrapper is the stable dict-in/dict-out API for candidate semantic core
    rules. It keeps the raw primary object and separates manipulated object from
    instrument context without touching video files or backend services.
    """

    del min_balance_frame_ratio
    enriched = deepcopy(dict(candidate or {}))
    micro = deepcopy(dict(micro_segment or {}))
    evidence_rows = _semantic_evidence_rows(enriched, micro, yolo_window_evidence)
    fields = enhance_material_semantics(
        enriched,
        micro=micro,
        evidence_rows=evidence_rows,
        min_balance_frames=min_balance_frames,
    )
    enriched.update(fields)

    raw_for_context = fields.get("raw_primary_object") or _canon(
        enriched.get("primary_object")
        or _mapping(enriched.get("interaction")).get("primary_object")
        or micro.get("primary_object")
        or _mapping(micro.get("interaction")).get("primary_object")
    )
    weighing_context = _mapping(fields.get("weighing_context")) or infer_weighing_context(
        evidence_rows,
        primary_object=raw_for_context,
        secondary_objects=_semantic_secondary_objects(enriched, micro),
        min_balance_frames=min_balance_frames,
    )
    balance_stable = bool(weighing_context.get("balance_stable"))
    balance_region_interaction = int(weighing_context.get("interaction_ref_count") or 0) > 0
    manipulated_objects = _unique(weighing_context.get("manipulated_objects") or [fields.get("manipulated_object")])
    manipulated_object = (
        manipulated_objects[0]
        if enriched.get("semantic_action") == WEIGHING_OPERATION and manipulated_objects
        else fields.get("manipulated_object") or (manipulated_objects[0] if manipulated_objects else None)
    )
    if weighing_context.get("matched"):
        enriched["semantic_action"] = WEIGHING_OPERATION
        enriched["semantic_action_source"] = enriched.get("semantic_action_source") or "weighing_priority_rule"
        enriched["semantic_correction_status"] = enriched.get("semantic_correction_status") or "rule_supported_by_yolo"
        enriched["instrument_context"] = "balance"
        title_objects = "/".join(_unique_display_names(manipulated_objects)) or _display_name(manipulated_object)
        business_title = _business_display_title(manipulated_objects[0] if manipulated_objects else manipulated_object)
        enriched["display_title"] = business_title or enriched.get("display_title") or f"天平称量-手与{title_objects}"
        if manipulated_objects:
            manipulated_object = manipulated_objects[0]
    if not enriched.get("semantic_action"):
        enriched["semantic_action"] = str(
            enriched.get("action_type")
            or _mapping(enriched.get("text_description")).get("action_type")
            or ("hand_object_interaction" if manipulated_object else "unknown_operation")
        )

    reasons: list[str] = []
    if balance_stable:
        reasons.append("stable_balance_context")
    if balance_region_interaction:
        reasons.append("hand_manipulated_object_in_balance_region")
    if enriched.get("semantic_action") == WEIGHING_OPERATION:
        reasons.append("weighing_operation_priority_rule")

    enriched["raw_primary_object"] = fields.get("raw_primary_object") or _canon(
        enriched.get("primary_object")
        or _mapping(enriched.get("interaction")).get("primary_object")
        or micro.get("primary_object")
        or _mapping(micro.get("interaction")).get("primary_object")
    ) or None
    enriched["manipulated_object"] = manipulated_object
    enriched["manipulated_objects"] = manipulated_objects or ([manipulated_object] if manipulated_object else [])
    enriched["instrument_context"] = fields.get("instrument_context") if fields.get("instrument_context") else None
    enriched["semantic_reasons"] = _unique([*(_as_list(enriched.get("semantic_reasons"))), *reasons])
    enriched["semantic_evidence"] = {
        "schema_version": SEMANTIC_SCHEMA_VERSION,
        "raw_primary_object": enriched["raw_primary_object"],
        "balance_stable": balance_stable,
        "balance_frame_count": int(weighing_context.get("balance_frame_count") or 0),
        "balance_region_interaction": balance_region_interaction,
        "balance_region_objects": manipulated_objects,
        "manipulated_object_candidates": manipulated_objects,
        "rule": "weighing_operation_priority_v1",
    }
    return enriched


def enrich_semantic_candidate(
    candidate: Mapping[str, Any] | None,
    micro_segment: Mapping[str, Any] | None = None,
    yolo_window_evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return enrich_semantic_action(candidate, micro_segment, yolo_window_evidence, **kwargs)


def infer_weighing_context(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    primary_object: Any = None,
    secondary_objects: Sequence[Any] | None = None,
    min_balance_frames: int = 2,
) -> dict[str, Any]:
    """Detect a weighing context from YOLO evidence only."""

    rows = [row for row in evidence_rows if isinstance(row, Mapping)]
    primary = _canon(primary_object)
    secondary = [_canon(item) for item in (secondary_objects or [])]
    balance_detections = _detections_for_labels(rows, BALANCE_LABELS)
    balance_stable = _stable_detections(balance_detections, min_frames=min_balance_frames)
    manipulated_candidates = _unique(label for label in [primary, *secondary] if label in WEIGHING_MANIPULATED_LABELS)
    region_labels: list[str] = []
    interaction_refs: list[dict[str, Any]] = []

    for row in rows:
        balance_boxes = [det["bbox"] for det in _row_detections_for_labels(row, BALANCE_LABELS) if _bbox(det.get("bbox"))]
        if not balance_boxes:
            continue
        region = _union_bbox(balance_boxes)
        if region is None:
            continue
        expanded = _expand_bbox(region, ratio=0.45)
        row_refs = _manipulated_refs_in_region(row, expanded)
        if primary in WEIGHING_MANIPULATED_LABELS and primary not in {ref["label"] for ref in row_refs}:
            row_refs.extend(_detection_refs_in_region(row, {primary}, expanded))
        for ref in row_refs:
            label = _canon(ref.get("label"))
            if label and label not in region_labels:
                region_labels.append(label)
            if label and label not in manipulated_candidates:
                manipulated_candidates.append(label)
            interaction_refs.append(
                {
                    **_evidence_ref(row),
                    "label": label,
                    "region_label": "balance",
                    "bbox": ref.get("bbox"),
                    "score": ref.get("score"),
                    "evidence_kind": ref.get("evidence_kind"),
                }
            )

    matched = bool(balance_stable["stable"] and interaction_refs)
    confidence = min(
        0.98,
        0.55
        + min(0.25, 0.05 * int(balance_stable["frame_count"]))
        + min(0.18, 0.04 * len(interaction_refs)),
    )
    return {
        "matched": matched,
        "instrument_context": "balance" if balance_stable["stable"] else None,
        "balance_stable": balance_stable["stable"],
        "balance_frame_count": balance_stable["frame_count"],
        "manipulated_objects": region_labels or manipulated_candidates,
        "interaction_ref_count": len(interaction_refs),
        "confidence": round(confidence if matched else 0.0, 3),
        "evidence_refs": interaction_refs[:12],
    }


def _evidence_supported_vlm_patch(vlm_semantics: Mapping[str, Any]) -> dict[str, Any]:
    semantic_action = str(vlm_semantics.get("semantic_action") or "").strip()
    corrected = _canon(vlm_semantics.get("corrected_primary_object"))
    refs = [dict(ref) for ref in _as_list(vlm_semantics.get("yolo_evidence_refs")) if isinstance(ref, Mapping)]
    if corrected and refs:
        return {
            "semantic_action": semantic_action or None,
            "semantic_action_source": "vlm_yolo_evidence_refs",
            "corrected_primary_object": corrected,
            "semantic_correction_status": "evidence_supported",
            "yolo_evidence_refs": refs[:12],
        }
    if corrected:
        return {
            "semantic_action": semantic_action or None,
            "semantic_action_source": "vlm_without_required_evidence_refs",
            "proposed_corrected_primary_object": corrected,
            "semantic_correction_status": "missing_yolo_evidence_refs",
        }
    return {"semantic_action": semantic_action or None, "semantic_action_source": "vlm_advisory" if semantic_action else None}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _semantic_evidence_rows(
    candidate: Mapping[str, Any],
    micro: Mapping[str, Any],
    yolo_window_evidence: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_rows_from_any(candidate.get("yolo_evidence")))
    rows.extend(_rows_from_any(micro.get("yolo_evidence")))

    top_level_interactions: list[dict[str, Any]] = []
    if isinstance(yolo_window_evidence, Mapping):
        for key in ("hand_object_interactions", "interactions"):
            top_level_interactions.extend(
                dict(item)
                for item in _as_list(yolo_window_evidence.get(key))
                if isinstance(item, Mapping)
            )
    window_rows = _rows_from_any(yolo_window_evidence)
    top_level_interaction_bundle = (
        isinstance(yolo_window_evidence, Mapping)
        and top_level_interactions
        and yolo_window_evidence.get("detections") is None
        and not any(key in yolo_window_evidence for key in ("frame_index", "frame_id", "local_time_sec", "time_sec", "view", "source_view"))
    )
    if not top_level_interaction_bundle:
        rows.extend(window_rows)
    if top_level_interactions:
        if rows:
            target = dict(rows[-1])
            target["hand_object_interactions"] = [
                *[dict(item) for item in _as_list(target.get("hand_object_interactions")) if isinstance(item, Mapping)],
                *top_level_interactions,
            ]
            rows[-1] = target
        else:
            rows.append({"hand_object_interactions": top_level_interactions})
    return rows


def _semantic_secondary_objects(candidate: Mapping[str, Any], micro: Mapping[str, Any]) -> list[str]:
    interaction = _mapping(micro.get("interaction"))
    return _unique(
        [
            *_as_list(candidate.get("secondary_objects")),
            *_as_list(micro.get("secondary_objects")),
            *_as_list(interaction.get("secondary_objects")),
        ]
    )


def _rows_from_any(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("frames", "rows", "frame_rows", "yolo_evidence", "evidence"):
            rows = _rows_from_any(value.get(key))
            if rows:
                return rows
        if value.get("detections") is not None or value.get("hand_object_interactions") is not None:
            return [dict(value)]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    return []


def _canon(value: Any) -> str:
    label = _canonical_label(value)
    text = str(label or value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text == "scale":
        return "balance"
    if text == "weighingpaper":
        return "weighing_paper"
    return text


def _canonical_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return LABEL_ALIASES.get(text, text)


def _display_name(label: Any) -> str:
    normalized = _canon(label)
    if normalized == "paper" or normalized == "weighing_paper":
        return "称量纸"
    if normalized == "spatula":
        return "药匙"
    if normalized == "balance":
        return "天平"
    if normalized in {"panel", "display"}:
        return "设备面板"
    return OBJECT_DISPLAY_NAMES.get(normalized, normalized or "对象").split("/")[0].strip()


def _hand_object_display_title(label: Any) -> str | None:
    normalized = _canon(label)
    if not normalized:
        return None
    business_title = _business_display_title(normalized)
    if business_title:
        return business_title
    name = _display_name(normalized)
    return f"手部与{name}操作" if name else None


def _business_display_title(label: Any) -> str | None:
    normalized = _canon(label)
    if normalized in BOTTLE_OPERATION_LABELS:
        return REAGENT_BOTTLE_OPERATION_TITLE
    if normalized in PAPER_OPERATION_LABELS:
        return WEIGHING_PAPER_OPERATION_TITLE
    if normalized in BALANCE_PANEL_LABELS:
        return BALANCE_PANEL_OPERATION_TITLE
    return None


def _unique_display_names(labels: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for label in labels:
        name = _display_name(label)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _semantic_title(
    *,
    semantic_action: Any,
    corrected_primary_object: str,
    manipulated_object: str,
    fallback: str,
) -> str | None:
    action = str(semantic_action or "").strip()
    if action == "weighing_operation" or corrected_primary_object == "balance":
        return f"天平称量-手与{_display_name(manipulated_object or corrected_primary_object)}"
    if action:
        return f"{action}-手与{_display_name(manipulated_object or corrected_primary_object)}"
    return fallback or None


def _first_manipulated(labels: Sequence[str]) -> str:
    for label in labels:
        if _canon(label) in WEIGHING_MANIPULATED_LABELS:
            return _canon(label)
    return ""


def _initial_manipulated(raw_primary: str, secondary: Sequence[str]) -> str:
    primary = _canon(raw_primary)
    if primary in BALANCE_LABELS:
        return _first_manipulated(secondary) or primary
    return primary or _first_manipulated(secondary)


def _unique(values: Sequence[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = _canon(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _detections_for_labels(rows: Sequence[Mapping[str, Any]], labels: set[str]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for row in rows:
        detections.extend(_row_detections_for_labels(row, labels))
    return detections


def _row_detections_for_labels(row: Mapping[str, Any], labels: set[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for detection in row.get("detections") or []:
        if not isinstance(detection, Mapping):
            continue
        label = _canon(detection.get("label") or detection.get("class_name") or detection.get("name"))
        if label in labels:
            result.append({**dict(detection), "label": label, "row": row})
    return result


def _stable_detections(detections: Sequence[Mapping[str, Any]], *, min_frames: int) -> dict[str, Any]:
    boxed = [det for det in detections if _bbox(det.get("bbox"))]
    if len(boxed) < min_frames:
        return {"stable": False, "frame_count": len(boxed), "reason": "not_enough_balance_frames"}
    centers = [_center(_bbox(det.get("bbox")) or [0, 0, 0, 0]) for det in boxed]
    sizes = [max(1.0, ((_bbox(det.get("bbox")) or [0, 0, 0, 0])[2] - (_bbox(det.get("bbox")) or [0, 0, 0, 0])[0])) for det in boxed]
    median_size = max(1.0, median(sizes))
    cx = median([center[0] for center in centers])
    cy = median([center[1] for center in centers])
    max_drift = max((((center[0] - cx) ** 2 + (center[1] - cy) ** 2) ** 0.5 for center in centers), default=0.0)
    drift_ratio = max_drift / median_size
    return {
        "stable": drift_ratio <= 0.75,
        "frame_count": len(boxed),
        "center_drift_ratio": round(drift_ratio, 4),
    }


def _manipulated_refs_in_region(row: Mapping[str, Any], region: list[float]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for interaction in row.get("hand_object_interactions") or []:
        if not isinstance(interaction, Mapping):
            continue
        label = _canon(interaction.get("object_label") or interaction.get("target_label") or interaction.get("object"))
        if label not in WEIGHING_MANIPULATED_LABELS:
            continue
        if _explicit_balance_region(interaction):
            refs.append(
                {
                    "label": label,
                    "bbox": _bbox(interaction.get("object_bbox")) or _bbox(interaction.get("bbox")) or _detection_bbox(row, label),
                    "score": _float(interaction.get("score"), _float(interaction.get("interaction_score"), 0.0)),
                    "evidence_kind": "explicit_hand_object_interaction_in_balance_region",
                }
            )
            continue
        bbox = _bbox(interaction.get("object_bbox")) or _bbox(interaction.get("bbox"))
        if bbox is None:
            bbox = _detection_bbox(row, label)
        if bbox is None or not _bbox_touches_region(bbox, region):
            continue
        refs.append(
            {
                "label": label,
                "bbox": bbox,
                "score": _float(interaction.get("score"), _float(interaction.get("interaction_score"), 0.0)),
                "evidence_kind": "hand_object_interaction",
            }
        )
    return refs


def _explicit_balance_region(interaction: Mapping[str, Any]) -> bool:
    for key in ("in_balance_region", "within_balance_region", "object_in_balance_region", "balance_region", "on_balance"):
        if interaction.get(key) is True:
            return True
    for key in ("region", "object_region", "context_object", "instrument_context", "instrument_label", "near_object"):
        if _canon(interaction.get(key)) in {"balance", "balance_area", "balance_region"}:
            return True
    return False


def _detection_refs_in_region(row: Mapping[str, Any], labels: set[str], region: list[float]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for det in _row_detections_for_labels(row, labels):
        bbox = _bbox(det.get("bbox"))
        if bbox is None or not _bbox_touches_region(bbox, region):
            continue
        refs.append(
            {
                "label": _canon(det.get("label")),
                "bbox": bbox,
                "score": _float(det.get("confidence"), _float(det.get("score"), 0.0)),
                "evidence_kind": "object_detection_in_balance_region",
            }
        )
    return refs


def _evidence_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_id": row.get("frame_id") or row.get("frame_index"),
        "local_time_sec": row.get("local_time_sec") if row.get("local_time_sec") is not None else row.get("time_sec"),
        "view": row.get("view") or row.get("source_view"),
        "candidate_id": row.get("candidate_id"),
        "micro_segment_id": row.get("micro_segment_id"),
    }


def _detection_bbox(row: Mapping[str, Any], label: str) -> list[float] | None:
    for det in _row_detections_for_labels(row, {label}):
        bbox = _bbox(det.get("bbox"))
        if bbox is not None:
            return bbox
    return None


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _center(bbox: Sequence[float]) -> tuple[float, float]:
    return ((float(bbox[0]) + float(bbox[2])) / 2.0, (float(bbox[1]) + float(bbox[3])) / 2.0)


def _union_bbox(boxes: Sequence[Sequence[float]]) -> list[float] | None:
    valid = [_bbox(box) for box in boxes]
    valid = [box for box in valid if box is not None]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def _expand_bbox(bbox: Sequence[float], *, ratio: float) -> list[float]:
    width = float(bbox[2]) - float(bbox[0])
    height = float(bbox[3]) - float(bbox[1])
    return [
        float(bbox[0]) - width * ratio,
        float(bbox[1]) - height * ratio,
        float(bbox[2]) + width * ratio,
        float(bbox[3]) + height * ratio,
    ]


def _bbox_touches_region(bbox: Sequence[float], region: Sequence[float]) -> bool:
    cx, cy = _center(bbox)
    if float(region[0]) <= cx <= float(region[2]) and float(region[1]) <= cy <= float(region[3]):
        return True
    x1 = max(float(bbox[0]), float(region[0]))
    y1 = max(float(bbox[1]), float(region[1]))
    x2 = min(float(bbox[2]), float(region[2]))
    y2 = min(float(bbox[3]), float(region[3]))
    return x2 > x1 and y2 > y1


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "SEMANTIC_SCHEMA_VERSION",
    "WEIGHING_OPERATION",
    "enhance_material_semantics",
    "enrich_semantic_action",
    "enrich_semantic_candidate",
    "infer_weighing_context",
]
