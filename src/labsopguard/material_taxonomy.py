from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


STANDARD_ACTION_TAXONOMY: Dict[str, Dict[str, str]] = {
    "hand-bottle": {
        "canonical_object": "bottle",
        "sop_phase": "reagent-bottle-handling",
        "display_label": "Hand-bottle",
    },
    "hand-balance": {
        "canonical_object": "balance",
        "sop_phase": "balance-weighing",
        "display_label": "Hand-balance",
    },
    "hand-spatula": {
        "canonical_object": "spatula",
        "sop_phase": "solid-transfer",
        "display_label": "Hand-spatula",
    },
    "hand-paper": {
        "canonical_object": "paper",
        "sop_phase": "weighing-paper-prep",
        "display_label": "Hand-paper",
    },
    "hand-container": {
        "canonical_object": "container",
        "sop_phase": "container-handling",
        "display_label": "Hand-container",
    },
}

_ACTION_ALIASES: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "hand-balance",
        (
            "balance",
            "electronic balance",
            "analytical balance",
            "scale",
            "weighing scale",
            "天平",
            "电子天平",
            "電子天平",
        ),
    ),
    (
        "hand-spatula",
        (
            "spatula",
            "scoopula",
            "scoop",
            "药匙",
            "藥匙",
            "勺",
        ),
    ),
    (
        "hand-paper",
        (
            "weighing paper",
            "weigh paper",
            "paper",
            "filter paper",
            "称量纸",
            "稱量紙",
            "纸",
            "紙",
        ),
    ),
    (
        "hand-bottle",
        (
            "reagent_bottle",
            "sample_bottle",
            "wash_bottle",
            "bottle",
            "vial",
            "试剂瓶",
            "試劑瓶",
            "样品瓶",
            "樣品瓶",
            "瓶",
        ),
    ),
    (
        "hand-container",
        (
            "container",
            "beaker",
            "flask",
            "tube",
            "centrifuge tube",
            "dish",
            "tray",
            "cup",
            "weighing boat",
            "容器",
            "烧杯",
            "燒杯",
            "离心管",
            "離心管",
            "称量舟",
            "稱量舟",
        ),
    ),
]


def _flatten(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, dict):
        for nested in value.values():
            yield from _flatten(nested)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            yield from _flatten(nested)
        return
    text = str(value).strip()
    if text:
        yield text


def _norm(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text)


def _match_action_type(values: Iterable[Any]) -> str | None:
    texts = [_norm(value) for value in values if str(value or "").strip()]
    if not texts:
        return None
    for action_type, aliases in _ACTION_ALIASES:
        for text in texts:
            for alias in aliases:
                normalized_alias = _norm(alias)
                if text == normalized_alias or normalized_alias in text:
                    return action_type
    return None


def _explicit_object_values(row: Dict[str, Any]) -> List[Any]:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
    yolo = row.get("yolo_recheck") if isinstance(row.get("yolo_recheck"), dict) else {}
    source = row.get("source_container") if isinstance(row.get("source_container"), dict) else {}
    target = row.get("target_container") if isinstance(row.get("target_container"), dict) else {}
    values: List[Any] = [
        row.get("canonical_object"),
        row.get("primary_object"),
        row.get("object_label"),
        row.get("primary_object_family"),
        interaction.get("primary_object"),
        interaction.get("primary_object_family"),
        yolo.get("primary_object"),
        source.get("class_name"),
        source.get("display_name"),
        target.get("class_name"),
        target.get("display_name"),
        action.get("primary_object"),
    ]
    values.extend(row.get("object_labels") or [])
    if isinstance(action.get("objects"), list):
        values.extend(action.get("objects") or [])
    return values


def _fallback_text_values(row: Dict[str, Any]) -> List[Any]:
    values: List[Any] = [
        row.get("canonical_action_type"),
        row.get("action_name"),
        row.get("event_type"),
        row.get("display_name"),
        row.get("stable_name"),
        row.get("stored_filename"),
        row.get("file_name"),
    ]
    values.extend(row.get("actions") or [])
    vlm = row.get("vlm_semantics") if isinstance(row.get("vlm_semantics"), dict) else {}
    values.extend(
        [
            vlm.get("physical_action"),
            vlm.get("description"),
            vlm.get("confirmed_objects"),
        ]
    )
    packet = vlm.get("evidence_packet") if isinstance(vlm.get("evidence_packet"), dict) else {}
    values.extend(
        [
            packet.get("allowed_confirmed_objects"),
            packet.get("hand_object_interactions"),
        ]
    )
    return list(_flatten(values))


def canonicalize_material_action(row: Dict[str, Any]) -> Dict[str, str]:
    """Return canonical taxonomy fields for hand-object material records."""

    action_type = _match_action_type(_explicit_object_values(row))
    if action_type is None:
        action_type = _match_action_type(_fallback_text_values(row))
    if action_type is None:
        action_type = "hand-container"

    spec = STANDARD_ACTION_TAXONOMY[action_type]
    return {
        "canonical_action_type": action_type,
        "canonical_object": spec["canonical_object"],
        "sop_phase": spec["sop_phase"],
        "interaction_family": "hand-object",
    }


def enrich_material_taxonomy(row: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(row)
    enriched.update(canonicalize_material_action(enriched))
    return enriched
