from __future__ import annotations

from typing import Any

HAND_TOUCH_BOTTLE = "\u624b\u78b0\u74f6\u5b50"
REAGENT_BOTTLE_OPERATION = "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c"
WEIGHING_PAPER_OPERATION = "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c"
BALANCE_PANEL_OPERATION = "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c"
WEIGHING = "\u79f0\u91cf"
USE_SPATULA = "\u4f7f\u7528\u522e\u52fa"
PIPETTING_ZH = "\u52a0\u6837"
SAMPLE_HANDLING = "sample_handling"
RECORDING = "recording"

BOTTLE_BUSINESS_OBJECTS = ["bottle", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "sample_bottle", "sample_bottle_blue"]
WEIGHING_PAPER_OBJECTS = ["paper", "weighing_paper"]
BALANCE_PANEL_OBJECTS = ["balance", "panel", "display"]


ACTION_ALIASES: dict[str, dict[str, list[str]]] = {
    HAND_TOUCH_BOTTLE: {
        "objects": ["bottle", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "sample_bottle", "sample_bottle_blue"],
        "interaction_types": [
            "hand_object_contact",
            "hand_bottle_contact",
            "hand_reagent_bottle_contact",
            "hand_sample_bottle_contact",
        ],
        "keywords": [
            "\u624b",
            "\u78b0",
            "\u63a5\u89e6",
            "\u74f6\u5b50",
            "\u8bd5\u5242\u74f6",
            "\u74f6\u76d6",
            "\u5f00\u74f6",
            REAGENT_BOTTLE_OPERATION,
            "\u6837\u54c1\u74f6",
        ],
    },
    WEIGHING: {
        "objects": ["balance", "paper", "weighing_paper", "bottle", "reagent_bottle", "sample_bottle", "sample_bottle_blue", "spatula"],
        "interaction_types": ["hand_balance_contact", "weighing_related"],
        "keywords": ["\u79f0\u91cf", "\u5929\u5e73", "\u8bfb\u6570", "\u91cd\u91cf", "\u8d28\u91cf"],
    },
    WEIGHING_PAPER_OPERATION: {
        "objects": WEIGHING_PAPER_OBJECTS,
        "interaction_types": ["hand_paper_contact", "hand_object_contact"],
        "keywords": ["weighing paper", "paper", "\u79f0\u91cf\u7eb8", WEIGHING_PAPER_OPERATION],
    },
    BALANCE_PANEL_OPERATION: {
        "objects": BALANCE_PANEL_OBJECTS,
        "interaction_types": ["equipment_panel_operation", "panel_reading", "hand_balance_contact"],
        "keywords": ["balance panel", "equipment panel", "panel", "display", "\u5929\u5e73", "\u9762\u677f", "\u8bbe\u5907\u9762\u677f", BALANCE_PANEL_OPERATION],
    },
    USE_SPATULA: {
        "objects": ["spatula"],
        "interaction_types": ["hand_spatula_contact", "spatula_sampling"],
        "keywords": ["\u522e\u52fa", "\u836f\u5319", "\u52fa", "\u53d6\u6837", "\u4f7f\u7528\u522e\u52fa"],
    },
    PIPETTING_ZH: {
        "objects": ["pipette", "pipette_tip", "tube", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "bottle", "sample_bottle", "sample_bottle_blue"],
        "interaction_types": ["pipetting", "sample_adding", "hand_pipette_contact", "hand_pipette_tip_contact"],
        "keywords": ["\u52a0\u6837", "\u79fb\u6db2", "\u79fb\u6db2\u67aa", "\u5fae\u5347", "\u52a0\u5165", "\u6ef4\u52a0"],
    },
    SAMPLE_HANDLING: {
        "objects": ["sample_bottle", "sample_bottle_blue", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "spatula", "paper", "bottle"],
        "interaction_types": ["hand_object_contact", "hand_sample_bottle_contact", "hand_reagent_bottle_contact", "hand_spatula_contact", "hand_paper_contact"],
        "keywords": ["sample", "sample handling", "sample bottle", "spatula", "weighing paper", "\u6837\u54c1", "\u6837\u54c1\u5904\u7406", "\u53d6\u6837", "\u6837\u54c1\u74f6"],
    },
    RECORDING: {
        "objects": ["balance", "panel", "paper", "display", "notebook"],
        "interaction_types": ["recording", "panel_reading", "hand_paper_contact"],
        "keywords": ["recording", "record", "readout", "balance readout", "reading", "\u8bb0\u5f55", "\u8bfb\u6570", "\u5929\u5e73\u8bfb\u6570"],
    },
}


OBJECT_DISPLAY_NAMES = {
    "reagent_bottle": "\u8bd5\u5242\u74f6",
    "reagent_bottle_open": "\u8bd5\u5242\u74f6",
    "bottle_cap": "\u8bd5\u5242\u74f6",
    "sample_bottle": "\u6837\u54c1\u74f6",
    "sample_bottle_blue": "\u84dd\u8272\u6837\u54c1\u74f6",
    "bottle": "\u74f6\u5b50",
    "balance": "\u5929\u5e73",
    "panel": "\u8bbe\u5907\u9762\u677f",
    "display": "\u8bbe\u5907\u9762\u677f",
    "spatula": "\u522e\u52fa / \u836f\u5319",
    "pipette": "\u79fb\u6db2\u67aa",
    "pipette_tip": "\u79fb\u6db2\u67aa\u67aa\u5934",
    "tube": "\u8bd5\u7ba1",
    "magnetic_stir_bar": "\u78c1\u529b\u6405\u62cc\u5b50",
    "magnetic_stirrer": "\u78c1\u529b\u6405\u62cc\u5668",
}


OBJECT_CHINESE_ALIASES: dict[str, list[str]] = {
    "gloved_hand": ["\u6234\u624b\u5957\u7684\u624b", "\u624b\u5957\u624b"],
    "hand": ["\u624b", "\u624b\u90e8"],
    "sample_bottle": ["\u6837\u54c1\u74f6", "\u6837\u54c1\u5bb9\u5668"],
    "sample_bottle_blue": ["\u84dd\u76d6\u6837\u54c1\u74f6", "\u84dd\u8272\u6837\u54c1\u74f6"],
    "reagent_bottle": ["\u8bd5\u5242\u74f6", "\u8bd5\u5242\u5bb9\u5668"],
    "reagent_bottle_open": ["\u8bd5\u5242\u74f6", "\u5df2\u6253\u5f00\u8bd5\u5242\u74f6", "\u5f00\u74f6", REAGENT_BOTTLE_OPERATION],
    "bottle_cap": ["\u74f6\u76d6", "\u8bd5\u5242\u74f6\u76d6", "\u8bd5\u5242\u74f6", REAGENT_BOTTLE_OPERATION],
    "balance": ["\u5929\u5e73", "\u7535\u5b50\u5929\u5e73", "\u79f0\u91cf\u4eea"],
    "spatula": ["\u522e\u52fa", "\u836f\u5319", "\u53d6\u6837\u52fa"],
    "pipette": ["\u79fb\u6db2\u67aa", "\u79fb\u6db2\u5668"],
    "pipette_tip": ["\u79fb\u6db2\u67aa\u5934", "\u67aa\u5934"],
    "paper": ["\u79f0\u91cf\u7eb8", "\u8bb0\u5f55\u7eb8", WEIGHING_PAPER_OPERATION],
    "weighing_paper": ["\u79f0\u91cf\u7eb8", WEIGHING_PAPER_OPERATION],
    "panel": ["\u8bbe\u5907\u9762\u677f", "\u5929\u5e73\u9762\u677f", "\u9762\u677f", BALANCE_PANEL_OPERATION],
    "display": ["\u8bbe\u5907\u9762\u677f", "\u663e\u793a\u9762\u677f", "\u8bfb\u6570", BALANCE_PANEL_OPERATION],
    "tube": ["\u8bd5\u7ba1", "\u79bb\u5fc3\u7ba1"],
    "magnetic_stir_bar": ["\u78c1\u529b\u6405\u62cc\u5b50", "\u6405\u62cc\u5b50", "\u78c1\u5b50"],
    "magnetic_stirrer": ["\u78c1\u529b\u6405\u62cc\u5668", "\u6405\u62cc\u5668"],
    "bottle": ["\u74f6\u5b50", "\u6837\u54c1\u74f6", "\u8bd5\u5242\u74f6"],
}


ACTION_TYPES_BY_OBJECT = {
    "balance": "equipment_panel_operation",
    "panel": "equipment_panel_operation",
    "display": "equipment_panel_operation",
    "paper": "weighing_paper_operation",
    "weighing_paper": "weighing_paper_operation",
    "spatula": "spatula_interaction",
    "pipette": "pipetting",
    "pipette_tip": "pipetting",
    "tube": "tube_interaction",
    "reagent_bottle": "reagent_bottle_interaction",
    "reagent_bottle_open": "reagent_bottle_interaction",
    "bottle_cap": "reagent_bottle_interaction",
    "sample_bottle": "reagent_bottle_interaction",
    "sample_bottle_blue": "reagent_bottle_interaction",
    "bottle": "reagent_bottle_interaction",
    "magnetic_stir_bar": "stirring_operation",
    "magnetic_stirrer": "stirring_operation",
}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def chinese_aliases_for_label(label: Any) -> list[str]:
    normalized = _norm(label)
    aliases = list(OBJECT_CHINESE_ALIASES.get(normalized, []))
    display = OBJECT_DISPLAY_NAMES.get(normalized)
    if display:
        aliases.extend(part.strip() for part in display.split("/") if part.strip())
    for info in ACTION_ALIASES.values():
        if normalized in {_norm(item) for item in info["objects"]}:
            aliases.extend(info["keywords"])
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        if alias and alias not in seen:
            result.append(alias)
            seen.add(alias)
    return result


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _query_payload(text: str, canonical: str, objects: list[str], interactions: list[str], keywords: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "canonical_action": canonical,
        "target_objects": objects,
        "target_interaction_types": interactions,
        "keywords": keywords,
    }
    for token in [text, canonical, *objects, *interactions, *keywords]:
        if token:
            payload[str(token)] = True
    return payload


def expand_query(query_text: str) -> dict[str, Any]:
    text = str(query_text or "")
    lower = text.lower()
    if "\u8bb0\u5f55" in text or ("\u8bfb\u6570" in text and "\u79f0\u91cf" not in text):
        return _query_payload(
            text,
            RECORDING,
            ["balance", "paper", "display", "notebook"],
            ["recording", "panel_reading", "hand_paper_contact"],
            ["recording", "record", "readout", "balance readout", "reading", "\u8bb0\u5f55", "\u8bfb\u6570", "\u5929\u5e73\u8bfb\u6570"],
        )
    if any(token in text for token in ("\u522e\u53d6", "\u522e\u52fa", "\u836f\u5319")):
        return _query_payload(
            text,
            USE_SPATULA,
            ["spatula"],
            ["hand_spatula_contact", "spatula_sampling"],
            ["spatula", "\u522e\u53d6", "\u522e\u52fa", "\u836f\u5319", "\u53d6\u6837", "\u6837\u54c1"],
        )
    if "\u79f0\u91cf\u7eb8" in text:
        return _query_payload(
            text,
            "weighing_paper_operation",
            ["paper", "weighing_paper", "spatula"],
            ["hand_paper_contact", "hand_spatula_contact", "spatula_sampling"],
            ["weighing paper", "sample transfer", "\u79f0\u91cf\u7eb8", WEIGHING_PAPER_OPERATION, "\u6837\u54c1", "\u52a0\u5165", "\u53d6\u6837"],
        )
    if (
        BALANCE_PANEL_OPERATION in text
        or "\u8bbe\u5907\u9762\u677f" in text
        or "\u5929\u5e73\u9762\u677f" in text
        or "balance panel" in lower
        or "equipment panel" in lower
    ):
        return _query_payload(
            text,
            "equipment_panel_operation",
            ["balance", "panel", "display"],
            ["equipment_panel_operation", "panel_reading", "hand_balance_contact"],
            ["balance panel", "equipment panel", "panel", "display", "\u5929\u5e73", "\u8bbe\u5907\u9762\u677f", BALANCE_PANEL_OPERATION],
        )
    if (
        REAGENT_BOTTLE_OPERATION in text
        or "\u8bd5\u5242\u74f6\u64cd\u4f5c" in text
        or "\u74f6\u76d6" in text
        or "\u5f00\u74f6" in text
        or "reagent bottle open" in lower
        or "bottle cap" in lower
    ):
        return _query_payload(
            text,
            "reagent_bottle_interaction",
            ["reagent_bottle", "reagent_bottle_open", "bottle_cap", "sample_bottle", "sample_bottle_blue", "bottle"],
            ["hand_bottle_contact", "hand_reagent_bottle_contact", "hand_object_contact"],
            ["reagent bottle", "bottle cap", "sample bottle", "open", REAGENT_BOTTLE_OPERATION, "\u8bd5\u5242\u74f6", "\u6837\u54c1\u74f6", "\u74f6\u76d6", "\u5f00\u74f6"],
        )
    if "\u79fb\u6db2" in text or "\u52a0\u6837" in text:
        return _query_payload(
            text,
            "pipetting",
            ["pipette", "pipette_tip", "tube", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "bottle", "sample_bottle", "sample_bottle_blue"],
            ["pipetting", "sample_adding", "hand_pipette_contact", "hand_pipette_tip_contact"],
            ["pipetting", "pipette", "liquid transfer", "sample adding", "\u79fb\u6db2", "\u52a0\u6837", "\u6ef4\u52a0"],
        )
    if "\u79f0\u91cf" in text or "\u5929\u5e73" in text:
        return _query_payload(
            text,
            "weighing",
            ["balance", "panel", "paper", "weighing_paper", "sample_bottle", "spatula"],
            ["weighing", "equipment_panel_operation", "hand_balance_contact", "hand_paper_contact", "spatula_sampling"],
            ["weighing", "balance", "balance panel", "weighing paper", "sample mass", "\u79f0\u91cf", "\u5929\u5e73", "\u8bbe\u5907\u9762\u677f", "\u91cd\u91cf"],
        )
    if "\u6837\u54c1\u5904\u7406" in text or ("\u6837\u54c1" in text and "\u79fb\u6db2" not in text and "\u52a0\u6837" not in text):
        return _query_payload(
            text,
            SAMPLE_HANDLING,
            ["sample_bottle", "sample_bottle_blue", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "spatula", "paper", "bottle"],
            ["hand_object_contact", "hand_sample_bottle_contact", "hand_reagent_bottle_contact", "hand_spatula_contact", "hand_paper_contact"],
            ["sample handling", "sample", "sample bottle", "spatula", "weighing paper", "\u6837\u54c1", "\u6837\u54c1\u5904\u7406", "\u53d6\u6837", "\u6837\u54c1\u74f6"],
        )
    if any(token in lower for token in ["balance weighing", "weighing", "weigh"]):
        return _query_payload(
            text,
            "weighing",
            ["balance", "panel", "paper", "weighing_paper", "sample_bottle", "spatula"],
            ["weighing", "equipment_panel_operation", "hand_balance_contact", "hand_paper_contact", "spatula_sampling"],
            ["weighing", "balance", "balance panel", "weighing paper", "sample mass"],
        )
    if any(token in lower for token in ["pipette", "pipetting", "liquid transfer"]):
        return _query_payload(
            text,
            "pipetting",
            ["pipette", "pipette_tip", "tube", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "bottle", "sample_bottle", "sample_bottle_blue"],
            ["pipetting", "sample_adding", "hand_pipette_contact", "hand_pipette_tip_contact"],
            ["pipetting", "pipette", "liquid transfer", "sample adding"],
        )
    if "sample handling" in lower or ("sample" in lower and "handling" in lower):
        return _query_payload(
            text,
            SAMPLE_HANDLING,
            ["sample_bottle", "sample_bottle_blue", "reagent_bottle", "reagent_bottle_open", "bottle_cap", "spatula", "paper", "bottle"],
            ["hand_object_contact", "hand_sample_bottle_contact", "hand_reagent_bottle_contact", "hand_spatula_contact", "hand_paper_contact"],
            ["sample handling", "sample", "sample bottle", "spatula", "weighing paper"],
        )
    if "recording" in lower or "readout" in lower or ("record" in lower and "balance" in lower):
        return _query_payload(
            text,
            RECORDING,
            ["balance", "paper", "display", "notebook"],
            ["recording", "panel_reading", "hand_paper_contact"],
            ["recording", "record", "readout", "balance readout", "reading"],
        )
    if "reagent bottle" in lower or ("bottle" in lower and ("touch" in lower or "contact" in lower or "open" in lower)):
        return _query_payload(
            text,
            "reagent_bottle_interaction",
            ["reagent_bottle", "reagent_bottle_open", "bottle_cap", "bottle", "sample_bottle", "sample_bottle_blue"],
            ["hand_bottle_contact", "hand_reagent_bottle_contact", "hand_sample_bottle_contact"],
            ["reagent bottle", "bottle cap", "bottle", "touch", "contact", "open"],
        )
    selected = None
    for action, info in ACTION_ALIASES.items():
        if action in text or any(keyword and keyword in text for keyword in info["keywords"]):
            selected = action
            break
    if selected is None:
        selected = text.strip() or "unknown"
        return _query_payload(text, selected, [], [], [part for part in text.split() if part])
    info = ACTION_ALIASES[selected]
    canonical = "pipetting" if selected == PIPETTING_ZH else selected
    return _query_payload(text, canonical, list(info["objects"]), list(info["interaction_types"]), list(info["keywords"]))


def infer_action_type_from_metadata(metadata: dict[str, Any], dialogue_text: str = "") -> str:
    text_description = metadata.get("text_description") if isinstance(metadata.get("text_description"), dict) else {}
    if text_description.get("action_type"):
        return str(text_description["action_type"])
    if metadata.get("action_type") and str(metadata.get("action_type")) != "unknown_operation":
        return str(metadata.get("action_type"))
    text = f"{metadata.get('index_text', '')} {dialogue_text}"
    primary = _norm(metadata.get("primary_object"))
    interaction_type = _norm(metadata.get("interaction_type"))
    detected = {_norm(item) for item in _as_text_list(metadata.get("detected_objects"))}
    if any(keyword in text for keyword in ACTION_ALIASES[PIPETTING_ZH]["keywords"]) or primary in {"pipette", "pipette_tip"}:
        return "pipetting"
    if any(keyword in text for keyword in ACTION_ALIASES[USE_SPATULA]["keywords"]) or primary == "spatula":
        return "spatula_interaction"
    if (
        BALANCE_PANEL_OPERATION in text
        or "\u8bbe\u5907\u9762\u677f" in text
        or "\u5929\u5e73\u9762\u677f" in text
        or "panel" in interaction_type
    ):
        return "equipment_panel_operation"
    if WEIGHING_PAPER_OPERATION in text or (primary in {"paper", "weighing_paper"} and "\u79f0\u91cf\u7eb8" in text):
        return "weighing_paper_operation"
    if any(keyword in text for keyword in ACTION_ALIASES[WEIGHING]["keywords"]):
        return "weighing"
    if primary in {"balance", "panel", "display"} or (not primary and detected & {"balance", "panel", "display"}):
        return "equipment_panel_operation"
    if primary in {"paper", "weighing_paper"}:
        return "weighing_paper_operation"
    if "pipette" in interaction_type:
        return "pipetting"
    return ACTION_TYPES_BY_OBJECT.get(primary, str(metadata.get("action_type") or "unknown_operation"))


def score_query_metadata_match(query_text: str, metadata: dict[str, Any]) -> dict[str, Any]:
    query = expand_query(query_text)
    canonical = query["canonical_action"]
    target_objects = {_norm(item) for item in query["target_objects"]}
    target_interactions = {_norm(item) for item in query["target_interaction_types"]}
    keywords = [str(item) for item in query["keywords"]]
    primary = _norm(metadata.get("primary_object"))
    interaction_type = _norm(metadata.get("interaction_type"))
    action_type = _norm(metadata.get("action_type"))
    index_text = str(metadata.get("index_text") or "")
    related_dialogue = " ".join(_as_text_list(metadata.get("related_dialogue")))
    detected = {_norm(item) for item in _as_text_list(metadata.get("detected_objects"))}
    tools = {_norm(item) for item in _as_text_list(metadata.get("tools"))}
    start_sec = _as_float(metadata.get("start_sec"))
    duration_sec = _as_float(metadata.get("duration_sec"))
    class_threshold = metadata.get("class_threshold") if isinstance(metadata.get("class_threshold"), dict) else {}
    query_boost = float(class_threshold.get("query_boost", 1.0) or 1.0)
    score = 0.0
    reasons: list[str] = []

    has_dialogue_keyword = any(keyword and keyword in related_dialogue for keyword in keywords)
    has_index_keyword = any(keyword and keyword in index_text for keyword in keywords)
    sample_adding_objects = {"pipette", "pipette_tip", "tube"}
    has_pipette_evidence = bool({primary, *tools} & sample_adding_objects) or any(
        label in interaction_type for label in sample_adding_objects
    ) or bool(detected & sample_adding_objects)
    if canonical in {PIPETTING_ZH, "pipetting"} and not has_pipette_evidence and not has_dialogue_keyword and not has_index_keyword:
        if metadata.get("index_level") == "segment":
            score += 0.08
            reasons.append("fallback_parent_segment_for_insufficient_pipette_or_asr")
        else:
            reasons.append("insufficient_pipette_or_dialogue_evidence")
            reasons.append("insufficient_sample_adding_evidence")
        return {
            "score": float(score),
            "rerank_score": float(score),
            "class_specific_query_boost": 0.0,
            "rerank_reasons": reasons,
        }

    if primary and primary in target_objects:
        score += 0.18
        reasons.append(f"matched_primary_object: {primary}")
    object_hits = detected & target_objects
    if object_hits:
        score += 0.08
        reasons.append("matched_yolo_labels: " + ",".join(sorted(object_hits)))
    if interaction_type and (interaction_type in target_interactions or any(part in interaction_type for part in target_interactions)):
        score += 0.12
        reasons.append(f"matched_interaction_type: {interaction_type}")
    elif target_interactions and "contact" in interaction_type and canonical in {HAND_TOUCH_BOTTLE, "bottle_interaction", "reagent_bottle_interaction"}:
        score += 0.08
        reasons.append("matched_contact_interaction")
    inferred_action = infer_action_type_from_metadata(metadata, related_dialogue)
    if inferred_action and inferred_action == action_type and action_type != "unknown_operation":
        score += 0.10
        reasons.append(f"action_type_match:{action_type}")
    if canonical in {action_type, inferred_action} and canonical:
        score += 0.12
        reasons.append(f"alias_overlap:{canonical}")
    if canonical == "weighing" and (primary == "spatula" or interaction_type == "hand_spatula_contact" or action_type == "spatula_interaction"):
        score += 0.24
        reasons.append("weighing_spatula_sample_transfer_candidate")
    if canonical == SAMPLE_HANDLING and action_type in {"spatula_interaction", "bottle_interaction", "reagent_bottle_interaction", "hand_object_interaction"}:
        score += 0.10
        reasons.append("sample_handling_action_family")
    if canonical == SAMPLE_HANDLING and primary in {"reagent_bottle", "reagent_bottle_open", "bottle_cap", "sample_bottle", "sample_bottle_blue", "spatula", "bottle"}:
        score += 0.12
        reasons.append("sample_handling_object_priority")
    if canonical == SAMPLE_HANDLING and primary == "paper":
        score -= 0.04
        reasons.append("sample_handling_deprioritize_paper_surface")
    if canonical in {"weighing_paper_transfer", "weighing_paper_operation"} and primary in {"paper", "weighing_paper"}:
        score += 0.18
        reasons.append("weighing_paper_surface_priority")
    if canonical in {"weighing_paper_transfer", "weighing_paper_operation"} and (primary == "spatula" or interaction_type == "hand_spatula_contact"):
        score += 0.12
        reasons.append("weighing_paper_spatula_transfer_candidate")
    if canonical == "equipment_panel_operation" and primary in {"balance", "panel", "display"}:
        score += 0.18
        reasons.append("balance_panel_operation_priority")
    if canonical == "equipment_panel_operation" and (
        "panel" in interaction_type or action_type in {"equipment_panel_operation", "panel_operation"}
    ):
        score += 0.14
        reasons.append("equipment_panel_action_family")
    if canonical == RECORDING and primary == "paper":
        score += 0.08
        reasons.append("recording_surface_candidate")
    if canonical == RECORDING and duration_sec is not None and duration_sec >= 8.0:
        score += 0.08
        reasons.append("recording_sustained_surface_window")
    if canonical == RECORDING and start_sec is not None and start_sec >= 45.0:
        score += 0.05
        reasons.append("recording_late_window_candidate")
    if has_dialogue_keyword:
        score += 0.12
        reasons.append("matched_dialogue_keyword")
    if has_index_keyword:
        score += 0.08
        reasons.append(f"matched_query_alias: {canonical}")
    if metadata.get("keyframes"):
        score += 0.03
        reasons.append("keyframe_evidence")
    if metadata.get("index_level") == "micro_segment":
        score += 0.05
        reasons.append("index_level_boost: micro_segment")

    from .evidence import LIMIT_MISSING_PIPETTE_TUBE, LIMIT_MISSING_TRANSCRIPT, evaluate_metadata_evidence

    evidence = evaluate_metadata_evidence(metadata, query_text=query_text)
    if (
        canonical in {PIPETTING_ZH, "pipetting", "sample_adding"}
        and LIMIT_MISSING_PIPETTE_TUBE in evidence.get("limitations", [])
        and LIMIT_MISSING_TRANSCRIPT in evidence.get("limitations", [])
    ):
        score -= 0.25
        reasons.append("insufficient_sample_adding_evidence")
        return {
            "score": float(score),
            "rerank_score": float(score),
            "class_specific_query_boost": 0.0,
            "rerank_reasons": reasons,
        }

    class_specific_query_boost = 0.0
    if canonical in {action_type, inferred_action} and canonical not in {"unknown", ""}:
        class_specific_query_boost = 0.35
    elif query_boost > 1.0 and primary in target_objects:
        class_specific_query_boost = min(0.18, (query_boost - 1.0) * 0.20)
    if class_specific_query_boost:
        score += class_specific_query_boost
        reasons.append(f"class_specific_query_boost:{class_specific_query_boost:.2f}")
    return {
        "score": float(score),
        "rerank_score": float(score),
        "class_specific_query_boost": float(class_specific_query_boost),
        "rerank_reasons": reasons,
    }
