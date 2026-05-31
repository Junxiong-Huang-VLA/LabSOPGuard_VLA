from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any

from .semantic_alias import ACTION_ALIASES, PIPETTING_ZH, USE_SPATULA, WEIGHING


SAMPLE_ADDING_KEYWORDS = ACTION_ALIASES[PIPETTING_ZH]["keywords"]
WEIGHING_KEYWORDS = ACTION_ALIASES[WEIGHING]["keywords"]
SPATULA_KEYWORDS = ACTION_ALIASES[USE_SPATULA]["keywords"]
RECORDING_KEYWORDS = ["记录", "读数", "读取", "记一下", "record"]
PIPETTE_OBJECTS = {"pipette", "pipette_tip", "tube"}
HAND_LABELS = {"hand", "gloved_hand", "hands", "glove", "gloves"}
SAMPLE_ADDING_ACTIONS = {"pipetting", "sample_adding"}
LIMIT_MISSING_PIPETTE_TUBE = "missing pipette or tube visual evidence"
LIMIT_MISSING_TRANSCRIPT = "missing transcript evidence"
LEVELS = {
    "trusted",
    "visual_confirmed",
    "transcript_supported",
    "visual_and_transcript_confirmed",
    "weak_visual_evidence",
    "insufficient",
    "insufficient_evidence",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return {key: getattr(obj, key) for key in getattr(obj, "__dataclass_fields__", {})}
    return {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _norm(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")


def _texts(value: Any) -> list[str]:
    texts: list[str] = []
    for item in _as_list(value):
        if isinstance(item, dict):
            text = item.get("text") or item.get("utterance")
        else:
            text = getattr(item, "text", item)
        if text:
            texts.append(str(text))
    return texts


def _dialogue_context_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for item in _as_list(metadata.get("dialogue_context")):
        if isinstance(item, dict):
            context.append(item)
        elif item:
            context.append({"text": str(item)})
    for item in _as_list(metadata.get("related_dialogue")):
        if isinstance(item, dict):
            context.append(item)
        elif item:
            context.append({"text": str(item)})
    return context


def _dialogue_text(dialogue_context: Any) -> str:
    return " ".join(_texts(dialogue_context))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            output.append(value)
            seen.add(value)
    return output


def _keyframe_count(keyframes: Any) -> int:
    if isinstance(keyframes, dict):
        return len([value for value in keyframes.values() if value])
    return len([value for value in _as_list(keyframes) if value])


def extract_dialogue_keywords(dialogue_context: Any, extra_keywords: list[str] | None = None) -> list[str]:
    text = _dialogue_text(dialogue_context)
    keywords = [*SAMPLE_ADDING_KEYWORDS, *WEIGHING_KEYWORDS, *SPATULA_KEYWORDS, *RECORDING_KEYWORDS]
    if extra_keywords:
        keywords.extend(extra_keywords)
    return _dedupe([keyword for keyword in keywords if keyword and keyword in text])


def _metadata_labels(metadata: dict[str, Any]) -> set[str]:
    labels = {
        _norm(metadata.get("primary_object")),
        *{_norm(item) for item in _as_list(metadata.get("detected_objects"))},
        *{_norm(item) for item in _as_list(metadata.get("tools"))},
        *{_norm(item) for item in _as_list(metadata.get("objects"))},
        *{_norm(item) for item in _as_list(metadata.get("visual_keywords"))},
    }
    interaction_type = _norm(metadata.get("interaction_type"))
    if interaction_type:
        labels.add(interaction_type)
        labels.update(part for part in interaction_type.split("_") if part)
    for key in ("interaction_events", "yolo_interactions"):
        for event in _as_list(metadata.get(key)):
            if isinstance(event, dict):
                labels.add(_norm(event.get("object_label")))
                labels.add(_norm(event.get("object_name")))
                labels.add(_norm(event.get("interaction")))
                for detection in _as_list(event.get("detections")):
                    if isinstance(detection, dict):
                        labels.add(_norm(detection.get("label")))
    for frame in _as_list(metadata.get("yolo_evidence")):
        if isinstance(frame, dict):
            labels.add(_norm(frame.get("primary_object")))
            for detection in _as_list(frame.get("detections")):
                if isinstance(detection, dict):
                    labels.add(_norm(detection.get("label")))
            for interaction in _as_list(frame.get("hand_object_interactions")):
                if isinstance(interaction, dict):
                    labels.add(_norm(interaction.get("object_label")))
                    labels.add(_norm(interaction.get("hand_label")))
    for frame in _as_list(metadata.get("interaction_keyframes")):
        if isinstance(frame, dict):
            labels.update(_norm(item) for item in _as_list(frame.get("labels")))
            labels.add(_norm(frame.get("interaction")))
    return {label for label in labels if label}


def _has_keyframe_or_yolo(metadata: dict[str, Any]) -> bool:
    return bool(
        _as_list(metadata.get("keyframes"))
        or _as_list(metadata.get("interaction_keyframes"))
        or _as_list(metadata.get("yolo_interactions"))
        or _as_list(metadata.get("yolo_evidence"))
        or _as_list(metadata.get("interaction_events"))
    )


def _metadata_action_type(metadata: dict[str, Any]) -> str:
    text_description = metadata.get("text_description")
    if isinstance(text_description, dict) and text_description.get("action_type"):
        return _norm(text_description.get("action_type"))
    return _norm(metadata.get("action_type"))


def _has_trusted_sample_adding_visual(metadata: dict[str, Any]) -> bool:
    primary = _norm(metadata.get("primary_object"))
    interaction_type = _norm(metadata.get("interaction_type"))
    if primary in PIPETTE_OBJECTS:
        return True
    if any(label in interaction_type for label in PIPETTE_OBJECTS):
        return True
    for key in ("interaction_events", "yolo_interactions"):
        for event in _as_list(metadata.get(key)):
            if not isinstance(event, dict):
                continue
            event_object = _norm(event.get("object_label") or event.get("object_name") or event.get("primary_object"))
            event_type = _norm(event.get("interaction") or event.get("interaction_type"))
            if event_object in PIPETTE_OBJECTS or any(label in event_type for label in PIPETTE_OBJECTS):
                return True
    return False


def _metadata_scores(metadata: dict[str, Any]) -> tuple[float, float]:
    interaction = metadata.get("interaction") if isinstance(metadata.get("interaction"), dict) else {}
    max_score = float(interaction.get("max_interaction_score", metadata.get("max_interaction_score", 0.0)) or 0.0)
    avg_score = float(interaction.get("avg_interaction_score", metadata.get("avg_interaction_score", 0.0)) or 0.0)
    return max_score, avg_score


def evaluate_metadata_evidence(metadata: dict[str, Any], query_text: str | None = None) -> dict[str, Any]:
    stored = metadata.get("evidence") if isinstance(metadata.get("evidence"), dict) else {}
    stored_level = str(stored.get("evidence_level") or metadata.get("evidence_level") or "")
    if stored_level in LEVELS:
        limitations = _dedupe(_texts(stored.get("limitations")) + _texts(metadata.get("limitations")))
        reasons = _dedupe(_texts(stored.get("evidence_reasons")) + _texts(metadata.get("evidence_reasons")))
        labels = _metadata_labels(metadata)
        dialogue_context = _dialogue_context_from_metadata(metadata)
        dialogue_keywords = extract_dialogue_keywords(dialogue_context)
        has_pipette_visual = _has_trusted_sample_adding_visual(metadata)
        query_text = str(query_text or "")
        sample_query = any(keyword in query_text for keyword in SAMPLE_ADDING_KEYWORDS)
        if sample_query and not has_pipette_visual:
            limitations.append(LIMIT_MISSING_PIPETTE_TUBE)
        if sample_query and not dialogue_context:
            limitations.append(LIMIT_MISSING_TRANSCRIPT)
        return {
            "evidence_level": stored_level,
            "evidence_reasons": reasons,
            "limitations": _dedupe(limitations),
            "has_pipette_or_tube_visual_evidence": has_pipette_visual,
            "has_transcript_evidence": bool(dialogue_context),
            "dialogue_keywords": dialogue_keywords,
        }

    labels = _metadata_labels(metadata)
    max_score, avg_score = _metadata_scores(metadata)
    dialogue_context = _dialogue_context_from_metadata(metadata)
    dialogue_keywords = extract_dialogue_keywords(dialogue_context)
    has_dialogue = bool(dialogue_context)
    has_transcript_keywords = bool(dialogue_keywords)
    has_keyframe = _has_keyframe_or_yolo(metadata)
    has_hand = bool(labels & HAND_LABELS) or "hand" in _norm(metadata.get("interaction_type"))
    primary = _norm(metadata.get("primary_object"))
    has_interaction = (has_hand and bool(primary)) or has_keyframe
    strong_visual = has_interaction and (max_score >= 0.50 or has_keyframe)
    weak_visual = has_interaction or max_score > 0.0 or avg_score > 0.0 or bool(labels)
    has_pipette_visual = _has_trusted_sample_adding_visual(metadata)
    action_type = _metadata_action_type(metadata)
    query_text = str(query_text or "")
    sample_context = (
        action_type in SAMPLE_ADDING_ACTIONS
        or any(keyword in query_text for keyword in SAMPLE_ADDING_KEYWORDS)
        or any(keyword in str(metadata.get("index_text") or "") for keyword in SAMPLE_ADDING_KEYWORDS)
    )
    sample_dialogue = any(keyword in SAMPLE_ADDING_KEYWORDS for keyword in dialogue_keywords)
    reasons: list[str] = []
    limitations: list[str] = []
    if primary:
        reasons.append(f"visual object matched: {primary}")
    if labels:
        reasons.append("matched visual labels: " + ",".join(sorted(labels)[:8]))
    if max_score:
        reasons.append(f"max interaction score: {max_score:.3f}")
    if has_keyframe:
        reasons.append("peak keyframe available")
    reasons.extend(f"dialogue keyword matched: {keyword}" for keyword in dialogue_keywords)
    if sample_context and has_pipette_visual:
        reasons.append("sample_adding_visual_evidence:pipette_or_tube")
    if sample_context and sample_dialogue:
        reasons.append("transcript_evidence")

    if sample_context:
        if has_pipette_visual and sample_dialogue:
            level = "trusted"
        elif sample_dialogue:
            level = "transcript_supported"
            limitations.append(LIMIT_MISSING_PIPETTE_TUBE)
        elif has_pipette_visual:
            level = "visual_confirmed" if strong_visual else "weak_visual_evidence"
            limitations.append(LIMIT_MISSING_TRANSCRIPT)
        elif weak_visual and (has_keyframe or max_score > 0.0 or avg_score > 0.0):
            level = "visual_confirmed" if strong_visual else "weak_visual_evidence"
            limitations.extend([LIMIT_MISSING_PIPETTE_TUBE, LIMIT_MISSING_TRANSCRIPT])
        else:
            level = "insufficient"
            limitations.extend([LIMIT_MISSING_PIPETTE_TUBE, LIMIT_MISSING_TRANSCRIPT])
    elif strong_visual and has_transcript_keywords:
        level = "visual_and_transcript_confirmed"
    elif strong_visual:
        level = "visual_confirmed"
    elif has_transcript_keywords:
        level = "transcript_supported"
    elif weak_visual:
        level = "weak_visual_evidence"
    else:
        level = "insufficient_evidence"
        limitations.append("insufficient visual and transcript evidence")

    if not has_dialogue:
        limitations.append(LIMIT_MISSING_TRANSCRIPT)
    if level == "weak_visual_evidence":
        limitations.append("low or sparse visual interaction evidence")

    return {
        "evidence_level": level,
        "evidence_reasons": _dedupe(reasons),
        "limitations": _dedupe(limitations),
        "has_pipette_or_tube_visual_evidence": has_pipette_visual,
        "has_transcript_evidence": has_dialogue,
        "dialogue_keywords": dialogue_keywords,
    }


def attach_evidence(
    metadata: dict[str, Any],
    query_text: str | None = None,
    rerank_reasons: list[str] | None = None,
) -> dict[str, Any]:
    enriched = dict(metadata)
    stored = enriched.get("evidence") if isinstance(enriched.get("evidence"), dict) else {}
    preserved_evidence = {
        key: value
        for key, value in stored.items()
        if key not in {"evidence_level", "evidence_reasons", "limitations"}
    }
    evidence = evaluate_metadata_evidence(enriched, query_text=query_text)
    reasons = _dedupe([*evidence["evidence_reasons"], *(rerank_reasons or [])])
    enriched["evidence"] = {
        "evidence_level": evidence["evidence_level"],
        "evidence_reasons": reasons,
        "limitations": evidence["limitations"],
        **preserved_evidence,
    }
    for key, value in preserved_evidence.items():
        enriched.setdefault(key, value)
    enriched["evidence_level"] = evidence["evidence_level"]
    enriched["evidence_reasons"] = reasons
    enriched["limitations"] = evidence["limitations"]
    enriched["dialogue_keywords"] = evidence.get("dialogue_keywords", [])
    return enriched


def build_micro_evidence(micro: Any) -> dict[str, Any]:
    interaction = _as_dict(_get(micro, "interaction", {}))
    text_description = _as_dict(_get(micro, "text_description", {}))
    quality = _as_dict(_get(micro, "quality", {}))
    metadata = {
        "index_level": "micro_segment",
        "primary_object": _get(interaction, "primary_object"),
        "interaction_type": _get(interaction, "interaction_type"),
        "detected_objects": _get(interaction, "detected_objects", []),
        "interaction": {
            "avg_interaction_score": _get(interaction, "avg_interaction_score"),
            "max_interaction_score": _get(interaction, "max_interaction_score"),
        },
        "keyframes": _get(micro, "keyframes", {}),
        "yolo_evidence": _get(micro, "yolo_evidence", []),
        "dialogue_context": _get(micro, "dialogue_context", []),
        "text_description": text_description,
        "action_type": _get(text_description, "action_type"),
        "quality": _get(quality, "confidence"),
    }
    evidence = evaluate_metadata_evidence(metadata)
    evidence["visual_evidence"] = {
        "has_hand_object_interaction": bool(_get(interaction, "interaction_type")),
        "primary_object": _norm(_get(interaction, "primary_object")),
        "detected_objects": [_norm(item) for item in _as_list(_get(interaction, "detected_objects", []))],
        "max_interaction_score": float(_get(interaction, "max_interaction_score", 0.0) or 0.0),
        "avg_interaction_score": float(_get(interaction, "avg_interaction_score", 0.0) or 0.0),
        "keyframe_count": _keyframe_count(_get(micro, "keyframes", {})),
        "confidence": _get(quality, "confidence"),
    }
    dialogue_context = _get(micro, "dialogue_context", [])
    evidence["transcript_evidence"] = {
        "has_dialogue": bool(dialogue_context),
        "matched_keywords": evidence.get("dialogue_keywords", []),
        "utterance_ids": [
            str(item.get("utterance_id"))
            for item in _as_list(dialogue_context)
            if isinstance(item, dict) and item.get("utterance_id")
        ],
    }
    action_type = str(_get(text_description, "action_type", "") or "")
    if evidence["evidence_level"] == "transcript_supported" and any(
        keyword in SAMPLE_ADDING_KEYWORDS for keyword in evidence.get("dialogue_keywords", [])
    ):
        evidence["inferred_action_type"] = "sample_adding_candidate"
    elif action_type in SAMPLE_ADDING_ACTIONS and evidence["evidence_level"] == "weak_visual_evidence":
        evidence["inferred_action_type"] = "possible_sample_handling"
    else:
        evidence["inferred_action_type"] = action_type
    return evidence


def build_segment_evidence(segment: Any) -> dict[str, Any]:
    metadata = {
        "index_level": "segment",
        "related_dialogue": _get(segment, "dialogue_context", []),
        "interaction_events": _get(segment, "interaction_events", []),
        "yolo_interactions": _get(segment, "yolo_interactions", []),
        "interaction_keyframes": _get(segment, "interaction_keyframes", []),
        "action_type": _get(_get(segment, "text_description", {}), "action_type"),
    }
    evidence = evaluate_metadata_evidence(metadata)
    evidence.setdefault(
        "visual_evidence",
        {
            "has_hand_object_interaction": bool(metadata["interaction_events"] or metadata["yolo_interactions"]),
            "primary_object": None,
            "detected_objects": [],
            "max_interaction_score": None,
            "keyframe_count": len(_as_list(metadata["interaction_keyframes"])),
        },
    )
    evidence.setdefault(
        "transcript_evidence",
        {
            "has_dialogue": bool(metadata["related_dialogue"]),
            "matched_keywords": evidence.get("dialogue_keywords", []),
            "utterance_ids": [],
        },
    )
    return evidence


def apply_micro_evidence(micro: Any) -> Any:
    evidence = build_micro_evidence(micro)
    setattr(micro, "evidence", evidence)
    setattr(micro, "dialogue_keywords", evidence.get("dialogue_keywords", []))
    setattr(micro, "dialogue_context_available", bool(evidence.get("transcript_evidence", {}).get("has_dialogue")))
    text_description = getattr(micro, "text_description", None)
    inferred_action = evidence.get("inferred_action_type")
    if text_description is not None and inferred_action:
        text_description.action_type = str(inferred_action)
        if text_description.index_text:
            text_description.index_text += (
                f"evidence_level: {evidence['evidence_level']}\n"
                f"evidence_reasons: {'; '.join(evidence['evidence_reasons'])}\n"
                f"limitations: {'; '.join(evidence['limitations']) if evidence['limitations'] else 'none'}\n"
                f"dialogue_keywords: {', '.join(evidence.get('dialogue_keywords', [])) if evidence.get('dialogue_keywords') else 'none'}\n"
            )
    return micro


def apply_segment_evidence(segment: Any) -> Any:
    evidence = build_segment_evidence(segment)
    setattr(segment, "evidence", evidence)
    setattr(segment, "dialogue_keywords", evidence.get("dialogue_keywords", []))
    setattr(segment, "dialogue_match_window_sec", 3.0)
    index = getattr(segment, "index", None)
    if index is not None and getattr(index, "index_text", ""):
        index.index_text += (
            f"evidence_level: {evidence['evidence_level']}\n"
            f"evidence_reasons: {'; '.join(evidence['evidence_reasons'])}\n"
            f"limitations: {'; '.join(evidence['limitations']) if evidence['limitations'] else 'none'}\n"
            f"dialogue_keywords: {', '.join(evidence.get('dialogue_keywords', [])) if evidence.get('dialogue_keywords') else 'none'}\n"
        )
    return segment


def explain_query_evidence(query_text: str, metadata: dict[str, Any], rerank_reasons: list[str] | None = None) -> dict[str, Any]:
    enriched = attach_evidence(metadata, query_text=query_text, rerank_reasons=rerank_reasons)
    limitations = list(enriched.get("limitations") or [])
    if any(keyword in str(query_text or "") for keyword in SAMPLE_ADDING_KEYWORDS):
        if LIMIT_MISSING_PIPETTE_TUBE in limitations and metadata.get("index_level") == "segment":
            limitations.append("returned parent segment because no trustworthy sample_adding micro was found")
    return {
        "evidence_level": enriched["evidence_level"],
        "evidence_reasons": _dedupe(list(enriched.get("evidence_reasons") or [])),
        "limitations": _dedupe(limitations),
    }
