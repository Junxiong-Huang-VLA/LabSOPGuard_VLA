from __future__ import annotations

from dataclasses import is_dataclass
from typing import Any, Mapping

from .semantic_alias import OBJECT_DISPLAY_NAMES, chinese_aliases_for_label


ACTION_DISPLAY_NAMES = {
    "pipetting": "移液/加样",
    "sample_adding": "加样候选",
    "sample_adding_candidate": "加样候选",
    "weighing": "称量",
    "spatula_interaction": "刮勺/药匙取样",
    "bottle_interaction": "瓶体操作",
    "reagent_bottle_interaction": "试剂瓶操作",
    "tube_interaction": "试管操作",
    "hand_object_interaction": "手物交互",
    "unknown_operation": "未知实验操作",
}


def refresh_segment_chinese_index(segment: Any) -> Any:
    text_description = _child(segment, "text_description")
    index_info = _child(segment, "index")
    action_type = _text(_get(text_description, "action_type")) or "unknown_operation"
    objects = _segment_objects(segment)
    tools = _unique(_as_list(_get(text_description, "tools")) + _tool_objects(objects))
    micro_count = len(_as_list(_get(segment, "micro_segments")))
    interaction_count = len(_as_list(_get(segment, "interaction_events"))) + len(_as_list(_get(segment, "yolo_interactions")))
    evidence = _as_mapping(_get(segment, "evidence"))
    summary = _segment_summary(action_type, objects, interaction_count, micro_count)
    _set(text_description, "summary", summary)
    _set(text_description, "tools", tools)
    _set(text_description, "objects", objects)
    index_text = _append_chinese_block(
        _text(_get(index_info, "index_text")),
        "segment",
        {
            "中文摘要": summary,
            "中文动作": _action_name(action_type),
            "动作类型": action_type,
            "对象": ", ".join(_object_terms(objects)) or "unknown",
            "工具": ", ".join(_object_terms(tools)) or "unknown",
            "时间范围": f"{_get(segment, 'global_start_time')} - {_get(segment, 'global_end_time')}",
            "素材类型": "segment_clip, keyframe, micro_clip, micro_keyframe",
            "证据级别": evidence.get("evidence_level") or _get(segment, "evidence_level") or "unknown",
            "置信依据": "; ".join(_as_texts(evidence.get("evidence_reasons"))) or "none",
            "限制说明": "; ".join(_as_texts(evidence.get("limitations"))) or "none",
        },
    )
    _set(index_info, "index_text", index_text)
    return segment


def refresh_micro_chinese_index(micro: Any) -> Any:
    interaction = _child(micro, "interaction")
    text_description = _child(micro, "text_description")
    action_type = _text(_get(text_description, "action_type")) or _action_from_primary(_get(interaction, "primary_object"))
    primary = _text(_get(interaction, "primary_object"))
    detected = _unique([primary, *_as_texts(_get(interaction, "detected_objects")), *_as_texts(_get(interaction, "secondary_objects"))])
    evidence = _as_mapping(_get(micro, "evidence"))
    summary = _micro_summary(action_type, primary, interaction)
    _set(text_description, "action_type", action_type)
    _set(text_description, "summary", summary)
    index_text = _append_chinese_block(
        _text(_get(text_description, "index_text")),
        "micro_segment",
        {
            "中文摘要": summary,
            "中文动作": _action_name(action_type),
            "动作类型": action_type,
            "主对象": ", ".join(_object_terms([primary])) or "unknown",
            "检测对象": ", ".join(_object_terms(detected)) or "unknown",
            "交互类型": _text(_get(interaction, "interaction_type")),
            "时间范围": f"{_get(micro, 'global_start_time')} - {_get(micro, 'global_end_time')}",
            "素材类型": "micro_clip, micro_keyframe, segment_clip",
            "证据级别": evidence.get("evidence_level") or _get(micro, "evidence_level") or "unknown",
            "置信依据": "; ".join(_as_texts(evidence.get("evidence_reasons"))) or "none",
            "限制说明": "; ".join(_as_texts(evidence.get("limitations"))) or "none",
        },
    )
    _set(text_description, "index_text", index_text)
    return micro


def refresh_micro_row_chinese_index(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    refresh_micro_chinese_index(updated)
    return updated


def _segment_summary(action_type: str, objects: list[str], interaction_count: int, micro_count: int) -> str:
    object_text = "、".join(_display_name(item) for item in objects[:6]) or "实验物品"
    action_text = _action_name(action_type)
    evidence_parts = []
    if interaction_count:
        evidence_parts.append(f"{interaction_count} 个手物接触证据")
    if micro_count:
        evidence_parts.append(f"{micro_count} 个微片段")
    evidence_text = "，".join(evidence_parts) if evidence_parts else "视觉活动和时间对齐证据"
    return f"该片段疑似发生{action_text}，涉及{object_text}；检索证据包含{evidence_text}。"


def _micro_summary(action_type: str, primary: str, interaction: Any) -> str:
    object_text = _display_name(primary) if primary else "目标物体"
    action_text = _action_name(action_type)
    score = _get(interaction, "max_interaction_score")
    score_text = f"，最高交互分数 {float(score):.3f}" if _is_number(score) else ""
    return f"该微片段检测到手与{object_text}发生接触或近距离操作，属于{action_text}{score_text}。"


def _append_chinese_block(index_text: str, level: str, fields: Mapping[str, Any]) -> str:
    marker = "\n中文检索索引:\n"
    base = str(index_text or "").split(marker, 1)[0].rstrip()
    lines = [marker.strip(), f"索引层级: {level}"]
    for key, value in fields.items():
        lines.append(f"{key}: {value}")
    return f"{base}\n" + "\n".join(lines) + "\n"


def _segment_objects(segment: Any) -> list[str]:
    text_description = _child(segment, "text_description")
    values: list[Any] = [
        _get(text_description, "objects"),
        _get(text_description, "tools"),
        _get(segment, "visual_keywords"),
    ]
    counts = _as_mapping(_get(segment, "yolo_label_counts"))
    values.extend(counts.keys())
    for event in _as_list(_get(segment, "interaction_events")) + _as_list(_get(segment, "yolo_interactions")):
        values.extend([_get(event, "object_label"), _get(event, "object_name"), _get(event, "labels")])
    for micro in _as_list(_get(segment, "micro_segments")):
        values.extend([_get(micro, "primary_object"), _get(micro, "primary_object_family")])
    return _unique(_as_texts(values))


def _tool_objects(objects: list[str]) -> list[str]:
    return [item for item in objects if item in {"balance", "pipette", "pipette_tip", "spatula"}]


def _object_terms(values: list[str]) -> list[str]:
    terms: list[str] = []
    for value in values:
        if not value:
            continue
        terms.append(value)
        terms.append(_display_name(value))
        terms.extend(chinese_aliases_for_label(value))
    return _unique(terms)


def _display_name(value: Any) -> str:
    label = _text(value).strip().lower().replace("-", "_").replace(" ", "_")
    return OBJECT_DISPLAY_NAMES.get(label, label)


def _action_name(value: Any) -> str:
    return ACTION_DISPLAY_NAMES.get(_text(value), _text(value) or "未知实验操作")


def _action_from_primary(primary: Any) -> str:
    label = _text(primary)
    if label in {"pipette", "pipette_tip"}:
        return "pipetting"
    if label == "balance":
        return "weighing"
    if label == "spatula":
        return "spatula_interaction"
    if "bottle" in label:
        return "bottle_interaction"
    if "tube" in label:
        return "tube_interaction"
    return "hand_object_interaction"


def _child(obj: Any, key: str) -> Any:
    value = _get(obj, key)
    if value is None and isinstance(obj, dict):
        obj[key] = {}
        value = obj[key]
    return value


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[key] = value
    elif obj is not None:
        setattr(obj, key, value)


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    return [value]


def _as_texts(value: Any) -> list[str]:
    values: list[str] = []
    for item in _as_list(value):
        if isinstance(item, Mapping):
            for key in ("label", "name", "object_label", "object_name", "primary_object", "interaction", "action_type", "text"):
                if item.get(key) is not None:
                    values.extend(_as_texts(item.get(key)))
        elif is_dataclass(item):
            for key in ("label", "name", "object_label", "object_name", "primary_object", "interaction", "action_type", "text"):
                if hasattr(item, key):
                    values.extend(_as_texts(getattr(item, key)))
        else:
            text = _text(item)
            if text:
                values.append(text)
    return values


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False
