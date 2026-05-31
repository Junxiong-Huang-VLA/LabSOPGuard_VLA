from __future__ import annotations

import re
from typing import Any

from .schemas import (
    KeyActionSegment,
    MicroSegment,
    MicroSegmentIndexInfo,
    MicroSegmentTextDescription,
    SegmentIndexInfo,
    TextDescription,
    TranscriptUtterance,
)
from .semantic_alias import OBJECT_DISPLAY_NAMES, infer_action_type_from_metadata


PIPETTING_KEYWORDS = ["加样", "移液", "移液枪", "微升", "加入", "滴加", "pipette", "pipetting", "uL", "ul"]
WEIGHING_KEYWORDS = ["称量", "天平", "读数", "重量", "质量", "balance", "weighing"]
RECORDING_KEYWORDS = ["记录", "读数", "读取", "记一下", "record"]


def _joined_dialogue(dialogue: list[TranscriptUtterance] | list[str] | list[dict[str, Any]]) -> str:
    texts: list[str] = []
    for item in dialogue:
        if isinstance(item, dict):
            texts.append(str(item.get("text") or item))
        elif hasattr(item, "text"):
            texts.append(str(item.text))
        else:
            texts.append(str(item))
    return " ".join(text for text in texts if text)


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def infer_action_type(text: str) -> str:
    source = str(text or "")
    if any(keyword in source for keyword in PIPETTING_KEYWORDS):
        return "pipetting"
    if any(keyword in source for keyword in WEIGHING_KEYWORDS):
        return "weighing"
    if any(keyword in source for keyword in RECORDING_KEYWORDS):
        return "reading_or_recording"
    return "unknown_operation"


def infer_tools(text: str) -> list[str]:
    mapping = {
        "移液枪": "pipette",
        "移液": "pipette",
        "天平": "balance",
        "面板": "panel",
        "称量纸": "paper",
        "刮勺": "spatula",
        "药匙": "spatula",
        "试剂瓶": "reagent_bottle",
        "磁力搅拌子": "magnetic_stir_bar",
        "试管": "tube",
        "pipette": "pipette",
        "balance": "balance",
        "spatula": "spatula",
    }
    return _ordered_unique([value for key, value in mapping.items() if key in str(text)])


def infer_objects(text: str) -> list[str]:
    mapping = {
        "样品": "sample",
        "试剂": "reagent",
        "试剂瓶": "reagent_bottle",
        "样品瓶": "sample_bottle",
        "称量纸": "paper",
        "磁力搅拌子": "magnetic_stir_bar",
        "瓶子": "bottle",
        "试管": "tube",
        "溶液": "solution",
        "bottle": "bottle",
        "tube": "tube",
    }
    return _ordered_unique([value for key, value in mapping.items() if key in str(text)])


def infer_numbers(text: str) -> list[str]:
    pattern = re.compile(r"\d+(?:\.\d+)?\s*(?:微升|毫升|分钟|秒|克|mL|ml|uL|µL|g)")
    return [match.group(0).strip() for match in pattern.finditer(str(text or ""))]


def _event_value(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _interaction_phrases(segment: KeyActionSegment) -> list[str]:
    phrases: list[str] = []
    for event in segment.interaction_events:
        phrase = str(_event_value(event, "interaction", "") or "")
        if phrase:
            phrases.append(phrase)
    return _ordered_unique(phrases)


def _interaction_tools_objects(segment: KeyActionSegment) -> tuple[list[str], list[str]]:
    tool_labels = {"balance", "panel", "spatula", "pipette", "pipette_tip", "magnetic_stir_bar"}
    object_labels = {
        "sample_bottle_blue",
        "sample_bottle",
        "bottle",
        "reagent_bottle",
        "reagent_bottle_open",
        "bottle_cap",
        "paper",
        "weighing_paper",
        "magnetic_stir_bar",
        "tube",
        "beaker",
        "cup",
    }
    tools: list[str] = []
    objects: list[str] = []
    for event in segment.interaction_events:
        label = str(_event_value(event, "object_label", "") or "")
        name = str(_event_value(event, "object_name", "") or "")
        if label in tool_labels:
            tools.append(label)
        if label in object_labels:
            objects.append(label)
        if name in {"天平", "电子天平"}:
            tools.append("balance")
        elif name in {"刮勺", "药匙"}:
            tools.append("spatula")
        elif name == "试剂瓶":
            objects.append("reagent_bottle")
        elif name == "瓶子":
            objects.append("bottle")
    return _ordered_unique(tools), _ordered_unique(objects)


def _action_from_interactions(segment: KeyActionSegment, current: str) -> str:
    if current != "unknown_operation":
        return current
    labels = {str(_event_value(event, "object_label", "") or "") for event in segment.interaction_events}
    if labels & {"pipette", "pipette_tip"}:
        return "pipetting"
    if labels & {"balance"}:
        return "weighing"
    if labels & {"spatula"}:
        return "spatula_interaction"
    return current


def _append_interaction_index_text(index_text: str, segment: KeyActionSegment) -> str:
    phrases = _interaction_phrases(segment)
    if not phrases and not segment.interaction_keyframes and not segment.yolo_interactions:
        return index_text
    keyframe_paths = [frame.path for frame in segment.interaction_keyframes]
    sampled_interactions = segment.yolo_interactions[:20]
    yolo_pairs = [
        f"{item.hand_label}->{item.object_label}@{item.local_time_sec:.3f}s"
        for item in sampled_interactions
    ]
    if len(segment.yolo_interactions) > len(sampled_interactions):
        yolo_pairs.append(f"...total={len(segment.yolo_interactions)}")
    return (
        f"{index_text}"
        f"视觉交互: {'; '.join(phrases) if phrases else 'none'}\n"
        f"交互关键帧: {', '.join(keyframe_paths) if keyframe_paths else 'none'}\n"
        f"YOLO交互标签: {', '.join(yolo_pairs) if yolo_pairs else 'none'}\n"
        f"可检索交互短语: {' '.join(phrases)} hand-object interaction\n"
    )


def build_segment_description(
    segment: KeyActionSegment,
    related_dialogue: list[TranscriptUtterance] | list[str],
    vector_store: str = "local_fallback",
) -> KeyActionSegment:
    dialogue_text = _joined_dialogue(related_dialogue)
    action_type = _action_from_interactions(segment, infer_action_type(dialogue_text))
    tools = infer_tools(dialogue_text)
    objects = infer_objects(dialogue_text)
    numbers = infer_numbers(dialogue_text)
    interaction_tools, interaction_objects = _interaction_tools_objects(segment)
    tools = _ordered_unique(tools + interaction_tools)
    objects = _ordered_unique(objects + interaction_objects)
    interaction_phrases = _interaction_phrases(segment)

    summary = "实验人员在实验台前进行操作，检测到该时间段内 ROI 区域存在持续运动。"
    if action_type == "weighing":
        summary = "实验人员在实验台前进行称量相关操作。"
    elif action_type == "pipetting":
        summary = "实验人员在实验台前使用移液枪进行加样或转移液体操作。"
    elif action_type == "reading_or_recording":
        summary = "实验人员在实验台前进行读数或记录操作。"
    elif action_type == "spatula_interaction":
        summary = "实验人员在实验台前使用刮勺或药匙进行取样相关操作。"
    if interaction_phrases:
        summary = f"{summary} 视觉关键帧显示 {'；'.join(interaction_phrases)}。"

    first_clip = segment.first_person.clip_path if segment.first_person else "N/A"
    index_text = (
        f"实验 session_id: {segment.session_id}\n"
        f"segment_id: {segment.segment_id}\n"
        f"index_level: segment\n"
        f"全局开始时间: {segment.global_start_time}\n"
        f"全局结束时间: {segment.global_end_time}\n"
        f"动作类型: {action_type}\n"
        f"动作摘要: {summary}\n"
        f"可见行为: ROI 区域持续运动，平均活跃分数 {segment.cv_detection.avg_active_score:.3f}\n"
        f"工具: {', '.join(tools) if tools else 'unknown'}\n"
        f"对象: {', '.join(objects) if objects else 'unknown'}\n"
        f"数值: {', '.join(numbers) if numbers else 'none'}\n"
        f"相关对话: {dialogue_text or '无'}\n"
        f"第三人称视频路径: {segment.third_person.clip_path}\n"
        f"第一人称视频路径: {first_clip}\n"
        f"第三人称局部时间: {segment.third_person.local_start_sec:.3f}-{segment.third_person.local_end_sec:.3f}\n"
    )
    if segment.first_person:
        index_text += (
            f"第一人称局部时间: {segment.first_person.local_start_sec:.3f}-"
            f"{segment.first_person.local_end_sec:.3f}\n"
        )
    index_text = _append_interaction_index_text(index_text, segment)

    segment.text_description = TextDescription(
        action_type=action_type,
        summary=summary,
        tools=tools,
        objects=objects,
        numbers=numbers,
    )
    segment.dialogue_context = [item.text if hasattr(item, "text") else str(item) for item in related_dialogue]
    segment.index = SegmentIndexInfo(
        embedding_id=f"emb_{segment.segment_id}",
        index_text=index_text,
        vector_store=vector_store,
    )
    return segment


def infer_micro_action_type(primary_object: str, dialogue_text: str = "") -> str:
    label = str(primary_object or "").lower()
    text = f"{label} {dialogue_text}"
    if any(keyword in text for keyword in PIPETTING_KEYWORDS) or "pipette" in label:
        return "pipetting"
    if "spatula" in label or "刮勺" in text or "药匙" in text:
        return "spatula_interaction"
    if "panel" in label or "面板" in text:
        return "equipment_panel_operation"
    if "balance" in label or "天平" in text:
        return "equipment_panel_operation"
    if "paper" in label or "称量纸" in text:
        return "weighing_paper_operation"
    if "reagent_bottle" in label or "bottle_cap" in label:
        return "reagent_bottle_interaction"
    if "sample_bottle" in label or "bottle" in label:
        return "reagent_bottle_interaction"
    if "magnetic_stir_bar" in label or "搅拌子" in text:
        return "stirring_operation"
    if "tube" in label or "试管" in text:
        return "tube_interaction"
    return "hand_object_interaction"


def _micro_dialogue_context(related_dialogue: list[TranscriptUtterance] | list[str] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for item in related_dialogue:
        if isinstance(item, dict):
            context.append(
                {
                    "utterance_id": item.get("utterance_id", ""),
                    "text": item.get("text", str(item)),
                    "global_start_time": item.get("global_start_time"),
                    "global_end_time": item.get("global_end_time"),
                }
            )
        elif hasattr(item, "text"):
            context.append(
                {
                    "utterance_id": getattr(item, "utterance_id", ""),
                    "text": getattr(item, "text", ""),
                    "global_start_time": getattr(item, "global_start_time", None),
                    "global_end_time": getattr(item, "global_end_time", None),
                }
            )
        else:
            context.append({"utterance_id": "", "text": str(item), "global_start_time": None, "global_end_time": None})
    return context


def build_micro_segment_description(
    micro_segment: MicroSegment,
    related_dialogue: list[TranscriptUtterance] | list[str] | list[dict[str, Any]] | None = None,
    vector_store: str = "local_fallback",
) -> MicroSegment:
    related_dialogue = related_dialogue or []
    dialogue_text = _joined_dialogue(related_dialogue)
    primary = micro_segment.interaction.primary_object
    primary_family = micro_segment.interaction.primary_object_family or ""
    interaction_type = micro_segment.interaction.interaction_type
    action_type = infer_action_type_from_metadata(
        {
            "primary_object": primary,
            "interaction_type": interaction_type,
            "detected_objects": micro_segment.interaction.detected_objects,
            "index_text": dialogue_text,
        },
        dialogue_text=dialogue_text,
    )
    if action_type == "unknown_operation":
        action_type = infer_micro_action_type(primary, dialogue_text)

    display_name = OBJECT_DISPLAY_NAMES.get(primary, primary)
    keyframe_values = [
        micro_segment.keyframes.contact_frame,
        micro_segment.keyframes.peak_frame,
        micro_segment.keyframes.release_frame,
    ]
    keyframes = [str(item) for item in keyframe_values if item]
    first_clip = micro_segment.first_person.clip_path if micro_segment.first_person else None
    third_clip = micro_segment.third_person.clip_path
    if action_type == "weighing":
        summary = "手在天平或称量区域附近发生交互，属于称量相关子动作。"
    elif action_type == "equipment_panel_operation":
        summary = "手在天平设备面板附近发生交互，属于设备面板操作。"
    elif action_type == "weighing_paper_operation":
        summary = "手与称量纸发生接触并持续交互。"
    elif action_type == "pipetting":
        summary = "手与移液或加样相关器具发生交互，属于加样或移液相关子动作。"
    elif action_type == "spatula_interaction":
        summary = "手与刮勺或药匙发生接触并持续交互。"
    elif action_type == "reagent_bottle_interaction":
        summary = "手部与试剂瓶发生接触并持续交互。"
    elif action_type == "stirring_operation":
        summary = "手与磁力搅拌子发生接触并持续交互。"
    else:
        summary = f"手与{display_name}发生接触并持续交互。"

    evidence = micro_segment.interaction
    class_threshold = micro_segment.class_threshold or {}
    phrases = ["hand object contact", primary, interaction_type, display_name]
    if "bottle" in primary:
        phrases.extend(["手碰瓶子", "手接触瓶子", "瓶子交互", "试剂瓶"])
    if primary == "spatula":
        phrases.extend(["使用刮勺", "刮勺交互", "药匙", "取样"])
    if primary == "balance":
        phrases.extend(["称量", "天平读数", "天平交互", "质量"])
    if primary in {"panel", "display"}:
        phrases.extend(["天平设备面板操作", "设备面板", "天平面板", "读数"])
    if primary in {"paper", "weighing_paper"}:
        phrases.extend(["手部与称量纸操作", "称量纸", "weighing paper"])
    if primary in {"magnetic_stir_bar", "magnetic_stirrer"}:
        phrases.extend(["磁力搅拌子", "搅拌子", "magnetic stir bar"])
    if "pipette" in primary:
        phrases.extend(["加样", "移液", "移液枪", "微升"])
    if "tube" in primary:
        phrases.extend(["试管交互", "加样"])

    index_text = (
        f"实验 session: {micro_segment.session_id}\n"
        f"父级片段: {micro_segment.parent_segment_id}\n"
        f"子动作片段: {micro_segment.micro_segment_id}\n"
        f"display_id: {micro_segment.display_id or micro_segment.micro_segment_id}\n"
        f"index_level: micro_segment\n"
        f"时间: {micro_segment.global_start_time} 到 {micro_segment.global_end_time}\n"
        f"主交互对象: {primary}\n"
        f"主对象家族: {primary_family or 'unknown'}\n"
        f"主对象仲裁: {micro_segment.interaction.primary_object_arbitration}\n"
        f"中文别名: {display_name}\n"
        f"交互类型: {interaction_type}\n"
        f"动作类型: {action_type}\n"
        f"class_threshold: interaction_threshold={class_threshold.get('interaction_threshold')} "
        f"min_duration_sec={class_threshold.get('min_duration_sec')} query_boost={class_threshold.get('query_boost')}\n"
        f"YOLO 物理证据: 检测到 hand 与 {primary} 在多帧中接近或重叠，"
        f"平均交互分数 {evidence.avg_interaction_score:.3f}，最高交互分数 {evidence.max_interaction_score:.3f}。\n"
        f"检测对象: {', '.join(evidence.detected_objects) if evidence.detected_objects else primary}\n"
        f"关键帧: {', '.join(keyframes) if keyframes else 'none'}\n"
        f"相关对话: {dialogue_text or '无'}\n"
        f"dialogue_context_available: {bool(dialogue_text)}\n"
        f"第一人称视频: {first_clip or 'N/A'}\n"
        f"第三人称视频: {third_clip or 'N/A'}\n"
        f"动作描述: {summary}\n"
        f"可检索短语: {' '.join(_ordered_unique(phrases))}\n"
    )
    micro_segment.dialogue_context = _micro_dialogue_context(related_dialogue)
    micro_segment.dialogue_context_available = bool(micro_segment.dialogue_context)
    micro_segment.text_description = MicroSegmentTextDescription(
        action_type=action_type,
        summary=summary,
        index_text=index_text,
    )
    micro_segment.index = MicroSegmentIndexInfo(
        index_level="micro_segment",
        embedding_id=f"emb_{micro_segment.micro_segment_id}",
    )
    return micro_segment
