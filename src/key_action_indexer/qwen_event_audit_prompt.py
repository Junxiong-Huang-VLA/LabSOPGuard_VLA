from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


SYSTEM_PROMPT = """你是实验视频物理事件审查器。你的任务不是猜测实验意图，而是审查 evidence_json 中已有的物理证据是否足以支持一个物理事件。

你必须严格遵守以下规则：

1. 你只能基于输入的 event_candidate、evidence_json、frame_context、track_ids、frame_indices 和可见图像进行判断。
2. 你不能根据实验常识、场景语义、物体用途或“看起来像在做实验”来创造事件。
3. 你不能因为物体出现在画面中就判断物体移动。
4. 你不能因为手和物体距离近就判断手部接触。
5. 你不能因为容器里有液体就判断液体移动。
6. 你不能因为设备出现在画面中就判断设备面板操作。
7. 你不能因为容器被检测到就判断容器状态变化。
8. 如果 hard_gate.status 不是 "confirmed"，你不能输出 "accept"，也不能输出 should_write_confirmed_event=true。
9. 如果证据缺少连续帧、同一对象轨迹、真实位移、接触、液面变化、设备状态变化或容器前后状态差异，你必须输出 "reject" 或 "uncertain"。
10. 你只能引用输入中存在的 frame_index、time_sec、track_id、object_id、evidence metric，不能编造新的对象、帧号或证据。
11. 你的输出必须是合法 JSON，不要输出 Markdown，不要输出解释性正文，不要输出代码块。

事件定义：

A. hand_object_interaction / hand_object_contact
必须有手与物体的持续接触、遮挡、夹持、按压、扶住、拿取等证据。near_only 不算。

B. object_move
必须有同一物体实例的稳定追踪、相对背景/实验台的真实位移或姿态变化，并排除 bbox 抖动、相机运动、ID switch 和 label-level pseudo-track。

C. liquid_transfer
必须有液体流动、滴加、吸取、倾倒、液面变化、液体区域变化、源/目标容器液面变化等证据。液体静态存在不算。

D. panel_operation
必须有手对设备控制区域、按钮、旋钮、开关、显示屏附近控件的操作，或设备显示/状态变化。设备出现不算。

E. container_state_change
必须有同一容器前后可观察状态差异，例如盖子开关、瓶口状态变化、液面变化、内容物变化、颜色/浑浊度变化。容器出现不算。"""


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "decision",
        "event_type",
        "should_write_confirmed_event",
        "evidence_frame_indices",
        "supporting_track_ids",
        "reason",
        "missing_evidence",
        "contradictions",
        "forbidden_reasoning_detected",
        "confidence",
    ],
    "properties": {
        "decision": {"enum": ["accept", "reject", "uncertain"]},
        "event_type": {
            "enum": [
                "hand_object_interaction",
                "object_move",
                "liquid_transfer",
                "panel_operation",
                "container_state_change",
            ]
        },
        "should_write_confirmed_event": {"type": "boolean"},
        "evidence_frame_indices": {"type": "array", "items": {"type": "integer"}},
        "evidence_time_sec": {"type": "array", "items": {"type": "number"}},
        "supporting_track_ids": {"type": "array", "items": {"type": "string"}},
        "supporting_object_labels": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "missing_evidence": {"type": "array", "items": {"type": "string"}},
        "contradictions": {"type": "array", "items": {"type": "string"}},
        "forbidden_reasoning_detected": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
}


FEW_SHOT_NEGATIVES: list[dict[str, Any]] = [
    {
        "event_type": "object_move",
        "evidence": {
            "object_label": "reagent_bottle",
            "hard_gate.status": "rejected",
            "reject_reason": "displacement_below_threshold",
            "stabilized_displacement_px": 3.2,
            "motion_threshold_px": 12.0,
        },
        "correct_output": {
            "decision": "reject",
            "should_write_confirmed_event": False,
            "reason": "The reagent bottle is detected but stabilized displacement is below the physical motion threshold.",
        },
    },
    {
        "event_type": "hand_object_interaction",
        "evidence": {"near_only": True, "contact_frames": 0},
        "correct_output": {
            "decision": "reject",
            "should_write_confirmed_event": False,
            "reason": "Hand-object proximity alone is not physical contact.",
        },
    },
    {
        "event_type": "liquid_transfer",
        "evidence": {"beaker_contains_liquid": True, "liquid_level_delta": None, "droplet": False, "stream": False},
        "correct_output": {
            "decision": "reject",
            "should_write_confirmed_event": False,
            "reason": "Static liquid presence does not prove liquid movement.",
        },
    },
    {
        "event_type": "panel_operation",
        "evidence": {"balance_detected": True, "hand_control_roi_contact": False, "display_change": False},
        "correct_output": {
            "decision": "reject",
            "should_write_confirmed_event": False,
            "reason": "Device presence alone is not panel operation.",
        },
    },
    {
        "event_type": "container_state_change",
        "evidence": {"bottle_detected": True, "pre_state": None, "post_state": None, "changed_fields": []},
        "correct_output": {
            "decision": "reject",
            "should_write_confirmed_event": False,
            "reason": "Container detection alone does not prove container state change.",
        },
    },
]


USER_PROMPT_TEMPLATE = """请审查下面这个实验视频物理事件候选。

你必须只根据 event_candidate 和 evidence_json 判断，不要根据实验常识补全不存在的证据。

event_candidate:
{event_candidate_json}

hard_gate:
{hard_gate_json}

evidence_json:
{evidence_json}

frame_context:
{frame_context_json}

allowed_track_ids:
{allowed_track_ids_json}

allowed_object_labels:
{allowed_object_labels_json}

如果有图像输入，请结合图像检查 evidence_json 是否与画面明显矛盾。
如果没有多帧图像，只能审查结构化 evidence_json，不能做时间动作推断。

禁止事项：
- 不要把 object presence 当作 object_move。
- 不要把 hand-object proximity 当作 contact。
- 不要把 static liquid presence 当作 liquid_transfer。
- 不要把 device presence 当作 panel_operation。
- 不要把 container presence 当作 container_state_change。
- 不要把 candidate / rejected / uncertain 的 hard_gate 提升成 confirmed。

请输出严格 JSON：

{{
  "decision": "accept | reject | uncertain",
  "event_type": "hand_object_interaction | object_move | liquid_transfer | panel_operation | container_state_change",
  "should_write_confirmed_event": true,
  "evidence_frame_indices": [],
  "evidence_time_sec": [],
  "supporting_track_ids": [],
  "supporting_object_labels": [],
  "reason": "",
  "missing_evidence": [],
  "contradictions": [],
  "forbidden_reasoning_detected": false,
  "confidence": 0.0
}}

字段要求：
- decision:
  - "accept" 只允许在 hard_gate.status == "confirmed" 且证据充分时使用。
  - hard_gate.status != "confirmed" 时必须是 "reject" 或 "uncertain"。
- should_write_confirmed_event:
  - 只有 decision == "accept" 且 hard_gate.status == "confirmed" 时才能为 true。
  - 其他情况必须为 false。
- evidence_frame_indices:
  - 只能使用输入中出现过的 frame_index。
- supporting_track_ids:
  - 只能使用 allowed_track_ids 中的 track_id。
- missing_evidence:
  - 如果证据不足，列出缺失项，例如 stable_instance_track、scene_stabilized_displacement、liquid_level_change、panel_state_change、container_pre_post_state。
- forbidden_reasoning_detected:
  - 如果 event_candidate 依赖物体出现、语义猜测、距离近、设备出现等非物理证据，则设为 true。

OUTPUT_SCHEMA:
{output_schema_json}

FEW_SHOT_NEGATIVES:
{few_shot_negatives_json}
"""


def build_physical_event_audit_prompt(
    event_candidate: Mapping[str, Any] | None = None,
    *,
    hard_gate: Mapping[str, Any] | None = None,
    evidence_json: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    frame_context: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    allowed_track_ids: Sequence[str] | None = None,
    allowed_object_labels: Sequence[str] | None = None,
    output_schema: Mapping[str, Any] | None = None,
    few_shot_negatives: Sequence[Mapping[str, Any]] | None = None,
    candidate_payload: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    event: Mapping[str, Any] | None = None,
    **extra_context: Any,
) -> dict[str, Any]:
    candidate_data = event_candidate or candidate_payload or candidate or event or {}
    hard_gate_data = hard_gate or _extract_hard_gate(candidate_data)
    evidence_data = evidence_json if evidence_json is not None else _extract_evidence(candidate_data)
    allowed_tracks = list(allowed_track_ids or _extract_track_ids(candidate_data, evidence_data))
    allowed_labels = list(allowed_object_labels or _extract_labels(candidate_data, evidence_data))
    frame_data = frame_context if frame_context is not None else extra_context.pop("frame_context_json", {})
    if extra_context:
        evidence_data = {"evidence": _json_safe(evidence_data), "extra_context": _json_safe(extra_context)}
    schema = _json_safe(output_schema or OUTPUT_SCHEMA)
    negatives = _json_safe(list(few_shot_negatives or FEW_SHOT_NEGATIVES))
    user_prompt = USER_PROMPT_TEMPLATE.format(
        event_candidate_json=_to_json(candidate_data),
        hard_gate_json=_to_json(hard_gate_data),
        evidence_json=_to_json(evidence_data),
        frame_context_json=_to_json(frame_data),
        allowed_track_ids_json=_to_json(allowed_tracks),
        allowed_object_labels_json=_to_json(allowed_labels),
        output_schema_json=_to_json(schema),
        few_shot_negatives_json=_to_json(negatives),
    )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "output_schema": schema,
        "temperature": 0.0,
    }


def _extract_hard_gate(candidate_data: Mapping[str, Any]) -> Any:
    gate = candidate_data.get("hard_gate") if isinstance(candidate_data, Mapping) else None
    if gate:
        return gate
    physical_gate = candidate_data.get("physical_event_gate") if isinstance(candidate_data, Mapping) else None
    if isinstance(physical_gate, Mapping):
        return physical_gate.get("hard_gate") or physical_gate
    return {}


def _extract_evidence(candidate_data: Mapping[str, Any]) -> Any:
    if not isinstance(candidate_data, Mapping):
        return {}
    return candidate_data.get("evidence_detail") or candidate_data.get("evidence") or candidate_data.get("metrics") or {}


def _extract_track_ids(candidate_data: Mapping[str, Any], evidence_data: Any) -> list[str]:
    values: list[str] = []
    if isinstance(candidate_data, Mapping):
        for key in ("actor_track_id", "primary_track_id", "tool_track_id"):
            if candidate_data.get(key):
                values.append(str(candidate_data[key]))
        for key in ("object_track_ids", "involved_track_ids", "supporting_track_ids"):
            values.extend(str(item) for item in candidate_data.get(key) or [] if item)
    if isinstance(evidence_data, Mapping):
        for key in ("track_id", "hand_track_id", "object_track_id", "tool_track_id", "source_container_id", "target_container_id"):
            if evidence_data.get(key):
                values.append(str(evidence_data[key]))
    return sorted(set(values))


def _extract_labels(candidate_data: Mapping[str, Any], evidence_data: Any) -> list[str]:
    values: list[str] = []
    if isinstance(candidate_data, Mapping):
        for key in ("event_type", "object_label", "primary_object", "dominant_object"):
            if candidate_data.get(key):
                values.append(str(candidate_data[key]))
        for key in ("object_labels", "involved_objects", "supporting_object_labels"):
            values.extend(str(item) for item in candidate_data.get(key) or [] if item)
    if isinstance(evidence_data, Mapping):
        for key in ("object_label",):
            if evidence_data.get(key):
                values.append(str(evidence_data[key]))
    return sorted(set(values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_safe(item) for item in value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _to_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2, sort_keys=True)


__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "OUTPUT_SCHEMA",
    "FEW_SHOT_NEGATIVES",
    "build_physical_event_audit_prompt",
]
