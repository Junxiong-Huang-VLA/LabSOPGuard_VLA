from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "sop_state_machine/v1"

ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "weighing": ("weigh", "balance", "mass", "称", "天平"),
    "pipetting": ("pipette", "transfer", "add sample", "liquid transfer", "移液", "加样", "加入"),
    "sample_handling": ("sample", "tube", "vial", "bottle", "container", "样品", "试管", "瓶", "容器"),
    "recording": ("record", "readout", "note", "记录", "读数"),
    "mixing": ("mix", "vortex", "shake", "stir", "混合", "振荡", "搅拌"),
    "incubating": ("incubate", "wait", "hold", "等待", "孵育", "静置"),
    "cleanup": ("clean", "discard", "dispose", "rinse", "清理", "丢弃", "冲洗"),
}

PARAM_PATTERN = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ul|uL|mL|ml|g|mg|s|sec|min|h|hour|C|°C|℃|微升|毫升|克|毫克|分钟|秒|小时)",
    re.IGNORECASE,
)
NUMBERED_STEP_PATTERN = re.compile(
    r"^\s*(?:step\s*)?(?P<num>\d+(?:\.\d+)*|[A-Za-z])[\).、:\s-]+(?P<body>.+?)\s*$",
    re.IGNORECASE,
)


def build_sop_state_machine(
    sop_source: str | Path | Mapping[str, Any] | list[Any] | None = None,
    *,
    context: Mapping[str, Any] | None = None,
    video_rows: list[Mapping[str, Any]] | None = None,
    asset_rows: list[Mapping[str, Any]] | None = None,
    history_model: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a deterministic SOP state machine from JSON, text, context, or history."""

    payload, source_ref = _load_sop_payload(sop_source)
    raw_steps = _steps_from_payload(payload, context or {}, video_rows or [], history_model or {})
    steps = [
        _normalize_step_template(
            step,
            index=index,
            source_ref=source_ref,
            context=context or {},
            asset_rows=asset_rows or [],
        )
        for index, step in enumerate(raw_steps)
    ]
    transitions = _build_transitions(steps)
    machine = {
        "schema_version": SCHEMA_VERSION,
        "source": source_ref,
        "step_count": len(steps),
        "initial_step_id": steps[0]["step_id"] if steps else None,
        "terminal_step_ids": [steps[-1]["step_id"]] if steps else [],
        "steps": steps,
        "transitions": transitions,
        "transition_priors": _transition_priors(history_model or {}),
        "status_values": [
            "not_observed",
            "in_progress",
            "waiting",
            "completed",
            "inferred_missing",
            "skipped",
            "branch_not_taken",
            "condition_failed",
            "human_confirmed",
            "human_rejected",
        ],
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(machine, ensure_ascii=False, indent=2), encoding="utf-8")
    return machine


def load_sop_state_machine(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8-sig"))


def _load_sop_payload(source: str | Path | Mapping[str, Any] | list[Any] | None) -> tuple[Any, dict[str, Any]]:
    if source is None:
        return None, {"type": "inferred"}
    if isinstance(source, Mapping):
        return dict(source), {"type": "mapping"}
    if isinstance(source, list):
        return list(source), {"type": "list"}
    path = Path(source)
    if path.exists():
        text = path.read_text(encoding="utf-8-sig")
        try:
            return json.loads(text), {"type": "file", "path": str(path), "format": "json"}
        except json.JSONDecodeError:
            return text, {"type": "file", "path": str(path), "format": "text"}
    text = str(source)
    try:
        return json.loads(text), {"type": "inline", "format": "json"}
    except json.JSONDecodeError:
        return text, {"type": "inline", "format": "text"}


def _steps_from_payload(
    payload: Any,
    context: Mapping[str, Any],
    video_rows: list[Mapping[str, Any]],
    history_model: Mapping[str, Any],
) -> list[Any]:
    if isinstance(payload, Mapping):
        for key in ("steps", "step_templates", "nodes"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        if isinstance(payload.get("state_machine"), Mapping):
            value = payload["state_machine"].get("steps")
            if isinstance(value, list):
                return value
        for key in ("sop_text", "text", "procedure_text", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return _parse_text_steps(value)
        if payload.get("expected_action") or payload.get("name"):
            return [payload]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, str) and payload.strip():
        return _parse_text_steps(payload)

    procedure = context.get("procedure_candidates") if isinstance(context.get("procedure_candidates"), list) else []
    actions = [str(item.get("action_type")) for item in procedure if isinstance(item, Mapping) and item.get("action_type")]
    if not actions:
        recommended = history_model.get("recommended_sop") if isinstance(history_model.get("recommended_sop"), list) else []
        if recommended:
            return recommended
    if not actions:
        actions = _dedupe(str(row.get("action_type") or "") for row in video_rows if row.get("action_type"))
    return [{"expected_action": action} for action in actions]


def _parse_text_steps(text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending_text: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = NUMBERED_STEP_PATTERN.match(line)
        if match:
            if pending_text:
                steps.append(_step_from_text(" ".join(pending_text), len(steps)))
                pending_text = []
            body = match.group("body")
            step = _step_from_text(body, len(steps))
            step["source_step_number"] = match.group("num")
            steps.append(step)
        elif line.startswith(("-", "*")):
            if pending_text:
                steps.append(_step_from_text(" ".join(pending_text), len(steps)))
                pending_text = []
            steps.append(_step_from_text(line.lstrip("-* ").strip(), len(steps)))
        else:
            pending_text.append(line)
    if pending_text:
        merged = " ".join(pending_text)
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.;。；])\s+", merged) if part.strip()]
        if len(sentence_parts) > 1:
            steps.extend(_step_from_text(part, len(steps) + idx) for idx, part in enumerate(sentence_parts))
        else:
            steps.append(_step_from_text(merged, len(steps)))
    return steps


def _step_from_text(text: str, index: int) -> dict[str, Any]:
    lowered = text.lower()
    action = _infer_action(text) or f"step_{index + 1:03d}"
    step: dict[str, Any] = {
        "step_id": f"step_{index + 1:03d}",
        "name": _short_name(text, action),
        "expected_action": action,
        "description": text,
        "completion_conditions": [{"any_actions": [action]}],
        "evidence_requirements": _default_evidence_requirements(action, text),
        "parameters": _parameters(text),
    }
    if any(token in lowered for token in ("if ", "when ", "unless ", "如果", "若", "当")):
        step["branch_condition"] = {"text_keywords": _condition_keywords(text)}
    if any(token in lowered for token in ("repeat", "until", "重复", "循环", "直至", "直到")):
        step["repeatable"] = True
        step["repeat_until"] = {"text_keywords": _condition_keywords(text)}
    if any(token in lowered for token in ("wait", "hold", "incubate", "等待", "静置", "孵育")):
        step["wait_conditions"] = {"text_keywords": _condition_keywords(text)}
    if any(token in lowered for token in ("parallel", "simultaneously", "meanwhile", "同时", "并行")):
        step["parallel_observations"] = [{"observation_id": "parallel_001", "condition": {"text_keywords": _condition_keywords(text)}}]
    return step


def _normalize_step_template(
    step: Any,
    *,
    index: int,
    source_ref: Mapping[str, Any],
    context: Mapping[str, Any],
    asset_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    data = dict(step) if isinstance(step, Mapping) else {"name": str(step)}
    expected = str(
        data.get("expected_action")
        or data.get("action_type")
        or _infer_action(str(data.get("description") or data.get("name") or ""))
        or data.get("id")
        or f"step_{index + 1:03d}"
    )
    step_id = str(data.get("step_id") or data.get("id") or f"step_{index + 1:03d}")
    conditions = data.get("conditions") if isinstance(data.get("conditions"), Mapping) else {}
    entry_conditions = data.get("entry_conditions")
    completion_conditions = data.get("completion_conditions")
    branch_condition = data.get("branch_condition")
    if entry_conditions is None:
        entry_conditions = conditions.get("entry_conditions") or conditions.get("entry")
    if completion_conditions is None:
        completion_conditions = conditions.get("completion_conditions") or conditions.get("completion")
    if branch_condition is None:
        branch_condition = conditions.get("branch_condition") or conditions.get("branch")
    repeat_until = data.get("repeat_until") or conditions.get("repeat_until")
    wait_conditions = data.get("wait_conditions") or data.get("wait_until") or conditions.get("wait_until")
    parallel_observations = data.get("parallel_observations") or data.get("parallel") or conditions.get("parallel_observations")
    allowed = _allowed_transitions(data)
    return {
        "step_id": step_id,
        "order": int(data.get("order") or index + 1),
        "name": str(data.get("name") or _default_name(expected)),
        "description": str(data.get("description") or data.get("text") or ""),
        "expected_action": expected,
        "entry_conditions": _as_list(entry_conditions),
        "completion_conditions": _as_list(completion_conditions) or [{"any_actions": [expected]}],
        "evidence_requirements": _as_list(data.get("evidence_requirements")) or _default_evidence_requirements(expected, str(data)),
        "allowed_transitions": allowed,
        "branch_condition": branch_condition or {},
        "branch_id": data.get("branch_id"),
        "branch_type": data.get("branch_type") or data.get("branch"),
        "repeatable": bool(data.get("repeatable") or repeat_until),
        "repeatable_explicit": "repeatable" in data,
        "min_repeats": int(data.get("min_repeats") or 1),
        "max_repeats": int(data.get("max_repeats") or 999999),
        "repeat_until": repeat_until or {},
        "wait_conditions": wait_conditions or {},
        "parallel_observations": _as_list(parallel_observations),
        "required_parallel_observations": bool(data.get("required_parallel_observations", True)),
        "required": bool(data.get("required", True)),
        "required_material": data.get("required_material") or data.get("required_materials") or data.get("requires_material"),
        "min_confidence": data.get("min_confidence"),
        "max_elapsed_sec": data.get("max_elapsed_sec"),
        "parameters": _as_list(data.get("parameters")),
        "context_support": _context_support(expected, context, asset_rows),
        "source_refs": _source_refs(step_id, source_ref),
    }


def _build_transitions(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    by_id = {str(step.get("step_id")) for step in steps}
    for index, step in enumerate(steps):
        source = str(step.get("step_id"))
        allowed = [item for item in _as_list(step.get("allowed_transitions")) if item]
        if not allowed and index + 1 < len(steps):
            allowed = [steps[index + 1]["step_id"]]
        if step.get("repeatable") and step.get("repeat_until"):
            transitions.append(
                {
                    "from_step_id": source,
                    "to_step_id": source,
                    "transition_type": "repeat_until",
                    "condition": step.get("repeat_until") or {},
                }
            )
        for destination in allowed:
            destination_id = str(destination.get("step_id") if isinstance(destination, Mapping) else destination)
            if destination_id not in by_id:
                continue
            transitions.append(
                {
                    "from_step_id": source,
                    "to_step_id": destination_id,
                    "transition_type": "normal",
                    "condition": destination.get("condition") if isinstance(destination, Mapping) else {},
                }
            )
    return transitions


def _allowed_transitions(data: Mapping[str, Any]) -> list[Any]:
    for key in ("allowed_transitions", "next_steps", "next_step_ids", "next", "on_success"):
        if key in data:
            return _as_list(data.get(key))
    return []


def _transition_priors(history_model: Mapping[str, Any]) -> dict[str, Any]:
    transitions = history_model.get("transition_probabilities")
    if not isinstance(transitions, Mapping):
        return {}
    return {
        str(source): {str(destination): float(probability or 0.0) for destination, probability in dict(options).items()}
        for source, options in transitions.items()
        if isinstance(options, Mapping)
    }


def _infer_action(text: str) -> str:
    lowered = str(text or "").lower()
    for action, keywords in ACTION_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return action
    return ""


def _default_name(action: str) -> str:
    return action.replace("_", " ").title()


def _short_name(text: str, action: str) -> str:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        return _default_name(action)
    return stripped[:80]


def _parameters(text: str) -> list[dict[str, Any]]:
    values = []
    for match in PARAM_PATTERN.finditer(text):
        values.append({"value": float(match.group("value")), "unit": match.group("unit"), "text": match.group(0)})
    return values


def _condition_keywords(text: str) -> list[str]:
    words = re.findall(r"[\w\u4e00-\u9fff]+", text)
    return [word for word in _dedupe(words) if len(word) >= 2][:8]


def _default_evidence_requirements(action: str, text: str) -> list[dict[str, Any]]:
    requirements = [{"type": "video_action", "action": action}]
    lowered = f"{action} {text}".lower()
    if any(token in lowered for token in ("pipette", "sample", "tube", "bottle", "container", "移液", "样品")):
        requirements.append({"type": "hand_object_interaction"})
    if any(token in lowered for token in ("state", "change", "open", "close", "level", "color", "状态", "变化")):
        requirements.append({"type": "state_change"})
    return requirements


def _context_support(expected: str, context: Mapping[str, Any], asset_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_types: set[str] = set()
    procedure_score = 0.0
    for item in _as_list(context.get("procedure_candidates")):
        if not isinstance(item, Mapping) or str(item.get("action_type") or "") != expected:
            continue
        procedure_score = max(procedure_score, float(item.get("score") or 0.0))
        source_types.update(str(source) for source in _as_list(item.get("source_types")))
    material_hits = []
    for row in asset_rows:
        haystack = " ".join(str(value) for value in _flatten_values(row)).lower()
        if expected.lower() in haystack:
            material_hits.append(row.get("asset_id") or row.get("path"))
    return {
        "procedure_score": procedure_score,
        "source_types": sorted(source_types),
        "asset_hits": [str(item) for item in material_hits if item][:5],
    }


def _source_refs(step_id: str, source_ref: Mapping[str, Any]) -> list[dict[str, Any]]:
    ref = {"type": "sop_step", "step_id": step_id}
    ref.update({key: value for key, value in source_ref.items() if value is not None})
    return [ref]


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        values: list[Any] = []
        for item in value.values():
            values.extend(_flatten_values(item))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_flatten_values(item))
        return values
    return [value]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


__all__ = ["SCHEMA_VERSION", "build_sop_state_machine", "load_sop_state_machine"]
