from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .history_learning import load_history_model, score_process_with_history, search_similar_history
from .schemas import read_jsonl, write_jsonl
from .sop_state_machine import build_sop_state_machine
from .time_alignment import parse_time


DEFAULT_ACTION_NAMES = {
    "weighing": "Weigh sample",
    "pipetting": "Transfer liquid with pipette",
    "sample_adding_candidate": "Add sample or liquid",
    "sample_handling": "Handle sample container",
    "recording": "Record readout",
}
NON_BLOCKING_VIDEO_ANOMALY_FLAGS = {
    "advanced_evidence_candidate",
    "evidence_limitation_missing_visual_or_transcript",
    "heuristic_candidate",
    "low_confidence_candidate_event",
    "not_container_open_close_confirmed",
    "not_panel_ocr_confirmed",
    "not_visual_liquid_flow_confirmed",
    "requires_human_confirmation",
    "visual_confirmation_limited",
}
STEP_FIELDS = (
    "step_id",
    "name",
    "expected_action",
    "status",
    "observed",
    "inferred",
    "completed",
    "skipped",
    "repeated",
    "abnormal",
    "confidence",
    "confidence_reasons",
    "entry_conditions",
    "completion_conditions",
    "global_start_time",
    "global_end_time",
    "evidence_refs",
    "missing_completion_reason",
    "next_step_hint",
    "requires_human_confirmation",
    "repeat_count",
    "conflict_flags",
    "branch_condition",
    "branch_enabled",
    "auto_confirmed",
    "confirmation_status",
    "condition_results",
    "history_prior",
    "history_deviation",
    "history_basis",
    "inference_source",
    "inference_reason",
    "inference_confidence",
    "inferred_from_step_ids",
    "evidence_level",
    "evidence_strength",
    "reasoning",
    "text_context_refs",
    "sop_refs",
    "history_refs",
    "wait_conditions",
    "repeat_until",
    "parallel_observations",
    "transition_options",
)


def build_experiment_process(
    session_dir: str | Path,
    sop_path: str | Path | None = None,
    output_path: str | Path | None = None,
    timeline_output_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    metadata = session / "metadata"
    video_rows = _read_jsonl_if_exists(metadata / "video_understanding.jsonl")
    state_rows = _read_jsonl_if_exists(metadata / "state_change_index.jsonl")
    asset_rows = _read_jsonl_if_exists(metadata / "material_asset_catalog.jsonl")
    sop_rows = _read_jsonl_if_exists(metadata / "sop_records.jsonl")
    context = _read_json_if_exists(metadata / "experiment_context.json")
    history_model = load_history_model(metadata / "history_model.json")
    target = Path(output_path) if output_path is not None else metadata / "experiment_process.json"
    timeline_target = Path(timeline_output_path) if timeline_output_path is not None else metadata / "experiment_process_timeline.jsonl"

    sop_source: Any = sop_path
    if sop_source is None and sop_rows:
        sop_source = {"steps": sop_rows}
    state_machine_path = metadata / "sop_state_machine.json"
    state_machine = build_sop_state_machine(
        sop_source,
        context=context,
        video_rows=video_rows,
        asset_rows=asset_rows,
        history_model=history_model,
        output_path=state_machine_path,
    )
    observed_actions = _observed_action_families(video_rows)
    sop_steps = [
        _normalize_sop_step(step, index, observed_actions, context, video_rows, asset_rows)
        for index, step in enumerate(state_machine.get("steps", []))
    ]
    observations = _observations_by_action(video_rows)
    steps = [_build_step(step, index, sop_steps, observations, video_rows, state_rows, asset_rows, context) for index, step in enumerate(sop_steps)]
    steps = _apply_elapsed_conditions(steps, sop_steps)
    if history_model:
        steps = _apply_history_priors(steps, history_model)
        steps = _mark_history_conflicts(steps)
    steps = _infer_missing_steps(steps)
    steps = _mark_order_conflicts(steps)
    current_step, next_step = _current_and_next(steps)
    timeline_rows = _process_timeline_rows(steps)
    write_jsonl(timeline_target, timeline_rows)

    status_counts = Counter(str(step.get("status") or "unknown") for step in steps)
    related_history_records = search_similar_history({"steps": steps}, history_model) if history_model else []
    result = {
        "session_id": context.get("session_id") or _session_id(video_rows, state_rows, asset_rows, session),
        "process_status": "completed" if steps and all(step.get("completed") or step.get("skipped") for step in steps) else "in_progress",
        "current_step_id": current_step.get("step_id") if current_step else None,
        "next_step_id": next_step.get("step_id") if next_step else None,
        "completed_step_ids": [step.get("step_id") for step in steps if step.get("completed") and not step.get("skipped")],
        "pending_confirmation_step_ids": [step.get("step_id") for step in steps if step.get("requires_human_confirmation")],
        "step_count": len(steps),
        "status_counts": dict(sorted(status_counts.items())),
        "steps": steps,
        "state_machine_path": str(state_machine_path),
        "state_machine": state_machine,
        "related_history_records": related_history_records,
        "conflict_report": _conflict_report(steps, context, related_history_records),
        "timeline_path": str(timeline_target),
        "evidence_index": _evidence_index(steps),
    }
    if history_model:
        history_score = score_process_with_history(result, history_model)
        result["history_prior"] = history_score.get("history_prior", {})
        result["history_deviation"] = history_score.get("history_deviation", {})
        result["history_score"] = history_score
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_experiment_process(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    return json.loads(source.read_text(encoding="utf-8-sig"))


def _load_sop_steps(
    sop_path: str | Path | None,
    context: Mapping[str, Any],
    video_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
    sop_rows: list[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    observed_actions = _observed_action_families(video_rows)
    if sop_path:
        data = json.loads(Path(sop_path).read_text(encoding="utf-8-sig"))
        steps = data.get("steps") if isinstance(data, Mapping) else data
        if isinstance(steps, list):
            return [_normalize_sop_step(step, index, observed_actions, context, video_rows, asset_rows) for index, step in enumerate(steps)]
    if sop_rows:
        return [_normalize_sop_step(step, index, observed_actions, context, video_rows, asset_rows) for index, step in enumerate(sop_rows)]
    procedure = context.get("procedure_candidates") if isinstance(context.get("procedure_candidates"), list) else []
    actions = [str(item.get("action_type")) for item in procedure if isinstance(item, Mapping) and item.get("action_type")]
    if not actions:
        actions = _dedupe(str(row.get("action_type") or "") for row in video_rows if row.get("action_type"))
    return [_normalize_sop_step({"expected_action": action}, index, observed_actions, context, video_rows, asset_rows) for index, action in enumerate(actions)]


def _normalize_sop_step(
    step: Any,
    index: int,
    observed_actions: set[str],
    context: Mapping[str, Any],
    video_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    data = dict(step) if isinstance(step, Mapping) else {"name": str(step)}
    expected = str(data.get("expected_action") or data.get("action_type") or data.get("id") or data.get("name") or f"step_{index + 1}")
    conditions = data.get("conditions") if isinstance(data.get("conditions"), Mapping) else {}
    condition_sections = {"entry", "completion", "branch", "entry_conditions", "completion_conditions", "branch_condition"}
    has_condition_sections = bool(condition_sections & set(conditions))
    branch_condition = data.get("branch_condition")
    if branch_condition is None and has_condition_sections:
        branch_condition = conditions.get("branch_condition") or conditions.get("branch")
    if branch_condition is None and not has_condition_sections:
        branch_condition = data.get("conditions")
    branch_condition = branch_condition or {}
    entry_conditions = data.get("entry_conditions")
    if entry_conditions is None and has_condition_sections:
        entry_conditions = conditions.get("entry_conditions") or conditions.get("entry")
    completion_conditions = data.get("completion_conditions")
    if completion_conditions is None and has_condition_sections:
        completion_conditions = conditions.get("completion_conditions") or conditions.get("completion")
    repeat_until = data.get("repeat_until") or conditions.get("repeat_until")
    wait_conditions = data.get("wait_conditions") or data.get("wait_until") or conditions.get("wait_until")
    parallel_observations = data.get("parallel_observations") or data.get("parallel") or conditions.get("parallel_observations")
    branch_result = _evaluate_sop_condition(
        branch_condition,
        all_events=video_rows,
        confidence_events=video_rows,
        context=context,
        asset_rows=asset_rows,
    )
    branch_enabled = branch_result["passed"]
    return {
        "step_id": str(data.get("step_id") or data.get("id") or f"step_{index + 1:03d}"),
        "name": str(data.get("name") or DEFAULT_ACTION_NAMES.get(expected, expected.replace("_", " ").title())),
        "expected_action": expected,
        "entry_conditions": entry_conditions or [],
        "completion_conditions": completion_conditions or [f"observe {expected}"],
        "evidence_requirements": data.get("evidence_requirements") or [],
        "repeatable": bool(data.get("repeatable", False)),
        "repeatable_explicit": bool(data.get("repeatable_explicit") if "repeatable_explicit" in data else "repeatable" in data),
        "min_repeats": int(data.get("min_repeats") or 1),
        "max_repeats": int(data.get("max_repeats") or 999999),
        "repeat_until": repeat_until or {},
        "wait_conditions": wait_conditions or {},
        "parallel_observations": _list(parallel_observations),
        "required_parallel_observations": bool(data.get("required_parallel_observations", True)),
        "required": bool(data.get("required", True)),
        "required_material": data.get("required_material") or data.get("required_materials") or data.get("requires_material"),
        "min_confidence": data.get("min_confidence"),
        "max_elapsed_sec": data.get("max_elapsed_sec"),
        "allowed_transitions": _list(data.get("allowed_transitions")),
        "source_refs": _list(data.get("source_refs")),
        "context_support": data.get("context_support") if isinstance(data.get("context_support"), Mapping) else {},
        "parameters": _list(data.get("parameters")),
        "branch_condition": branch_condition,
        "branch_enabled": branch_enabled,
        "branch_condition_result": branch_result,
    }


def _observations_by_action(video_rows: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in video_rows:
        action = str(row.get("action_type") or "")
        event_type = str(row.get("event_type") or "")
        keys = _dedupe([action, event_type, _action_family(action), _action_family(event_type)])
        for key in keys:
            if key:
                observations[key].append(dict(row))
    return observations


def _observed_action_families(video_rows: list[Mapping[str, Any]]) -> set[str]:
    values: set[str] = set()
    for row in video_rows:
        for value in (row.get("action_type"), row.get("event_type")):
            text = str(value or "")
            if text:
                values.add(text)
                values.add(_action_family(text))
    return {value for value in values if value}


def _branch_enabled(condition: Any, observed_actions: set[str], context: Mapping[str, Any]) -> bool:
    if not condition:
        return True
    data = dict(condition) if isinstance(condition, Mapping) else {}
    any_actions = {_action_family(str(item)) for item in _list(data.get("when_any_action_observed"))}
    if any_actions and not (any_actions & observed_actions):
        return False
    unless_actions = {_action_family(str(item)) for item in _list(data.get("unless_action_observed"))}
    if unless_actions and unless_actions & observed_actions:
        return False
    required_materials = {str(item) for item in _list(data.get("requires_material"))}
    if required_materials:
        materials = {str(item.get("name")) for item in _list(context.get("materials")) if isinstance(item, Mapping)}
        if not required_materials & materials:
            return False
    return True


def _completion_condition_for_step(sop_step: Mapping[str, Any]) -> list[Any]:
    conditions = list(_list(sop_step.get("completion_conditions")))
    extra = {
        key: sop_step.get(key)
        for key in ("required_material", "min_confidence", "max_elapsed_sec")
        if sop_step.get(key) is not None
    }
    if extra:
        conditions.append(extra)
    return conditions


def _evaluate_sop_condition(
    condition: Any,
    *,
    all_events: list[Mapping[str, Any]],
    confidence_events: list[Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
    asset_rows: list[Mapping[str, Any]] | None = None,
    state_rows: list[Mapping[str, Any]] | None = None,
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    facts = _condition_facts(
        all_events=all_events,
        confidence_events=confidence_events if confidence_events is not None else all_events,
        context=context or {},
        asset_rows=asset_rows or [],
        state_rows=state_rows or [],
        elapsed_sec=elapsed_sec,
    )
    passed, failures, reasons = _condition_passed(condition, facts)
    return {"passed": passed, "failures": _dedupe(failures), "reasons": _dedupe(reasons)}


def _condition_passed(condition: Any, facts: Mapping[str, Any]) -> tuple[bool, list[str], list[str]]:
    if not condition:
        return True, [], []
    if isinstance(condition, str):
        return True, [], []
    if isinstance(condition, list):
        passed = True
        failures: list[str] = []
        reasons: list[str] = []
        for item in condition:
            item_passed, item_failures, item_reasons = _condition_passed(item, facts)
            passed = passed and item_passed
            failures.extend(item_failures)
            reasons.extend(item_reasons)
        return passed, failures, reasons
    if not isinstance(condition, Mapping):
        return True, [], []

    data = dict(condition)
    passed = True
    failures: list[str] = []
    reasons: list[str] = []

    if "all" in data:
        item_passed, item_failures, item_reasons = _logical_condition_passed(data["all"], facts, mode="all")
        passed = passed and item_passed
        failures.extend(item_failures)
        reasons.extend(item_reasons)
    if "any" in data:
        item_passed, item_failures, item_reasons = _logical_condition_passed(data["any"], facts, mode="any")
        passed = passed and item_passed
        failures.extend(item_failures)
        reasons.extend(item_reasons)
    if "not" in data:
        item_passed, item_failures, item_reasons = _not_condition_passed(data["not"], facts)
        passed = passed and item_passed
        failures.extend(item_failures)
        reasons.extend(item_reasons)

    all_actions = _condition_values(data, ("all_actions", "all_action", "action_all", "when_all_action_observed"))
    if all_actions:
        missing = [action for action in all_actions if not _action_present(action, facts)]
        if missing:
            passed = False
            failures.append("all_actions_not_observed")
            reasons.append(f"missing_actions={','.join(str(item) for item in missing)}")
        else:
            reasons.append(f"all_actions_observed={','.join(str(item) for item in all_actions)}")

    any_actions = _condition_values(data, ("any_actions", "any_action", "action_any", "when_any_action_observed", "action"))
    if any_actions:
        if not any(_action_present(action, facts) for action in any_actions):
            passed = False
            failures.append("any_action_not_observed")
            reasons.append(f"required_any_action={','.join(str(item) for item in any_actions)}")
        else:
            reasons.append(f"any_action_observed={','.join(str(item) for item in any_actions)}")

    not_actions = _condition_values(data, ("not_actions", "not_action", "action_not", "unless_action_observed"))
    if not_actions:
        observed = [action for action in not_actions if _action_present(action, facts)]
        if observed:
            passed = False
            failures.append("not_action_observed")
            reasons.append(f"forbidden_actions={','.join(str(item) for item in observed)}")
        else:
            reasons.append(f"not_actions_absent={','.join(str(item) for item in not_actions)}")

    materials = _condition_values(data, ("required_material", "required_materials", "requires_material"))
    if materials:
        if not any(_material_present(material, facts) for material in materials):
            passed = False
            failures.append("required_material_missing")
            reasons.append(f"required_material={','.join(str(item) for item in materials)}")
        else:
            reasons.append(f"required_material_present={','.join(str(item) for item in materials)}")

    all_materials = _condition_values(data, ("all_required_materials", "requires_all_materials"))
    if all_materials:
        missing_materials = [material for material in all_materials if not _material_present(material, facts)]
        if missing_materials:
            passed = False
            failures.append("all_required_materials_missing")
            reasons.append(f"missing_materials={','.join(str(item) for item in missing_materials)}")
        else:
            reasons.append(f"all_required_materials_present={','.join(str(item) for item in all_materials)}")

    objects = _condition_values(data, ("object", "objects", "any_objects", "primary_object", "required_object", "required_objects"))
    if objects:
        if not any(_object_present(obj, facts) for obj in objects):
            passed = False
            failures.append("required_object_missing")
            reasons.append(f"required_object={','.join(str(item) for item in objects)}")
        else:
            reasons.append(f"required_object_present={','.join(str(item) for item in objects)}")

    all_objects = _condition_values(data, ("all_objects", "requires_all_objects"))
    if all_objects:
        missing_objects = [obj for obj in all_objects if not _object_present(obj, facts)]
        if missing_objects:
            passed = False
            failures.append("all_required_objects_missing")
            reasons.append(f"missing_objects={','.join(str(item) for item in missing_objects)}")
        else:
            reasons.append(f"all_required_objects_present={','.join(str(item) for item in all_objects)}")

    states = _condition_values(data, ("state", "state_type", "state_change", "any_states", "any_state_types"))
    if states:
        if not any(_state_present(state, facts) for state in states):
            passed = False
            failures.append("required_state_missing")
            reasons.append(f"required_state={','.join(str(item) for item in states)}")
        else:
            reasons.append(f"required_state_present={','.join(str(item) for item in states)}")

    all_states = _condition_values(data, ("all_states", "all_state_types"))
    if all_states:
        missing_states = [state for state in all_states if not _state_present(state, facts)]
        if missing_states:
            passed = False
            failures.append("all_required_states_missing")
            reasons.append(f"missing_states={','.join(str(item) for item in missing_states)}")
        else:
            reasons.append(f"all_required_states_present={','.join(str(item) for item in all_states)}")

    any_text = _condition_values(data, ("text_keywords", "any_text", "any_text_keywords"))
    if any_text:
        if not any(_text_present(keyword, facts) for keyword in any_text):
            passed = False
            failures.append("required_text_missing")
            reasons.append(f"required_text={','.join(str(item) for item in any_text)}")
        else:
            reasons.append(f"required_text_present={','.join(str(item) for item in any_text)}")

    all_text = _condition_values(data, ("all_text", "all_text_keywords"))
    if all_text:
        missing_text = [keyword for keyword in all_text if not _text_present(keyword, facts)]
        if missing_text:
            passed = False
            failures.append("all_required_text_missing")
            reasons.append(f"missing_text={','.join(str(item) for item in missing_text)}")
        else:
            reasons.append(f"all_required_text_present={','.join(str(item) for item in all_text)}")

    min_confidence = _condition_float(data.get("min_confidence"))
    if min_confidence is not None:
        max_confidence = float(facts.get("max_confidence") or 0.0)
        if max_confidence < min_confidence:
            passed = False
            failures.append("min_confidence_not_met")
            reasons.append(f"max_confidence={max_confidence:.3f}<min_confidence={min_confidence:.3f}")
        else:
            reasons.append(f"max_confidence={max_confidence:.3f}>=min_confidence={min_confidence:.3f}")

    max_elapsed_sec = _condition_float(data.get("max_elapsed_sec"))
    if max_elapsed_sec is not None:
        elapsed = facts.get("elapsed_sec")
        if elapsed is None:
            reasons.append("elapsed_sec_unavailable")
        elif float(elapsed) > max_elapsed_sec:
            passed = False
            failures.append("max_elapsed_sec_exceeded")
            reasons.append(f"elapsed_sec={float(elapsed):.3f}>max_elapsed_sec={max_elapsed_sec:.3f}")
        else:
            reasons.append(f"elapsed_sec={float(elapsed):.3f}<=max_elapsed_sec={max_elapsed_sec:.3f}")

    return passed, failures, reasons


def _logical_condition_passed(value: Any, facts: Mapping[str, Any], mode: str) -> tuple[bool, list[str], list[str]]:
    items = _list(value)
    if items and all(not isinstance(item, Mapping) for item in items):
        if mode == "all":
            missing = [item for item in items if not _action_present(item, facts)]
            if missing:
                return False, ["all_actions_not_observed"], [f"missing_actions={','.join(str(item) for item in missing)}"]
            return True, [], [f"all_actions_observed={','.join(str(item) for item in items)}"]
        if not any(_action_present(item, facts) for item in items):
            return False, ["any_action_not_observed"], [f"required_any_action={','.join(str(item) for item in items)}"]
        return True, [], [f"any_action_observed={','.join(str(item) for item in items)}"]

    outcomes = [_condition_passed(item, facts) for item in items]
    if not outcomes:
        return True, [], []
    if mode == "all":
        return (
            all(item[0] for item in outcomes),
            [failure for item in outcomes for failure in item[1]],
            [reason for item in outcomes for reason in item[2]],
        )
    if any(item[0] for item in outcomes):
        return True, [], [reason for item in outcomes for reason in item[2]]
    failures = [failure for item in outcomes for failure in item[1]]
    reasons = [reason for item in outcomes for reason in item[2]]
    return False, failures or ["any_condition_not_met"], reasons


def _not_condition_passed(value: Any, facts: Mapping[str, Any]) -> tuple[bool, list[str], list[str]]:
    items = _list(value)
    if items and all(not isinstance(item, Mapping) for item in items):
        observed = [item for item in items if _action_present(item, facts)]
        if observed:
            return False, ["not_action_observed"], [f"forbidden_actions={','.join(str(item) for item in observed)}"]
        return True, [], [f"not_actions_absent={','.join(str(item) for item in items)}"]
    nested_passed, _failures, reasons = _condition_passed(value, facts)
    if nested_passed:
        return False, ["not_condition_observed"], reasons
    return True, [], reasons


def _condition_facts(
    *,
    all_events: list[Mapping[str, Any]],
    confidence_events: list[Mapping[str, Any]],
    context: Mapping[str, Any],
    asset_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
    elapsed_sec: float | None,
) -> dict[str, Any]:
    return {
        "actions": _observed_action_families(all_events),
        "materials": _material_labels(context, all_events, asset_rows, state_rows),
        "objects": _object_labels(all_events, asset_rows, state_rows),
        "states": _state_labels(all_events, state_rows),
        "texts": _text_labels(context, all_events, asset_rows, state_rows),
        "max_confidence": _max_confidence(confidence_events),
        "elapsed_sec": elapsed_sec,
    }


def _condition_values(data: Mapping[str, Any], aliases: tuple[str, ...]) -> list[Any]:
    values: list[Any] = []
    for alias in aliases:
        if alias not in data:
            continue
        for item in _list(data.get(alias)):
            if isinstance(item, Mapping):
                for key in ("action", "name", "id", "label", "material"):
                    if item.get(key):
                        values.append(item[key])
                        break
            else:
                values.append(item)
    return [value for value in values if value is not None and str(value) != ""]


def _condition_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _action_present(action: Any, facts: Mapping[str, Any]) -> bool:
    action_text = str(action or "").strip()
    if not action_text:
        return False
    actions = {str(item).lower() for item in facts.get("actions", set())}
    return action_text.lower() in actions or _action_family(action_text).lower() in actions


def _material_present(material: Any, facts: Mapping[str, Any]) -> bool:
    required = _normalize_label(material)
    if not required:
        return False
    for value in facts.get("materials", set()):
        candidate = _normalize_label(value)
        if candidate and (required == candidate or required in candidate or candidate in required):
            return True
    return False


def _object_present(obj: Any, facts: Mapping[str, Any]) -> bool:
    required = _normalize_label(obj)
    if not required:
        return False
    for value in facts.get("objects", set()):
        candidate = _normalize_label(value)
        if candidate and (required == candidate or required in candidate or candidate in required):
            return True
    return _material_present(obj, facts)


def _state_present(state: Any, facts: Mapping[str, Any]) -> bool:
    required = _normalize_label(state)
    if not required:
        return False
    return any(required == _normalize_label(value) or required in _normalize_label(value) for value in facts.get("states", set()))


def _text_present(keyword: Any, facts: Mapping[str, Any]) -> bool:
    needle = str(keyword or "").strip().lower()
    if not needle:
        return False
    return any(needle in str(text or "").lower() for text in facts.get("texts", []))


def _material_labels(
    context: Mapping[str, Any],
    video_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
) -> set[str]:
    values: list[Any] = []
    for item in _list(context.get("materials")):
        values.extend(_label_values(item))
    for row in [*video_rows, *asset_rows, *state_rows]:
        values.extend(_label_values(row))
    return {_normalize_label(value) for value in values if _normalize_label(value)}


def _object_labels(
    video_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
) -> set[str]:
    values: list[Any] = []
    for row in [*video_rows, *asset_rows, *state_rows]:
        for key in ("primary_object", "object_label", "object_labels", "objects", "detected_objects", "asset_id", "label"):
            if key in row:
                values.extend(_label_values(row.get(key)))
    return {_normalize_label(value) for value in values if _normalize_label(value)}


def _state_labels(video_rows: list[Mapping[str, Any]], state_rows: list[Mapping[str, Any]]) -> set[str]:
    values: list[Any] = []
    for row in [*video_rows, *state_rows]:
        for key in ("state_type", "state", "state_change_types", "event_type", "visual_confirmation_level", "confirmation_level"):
            if key in row:
                values.extend(_label_values(row.get(key)))
    return {_normalize_label(value) for value in values if _normalize_label(value)}


def _text_labels(
    context: Mapping[str, Any],
    video_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
) -> list[str]:
    texts: list[str] = []
    for key in ("text_evidence", "upload_evidence", "ai_evidence", "transcript_evidence", "database_evidence", "video_evidence"):
        for row in _list(context.get(key)):
            if isinstance(row, Mapping) and row.get("text"):
                texts.append(str(row.get("text") or ""))
    for row in [*video_rows, *asset_rows, *state_rows]:
        for key in ("text", "summary", "description", "search_text"):
            if row.get(key):
                texts.append(str(row[key]))
    return texts


def _label_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        values: list[Any] = []
        for key in (
            "name",
            "id",
            "asset_id",
            "label",
            "material",
            "primary_object",
            "object_label",
            "object_labels",
            "materials",
            "objects",
            "detected_objects",
            "path",
        ):
            if key in value:
                values.extend(_label_values(value.get(key)))
        for asset_ref in _list(value.get("asset_refs")):
            values.extend(_label_values(asset_ref))
        return values
    if isinstance(value, list):
        values = []
        for item in value:
            values.extend(_label_values(item))
        return values
    return [value]


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _max_confidence(rows: list[Mapping[str, Any]]) -> float:
    scores = []
    for row in rows:
        try:
            scores.append(float(row.get("confidence") or 0.0))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else 0.0


def _missing_flags(existing: list[str], failures: list[str]) -> list[str]:
    current = set(existing)
    output = []
    for failure in failures:
        if failure not in current:
            current.add(failure)
            output.append(failure)
    return output


def _missing_completion_reason(status: str, observed: bool, entry_result: Mapping[str, Any], completion_result: Mapping[str, Any]) -> str:
    if status == "branch_not_taken":
        return "branch_condition_not_met"
    if status == "entry_condition_not_met":
        return ";".join(str(item) for item in entry_result.get("failures") or ["entry_condition_not_met"])
    if status == "condition_failed":
        failures = list(entry_result.get("failures") or []) + list(completion_result.get("failures") or [])
        return ";".join(str(item) for item in failures) if failures else "condition_failed"
    if status == "waiting":
        return "wait_condition_pending"
    if status == "in_progress":
        return "completion_pending"
    return "" if observed else "no_direct_video_observation"


def _is_blocking_video_anomaly(value: Any) -> bool:
    flag = str(value or "")
    return bool(flag and flag not in NON_BLOCKING_VIDEO_ANOMALY_FLAGS)


def _build_step(
    sop_step: Mapping[str, Any],
    index: int,
    sop_steps: list[Mapping[str, Any]],
    observations: Mapping[str, list[dict[str, Any]]],
    video_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
    asset_rows: list[Mapping[str, Any]],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    expected = str(sop_step["expected_action"])
    observed_events = _matching_observations(expected, observations)
    observed = bool(observed_events)
    repeat_count = len({row.get("micro_segment_id") or row.get("video_event_id") for row in observed_events})
    repeated = repeat_count > 1
    conflict_flags: list[str] = []
    if repeated and sop_step.get("repeatable") is False and sop_step.get("repeatable_explicit"):
        conflict_flags.append("unexpected_repeat")
    if repeat_count > int(sop_step.get("max_repeats") or 999999):
        conflict_flags.append("repeat_count_exceeds_max")
    if repeat_count < int(sop_step.get("min_repeats") or 1) and observed:
        conflict_flags.append("repeat_count_below_min")
    abnormal = any(
        _is_blocking_video_anomaly(anomaly)
        for row in observed_events
        for anomaly in _list(row.get("anomaly_flags"))
    ) or bool(conflict_flags)
    evidence_refs = _evidence_refs(observed_events, state_rows, asset_rows)
    sop_refs = _sop_evidence_refs(sop_step, index)
    text_context_refs = _context_evidence_refs(expected, context)
    evidence_refs = _dedupe_refs([*evidence_refs, *sop_refs, *text_context_refs])
    confidence, reasons = _step_confidence(observed_events, evidence_refs)
    confidence, context_reasons = _fuse_step_confidence(confidence, sop_step, text_context_refs)
    reasons.extend(context_reasons)
    entry_result = _evaluate_sop_condition(
        sop_step.get("entry_conditions"),
        all_events=video_rows,
        confidence_events=video_rows,
        context=context,
        asset_rows=asset_rows,
        state_rows=state_rows,
    )
    completion_result = _evaluate_sop_condition(
        _completion_condition_for_step(sop_step),
        all_events=video_rows,
        confidence_events=observed_events,
        context=context,
        asset_rows=asset_rows,
        state_rows=state_rows,
    )
    wait_result = _evaluate_sop_condition(
        sop_step.get("wait_conditions"),
        all_events=video_rows,
        confidence_events=video_rows,
        context=context,
        asset_rows=asset_rows,
        state_rows=state_rows,
    )
    repeat_until_result = _evaluate_sop_condition(
        sop_step.get("repeat_until"),
        all_events=video_rows,
        confidence_events=observed_events,
        context=context,
        asset_rows=asset_rows,
        state_rows=state_rows,
    )
    parallel_results = _parallel_observation_results(
        sop_step.get("parallel_observations"),
        video_rows=video_rows,
        observed_events=observed_events,
        context=context,
        asset_rows=asset_rows,
        state_rows=state_rows,
    )
    if not entry_result["passed"]:
        conflict_flags.extend(_missing_flags(conflict_flags, entry_result["failures"]))
    if observed and not completion_result["passed"]:
        conflict_flags.extend(_missing_flags(conflict_flags, completion_result["failures"]))
    missing_parallel = [row for row in parallel_results if row.get("required", True) and not row.get("result", {}).get("passed")]
    if observed and missing_parallel and sop_step.get("required_parallel_observations", True):
        conflict_flags.extend(_missing_flags(conflict_flags, ["parallel_observation_missing"]))
    for event in observed_events:
        for anomaly in _list(event.get("anomaly_flags")):
            if _is_blocking_video_anomaly(anomaly):
                conflict_flags.extend(_missing_flags(conflict_flags, [f"video_anomaly:{anomaly}"]))
    abnormal = abnormal or bool(conflict_flags)
    if not sop_step.get("branch_enabled", True):
        status = "branch_not_taken"
        observed = False
        confidence = 0.6
        reasons = ["branch condition not met"]
    elif not entry_result["passed"]:
        status = "entry_condition_not_met" if not observed else "condition_failed"
        reasons.extend(entry_result["reasons"])
    elif sop_step.get("wait_conditions") and not wait_result["passed"]:
        status = "waiting"
        reasons.extend(wait_result["reasons"] or ["wait condition pending"])
    elif observed and sop_step.get("repeat_until") and not repeat_until_result["passed"]:
        status = "in_progress"
        reasons.extend(repeat_until_result["reasons"] or ["repeat-until condition pending"])
    elif observed and missing_parallel and sop_step.get("required_parallel_observations", True):
        status = "condition_failed"
        reasons.extend([f"missing_parallel={row.get('observation_id')}" for row in missing_parallel])
    elif observed and not completion_result["passed"]:
        status = "condition_failed"
        reasons.extend(completion_result["reasons"])
    else:
        status = "completed" if observed else "not_observed"
    completed = observed and status == "completed"
    if status == "branch_not_taken":
        completed = True
    branch_result = sop_step.get("branch_condition_result") if isinstance(sop_step.get("branch_condition_result"), Mapping) else {}
    step = {
        "step_id": sop_step["step_id"],
        "name": sop_step["name"],
        "expected_action": expected,
        "status": status,
        "observed": observed,
        "inferred": False,
        "completed": completed,
        "skipped": status == "branch_not_taken",
        "repeated": repeated,
        "abnormal": abnormal,
        "confidence": confidence,
        "confidence_reasons": reasons,
        "entry_conditions": sop_step.get("entry_conditions") or [],
        "completion_conditions": sop_step.get("completion_conditions") or [],
        "wait_conditions": sop_step.get("wait_conditions") or {},
        "repeat_until": sop_step.get("repeat_until") or {},
        "parallel_observations": parallel_results,
        "global_start_time": _min_time(observed_events),
        "global_end_time": _max_time(observed_events),
        "evidence_refs": evidence_refs,
        "missing_completion_reason": _missing_completion_reason(status, observed, entry_result, completion_result),
        "next_step_hint": str(sop_steps[index + 1]["step_id"]) if index + 1 < len(sop_steps) else None,
        "requires_human_confirmation": abnormal or status in {"condition_failed", "entry_condition_not_met"} or (not observed and status not in {"branch_not_taken", "waiting"}),
        "repeat_count": repeat_count,
        "conflict_flags": conflict_flags,
        "branch_condition": sop_step.get("branch_condition") or {},
        "branch_enabled": bool(sop_step.get("branch_enabled", True)),
        "auto_confirmed": bool(status == "completed" and observed and confidence >= 0.75 and not abnormal),
        "confirmation_status": "pending"
        if abnormal or status in {"condition_failed", "entry_condition_not_met"} or (not observed and status not in {"branch_not_taken", "waiting"})
        else "auto_confirmed",
        "condition_results": {
            "branch": branch_result or {"passed": bool(sop_step.get("branch_enabled", True)), "failures": [], "reasons": []},
            "entry": entry_result,
            "completion": completion_result,
            "wait": wait_result,
            "repeat_until": repeat_until_result,
            "parallel_observations": parallel_results,
        },
        "history_prior": {},
        "history_deviation": {},
        "history_basis": [],
        "reasoning": _step_reasoning(
            status=status,
            observed=observed,
            confidence=confidence,
            evidence_refs=evidence_refs,
            conflict_flags=conflict_flags,
            condition_results={
                "entry": entry_result,
                "completion": completion_result,
                "wait": wait_result,
                "repeat_until": repeat_until_result,
                "parallel_observations": parallel_results,
            },
            confidence_reasons=reasons,
        ),
        "text_context_refs": text_context_refs,
        "sop_refs": sop_refs,
        "history_refs": [],
        "transition_options": _list(sop_step.get("allowed_transitions")),
        "inference_source": [],
        "inference_reason": "",
        "inference_confidence": None,
        "inferred_from_step_ids": [],
        "evidence_level": _strongest_evidence_level(evidence_refs),
        "evidence_strength": _evidence_strength_summary(evidence_refs),
    }
    return {field: step.get(field) for field in STEP_FIELDS}


def _infer_missing_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed_indices = [idx for idx, step in enumerate(steps) if step.get("observed")]
    if len(observed_indices) < 2:
        return steps
    first, last = min(observed_indices), max(observed_indices)
    for index, step in enumerate(steps):
        if step.get("observed"):
            continue
        if step.get("status") == "branch_not_taken":
            continue
        if first < index < last:
            history_basis = _history_basis_for_missing_step(step)
            inferred_from_step_ids = _neighbor_observed_step_ids(steps, index)
            inference_confidence = _history_adjusted_inference_confidence(0.35, history_basis)
            step["status"] = "inferred_missing"
            step["inferred"] = True
            step["completed"] = True
            step["skipped"] = False
            step["confidence"] = inference_confidence
            step["confidence_reasons"] = ["inferred from surrounding completed SOP steps", "no direct observation"] + _history_confidence_reasons(history_basis)
            step["missing_completion_reason"] = "step not directly observed but surrounding SOP steps were observed"
            if history_basis:
                step["missing_completion_reason"] += "; historical priors support this step"
                step["history_basis"] = history_basis
            step["inference_source"] = _dedupe(["surrounding_observed_steps", "sop_order", "history_prior" if history_basis else ""])
            step["inference_reason"] = step["missing_completion_reason"]
            step["inference_confidence"] = inference_confidence
            step["inferred_from_step_ids"] = inferred_from_step_ids
            _append_inference_evidence_refs(step, inferred_from_step_ids=inferred_from_step_ids, history_basis=history_basis)
            step["requires_human_confirmation"] = True
            step["confirmation_status"] = "pending"
            step["reasoning"] = _step_reasoning(
                status=str(step.get("status")),
                observed=bool(step.get("observed")),
                confidence=float(step.get("confidence") or 0.0),
                evidence_refs=list(step.get("evidence_refs") or []),
                conflict_flags=list(step.get("conflict_flags") or []),
                condition_results=step.get("condition_results") if isinstance(step.get("condition_results"), Mapping) else {},
                confidence_reasons=list(step.get("confidence_reasons") or []),
            )
        elif index < last:
            history_basis = _history_basis_for_missing_step(step)
            step["status"] = "skipped_or_unobserved"
            step["skipped"] = True
            step["confidence"] = 0.2
            step["confidence_reasons"] = ["later SOP step observed before this step"] + _history_confidence_reasons(history_basis)
            step["missing_completion_reason"] = "later step observed; this step may be skipped or missing"
            if history_basis:
                step["missing_completion_reason"] += "; historical priors recorded for review"
                step["history_basis"] = history_basis
            step["inference_source"] = _dedupe(["later_observed_step", "sop_order", "history_prior" if history_basis else ""])
            step["inference_reason"] = step["missing_completion_reason"]
            step["inference_confidence"] = 0.2
            step["inferred_from_step_ids"] = _neighbor_observed_step_ids(steps, index)
            _append_inference_evidence_refs(step, inferred_from_step_ids=step["inferred_from_step_ids"], history_basis=history_basis)
            step["requires_human_confirmation"] = True
            step["confirmation_status"] = "pending"
            step["reasoning"] = _step_reasoning(
                status=str(step.get("status")),
                observed=bool(step.get("observed")),
                confidence=float(step.get("confidence") or 0.0),
                evidence_refs=list(step.get("evidence_refs") or []),
                conflict_flags=list(step.get("conflict_flags") or []),
                condition_results=step.get("condition_results") if isinstance(step.get("condition_results"), Mapping) else {},
                confidence_reasons=list(step.get("confidence_reasons") or []),
            )
        step["evidence_level"] = _strongest_evidence_level(step.get("evidence_refs") or [])
        step["evidence_strength"] = _evidence_strength_summary(step.get("evidence_refs") or [])
    return steps


def _apply_elapsed_conditions(steps: list[dict[str, Any]], sop_steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    previous_end_time: str | None = None
    for step, sop_step in zip(steps, sop_steps):
        if not step.get("observed"):
            continue
        start_time = str(step.get("global_start_time") or "")
        elapsed_sec = _elapsed_sec(previous_end_time, start_time) if previous_end_time and start_time else None
        max_elapsed_sec = _max_elapsed_condition(_completion_condition_for_step(sop_step))
        if max_elapsed_sec is not None:
            _record_elapsed_result(step, elapsed_sec, max_elapsed_sec)
            if elapsed_sec is not None and elapsed_sec > max_elapsed_sec:
                _fail_step_condition(
                    step,
                    flag="max_elapsed_sec_exceeded",
                    reason=f"elapsed_sec={elapsed_sec:.3f}>max_elapsed_sec={max_elapsed_sec:.3f}",
                )
        previous_end_time = str(step.get("global_end_time") or step.get("global_start_time") or previous_end_time or "")
    return steps


def _record_elapsed_result(step: dict[str, Any], elapsed_sec: float | None, max_elapsed_sec: float) -> None:
    condition_results = dict(step.get("condition_results") or {})
    completion = dict(condition_results.get("completion") or {"passed": True, "failures": [], "reasons": []})
    reasons = list(completion.get("reasons") or [])
    if elapsed_sec is None:
        reasons.append("elapsed_sec_unavailable")
    elif elapsed_sec <= max_elapsed_sec:
        reasons.append(f"elapsed_sec={elapsed_sec:.3f}<=max_elapsed_sec={max_elapsed_sec:.3f}")
    completion["reasons"] = _dedupe(reasons)
    condition_results["completion"] = completion
    step["condition_results"] = condition_results
    step["reasoning"] = _step_reasoning(
        status=str(step.get("status") or ""),
        observed=bool(step.get("observed")),
        confidence=float(step.get("confidence") or 0.0),
        evidence_refs=list(step.get("evidence_refs") or []),
        conflict_flags=list(step.get("conflict_flags") or []),
        condition_results=condition_results,
        confidence_reasons=list(step.get("confidence_reasons") or []),
    )


def _fail_step_condition(step: dict[str, Any], *, flag: str, reason: str) -> None:
    flags = list(step.get("conflict_flags") or [])
    if flag not in flags:
        flags.append(flag)
    step["conflict_flags"] = flags
    step["abnormal"] = True
    step["completed"] = False
    step["requires_human_confirmation"] = True
    step["auto_confirmed"] = False
    step["confirmation_status"] = "pending"
    if step.get("status") == "completed":
        step["status"] = "condition_failed"
    existing_reason = str(step.get("missing_completion_reason") or "")
    step["missing_completion_reason"] = ";".join([item for item in (existing_reason, flag) if item])
    condition_results = dict(step.get("condition_results") or {})
    completion = dict(condition_results.get("completion") or {"passed": True, "failures": [], "reasons": []})
    failures = list(completion.get("failures") or [])
    if flag not in failures:
        failures.append(flag)
    reasons = list(completion.get("reasons") or [])
    reasons.append(reason)
    completion["passed"] = False
    completion["failures"] = _dedupe(failures)
    completion["reasons"] = _dedupe(reasons)
    condition_results["completion"] = completion
    step["condition_results"] = condition_results


def _elapsed_sec(previous_end_time: str | None, current_start_time: str | None) -> float | None:
    if not previous_end_time or not current_start_time:
        return None
    try:
        return (parse_time(current_start_time) - parse_time(previous_end_time)).total_seconds()
    except (TypeError, ValueError):
        return None


def _max_elapsed_condition(condition: Any) -> float | None:
    if isinstance(condition, Mapping):
        value = _condition_float(condition.get("max_elapsed_sec"))
        if value is not None:
            return value
        for nested in condition.values():
            value = _max_elapsed_condition(nested)
            if value is not None:
                return value
    if isinstance(condition, list):
        for item in condition:
            value = _max_elapsed_condition(item)
            if value is not None:
                return value
    return None


def _apply_history_priors(steps: list[dict[str, Any]], history_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    action_counts = _history_action_counts(history_model)
    action_probabilities = _history_action_probabilities(history_model, action_counts)
    transitions = history_model.get("transition_probabilities") if isinstance(history_model.get("transition_probabilities"), Mapping) else {}
    total_count = sum(action_counts.values())
    if total_count <= 0 and not transitions:
        for step in steps:
            step["history_prior"] = {
                "available": False,
                "history_event_count": 0,
                "history_session_count": history_model.get("session_count", 0),
            }
            step["history_deviation"] = {"score": 0.0, "flags": []}
            step["history_refs"] = []
        return steps
    for index, step in enumerate(steps):
        action = str(step.get("expected_action") or "")
        previous_action = str(steps[index - 1].get("expected_action") or "") if index > 0 else ""
        next_action = str(steps[index + 1].get("expected_action") or "") if index + 1 < len(steps) else ""
        action_probability = _history_action_probability(action, action_probabilities)
        previous_probability = _history_transition_probability(transitions, previous_action, action) if previous_action else None
        next_probability = _history_transition_probability(transitions, action, next_action) if next_action else None
        support_count = _history_action_count(action, action_counts)
        score_parts = [value for value in (action_probability, previous_probability, next_probability) if value is not None]
        prior_score = round(sum(score_parts) / len(score_parts), 4) if score_parts else 0.0
        prior = {
            "score": prior_score,
            "action": action,
            "action_probability": round(action_probability, 6),
            "support_count": support_count,
            "history_event_count": total_count,
            "history_session_count": history_model.get("session_count", 0),
            "previous_transition": f"{previous_action}->{action}" if previous_action else None,
            "previous_transition_probability": round(previous_probability, 6) if previous_probability is not None else None,
            "next_transition": f"{action}->{next_action}" if next_action else None,
            "next_transition_probability": round(next_probability, 6) if next_probability is not None else None,
        }
        flags = _history_deviation_flags(action, action_probability, previous_probability, next_probability)
        step["history_prior"] = prior
        step["history_deviation"] = {"score": round(max(0.0, min(1.0, 1.0 - prior_score)), 4), "flags": flags}
        history_refs = _history_prior_refs(step, prior)
        step["history_refs"] = history_refs
        step["evidence_refs"] = _dedupe_refs([*(step.get("evidence_refs") or []), *history_refs])
        step["evidence_level"] = _strongest_evidence_level(step.get("evidence_refs") or [])
        step["evidence_strength"] = _evidence_strength_summary(step.get("evidence_refs") or [])
        if step.get("reasoning"):
            step["reasoning"] = f"{step['reasoning']} | history_prior_score={prior_score:.3f}"
    return steps


def _mark_history_conflicts(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for step in steps:
        deviation = step.get("history_deviation") if isinstance(step.get("history_deviation"), Mapping) else {}
        flags = [str(flag) for flag in deviation.get("flags") or [] if flag]
        if not flags:
            continue
        existing = [str(flag) for flag in step.get("conflict_flags") or [] if flag]
        step["conflict_flags"] = _dedupe([*existing, *(f"history_{flag}" for flag in flags)])
        reasons = [str(reason) for reason in step.get("confidence_reasons") or [] if reason]
        step["confidence_reasons"] = _dedupe([*reasons, *(f"history_deviation:{flag}" for flag in flags)])
        step["abnormal"] = True
        step["requires_human_confirmation"] = True
        step["confirmation_status"] = "pending"
        step["auto_confirmed"] = False
        if step.get("reasoning"):
            step["reasoning"] = f"{step['reasoning']} | history_conflicts={','.join(flags)}"
    return steps


def _history_prior_refs(step: Mapping[str, Any], prior: Mapping[str, Any]) -> list[dict[str, Any]]:
    step_id = str(step.get("step_id") or "")
    refs = []
    action_probability = float(prior.get("action_probability") or 0.0)
    if int(prior.get("support_count") or 0) > 0 or action_probability > 0.0:
        refs.append(
            {
                "type": "database_record",
                "evidence_id": f"history_prior:{step_id}:action",
                "history_id": f"history_prior:{step_id}:action",
                "source": "history_model",
                "action": prior.get("action"),
                "support_count": prior.get("support_count"),
                "confidence": action_probability,
                "evidence_level": "database_record",
            }
        )
    for key, transition_key in (("previous_transition_probability", "previous_transition"), ("next_transition_probability", "next_transition")):
        probability = prior.get(key)
        transition = prior.get(transition_key)
        if probability is None or not transition:
            continue
        refs.append(
            {
                "type": "database_record",
                "evidence_id": f"history_prior:{step_id}:{transition_key}",
                "history_id": f"history_prior:{step_id}:{transition_key}",
                "source": "history_model",
                "transition": transition,
                "confidence": probability,
                "evidence_level": "database_record",
            }
        )
    return refs


def _history_action_counts(history_model: Mapping[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    raw = history_model.get("action_counts") if isinstance(history_model.get("action_counts"), Mapping) else {}
    for action, count in raw.items():
        try:
            counts[str(action)] = int(float(count or 0))
        except (TypeError, ValueError):
            counts[str(action)] = 0
    return counts


def _history_action_probabilities(history_model: Mapping[str, Any], counts: Mapping[str, int]) -> dict[str, float]:
    raw = history_model.get("action_probabilities")
    if isinstance(raw, Mapping):
        probabilities: dict[str, float] = {}
        for action, probability in raw.items():
            try:
                probabilities[str(action)] = float(probability or 0.0)
            except (TypeError, ValueError):
                probabilities[str(action)] = 0.0
        return probabilities
    total = sum(counts.values())
    if not total:
        return {}
    return {action: count / total for action, count in counts.items()}


def _history_action_probability(action: str, probabilities: Mapping[str, float]) -> float:
    if action in probabilities:
        return float(probabilities[action] or 0.0)
    family = _action_family(action)
    return float(probabilities.get(family, 0.0) or 0.0)


def _history_action_count(action: str, counts: Mapping[str, int]) -> int:
    if action in counts:
        return int(counts[action])
    family = _action_family(action)
    return int(counts.get(family, 0) or 0)


def _history_transition_probability(transitions: Mapping[str, Any], source: str, destination: str) -> float:
    for source_key in _dedupe([source, _action_family(source)]):
        options = transitions.get(source_key)
        if not isinstance(options, Mapping):
            continue
        for destination_key in _dedupe([destination, _action_family(destination)]):
            try:
                return float(options.get(destination_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _history_deviation_flags(
    action: str,
    action_probability: float,
    previous_probability: float | None,
    next_probability: float | None,
) -> list[str]:
    flags: list[str] = []
    if action and action_probability <= 0.0:
        flags.append("unseen_action")
    elif action_probability < 0.05:
        flags.append("rare_action")
    if previous_probability is not None and previous_probability < 0.15:
        flags.append("rare_previous_transition")
    if next_probability is not None and next_probability < 0.15:
        flags.append("rare_next_transition")
    return flags


def _history_basis_for_missing_step(step: Mapping[str, Any]) -> list[dict[str, Any]]:
    prior = step.get("history_prior") if isinstance(step.get("history_prior"), Mapping) else {}
    basis: list[dict[str, Any]] = []
    action_probability = prior.get("action_probability")
    support_count = prior.get("support_count")
    if action_probability is not None and (float(action_probability or 0.0) > 0.0 or int(support_count or 0) > 0):
        basis.append(
            {
                "type": "action_prior",
                "action": prior.get("action"),
                "probability": action_probability,
                "support_count": support_count,
                "history_session_count": prior.get("history_session_count"),
            }
        )
    for key, direction in (("previous_transition_probability", "previous"), ("next_transition_probability", "next")):
        probability = prior.get(key)
        transition = prior.get("previous_transition" if direction == "previous" else "next_transition")
        if probability is not None and float(probability or 0.0) > 0.0:
            basis.append({"type": "transition_prior", "direction": direction, "transition": transition, "probability": probability})
    return basis


def _history_confidence_reasons(history_basis: list[Mapping[str, Any]]) -> list[str]:
    reasons = []
    for item in history_basis:
        if item.get("type") == "action_prior":
            reasons.append(f"history_action_prior={float(item.get('probability') or 0.0):.3f}")
        elif item.get("type") == "transition_prior":
            reasons.append(f"history_transition_prior={item.get('transition')}:{float(item.get('probability') or 0.0):.3f}")
    return reasons


def _history_adjusted_inference_confidence(base_confidence: float, history_basis: list[Mapping[str, Any]]) -> float:
    probabilities = [float(item.get("probability") or 0.0) for item in history_basis]
    if not probabilities:
        return base_confidence
    max_probability = max(probabilities)
    if max_probability >= 0.75:
        return 0.45
    if max_probability >= 0.4:
        return 0.4
    return base_confidence


def _mark_order_conflicts(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_time = ""
    for step in steps:
        start = str(step.get("global_start_time") or "")
        if step.get("observed") and latest_time and start and start < latest_time:
            flags = list(step.get("conflict_flags") or [])
            if "observed_out_of_sop_order" not in flags:
                flags.append("observed_out_of_sop_order")
            step["conflict_flags"] = flags
            step["abnormal"] = True
            step["requires_human_confirmation"] = True
            step["confirmation_status"] = "pending"
            if step.get("reasoning"):
                step["reasoning"] = f"{step['reasoning']} | order_conflict=observed_out_of_sop_order"
        if step.get("observed") and start and start > latest_time:
            latest_time = start
    return steps


def _current_and_next(steps: list[Mapping[str, Any]]) -> tuple[Mapping[str, Any] | None, Mapping[str, Any] | None]:
    if not steps:
        return None, None
    active_statuses = {"in_progress", "waiting", "condition_failed", "entry_condition_not_met", "human_rejected"}
    active = next((step for step in steps if str(step.get("status") or "") in active_statuses and not step.get("skipped")), None)
    if active is not None:
        return active, active
    last_observed = None
    for step in steps:
        if step.get("observed") or step.get("completed"):
            last_observed = step
    next_actionable = next((step for step in steps if not step.get("completed") and not step.get("skipped")), None)
    if last_observed is None:
        return next_actionable or steps[0], next_actionable or steps[0]
    return last_observed, next_actionable


def _conflict_report(
    steps: list[Mapping[str, Any]],
    context: Mapping[str, Any],
    related_history_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    by_flag: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        step_id = str(step.get("step_id") or "")
        for flag in step.get("conflict_flags") or []:
            by_flag[str(flag)].append(step_id)
    sop_actions = {str(step.get("expected_action") or "") for step in steps if step.get("expected_action")}
    context_actions = {
        str(item.get("action_type") or "")
        for item in _list(context.get("procedure_candidates"))
        if isinstance(item, Mapping) and item.get("action_type")
    }
    context_only = sorted(action for action in context_actions - sop_actions if action)
    sop_only = sorted(action for action in sop_actions - context_actions if action and context_actions)
    if context_only:
        by_flag["context_action_not_in_sop"] = context_only
    if sop_only:
        by_flag["sop_action_not_in_context"] = sop_only
    return {
        "conflict_count": sum(len(values) for values in by_flag.values()),
        "flags": {flag: values for flag, values in sorted(by_flag.items())},
        "requires_human_confirmation_step_ids": [str(step.get("step_id")) for step in steps if step.get("requires_human_confirmation")],
        "resolution_policy": {
            "do_not_silently_drop_conflicts": True,
            "low_confidence_or_conflicted_steps_require_confirmation": True,
            "human_decisions_are_audited": True,
        },
        "related_history_record_ids": [record.get("record_id") for record in related_history_records if record.get("record_id")],
    }


def _process_timeline_rows(steps: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for order, step in enumerate(steps, start=1):
        rows.append(
            {
                "timeline_event_id": step.get("step_id"),
                "event_type": "experiment_step",
                "order": order,
                "global_time": step.get("global_start_time"),
                "global_end_time": step.get("global_end_time"),
                "text": step.get("name"),
                "status": step.get("status"),
                "confidence": step.get("confidence"),
                "evidence_refs": step.get("evidence_refs"),
                "conflict_flags": step.get("conflict_flags"),
                "requires_human_confirmation": step.get("requires_human_confirmation"),
                "reasoning": step.get("reasoning"),
            }
        )
    return rows


def _matching_observations(expected: str, observations: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    keys = _dedupe([expected, _action_family(expected)])
    rows: list[dict[str, Any]] = []
    for key in keys:
        rows.extend(observations.get(key, []))
    return _dedupe_events(rows)


def _action_family(value: str) -> str:
    text = str(value or "").lower()
    if any(token in text for token in ("pipette", "pipetting", "sample_adding", "liquid_transfer", "加样", "移液")):
        return "pipetting"
    if any(token in text for token in ("weigh", "balance", "称量", "天平")):
        return "weighing"
    if any(token in text for token in ("record", "readout", "记录", "读数")):
        return "recording"
    if any(token in text for token in ("sample", "bottle", "tube", "vial", "container")):
        return "sample_handling"
    return text


def _evidence_refs(video_events: list[Mapping[str, Any]], state_rows: list[Mapping[str, Any]], asset_rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    micro_ids = {str(row.get("micro_segment_id")) for row in video_events if row.get("micro_segment_id")}
    asset_by_id = {str(row.get("asset_id")): row for row in asset_rows if row.get("asset_id")}
    for event in video_events:
        refs.append(
            {
                "type": "video_event",
                "evidence_id": event.get("video_event_id"),
                "video_event_id": event.get("video_event_id"),
                "segment_id": event.get("segment_id"),
                "micro_segment_id": event.get("micro_segment_id"),
                "global_start_time": event.get("global_start_time"),
                "global_end_time": event.get("global_end_time"),
                "confidence": event.get("confidence"),
                "evidence_level": _evidence_level_from_video_event(event),
            }
        )
        for asset_ref in event.get("asset_refs") or []:
            if isinstance(asset_ref, Mapping) and asset_ref.get("asset_id"):
                asset = asset_by_id.get(str(asset_ref.get("asset_id")), {})
                refs.append(_asset_evidence_ref(asset_ref, asset))
    for state in state_rows:
        if str(state.get("micro_segment_id")) in micro_ids:
            refs.append(
                {
                    "type": "state_change",
                    "evidence_id": state.get("state_change_id"),
                    "state_change_id": state.get("state_change_id"),
                    "state_type": state.get("state_type"),
                    "segment_id": state.get("segment_id"),
                    "micro_segment_id": state.get("micro_segment_id"),
                    "global_start_time": state.get("global_start_time") or state.get("global_time"),
                    "global_end_time": state.get("global_end_time") or state.get("global_time"),
                    "confidence": state.get("confidence"),
                    "evidence_level": "state_change",
                }
            )
            for asset_ref in state.get("asset_refs") or []:
                if isinstance(asset_ref, Mapping) and asset_ref.get("asset_id"):
                    asset = asset_by_id.get(str(asset_ref.get("asset_id")), {})
                    refs.append(_asset_evidence_ref(asset_ref, asset))
    return _dedupe_refs(refs)


def _sop_evidence_ref(sop_step: Mapping[str, Any], index: int) -> dict[str, Any]:
    step_id = str(sop_step.get("step_id") or f"step_{index + 1:03d}")
    return {
        "type": "sop_step",
        "evidence_id": f"sop:{step_id}",
        "sop_step_id": step_id,
        "expected_action": sop_step.get("expected_action"),
        "name": sop_step.get("name"),
        "confidence": 1.0,
        "evidence_level": "text_support",
        "source": "sop",
    }


def _sop_evidence_refs(sop_step: Mapping[str, Any], index: int) -> list[dict[str, Any]]:
    refs = [_sop_evidence_ref(sop_step, index)]
    for ref in _list(sop_step.get("source_refs")):
        if isinstance(ref, Mapping):
            item = dict(ref)
            item.setdefault("type", "sop_step")
            item.setdefault("evidence_id", f"sop:{sop_step.get('step_id')}")
            item.setdefault("evidence_level", "text_support")
            item.setdefault("confidence", 1.0)
            refs.append(item)
    return _dedupe_refs(refs)


def _context_evidence_refs(expected: str, context: Mapping[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    expected_family = _action_family(expected)
    for item in _list(context.get("procedure_candidates")):
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action_type") or "")
        if action != expected and _action_family(action) != expected_family:
            continue
        refs.append(
            {
                "type": "text_context",
                "evidence_id": f"context:procedure:{action}",
                "context_key": "procedure_candidates",
                "action_type": action,
                "confidence": min(1.0, float(item.get("score") or 0.0) / 5.0) if item.get("score") is not None else None,
                "source_types": list(item.get("source_types") or []),
                "evidence_level": "text_support",
            }
        )
    for key in ("text_evidence", "upload_evidence", "ai_evidence", "transcript_evidence", "database_evidence", "video_evidence"):
        for row in _list(context.get(key)):
            if not isinstance(row, Mapping):
                continue
            text = str(row.get("text") or "")
            action = str(row.get("action_type") or "")
            haystack = f"{text} {action}".lower()
            if expected.lower() not in haystack and expected_family.lower() not in haystack:
                continue
            refs.append(
                {
                    "type": "text_context",
                    "evidence_id": f"context:{key}:{row.get('id') or len(refs) + 1}",
                    "context_key": key,
                    "source": row.get("source"),
                    "text": text[:240],
                    "global_time": row.get("global_time"),
                    "action_type": action or None,
                    "confidence": row.get("confidence"),
                    "evidence_level": "text_support",
                }
            )
            if len(refs) >= limit:
                return _dedupe_refs(refs)
    for record in _list(context.get("related_records")):
        if not isinstance(record, Mapping):
            continue
        matched_actions = {str(item) for item in record.get("matched_actions") or []}
        sequence = {str(item) for item in record.get("transition_sequence") or []}
        if expected not in matched_actions and expected not in sequence and expected_family not in matched_actions and expected_family not in sequence:
            continue
        refs.append(
            {
                "type": "database_record",
                "evidence_id": f"related_record:{record.get('record_id')}",
                "record_id": record.get("record_id"),
                "session_id": record.get("session_id"),
                "confidence": min(1.0, float(record.get("score") or 0.0) / 5.0),
                "evidence_level": "database_record",
            }
        )
    return _dedupe_refs(refs[:limit])


def _fuse_step_confidence(confidence: float, sop_step: Mapping[str, Any], text_context_refs: list[Mapping[str, Any]]) -> tuple[float, list[str]]:
    score = float(confidence or 0.0)
    reasons: list[str] = []
    context_support = sop_step.get("context_support") if isinstance(sop_step.get("context_support"), Mapping) else {}
    if text_context_refs:
        score += 0.05
        reasons.append(f"text_context_refs={len(text_context_refs)}")
    if float(context_support.get("procedure_score") or 0.0) > 0:
        score += 0.03
        reasons.append(f"context_procedure_score={float(context_support.get('procedure_score') or 0.0):.3f}")
    if sop_step.get("source_refs"):
        score += 0.02
        reasons.append("sop_template_support")
    return round(max(0.0, min(1.0, score)), 4), reasons


def _parallel_observation_results(
    observations: Any,
    *,
    video_rows: list[Mapping[str, Any]],
    observed_events: list[Mapping[str, Any]],
    context: Mapping[str, Any],
    asset_rows: list[Mapping[str, Any]],
    state_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, item in enumerate(_list(observations), start=1):
        if isinstance(item, Mapping):
            condition = item.get("condition") or item.get("conditions") or item
            observation_id = str(item.get("observation_id") or item.get("id") or f"parallel_{index:03d}")
            required = bool(item.get("required", True))
            name = item.get("name")
        else:
            condition = item
            observation_id = f"parallel_{index:03d}"
            required = True
            name = None
        result = _evaluate_sop_condition(
            condition,
            all_events=video_rows,
            confidence_events=observed_events or video_rows,
            context=context,
            asset_rows=asset_rows,
            state_rows=state_rows,
        )
        results.append(
            {
                "observation_id": observation_id,
                "name": name,
                "required": required,
                "condition": condition,
                "result": result,
            }
        )
    return results


def _step_reasoning(
    *,
    status: str,
    observed: bool,
    confidence: float,
    evidence_refs: list[Mapping[str, Any]],
    conflict_flags: list[str],
    condition_results: Mapping[str, Any],
    confidence_reasons: list[str],
) -> str:
    evidence_counts = Counter(str(ref.get("type") or "unknown") for ref in evidence_refs if isinstance(ref, Mapping))
    parts = [
        f"status={status}",
        f"observed={bool(observed)}",
        f"confidence={float(confidence or 0.0):.3f}",
        f"evidence_counts={dict(sorted(evidence_counts.items()))}",
    ]
    if conflict_flags:
        parts.append(f"conflicts={','.join(conflict_flags)}")
    for key in ("entry", "completion", "wait", "repeat_until"):
        result = condition_results.get(key) if isinstance(condition_results.get(key), Mapping) else {}
        if result and result.get("failures"):
            parts.append(f"{key}_failures={','.join(str(item) for item in result.get('failures') or [])}")
    if confidence_reasons:
        parts.append(f"confidence_reasons={';'.join(confidence_reasons[:5])}")
    return " | ".join(parts)


def _asset_evidence_ref(asset_ref: Mapping[str, Any], asset: Mapping[str, Any]) -> dict[str, Any]:
    asset_id = asset_ref.get("asset_id") or asset.get("asset_id")
    return {
        "type": "asset",
        "evidence_id": asset_id,
        "asset_id": asset_id,
        "path": asset_ref.get("path") or asset.get("path"),
        "asset_type": asset_ref.get("asset_type") or asset.get("asset_type"),
        "segment_id": asset_ref.get("segment_id") or asset.get("segment_id"),
        "micro_segment_id": asset_ref.get("micro_segment_id") or asset.get("micro_segment_id"),
        "confidence": asset_ref.get("confidence") or asset.get("confidence"),
        "evidence_level": str(asset.get("evidence_level") or asset_ref.get("evidence_level") or "visual_asset"),
    }


def _evidence_level_from_video_event(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type") or "")
    confidence = float(event.get("confidence") or 0.0)
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    if payload.get("source") in {"model_observation", "advanced_vision_evidence"}:
        return "model_confirmed" if confidence >= 0.65 else "model_candidate"
    if event_type in {"hand_object_contact", "experiment_action_classification"} and confidence >= 0.75:
        return "direct_visual"
    if "state" in event_type or "liquid" in event_type or "equipment" in event_type or "container" in event_type:
        return "model_confirmed" if confidence >= 0.65 else "model_candidate"
    return "visual_candidate"


def _neighbor_observed_step_ids(steps: list[Mapping[str, Any]], index: int) -> list[str]:
    previous_id = ""
    next_id = ""
    for row in reversed(steps[:index]):
        if row.get("observed"):
            previous_id = str(row.get("step_id") or "")
            break
    for row in steps[index + 1 :]:
        if row.get("observed"):
            next_id = str(row.get("step_id") or "")
            break
    return _dedupe([previous_id, next_id])


def _append_inference_evidence_refs(
    step: dict[str, Any],
    *,
    inferred_from_step_ids: list[str],
    history_basis: list[Mapping[str, Any]],
) -> None:
    step_id = str(step.get("step_id") or "")
    refs = list(step.get("evidence_refs") or [])
    refs.append(
        {
            "type": "inference",
            "evidence_id": f"inference:{step_id}",
            "inference_id": f"inference:{step_id}",
            "source_step_ids": inferred_from_step_ids,
            "confidence": step.get("confidence"),
            "evidence_level": "inferred_support",
            "reason": step.get("missing_completion_reason"),
        }
    )
    for index, basis in enumerate(history_basis, start=1):
        refs.append(
            {
                "type": "database_record",
                "evidence_id": f"history:{step_id}:{index}",
                "history_id": f"history:{step_id}:{index}",
                "source": "history_model",
                "history_basis": dict(basis),
                "confidence": basis.get("probability"),
                "evidence_level": "database_record",
            }
        )
    step["evidence_refs"] = _dedupe_refs(refs)


def _strongest_evidence_level(refs: Iterable[Mapping[str, Any]]) -> str:
    summary = _evidence_strength_summary(refs)
    return str(summary.get("strongest_level") or "none")


def _evidence_strength_summary(refs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    levels = [str(ref.get("evidence_level") or _fallback_evidence_level(ref)) for ref in refs if isinstance(ref, Mapping)]
    scores = [_evidence_level_score(level) for level in levels]
    return {
        "strongest_level": levels[scores.index(max(scores))] if scores else "none",
        "strongest_score": max(scores) if scores else 0.0,
        "level_counts": dict(sorted(Counter(levels).items())),
    }


def _fallback_evidence_level(ref: Mapping[str, Any]) -> str:
    ref_type = str(ref.get("type") or "")
    if ref_type == "video_event":
        return "visual_candidate"
    if ref_type == "state_change":
        return "state_change"
    if ref_type == "asset":
        return "visual_asset"
    if ref_type == "sop_step":
        return "text_support"
    if ref_type == "database_record":
        return "database_record"
    if ref_type == "inference":
        return "inferred_support"
    return "weak"


def _evidence_level_score(level: str) -> float:
    return {
        "direct_visual": 1.0,
        "visual_confirmed": 0.95,
        "trajectory_confirmed": 0.9,
        "model_confirmed": 0.85,
        "state_change": 0.75,
        "visual_asset": 0.7,
        "text_support": 0.55,
        "database_record": 0.5,
        "model_candidate": 0.45,
        "visual_candidate": 0.4,
        "inferred_support": 0.35,
        "weak": 0.2,
        "none": 0.0,
    }.get(str(level or "weak"), 0.3)


def _step_confidence(observed_events: list[Mapping[str, Any]], evidence_refs: list[Mapping[str, Any]]) -> tuple[float, list[str]]:
    if not observed_events:
        return 0.0, ["no direct observation"]
    scores = [float(row.get("confidence") or 0.0) for row in observed_events]
    score = max(scores) if scores else 0.0
    reasons = [f"video_event_count={len(observed_events)}", f"max_video_confidence={score:.3f}"]
    if any(ref.get("type") == "state_change" for ref in evidence_refs):
        score += 0.05
        reasons.append("state_change_evidence")
    if any(ref.get("type") == "asset" for ref in evidence_refs):
        score += 0.05
        reasons.append("asset_evidence")
    if any(row.get("anomaly_flags") for row in observed_events):
        score -= 0.1
        reasons.append("anomaly_flags_present")
    return round(max(0.0, min(1.0, score)), 4), reasons


def _evidence_index(steps: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for step in steps:
        step_id = str(step.get("step_id"))
        for ref in step.get("evidence_refs") or []:
            if isinstance(ref, Mapping):
                key = str(
                    ref.get("evidence_id")
                    or ref.get("video_event_id")
                    or ref.get("state_change_id")
                    or ref.get("asset_id")
                    or ref.get("sop_step_id")
                    or ref.get("history_id")
                    or ref.get("inference_id")
                    or ""
                )
                if key:
                    index[key].append(step_id)
    return dict(sorted(index.items()))


def _sop_evidence_ref(sop_step: Mapping[str, Any], index: int) -> dict[str, Any]:
    step_id = str(sop_step.get("step_id") or f"step_{index + 1:03d}")
    return {
        "type": "sop_step",
        "evidence_id": f"sop:{step_id}",
        "sop_step_id": step_id,
        "expected_action": sop_step.get("expected_action"),
        "name": sop_step.get("name"),
        "confidence": 1.0,
        "evidence_level": "text_support",
        "source": "sop",
    }


def _asset_evidence_ref(asset_ref: Mapping[str, Any], asset: Mapping[str, Any]) -> dict[str, Any]:
    evidence_level = str(asset.get("evidence_level") or asset_ref.get("evidence_level") or "visual_asset")
    asset_id = asset_ref.get("asset_id") or asset.get("asset_id")
    return {
        "type": "asset",
        "evidence_id": asset_id,
        "asset_id": asset_id,
        "path": asset_ref.get("path") or asset.get("path"),
        "asset_type": asset_ref.get("asset_type") or asset.get("asset_type"),
        "segment_id": asset_ref.get("segment_id") or asset.get("segment_id"),
        "micro_segment_id": asset_ref.get("micro_segment_id") or asset.get("micro_segment_id"),
        "confidence": asset_ref.get("confidence") or asset.get("confidence"),
        "evidence_level": evidence_level,
    }


def _evidence_level_from_video_event(event: Mapping[str, Any]) -> str:
    event_type = str(event.get("event_type") or "")
    confidence = float(event.get("confidence") or 0.0)
    payload = event.get("payload") if isinstance(event.get("payload"), Mapping) else {}
    if payload.get("source") in {"model_observation", "advanced_vision_evidence"}:
        return "model_confirmed" if confidence >= 0.65 else "model_candidate"
    if event_type in {"hand_object_contact", "experiment_action_classification"} and confidence >= 0.75:
        return "direct_visual"
    if "state" in event_type or "liquid" in event_type or "equipment" in event_type or "container" in event_type:
        return "model_confirmed" if confidence >= 0.65 else "model_candidate"
    return "visual_candidate"


def _neighbor_observed_step_ids(steps: list[Mapping[str, Any]], index: int) -> list[str]:
    previous_id = ""
    next_id = ""
    for row in reversed(steps[:index]):
        if row.get("observed"):
            previous_id = str(row.get("step_id") or "")
            break
    for row in steps[index + 1 :]:
        if row.get("observed"):
            next_id = str(row.get("step_id") or "")
            break
    return _dedupe([previous_id, next_id])


def _append_inference_evidence_refs(
    step: dict[str, Any],
    *,
    inferred_from_step_ids: list[str],
    history_basis: list[Mapping[str, Any]],
) -> None:
    step_id = str(step.get("step_id") or "")
    refs = list(step.get("evidence_refs") or [])
    refs.append(
        {
            "type": "inference",
            "evidence_id": f"inference:{step_id}",
            "inference_id": f"inference:{step_id}",
            "source_step_ids": inferred_from_step_ids,
            "confidence": step.get("confidence"),
            "evidence_level": "inferred_support",
            "reason": step.get("missing_completion_reason"),
        }
    )
    for index, basis in enumerate(history_basis, start=1):
        refs.append(
            {
                "type": "database_record",
                "evidence_id": f"history:{step_id}:{index}",
                "history_id": f"history:{step_id}:{index}",
                "source": "history_model",
                "history_basis": dict(basis),
                "confidence": basis.get("probability"),
                "evidence_level": "database_record",
            }
        )
    step["evidence_refs"] = _dedupe_refs(refs)


def _strongest_evidence_level(refs: Iterable[Mapping[str, Any]]) -> str:
    summary = _evidence_strength_summary(refs)
    return str(summary.get("strongest_level") or "none")


def _evidence_strength_summary(refs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    levels = [str(ref.get("evidence_level") or _fallback_evidence_level(ref)) for ref in refs if isinstance(ref, Mapping)]
    scores = [_evidence_level_score(level) for level in levels]
    return {
        "strongest_level": levels[scores.index(max(scores))] if scores else "none",
        "strongest_score": max(scores) if scores else 0.0,
        "level_counts": dict(sorted(Counter(levels).items())),
    }


def _fallback_evidence_level(ref: Mapping[str, Any]) -> str:
    ref_type = str(ref.get("type") or "")
    if ref_type == "video_event":
        return "visual_candidate"
    if ref_type == "state_change":
        return "state_change"
    if ref_type == "asset":
        return "visual_asset"
    if ref_type == "sop_step":
        return "text_support"
    if ref_type == "database_record":
        return "database_record"
    if ref_type == "inference":
        return "inferred_support"
    return "weak"


def _evidence_level_score(level: str) -> float:
    return {
        "direct_visual": 1.0,
        "visual_confirmed": 0.95,
        "trajectory_confirmed": 0.9,
        "model_confirmed": 0.85,
        "state_change": 0.75,
        "visual_asset": 0.7,
        "text_support": 0.55,
        "database_record": 0.5,
        "model_candidate": 0.45,
        "visual_candidate": 0.4,
        "inferred_support": 0.35,
        "weak": 0.2,
        "none": 0.0,
    }.get(str(level or "weak"), 0.3)


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path) if path.exists() else []


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _dedupe_events(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in rows:
        key = str(row.get("video_event_id") or row)
        if key not in seen:
            seen.add(key)
            output.append(dict(row))
    return output


def _dedupe_refs(refs: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output = []
    for ref in refs:
        key = (
            str(ref.get("type") or ""),
            str(
                ref.get("evidence_id")
                or ref.get("video_event_id")
                or ref.get("state_change_id")
                or ref.get("asset_id")
                or ref.get("sop_step_id")
                or ref.get("history_id")
                or ref.get("inference_id")
                or ""
            ),
            str(ref.get("path") or ""),
        )
        if key not in seen:
            seen.add(key)
            output.append(dict(ref))
    return output


def _min_time(rows: list[Mapping[str, Any]]) -> str | None:
    times = [str(row.get("global_start_time")) for row in rows if row.get("global_start_time")]
    return min(times) if times else None


def _max_time(rows: list[Mapping[str, Any]]) -> str | None:
    times = [str(row.get("global_end_time") or row.get("global_start_time")) for row in rows if row.get("global_end_time") or row.get("global_start_time")]
    return max(times) if times else None


def _session_id(*sources: Any) -> str:
    for source in sources[:-1]:
        for row in source:
            if isinstance(row, Mapping) and row.get("session_id"):
                return str(row["session_id"])
    return Path(sources[-1]).name


__all__ = ["build_experiment_process", "load_experiment_process"]
