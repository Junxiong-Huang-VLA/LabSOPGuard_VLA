from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List


def _infer_step_from_summary(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["glove", "goggle", "ppe", "lab coat"]):
        return "wear_ppe"
    if any(k in t for k in ["label", "reagent", "bottle text"]):
        return "verify_reagent_label"
    if any(k in t for k in ["pipette", "dropper", "transfer tool"]):
        return "prepare_transfer_tool"
    if any(k in t for k in ["transfer", "pour", "dispense"]):
        return "execute_transfer"
    if any(k in t for k in ["close", "cap", "lid"]):
        return "close_container"
    if any(k in t for k in ["clean", "wipe", "sanitize"]):
        return "clean_workspace"
    if any(k in t for k in ["waste", "dispose", "trash"]):
        return "dispose_waste"
    return "unknown"


def _build_context(
    observed_steps: List[str],
    keyframe_analysis: List[Dict[str, Any]],
    hand_summary: Dict[str, Any] | None,
) -> Dict[str, bool]:
    text_joined = " ".join(str(x.get("summary", "")).lower() for x in keyframe_analysis)

    ctx = {
        "gloves": ("glove" in text_joined),
        "goggles": ("goggle" in text_joined) or ("safety glasses" in text_joined),
        "lab_coat": ("lab coat" in text_joined) or ("labcoat" in text_joined),
        "label_verified": "verify_reagent_label" in observed_steps,
        "transfer_started": "execute_transfer" in observed_steps,
        "transfer_in_restricted_zone": ("restricted zone" in text_joined) or ("unsafe zone" in text_joined),
        "container_closed": "close_container" in observed_steps,
        "waste_disposed": "dispose_waste" in observed_steps,
        "experiment_end": len(observed_steps) > 0,
    }

    if hand_summary:
        # If no hands detected during operational frames, treat as potential execution uncertainty.
        ratio = float(hand_summary.get("hand_presence_ratio", 0.0) or 0.0)
        if ratio < 0.05:
            ctx["transfer_started"] = ctx["transfer_started"] or ("transfer" in text_joined)

    return ctx


def _safe_eval_condition(expr: str, context: Dict[str, bool]) -> bool:
    if not expr or not isinstance(expr, str):
        return False

    node = ast.parse(expr, mode="eval")
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Constant,
    )
    for n in ast.walk(node):
        if not isinstance(n, allowed_nodes):
            raise ValueError(f"Unsupported condition syntax: {type(n).__name__}")
        if isinstance(n, ast.Name) and n.id not in context:
            raise ValueError(f"Unknown variable in condition: {n.id}")

    return bool(eval(compile(node, "<rule_condition>", "eval"), {"__builtins__": {}}, context))


def run_step_check(
    output_dir: Path,
    keyframe_meta: List[Dict[str, Any]],
    keyframe_analysis: List[Dict[str, Any]],
    expected_steps: List[str],
    rules: Dict[str, Any] | None = None,
    hand_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    alarms: List[Dict[str, Any]] = []
    observed_steps: List[str] = []

    for idx, item in enumerate(keyframe_analysis):
        summary = str(item.get("summary", ""))
        step = _infer_step_from_summary(summary)
        if step != "unknown":
            observed_steps.append(step)

        kf = keyframe_meta[idx] if idx < len(keyframe_meta) else {}
        frame_id = int(kf.get("frame_id", idx))
        timestamp = float(kf.get("timestamp", idx))

        if "spill" in summary.lower() or "unsafe" in summary.lower():
            alarms.append(
                {
                    "alarm_type": "abnormal_behavior",
                    "frame": frame_id,
                    "timestamp": timestamp,
                    "description": summary[:180],
                    "severity": "high",
                    "evidence": {"image": item.get("image", "")},
                }
            )

    pos = {step: i for i, step in enumerate(expected_steps)}
    for i in range(1, len(observed_steps)):
        prev_step = observed_steps[i - 1]
        curr_step = observed_steps[i]
        if prev_step in pos and curr_step in pos and pos[curr_step] < pos[prev_step]:
            alarms.append(
                {
                    "alarm_type": "step_order_error",
                    "frame": keyframe_meta[min(i, len(keyframe_meta) - 1)].get("frame_id", i),
                    "timestamp": keyframe_meta[min(i, len(keyframe_meta) - 1)].get("timestamp", i),
                    "description": f"Step order issue: {curr_step} happened after {prev_step}.",
                    "severity": "medium",
                    "evidence": {"prev": prev_step, "current": curr_step},
                }
            )

    missing = [s for s in expected_steps if s not in observed_steps]
    for m in missing:
        alarms.append(
            {
                "alarm_type": "missing_step",
                "frame": -1,
                "timestamp": -1,
                "description": f"Expected step not observed: {m}",
                "severity": "medium",
                "evidence": {"expected_step": m},
            }
        )

    # Evaluate configured violation rules from configs/sop/rules.yaml when provided.
    rule_eval_errors: List[str] = []
    violation_rules = (rules or {}).get("violation_rules", {}) if isinstance(rules, dict) else {}
    context = _build_context(observed_steps, keyframe_analysis, hand_summary)
    last_meta = keyframe_meta[-1] if keyframe_meta else {"frame_id": -1, "timestamp": -1}

    if isinstance(violation_rules, dict):
        for rule_id, rule in violation_rules.items():
            if not isinstance(rule, dict):
                continue
            condition = str(rule.get("condition", "")).strip()
            severity = str(rule.get("severity", "medium"))
            message = str(rule.get("message", rule_id))
            try:
                triggered = _safe_eval_condition(condition, context)
            except Exception as exc:
                rule_eval_errors.append(f"{rule_id}: {exc}")
                continue
            if triggered:
                alarms.append(
                    {
                        "alarm_type": str(rule_id),
                        "frame": int(last_meta.get("frame_id", -1)),
                        "timestamp": float(last_meta.get("timestamp", -1)),
                        "description": message,
                        "severity": severity,
                        "evidence": {
                            "condition": condition,
                            "context": context,
                            "sop_step": rule.get("sop_step", "unknown"),
                        },
                    }
                )

    payload = {
        "expected_steps": expected_steps,
        "observed_steps": observed_steps,
        "context": context,
        "rule_eval_errors": rule_eval_errors,
        "alarm_count": len(alarms),
        "alarms": alarms,
    }

    out = output_dir / "alarm_log.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
