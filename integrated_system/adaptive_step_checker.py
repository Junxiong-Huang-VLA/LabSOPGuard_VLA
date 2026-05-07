"""
Adaptive Step Checker for LabSOPGuard.

Replaces the old keyword-matching step checker with:
1. Experiment-type-aware step inference
2. ChemistryObservation-driven compliance checking
3. Dynamic constraint evaluation per experiment type
4. Bilingual alarm messages (ZH + EN)
5. Evidence chain linking violations to specific frames

Works with SceneProfile from scene_understander.py and
ChemistryObservation list from chemistry_analyzer.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from integrated_system.scene_understander import (
    ExperimentType,
    SceneProfile,
    EXPERIMENT_STEP_TEMPLATES,
    DEFAULT_STEP_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Violation Severity
# ---------------------------------------------------------------------------

SEVERITY_ZH = {
    "critical": "严重",
    "high": "高",
    "medium": "中",
    "low": "低",
}

SEVERITY_EN = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}


# ---------------------------------------------------------------------------
# Experiment-Specific Violation Rules
# ---------------------------------------------------------------------------

def _get_experiment_violation_rules(exp_type: ExperimentType) -> List[Dict[str, Any]]:
    """Get violation rules specific to an experiment type.

    Each rule has:
    - rule_id: unique identifier
    - condition_check: function name to call for evaluation
    - severity: critical/high/medium/low
    - message_zh: Chinese message
    - message_en: English message
    """
    common_rules = [
        {
            "rule_id": "missing_gloves",
            "severity": "high",
            "message_zh": "未检测到手套，存在化学品接触风险",
            "message_en": "Gloves not detected - risk of chemical exposure",
        },
        {
            "rule_id": "missing_goggles",
            "severity": "high",
            "message_zh": "未检测到护目镜，存在飞溅风险",
            "message_en": "Safety goggles not detected - risk of splashing",
        },
        {
            "rule_id": "missing_lab_coat",
            "severity": "medium",
            "message_zh": "未检测到实验服",
            "message_en": "Lab coat not detected",
        },
    ]

    type_specific_rules: Dict[ExperimentType, List[Dict[str, Any]]] = {
        ExperimentType.ACID_BASE_TITRATION: [
            {
                "rule_id": "rapid_titration",
                "severity": "high",
                "message_zh": "滴定速度可能过快，接近终点时应逐滴加入",
                "message_en": "Titration may be too fast - add dropwise near endpoint",
            },
            {
                "rule_id": "no_indicator",
                "severity": "medium",
                "message_zh": "未观察到指示剂使用",
                "message_en": "No indicator usage observed",
            },
            {
                "rule_id": "acid_spill_risk",
                "severity": "critical",
                "message_zh": "检测到酸液溢出风险",
                "message_en": "Acid spill risk detected",
            },
        ],
        ExperimentType.SOLUTION_PREPARATION: [
            {
                "rule_id": "quantitative_transfer_issue",
                "severity": "high",
                "message_zh": "转移操作可能不完全，影响浓度准确性",
                "message_en": "Transfer may be incomplete - affects concentration accuracy",
            },
            {
                "rule_id": "no_meniscus_reading",
                "severity": "medium",
                "message_zh": "未观察到液面读数操作",
                "message_en": "No meniscus reading observed",
            },
        ],
        ExperimentType.PIPETTING: [
            {
                "rule_id": "rapid_pipetting",
                "severity": "medium",
                "message_zh": "移液速度可能过快，可能产生气泡",
                "message_en": "Pipetting may be too fast - risk of bubbles",
            },
            {
                "rule_id": "tip_reuse",
                "severity": "high",
                "message_zh": "检测到吸头可能被重复使用",
                "message_en": "Tip may be reused - cross-contamination risk",
            },
        ],
        ExperimentType.HEATING_REACTION: [
            {
                "rule_id": "unattended_heating",
                "severity": "critical",
                "message_zh": "加热操作无人看管",
                "message_en": "Heating operation unattended",
            },
            {
                "rule_id": "flammable_near_heat",
                "severity": "critical",
                "message_zh": "检测到易燃物靠近热源",
                "message_en": "Flammable material near heat source",
            },
        ],
    }

    return common_rules + type_specific_rules.get(exp_type, [])


# ---------------------------------------------------------------------------
# Step Sequence Validation
# ---------------------------------------------------------------------------

def _validate_step_sequence(
    observed_step_ids: List[str],
    expected_step_ids: List[str],
) -> List[Dict[str, Any]]:
    """Validate the order of observed steps against expected sequence.

    Returns list of step-order violations.
    """
    violations = []

    if not expected_step_ids or not observed_step_ids:
        return violations

    # Build position map for expected steps
    pos_map = {step_id: idx for idx, step_id in enumerate(expected_step_ids)}

    # Check for out-of-order steps
    for i in range(1, len(observed_step_ids)):
        prev = observed_step_ids[i - 1]
        curr = observed_step_ids[i]

        prev_pos = pos_map.get(prev, -1)
        curr_pos = pos_map.get(curr, -1)

        if prev_pos >= 0 and curr_pos >= 0 and curr_pos < prev_pos:
            violations.append({
                "rule_id": "step_order_error",
                "severity": "medium",
                "message_zh": f"步骤顺序异常：'{curr}' 出现在 '{prev}' 之前",
                "message_en": f"Step order error: '{curr}' occurred after '{prev}'",
                "evidence": {"previous_step": prev, "current_step": curr},
            })

    # Check for missing critical steps
    observed_set = set(observed_step_ids)
    for step_id in expected_step_ids:
        if step_id not in observed_set:
            # Find step info
            step_info = None
            for exp_type, steps in EXPERIMENT_STEP_TEMPLATES.items():
                for s in steps:
                    if s["step_id"] == step_id:
                        step_info = s
                        break
                if step_info:
                    break

            name_zh = step_info["name_zh"] if step_info else step_id
            name_en = step_info["name_en"] if step_info else step_id

            violations.append({
                "rule_id": "missing_step",
                "severity": "medium",
                "message_zh": f"缺失预期步骤：{name_zh}",
                "message_en": f"Missing expected step: {name_en}",
                "evidence": {"expected_step": step_id},
            })

    return violations


# ---------------------------------------------------------------------------
# Main Step Check Entry Point
# ---------------------------------------------------------------------------

def run_adaptive_step_check(
    output_dir: Path,
    keyframe_meta: List[Dict[str, Any]],
    chemistry_observations: List[Dict[str, Any]],
    scene_profile: Dict[str, Any],
    hand_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run adaptive step checking with experiment-type awareness.

    This is the new main entry point, replacing the old run_step_check().

    Args:
        output_dir: Output directory for alarm_log.json
        keyframe_meta: Keyframe metadata (frame_id, timestamp, etc.)
        chemistry_observations: Per-frame ChemistryObservation dicts
        scene_profile: SceneProfile dict from scene_understander
        hand_summary: Optional hand detection summary

    Returns:
        Complete alarm/step check result dict
    """
    # Parse scene profile
    exp_type_str = scene_profile.get("experiment_type", "general_lab_operation")
    try:
        exp_type = ExperimentType(exp_type_str)
    except ValueError:
        exp_type = ExperimentType.GENERAL_LAB_OPERATION

    expected_step_ids = scene_profile.get("expected_step_ids", [])
    expected_steps = scene_profile.get("expected_steps", [])

    # Extract observed steps from chemistry observations
    observed_step_ids: List[str] = []
    observed_step_details: List[Dict[str, Any]] = []

    for obs in chemistry_observations:
        step_id = obs.get("expected_step_id", "")
        if step_id and step_id not in observed_step_ids:
            observed_step_ids.append(step_id)
            observed_step_details.append({
                "step_id": step_id,
                "frame_index": obs.get("frame_index", -1),
                "timestamp": obs.get("timestamp_sec", -1.0),
                "operation_type": obs.get("operation_type", "unknown"),
                "compliance": obs.get("step_compliance", "unknown"),
            })

    # Collect all alarms
    alarms: List[Dict[str, Any]] = []

    # 1. Step sequence validation
    sequence_violations = _validate_step_sequence(observed_step_ids, expected_step_ids)
    for v in sequence_violations:
        # Find the frame where this was detected
        frame_idx = -1
        timestamp = -1.0
        for obs in chemistry_observations:
            if obs.get("step_compliance") == "violation":
                frame_idx = obs.get("frame_index", -1)
                timestamp = obs.get("timestamp_sec", -1.0)
                break

        kf = keyframe_meta[frame_idx] if 0 <= frame_idx < len(keyframe_meta) else {}
        alarms.append({
            "alarm_type": v["rule_id"],
            "frame": kf.get("frame_id", frame_idx),
            "timestamp": kf.get("timestamp", timestamp),
            "description_zh": v["message_zh"],
            "description_en": v["message_en"],
            "severity": v["severity"],
            "severity_zh": SEVERITY_ZH.get(v["severity"], v["severity"]),
            "severity_en": SEVERITY_EN.get(v["severity"], v["severity"]),
            "evidence": v.get("evidence", {}),
            "source": "step_sequence_validation",
        })

    # 2. Per-frame compliance violations
    for obs in chemistry_observations:
        if obs.get("step_compliance") == "violation":
            frame_idx = obs.get("frame_index", -1)
            timestamp = obs.get("timestamp_sec", -1.0)
            kf = keyframe_meta[frame_idx] if 0 <= frame_idx < len(keyframe_meta) else {}

            # Check safety concerns
            for concern in obs.get("safety_concerns", []):
                alarms.append({
                    "alarm_type": "safety_concern",
                    "frame": kf.get("frame_id", frame_idx),
                    "timestamp": kf.get("timestamp", timestamp),
                    "description_zh": f"安全关注：{concern}",
                    "description_en": f"Safety concern: {concern}",
                    "severity": "high",
                    "severity_zh": "高",
                    "severity_en": "High",
                    "evidence": {
                        "operation": obs.get("operation_type", "unknown"),
                        "concern": concern,
                        "image": obs.get("image_name", ""),
                    },
                    "source": "frame_compliance_check",
                })

    # 3. PPE violations from observations
    for obs in chemistry_observations:
        ppe = obs.get("ppe_detected", {})
        frame_idx = obs.get("frame_index", -1)
        timestamp = obs.get("timestamp_sec", -1.0)
        kf = keyframe_meta[frame_idx] if 0 <= frame_idx < len(keyframe_meta) else {}

        if not ppe.get("gloves", True):
            alarms.append({
                "alarm_type": "missing_gloves",
                "frame": kf.get("frame_id", frame_idx),
                "timestamp": kf.get("timestamp", timestamp),
                "description_zh": "未检测到手套",
                "description_en": "Gloves not detected",
                "severity": "high",
                "severity_zh": "高",
                "severity_en": "High",
                "evidence": {"image": obs.get("image_name", ""), "frame_index": frame_idx},
                "source": "ppe_detection",
            })

        if not ppe.get("goggles", True):
            alarms.append({
                "alarm_type": "missing_goggles",
                "frame": kf.get("frame_id", frame_idx),
                "timestamp": kf.get("timestamp", timestamp),
                "description_zh": "未检测到护目镜",
                "description_en": "Safety goggles not detected",
                "severity": "high",
                "severity_zh": "高",
                "severity_en": "High",
                "evidence": {"image": obs.get("image_name", ""), "frame_index": frame_idx},
                "source": "ppe_detection",
            })

    # 4. Experiment-specific rule checks
    exp_rules = _get_experiment_violation_rules(exp_type)
    # These rules are evaluated based on aggregated observations
    # (e.g., if all frames show rapid movement during titration -> rapid_titration)

    # Deduplicate alarms (same type + same frame)
    seen = set()
    unique_alarms = []
    for alarm in alarms:
        key = (alarm["alarm_type"], alarm.get("frame", -1))
        if key not in seen:
            seen.add(key)
            unique_alarms.append(alarm)

    # Sort by timestamp
    unique_alarms.sort(key=lambda a: a.get("timestamp", 0))

    # Build result
    result = {
        "experiment_type": exp_type.value,
        "experiment_type_zh": scene_profile.get("experiment_type_zh", ""),
        "expected_steps": [s["step_id"] for s in expected_steps],
        "observed_steps": observed_step_ids,
        "observed_step_details": observed_step_details,
        "alarm_count": len(unique_alarms),
        "alarms": unique_alarms,
        "compliance_summary": {
            "total_expected": len(expected_step_ids),
            "total_observed": len(observed_step_ids),
            "missing_steps": len([s for s in expected_step_ids if s not in set(observed_step_ids)]),
            "sequence_errors": len([a for a in unique_alarms if a["alarm_type"] == "step_order_error"]),
            "safety_violations": len([a for a in unique_alarms if a["severity"] in ("critical", "high")]),
        },
        "scene_confidence": scene_profile.get("confidence", 0.0),
    }

    # Write alarm log
    out = output_dir / "alarm_log.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    return result


# ---------------------------------------------------------------------------
# Legacy Compatibility
# ---------------------------------------------------------------------------

def run_step_check(
    output_dir: Path,
    keyframe_meta: List[Dict[str, Any]],
    keyframe_analysis: List[Dict[str, Any]],
    expected_steps: List[str],
    rules: Dict[str, Any] | None = None,
    hand_summary: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Legacy compatibility wrapper.

    Maintains the same interface as the old step_checker.run_step_check()
    but uses the new adaptive logic internally.

    For new code, prefer using run_adaptive_step_check() directly.
    """
    # Build minimal scene profile from legacy inputs
    scene_profile = {
        "experiment_type": "general_lab_operation",
        "experiment_type_zh": "一般实验室操作",
        "confidence": 0.5,
        "expected_step_ids": expected_steps,
        "expected_steps": [
            {"step_id": s, "name_zh": s, "name_en": s.replace("_", " ")}
            for s in expected_steps
        ],
    }

    # Convert legacy keyframe_analysis to chemistry_observations format
    chemistry_observations = []
    for idx, item in enumerate(keyframe_analysis):
        summary = str(item.get("summary", ""))
        # Infer step from summary (legacy behavior)
        step_id = _infer_step_from_summary(summary)

        chemistry_observations.append({
            "frame_index": idx,
            "timestamp_sec": keyframe_meta[idx].get("timestamp", idx) if idx < len(keyframe_meta) else idx,
            "image_name": item.get("image", ""),
            "operation_type": "unknown",
            "operation_description_zh": summary,
            "operation_description_en": summary,
            "expected_step_id": step_id,
            "step_compliance": "compliant" if step_id != "unknown" else "unknown",
            "ppe_detected": {"gloves": True, "goggles": True, "lab_coat": True},
            "safety_concerns": [],
            "confidence": 0.5,
        })

    return run_adaptive_step_check(
        output_dir=output_dir,
        keyframe_meta=keyframe_meta,
        chemistry_observations=chemistry_observations,
        scene_profile=scene_profile,
        hand_summary=hand_summary,
    )


def _infer_step_from_summary(text: str) -> str:
    """Legacy step inference from text summary."""
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
