from __future__ import annotations

import pytest

from integrated_system.step_checker import _safe_eval_condition, run_step_check


def test_safe_eval_condition_basic_boolean_logic() -> None:
    context = {
        "gloves": True,
        "goggles": True,
        "lab_coat": False,
        "transfer_started": True,
        "transfer_in_restricted_zone": False,
        "container_closed": True,
        "waste_disposed": False,
        "label_verified": True,
        "experiment_end": True,
    }
    assert _safe_eval_condition("gloves and goggles and not transfer_in_restricted_zone", context) is True
    assert _safe_eval_condition("lab_coat or waste_disposed", context) is False


def test_safe_eval_condition_rejects_unknown_variable() -> None:
    with pytest.raises(ValueError, match="Unknown variable"):
        _safe_eval_condition("non_existing_flag and gloves", {"gloves": True})


def test_safe_eval_condition_rejects_call_syntax() -> None:
    with pytest.raises(ValueError, match="Unsupported condition syntax: Call"):
        _safe_eval_condition("__import__('os').system('echo x')", {"gloves": True})


def test_run_step_check_collects_rule_eval_errors_without_executing(tmp_path) -> None:
    payload = run_step_check(
        output_dir=tmp_path,
        keyframe_meta=[{"frame_id": 1, "timestamp": 1.0}],
        keyframe_analysis=[{"summary": "Operator wears gloves and starts transfer"}],
        expected_steps=["wear_ppe", "execute_transfer"],
        rules={
            "violation_rules": {
                "bad_rule": {
                    "condition": "__import__('os').system('echo should_not_run')",
                    "severity": "high",
                    "message": "bad",
                }
            }
        },
        hand_summary=None,
    )

    assert payload["alarm_count"] >= 0
    assert any(item.startswith("bad_rule:") for item in payload.get("rule_eval_errors", []))
