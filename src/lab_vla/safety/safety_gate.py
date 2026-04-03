from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

try:
    from lab_titration_vla.control.recovery_controller.controller import build_recovery_plan
except Exception:
    def build_recovery_plan(error_code: str):
        if error_code == "missing_target_xyz":
            return [
                {"type": "logic", "command": "pause"},
                {"type": "logic", "command": "request_relocalization"},
                {"type": "logic", "command": "notify_operator"},
            ]
        return [
            {"type": "logic", "command": "pause"},
            {"type": "cartesian_move", "command": "retreat_safe_pose"},
            {"type": "logic", "command": "notify_operator"},
        ]

try:
    from lab_titration_vla.vlm.safety_checker.checker import check_safety
except Exception:
    def check_safety(plan: Dict[str, Any], ppe_state: Dict[str, bool]) -> Dict[str, Any]:
        violations = []
        if not bool(ppe_state.get("wear_gloves", False)):
            violations.append("no_gloves")
        if not bool(ppe_state.get("wear_goggles", False)):
            violations.append("no_goggles")
        return {
            "safe_to_execute": len(violations) == 0,
            "violations": violations,
            "recommendation": "pause_and_alert" if violations else "proceed",
            "plan": plan,
        }

from project_name.monitoring.sop_engine import SOPComplianceEngine

from lab_vla.core.contracts import PerceptionPacket, SafetyDecision, SceneState


class SafetyGate:
    def __init__(self, rules: Dict[str, Any], cooldown_seconds: float = 0.0) -> None:
        self.engine = SOPComplianceEngine(rules=rules, cooldown_seconds=cooldown_seconds)

    def evaluate(self, perception: PerceptionPacket, scene: SceneState, plan_payload: Dict[str, Any]) -> SafetyDecision:
        detection_like = SimpleNamespace(
            frame_id=perception.frame_id,
            timestamp_sec=perception.timestamp_sec,
            ppe=perception.ppe,
            objects=[{"label": x.label} for x in perception.objects],
            actions=perception.actions,
            layer_outputs=perception.layer_outputs,
        )
        violations = self.engine.update(detection_like)
        vlm_check = check_safety(plan_payload, ppe_state=perception.ppe)

        violation_ids = [x.rule_id for x in violations]
        violation_ids.extend(list(vlm_check.get("violations", [])))
        safe = len(violation_ids) == 0 and bool(scene.ppe_ok)
        if safe:
            return SafetyDecision(safe_to_execute=True, reason="ok")

        recovery = build_recovery_plan(error_code="missing_target_xyz" if scene.target_bbox is None else "safety_violation")
        return SafetyDecision(
            safe_to_execute=False,
            reason="safety_blocked",
            violations=violation_ids,
            recovery_plan=recovery,
            metadata={"vlm_recommendation": vlm_check.get("recommendation", "pause_and_alert")},
        )

    def status(self) -> Dict[str, Any]:
        return self.engine.build_status()
