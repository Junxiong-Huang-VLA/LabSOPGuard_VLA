from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from lab_vla.adapters.robot_adapter import RobotCommandService
from lab_vla.core.contracts import ExecutionResult, SafetyDecision, SceneState, SkillPlan, TaskCommand


class Executor:
    def __init__(self, bridge_cfg: Dict[str, Any], adapter_name: str) -> None:
        self.robot = RobotCommandService(bridge_cfg=bridge_cfg, adapter_name=adapter_name)

    def run(
        self,
        task: TaskCommand,
        scene: SceneState,
        plan: SkillPlan,
        safety: SafetyDecision,
        calibration: Dict[str, Any],
    ) -> ExecutionResult:
        if not safety.safe_to_execute:
            return ExecutionResult(
                status="blocked",
                adapter=self.robot.adapter.name,
                command={"recovery_plan": safety.recovery_plan},
                verification={"violations": safety.violations},
                error=safety.reason,
            )

        result = self.robot.build_and_execute(
            command=task,
            scene=scene,
            plan=plan,
            calibration=calibration,
        )
        ok = str(result.get("status", "")).lower() in {"ready", "ok"}
        verification = {
            "target_visible": scene.target_bbox is not None,
            "plan_steps": [asdict(x) for x in plan.steps],
        }
        return ExecutionResult(
            status="ok" if ok else "failed",
            adapter=str(result.get("adapter", self.robot.adapter.name)),
            command=result,
            verification=verification,
            error=None if ok else "adapter_failed",
        )

