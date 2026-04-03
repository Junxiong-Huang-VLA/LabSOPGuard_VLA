from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Protocol

try:
    from lab_titration_vla.control.moveit_interface.command_bridge import build_moveit_goal
except Exception:
    def build_moveit_goal(robot_command: Dict[str, Any]) -> Dict[str, Any]:
        if robot_command.get("status") != "ready":
            return {"goal_status": "blocked", "reason": robot_command.get("reason", "not_ready")}
        targets = robot_command.get("robot_targets", {})
        return {
            "goal_status": "ready",
            "planner": "moveit2",
            "target_xyz_m": targets.get("target_xyz_m"),
            "pre_grasp_xyz_m": targets.get("pre_grasp_xyz_m"),
            "place_xyz_m": targets.get("place_xyz_m"),
            "speed_scale": robot_command.get("safety_constraints", {}).get("max_speed_scale", 0.2),
            "execution_plan": robot_command.get("execution_plan", []),
        }

from project_name.action.policy import ActionPolicy
from project_name.action.robot_command_builder import RobotCommandBuilder
from project_name.common.schemas import PerceptionResult
from project_name.language.instruction_parser import ParsedInstruction

from lab_vla.core.contracts import SceneState, SkillPlan, TaskCommand


class RobotAdapter(Protocol):
    name: str

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        ...


def _scene_to_perception(scene: SceneState, command: TaskCommand) -> PerceptionResult:
    bbox = scene.target_bbox or [0, 0, 0, 0]
    cx = float((bbox[0] + bbox[2]) / 2.0) if bbox else 0.0
    cy = float((bbox[1] + bbox[3]) / 2.0) if bbox else 0.0
    return PerceptionResult(
        target_name=command.target_object,
        bbox=bbox,
        center_point=[cx, cy],
        depth_info={
            "center_depth": 0.0,
            "region_depth_mean": 0.0,
            "region_depth_median": 0.0,
            "valid_depth_ratio": 0.0,
        },
        confidence=float(scene.confidence),
        xyz=scene.target_xyz_m if scene.target_xyz_m is not None else [0.22, 0.04, 0.12],
        target_representation={
            "scene_frame_id": scene.frame_id,
            "object_labels": scene.object_labels,
        },
    )


class MoveItAdapter:
    name = "moveit"

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        goal = build_moveit_goal(command)
        return {"adapter": self.name, "goal": goal, "status": goal.get("goal_status", "unknown")}


class MockRobotAdapter:
    name = "mock"

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "status": "ready" if command.get("status") == "ready" else "blocked",
            "echo": command,
        }


class RobotCommandService:
    def __init__(self, bridge_cfg: Dict[str, Any], adapter_name: str) -> None:
        self.builder = RobotCommandBuilder(config=bridge_cfg)
        self.policy = ActionPolicy()
        self.adapter: RobotAdapter = MoveItAdapter() if adapter_name.lower() == "moveit" else MockRobotAdapter()

    def build_and_execute(
        self,
        command: TaskCommand,
        scene: SceneState,
        plan: SkillPlan,
        calibration: Dict[str, Any],
    ) -> Dict[str, Any]:
        parsed = ParsedInstruction(
            intent=str(command.metadata.get("intent", "move")),
            target_object=command.target_object,
            source_zone=command.source_zone,
            target_zone=command.target_zone,
            constraints=list(command.constraints),
        )
        perception = _scene_to_perception(scene=scene, command=command)
        action_plan = self.policy.plan(parsed, perception)
        action_plan.metadata["skill_plan"] = [asdict(x) for x in plan.steps]

        cmd = self.builder.build(
            sample_id=command.task_id,
            parsed_instruction=parsed,
            perception=perception,
            action_plan=action_plan,
            camera_intrinsics=calibration.get("camera_intrinsics"),
            hand_eye_extrinsics=calibration.get("hand_eye_extrinsics"),
        )
        return self.adapter.execute(cmd.command)
