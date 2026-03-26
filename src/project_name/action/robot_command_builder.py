from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from project_name.common.schemas import ActionPlan, PerceptionResult
from project_name.language.instruction_parser import ParsedInstruction
from project_name.utils.spatial import camera_to_robot_xyz, pixel_to_camera_xyz


@dataclass
class RobotCommandResult:
    command: Dict[str, Any]
    status: str


class RobotCommandBuilder:
    """Build executable-ish robot command payload from perception + action plan."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.ee_lift_offset_m = float(cfg.get("ee_lift_offset_m", 0.05))
        self.default_gripper_open = float(cfg.get("gripper_open_width_m", 0.08))
        self.default_gripper_close = float(cfg.get("gripper_close_width_m", 0.018))
        self.default_speed = float(cfg.get("speed_scale_default", 0.3))
        self.safe_speed = float(cfg.get("speed_scale_safe", 0.15))
        self.workspace = cfg.get("workspace", {})

    def build(
        self,
        sample_id: str,
        parsed_instruction: ParsedInstruction,
        perception: PerceptionResult,
        action_plan: ActionPlan,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        hand_eye_extrinsics: Optional[Dict[str, List[float]]] = None,
    ) -> RobotCommandResult:
        target_xyz = self._resolve_robot_xyz(
            perception=perception,
            camera_intrinsics=camera_intrinsics,
            hand_eye_extrinsics=hand_eye_extrinsics,
        )
        if target_xyz is None:
            return RobotCommandResult(
                command={
                    "sample_id": sample_id,
                    "status": "blocked",
                    "reason": "missing_target_xyz",
                    "instruction_intent": parsed_instruction.intent,
                    "action_sequence": action_plan.action_sequence,
                },
                status="blocked",
            )

        speed = self.safe_speed if "slow_motion" in parsed_instruction.constraints else self.default_speed
        ee_pre_grasp = [target_xyz[0], target_xyz[1], target_xyz[2] + self.ee_lift_offset_m]
        ee_place = [target_xyz[0] + 0.10, target_xyz[1], target_xyz[2] + 0.01]

        cmd = {
            "sample_id": sample_id,
            "status": "ready",
            "instruction_intent": parsed_instruction.intent,
            "target_object": perception.target_name,
            "target_representation": perception.target_representation,
            "safety_constraints": {
                "constraints": parsed_instruction.constraints,
                "workspace": self.workspace,
                "max_speed_scale": speed,
            },
            "robot_targets": {
                "target_xyz_m": [round(float(v), 4) for v in target_xyz],
                "pre_grasp_xyz_m": [round(float(v), 4) for v in ee_pre_grasp],
                "place_xyz_m": [round(float(v), 4) for v in ee_place],
                "gripper_open_width_m": self.default_gripper_open,
                "gripper_close_width_m": self.default_gripper_close,
            },
            "execution_plan": self._to_motion_primitives(
                actions=action_plan.action_sequence,
                speed_scale=speed,
            ),
        }
        return RobotCommandResult(command=cmd, status="ready")

    def _resolve_robot_xyz(
        self,
        perception: PerceptionResult,
        camera_intrinsics: Optional[Dict[str, float]],
        hand_eye_extrinsics: Optional[Dict[str, List[float]]],
    ) -> Optional[List[float]]:
        if perception.xyz is not None and len(perception.xyz) == 3:
            cam_xyz = [float(v) for v in perception.xyz]
        else:
            z = float(perception.depth_info.get("center_depth", 0.0))
            cam_xyz = pixel_to_camera_xyz(
                u=float(perception.center_point[0]),
                v=float(perception.center_point[1]),
                z_m=z,
                intrinsics=camera_intrinsics,
            )
        return camera_to_robot_xyz(cam_xyz, hand_eye_extrinsics)

    def _to_motion_primitives(self, actions: List[str], speed_scale: float) -> List[Dict[str, Any]]:
        primitives: List[Dict[str, Any]] = []
        for act in actions:
            if act == "open_gripper":
                primitives.append({"type": "gripper", "command": "open", "speed_scale": speed_scale})
            elif act == "close_gripper":
                primitives.append({"type": "gripper", "command": "close", "speed_scale": speed_scale})
            elif act in {"approach_target", "align_gripper", "lift", "lower", "retreat", "move_to_target_zone"}:
                primitives.append({"type": "cartesian_move", "command": act, "speed_scale": speed_scale})
            else:
                primitives.append({"type": "logic", "command": act, "speed_scale": speed_scale})
        return primitives
