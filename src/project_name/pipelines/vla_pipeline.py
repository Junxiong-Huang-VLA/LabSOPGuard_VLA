from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from project_name.action.policy import ActionPolicy
from project_name.action.robot_command_builder import RobotCommandBuilder
from project_name.common.config import load_yaml
from project_name.common.schemas import VLAResult
from project_name.language.instruction_parser import InstructionParser
from project_name.perception.perception_engine import PerceptionEngine

_log = logging.getLogger(__name__)


class VLAPipeline:
    def __init__(self, robot_config_path: str = "configs/robot/bridge.yaml") -> None:
        self.perception = PerceptionEngine()
        self.parser = InstructionParser()
        self.policy = ActionPolicy()

        try:
            robot_cfg = load_yaml(robot_config_path)
        except FileNotFoundError:
            _log.warning("Robot config not found at '%s'; using empty config.", robot_config_path)
            robot_cfg = {}
        except Exception as exc:
            _log.warning("Failed to load robot config '%s': %s", robot_config_path, exc)
            robot_cfg = {}
        bridge_cfg = robot_cfg.get("bridge", {}) if isinstance(robot_cfg, dict) else {}
        self.robot_builder = RobotCommandBuilder(config=bridge_cfg)

    def run(
        self,
        sample_id: str,
        instruction: str,
        rgb_path: str,
        depth_path: str | None = None,
        raw_target: Optional[Dict[str, Any]] = None,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        hand_eye_extrinsics: Optional[Dict[str, Any]] = None,
    ) -> VLAResult:
        perception_result = self.perception.infer(
            rgb_path=rgb_path,
            depth_path=depth_path,
            raw_target=raw_target,
            camera_intrinsics=camera_intrinsics,
            sample_id=sample_id,
        )
        parsed = self.parser.parse(instruction)
        action_plan = self.policy.plan(parsed, perception_result)

        robot_cmd = self.robot_builder.build(
            sample_id=sample_id,
            parsed_instruction=parsed,
            perception=perception_result,
            action_plan=action_plan,
            camera_intrinsics=camera_intrinsics,
            hand_eye_extrinsics=hand_eye_extrinsics,
        )
        action_plan.robot_command = robot_cmd.command
        action_plan.metadata["robot_command_status"] = robot_cmd.status

        return VLAResult(
            sample_id=sample_id,
            instruction=instruction,
            perception=perception_result,
            action=action_plan,
        )
