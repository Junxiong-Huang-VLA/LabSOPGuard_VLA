from __future__ import annotations

from typing import List

from project_name.common.schemas import ActionPlan, PerceptionResult
from project_name.language.instruction_parser import ParsedInstruction


class ActionPolicy:
    def plan(self, parsed_instruction: ParsedInstruction, perception: PerceptionResult) -> ActionPlan:
        seq: List[str] = ["approach_target"]

        if parsed_instruction.intent == "pick":
            seq.extend(["open_gripper", "align_gripper", "close_gripper", "lift"])
        elif parsed_instruction.intent == "place":
            seq.extend(["move_to_target_zone", "lower", "open_gripper", "retreat"])
        elif parsed_instruction.intent == "sort":
            seq.extend(["scan_all_samples", "rank_samples", "execute_reorder"])
        else:
            seq.extend(["align_gripper", "close_gripper", "move_to_target_zone", "open_gripper"])

        center_depth = float(perception.depth_info.get("center_depth", 0.0))

        if perception.xyz is not None:
            grasp_point_xyz = [round(float(v), 4) for v in perception.xyz]
        else:
            cx, cy = perception.center_point
            grasp_point_xyz = [round(cx / 1000.0, 4), round(cy / 1000.0, 4), round(center_depth, 4)]

        end_effector_target_xyz = [
            grasp_point_xyz[0],
            grasp_point_xyz[1],
            round(grasp_point_xyz[2] + 0.05, 4),
        ]

        return ActionPlan(
            action_sequence=seq,
            grasp_point_xyz=grasp_point_xyz,
            end_effector_target_xyz=end_effector_target_xyz,
            metadata={
                "constraints": parsed_instruction.constraints,
                "target_representation": perception.target_representation,
            },
        )
