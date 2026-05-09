from __future__ import annotations

from typing import List

from lab_vla.core.contracts import SkillStep, TaskCommand


class SkillLibrary:
    def build_steps(self, command: TaskCommand, frame_id: int) -> List[SkillStep]:
        steps: List[SkillStep] = []
        steps.append(
            SkillStep(
                step_id=f"{command.task_id}-detect-{frame_id}",
                skill_name="detect_target",
                command="detect",
                args={"target_object": command.target_object},
                constraints=command.constraints,
            )
        )
        steps.append(
            SkillStep(
                step_id=f"{command.task_id}-locate-{frame_id}",
                skill_name="locate_target",
                command="locate",
                args={"target_object": command.target_object},
                constraints=command.constraints,
            )
        )
        steps.append(
            SkillStep(
                step_id=f"{command.task_id}-pick-{frame_id}",
                skill_name="pick_object",
                command="pick",
                args={"target_object": command.target_object},
                constraints=command.constraints,
            )
        )
        steps.append(
            SkillStep(
                step_id=f"{command.task_id}-place-{frame_id}",
                skill_name="place_object",
                command="place",
                args={"target_zone": command.target_zone},
                constraints=command.constraints,
            )
        )
        steps.append(
            SkillStep(
                step_id=f"{command.task_id}-verify-{frame_id}",
                skill_name="verify_result",
                command="verify",
                args={"expected_zone": command.target_zone},
                constraints=command.constraints,
            )
        )
        return steps

