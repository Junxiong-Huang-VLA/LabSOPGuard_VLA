from __future__ import annotations

from lab_vla.core.contracts import SceneState, SkillPlan, TaskCommand
from lab_vla.skills.library import SkillLibrary


class SkillPlanner:
    def __init__(self) -> None:
        self.library = SkillLibrary()

    def plan(self, command: TaskCommand, scene: SceneState) -> SkillPlan:
        steps = self.library.build_steps(command=command, frame_id=scene.frame_id)
        return SkillPlan(
            plan_id=f"plan-{command.task_id}-{scene.frame_id}",
            task_id=command.task_id,
            steps=steps,
            planner_backend="deterministic_skill_planner",
            metadata={
                "scene_confidence": scene.confidence,
                "target_visible": scene.target_bbox is not None,
            },
        )

