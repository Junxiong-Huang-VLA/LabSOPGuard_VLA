from __future__ import annotations

from typing import Protocol

from lab_vla.core.contracts import SceneState, SkillPlan, TaskCommand


class TaskPlannerBackend(Protocol):
    def plan(self, command: TaskCommand, scene: SceneState) -> SkillPlan:
        ...

