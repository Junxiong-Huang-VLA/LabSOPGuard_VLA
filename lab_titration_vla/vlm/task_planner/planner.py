from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from project_name.action.policy import ActionPolicy
from project_name.common.schemas import PerceptionResult
from project_name.language.instruction_parser import InstructionParser


class VLMTaskPlanner:
    def __init__(self) -> None:
        self.parser = InstructionParser()
        self.policy = ActionPolicy()

    def plan(self, instruction: str, perception: PerceptionResult) -> Dict[str, Any]:
        parsed = self.parser.parse(instruction)
        plan = self.policy.plan(parsed, perception)
        return asdict(plan)

