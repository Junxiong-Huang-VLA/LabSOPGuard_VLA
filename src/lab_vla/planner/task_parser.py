from __future__ import annotations

from typing import Any, Dict

try:
    from lab_titration_vla.vlm.qwen3_vl_parser.parser import Qwen3VLParser
except Exception:
    from project_name.language.instruction_parser import InstructionParser

    class Qwen3VLParser:
        def __init__(self) -> None:
            self._fallback = InstructionParser()

        def parse_instruction(self, instruction: str) -> Dict[str, object]:
            parsed = self._fallback.parse(instruction)
            return self._fallback.to_dict(parsed)

from lab_vla.core.contracts import TaskCommand


class TaskParser:
    def __init__(self) -> None:
        self.parser = Qwen3VLParser()

    def parse(self, task_id: str, instruction: str) -> TaskCommand:
        parsed: Dict[str, Any] = self.parser.parse_instruction(instruction)
        return TaskCommand(
            task_id=task_id,
            instruction=instruction,
            target_object=str(parsed.get("target_object", "sample_container")),
            source_zone=str(parsed.get("source_zone", "workbench_left")),
            target_zone=str(parsed.get("target_zone", "workbench_right")),
            constraints=list(parsed.get("constraints", [])),
            metadata={"intent": parsed.get("intent", "move"), "parser_backend": "qwen3_interface"},
        )
