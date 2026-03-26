from __future__ import annotations

from typing import Dict

from project_name.language.instruction_parser import InstructionParser


class Qwen3VLParser:
    """
    Interface parser: currently uses local deterministic parser.
    Can be replaced with Qwen3-VL JSON parser API later.
    """

    def __init__(self) -> None:
        self._fallback = InstructionParser()

    def parse_instruction(self, instruction: str) -> Dict[str, object]:
        parsed = self._fallback.parse(instruction)
        return self._fallback.to_dict(parsed)

