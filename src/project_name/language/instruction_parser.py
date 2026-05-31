from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ParsedInstruction:
    intent: str
    target_object: str
    source_zone: str
    target_zone: str
    constraints: List[str]


class InstructionParser:
    def parse(self, instruction: str) -> ParsedInstruction:
        text = instruction.lower().strip()

        intent = "move"
        if "pick" in text or "grab" in text:
            intent = "pick"
        elif "place" in text or "put" in text:
            intent = "place"
        elif "sort" in text or "reorder" in text:
            intent = "sort"

        target_object = "sample_container"
        source_zone = "workbench_left"
        target_zone = "workbench_right"

        constraints: List[str] = []
        if "carefully" in text:
            constraints.append("slow_motion")
        if "avoid" in text:
            constraints.append("obstacle_avoidance")

        return ParsedInstruction(
            intent=intent,
            target_object=target_object,
            source_zone=source_zone,
            target_zone=target_zone,
            constraints=constraints,
        )

    def to_dict(self, parsed: ParsedInstruction) -> Dict[str, str | List[str]]:
        return {
            "intent": parsed.intent,
            "target_object": parsed.target_object,
            "source_zone": parsed.source_zone,
            "target_zone": parsed.target_zone,
            "constraints": parsed.constraints,
        }
