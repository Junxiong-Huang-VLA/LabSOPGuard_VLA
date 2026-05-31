from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass
class StepGraphNode:
    step_id: str
    step_name: str
    stage_name: str
    order_index: int


class StepGraphReasoner:
    def parse_protocol(self, protocol_text: str) -> List[StepGraphNode]:
        nodes: List[StepGraphNode] = []
        for idx, raw in enumerate(protocol_text.splitlines(), start=1):
            line = raw.strip()
            if not line:
                continue
            normalized = re.sub(r"^[\\-\\*\\d\\.\\)\\s]+", "", line).strip()
            if not normalized:
                continue
            nodes.append(
                StepGraphNode(
                    step_id=f"protocol_step_{idx}",
                    step_name=normalized,
                    stage_name=self._guess_stage_name(normalized),
                    order_index=len(nodes),
                )
            )
        return nodes

    def _guess_stage_name(self, text: str) -> str:
        lowered = text.lower()
        if any(token in lowered for token in ["prepare", "准备", "取出", "setup"]):
            return "preparation"
        if any(token in lowered for token in ["mix", "add", "transfer", "pipette", "加入", "转移", "移液", "倾倒"]):
            return "operation"
        if any(token in lowered for token in ["observe", "record", "measure", "观察", "记录", "测量"]):
            return "observation"
        if any(token in lowered for token in ["clean", "dispose", "结束", "清理"]):
            return "cleanup"
        return "execution"

    def match_timeline_steps(
        self,
        protocol_nodes: List[StepGraphNode],
        steps: Iterable[Any],
    ) -> List[Dict[str, Any]]:
        protocol_iter = iter(protocol_nodes)
        current_node = next(protocol_iter, None)
        matched: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            step_name = getattr(step, "step_name", "")
            stage_name = getattr(step, "metadata", {}).get("stage_name")
            if current_node and (current_node.step_name in step_name or step_name in current_node.step_name):
                stage_name = current_node.stage_name
                current_node = next(protocol_iter, None)
            matched.append(
                {
                    "step_id": getattr(step, "step_id", f"step_{idx}"),
                    "step_name": step_name,
                    "stage_name": stage_name or (current_node.stage_name if current_node else "execution"),
                    "status": getattr(step, "status", "candidate"),
                    "completion_type": "inferred" if getattr(step, "completed_by_inference", False) else "observed",
                    "confidence": getattr(step, "confidence", 0.0),
                }
            )
        return matched
