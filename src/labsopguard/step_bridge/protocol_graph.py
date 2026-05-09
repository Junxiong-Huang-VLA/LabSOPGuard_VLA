from __future__ import annotations

from typing import Any, Dict, List

from .sop_schema_validator import SopSchemaValidator
from .schemas import ProtocolStepNode


def _text(step: Dict[str, Any]) -> str:
    return " ".join(str(step.get(key) or "") for key in ("step_name", "step_description", "notes", "evidence_notes")).lower()


class ProtocolGraphBuilder:
    def __init__(self) -> None:
        self.validator = SopSchemaValidator()

    def build(self, steps: List[Dict[str, Any]], protocol_payload: Dict[str, Any] | None = None) -> List[ProtocolStepNode]:
        validation = self.validator.validate(steps, output_dir=(protocol_payload or {}).get("output_dir") if isinstance(protocol_payload, dict) else None, fail_fast=False)
        if validation.errors and self._has_explicit_schema(steps):
            raise ValueError("Invalid SOP schema: " + "; ".join(validation.errors))
        steps = validation.normalized_schema.get("steps") or steps
        nodes: List[ProtocolStepNode] = []
        for index, step in enumerate(steps):
            explicit = self._explicit_schema(step)
            step_id = str(step.get("step_id") or step.get("protocol_step_id") or f"protocol_step_{index:03d}")
            name = str(step.get("step_name") or step.get("protocol_step_name") or f"Protocol step {index + 1}")
            required, optional, critical = self._constraints(step, explicit)
            nodes.append(
                ProtocolStepNode(
                    protocol_step_id=step_id,
                    protocol_step_name=name,
                    step_index=int(step.get("step_index", index)),
                    predecessor_ids=[str(item) for item in explicit.get("predecessors", [])],
                    successor_ids=[str(item) for item in explicit.get("successors", [])],
                    required_event_types=required,
                    optional_event_types=optional,
                    critical_fields=critical,
                    promotion_rules=dict(explicit.get("promotion_rules") or {}),
                    blocking_conditions=[str(item) for item in explicit.get("blocking_conditions", [])],
                    event_reuse_policy=str(explicit.get("event_reuse_policy") or "prefer_unique"),
                    order_constraints=dict(explicit.get("order_constraints") or {"must_follow_predecessor": True, "allow_inferred_gap": True}),
                )
            )
        nodes.sort(key=lambda item: item.step_index)
        for idx, node in enumerate(nodes):
            if idx > 0 and not node.predecessor_ids:
                node.predecessor_ids.append(nodes[idx - 1].protocol_step_id)
            if idx + 1 < len(nodes) and not node.successor_ids:
                node.successor_ids.append(nodes[idx + 1].protocol_step_id)
        return nodes

    @staticmethod
    def _has_explicit_schema(steps: List[Dict[str, Any]]) -> bool:
        keys = {"required_event_types", "optional_event_types", "critical_fields", "predecessors", "successors", "promotion_rules", "blocking_conditions", "event_reuse_policy", "sop_schema", "protocol_schema", "step_schema"}
        return any(any(key in step for key in keys) for step in steps)

    @staticmethod
    def _explicit_schema(step: Dict[str, Any]) -> Dict[str, Any]:
        schema = step.get("sop_schema") or step.get("protocol_schema") or step.get("step_schema") or {}
        explicit: Dict[str, Any] = dict(schema) if isinstance(schema, dict) else {}
        for key in (
            "required_event_types",
            "optional_event_types",
            "critical_fields",
            "predecessors",
            "successors",
            "promotion_rules",
            "blocking_conditions",
            "order_constraints",
            "event_reuse_policy",
        ):
            if key in step and step.get(key) is not None:
                explicit[key] = step.get(key)
        return explicit

    def _constraints(self, step: Dict[str, Any], explicit: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        if any(key in explicit for key in ("required_event_types", "optional_event_types", "critical_fields")):
            return (
                self._list(explicit.get("required_event_types")),
                self._list(explicit.get("optional_event_types")),
                self._list(explicit.get("critical_fields")),
            )
        return self._infer_constraints(step)

    @staticmethod
    def _list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, "")]
        return []

    @staticmethod
    def _infer_constraints(step: Dict[str, Any]) -> tuple[List[str], List[str], List[str]]:
        text = _text(step)
        required: List[str] = []
        optional: List[str] = []
        critical: List[str] = []
        if any(token in text for token in ["transfer", "pour", "pipette", "dispense", "add", "转移", "倾倒", "加样", "加液"]):
            required.append("liquid_transfer")
            optional.extend(["hand_object_interaction", "object_move"])
            critical.extend(["source_container", "target_container", "direction_status"])
        if any(token in text for token in ["open", "close", "cap", "lid", "container", "开", "关", "盖", "容器"]):
            required.append("container_state_change")
            optional.append("hand_object_interaction")
            critical.extend(["state_change_type", "state_confidence"])
        if any(token in text for token in ["move", "place", "prepare", "pick", "remove", "放置", "移动", "准备", "拿取"]):
            required.append("object_move")
            optional.append("hand_object_interaction")
            critical.append("actor_track_id")
        if any(token in text for token in ["balance", "scale", "screen", "button", "panel", "record", "称量", "面板", "按钮", "记录"]):
            required.append("panel_operation")
            optional.append("hand_object_interaction")
            critical.append("actor_track_id")
        if not required:
            optional.extend(["hand_object_interaction", "object_move", "container_state_change", "liquid_transfer", "panel_operation"])
        return list(dict.fromkeys(required)), list(dict.fromkeys(optional)), list(dict.fromkeys(critical))
