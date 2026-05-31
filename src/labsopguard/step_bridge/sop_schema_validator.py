from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Set

from labsopguard.step_review.schemas import SopSchemaValidationResult, write_json

VALID_EVENT_TYPES: Set[str] = {
    "hand_object_interaction",
    "object_move",
    "liquid_transfer",
    "panel_operation",
    "container_state_change",
}
VALID_REUSE_POLICIES = {"prefer_unique", "unique", "allow_reuse"}
VALID_BLOCKING_CONDITIONS = {
    "direction_unknown",
    "missing_required_event",
    "track_quality_too_low",
    "predecessor_not_confirmed",
    "state_unknown",
}
VALID_PROMOTION_RULE_KEYS = {"min_score", "min_evidence_grade", "require_confirmed_direction", "require_state_change"}


class SopSchemaValidator:
    def validate(self, steps: List[Dict[str, Any]], *, output_dir: str | Path | None = None, fail_fast: bool = False) -> SopSchemaValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        normalized_steps: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        for index, step in enumerate(steps):
            step_id = str(step.get("step_id") or step.get("protocol_step_id") or f"protocol_step_{index:03d}")
            if step_id in seen_ids:
                errors.append(f"duplicate_step_id:{step_id}")
            seen_ids.add(step_id)
            normalized = dict(step)
            normalized["step_id"] = step_id
            normalized.setdefault("step_name", step.get("protocol_step_name") or f"Protocol step {index + 1}")
            normalized.setdefault("step_index", int(step.get("step_index", index)))
            normalized_steps.append(normalized)

        known_ids = {str(step["step_id"]) for step in normalized_steps}
        for step in normalized_steps:
            schema = self._explicit_schema(step)
            for field in ("predecessors", "successors"):
                for ref in self._list(schema.get(field)):
                    if ref not in known_ids:
                        errors.append(f"{field[:-1]}_not_found:{step['step_id']}->{ref}")
            for field in ("required_event_types", "optional_event_types"):
                for event_type in self._list(schema.get(field)):
                    if event_type not in VALID_EVENT_TYPES:
                        errors.append(f"invalid_{field}:{step['step_id']}:{event_type}")
            policy = schema.get("event_reuse_policy")
            if policy is not None and str(policy) not in VALID_REUSE_POLICIES:
                errors.append(f"invalid_event_reuse_policy:{step['step_id']}:{policy}")
            for condition in self._list(schema.get("blocking_conditions")):
                if condition not in VALID_BLOCKING_CONDITIONS:
                    warnings.append(f"unknown_blocking_condition:{step['step_id']}:{condition}")
            promotion = schema.get("promotion_rules") or {}
            if promotion and not isinstance(promotion, dict):
                errors.append(f"invalid_promotion_rules:{step['step_id']}:not_object")
            else:
                for key in promotion.keys():
                    if key not in VALID_PROMOTION_RULE_KEYS:
                        warnings.append(f"unknown_promotion_rule:{step['step_id']}:{key}")
                min_score = promotion.get("min_score") if isinstance(promotion, dict) else None
                if min_score is not None and not 0.0 <= float(min_score) <= 1.0:
                    errors.append(f"invalid_promotion_rules:{step['step_id']}:min_score_out_of_range")

        self._check_graph(normalized_steps, errors, warnings)
        result = SopSchemaValidationResult(
            schema_id="sop_schema_" + hashlib.sha1(str([(s.get("step_id"), s.get("step_index")) for s in normalized_steps]).encode("utf-8")).hexdigest()[:12],
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            normalized_schema={"steps": normalized_steps},
        )
        if output_dir is not None:
            write_json(Path(output_dir) / "sop_schema_validation.json", result.to_dict())
        if fail_fast and errors:
            raise ValueError("; ".join(errors))
        return result

    @staticmethod
    def _explicit_schema(step: Dict[str, Any]) -> Dict[str, Any]:
        schema = step.get("sop_schema") or step.get("protocol_schema") or step.get("step_schema") or {}
        explicit = dict(schema) if isinstance(schema, dict) else {}
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

    @staticmethod
    def _list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value if item not in (None, "")]
        return []

    def _check_graph(self, steps: List[Dict[str, Any]], errors: List[str], warnings: List[str]) -> None:
        if not steps:
            errors.append("empty_sop_schema")
            return
        ids = [str(step["step_id"]) for step in steps]
        edges: Dict[str, List[str]] = {step_id: [] for step_id in ids}
        explicit_edge_count = 0
        for idx, step in enumerate(steps):
            schema = self._explicit_schema(step)
            successors = self._list(schema.get("successors"))
            predecessors = self._list(schema.get("predecessors"))
            if not successors and idx + 1 < len(steps):
                successors = [str(steps[idx + 1]["step_id"])]
            if successors or predecessors:
                explicit_edge_count += len(successors) + len(predecessors)
            edges[str(step["step_id"])].extend([s for s in successors if s in edges])
            for predecessor in predecessors:
                if predecessor in edges and str(step["step_id"]) not in edges[predecessor]:
                    warnings.append(f"predecessor_successor_inferred:{predecessor}->{step['step_id']}")
                    edges[predecessor].append(str(step["step_id"]))
        if len(steps) > 1 and explicit_edge_count == 0:
            warnings.append("no_explicit_edges_using_linear_fallback")
        visited: Set[str] = set()
        stack: Set[str] = set()

        def visit(node: str) -> None:
            if node in stack:
                errors.append(f"cycle_detected:{node}")
                return
            if node in visited:
                return
            stack.add(node)
            for child in edges.get(node, []):
                visit(child)
            stack.remove(node)
            visited.add(node)

        for node in ids:
            visit(node)
        connected = {ids[0]}
        changed = True
        while changed:
            changed = False
            for parent, children in edges.items():
                if parent in connected:
                    for child in children:
                        if child not in connected:
                            connected.add(child)
                            changed = True
                if any(child in connected for child in children) and parent not in connected:
                    connected.add(parent)
                    changed = True
        for step_id in ids:
            if step_id not in connected:
                warnings.append(f"isolated_step:{step_id}")
