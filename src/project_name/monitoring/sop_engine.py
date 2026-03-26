from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from project_name.detection.multi_level_detector import DetectionEvent


@dataclass
class ViolationEvent:
    frame_id: int
    timestamp_sec: float
    rule_id: str
    severity: str
    message: str


class SOPComplianceEngine:
    """Rule-based compliance tracker for SOP events."""

    def __init__(self, rules: Dict[str, Any]) -> None:
        self.rules = rules
        self.step_state: Dict[str, bool] = {
            "wear_gloves": False,
            "wear_goggles": False,
            "verify_label": False,
            "pipette_transfer": False,
            "cap_container": False,
            "dispose_waste": False,
        }

    def update(self, detection: DetectionEvent) -> List[ViolationEvent]:
        self.step_state["wear_gloves"] = bool(detection.ppe.get("wear_gloves", False))
        self.step_state["wear_goggles"] = bool(detection.ppe.get("wear_goggles", False))

        for step in detection.actions:
            self.step_state[step] = True

        violations: List[ViolationEvent] = []

        if not (self.step_state["wear_gloves"] and self.step_state["wear_goggles"]):
            violations.append(
                ViolationEvent(
                    frame_id=detection.frame_id,
                    timestamp_sec=detection.timestamp_sec,
                    rule_id="missing_ppe",
                    severity="high",
                    message="Gloves and goggles are both required.",
                )
            )

        if self.step_state["pipette_transfer"] and not self.step_state["verify_label"]:
            violations.append(
                ViolationEvent(
                    frame_id=detection.frame_id,
                    timestamp_sec=detection.timestamp_sec,
                    rule_id="unsafe_transfer",
                    severity="high",
                    message="Pipette transfer before label verification.",
                )
            )

        return violations

    def build_status(self) -> Dict[str, Any]:
        completed = [k for k, v in self.step_state.items() if v]
        pending = [k for k, v in self.step_state.items() if not v]
        return {
            "completed_steps": completed,
            "pending_steps": pending,
            "compliance_ratio": float(len(completed) / max(len(self.step_state), 1)),
        }
