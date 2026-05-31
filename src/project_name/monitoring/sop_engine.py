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

    def __init__(self, rules: Dict[str, Any], cooldown_seconds: float = 0.0) -> None:
        self.rules = rules
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._last_emitted_ts: Dict[str, float] = {}
        self.required_ppe_keys = self._resolve_required_ppe_keys(rules)
        self.step_state: Dict[str, bool] = {}
        self.reset()

    @staticmethod
    def _resolve_required_ppe_keys(rules: Dict[str, Any]) -> List[str]:
        ppe_cfg = rules.get("ppe_requirements", {}) if isinstance(rules, dict) else {}
        must_wear = ppe_cfg.get("must_wear", []) if isinstance(ppe_cfg, dict) else []
        key_map = {
            "gloves": "wear_gloves",
            "goggles": "wear_goggles",
            "lab_coat": "wear_lab_coat",
        }
        resolved = [key_map[str(x).strip().lower()] for x in must_wear if str(x).strip().lower() in key_map]
        return resolved or ["wear_gloves", "wear_goggles"]

    def reset(self) -> None:
        self._last_emitted_ts = {}
        self.step_state: Dict[str, bool] = {
            "wear_gloves": False,
            "wear_goggles": False,
            "verify_label": False,
            "pipette_transfer": False,
            "cap_container": False,
            "dispose_waste": False,
        }

    def _rule_meta(self, rule_id: str, fallback_severity: str, fallback_message: str) -> tuple[str, str]:
        vcfg = self.rules.get("violation_rules", {}) if isinstance(self.rules, dict) else {}
        rule = vcfg.get(rule_id, {}) if isinstance(vcfg, dict) else {}
        severity = str(rule.get("severity", fallback_severity))
        message = str(rule.get("message", fallback_message))
        return severity, message

    def _emit_if_due(
        self,
        violations: List[ViolationEvent],
        detection: DetectionEvent,
        rule_id: str,
        severity: str,
        message: str,
    ) -> None:
        now = float(detection.timestamp_sec)
        last = self._last_emitted_ts.get(rule_id)
        if last is not None and (now - last) < self.cooldown_seconds:
            return
        violations.append(
            ViolationEvent(
                frame_id=detection.frame_id,
                timestamp_sec=detection.timestamp_sec,
                rule_id=rule_id,
                severity=severity,
                message=message,
            )
        )
        self._last_emitted_ts[rule_id] = now

    @staticmethod
    def _has_human_presence(detection: DetectionEvent) -> bool:
        layer1 = detection.layer_outputs.get("layer1_realtime_pose", {})
        pose_instances = int(layer1.get("pose_instances", 0)) if isinstance(layer1, dict) else 0
        if pose_instances > 0:
            return True
        person_labels = {"person", "human", "operator", "worker"}
        labels = {str(o.get("label", "")).strip().lower() for o in detection.objects}
        return bool(labels.intersection(person_labels))

    def _is_ppe_complete(self, detection: DetectionEvent) -> bool:
        ppe = detection.ppe if isinstance(detection.ppe, dict) else {}
        return all(bool(ppe.get(k, False)) for k in self.required_ppe_keys)

    def update(self, detection: DetectionEvent) -> List[ViolationEvent]:
        self.step_state["wear_gloves"] = bool(detection.ppe.get("wear_gloves", False))
        self.step_state["wear_goggles"] = bool(detection.ppe.get("wear_goggles", False))

        for step in detection.actions:
            self.step_state[step] = True

        violations: List[ViolationEvent] = []

        if self._has_human_presence(detection) and not self._is_ppe_complete(detection):
            sev, msg = self._rule_meta(
                rule_id="missing_ppe",
                fallback_severity="high",
                fallback_message="Gloves and goggles are both required.",
            )
            self._emit_if_due(
                violations=violations,
                detection=detection,
                rule_id="missing_ppe",
                severity=sev,
                message=msg,
            )

        if self.step_state["pipette_transfer"] and not self.step_state["verify_label"]:
            sev, msg = self._rule_meta(
                rule_id="reagent_unverified",
                fallback_severity="high",
                fallback_message="Pipette transfer before label verification.",
            )
            self._emit_if_due(
                violations=violations,
                detection=detection,
                rule_id="reagent_unverified",
                severity=sev,
                message=msg,
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
