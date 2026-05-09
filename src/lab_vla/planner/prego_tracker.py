"""
PREGO-Style Program Step Prediction & Anomaly Detection for LabSOPGuard.

PREGO (Program Recognition for Experimental General Operations) framework:
- Maintains a procedural state machine for the current experiment
- Predicts the next expected step based on observed actions and current state
- Detects anomalies: skipped steps, out-of-order operations, missing prerequisites
- Generates structured explanations for detected issues

Works without API calls using structured rule-based reasoning.
When VLM/LLM is available, can enhance predictions with semantic understanding.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from integrated_system.scene_understander import (
    ExperimentType,
    EXPERIMENT_STEP_TEMPLATES,
    DEFAULT_STEP_TEMPLATE,
)


class AnomalyType(str, Enum):
    STEP_SKIPPED = "step_skipped"
    STEP_OUT_OF_ORDER = "step_out_of_order"
    PREREQUISITE_MISSING = "prerequisite_missing"
    REDUNDANT_OPERATION = "redundant_operation"
    DANGEROUS_SEQUENCE = "dangerous_sequence"
    EXTRA_STEP = "extra_step"
    NO_PROGRESS = "no_progress"
    TIMEOUT = "timeout"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ProceduralStep:
    """A step in the experiment procedure."""
    step_id: str
    name_zh: str
    name_en: str
    order: int
    prerequisites: List[str] = field(default_factory=list)
    expected_actions: List[str] = field(default_factory=list)
    forbidden_before: List[str] = field(default_factory=list)  # Cannot happen before these
    timeout_sec: float = 0.0  # 0 = no timeout


@dataclass
class ObservedEvent:
    """An observed action/event in the experiment."""
    timestamp_sec: float
    frame_id: int
    action_type: str  # From LabAction or operation_type
    step_id: str  # Which step this maps to (if any)
    confidence: float
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """A detected procedural anomaly."""
    anomaly_type: AnomalyType
    severity: Severity
    description_zh: str
    description_en: str
    timestamp_sec: float
    frame_id: int
    related_step: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation_zh: str = ""
    recommendation_en: str = ""


@dataclass
class PREGOState:
    """Current state of the PREGO procedural tracker."""
    experiment_type: str
    current_step_index: int = 0
    completed_steps: List[str] = field(default_factory=list)
    active_step: str = ""
    predicted_next_steps: List[str] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    progress_pct: float = 0.0
    is_complete: bool = False
    last_event_time: float = 0.0


# ---------------------------------------------------------------------------
# Prerequisite Rules Engine
# ---------------------------------------------------------------------------

def _build_prerequisite_map(steps: List[ProceduralStep]) -> Dict[str, ProceduralStep]:
    """Build lookup map from step list."""
    return {s.step_id: s for s in steps}


def _build_step_sequence(steps: List[ProceduralStep]) -> List[str]:
    """Get ordered step ID list."""
    return [s.step_id for s in sorted(steps, key=lambda s: s.order)]


def _get_step_prerequisites(step_id: str, step_map: Dict[str, ProceduralStep]) -> Set[str]:
    """Get all transitive prerequisites for a step."""
    if step_id not in step_map:
        return set()

    step = step_map[step_id]
    all_prereqs = set(step.prerequisites)

    # Transitive: prerequisites of prerequisites
    for prereq in step.prerequisites:
        all_prereqs.update(_get_step_prerequisites(prereq, step_map))

    return all_prereqs


# ---------------------------------------------------------------------------
# PREGO Procedural Tracker
# ---------------------------------------------------------------------------

class PREGOTracker:
    """PREGO-style procedural state tracker for laboratory experiments.

    Tracks observed actions against expected procedure, detects anomalies,
    and predicts next steps.
    """

    def __init__(
        self,
        experiment_type: ExperimentType,
        custom_steps: Optional[List[Dict[str, Any]]] = None,
    ):
        self.experiment_type = experiment_type

        # Build procedural steps
        if custom_steps:
            self.steps = self._build_steps_from_dict(custom_steps)
        else:
            template = EXPERIMENT_STEP_TEMPLATES.get(experiment_type, DEFAULT_STEP_TEMPLATE)
            self.steps = self._build_steps_from_template(template)

        self.step_map = _build_prerequisite_map(self.steps)
        self.sequence = _build_step_sequence(self.steps)

        # State
        self.state = PREGOState(experiment_type=experiment_type.value)
        self._events: List[ObservedEvent] = []
        self._step_first_seen: Dict[str, float] = {}
        self._step_last_seen: Dict[str, float] = {}

    def _build_steps_from_template(self, template: List[Dict[str, Any]]) -> List[ProceduralStep]:
        """Build ProceduralSteps from EXPERIMENT_STEP_TEMPLATES format."""
        steps = []
        for i, t in enumerate(template):
            step_id = t["step_id"]
            prerequisites = []
            if i > 0:
                prerequisites = [template[i - 1]["step_id"]]

            steps.append(ProceduralStep(
                step_id=step_id,
                name_zh=t.get("name_zh", step_id),
                name_en=t.get("name_en", step_id.replace("_", " ")),
                order=i,
                prerequisites=prerequisites,
                expected_actions=t.get("keywords", []),
                forbidden_before=template[:max(0, i - 1)],
            ))
        return steps

    def _build_steps_from_dict(self, dicts: List[Dict[str, Any]]) -> List[ProceduralStep]:
        """Build ProceduralSteps from custom dict list."""
        steps = []
        for i, d in enumerate(dicts):
            steps.append(ProceduralStep(
                step_id=d.get("step_id", f"step_{i}"),
                name_zh=d.get("name_zh", ""),
                name_en=d.get("name_en", ""),
                order=d.get("order", i),
                prerequisites=d.get("prerequisites", []),
                expected_actions=d.get("expected_actions", []),
                forbidden_before=d.get("forbidden_before", []),
                timeout_sec=d.get("timeout_sec", 0.0),
            ))
        return steps

    def process_event(self, event: ObservedEvent) -> List[Anomaly]:
        """Process an observed event and check for anomalies.

        Returns list of newly detected anomalies.
        """
        self._events.append(event)
        new_anomalies: List[Anomaly] = []

        step_id = event.step_id
        if not step_id or step_id == "unknown":
            return new_anomalies

        # Track step timing
        if step_id not in self._step_first_seen:
            self._step_first_seen[step_id] = event.timestamp_sec
        self._step_last_seen[step_id] = event.timestamp_sec
        self.state.last_event_time = event.timestamp_sec

        # Check 1: Step out of order
        if step_id in self.sequence:
            step_idx = self.sequence.index(step_id)
            expected_idx = self.state.current_step_index

            if step_idx < expected_idx - 1:
                # Already completed, redundant
                if step_id not in self.state.completed_steps:
                    anomaly = Anomaly(
                        anomaly_type=AnomalyType.STEP_OUT_OF_ORDER,
                        severity=Severity.MEDIUM,
                        description_zh=f"步骤顺序异常：'{self.step_map[step_id].name_zh}' 出现在预期之前",
                        description_en=f"Step out of order: '{self.step_map[step_id].name_en}' occurred earlier than expected",
                        timestamp_sec=event.timestamp_sec,
                        frame_id=event.frame_id,
                        related_step=step_id,
                        evidence={"expected_index": expected_idx, "actual_index": step_idx},
                        recommendation_zh=f"请确认是否跳过了中间步骤",
                        recommendation_en=f"Verify if intermediate steps were skipped",
                    )
                    new_anomalies.append(anomaly)

        # Check 2: Prerequisites not met
        if step_id in self.step_map:
            prereqs = _get_step_prerequisites(step_id, self.step_map)
            missing_prereqs = prereqs - set(self.state.completed_steps)

            if missing_prereqs and step_id not in self.state.completed_steps:
                missing_names = [
                    self.step_map[p].name_zh for p in missing_prereqs if p in self.step_map
                ]
                anomaly = Anomaly(
                    anomaly_type=AnomalyType.PREREQUISITE_MISSING,
                    severity=Severity.HIGH,
                    description_zh=f"'{self.step_map[step_id].name_zh}' 的前置步骤未完成：{', '.join(missing_names)}",
                    description_en=f"Prerequisites for '{self.step_map[step_id].name_en}' not met: {', '.join(missing_names)}",
                    timestamp_sec=event.timestamp_sec,
                    frame_id=event.frame_id,
                    related_step=step_id,
                    evidence={"missing_prereqs": list(missing_prereqs), "completed": self.state.completed_steps},
                    recommendation_zh=f"请先完成前置步骤：{', '.join(missing_names)}",
                    recommendation_en=f"Complete prerequisites first: {', '.join(missing_names)}",
                )
                new_anomalies.append(anomaly)

        # Check 3: Forbidden-before violation
        if step_id in self.step_map:
            step = self.step_map[step_id]
            for forbidden in step.forbidden_before:
                if forbidden not in self.state.completed_steps and forbidden in self._step_first_seen:
                    anomaly = Anomaly(
                        anomaly_type=AnomalyType.DANGEROUS_SEQUENCE,
                        severity=Severity.CRITICAL,
                        description_zh=f"危险操作序列：'{step.name_zh}' 在 '{self.step_map[forbidden].name_zh}' 之前执行",
                        description_en=f"Dangerous sequence: '{step.name_en}' performed before '{self.step_map[forbidden].name_en}'",
                        timestamp_sec=event.timestamp_sec,
                        frame_id=event.frame_id,
                        related_step=step_id,
                        evidence={"forbidden_step": forbidden},
                        recommendation_zh=f"立即停止操作，按照正确顺序执行",
                        recommendation_en=f"Stop immediately, follow correct procedure order",
                    )
                    new_anomalies.append(anomaly)

        # Update state
        if step_id not in self.state.completed_steps:
            self.state.completed_steps.append(step_id)

            # Advance current step index
            if step_id in self.sequence:
                step_idx = self.sequence.index(step_id)
                if step_idx >= self.state.current_step_index:
                    self.state.current_step_index = step_idx + 1

        # Update active step and predictions
        if self.state.current_step_index < len(self.sequence):
            self.state.active_step = self.sequence[self.state.current_step_index]
            remaining = self.sequence[self.state.current_step_index:]
            self.state.predicted_next_steps = remaining[:3]  # Top 3 predictions
        else:
            self.state.active_step = ""
            self.state.predicted_next_steps = []
            self.state.is_complete = True

        self.state.progress_pct = len(self.state.completed_steps) / max(len(self.sequence), 1) * 100

        # Add anomalies to state
        self.state.anomalies.extend(new_anomalies)

        return new_anomalies

    def check_timeouts(self, current_time: float) -> List[Anomaly]:
        """Check for step timeouts."""
        anomalies = []
        for step_id, first_seen in self._step_first_seen.items():
            if step_id not in self.step_map:
                continue
            step = self.step_map[step_id]
            if step.timeout_sec > 0 and step_id not in self.state.completed_steps:
                elapsed = current_time - first_seen
                if elapsed > step.timeout_sec:
                    anomalies.append(Anomaly(
                        anomaly_type=AnomalyType.TIMEOUT,
                        severity=Severity.MEDIUM,
                        description_zh=f"步骤超时：'{step.name_zh}' 已耗时 {elapsed:.0f}秒（限制 {step.timeout_sec:.0f}秒）",
                        description_en=f"Step timeout: '{step.name_en}' took {elapsed:.0f}s (limit {step.timeout_sec:.0f}s)",
                        timestamp_sec=current_time,
                        frame_id=-1,
                        related_step=step_id,
                        evidence={"elapsed_sec": elapsed, "limit_sec": step.timeout_sec},
                    ))
        return anomalies

    def check_missing_steps(self, final_time: float) -> List[Anomaly]:
        """Check for missing steps at experiment end."""
        anomalies = []
        for step_id in self.sequence:
            if step_id not in self.state.completed_steps:
                step = self.step_map[step_id]
                anomalies.append(Anomaly(
                    anomaly_type=AnomalyType.STEP_SKIPPED,
                    severity=Severity.HIGH,
                    description_zh=f"缺失步骤：'{step.name_zh}' 未被观察到",
                    description_en=f"Missing step: '{step.name_en}' was not observed",
                    timestamp_sec=final_time,
                    frame_id=-1,
                    related_step=step_id,
                    evidence={"expected_step": step_id, "completed_steps": self.state.completed_steps},
                    recommendation_zh=f"请补充执行步骤：{step.name_zh}",
                    recommendation_en=f"Please perform the missing step: {step.name_en}",
                ))
        return anomalies

    def get_state(self) -> PREGOState:
        """Get current procedural state."""
        return self.state

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of PREGO analysis."""
        return {
            "experiment_type": self.state.experiment_type,
            "total_steps": len(self.sequence),
            "completed_steps": len(self.state.completed_steps),
            "progress_pct": self.state.progress_pct,
            "is_complete": self.state.is_complete,
            "active_step": self.state.active_step,
            "predicted_next": self.state.predicted_next_steps,
            "anomaly_count": len(self.state.anomalies),
            "anomalies_by_severity": {
                "critical": len([a for a in self.state.anomalies if a.severity == Severity.CRITICAL]),
                "high": len([a for a in self.state.anomalies if a.severity == Severity.HIGH]),
                "medium": len([a for a in self.state.anomalies if a.severity == Severity.MEDIUM]),
                "low": len([a for a in self.state.anomalies if a.severity == Severity.LOW]),
            },
            "completed_step_ids": self.state.completed_steps,
            "missing_step_ids": [s for s in self.sequence if s not in self.state.completed_steps],
        }

    def to_json(self, path: str | Path) -> None:
        """Save PREGO state to JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = self.get_summary()
        # Convert anomalies to dicts
        data["anomalies"] = [
            {
                "type": a.anomaly_type.value,
                "severity": a.severity.value,
                "description_zh": a.description_zh,
                "description_en": a.description_en,
                "timestamp": a.timestamp_sec,
                "frame_id": a.frame_id,
                "related_step": a.related_step,
                "recommendation_zh": a.recommendation_zh,
                "recommendation_en": a.recommendation_en,
            }
            for a in self.state.anomalies
        ]

        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Integration with Chemistry Observations
# ---------------------------------------------------------------------------

def run_prego_analysis(
    experiment_type: ExperimentType,
    chemistry_observations: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Run PREGO analysis on chemistry observations.

    Args:
        experiment_type: Detected experiment type
        chemistry_observations: List of ChemistryObservation dicts
        output_dir: Output directory for prego_result.json

    Returns:
        PREGO summary dict
    """
    tracker = PREGOTracker(experiment_type)

    for obs in chemistry_observations:
        step_id = obs.get("expected_step_id", "")
        if not step_id or step_id == "unknown":
            continue

        event = ObservedEvent(
            timestamp_sec=obs.get("timestamp_sec", 0.0),
            frame_id=obs.get("frame_index", 0),
            action_type=obs.get("operation_type", "unknown"),
            step_id=step_id,
            confidence=obs.get("confidence", 0.5),
            description=obs.get("operation_description_en", ""),
        )

        tracker.process_event(event)

    # Final check for missing steps
    if chemistry_observations:
        last_time = max(o.get("timestamp_sec", 0.0) for o in chemistry_observations)
        missing = tracker.check_missing_steps(last_time)
        tracker.state.anomalies.extend(missing)

    # Save result
    result_path = output_dir / "prego_result.json"
    tracker.to_json(result_path)

    return tracker.get_summary()
