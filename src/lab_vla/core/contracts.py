from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskCommand:
    task_id: str
    instruction: str
    target_object: str
    source_zone: str
    target_zone: str
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionObject:
    label: str
    bbox: List[int]
    score: float


@dataclass
class PerceptionPacket:
    frame_id: int
    timestamp_sec: float
    ppe: Dict[str, bool]
    objects: List[DetectionObject]
    actions: List[str]
    confidence: float
    layer_outputs: Dict[str, Any]


@dataclass
class SceneState:
    sample_id: str
    frame_id: int
    timestamp_sec: float
    target_object: str
    target_bbox: Optional[List[int]]
    target_xyz_m: Optional[List[float]]
    ppe_ok: bool
    object_labels: List[str]
    action_hints: List[str]
    confidence: float
    layer_outputs: Dict[str, Any]


@dataclass
class SkillStep:
    step_id: str
    skill_name: str
    command: str
    args: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)


@dataclass
class SkillPlan:
    plan_id: str
    task_id: str
    steps: List[SkillStep] = field(default_factory=list)
    planner_backend: str = "deterministic"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyDecision:
    safe_to_execute: bool
    reason: str
    violations: List[str] = field(default_factory=list)
    recovery_plan: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    status: str
    adapter: str
    command: Dict[str, Any]
    verification: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class RuntimeMetrics:
    frames_total: int = 0
    frames_with_target: int = 0
    plans_total: int = 0
    safety_blocks: int = 0
    executions_ok: int = 0
    executions_failed: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "frames_total": self.frames_total,
            "frames_with_target": self.frames_with_target,
            "plans_total": self.plans_total,
            "safety_blocks": self.safety_blocks,
            "executions_ok": self.executions_ok,
            "executions_failed": self.executions_failed,
        }
