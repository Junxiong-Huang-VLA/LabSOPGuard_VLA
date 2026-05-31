from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MonitoringStatus:
    completed_steps: List[str]
    pending_steps: List[str]
    compliance_ratio: float


@dataclass
class MonitoringResult:
    sample_id: str
    sop_id: str
    status: MonitoringStatus
    violations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ActionPlan:
    action_sequence: List[str]
    grasp_point_xyz: Optional[List[float]] = None
    end_effector_target_xyz: Optional[List[float]] = None
    robot_command: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionResult:
    target_name: str
    bbox: List[int]
    center_point: List[float]
    depth_info: Dict[str, float]
    confidence: float
    segmentation: Optional[List[List[float]]] = None
    region_reference: Optional[Dict[str, Any]] = None
    xyz: Optional[List[float]] = None
    target_representation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VLAResult:
    sample_id: str
    instruction: str
    perception: PerceptionResult
    action: ActionPlan
