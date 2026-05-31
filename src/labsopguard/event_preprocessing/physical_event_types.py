from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class GateStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


class EventType(str, Enum):
    HAND_OBJECT_INTERACTION = "hand_object_interaction"
    HAND_OBJECT_CONTACT = "hand_object_contact"
    OBJECT_MOVE = "object_move"
    LIQUID_TRANSFER = "liquid_transfer"
    PANEL_OPERATION = "panel_operation"
    CONTAINER_STATE_CHANGE = "container_state_change"


@dataclass
class HardGate:
    passed: bool
    gate_name: str
    required_evidence: List[str] = field(default_factory=list)
    passed_evidence: List[str] = field(default_factory=list)
    failed_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateDecision:
    status: str
    event_type: str
    confidence: float
    hard_gate: HardGate | Dict[str, Any]
    evidence: Dict[str, Any] = field(default_factory=dict)
    reject_reasons: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if isinstance(self.hard_gate, HardGate):
            data["hard_gate"] = self.hard_gate.to_dict()
        return data


@dataclass
class TrackEvidence:
    track_id: str
    track_type: str
    object_label: str
    source_view: str = ""
    point_count: int = 0
    identity_confidence: float = 0.0
    id_switch_risk: float = 1.0
    median_bbox_size: float = 0.0
    raw_displacement_px: float = 0.0
    path_length_px: float = 0.0
    stabilized_displacement_px: Optional[float] = None
    motion_persistent: bool = False
    can_confirm_motion: bool = True
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneMotionEvidence:
    is_camera_motion: bool = False
    is_scene_cut: bool = False
    background_shift_px: float = 0.0
    homography_confidence: float = 0.0
    global_motion_ratio: float = 0.0
    method: str = "none"
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class JitterProfile:
    object_label: str
    sigma_px: float
    source: str = "fallback"
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventCandidate:
    event_type: str
    time_start: float
    time_end: float
    source_view: str = ""
    actor_track_id: Optional[str] = None
    object_track_ids: List[str] = field(default_factory=list)
    raw_evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    candidate_source: str = "yolo"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

