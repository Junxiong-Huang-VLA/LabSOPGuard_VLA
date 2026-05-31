from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

METADATA_VERSION = "step_bridge.v1"


@dataclass
class StepCandidate:
    step_candidate_id: str
    experiment_id: str
    protocol_step_id: str
    protocol_step_name: str
    matched_event_ids: List[str]
    matched_event_types: List[str]
    candidate_score: float
    candidate_status: str
    evidence_grade: str
    review_status: str
    reasoning_summary: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepEvidenceBundle:
    protocol_step_id: str
    linked_event_ids: List[str]
    linked_asset_paths: List[str]
    direction_signals: List[Dict[str, Any]]
    state_signals: List[Dict[str, Any]]
    track_quality_summary: Dict[str, Any]
    evidence_grade: str
    evidence_summary: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepPromotionDecision:
    protocol_step_id: str
    decision: str
    score: float
    rationale: str
    blocking_issues: List[str]
    recommendation: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProtocolStepNode:
    protocol_step_id: str
    protocol_step_name: str
    step_index: int
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)
    required_event_types: List[str] = field(default_factory=list)
    optional_event_types: List[str] = field(default_factory=list)
    critical_fields: List[str] = field(default_factory=list)
    promotion_rules: Dict[str, Any] = field(default_factory=dict)
    blocking_conditions: List[str] = field(default_factory=list)
    event_reuse_policy: str = "prefer_unique"
    order_constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_json(path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
