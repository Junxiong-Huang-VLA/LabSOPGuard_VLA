from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

METADATA_VERSION = "operational_step_governance.v1"


@dataclass
class OfficialStepRecord:
    official_step_id: str
    experiment_id: str
    protocol_step_id: str
    protocol_step_name: str
    source_step_candidate_id: str
    status: str
    linked_event_ids: List[str]
    evidence_bundle: Dict[str, Any]
    created_at: str
    locked: bool
    version: int
    lifecycle_status: str = "approved"
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepReviewDecision:
    review_decision_id: str
    step_candidate_id: str
    decision: str
    rationale: str
    operator: str
    operator_role: str
    created_at: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepRevision:
    revision_id: str
    official_step_id: str
    previous_payload: Dict[str, Any]
    new_payload: Dict[str, Any]
    change_reason: str
    operator: str
    operator_role: str
    created_at: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SopSchemaValidationResult:
    schema_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_schema: Dict[str, Any]
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OfficialStepLifecycleEvent:
    lifecycle_event_id: str
    official_step_id: str
    from_status: str
    to_status: str
    rationale: str
    operator: str
    operator_role: str
    created_at: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepGovernanceDecision:
    governance_decision_id: str
    official_step_id: str
    decision: str
    rationale: str
    operator: str
    operator_role: str
    created_at: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))
