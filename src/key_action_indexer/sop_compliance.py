from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_package import EvidencePackage
from .physical_event_types import EventType
from .schemas import read_jsonl, write_jsonl
from .sop_state_machine import build_sop_state_machine


SCHEMA_VERSION = "sop_compliance_report.v1"
EVENT_SCHEMA_VERSION = "sop_compliance_event.v1"


class SOPSeverity(str, Enum):
    CRITICAL = "Critical"
    MAJOR = "Major"
    MINOR = "Minor"


class SOPReviewStatus(str, Enum):
    AUTO_CONFIRMED = "auto_confirmed"
    NEEDS_REVIEW = "needs_review"
    HUMAN_CONFIRMED = "human_confirmed"
    HUMAN_REJECTED = "human_rejected"


@dataclass
class SOPComplianceEvent:
    sop_ref: dict[str, Any]
    severity: str
    description: str
    recommendation: str
    evidence_refs: list[dict[str, Any]]
    confidence: float
    review_status: str
    event_id: str = ""
    schema_version: str = EVENT_SCHEMA_VERSION
    compliance_status: str = "needs_review"
    source_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 4)
        return data


@dataclass
class SOPObservation:
    source_kind: str
    row: dict[str, Any]
    tokens: set[str]
    evidence_ref: dict[str, Any]
    confidence: float


ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "weighing": ("weigh", "weighing", "balance", "mass", "scale", "measure_mass"),
    "pipetting": ("pipette", "pipetting", "liquid_transfer", "transfer", "add_sample", "add_liquid", "dose"),
    "sample_handling": ("sample", "tube", "vial", "bottle", "container", "hand_object_interaction", "object_move"),
    "recording": ("record", "recording", "readout", "note", "panel_operation", "display"),
    "mixing": ("mix", "mixing", "vortex", "shake", "stir"),
    "incubating": ("incubate", "incubating", "wait", "hold", "settle"),
    "cleanup": ("clean", "cleanup", "discard", "dispose", "rinse", "waste"),
    EventType.HAND_OBJECT_INTERACTION.value: ("hand", "contact", "grasp", "hand_object_contact"),
    EventType.HAND_OBJECT_CONTACT.value: ("hand", "contact", "grasp", "hand_object_interaction"),
    EventType.OBJECT_MOVE.value: ("move", "movement", "object_movement", "relocate", "return"),
    EventType.LIQUID_TRANSFER.value: ("pipette", "pour", "transfer", "liquid", "add_liquid"),
    EventType.PANEL_OPERATION.value: ("panel", "button", "knob", "display", "readout"),
    EventType.CONTAINER_STATE_CHANGE.value: ("open", "close", "cap", "lid", "container"),
}

HIGH_RISK_ACTIONS = {
    "pipetting",
    EventType.LIQUID_TRANSFER.value,
    EventType.CONTAINER_STATE_CHANGE.value,
}


def build_sop_compliance_report(
    sop_source: str | Path | Mapping[str, Any] | Sequence[Any] | None,
    *,
    key_actions: Any = None,
    evidence_refs: Any = None,
    physical_changes: Any = None,
    package_dir: str | Path | None = None,
    min_confidence: float = 0.5,
    include_unmapped_observations: bool = True,
    output_path: str | Path | None = None,
    events_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Map local key-action/evidence rows to SOP compliance events.

    This module is intentionally independent from LabSOPGuard. It accepts local
    JSON/JSONL files or in-memory rows and does not call external services.
    """

    state_machine = _coerce_sop_state_machine(sop_source)
    package_refs: list[dict[str, Any]] = []
    package_changes: list[dict[str, Any]] = []
    package_id = ""
    if package_dir is not None:
        package = EvidencePackage.load(package_dir)
        package_refs = list(package.references)
        package_changes = list(package.physical_changes)
        package_id = package.package_id

    evidence_rows = _coerce_rows(evidence_refs)
    physical_change_rows = _coerce_rows(physical_changes)
    if package_refs:
        evidence_rows.extend(package_refs)
    if package_changes:
        physical_change_rows.extend(package_changes)

    events = map_sop_compliance_events(
        state_machine,
        key_action_rows=_coerce_rows(key_actions),
        evidence_rows=evidence_rows,
        physical_change_rows=physical_change_rows,
        min_confidence=min_confidence,
        include_unmapped_observations=include_unmapped_observations,
    )
    event_rows = [event.to_dict() for event in events]
    coverage = _coverage_summary(state_machine, event_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "package_id": package_id,
        "backend_required": False,
        "external_services_required": False,
        "dry_run_compatible": True,
        "min_confidence": float(min_confidence),
        "sop": {
            "schema_version": state_machine.get("schema_version"),
            "source": state_machine.get("source"),
            "step_count": len(_sop_steps(state_machine)),
        },
        "source_counts": {
            "key_actions": len(_coerce_rows(key_actions)),
            "evidence_refs": len(evidence_rows),
            "physical_changes": len(physical_change_rows),
        },
        "coverage": coverage,
        "event_count": len(event_rows),
        "events": event_rows,
    }
    if output_path is not None:
        _write_json(output_path, report)
    if events_output_path is not None:
        write_jsonl(events_output_path, event_rows)
    return report


def map_sop_compliance_events(
    sop_state_machine: Mapping[str, Any],
    *,
    key_action_rows: Sequence[Mapping[str, Any]] | None = None,
    evidence_rows: Sequence[Mapping[str, Any]] | None = None,
    physical_change_rows: Sequence[Mapping[str, Any]] | None = None,
    min_confidence: float = 0.5,
    include_unmapped_observations: bool = True,
) -> list[SOPComplianceEvent]:
    steps = _sop_steps(sop_state_machine)
    observations = _build_observations(
        key_action_rows=key_action_rows or [],
        evidence_rows=evidence_rows or [],
        physical_change_rows=physical_change_rows or [],
    )
    used_observation_ids: set[str] = set()
    events: list[SOPComplianceEvent] = []

    for step in steps:
        matches = sorted(
            (_match_step_to_observation(step, observation) for observation in observations),
            key=lambda item: (item["score"], item["confidence"]),
            reverse=True,
        )
        best = matches[0] if matches and matches[0]["score"] > 0 else None
        if best is None:
            events.append(_missing_step_event(step))
            continue

        observation = best["observation"]
        used_observation_ids.add(str(observation.evidence_ref.get("ref_id")))
        confidence = float(best["confidence"])
        if confidence < min_confidence:
            events.append(_low_confidence_event(step, observation, confidence, best["score"], min_confidence))
        else:
            events.append(_mapped_step_event(step, observation, confidence, best["score"]))

    if include_unmapped_observations:
        for observation in observations:
            ref_id = str(observation.evidence_ref.get("ref_id"))
            if ref_id in used_observation_ids:
                continue
            events.append(_unmapped_observation_event(observation))

    return events


def _coerce_sop_state_machine(source: str | Path | Mapping[str, Any] | Sequence[Any] | None) -> dict[str, Any]:
    if isinstance(source, Mapping):
        data = dict(source)
        if str(data.get("schema_version") or "").startswith("sop_state_machine") and isinstance(data.get("steps"), list):
            return data
        machine = build_sop_state_machine(source)
        return _merge_step_compliance_hints(machine, data.get("steps"))
    if isinstance(source, list):
        machine = build_sop_state_machine(source)
        return _merge_step_compliance_hints(machine, source)
    return build_sop_state_machine(source)


def _merge_step_compliance_hints(machine: dict[str, Any], raw_steps: Any) -> dict[str, Any]:
    if not isinstance(raw_steps, list):
        return machine
    hints_by_id: dict[str, dict[str, Any]] = {}
    hints_by_index: dict[int, dict[str, Any]] = {}
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            continue
        hints = {
            key: raw_step.get(key)
            for key in ("severity", "compliance_severity", "critical", "required", "review_status")
            if raw_step.get(key) is not None
        }
        if not hints:
            continue
        step_id = str(raw_step.get("step_id") or raw_step.get("id") or "")
        if step_id:
            hints_by_id[step_id] = hints
        hints_by_index[index] = hints
    for index, step in enumerate(_sop_steps(machine)):
        hints = hints_by_id.get(str(step.get("step_id") or "")) or hints_by_index.get(index) or {}
        step.update(hints)
        machine["steps"][index] = step
    return machine


def _sop_steps(state_machine: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(step) for step in state_machine.get("steps", []) if isinstance(step, Mapping)]


def _build_observations(
    *,
    key_action_rows: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    physical_change_rows: Sequence[Mapping[str, Any]],
) -> list[SOPObservation]:
    observations: list[SOPObservation] = []
    for source_kind, rows in (
        ("key_action", key_action_rows),
        ("evidence_ref", evidence_rows),
        ("physical_change", physical_change_rows),
    ):
        for index, row in enumerate(rows, start=1):
            normalized = _normalize_row(row)
            evidence_ref = _evidence_ref(source_kind, normalized, index)
            observations.append(
                SOPObservation(
                    source_kind=source_kind,
                    row=normalized,
                    tokens=_row_tokens(normalized),
                    evidence_ref=evidence_ref,
                    confidence=_row_confidence(normalized, source_kind),
                )
            )
    return observations


def _match_step_to_observation(step: Mapping[str, Any], observation: SOPObservation) -> dict[str, Any]:
    step_tokens = _step_tokens(step)
    if not step_tokens or not observation.tokens:
        return {"score": 0.0, "confidence": 0.0, "observation": observation}

    expected = _canonical_action(str(step.get("expected_action") or ""))
    obs_actions = _observation_actions(observation)
    score = 0.0
    if expected and expected in obs_actions:
        score += 0.6
    elif expected and _expanded_tokens([expected]) & observation.tokens:
        score += 0.45

    requirement_tokens = _requirement_tokens(step)
    if requirement_tokens & observation.tokens:
        score += 0.25

    overlap = step_tokens & observation.tokens
    if overlap:
        score += min(0.25, len(overlap) / max(8, len(step_tokens)))

    step_objects = _object_tokens(step)
    obs_objects = _object_tokens(observation.row)
    if step_objects and obs_objects and step_objects & obs_objects:
        score += 0.15

    score = min(1.0, score)
    confidence = min(1.0, observation.confidence * (0.55 + 0.45 * score))
    if score < 0.18:
        score = 0.0
        confidence = 0.0
    return {"score": score, "confidence": confidence, "observation": observation}


def _mapped_step_event(step: Mapping[str, Any], observation: SOPObservation, confidence: float, score: float) -> SOPComplianceEvent:
    sop_ref = _sop_ref(step)
    evidence_ref = {**observation.evidence_ref, "match_score": round(score, 4)}
    event_id = _event_id("sop_ok", sop_ref.get("step_id"), evidence_ref.get("ref_id"))
    return SOPComplianceEvent(
        event_id=event_id,
        sop_ref=sop_ref,
        severity=SOPSeverity.MINOR.value,
        description=f"SOP step '{sop_ref.get('step_name')}' is supported by local evidence.",
        recommendation="Retain the mapped evidence references for audit traceability.",
        evidence_refs=[evidence_ref],
        confidence=confidence,
        review_status=SOPReviewStatus.AUTO_CONFIRMED.value,
        compliance_status="compliant",
        source_events=[_source_event_summary(observation, score)],
    )


def _low_confidence_event(
    step: Mapping[str, Any],
    observation: SOPObservation,
    confidence: float,
    score: float,
    min_confidence: float,
) -> SOPComplianceEvent:
    sop_ref = _sop_ref(step)
    evidence_ref = {**observation.evidence_ref, "match_score": round(score, 4)}
    event_id = _event_id("sop_low_conf", sop_ref.get("step_id"), evidence_ref.get("ref_id"))
    return SOPComplianceEvent(
        event_id=event_id,
        sop_ref=sop_ref,
        severity=_declared_or_default_severity(step, default=SOPSeverity.MAJOR),
        description=(
            f"SOP step '{sop_ref.get('step_name')}' has matching evidence, "
            f"but confidence {confidence:.2f} is below threshold {min_confidence:.2f}."
        ),
        recommendation="Review the mapped evidence before accepting this SOP step as complete.",
        evidence_refs=[evidence_ref],
        confidence=confidence,
        review_status=SOPReviewStatus.NEEDS_REVIEW.value,
        compliance_status="low_confidence",
        source_events=[_source_event_summary(observation, score)],
    )


def _missing_step_event(step: Mapping[str, Any]) -> SOPComplianceEvent:
    sop_ref = _sop_ref(step)
    event_id = _event_id("sop_missing", sop_ref.get("step_id"), sop_ref.get("expected_action"))
    return SOPComplianceEvent(
        event_id=event_id,
        sop_ref=sop_ref,
        severity=_missing_severity(step),
        description=f"No local key action, evidence reference, or physical_change_log row mapped to SOP step '{sop_ref.get('step_name')}'.",
        recommendation="Inspect the source video/evidence package and add or confirm evidence for this SOP step.",
        evidence_refs=[],
        confidence=0.0,
        review_status=SOPReviewStatus.NEEDS_REVIEW.value,
        compliance_status="missing_evidence",
    )


def _unmapped_observation_event(observation: SOPObservation) -> SOPComplianceEvent:
    evidence_ref = observation.evidence_ref
    event_id = _event_id("sop_unmapped", evidence_ref.get("ref_id"), evidence_ref.get("event_type"))
    label = evidence_ref.get("event_type") or evidence_ref.get("ref_id") or observation.source_kind
    return SOPComplianceEvent(
        event_id=event_id,
        sop_ref={"type": "unmapped_observation", "step_id": None, "expected_action": None},
        severity=_unmapped_severity(observation),
        description=f"Observation '{label}' did not map to any SOP step.",
        recommendation="Confirm whether this observation is an allowed extra action or should be linked to an SOP step.",
        evidence_refs=[evidence_ref],
        confidence=observation.confidence,
        review_status=SOPReviewStatus.NEEDS_REVIEW.value,
        compliance_status="unexpected_evidence",
        source_events=[_source_event_summary(observation, 0.0)],
    )


def _sop_ref(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "type": "sop_step",
            "step_id": step.get("step_id"),
            "step_order": step.get("order"),
            "step_name": step.get("name") or step.get("description") or step.get("expected_action"),
            "expected_action": step.get("expected_action"),
            "required": bool(step.get("required", True)),
            "source_refs": step.get("source_refs") or [],
        }.items()
        if value not in (None, "", [])
    }


def _source_event_summary(observation: SOPObservation, score: float) -> dict[str, Any]:
    row = observation.row
    return {
        key: value
        for key, value in {
            "source_kind": observation.source_kind,
            "ref_id": observation.evidence_ref.get("ref_id"),
            "event_type": row.get("event_type"),
            "actions": _string_list(row.get("actions") or row.get("secondary_actions")),
            "objects": _string_list(row.get("objects") or row.get("secondary_objects")),
            "confidence": observation.confidence,
            "match_score": round(score, 4),
        }.items()
        if value not in (None, "", [])
    }


def _evidence_ref(source_kind: str, row: Mapping[str, Any], index: int) -> dict[str, Any]:
    ref_id = _first_text(
        row,
        "material_id",
        "change_id",
        "micro_segment_id",
        "segment_id",
        "event_id",
        "video_event_id",
        "id",
    )
    if not ref_id:
        ref_id = _event_id(source_kind, index, row.get("event_type"), row.get("start_sec"), row.get("end_sec"))
    ref = {
        "type": source_kind,
        "ref_id": ref_id,
        "event_type": row.get("event_type") or row.get("action_type") or row.get("canonical_action_type"),
        "segment_id": row.get("segment_id") or row.get("parent_segment_id"),
        "micro_segment_id": row.get("micro_segment_id"),
        "material_id": row.get("material_id"),
        "change_id": row.get("change_id"),
        "start_sec": _float_or_none(row.get("start_sec") or row.get("local_start_sec")),
        "end_sec": _float_or_none(row.get("end_sec") or row.get("local_end_sec")),
        "source_view": row.get("source_view") or row.get("view") or row.get("camera_view"),
        "confidence": _row_confidence(row, source_kind),
    }
    material_ids = _string_list(row.get("evidence_material_ids"))
    if material_ids:
        ref["evidence_material_ids"] = material_ids
    nested_refs = _list_of_mappings(row.get("evidence_refs"))
    if nested_refs:
        ref["nested_evidence_refs"] = nested_refs
    return {key: value for key, value in ref.items() if value not in (None, "", [])}


def _step_tokens(step: Mapping[str, Any]) -> set[str]:
    texts = [
        step.get("step_id"),
        step.get("name"),
        step.get("description"),
        step.get("expected_action"),
        step.get("branch_type"),
    ]
    texts.extend(_flatten_texts(step.get("completion_conditions")))
    texts.extend(_flatten_texts(step.get("evidence_requirements")))
    texts.extend(_flatten_texts(step.get("required_material")))
    return _expanded_tokens(texts)


def _row_tokens(row: Mapping[str, Any]) -> set[str]:
    texts = [
        row.get("event_type"),
        row.get("action_type"),
        row.get("canonical_action_type"),
        row.get("action_name"),
        row.get("summary"),
        row.get("index_text"),
        row.get("searchable_text"),
        row.get("primary_object"),
        row.get("object_label"),
    ]
    for key in ("actions", "secondary_actions", "objects", "secondary_objects", "detected_objects"):
        texts.extend(_flatten_texts(row.get(key)))
    text_description = row.get("text_description")
    if isinstance(text_description, Mapping):
        texts.extend(_flatten_texts(text_description))
    return _expanded_tokens(texts)


def _requirement_tokens(step: Mapping[str, Any]) -> set[str]:
    return _expanded_tokens(_flatten_texts(step.get("completion_conditions")) + _flatten_texts(step.get("evidence_requirements")))


def _observation_actions(observation: SOPObservation) -> set[str]:
    row = observation.row
    values = [
        row.get("event_type"),
        row.get("action_type"),
        row.get("canonical_action_type"),
    ]
    values.extend(_flatten_texts(row.get("actions")))
    text_description = row.get("text_description")
    if isinstance(text_description, Mapping):
        values.append(text_description.get("action_type"))
    return {_canonical_action(token) for token in _expanded_tokens(values)}


def _object_tokens(value: Mapping[str, Any]) -> set[str]:
    texts = []
    for key in (
        "objects",
        "secondary_objects",
        "detected_objects",
        "primary_object",
        "object_label",
        "required_material",
        "required_materials",
    ):
        texts.extend(_flatten_texts(value.get(key)))
    return _expanded_tokens(texts)


def _expanded_tokens(values: Sequence[Any]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        for token in re.findall(r"[a-zA-Z0-9_./:-]+", text):
            normalized = token.strip(" ./:;-").replace("-", "_")
            if normalized:
                tokens.add(normalized)
                tokens.add(_canonical_action(normalized))
    for token in list(tokens):
        aliases = ACTION_ALIASES.get(token, ())
        tokens.update(alias.lower().replace("-", "_") for alias in aliases)
    return {token for token in tokens if token}


def _canonical_action(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text.endswith("_candidate"):
        text = text[: -len("_candidate")]
    direct = {
        "object_movement": EventType.OBJECT_MOVE.value,
        "object_trajectory_movement": EventType.OBJECT_MOVE.value,
        "liquid_movement": EventType.LIQUID_TRANSFER.value,
        "equipment_panel_operation": EventType.PANEL_OPERATION.value,
        "container_interaction": EventType.CONTAINER_STATE_CHANGE.value,
        "hand_spatula": EventType.HAND_OBJECT_INTERACTION.value,
    }
    if text in direct:
        return direct[text]
    if text in ACTION_ALIASES:
        return text
    for action, aliases in ACTION_ALIASES.items():
        alias_set = {action, *(alias.lower().replace("-", "_") for alias in aliases)}
        if text in alias_set:
            return action
    return text


def _missing_severity(step: Mapping[str, Any]) -> str:
    declared = _declared_severity(step)
    if declared:
        return declared
    expected = _canonical_action(str(step.get("expected_action") or ""))
    if bool(step.get("required", True)) and expected in HIGH_RISK_ACTIONS:
        return SOPSeverity.CRITICAL.value
    if bool(step.get("required", True)):
        return SOPSeverity.MAJOR.value
    return SOPSeverity.MINOR.value


def _declared_or_default_severity(step: Mapping[str, Any], *, default: SOPSeverity) -> str:
    return _declared_severity(step) or default.value


def _declared_severity(step: Mapping[str, Any]) -> str:
    raw = str(step.get("severity") or step.get("compliance_severity") or "").strip().lower()
    for severity in SOPSeverity:
        if raw == severity.value.lower():
            return severity.value
    if bool(step.get("critical")):
        return SOPSeverity.CRITICAL.value
    return ""


def _unmapped_severity(observation: SOPObservation) -> str:
    event_type = _canonical_action(str(observation.row.get("event_type") or ""))
    if event_type in HIGH_RISK_ACTIONS and observation.confidence >= 0.8:
        return SOPSeverity.MAJOR.value
    return SOPSeverity.MINOR.value


def _coverage_summary(state_machine: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    step_count = len(_sop_steps(state_machine))
    mapped = sum(1 for event in events if event.get("compliance_status") == "compliant")
    low_confidence = sum(1 for event in events if event.get("compliance_status") == "low_confidence")
    missing = sum(1 for event in events if event.get("compliance_status") == "missing_evidence")
    unmapped = sum(1 for event in events if event.get("compliance_status") == "unexpected_evidence")
    return {
        "step_count": step_count,
        "mapped_step_count": mapped,
        "low_confidence_step_count": low_confidence,
        "missing_step_count": missing,
        "unmapped_observation_count": unmapped,
        "mapped_step_ratio": round(mapped / step_count, 4) if step_count else 0.0,
        "needs_review_count": sum(1 for event in events if event.get("review_status") == SOPReviewStatus.NEEDS_REVIEW.value),
    }


def _coerce_rows(source: Any) -> list[dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        return _rows_from_path(Path(source))
    if isinstance(source, Mapping):
        return _rows_from_payload(source)
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        rows: list[dict[str, Any]] = []
        for item in source:
            if isinstance(item, (str, Path)):
                rows.extend(_rows_from_path(Path(item)))
            elif isinstance(item, Mapping):
                rows.extend(_rows_from_payload(item))
            elif is_dataclass(item):
                rows.append(asdict(item))
        return rows
    if is_dataclass(source):
        return [asdict(source)]
    return []


def _rows_from_path(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"SOP compliance input does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        return [dict(row) for row in read_jsonl(path)]
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return _rows_from_payload(payload)


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in (
        "events",
        "rows",
        "references",
        "physical_changes",
        "segments",
        "micro_segments",
        "evidence",
        "items",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, Mapping)]
    return [dict(payload)]


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    text_description = data.get("text_description")
    if is_dataclass(text_description):
        data["text_description"] = asdict(text_description)
    return data


def _row_confidence(row: Mapping[str, Any], source_kind: str) -> float:
    for key in ("confidence", "final_score", "quality_score", "score", "avg_confidence", "max_interaction_score"):
        value = _float_or_none(row.get(key))
        if value is not None:
            return _clamp(value)
    if source_kind == "physical_change" and (row.get("event_type") or row.get("evidence_material_ids")):
        return 0.65
    if source_kind == "evidence_ref" and (row.get("material_id") or row.get("formal_clip_path") or row.get("stored_file")):
        return 0.6
    return 0.0


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _flatten_texts(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        texts: list[Any] = []
        for item in value.values():
            texts.extend(_flatten_texts(item))
        return texts
    if isinstance(value, (list, tuple, set)):
        texts = []
        for item in value:
            texts.extend(_flatten_texts(item))
        return texts
    return [value]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _flatten_texts(value) if str(item or "").strip()]


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _event_id(prefix: str, *parts: Any) -> str:
    text = "|".join(str(part) for part in parts if part not in (None, ""))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SOPComplianceEvent",
    "SOPReviewStatus",
    "SOPSeverity",
    "build_sop_compliance_report",
    "map_sop_compliance_events",
]
