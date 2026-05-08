from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "key_action_artifacts.v1"


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_type: str
    relative_path: str
    file_format: str
    schema: dict[str, Any]


def _nullable(schema_type: str) -> dict[str, Any]:
    return {"type": [schema_type, "null"]}


def _array_of(schema_type: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": schema_type}}


def _array_of_objects() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "object"}}


VIDEO_UNDERSTANDING_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.video_understanding.v1",
    "title": "Video understanding event",
    "type": "object",
    "required": [
        "video_event_id",
        "event_type",
        "confidence",
        "confidence_reasons",
        "anomaly_flags",
        "asset_refs",
        "payload",
    ],
    "properties": {
        "video_event_id": {"type": "string"},
        "session_id": _nullable("string"),
        "segment_id": _nullable("string"),
        "micro_segment_id": _nullable("string"),
        "event_type": {"type": "string"},
        "global_start_time": _nullable("string"),
        "global_end_time": _nullable("string"),
        "primary_object": _nullable("string"),
        "action_type": _nullable("string"),
        "state_change_types": _array_of("string"),
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "confidence_reasons": _array_of("string"),
        "anomaly_flags": _array_of("string"),
        "asset_refs": _array_of_objects(),
        "evidence_refs": _array_of_objects(),
        "conclusion_status": _nullable("string"),
        "normalized_object": {"type": ["object", "null"]},
        "object_category": _nullable("string"),
        "action_classification": {"type": ["object", "null"]},
        "semantic_description": _nullable("string"),
        "extracted_entities": {"type": ["object", "null"]},
        "text": _nullable("string"),
        "payload": {"type": "object"},
    },
    "additionalProperties": True,
}

MODEL_OBSERVATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.model_observation_events.v1",
    "title": "External model observation event",
    "type": "object",
    "required": [
        "observation_id",
        "source_file",
        "source_type",
        "observation_type",
        "event_type",
        "confirmation_level",
        "confidence",
        "evidence_reasons",
        "limitations",
        "metrics",
        "asset_refs",
        "payload",
    ],
    "properties": {
        "observation_id": {"type": "string"},
        "session_id": _nullable("string"),
        "segment_id": _nullable("string"),
        "micro_segment_id": _nullable("string"),
        "source_file": {"type": "string"},
        "source_type": {"type": "string"},
        "observation_type": {"type": "string"},
        "event_type": {"type": "string"},
        "global_start_time": _nullable("string"),
        "global_end_time": _nullable("string"),
        "start_sec": {"type": ["number", "null"]},
        "end_sec": {"type": ["number", "null"]},
        "view": _nullable("string"),
        "object_label": _nullable("string"),
        "track_id": _nullable("string"),
        "state": _nullable("string"),
        "measurement": {"type": "object"},
        "confirmation_level": {"type": "string"},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "evidence_reasons": _array_of("string"),
        "limitations": _array_of("string"),
        "metrics": {"type": "object"},
        "asset_refs": _array_of_objects(),
        "payload": {"type": "object"},
    },
    "additionalProperties": True,
}

EXPERIMENT_CONTEXT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.experiment_context.v1",
    "title": "Experiment context",
    "type": "object",
    "required": [
        "session_id",
        "purpose",
        "procedure_candidates",
        "materials",
        "parameters",
        "source_counts",
        "fused_context",
        "confidence",
        "gaps",
    ],
    "properties": {
        "session_id": {"type": "string"},
        "purpose": {"type": "string"},
        "procedure_candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["action_type", "score", "source_types"],
                "properties": {
                    "action_type": {"type": "string"},
                    "score": {"type": ["number", "integer"]},
                    "source_types": _array_of("string"),
                },
                "additionalProperties": True,
            },
        },
        "materials": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "score", "source_types"],
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": ["number", "integer"]},
                    "source_types": _array_of("string"),
                },
                "additionalProperties": True,
            },
        },
        "reagents": _array_of_objects(),
        "equipment": _array_of_objects(),
        "parameters": _array_of_objects(),
        "related_records": _array_of_objects(),
        "transition_priors": {"type": "object"},
        "source_counts": {"type": "object"},
        "text_evidence": _array_of_objects(),
        "upload_evidence": _array_of_objects(),
        "ai_evidence": _array_of_objects(),
        "transcript_evidence": _array_of_objects(),
        "database_evidence": _array_of_objects(),
        "video_evidence": _array_of_objects(),
        "fused_context": {"type": "object"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "gaps": _array_of("string"),
    },
    "additionalProperties": True,
}

EXPERIMENT_PROCESS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.experiment_process.v1",
    "title": "Experiment process",
    "type": "object",
    "required": [
        "session_id",
        "process_status",
        "step_count",
        "status_counts",
        "steps",
        "timeline_path",
        "evidence_index",
    ],
    "properties": {
        "session_id": {"type": "string"},
        "process_status": {"type": "string"},
        "current_step_id": _nullable("string"),
        "next_step_id": _nullable("string"),
        "step_count": {"type": "integer", "minimum": 0},
        "status_counts": {"type": "object"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "step_id",
                    "name",
                    "expected_action",
                    "status",
                    "completed",
                    "requires_human_confirmation",
                    "confidence",
                    "evidence_refs",
                ],
                "properties": {
                    "step_id": {"type": "string"},
                    "name": {"type": "string"},
                    "expected_action": {"type": "string"},
                    "status": {"type": "string"},
                    "observed": {"type": "boolean"},
                    "inferred": {"type": "boolean"},
                    "completed": {"type": "boolean"},
                    "skipped": {"type": "boolean"},
                    "abnormal": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "confidence_reasons": _array_of("string"),
                    "evidence_refs": _array_of_objects(),
                    "reasoning": _nullable("string"),
                    "requires_human_confirmation": {"type": "boolean"},
                    "repeat_count": {"type": ["integer", "null"], "minimum": 0},
                    "conflict_flags": _array_of("string"),
                    "confirmation_status": _nullable("string"),
                    "condition_results": {"type": "object"},
                },
                "additionalProperties": True,
            },
        },
        "state_machine_path": _nullable("string"),
        "state_machine": {"type": "object"},
        "conflict_report": {"type": "object"},
        "related_history_records": _array_of_objects(),
        "timeline_path": _nullable("string"),
        "evidence_index": {"type": "object"},
    },
    "additionalProperties": True,
}

SOP_STATE_MACHINE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.sop_state_machine.v1",
    "title": "SOP state machine",
    "type": "object",
    "required": ["schema_version", "step_count", "steps", "transitions"],
    "properties": {
        "schema_version": {"type": "string"},
        "source": {"type": "object"},
        "step_count": {"type": "integer", "minimum": 0},
        "initial_step_id": _nullable("string"),
        "terminal_step_ids": _array_of("string"),
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "step_id",
                    "name",
                    "expected_action",
                    "entry_conditions",
                    "completion_conditions",
                    "evidence_requirements",
                    "allowed_transitions",
                ],
                "properties": {
                    "step_id": {"type": "string"},
                    "order": {"type": "integer"},
                    "name": {"type": "string"},
                    "expected_action": {"type": "string"},
                    "entry_conditions": _array_of_objects(),
                    "completion_conditions": _array_of_objects(),
                    "evidence_requirements": _array_of_objects(),
                    "allowed_transitions": {"type": "array"},
                    "branch_condition": {"type": "object"},
                    "repeat_until": {"type": "object"},
                    "wait_conditions": {"type": "object"},
                    "parallel_observations": {"type": "array"},
                },
                "additionalProperties": True,
            },
        },
        "transitions": _array_of_objects(),
        "transition_priors": {"type": "object"},
    },
    "additionalProperties": True,
}

PROCESS_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.process_record.v1",
    "title": "Final step-level process record",
    "type": "object",
    "required": [
        "schema_version",
        "session_id",
        "generated_at",
        "summary",
        "steps",
        "evidence",
        "evidence_index",
        "step_index",
        "audit_report_path",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "session_id": {"type": "string"},
        "generated_at": {"type": "string"},
        "source_paths": {"type": "object"},
        "summary": {
            "type": "object",
            "required": ["step_count", "inferred_step_count", "pending_confirmation_count", "weak_evidence_step_count"],
            "properties": {
                "step_count": {"type": "integer", "minimum": 0},
                "inferred_step_count": {"type": "integer", "minimum": 0},
                "pending_confirmation_count": {"type": "integer", "minimum": 0},
                "weak_evidence_step_count": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": True,
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": [
                    "step_id",
                    "order",
                    "status",
                    "observed",
                    "inferred",
                    "completed",
                    "confidence",
                    "evidence_refs",
                    "evidence_summary",
                    "inference",
                    "confirmation",
                    "audit_flags",
                ],
                "properties": {
                    "step_id": {"type": "string"},
                    "order": {"type": "integer", "minimum": 1},
                    "name": _nullable("string"),
                    "expected_action": _nullable("string"),
                    "status": _nullable("string"),
                    "observed": {"type": "boolean"},
                    "inferred": {"type": "boolean"},
                    "completed": {"type": "boolean"},
                    "skipped": {"type": "boolean"},
                    "abnormal": {"type": "boolean"},
                    "global_start_time": _nullable("string"),
                    "global_end_time": _nullable("string"),
                    "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
                    "confidence_reasons": _array_of("string"),
                    "evidence_refs": _array_of_objects(),
                    "evidence_summary": {"type": "object"},
                    "inference": {"type": "object"},
                    "reasoning": {"type": "object"},
                    "confirmation": {"type": "object"},
                    "audit_flags": _array_of("string"),
                },
                "additionalProperties": True,
            },
        },
        "evidence": _array_of_objects(),
        "evidence_index": {"type": "object"},
        "step_index": {"type": "object"},
        "confirmation_review": {"type": "object"},
        "audit_report_path": {"type": "string"},
    },
    "additionalProperties": True,
}

ASSET_CATALOG_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.asset_catalog.v1",
    "title": "Material asset catalog row",
    "type": "object",
    "required": [
        "asset_id",
        "session_id",
        "asset_type",
        "path",
        "exists",
        "size_bytes",
        "dry_run_placeholder",
        "source_type",
        "source_id",
        "objects",
        "actions",
        "state_tags",
        "quality",
    ],
    "properties": {
        "asset_id": {"type": "string"},
        "session_id": {"type": "string"},
        "asset_type": {"type": "string"},
        "path": {"type": "string"},
        "exists": {"type": "boolean"},
        "size_bytes": {"type": "integer", "minimum": 0},
        "dry_run_placeholder": {"type": "boolean"},
        "source_type": {"type": "string"},
        "source_id": {"type": "string"},
        "segment_id": _nullable("string"),
        "micro_segment_id": _nullable("string"),
        "global_start_time": _nullable("string"),
        "global_end_time": _nullable("string"),
        "objects": _array_of("string"),
        "actions": _array_of("string"),
        "state_tags": _array_of("string"),
        "evidence_level": _nullable("string"),
        "search_text": {"type": "string"},
        "quality": {"type": "object"},
        "payload_ref": {"type": "object"},
        "source_path": _nullable("string"),
        "sha256": _nullable("string"),
        "created_at": _nullable("string"),
        "schema_version": _nullable("string"),
        "privacy_level": _nullable("string"),
        "audit_trail": _array_of_objects(),
    },
    "additionalProperties": True,
}

CONFIRMATION_QUEUE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.confirmation_queue.v1",
    "title": "Human confirmation queue row",
    "type": "object",
    "required": [
        "confirmation_id",
        "session_id",
        "item_type",
        "item_id",
        "status",
        "reason",
        "proposed_update",
        "evidence_refs",
        "created_at",
        "decision",
    ],
    "properties": {
        "confirmation_id": {"type": "string"},
        "session_id": {"type": "string"},
        "item_type": {"type": "string"},
        "item_id": _nullable("string"),
        "status": {"type": "string", "enum": ["pending", "approved", "rejected", "needs_review"]},
        "reason": {"type": "string"},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "summary": _nullable("string"),
        "proposed_update": {"type": "object"},
        "evidence_refs": _array_of_objects(),
        "created_at": {"type": "string"},
        "decision": {"type": "object"},
    },
    "additionalProperties": True,
}

PROCESS_QUALITY_REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.process_quality_report.v1",
    "title": "Process quality report",
    "type": "object",
    "required": [
        "metadata_version",
        "session_id",
        "overall_status",
        "overall_score",
        "checks",
    ],
    "properties": {
        "metadata_version": {"type": "string"},
        "session_id": {"type": "string"},
        "overall_status": {"type": "string"},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "scorecard": {"type": "object"},
        "checks": _array_of_objects(),
        "diagnostics": {"type": "object"},
    },
    "additionalProperties": True,
}

PIPELINE_EVALUATION_REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "key_action_artifacts.pipeline_evaluation_report.v1",
    "title": "Pipeline evaluation report",
    "type": "object",
    "required": [
        "schema_version",
        "session_dir",
        "scores",
        "overall_score",
    ],
    "properties": {
        "schema_version": {"type": "string"},
        "session_dir": {"type": "string"},
        "ground_truth_dir": _nullable("string"),
        "scores": {"type": "object"},
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "schema_validation": {"type": "object"},
    },
    "additionalProperties": True,
}

ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "model_observation_events": ArtifactSpec(
        artifact_type="model_observation_events",
        relative_path="metadata/model_observation_events.jsonl",
        file_format="jsonl",
        schema=MODEL_OBSERVATION_SCHEMA,
    ),
    "video_understanding": ArtifactSpec(
        artifact_type="video_understanding",
        relative_path="metadata/video_understanding.jsonl",
        file_format="jsonl",
        schema=VIDEO_UNDERSTANDING_SCHEMA,
    ),
    "experiment_context": ArtifactSpec(
        artifact_type="experiment_context",
        relative_path="metadata/experiment_context.json",
        file_format="json",
        schema=EXPERIMENT_CONTEXT_SCHEMA,
    ),
    "experiment_process": ArtifactSpec(
        artifact_type="experiment_process",
        relative_path="metadata/experiment_process.json",
        file_format="json",
        schema=EXPERIMENT_PROCESS_SCHEMA,
    ),
    "process_record": ArtifactSpec(
        artifact_type="process_record",
        relative_path="exports/process_record.json",
        file_format="json",
        schema=PROCESS_RECORD_SCHEMA,
    ),
    "asset_catalog": ArtifactSpec(
        artifact_type="asset_catalog",
        relative_path="metadata/material_asset_catalog.jsonl",
        file_format="jsonl",
        schema=ASSET_CATALOG_SCHEMA,
    ),
    "confirmation_queue": ArtifactSpec(
        artifact_type="confirmation_queue",
        relative_path="metadata/human_confirmation_queue.jsonl",
        file_format="jsonl",
        schema=CONFIRMATION_QUEUE_SCHEMA,
    ),
}

OPTIONAL_ARTIFACT_SPECS: dict[str, ArtifactSpec] = {
    "sop_state_machine": ArtifactSpec(
        artifact_type="sop_state_machine",
        relative_path="metadata/sop_state_machine.json",
        file_format="json",
        schema=SOP_STATE_MACHINE_SCHEMA,
    ),
    "process_quality_report": ArtifactSpec(
        artifact_type="process_quality_report",
        relative_path="metadata/process_quality_report.json",
        file_format="json",
        schema=PROCESS_QUALITY_REPORT_SCHEMA,
    ),
    "pipeline_evaluation_report": ArtifactSpec(
        artifact_type="pipeline_evaluation_report",
        relative_path="evaluation/pipeline_evaluation_report.json",
        file_format="json",
        schema=PIPELINE_EVALUATION_REPORT_SCHEMA,
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _issue(severity: str, message: str, path: str) -> dict[str, str]:
    return {"severity": severity, "message": message, "path": path}


def _as_type_names(schema_type: Any) -> list[str]:
    if schema_type is None:
        return []
    if isinstance(schema_type, list):
        return [str(item) for item in schema_type]
    return [str(schema_type)]


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "any":
        return True
    if schema_type == "null":
        return value is None
    if schema_type == "object":
        return isinstance(value, Mapping)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True


def _type_label(type_names: Iterable[str]) -> str:
    return " or ".join(type_names)


def _validate_value(value: Any, schema: Mapping[str, Any], path: str) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    type_names = _as_type_names(schema.get("type"))
    if type_names and not any(_matches_type(value, type_name) for type_name in type_names):
        issues.append(_issue("error", f"expected {_type_label(type_names)}", path))
        return issues

    if value is None:
        return issues

    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(str(item) for item in schema["enum"])
        issues.append(_issue("error", f"expected one of: {allowed}", path))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < float(schema["minimum"]):
            issues.append(_issue("error", f"expected >= {schema['minimum']}", path))
        if "maximum" in schema and value > float(schema["maximum"]):
            issues.append(_issue("error", f"expected <= {schema['maximum']}", path))

    if isinstance(value, Mapping):
        required = [str(item) for item in schema.get("required", [])]
        for field in required:
            if field not in value:
                issues.append(_issue("error", "missing required field", f"{path}.{field}"))
        properties = schema.get("properties") or {}
        if isinstance(properties, Mapping):
            for field, child_schema in properties.items():
                if field in value and isinstance(child_schema, Mapping):
                    issues.extend(_validate_value(value[field], child_schema, f"{path}.{field}"))
        if schema.get("additionalProperties") is False and isinstance(properties, Mapping):
            allowed = set(properties)
            for field in value:
                if field not in allowed:
                    issues.append(_issue("error", "unexpected field", f"{path}.{field}"))

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            issues.append(_issue("error", f"expected at least {schema['minItems']} items", path))
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                issues.extend(_validate_value(item, item_schema, f"{path}[{index}]"))

    return issues


def get_artifact_schema(artifact_type: str) -> dict[str, Any]:
    return dict(_artifact_spec(artifact_type).schema)


def get_artifact_spec(artifact_type: str) -> ArtifactSpec:
    return _artifact_spec(artifact_type)


def available_artifact_types(include_optional: bool = True) -> list[str]:
    specs = _artifact_specs(include_optional=include_optional)
    return sorted(specs)


def list_artifact_specs(include_optional: bool = True) -> list[dict[str, str]]:
    return [
        {
            "artifact_type": spec.artifact_type,
            "relative_path": spec.relative_path,
            "file_format": spec.file_format,
        }
        for spec in _artifact_specs(include_optional=include_optional).values()
    ]


def validate_artifact_record(artifact_type: str, record: Mapping[str, Any], path: str = "$") -> list[dict[str, str]]:
    spec = _artifact_spec(artifact_type)
    return _validate_value(record, spec.schema, path)


def _parse_json_file(path: Path) -> tuple[Any, list[dict[str, str]]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), []
    except OSError as exc:
        return None, [_issue("error", f"cannot read file: {exc}", str(path))]
    except json.JSONDecodeError as exc:
        return None, [_issue("error", f"invalid JSON: {exc}", str(path))]


def _result(
    *,
    artifact_type: str,
    path: Path,
    file_format: str,
    exists: bool,
    record_count: int,
    issues: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "path": str(path),
        "file_format": file_format,
        "exists": exists,
        "record_count": record_count,
        "valid": not any(issue.get("severity") == "error" for issue in issues),
        "issues": issues,
    }


def _artifact_specs(include_optional: bool = True) -> dict[str, ArtifactSpec]:
    if not include_optional:
        return dict(ARTIFACT_SPECS)
    return {**ARTIFACT_SPECS, **OPTIONAL_ARTIFACT_SPECS}


def _artifact_spec(artifact_type: str) -> ArtifactSpec:
    specs = _artifact_specs(include_optional=True)
    if artifact_type not in specs:
        raise KeyError(artifact_type)
    return specs[artifact_type]


def validate_artifact_file(path: str | Path, artifact_type: str) -> dict[str, Any]:
    spec = _artifact_spec(artifact_type)
    source = Path(path)
    if not source.exists():
        return _result(
            artifact_type=artifact_type,
            path=source,
            file_format=spec.file_format,
            exists=False,
            record_count=0,
            issues=[_issue("error", "artifact file does not exist", str(source))],
        )

    if spec.file_format == "json":
        data, issues = _parse_json_file(source)
        if not issues:
            issues.extend(_validate_value(data, spec.schema, "$"))
        return _result(
            artifact_type=artifact_type,
            path=source,
            file_format=spec.file_format,
            exists=True,
            record_count=1 if data is not None else 0,
            issues=issues,
        )

    issues: list[dict[str, str]] = []
    record_count = 0
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                record_count += 1
                try:
                    row = json.loads(text)
                except json.JSONDecodeError as exc:
                    issues.append(_issue("error", f"invalid JSONL record: {exc}", f"{source}:line {line_no}"))
                    continue
                row_issues = _validate_value(row, spec.schema, "$")
                for issue in row_issues:
                    issues.append(
                        {
                            **issue,
                            "path": f"{source}:line {line_no} {issue['path']}",
                        }
                    )
    except OSError as exc:
        issues.append(_issue("error", f"cannot read file: {exc}", str(source)))

    return _result(
        artifact_type=artifact_type,
        path=source,
        file_format=spec.file_format,
        exists=True,
        record_count=record_count,
        issues=issues,
    )


def validate_session_artifacts(
    session_dir: str | Path,
    artifact_types: Iterable[str] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    selected = list(artifact_types) if artifact_types is not None else list(ARTIFACT_SPECS)
    artifacts = []
    for artifact_type in selected:
        spec = _artifact_spec(artifact_type)
        artifacts.append(validate_artifact_file(session / spec.relative_path, artifact_type))

    issue_count = sum(len(item["issues"]) for item in artifacts)
    error_count = sum(1 for item in artifacts for issue in item["issues"] if issue.get("severity") == "error")
    result = {
        "schema_version": SCHEMA_VERSION,
        "validated_at": _now(),
        "session_dir": str(session),
        "artifact_count": len(artifacts),
        "valid_artifact_count": sum(1 for item in artifacts if item["valid"]),
        "missing_artifact_count": sum(1 for item in artifacts if not item["exists"]),
        "record_count": sum(int(item["record_count"]) for item in artifacts),
        "issue_count": issue_count,
        "error_count": error_count,
        "valid": error_count == 0,
        "artifacts": artifacts,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


__all__ = [
    "ARTIFACT_SPECS",
    "OPTIONAL_ARTIFACT_SPECS",
    "SCHEMA_VERSION",
    "ASSET_CATALOG_SCHEMA",
    "CONFIRMATION_QUEUE_SCHEMA",
    "EXPERIMENT_CONTEXT_SCHEMA",
    "EXPERIMENT_PROCESS_SCHEMA",
    "PROCESS_RECORD_SCHEMA",
    "MODEL_OBSERVATION_SCHEMA",
    "PIPELINE_EVALUATION_REPORT_SCHEMA",
    "PROCESS_QUALITY_REPORT_SCHEMA",
    "SOP_STATE_MACHINE_SCHEMA",
    "VIDEO_UNDERSTANDING_SCHEMA",
    "available_artifact_types",
    "get_artifact_schema",
    "get_artifact_spec",
    "list_artifact_specs",
    "validate_artifact_file",
    "validate_artifact_record",
    "validate_session_artifacts",
]
