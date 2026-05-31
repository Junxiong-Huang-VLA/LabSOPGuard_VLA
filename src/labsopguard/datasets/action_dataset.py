from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ACTION_DATASET_SCHEMA_VERSION = "lab_action_dataset.v1"
SUPPORTED_EVENT_TYPES = {
    "hand_object_interaction",
    "object_move",
    "liquid_transfer",
    "panel_operation",
    "container_state_change",
}


@dataclass
class ActionEventAnnotation:
    event_id: str
    event_type: str
    start_time_sec: float
    end_time_sec: float
    involved_objects: List[str] = field(default_factory=list)
    actor_track_id: Optional[str] = None
    source_container: Optional[Dict[str, Any]] = None
    target_container: Optional[Dict[str, Any]] = None
    state_before: Optional[str] = None
    state_after: Optional[str] = None
    state_change_type: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActionVideoRecord:
    video_id: str
    video_path: str
    experiment_id: Optional[str] = None
    experiment_type: Optional[str] = None
    duration_sec: Optional[float] = None
    annotations: List[ActionEventAnnotation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["annotations"] = [item.to_dict() for item in self.annotations]
        return data


@dataclass
class ActionDataset:
    dataset_id: str
    records: List[ActionVideoRecord]
    schema_version: str = ACTION_DATASET_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "records": [item.to_dict() for item in self.records],
        }


def load_action_dataset(path: str | Path) -> ActionDataset:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records: List[ActionVideoRecord] = []
    for raw_record in payload.get("records") or payload.get("videos") or []:
        annotations = [
            ActionEventAnnotation(
                event_id=str(item.get("event_id") or f"ann_{idx:04d}"),
                event_type=str(item.get("event_type")),
                start_time_sec=float(item.get("start_time_sec") or item.get("time_start") or 0.0),
                end_time_sec=float(item.get("end_time_sec") or item.get("time_end") or 0.0),
                involved_objects=[str(value) for value in item.get("involved_objects") or item.get("objects") or []],
                actor_track_id=item.get("actor_track_id"),
                source_container=item.get("source_container"),
                target_container=item.get("target_container"),
                state_before=item.get("state_before"),
                state_after=item.get("state_after"),
                state_change_type=item.get("state_change_type"),
                notes=str(item.get("notes") or ""),
            )
            for idx, item in enumerate(raw_record.get("annotations") or raw_record.get("events") or [])
        ]
        records.append(
            ActionVideoRecord(
                video_id=str(raw_record.get("video_id") or raw_record.get("id") or Path(str(raw_record.get("video_path") or "")).stem),
                video_path=str(raw_record.get("video_path") or raw_record.get("path") or ""),
                experiment_id=raw_record.get("experiment_id"),
                experiment_type=raw_record.get("experiment_type"),
                duration_sec=float(raw_record["duration_sec"]) if raw_record.get("duration_sec") is not None else None,
                annotations=annotations,
            )
        )
    return ActionDataset(dataset_id=str(payload.get("dataset_id") or Path(path).stem), records=records, schema_version=str(payload.get("schema_version") or ACTION_DATASET_SCHEMA_VERSION))


def validate_action_dataset(dataset: ActionDataset, *, require_video_files: bool = False, root: str | Path | None = None) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    seen_event_ids: set[str] = set()
    base = Path(root) if root else None
    for record in dataset.records:
        if not record.video_id:
            errors.append("video_id_missing")
        if not record.video_path:
            errors.append(f"{record.video_id}:video_path_missing")
        elif require_video_files:
            candidate = Path(record.video_path)
            if not candidate.is_absolute() and base:
                candidate = base / candidate
            if not candidate.exists():
                errors.append(f"{record.video_id}:video_file_missing:{candidate}")
        for ann in record.annotations:
            if ann.event_id in seen_event_ids:
                errors.append(f"duplicate_event_id:{ann.event_id}")
            seen_event_ids.add(ann.event_id)
            if ann.event_type not in SUPPORTED_EVENT_TYPES:
                errors.append(f"{ann.event_id}:unsupported_event_type:{ann.event_type}")
            if ann.end_time_sec <= ann.start_time_sec:
                errors.append(f"{ann.event_id}:invalid_time_window")
            if ann.event_type == "liquid_transfer" and (not ann.source_container or not ann.target_container):
                warnings.append(f"{ann.event_id}:liquid_transfer_missing_source_or_target")
            if ann.event_type == "container_state_change" and not ann.state_change_type:
                warnings.append(f"{ann.event_id}:container_state_change_missing_state_change_type")
    return {
        "schema_version": "action_dataset_validation.v1",
        "dataset_id": dataset.dataset_id,
        "is_valid": not errors,
        "record_count": len(dataset.records),
        "annotation_count": sum(len(record.annotations) for record in dataset.records),
        "errors": errors,
        "warnings": warnings,
    }


def write_action_dataset_template(path: str | Path) -> None:
    template = ActionDataset(
        dataset_id="lab_actions_template",
        records=[
            ActionVideoRecord(
                video_id="video_001",
                video_path="videos/video_001.mp4",
                experiment_type="solid_weighing",
                annotations=[
                    ActionEventAnnotation(
                        event_id="ann_001",
                        event_type="object_move",
                        start_time_sec=81.0,
                        end_time_sec=84.0,
                        involved_objects=["hand", "bottle"],
                    )
                ],
            )
        ],
    )
    Path(path).write_text(json.dumps(template.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
