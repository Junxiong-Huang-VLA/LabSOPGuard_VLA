from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


DEFAULT_VLM_MODEL = "qwen3.6-plus"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uuid() -> str:
    return str(uuid.uuid4())


class ExperimentStatus(str, Enum):
    CREATED = "created"
    DRAFT = "created"
    VIDEO_UPLOADED = "video_uploaded"
    QUEUED = "queued"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    REVIEWED = "reviewed"
    PENDING = "created"
    PROCESSING = "analyzing"
    COMPLETED = "analyzed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    CONFIRMED = "confirmed"
    CANDIDATE = "candidate"
    INFERRED = "inferred"
    SKIPPED = "skipped"


class StepConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceType(str, Enum):
    VIDEO_FRAME = "video_frame"
    VIDEO_TIMESTAMP = "video_timestamp"
    CONVERSATION = "conversation"
    PROTOCOL = "protocol"
    SENSOR = "sensor"
    MANUAL = "manual"
    INFERRED = "inferred"


class ContextSource(str, Enum):
    VIDEO = "video"
    CONVERSATION = "conversation"
    PROTOCOL = "protocol"
    SENSOR = "sensor"
    MANUAL = "manual"
    SYSTEM = "system"


class MediaType(str, Enum):
    VIDEO = "video"
    IMAGE = "image"
    TEXT = "text"
    CONVERSATION = "conversation"
    PROTOCOL = "protocol"
    SENSOR_DATA = "sensor_data"


class ProcessStage(str, Enum):
    INGESTION = "ingestion"
    VIDEO_UNDERSTANDING = "video_understanding"
    CONTEXT_INTEGRATION = "context_integration"
    STEP_REASONING = "step_reasoning"
    EVIDENCE_LINKING = "evidence_linking"
    OUTPUT_GENERATION = "output_generation"


@dataclass
class ProvenanceInfo:
    source: str
    source_id: Optional[str] = None
    confidence: float = 1.0
    is_inferred: bool = False
    inference_method: Optional[str] = None
    model_used: Optional[str] = None
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_video_frame(cls, frame_id: int, timestamp_sec: float, confidence: float = 1.0) -> "ProvenanceInfo":
        return cls(
            source="video",
            source_id=f"frame_{frame_id}",
            confidence=confidence,
            is_inferred=False,
            timestamp=_now_iso(),
        )

    @classmethod
    def from_inference(cls, inference_method: str, model_used: str, confidence: float) -> "ProvenanceInfo":
        return cls(
            source="system",
            confidence=confidence,
            is_inferred=True,
            inference_method=inference_method,
            model_used=model_used,
            timestamp=_now_iso(),
        )


@dataclass
class EvidenceRef:
    evidence_id: str = field(default_factory=_uuid)
    evidence_type: EvidenceType = EvidenceType.VIDEO_FRAME
    media_asset_id: Optional[str] = None
    source: str = ""
    frame_id: Optional[int] = None
    timestamp_sec: Optional[float] = None
    text_snippet: Optional[str] = None
    bbox: Optional[List[int]] = None
    confidence: float = 1.0
    description: Optional[str] = None
    provenance: Optional[ProvenanceInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence_type"] = self.evidence_type.value if isinstance(self.evidence_type, Enum) else self.evidence_type
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        return data


@dataclass
class StepParameter:
    name: str
    value: Any
    unit: Optional[str] = None
    source: str = "observed"
    provenance: Optional[ProvenanceInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        return data


@dataclass
class MediaAsset:
    asset_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    media_type: MediaType = MediaType.VIDEO
    file_path: Optional[str] = None
    url: Optional[str] = None
    filename: str = ""
    mime_type: Optional[str] = None
    duration_sec: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    size_bytes: Optional[int] = None
    hash_sha256: Optional[str] = None
    created_at: str = field(default_factory=_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["media_type"] = self.media_type.value if isinstance(self.media_type, Enum) else self.media_type
        return data


@dataclass
class PhysicalEvent:
    event_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    event_type: str = ""
    timestamp_sec: float = 0.0
    end_timestamp_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    location: Optional[str] = None
    device_id: Optional[str] = None
    parameters: List[StepParameter] = field(default_factory=list)
    confidence: float = 1.0
    provenance: Optional[ProvenanceInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["parameters"] = [p.to_dict() for p in self.parameters]
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        return data


@dataclass
class ContextEvent:
    event_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    context_source: ContextSource = ContextSource.VIDEO
    event_type: str = ""
    timestamp_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    content: str = ""
    raw_content: Optional[str] = None
    confidence: float = 1.0
    provenance: Optional[ProvenanceInfo] = None
    linked_step_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["context_source"] = self.context_source.value if isinstance(self.context_source, Enum) else self.context_source
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        return data


@dataclass
class StepRecord:
    step_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    step_index: int = 0
    step_name: str = ""
    step_description: str = ""
    status: StepStatus = StepStatus.CONFIRMED
    start_time_sec: float = 0.0
    end_time_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    confidence: float = 1.0
    step_confidence: StepConfidence = StepConfidence.HIGH
    completed_by_inference: bool = False
    inference_method: Optional[str] = None
    inference_model: Optional[str] = None
    evidence_refs: List[EvidenceRef] = field(default_factory=list)
    parameters: List[StepParameter] = field(default_factory=list)
    linked_context_events: List[str] = field(default_factory=list)
    linked_physical_events: List[str] = field(default_factory=list)
    linked_media_assets: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    evidence_notes: Optional[str] = None
    provenance: Optional[ProvenanceInfo] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "experiment_id": self.experiment_id,
            "step_index": self.step_index,
            "step_name": self.step_name,
            "step_description": self.step_description,
            "status": self.status.value if isinstance(self.status, Enum) else self.status,
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "duration_sec": self.duration_sec,
            "confidence": round(self.confidence, 4),
            "step_confidence": self.step_confidence.value if isinstance(self.step_confidence, Enum) else self.step_confidence,
            "completed_by_inference": self.completed_by_inference,
            "inference_method": self.inference_method,
            "inference_model": self.inference_model,
            "evidence_refs": [e.to_dict() for e in self.evidence_refs],
            "parameters": [p.to_dict() for p in self.parameters],
            "linked_context_events": self.linked_context_events,
            "linked_physical_events": self.linked_physical_events,
            "linked_media_assets": self.linked_media_assets,
            "notes": self.notes,
            "evidence_notes": self.evidence_notes,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def summary(self) -> str:
        flag = " [INFERRED]" if self.completed_by_inference else ""
        return f"[{self.step_index}] {self.step_name} ({self.status.value}){flag} @{self.start_time_sec:.1f}s conf={self.confidence:.2f}"


@dataclass
class ExperimentTimeline:
    timeline_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    title: str = ""
    steps: List[StepRecord] = field(default_factory=list)
    total_steps: int = 0
    confirmed_steps: int = 0
    candidate_steps: int = 0
    inferred_steps: int = 0
    skipped_steps: int = 0
    avg_confidence: Optional[float] = None
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    total_duration_sec: float = 0.0
    video_asset_id: Optional[str] = None
    video_duration_sec: Optional[float] = None
    video_coverage_ratio: float = 0.0
    context_summary: Optional[str] = None
    protocol_id: Optional[str] = None
    protocol_name: Optional[str] = None
    protocol_text: Optional[str] = None
    processing_stage: ProcessStage = ProcessStage.OUTPUT_GENERATION
    models_used: List[str] = field(default_factory=list)
    inference_count: int = 0
    media_assets: List[str] = field(default_factory=list)
    context_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def compute_stats(self) -> None:
        self.total_steps = len(self.steps)
        self.confirmed_steps = sum(1 for s in self.steps if s.status == StepStatus.CONFIRMED)
        self.candidate_steps = sum(1 for s in self.steps if s.status == StepStatus.CANDIDATE)
        self.inferred_steps = sum(1 for s in self.steps if s.status == StepStatus.INFERRED)
        self.skipped_steps = sum(1 for s in self.steps if s.status == StepStatus.SKIPPED)
        self.inference_count = sum(1 for s in self.steps if s.completed_by_inference)

        if self.steps:
            self.avg_confidence = round(sum(s.confidence for s in self.steps) / len(self.steps), 4)
            self.start_time_sec = min(s.start_time_sec for s in self.steps)
            self.end_time_sec = max((s.end_time_sec or s.start_time_sec) for s in self.steps)
            self.total_duration_sec = self.end_time_sec - self.start_time_sec
            if self.video_duration_sec and self.video_duration_sec > 0:
                self.video_coverage_ratio = round(self.total_duration_sec / self.video_duration_sec, 4)
            else:
                self.video_coverage_ratio = 1.0
        else:
            self.avg_confidence = None
            self.start_time_sec = 0.0
            self.end_time_sec = 0.0
            self.total_duration_sec = 0.0
            self.video_coverage_ratio = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["steps"] = [s.to_dict() for s in self.steps]
        data["processing_stage"] = self.processing_stage.value if isinstance(self.processing_stage, Enum) else self.processing_stage
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def step_summary(self) -> str:
        avg_conf = f"{self.avg_confidence:.2f}" if self.avg_confidence is not None else "N/A"
        lines = [f"Timeline: {self.title} ({self.experiment_id})"]
        lines.append(f"  Steps: {self.total_steps} total, {self.confirmed_steps} confirmed, {self.candidate_steps} candidate, {self.inferred_steps} inferred")
        lines.append(f"  Coverage: {self.total_duration_sec:.1f}s, avg_conf={avg_conf}, video_coverage={self.video_coverage_ratio:.1%}")
        for step in self.steps:
            lines.append(f"  {step.summary()}")
        return "\n".join(lines)


@dataclass
class Experiment:
    experiment_id: str = field(default_factory=_uuid)
    title: str = ""
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_at: str = field(default_factory=_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    analyzed_at: Optional[str] = None
    video_assets: List[MediaAsset] = field(default_factory=list)
    context_inputs: List[Dict[str, Any]] = field(default_factory=list)
    protocol_text: Optional[str] = None
    protocol_id: Optional[str] = None
    video_asset_id: Optional[str] = None
    analysis_job_id: Optional[str] = None
    timeline: Optional[ExperimentTimeline] = None
    context_events: List[ContextEvent] = field(default_factory=list)
    physical_events: List[PhysicalEvent] = field(default_factory=list)
    processing_stage: ProcessStage = ProcessStage.INGESTION
    processing_error: Optional[str] = None
    models_used: List[str] = field(default_factory=list)
    total_steps: int = 0
    inferred_steps: int = 0
    avg_confidence: Optional[float] = None
    evidence_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    owner: Optional[str] = None

    def sync_stats(self) -> None:
        if self.timeline:
            self.total_steps = self.timeline.total_steps
            self.inferred_steps = self.timeline.inferred_steps
            self.avg_confidence = self.timeline.avg_confidence
            self.evidence_count = sum(len(s.evidence_refs) for s in self.timeline.steps)
        else:
            self.total_steps = 0
            self.inferred_steps = 0
            self.avg_confidence = None
            self.evidence_count = 0
        if self.status == ExperimentStatus.ANALYZING and not self.started_at:
            self.started_at = _now_iso()
        if self.status in {ExperimentStatus.ANALYZED, ExperimentStatus.REVIEWED} and not self.completed_at:
            self.completed_at = _now_iso()
        if self.status in {ExperimentStatus.ANALYZED, ExperimentStatus.REVIEWED} and not self.analyzed_at:
            self.analyzed_at = _now_iso()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value if isinstance(self.status, Enum) else self.status
        data["processing_stage"] = self.processing_stage.value if isinstance(self.processing_stage, Enum) else self.processing_stage
        data["video_assets"] = [a.to_dict() for a in self.video_assets]
        data["context_events"] = [e.to_dict() for e in self.context_events]
        data["physical_events"] = [e.to_dict() for e in self.physical_events]
        if self.timeline:
            data["timeline"] = self.timeline.to_dict()
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class MultimodalMaterialStreamItem:
    schema_version: str = "material_stream.v1"
    item_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    timestamp_sec: float = 0.0
    local_timestamp_sec: Optional[float] = None
    global_timestamp_sec: Optional[float] = None
    camera_id: Optional[str] = None
    view_type: Optional[str] = None
    source_group: Optional[str] = None
    source_type: Optional[str] = None
    sync_group: Optional[str] = None
    alignment_confidence: Optional[float] = None
    media_asset_id: Optional[str] = None
    stream_id: Optional[str] = None
    frame_id: int = 0
    local_frame_id: Optional[int] = None
    frame_bgr_path: Optional[str] = None
    frame_embedding: Optional[List[float]] = None
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    pose_keypoints: Optional[List[List[float]]] = None
    scene_description: Optional[str] = None
    detected_activities: List[str] = field(default_factory=list)
    object_labels: List[str] = field(default_factory=list)
    ppe_status: Dict[str, bool] = field(default_factory=dict)
    transcript_segment: Optional[str] = None
    conversation_context: Optional[str] = None
    linked_context_event_ids: List[str] = field(default_factory=list)
    inference_model: Optional[str] = None
    confidence: float = 1.0
    is_key_frame: bool = False
    key_frame_reason: Optional[str] = None
    change_score: float = 0.0
    clip_id: Optional[str] = None
    analysis: Dict[str, Any] = field(default_factory=dict)
    provenance: Optional[ProvenanceInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.provenance:
            data["provenance"] = self.provenance.to_dict()
        return data


def make_confirmed_step(
    experiment_id: str,
    step_index: int,
    step_name: str,
    start_time_sec: float,
    end_time_sec: Optional[float] = None,
    evidence_refs: Optional[List[EvidenceRef]] = None,
    description: str = "",
    parameters: Optional[List[StepParameter]] = None,
) -> StepRecord:
    return StepRecord(
        experiment_id=experiment_id,
        step_index=step_index,
        step_name=step_name,
        step_description=description,
        status=StepStatus.CONFIRMED,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        duration_sec=(end_time_sec - start_time_sec) if end_time_sec is not None else None,
        confidence=1.0,
        step_confidence=StepConfidence.HIGH,
        completed_by_inference=False,
        evidence_refs=evidence_refs or [],
        parameters=parameters or [],
        provenance=ProvenanceInfo(source="video", confidence=1.0, is_inferred=False),
    )


def make_inferred_step(
    experiment_id: str,
    step_index: int,
    step_name: str,
    start_time_sec: float,
    end_time_sec: Optional[float] = None,
    confidence: float = 0.6,
    inference_method: str = "qwen_vl_temporal_reasoning",
    inference_model: str = DEFAULT_VLM_MODEL,
    description: str = "",
    evidence_refs: Optional[List[EvidenceRef]] = None,
    linked_context: Optional[List[str]] = None,
    parameters: Optional[List[StepParameter]] = None,
    evidence_notes: Optional[str] = None,
) -> StepRecord:
    if confidence >= 0.8:
        step_conf = StepConfidence.HIGH
    elif confidence >= 0.5:
        step_conf = StepConfidence.MEDIUM
    else:
        step_conf = StepConfidence.LOW

    status = StepStatus.INFERRED if confidence < 0.7 else StepStatus.CANDIDATE
    return StepRecord(
        experiment_id=experiment_id,
        step_index=step_index,
        step_name=step_name,
        step_description=description,
        status=status,
        start_time_sec=start_time_sec,
        end_time_sec=end_time_sec,
        duration_sec=(end_time_sec - start_time_sec) if end_time_sec is not None else None,
        confidence=round(confidence, 4),
        step_confidence=step_conf,
        completed_by_inference=True,
        inference_method=inference_method,
        inference_model=inference_model,
        evidence_refs=evidence_refs or [],
        parameters=parameters or [],
        linked_context_events=linked_context or [],
        evidence_notes=evidence_notes,
        provenance=ProvenanceInfo.from_inference(inference_method, inference_model, confidence),
    )


def make_inferred_parameter(
    name: str,
    value: Any,
    unit: Optional[str] = None,
    confidence: float = 0.7,
    method: str = "qwen_vl_ocr",
    model: str = DEFAULT_VLM_MODEL,
) -> StepParameter:
    return StepParameter(
        name=name,
        value=value,
        unit=unit,
        source="inferred",
        provenance=ProvenanceInfo.from_inference(method, model, confidence),
    )


def get_step_record_schema() -> Dict[str, Any]:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": [
            "step_id",
            "experiment_id",
            "step_index",
            "step_name",
            "status",
            "start_time_sec",
            "confidence",
            "completed_by_inference",
            "created_at",
        ],
        "properties": {
            "step_id": {"type": "string", "format": "uuid"},
            "experiment_id": {"type": "string"},
            "step_index": {"type": "integer", "minimum": 0},
            "step_name": {"type": "string", "minLength": 1},
            "step_description": {"type": "string"},
            "status": {"type": "string", "enum": ["confirmed", "candidate", "inferred", "skipped"]},
            "start_time_sec": {"type": "number", "minimum": 0},
            "end_time_sec": {"type": ["number", "null"], "minimum": 0},
            "duration_sec": {"type": ["number", "null"], "minimum": 0},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "step_confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "completed_by_inference": {"type": "boolean"},
            "inference_method": {"type": ["string", "null"]},
            "inference_model": {"type": ["string", "null"]},
            "evidence_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["evidence_id", "evidence_type", "source"],
                    "properties": {
                        "evidence_id": {"type": "string"},
                        "evidence_type": {"type": "string"},
                        "source": {"type": "string"},
                        "frame_id": {"type": ["integer", "null"]},
                        "timestamp_sec": {"type": ["number", "null"]},
                        "confidence": {"type": "number"},
                    },
                },
            },
            "parameters": {"type": "array"},
            "linked_context_events": {"type": "array", "items": {"type": "string"}},
            "linked_media_assets": {"type": "array", "items": {"type": "string"}},
            "notes": {"type": ["string", "null"]},
            "evidence_notes": {"type": ["string", "null"]},
            "provenance": {
                "type": ["object", "null"],
                "required": ["source", "confidence", "is_inferred"],
                "properties": {
                    "source": {"type": "string"},
                    "confidence": {"type": "number"},
                    "is_inferred": {"type": "boolean"},
                    "inference_method": {"type": ["string", "null"]},
                    "model_used": {"type": ["string", "null"]},
                },
            },
            "created_at": {"type": "string", "format": "date-time"},
            "updated_at": {"type": "string", "format": "date-time"},
        },
    }
