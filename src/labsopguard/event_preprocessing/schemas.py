from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

METADATA_VERSION = "event_preprocessing.v1"


@dataclass
class DetectionBox:
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    track_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data


@dataclass
class DetectionFrame:
    frame_idx: int
    timestamp_sec: float
    detections: List[DetectionBox] = field(default_factory=list)
    semantic_activities: List[str] = field(default_factory=list)
    semantic_objects: List[str] = field(default_factory=list)
    scene_description: str = ""
    change_score: float = 0.0
    active_track_ids: List[str] = field(default_factory=list)
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": self.timestamp_sec,
            "detections": [item.to_dict() for item in self.detections],
            "semantic_activities": self.semantic_activities,
            "semantic_objects": self.semantic_objects,
            "scene_description": self.scene_description,
            "change_score": self.change_score,
            "active_track_ids": self.active_track_ids,
            "metadata_version": self.metadata_version,
        }


@dataclass
class Tracklet:
    track_id: str
    class_name: str
    start_frame_idx: int
    end_frame_idx: int
    start_time_sec: float
    end_time_sec: float
    frame_indices: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    mean_confidence: float
    displacement_px: float
    fragment_count: int = 1
    recovered_from_fragment: bool = False
    tracking_backend: str = "unknown"
    tracking_backend_version: str = "unknown"
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bboxes"] = [list(bbox) for bbox in self.bboxes]
        return data


@dataclass
class TrackedObject:
    track_id: str
    experiment_id: str
    source_video_id: str
    class_name: str
    display_name: str
    start_time_sec: float
    end_time_sec: float
    frame_indices: List[int]
    bboxes: List[Tuple[int, int, int, int]]
    centroids: List[Tuple[float, float]]
    confidence_stats: Dict[str, float]
    velocity_stats: Dict[str, float]
    track_confidence: float
    id_switch_risk: float
    occlusion_ratio: float
    fragment_count: int
    recovered_from_fragment: bool
    tracking_backend: str
    tracking_backend_version: str
    state_labels: List[str] = field(default_factory=list)
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["bboxes"] = [list(bbox) for bbox in self.bboxes]
        data["centroids"] = [list(point) for point in self.centroids]
        return data


@dataclass
class TrackRelation:
    relation_id: str
    relation_type: str
    subject_track_id: str
    object_track_id: str
    start_time_sec: float
    end_time_sec: float
    confidence: float
    relation_features: Dict[str, Any]
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContainerRole:
    object_name: str
    track_id: Optional[str] = None
    role_confidence: float = 0.0
    role_source: str = "heuristic_container_role_resolver"
    bbox: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data


@dataclass
class EventProposal:
    proposal_id: str
    event_type: str
    start_frame_idx: int
    end_frame_idx: int
    start_time_sec: float
    end_time_sec: float
    evidence_frame_indices: List[int]
    involved_objects: List[str]
    dominant_object: Optional[str]
    involved_track_ids: List[str]
    primary_track_id: Optional[str]
    source_container: Optional[ContainerRole]
    target_container: Optional[ContainerRole]
    track_motion_summary: Dict[str, Any]
    actor_track_id: Optional[str]
    tool_track_id: Optional[str]
    related_tracks: List[str]
    transfer_mode: Optional[str]
    action_resolution_source: str
    action_resolution_notes: str
    supporting_relation_ids: List[str]
    direction_confidence: Optional[float]
    direction_status: Optional[str]
    direction_evidence: List[str]
    state_before: Optional[str]
    state_after: Optional[str]
    state_change_type: Optional[str]
    state_confidence: Optional[float]
    state_evidence: List[str]
    evidence_grade: str
    review_status: str
    evidence_summary: str
    related_detection_classes: List[str]
    confidence: float
    proposal_source: str = "temporal_detection_heuristic"
    notes: str = ""
    metadata_version: str = METADATA_VERSION
    # gloved_hand × 具体物体类别 的细粒度交互专用字段
    contact_target_class: str = ""   # 目标物体的英文类别名，如 "balance"
    contact_target_zh: str = ""      # 目标物体的中文名，如 "天平"
    status: str = "candidate"
    hard_gate: Dict[str, Any] = field(default_factory=dict)
    evidence_detail: Dict[str, Any] = field(default_factory=dict)
    reject_reasons: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    qwen_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PhysicalEvent:
    event_id: str
    experiment_id: str
    source_video_id: str
    event_type: str
    stable_name: str
    display_name: str
    actor_name: str
    start_time_sec: float
    end_time_sec: float
    duration_sec: float
    key_timestamps: List[float]
    involved_objects: List[str]
    dominant_object: Optional[str]
    involved_track_ids: List[str]
    primary_track_id: Optional[str]
    source_container: Optional[Dict[str, Any]]
    target_container: Optional[Dict[str, Any]]
    track_motion_summary: Dict[str, Any]
    actor_track_id: Optional[str]
    tool_track_id: Optional[str]
    related_tracks: List[str]
    transfer_mode: Optional[str]
    action_resolution_source: str
    action_resolution_notes: str
    supporting_relation_ids: List[str]
    direction_confidence: Optional[float]
    direction_status: Optional[str]
    direction_evidence: List[str]
    state_before: Optional[str]
    state_after: Optional[str]
    state_change_type: Optional[str]
    state_confidence: Optional[float]
    state_evidence: List[str]
    evidence_grade: str
    review_status: str
    evidence_summary: str
    confidence: float
    event_status: str
    proposal_source: str
    evidence_frame_indices: List[int]
    related_detection_classes: List[str]
    notes: str = ""
    metadata_version: str = METADATA_VERSION
    asset_pack: Optional[Dict[str, Any]] = None
    status: str = "candidate"
    hard_gate: Dict[str, Any] = field(default_factory=dict)
    evidence_detail: Dict[str, Any] = field(default_factory=dict)
    reject_reasons: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    qwen_audit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EventAssetPack:
    event_id: str
    clip_path: str
    preview_path: str
    keyframe_paths: List[str]
    event_json_path: str
    overlay_mode: str
    asset_status: str
    quality_score: float = 0.0
    quality_grade: str = "unknown"
    quality_reasons: List[str] = field(default_factory=list)
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IndexedMaterialRecord:
    material_id: str
    experiment_id: str
    event_id: str
    event_type: str
    display_name: str
    stable_name: str
    actor_name: str
    source_container_json: str
    target_container_json: str
    source_container_class: Optional[str]
    source_container_track_id: Optional[str]
    target_container_class: Optional[str]
    target_container_track_id: Optional[str]
    actor_track_id: Optional[str]
    tool_track_id: Optional[str]
    transfer_mode: Optional[str]
    direction_confidence: Optional[float]
    direction_status: Optional[str]
    evidence_grade: Optional[str]
    review_status: Optional[str]
    time_start: float
    time_end: float
    duration_sec: float
    semantic_tags: List[str]
    involved_objects_json: str
    clip_path: str
    preview_path: str
    keyframe_count: int
    quality_score: float
    quality_grade: str
    quality_reasons_json: str
    qwen_summary: Optional[str]
    linked_step_id: Optional[str]
    searchable_text: str
    created_at: str
    metadata_version: str = METADATA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def dump_json(path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
