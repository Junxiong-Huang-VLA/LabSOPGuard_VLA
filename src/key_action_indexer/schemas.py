from __future__ import annotations



import json

from dataclasses import asdict, dataclass, field, is_dataclass

from datetime import datetime

from pathlib import Path

from typing import Any, Optional





def _jsonable(value: Any) -> Any:

    if isinstance(value, datetime):

        return value.isoformat()

    if isinstance(value, Path):

        return str(value)

    if is_dataclass(value):

        return {key: _jsonable(item) for key, item in asdict(value).items()}

    if isinstance(value, dict):

        return {key: _jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):

        return [_jsonable(item) for item in value]

    return value





@dataclass

class WorkbenchROI:

    x: int

    y: int

    w: int

    h: int



    @classmethod

    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["WorkbenchROI"]:

        if not data:

            return None

        return cls(x=int(data["x"]), y=int(data["y"]), w=int(data["w"]), h=int(data["h"]))





@dataclass

class DetectionConfig:

    sample_fps: float = 2.0

    parent_sample_fps: Optional[float] = None

    micro_refine_sample_fps: float = 8.0

    enable_micro_refine_rescan: bool = False

    start_threshold: float = 0.6

    end_threshold: float = 0.3

    start_min_duration_sec: float = 2.0

    end_min_duration_sec: float = 5.0

    merge_gap_sec: float = 5.0

    min_segment_duration_sec: float = 5.0

    buffer_sec: float = 2.0

    motion_normalization: str = "adaptive"

    roi_mode: str = "manifest_or_default"

    detector_backend: str = "motion"

    yolo_preferred_view: str = "first_person"

    yolo_scan_both_views: bool = False

    yolo_model_path: Optional[str] = None

    yolo_first_person_model_path: Optional[str] = None

    yolo_third_person_model_path: Optional[str] = None

    yolo_conf: float = 0.25

    yolo_iou: float = 0.45

    yolo_device: str = "auto"

    yolo_imgsz: Optional[int] = None

    yolo_first_person_imgsz: Optional[int] = None

    yolo_third_person_imgsz: Optional[int] = None

    yolo_adaptive_imgsz: bool = True

    yolo_min_imgsz: int = 960

    yolo_max_imgsz: int = 1280

    yolo_continuity_frames: int = 3

    yolo_fallback_to_motion: bool = True

    yolo_class_thresholds: dict[str, float] = field(default_factory=dict)

    long_video_chunk_sec: float = 1800.0

    long_video_cache_enabled: bool = True

    long_video_resume: bool = True

    long_video_two_stage_sampling: bool = True

    long_video_stage1_sample_fps: Optional[float] = None

    long_video_stage2_sample_fps: Optional[float] = None

    long_video_cache_dir: Optional[str] = None

    expected_experiment_count: Optional[int] = None



    @classmethod

    def from_dict(cls, data: Optional[dict[str, Any]]) -> "DetectionConfig":

        if not data:

            return cls()

        base = cls()

        values = asdict(base)

        yolo_config = data.get("yolo_config") if isinstance(data.get("yolo_config"), dict) else {}

        long_video_config = data.get("long_video") if isinstance(data.get("long_video"), dict) else {}

        for key in (

            "parent_sample_fps",

            "micro_refine_sample_fps",

            "enable_micro_refine_rescan",

            "first_person_model_path",

            "third_person_model_path",

            "yolo_first_person_model_path",

            "yolo_third_person_model_path",

            "yolo_imgsz",

            "yolo_first_person_imgsz",

            "yolo_third_person_imgsz",

            "yolo_adaptive_imgsz",

            "yolo_min_imgsz",

            "yolo_max_imgsz",

        ):

            if key in yolo_config and yolo_config[key] is not None:

                if key == "first_person_model_path":

                    values["yolo_first_person_model_path"] = yolo_config[key]

                elif key == "third_person_model_path":

                    values["yolo_third_person_model_path"] = yolo_config[key]

                else:

                    values[key] = yolo_config[key]

        for key in (

            "long_video_chunk_sec",

            "long_video_cache_enabled",

            "long_video_resume",

            "long_video_two_stage_sampling",

            "long_video_stage1_sample_fps",

            "long_video_stage2_sample_fps",

            "long_video_cache_dir",

            "expected_experiment_count",

        ):

            short_key = key.replace("long_video_", "")

            if key in long_video_config and long_video_config[key] is not None:

                values[key] = long_video_config[key]

            elif short_key in long_video_config and long_video_config[short_key] is not None:

                values[key] = long_video_config[short_key]

        if values.get("parent_sample_fps") is not None and "sample_fps" not in data:

            values["sample_fps"] = values["parent_sample_fps"]

        for key in values:

            if key in data and data[key] is not None:

                values[key] = data[key]

        if values.get("parent_sample_fps") is not None:

            values["sample_fps"] = values["parent_sample_fps"]

        return cls(

            sample_fps=float(values["sample_fps"]),

            parent_sample_fps=float(values["parent_sample_fps"]) if values.get("parent_sample_fps") is not None else None,

            micro_refine_sample_fps=float(values["micro_refine_sample_fps"]),

            enable_micro_refine_rescan=bool(values["enable_micro_refine_rescan"]),

            start_threshold=float(values["start_threshold"]),

            end_threshold=float(values["end_threshold"]),

            start_min_duration_sec=float(values["start_min_duration_sec"]),

            end_min_duration_sec=float(values["end_min_duration_sec"]),

            merge_gap_sec=float(values["merge_gap_sec"]),

            min_segment_duration_sec=float(values["min_segment_duration_sec"]),

            buffer_sec=float(values["buffer_sec"]),

            motion_normalization=str(values["motion_normalization"]),

            roi_mode=str(values["roi_mode"]),

            detector_backend=str(values["detector_backend"]),

            yolo_preferred_view=str(values["yolo_preferred_view"]),

            yolo_scan_both_views=bool(values["yolo_scan_both_views"]),

            yolo_model_path=str(values["yolo_model_path"]) if values["yolo_model_path"] else None,

            yolo_first_person_model_path=str(values["yolo_first_person_model_path"]) if values["yolo_first_person_model_path"] else None,

            yolo_third_person_model_path=str(values["yolo_third_person_model_path"]) if values["yolo_third_person_model_path"] else None,

            yolo_conf=float(values["yolo_conf"]),

            yolo_iou=float(values["yolo_iou"]),

            yolo_device=str(values["yolo_device"]),

            yolo_imgsz=int(values["yolo_imgsz"]) if values.get("yolo_imgsz") is not None else None,

            yolo_first_person_imgsz=int(values["yolo_first_person_imgsz"]) if values.get("yolo_first_person_imgsz") is not None else None,

            yolo_third_person_imgsz=int(values["yolo_third_person_imgsz"]) if values.get("yolo_third_person_imgsz") is not None else None,

            yolo_adaptive_imgsz=bool(values["yolo_adaptive_imgsz"]),

            yolo_min_imgsz=int(values["yolo_min_imgsz"]),

            yolo_max_imgsz=int(values["yolo_max_imgsz"]),

            yolo_continuity_frames=int(values["yolo_continuity_frames"]),

            yolo_fallback_to_motion=bool(values["yolo_fallback_to_motion"]),

            yolo_class_thresholds={

                str(key): float(value)

                for key, value in dict(values.get("yolo_class_thresholds") or {}).items()

            },

            long_video_chunk_sec=float(values["long_video_chunk_sec"]),

            long_video_cache_enabled=bool(values["long_video_cache_enabled"]),

            long_video_resume=bool(values["long_video_resume"]),

            long_video_two_stage_sampling=bool(values["long_video_two_stage_sampling"]),

            long_video_stage1_sample_fps=float(values["long_video_stage1_sample_fps"])

            if values.get("long_video_stage1_sample_fps") is not None

            else None,

            long_video_stage2_sample_fps=float(values["long_video_stage2_sample_fps"])

            if values.get("long_video_stage2_sample_fps") is not None

            else None,

            long_video_cache_dir=str(values["long_video_cache_dir"]) if values.get("long_video_cache_dir") else None,

            expected_experiment_count=int(values["expected_experiment_count"])
            if values.get("expected_experiment_count") is not None
            else None,

        )





@dataclass

class MicroSegmentConfig:

    extract_micro_clips: bool = True

    extract_micro_keyframes: bool = True

    default_interaction_threshold: float = 0.50

    default_min_duration_sec: float = 0.8

    default_merge_gap_sec: float = 1.5

    class_thresholds: dict[str, dict[str, float]] = field(

        default_factory=lambda: {

            "balance": {"interaction_threshold": 0.55, "min_duration_sec": 0.8, "query_boost": 1.30},

            "reagent_bottle": {"interaction_threshold": 0.50, "min_duration_sec": 0.6, "query_boost": 1.25},

            "sample_bottle": {"interaction_threshold": 0.50, "min_duration_sec": 0.6, "query_boost": 1.25},

            "sample_bottle_blue": {"interaction_threshold": 0.50, "min_duration_sec": 0.6, "query_boost": 1.25},

            "bottle": {"interaction_threshold": 0.50, "min_duration_sec": 0.6, "query_boost": 1.20},

            "spatula": {"interaction_threshold": 0.50, "min_duration_sec": 0.4, "query_boost": 1.50},

            "pipette": {"interaction_threshold": 0.50, "min_duration_sec": 0.4, "query_boost": 1.55},

            "pipette_tip": {"interaction_threshold": 0.50, "min_duration_sec": 0.3, "query_boost": 1.60},

            "tube": {"interaction_threshold": 0.50, "min_duration_sec": 0.4, "query_boost": 1.35},

            "paper": {"interaction_threshold": 0.55, "min_duration_sec": 0.4, "query_boost": 1.10},

            "beaker": {"interaction_threshold": 0.50, "min_duration_sec": 0.6, "query_boost": 1.20},

            "container": {"interaction_threshold": 0.50, "min_duration_sec": 0.6, "query_boost": 1.20},

        }

    )

    micro_interaction_threshold: float = 0.50

    micro_object_thresholds: dict[str, float] = field(

        default_factory=lambda: {

            "spatula": 0.50,

            "pipette": 0.50,

            "pipette_tip": 0.50,

        }

    )

    micro_min_duration_sec: float = 1.0

    micro_merge_gap_sec: float = 1.5

    micro_object_change_split: bool = True

    single_primary_object_timeline: bool = True

    allow_secondary_interaction_candidates: bool = False

    allowed_primary_objects: list[str] = field(default_factory=list)

    disabled_primary_objects: list[str] = field(default_factory=list)

    primary_switch_margin: float = 0.08

    primary_min_stable_sec: float = 0.3

    tracklet_primary_vote_enabled: bool = True

    tracklet_vote_window_sec: float = 0.8

    tracklet_vote_min_count: int = 2

    tracklet_vote_margin: float = 0.12

    tracklet_view_weights: dict[str, float] = field(default_factory=lambda: {"first_person": 1.0, "third_person": 0.95})

    tracklet_family_vote_enabled: bool = True

    tracklet_family_vote_weight: float = 0.25

    tracklet_family_vote_margin: float = 0.08

    tracklet_family_run_merge_enabled: bool = True

    micro_pre_buffer_sec: float = 0.5

    micro_post_buffer_sec: float = 1.0

    micro_peak_keyframe_required: bool = True

    sop_action_backcheck_enabled: bool = True

    sop_action_backcheck_objects: list[str] = field(default_factory=list)

    sop_action_min_valid_frames: int = 4

    same_object_merge_enabled: bool = True

    same_object_merge_gap_sec: float = 1.0

    max_merged_micro_duration_sec: float = 8.0

    merge_low_confidence_adjacent: bool = True



    @classmethod

    def from_dict(cls, data: Optional[dict[str, Any]]) -> "MicroSegmentConfig":

        if not data:

            return cls()

        base = asdict(cls())

        for key in base:

            if key in data and data[key] is not None:

                base[key] = data[key]

        if "default_interaction_threshold" not in data and "micro_interaction_threshold" in data:

            base["default_interaction_threshold"] = data["micro_interaction_threshold"]

        if "default_min_duration_sec" not in data and "micro_min_duration_sec" in data:

            base["default_min_duration_sec"] = data["micro_min_duration_sec"]

        if "default_merge_gap_sec" not in data and "micro_merge_gap_sec" in data:

            base["default_merge_gap_sec"] = data["micro_merge_gap_sec"]

        legacy_thresholds = dict(base.get("micro_object_thresholds") or {})

        class_thresholds = {

            str(key): {

                str(inner_key): float(inner_value)

                for inner_key, inner_value in dict(value or {}).items()

            }

            for key, value in dict(base.get("class_thresholds") or {}).items()

        }

        for key, value in legacy_thresholds.items():

            class_thresholds.setdefault(str(key), {})["interaction_threshold"] = float(value)

        return cls(

            extract_micro_clips=bool(base["extract_micro_clips"]),

            extract_micro_keyframes=bool(base["extract_micro_keyframes"]),

            default_interaction_threshold=float(base["default_interaction_threshold"]),

            default_min_duration_sec=float(base["default_min_duration_sec"]),

            default_merge_gap_sec=float(base["default_merge_gap_sec"]),

            class_thresholds=class_thresholds,

            micro_interaction_threshold=float(base["micro_interaction_threshold"]),

            micro_object_thresholds={

                str(key): float(value)

                for key, value in dict(base.get("micro_object_thresholds") or {}).items()

            },

            micro_min_duration_sec=float(base["micro_min_duration_sec"]),

            micro_merge_gap_sec=float(base["micro_merge_gap_sec"]),

            micro_object_change_split=bool(base["micro_object_change_split"]),

            single_primary_object_timeline=bool(base["single_primary_object_timeline"]),

            allow_secondary_interaction_candidates=bool(base["allow_secondary_interaction_candidates"]),

            allowed_primary_objects=[str(item) for item in list(base.get("allowed_primary_objects") or [])],

            disabled_primary_objects=[str(item) for item in list(base.get("disabled_primary_objects") or [])],

            primary_switch_margin=float(base["primary_switch_margin"]),

            primary_min_stable_sec=float(base["primary_min_stable_sec"]),

            tracklet_primary_vote_enabled=bool(base["tracklet_primary_vote_enabled"]),

            tracklet_vote_window_sec=float(base["tracklet_vote_window_sec"]),

            tracklet_vote_min_count=int(base["tracklet_vote_min_count"]),

            tracklet_vote_margin=float(base["tracklet_vote_margin"]),

            tracklet_view_weights={

                str(key): float(value)

                for key, value in dict(base.get("tracklet_view_weights") or {}).items()

            },

            tracklet_family_vote_enabled=bool(base["tracklet_family_vote_enabled"]),

            tracklet_family_vote_weight=float(base["tracklet_family_vote_weight"]),

            tracklet_family_vote_margin=float(base["tracklet_family_vote_margin"]),

            tracklet_family_run_merge_enabled=bool(base["tracklet_family_run_merge_enabled"]),

            micro_pre_buffer_sec=float(base["micro_pre_buffer_sec"]),

            micro_post_buffer_sec=float(base["micro_post_buffer_sec"]),

            micro_peak_keyframe_required=bool(base["micro_peak_keyframe_required"]),

            sop_action_backcheck_enabled=bool(base["sop_action_backcheck_enabled"]),

            sop_action_backcheck_objects=[str(item) for item in list(base.get("sop_action_backcheck_objects") or [])],

            sop_action_min_valid_frames=int(base["sop_action_min_valid_frames"]),

            same_object_merge_enabled=bool(base["same_object_merge_enabled"]),

            same_object_merge_gap_sec=float(base["same_object_merge_gap_sec"]),

            max_merged_micro_duration_sec=float(base["max_merged_micro_duration_sec"]),

            merge_low_confidence_adjacent=bool(base["merge_low_confidence_adjacent"]),

        )





@dataclass

class VideoSource:

    name: str

    path: str

    start_time: str

    fps: Optional[float] = None

    offset_sec: float = 0.0

    role: Optional[str] = None

    camera_id: Optional[str] = None

    duration_sec: Optional[float] = None

    frames_csv_path: Optional[str] = None

    capture_start_time: Optional[str] = None

    capture_start_source: Optional[str] = None

    capture_start_status: Optional[str] = None



    @classmethod

    def from_dict(cls, name: str, data: dict[str, Any]) -> "VideoSource":

        return cls(

            name=str(data.get("view_id") or data.get("name") or name),

            path=str(data["path"]),

            start_time=str(data["start_time"]),

            fps=float(data["fps"]) if data.get("fps") is not None else None,

            offset_sec=float(data.get("offset_sec", 0.0)),

            role=str(data["role"]) if data.get("role") is not None else None,

            camera_id=str(data["camera_id"]) if data.get("camera_id") is not None else None,

            duration_sec=float(data["duration_sec"]) if data.get("duration_sec") is not None else None,

            frames_csv_path=str(data["frames_csv_path"]) if data.get("frames_csv_path") is not None else None,

            capture_start_time=str(data["capture_start_time"]) if data.get("capture_start_time") is not None else None,

            capture_start_source=str(data["capture_start_source"]) if data.get("capture_start_source") is not None else None,

            capture_start_status=str(data["capture_start_status"]) if data.get("capture_start_status") is not None else None,

        )





@dataclass

class TranscriptSource:

    path: str

    start_time: str

    offset_sec: float = 0.0



    @classmethod

    def from_dict(cls, data: Optional[dict[str, Any]]) -> Optional["TranscriptSource"]:

        if not data:

            return None

        return cls(

            path=str(data["path"]),

            start_time=str(data["start_time"]),

            offset_sec=float(data.get("offset_sec", 0.0)),

        )





@dataclass

class SessionVideos:

    third_person: VideoSource

    first_person: Optional[VideoSource] = None

    extra_views: dict[str, VideoSource] = field(default_factory=dict)



    def all_sources(self) -> dict[str, VideoSource]:

        sources = {"third_person": self.third_person}

        if self.first_person is not None:

            sources["first_person"] = self.first_person

        for view_id, source in self.extra_views.items():

            if view_id not in sources:

                sources[view_id] = source

        return sources



    def get(self, view_id: str | None) -> Optional[VideoSource]:

        if not view_id:

            return None

        return self.all_sources().get(str(view_id))





@dataclass

class InputEventSource:

    path: str

    source_type: str

    event_type: str

    modality: Optional[str] = None

    start_time: Optional[str] = None

    offset_sec: float = 0.0

    latency_sec: float = 0.0

    required: bool = False



    @classmethod

    def from_dict(cls, name: str, data: str | dict[str, Any]) -> "InputEventSource":

        values = {"path": data} if isinstance(data, str) else dict(data)

        source_type, event_type, modality = _input_source_defaults(name)

        return cls(

            path=str(values["path"]),

            source_type=str(values.get("source_type") or values.get("source") or source_type),

            event_type=str(values.get("event_type") or event_type),

            modality=str(values["modality"]) if values.get("modality") is not None else modality,

            start_time=str(values["start_time"]) if values.get("start_time") is not None else None,

            offset_sec=float(values.get("offset_sec", 0.0) or 0.0),

            latency_sec=float(values.get("latency_sec", 0.0) or 0.0),

            required=bool(values.get("required", False)),

        )





def _input_source_defaults(name: str) -> tuple[str, str, str | None]:

    normalized = str(name).strip().lower()

    if normalized in {"user", "user_text", "user_text_events", "user_events", "manual_notes"}:

        return "user_text", "user_text", "text"

    if normalized in {"ai", "assistant", "ai_reply", "ai_reply_events", "ai_events"}:

        return "ai_reply", "ai_reply", "text"

    if normalized in {"upload", "uploads", "upload_events"}:

        return "upload", "upload", None

    if normalized in {"sop", "sops", "sop_records", "sop_source", "sop_sources"}:

        return "sop", "sop_record", "text"

    if normalized in {

        "database",

        "databases",

        "database_records",

        "history",

        "history_database",

        "history_records",

        "history_sources",

    }:

        return "database", "database_record", "text"

    return normalized, normalized, None





def _parse_input_sources(data: dict[str, Any]) -> dict[str, InputEventSource]:

    sources: dict[str, InputEventSource] = {}

    raw_sources = data.get("input_sources")

    if isinstance(raw_sources, dict):

        for name, value in raw_sources.items():

            if value:

                sources[str(name)] = InputEventSource.from_dict(str(name), value)



    legacy_keys = {

        "user_text_events": "user_text_events",

        "user_events": "user_text_events",

        "user_events_path": "user_text_events",

        "ai_reply_events": "ai_reply_events",

        "ai_events": "ai_reply_events",

        "ai_events_path": "ai_reply_events",

        "upload_events": "upload_events",

        "uploads": "upload_events",

        "uploads_path": "upload_events",

        "sop": "sop",

        "sops": "sop",

        "sop_path": "sop",

        "sop_paths": "sop",

        "sop_sources": "sop",

        "database": "database",

        "databases": "database",

        "database_path": "database",

        "database_paths": "database",

        "history_database": "database",

        "history_source": "database",

        "history_sources": "database",

    }

    for key, source_name in legacy_keys.items():

        value = data.get(key)

        if not value or source_name in sources:

            continue

        if isinstance(value, list):

            for index, item in enumerate(value, start=1):

                if item:

                    sources[f"{source_name}_{index:03d}"] = InputEventSource.from_dict(source_name, item)

        else:

            sources[source_name] = InputEventSource.from_dict(source_name, value)

    return sources





@dataclass

class SessionManifest:

    session_id: str

    session_start_time: str

    videos: SessionVideos

    transcript: Optional[TranscriptSource] = None

    workbench_roi: Optional[WorkbenchROI] = None

    detection_config: DetectionConfig = field(default_factory=DetectionConfig)

    micro_segment_config: MicroSegmentConfig = field(default_factory=MicroSegmentConfig)

    input_sources: dict[str, InputEventSource] = field(default_factory=dict)

    config: dict[str, Any] = field(default_factory=dict)

    config_path: Optional[str] = None

    output_dir: str = "data/sessions/default"



    @classmethod

    def from_dict(cls, data: dict[str, Any]) -> "SessionManifest":

        videos_data = data["videos"]

        known_video_keys = {"third_person", "first_person"}

        extra_views = {

            str(key): VideoSource.from_dict(str(key), value)

            for key, value in videos_data.items()

            if key not in known_video_keys and isinstance(value, dict)

        }

        videos = SessionVideos(

            third_person=VideoSource.from_dict("third_person", videos_data["third_person"]),

            first_person=VideoSource.from_dict("first_person", videos_data["first_person"])

            if videos_data.get("first_person")

            else None,

            extra_views=extra_views,

        )

        input_sources = _parse_input_sources(data)

        return cls(

            session_id=str(data["session_id"]),

            session_start_time=str(data["session_start_time"]),

            videos=videos,

            transcript=TranscriptSource.from_dict(data.get("transcript")),

            workbench_roi=WorkbenchROI.from_dict(data.get("workbench_roi")),

            detection_config=DetectionConfig.from_dict(data.get("detection_config")),

            micro_segment_config=MicroSegmentConfig.from_dict(data.get("micro_segment_config")),

            input_sources=input_sources,

            config=dict(data.get("config") or {}),

            config_path=str(data["config_path"]) if data.get("config_path") is not None else None,

            output_dir=str(data.get("output_dir") or f"data/sessions/{data['session_id']}"),

        )



    @classmethod

    def load(cls, path: str | Path) -> "SessionManifest":

        with Path(path).open("r", encoding="utf-8-sig") as handle:

            return cls.from_dict(json.load(handle))



    def to_json_dict(self) -> dict[str, Any]:

        return _jsonable(self)





@dataclass

class TranscriptUtterance:

    utterance_id: str

    start_sec: float

    end_sec: float

    text: str

    global_start_time: str

    global_end_time: str



    @classmethod

    def from_raw(cls, data: dict[str, Any], global_start_time: datetime, global_end_time: datetime) -> "TranscriptUtterance":

        return cls(

            utterance_id=str(data.get("utterance_id") or data.get("id")),

            start_sec=float(data["start_sec"]),

            end_sec=float(data["end_sec"]),

            text=str(data.get("text") or ""),

            global_start_time=global_start_time.isoformat(),

            global_end_time=global_end_time.isoformat(),

        )





@dataclass

class DetectedSegment:

    segment_id: str

    start_sec: float

    end_sec: float

    duration_sec: float

    global_start_time: str

    global_end_time: str

    avg_motion_score: float

    avg_active_score: float

    start_reason: str

    end_reason: str

    review_required: bool = False

    detector_backend: str = "motion"

    detector_source_view: str = "third_person"

    yolo_label_counts: dict[str, int] = field(default_factory=dict)

    yolo_interaction_count: int = 0

    boundary_confidence: float = 0.0

    boundary_support_count: int = 0

    boundary_source: str = ""

    start_ms: float | None = None

    end_ms: float | None = None

    decision_path: str = ""

    decision_trace: list[Any] = field(default_factory=list)

    fallback_used: bool = False

    fallback_reason: str = ""

    reason_code: str = ""

    raw_score: float = 0.0

    score: float = 0.0

    source: str = "segment"

    source_view: str | None = None

    detector_version: str = ""

    final_score: float = 0.0

    run_manifest_id: str | None = None

    evidence_link: str | None = None

    retrieval_boost_factors: dict[str, Any] = field(default_factory=dict)

    alignment_health: dict[str, Any] | None = None

    alignment_report: dict[str, Any] | None = None





@dataclass

class ClipReference:

    video_path: str

    clip_path: str

    local_start_sec: float

    local_end_sec: float





@dataclass

class CVDetectionSummary:

    avg_motion_score: float

    avg_active_score: float

    start_reason: str

    end_reason: str

    start_sec: float = 0.0

    end_sec: float = 0.0

    confidence: float = 0.0





@dataclass

class TextDescription:

    action_type: str = "unknown_operation"

    summary: str = ""

    tools: list[str] = field(default_factory=list)

    objects: list[str] = field(default_factory=list)

    numbers: list[str] = field(default_factory=list)





@dataclass

class SegmentIndexInfo:

    embedding_id: str

    index_text: str

    vector_store: str





@dataclass

class InteractionKeyframe:

    path: str

    view: str

    local_time_sec: float

    global_time: Optional[str] = None

    interaction: str = ""

    event_id: str = ""

    source: str = "yolo_frame_rows"

    labels: list[str] = field(default_factory=list)





@dataclass

class InteractionEvent:

    event_id: str

    view: str

    local_time_sec: float

    global_time: Optional[str]

    interaction: str

    hand_label: str

    object_label: str

    object_name: str

    confidence: float

    keyframe_path: Optional[str] = None

    source: str = "yolo_frame_rows"





@dataclass

class YoloInteraction:

    view: str

    local_time_sec: float

    global_time: Optional[str]

    interaction: str

    hand_label: str

    object_label: str

    object_name: str

    confidence: float

    source: str = "yolo_frame_rows"

    source_image_path: Optional[str] = None

    detections: list[dict[str, Any]] = field(default_factory=list)





@dataclass

class MicroSegmentView:

    clip_path: Optional[str]

    local_start_sec: float

    local_end_sec: float

    annotated_clip_path: Optional[str] = None





@dataclass

class MicroSegmentInteraction:

    interaction_type: str

    primary_object: str

    secondary_objects: list[str]

    detected_objects: list[str]

    avg_interaction_score: float

    max_interaction_score: float

    contact_start_sec: float

    peak_interaction_sec: float

    contact_end_sec: float

    evidence_frame_indices: list[int]

    avg_hand_object_distance: Optional[float] = None

    max_bbox_overlap: float = 0.0

    primary_object_family: Optional[str] = None

    primary_object_arbitration: str = "peak_frame"

    primary_object_vote_score: Optional[float] = None

    primary_object_vote_margin: Optional[float] = None

    primary_object_vote_counts: dict[str, int] = field(default_factory=dict)

    primary_object_vote_scores: dict[str, float] = field(default_factory=dict)

    peak_primary_object: Optional[str] = None

    secondary_actions: list[str] = field(default_factory=list)





@dataclass

class MicroSegmentKeyframes:

    contact_frame: Optional[str] = None

    peak_frame: Optional[str] = None

    release_frame: Optional[str] = None

    contact_frame_time_sec: Optional[float] = None

    peak_frame_time_sec: Optional[float] = None





@dataclass

class MicroSegmentQuality:

    confidence: str = "low"

    warnings: list[str] = field(default_factory=list)





@dataclass

class MicroSegmentTextDescription:

    action_type: str

    summary: str

    index_text: str





@dataclass

class MicroSegmentIndexInfo:

    index_level: str

    embedding_id: str





@dataclass

class MicroSegment:

    micro_segment_id: str

    parent_segment_id: str

    session_id: str

    display_order: int

    display_id: str

    start_sec: float

    end_sec: float

    duration_sec: float

    global_start_time: str

    global_end_time: str

    first_person: Optional[MicroSegmentView]

    third_person: MicroSegmentView

    interaction: MicroSegmentInteraction

    keyframes: MicroSegmentKeyframes

    dialogue_context: list[dict[str, Any]]

    text_description: MicroSegmentTextDescription

    index: MicroSegmentIndexInfo

    quality: MicroSegmentQuality = field(default_factory=MicroSegmentQuality)

    class_threshold: dict[str, float] = field(default_factory=dict)

    dialogue_context_available: bool = False

    dialogue_match_window_sec: float = 2.0

    dialogue_keywords: list[str] = field(default_factory=list)

    evidence: dict[str, Any] = field(default_factory=dict)

    window_audit: dict[str, Any] = field(default_factory=dict)

    asset_bindings: list[dict[str, Any]] = field(default_factory=list)

    yolo_evidence: list[dict[str, Any]] = field(default_factory=list)

    manual_corrected: bool = False

    manual_correction_note: Optional[str] = None





@dataclass

class KeyActionSegment:

    session_id: str

    segment_id: str

    global_start_time: str

    global_end_time: str

    duration_sec: float

    third_person: ClipReference

    first_person: Optional[ClipReference]

    cv_detection: CVDetectionSummary

    text_description: TextDescription

    dialogue_context: list[str]

    index: SegmentIndexInfo

    interaction_keyframes: list[InteractionKeyframe] = field(default_factory=list)

    interaction_events: list[InteractionEvent] = field(default_factory=list)

    yolo_interactions: list[YoloInteraction] = field(default_factory=list)

    yolo_label_counts: dict[str, int] = field(default_factory=dict)

    yolo_interaction_count: int = 0

    detector_backend: str = "motion"

    detector_source_view: str = "third_person"

    decision_path: str = ""

    decision_trace: list[str] = field(default_factory=list)

    fallback_used: bool = False

    fallback_reason: str = ""

    reason_code: str = ""

    raw_score: float = 0.0

    final_score: float = 0.0

    run_manifest_id: str | None = None

    evidence_link: str | None = None

    alignment_health: dict[str, Any] | None = None

    alignment_report: dict[str, Any] | None = None

    retrieval_boost_factors: dict[str, Any] = field(default_factory=dict)

    micro_segments: list[dict[str, Any]] = field(default_factory=list)

    asset_bindings: list[dict[str, Any]] = field(default_factory=list)

    dialogue_match_window_sec: float = 3.0

    dialogue_keywords: list[str] = field(default_factory=list)

    evidence: dict[str, Any] = field(default_factory=dict)





@dataclass

class VectorMetadata:

    embedding_id: str

    segment_id: str

    session_id: str

    index_text: str

    global_start_time: str

    global_end_time: str

    third_person_clip: str

    first_person_clip: Optional[str]

    related_dialogue: list[str]

    action_type: str

    detector_backend: str = "motion"

    detector_source_view: str = "third_person"

    source_view: str | None = None

    decision_path: str = ""

    decision_trace: list[str] = field(default_factory=list)

    fallback_used: bool = False

    fallback_reason: str = ""

    reason_code: str = ""

    raw_score: float = 0.0

    final_score: float = 0.0

    run_manifest_id: str | None = None

    evidence_link: str | None = None

    alignment_health: dict[str, Any] | None = None

    alignment_report: dict[str, Any] | None = None

    retrieval_boost_factors: dict[str, Any] = field(default_factory=dict)

    interaction_keyframes: list[dict[str, Any]] = field(default_factory=list)

    interaction_events: list[dict[str, Any]] = field(default_factory=list)

    yolo_interactions: list[dict[str, Any]] = field(default_factory=list)

    yolo_evidence: list[dict[str, Any]] = field(default_factory=list)

    asset_bindings: list[dict[str, Any]] = field(default_factory=list)

    visual_keywords: list[str] = field(default_factory=list)

    index_level: str = "segment"

    micro_segment_id: Optional[str] = None

    parent_segment_id: Optional[str] = None

    primary_object: Optional[str] = None

    interaction_type: Optional[str] = None

    secondary_objects: list[str] = field(default_factory=list)

    secondary_actions: list[str] = field(default_factory=list)

    window_audit: dict[str, Any] = field(default_factory=dict)

    detected_objects: list[str] = field(default_factory=list)

    keyframes: list[str] = field(default_factory=list)

    evidence: dict[str, Any] = field(default_factory=dict)

    evidence_level: Optional[str] = None

    evidence_reasons: list[str] = field(default_factory=list)

    limitations: list[str] = field(default_factory=list)

    dialogue_context_available: bool = False

    dialogue_match_window_sec: Optional[float] = None

    dialogue_keywords: list[str] = field(default_factory=list)





@dataclass

class FrameScore:

    time_sec: float

    motion_score: float

    active_score: float

    raw_score: float = 0.0

    probability: float = 0.0

    prob: float = 0.0

    prob_score: float = 0.0

    local_time_sec: Optional[float] = None

    global_time: Optional[str] = None

    raw_prob: float = 0.0

    motion_prob: float = 0.0

    keep: bool = False

    keep_score: float = 0.0

    roi: Optional[dict[str, int]] = None

    is_active: bool = False

    frame_index: int = 0





def to_json_dict(value: Any) -> dict[str, Any]:

    return _jsonable(value)





def write_jsonl(path: str | Path, rows: list[Any]) -> None:

    target = Path(path)

    target.parent.mkdir(parents=True, exist_ok=True)

    with target.open("w", encoding="utf-8") as handle:

        for row in rows:

            handle.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")





def read_jsonl(path: str | Path) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    with Path(path).open("r", encoding="utf-8") as handle:

        for line in handle:

            line = line.strip()

            if line:

                rows.append(json.loads(line))

    return rows
