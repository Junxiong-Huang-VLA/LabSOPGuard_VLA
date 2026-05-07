from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ReadStatus(str, Enum):
    OK = "ok"
    TIMEOUT = "timeout"
    CAMERA_OFFLINE = "camera_offline"
    STREAM_UNAVAILABLE = "stream_unavailable"
    PAIR_FAILED = "pair_failed"
    CALIBRATION_MISSING = "calibration_missing"
    INTERNAL_ERROR = "internal_error"


class PairMode(str, Enum):
    STRICT = "strict"
    REALTIME = "realtime"


class StreamName(str, Enum):
    RGB = "rgb"
    DEPTH_RAW = "depth_raw"
    DEPTH_ALIGNED_TO_RGB = "depth_aligned_to_rgb"
    DEPTH_COLOR = "depth_color"


@dataclass(frozen=True)
class SenderInfo:
    sender_id: str
    sender_name: str
    online: bool
    camera_count: int


@dataclass(frozen=True)
class CameraInfo:
    sender_id: str
    sender_name: str
    camera_id: str
    camera_name: str
    online: bool


@dataclass(frozen=True)
class FrameMeta:
    sender_id: str
    camera_id: str
    stream_name: str
    frame_id: int
    timestamp_us: int
    online: bool
    status_code: str | None = None


@dataclass(frozen=True)
class PairMeta:
    sender_id: str
    camera_id: str
    rgb_frame_id: int
    depth_frame_id: int
    rgb_timestamp_us: int
    depth_timestamp_us: int
    pair_mode: str
    pair_complete: bool
    aligned: bool
    online: bool
    pair_delta_ms: float | None = None
    drop_count: int | None = None
    status_code: str | None = None


@dataclass(frozen=True)
class CalibrationInfo:
    rgb_intrinsics: dict
    depth_intrinsics: dict
    rgb_to_depth: dict
    depth_to_rgb: dict
    depth_scale: float
    depth_unit: str


@dataclass(frozen=True)
class CameraCapabilities:
    supports_rgb: bool
    supports_depth_raw: bool
    supports_depth_aligned_to_rgb: bool
    supports_depth_color: bool
    supports_strict_pair: bool
    has_calibration: bool


@dataclass(frozen=True)
class StreamProfile:
    stream_name: str
    width: int
    height: int
    fps: float
    pixel_format: str


@dataclass(frozen=True)
class CameraStatus:
    sender_id: str
    camera_id: str
    online: bool
    rgb_available: bool
    depth_available: bool
    depth_color_available: bool
    pair_mode_supported: list[str]
    calibration_available: bool
    last_frame_ts_us: int | None
    last_status_code: str | None
