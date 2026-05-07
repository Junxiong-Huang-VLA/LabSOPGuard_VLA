from __future__ import annotations

from typing import Any, Dict, List, Tuple


VIDEO_INPUT_SCHEMA_VERSION = "video_input.v1"
ALLOWED_SOURCE_TYPES = {"file", "rtsp", "usb", "http", "rtmp", "udp"}
LIVE_SOURCE_TYPES = {"rtsp", "usb", "http", "rtmp", "udp"}


class VideoInputValidationError(ValueError):
    pass


def _as_float(value: Any, field_name: str, errors: List[str]) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        errors.append(f"{field_name} must be numeric")
        return None


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_sync_anchors(value: Any, errors: List[str]) -> List[Dict[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        errors.append("sync_anchors must be a list")
        return []

    anchors: List[Dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"sync_anchors[{index}] must be an object")
            continue
        local_time = _as_float(item.get("local_time_sec"), f"sync_anchors[{index}].local_time_sec", errors)
        reference_time = _as_float(item.get("reference_time_sec"), f"sync_anchors[{index}].reference_time_sec", errors)
        if local_time is None or reference_time is None:
            continue
        anchor = dict(item)
        anchor["local_time_sec"] = round(local_time, 6)
        anchor["reference_time_sec"] = round(reference_time, 6)
        anchor["method"] = _clean_text(anchor.get("method")) or "manual"
        if anchor.get("confidence") is not None:
            confidence = _as_float(anchor.get("confidence"), f"sync_anchors[{index}].confidence", errors)
            if confidence is not None:
                anchor["confidence"] = max(0.0, min(1.0, confidence))
        anchors.append(anchor)
    return anchors


def normalize_video_input(
    item: Dict[str, Any],
    index: int = 0,
    *,
    strict: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(item, dict):
        raise VideoInputValidationError("video input must be an object")

    errors: List[str] = []
    warnings: List[str] = []
    source = _clean_text(item.get("video_path") or item.get("file_path") or item.get("path") or item.get("source"))
    if not source:
        errors.append("source/video_path is required")

    source_type = (_clean_text(item.get("source_type") or item.get("ingest_mode")) or "file").lower()
    if source_type not in ALLOWED_SOURCE_TYPES:
        errors.append(f"source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}")

    video_index = item.get("video_index", index)
    try:
        video_index = int(video_index)
    except (TypeError, ValueError):
        errors.append("video_index must be an integer")
        video_index = index

    has_start_offset = "start_offset_sec" in item and item.get("start_offset_sec") not in (None, "")
    start_offset = _as_float(item.get("start_offset_sec"), "start_offset_sec", errors)
    capture_duration = _as_float(item.get("capture_duration_sec"), "capture_duration_sec", errors)
    clock_scale = _as_float(item.get("clock_scale"), "clock_scale", errors)
    clock_drift = _as_float(item.get("clock_drift_ppm"), "clock_drift_ppm", errors)
    hardware_timecode = _as_float(item.get("hardware_timecode_start_sec"), "hardware_timecode_start_sec", errors)
    sync_board_offset = _as_float(item.get("sync_board_offset_sec"), "sync_board_offset_sec", errors)
    sync_anchors = _normalize_sync_anchors(item.get("sync_anchors"), errors)

    if source_type in LIVE_SOURCE_TYPES and capture_duration is not None and capture_duration <= 0:
        errors.append("capture_duration_sec must be positive for live sources")
    if clock_scale is not None and clock_scale <= 0:
        errors.append("clock_scale must be positive")

    camera_id = _clean_text(item.get("camera_id"))
    if not camera_id:
        camera_id = f"camera_{video_index:02d}"
        warnings.append("camera_id was generated from video_index")

    if errors and strict:
        raise VideoInputValidationError("; ".join(errors))

    normalized = dict(item)
    normalized["schema_version"] = VIDEO_INPUT_SCHEMA_VERSION
    normalized["video_index"] = video_index
    normalized["video_path"] = source or ""
    normalized["source"] = _clean_text(item.get("source")) or source or ""
    normalized["source_type"] = source_type
    normalized["ingest_mode"] = _clean_text(item.get("ingest_mode")) or source_type
    normalized["camera_id"] = camera_id
    normalized["sync_group"] = _clean_text(item.get("sync_group"))
    normalized["is_live_source"] = source_type in LIVE_SOURCE_TYPES
    normalized["start_offset_sec"] = round(start_offset or 0.0, 6)
    normalized["offset_source"] = (
        _clean_text(item.get("offset_source"))
        or ("explicit" if has_start_offset else "default_zero")
    )

    if capture_duration is not None:
        normalized["capture_duration_sec"] = round(capture_duration, 6)
    if sync_anchors:
        normalized["sync_anchors"] = sync_anchors
    elif "sync_anchors" in normalized:
        normalized["sync_anchors"] = []
    if _clean_text(item.get("sync_method")):
        normalized["sync_method"] = _clean_text(item.get("sync_method"))
    if hardware_timecode is not None:
        normalized["hardware_timecode_start_sec"] = round(hardware_timecode, 6)
    if sync_board_offset is not None:
        normalized["sync_board_offset_sec"] = round(sync_board_offset, 6)
    if clock_scale is not None:
        normalized["clock_scale"] = round(clock_scale, 9)
    if clock_drift is not None:
        normalized["clock_drift_ppm"] = round(clock_drift, 6)
    if errors:
        normalized["validation_errors"] = errors
    if warnings:
        normalized["validation_warnings"] = warnings
    return normalized, warnings


def normalize_video_inputs(
    items: List[Dict[str, Any]],
    *,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    normalized: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for index, item in enumerate(items or []):
        normalized_item, item_warnings = normalize_video_input(item, index=index, strict=strict)
        normalized.append(normalized_item)
        warnings.extend(f"video_inputs[{index}]: {warning}" for warning in item_warnings)
    return normalized, warnings
