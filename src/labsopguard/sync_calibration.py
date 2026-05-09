from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from labsopguard.time_sync import SyncAnchor, TimeSyncCalibrator


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def anchors_from_hardware_metadata(video_input: Dict[str, Any]) -> List[SyncAnchor]:
    camera_id = str(video_input.get("camera_id") or "camera")
    anchors = TimeSyncCalibrator.anchors_from_descriptor(camera_id, video_input)
    for field, method in (
        ("hardware_timecode_start_sec", "hardware_timecode"),
        ("sync_board_offset_sec", "sync_board"),
        ("ptp_epoch_start_sec", "ptp"),
        ("ntp_epoch_start_sec", "ntp"),
    ):
        value = _as_float(video_input.get(field))
        if value is None:
            continue
        anchors.append(
            SyncAnchor(
                camera_id=camera_id,
                local_time_sec=0.0,
                reference_time_sec=value,
                method=method,
                confidence=0.95 if method in {"hardware_timecode", "ptp"} else 0.85,
                metadata={"source_field": field},
            )
        )
    drift = _as_float(video_input.get("clock_drift_ppm"))
    if drift is not None:
        duration = _as_float(video_input.get("capture_duration_sec")) or 3600.0
        scale = 1.0 + drift / 1_000_000.0
        offset = (
            _as_float(video_input.get("hardware_timecode_start_sec"))
            or _as_float(video_input.get("sync_board_offset_sec"))
            or _as_float(video_input.get("start_offset_sec"))
            or 0.0
        )
        anchors.append(
            SyncAnchor(
                camera_id=camera_id,
                local_time_sec=duration,
                reference_time_sec=duration * scale + offset,
                method="hardware_timecode",
                confidence=0.7,
                metadata={"source_field": "clock_drift_ppm", "clock_drift_ppm": drift},
            )
        )
    return anchors


def detect_flash_anchors_from_video(
    video_path: str | Path,
    *,
    camera_id: str,
    sample_interval_sec: float = 0.1,
    z_threshold: float = 3.0,
    max_frames: int = 3000,
) -> List[SyncAnchor]:
    path = Path(video_path)
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps * sample_interval_sec)))
    frames = []
    frame_index = 0
    sampled = 0
    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                frames.append((frame_index / fps, frame))
                sampled += 1
            frame_index += 1
    finally:
        cap.release()
    return TimeSyncCalibrator.detect_flash_anchors(frames, camera_id=camera_id, z_threshold=z_threshold)


def build_sync_calibration_report(
    video_inputs: List[Dict[str, Any]],
    *,
    reference_camera_id: str = "global",
    auto_flash: bool = False,
) -> Dict[str, Any]:
    calibrator = TimeSyncCalibrator(reference_camera_id=reference_camera_id)
    anchors: List[SyncAnchor] = []
    for item in video_inputs:
        camera_id = str(item.get("camera_id") or "camera")
        camera_anchors = anchors_from_hardware_metadata(item)
        if auto_flash and item.get("source_type") == "file":
            camera_anchors.extend(detect_flash_anchors_from_video(item.get("video_path") or item.get("source"), camera_id=camera_id))
        anchors.extend(camera_anchors)
    calibrator.extend(anchors)
    camera_ids = sorted({str(item.get("camera_id") or "camera") for item in video_inputs})
    profiles = [calibrator.fit(camera_id).to_dict() for camera_id in camera_ids]
    return {
        "schema_version": "sync_calibration_report.v1",
        "reference_camera_id": reference_camera_id,
        "anchor_count": len(anchors),
        "anchors": [anchor.to_dict() for anchor in anchors],
        "profiles": profiles,
        "max_residual_error_sec": max([profile.get("residual_error_sec", 0.0) for profile in profiles] or [0.0]),
        "calibration_modes": sorted({profile.get("method", "manual") for profile in profiles}),
    }


def build_sync_calibration_report_from_file(path: str | Path, *, auto_flash: bool = False) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        video_inputs = payload.get("video_inputs") or payload.get("streams") or payload.get("cameras") or []
    else:
        video_inputs = payload
    return build_sync_calibration_report(video_inputs, auto_flash=auto_flash)


def write_sync_calibration_report(report: Dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
