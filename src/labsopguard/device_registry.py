from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from labsopguard.video_input_schema import normalize_video_inputs


def load_device_registry(path: str | Path) -> Dict[str, Any]:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.suffix.lower() in {".yaml", ".yml"} else json.loads(registry_path.read_text(encoding="utf-8"))
    return payload or {}


def video_inputs_from_registry(registry: Dict[str, Any], *, strict: bool = True) -> List[Dict[str, Any]]:
    cameras = registry.get("cameras") or []
    video_inputs = []
    for index, camera in enumerate(cameras):
        if not camera.get("enabled", True):
            continue
        metadata = camera.get("metadata") or {}
        video_inputs.append(
            {
                "video_index": camera.get("video_index", index),
                "camera_id": camera.get("camera_id") or camera.get("device_id") or f"camera_{index:02d}",
                "source": camera.get("source"),
                "video_path": camera.get("source"),
                "source_type": camera.get("source_type") or camera.get("transport") or "rtsp",
                "sync_group": camera.get("sync_group"),
                "expected_fps": camera.get("expected_fps") or camera.get("target_fps"),
                "hardware_timecode_start_sec": camera.get("hardware_timecode_start_sec"),
                "sync_board_offset_sec": camera.get("sync_board_offset_sec"),
                "ptp_epoch_start_sec": camera.get("ptp_epoch_start_sec"),
                "ntp_epoch_start_sec": camera.get("ntp_epoch_start_sec"),
                "clock_drift_ppm": camera.get("clock_drift_ppm"),
                "sync_anchors": camera.get("sync_anchors") or [],
                "metadata": {
                    "vendor": camera.get("vendor") or metadata.get("vendor"),
                    "model": camera.get("model") or metadata.get("model"),
                    "serial_number": camera.get("serial_number") or metadata.get("serial_number"),
                    "sdk_adapter": camera.get("sdk_adapter") or metadata.get("sdk_adapter"),
                    **metadata,
                },
            }
        )
    normalized, _warnings = normalize_video_inputs(video_inputs, strict=strict)
    return normalized


def write_video_inputs_from_registry(registry_path: str | Path, output_path: str | Path, *, strict: bool = True) -> List[Dict[str, Any]]:
    registry = load_device_registry(registry_path)
    video_inputs = video_inputs_from_registry(registry, strict=strict)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(video_inputs, ensure_ascii=False, indent=2), encoding="utf-8")
    return video_inputs
