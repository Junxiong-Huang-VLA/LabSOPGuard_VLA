from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CameraDevice:
    device_id: str
    mode: str
    source: str
    depth_source: Optional[str]
    target_fps: float


@dataclass
class RobotDevice:
    device_id: str
    adapter: str
    enabled: bool
    metadata: Dict[str, Any]


@dataclass
class DeviceContext:
    camera: CameraDevice
    robot: RobotDevice
    calibration: Dict[str, Any]

    @classmethod
    def from_configs(
        cls,
        devices_cfg: Dict[str, Any],
        calibration_cfg: Dict[str, Any],
    ) -> "DeviceContext":
        camera_cfg = devices_cfg.get("camera", {})
        robot_cfg = devices_cfg.get("robot", {})
        camera = CameraDevice(
            device_id=str(camera_cfg.get("device_id", "camera_default")),
            mode=str(camera_cfg.get("mode", "mock")),
            source=str(camera_cfg.get("source", "mock")),
            depth_source=camera_cfg.get("depth_source"),
            target_fps=float(camera_cfg.get("target_fps", 10.0)),
        )
        robot = RobotDevice(
            device_id=str(robot_cfg.get("device_id", "robot_default")),
            adapter=str(robot_cfg.get("adapter", "mock")),
            enabled=bool(robot_cfg.get("enabled", False)),
            metadata=dict(robot_cfg.get("metadata", {})),
        )
        return cls(camera=camera, robot=robot, calibration=calibration_cfg)

