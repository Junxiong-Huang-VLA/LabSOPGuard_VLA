"""
RealSense D435i Integration for LabSOPGuard.

Provides:
1. RealSense live stream input (RGB + Depth)
2. Depth processing and 3D spatial analysis
3. Camera intrinsics/extrinsics management
4. Integration with lab_vla perception modules

Supports both live camera mode and recorded video mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RealSenseConfig:
    """RealSense camera configuration."""
    mode: str = "file"  # "live", "file", "mock"
    source: str = ""  # video file path or camera serial number
    width: int = 640
    height: int = 480
    fps: float = 30.0
    enable_depth: bool = True
    enable_point_cloud: bool = False

    # Calibration
    camera_intrinsics: Optional[Dict[str, Any]] = None
    hand_eye_extrinsics: Optional[Dict[str, Any]] = None

    # Depth processing
    depth_min_m: float = 0.1
    depth_max_m: float = 2.0
    depth_window_size: int = 5

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any]) -> "RealSenseConfig":
        """Load config from YAML dict (from configs/devices/default_devices.yaml)."""
        camera_cfg = yaml_data.get("camera", {})
        depth_cfg = yaml_data.get("depth_processing", {})
        calib_cfg = yaml_data.get("calibration", {})

        return cls(
            mode=str(camera_cfg.get("mode", "file")),
            source=str(camera_cfg.get("source", "")),
            width=int(camera_cfg.get("width", 640)),
            height=int(camera_cfg.get("height", 480)),
            fps=float(camera_cfg.get("fps", 30.0)),
            enable_depth=bool(camera_cfg.get("enable_depth", True)),
            enable_point_cloud=bool(camera_cfg.get("enable_point_cloud", False)),
            depth_min_m=float(depth_cfg.get("depth_min_m", 0.1)),
            depth_max_m=float(depth_cfg.get("depth_max_m", 2.0)),
            depth_window_size=int(depth_cfg.get("depth_window_size", 5)),
            camera_intrinsics=calib_cfg.get("camera_intrinsics"),
            hand_eye_extrinsics=calib_cfg.get("hand_eye_extrinsics"),
        )


@dataclass
class FrameData:
    """A single frame with RGB, depth, and metadata."""
    frame_id: int
    timestamp_sec: float
    rgb: np.ndarray  # BGR image
    depth: Optional[np.ndarray] = None  # 16-bit depth in mm
    point_cloud: Optional[np.ndarray] = None  # Nx3 array

    # Derived data
    depth_center_m: float = 0.0  # Depth at frame center
    depth_mean_m: float = 0.0  # Mean valid depth
    valid_depth_ratio: float = 0.0  # Ratio of valid depth pixels

    def has_depth(self) -> bool:
        return self.depth is not None


# ---------------------------------------------------------------------------
# Depth Processing
# ---------------------------------------------------------------------------

def _compute_depth_stats(
    depth: np.ndarray,
    min_m: float = 0.1,
    max_m: float = 2.0,
    window_size: int = 5,
) -> Dict[str, float]:
    """Compute depth statistics with robust window sampling.

    Returns dict with: center_depth_m, mean_depth_m, valid_depth_ratio, depth_std
    """
    if depth is None or depth.size == 0:
        return {"center_depth_m": 0.0, "mean_depth_m": 0.0,
                "valid_depth_ratio": 0.0, "depth_std": 0.0}

    # Convert to meters (RealSense depth is in mm, uint16)
    depth_m = depth.astype(np.float64) / 1000.0

    # Filter valid range
    valid_mask = (depth_m > min_m) & (depth_m < max_m) & np.isfinite(depth_m)
    valid_depth = depth_m[valid_mask]

    if valid_depth.size == 0:
        return {"center_depth_m": 0.0, "mean_depth_m": 0.0,
                "valid_depth_ratio": 0.0, "depth_std": 0.0}

    valid_ratio = float(valid_depth.size / depth_m.size)

    # Center depth (with window averaging for robustness)
    h, w = depth_m.shape[:2]
    cy, cx = h // 2, w // 2
    half_w = window_size // 2
    center_region = depth_m[
        max(0, cy - half_w):min(h, cy + half_w + 1),
        max(0, cx - half_w):min(w, cx + half_w + 1),
    ]
    center_valid = center_region[valid_mask[
        max(0, cy - half_w):min(h, cy + half_w + 1),
        max(0, cx - half_w):min(w, cx + half_w + 1),
    ]]
    center_depth = float(np.median(center_valid)) if center_valid.size > 0 else 0.0

    return {
        "center_depth_m": center_depth,
        "mean_depth_m": float(np.mean(valid_depth)),
        "valid_depth_ratio": valid_ratio,
        "depth_std": float(np.std(valid_depth)),
    }


def depth_to_point_cloud(
    depth: np.ndarray,
    fx: float, fy: float,
    cx: float, cy: float,
    depth_scale: float = 0.001,
) -> np.ndarray:
    """Convert depth image to 3D point cloud in camera frame.

    Args:
        depth: uint16 depth image (mm)
        fx, fy: focal lengths
        cx, cy: principal point
        depth_scale: scale factor (mm -> m for RealSense = 0.001)

    Returns:
        Nx3 point cloud array in meters
    """
    h, w = depth.shape[:2]
    u_coords = np.arange(w)
    v_coords = np.arange(h)
    u_grid, v_grid = np.meshgrid(u_coords, v_coords)

    depth_m = depth.astype(np.float64) * depth_scale
    valid = (depth_m > 0.01) & (depth_m < 5.0) & np.isfinite(depth_m)

    z = depth_m[valid]
    x = (u_grid[valid] - cx) * z / fx
    y = (v_grid[valid] - cy) * z / fy

    return np.stack([x, y, z], axis=-1)


def transform_points_to_base(
    points_camera: np.ndarray,
    rotation: List[float],
    translation: List[float],
) -> np.ndarray:
    """Transform points from camera frame to robot base frame.

    Args:
        points_camera: Nx3 points in camera frame
        rotation: 9-element rotation matrix (row-major)
        translation: 3-element translation vector

    Returns:
        Nx3 points in robot base frame
    """
    R = np.array(rotation).reshape(3, 3)
    t = np.array(translation).reshape(3, 1)

    return (R @ points_camera.T + t).T


# ---------------------------------------------------------------------------
# Camera Input Sources
# ---------------------------------------------------------------------------

class RealSenseStream:
    """RealSense D435i live stream or file-based input."""

    def __init__(self, config: RealSenseConfig):
        self.config = config
        self._pipeline = None
        self._align = None

    def _init_live(self) -> bool:
        """Initialize live RealSense pipeline."""
        try:
            import pyrealsense2 as rs

            self._pipeline = rs.pipeline()
            rs_config = rs.config()

            if self.config.source:
                rs_config.enable_device(self.config.source)

            rs_config.enable_stream(
                rs.stream.color,
                self.config.width, self.config.height,
                rs.format.bgr8, int(self.config.fps),
            )

            if self.config.enable_depth:
                rs_config.enable_stream(
                    rs.stream.depth,
                    self.config.width, self.config.height,
                    rs.format.z16, int(self.config.fps),
                )

            self._pipeline.start(rs_config)
            self._align = rs.align(rs.stream.color)
            return True

        except Exception:
            return False

    def _init_file(self) -> bool:
        """Initialize from video file using OpenCV."""
        try:
            import cv2
            self._cap = cv2.VideoCapture(self.config.source)
            return self._cap.isOpened()
        except Exception:
            return False

    def frames(self, max_frames: Optional[int] = None) -> Generator[FrameData, None, None]:
        """Generate frames from the configured source."""
        if self.config.mode == "live":
            yield from self._live_frames(max_frames)
        elif self.config.mode == "file":
            yield from self._file_frames(max_frames)
        else:
            yield from self._mock_frames(max_frames)

    def _live_frames(self, max_frames: Optional[int]) -> Generator[FrameData, None, None]:
        """Yield frames from live RealSense camera."""
        if not self._init_live():
            # Fall back to file mode if live fails
            yield from self._mock_frames(max_frames)
            return

        try:
            import pyrealsense2 as rs

            frame_id = 0
            while True:
                if max_frames and frame_id >= max_frames:
                    break

                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
                if self._align:
                    frames = self._align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                rgb = np.asanyarray(color_frame.get_data())
                timestamp = color_frame.get_timestamp() / 1000.0

                depth = None
                depth_stats = {}
                if self.config.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth = np.asanyarray(depth_frame.get_data())
                        depth_stats = _compute_depth_stats(
                            depth,
                            self.config.depth_min_m,
                            self.config.depth_max_m,
                            self.config.depth_window_size,
                        )

                pc = None
                if self.config.enable_point_cloud and depth is not None:
                    # Get intrinsics for point cloud
                    profile = self._pipeline.get_active_profile()
                    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                    intrinsics = color_profile.get_intrinsics()
                    pc = depth_to_point_cloud(
                        depth,
                        intrinsics.fx, intrinsics.fy,
                        intrinsics.ppx, intrinsics.ppy,
                    )

                yield FrameData(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    rgb=rgb,
                    depth=depth,
                    point_cloud=pc,
                    depth_center_m=depth_stats.get("center_depth_m", 0.0),
                    depth_mean_m=depth_stats.get("mean_depth_m", 0.0),
                    valid_depth_ratio=depth_stats.get("valid_depth_ratio", 0.0),
                )
                frame_id += 1

        except Exception:
            pass
        finally:
            if self._pipeline:
                self._pipeline.stop()

    def _file_frames(self, max_frames: Optional[int]) -> Generator[FrameData, None, None]:
        """Yield frames from video file (RGB only, no real depth)."""
        import cv2

        cap = cv2.VideoCapture(self.config.source)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.fps
        frame_id = 0

        try:
            while True:
                if max_frames and frame_id >= max_frames:
                    break

                ok, rgb = cap.read()
                if not ok:
                    break

                timestamp = frame_id / fps
                yield FrameData(
                    frame_id=frame_id,
                    timestamp_sec=timestamp,
                    rgb=rgb,
                    depth=None,
                    depth_center_m=0.0,
                    depth_mean_m=0.0,
                    valid_depth_ratio=0.0,
                )
                frame_id += 1
        finally:
            cap.release()

    def _mock_frames(self, max_frames: Optional[int]) -> Generator[FrameData, None, None]:
        """Yield mock frames for testing."""
        count = max_frames or 10
        for i in range(count):
            rgb = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            yield FrameData(
                frame_id=i,
                timestamp_sec=float(i / self.config.fps),
                rgb=rgb,
                depth=None,
            )

    def stop(self) -> None:
        """Release resources."""
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 3D Spatial Analysis
# ---------------------------------------------------------------------------

@dataclass
class SpatialAnalysis:
    """3D spatial analysis result for a frame."""
    frame_id: int
    timestamp_sec: float

    # Object positions in camera frame
    object_positions: Dict[str, List[float]] = field(default_factory=dict)  # label -> [x, y, z]
    object_positions_base: Dict[str, List[float]] = field(default_factory=dict)  # in robot base frame

    # Hand-object relations
    hand_object_distances: Dict[str, float] = field(default_factory=dict)  # label -> distance_m
    nearest_object: str = ""
    nearest_object_distance: float = 0.0

    # Workspace analysis
    workspace_center_m: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    workspace_radius_m: float = 0.0

    # Quality metrics
    depth_quality: str = "ok"  # "ok", "poor", "unavailable"
    calibration_confidence: float = 0.0


def analyze_spatial_relations(
    frame_data: FrameData,
    detections: List[Dict[str, Any]],
    config: RealSenseConfig,
) -> SpatialAnalysis:
    """Analyze 3D spatial relationships between detected objects.

    Args:
        frame_data: Frame with optional depth data
        detections: List of detection results with bbox and label
        config: Camera configuration with calibration

    Returns:
        SpatialAnalysis with 3D positions and relationships
    """
    analysis = SpatialAnalysis(
        frame_id=frame_data.frame_id,
        timestamp_sec=frame_data.timestamp_sec,
    )

    if not frame_data.has_depth():
        analysis.depth_quality = "unavailable"
        return analysis

    depth = frame_data.depth
    if depth is None:
        analysis.depth_quality = "unavailable"
        return analysis

    # Get camera intrinsics (from config or defaults)
    intrinsics = config.camera_intrinsics or {}
    fx = intrinsics.get("fx", 615.0)
    fy = intrinsics.get("fy", 615.0)
    cx = intrinsics.get("cx", config.width / 2.0)
    cy = intrinsics.get("cy", config.height / 2.0)

    # Compute depth quality
    stats = _compute_depth_stats(depth, config.depth_min_m, config.depth_max_m)
    if stats["valid_depth_ratio"] > 0.7:
        analysis.depth_quality = "ok"
    elif stats["valid_depth_ratio"] > 0.3:
        analysis.depth_quality = "poor"
    else:
        analysis.depth_quality = "unavailable"
        return analysis

    # For each detection, compute 3D position from bbox center depth
    half_w = config.depth_window_size // 2

    for det in detections:
        label = det.get("label", "unknown")
        bbox = det.get("bbox", [0, 0, 0, 0])
        if len(bbox) != 4:
            continue

        # Bbox center
        u = int((bbox[0] + bbox[2]) / 2)
        v = int((bbox[1] + bbox[3]) / 2)

        # Get depth at center (with window averaging)
        h, w = depth.shape[:2]
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        region = depth[
            max(0, v - half_w):min(h, v + half_w + 1),
            max(0, u - half_w):min(w, u + half_w + 1),
        ]
        depth_m = region.astype(np.float64) / 1000.0
        valid = depth_m[(depth_m > config.depth_min_m) & (depth_m < config.depth_max_m)]

        if valid.size == 0:
            continue

        z = float(np.median(valid))
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        analysis.object_positions[label] = [x, y, z]

        # Transform to base frame if calibration available
        extrinsics = config.hand_eye_extrinsics
        if extrinsics:
            rotation = extrinsics.get("rotation", [1, 0, 0, 0, 1, 0, 0, 0, 1])
            translation = extrinsics.get("translation", [0, 0, 0])
            pts = np.array([[x, y, z]])
            transformed = transform_points_to_base(pts, rotation, translation)
            analysis.object_positions_base[label] = transformed[0].tolist()

    return analysis
