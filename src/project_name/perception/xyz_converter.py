from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class XYZConverter:
    """Convert pixel + depth into camera-frame xyz when intrinsics are available."""

    def to_xyz(
        self,
        center_point: List[float],
        center_depth: float,
        camera_intrinsics: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[List[float]], str]:
        if center_depth <= 0:
            return None, "invalid_depth"

        if not camera_intrinsics:
            return None, "missing_intrinsics"

        required = ["fx", "fy", "cx", "cy"]
        if any(k not in camera_intrinsics for k in required):
            return None, "incomplete_intrinsics"

        fx = float(camera_intrinsics["fx"])
        fy = float(camera_intrinsics["fy"])
        cx = float(camera_intrinsics["cx"])
        cy = float(camera_intrinsics["cy"])
        if fx == 0 or fy == 0:
            return None, "invalid_intrinsics"

        u, v = float(center_point[0]), float(center_point[1])
        z = float(center_depth)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return [x, y, z], "ok"
