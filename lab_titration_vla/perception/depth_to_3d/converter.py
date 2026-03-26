from __future__ import annotations

from typing import Dict, List, Optional

from project_name.utils.spatial import camera_to_robot_xyz, pixel_to_camera_xyz


class DepthTo3DConverter:
    def center_to_robot_xyz(
        self,
        center_point: List[float],
        center_depth_m: float,
        intrinsics: Optional[Dict[str, float]],
        hand_eye_extrinsics: Optional[Dict[str, List[float]]],
    ) -> Optional[List[float]]:
        cam_xyz = pixel_to_camera_xyz(
            u=float(center_point[0]),
            v=float(center_point[1]),
            z_m=float(center_depth_m),
            intrinsics=intrinsics,
        )
        return camera_to_robot_xyz(cam_xyz, hand_eye_extrinsics)

