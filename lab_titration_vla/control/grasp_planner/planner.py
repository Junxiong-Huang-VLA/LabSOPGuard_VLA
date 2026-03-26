from __future__ import annotations

from typing import Dict, List


def plan_grasp(target_xyz_m: List[float], pre_offset_m: float = 0.05) -> Dict[str, List[float]]:
    return {
        "pre_grasp_xyz_m": [target_xyz_m[0], target_xyz_m[1], target_xyz_m[2] + pre_offset_m],
        "grasp_xyz_m": [target_xyz_m[0], target_xyz_m[1], target_xyz_m[2]],
    }

