from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


class CenterDepthExtractor:
    def extract(
        self,
        depth_m: np.ndarray | None,
        center_point: List[float] | None,
        region_fallback_depth: float | None = None,
        max_radius: int = 8,
    ) -> Dict[str, float | int | str]:
        if depth_m is None or center_point is None:
            return {
                "center_depth": float(region_fallback_depth or 0.0),
                "center_depth_source": "no_depth_or_center",
                "center_depth_radius": 0,
                "valid_depth_ratio": 0.0,
            }

        h, w = depth_m.shape[:2]
        cx = max(0, min(int(round(center_point[0])), w - 1))
        cy = max(0, min(int(round(center_point[1])), h - 1))

        center = float(depth_m[cy, cx])
        if np.isfinite(center) and center > 0:
            return {
                "center_depth": center,
                "center_depth_source": "center_pixel",
                "center_depth_radius": 0,
                "valid_depth_ratio": 1.0,
            }

        for r in range(1, max_radius + 1):
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r + 1), min(h, cy + r + 1)
            patch = depth_m[y1:y2, x1:x2]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            ratio = float(valid.size / patch.size) if patch.size else 0.0
            if valid.size > 0:
                return {
                    "center_depth": float(np.median(valid)),
                    "center_depth_source": "neighbor_median",
                    "center_depth_radius": r,
                    "valid_depth_ratio": ratio,
                }

        return {
            "center_depth": float(region_fallback_depth or 0.0),
            "center_depth_source": "region_fallback",
            "center_depth_radius": max_radius,
            "valid_depth_ratio": 0.0,
        }
