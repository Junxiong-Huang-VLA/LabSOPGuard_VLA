from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _polygon_to_mask(height: int, width: int, polygon: List[List[float]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if len(polygon) < 3:
        return mask

    # Ray-casting point-in-polygon test on bounding rectangle.
    xs = np.array([p[0] for p in polygon], dtype=np.float32)
    ys = np.array([p[1] for p in polygon], dtype=np.float32)
    min_x = max(0, int(np.floor(xs.min())))
    max_x = min(width - 1, int(np.ceil(xs.max())))
    min_y = max(0, int(np.floor(ys.min())))
    max_y = min(height - 1, int(np.ceil(ys.max())))
    if min_x > max_x or min_y > max_y:
        return mask

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            inside = False
            j = len(polygon) - 1
            for i in range(len(polygon)):
                xi, yi = polygon[i]
                xj, yj = polygon[j]
                intersects = ((yi > y) != (yj > y)) and (
                    x < (xj - xi) * (y - yi) / max((yj - yi), 1e-6) + xi
                )
                if intersects:
                    inside = not inside
                j = i
            mask[y, x] = inside
    return mask


class RegionDepthStats:
    """Robust depth stats in target region with segmentation->bbox fallback."""

    def compute(
        self,
        depth_m: np.ndarray | None,
        bbox: List[int],
        segmentation: Optional[List[List[float]]] = None,
    ) -> Dict[str, float | str]:
        if depth_m is None:
            return {
                "region_depth_mean": 0.0,
                "region_depth_median": 0.0,
                "valid_depth_ratio": 0.0,
                "depth_region_source": "no_depth",
            }

        h, w = depth_m.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return {
                "region_depth_mean": 0.0,
                "region_depth_median": 0.0,
                "valid_depth_ratio": 0.0,
                "depth_region_source": "invalid_bbox",
            }

        region = depth_m[y1 : y2 + 1, x1 : x2 + 1]
        source = "bbox"

        if segmentation:
            mask_full = _polygon_to_mask(h, w, segmentation)
            mask = mask_full[y1 : y2 + 1, x1 : x2 + 1]
            seg_values = region[mask]
            valid_seg = seg_values[np.isfinite(seg_values) & (seg_values > 0)]
            if valid_seg.size > 0:
                values = valid_seg
                source = "segmentation"
            else:
                values = region[np.isfinite(region) & (region > 0)]
                source = "segmentation_fallback_bbox"
        else:
            values = region[np.isfinite(region) & (region > 0)]

        total_pixels = float(region.size)
        valid_ratio = float(values.size / total_pixels) if total_pixels > 0 else 0.0

        if values.size == 0:
            return {
                "region_depth_mean": 0.0,
                "region_depth_median": 0.0,
                "valid_depth_ratio": 0.0,
                "depth_region_source": f"{source}_no_valid",
            }

        median = float(np.median(values))
        # Trim extremes to reduce transparent/reflective outliers.
        q10 = float(np.quantile(values, 0.10))
        q90 = float(np.quantile(values, 0.90))
        trimmed = values[(values >= q10) & (values <= q90)]
        robust_mean = float(np.mean(trimmed)) if trimmed.size > 0 else float(np.mean(values))

        return {
            "region_depth_mean": robust_mean,
            "region_depth_median": median,
            "valid_depth_ratio": valid_ratio,
            "depth_region_source": source,
        }
