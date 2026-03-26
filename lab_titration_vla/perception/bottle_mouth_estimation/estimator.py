from __future__ import annotations

from typing import Dict, List, Optional


def estimate_bottle_mouth_point(bbox_xyxy: List[int], keypoints17: Optional[List[List[float]]] = None) -> Dict[str, float]:
    """
    Lightweight bottle-mouth proxy.
    Priority:
    1) use top-most confident keypoint (if any)
    2) use bbox top-center
    """
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    if keypoints17:
        valid = [kp for kp in keypoints17 if isinstance(kp, list) and len(kp) >= 3 and float(kp[2]) > 0.2]
        if valid:
            top = min(valid, key=lambda p: float(p[1]))
            return {"x": float(top[0]), "y": float(top[1]), "source": "keypoint"}
    return {"x": float((x1 + x2) / 2.0), "y": float(y1), "source": "bbox_top_center"}

