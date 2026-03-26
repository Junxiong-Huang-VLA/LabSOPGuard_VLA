from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


TRANSPARENT_CLASSES = {"glass_bottle", "beaker", "flask", "transparent_container"}


@dataclass
class ParsedObject:
    class_name: str
    confidence: float
    bbox: Optional[List[int]]
    region: Dict[str, Any]
    center_point: Optional[List[float]]
    quality_flags: List[str]


class ObjectParser:
    """Normalize detector outputs to stable, report-ready object representations."""

    def parse_object(self, obj: Dict[str, Any], image_shape: Tuple[int, int]) -> ParsedObject:
        h, w = image_shape
        class_name = str(obj.get("label", obj.get("class_name", "unknown")))
        confidence = float(obj.get("score", obj.get("confidence", 0.0)))
        bbox_raw = obj.get("bbox")
        segmentation = obj.get("segmentation")

        flags: List[str] = []
        bbox = None
        center = None

        if isinstance(bbox_raw, list) and len(bbox_raw) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox_raw]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                flags.append("invalid_bbox")
            else:
                bbox = [x1, y1, x2, y2]
                center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
        else:
            flags.append("missing_bbox")

        region: Dict[str, Any]
        if isinstance(segmentation, list) and len(segmentation) >= 3:
            region = {"type": "segmentation", "points": segmentation}
        elif bbox is not None:
            region = {"type": "bbox", "bbox": bbox}
        else:
            region = {"type": "unknown", "value": None}

        if class_name in TRANSPARENT_CLASSES:
            flags.append("depth_sensitive_transparent_or_reflective")

        return ParsedObject(
            class_name=class_name,
            confidence=confidence,
            bbox=bbox,
            region=region,
            center_point=center,
            quality_flags=flags,
        )

    def parse_batch(self, objects: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> List[ParsedObject]:
        return [self.parse_object(obj, image_shape=image_shape) for obj in objects]
