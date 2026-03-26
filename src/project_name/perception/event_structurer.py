from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from project_name.perception.object_parser import ParsedObject


def _iso_ts(ts_sec: float) -> str:
    return datetime.fromtimestamp(ts_sec, tz=timezone.utc).isoformat()


class EventStructurer:
    """Build traceable event schema for detection and violation events."""

    def build_detection_event(
        self,
        sample_id: str,
        camera_id: str,
        frame_id: int,
        timestamp_sec: float,
        obj: ParsedObject,
        sop_step: str = "unknown",
        depth_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "sample_id": sample_id,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "timestamp": _iso_ts(timestamp_sec),
            "timestamp_sec": float(timestamp_sec),
            "class_name": obj.class_name,
            "confidence": float(obj.confidence),
            "bbox": obj.bbox,
            "region": obj.region,
            "center_point": obj.center_point,
            "depth_info": depth_info,
            "event_type": "detection",
            "sop_step": sop_step,
            "violation_flag": False,
            "severity_level": "none",
            "trace": {
                "quality_flags": obj.quality_flags,
                "event_version": "v1",
            },
        }

    def build_violation_event(
        self,
        sample_id: str,
        camera_id: str,
        frame_id: int,
        timestamp_sec: float,
        violation: Dict[str, Any],
        related_target: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "sample_id": sample_id,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "timestamp": _iso_ts(timestamp_sec),
            "timestamp_sec": float(timestamp_sec),
            "class_name": related_target.get("class_name") if related_target else "sop_event",
            "confidence": float(related_target.get("confidence", 1.0)) if related_target else 1.0,
            "bbox": related_target.get("bbox") if related_target else None,
            "region": related_target.get("region") if related_target else {"type": "rule"},
            "center_point": related_target.get("center_point") if related_target else None,
            "depth_info": related_target.get("depth_info") if related_target else None,
            "event_type": str(violation.get("rule_id", "violation")),
            "sop_step": str(violation.get("rule_id", "unknown")),
            "violation_flag": True,
            "severity_level": str(violation.get("severity", "medium")),
            "violation_message": str(violation.get("message", "")),
            "trace": {
                "event_version": "v1",
                "source": "sop_engine",
            },
        }

    def align_multi_camera_timestamp(self, events: List[Dict[str, Any]], camera_offsets_ms: Dict[str, int]) -> List[Dict[str, Any]]:
        aligned = []
        for e in events:
            cam = str(e.get("camera_id", "cam0"))
            offset_ms = int(camera_offsets_ms.get(cam, 0))
            ts = float(e.get("timestamp_sec", 0.0)) + offset_ms / 1000.0
            e2 = dict(e)
            e2["timestamp_sec"] = ts
            e2["timestamp"] = _iso_ts(ts)
            e2.setdefault("trace", {})["camera_offset_ms"] = offset_ms
            aligned.append(e2)
        return aligned
