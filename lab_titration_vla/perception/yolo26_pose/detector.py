from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from project_name.detection.multi_level_detector import MultiLevelDetector
from project_name.video.capture import FramePacket


class YOLO26PoseDetector:
    """Layer-1 realtime detector wrapper for the new structure."""

    def __init__(self, confidence_threshold: float = 0.25) -> None:
        self.detector = MultiLevelDetector(confidence_threshold=confidence_threshold)

    def infer_frame(self, frame: FramePacket) -> Dict[str, Any]:
        det = self.detector.detect(frame)
        return asdict(det)

