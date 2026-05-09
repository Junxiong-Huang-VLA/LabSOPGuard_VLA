from __future__ import annotations

from typing import List

from labsopguard.event_preprocessing.schemas import DetectionFrame, Tracklet
from labsopguard.event_preprocessing.tracking.multi_object_tracker import IouMultiObjectTracker

from .base import TrackingBackendInfo, TrackingProvider


class IouBaselineTrackingProvider(TrackingProvider):
    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 3) -> None:
        self.backend_info = TrackingBackendInfo(
            name="iou_baseline",
            version="1.0",
            notes="Class-aware IoU association baseline; replaceable provider contract.",
        )
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def track(self, frames: List[DetectionFrame]) -> List[Tracklet]:
        tracklets = IouMultiObjectTracker(iou_threshold=self.iou_threshold, max_missed=self.max_missed).apply(frames)
        for track in tracklets:
            track.tracking_backend = self.backend_info.name
            track.tracking_backend_version = self.backend_info.version
        return tracklets
