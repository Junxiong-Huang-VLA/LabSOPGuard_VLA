from __future__ import annotations

from typing import List

from labsopguard.event_preprocessing.schemas import DetectionFrame, Tracklet

from .base import TrackingBackendInfo, TrackingProvider
from .iou_baseline import IouBaselineTrackingProvider


class BoTSORTAdapter(TrackingProvider):
    def __init__(self, fallback: TrackingProvider | None = None) -> None:
        self.fallback = fallback or IouBaselineTrackingProvider()
        self.backend_info = TrackingBackendInfo(
            name="botsort_adapter",
            version="adapter_stub.v1",
            available=False,
            notes="BoT-SORT dependency/model not configured; falling back to iou_baseline.",
        )

    def track(self, frames: List[DetectionFrame]) -> List[Tracklet]:
        tracklets = self.fallback.track(frames)
        for track in tracklets:
            track.tracking_backend = f"{self.backend_info.name}->fallback:{getattr(self.fallback, 'backend_info').name}"
            track.tracking_backend_version = self.backend_info.version
        return tracklets
