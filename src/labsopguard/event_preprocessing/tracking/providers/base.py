from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

from labsopguard.event_preprocessing.schemas import DetectionFrame, Tracklet


@dataclass
class TrackingBackendInfo:
    name: str
    version: str
    available: bool = True
    notes: str = ""


class TrackingProvider(Protocol):
    backend_info: TrackingBackendInfo

    def track(self, frames: List[DetectionFrame]) -> List[Tracklet]:
        ...
