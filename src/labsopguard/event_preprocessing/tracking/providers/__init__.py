from __future__ import annotations

import os

from .base import TrackingBackendInfo, TrackingProvider
from .botsort_adapter import BoTSORTAdapter
from .bytetrack_adapter import ByteTrackAdapter
from .iou_baseline import IouBaselineTrackingProvider
from .ocsort_adapter import OCSORTAdapter
from .strong_sort_lite import StrongSortLiteTrackingProvider


def build_tracking_provider(name: str | None = None) -> TrackingProvider:
    requested = (name or os.getenv("LABSOPGUARD_TRACKING_BACKEND") or "strong_sort_lite").strip().lower()
    if requested in {"strong_sort_lite", "strongsort_lite", "strongsort", "motion"}:
        return StrongSortLiteTrackingProvider()
    if requested in {"iou", "iou_baseline", "baseline"}:
        return IouBaselineTrackingProvider()
    if requested in {"bytetrack", "byte_track"}:
        return ByteTrackAdapter()
    if requested in {"botsort", "bo_t_sort", "bot-sort"}:
        return BoTSORTAdapter()
    if requested in {"ocsort", "oc-sort"}:
        return OCSORTAdapter()
    return IouBaselineTrackingProvider()


__all__ = [
    "TrackingBackendInfo",
    "TrackingProvider",
    "IouBaselineTrackingProvider",
    "ByteTrackAdapter",
    "BoTSORTAdapter",
    "OCSORTAdapter",
    "StrongSortLiteTrackingProvider",
    "build_tracking_provider",
]
