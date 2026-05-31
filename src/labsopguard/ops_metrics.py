from __future__ import annotations

from typing import Any, Optional


try:
    from prometheus_client import Gauge
except Exception:  # pragma: no cover - optional dependency
    Gauge = None


def _make_gauge(name: str, description: str, labels: list[str]):
    if Gauge is None:
        return None
    try:
        return Gauge(name, description, labels)
    except ValueError:
        # The module can be imported by tests/reloaders more than once.
        return None


CAPTURE_ACTUAL_FPS = _make_gauge("labsopguard_capture_actual_fps", "Actual capture FPS per camera", ["experiment_id", "camera_id"])
CAPTURE_FRAME_COUNT = _make_gauge("labsopguard_capture_frame_count", "Captured frame count per camera", ["experiment_id", "camera_id"])
CAPTURE_DROP_RATE = _make_gauge("labsopguard_capture_drop_rate", "Estimated dropped-frame rate per camera", ["experiment_id", "camera_id"])
CAPTURE_DECODE_ERRORS = _make_gauge("labsopguard_capture_decode_errors", "Decode/read errors per camera", ["experiment_id", "camera_id"])
CAPTURE_RECONNECTS = _make_gauge("labsopguard_capture_reconnects", "Reconnect count per camera", ["experiment_id", "camera_id"])
CAPTURE_SEGMENTS = _make_gauge("labsopguard_capture_segments", "Recorded segment count per camera", ["experiment_id", "camera_id"])
CAPTURE_DISK_FREE_BYTES = _make_gauge("labsopguard_capture_disk_free_bytes", "Free disk bytes for capture output", ["experiment_id"])
MATERIAL_TOTAL_ITEMS = _make_gauge("labsopguard_material_total_items", "Total indexed material items", ["scope"])
MATERIAL_BROKEN_CLIPS = _make_gauge("labsopguard_material_broken_clip_refs", "Broken material clip references", ["scope"])


def set_capture_snapshot_metrics(snapshot: Any) -> None:
    labels = [str(snapshot.experiment_id), str(snapshot.camera_id)]
    total_slots = int(snapshot.frame_count) + int(snapshot.dropped_frame_count)
    drop_rate = int(snapshot.dropped_frame_count) / total_slots if total_slots else 0.0
    for gauge, value in (
        (CAPTURE_ACTUAL_FPS, snapshot.actual_fps),
        (CAPTURE_FRAME_COUNT, snapshot.frame_count),
        (CAPTURE_DROP_RATE, drop_rate),
        (CAPTURE_DECODE_ERRORS, snapshot.decode_error_count),
        (CAPTURE_RECONNECTS, snapshot.reconnect_count),
        (CAPTURE_SEGMENTS, snapshot.segment_count),
    ):
        if gauge is not None:
            gauge.labels(*labels).set(float(value or 0.0))


def set_capture_disk_free(experiment_id: str, free_bytes: Optional[int]) -> None:
    if CAPTURE_DISK_FREE_BYTES is not None and free_bytes is not None:
        CAPTURE_DISK_FREE_BYTES.labels(str(experiment_id)).set(float(free_bytes))


def set_material_health_metrics(scope: str, health: dict[str, Any]) -> None:
    if MATERIAL_TOTAL_ITEMS is not None:
        MATERIAL_TOTAL_ITEMS.labels(str(scope)).set(float(health.get("total_items") or 0))
    if MATERIAL_BROKEN_CLIPS is not None:
        MATERIAL_BROKEN_CLIPS.labels(str(scope)).set(float(health.get("broken_clip_reference_count") or 0))
