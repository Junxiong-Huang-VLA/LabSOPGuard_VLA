from __future__ import annotations

import os
import math
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping

from .action_detector import build_segments_from_scores
from .config import DetectorConfig
from .schemas import DetectedSegment, FrameScore, VideoSource


HAND_LABELS = frozenset({"gloved_hand", "hand"})
INTERACTION_OBJECT_LABELS = frozenset(
    {
        "sample_bottle",
        "sample_bottle_blue",
        "reagent_bottle",
        "reagent_bottle_open",
        "bottle_cap",
        "balance",
        "beaker",
        "container",
        "magnetic_stirrer",
        "magnetic_stir_bar",
        "spatula",
        "pipette",
        "pipette_tip",
        "paper",
        "tube",
        "tube_cap",
        "tube_rack",
    }
)
EXPERIMENT_CONTEXT_LABELS = HAND_LABELS | INTERACTION_OBJECT_LABELS | frozenset({"lab_coat", "ppe_storage"})
PHYSICAL_EVIDENCE_LABELS = HAND_LABELS | INTERACTION_OBJECT_LABELS | frozenset({"lab_coat", "ppe_storage"})
_YOLO_MODEL_CACHE: dict[str, Any] = {}
_YOLO_MODEL_CACHE_LOCK = threading.Lock()
_YOLO_MODEL_PREDICT_LOCKS: dict[int, threading.Lock] = {}
_YOLO_MODEL_PREDICT_LOCKS_LOCK = threading.Lock()
_YOLO_GPU_PREDICT_LOCK = threading.Lock()

_LABEL_ALIASES = {
    "glove": "gloved_hand",
    "gloves": "gloved_hand",
    "gloved_hands": "gloved_hand",
    "person_hand": "hand",
    "hands": "hand",
    "sample_bottle_blue": "sample_bottle_blue",
    "blue_sample_bottle": "sample_bottle_blue",
    "sample_bottle": "sample_bottle",
    "sample_vial": "sample_bottle",
    "vial": "sample_bottle",
    "bottle": "sample_bottle",
    "reagent": "reagent_bottle",
    "reagent_bottle": "reagent_bottle",
    "reagent_bottle_open": "reagent_bottle_open",
    "open_reagent_bottle": "reagent_bottle_open",
    "bottle_cap": "bottle_cap",
    "cap": "bottle_cap",
    "electronic_balance": "balance",
    "scale": "balance",
    "weighing_scale": "balance",
    "panel": "panel",
    "display_panel": "panel",
    "equipment_panel": "panel",
    "weighing_paper": "paper",
    "paper": "paper",
    "spoon": "spatula",
    "scoop": "spatula",
    "spatula": "spatula",
    "pipette": "pipette",
    "pipette_tip": "pipette_tip",
    "spearhead": "pipette_tip",
    "tube": "tube",
    "test_tube": "tube",
    "tube_cap": "tube_cap",
    "tube-cap": "tube_cap",
    "tube_rack": "tube_rack",
    "rack": "tube_rack",
    "beaker": "beaker",
    "container": "container",
    "magnetic_stirrer": "magnetic_stirrer",
    "magnetic_stir_bar": "magnetic_stir_bar",
    "stir_bar": "magnetic_stir_bar",
    "magnetic_bar": "magnetic_stir_bar",
    "stirrer": "magnetic_stirrer",
    "ppe_storage": "ppe_storage",
    "ppe": "ppe_storage",
    "ppe_storage_box": "ppe_storage",
}


YoloFrameDetector = Callable[[Any], list[dict[str, Any]]]


@dataclass
class YoloScanResult:
    rows: list[dict[str, Any]]
    source_view: str
    video_path: str
    fps: float
    sample_fps: float
    sampled_frames: int
    source_frame_count: int
    duration_sec: float


def canonical_yolo_label(label: Any) -> str:
    value = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    while "__" in value:
        value = value.replace("__", "_")
    return _LABEL_ALIASES.get(value, value)


def _coerce_bbox(value: Any) -> list[float]:
    if value is None:
        return [0.0, 0.0, 0.0, 0.0]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    values = list(value)[:4]
    if len(values) < 4:
        values = values + [0.0] * (4 - len(values))
    x1, y1, x2, y2 = [float(item) for item in values]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _to_scalar(value: Any) -> float:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else 0.0
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def normalize_yolo_detection(detection: dict[str, Any]) -> dict[str, Any]:
    raw_label = detection.get("raw_label", detection.get("label", detection.get("name", "")))
    label = canonical_yolo_label(raw_label)
    bbox = _coerce_bbox(detection.get("bbox", detection.get("xyxy", [0, 0, 0, 0])))
    normalized = {
        "label": label,
        "raw_label": str(raw_label or label),
        "class_id": detection.get("class_id", detection.get("cls")),
        "confidence": round(float(detection.get("confidence", detection.get("conf", 0.0))), 6),
        "bbox": [round(float(value), 3) for value in bbox],
    }
    normalized["raw_score"] = normalized["confidence"]
    normalized["probability"] = normalized["confidence"]
    normalized["prob"] = normalized["confidence"]
    normalized["prob_score"] = normalized["confidence"]
    normalized["keep"] = bool(normalized["confidence"] >= float(detection.get("threshold", 0.0)))
    normalized["keep_score"] = 1.0 if normalized["keep"] else 0.0
    for key in (
        "track_id",
        "object_track_id",
        "tracklet_id",
        "tracklet_source",
        "source",
        "view",
        "source_view",
        "local_time_sec",
        "time_sec",
        "frame_index",
    ):
        if detection.get(key) is not None:
            normalized[key] = detection.get(key)
    return normalized


def _bbox_size_ratios(detection: dict[str, Any], frame_width: int | None, frame_height: int | None) -> tuple[float, float, float]:
    if not frame_width or not frame_height:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])[:4]]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    frame_area = max(1.0, float(frame_width) * float(frame_height))
    return width / max(1.0, float(frame_width)), height / max(1.0, float(frame_height)), (width * height) / frame_area


def _bbox_aspect_ratio(detection: dict[str, Any]) -> float:
    x1, y1, x2, y2 = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])[:4]]
    width = max(0.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return width / height


def _bbox_edge_contact(detection: dict[str, Any], frame_width: int | None, frame_height: int | None) -> tuple[bool, bool]:
    if not frame_width or not frame_height:
        return False, False
    x1, y1, x2, y2 = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])[:4]]
    margin_x = max(2.0, float(frame_width) * 0.015)
    margin_y = max(2.0, float(frame_height) * 0.015)
    touches_x = x1 <= margin_x or x2 >= float(frame_width) - margin_x
    touches_y = y1 <= margin_y or y2 >= float(frame_height) - margin_y
    return touches_x, touches_y


def _frame_size(frame: Any | None) -> tuple[int | None, int | None]:
    if frame is None:
        return None, None
    try:
        return int(frame.shape[1]), int(frame.shape[0])
    except Exception:
        return None, None


def _crop_stats(
    frame: Any | None,
    detection: dict[str, Any],
    *,
    cache: dict[tuple[int, int, int, int], dict[str, float | bool]] | None = None,
) -> dict[str, float | bool]:
    if frame is None:
        return {}
    try:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in detection.get("bbox", [0, 0, 0, 0])[:4]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            return {}
        cache_key = (x1, y1, x2, y2)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {}
        crop_h, crop_w = crop.shape[:2]
        max_dim = max(crop_w, crop_h)
        max_pixels = 96 * 96
        if max_dim > 128 or (crop_w * crop_h) > max_pixels:
            scale = min(128.0 / float(max_dim), (float(max_pixels) / float(crop_w * crop_h)) ** 0.5)
            new_w = max(1, int(round(crop_w * scale)))
            new_h = max(1, int(round(crop_h * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pixels = crop.reshape(-1, 3).astype("float32")
        mean_b, mean_g, mean_r = [float(value) for value in pixels.mean(axis=0)]
        channel_std = float(pixels.std(axis=0).mean())
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / max(1.0, float(edges.size))
        brightness = (mean_b + mean_g + mean_r) / 3.0
        channel_range = max(mean_b, mean_g, mean_r) - min(mean_b, mean_g, mean_r)
        blue_family = (mean_b >= mean_r + 25.0 and mean_g >= mean_r + 12.0) or (
            mean_b >= mean_g + 12.0 and mean_b >= mean_r + 28.0
        )
        neutral_family = channel_range <= 42.0
        low_texture = channel_std <= 42.0 and edge_density <= 0.075
        stats: dict[str, float | bool] = {
            "mean_b": mean_b,
            "mean_g": mean_g,
            "mean_r": mean_r,
            "brightness": brightness,
            "channel_range": channel_range,
            "channel_std": channel_std,
            "edge_density": edge_density,
            "blue_family": bool(blue_family),
            "neutral_family": bool(neutral_family),
            "low_texture": bool(low_texture),
        }
        if cache is not None:
            cache[cache_key] = stats
        return stats
    except Exception:
        return {}


def _is_flat_blue_background_like(
    detection: dict[str, Any],
    *,
    frame: Any | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    crop_stats_cache: dict[tuple[int, int, int, int], dict[str, float | bool]] | None = None,
) -> bool:
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if area_ratio < 0.006 or width_ratio < 0.06 or height_ratio < 0.04:
        return False
    stats = _crop_stats(frame, detection, cache=crop_stats_cache)
    if not stats:
        return False
    edge_density = float(stats.get("edge_density", 1.0) or 0.0)
    channel_std = float(stats.get("channel_std", 255.0) or 0.0)
    low_texture = bool(stats.get("low_texture")) or (channel_std <= 50.0 and edge_density <= 0.055)
    return bool(stats.get("blue_family")) and low_texture


def _paper_crop_quality_reason(
    detection: dict[str, Any],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    crop_stats_cache: dict[tuple[int, int, int, int], dict[str, float | bool]] | None = None,
) -> str | None:
    if str(detection.get("label") or "") != "paper":
        return None
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if area_ratio <= 0.0:
        return None
    aspect = _bbox_aspect_ratio(detection)
    touches_x, touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
    if area_ratio > 0.24 and (touches_x or touches_y):
        return "edge_spanning_paper_background_bbox"
    if aspect < 0.18 or aspect > 6.2:
        return "implausible_paper_shape"
    stats = _crop_stats(frame, detection, cache=crop_stats_cache)
    if not stats:
        return None
    brightness = float(stats.get("brightness", 0.0) or 0.0)
    edge_density = float(stats.get("edge_density", 0.0) or 0.0)
    channel_range = float(stats.get("channel_range", 0.0) or 0.0)
    if bool(stats.get("blue_family")) and (bool(stats.get("low_texture")) or edge_density <= 0.11 or brightness < 205.0):
        return "blue_workbench_as_paper"
    if brightness < 58.0 and edge_density <= 0.12:
        return "dark_background_as_paper"
    if channel_range > 82.0 and brightness < 185.0 and edge_density <= 0.10:
        return "colored_background_as_paper"
    return None


def _physical_object_quality_reason(
    detection: dict[str, Any],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    crop_stats_cache: dict[tuple[int, int, int, int], dict[str, float | bool]] | None = None,
) -> str | None:
    label = str(detection.get("label") or "")
    if label not in PHYSICAL_EVIDENCE_LABELS:
        return None
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if area_ratio <= 0.0:
        return None
    aspect = _bbox_aspect_ratio(detection)
    touches_x, touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
    stats = _crop_stats(frame, detection, cache=crop_stats_cache)

    if label == "paper":
        return _paper_crop_quality_reason(
            detection,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame,
            crop_stats_cache=crop_stats_cache,
        )

    if label == "sample_bottle_blue":
        if _is_flat_blue_background_like(
            detection,
            frame=frame,
            frame_width=frame_width,
            frame_height=frame_height,
            crop_stats_cache=crop_stats_cache,
        ) and (area_ratio > 0.075 or touches_x or touches_y):
            return "flat_blue_background_as_blue_bottle"
    elif _is_flat_blue_background_like(
        detection,
        frame=frame,
        frame_width=frame_width,
        frame_height=frame_height,
        crop_stats_cache=crop_stats_cache,
    ):
        return "flat_blue_workbench_background"

    if label in {"pipette", "pipette_tip", "spatula", "tube"} and area_ratio > 0.012 and 0.5 <= aspect <= 2.0:
        return "implausible_tool_shape"
    if label in {"pipette", "spatula"} and area_ratio > 0.010 and (aspect < 0.14 or aspect > 12.0):
        return "implausible_tool_shape"
    if label in {"sample_bottle", "sample_bottle_blue", "reagent_bottle"} and area_ratio > 0.012 and (aspect < 0.22 or aspect > 3.6):
        return "implausible_bottle_shape"
    if label in {"beaker", "container"} and area_ratio > 0.025 and (aspect < 0.22 or aspect > 4.5):
        return "implausible_container_shape"

    if stats and label not in {"balance", "ppe_storage"}:
        low_texture = bool(stats.get("low_texture"))
        edge_density = float(stats.get("edge_density", 0.0) or 0.0)
        if low_texture and area_ratio >= 0.16:
            return "large_low_texture_background_bbox"
        if low_texture and area_ratio >= 0.055 and (touches_x or touches_y):
            return "edge_low_texture_background_bbox"
        if edge_density <= 0.025 and area_ratio >= 0.08 and (width_ratio > 0.32 or height_ratio > 0.32):
            return "textureless_background_bbox"

    return None


def _is_implausible_hand_detection(
    detection: dict[str, Any],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    source_view: str | None = None,
    crop_stats_cache: dict[tuple[int, int, int, int], dict[str, float | bool]] | None = None,
) -> bool:
    if str(detection.get("label") or "") not in HAND_LABELS:
        return False
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if area_ratio <= 0.0:
        return False
    confidence = float(detection.get("confidence", 0.0) or 0.0)
    view = str(source_view or "").strip().lower()
    touches_x, touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
    if confidence < 0.45:
        return True
    flat_blue = _is_flat_blue_background_like(
        detection,
        frame=frame,
        frame_width=frame_width,
        frame_height=frame_height,
        crop_stats_cache=crop_stats_cache,
    )
    if flat_blue:
        first_person_bottom_hand = (
            view == "first_person"
            and confidence >= 0.90
            and area_ratio <= 0.13
            and height_ratio <= 0.34
            and not touches_x
            and touches_y
        )
        if not first_person_bottom_hand:
            return True
    stats = _crop_stats(frame, detection, cache=crop_stats_cache)
    if view == "first_person" and stats:
        blue_family = bool(stats.get("blue_family"))
        mean_b = float(stats.get("mean_b", 0.0) or 0.0)
        mean_g = float(stats.get("mean_g", 0.0) or 0.0)
        mean_r = float(stats.get("mean_r", 0.0) or 0.0)
        brightness = float(stats.get("brightness", 255.0) or 0.0)
        channel_std = float(stats.get("channel_std", 255.0) or 0.0)
        edge_density = float(stats.get("edge_density", 1.0) or 0.0)
        low_res = bool(frame_width and frame_height and min(int(frame_width), int(frame_height)) <= 540)
        blue_glove_like = blue_family or (
            mean_b >= mean_r + 18.0 and mean_b >= mean_g + 4.0 and brightness >= 42.0
        )
        if str(detection.get("label") or "") == "gloved_hand" and not blue_glove_like:
            return True
        if (
            low_res
            and blue_family
            and brightness < 92.0
            and channel_std < 36.0
            and edge_density < 0.12
            and area_ratio > 0.11
            and confidence < 0.82
            and not touches_y
        ):
            return True
    if view == "third_person":
        if confidence < 0.55:
            return True
        if area_ratio > 0.18:
            return True
        if area_ratio > 0.12 and (width_ratio > 0.36 or height_ratio > 0.44):
            return True
        if touches_x and area_ratio > 0.03 and confidence < 0.68:
            return True
    # Blue tabletops and PPE-box regions often appear as very large gloved_hand boxes.
    # Keep the filter conservative enough for close-up hands, but do not let large flat
    # blue regions drive hand-object interactions or key action segmentation.
    if area_ratio > 0.28:
        return True
    if area_ratio > 0.20 and (width_ratio > 0.42 or height_ratio > 0.60):
        return True
    if area_ratio > 0.20 and width_ratio > 0.32 and height_ratio > 0.56:
        return True
    if area_ratio > 0.17 and (width_ratio > 0.40 or height_ratio > 0.56):
        return True
    if area_ratio > 0.14 and width_ratio > 0.48:
        return True
    if area_ratio > 0.16 and touches_x and (height_ratio > 0.52 or width_ratio > 0.36):
        return True
    if area_ratio > 0.14 and touches_x and touches_y:
        return True
    if area_ratio > 0.16 and confidence < 0.65 and (width_ratio > 0.35 or height_ratio > 0.52):
        return True
    return False


def _implausible_detection_reason(
    detection: dict[str, Any],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    source_view: str | None = None,
    crop_stats_cache: dict[tuple[int, int, int, int], dict[str, float | bool]] | None = None,
) -> str | None:
    label = str(detection.get("label") or "")
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if label in HAND_LABELS and _is_implausible_hand_detection(
        detection,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
        crop_stats_cache=crop_stats_cache,
    ):
        return "implausible_hand_bbox_or_background"
    if label in HAND_LABELS and str(source_view or "").strip().lower() == "first_person":
        return None

    if area_ratio <= 0.0:
        return None
    aspect = _bbox_aspect_ratio(detection)
    touches_x, touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
    object_quality_reason = _physical_object_quality_reason(
        detection,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        crop_stats_cache=crop_stats_cache,
    )
    if object_quality_reason:
        return object_quality_reason
    flat_blue = _is_flat_blue_background_like(
        detection,
        frame=frame,
        frame_width=frame_width,
        frame_height=frame_height,
        crop_stats_cache=crop_stats_cache,
    )

    if flat_blue and label != "sample_bottle_blue":
        return "flat_blue_workbench_background"
    if flat_blue and area_ratio > 0.075:
        return "oversized_flat_blue_object"

    max_area_by_label = {
        "balance": 0.30,
        "beaker": 0.28,
        "container": 0.28,
        "sample_bottle": 0.22,
        "sample_bottle_blue": 0.18,
        "reagent_bottle": 0.22,
        "paper": 0.32,
        "tube": 0.12,
        "tube_cap": 0.08,
        "pipette": 0.14,
        "pipette_tip": 0.07,
        "spatula": 0.12,
        "ppe_storage": 0.34,
    }
    max_area = max_area_by_label.get(label)
    if max_area is not None and area_ratio > max_area:
        return "implausible_large_object_bbox"
    if label in {"pipette", "pipette_tip", "spatula", "tube"} and area_ratio > 0.035 and 0.55 <= aspect <= 1.85:
        return "implausible_tool_shape"
    if label in {"tube_cap", "pipette_tip"} and (width_ratio > 0.24 or height_ratio > 0.24):
        return "implausible_small_part_size"
    if label in INTERACTION_OBJECT_LABELS and area_ratio > 0.18 and touches_x and touches_y:
        return "edge_spanning_background_bbox"
    if label in INTERACTION_OBJECT_LABELS and area_ratio > 0.16 and touches_x and (width_ratio > 0.45 or height_ratio > 0.45):
        return "edge_background_bbox"
    return None


def filter_implausible_hand_detections(
    detections: Iterable[dict[str, Any]],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return filter_implausible_detections(
        detections,
        frame_width=frame_width,
        frame_height=frame_height,
        hand_only=True,
    )


def filter_implausible_detections(
    detections: Iterable[dict[str, Any]],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    source_view: str | None = None,
    hand_only: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_w, frame_h = _frame_size(frame)
    frame_width = frame_width or frame_w
    frame_height = frame_height or frame_h
    kept: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []
    crop_stats_cache: dict[tuple[int, int, int, int], dict[str, float | bool]] = {}
    for detection in detections:
        normalized = normalize_yolo_detection(detection)
        reason: str | None
        if hand_only:
            reason = (
                "implausible_large_hand_bbox"
                if _is_implausible_hand_detection(
                    normalized,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    frame=frame,
                    source_view=source_view,
                    crop_stats_cache=crop_stats_cache,
                )
                else None
            )
        else:
            reason = _implausible_detection_reason(
                normalized,
                frame_width=frame_width,
                frame_height=frame_height,
                frame=frame,
                source_view=source_view,
                crop_stats_cache=crop_stats_cache,
            )
        if reason:
            ignored_item = dict(normalized)
            ignored_item["ignore_reason"] = reason
            ignored.append(ignored_item)
            continue
        kept.append(normalized)
    return kept, ignored


def _filter_detections(
    detections: Iterable[dict[str, Any]],
    *,
    conf: float = 0.25,
    class_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    thresholds = {canonical_yolo_label(key): float(value) for key, value in (class_thresholds or {}).items()}
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        normalized = normalize_yolo_detection(detection)
        threshold = thresholds.get(str(normalized["label"]), float(conf))
        if float(normalized.get("confidence", 0.0)) >= threshold:
            filtered.append(normalized)
    return filtered


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def _bbox_intersection_area(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_area(box: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_diag(box: list[float]) -> float:
    x1, y1, x2, y2 = [float(value) for value in box]
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def _center_distance(box_a: list[float], box_b: list[float]) -> float:
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _proximity_score(
    hand_bbox: list[float],
    object_bbox: list[float],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[float, float]:
    distance_px = _center_distance(hand_bbox, object_bbox)
    hand_diag = _bbox_diag(hand_bbox)
    object_diag = _bbox_diag(object_bbox)
    local_scale = max(24.0, (hand_diag + object_diag) * 0.55)
    if frame_width and frame_height:
        frame_diag = (float(frame_width) ** 2 + float(frame_height) ** 2) ** 0.5
        local_scale = max(local_scale, frame_diag * 0.08)
    score = max(0.0, 1.0 - (distance_px / local_scale))
    return float(score), float(distance_px)


def find_hand_object_interactions(
    detections: list[dict[str, Any]],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    source_view: str | None = None,
    min_interaction_score: float = 0.1,
) -> list[dict[str, Any]]:
    normalized, _ignored = filter_implausible_detections(
        detections,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    )
    hands = [item for item in normalized if item["label"] in HAND_LABELS]
    objects = [item for item in normalized if item["label"] in INTERACTION_OBJECT_LABELS]
    interactions: list[dict[str, Any]] = []
    for hand in hands:
        for obj in objects:
            iou = bbox_iou(hand["bbox"], obj["bbox"])
            intersection_area = _bbox_intersection_area(hand["bbox"], obj["bbox"])
            object_area = _bbox_area(obj["bbox"])
            hand_area = _bbox_area(hand["bbox"])
            object_overlap_ratio = 0.0 if object_area <= 0 else intersection_area / object_area
            hand_overlap_ratio = 0.0 if hand_area <= 0 else intersection_area / hand_area
            proximity, distance_px = _proximity_score(
                hand["bbox"],
                obj["bbox"],
                frame_width=frame_width,
                frame_height=frame_height,
            )
            iou_score = min(1.0, iou / 0.12) if iou > 0 else 0.0
            object_overlap_score = min(1.0, object_overlap_ratio / 0.18) if object_overlap_ratio > 0 else 0.0
            score = max(iou_score, object_overlap_score, proximity)
            score *= min(1.0, (float(hand.get("confidence", 0.0)) + float(obj.get("confidence", 0.0))) / 2.0 / 0.65)
            if score < min_interaction_score:
                continue
            interactions.append(
                {
                    "hand_label": hand["label"],
                    "object_label": obj["label"],
                    "hand_bbox": hand["bbox"],
                    "object_bbox": obj["bbox"],
                    "iou": round(float(iou), 6),
                    "object_overlap_ratio": round(float(object_overlap_ratio), 6),
                    "hand_overlap_ratio": round(float(hand_overlap_ratio), 6),
                    "distance_px": round(float(distance_px), 3),
                    "proximity_score": round(float(proximity), 6),
                    "score": round(float(min(1.0, score)), 6),
                }
            )
    interactions.sort(key=lambda item: float(item["score"]), reverse=True)
    return interactions


def _label_counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(item.get("label") or "") for item in detections)
    counts.pop("", None)
    return dict(counts)


def _presence_score(label_counts: dict[str, int]) -> float:
    hand_count = sum(int(label_counts.get(label, 0)) for label in HAND_LABELS)
    object_count = sum(int(label_counts.get(label, 0)) for label in INTERACTION_OBJECT_LABELS)
    context_count = sum(int(label_counts.get(label, 0)) for label in EXPERIMENT_CONTEXT_LABELS)
    hand_score = 1.0 if hand_count else 0.0
    object_score = min(1.0, object_count / 2.0)
    density_score = min(1.0, context_count / 4.0)
    return min(1.0, 0.45 * hand_score + 0.4 * object_score + 0.15 * density_score)


class YoloActivityScorer:
    def __init__(self, *, active_threshold: float = 0.55, continuity_frames: int = 3) -> None:
        self.active_threshold = float(active_threshold)
        self.continuity_frames = max(1, int(continuity_frames))
        self._evidence_streak = 0

    def score(
        self,
        detections: list[dict[str, Any]],
        *,
        frame_width: int | None = None,
        frame_height: int | None = None,
        frame: Any | None = None,
        source_view: str | None = None,
    ) -> dict[str, Any]:
        normalized, ignored = filter_implausible_detections(
            detections,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame,
            source_view=source_view,
        )
        counts = _label_counts(normalized)
        interactions = find_hand_object_interactions(
            normalized,
            frame_width=frame_width,
            frame_height=frame_height,
            frame=frame,
            source_view=source_view,
        )
        interaction_score = max([float(item["score"]) for item in interactions], default=0.0)
        presence_score = _presence_score(counts)
        has_evidence = presence_score >= 0.35 or interaction_score >= 0.1
        if has_evidence:
            self._evidence_streak += 1
        else:
            self._evidence_streak = max(0, self._evidence_streak - 1)
        continuity_score = min(1.0, self._evidence_streak / self.continuity_frames)
        active_score = min(1.0, 0.4 * presence_score + 0.45 * interaction_score + 0.15 * continuity_score)
        return {
            "detections": normalized,
            "ignored_detections": ignored,
            "label_counts": counts,
            "hand_object_interactions": interactions,
            "interaction_score": round(float(interaction_score), 6),
            "presence_score": round(float(presence_score), 6),
            "continuity_score": round(float(continuity_score), 6),
            "active_score": round(float(active_score), 6),
            "is_experiment_active": bool(active_score >= self.active_threshold),
        }


def _row_from_detections(
    detections: list[dict[str, Any]],
    *,
    scorer: YoloActivityScorer,
    source_view: str,
    video_path: str | Path,
    frame_index: int,
    sample_index: int,
    time_sec: float,
    sample_fps: float,
    source_fps: float | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
    video_start_time: str | None = None,
    frame: Any | None = None,
) -> dict[str, Any]:
    scored = scorer.score(
        detections,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    )
    row = {
        "source_view": source_view,
        "video_path": str(video_path),
        "frame_index": int(frame_index),
        "sample_index": int(sample_index),
        "time_sec": round(float(time_sec), 6),
        "local_time_sec": round(float(time_sec), 6),
        "sample_fps": float(sample_fps),
        "interaction_score": float(scored.get("interaction_score", scored.get("active_score", 0.0))),
        "active_score": float(scored.get("active_score", scored.get("interaction_score", 0.0))),
        **scored,
    }
    raw_score = max(float(row.get("interaction_score", 0.0)), float(row.get("active_score", 0.0)))
    active_threshold = float(getattr(scorer, "active_threshold", 0.55))
    is_active = bool(row["is_experiment_active"]) if "is_experiment_active" in row else raw_score >= active_threshold
    row.setdefault("is_experiment_active", is_active)
    row["raw_score"] = float(row.get("raw_score", raw_score))
    row["probability"] = float(row.get("probability", row.get("interaction_score", raw_score)))
    row["prob"] = float(row.get("prob", row.get("probability")))
    row["prob_score"] = float(row.get("prob_score", row.get("probability")))
    row["raw_prob"] = float(row.get("raw_prob", row.get("interaction_score", raw_score)))
    row["motion_prob"] = float(row.get("motion_prob", row.get("active_score", raw_score)))
    row["keep"] = bool(row.get("keep", is_active))
    row["keep_score"] = float(row.get("keep_score", 1.0 if is_active else 0.0))
    if source_fps is not None:
        row["source_fps"] = float(source_fps)
    if frame_width is not None and frame_height is not None:
        row["frame_width"] = int(frame_width)
        row["frame_height"] = int(frame_height)
    if video_start_time:
        row["video_start_time"] = str(video_start_time)
    return row


def normalize_yolo_frame_rows(
    rows: Iterable[dict[str, Any]],
    *,
    active_threshold: float = 0.55,
    continuity_frames: int = 3,
) -> list[dict[str, Any]]:
    scorer = YoloActivityScorer(active_threshold=active_threshold, continuity_frames=continuity_frames)
    normalized_rows: list[dict[str, Any]] = []
    for sample_index, row in enumerate(sorted(rows, key=lambda item: float(item.get("time_sec", 0.0)))):
        if "detections" in row:
            scored = scorer.score(
                list(row.get("detections") or []),
                frame_width=row.get("frame_width"),
                frame_height=row.get("frame_height"),
                source_view=str(row.get("source_view") or row.get("view") or ""),
            )
            merged = dict(row)
            merged.update(scored)
            merged.setdefault("sample_index", sample_index)
            merged.setdefault("local_time_sec", float(merged.get("time_sec", 0.0)))
            normalized_rows.append(merged)
            continue
        merged = dict(row)
        merged.setdefault("detections", [])
        merged.setdefault("label_counts", {})
        merged.setdefault("hand_object_interactions", [])
        merged.setdefault("interaction_score", 0.0)
        merged.setdefault("active_score", 0.0)
        merged.setdefault("is_experiment_active", bool(float(merged.get("active_score", 0.0)) >= active_threshold))
        merged.setdefault("sample_index", sample_index)
        merged.setdefault("local_time_sec", float(merged.get("time_sec", 0.0)))
        normalized_rows.append(merged)
    return normalized_rows


def mock_yolo_frame_rows(
    *,
    duration_sec: float = 960.0,
    sample_fps: float = 1.0,
    source_view: str = "first_person",
    video_path: str | Path = "dry_run.mp4",
    active_windows: list[tuple[float, float]] | None = None,
    active_threshold: float = 0.55,
    continuity_frames: int = 3,
) -> list[dict[str, Any]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")
    windows = active_windows or [(598.0, 610.0), (618.0, 632.0), (898.0, 912.0)]
    scorer = YoloActivityScorer(active_threshold=active_threshold, continuity_frames=continuity_frames)
    rows: list[dict[str, Any]] = []
    frame_width, frame_height = 1280, 720
    step = 1.0 / float(sample_fps)
    sample_index = 0
    t = 0.0
    while t <= duration_sec + 1e-9:
        active = any(start <= t <= end for start, end in windows)
        detections = (
            [
                {"label": "gloved_hand", "confidence": 0.86, "bbox": [410, 300, 500, 420]},
                {"label": "sample_bottle", "confidence": 0.78, "bbox": [455, 330, 535, 475]},
                {"label": "balance", "confidence": 0.72, "bbox": [545, 360, 760, 520]},
            ]
            if active
            else [{"label": "paper", "confidence": 0.45, "bbox": [720, 430, 850, 520]}]
        )
        rows.append(
            _row_from_detections(
                detections,
                scorer=scorer,
                source_view=source_view,
                video_path=video_path,
                frame_index=int(round(t * 30.0)),
                sample_index=sample_index,
                time_sec=t,
                sample_fps=sample_fps,
                source_fps=30.0,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )
        sample_index += 1
        t = sample_index * step
    return rows


def _select_video_source(
    video_path: str | Path | VideoSource | None = None,
    *,
    first_person_path: str | Path | VideoSource | None = None,
    third_person_path: str | Path | VideoSource | None = None,
    preferred_view: str = "first_person",
    source_view: str | None = None,
    dry_run: bool = False,
) -> tuple[str, Path, str | None, float | None]:
    def unpack(value: str | Path | VideoSource | None, default_view: str) -> tuple[str, Path, str | None, float | None] | None:
        if value is None:
            return None
        if isinstance(value, VideoSource):
            return value.name or default_view, Path(value.path), value.start_time, value.fps
        return default_view, Path(value), None, None

    direct = unpack(video_path, source_view or preferred_view or "video")
    if direct is not None:
        return direct

    candidates = {
        "first_person": unpack(first_person_path, "first_person"),
        "third_person": unpack(third_person_path, "third_person"),
    }
    order = [preferred_view, "third_person" if preferred_view == "first_person" else "first_person"]
    for view in order:
        candidate = candidates.get(view)
        if candidate is None:
            continue
        if dry_run or candidate[1].exists():
            return candidate
    for candidate in candidates.values():
        if candidate is not None:
            return candidate
    raise ValueError("video_path or first_person_path/third_person_path is required")


def _load_yolo_model(model: Any | None, model_path: str | Path | None) -> Any:
    if model is not None:
        return model
    if model_path is None:
        raise RuntimeError("model, model_path, or detector callable is required for real YOLO scanning")
    cache_enabled = str(os.environ.get("KEY_ACTION_YOLO_MODEL_CACHE", "1")).strip().lower() not in {"", "0", "false", "no", "off"}
    cache_key = str(Path(model_path).resolve())
    if cache_enabled:
        with _YOLO_MODEL_CACHE_LOCK:
            cached = _YOLO_MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached
            try:
                from ultralytics import YOLO
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("ultralytics is required when detector is not provided") from exc
            loaded = YOLO(str(model_path))
            _YOLO_MODEL_CACHE[cache_key] = loaded
            return loaded
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required when detector is not provided") from exc
    loaded = YOLO(str(model_path))
    return loaded


def _yolo_predict_lock_scope(device: str) -> str:
    value = str(os.environ.get("KEY_ACTION_YOLO_PREDICT_LOCK_SCOPE", "model")).strip().lower().replace("-", "_")
    if value in {"", "0", "false", "no", "off", "none", "disabled"}:
        return "off"
    if value in {"gpu", "cuda", "device"}:
        return "gpu"
    if value in {"global", "process"}:
        return "global"
    return "model"


def _model_predict_lock(model: Any) -> threading.Lock:
    key = id(model)
    with _YOLO_MODEL_PREDICT_LOCKS_LOCK:
        lock = _YOLO_MODEL_PREDICT_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _YOLO_MODEL_PREDICT_LOCKS[key] = lock
        return lock


def _predict_with_lock(model: Any, predict_kwargs: dict[str, Any], *, device: str) -> Any:
    scope = _yolo_predict_lock_scope(device)
    resolved_device = str(predict_kwargs.get("device") or device or "").strip().lower()
    if scope == "off":
        return model.predict(**predict_kwargs)
    if scope in {"gpu", "global"} and resolved_device not in {"", "cpu"}:
        with _YOLO_GPU_PREDICT_LOCK:
            return model.predict(**predict_kwargs)
    if scope == "global":
        with _YOLO_GPU_PREDICT_LOCK:
            return model.predict(**predict_kwargs)
    with _model_predict_lock(model):
        return model.predict(**predict_kwargs)


def _requested_cuda_indices(device: str) -> list[int] | None:
    value = device.strip().lower()
    if value == "cuda":
        return [0]
    if value.startswith("cuda:"):
        value = value.split(":", 1)[1]
    if value and all(part.strip().isdigit() for part in value.split(",")):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return None


def _is_yolo_gpu_device(device: Any) -> bool:
    value = str(device or "").strip().lower()
    if not value:
        return False
    if value in {"mps"} or value.startswith(("cuda", "gpu")):
        return True
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return bool(parts) and all(part.isdigit() for part in parts)


def _record_batch_stat(stats: dict[str, Any] | None, key: str, amount: int = 1) -> None:
    if stats is None:
        return
    stats[key] = int(stats.get(key) or 0) + int(amount)


def _record_batch_sizes(stats: dict[str, Any] | None, batch_size: int) -> None:
    if stats is None:
        return
    for key in (
        "predict_call_count",
        "batch_predict_attempts",
        "batch_predict_calls",
        "frame_predict_calls",
        "custom_detector_calls",
        "batch_fallback_count",
        "batch_count",
    ):
        stats.setdefault(key, 0)
    sizes = stats.setdefault("actual_batch_sizes", [])
    if isinstance(sizes, list):
        sizes.append(int(batch_size))
    _record_batch_stat(stats, "batch_count", 1)


def _record_batch_error(stats: dict[str, Any] | None, exc: Exception) -> None:
    if stats is None:
        return
    errors = stats.setdefault("batch_fallback_errors", [])
    if isinstance(errors, list) and len(errors) < 5:
        errors.append(str(exc))


def _batch_diagnostics(stats: dict[str, Any], requested_batch_size: int) -> dict[str, Any]:
    sizes = [int(value) for value in stats.get("actual_batch_sizes", []) if int(value or 0) > 0]
    batch_count = int(stats.get("batch_count") or len(sizes))
    underfilled = sum(1 for value in sizes if value < int(requested_batch_size))
    avg_batch = (sum(sizes) / len(sizes)) if sizes else 0.0
    return {
        "requested_batch_size": int(requested_batch_size),
        "batch_size": int(requested_batch_size),
        "actual_batch_sizes": sizes,
        "batch_count": int(batch_count),
        "max_actual_batch_size": max(sizes, default=0),
        "avg_actual_batch_size": round(avg_batch, 6),
        "underfilled_batch_count": int(underfilled),
        "yolo_predict_call_count": int(stats.get("predict_call_count") or 0),
        "yolo_batch_predict_attempts": int(stats.get("batch_predict_attempts") or 0),
        "yolo_batch_predict_calls": int(stats.get("batch_predict_calls") or 0),
        "yolo_frame_predict_calls": int(stats.get("frame_predict_calls") or 0),
        "yolo_custom_detector_calls": int(stats.get("custom_detector_calls") or 0),
        "yolo_batch_fallback_count": int(stats.get("batch_fallback_count") or 0),
        "yolo_batch_fallback_errors": list(stats.get("batch_fallback_errors") or []),
    }


def _cuda_runtime_error(message: str, *, device: str, torch_module: Any | None = None) -> RuntimeError:
    details = [message, f"requested_device={device!r}"]
    if torch_module is not None:
        details.append(f"torch={getattr(torch_module, '__version__', 'unknown')}")
        details.append(f"torch_cuda={getattr(getattr(torch_module, 'version', None), 'cuda', None)}")
    return RuntimeError("; ".join(details))


def _resolve_yolo_device(device: Any, *, torch_module: Any | None = None) -> str:
    value = "auto" if device is None else str(device).strip()
    lower = value.lower()
    if lower == "cpu":
        return "cpu"

    if not lower or lower in {"none", "auto"}:
        try:
            torch_module = torch_module or __import__("torch")
            cuda = getattr(torch_module, "cuda", None)
            return "0" if cuda is not None and cuda.is_available() and cuda.device_count() > 0 else "cpu"
        except Exception:
            return "cpu"

    requested_indices = _requested_cuda_indices(value)
    if requested_indices is None:
        return value

    loaded_torch = torch_module
    try:
        loaded_torch = loaded_torch or __import__("torch")
        cuda = getattr(loaded_torch, "cuda", None)
        if cuda is None or not cuda.is_available():
            raise _cuda_runtime_error(
                "explicit CUDA YOLO device was requested, but this Python runtime has no CUDA-enabled torch",
                device=value,
                torch_module=loaded_torch,
            )
        device_count = int(cuda.device_count())
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "explicit CUDA YOLO device was requested, but torch CUDA runtime could not be inspected; "
            f"requested_device={value!r}; error={exc}"
        ) from exc
    if requested_indices and max(requested_indices) < device_count:
        return value
    raise _cuda_runtime_error(
        f"explicit CUDA YOLO device index is out of range; cuda_device_count={device_count}",
        device=value,
        torch_module=loaded_torch,
    )


def _is_long_video_coarse_scan(model_ref: dict[str, Any] | None) -> bool:
    return str((model_ref or {}).get("scan_role") or "").strip().lower() == "long_video_coarse"


def _coarse_seek_scan_enabled() -> bool:
    for name in ("KEY_ACTION_FAST_LOCATE_COARSE_SEEK_SCAN", "KEY_ACTION_YOLO_COARSE_SEEK_SCAN"):
        if os.environ.get(name) is not None:
            return _env_bool(name, True)
    return True


def _detections_from_model(
    model: Any,
    frame: Any,
    *,
    conf: float,
    iou: float,
    device: str,
    class_thresholds: dict[str, float] | None = None,
    imgsz: int | None = None,
) -> list[dict[str, Any]]:
    model_conf = float(conf)
    if class_thresholds:
        try:
            model_conf = min([model_conf, *[float(value) for value in class_thresholds.values()]])
        except Exception:
            model_conf = float(conf)
    predict_kwargs: dict[str, Any] = {
        "source": frame,
        "conf": max(0.01, model_conf),
        "iou": float(iou),
        "device": _resolve_yolo_device(device),
        "verbose": False,
    }
    if imgsz:
        predict_kwargs["imgsz"] = int(imgsz)
    results = _predict_with_lock(model, predict_kwargs, device=device)
    detections: list[dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(_to_scalar(getattr(box, "cls", [0])))
            raw_label = names.get(cls_id, cls_id)
            detections.append(
                {
                    "label": raw_label,
                    "class_id": cls_id,
                    "confidence": _to_scalar(getattr(box, "conf", [0.0])),
                    "bbox": _coerce_bbox(getattr(box, "xyxy", [0, 0, 0, 0])),
                }
            )
    filtered = _filter_detections(detections, conf=conf, class_thresholds=class_thresholds)
    kept, _ignored = filter_implausible_detections(filtered, frame=frame)
    return kept


def _detections_from_model_batch(
    model: Any,
    frames: list[Any],
    *,
    conf: float,
    iou: float,
    device: str,
    class_thresholds: dict[str, float] | None = None,
    imgsz: int | None = None,
    stats: dict[str, Any] | None = None,
) -> list[list[dict[str, Any]]]:
    if not frames:
        return []
    _record_batch_sizes(stats, len(frames))
    model_conf = float(conf)
    if class_thresholds:
        try:
            model_conf = min([model_conf, *[float(value) for value in class_thresholds.values()]])
        except Exception:
            model_conf = float(conf)
    try:
        predict_kwargs: dict[str, Any] = {
            "source": frames,
            "conf": max(0.01, model_conf),
            "iou": float(iou),
            "device": _resolve_yolo_device(device),
            "batch": len(frames),
            "verbose": False,
        }
        if imgsz:
            predict_kwargs["imgsz"] = int(imgsz)
        _record_batch_stat(stats, "batch_predict_attempts", 1)
        _record_batch_stat(stats, "predict_call_count", 1)
        results = _predict_with_lock(model, predict_kwargs, device=device)
        _record_batch_stat(stats, "batch_predict_calls", 1)
    except Exception as exc:
        _record_batch_stat(stats, "batch_fallback_count", 1)
        _record_batch_stat(stats, "frame_predict_calls", len(frames))
        _record_batch_stat(stats, "predict_call_count", len(frames))
        _record_batch_error(stats, exc)
        return [
            _detections_from_model(
                model,
                frame,
                conf=conf,
                iou=iou,
                device=device,
                class_thresholds=class_thresholds,
                imgsz=imgsz,
            )
            for frame in frames
        ]
    batched: list[list[dict[str, Any]]] = []
    for frame, result in zip(frames, results):
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(_to_scalar(getattr(box, "cls", [0])))
                raw_label = names.get(cls_id, cls_id)
                detections.append(
                    {
                        "label": raw_label,
                        "class_id": cls_id,
                        "confidence": _to_scalar(getattr(box, "conf", [0.0])),
                        "bbox": _coerce_bbox(getattr(box, "xyxy", [0, 0, 0, 0])),
                    }
                )
        filtered = _filter_detections(detections, conf=conf, class_thresholds=class_thresholds)
        kept, _ignored = filter_implausible_detections(filtered, frame=frame)
        batched.append(kept)
    if len(batched) < len(frames):
        batched.extend([[] for _ in range(len(frames) - len(batched))])
    return batched


def _yolo_batch_size_env_names(model_ref: Mapping[str, Any] | None = None) -> list[str]:
    scan_role = str((model_ref or {}).get("scan_role") or "").strip().lower()
    env_names: list[str] = []
    if scan_role == "long_video_coarse":
        env_names.extend(["KEY_ACTION_FAST_LOCATE_COARSE_YOLO_BATCH_SIZE", "KEY_ACTION_YOLO_COARSE_BATCH_SIZE"])
    elif scan_role in {"micro_refine", "paired_micro_refine"}:
        env_names.extend(["KEY_ACTION_FAST_LOCATE_FINE_YOLO_BATCH_SIZE", "KEY_ACTION_YOLO_FINE_BATCH_SIZE"])
    env_names.append("KEY_ACTION_YOLO_BATCH_SIZE")
    return env_names


def _yolo_batch_size_env_configured(model_ref: Mapping[str, Any] | None = None) -> bool:
    return any(os.environ.get(env_name) is not None for env_name in _yolo_batch_size_env_names(model_ref))


def _default_yolo_batch_size(model_ref: Mapping[str, Any] | None = None) -> int:
    for env_name in _yolo_batch_size_env_names(model_ref):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            return max(1, int(float(raw)))
        except Exception:
            continue
    return 16


def _gpu_yolo_batch_size_default(model_ref: Mapping[str, Any] | None = None) -> int:
    scan_role = str((model_ref or {}).get("scan_role") or "").strip().lower()
    env_names: list[str] = []
    if scan_role == "long_video_coarse":
        env_names.extend(["KEY_ACTION_FAST_LOCATE_COARSE_GPU_YOLO_BATCH_SIZE", "KEY_ACTION_YOLO_COARSE_GPU_BATCH_SIZE"])
        default = 24
    elif scan_role in {"micro_refine", "paired_micro_refine"}:
        env_names.extend(["KEY_ACTION_FAST_LOCATE_FINE_GPU_YOLO_BATCH_SIZE", "KEY_ACTION_YOLO_FINE_GPU_BATCH_SIZE"])
        default = 16
    else:
        default = 16
    env_names.append("KEY_ACTION_YOLO_GPU_BATCH_SIZE")
    for env_name in env_names:
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            return max(1, int(float(raw)))
        except Exception:
            continue
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def _imgsz_multiple(value: int) -> int:
    value = max(32, int(value))
    return int(((value + 31) // 32) * 32)


def resolve_adaptive_yolo_imgsz(
    frame_width: int | None,
    frame_height: int | None,
    *,
    configured_imgsz: int | None = None,
    adaptive: bool = True,
    min_imgsz: int | None = None,
    max_imgsz: int | None = None,
) -> int | None:
    """Choose a stable YOLO input size for mixed-resolution experiment videos."""
    explicit = configured_imgsz
    if explicit is None:
        env_value = os.environ.get("KEY_ACTION_YOLO_IMGSZ")
        if env_value:
            explicit = _env_int("KEY_ACTION_YOLO_IMGSZ", 960)
    if explicit:
        return _imgsz_multiple(int(explicit))
    if not adaptive:
        return None

    min_size = _imgsz_multiple(min_imgsz if min_imgsz is not None else _env_int("KEY_ACTION_YOLO_MIN_IMGSZ", 960))
    max_size = _imgsz_multiple(max_imgsz if max_imgsz is not None else _env_int("KEY_ACTION_YOLO_MAX_IMGSZ", 1280))
    if max_size < min_size:
        max_size = min_size
    if not frame_width or not frame_height:
        return min_size

    long_side = max(int(frame_width), int(frame_height))
    short_side = min(int(frame_width), int(frame_height))
    target = min_size
    if long_side >= 1600 or short_side >= 900:
        target = max(target, 1280)
    if short_side <= 540:
        target = max(target, min_size)
    return min(max_size, _imgsz_multiple(target))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _use_ffmpeg_sparse_scan(
    sample_fps: float,
    max_sampled_frames: int | None,
    scan_duration_sec: float | None = None,
) -> bool:
    if max_sampled_frames is not None:
        return False
    if not _env_bool("KEY_ACTION_YOLO_FFMPEG_SPARSE_SCAN", True):
        return _env_bool("KEY_ACTION_YOLO_FFMPEG_FORCE_SPARSE_SCAN", False)
    if _env_bool("KEY_ACTION_YOLO_FFMPEG_FORCE_SPARSE_SCAN", False):
        return True
    if scan_duration_sec is not None and float(scan_duration_sec) <= 0.0:
        return False
    max_sparse_fps = _env_float("KEY_ACTION_YOLO_FFMPEG_SPARSE_MAX_FPS", 1.25)
    return float(sample_fps) <= max_sparse_fps


def _use_sparse_scan_for_scan(
    sample_fps: float,
    max_sampled_frames: int | None,
    scan_duration_sec: float | None,
    model_ref: dict[str, Any] | None,
) -> bool:
    if max_sampled_frames is not None:
        return False
    if _is_long_video_coarse_scan(model_ref):
        if scan_duration_sec is not None and float(scan_duration_sec) <= 0.0:
            return False
        return _coarse_seek_scan_enabled()
    return _use_ffmpeg_sparse_scan(sample_fps, max_sampled_frames, scan_duration_sec)


def _resolve_ffmpeg_sparse_mode(sample_fps: float, scan_duration_sec: float | None = None) -> str:
    raw_mode = os.environ.get("KEY_ACTION_YOLO_FFMPEG_SPARSE_MODE")
    mode = str(raw_mode or "auto").strip().lower()
    if mode in {"opencv_seek", "cv2_seek", "cv_seek"}:
        return "opencv_seek"
    if mode in {"seek", "seek_parallel", "parallel_seek", "frame_seek", "per_frame_seek", "legacy_seek", "ffmpeg_sparse_seek"}:
        return "seek"
    if mode in {"chunk", "chunks", "ffmpeg_chunks", "ffmpeg_sparse_chunks"}:
        return "chunks"
    if mode not in {"", "auto", "default"}:
        mode = "auto"

    fps = max(0.001, float(sample_fps))
    duration = max(0.0, float(scan_duration_sec or 0.0))
    seek_max_fps = _env_float("KEY_ACTION_YOLO_FFMPEG_AUTO_SEEK_MAX_FPS", 0.25)
    seek_max_frames = max(1, _env_int("KEY_ACTION_YOLO_FFMPEG_AUTO_SEEK_MAX_FRAMES", 80))
    short_seek_window_sec = _env_float("KEY_ACTION_YOLO_FFMPEG_AUTO_SEEK_WINDOW_SEC", 15.0)
    short_seek_max_fps = _env_float("KEY_ACTION_YOLO_FFMPEG_AUTO_SEEK_WINDOW_MAX_FPS", 0.5)
    expected_frames = int(duration * fps) if duration > 0.0 else 0
    if fps <= seek_max_fps and (expected_frames <= 0 or expected_frames <= seek_max_frames):
        return "seek"
    if duration > 0.0 and duration <= short_seek_window_sec and fps <= short_seek_max_fps:
        return "seek"
    return "chunks"


def _resolve_ffmpeg_sparse_mode_for_scan(
    sample_fps: float,
    scan_duration_sec: float | None,
    model_ref: dict[str, Any] | None,
) -> str:
    scan_role = str((model_ref or {}).get("scan_role") or "").strip().lower()
    if scan_role == "long_video_coarse":
        raw_mode = os.environ.get("KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SPARSE_MODE") or os.environ.get(
            "KEY_ACTION_YOLO_COARSE_FFMPEG_SPARSE_MODE"
        )
    elif scan_role == "micro_refine":
        raw_mode = os.environ.get("KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_SPARSE_MODE") or os.environ.get(
            "KEY_ACTION_YOLO_FINE_FFMPEG_SPARSE_MODE"
        )
    else:
        raw_mode = None
    if raw_mode is not None:
        mode = str(raw_mode).strip().lower()
        if mode in {"opencv_seek", "cv2_seek", "cv_seek"}:
            return "opencv_seek"
        if mode in {"frame_seek", "per_frame_seek", "legacy_seek", "ffmpeg_sparse_seek"}:
            return "seek"
        if mode in {"seek", "seek_parallel", "parallel_seek"}:
            return "seek"
        if mode in {"chunk", "chunks", "ffmpeg_chunks", "ffmpeg_sparse_chunks"}:
            return "chunks"
    if scan_role == "long_video_coarse":
        # Long-video coarse locate is sampling sparse evidence across many minutes.
        # Per-frame OpenCV seeking is easy to reason about but very slow on long
        # MP4/GOP streams. The default fast path seeks once per coarse chunk and
        # lets ffmpeg emit low-fps frames through the raw pipe; explicit env vars
        # can still force seek/opencv modes for debugging.
        return "chunks" if _coarse_seek_scan_enabled() else "chunks"
    return _resolve_ffmpeg_sparse_mode(sample_fps, scan_duration_sec)


def _ffmpeg_fps_value(sample_fps: float) -> str:
    return f"{max(0.001, float(sample_fps)):.6f}".rstrip("0").rstrip(".")


def _ffmpeg_hwaccel_args() -> list[str]:
    raw = os.environ.get("KEY_ACTION_YOLO_FFMPEG_HWACCEL")
    if raw is None:
        return []
    hwaccel = str(raw).strip().lower()
    if hwaccel in {"", "0", "false", "no", "off", "none", "cpu"}:
        return []
    args = ["-hwaccel", hwaccel]
    output_format = os.environ.get("KEY_ACTION_YOLO_FFMPEG_HWACCEL_OUTPUT_FORMAT")
    if output_format and output_format.strip():
        args += ["-hwaccel_output_format", output_format.strip()]
    return args


def _ffmpeg_worker_count_for_scan(model_ref: dict[str, Any] | None) -> int:
    scan_role = str((model_ref or {}).get("scan_role") or "").strip().lower()
    env_names: list[str] = []
    if scan_role == "long_video_coarse":
        env_names.extend(
            [
                "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_WORKERS",
                "KEY_ACTION_YOLO_COARSE_FFMPEG_WORKERS",
            ]
        )
    elif scan_role == "micro_refine":
        env_names.extend(
            [
                "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_WORKERS",
                "KEY_ACTION_YOLO_FINE_FFMPEG_WORKERS",
            ]
        )
    env_names.append("KEY_ACTION_YOLO_FFMPEG_WORKERS")
    default_workers = 4 if scan_role in {"long_video_coarse", "micro_refine"} else 4
    for name in env_names:
        if os.environ.get(name) is not None:
            return max(1, _env_int(name, default_workers))
    return max(1, int(default_workers))


def _ffmpeg_scale_width_for_scan(model_ref: dict[str, Any] | None, default_width: int) -> int:
    scan_role = str((model_ref or {}).get("scan_role") or "").strip().lower()
    env_names: list[str] = []
    if scan_role == "long_video_coarse":
        env_names.extend(
            [
                "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SCALE_WIDTH",
                "KEY_ACTION_YOLO_COARSE_FFMPEG_SCALE_WIDTH",
            ]
        )
    elif scan_role == "micro_refine":
        env_names.extend(
            [
                "KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_SCALE_WIDTH",
                "KEY_ACTION_YOLO_FINE_FFMPEG_SCALE_WIDTH",
            ]
        )
    env_names.append("KEY_ACTION_YOLO_FFMPEG_SCALE_WIDTH")
    safe_default = int(default_width or 640)
    for name in env_names:
        if os.environ.get(name) is not None:
            return _env_int(name, safe_default)
    return safe_default


def _ffmpeg_chunk_sec_for_scan(model_ref: dict[str, Any] | None) -> float:
    scan_role = str((model_ref or {}).get("scan_role") or "").strip().lower()
    if scan_role == "long_video_coarse":
        for name in ("KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_CHUNK_SEC", "KEY_ACTION_YOLO_COARSE_FFMPEG_CHUNK_SEC"):
            if os.environ.get(name) is not None:
                return max(5.0, _env_float(name, 600.0))
        return max(5.0, _env_float("KEY_ACTION_YOLO_FFMPEG_CHUNK_SEC", 600.0))
    if scan_role == "micro_refine":
        for name in ("KEY_ACTION_FAST_LOCATE_FINE_FFMPEG_CHUNK_SEC", "KEY_ACTION_YOLO_FINE_FFMPEG_CHUNK_SEC"):
            if os.environ.get(name) is not None:
                return max(1.0, _env_float(name, 30.0))
        return max(1.0, _env_float("KEY_ACTION_YOLO_FFMPEG_CHUNK_SEC", 30.0))
    return max(1.0, _env_float("KEY_ACTION_YOLO_FFMPEG_CHUNK_SEC", 180.0))


def _ffmpeg_sparse_chunks(
    *,
    start_sec: float,
    end_sec: float,
    chunk_sec: float,
) -> list[tuple[int, float, float]]:
    chunks: list[tuple[int, float, float]] = []
    cursor = max(0.0, float(start_sec))
    end = max(cursor, float(end_sec))
    step = max(1.0, float(chunk_sec))
    index = 0
    while cursor < end:
        chunk_end = min(end, cursor + step)
        chunks.append((index, cursor, max(0.001, chunk_end - cursor)))
        cursor = chunk_end
        index += 1
    return chunks


def _extract_ffmpeg_sparse_chunk(
    *,
    ffmpeg_path: str,
    video_path: Path,
    output_root: Path,
    chunk_index: int,
    chunk_start_sec: float,
    chunk_duration_sec: float,
    sample_fps: float,
    scale_width: int | None,
    quality: int,
) -> dict[str, Any]:
    chunk_dir = output_root / f"chunk_{chunk_index:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    vf_parts = [f"fps={_ffmpeg_fps_value(sample_fps)}"]
    if scale_width and scale_width > 0:
        vf_parts.append(f"scale={int(scale_width)}:-2")
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *_ffmpeg_hwaccel_args(),
        "-ss",
        f"{max(0.0, chunk_start_sec):.3f}",
        "-t",
        f"{max(0.001, chunk_duration_sec):.3f}",
        "-i",
        str(video_path),
        "-vf",
        ",".join(vf_parts),
        "-q:v",
        str(max(2, min(31, int(quality)))),
        str(chunk_dir / "frame_%06d.jpg"),
    ]
    started = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall_sec = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg sparse extraction failed for chunk {chunk_index}: {result.stderr[-800:]}")
    frames = sorted(chunk_dir.glob("frame_*.jpg"))
    return {
        "chunk_index": chunk_index,
        "chunk_start_sec": float(chunk_start_sec),
        "chunk_duration_sec": float(chunk_duration_sec),
        "wall_sec": wall_sec,
        "frame_count": len(frames),
        "frames": [str(path) for path in frames],
    }


def _scaled_frame_dimensions(
    frame_width: int | None,
    frame_height: int | None,
    scale_width: int | None,
) -> tuple[int | None, int | None]:
    if not frame_width or not frame_height:
        return None, None
    if not scale_width or scale_width <= 0:
        return int(frame_width), int(frame_height)
    output_width = int(scale_width)
    scaled_height = int(round((float(frame_height) * float(output_width) / max(1.0, float(frame_width))) / 2.0) * 2)
    return output_width, max(2, scaled_height)


def _extract_ffmpeg_sparse_chunk_pipe(
    *,
    ffmpeg_path: str,
    video_path: Path,
    chunk_index: int,
    chunk_start_sec: float,
    chunk_duration_sec: float,
    sample_fps: float,
    scale_width: int | None,
    frame_width: int | None,
    frame_height: int | None,
) -> dict[str, Any]:
    output_width, output_height = _scaled_frame_dimensions(frame_width, frame_height, scale_width)
    if not output_width or not output_height:
        raise RuntimeError("ffmpeg raw pipe extraction requires known output dimensions")
    vf_parts = [f"fps={_ffmpeg_fps_value(sample_fps)}"]
    if scale_width and scale_width > 0:
        vf_parts.append(f"scale={int(scale_width)}:-2")
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *_ffmpeg_hwaccel_args(),
        "-ss",
        f"{max(0.0, chunk_start_sec):.3f}",
        "-t",
        f"{max(0.001, chunk_duration_sec):.3f}",
        "-i",
        str(video_path),
        "-vf",
        ",".join(vf_parts),
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    started = time.perf_counter()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_size = int(output_width) * int(output_height) * 3
    frames: list[dict[str, Any]] = []
    try:
        if process.stdout is None:
            raise RuntimeError("ffmpeg raw pipe stdout unavailable")
        import numpy as np

        while True:
            data = process.stdout.read(frame_size)
            if not data:
                break
            if len(data) != frame_size:
                raise RuntimeError(f"incomplete raw frame from ffmpeg pipe: {len(data)} of {frame_size} bytes")
            frame_number = len(frames)
            frame = np.frombuffer(data, dtype=np.uint8).reshape((int(output_height), int(output_width), 3)).copy()
            frames.append({"frame_number": frame_number, "frame": frame})
        stderr_bytes = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
    wall_sec = time.perf_counter() - started
    if return_code != 0:
        stderr_text = stderr_bytes.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg sparse pipe extraction failed for chunk {chunk_index}: {stderr_text[-800:]}")
    return {
        "chunk_index": chunk_index,
        "chunk_start_sec": float(chunk_start_sec),
        "chunk_duration_sec": float(chunk_duration_sec),
        "wall_sec": wall_sec,
        "frame_count": len(frames),
        "frames": frames,
        "pipe_output_width": int(output_width),
        "pipe_output_height": int(output_height),
    }


def _extract_ffmpeg_seek_frame(
    *,
    ffmpeg_path: str,
    video_path: Path,
    output_root: Path,
    frame_index: int,
    target_time_sec: float,
    scale_width: int | None,
    quality: int,
) -> dict[str, Any]:
    output_path = output_root / f"seek_{frame_index:06d}.jpg"
    vf_parts = []
    if scale_width and scale_width > 0:
        vf_parts.append(f"scale={int(scale_width)}:-2")
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *_ffmpeg_hwaccel_args(),
        "-ss",
        f"{max(0.0, target_time_sec):.3f}",
        "-noaccurate_seek",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
    ]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-q:v", str(max(2, min(31, int(quality)))), str(output_path)]
    started = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wall_sec = time.perf_counter() - started
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(f"ffmpeg seek extraction failed at {target_time_sec:.3f}s: {result.stderr[-800:]}")
    return {
        "frame_index": frame_index,
        "target_time_sec": float(target_time_sec),
        "wall_sec": wall_sec,
        "path": str(output_path),
    }


def _ffmpeg_seek_times(start_sec: float, end_sec: float, sample_fps: float) -> list[float]:
    step = 1.0 / max(0.001, float(sample_fps))
    cursor = max(0.0, float(start_sec))
    end = max(cursor, float(end_sec))
    times: list[float] = []
    while cursor <= end + 0.001:
        times.append(round(cursor, 3))
        cursor += step
    return times


def scan_yolo_video(
    video_path: str | Path | VideoSource | None = None,
    *,
    first_person_path: str | Path | VideoSource | None = None,
    third_person_path: str | Path | VideoSource | None = None,
    preferred_view: str = "first_person",
    source_view: str | None = None,
    model: Any | None = None,
    model_path: str | Path | None = None,
    detector: YoloFrameDetector | None = None,
    sample_fps: float = 1.0,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "auto",
    imgsz: int | None = None,
    adaptive_imgsz: bool = True,
    min_imgsz: int | None = None,
    max_imgsz: int | None = None,
    class_thresholds: dict[str, float] | None = None,
    model_ref: dict[str, Any] | None = None,
    class_schema: dict[str, Any] | None = None,
    annotation_asset_refs: list[dict[str, Any]] | None = None,
    active_threshold: float = 0.55,
    continuity_frames: int = 3,
    dry_run: bool = False,
    mock_rows: list[dict[str, Any]] | None = None,
    mock_duration_sec: float = 960.0,
    batch_size: int | None = None,
    max_sampled_frames: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    timing_callback: Callable[[dict[str, Any]], None] | None = None,
    scan_start_sec: float | None = None,
    scan_end_sec: float | None = None,
) -> list[dict[str, Any]]:
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")
    selected_view, selected_path, video_start_time, source_fps_hint = _select_video_source(
        video_path,
        first_person_path=first_person_path,
        third_person_path=third_person_path,
        preferred_view=preferred_view,
        source_view=source_view,
        dry_run=dry_run,
    )
    early_timing_start = time.perf_counter()
    requested_device = "auto" if device is None else str(device)
    actual_device: str | None = "dry_run" if dry_run or mock_rows is not None else None
    batch_size_env_configured = _yolo_batch_size_env_configured(model_ref)
    resolved_batch_size = max(1, int(batch_size or _default_yolo_batch_size(model_ref)))
    batch_size_source = "argument" if batch_size is not None else ("environment" if batch_size_env_configured else "default")
    gpu_batch_default_applied = False

    def emit_early_timing(rows: list[dict[str, Any]], *, scan_backend: str) -> None:
        if timing_callback is None:
            return
        window_start_sec = max(0.0, float(scan_start_sec or 0.0))
        window_end_sec = float(scan_end_sec) if scan_end_sec is not None else None
        wall_sec = time.perf_counter() - early_timing_start
        try:
            timing_callback(
                {
                    "stage": "yolo_scan",
                    "source_view": selected_view,
                    "video_path": str(selected_path),
                    "scan_start_sec": window_start_sec,
                    "scan_end_sec": window_end_sec,
                    "scan_duration_sec": round(max(0.0, float((window_end_sec or 0.0) - window_start_sec)), 6)
                    if window_end_sec is not None
                    else None,
                    "sample_fps": float(sample_fps),
                    "sampled_frames": len(rows),
                    "read_frames": 0,
                    "grab_frames": 0,
                    "decode_sec": 0.0,
                    "inference_sec": 0.0,
                    "postprocess_sec": 0.0,
                    "wall_sec": round(wall_sec, 6),
                    "effective_sampled_fps": round(len(rows) / wall_sec, 6) if wall_sec > 0 else 0.0,
                    "batch_size": resolved_batch_size,
                    "batch_size_source": batch_size_source,
                    "gpu_batch_default_applied": bool(gpu_batch_default_applied),
                    "scan_backend": scan_backend,
                    "requested_device": requested_device,
                    "actual_device": actual_device,
                }
            )
        except Exception:
            pass

    if mock_rows is not None:
        rows = normalize_yolo_frame_rows(mock_rows, active_threshold=active_threshold, continuity_frames=continuity_frames)
        for row in rows:
            row.setdefault("source_view", selected_view)
            row.setdefault("video_path", str(selected_path))
            row.setdefault("sample_fps", float(sample_fps))
            row.setdefault("requested_yolo_device", requested_device)
            row.setdefault("actual_yolo_device", str(actual_device or ""))
            if video_start_time:
                row.setdefault("video_start_time", video_start_time)
        filtered_rows = _filter_rows_to_scan_window(rows, scan_start_sec=scan_start_sec, scan_end_sec=scan_end_sec)
        emit_early_timing(filtered_rows, scan_backend="mock_rows")
        return filtered_rows
    if dry_run:
        rows = mock_yolo_frame_rows(
            duration_sec=mock_duration_sec,
            sample_fps=sample_fps,
            source_view=selected_view,
            video_path=selected_path,
            active_threshold=active_threshold,
            continuity_frames=continuity_frames,
        )
        for row in rows:
            row.setdefault("requested_yolo_device", requested_device)
            row.setdefault("actual_yolo_device", str(actual_device or ""))
        filtered_rows = _filter_rows_to_scan_window(rows, scan_start_sec=scan_start_sec, scan_end_sec=scan_end_sec)
        emit_early_timing(filtered_rows, scan_backend="dry_run_mock")
        return filtered_rows

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for real YOLO video scanning") from exc

    loaded_model: Any | None = None
    if detector is None:
        actual_device = _resolve_yolo_device(device)
    else:
        actual_device = "custom_detector"
    if batch_size is None and not batch_size_env_configured and _is_yolo_gpu_device(actual_device):
        gpu_batch_size = _gpu_yolo_batch_size_default(model_ref)
        if int(gpu_batch_size) > int(resolved_batch_size):
            resolved_batch_size = int(gpu_batch_size)
            batch_size_source = "gpu_default"
            gpu_batch_default_applied = True
    if detector is None:
        loaded_model = _load_yolo_model(model, model_path)

        def detector(frame: Any) -> list[dict[str, Any]]:
            frame_w, frame_h = _frame_size(frame)
            effective_imgsz = resolve_adaptive_yolo_imgsz(
                frame_w,
                frame_h,
                configured_imgsz=imgsz,
                adaptive=adaptive_imgsz,
                min_imgsz=min_imgsz,
                max_imgsz=max_imgsz,
            )
            return _detections_from_model(
                loaded_model,
                frame,
                conf=conf,
                iou=iou,
                device=actual_device,
                class_thresholds=class_thresholds,
                imgsz=effective_imgsz,
            )

    cap = cv2.VideoCapture(str(selected_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for YOLO scanning: {selected_path}")

    rows: list[dict[str, Any]] = []
    scorer = YoloActivityScorer(active_threshold=active_threshold, continuity_frames=continuity_frames)
    timing_wall_start = time.perf_counter()
    timing_decode_sec = 0.0
    timing_inference_sec = 0.0
    timing_postprocess_sec = 0.0
    timing_read_frames = 0
    timing_grab_frames = 0
    timing_extra: dict[str, Any] = {}
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or source_fps_hint or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_sec = frame_count / fps if fps > 0 and frame_count > 0 else None
        window_start_sec = max(0.0, float(scan_start_sec or 0.0))
        window_end_sec = float(scan_end_sec) if scan_end_sec is not None else (duration_sec if duration_sec is not None else None)
        if window_end_sec is not None:
            window_end_sec = max(window_start_sec, float(window_end_sec))
        start_frame = max(0, int(math.floor(window_start_sec * fps))) if fps > 0 else 0
        if frame_count > 0:
            start_frame = min(start_frame, max(frame_count - 1, 0))
        end_frame = None
        if window_end_sec is not None and fps > 0:
            end_frame = int(math.ceil(window_end_sec * fps))
            if frame_count > 0:
                end_frame = min(end_frame, frame_count - 1)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        sample_every = max(1, int(round(fps / float(sample_fps))))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        effective_imgsz = resolve_adaptive_yolo_imgsz(
            frame_width,
            frame_height,
            configured_imgsz=imgsz,
            adaptive=adaptive_imgsz,
            min_imgsz=min_imgsz,
            max_imgsz=max_imgsz,
        )
        resolution_profile = {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "yolo_imgsz": effective_imgsz,
            "adaptive_imgsz": bool(adaptive_imgsz and not imgsz and not os.environ.get("KEY_ACTION_YOLO_IMGSZ")),
            "low_resolution_input": bool(frame_width and frame_height and min(int(frame_width), int(frame_height)) < 540),
            "high_resolution_input": bool(frame_width and frame_height and (max(int(frame_width), int(frame_height)) >= 1600 or min(int(frame_width), int(frame_height)) >= 900)),
        }
        frame_index = start_frame
        sample_index = 0
        effective_batch_size = resolved_batch_size
        use_model_batch = loaded_model is not None and effective_batch_size > 1
        pending_frames: list[tuple[int, int, Any]] = []
        batch_stats: dict[str, Any] = {}
        last_progress_emit = 0.0

        def emit_progress(*, force: bool = False) -> None:
            nonlocal last_progress_emit
            if progress_callback is None:
                return
            now = time.perf_counter()
            if not force and now - last_progress_emit < 1.5:
                return
            last_progress_emit = now
            frame_progress = (
                max(0.0, min(1.0, float(frame_index - start_frame) / float(max(1, (end_frame or frame_count) - start_frame))))
                if frame_count > 0 or end_frame is not None
                else 0.0
            )
            if max_sampled_frames is not None and int(max_sampled_frames) > 0:
                sample_progress = max(0.0, min(1.0, float(sample_index) / float(max_sampled_frames)))
                scan_progress = max(frame_progress, sample_progress)
            else:
                scan_progress = frame_progress
            try:
                progress_callback(
                    {
                        "stage": "yolo_scan",
                        "source_view": selected_view,
                        "video_path": str(selected_path),
                        "frame_index": frame_index,
                        "frame_count": frame_count,
                        "sample_index": sample_index,
                        "sample_fps": float(sample_fps),
                        "sample_every": sample_every,
                        "batch_size": effective_batch_size,
                        "batch_size_source": batch_size_source,
                        "gpu_batch_default_applied": bool(gpu_batch_default_applied),
                        "yolo_imgsz": effective_imgsz,
                        "requested_device": requested_device,
                        "actual_device": actual_device,
                        "resolution_profile": resolution_profile,
                        "progress": scan_progress,
                        "rows": len(rows),
                        "scan_start_sec": window_start_sec,
                        "scan_end_sec": window_end_sec,
                    }
                )
            except Exception:
                pass

        def append_row(row_frame_index: int, row_sample_index: int, row_frame: Any, raw_detections: list[dict[str, Any]]) -> None:
            detections = _filter_detections(raw_detections, conf=conf, class_thresholds=class_thresholds)
            row_frame_width, row_frame_height = _frame_size(row_frame)
            row = _row_from_detections(
                detections,
                scorer=scorer,
                source_view=selected_view,
                video_path=selected_path,
                frame_index=row_frame_index,
                sample_index=row_sample_index,
                time_sec=row_frame_index / fps if fps > 0 else 0.0,
                sample_fps=sample_fps,
                source_fps=fps,
                frame_width=row_frame_width or frame_width,
                frame_height=row_frame_height or frame_height,
                video_start_time=video_start_time,
                frame=row_frame,
            )
            if model_ref:
                row["model_ref"] = dict(model_ref)
            if class_schema:
                row["class_schema"] = dict(class_schema)
            if annotation_asset_refs:
                row["annotation_asset_refs"] = list(annotation_asset_refs)
            if use_model_batch:
                row["yolo_batch_size"] = effective_batch_size
            if effective_imgsz:
                row["yolo_imgsz"] = int(effective_imgsz)
            row["requested_yolo_device"] = requested_device
            row["actual_yolo_device"] = str(actual_device or "")
            row["resolution_profile"] = dict(resolution_profile)
            rows.append(row)

        def flush_pending() -> None:
            nonlocal timing_inference_sec, timing_postprocess_sec
            if not pending_frames:
                return
            inference_start = time.perf_counter()
            if use_model_batch and loaded_model is not None:
                batch_frames = [item[2] for item in pending_frames]
                batch_detections = _detections_from_model_batch(
                    loaded_model,
                    batch_frames,
                    conf=conf,
                    iou=iou,
                    device=actual_device,
                    class_thresholds=class_thresholds,
                    imgsz=effective_imgsz,
                    stats=batch_stats,
                )
            else:
                _record_batch_sizes(batch_stats, len(pending_frames))
                if loaded_model is not None:
                    _record_batch_stat(batch_stats, "frame_predict_calls", len(pending_frames))
                    _record_batch_stat(batch_stats, "predict_call_count", len(pending_frames))
                else:
                    _record_batch_stat(batch_stats, "custom_detector_calls", len(pending_frames))
                batch_detections = [detector(item[2]) for item in pending_frames]
            timing_inference_sec += time.perf_counter() - inference_start
            postprocess_start = time.perf_counter()
            for (row_frame_index, row_sample_index, row_frame), raw_detections in zip(pending_frames, batch_detections):
                append_row(row_frame_index, row_sample_index, row_frame, raw_detections)
            timing_postprocess_sec += time.perf_counter() - postprocess_start
            pending_frames.clear()

        scan_end_for_chunks = float(window_end_sec) if window_end_sec is not None else float(duration_sec or 0.0)
        scan_duration_for_ffmpeg = max(0.0, scan_end_for_chunks - float(window_start_sec))
        model_ref_dict = model_ref if isinstance(model_ref, dict) else None
        sparse_scan_enabled = _use_sparse_scan_for_scan(
            sample_fps,
            max_sampled_frames,
            scan_duration_for_ffmpeg,
            model_ref_dict,
        )
        sparse_mode = _resolve_ffmpeg_sparse_mode_for_scan(sample_fps, scan_duration_for_ffmpeg, model_ref_dict)
        ffmpeg_path = shutil.which("ffmpeg")
        if sparse_scan_enabled and sparse_mode != "opencv_seek" and not ffmpeg_path and _is_long_video_coarse_scan(model_ref_dict):
            sparse_mode = "opencv_seek"
        if sparse_scan_enabled and (sparse_mode == "opencv_seek" or ffmpeg_path):
            cap.release()
            ffmpeg_chunk_sec = _ffmpeg_chunk_sec_for_scan(model_ref_dict)
            ffmpeg_workers = _ffmpeg_worker_count_for_scan(model_ref_dict)
            ffmpeg_quality = _env_int("KEY_ACTION_YOLO_FFMPEG_JPEG_QUALITY", 5)
            scale_width = _ffmpeg_scale_width_for_scan(
                model_ref_dict,
                int(effective_imgsz or 640),
            )
            if (
                scale_width
                and scale_width > 0
                and imgsz is None
                and effective_imgsz
                and int(scale_width) < int(effective_imgsz)
            ):
                effective_imgsz = max(32, int(round(float(scale_width) / 32.0)) * 32)
                resolution_profile["yolo_imgsz"] = effective_imgsz
                resolution_profile["sparse_scaled_input"] = True
                resolution_profile["sparse_scale_width"] = int(scale_width)
            extract_started = time.perf_counter()
            with tempfile.TemporaryDirectory(prefix="key_action_yolo_sparse_") as temp_dir:
                temp_root = Path(temp_dir)
                if sparse_mode == "opencv_seek":
                    seek_times = _ffmpeg_seek_times(window_start_sec, scan_end_for_chunks, float(sample_fps))
                    seek_cap = cv2.VideoCapture(str(selected_path))
                    if not seek_cap.isOpened():
                        raise RuntimeError(f"Cannot open video for OpenCV sparse seek: {selected_path}")
                    seek_wall_min: float | None = None
                    seek_wall_max = 0.0
                    try:
                        for target_time in seek_times:
                            seek_start = time.perf_counter()
                            seek_cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(target_time)) * 1000.0)
                            ok, frame = seek_cap.read()
                            seek_wall = time.perf_counter() - seek_start
                            seek_wall_min = seek_wall if seek_wall_min is None else min(seek_wall_min, seek_wall)
                            seek_wall_max = max(seek_wall_max, seek_wall)
                            if not ok or frame is None:
                                continue
                            if scale_width and scale_width > 0:
                                current_h, current_w = frame.shape[:2]
                                scaled_w, scaled_h = _scaled_frame_dimensions(current_w, current_h, scale_width)
                                if scaled_w and scaled_h and (int(scaled_w) != current_w or int(scaled_h) != current_h):
                                    frame = cv2.resize(frame, (int(scaled_w), int(scaled_h)), interpolation=cv2.INTER_AREA)
                            local_time_sec = float(target_time)
                            if window_end_sec is not None and local_time_sec > float(window_end_sec) + 0.001:
                                continue
                            row_frame_index = int(round(local_time_sec * fps)) if fps > 0 else sample_index
                            frame_index = row_frame_index
                            pending_frames.append((row_frame_index, sample_index, frame))
                            timing_read_frames += 1
                            if len(pending_frames) >= effective_batch_size:
                                flush_pending()
                            sample_index += 1
                            emit_progress()
                    finally:
                        seek_cap.release()
                    timing_decode_sec += time.perf_counter() - extract_started
                    timing_extra = {
                        "scan_backend": "opencv_sparse_seek",
                        "opencv_seek_count": len(seek_times),
                        "opencv_extracted_frames": timing_read_frames,
                        "opencv_extract_wall_sec": round(timing_decode_sec, 6),
                        "opencv_seek_step_sec": round(1.0 / max(float(sample_fps), 0.001), 6),
                        "opencv_seek_wall_sec_min": round(float(seek_wall_min or 0.0), 6),
                        "opencv_seek_wall_sec_max": round(float(seek_wall_max), 6),
                        "opencv_scale_width": scale_width,
                    }
                elif sparse_mode == "seek":
                    if not ffmpeg_path:
                        raise RuntimeError("ffmpeg sparse seek requested but ffmpeg is not available")
                    seek_times = _ffmpeg_seek_times(window_start_sec, scan_end_for_chunks, float(sample_fps))
                    seek_results: list[dict[str, Any]] = []
                    with ThreadPoolExecutor(max_workers=min(ffmpeg_workers, max(1, len(seek_times)))) as executor:
                        futures = [
                            executor.submit(
                                _extract_ffmpeg_seek_frame,
                                ffmpeg_path=ffmpeg_path,
                                video_path=Path(selected_path),
                                output_root=temp_root,
                                frame_index=index,
                                target_time_sec=target_time,
                                scale_width=scale_width,
                                quality=ffmpeg_quality,
                            )
                            for index, target_time in enumerate(seek_times)
                        ]
                        for future in as_completed(futures):
                            seek_results.append(future.result())
                    timing_decode_sec += time.perf_counter() - extract_started
                    seek_results.sort(key=lambda item: int(item.get("frame_index", 0)))
                    timing_extra = {
                        "scan_backend": "ffmpeg_sparse_seek",
                        "ffmpeg_workers": ffmpeg_workers,
                        "ffmpeg_scale_width": scale_width,
                        "ffmpeg_seek_count": len(seek_results),
                        "ffmpeg_extracted_frames": len(seek_results),
                        "ffmpeg_extract_wall_sec": round(timing_decode_sec, 6),
                        "ffmpeg_seek_step_sec": round(1.0 / max(float(sample_fps), 0.001), 6),
                        "ffmpeg_seek_wall_sec_min": round(min((float(item.get("wall_sec") or 0.0) for item in seek_results), default=0.0), 6),
                        "ffmpeg_seek_wall_sec_max": round(max((float(item.get("wall_sec") or 0.0) for item in seek_results), default=0.0), 6),
                    }
                    for item in seek_results:
                        local_time_sec = float(item.get("target_time_sec") or 0.0)
                        if window_end_sec is not None and local_time_sec > float(window_end_sec) + 0.001:
                            continue
                        frame = cv2.imread(str(item.get("path")))
                        if frame is None:
                            continue
                        row_frame_index = int(round(local_time_sec * fps)) if fps > 0 else sample_index
                        frame_index = row_frame_index
                        pending_frames.append((row_frame_index, sample_index, frame))
                        timing_read_frames += 1
                        if len(pending_frames) >= effective_batch_size:
                            flush_pending()
                        sample_index += 1
                        emit_progress()
                else:
                    if not ffmpeg_path:
                        raise RuntimeError("ffmpeg sparse chunk extraction requested but ffmpeg is not available")
                    chunk_specs = _ffmpeg_sparse_chunks(
                        start_sec=window_start_sec,
                        end_sec=scan_end_for_chunks,
                        chunk_sec=ffmpeg_chunk_sec,
                    )
                    chunk_results: list[dict[str, Any]] = []
                    max_pipe_frames = max(1, _env_int("KEY_ACTION_YOLO_FFMPEG_PIPE_MAX_FRAMES_PER_CHUNK", 450))
                    pipe_scan = _env_bool("KEY_ACTION_YOLO_FFMPEG_PIPE_SCAN", True) and bool(frame_width and frame_height)
                    pipe_scan = bool(
                        pipe_scan
                        and all(
                            int(math.ceil(float(chunk_duration) * max(float(sample_fps), 0.001))) <= max_pipe_frames
                            for _chunk_index, _chunk_start, chunk_duration in chunk_specs
                        )
                    )
                    extractor = _extract_ffmpeg_sparse_chunk_pipe if pipe_scan else _extract_ffmpeg_sparse_chunk

                    def consume_chunk_result(chunk: dict[str, Any]) -> None:
                        nonlocal sample_index, timing_read_frames
                        chunk_start = float(chunk.get("chunk_start_sec") or 0.0)
                        frame_items = list(chunk.get("frames") or [])
                        for frame_item in frame_items:
                            frame = None
                            if pipe_scan:
                                frame_number = int(frame_item.get("frame_number") or 0)
                                frame = frame_item.get("frame")
                            else:
                                frame_path = Path(str(frame_item))
                                try:
                                    frame_number = max(0, int(frame_path.stem.split("_")[-1]) - 1)
                                except ValueError:
                                    frame_number = 0
                                frame = cv2.imread(str(frame_path))
                            local_time_sec = chunk_start + (float(frame_number) / max(float(sample_fps), 0.001))
                            if window_end_sec is not None and local_time_sec > float(window_end_sec) + 0.001:
                                continue
                            if frame is None:
                                continue
                            row_frame_index = int(round(local_time_sec * fps)) if fps > 0 else sample_index
                            pending_frames.append((row_frame_index, sample_index, frame))
                            timing_read_frames += 1
                            if len(pending_frames) >= effective_batch_size:
                                flush_pending()
                            sample_index += 1
                            emit_progress()

                    with ThreadPoolExecutor(max_workers=min(ffmpeg_workers, max(1, len(chunk_specs)))) as executor:
                        futures = []
                        for chunk_index, chunk_start, chunk_duration in chunk_specs:
                            common_kwargs = {
                                "ffmpeg_path": ffmpeg_path,
                                "video_path": Path(selected_path),
                                "chunk_index": chunk_index,
                                "chunk_start_sec": chunk_start,
                                "chunk_duration_sec": chunk_duration,
                                "sample_fps": float(sample_fps),
                                "scale_width": scale_width,
                            }
                            if pipe_scan:
                                common_kwargs.update({"frame_width": frame_width, "frame_height": frame_height})
                            else:
                                common_kwargs.update({"output_root": temp_root, "quality": ffmpeg_quality})
                            futures.append(executor.submit(extractor, **common_kwargs))
                        for future in as_completed(futures):
                            chunk = future.result()
                            chunk_results.append(chunk)
                            consume_chunk_result(chunk)
                    timing_decode_sec += time.perf_counter() - extract_started
                    chunk_results.sort(key=lambda item: int(item.get("chunk_index", 0)))
                    timing_extra = {
                        "scan_backend": "ffmpeg_sparse_pipe_chunks" if pipe_scan else "ffmpeg_sparse_chunks",
                        "ffmpeg_chunk_sec": ffmpeg_chunk_sec,
                        "ffmpeg_workers": ffmpeg_workers,
                        "ffmpeg_scale_width": scale_width,
                        "ffmpeg_chunk_count": len(chunk_results),
                        "ffmpeg_extracted_frames": sum(int(item.get("frame_count") or 0) for item in chunk_results),
                        "ffmpeg_extract_wall_sec": round(timing_decode_sec, 6),
                        "ffmpeg_pipe_scan": bool(pipe_scan),
                        "ffmpeg_pipe_max_frames_per_chunk": max_pipe_frames,
                        "ffmpeg_pipe_output_width": int(chunk_results[0].get("pipe_output_width") or 0) if pipe_scan and chunk_results else None,
                        "ffmpeg_pipe_output_height": int(chunk_results[0].get("pipe_output_height") or 0) if pipe_scan and chunk_results else None,
                        "ffmpeg_chunk_timings": [
                            {
                                "chunk_index": int(item.get("chunk_index", 0)),
                                "chunk_start_sec": float(item.get("chunk_start_sec", 0.0)),
                                "chunk_duration_sec": float(item.get("chunk_duration_sec", 0.0)),
                                "wall_sec": round(float(item.get("wall_sec") or 0.0), 6),
                                "frame_count": int(item.get("frame_count") or 0),
                            }
                            for item in chunk_results
                        ],
                    }
                flush_pending()
                emit_progress(force=True)
        else:
            timing_extra = {
                "scan_backend": "opencv_frame_skip",
                "opencv_sample_every": int(sample_every),
            }
            while True:
                if end_frame is not None and frame_index > end_frame:
                    break
                if (frame_index - start_frame) % sample_every == 0:
                    decode_start = time.perf_counter()
                    ok, frame = cap.read()
                    timing_decode_sec += time.perf_counter() - decode_start
                    if not ok:
                        break
                    timing_read_frames += 1
                    pending_frames.append((frame_index, sample_index, frame))
                    if len(pending_frames) >= effective_batch_size:
                        flush_pending()
                    sample_index += 1
                    emit_progress()
                    if max_sampled_frames is not None and sample_index >= int(max_sampled_frames):
                        break
                else:
                    decode_start = time.perf_counter()
                    ok = cap.grab()
                    timing_decode_sec += time.perf_counter() - decode_start
                    if not ok:
                        break
                    timing_grab_frames += 1
                frame_index += 1
            flush_pending()
            emit_progress(force=True)
    finally:
        cap.release()
    if timing_callback is not None:
        try:
            wall_sec = time.perf_counter() - timing_wall_start
            timing_callback(
                {
                    "stage": "yolo_scan",
                    "source_view": selected_view,
                    "video_path": str(selected_path),
                    "scan_start_sec": float(window_start_sec),
                    "scan_end_sec": float(window_end_sec) if window_end_sec is not None else None,
                    "scan_duration_sec": round(max(0.0, float((window_end_sec or 0.0) - window_start_sec)), 6)
                    if window_end_sec is not None
                    else None,
                    "sample_fps": float(sample_fps),
                    "sampled_frames": len(rows),
                    "read_frames": timing_read_frames,
                    "grab_frames": timing_grab_frames,
                    "decode_sec": round(timing_decode_sec, 6),
                    "inference_sec": round(timing_inference_sec, 6),
                    "postprocess_sec": round(timing_postprocess_sec, 6),
                    "wall_sec": round(wall_sec, 6),
                    "effective_sampled_fps": round(len(rows) / wall_sec, 6) if wall_sec > 0 else 0.0,
                    "batch_size": effective_batch_size,
                    "batch_size_source": batch_size_source,
                    "gpu_batch_default_applied": bool(gpu_batch_default_applied),
                    "gpu_device_observed": _is_yolo_gpu_device(actual_device),
                    "device_fallback": (
                        "auto_to_cpu"
                        if str(requested_device or "").strip().lower() in {"", "none", "auto"}
                        and str(actual_device or "").strip().lower() == "cpu"
                        else ""
                    ),
                    "requested_device": requested_device,
                    "actual_device": actual_device,
                    **_batch_diagnostics(batch_stats, effective_batch_size),
                    **timing_extra,
                }
            )
        except Exception:
            pass
    return rows


def _filter_rows_to_scan_window(
    rows: list[dict[str, Any]],
    *,
    scan_start_sec: float | None = None,
    scan_end_sec: float | None = None,
) -> list[dict[str, Any]]:
    if scan_start_sec is None and scan_end_sec is None:
        return rows
    start = float(scan_start_sec or 0.0)
    end = float(scan_end_sec) if scan_end_sec is not None else None
    filtered = []
    for row in rows:
        try:
            time_sec = float(row.get("time_sec", row.get("local_time_sec", 0.0)) or 0.0)
        except (TypeError, ValueError):
            continue
        if time_sec < start:
            continue
        if end is not None and time_sec > end:
            continue
        filtered.append(row)
    return filtered


def scan_yolo_video_with_summary(*args: Any, **kwargs: Any) -> YoloScanResult:
    rows = scan_yolo_video(*args, **kwargs)
    if not rows:
        return YoloScanResult(
            rows=[],
            source_view="",
            video_path="",
            fps=0.0,
            sample_fps=float(kwargs.get("sample_fps", 1.0)),
            sampled_frames=0,
            source_frame_count=0,
            duration_sec=0.0,
        )
    sample_period = _estimate_sample_period(rows)
    duration_sec = max(float(row.get("time_sec", 0.0)) for row in rows) + sample_period
    source_fps = float(rows[0].get("source_fps", 0.0) or 0.0)
    source_frame_count = int(round(duration_sec * source_fps)) if source_fps > 0 else 0
    return YoloScanResult(
        rows=rows,
        source_view=str(rows[0].get("source_view") or ""),
        video_path=str(rows[0].get("video_path") or ""),
        fps=source_fps,
        sample_fps=float(rows[0].get("sample_fps", kwargs.get("sample_fps", 1.0))),
        sampled_frames=len(rows),
        source_frame_count=source_frame_count,
        duration_sec=float(duration_sec),
    )


def _estimate_sample_period(rows: list[dict[str, Any]]) -> float:
    times = sorted(float(row.get("time_sec", 0.0)) for row in rows)
    if len(times) < 2:
        return float(1.0 / float(rows[0].get("sample_fps", 1.0) or 1.0)) if rows else 0.0
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    if not deltas:
        return float(1.0 / float(rows[0].get("sample_fps", 1.0) or 1.0))
    return float(median(deltas))


def _video_source_from_rows(rows: list[dict[str, Any]], video_source: VideoSource | None = None) -> VideoSource:
    if video_source is not None:
        return video_source
    first = rows[0] if rows else {}
    return VideoSource(
        name=str(first.get("source_view") or "yolo"),
        path=str(first.get("video_path") or "unknown.mp4"),
        start_time=str(first.get("video_start_time") or "1970-01-01T00:00:00+00:00"),
        fps=float(first.get("source_fps")) if first.get("source_fps") is not None else None,
    )


def build_segments_from_yolo_frame_rows(
    rows: list[dict[str, Any]],
    *,
    video_source: VideoSource | None = None,
    duration_sec: float | None = None,
    start_threshold: float = 0.6,
    end_threshold: float = 0.3,
    start_min_duration_sec: float = 2.0,
    end_min_duration_sec: float = 5.0,
    merge_gap_sec: float = 5.0,
    min_segment_duration_sec: float = 5.0,
    buffer_sec: float = 2.0,
) -> list[DetectedSegment]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda item: float(item.get("time_sec", 0.0)))
    source = _video_source_from_rows(ordered, video_source=video_source)
    scores: list[FrameScore] = []
    for row in ordered:
        active_score = float(row.get("active_score", 0.0))
        interaction_score = float(row.get("interaction_score", active_score))
        keep_score = float(row.get("keep_score", 1.0 if bool(row.get("is_experiment_active", active_score >= start_threshold)) else 0.0))
        scores.append(
            FrameScore(
                time_sec=float(row.get("time_sec", 0.0)),
                frame_index=int(row.get("frame_index", 0)),
                local_time_sec=float(row.get("local_time_sec", row.get("time_sec", 0.0))),
                global_time=row.get("global_time"),
                motion_score=interaction_score,
                raw_score=float(row.get("raw_score", interaction_score)),
                probability=float(row.get("probability", interaction_score)),
                prob=float(row.get("prob", interaction_score)),
                prob_score=float(row.get("prob_score", interaction_score)),
                raw_prob=float(row.get("raw_prob", interaction_score)),
                motion_prob=float(row.get("motion_prob", active_score)),
                active_score=active_score,
                keep=bool(row.get("keep", active_score >= start_threshold)),
                keep_score=keep_score,
                roi=row.get("roi"),
                is_active=bool(row.get("is_experiment_active", active_score >= start_threshold)),
            )
        )
    inferred_duration = max(float(row.get("time_sec", 0.0)) for row in ordered) + _estimate_sample_period(ordered)
    config = DetectorConfig(
        sample_fps=float(ordered[0].get("sample_fps", 1.0) or 1.0),
        start_threshold=float(start_threshold),
        end_threshold=float(end_threshold),
        start_min_duration_sec=float(start_min_duration_sec),
        end_min_duration_sec=float(end_min_duration_sec),
        merge_gap_sec=float(merge_gap_sec),
        min_segment_duration_sec=float(min_segment_duration_sec),
        buffer_sec=float(buffer_sec),
    )
    return build_segments_from_scores(scores, source, duration_sec=float(duration_sec or inferred_duration), config=config)


detect_segments_from_yolo_rows = build_segments_from_yolo_frame_rows
scan_video_with_yolo = scan_yolo_video
