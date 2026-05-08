from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable

from .action_detector import build_segments_from_scores
from .config import DetectorConfig
from .schemas import DetectedSegment, FrameScore, VideoSource


HAND_LABELS = frozenset({"gloved_hand", "hand"})
INTERACTION_OBJECT_LABELS = frozenset(
    {
        "sample_bottle",
        "sample_bottle_blue",
        "reagent_bottle",
        "balance",
        "beaker",
        "container",
        "spatula",
        "pipette",
        "pipette_tip",
        "paper",
        "tube",
        "tube_cap",
    }
)
EXPERIMENT_CONTEXT_LABELS = HAND_LABELS | INTERACTION_OBJECT_LABELS | frozenset({"lab_coat", "ppe_storage"})

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
    "electronic_balance": "balance",
    "scale": "balance",
    "weighing_scale": "balance",
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
    "beaker": "beaker",
    "container": "container",
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
    if detection.get("track_id") is not None:
        normalized["track_id"] = detection.get("track_id")
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


def _crop_stats(frame: Any | None, detection: dict[str, Any]) -> dict[str, float | bool]:
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
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {}
        pixels = crop.reshape(-1, 3).astype("float32")
        mean_b, mean_g, mean_r = [float(value) for value in pixels.mean(axis=0)]
        channel_std = float(pixels.std(axis=0).mean())
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.count_nonzero(edges)) / max(1.0, float(edges.size))
        brightness = (mean_b + mean_g + mean_r) / 3.0
        blue_family = (mean_b >= mean_r + 25.0 and mean_g >= mean_r + 12.0) or (
            mean_b >= mean_g + 12.0 and mean_b >= mean_r + 28.0
        )
        low_texture = channel_std <= 42.0 and edge_density <= 0.075
        return {
            "mean_b": mean_b,
            "mean_g": mean_g,
            "mean_r": mean_r,
            "brightness": brightness,
            "channel_std": channel_std,
            "edge_density": edge_density,
            "blue_family": bool(blue_family),
            "low_texture": bool(low_texture),
        }
    except Exception:
        return {}


def _is_flat_blue_background_like(
    detection: dict[str, Any],
    *,
    frame: Any | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> bool:
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if area_ratio < 0.012 or width_ratio < 0.10 or height_ratio < 0.06:
        return False
    stats = _crop_stats(frame, detection)
    if not stats:
        return False
    return bool(stats.get("blue_family")) and bool(stats.get("low_texture")) and area_ratio >= 0.018


def _is_implausible_hand_detection(
    detection: dict[str, Any],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    source_view: str | None = None,
) -> bool:
    if str(detection.get("label") or "") not in HAND_LABELS:
        return False
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if area_ratio <= 0.0:
        return False
    confidence = float(detection.get("confidence", 0.0) or 0.0)
    if _is_flat_blue_background_like(detection, frame=frame, frame_width=frame_width, frame_height=frame_height):
        return True
    view = str(source_view or "").strip().lower()
    if view == "third_person":
        if confidence < 0.55:
            return True
        if area_ratio > 0.18:
            return True
        if area_ratio > 0.12 and (width_ratio > 0.36 or height_ratio > 0.44):
            return True
        touches_x, _touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
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
    touches_x, touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
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
) -> str | None:
    label = str(detection.get("label") or "")
    width_ratio, height_ratio, area_ratio = _bbox_size_ratios(detection, frame_width, frame_height)
    if label in HAND_LABELS and _is_implausible_hand_detection(
        detection,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    ):
        return "implausible_hand_bbox_or_background"

    if area_ratio <= 0.0:
        return None
    aspect = _bbox_aspect_ratio(detection)
    touches_x, touches_y = _bbox_edge_contact(detection, frame_width, frame_height)
    flat_blue = _is_flat_blue_background_like(detection, frame=frame, frame_width=frame_width, frame_height=frame_height)

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
            proximity, distance_px = _proximity_score(
                hand["bbox"],
                obj["bbox"],
                frame_width=frame_width,
                frame_height=frame_height,
            )
            iou_score = min(1.0, iou / 0.12) if iou > 0 else 0.0
            score = max(iou_score, proximity)
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
        **scored,
    }
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
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ultralytics is required when detector is not provided") from exc
    return YOLO(str(model_path))


def _detections_from_model(
    model: Any,
    frame: Any,
    *,
    conf: float,
    iou: float,
    device: str,
    class_thresholds: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    model_conf = float(conf)
    if class_thresholds:
        try:
            model_conf = min([model_conf, *[float(value) for value in class_thresholds.values()]])
        except Exception:
            model_conf = float(conf)
    results = model.predict(source=frame, conf=max(0.01, model_conf), iou=float(iou), device=device, verbose=False)
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
    device: str = "cpu",
    class_thresholds: dict[str, float] | None = None,
    model_ref: dict[str, Any] | None = None,
    class_schema: dict[str, Any] | None = None,
    annotation_asset_refs: list[dict[str, Any]] | None = None,
    active_threshold: float = 0.55,
    continuity_frames: int = 3,
    dry_run: bool = False,
    mock_rows: list[dict[str, Any]] | None = None,
    mock_duration_sec: float = 960.0,
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
    if mock_rows is not None:
        rows = normalize_yolo_frame_rows(mock_rows, active_threshold=active_threshold, continuity_frames=continuity_frames)
        for row in rows:
            row.setdefault("source_view", selected_view)
            row.setdefault("video_path", str(selected_path))
            row.setdefault("sample_fps", float(sample_fps))
            if video_start_time:
                row.setdefault("video_start_time", video_start_time)
        return rows
    if dry_run:
        return mock_yolo_frame_rows(
            duration_sec=mock_duration_sec,
            sample_fps=sample_fps,
            source_view=selected_view,
            video_path=selected_path,
            active_threshold=active_threshold,
            continuity_frames=continuity_frames,
        )

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for real YOLO video scanning") from exc

    if detector is None:
        loaded_model = _load_yolo_model(model, model_path)

        def detector(frame: Any) -> list[dict[str, Any]]:
            return _detections_from_model(
                loaded_model,
                frame,
                conf=conf,
                iou=iou,
                device=device,
                class_thresholds=class_thresholds,
            )

    cap = cv2.VideoCapture(str(selected_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for YOLO scanning: {selected_path}")

    rows: list[dict[str, Any]] = []
    scorer = YoloActivityScorer(active_threshold=active_threshold, continuity_frames=continuity_frames)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or source_fps_hint or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_every = max(1, int(round(fps / float(sample_fps))))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        frame_index = 0
        sample_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue
            raw_detections = detector(frame)
            detections = _filter_detections(raw_detections, conf=conf, class_thresholds=class_thresholds)
            row = _row_from_detections(
                detections,
                scorer=scorer,
                source_view=selected_view,
                video_path=selected_path,
                frame_index=frame_index,
                sample_index=sample_index,
                time_sec=frame_index / fps if fps > 0 else 0.0,
                sample_fps=sample_fps,
                source_fps=fps,
                frame_width=frame_width,
                frame_height=frame_height,
                video_start_time=video_start_time,
                frame=frame,
            )
            if model_ref:
                row["model_ref"] = dict(model_ref)
            if class_schema:
                row["class_schema"] = dict(class_schema)
            if annotation_asset_refs:
                row["annotation_asset_refs"] = list(annotation_asset_refs)
            rows.append(row)
            sample_index += 1
            frame_index += 1
    finally:
        cap.release()
    return rows


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
        scores.append(
            FrameScore(
                time_sec=float(row.get("time_sec", 0.0)),
                frame_index=int(row.get("frame_index", 0)),
                local_time_sec=float(row.get("local_time_sec", row.get("time_sec", 0.0))),
                global_time=row.get("global_time"),
                motion_score=float(row.get("interaction_score", active_score)),
                active_score=active_score,
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
