from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def render_segmentation_preview(
    *,
    video_path: str | Path,
    segmentation: Any,
    output_path: str | Path,
    sample_interval_sec: Optional[float] = None,
    output_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    max_width: Optional[int] = None,
    time_range: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
    detector: Optional[Any] = None,
    yolo_overlay: Optional[bool] = None,
) -> Dict[str, Any]:
    """Render a fast time-lapse video that visualizes detected experiment segments."""
    source = Path(video_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for segmentation preview: {source}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    source_duration = total_frames / source_fps if source_fps > 0 and total_frames > 0 else 0.0
    if source_duration <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid video metadata for segmentation preview: {source}")

    if time_range is not None:
        range_start = max(0.0, min(float(time_range[0]), source_duration))
        range_end = max(range_start, min(float(time_range[1]), source_duration))
    else:
        range_start = 0.0
        range_end = source_duration
    render_duration = max(range_end - range_start, 0.001)

    target_fps = float(output_fps or os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_FPS", "6"))
    frame_budget = int(max_frames or os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_MAX_FRAMES", "720"))
    min_interval = float(sample_interval_sec or os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_INTERVAL_SEC", "5"))
    sample_interval = max(min_interval, render_duration / max(frame_budget, 1))

    target_width = int(max_width or os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_MAX_WIDTH", "960"))
    scale = min(1.0, target_width / max(width, 1))
    out_w = max(320, int(width * scale))
    out_h = max(180, int(height * scale))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1

    raw_output = output.with_name(f"{output.stem}.raw{output.suffix}")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open segmentation preview writer: {raw_output}")

    segments = _segments_from(segmentation)
    total_segments = len(segments)
    written = 0
    detection_count = 0
    interaction_count = 0
    action_labels: Dict[str, int] = {}
    detector_errors = 0
    overlay_enabled = bool(detector) and (
        yolo_overlay
        if yolo_overlay is not None
        else os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_YOLO_ENABLED", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    sample_count = min(frame_budget, int(math.ceil(render_duration / sample_interval)) + 1)

    try:
        for sample_index in range(sample_count):
            timestamp = min(range_end, range_start + sample_index * sample_interval)
            frame_idx = min(total_frames - 1, int(round(timestamp * source_fps)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            original_h, original_w = frame.shape[:2]
            detections: List[Dict[str, Any]] = []
            interactions: List[Dict[str, Any]] = []
            action_label = ""
            if overlay_enabled:
                try:
                    raw_detections = _run_detector(detector, frame, frame_idx, timestamp)
                    detections = _scale_detections(raw_detections, original_w, original_h, out_w, out_h)
                    interactions = _find_interactions(detections)
                    action_label = _key_action_label(interactions, detections)
                    detection_count += len(detections)
                    interaction_count += len(interactions)
                    if action_label:
                        action_labels[action_label] = action_labels.get(action_label, 0) + 1
                except Exception as exc:
                    detector_errors += 1
                    logger.warning("YOLO preview overlay failed at %.2fs: %s", timestamp, exc)
            frame = cv2.resize(frame, (out_w, out_h))
            if overlay_enabled:
                frame = _draw_yolo_evidence_overlay(frame, detections, interactions, action_label)
            frame = _draw_overlay(frame, timestamp, source_duration, segments, total_segments, title_override=title)
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()

    _transcode_for_browser(raw_output, output)
    if raw_output.exists() and raw_output != output:
        raw_output.unlink(missing_ok=True)

    manifest = {
        "schema_version": "segmentation_preview.v1",
        "video_path": str(source),
        "output_path": str(output),
        "source_duration_sec": round(source_duration, 3),
        "render_start_sec": round(range_start, 3),
        "render_end_sec": round(range_end, 3),
        "sample_interval_sec": round(sample_interval, 3),
        "output_fps": round(target_fps, 3),
        "frame_count": written,
        "experiment_segment_count": total_segments,
        "yolo_overlay_enabled": overlay_enabled,
        "yolo_detection_count": detection_count,
        "hand_object_interaction_count": interaction_count,
        "key_action_labels": action_labels,
        "detector_error_count": detector_errors,
        "annotation_mode": "yolo_hand_object_key_action" if overlay_enabled else "segmentation_timeline",
    }
    return manifest


def _segments_from(segmentation: Any) -> List[Dict[str, Any]]:
    raw_segments: Iterable[Any] = getattr(segmentation, "segments", []) or []
    segments: List[Dict[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        if isinstance(segment, dict):
            start = float(segment.get("start_sec") or 0.0)
            end = float(segment.get("end_sec") or start)
            seg_index = int(segment.get("index") or index)
            display_name = str(segment.get("display_name") or "")
        else:
            start = float(getattr(segment, "start_sec", 0.0))
            end = float(getattr(segment, "end_sec", start))
            seg_index = int(getattr(segment, "index", index))
            display_name = str(getattr(segment, "display_name", "") or "")
        if end > start:
            segments.append({"index": float(seg_index), "start_sec": start, "end_sec": end, "display_name": display_name})
    return sorted(segments, key=lambda item: item["start_sec"])


def _run_detector(detector: Any, frame: np.ndarray, frame_idx: int, timestamp: float) -> List[Any]:
    if detector is None:
        return []
    if hasattr(detector, "detect_frame"):
        return list(detector.detect_frame(frame) or [])
    if hasattr(detector, "_run_yolo"):
        return list(detector._run_yolo(frame, frame_idx, timestamp) or [])
    return []


def _scale_detections(
    raw_detections: List[Any],
    source_w: int,
    source_h: int,
    target_w: int,
    target_h: int,
) -> List[Dict[str, Any]]:
    sx = target_w / max(source_w, 1)
    sy = target_h / max(source_h, 1)
    max_boxes = int(os.environ.get("LABSOPGUARD_SEGMENTATION_PREVIEW_MAX_BOXES", "12"))
    detections: List[Dict[str, Any]] = []
    for raw in raw_detections:
        item = _normalize_detection(raw)
        if item is None:
            continue
        x1, y1, x2, y2 = item["bbox"]
        item["bbox"] = _clip_bbox((x1 * sx, y1 * sy, x2 * sx, y2 * sy), target_w, target_h)
        if item["bbox"][2] - item["bbox"][0] < 2 or item["bbox"][3] - item["bbox"][1] < 2:
            continue
        detections.append(item)
    detections.sort(key=lambda det: float(det.get("confidence") or 0.0), reverse=True)
    return detections[:max(1, max_boxes)]


def _normalize_detection(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        bbox = raw.get("bbox") or raw.get("xyxy")
        label = raw.get("class_name") or raw.get("label") or raw.get("object_type")
        confidence = raw.get("confidence", raw.get("score", 0.0))
    else:
        bbox = getattr(raw, "bbox", None)
        label = getattr(raw, "class_name", None) or getattr(raw, "label", None) or getattr(raw, "object_type", None)
        confidence = getattr(raw, "confidence", getattr(raw, "score", 0.0))
    if bbox is None or label is None:
        return None
    try:
        values = [float(v) for v in list(bbox)[:4]]
    except Exception:
        return None
    if len(values) != 4:
        return None
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return {
        "bbox": (x1, y1, x2, y2),
        "class_name": str(label),
        "confidence": _safe_float(confidence, 0.0),
    }


def _clip_bbox(bbox: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def _find_interactions(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    hands = [det for det in detections if _is_hand(det["class_name"])]
    objects = [det for det in detections if _is_interaction_object(det["class_name"])]
    interactions: List[Dict[str, Any]] = []
    for hand in hands:
        for obj in objects:
            distance = _bbox_edge_distance(hand["bbox"], obj["bbox"])
            iou = _bbox_iou(hand["bbox"], obj["bbox"])
            if iou >= 0.01 or distance <= 36:
                interactions.append(
                    {
                        "hand": hand,
                        "object": obj,
                        "iou": round(iou, 4),
                        "distance_px": round(distance, 2),
                        "confidence": round(min(0.98, (hand["confidence"] + obj["confidence"]) / 2 + (0.18 if iou > 0 else 0.08)), 3),
                    }
                )
    interactions.sort(key=lambda item: (float(item["confidence"]), -float(item["distance_px"])), reverse=True)
    return interactions[:4]


def _key_action_label(interactions: List[Dict[str, Any]], detections: List[Dict[str, Any]]) -> str:
    if not interactions:
        if any(_is_hand(det["class_name"]) for det in detections):
            return "Key action: hand visible"
        return ""
    object_labels = [_norm(item["object"]["class_name"]) for item in interactions]
    if any(_is_tool(label) for label in object_labels):
        return "Key action: tool handling"
    if any(_is_panel(label) for label in object_labels):
        return "Key action: device operation"
    if any(_is_container(label) for label in object_labels):
        return "Key action: sample/container handling"
    return "Key action: hand-object interaction"


def _draw_yolo_evidence_overlay(
    frame: np.ndarray,
    detections: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
    action_label: str,
) -> np.ndarray:
    if not detections and not action_label:
        return frame

    for det in detections:
        _draw_detection_box(frame, det, highlight=False)

    for interaction in interactions:
        hand = interaction["hand"]
        obj = interaction["object"]
        _draw_detection_box(frame, hand, highlight=True)
        _draw_detection_box(frame, obj, highlight=True)
        hx, hy = _bbox_center(hand["bbox"])
        ox, oy = _bbox_center(obj["bbox"])
        cv2.line(frame, (hx, hy), (ox, oy), (255, 40, 210), 3, cv2.LINE_AA)
        mx1 = min(hand["bbox"][0], obj["bbox"][0])
        my1 = min(hand["bbox"][1], obj["bbox"][1])
        mx2 = max(hand["bbox"][2], obj["bbox"][2])
        my2 = max(hand["bbox"][3], obj["bbox"][3])
        cv2.rectangle(frame, (mx1, my1), (mx2, my2), (255, 40, 210), 2)

    if action_label:
        _draw_banner(frame, action_label, len(detections), len(interactions))
    return frame


def _draw_detection_box(frame: np.ndarray, det: Dict[str, Any], *, highlight: bool) -> None:
    x1, y1, x2, y2 = det["bbox"]
    label = _label_text(det["class_name"], det["confidence"])
    color = (255, 40, 210) if highlight else _class_color(det["class_name"])
    thickness = 3 if highlight else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    ly1 = max(0, y1 - th - 8)
    cv2.rectangle(frame, (x1, ly1), (min(frame.shape[1] - 1, x1 + tw + 8), y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, max(th + 2, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_banner(frame: np.ndarray, action_label: str, detection_count: int, interaction_count: int) -> None:
    text = f"{action_label} | YOLO {detection_count} | contacts {interaction_count}"
    h, w = frame.shape[:2]
    y = 82
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
    cv2.rectangle(frame, (14, y), (min(w - 14, 34 + tw), y + th + 16), (16, 18, 24), -1)
    cv2.rectangle(frame, (14, y), (min(w - 14, 34 + tw), y + th + 16), (255, 40, 210), 2)
    cv2.putText(frame, text, (24, y + th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 2, cv2.LINE_AA)


def _label_text(label: str, confidence: float) -> str:
    value = _norm(label) or "object"
    return f"{value[:22]} {confidence:.2f}"


def _norm(value: str) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_hand(value: str) -> bool:
    label = _norm(value)
    return any(term in label for term in ("hand", "glove", "gloved_hand", "left_hand", "right_hand", "arm"))


def _is_container(value: str) -> bool:
    label = _norm(value)
    return any(term in label for term in ("bottle", "beaker", "vial", "tube", "cup", "jar", "container", "flask", "reagent", "sample"))


def _is_tool(value: str) -> bool:
    label = _norm(value)
    return any(term in label for term in ("pipette", "dropper", "spatula", "spearhead", "tool", "spoon"))


def _is_panel(value: str) -> bool:
    label = _norm(value)
    return any(term in label for term in ("panel", "button", "screen", "display", "balance", "scale", "device"))


def _is_interaction_object(value: str) -> bool:
    label = _norm(value)
    if not label or _is_hand(label):
        return False
    if label in {"lab_coat", "paper", "reagent_label", "label", "bottle_label"}:
        return False
    return _is_container(label) or _is_tool(label) or _is_panel(label) or True


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _bbox_edge_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return float((dx * dx + dy * dy) ** 0.5)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _class_color(class_name: str) -> Tuple[int, int, int]:
    label = _norm(class_name)
    if _is_hand(label):
        return (255, 190, 80)
    if _is_tool(label):
        return (70, 210, 255)
    if _is_container(label):
        return (90, 220, 150)
    if _is_panel(label):
        return (220, 170, 255)
    return (80, 190, 255)


def _draw_overlay(
    frame: np.ndarray,
    timestamp: float,
    duration: float,
    segments: List[Dict[str, Any]],
    total_segments: int,
    title_override: Optional[str] = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 72), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    current = _current_segment(timestamp, segments)
    title = title_override or f"{max(total_segments, 1)} experiment segment{'s' if total_segments != 1 else ''} detected"
    state = "Idle / transition"
    color = (170, 170, 170)
    if current is not None:
        state = _overlay_text(str(current.get("display_name") or f"Experiment {int(current['index']) + 1}"))
        color = _segment_color(int(current["index"]))

    cv2.putText(frame, _overlay_text(title), (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{state}  |  {format_timestamp(timestamp)} / {format_timestamp(duration)}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )

    track_y = h - 34
    track_x = 18
    track_w = w - 36
    cv2.rectangle(frame, (track_x, track_y), (track_x + track_w, track_y + 12), (55, 55, 55), -1)
    for segment in segments:
        x1 = track_x + int((segment["start_sec"] / max(duration, 1.0)) * track_w)
        x2 = track_x + int((segment["end_sec"] / max(duration, 1.0)) * track_w)
        cv2.rectangle(frame, (x1, track_y), (max(x2, x1 + 2), track_y + 12), _segment_color(int(segment["index"])), -1)
    cursor_x = track_x + int((timestamp / max(duration, 1.0)) * track_w)
    cv2.line(frame, (cursor_x, track_y - 8), (cursor_x, track_y + 20), (255, 255, 255), 2)
    return frame


def _current_segment(timestamp: float, segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for segment in segments:
        if segment["start_sec"] <= timestamp <= segment["end_sec"]:
            return segment
    return None


def _overlay_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.isascii():
        return text[:80]
    return "Experiment segment"


def _segment_color(index: int) -> tuple[int, int, int]:
    palette = [
        (80, 190, 255),
        (90, 220, 150),
        (220, 170, 255),
        (255, 190, 90),
        (120, 210, 220),
    ]
    return palette[index % len(palette)]


def format_timestamp(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _transcode_for_browser(raw_output: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        if raw_output != output:
            raw_output.replace(output)
        return
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_output),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        if result.returncode == 0 and output.exists():
            return
        logger.warning("ffmpeg segmentation preview transcode failed: %s", result.stderr[-600:] if result.stderr else result.returncode)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("ffmpeg segmentation preview transcode error: %s", exc)
    if raw_output != output:
        raw_output.replace(output)
