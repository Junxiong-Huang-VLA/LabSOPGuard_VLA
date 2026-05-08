from __future__ import annotations

import json
import os
import hashlib
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .experiment_focus import extract_experiment_focus_clips
from .schemas import read_jsonl, write_jsonl
from .vector_index import VectorIndex
from .yolo_detector import (
    HAND_LABELS,
    INTERACTION_OBJECT_LABELS,
    YoloActivityScorer,
    bbox_iou as yolo_bbox_iou,
    build_segments_from_yolo_frame_rows,
    canonical_yolo_label,
    detect_segments_from_yolo_rows,
    filter_implausible_detections,
    filter_implausible_hand_detections,
    find_hand_object_interactions,
    mock_yolo_frame_rows,
    normalize_yolo_frame_rows,
    scan_video_with_yolo,
    scan_yolo_video,
)


_CLASS_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (0, 180, 255),    # orange
    (80, 220, 80),    # green
    (255, 120, 60),   # blue
    (220, 80, 220),   # magenta
    (60, 220, 220),   # yellow
    (240, 160, 40),   # cyan-ish
    (120, 120, 255),  # red-ish
    (180, 220, 40),   # lime
    (255, 90, 180),   # purple
    (80, 170, 255),   # amber
]

_LABEL_SYNONYMS: dict[str, list[str]] = {
    "gloved_hand": ["gloved_hand", "手套", "戴手套", "手部", "实验人员手部"],
    "hand": ["hand", "手", "手部"],
    "balance": ["balance", "天平", "电子天平", "称量", "称重"],
    "sample_bottle_blue": ["sample_bottle_blue", "蓝色样品瓶", "蓝盖样品瓶", "样品瓶"],
    "sample_bottle": ["sample_bottle", "样品瓶", "样品容器", "样品"],
    "reagent_bottle": ["reagent_bottle", "试剂瓶", "试剂", "试剂容器"],
    "spatula": ["spatula", "刮勺", "药匙", "取样勺"],
    "paper": ["paper", "称量纸", "纸张", "记录纸"],
    "lab_coat": ["lab_coat", "实验服", "白大褂"],
    "pipette": ["pipette", "移液枪", "加样", "移液"],
    "pipette_tip": ["pipette_tip", "枪头", "移液枪头"],
}

DEFAULT_CLASS_THRESHOLDS: dict[str, float] = {
    "gloved_hand": 0.40,
    "hand": 0.40,
    "balance": 0.50,
    "reagent_bottle": 0.45,
    "sample_bottle": 0.45,
    "sample_bottle_blue": 0.45,
    "spatula": 0.45,
    "paper": 0.45,
    "lab_coat": 0.35,
    "pipette": 0.45,
    "pipette_tip": 0.45,
    "beaker": 0.45,
    "container": 0.45,
    "tube": 0.45,
    "tube_cap": 0.45,
}


def _color_for_label(label: str) -> tuple[int, int, int]:
    value = str(label or "unknown")
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(_CLASS_COLOR_PALETTE)
    return _CLASS_COLOR_PALETTE[index]


def _text_color_for_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    b, g, r = color
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (20, 20, 20) if luminance > 150 else (255, 255, 255)


def _synonyms_for_label(label: str) -> list[str]:
    normalized = str(label or "").strip()
    return _LABEL_SYNONYMS.get(normalized, [normalized] if normalized else [])


def parse_class_thresholds(value: str | dict[str, Any] | None = None) -> dict[str, float]:
    thresholds = dict(DEFAULT_CLASS_THRESHOLDS)
    raw = value if value is not None else os.environ.get("KEY_ACTION_YOLO_CLASS_THRESHOLDS")
    if not raw:
        return thresholds
    try:
        data = json.loads(raw) if isinstance(raw, str) and raw.strip().startswith("{") else raw
    except Exception:
        data = raw
    if isinstance(data, dict):
        for key, item in data.items():
            try:
                thresholds[str(key)] = float(item)
            except Exception:
                continue
        return thresholds
    if isinstance(raw, str):
        for item in raw.split(","):
            if "=" not in item:
                continue
            key, number = item.split("=", 1)
            try:
                thresholds[key.strip()] = float(number.strip())
            except Exception:
                continue
    return thresholds


def _effective_model_conf(conf: float, class_thresholds: dict[str, float] | None) -> float:
    values = [float(conf)]
    if class_thresholds:
        values.extend(float(value) for value in class_thresholds.values())
    return max(0.01, min(values))


def filter_detections_by_class_threshold(
    detections: list[dict[str, Any]],
    class_thresholds: dict[str, float] | None = None,
    default_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    thresholds = class_thresholds or {}
    filtered: list[dict[str, Any]] = []
    for detection in detections:
        label = str(detection.get("label") or "")
        threshold = float(thresholds.get(label, default_threshold))
        if float(detection.get("confidence", 0.0)) >= threshold:
            filtered.append(detection)
    return filtered


def filter_detections_by_allowed_labels(
    detections: list[dict[str, Any]],
    allowed_labels: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[dict[str, Any]]:
    if not allowed_labels:
        return detections
    allowed = {str(label).strip().lower() for label in allowed_labels if str(label).strip()}
    if not allowed:
        return detections
    return [
        detection
        for detection in detections
        if str(detection.get("label") or "").strip().lower() in allowed
    ]


def filter_implausible_hands_for_frame(
    detections: list[dict[str, Any]],
    *,
    frame_width: int | None = None,
    frame_height: int | None = None,
    frame: Any | None = None,
    source_view: str | None = None,
) -> list[dict[str, Any]]:
    kept, _ignored = filter_implausible_detections(
        detections,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    )
    return kept


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
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


def _bbox_center_distance_ratio(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    distance = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    aw, ah = max(1.0, ax2 - ax1), max(1.0, ay2 - ay1)
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    scale = max((aw**2 + ah**2) ** 0.5, (bw**2 + bh**2) ** 0.5, 1.0)
    return float(distance / scale)


class TemporalDetectionSmoother:
    def __init__(
        self,
        *,
        min_hits: int = 2,
        hold_frames: int = 10,
        iou_threshold: float = 0.15,
        bbox_alpha: float = 0.72,
        center_distance_threshold: float = 0.85,
    ) -> None:
        self.min_hits = max(1, int(min_hits))
        self.hold_frames = max(0, int(hold_frames))
        self.iou_threshold = float(iou_threshold)
        self.bbox_alpha = min(0.95, max(0.0, float(bbox_alpha)))
        self.center_distance_threshold = max(0.0, float(center_distance_threshold))
        self._tracks: list[dict[str, Any]] = []
        self._next_track_id = 1

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        new_track_ids: set[int] = set()
        for det_idx, detection in enumerate(detections):
            label = str(detection.get("label") or "")
            bbox = [float(value) for value in detection.get("bbox", [0, 0, 0, 0])]
            best_track_idx: int | None = None
            best_iou = 0.0
            best_distance_ratio = 999.0
            best_match_score = -1.0
            for track_idx, track in enumerate(self._tracks):
                if track_idx in matched_tracks or str(track.get("label")) != label:
                    continue
                iou = _bbox_iou(track["bbox"], bbox)
                distance_ratio = _bbox_center_distance_ratio(track["bbox"], bbox)
                proximity_score = max(0.0, 1.0 - min(distance_ratio, 2.0) / 2.0)
                match_score = iou * 0.7 + proximity_score * 0.3
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_iou = iou
                    best_distance_ratio = distance_ratio
                    best_track_idx = track_idx
            matched_by_iou = best_iou >= self.iou_threshold
            matched_by_center = best_distance_ratio <= self.center_distance_threshold
            if best_track_idx is None or not (matched_by_iou or matched_by_center):
                continue
            track = self._tracks[best_track_idx]
            adaptive_alpha = self.bbox_alpha
            if not matched_by_iou or best_distance_ratio > self.center_distance_threshold * 0.65:
                adaptive_alpha = min(adaptive_alpha, 0.55)
            track["bbox"] = [
                round(adaptive_alpha * float(old) + (1.0 - adaptive_alpha) * float(new), 3)
                for old, new in zip(track["bbox"], bbox)
            ]
            track["confidence"] = max(float(track.get("confidence", 0.0)) * 0.75, float(detection.get("confidence", 0.0)))
            track["hits"] = int(track.get("hits", 0)) + 1
            track["missed"] = 0
            track["class_id"] = detection.get("class_id")
            track["last_iou"] = round(best_iou, 6)
            track["last_center_distance_ratio"] = round(best_distance_ratio, 6)
            track["last_match_score"] = round(best_match_score, 6)
            matched_tracks.add(best_track_idx)
            matched_detections.add(det_idx)

        for det_idx, detection in enumerate(detections):
            if det_idx in matched_detections:
                continue
            track_id = self._next_track_id
            self._tracks.append(
                {
                    "track_id": track_id,
                    "label": str(detection.get("label") or ""),
                    "class_id": detection.get("class_id"),
                    "confidence": float(detection.get("confidence", 0.0)),
                    "bbox": [float(value) for value in detection.get("bbox", [0, 0, 0, 0])],
                    "hits": 1,
                    "missed": 0,
                }
            )
            new_track_ids.add(track_id)
            self._next_track_id += 1

        for track_idx, track in enumerate(self._tracks):
            if track_idx in matched_tracks or int(track.get("track_id", 0)) in new_track_ids:
                continue
            track["missed"] = int(track.get("missed", 0)) + 1

        live_tracks: list[dict[str, Any]] = []
        for track in self._tracks:
            if int(track.get("missed", 0)) <= self.hold_frames:
                live_tracks.append(track)
        self._tracks = live_tracks

        stable: list[dict[str, Any]] = []
        for track in self._tracks:
            if int(track.get("hits", 0)) < self.min_hits:
                continue
            stable.append(
                {
                    "label": track.get("label"),
                    "class_id": track.get("class_id"),
                    "confidence": round(float(track.get("confidence", 0.0)), 6),
                    "bbox": [round(float(value), 3) for value in track.get("bbox", [0, 0, 0, 0])],
                    "track_id": int(track.get("track_id", 0)),
                    "stable": True,
                    "missed_frames": int(track.get("missed", 0)),
                    "hits": int(track.get("hits", 0)),
                    "last_iou": float(track.get("last_iou", 0.0)),
                    "last_center_distance_ratio": float(track.get("last_center_distance_ratio", 0.0)),
                    "last_match_score": float(track.get("last_match_score", 0.0)),
                }
            )
        return stable


def resolve_default_yolo_model(explicit_path: str | Path | None = None, project_root: str | Path | None = None) -> Path | None:
    try:
        from .model_inventory import resolve_best_model_path

        resolved = resolve_best_model_path(explicit_path, project_root=project_root)
        if resolved is not None:
            return resolved
    except Exception:
        pass
    candidates: list[str | Path | None] = [
        explicit_path,
        os.environ.get("KEY_ACTION_YOLO_MODEL"),
        os.environ.get("LABSOPGUARD_YOLO_MODEL"),
    ]
    roots = []
    if project_root:
        roots.append(Path(project_root))
    cwd = Path.cwd()
    roots.extend([cwd, cwd / "LabSOPGuard", cwd.parent, cwd.parent / "LabSOPGuard"])
    relative_candidates = [
        "outputs/training/yolo26s_pose_lab_v4_focus_auto/weights/best.pt",
        "LabSOPGuard/outputs/training/yolo26s_pose_lab_v4_focus_auto/weights/best.pt",
        "outputs/training/yolo26s_pose_lab_v4_focus_auto_stage2/weights/best.pt",
        "LabSOPGuard/outputs/training/yolo26s_pose_lab_v4_focus_auto_stage2/weights/best.pt",
        "yolo26s.pt",
        "yolo26s-pose.pt",
        "yolo26n.pt",
        "yolov8n-pose.pt",
        "yolov8n.pt",
    ]
    candidates.extend(root / name for root in roots for name in relative_candidates)
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and path.is_dir():
            for nested in ("weights/best.pt", "best.pt", "weights/best.onnx", "best.onnx"):
                nested_path = path / nested
                if nested_path.exists() and nested_path.is_file():
                    return nested_path.resolve()
        if not path.is_absolute() and not path.exists() and project_root:
            path = Path(project_root) / path
            if path.exists() and path.is_dir():
                for nested in ("weights/best.pt", "best.pt", "weights/best.onnx", "best.onnx"):
                    nested_path = path / nested
                    if nested_path.exists() and nested_path.is_file():
                        return nested_path.resolve()
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _detect_image(
    model: Any,
    image_path: Path,
    conf: float,
    iou: float,
    device: str,
    class_thresholds: dict[str, float] | None = None,
    allowed_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    source_view: str | None = None,
) -> list[dict[str, Any]]:
    results = model.predict(source=str(image_path), conf=_effective_model_conf(conf, class_thresholds), iou=iou, device=device, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id))
            confidence = float(box.conf[0])
            bbox = [float(value) for value in box.xyxy[0].detach().cpu().numpy().tolist()]
            detections.append(
                {
                    "label": label,
                    "class_id": cls_id,
                    "confidence": round(confidence, 6),
                    "bbox": [round(value, 3) for value in bbox],
                }
            )
    filtered = filter_detections_by_class_threshold(detections, class_thresholds, default_threshold=conf)
    allowed = filter_detections_by_allowed_labels(filtered, allowed_labels)
    frame_width: int | None = None
    frame_height: int | None = None
    image: Any | None = None
    try:
        import cv2

        image = cv2.imread(str(image_path))
        if image is not None:
            frame_height, frame_width = int(image.shape[0]), int(image.shape[1])
    except Exception:
        pass
    return filter_implausible_hands_for_frame(
        allowed,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=image,
        source_view=source_view,
    )


def _detect_frame(
    model: Any,
    frame: Any,
    conf: float,
    iou: float,
    device: str,
    class_thresholds: dict[str, float] | None = None,
    allowed_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    source_view: str | None = None,
) -> list[dict[str, Any]]:
    results = model.predict(source=frame, conf=_effective_model_conf(conf, class_thresholds), iou=iou, device=device, verbose=False)
    detections: list[dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            label = str(names.get(cls_id, cls_id))
            confidence = float(box.conf[0])
            bbox = [float(value) for value in box.xyxy[0].detach().cpu().numpy().tolist()]
            detections.append(
                {
                    "label": label,
                    "class_id": cls_id,
                    "confidence": round(confidence, 6),
                    "bbox": [round(value, 3) for value in bbox],
                }
            )
    filtered = filter_detections_by_class_threshold(detections, class_thresholds, default_threshold=conf)
    allowed = filter_detections_by_allowed_labels(filtered, allowed_labels)
    frame_height: int | None = None
    frame_width: int | None = None
    try:
        frame_height, frame_width = int(frame.shape[0]), int(frame.shape[1])
    except Exception:
        pass
    return filter_implausible_hands_for_frame(
        allowed,
        frame_width=frame_width,
        frame_height=frame_height,
        frame=frame,
        source_view=source_view,
    )


def _draw_annotated(image_path: Path, detections: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import cv2
    except Exception:
        return
    image = cv2.imread(str(image_path))
    if image is None:
        return
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.get("bbox", [0, 0, 0, 0])]
        color = _color_for_label(str(detection.get("label")))
        text_color = _text_color_for_bgr(color)
        label = f"{detection.get('label')} {float(detection.get('confidence', 0.0)):.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label_width = max(90, len(label) * 9)
        cv2.rectangle(image, (x1, max(0, y1 - 24)), (min(image.shape[1] - 1, x1 + label_width), y1), color, -1)
        cv2.putText(image, label, (x1 + 4, max(17, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def _draw_detections_on_frame(frame: Any, detections: list[dict[str, Any]]) -> Any:
    import cv2

    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.get("bbox", [0, 0, 0, 0])]
        color = _color_for_label(str(detection.get("label")))
        text_color = _text_color_for_bgr(color)
        label = f"{detection.get('label')} {float(detection.get('confidence', 0.0)):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (min(frame.shape[1] - 1, x1 + max(90, len(label) * 9)), y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return frame


def _find_ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _transcode_to_browser_h264(path: Path) -> bool:
    ffmpeg_exe = _find_ffmpeg_exe()
    if not ffmpeg_exe or not path.exists() or path.stat().st_size <= 0:
        return False
    tmp = path.with_name(f"{path.stem}.browser_tmp_{os.getpid()}{path.suffix}")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp),
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
            shutil.move(str(tmp), str(path))
            return True
    finally:
        tmp.unlink(missing_ok=True)
    return False


def annotate_clip_with_yolo(
    input_clip_path: str | Path,
    output_clip_path: str | Path,
    model: Any,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    detect_fps: float = 5.0,
    class_thresholds: dict[str, float] | None = None,
    allowed_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    source_view: str | None = None,
    smoothing_min_hits: int = 2,
    smoothing_hold_frames: int = 10,
    smoothing_iou: float = 0.15,
    smoothing_bbox_alpha: float = 0.72,
    smoothing_center_distance: float = 0.85,
) -> dict[str, Any]:
    import cv2

    source = Path(input_clip_path)
    target = Path(output_clip_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip for YOLO annotation: {source}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid clip dimensions: {source}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(target), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create annotated clip: {target}")
        frame_idx = 0
        detect_every = 1 if detect_fps <= 0 or detect_fps >= fps else max(1, int(round(fps / detect_fps)))
        last_detections: list[dict[str, Any]] = []
        smoother = TemporalDetectionSmoother(
            min_hits=smoothing_min_hits,
            hold_frames=smoothing_hold_frames,
            iou_threshold=smoothing_iou,
            bbox_alpha=smoothing_bbox_alpha,
            center_distance_threshold=smoothing_center_distance,
        )
        label_counts: Counter[str] = Counter()
        raw_label_counts: Counter[str] = Counter()
        detection_frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % detect_every == 0:
                raw_detections = _detect_frame(
                    model,
                    frame,
                    conf=conf,
                    iou=iou,
                    device=device,
                    class_thresholds=class_thresholds,
                    allowed_labels=allowed_labels,
                    source_view=source_view,
                )
                detection_frames += 1
                for detection in raw_detections:
                    raw_label_counts[str(detection.get("label"))] += 1
                last_detections = smoother.update(raw_detections)
            elif detect_every > 1:
                last_detections = smoother.update([])
            for detection in last_detections:
                label_counts[str(detection.get("label"))] += 1
            annotated = _draw_detections_on_frame(frame, last_detections)
            writer.write(annotated)
            frame_idx += 1
        writer.release()
    finally:
        cap.release()
    browser_h264 = _transcode_to_browser_h264(target)
    return {
        "input_clip_path": str(source),
        "annotated_clip_path": str(target),
        "browser_h264": browser_h264,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_idx,
        "source_frame_count": total_frames,
        "detection_frames": detection_frames,
        "label_counts": dict(label_counts),
        "raw_label_counts": dict(raw_label_counts),
        "detections": int(sum(label_counts.values())),
        "raw_detections": int(sum(raw_label_counts.values())),
        "class_thresholds": class_thresholds or {},
        "allowed_labels": list(allowed_labels or []),
        "temporal_smoothing": {
            "min_hits": int(smoothing_min_hits),
            "hold_frames": int(smoothing_hold_frames),
            "iou_threshold": float(smoothing_iou),
            "bbox_alpha": float(smoothing_bbox_alpha),
            "center_distance_threshold": float(smoothing_center_distance),
        },
    }


def run_yolo_on_keyframes(
    session_dir: str | Path,
    model_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    preferred_view: str = "first_person",
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    class_thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    session_root = Path(session_dir).resolve()
    resolved_model = resolve_default_yolo_model(model_path, project_root=project_root)
    thresholds = parse_class_thresholds(class_thresholds)
    output_path = session_root / "metadata" / "yolo_detections.jsonl"
    summary_path = session_root / "metadata" / "yolo_summary.json"
    if resolved_model is None:
        summary = {"available": False, "error": "No YOLO weights found", "detections": 0, "model_path": None}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(output_path, [])
        return summary

    try:
        from ultralytics import YOLO

        model = YOLO(str(resolved_model))
    except Exception as exc:
        summary = {"available": False, "error": str(exc), "detections": 0, "model_path": str(resolved_model)}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(output_path, [])
        return summary

    segments = read_jsonl(session_root / "metadata" / "key_action_segments.jsonl") if (session_root / "metadata" / "key_action_segments.jsonl").exists() else []
    rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    segment_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for segment in segments:
        segment_id = str(segment.get("segment_id"))
        keyframe_dir = session_root / "keyframes" / segment_id
        view = preferred_view
        frame_paths = [keyframe_dir / f"{view}_{phase}.jpg" for phase in ("start", "middle", "end")]
        if not any(path.exists() for path in frame_paths):
            view = "third_person"
            frame_paths = [keyframe_dir / f"{view}_{phase}.jpg" for phase in ("start", "middle", "end")]
        for phase, image_path in zip(("start", "middle", "end"), frame_paths):
            if not image_path.exists():
                continue
            detections = _detect_image(
                model,
                image_path,
                conf=conf,
                iou=iou,
                device=device,
                class_thresholds=thresholds,
                source_view=view,
            )
            annotated_path = session_root / "debug" / "yolo_annotated" / f"{segment_id}_{view}_{phase}.jpg"
            _draw_annotated(image_path, detections, annotated_path)
            for detection in detections:
                label = str(detection.get("label"))
                label_counts[label] += 1
                segment_counts[segment_id][label] += 1
            rows.append(
                {
                    "segment_id": segment_id,
                    "view": view,
                    "phase": phase,
                    "image_path": str(image_path),
                    "annotated_image_path": str(annotated_path),
                    "model_path": str(resolved_model),
                    "detections": detections,
                }
            )
    write_jsonl(output_path, rows)
    summary = {
        "available": True,
        "model_path": str(resolved_model),
        "preferred_view": preferred_view,
        "frame_count": len(rows),
        "detections": int(sum(label_counts.values())),
        "label_counts": dict(label_counts),
        "segment_label_counts": {key: dict(value) for key, value in segment_counts.items()},
        "class_thresholds": thresholds,
        "output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_yolo_on_segment_clips(
    session_dir: str | Path,
    model_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    preferred_view: str = "first_person",
    views: list[str] | tuple[str, ...] | None = None,
    model_paths_by_view: dict[str, str | Path | None] | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    detect_fps: float = 5.0,
    class_thresholds: dict[str, float] | None = None,
    allowed_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    smoothing_min_hits: int = 2,
    smoothing_hold_frames: int = 10,
    smoothing_iou: float = 0.15,
    smoothing_bbox_alpha: float = 0.72,
    smoothing_center_distance: float = 0.85,
) -> dict[str, Any]:
    session_root = Path(session_dir).resolve()
    thresholds = parse_class_thresholds(class_thresholds)
    output_path = session_root / "metadata" / "yolo_annotated_clips.jsonl"
    summary_path = session_root / "metadata" / "yolo_clip_summary.json"

    requested_views = [str(view) for view in (views or [preferred_view]) if str(view)]
    if not requested_views:
        requested_views = [preferred_view]

    def _resolve_view_model_path(view: str) -> Path | None:
        view_model_path = None
        if model_paths_by_view:
            view_model_path = model_paths_by_view.get(view)
        return resolve_default_yolo_model(view_model_path or model_path, project_root=project_root)

    resolved_by_view = {view: _resolve_view_model_path(view) for view in requested_views}
    if not any(resolved_by_view.values()):
        summary = {"available": False, "error": "No YOLO weights found", "clips": 0, "detections": 0, "model_path": None}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(output_path, [])
        return summary

    model_cache: dict[str, Any] = {}
    errors: list[str] = []

    def _model_for_view(view: str) -> tuple[Path | None, Any | None]:
        resolved = resolved_by_view.get(view)
        if resolved is None:
            errors.append(f"{view}: no YOLO weights found")
            return None, None
        key = str(resolved)
        if key in model_cache:
            return resolved, model_cache[key]
        try:
            from ultralytics import YOLO

            model_cache[key] = YOLO(key)
            return resolved, model_cache[key]
        except Exception as exc:
            errors.append(f"{view}: {exc}")
            return resolved, None

    # Preserve the previous single-view fallback behavior: when the requested
    # preferred view is missing on an older single-view run, annotate the
    # available third-person clip with the same model. Explicit multiview calls
    # skip missing views instead of silently changing the requested view.
    legacy_single_view = views is None and not model_paths_by_view
    try:
        segments = read_jsonl(session_root / "metadata" / "key_action_segments.jsonl") if (session_root / "metadata" / "key_action_segments.jsonl").exists() else []
        rows: list[dict[str, Any]] = []
        total_labels: Counter[str] = Counter()
        label_counts_by_view: dict[str, Counter[str]] = defaultdict(Counter)
        clip_counts_by_view: Counter[str] = Counter()
        for segment in segments:
            segment_id = str(segment.get("segment_id"))
            for requested_view in requested_views:
                view = requested_view
                view_ref = segment.get(requested_view)
                if not view_ref and legacy_single_view:
                    view_ref = segment.get("third_person")
                    view = "third_person"
                if not view_ref:
                    continue
                clip_path = Path(str((view_ref or {}).get("clip_path", "")))
                if not clip_path.is_absolute():
                    clip_path = Path.cwd() / clip_path
                    if not clip_path.exists():
                        clip_path = session_root / str((view_ref or {}).get("clip_path", ""))
                if not clip_path.exists():
                    errors.append(f"{segment_id}/{view}: clip not found: {clip_path}")
                    continue
                resolved_model, model = _model_for_view(requested_view)
                if model is None:
                    continue
                output_clip = session_root / "clips" / segment_id / f"{view}_yolo_annotated.mp4"
                clip_result = annotate_clip_with_yolo(
                    clip_path,
                    output_clip,
                    model,
                    conf=conf,
                    iou=iou,
                    device=device,
                    detect_fps=detect_fps,
                    class_thresholds=thresholds,
                    allowed_labels=allowed_labels,
                    source_view=view,
                    smoothing_min_hits=smoothing_min_hits,
                    smoothing_hold_frames=smoothing_hold_frames,
                    smoothing_iou=smoothing_iou,
                    smoothing_bbox_alpha=smoothing_bbox_alpha,
                    smoothing_center_distance=smoothing_center_distance,
                )
                total_labels.update(clip_result.get("label_counts", {}))
                label_counts_by_view[view].update(clip_result.get("label_counts", {}))
                clip_counts_by_view[view] += 1
                rows.append(
                    {
                        "segment_id": segment_id,
                        "view": view,
                        "requested_view": requested_view,
                        "model_path": str(resolved_model) if resolved_model else None,
                        **clip_result,
                    }
                )
    except Exception as exc:
        summary = {
            "available": False,
            "error": str(exc),
            "clips": 0,
            "detections": 0,
            "model_path": str(next((path for path in resolved_by_view.values() if path), "")) or None,
            "model_paths_by_view": {view: str(path) if path else None for view, path in resolved_by_view.items()},
            "views": requested_views,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(output_path, [])
        return summary
    write_jsonl(output_path, rows)
    _attach_annotated_clip_refs(session_root, rows)
    summary = {
        "available": True,
        "model_path": str(resolved_by_view.get(preferred_view)) if resolved_by_view.get(preferred_view) else None,
        "model_paths_by_view": {view: str(path) if path else None for view, path in resolved_by_view.items()},
        "preferred_view": preferred_view,
        "views": requested_views,
        "clips": len(rows),
        "clips_by_view": dict(clip_counts_by_view),
        "detections": int(sum(total_labels.values())),
        "label_counts": dict(total_labels),
        "label_counts_by_view": {key: dict(value) for key, value in label_counts_by_view.items()},
        "class_thresholds": thresholds,
        "allowed_labels": list(allowed_labels or []),
        "temporal_smoothing": {
            "min_hits": int(smoothing_min_hits),
            "hold_frames": int(smoothing_hold_frames),
            "iou_threshold": float(smoothing_iou),
            "bbox_alpha": float(smoothing_bbox_alpha),
            "center_distance_threshold": float(smoothing_center_distance),
        },
        "errors": errors,
        "output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_yolo_on_experiment_focus_clips(
    session_dir: str | Path,
    model_path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
    preferred_view: str = "first_person",
    views: list[str] | tuple[str, ...] | None = None,
    model_paths_by_view: dict[str, str | Path | None] | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cpu",
    detect_fps: float = 5.0,
    class_thresholds: dict[str, float] | None = None,
    allowed_labels: list[str] | tuple[str, ...] | set[str] | None = None,
    source_view: str | None = None,
    smoothing_min_hits: int = 2,
    smoothing_hold_frames: int = 10,
    smoothing_iou: float = 0.15,
    smoothing_bbox_alpha: float = 0.72,
    smoothing_center_distance: float = 0.85,
) -> dict[str, Any]:
    session_root = Path(session_dir).resolve()
    thresholds = parse_class_thresholds(class_thresholds)
    output_path = session_root / "metadata" / "yolo_experiment_focus_clips.jsonl"
    summary_path = session_root / "metadata" / "yolo_experiment_focus_summary.json"

    requested_views = [str(view) for view in (views or [preferred_view]) if str(view)]
    if not requested_views:
        requested_views = [preferred_view]

    def _resolve_view_model_path(view: str) -> Path | None:
        view_model_path = None
        if model_paths_by_view:
            view_model_path = model_paths_by_view.get(view)
        return resolve_default_yolo_model(view_model_path or model_path, project_root=project_root)

    resolved_by_view = {view: _resolve_view_model_path(view) for view in requested_views}
    if not any(resolved_by_view.values()):
        summary = {"available": False, "error": "No YOLO weights found", "clips": 0, "detections": 0, "model_path": None}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(output_path, [])
        return summary

    try:
        focus_summary = extract_experiment_focus_clips(session_root, dry_run=False)
    except Exception as exc:
        summary = {
            "available": False,
            "error": str(exc),
            "clips": 0,
            "detections": 0,
            "model_paths_by_view": {view: str(path) if path else None for view, path in resolved_by_view.items()},
            "views": requested_views,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_jsonl(output_path, [])
        return summary

    model_cache: dict[str, Any] = {}
    errors: list[str] = []

    def _model_for_view(view: str) -> tuple[Path | None, Any | None]:
        resolved = resolved_by_view.get(view)
        if resolved is None:
            errors.append(f"{view}: no YOLO weights found")
            return None, None
        key = str(resolved)
        if key in model_cache:
            return resolved, model_cache[key]
        try:
            from ultralytics import YOLO

            model_cache[key] = YOLO(key)
            return resolved, model_cache[key]
        except Exception as exc:
            errors.append(f"{view}: {exc}")
            return resolved, None

    rows: list[dict[str, Any]] = []
    total_labels: Counter[str] = Counter()
    label_counts_by_view: dict[str, Counter[str]] = defaultdict(Counter)
    clip_counts_by_view: Counter[str] = Counter()
    clips_by_view = focus_summary.get("clips_by_view") if isinstance(focus_summary.get("clips_by_view"), dict) else {}
    for requested_view in requested_views:
        view_ref = clips_by_view.get(requested_view)
        if not isinstance(view_ref, dict):
            continue
        clip_path = Path(str(view_ref.get("clip_path") or ""))
        if not clip_path.exists():
            errors.append(f"{requested_view}: focus clip not found: {clip_path}")
            continue
        resolved_model, model = _model_for_view(requested_view)
        if model is None:
            continue
        output_clip = session_root / "clips" / "experiment_focus" / f"{requested_view}_yolo_annotated.mp4"
        clip_result = annotate_clip_with_yolo(
            clip_path,
            output_clip,
            model,
            conf=conf,
            iou=iou,
            device=device,
            detect_fps=detect_fps,
            class_thresholds=thresholds,
            allowed_labels=allowed_labels,
            source_view=requested_view,
            smoothing_min_hits=smoothing_min_hits,
            smoothing_hold_frames=smoothing_hold_frames,
            smoothing_iou=smoothing_iou,
            smoothing_bbox_alpha=smoothing_bbox_alpha,
            smoothing_center_distance=smoothing_center_distance,
        )
        total_labels.update(clip_result.get("label_counts", {}))
        label_counts_by_view[requested_view].update(clip_result.get("label_counts", {}))
        clip_counts_by_view[requested_view] += 1
        rows.append(
            {
                "segment_id": "experiment_focus",
                "view": requested_view,
                "requested_view": requested_view,
                "model_path": str(resolved_model) if resolved_model else None,
                "time_start_sec": view_ref.get("time_start_sec"),
                "time_end_sec": view_ref.get("time_end_sec"),
                "global_start_time": view_ref.get("global_start_time"),
                "global_end_time": view_ref.get("global_end_time"),
                **clip_result,
            }
        )

    write_jsonl(output_path, rows)
    summary = {
        "available": bool(rows),
        "model_path": str(resolved_by_view.get(preferred_view)) if resolved_by_view.get(preferred_view) else None,
        "model_paths_by_view": {view: str(path) if path else None for view, path in resolved_by_view.items()},
        "preferred_view": preferred_view,
        "views": requested_views,
        "clips": len(rows),
        "clips_by_view": dict(clip_counts_by_view),
        "detections": int(sum(total_labels.values())),
        "label_counts": dict(total_labels),
        "label_counts_by_view": {key: dict(value) for key, value in label_counts_by_view.items()},
        "class_thresholds": thresholds,
        "allowed_labels": list(allowed_labels or []),
        "focus": focus_summary,
        "errors": errors,
        "output_path": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _attach_annotated_clip_refs(session_root: Path, rows: list[dict[str, Any]]) -> None:
    segments_path = session_root / "metadata" / "key_action_segments.jsonl"
    if not rows or not segments_path.exists():
        return
    segments = read_jsonl(segments_path)
    if not segments:
        return
    by_segment_view: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        segment_id = str(row.get("segment_id") or "")
        view = str(row.get("view") or row.get("requested_view") or "")
        annotated_path = row.get("annotated_clip_path")
        if not segment_id or not view or not annotated_path:
            continue
        by_segment_view[(segment_id, view)] = row

    changed = False
    stale_markers = (
        "annotated clips are aligned preview copies",
        "rerun regenerates YOLO overlays",
        "标注视频只是对齐预览",
    )
    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        for view in ("first_person", "third_person"):
            row = by_segment_view.get((segment_id, view))
            view_ref = segment.get(view)
            if not row or not isinstance(view_ref, dict):
                continue
            view_ref["annotated_clip_path"] = row.get("annotated_clip_path")
            view_ref["yolo_label_counts"] = row.get("label_counts") or {}
            view_ref["yolo_detection_count"] = int(row.get("detections") or 0)
            if row.get("model_path"):
                view_ref["yolo_model_path"] = row.get("model_path")
            changed = True

        evidence = segment.get("evidence")
        if isinstance(evidence, dict):
            limitations = evidence.get("limitations")
            if isinstance(limitations, list):
                filtered = [
                    item
                    for item in limitations
                    if not any(marker in str(item) for marker in stale_markers)
                ]
                if filtered != limitations:
                    evidence["limitations"] = filtered
                    changed = True

    if changed:
        write_jsonl(segments_path, segments)


def _aggregate_yolo_counts(session_root: Path) -> dict[str, Counter[str]]:
    counts_by_segment: dict[str, Counter[str]] = defaultdict(Counter)
    clip_rows = read_jsonl(session_root / "metadata" / "yolo_annotated_clips.jsonl") if (session_root / "metadata" / "yolo_annotated_clips.jsonl").exists() else []
    for row in clip_rows:
        segment_id = str(row.get("segment_id") or "")
        if not segment_id:
            continue
        counts_by_segment[segment_id].update({str(k): int(v) for k, v in (row.get("label_counts") or {}).items()})
    if counts_by_segment:
        return counts_by_segment

    keyframe_rows = read_jsonl(session_root / "metadata" / "yolo_detections.jsonl") if (session_root / "metadata" / "yolo_detections.jsonl").exists() else []
    for row in keyframe_rows:
        segment_id = str(row.get("segment_id") or "")
        if not segment_id:
            continue
        for detection in row.get("detections") or []:
            label = str(detection.get("label") or "")
            if label:
                counts_by_segment[segment_id][label] += 1
    return counts_by_segment


def _visual_keywords(label_counts: Counter[str]) -> list[str]:
    keywords: list[str] = []
    seen: set[str] = set()
    for label, _count in label_counts.most_common():
        for synonym in _synonyms_for_label(label):
            if synonym and synonym not in seen:
                keywords.append(synonym)
                seen.add(synonym)
    return keywords


def _action_type_from_labels(label_counts: Counter[str], current: str) -> str:
    if current and current != "unknown_operation":
        return current
    labels = set(label_counts)
    if {"pipette", "pipette_tip"} & labels:
        return "pipetting"
    if "balance" in labels or "spatula" in labels:
        return "weighing"
    return current or "unknown_operation"


def _append_yolo_index_text(index_text: str, label_counts: Counter[str]) -> str:
    marker = "\nYOLO视觉索引:\n"
    base = str(index_text or "").split(marker, 1)[0].rstrip()
    keywords = _visual_keywords(label_counts)
    counts_text = ", ".join(f"{label}={count}" for label, count in label_counts.most_common()) or "none"
    keyword_text = " ".join(keywords) if keywords else "none"
    return (
        f"{base}{marker}"
        f"YOLO标签计数: {counts_text}\n"
        f"YOLO可检索关键词: {keyword_text}\n"
        f"视觉动作线索: {keyword_text}\n"
    )


def _replace_index_line(index_text: str, prefix: str, value: str) -> str:
    lines = str(index_text or "").splitlines()
    replaced = False
    for idx, line in enumerate(lines):
        if line.startswith(prefix):
            lines[idx] = f"{prefix} {value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{prefix} {value}")
    return "\n".join(lines) + "\n"


def _tools_objects_from_labels(label_counts: Counter[str]) -> tuple[list[str], list[str]]:
    tool_labels = {"balance", "pipette", "pipette_tip", "spatula"}
    object_labels = {"sample_bottle_blue", "sample_bottle", "reagent_bottle", "paper", "lab_coat", "gloved_hand", "hand"}
    tools = [label for label in label_counts if label in tool_labels]
    objects = [label for label in label_counts if label in object_labels]
    return tools, objects


def enrich_key_action_index_with_yolo(session_dir: str | Path) -> dict[str, Any]:
    session_root = Path(session_dir).resolve()
    metadata_dir = session_root / "metadata"
    key_segments_path = metadata_dir / "key_action_segments.jsonl"
    vector_metadata_path = metadata_dir / "vector_metadata.jsonl"
    if not key_segments_path.exists() or not vector_metadata_path.exists():
        return {"available": False, "error": "key_action_segments.jsonl or vector_metadata.jsonl is missing"}

    counts_by_segment = _aggregate_yolo_counts(session_root)
    segments = read_jsonl(key_segments_path)
    vector_metadata = read_jsonl(vector_metadata_path)
    metadata_by_segment = {str(item.get("segment_id")): item for item in vector_metadata}
    enriched_count = 0

    for segment in segments:
        segment_id = str(segment.get("segment_id") or "")
        label_counts = counts_by_segment.get(segment_id, Counter())
        if not label_counts:
            continue
        enriched_count += 1
        labels = [label for label, _count in label_counts.most_common()]
        keywords = _visual_keywords(label_counts)
        tools, objects = _tools_objects_from_labels(label_counts)

        text_description = segment.setdefault("text_description", {})
        current_action = str(text_description.get("action_type") or "unknown_operation")
        action_type = _action_type_from_labels(label_counts, current_action)
        text_description["action_type"] = action_type
        existing_tools = list(text_description.get("tools") or [])
        existing_objects = list(text_description.get("objects") or [])
        text_description["tools"] = sorted(set(existing_tools + tools))
        text_description["objects"] = sorted(set(existing_objects + objects))
        visual_summary = "YOLO识别到: " + ", ".join(f"{label}({count})" for label, count in label_counts.most_common())
        summary = str(text_description.get("summary") or "").split(" YOLO识别到:", 1)[0].rstrip()
        text_description["summary"] = f"{summary} {visual_summary}".strip()
        segment["yolo_labels"] = labels
        segment["yolo_label_counts"] = dict(label_counts)
        segment["visual_keywords"] = keywords
        index_info = segment.setdefault("index", {})
        index_text = str(index_info.get("index_text") or "")
        index_text = _replace_index_line(index_text, "动作类型:", action_type)
        index_text = _replace_index_line(index_text, "动作摘要:", str(text_description.get("summary") or ""))
        index_text = _replace_index_line(index_text, "工具:", ", ".join(text_description["tools"]) if text_description["tools"] else "unknown")
        index_text = _replace_index_line(index_text, "对象:", ", ".join(text_description["objects"]) if text_description["objects"] else "unknown")
        index_info["index_text"] = _append_yolo_index_text(index_text, label_counts)

        vector_item = metadata_by_segment.get(segment_id)
        if vector_item is not None:
            vector_item["index_text"] = index_info["index_text"]
            vector_item["action_type"] = action_type
            vector_item["yolo_labels"] = labels
            vector_item["yolo_label_counts"] = dict(label_counts)
            vector_item["visual_keywords"] = keywords

    write_jsonl(key_segments_path, segments)
    write_jsonl(vector_metadata_path, vector_metadata)
    index_dir = session_root / "index"
    index = VectorIndex()
    index.build([str(item.get("index_text") or "") for item in vector_metadata], vector_metadata)
    index.save(index_dir)
    write_jsonl(index_dir / "docstore.jsonl", vector_metadata)

    summary = {
        "available": True,
        "enriched_segments": enriched_count,
        "segment_count": len(segments),
        "labels_by_segment": {segment_id: dict(counts) for segment_id, counts in counts_by_segment.items()},
        "vector_metadata_path": str(vector_metadata_path),
        "index_dir": str(index_dir),
    }
    (metadata_dir / "yolo_index_enrichment.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
