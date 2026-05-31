from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "model" / "detection_runtime.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "2593eaf50893655ac677c1b1c95d1afc_first15s_progressive_boxes.mp4"
DEFAULT_SOURCE = PROJECT_ROOT / "outputs" / "2593eaf50893655ac677c1b1c95d1afc.mp4"
DEFAULT_FALLBACK_WEIGHTS = (
    PROJECT_ROOT / "yolo26s.pt",
    PROJECT_ROOT / "yolo26n.pt",
    PROJECT_ROOT / "yolo26s-pose.pt",
)


@dataclass
class Detection:
    box: np.ndarray
    class_id: int
    class_name: str
    confidence: float


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    latest_box: np.ndarray
    smoothed_box: np.ndarray
    render_box: np.ndarray
    class_scores: Dict[int, float]
    class_names: Dict[int, str]
    seed_box: Optional[np.ndarray]
    animation_target_box: Optional[np.ndarray]
    feature_point: Optional[Tuple[int, int]]
    confidence: float
    first_seen_frame: int
    last_seen_frame: int
    animation_start_frame: Optional[int]
    animation_played: bool
    missing_count: int
    hits: int
    confirmed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 15-second progressive detection box demo.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Input video path.")
    parser.add_argument("--weights", default=None, help="YOLO weights path. If omitted, resolve from project config.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output mp4 path.")
    parser.add_argument("--duration", type=float, default=15.0, help="Only render the first N seconds.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for tracking association.")
    parser.add_argument("--anim-frames", type=int, default=8, help="Frames used for seed-to-final box animation.")
    parser.add_argument("--seed-scale", type=float, default=0.22, help="Seed box size relative to final box.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--ema", type=float, default=0.18, help="EMA update factor for box smoothing.")
    parser.add_argument("--center-smooth", type=float, default=0.14, help="Display smoothing factor for box center.")
    parser.add_argument("--size-smooth", type=float, default=0.1, help="Display smoothing factor for box width and height.")
    parser.add_argument("--min-hits", type=int, default=4, help="Frames required before a track is considered stable.")
    parser.add_argument("--max-missing", type=int, default=12, help="Frames to keep unmatched tracks alive.")
    parser.add_argument("--hold-missing", type=int, default=3, help="Keep rendering a confirmed track for this many missed frames.")
    parser.add_argument("--reanimate-missing", type=int, default=5, help="Replay animation after this many missed frames.")
    return parser.parse_args()


def _load_runtime_config() -> Dict[str, object]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def resolve_weights(override: Optional[str]) -> Path:
    candidates: List[Path] = []
    runtime_cfg = _load_runtime_config()
    config_model = runtime_cfg.get("model")
    env_override = os.environ.get("LABSOPGUARD_YOLO_MODEL")
    for candidate in (override, env_override, config_model):
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        candidates.append(path)
    candidates.extend(DEFAULT_FALLBACK_WEIGHTS)

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("No YOLO weights could be resolved from config or project fallbacks.")


def clamp_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1 = float(np.clip(x1, 0, max(0, width - 2)))
    y1 = float(np.clip(y1, 0, max(0, height - 2)))
    x2 = float(np.clip(x2, x1 + 1, max(1, width - 1)))
    y2 = float(np.clip(y2, y1 + 1, max(1, height - 1)))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter_area
    return float(inter_area / denom) if denom > 0 else 0.0


def box_to_center_size(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return np.array([center_x, center_y, width, height], dtype=np.float32)


def center_size_to_box(values: Sequence[float]) -> np.ndarray:
    center_x, center_y, width, height = values
    half_w = max(0.5, width / 2.0)
    half_h = max(0.5, height / 2.0)
    return np.array(
        [center_x - half_w, center_y - half_h, center_x + half_w, center_y + half_h],
        dtype=np.float32,
    )


def smooth_box_components(
    previous_box: Sequence[float],
    target_box: Sequence[float],
    center_alpha: float,
    size_alpha: float,
) -> np.ndarray:
    prev_state = box_to_center_size(previous_box)
    target_state = box_to_center_size(target_box)
    center_distance = float(np.linalg.norm(prev_state[:2] - target_state[:2]))
    size_delta = float(np.linalg.norm(prev_state[2:] - target_state[2:]))
    motion_scale = max(24.0, (prev_state[2] + prev_state[3] + target_state[2] + target_state[3]) * 0.2)
    motion_ratio = float(np.clip(center_distance / motion_scale, 0.0, 1.0))
    size_ratio = float(np.clip(size_delta / motion_scale, 0.0, 1.0))
    adaptive_center_alpha = float(np.clip(center_alpha + motion_ratio * 0.18, center_alpha, 0.36))
    adaptive_size_alpha = float(np.clip(size_alpha + size_ratio * 0.12, size_alpha, 0.26))
    prev_state[:2] = prev_state[:2] * (1.0 - adaptive_center_alpha) + target_state[:2] * adaptive_center_alpha
    prev_state[2:] = prev_state[2:] * (1.0 - adaptive_size_alpha) + target_state[2:] * adaptive_size_alpha
    return center_size_to_box(prev_state)


def normalized_center_affinity(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    state_a = box_to_center_size(box_a)
    state_b = box_to_center_size(box_b)
    center_distance = float(np.linalg.norm(state_a[:2] - state_b[:2]))
    normalizer = max(20.0, (state_a[2] + state_b[2] + state_a[3] + state_b[3]) * 0.25)
    return float(np.clip(1.0 - center_distance / normalizer, 0.0, 1.0))


def ease_out_cubic(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return 1.0 - (1.0 - progress) ** 3


def ease_out_quint(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return 1.0 - (1.0 - progress) ** 5


def lerp_box(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start * (1.0 - alpha) + end * alpha


def remap_progress(progress: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0
    return float(np.clip((progress - start) / (end - start), 0.0, 1.0))


def class_color(class_name: str) -> Tuple[int, int, int]:
    palette_rgb = [
        (0, 191, 255),
        (76, 175, 80),
        (255, 167, 38),
        (244, 81, 30),
        (126, 87, 194),
        (38, 198, 218),
        (171, 71, 188),
        (255, 112, 67),
    ]
    digest = hashlib.md5(class_name.lower().encode("utf-8")).hexdigest()
    rgb = palette_rgb[int(digest[:2], 16) % len(palette_rgb)]
    return (rgb[2], rgb[1], rgb[0])


def blend_rect(
    frame: np.ndarray,
    box: Sequence[float],
    color: Tuple[int, int, int],
    thickness: int,
    alpha: float,
    radius_pad: int = 0,
) -> None:
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, x1 - radius_pad)
    y1 = max(0, y1 - radius_pad)
    x2 = min(frame_w - 1, x2 + radius_pad)
    y2 = min(frame_h - 1, y2 + radius_pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def expand_box(box: Sequence[float], pad: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad], dtype=np.float32)


def draw_corner_guides(
    frame: np.ndarray,
    box: Sequence[float],
    color: Tuple[int, int, int],
    alpha: float,
    thickness: int = 1,
) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    guide = max(8, int(round(min(w, h) * 0.14)))
    overlay = frame.copy()
    segments = (
        ((x1, y1), (x1 + guide, y1)),
        ((x1, y1), (x1, y1 + guide)),
        ((x2, y1), (x2 - guide, y1)),
        ((x2, y1), (x2, y1 + guide)),
        ((x1, y2), (x1 + guide, y2)),
        ((x1, y2), (x1, y2 - guide)),
        ((x2, y2), (x2 - guide, y2)),
        ((x2, y2), (x2, y2 - guide)),
    )
    for start, end in segments:
        cv2.line(overlay, start, end, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def draw_label(frame: np.ndarray, box: Sequence[float], label: str, color: Tuple[int, int, int], alpha: float) -> None:
    x1, y1, _, _ = [int(round(v)) for v in box]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad_x = 7
    pad_y = 5
    rect_x1 = max(4, x1)
    rect_y1 = max(4, y1 - text_h - baseline - 14)
    rect_x2 = rect_x1 + text_w + pad_x * 2
    rect_y2 = rect_y1 + text_h + baseline + pad_y * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)
    cv2.putText(
        frame,
        label,
        (rect_x1 + pad_x, rect_y2 - baseline - pad_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_feature_hint(frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int], alpha: float) -> None:
    overlay = frame.copy()
    px, py = point
    cv2.circle(overlay, (px, py), 2, color, -1, cv2.LINE_AA)
    cv2.circle(overlay, (px, py), 6, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def build_seed_box(
    frame: np.ndarray,
    final_box: Sequence[float],
    seed_scale: float,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in final_box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    crop = frame[y1:y2, x1:x2]
    feature_point = ((x1 + x2) // 2, (y1 + y2) // 2)

    if crop.size > 0:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=8,
            qualityLevel=0.04,
            minDistance=max(6, int(min(gray.shape[:2]) * 0.12)),
            blockSize=5,
        )
        if corners is not None and len(corners) > 0:
            corner = corners[0][0]
            feature_point = (x1 + int(round(corner[0])), y1 + int(round(corner[1])))
        else:
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(grad_x, grad_y)
            _, _, _, max_loc = cv2.minMaxLoc(magnitude)
            if max_loc is not None:
                feature_point = (x1 + int(max_loc[0]), y1 + int(max_loc[1]))

    box_w = max(10.0, (x2 - x1) * float(np.clip(seed_scale, 0.18, 0.28)))
    box_h = max(10.0, (y2 - y1) * float(np.clip(seed_scale, 0.18, 0.28)))
    seed_x1 = feature_point[0] - box_w / 2.0
    seed_y1 = feature_point[1] - box_h / 2.0
    seed_x2 = feature_point[0] + box_w / 2.0
    seed_y2 = feature_point[1] + box_h / 2.0
    seed_box = clamp_box((seed_x1, seed_y1, seed_x2, seed_y2), width, height)
    return seed_box, feature_point


def detect_frame(model, frame: np.ndarray, conf: float, iou: float, imgsz: int) -> List[Detection]:
    try:
        results = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    except TypeError:
        results = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)
    detections: List[Detection] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls.item())
            xyxy = np.array(box.xyxy[0].tolist(), dtype=np.float32)
            detections.append(
                Detection(
                    box=xyxy,
                    class_id=class_id,
                    class_name=str(result.names.get(class_id, class_id)),
                    confidence=float(box.conf.item()),
                )
            )
    return detections


def update_track_label(track: Track, detection: Detection, decay: float = 0.97) -> None:
    for class_id in list(track.class_scores):
        track.class_scores[class_id] *= decay
        if track.class_scores[class_id] < 0.05:
            track.class_scores.pop(class_id, None)
            track.class_names.pop(class_id, None)
    track.class_scores[detection.class_id] = track.class_scores.get(detection.class_id, 0.0) + detection.confidence
    track.class_names[detection.class_id] = detection.class_name
    stable_class_id = max(track.class_scores, key=track.class_scores.get)
    current_score = track.class_scores.get(track.class_id, 0.0)
    stable_score = track.class_scores.get(stable_class_id, 0.0)
    if (
        stable_class_id == track.class_id
        or current_score <= 0.0
        or stable_score > current_score * 1.35
        or stable_score - current_score > 0.55
    ):
        track.class_id = stable_class_id
        track.class_name = track.class_names.get(stable_class_id, track.class_name)


def match_detections(
    tracks: Dict[int, Track],
    detections: Sequence[Detection],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    candidates: List[Tuple[float, int, int]] = []
    track_ids = list(tracks.keys())
    for det_idx, det in enumerate(detections):
        for track_id in track_ids:
            track = tracks[track_id]
            reference_box = track.render_box if track.confirmed else track.smoothed_box
            iou_score = compute_iou(reference_box, det.box)
            center_affinity = normalized_center_affinity(reference_box, det.box)
            class_bonus = 0.04 if track.class_id == det.class_id else 0.0
            score = iou_score * 0.84 + center_affinity * 0.16 + class_bonus
            if iou_score >= max(0.28, iou_threshold * 0.7) or (iou_score >= 0.18 and center_affinity >= 0.78):
                candidates.append((score, track_id, det_idx))
    candidates.sort(reverse=True)

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    matches: List[Tuple[int, int]] = []
    for _, track_id, det_idx in candidates:
        if track_id in matched_tracks or det_idx in matched_dets:
            continue
        matched_tracks.add(track_id)
        matched_dets.add(det_idx)
        matches.append((track_id, det_idx))

    unmatched_tracks = [track_id for track_id in track_ids if track_id not in matched_tracks]
    unmatched_dets = [det_idx for det_idx in range(len(detections)) if det_idx not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def trigger_animation(track: Track, frame: np.ndarray, frame_idx: int, seed_scale: float) -> None:
    track.animation_target_box = track.render_box.copy()
    seed_box, feature_point = build_seed_box(frame, track.animation_target_box, seed_scale)
    track.seed_box = seed_box
    track.feature_point = feature_point
    track.animation_start_frame = frame_idx
    track.animation_played = True


def update_tracks(
    tracks: Dict[int, Track],
    detections: Sequence[Detection],
    frame: np.ndarray,
    frame_idx: int,
    next_track_id: int,
    iou_threshold: float,
    ema_alpha: float,
    render_center_alpha: float,
    render_size_alpha: float,
    min_hits: int,
    max_missing: int,
    reanimate_missing: int,
    seed_scale: float,
) -> int:
    matches, unmatched_tracks, unmatched_dets = match_detections(tracks, detections, iou_threshold)

    for track_id, det_idx in matches:
        track = tracks[track_id]
        detection = detections[det_idx]
        missed_before_match = track.missing_count
        track.latest_box = detection.box.copy()
        track.smoothed_box = smooth_box_components(
            track.smoothed_box,
            detection.box,
            center_alpha=ema_alpha,
            size_alpha=max(0.05, ema_alpha * 0.82),
        )
        track.render_box = smooth_box_components(
            track.render_box,
            track.smoothed_box,
            center_alpha=render_center_alpha,
            size_alpha=render_size_alpha,
        )
        track.confidence = track.confidence * 0.75 + detection.confidence * 0.25
        track.last_seen_frame = frame_idx
        track.missing_count = 0
        track.hits += 1
        update_track_label(track, detection)

        if not track.confirmed and track.hits >= min_hits:
            track.confirmed = True
            trigger_animation(track, frame, frame_idx, seed_scale)
        elif track.confirmed and missed_before_match >= reanimate_missing:
            trigger_animation(track, frame, frame_idx, seed_scale)

    for track_id in unmatched_tracks:
        track = tracks[track_id]
        track.missing_count += 1
        track.render_box = smooth_box_components(
            track.render_box,
            track.smoothed_box,
            center_alpha=render_center_alpha * 0.55,
            size_alpha=render_size_alpha * 0.45,
        )

    for det_idx in unmatched_dets:
        detection = detections[det_idx]
        tracks[next_track_id] = Track(
            track_id=next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            latest_box=detection.box.copy(),
            smoothed_box=detection.box.copy(),
            render_box=detection.box.copy(),
            class_scores={detection.class_id: detection.confidence},
            class_names={detection.class_id: detection.class_name},
            seed_box=None,
            animation_target_box=None,
            feature_point=None,
            confidence=detection.confidence,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
            animation_start_frame=None,
            animation_played=False,
            missing_count=0,
            hits=1,
            confirmed=False,
        )
        next_track_id += 1

    stale_track_ids = [track_id for track_id, track in tracks.items() if track.missing_count > max_missing]
    for track_id in stale_track_ids:
        tracks.pop(track_id, None)
    return next_track_id


def render_track(frame: np.ndarray, track: Track, frame_idx: int, anim_frames: int, hold_missing: int) -> None:
    if not track.confirmed or track.missing_count > hold_missing:
        return

    final_box = track.render_box
    current_box = final_box
    color = class_color(track.class_name)
    fade_ratio = 1.0 - min(track.missing_count, hold_missing) / max(1, hold_missing + 1)
    label_alpha = 0.76 * fade_ratio
    label = track.class_name

    if track.animation_start_frame is not None:
        raw_progress = (frame_idx - track.animation_start_frame) / max(1, anim_frames)
        seed_hold = min(0.18, 1.5 / max(3, anim_frames))

        if track.seed_box is not None and raw_progress < 1.0:
            if raw_progress < seed_hold:
                settle = ease_out_cubic(remap_progress(raw_progress, 0.0, seed_hold))
                current_box = lerp_box(track.seed_box, expand_box(track.seed_box, 1.5), 0.18 + settle * 0.12)
                blend_rect(frame, current_box, color, thickness=1, alpha=0.42)
                draw_corner_guides(frame, current_box, color, alpha=0.24 + settle * 0.08, thickness=1)
                if track.feature_point is not None:
                    draw_feature_hint(frame, track.feature_point, color, alpha=0.2 + settle * 0.08)
                label_alpha = 0.62
            else:
                expand_progress = remap_progress(raw_progress, seed_hold, 1.0)
                eased = ease_out_quint(expand_progress)
                animation_target = track.animation_target_box if track.animation_target_box is not None else final_box
                current_box = lerp_box(track.seed_box, animation_target, eased)
                halo_box = lerp_box(track.seed_box, animation_target, min(1.0, eased + 0.08))
                blend_rect(frame, animation_target, color, thickness=1, alpha=0.05)
                blend_rect(frame, halo_box, color, thickness=1, alpha=0.08, radius_pad=int(round((1.0 - eased) * 2)))
                blend_rect(frame, current_box, color, thickness=2, alpha=0.7)
                draw_corner_guides(frame, current_box, color, alpha=0.14 + (1.0 - eased) * 0.16, thickness=1)
                if track.feature_point is not None:
                    draw_feature_hint(frame, track.feature_point, color, alpha=0.1 + (1.0 - eased) * 0.1)
                if eased < 0.42:
                    label_alpha = 0.68
        elif 1.0 <= raw_progress < 1.18:
            lock_alpha = 1.0 - remap_progress(raw_progress, 1.0, 1.18)
            lock_box = track.animation_target_box if track.animation_target_box is not None else final_box
            blend_rect(frame, expand_box(lock_box, 1.5 + lock_alpha * 2.0), color, thickness=1, alpha=0.04 + lock_alpha * 0.05)
            draw_corner_guides(frame, expand_box(lock_box, 1.0), color, alpha=0.06 + lock_alpha * 0.06, thickness=1)
        elif raw_progress >= 1.18:
            track.seed_box = None
            track.animation_target_box = None
            track.feature_point = None
            track.animation_start_frame = None

    blend_rect(frame, current_box, color, thickness=3, alpha=0.9 * fade_ratio)
    draw_label(frame, current_box, label, color, label_alpha)


def render_demo(args: argparse.Namespace) -> Path:
    source_path = Path(args.source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Video not found: {source_path}")

    weights_path = resolve_weights(args.weights)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    local_cfg = PROJECT_ROOT / ".ultralytics"
    local_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cfg.resolve()))

    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frames = min(total_frames if total_frames > 0 else int(round(args.duration * fps)), int(round(args.duration * fps)))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
        True,
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    tracks: Dict[int, Track] = {}
    next_track_id = 1
    frame_idx = 0

    while frame_idx < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_frame(model, frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz)
        next_track_id = update_tracks(
            tracks=tracks,
            detections=detections,
            frame=frame,
            frame_idx=frame_idx,
            next_track_id=next_track_id,
            iou_threshold=args.iou,
            ema_alpha=float(np.clip(args.ema, 0.05, 0.95)),
            render_center_alpha=float(np.clip(args.center_smooth, 0.05, 0.6)),
            render_size_alpha=float(np.clip(args.size_smooth, 0.03, 0.5)),
            min_hits=max(1, args.min_hits),
            max_missing=max(1, args.max_missing),
            reanimate_missing=max(1, args.reanimate_missing),
            seed_scale=float(np.clip(args.seed_scale, 0.18, 0.28)),
        )

        annotated = frame.copy()
        for track_id in sorted(tracks):
            render_track(annotated, tracks[track_id], frame_idx, max(1, args.anim_frames), max(0, args.hold_missing))

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    if frame_idx == 0:
        raise RuntimeError("No frames were rendered.")
    return output_path


def main() -> int:
    args = parse_args()
    output_path = render_demo(args)
    print(f"output_video={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
