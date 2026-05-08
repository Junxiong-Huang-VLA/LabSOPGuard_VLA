from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .schemas import (
    CVDetectionSummary,
    ClipReference,
    DetectedSegment,
    InteractionEvent,
    InteractionKeyframe,
    KeyActionSegment,
    SegmentIndexInfo,
    SessionManifest,
    TextDescription,
    VideoSource,
    YoloInteraction,
)
from .time_alignment import global_time_to_local_sec, local_sec_to_global_time


_HAND_LABELS = {
    "hand",
    "hands",
    "gloved_hand",
    "glove",
    "gloved hand",
    "person_hand",
    "手",
    "手部",
    "戴手套",
    "手套",
}
_OBJECT_LABEL_ALIASES = {
    "sample_bottle_blue": "瓶子",
    "sample_bottle": "瓶子",
    "bottle": "瓶子",
    "blue_bottle": "瓶子",
    "reagent_bottle": "试剂瓶",
    "reagent bottle": "试剂瓶",
    "balance": "天平",
    "scale": "天平",
    "weighing_scale": "天平",
    "spatula": "刮勺",
    "scoopula": "刮勺",
    "药匙": "刮勺",
    "pipette": "移液枪",
    "pipette_tip": "移液枪枪头",
    "tube": "试管",
    "test_tube": "试管",
    "beaker": "烧杯",
    "cup": "杯子",
}
_PHASE_FRACTIONS = {"start": 0.0, "middle": 0.5, "mid": 0.5, "end": 1.0}


def extract_clip_ffmpeg(
    input_video_path: str | Path,
    local_start_sec: float,
    local_end_sec: float,
    output_clip_path: str | Path,
    dry_run: bool = False,
) -> Path:
    output_path = Path(output_clip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        output_path.write_bytes(
            f"DRY RUN CLIP {input_video_path} {local_start_sec:.3f}-{local_end_sec:.3f}\n".encode("utf-8")
        )
        return output_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required for real clip extraction. Install ffmpeg, add it to PATH, "
            "or install imageio-ffmpeg. Re-run with --dry-run to validate metadata only."
        )
    if local_end_sec <= local_start_sec:
        raise ValueError(f"Invalid clip range: {local_start_sec} >= {local_end_sec}")

    copy_cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{local_start_sec:.3f}",
        "-to",
        f"{local_end_sec:.3f}",
        "-i",
        str(input_video_path),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    reencode_cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{local_start_sec:.3f}",
        "-to",
        f"{local_end_sec:.3f}",
        "-i",
        str(input_video_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(output_path),
    ]
    result = subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg clip extraction failed: {result.stderr[-1200:]}")
    return output_path


def extract_keyframes(
    input_video_path: str | Path,
    local_start_sec: float,
    local_end_sec: float,
    output_dir: str | Path,
    prefix: str,
    dry_run: bool = False,
) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    times = [
        ("start", local_start_sec),
        ("middle", (local_start_sec + local_end_sec) / 2.0),
        ("end", max(local_start_sec, local_end_sec - 0.05)),
    ]
    paths = [target_dir / f"{prefix}_{name}.jpg" for name, _ in times]
    if dry_run:
        for path in paths:
            path.write_bytes(b"DRY RUN KEYFRAME\n")
        return paths

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for keyframe extraction") from exc

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for keyframes: {input_video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration_sec = (frame_count / fps) if fps > 0 and frame_count > 0 else None
        for (name, time_sec), path in zip(times, paths):
            seek_sec = max(0.0, time_sec)
            if duration_sec is not None:
                seek_sec = min(seek_sec, max(0.0, duration_sec - 0.08))
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000.0)
            ok, frame = cap.read()
            if not ok and frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count) - 1))
                ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Cannot read {name} keyframe at {time_sec:.3f}s from {input_video_path}")
            cv2.imwrite(str(path), frame)
    finally:
        cap.release()
    return paths


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalized_label(label: Any) -> str:
    return str(label or "").strip().lower().replace("-", "_")


def _detection_label(detection: dict[str, Any]) -> str:
    return str(
        detection.get("label")
        or detection.get("class_name")
        or detection.get("name")
        or detection.get("category")
        or detection.get("class")
        or ""
    ).strip()


def _detection_confidence(detection: dict[str, Any]) -> float:
    return float(
        _numeric(detection.get("confidence"))
        or _numeric(detection.get("conf"))
        or _numeric(detection.get("score"))
        or 1.0
    )


def _detection_bbox(detection: dict[str, Any]) -> list[float] | None:
    raw = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    if raw is None and {"x1", "y1", "x2", "y2"} <= set(detection):
        raw = [detection.get("x1"), detection.get("y1"), detection.get("x2"), detection.get("y2")]
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    values = [_numeric(item) for item in raw[:4]]
    if any(item is None for item in values):
        return None
    x1, y1, x2, y2 = [float(item) for item in values]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _bbox_touch_score(box_a: list[float], box_b: list[float]) -> tuple[bool, float, float]:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = 0.0 if union <= 0 else float(intersection / union)
    x_gap = max(0.0, max(bx1 - ax2, ax1 - bx2))
    y_gap = max(0.0, max(by1 - ay2, ay1 - by2))
    gap = (x_gap**2 + y_gap**2) ** 0.5
    max_coord = max(abs(value) for value in [*box_a, *box_b])
    span = max(ax2 - ax1, ay2 - ay1, bx2 - bx1, by2 - by1, 1.0)
    threshold = 0.05 if max_coord <= 1.5 else max(18.0, min(80.0, span * 0.08))
    if iou > 0:
        return True, 1.0, gap
    if gap <= threshold:
        return True, max(0.1, 1.0 - gap / max(threshold * 2.0, 1e-6)), gap
    return False, 0.0, gap


def _is_hand_label(label: str) -> bool:
    normalized = _normalized_label(label)
    return normalized in _HAND_LABELS or "hand" in normalized or "手" in str(label)


def _object_name(label: str) -> str | None:
    normalized = _normalized_label(label)
    if _is_hand_label(label):
        return None
    if normalized in _OBJECT_LABEL_ALIASES:
        return _OBJECT_LABEL_ALIASES[normalized]
    if "bottle" in normalized or "瓶" in str(label):
        return "瓶子"
    if "balance" in normalized or "scale" in normalized or "天平" in str(label):
        return "天平"
    if "spatula" in normalized or "scoop" in normalized or "刮勺" in str(label) or "药匙" in str(label):
        return "刮勺"
    return None


def _interaction_text(object_name: str) -> str:
    return f"手与{object_name}交互"


def _parse_global_time(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _row_global_time(row: dict[str, Any]) -> str | None:
    value = row.get("global_time") or row.get("global_timestamp") or row.get("timestamp_global")
    return str(value) if value else None


def _row_view(row: dict[str, Any]) -> str:
    raw = str(row.get("source_view") or row.get("view") or row.get("camera") or row.get("stream") or row.get("video") or "third_person").strip().lower()
    raw = raw.replace("-", "_").replace(" ", "_")
    if raw in {"first", "first_person", "egocentric", "fpv", "bottom", "bottom_view", "operator", "head", "wrist"}:
        return "first_person"
    if raw in {"third", "third_person", "top", "top_view", "overview", "external", "scene"}:
        return "third_person"
    return "third_person"


def _row_source(row: dict[str, Any]) -> str:
    return str(row.get("source") or row.get("source_file") or "yolo_frame_rows")


def _row_source_image_path(row: dict[str, Any]) -> str | None:
    value = row.get("image_path") or row.get("frame_path") or row.get("source_image_path")
    return str(value) if value else None


def _row_segment_matches(row: dict[str, Any], segment: DetectedSegment) -> bool:
    row_segment_id = row.get("segment_id") or row.get("segment")
    return not row_segment_id or str(row_segment_id) == segment.segment_id


def _time_from_phase(row: dict[str, Any], ref: ClipReference) -> float | None:
    phase = str(row.get("phase") or "").strip().lower()
    if phase not in _PHASE_FRACTIONS:
        return None
    if phase == "end":
        return max(ref.local_start_sec, ref.local_end_sec - 0.05)
    return ref.local_start_sec + (ref.local_end_sec - ref.local_start_sec) * _PHASE_FRACTIONS[phase]


def _row_local_time_sec(row: dict[str, Any], segment: DetectedSegment, ref: ClipReference) -> float | None:
    for key in ("clip_time_sec", "segment_time_sec", "segment_offset_sec", "offset_in_segment_sec"):
        value = _numeric(row.get(key))
        if value is not None:
            return ref.local_start_sec + value
    phase_time = _time_from_phase(row, ref)
    if phase_time is not None:
        return phase_time
    for key in ("local_time_sec", "source_local_time_sec", "video_time_sec", "time_sec", "timestamp_sec", "sec", "t"):
        value = _numeric(row.get(key))
        if value is not None:
            if _row_segment_matches(row, segment) and 0.0 <= value <= segment.duration_sec + 0.25:
                if value < ref.local_start_sec - 0.25 or value > ref.local_end_sec + 0.25:
                    return ref.local_start_sec + value
            return value
    frame_index = _numeric(row.get("frame_index"))
    fps = _numeric(row.get("fps"))
    if frame_index is not None and fps and fps > 0:
        return frame_index / fps
    if _row_segment_matches(row, segment):
        return (ref.local_start_sec + ref.local_end_sec) / 2.0
    return None


def _row_inside_segment(row: dict[str, Any], segment: DetectedSegment, ref: ClipReference) -> bool:
    if not _row_segment_matches(row, segment):
        return False
    row_global = _parse_global_time(_row_global_time(row))
    start_global = _parse_global_time(segment.global_start_time)
    end_global = _parse_global_time(segment.global_end_time)
    if row_global and start_global and end_global:
        return start_global <= row_global <= end_global
    local_time = _row_local_time_sec(row, segment, ref)
    if local_time is None:
        return _row_segment_matches(row, segment)
    return ref.local_start_sec - 0.25 <= local_time <= ref.local_end_sec + 0.25


def _normalized_detection(detection: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(detection)
    normalized["label"] = _detection_label(detection)
    normalized["confidence"] = round(_detection_confidence(detection), 6)
    bbox = _detection_bbox(detection)
    if bbox is not None:
        normalized["bbox"] = [round(value, 3) for value in bbox]
    return normalized


def _explicit_interaction_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for key in ("interactions", "yolo_interactions", "hand_object_interactions", "interaction_events"):
        value = row.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    if any(key in row for key in ("interaction", "object_label", "target_label", "hand_label")):
        candidates.append(row)
    return candidates


def _interaction_from_candidate(
    row: dict[str, Any],
    candidate: dict[str, Any],
    *,
    view: str,
    local_time_sec: float,
    global_time: str | None,
) -> YoloInteraction | None:
    object_label = str(
        candidate.get("object_label")
        or candidate.get("target_label")
        or candidate.get("object")
        or candidate.get("label")
        or ""
    ).strip()
    object_name = str(candidate.get("object_name") or "") or (_object_name(object_label) if object_label else None)
    if not object_label or not object_name:
        return None
    hand_label = str(candidate.get("hand_label") or candidate.get("hand") or "hand")
    confidence = float(_numeric(candidate.get("confidence")) or _numeric(candidate.get("score")) or _numeric(row.get("confidence")) or 1.0)
    interaction = str(candidate.get("interaction") or candidate.get("description") or _interaction_text(object_name))
    detections = candidate.get("detections") or row.get("detections") or []
    normalized_detections = []
    hand_bbox = candidate.get("hand_bbox")
    object_bbox = candidate.get("object_bbox")
    if isinstance(hand_bbox, list) and len(hand_bbox) >= 4:
        normalized_detections.append(
            {
                "label": hand_label,
                "confidence": round(max(0.0, min(confidence, 1.0)), 6),
                "bbox": [round(float(value), 3) for value in hand_bbox[:4]],
            }
        )
    if isinstance(object_bbox, list) and len(object_bbox) >= 4:
        normalized_detections.append(
            {
                "label": object_label,
                "confidence": round(max(0.0, min(confidence, 1.0)), 6),
                "bbox": [round(float(value), 3) for value in object_bbox[:4]],
            }
        )
    normalized_detections.extend(_normalized_detection(item) for item in detections if isinstance(item, dict))
    return YoloInteraction(
        view=view,
        local_time_sec=float(local_time_sec),
        global_time=global_time,
        interaction=interaction,
        hand_label=hand_label,
        object_label=object_label,
        object_name=object_name,
        confidence=round(max(0.0, min(confidence, 1.0)), 6),
        source=_row_source(row),
        source_image_path=_row_source_image_path(row),
        detections=normalized_detections,
    )


def _interactions_from_detections(
    row: dict[str, Any],
    *,
    view: str,
    local_time_sec: float,
    global_time: str | None,
) -> list[YoloInteraction]:
    raw_detections = row.get("detections") or row.get("objects") or row.get("boxes") or []
    if not isinstance(raw_detections, list):
        return []
    detections = [_normalized_detection(item) for item in raw_detections if isinstance(item, dict)]
    hands = [item for item in detections if _is_hand_label(str(item.get("label") or ""))]
    objects = [item for item in detections if _object_name(str(item.get("label") or ""))]
    interactions: list[YoloInteraction] = []
    for hand in hands:
        hand_box = _detection_bbox(hand)
        if hand_box is None:
            continue
        for obj in objects:
            object_label = str(obj.get("label") or "")
            object_name = _object_name(object_label)
            object_box = _detection_bbox(obj)
            if object_name is None or object_box is None:
                continue
            touching, proximity, _gap = _bbox_touch_score(hand_box, object_box)
            if not touching:
                continue
            confidence = ((_detection_confidence(hand) + _detection_confidence(obj)) / 2.0) * proximity
            interactions.append(
                YoloInteraction(
                    view=view,
                    local_time_sec=float(local_time_sec),
                    global_time=global_time,
                    interaction=_interaction_text(object_name),
                    hand_label=str(hand.get("label") or "hand"),
                    object_label=object_label,
                    object_name=object_name,
                    confidence=round(max(0.0, min(confidence, 1.0)), 6),
                    source=_row_source(row),
                    source_image_path=_row_source_image_path(row),
                    detections=[hand, obj],
                )
            )
    return interactions


def collect_yolo_interactions_for_segment(
    segment: DetectedSegment,
    yolo_frame_rows: list[dict[str, Any]] | None,
    view_refs: dict[str, ClipReference | None],
    view_sources: dict[str, VideoSource],
) -> list[YoloInteraction]:
    if not yolo_frame_rows:
        return []
    interactions: list[YoloInteraction] = []
    for row in yolo_frame_rows:
        if not isinstance(row, dict):
            continue
        view = _row_view(row)
        ref = view_refs.get(view) or view_refs.get("third_person")
        source = view_sources.get(view) or view_sources.get("third_person")
        if ref is None or source is None or not _row_inside_segment(row, segment, ref):
            continue
        local_time_sec = _row_local_time_sec(row, segment, ref)
        if local_time_sec is None:
            continue
        global_time = _row_global_time(row) or local_sec_to_global_time(source, local_time_sec).isoformat()
        candidates = _explicit_interaction_candidates(row)
        if candidates:
            for candidate in candidates:
                interaction = _interaction_from_candidate(
                    row,
                    candidate,
                    view=view,
                    local_time_sec=local_time_sec,
                    global_time=global_time,
                )
                if interaction is not None:
                    interactions.append(interaction)
            continue
        interactions.extend(
            _interactions_from_detections(
                row,
                view=view,
                local_time_sec=local_time_sec,
                global_time=global_time,
            )
        )
    interactions.sort(key=lambda item: (item.local_time_sec, -item.confidence, item.object_label))
    return interactions


def _interaction_hand_object_detections(interaction: YoloInteraction) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    hand_det = None
    object_det = None
    for detection in interaction.detections:
        label = str(detection.get("label") or "")
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        if hand_det is None and _is_hand_label(label):
            hand_det = detection
        if object_det is None and label == interaction.object_label:
            object_det = detection
    return hand_det, object_det


def _interaction_has_hand_object_boxes(interaction: YoloInteraction) -> bool:
    hand_det, object_det = _interaction_hand_object_detections(interaction)
    return hand_det is not None and object_det is not None


def _select_interaction_events(
    segment_id: str,
    interactions: list[YoloInteraction],
    *,
    max_keyframes: int = 5,
    min_gap_sec: float = 0.75,
) -> list[InteractionEvent]:
    selected: list[YoloInteraction] = []
    evidence_interactions = [item for item in interactions if _interaction_has_hand_object_boxes(item)]
    ranked_interactions = evidence_interactions or interactions
    for interaction in sorted(ranked_interactions, key=lambda item: (-item.confidence, item.local_time_sec)):
        duplicate = any(
            existing.view == interaction.view
            and existing.object_label == interaction.object_label
            and abs(existing.local_time_sec - interaction.local_time_sec) < min_gap_sec
            for existing in selected
        )
        if duplicate:
            continue
        selected.append(interaction)
        if len(selected) >= max_keyframes:
            break
    selected.sort(key=lambda item: (item.local_time_sec, item.object_label))
    events: list[InteractionEvent] = []
    for index, interaction in enumerate(selected, start=1):
        events.append(
            InteractionEvent(
                event_id=f"{segment_id}_interaction_{index:03d}",
                view=interaction.view,
                local_time_sec=float(interaction.local_time_sec),
                global_time=interaction.global_time,
                interaction=interaction.interaction,
                hand_label=interaction.hand_label,
                object_label=interaction.object_label,
                object_name=interaction.object_name,
                confidence=interaction.confidence,
                source=interaction.source,
            )
        )
    return events


def _write_keyframe_at_time(
    input_video_path: str | Path,
    local_time_sec: float,
    output_path: str | Path,
    *,
    dry_run: bool = False,
    source_image_path: str | Path | None = None,
    interaction: YoloInteraction | None = None,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        target.write_bytes(f"DRY RUN INTERACTION KEYFRAME {local_time_sec:.3f}\n".encode("utf-8"))
        return target
    has_evidence_boxes = interaction is not None and _interaction_has_hand_object_boxes(interaction)
    if source_image_path and Path(source_image_path).exists() and not has_evidence_boxes:
        shutil.copy2(source_image_path, target)
        return target

    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for interaction keyframe extraction") from exc

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video for interaction keyframe: {input_video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration_sec = (frame_count / fps) if fps > 0 and frame_count > 0 else None
        seek_sec = max(0.0, local_time_sec)
        if duration_sec is not None:
            seek_sec = min(seek_sec, max(0.0, duration_sec - 0.08))
        cap.set(cv2.CAP_PROP_POS_MSEC, seek_sec * 1000.0)
        ok, frame = cap.read()
        if not ok and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_count) - 1))
            ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read interaction keyframe at {local_time_sec:.3f}s from {input_video_path}")
        if interaction is not None:
            frame = _draw_interaction_evidence_boxes(frame, interaction)
        cv2.imwrite(str(target), frame)
    finally:
        cap.release()
    return target


def _draw_interaction_evidence_boxes(frame: Any, interaction: YoloInteraction) -> Any:
    import cv2

    hand_det, object_det = _interaction_hand_object_detections(interaction)
    if hand_det is None or object_det is None:
        return frame

    def _draw_box(detection: dict[str, Any], label: str, color: tuple[int, int, int]) -> None:
        bbox = detection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            return
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
        y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            return
        confidence = detection.get("confidence")
        label_text = f"{label} {float(confidence):.2f}" if confidence is not None else label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_top = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, text_top), (min(width - 1, x1 + text_w + 8), y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 4, max(text_h + 1, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    _draw_box(object_det, interaction.object_label, (0, 128, 255))
    _draw_box(hand_det, interaction.hand_label or "hand", (0, 220, 0))
    footer = (
        f"physical evidence: {interaction.hand_label or 'hand'} -> {interaction.object_label}"
        f" | score={interaction.confidence:.2f} | t={interaction.local_time_sec:.2f}s"
    )
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1] - 1, 800), 30), (20, 20, 20), -1)
    cv2.putText(frame, footer, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return frame


def extract_interaction_keyframes(
    segment: DetectedSegment,
    segment_keyframes_dir: str | Path,
    yolo_frame_rows: list[dict[str, Any]] | None,
    view_refs: dict[str, ClipReference | None],
    view_sources: dict[str, VideoSource],
    *,
    dry_run: bool = False,
) -> tuple[list[InteractionKeyframe], list[InteractionEvent], list[YoloInteraction]]:
    interactions = collect_yolo_interactions_for_segment(segment, yolo_frame_rows, view_refs, view_sources)
    if not interactions:
        return [], [], []
    events = _select_interaction_events(segment.segment_id, interactions)
    keyframes: list[InteractionKeyframe] = []
    output_dir = Path(segment_keyframes_dir)
    for index, event in enumerate(events, start=1):
        source = view_sources.get(event.view) or view_sources.get("third_person")
        interaction = next(
            (
                item
                for item in interactions
                if item.view == event.view
                and item.object_label == event.object_label
                and abs(item.local_time_sec - event.local_time_sec) < 1e-6
            ),
            None,
        )
        if source is None:
            continue
        output_path = output_dir / f"interaction_{index:03d}.jpg"
        saved_path = _write_keyframe_at_time(
            source.path,
            event.local_time_sec,
            output_path,
            dry_run=dry_run,
            source_image_path=interaction.source_image_path if interaction else None,
            interaction=interaction,
        )
        event.keyframe_path = str(saved_path)
        labels = [event.hand_label, event.object_label]
        keyframes.append(
            InteractionKeyframe(
                path=str(saved_path),
                view=event.view,
                local_time_sec=event.local_time_sec,
                global_time=event.global_time,
                interaction=event.interaction,
                event_id=event.event_id,
                source=event.source,
                labels=labels,
            )
        )
    return keyframes, events, interactions


def _segment_cv_confidence(segment: DetectedSegment) -> float:
    boundary_confidence = float(getattr(segment, "boundary_confidence", 0.0) or 0.0)
    score_confidence = float(segment.avg_active_score or segment.avg_motion_score or 0.0)
    support_bonus = min(0.15, 0.02 * float(getattr(segment, "yolo_interaction_count", 0) or 0))
    value = max(boundary_confidence, min(1.0, score_confidence + support_bonus))
    return round(max(0.0, min(1.0, value)), 6)


def _segment_asset_binding(
    *,
    segment: DetectedSegment,
    view: str,
    source: VideoSource,
    clip: Path,
    local_start_sec: float,
    local_end_sec: float,
    keyframe_paths: list[Path],
) -> dict[str, Any]:
    keyframes = {
        name: str(path)
        for name, path in zip(("start", "middle", "end"), keyframe_paths)
    }
    return {
        "level": "segment",
        "segment_id": segment.segment_id,
        "view": view,
        "video_path": source.path,
        "global_start_time": segment.global_start_time,
        "global_end_time": segment.global_end_time,
        "local_start_sec": float(local_start_sec),
        "local_end_sec": float(local_end_sec),
        "clip_path": str(clip),
        "keyframe_path": keyframes.get("middle") or keyframes.get("start"),
        "keyframe_paths": [str(path) for path in keyframe_paths],
        "keyframes": keyframes,
        "confidence": _segment_cv_confidence(segment),
        "evidence_source": segment.detector_backend,
        "detector_source_view": segment.detector_source_view,
    }


def extract_multiview_clips(
    manifest: SessionManifest,
    segment: DetectedSegment,
    clips_dir: str | Path,
    keyframes_dir: str | Path,
    yolo_frame_rows: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> KeyActionSegment:
    clips_root = Path(clips_dir)
    keyframes_root = Path(keyframes_dir)
    segment_clip_dir = clips_root / segment.segment_id
    segment_keyframes_dir = keyframes_root / segment.segment_id
    segment_clip_dir.mkdir(parents=True, exist_ok=True)
    segment_keyframes_dir.mkdir(parents=True, exist_ok=True)

    third = manifest.videos.third_person
    third_start = global_time_to_local_sec(third, segment.global_start_time)
    third_end = global_time_to_local_sec(third, segment.global_end_time)
    third_clip = segment_clip_dir / "third_person.mp4"
    extract_clip_ffmpeg(third.path, third_start, third_end, third_clip, dry_run=dry_run)
    third_keyframes = extract_keyframes(third.path, third_start, third_end, segment_keyframes_dir, "third_person", dry_run=dry_run)
    third_ref = ClipReference(
        video_path=third.path,
        clip_path=str(third_clip),
        local_start_sec=float(third_start),
        local_end_sec=float(third_end),
    )
    asset_bindings = [
        _segment_asset_binding(
            segment=segment,
            view="third_person",
            source=third,
            clip=third_clip,
            local_start_sec=third_start,
            local_end_sec=third_end,
            keyframe_paths=third_keyframes,
        )
    ]

    first_ref = None
    if manifest.videos.first_person is not None:
        first = manifest.videos.first_person
        first_start = global_time_to_local_sec(first, segment.global_start_time)
        first_end = global_time_to_local_sec(first, segment.global_end_time)
        first_clip = segment_clip_dir / "first_person.mp4"
        extract_clip_ffmpeg(first.path, first_start, first_end, first_clip, dry_run=dry_run)
        first_keyframes = extract_keyframes(first.path, first_start, first_end, segment_keyframes_dir, "first_person", dry_run=dry_run)
        first_ref = ClipReference(
            video_path=first.path,
            clip_path=str(first_clip),
            local_start_sec=float(first_start),
            local_end_sec=float(first_end),
        )
        asset_bindings.append(
            _segment_asset_binding(
                segment=segment,
                view="first_person",
                source=first,
                clip=first_clip,
                local_start_sec=first_start,
                local_end_sec=first_end,
                keyframe_paths=first_keyframes,
            )
        )

    view_refs: dict[str, ClipReference | None] = {"third_person": third_ref, "first_person": first_ref}
    view_sources: dict[str, VideoSource] = {"third_person": third}
    if manifest.videos.first_person is not None:
        view_sources["first_person"] = manifest.videos.first_person
    interaction_keyframes, interaction_events, yolo_interactions = extract_interaction_keyframes(
        segment=segment,
        segment_keyframes_dir=segment_keyframes_dir,
        yolo_frame_rows=yolo_frame_rows,
        view_refs=view_refs,
        view_sources=view_sources,
        dry_run=dry_run,
    )

    return KeyActionSegment(
        session_id=manifest.session_id,
        segment_id=segment.segment_id,
        global_start_time=segment.global_start_time,
        global_end_time=segment.global_end_time,
        duration_sec=segment.duration_sec,
        third_person=third_ref,
        first_person=first_ref,
        cv_detection=CVDetectionSummary(
            avg_motion_score=segment.avg_motion_score,
            avg_active_score=segment.avg_active_score,
            start_reason=segment.start_reason,
            end_reason=segment.end_reason,
            start_sec=float(segment.start_sec),
            end_sec=float(segment.end_sec),
            confidence=_segment_cv_confidence(segment),
        ),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo(embedding_id="", index_text="", vector_store=""),
        interaction_keyframes=interaction_keyframes,
        interaction_events=interaction_events,
        yolo_interactions=yolo_interactions,
        asset_bindings=asset_bindings,
    )
