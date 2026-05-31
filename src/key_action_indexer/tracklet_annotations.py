from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from .yolo_detector import HAND_LABELS, canonical_yolo_label, filter_implausible_detections


TRACKLET_SCHEMA_VERSION = "yolo_tracklet_annotation.v1"
TRACKLET_ANNOTATION_SCHEMA_VERSION = TRACKLET_SCHEMA_VERSION
TRACKLET_RENDER_PLAN_SCHEMA_VERSION = "yolo_tracklet_render_plan.v1"


@dataclass(frozen=True)
class TrackletAnnotationConfig:
    min_iou: float = 0.1
    max_center_distance_px: float = 80.0
    max_missing_frames: int = 1
    interpolation_confidence_decay: float = 0.85
    smoothing_radius: int = 1
    outlier_jump_ratio: float = 2.4
    outlier_return_ratio: float = 1.25
    max_single_frame_size_change_ratio: float = 1.9
    tracklet_id_prefix: str = "yolo_tracklet"


@dataclass
class _AnnotationFrame:
    input_index: int
    view_order: int
    view: str
    time_sec: float | None
    frame_index: Any
    global_time: Any
    detections: list[dict[str, Any]]


@dataclass
class _AnnotationTrack:
    label: str
    view: str
    ordinal: int
    tracklet_id: str
    points: list[dict[str, Any]]

    @property
    def last_bbox(self) -> list[float]:
        return list(self.points[-1]["raw_bbox"])

    @property
    def last_view_order(self) -> int:
        return int(self.points[-1]["view_order"])


def build_tracklet_annotations(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
    labels: Iterable[str] | None = None,
    views: Iterable[str] | None = None,
    config: TrackletAnnotationConfig | None = None,
    min_iou: float | None = None,
    max_center_distance_px: float | None = None,
    max_missing_frames: int | None = None,
    interpolation_confidence_decay: float | None = None,
    smoothing_radius: int | None = None,
    tracklet_id_prefix: str | None = None,
    target_labels: Sequence[Any] | None = None,
    include_hands: bool | None = None,
    max_gap_sec: float | None = None,
    iou_threshold: float | None = None,
    center_distance_ratio: float | None = None,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Build per-frame tracklet annotations, with legacy summary compatibility.

    Calls that pass ``target_labels``/``include_hands`` keep the existing
    material_references summary contract. New callers get a flat per-frame list
    with ``tracklet_id``, ``object_track_id``, bbox, source, and confidence.
    """

    if (
        target_labels is not None
        or include_hands is not None
        or max_gap_sec is not None
        or iou_threshold is not None
        or center_distance_ratio is not None
    ):
        return build_tracklet_summary(
            frame_rows,
            target_labels=target_labels,
            include_hands=True if include_hands is None else include_hands,
            max_gap_sec=1.25 if max_gap_sec is None else max_gap_sec,
            iou_threshold=0.08 if iou_threshold is None else iou_threshold,
            center_distance_ratio=1.6 if center_distance_ratio is None else center_distance_ratio,
        )

    settings = _annotation_settings(
        config,
        min_iou=min_iou,
        max_center_distance_px=max_center_distance_px,
        max_missing_frames=max_missing_frames,
        interpolation_confidence_decay=interpolation_confidence_decay,
        smoothing_radius=smoothing_radius,
        tracklet_id_prefix=tracklet_id_prefix,
    )
    frames = _annotation_frames(
        frame_rows,
        window_start_sec=window_start_sec,
        window_end_sec=window_end_sec,
        labels={_canon(label) for label in (labels or []) if _canon(label)},
        views={str(view) for view in (views or []) if str(view).strip()},
    )
    if not frames:
        return []

    frames_by_view: dict[str, list[_AnnotationFrame]] = defaultdict(list)
    labels_by_view: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        frames_by_view[frame.view].append(frame)
        for detection in frame.detections:
            labels_by_view[frame.view].add(str(detection["label"]))

    tracks: list[_AnnotationTrack] = []
    for view in sorted(frames_by_view):
        view_frames = sorted(frames_by_view[view], key=lambda item: item.view_order)
        for label in sorted(labels_by_view[view]):
            tracks.extend(_annotation_tracks_for_label_view(view_frames, view, label, settings))

    frames_by_key = {(frame.view, frame.view_order): frame for frame in frames}
    rows: list[dict[str, Any]] = []
    for track in tracks:
        points = _annotation_interpolated_points(track, frames_by_key, settings)
        points = _annotation_smoothed_points(points, settings.smoothing_radius)
        confidence_values = [float(point.get("confidence", 0.0) or 0.0) for point in points]
        track_confidence = round(mean(confidence_values), 4) if confidence_values else 0.0
        for point_index, point in enumerate(points):
            rows.append(
                _annotation_row(
                    point,
                    track=track,
                    point_index=point_index,
                    point_count=len(points),
                    track_confidence=track_confidence,
                    settings=settings,
                    window_start_sec=window_start_sec,
                    window_end_sec=window_end_sec,
                )
            )
    return sorted(rows, key=_annotation_sort_key)


def build_tracklet_summary(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    target_labels: Sequence[Any] | None = None,
    include_hands: bool = True,
    max_gap_sec: float = 1.25,
    iou_threshold: float = 0.08,
    center_distance_ratio: float = 1.6,
) -> dict[str, Any]:
    """Build lightweight YOLO tracklets from sampled evidence rows.

    This is intentionally dependency-free so dry-run and unit tests do not need
    ffmpeg, OpenCV, or a tracker package.  It provides stable per-frame bbox
    candidates for rendering; the detector remains the evidence source.
    """

    wanted = {_canon(label) for label in (target_labels or []) if _canon(label)}
    if include_hands:
        wanted.update(_canon(label) for label in HAND_LABELS)
    observations = _collect_observations(evidence_rows, wanted)
    observations.sort(key=lambda item: (item["label"], item["view"], item["time_sec"], -float(item["confidence"])))

    active: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []
    next_id = 1
    for obs in observations:
        candidates = [
            track
            for track in active
            if track["label"] == obs["label"]
            and track["view"] == obs["view"]
            and 0 <= obs["time_sec"] - track["last_time_sec"] <= max_gap_sec
        ]
        selected = _best_track(candidates, obs, iou_threshold=iou_threshold, center_distance_ratio=center_distance_ratio)
        if selected is None:
            selected = {
                "tracklet_id": f"trk_{next_id:04d}",
                "label": obs["label"],
                "view": obs["view"],
                "observations": [],
                "last_bbox": obs["bbox"],
                "last_time_sec": obs["time_sec"],
                "source_track_id": obs.get("source_track_id"),
            }
            next_id += 1
            active.append(selected)
        selected["observations"].append(obs)
        selected["last_bbox"] = obs["bbox"]
        selected["last_time_sec"] = obs["time_sec"]
        if not selected.get("source_track_id") and obs.get("source_track_id"):
            selected["source_track_id"] = obs.get("source_track_id")

        still_active: list[dict[str, Any]] = []
        for track in active:
            if obs["time_sec"] - track["last_time_sec"] > max_gap_sec * 2:
                completed.append(track)
            else:
                still_active.append(track)
        active = still_active
    completed.extend(active)
    completed = _absorb_single_observation_outlier_tracks(completed)

    tracklets = [_finalize_track(track) for track in completed if track.get("observations")]
    tracklets.sort(key=lambda item: (item["label"], item["view"], item["start_sec"], item["tracklet_id"]))
    return {
        "schema_version": TRACKLET_SCHEMA_VERSION,
        "tracklet_count": len(tracklets),
        "target_labels": sorted(wanted),
        "tracklets": tracklets,
    }


def select_tracklet_annotations(
    annotations: Iterable[Mapping[str, Any]],
    *,
    view: str | None = None,
    labels: Iterable[str] | None = None,
    tracklet_ids: Iterable[str] | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    time_sec: float | None = None,
    time_tolerance_sec: float | None = None,
    frame_index: Any | None = None,
    include_interpolated: bool = True,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    label_filter = {_canon(label) for label in (labels or []) if _canon(label)}
    tracklet_filter = {str(tracklet_id) for tracklet_id in (tracklet_ids or []) if str(tracklet_id).strip()}
    selected: list[dict[str, Any]] = []
    for row in annotations:
        if view is not None and str(row.get("view") or "") != str(view):
            continue
        if label_filter and _canon(row.get("label") or row.get("object_label")) not in label_filter:
            continue
        if tracklet_filter and str(row.get("tracklet_id") or "") not in tracklet_filter:
            continue
        if not include_interpolated and str(row.get("source") or "") == "interpolated":
            continue
        row_time = _annotation_float(row.get("time_sec"))
        if start_sec is not None and (row_time is None or row_time < float(start_sec)):
            continue
        if end_sec is not None and (row_time is None or row_time > float(end_sec)):
            continue
        if time_sec is not None:
            if row_time is None:
                continue
            distance = abs(row_time - float(time_sec))
            if time_tolerance_sec is not None and distance > float(time_tolerance_sec):
                continue
        if frame_index is not None and row.get("frame_index") != frame_index:
            continue
        selected.append(dict(row))

    selected.sort(key=lambda row: _annotation_selection_sort_key(row, time_sec=time_sec))
    if max_rows is not None:
        return selected[: max(0, int(max_rows))]
    return selected


def prepare_tracklet_render_plan(
    annotations: Iterable[Mapping[str, Any]],
    *,
    view: str | None = None,
    labels: Iterable[str] | None = None,
    tracklet_ids: Iterable[str] | None = None,
    start_sec: float | None = None,
    end_sec: float | None = None,
    time_sec: float | None = None,
    time_tolerance_sec: float | None = None,
    frame_index: Any | None = None,
    include_interpolated: bool = True,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    selected = select_tracklet_annotations(
        annotations,
        view=view,
        labels=labels,
        tracklet_ids=tracklet_ids,
        start_sec=start_sec,
        end_sec=end_sec,
        time_sec=time_sec,
        time_tolerance_sec=time_tolerance_sec,
        frame_index=frame_index,
        include_interpolated=include_interpolated,
        max_rows=max_rows,
    )
    grouped: dict[tuple[str, Any, float | None], list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        grouped[(str(row.get("view") or "unknown"), row.get("frame_index"), _annotation_float(row.get("time_sec")))].append(row)

    plans: list[dict[str, Any]] = []
    for (row_view, row_frame_index, row_time), rows in sorted(grouped.items(), key=_annotation_render_group_sort_key):
        rows.sort(key=lambda row: (str(row.get("label") or ""), str(row.get("tracklet_id") or "")))
        plans.append(
            {
                "schema_version": TRACKLET_RENDER_PLAN_SCHEMA_VERSION,
                "source": "tracklet_annotations",
                "view": row_view,
                "frame_index": row_frame_index,
                "time_sec": row_time,
                "global_time": _first_non_empty(row.get("global_time") for row in rows),
                "annotations": [_render_annotation(row) for row in rows],
            }
        )
    return plans


def detections_for_time(
    annotation: Mapping[str, Any] | None,
    local_time_sec: float,
    *,
    hold_sec: float = 0.75,
) -> list[dict[str, Any]]:
    if not annotation:
        return []
    detections: list[dict[str, Any]] = []
    for track in annotation.get("tracklets") or []:
        if not isinstance(track, Mapping):
            continue
        det = _track_detection_at(track, local_time_sec, hold_sec=hold_sec)
        if det is not None:
            detections.append(det)
    detections.sort(key=lambda item: (0 if _canon(item.get("label")) not in {_canon(label) for label in HAND_LABELS} else 1, item["label"]))
    return detections


def summarize_tracklets(annotation: Mapping[str, Any] | None) -> dict[str, Any]:
    if not annotation:
        return {"schema_version": TRACKLET_SCHEMA_VERSION, "tracklet_count": 0}
    tracklets = [track for track in annotation.get("tracklets") or [] if isinstance(track, Mapping)]
    return {
        "schema_version": str(annotation.get("schema_version") or TRACKLET_SCHEMA_VERSION),
        "tracklet_count": len(tracklets),
        "labels": sorted({_canon(track.get("label")) for track in tracklets if _canon(track.get("label"))}),
        "interpolated_tracklet_count": sum(1 for track in tracklets if int(track.get("interpolated_count") or 0) > 0),
    }


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    return _annotation_iou(box_a, box_b)


def center_distance(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax, ay = _annotation_center(box_a)
    bx, by = _annotation_center(box_b)
    return math.dist((ax, ay), (bx, by))


def _annotation_settings(
    config: TrackletAnnotationConfig | None,
    *,
    min_iou: float | None,
    max_center_distance_px: float | None,
    max_missing_frames: int | None,
    interpolation_confidence_decay: float | None,
    smoothing_radius: int | None,
    tracklet_id_prefix: str | None,
) -> TrackletAnnotationConfig:
    base = config or TrackletAnnotationConfig()
    return TrackletAnnotationConfig(
        min_iou=float(base.min_iou if min_iou is None else min_iou),
        max_center_distance_px=float(base.max_center_distance_px if max_center_distance_px is None else max_center_distance_px),
        max_missing_frames=max(0, int(base.max_missing_frames if max_missing_frames is None else max_missing_frames)),
        interpolation_confidence_decay=max(
            0.0,
            min(1.0, float(base.interpolation_confidence_decay if interpolation_confidence_decay is None else interpolation_confidence_decay)),
        ),
        smoothing_radius=max(0, int(base.smoothing_radius if smoothing_radius is None else smoothing_radius)),
        outlier_jump_ratio=float(base.outlier_jump_ratio),
        outlier_return_ratio=float(base.outlier_return_ratio),
        max_single_frame_size_change_ratio=float(base.max_single_frame_size_change_ratio),
        tracklet_id_prefix=str(base.tracklet_id_prefix if tracklet_id_prefix is None else tracklet_id_prefix),
    )


def _annotation_frames(
    frame_rows: Sequence[Mapping[str, Any]],
    *,
    window_start_sec: float | None,
    window_end_sec: float | None,
    labels: set[str],
    views: set[str],
) -> list[_AnnotationFrame]:
    collected: list[_AnnotationFrame] = []
    for input_index, row in enumerate(frame_rows):
        if not isinstance(row, Mapping):
            continue
        view = str(_first_non_empty([row.get("view"), row.get("source_view"), row.get("camera_view"), row.get("camera")]) or "unknown")
        if views and view not in views:
            continue
        time_sec = _annotation_row_time(row)
        if window_start_sec is not None and (time_sec is None or time_sec < float(window_start_sec)):
            continue
        if window_end_sec is not None and (time_sec is None or time_sec > float(window_end_sec)):
            continue
        detections: list[dict[str, Any]] = []
        for detection_index, detection in enumerate(_annotation_detections(row)):
            normalized = _annotation_detection(detection, detection_index=detection_index)
            if normalized is None:
                continue
            if labels and str(normalized["label"]) not in labels:
                continue
            detections.append(normalized)
        collected.append(
            _AnnotationFrame(
                input_index=input_index,
                view_order=-1,
                view=view,
                time_sec=time_sec,
                frame_index=row.get("frame_index"),
                global_time=row.get("global_time"),
                detections=detections,
            )
        )

    collected.sort(key=_annotation_frame_sort_key)
    view_counts: dict[str, int] = defaultdict(int)
    for frame in collected:
        frame.view_order = view_counts[frame.view]
        view_counts[frame.view] += 1
    return collected


def _annotation_tracks_for_label_view(
    frames: Sequence[_AnnotationFrame],
    view: str,
    label: str,
    settings: TrackletAnnotationConfig,
) -> list[_AnnotationTrack]:
    tracks: list[_AnnotationTrack] = []
    active: list[_AnnotationTrack] = []
    next_ordinal = 1
    for frame in frames:
        detections = [detection for detection in frame.detections if detection["label"] == label]
        matches = _annotation_matches(active, detections, frame.view_order, settings)
        matched_tracks = {track_index for track_index, _detection_index in matches}
        matched_detections = {detection_index for _track_index, detection_index in matches}

        for track_index, detection_index in matches:
            active[track_index].points.append(_annotation_detected_point(frame, detections[detection_index]))

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            tracklet_id = f"{settings.tracklet_id_prefix}:{_stable_token(view)}:{_stable_token(label)}:{next_ordinal:03d}"
            track = _AnnotationTrack(
                label=label,
                view=view,
                ordinal=next_ordinal,
                tracklet_id=tracklet_id,
                points=[_annotation_detected_point(frame, detection)],
            )
            next_ordinal += 1
            tracks.append(track)
            active.append(track)

        active = [
            track
            for index, track in enumerate(active)
            if index in matched_tracks or frame.view_order - track.last_view_order <= settings.max_missing_frames
        ]
    return tracks


def _annotation_matches(
    active: Sequence[_AnnotationTrack],
    detections: Sequence[Mapping[str, Any]],
    view_order: int,
    settings: TrackletAnnotationConfig,
) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for track_index, track in enumerate(active):
        missing = view_order - track.last_view_order - 1
        if missing < 0 or missing > settings.max_missing_frames:
            continue
        for detection_index, detection in enumerate(detections):
            score = _annotation_association_score(track.last_bbox, list(detection["bbox"]), settings)
            if score is not None:
                candidates.append((score, track_index, detection_index))

    candidates.sort(reverse=True)
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _score, track_index, detection_index in candidates:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        matches.append((track_index, detection_index))
    return sorted(matches)


def _annotation_association_score(
    previous_bbox: Sequence[float],
    current_bbox: Sequence[float],
    settings: TrackletAnnotationConfig,
) -> float | None:
    iou = _annotation_iou(previous_bbox, current_bbox)
    distance = center_distance(previous_bbox, current_bbox)
    if iou < settings.min_iou and distance > settings.max_center_distance_px:
        return None
    distance_score = max(0.0, 1.0 - (distance / max(settings.max_center_distance_px, 1.0)))
    return round(iou * 2.0 + distance_score, 8)


def _annotation_detected_point(frame: _AnnotationFrame, detection: Mapping[str, Any]) -> dict[str, Any]:
    bbox = [round(float(value), 4) for value in detection["bbox"]]
    return {
        "source": "detected",
        "label": detection["label"],
        "raw_label": detection.get("raw_label"),
        "raw_bbox": bbox,
        "bbox": bbox,
        "confidence": detection.get("confidence"),
        "time_sec": frame.time_sec,
        "frame_index": frame.frame_index,
        "global_time": frame.global_time,
        "view": frame.view,
        "view_order": frame.view_order,
        "input_index": frame.input_index,
        "source_row_index": frame.input_index,
        "source_detection_index": detection.get("detection_index"),
        "source_detection_id": detection.get("source_detection_id"),
        "source_track_id": detection.get("source_track_id"),
    }


def _annotation_interpolated_points(
    track: _AnnotationTrack,
    frames_by_key: Mapping[tuple[str, int], _AnnotationFrame],
    settings: TrackletAnnotationConfig,
) -> list[dict[str, Any]]:
    detected = sorted(track.points, key=lambda point: int(point["view_order"]))
    if len(detected) < 2:
        return [dict(point) for point in detected]

    points: list[dict[str, Any]] = []
    for previous, current in zip(detected, detected[1:]):
        if not points:
            points.append(dict(previous))
        gap = int(current["view_order"]) - int(previous["view_order"])
        if 1 < gap <= settings.max_missing_frames + 1:
            for step in range(1, gap):
                frame = frames_by_key.get((track.view, int(previous["view_order"]) + step))
                if frame is not None:
                    points.append(_annotation_interpolated_point(previous, current, frame, step / float(gap), settings))
        points.append(dict(current))
    return _annotation_stabilize_points(points, settings)


def _annotation_stabilize_points(points: Sequence[Mapping[str, Any]], settings: TrackletAnnotationConfig) -> list[dict[str, Any]]:
    if len(points) < 3:
        return [dict(point) for point in points]
    stabilized = [dict(point) for point in points]
    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        current = points[index]
        following = points[index + 1]
        previous_bbox = _annotation_bbox(previous.get("raw_bbox") or previous.get("bbox"))
        current_bbox = _annotation_bbox(current.get("raw_bbox") or current.get("bbox"))
        following_bbox = _annotation_bbox(following.get("raw_bbox") or following.get("bbox"))
        if previous_bbox is None or current_bbox is None or following_bbox is None:
            continue
        previous_next_distance = center_distance(previous_bbox, following_bbox)
        previous_current_distance = center_distance(previous_bbox, current_bbox)
        current_next_distance = center_distance(current_bbox, following_bbox)
        scale = max(_bbox_extent(previous_bbox), _bbox_extent(current_bbox), _bbox_extent(following_bbox), 1.0)
        jump_limit = max(settings.max_center_distance_px, settings.outlier_jump_ratio * scale)
        return_limit = max(settings.max_center_distance_px * 0.75, settings.outlier_return_ratio * scale)
        size_ratio = _bbox_size_ratio(current_bbox, previous_bbox, following_bbox)
        jump_out = previous_current_distance > jump_limit and current_next_distance > jump_limit
        returns_to_track = previous_next_distance <= return_limit
        size_out = size_ratio > settings.max_single_frame_size_change_ratio and returns_to_track
        if not ((jump_out and returns_to_track) or size_out):
            continue
        repaired_bbox = _lerp_bbox(previous_bbox, following_bbox, 0.5)
        stabilized[index]["raw_bbox"] = list(current_bbox)
        stabilized[index]["bbox"] = repaired_bbox
        stabilized[index]["source"] = "stabilized_outlier"
        stabilized[index]["confidence"] = round(
            min(float(previous.get("confidence", 0.0) or 0.0), float(following.get("confidence", 0.0) or 0.0)) * 0.88,
            4,
        )
        stabilized[index]["stabilization"] = {
            "method": "single_frame_outlier_interpolation",
            "outlier_bbox": list(current_bbox),
            "repaired_bbox": repaired_bbox,
            "previous_frame_index": previous.get("frame_index"),
            "next_frame_index": following.get("frame_index"),
            "previous_time_sec": previous.get("time_sec"),
            "next_time_sec": following.get("time_sec"),
            "prev_current_distance_px": round(previous_current_distance, 4),
            "current_next_distance_px": round(current_next_distance, 4),
            "prev_next_distance_px": round(previous_next_distance, 4),
            "size_ratio": round(size_ratio, 4),
        }
    return stabilized


def _annotation_interpolated_point(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    frame: _AnnotationFrame,
    fraction: float,
    settings: TrackletAnnotationConfig,
) -> dict[str, Any]:
    bbox = [
        round(float(left) + (float(right) - float(left)) * fraction, 4)
        for left, right in zip(previous["raw_bbox"], current["raw_bbox"])
    ]
    confidence = round(
        min(float(previous.get("confidence", 0.0) or 0.0), float(current.get("confidence", 0.0) or 0.0))
        * settings.interpolation_confidence_decay,
        4,
    )
    return {
        "source": "interpolated",
        "label": previous.get("label"),
        "raw_label": previous.get("raw_label"),
        "raw_bbox": bbox,
        "bbox": bbox,
        "confidence": confidence,
        "time_sec": frame.time_sec,
        "frame_index": frame.frame_index,
        "global_time": frame.global_time,
        "view": frame.view,
        "view_order": frame.view_order,
        "input_index": frame.input_index,
        "source_row_index": None,
        "source_detection_index": None,
        "source_detection_id": None,
        "source_track_id": previous.get("source_track_id") or current.get("source_track_id"),
        "interpolation": {
            "fraction": round(fraction, 4),
            "previous_time_sec": previous.get("time_sec"),
            "next_time_sec": current.get("time_sec"),
            "previous_frame_index": previous.get("frame_index"),
            "next_frame_index": current.get("frame_index"),
        },
    }


def _annotation_smoothed_points(points: Sequence[Mapping[str, Any]], radius: int) -> list[dict[str, Any]]:
    if radius <= 0 or len(points) < 3:
        return [dict(point) for point in points]
    smoothed: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        row = dict(point)
        if index == 0 or index == len(points) - 1:
            smoothed.append(row)
            continue
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        neighbors = [points[item].get("bbox") or points[item]["raw_bbox"] for item in range(start, end)]
        row["bbox"] = [round(mean(float(bbox[axis]) for bbox in neighbors), 4) for axis in range(4)]
        row["smoothing"] = {"method": "rolling_mean", "radius": radius, "source_bbox": list(point["raw_bbox"])}
        smoothed.append(row)
    return smoothed


def _annotation_row(
    point: Mapping[str, Any],
    *,
    track: _AnnotationTrack,
    point_index: int,
    point_count: int,
    track_confidence: float,
    settings: TrackletAnnotationConfig,
    window_start_sec: float | None,
    window_end_sec: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": TRACKLET_ANNOTATION_SCHEMA_VERSION,
        "tracklet_id": track.tracklet_id,
        "object_track_id": track.tracklet_id,
        "track_id": track.tracklet_id,
        "tracklet_index": track.ordinal,
        "tracklet_point_index": point_index,
        "tracklet_point_count": point_count,
        "label": track.label,
        "object_label": track.label,
        "view": track.view,
        "time_sec": point.get("time_sec"),
        "frame_index": point.get("frame_index"),
        "global_time": point.get("global_time"),
        "bbox": [round(float(value), 4) for value in point.get("bbox", [])],
        "raw_bbox": [round(float(value), 4) for value in point.get("raw_bbox", point.get("bbox", []))],
        "source": point.get("source"),
        "confidence": round(float(point.get("confidence", 0.0) or 0.0), 4),
        "tracklet_confidence": track_confidence,
        "source_row_index": point.get("source_row_index"),
        "source_detection_index": point.get("source_detection_index"),
        "source_detection_id": point.get("source_detection_id"),
        "source_track_id": point.get("source_track_id"),
        "interpolation": point.get("interpolation"),
        "smoothing": point.get("smoothing"),
        "settings": {
            "min_iou": settings.min_iou,
            "max_center_distance_px": settings.max_center_distance_px,
            "max_missing_frames": settings.max_missing_frames,
            "interpolation_confidence_decay": settings.interpolation_confidence_decay,
            "smoothing_radius": settings.smoothing_radius,
        },
        "window_start_sec": window_start_sec,
        "window_end_sec": window_end_sec,
    }


def _render_annotation(row: Mapping[str, Any]) -> dict[str, Any]:
    tracklet_id = str(row.get("tracklet_id") or row.get("object_track_id") or "")
    label = str(row.get("label") or row.get("object_label") or "object")
    source = str(row.get("source") or "detected")
    confidence = round(float(row.get("confidence", 0.0) or 0.0), 4)
    return {
        "tracklet_id": tracklet_id,
        "object_track_id": row.get("object_track_id") or tracklet_id,
        "label": label,
        "bbox": [round(float(value), 4) for value in row.get("bbox", [])],
        "source": source,
        "confidence": confidence,
        "text": f"{label} {tracklet_id.rsplit(':', 1)[-1]} {source} {confidence:.2f}",
        "style": {
            "stroke": _track_color(tracklet_id),
            "stroke_width": 2 if source == "detected" else 1,
            "dash": [] if source == "detected" else [4, 3],
        },
    }


def _annotation_detection(detection: Mapping[str, Any], *, detection_index: int) -> dict[str, Any] | None:
    bbox = _annotation_bbox(detection.get("bbox", detection.get("box", detection.get("xyxy"))))
    label = _canon(_first_non_empty([detection.get("label"), detection.get("object_label"), detection.get("raw_label")]))
    if bbox is None or not label:
        return None
    return {
        "label": label,
        "raw_label": _first_non_empty([detection.get("raw_label"), detection.get("label"), detection.get("object_label")]),
        "bbox": bbox,
        "confidence": _annotation_confidence(detection),
        "detection_index": detection_index,
        "source_detection_id": _first_non_empty([detection.get("detection_id"), detection.get("id")]),
        "source_track_id": _first_non_empty([detection.get("track_id"), detection.get("object_track_id"), detection.get("tracklet_id")]),
    }


def _annotation_detections(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    detections = row.get("detections")
    return [item for item in detections if isinstance(item, Mapping)] if isinstance(detections, list) else []


def _annotation_bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
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
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 == x1 or y2 == y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def _annotation_confidence(row: Mapping[str, Any]) -> float:
    value = _first_annotation_float(row.get("confidence"), row.get("score"), row.get("probability"), row.get("prob"), row.get("raw_score"))
    if value is None:
        return 0.5
    return round(max(0.0, min(1.0, value)), 4)


def _annotation_row_time(row: Mapping[str, Any]) -> float | None:
    return _first_annotation_float(row.get("alignment_time_sec"), row.get("session_time_sec"), row.get("local_time_sec"), row.get("time_sec"))


def _annotation_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - inter
    return 0.0 if union <= 0.0 else inter / union


def _annotation_center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _annotation_frame_sort_key(frame: _AnnotationFrame) -> tuple[str, float, float, int]:
    time_value = frame.time_sec if frame.time_sec is not None else float("inf")
    frame_value = _annotation_float(frame.frame_index, float("inf"))
    return (frame.view, time_value, frame_value, frame.input_index)


def _annotation_sort_key(row: Mapping[str, Any]) -> tuple[str, float, float, str, int]:
    return (
        str(row.get("view") or ""),
        _annotation_float(row.get("time_sec"), float("inf")),
        _annotation_float(row.get("frame_index"), float("inf")),
        str(row.get("tracklet_id") or ""),
        int(row.get("tracklet_point_index") or 0),
    )


def _annotation_selection_sort_key(row: Mapping[str, Any], *, time_sec: float | None) -> tuple[float, int, float, str]:
    row_time = _annotation_float(row.get("time_sec"), float("inf"))
    distance = abs(row_time - float(time_sec)) if time_sec is not None and row_time != float("inf") else 0.0
    source_rank = 0 if str(row.get("source") or "") == "detected" else 1
    return (distance, source_rank, -float(row.get("confidence", 0.0) or 0.0), str(row.get("tracklet_id") or ""))


def _annotation_render_group_sort_key(item: tuple[tuple[str, Any, float | None], list[dict[str, Any]]]) -> tuple[str, float, float]:
    (view, frame_index, time_sec), _rows = item
    return (view, float("inf") if time_sec is None else float(time_sec), _annotation_float(frame_index, float("inf")))


def _track_color(tracklet_id: str) -> str:
    digest = hashlib.sha1(tracklet_id.encode("utf-8")).hexdigest()
    palette = ("#2f80ed", "#27ae60", "#eb5757", "#9b51e0", "#f2994a", "#00a6a6", "#b83280", "#4f4f4f")
    return palette[int(digest[:2], 16) % len(palette)]


def _stable_token(value: str) -> str:
    return str(value or "unknown").replace(":", "_")


def _first_annotation_float(*values: Any) -> float | None:
    for value in values:
        numeric = _annotation_float(value)
        if numeric is not None:
            return numeric
    return None


def _annotation_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_non_empty(values: Iterable[Any]) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _collect_observations(evidence_rows: Sequence[Mapping[str, Any]], wanted: set[str]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for row_index, row in enumerate(evidence_rows):
        if not isinstance(row, Mapping):
            continue
        time_sec = _float(row.get("local_time_sec"), _float(row.get("time_sec"), 0.0))
        view = str(row.get("view") or row.get("source_view") or "")
        raw_detections = [item for item in row.get("detections") or [] if isinstance(item, Mapping)]
        detections, _ignored = filter_implausible_detections(
            raw_detections,
            frame_width=_optional_int(row.get("frame_width")),
            frame_height=_optional_int(row.get("frame_height")),
            source_view=view,
        )
        seen: set[tuple[str, tuple[int, int, int, int]]] = set()
        for detection in detections:
            if not isinstance(detection, Mapping):
                continue
            label = _canon(detection.get("label") or detection.get("class_name") or detection.get("name"))
            bbox = _bbox(detection.get("bbox"))
            if not label or bbox is None or (wanted and label not in wanted):
                continue
            key = (label, tuple(int(round(value)) for value in bbox))
            if key in seen:
                continue
            seen.add(key)
            observations.append(
                {
                    "label": label,
                    "bbox": bbox,
                    "confidence": _float(detection.get("confidence"), _float(detection.get("score"), 0.0)),
                    "time_sec": time_sec,
                    "view": view,
                    "frame_id": row.get("frame_id") or row.get("frame_index") or row_index,
                    "source": "detected",
                    "source_track_id": _first_non_empty([detection.get("track_id"), detection.get("object_track_id"), detection.get("tracklet_id")]),
                }
            )
        for interaction in row.get("hand_object_interactions") or []:
            if not isinstance(interaction, Mapping):
                continue
            object_label = _canon(interaction.get("object_label") or interaction.get("target_label") or interaction.get("object"))
            hand_label = _canon(interaction.get("hand_label") or "gloved_hand")
            if not _tracklet_interaction_supported_by_detections(interaction, object_label, detections):
                continue
            for label, bbox_value in ((object_label, interaction.get("object_bbox")), (hand_label, interaction.get("hand_bbox"))):
                bbox = _bbox(bbox_value)
                if not label or bbox is None or (wanted and label not in wanted):
                    continue
                key = (label, tuple(int(round(value)) for value in bbox))
                if key in seen:
                    continue
                seen.add(key)
                observations.append(
                    {
                        "label": label,
                        "bbox": bbox,
                        "confidence": _float(
                            interaction.get("score"),
                            _float(interaction.get("interaction_score"), _float(interaction.get("confidence"), 0.0)),
                        ),
                        "time_sec": time_sec,
                        "view": view,
                        "frame_id": row.get("frame_id") or row.get("frame_index") or row_index,
                        "source": "interaction_bbox",
                        "source_track_id": _first_non_empty([interaction.get("track_id"), interaction.get("object_track_id"), interaction.get("tracklet_id")]),
                    }
                )
    return observations


def _tracklet_interaction_supported_by_detections(
    interaction: Mapping[str, Any],
    object_label: str,
    detections: Sequence[Mapping[str, Any]],
) -> bool:
    if not object_label:
        return False
    return _has_matching_detection(interaction.get("hand_bbox"), {_canon(label) for label in HAND_LABELS}, detections) and _has_matching_detection(
        interaction.get("object_bbox"),
        {object_label},
        detections,
    )


def _has_matching_detection(
    bbox_value: Any,
    labels: set[str],
    detections: Sequence[Mapping[str, Any]],
) -> bool:
    bbox = _bbox(bbox_value)
    if bbox is None:
        return False
    rounded = tuple(int(round(value)) for value in bbox)
    for detection in detections:
        label = _canon(detection.get("label") or detection.get("object_label") or detection.get("name"))
        if label not in labels:
            continue
        det_bbox = _bbox(detection.get("bbox"))
        if det_bbox is None:
            continue
        if tuple(int(round(value)) for value in det_bbox) == rounded:
            return True
    return False


def _best_track(
    tracks: Sequence[Mapping[str, Any]],
    obs: Mapping[str, Any],
    *,
    iou_threshold: float,
    center_distance_ratio: float,
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    for track in tracks:
        last_bbox = track.get("last_bbox")
        if not isinstance(last_bbox, list):
            continue
        iou = _iou(last_bbox, obs["bbox"])
        distance = _center_distance_ratio(last_bbox, obs["bbox"])
        track_source_id = str(track.get("source_track_id") or "")
        obs_source_id = str(obs.get("source_track_id") or "")
        if track_source_id and obs_source_id and track_source_id != obs_source_id:
            continue
        if iou < iou_threshold and distance > center_distance_ratio:
            continue
        source_bonus = 0.75 if track_source_id and obs_source_id and track_source_id == obs_source_id else 0.0
        score = source_bonus + iou + max(0.0, center_distance_ratio - distance) * 0.1
        if best is None or score > best[0]:
            best = (score, track)  # type: ignore[arg-type]
    return best[1] if best else None


def _absorb_single_observation_outlier_tracks(tracks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    mutable = [{**dict(track), "observations": [dict(obs) for obs in track.get("observations") or []]} for track in tracks]
    absorbed: set[int] = set()
    for index, track in enumerate(mutable):
        observations = track.get("observations") or []
        if len(observations) != 1:
            continue
        obs = observations[0]
        obs_bbox = _bbox(obs.get("bbox"))
        obs_time = _float(obs.get("time_sec"), 0.0)
        if obs_bbox is None:
            continue
        best_host: tuple[float, int, dict[str, Any]] | None = None
        for host_index, host in enumerate(mutable):
            if host_index == index or host_index in absorbed:
                continue
            if host.get("label") != track.get("label") or host.get("view") != track.get("view"):
                continue
            host_points = sorted([dict(item) for item in host.get("observations") or []], key=lambda item: _float(item.get("time_sec"), 0.0))
            before = max((item for item in host_points if _float(item.get("time_sec"), 0.0) < obs_time), key=lambda item: _float(item.get("time_sec"), 0.0), default=None)
            after = min((item for item in host_points if _float(item.get("time_sec"), 0.0) > obs_time), key=lambda item: _float(item.get("time_sec"), 0.0), default=None)
            if before is None or after is None:
                continue
            before_bbox = _bbox(before.get("bbox"))
            after_bbox = _bbox(after.get("bbox"))
            if before_bbox is None or after_bbox is None:
                continue
            before_after_distance = _center_distance_px(before_bbox, after_bbox)
            before_obs_distance = _center_distance_px(before_bbox, obs_bbox)
            obs_after_distance = _center_distance_px(obs_bbox, after_bbox)
            scale = max(_bbox_extent(before_bbox), _bbox_extent(after_bbox), _bbox_extent(obs_bbox), 1.0)
            if before_after_distance > max(48.0, scale * 1.25):
                continue
            if before_obs_distance <= max(64.0, scale * 2.0) or obs_after_distance <= max(64.0, scale * 2.0):
                continue
            score = before_obs_distance + obs_after_distance - before_after_distance
            repaired = {
                **obs,
                "raw_bbox": list(obs_bbox),
                "bbox": _lerp_bbox(before_bbox, after_bbox, 0.5),
                "source": "stabilized_outlier",
                "confidence": round(min(_float(before.get("confidence"), 0.0), _float(after.get("confidence"), 0.0)) * 0.88, 4),
                "stabilization": {
                    "method": "single_frame_track_absorption",
                    "absorbed_tracklet_id": track.get("tracklet_id"),
                    "outlier_bbox": list(obs_bbox),
                    "previous_time_sec": before.get("time_sec"),
                    "next_time_sec": after.get("time_sec"),
                    "repaired_bbox": _lerp_bbox(before_bbox, after_bbox, 0.5),
                },
            }
            if best_host is None or score > best_host[0]:
                best_host = (score, host_index, repaired)
        if best_host is None:
            continue
        _score, host_index, repaired = best_host
        mutable[host_index]["observations"].append(repaired)
        absorbed.add(index)
    return [track for index, track in enumerate(mutable) if index not in absorbed]


def _finalize_track(track: Mapping[str, Any]) -> dict[str, Any]:
    observations = sorted([dict(obs) for obs in track.get("observations") or []], key=lambda item: item["time_sec"])
    stabilized = _stabilize_observation_outliers(observations)
    smoothed = _smooth_observations(stabilized)
    interpolated = _interpolate_missing(smoothed)
    all_points = sorted([*smoothed, *interpolated], key=lambda item: (item["time_sec"], item["source"]))
    quality = _tracklet_quality(all_points)
    return {
        "tracklet_id": track["tracklet_id"],
        "object_track_id": track["tracklet_id"],
        "label": track["label"],
        "view": track.get("view") or None,
        "start_sec": all_points[0]["time_sec"],
        "end_sec": all_points[-1]["time_sec"],
        "detected_count": len(smoothed),
        "interpolated_count": len(interpolated),
        "stabilized_outlier_count": sum(1 for point in all_points if point.get("stabilization")),
        "quality": quality,
        "points": all_points,
    }


def _stabilize_observation_outliers(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = [{**dict(obs), "raw_bbox": list(obs.get("raw_bbox") or obs["bbox"])} for obs in observations]
    if len(result) < 3:
        return result
    for index in range(1, len(result) - 1):
        previous = result[index - 1]
        current = result[index]
        following = result[index + 1]
        previous_bbox = _bbox(previous.get("bbox"))
        current_bbox = _bbox(current.get("bbox"))
        following_bbox = _bbox(following.get("bbox"))
        if previous_bbox is None or current_bbox is None or following_bbox is None:
            continue
        previous_next_distance = _center_distance_px(previous_bbox, following_bbox)
        previous_current_distance = _center_distance_px(previous_bbox, current_bbox)
        current_next_distance = _center_distance_px(current_bbox, following_bbox)
        scale = max(_bbox_extent(previous_bbox), _bbox_extent(current_bbox), _bbox_extent(following_bbox), 1.0)
        jump_limit = max(64.0, scale * 2.4)
        return_limit = max(48.0, scale * 1.25)
        size_ratio = _bbox_size_ratio(current_bbox, previous_bbox, following_bbox)
        jump_out = previous_current_distance > jump_limit and current_next_distance > jump_limit
        returns_to_track = previous_next_distance <= return_limit
        size_out = size_ratio > 1.9 and returns_to_track
        if not ((jump_out and returns_to_track) or size_out):
            continue
        repaired_bbox = _lerp_bbox(previous_bbox, following_bbox, 0.5)
        current["raw_bbox"] = list(current_bbox)
        current["bbox"] = repaired_bbox
        current["source"] = "stabilized_outlier"
        current["confidence"] = round(min(float(previous.get("confidence", 0.0) or 0.0), float(following.get("confidence", 0.0) or 0.0)) * 0.88, 4)
        current["stabilization"] = {
            "method": "single_frame_outlier_interpolation",
            "outlier_bbox": list(current_bbox),
            "repaired_bbox": repaired_bbox,
            "previous_time_sec": previous.get("time_sec"),
            "next_time_sec": following.get("time_sec"),
            "prev_current_distance_px": round(previous_current_distance, 4),
            "current_next_distance_px": round(current_next_distance, 4),
            "prev_next_distance_px": round(previous_next_distance, 4),
            "size_ratio": round(size_ratio, 4),
        }
    return result


def _smooth_observations(observations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous_bbox: list[float] | None = None
    for obs in observations:
        bbox = list(obs["bbox"])
        raw_bbox = list(obs.get("raw_bbox") or bbox)
        if previous_bbox is not None and obs.get("source") != "stabilized_outlier":
            bbox = [round(previous_bbox[index] * 0.25 + bbox[index] * 0.75, 3) for index in range(4)]
        previous_bbox = bbox
        result.append({**dict(obs), "raw_bbox": raw_bbox, "bbox": bbox, "source": obs.get("source") or "detected"})
    return result


def _interpolate_missing(observations: Sequence[Mapping[str, Any]], *, max_step_sec: float = 0.34) -> list[dict[str, Any]]:
    interpolated: list[dict[str, Any]] = []
    for left, right in zip(observations, observations[1:]):
        gap = float(right["time_sec"]) - float(left["time_sec"])
        if gap <= max_step_sec:
            continue
        steps = int(gap // max_step_sec)
        for step in range(1, steps + 1):
            ratio = min(0.95, (step * max_step_sec) / gap)
            time_sec = round(float(left["time_sec"]) + gap * ratio, 6)
            if time_sec >= float(right["time_sec"]):
                continue
            interpolated.append(
                {
                    "label": left["label"],
                    "bbox": _lerp_bbox(left["bbox"], right["bbox"], ratio),
                    "confidence": round(min(float(left["confidence"]), float(right["confidence"])) * 0.92, 4),
                    "time_sec": time_sec,
                    "view": left.get("view") or right.get("view") or "",
                    "frame_id": None,
                    "source": "interpolated",
                }
            )
    return interpolated


def _track_detection_at(track: Mapping[str, Any], time_sec: float, *, hold_sec: float) -> dict[str, Any] | None:
    points = [point for point in track.get("points") or [] if isinstance(point, Mapping)]
    if not points:
        return None
    before = max((point for point in points if float(point["time_sec"]) <= time_sec), key=lambda item: float(item["time_sec"]), default=None)
    after = min((point for point in points if float(point["time_sec"]) >= time_sec), key=lambda item: float(item["time_sec"]), default=None)
    if before and after and before is not after:
        span = max(1e-6, float(after["time_sec"]) - float(before["time_sec"]))
        ratio = (time_sec - float(before["time_sec"])) / span
        return _point_detection(track, before, bbox=_lerp_bbox(before["bbox"], after["bbox"], ratio), source="interpolated")
    nearest = before or after
    if nearest is None:
        return None
    if abs(float(nearest["time_sec"]) - time_sec) > hold_sec:
        return None
    source = str(nearest.get("source") or "")
    if source in {"detected", "interaction_bbox", "stabilized_outlier"}:
        source = "detected" if source == "interaction_bbox" else source
    else:
        source = "interpolated"
    return _point_detection(track, nearest, bbox=list(nearest["bbox"]), source=source)


def _point_detection(track: Mapping[str, Any], point: Mapping[str, Any], *, bbox: list[float], source: str) -> dict[str, Any]:
    detection = {
        "label": track["label"],
        "bbox": bbox,
        "confidence": point.get("confidence"),
        "tracklet_id": track["tracklet_id"],
        "object_track_id": track["tracklet_id"],
        "tracklet_source": source,
        "local_time_sec": point.get("time_sec"),
        "view": track.get("view"),
    }
    if point.get("stabilization"):
        detection["stabilization"] = point.get("stabilization")
    if track.get("quality"):
        detection["tracklet_quality"] = track.get("quality")
    return detection


def _canon(value: Any) -> str:
    return str(canonical_yolo_label(value) or value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(1e-6, left_area + right_area - inter)


def _center_distance_ratio(left: Sequence[float], right: Sequence[float]) -> float:
    lx = (float(left[0]) + float(left[2])) / 2.0
    ly = (float(left[1]) + float(left[3])) / 2.0
    rx = (float(right[0]) + float(right[2])) / 2.0
    ry = (float(right[1]) + float(right[3])) / 2.0
    distance = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    scale = max(1.0, ((float(left[2]) - float(left[0])) + (float(right[2]) - float(right[0]))) / 2.0)
    return distance / scale


def _center_distance_px(left: Sequence[float], right: Sequence[float]) -> float:
    lx = (float(left[0]) + float(left[2])) / 2.0
    ly = (float(left[1]) + float(left[3])) / 2.0
    rx = (float(right[0]) + float(right[2])) / 2.0
    ry = (float(right[1]) + float(right[3])) / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def _bbox_extent(bbox: Sequence[float]) -> float:
    return max(abs(float(bbox[2]) - float(bbox[0])), abs(float(bbox[3]) - float(bbox[1])), 1.0)


def _bbox_size_ratio(current: Sequence[float], previous: Sequence[float], following: Sequence[float]) -> float:
    current_size = max(1.0, _bbox_extent(current))
    neighbor_size = max(1.0, (_bbox_extent(previous) + _bbox_extent(following)) / 2.0)
    return max(current_size / neighbor_size, neighbor_size / current_size)


def _tracklet_quality(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    centers = [
        ((float(point["bbox"][0]) + float(point["bbox"][2])) / 2.0, (float(point["bbox"][1]) + float(point["bbox"][3])) / 2.0)
        for point in points
        if isinstance(point.get("bbox"), list) and len(point.get("bbox") or []) >= 4
    ]
    jumps = [math.dist(left, right) for left, right in zip(centers, centers[1:])]
    return {
        "max_center_jump_px": round(max(jumps, default=0.0), 4),
        "mean_center_jump_px": round(mean(jumps), 4) if jumps else 0.0,
        "stabilized_outlier_count": sum(1 for point in points if point.get("stabilization")),
        "point_count": len(points),
    }


def _lerp_bbox(left: Sequence[float], right: Sequence[float], ratio: float) -> list[float]:
    bounded = max(0.0, min(1.0, float(ratio)))
    return [round(float(left[index]) + (float(right[index]) - float(left[index])) * bounded, 3) for index in range(4)]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
