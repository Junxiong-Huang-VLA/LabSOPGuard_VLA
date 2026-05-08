from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DetectorConfig
from .schemas import DetectedSegment, FrameScore, VideoSource, WorkbenchROI, write_jsonl
from .time_alignment import local_sec_to_global_time
from .video_utils import default_roi, get_video_duration_sec


@dataclass
class RawInterval:
    start_sec: float
    end_sec: float
    start_reason: str
    end_reason: str


def _segment_id(index: int) -> str:
    return f"seg_{index:06d}"


def _roi_dict(roi_tuple: tuple[int, int, int, int] | None) -> dict[str, int] | None:
    if roi_tuple is None:
        return None
    x, y, w, h = roi_tuple
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def _resolved_roi_tuple(frame_width: int, frame_height: int, roi: WorkbenchROI | None) -> tuple[int, int, int, int]:
    if roi is None:
        return default_roi(frame_width, frame_height)
    x = max(0, min(int(roi.x), frame_width - 1))
    y = max(0, min(int(roi.y), frame_height - 1))
    w = max(1, min(int(roi.w), frame_width - x))
    h = max(1, min(int(roi.h), frame_height - y))
    return x, y, w, h


def mock_frame_scores(
    video_source: VideoSource | None = None,
    duration_sec: float = 960.0,
    roi: WorkbenchROI | None = None,
    config: DetectorConfig | None = None,
) -> list[FrameScore]:
    config = config or DetectorConfig()
    scores: list[FrameScore] = []
    active_windows = [(598.0, 610.0, 0.78), (618.0, 632.0, 0.86), (898.0, 912.0, 0.8)]
    roi_info = _roi_dict(_resolved_roi_tuple(1280, 720, roi))
    source = video_source or VideoSource("third_person", "dry_run.mp4", "2026-04-29T17:25:00+08:00")
    for time_sec in np.arange(0.0, duration_sec + 0.5, 0.5):
        score = 0.05
        for start, end, active in active_windows:
            if start <= time_sec <= end:
                score = active
                break
        scores.append(
            FrameScore(
                time_sec=float(time_sec),
                frame_index=int(round(float(time_sec) * float(source.fps or 30.0))),
                local_time_sec=float(time_sec),
                global_time=local_sec_to_global_time(source, float(time_sec)).isoformat(),
                motion_score=score,
                active_score=score,
                roi=roi_info,
                is_active=bool(score > config.start_threshold),
            )
        )
    return scores


def _avg_score(scores: list[FrameScore], start_sec: float, end_sec: float, field: str) -> float:
    values = [getattr(score, field) for score in scores if start_sec <= score.time_sec <= end_sec]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _detect_raw_intervals(scores: list[FrameScore], duration_sec: float, config: DetectorConfig) -> list[RawInterval]:
    intervals: list[RawInterval] = []
    in_segment = False
    active_since: float | None = None
    inactive_since: float | None = None
    segment_start = 0.0

    for score in scores:
        t = score.time_sec
        active = score.active_score
        if not in_segment:
            if active > config.start_threshold:
                if active_since is None:
                    active_since = t
                if t - active_since >= config.start_min_duration_sec:
                    in_segment = True
                    segment_start = active_since
                    inactive_since = None
            else:
                active_since = None
            continue

        if active < config.end_threshold:
            if inactive_since is None:
                inactive_since = t
            if t - inactive_since >= config.end_min_duration_sec:
                intervals.append(
                    RawInterval(
                        start_sec=segment_start,
                        end_sec=inactive_since,
                        start_reason="active_score_above_threshold",
                        end_reason="active_score_below_threshold",
                    )
                )
                in_segment = False
                active_since = None
                inactive_since = None
        else:
            inactive_since = None

    if in_segment:
        intervals.append(
            RawInterval(
                start_sec=segment_start,
                end_sec=duration_sec,
                start_reason="active_score_above_threshold",
                end_reason="video_end",
            )
        )
    return intervals


def _merge_intervals(intervals: list[RawInterval], merge_gap_sec: float) -> list[RawInterval]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: item.start_sec)
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start_sec - previous.end_sec <= merge_gap_sec:
            previous.end_sec = max(previous.end_sec, current.end_sec)
            previous.end_reason = current.end_reason
        else:
            merged.append(current)
    return merged


def build_segments_from_scores(
    scores: list[FrameScore],
    video_source: VideoSource,
    duration_sec: float,
    config: DetectorConfig | None = None,
) -> list[DetectedSegment]:
    config = config or DetectorConfig()
    raw = _detect_raw_intervals(scores, duration_sec=duration_sec, config=config)
    merged = _merge_intervals(raw, merge_gap_sec=config.merge_gap_sec)
    segments: list[DetectedSegment] = []
    for interval in merged:
        raw_duration = interval.end_sec - interval.start_sec
        if raw_duration < config.min_segment_duration_sec:
            continue
        start_sec = max(0.0, interval.start_sec - config.buffer_sec)
        end_sec = min(duration_sec, interval.end_sec + config.buffer_sec)
        if end_sec <= start_sec:
            continue
        global_start = local_sec_to_global_time(video_source, start_sec)
        global_end = local_sec_to_global_time(video_source, end_sec)
        segment = DetectedSegment(
            segment_id=_segment_id(len(segments) + 1),
            start_sec=float(start_sec),
            end_sec=float(end_sec),
            duration_sec=float(end_sec - start_sec),
            global_start_time=global_start.isoformat(),
            global_end_time=global_end.isoformat(),
            avg_motion_score=_avg_score(scores, start_sec, end_sec, "motion_score"),
            avg_active_score=_avg_score(scores, start_sec, end_sec, "active_score"),
            start_reason=interval.start_reason,
            end_reason=interval.end_reason,
            review_required=False,
        )
        segments.append(segment)
    return segments


def _crop_roi(frame: np.ndarray, roi: WorkbenchROI | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height, width = frame.shape[:2]
    x, y, w, h = _resolved_roi_tuple(width, height, roi)
    return frame[y : y + h, x : x + w], (x, y, w, h)


def compute_frame_scores(
    video_source: VideoSource,
    roi: WorkbenchROI | None = None,
    config: DetectorConfig | None = None,
) -> tuple[list[FrameScore], float]:
    config = config or DetectorConfig()
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for real video detection") from exc

    video_path = video_source.path
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration_sec = frame_count / fps if fps > 0 else get_video_duration_sec(video_path)
        sample_every = max(1, int(round(fps / config.sample_fps)))
        scores: list[FrameScore] = []
        raw_motion_scores: list[float] = []
        previous_gray = None
        frame_index = 0
        resolved_roi = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % sample_every != 0:
                frame_index += 1
                continue
            roi_frame, resolved_roi = _crop_roi(frame, roi)
            gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if previous_gray is None:
                raw_motion_score = 0.0
            else:
                diff = cv2.absdiff(gray, previous_gray)
                raw_motion_score = float(np.mean(diff) / 255.0 * 6.0)
            motion_score = min(1.0, raw_motion_score)
            raw_motion_scores.append(raw_motion_score)
            time_sec = frame_index / fps
            scores.append(
                FrameScore(
                    time_sec=float(time_sec),
                    frame_index=int(frame_index),
                    local_time_sec=float(time_sec),
                    global_time=local_sec_to_global_time(video_source, float(time_sec)).isoformat(),
                    motion_score=motion_score,
                    active_score=motion_score,
                    roi=_roi_dict(resolved_roi),
                    is_active=bool(motion_score > config.start_threshold),
                )
            )
            previous_gray = gray
            frame_index += 1
        if scores and str(config.motion_normalization).lower() == "adaptive":
            values = np.asarray(raw_motion_scores, dtype=float)
            low = float(np.percentile(values, 20))
            high = float(np.percentile(values, 95))
            scale = max(1e-6, high - low)
            normalized = np.clip((values - low) / scale, 0.0, 1.0)
            for score, value in zip(scores, normalized):
                score.motion_score = float(value)
                score.active_score = float(value)
                score.is_active = bool(value > config.start_threshold)
        return scores, float(duration_sec)
    finally:
        cap.release()


def detect_key_action_segments(
    video_source: VideoSource,
    roi: WorkbenchROI | None = None,
    config: DetectorConfig | None = None,
    dry_run: bool = False,
    frame_scores_output_path: str | Path | None = None,
) -> tuple[list[DetectedSegment], list[FrameScore]]:
    config = config or DetectorConfig()
    if dry_run:
        scores = mock_frame_scores(video_source=video_source, roi=roi, config=config)
        duration_sec = scores[-1].time_sec if scores else 960.0
    else:
        scores, duration_sec = compute_frame_scores(video_source, roi=roi, config=config)
    if frame_scores_output_path is not None:
        write_jsonl(frame_scores_output_path, scores)
    segments = build_segments_from_scores(scores, video_source=video_source, duration_sec=duration_sec, config=config)
    return segments, scores
