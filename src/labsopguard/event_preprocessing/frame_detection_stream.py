from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .activity_presegmenter import ActivityPreSegmenter, ActivitySegment, PresegmentConfig
from .detection_cache import DetectionCache, compute_cache_key
from .schemas import DetectionBox, DetectionFrame, Tracklet
from .tracking.providers import TrackingProvider, build_tracking_provider

logger = logging.getLogger(__name__)


class DetectionFrameStreamBuilder:
    """Builds the private frame-level perception stream used by event extraction."""

    def __init__(
        self,
        pipeline: Any,
        interval_sec: float = 0.5,
        max_frames: int = 360,
        tracking_provider: Optional[TrackingProvider] = None,
        presegment_config: Optional[PresegmentConfig] = None,
        cache_dir: Optional[Path] = None,
        batch_size: int = 8,
    ) -> None:
        self.pipeline = pipeline
        self.interval_sec = max(0.2, float(interval_sec or 0.5))
        self.max_frames = max(1, int(max_frames or 360))
        self.tracking_provider = tracking_provider or build_tracking_provider()
        self.presegment_config = presegment_config or PresegmentConfig()
        self.presegmenter = ActivityPreSegmenter(self.presegment_config)
        self.cache: Optional[DetectionCache] = DetectionCache(cache_dir) if cache_dir else None
        self.batch_size = max(1, batch_size)
        self._last_segments: List[ActivitySegment] = []

    @property
    def last_presegment_result(self) -> List[ActivitySegment]:
        return self._last_segments

    def build(
        self,
        video_path: str | Path,
        semantic_analyses: Optional[List[Dict[str, Any]]] = None,
        material_stream: Optional[List[Dict[str, Any]]] = None,
        time_range: Optional[tuple] = None,
    ) -> tuple[List[DetectionFrame], List[Tracklet]]:
        video_path = Path(video_path)
        semantic_by_time = self._semantic_snapshots(semantic_analyses or [], material_stream or [])

        # --- Try cache ---
        cache_key = None
        if self.cache:
            settings = getattr(self.pipeline, "settings", None)
            cache_key = compute_cache_key(
                video_path,
                getattr(settings, "yolo_model_path", None) if settings else None,
                getattr(settings, "yolo_imgsz", 960) if settings else 960,
                getattr(settings, "confidence_threshold", 0.25) if settings else 0.25,
                self.interval_sec,
                extra_components=self._cache_extra_components(time_range),
            )
            cached_frames = self.cache.load(cache_key)
            if cached_frames is not None:
                self._apply_semantic_snapshots(cached_frames, semantic_by_time)
                tracklets = self.tracking_provider.track(cached_frames)
                return cached_frames, tracklets

        # --- Open video ---
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps else 0.0

        # --- Layer 0: Pre-segmentation ---
        frame_indices = self._compute_frame_indices(
            video_path, fps, total_frames, duration, time_range=time_range
        )

        frames: List[DetectionFrame] = []
        previous_gray = None

        # --- Batch YOLO inference ---
        batch_buffer: List[tuple] = []  # (frame_idx, ts, frame_array)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ts = frame_idx / fps if fps else min(duration, frame_idx * self.interval_sec)
            batch_buffer.append((frame_idx, ts, frame))

            if len(batch_buffer) >= self.batch_size:
                self._process_batch(batch_buffer, frames, semantic_by_time, previous_gray)
                if frames:
                    previous_gray = cv2.cvtColor(
                        batch_buffer[-1][2], cv2.COLOR_BGR2GRAY
                    )
                    previous_gray = cv2.resize(previous_gray, (160, 120))
                batch_buffer = []

        # Process remaining frames
        if batch_buffer:
            self._process_batch(batch_buffer, frames, semantic_by_time, previous_gray)

        cap.release()

        # --- Save cache ---
        if self.cache and cache_key and frames:
            self.cache.save(cache_key, frames, metadata={
                "video_path": str(video_path),
                "duration_sec": duration,
                "interval_sec": self.interval_sec,
            })

        tracklets = self.tracking_provider.track(frames)
        return frames, tracklets

    def _compute_frame_indices(
        self, video_path: Path, fps: float, total_frames: int, duration: float,
        time_range: Optional[tuple] = None,
    ) -> List[int]:
        """Compute frame indices, optionally filtered by presegmentation and time_range."""
        # Apply time_range constraint first
        range_start = 0.0
        range_end = duration
        if time_range:
            range_start = max(0.0, float(time_range[0]))
            range_end = min(duration, float(time_range[1]))

        should_presegment = (
            self.presegment_config.enabled
            and duration >= self.presegment_config.skip_if_video_shorter_than
        )
        if should_presegment:
            segments = self.presegmenter.segment(video_path)
            # Filter segments to only those overlapping with time_range
            clipped_segments = [
                ActivitySegment(
                    start_sec=max(seg.start_sec, range_start),
                    end_sec=min(seg.end_sec, range_end),
                    peak_score=seg.peak_score,
                    avg_score=seg.avg_score,
                    trigger=seg.trigger,
                    stream_id=seg.stream_id,
                )
                for seg in segments
                if seg.end_sec > range_start and seg.start_sec < range_end
            ]
            clipped_segments = [seg for seg in clipped_segments if seg.end_sec > seg.start_sec]
            self._last_segments = clipped_segments
            active_ranges = [(seg.start_sec, seg.end_sec) for seg in clipped_segments]
        else:
            self._last_segments = []
            active_ranges = [(range_start, range_end)]

        step = max(1, int(round(fps * self.interval_sec)))
        frame_indices: List[int] = []
        required_indices = set()

        for start_sec, end_sec in active_ranges:
            start_frame = int(start_sec * fps)
            end_frame = min(int(math.ceil(end_sec * fps)), total_frames)
            if end_frame <= start_frame:
                continue
            for idx in range(start_frame, end_frame, step):
                frame_indices.append(idx)
            required_indices.add(min(end_frame - 1, total_frames - 1))

        # Deduplicate and sort
        frame_indices = sorted(set(frame_indices))

        # Apply max_frames limit
        if len(frame_indices) > self.max_frames:
            stride = math.ceil(len(frame_indices) / self.max_frames)
            frame_indices = frame_indices[::stride]
            frame_indices.extend(idx for idx in required_indices if idx not in frame_indices)
            frame_indices = sorted(set(frame_indices))

        logger.info(
            "Frame selection: %d frames from %.1fs video (presegment: %s, active_ranges: %d)",
            len(frame_indices),
            duration,
            "enabled" if self._last_segments else "disabled",
            len(active_ranges),
        )
        return frame_indices

    def _cache_extra_components(self, time_range: Optional[tuple]) -> Dict[str, Any]:
        settings = getattr(self.pipeline, "settings", None)
        labels = getattr(settings, "allowed_detection_labels", []) if settings else []
        return {
            "time_range": self._normalize_time_range(time_range),
            "max_frames": self.max_frames,
            "presegment": {
                "enabled": self.presegment_config.enabled,
                "scan_fps": self.presegment_config.scan_fps,
                "scan_resolution": list(self.presegment_config.scan_resolution),
                "motion_threshold_mode": self.presegment_config.motion_threshold_mode,
                "motion_fixed_threshold": self.presegment_config.motion_fixed_threshold,
                "min_segment_sec": self.presegment_config.min_segment_sec,
                "merge_gap_sec": self.presegment_config.merge_gap_sec,
                "padding_sec": self.presegment_config.padding_sec,
                "skip_if_video_shorter_than": self.presegment_config.skip_if_video_shorter_than,
                "forced_sample_interval_sec": self.presegment_config.forced_sample_interval_sec,
            },
            "yolo": {
                "iou_threshold": getattr(settings, "iou_threshold", None) if settings else None,
                "max_detections": getattr(settings, "max_detections", None) if settings else None,
                "allowed_labels": sorted(str(item) for item in labels),
            },
        }

    @staticmethod
    def _normalize_time_range(time_range: Optional[tuple]) -> Optional[tuple[float, float]]:
        if not time_range:
            return None
        return (round(float(time_range[0]), 3), round(float(time_range[1]), 3))

    def _process_batch(
        self,
        batch: List[tuple],
        frames: List[DetectionFrame],
        semantic_by_time: List[Dict[str, Any]],
        prev_gray: Optional[np.ndarray],
    ) -> None:
        """Process a batch of frames through YOLO and build DetectionFrames."""
        # Try batch inference if pipeline supports it
        batch_detections = self._run_yolo_batch(batch)

        for i, (frame_idx, ts, frame_array) in enumerate(batch):
            detections = []
            for det in batch_detections[i]:
                detections.append(
                    DetectionBox(
                        bbox=tuple(int(v) for v in det.bbox),
                        class_name=str(det.class_name),
                        confidence=float(det.confidence),
                    )
                )

            gray = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (160, 120))
            change_score = 0.0
            if prev_gray is not None:
                change_score = float(cv2.absdiff(prev_gray, small_gray).mean() / 255.0)
            prev_gray = small_gray

            snapshot = self._nearest_snapshot(ts, semantic_by_time)
            frames.append(
                DetectionFrame(
                    frame_idx=frame_idx,
                    timestamp_sec=round(ts, 3),
                    detections=detections,
                    semantic_activities=list(snapshot.get("activities", [])),
                    semantic_objects=list(snapshot.get("objects", [])),
                    scene_description=str(snapshot.get("description", "")),
                    change_score=round(change_score, 4),
                )
            )

    def _run_yolo_batch(self, batch: List[tuple]) -> List[list]:
        """Run YOLO on a batch of frames. Falls back to sequential if batch not supported."""
        frames_array = [item[2] for item in batch]
        frame_indices = [item[0] for item in batch]
        timestamps = [item[1] for item in batch]

        # Try batch inference via pipeline
        if hasattr(self.pipeline, "_run_yolo_batch"):
            try:
                return self.pipeline._run_yolo_batch(frames_array, frame_indices, timestamps)
            except (TypeError, AttributeError):
                pass

        # Fallback: sequential inference
        results = []
        for frame_array, frame_idx, ts in zip(frames_array, frame_indices, timestamps):
            dets = self.pipeline._run_yolo(frame_array, frame_idx, ts)
            results.append(dets)
        return results

    @staticmethod
    def _semantic_snapshots(semantic_analyses: List[Dict[str, Any]], material_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        snapshots: List[Dict[str, Any]] = []
        for item in semantic_analyses:
            snapshots.append(
                {
                    "timestamp_sec": float(item.get("timestamp_sec") or 0.0),
                    "activities": item.get("detected_activities") or [],
                    "objects": item.get("object_labels") or [],
                    "description": item.get("scene_description") or item.get("description") or "",
                }
            )
        for item in material_stream:
            snapshots.append(
                {
                    "timestamp_sec": float(item.get("timestamp_sec") or 0.0),
                    "activities": item.get("detected_activities") or [],
                    "objects": item.get("object_labels") or item.get("detected_objects") or [],
                    "description": item.get("scene_description") or "",
                }
            )
        snapshots.sort(key=lambda row: row["timestamp_sec"])
        return snapshots

    @staticmethod
    def _nearest_snapshot(ts: float, snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not snapshots:
            return {}
        return min(snapshots, key=lambda row: abs(float(row.get("timestamp_sec") or 0.0) - ts))

    def _apply_semantic_snapshots(
        self,
        frames: List[DetectionFrame],
        snapshots: List[Dict[str, Any]],
    ) -> None:
        """Refresh cached detection frames with the current semantic context."""
        for frame in frames:
            snapshot = self._nearest_snapshot(frame.timestamp_sec, snapshots)
            frame.semantic_activities = list(snapshot.get("activities", []))
            frame.semantic_objects = list(snapshot.get("objects", []))
            frame.scene_description = str(snapshot.get("description", ""))
