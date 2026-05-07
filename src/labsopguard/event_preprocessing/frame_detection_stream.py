from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .schemas import DetectionBox, DetectionFrame, Tracklet
from .tracking.providers import TrackingProvider, build_tracking_provider


class DetectionFrameStreamBuilder:
    """Builds the private frame-level perception stream used by event extraction."""

    def __init__(
        self,
        pipeline: Any,
        interval_sec: float = 0.5,
        max_frames: int = 360,
        tracking_provider: Optional[TrackingProvider] = None,
    ) -> None:
        self.pipeline = pipeline
        self.interval_sec = max(0.2, float(interval_sec or 0.5))
        self.max_frames = max(1, int(max_frames or 360))
        self.tracking_provider = tracking_provider or build_tracking_provider()

    def build(
        self,
        video_path: str | Path,
        semantic_analyses: Optional[List[Dict[str, Any]]] = None,
        material_stream: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[List[DetectionFrame], List[Tracklet]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps else 0.0
        step = max(1, int(round(fps * self.interval_sec)))
        frame_indices = list(range(0, max(total_frames, 1), step))
        if total_frames > 0 and (total_frames - 1) not in frame_indices:
            frame_indices.append(total_frames - 1)
        if len(frame_indices) > self.max_frames:
            stride = math.ceil(len(frame_indices) / self.max_frames)
            frame_indices = frame_indices[::stride]
            if total_frames > 0 and (total_frames - 1) not in frame_indices:
                frame_indices.append(total_frames - 1)

        semantic_by_time = self._semantic_snapshots(semantic_analyses or [], material_stream or [])
        frames: List[DetectionFrame] = []
        previous_gray = None
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            ts = frame_idx / fps if fps else min(duration, frame_idx * self.interval_sec)
            detections = []
            for det in self.pipeline._run_yolo(frame, frame_idx, ts):
                detections.append(
                    DetectionBox(
                        bbox=tuple(int(v) for v in det.bbox),
                        class_name=str(det.class_name),
                        confidence=float(det.confidence),
                    )
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            change_score = 0.0
            if previous_gray is not None:
                resized = cv2.resize(gray, (previous_gray.shape[1], previous_gray.shape[0]))
                change_score = float(cv2.absdiff(previous_gray, resized).mean() / 255.0)
            previous_gray = gray
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
        cap.release()
        tracklets = self.tracking_provider.track(frames)
        return frames, tracklets

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
