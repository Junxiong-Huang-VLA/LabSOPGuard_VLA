from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from labsopguard.event_preprocessing.schemas import DetectionBox, DetectionFrame, Tracklet
from labsopguard.event_preprocessing.tracking.multi_object_tracker import center, iou

from .base import TrackingBackendInfo, TrackingProvider

BBox = Tuple[int, int, int, int]


@dataclass
class _MotionTrack:
    track_id: str
    class_name: str
    last_bbox: BBox
    last_frame_idx: int
    last_time_sec: float
    velocity_xy: Tuple[float, float] = (0.0, 0.0)
    missed: int = 0
    frame_indices: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    fragment_count: int = 1
    recovered_from_fragment: bool = False

    def predicted_bbox(self, frame: DetectionFrame) -> BBox:
        dt = max(1.0, frame.timestamp_sec - self.last_time_sec)
        dx, dy = self.velocity_xy[0] * dt, self.velocity_xy[1] * dt
        x1, y1, x2, y2 = self.last_bbox
        return (int(round(x1 + dx)), int(round(y1 + dy)), int(round(x2 + dx)), int(round(y2 + dy)))

    def append(self, frame: DetectionFrame, det: DetectionBox) -> None:
        prev_center = center(self.last_bbox)
        new_center = center(det.bbox)
        dt = max(1e-3, frame.timestamp_sec - self.last_time_sec)
        measured_velocity = ((new_center[0] - prev_center[0]) / dt, (new_center[1] - prev_center[1]) / dt)
        self.velocity_xy = (
            self.velocity_xy[0] * 0.55 + measured_velocity[0] * 0.45,
            self.velocity_xy[1] * 0.55 + measured_velocity[1] * 0.45,
        )
        if self.missed:
            self.fragment_count += 1
            self.recovered_from_fragment = True
        self.last_bbox = det.bbox
        self.last_frame_idx = frame.frame_idx
        self.last_time_sec = frame.timestamp_sec
        self.missed = 0
        self.frame_indices.append(frame.frame_idx)
        self.timestamps.append(frame.timestamp_sec)
        self.bboxes.append(det.bbox)
        self.confidences.append(float(det.confidence))
        det.track_id = self.track_id

    def to_tracklet(self) -> Tracklet:
        first_center = center(self.bboxes[0]) if self.bboxes else (0.0, 0.0)
        last_center = center(self.bboxes[-1]) if self.bboxes else first_center
        return Tracklet(
            track_id=self.track_id,
            class_name=self.class_name,
            start_frame_idx=self.frame_indices[0] if self.frame_indices else self.last_frame_idx,
            end_frame_idx=self.frame_indices[-1] if self.frame_indices else self.last_frame_idx,
            start_time_sec=self.timestamps[0] if self.timestamps else self.last_time_sec,
            end_time_sec=self.timestamps[-1] if self.timestamps else self.last_time_sec,
            frame_indices=list(self.frame_indices),
            bboxes=list(self.bboxes),
            mean_confidence=round(sum(self.confidences) / len(self.confidences), 4) if self.confidences else 0.0,
            displacement_px=round(math.dist(first_center, last_center), 3),
            fragment_count=max(1, self.fragment_count),
            recovered_from_fragment=self.recovered_from_fragment,
            tracking_backend="strong_sort_lite",
            tracking_backend_version="1.0",
        )


class StrongSortLiteTrackingProvider(TrackingProvider):
    """Dependency-free MOT provider using motion prediction plus detection geometry.

    It is not a replacement for ByteTrack/BoT-SORT on production footage, but it is
    materially stronger than pure IoU matching for fast object movement and short
    occlusions because tracks can be recovered by predicted center proximity.
    """

    def __init__(self, score_threshold: float = 0.28, max_missed: int = 6, distance_gate_px: float = 180.0) -> None:
        self.score_threshold = float(score_threshold)
        self.max_missed = int(max_missed)
        self.distance_gate_px = float(distance_gate_px)
        self.backend_info = TrackingBackendInfo(
            name="strong_sort_lite",
            version="1.0",
            available=True,
            notes="Dependency-free motion-aware tracker; use ByteTrack/BoT-SORT when installed for production MOT.",
        )
        self._next_id = 1
        self._live: Dict[str, _MotionTrack] = {}
        self._finished: List[_MotionTrack] = []

    def track(self, frames: List[DetectionFrame]) -> List[Tracklet]:
        self._next_id = 1
        self._live = {}
        self._finished = []
        for frame in frames:
            self._update(frame)
        self._finished.extend(self._live.values())
        self._live = {}
        return [track.to_tracklet() for track in self._finished if track.frame_indices]

    def _update(self, frame: DetectionFrame) -> None:
        unmatched_dets = set(range(len(frame.detections)))
        unmatched_tracks = set(self._live.keys())
        candidates: List[Tuple[float, str, int]] = []
        for track_id, track in self._live.items():
            predicted = track.predicted_bbox(frame)
            predicted_center = center(predicted)
            for det_idx, det in enumerate(frame.detections):
                if track.class_name != det.class_name:
                    continue
                det_center = center(det.bbox)
                dist = math.dist(predicted_center, det_center)
                if dist > self.distance_gate_px:
                    continue
                iou_score = max(iou(track.last_bbox, det.bbox), iou(predicted, det.bbox))
                distance_score = max(0.0, 1.0 - dist / self.distance_gate_px)
                score = iou_score * 0.50 + distance_score * 0.35 + float(det.confidence) * 0.15
                if score >= self.score_threshold:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True, key=lambda item: item[0])

        used_tracks: set[str] = set()
        used_dets: set[int] = set()
        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            self._live[track_id].append(frame, frame.detections[det_idx])
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            unmatched_tracks.discard(track_id)
            unmatched_dets.discard(det_idx)

        for track_id in list(unmatched_tracks):
            track = self._live[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                self._finished.append(track)
                del self._live[track_id]

        for det_idx in sorted(unmatched_dets):
            det = frame.detections[det_idx]
            track_id = f"trk_{self._next_id:05d}"
            self._next_id += 1
            track = _MotionTrack(track_id, det.class_name, det.bbox, frame.frame_idx, frame.timestamp_sec)
            self._live[track_id] = track
            track.append(frame, det)

        frame.active_track_ids = [det.track_id for det in frame.detections if det.track_id]
