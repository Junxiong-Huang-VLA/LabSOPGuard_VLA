from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from labsopguard.event_preprocessing.schemas import DetectionBox, DetectionFrame, Tracklet

BBox = Tuple[int, int, int, int]


def center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


@dataclass
class _LiveTrack:
    track_id: str
    class_name: str
    last_bbox: BBox
    last_frame_idx: int
    last_time_sec: float
    missed: int = 0
    frame_indices: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    bboxes: List[BBox] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)

    def append(self, frame: DetectionFrame, det: DetectionBox) -> None:
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
            fragment_count=1,
            recovered_from_fragment=False,
            tracking_backend="iou_baseline",
            tracking_backend_version="1.0",
        )


class IouMultiObjectTracker:
    """Dependency-free MOT baseline behind the tracker interface."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 3) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._next_id = 1
        self._live: Dict[str, _LiveTrack] = {}
        self._finished: List[_LiveTrack] = []

    def apply(self, frames: List[DetectionFrame]) -> List[Tracklet]:
        for frame in frames:
            self._update(frame)
        self._finished.extend(self._live.values())
        self._live = {}
        return [track.to_tracklet() for track in self._finished if track.frame_indices]

    def _update(self, frame: DetectionFrame) -> None:
        unmatched_detections = list(range(len(frame.detections)))
        unmatched_tracks = set(self._live.keys())
        candidates: List[Tuple[float, str, int]] = []
        for track_id, track in self._live.items():
            for det_idx, det in enumerate(frame.detections):
                if track.class_name != det.class_name:
                    continue
                score = iou(track.last_bbox, det.bbox)
                if score >= self.iou_threshold:
                    candidates.append((score, track_id, det_idx))
        candidates.sort(reverse=True, key=lambda item: item[0])

        used_tracks = set()
        used_dets = set()
        for _, track_id, det_idx in candidates:
            if track_id in used_tracks or det_idx in used_dets:
                continue
            self._live[track_id].append(frame, frame.detections[det_idx])
            used_tracks.add(track_id)
            used_dets.add(det_idx)
            unmatched_tracks.discard(track_id)
            if det_idx in unmatched_detections:
                unmatched_detections.remove(det_idx)

        for track_id in list(unmatched_tracks):
            track = self._live[track_id]
            track.missed += 1
            if track.missed > self.max_missed:
                self._finished.append(track)
                del self._live[track_id]

        for det_idx in unmatched_detections:
            det = frame.detections[det_idx]
            track_id = f"trk_{self._next_id:05d}"
            self._next_id += 1
            track = _LiveTrack(track_id, det.class_name, det.bbox, frame.frame_idx, frame.timestamp_sec)
            self._live[track_id] = track
            track.append(frame, det)

        frame.active_track_ids = [det.track_id for det in frame.detections if det.track_id]
