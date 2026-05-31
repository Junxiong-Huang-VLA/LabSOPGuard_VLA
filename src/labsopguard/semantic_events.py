from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


BBox = Tuple[float, float, float, float]


def _bbox_from_detection(detection: Dict[str, Any]) -> Optional[BBox]:
    bbox = detection.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    return x1, y1, x2, y2


def _center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _distance(a: BBox, b: BBox) -> float:
    ca = _center(a)
    cb = _center(b)
    return float(np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2))


@dataclass
class ObjectTrack:
    track_id: str
    label: str
    bbox: BBox
    first_seen_sec: float
    last_seen_sec: float
    observations: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[float, float]:
        return _center(self.bbox)


class StableObjectTracker:
    def __init__(self, iou_threshold: float = 0.2, max_center_distance: float = 80.0, max_missing_sec: float = 5.0) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_missing_sec = max_missing_sec
        self._next_id = 1
        self._tracks: Dict[str, ObjectTrack] = {}

    def update(self, detections: Sequence[Dict[str, Any]], timestamp_sec: float) -> List[ObjectTrack]:
        active: List[ObjectTrack] = []
        used_track_ids: set[str] = set()
        for detection in detections:
            bbox = _bbox_from_detection(detection)
            if bbox is None:
                continue
            label = str(detection.get("label") or detection.get("object_type") or detection.get("name") or "object").lower()
            match = self._match(label, bbox, timestamp_sec, used_track_ids)
            if match is None:
                match = ObjectTrack(
                    track_id=f"obj_{self._next_id:05d}",
                    label=label,
                    bbox=bbox,
                    first_seen_sec=timestamp_sec,
                    last_seen_sec=timestamp_sec,
                    metadata={"score": detection.get("score", detection.get("confidence"))},
                )
                self._next_id += 1
                self._tracks[match.track_id] = match
            else:
                match.bbox = bbox
                match.last_seen_sec = timestamp_sec
                match.observations += 1
                match.metadata.update({"score": detection.get("score", detection.get("confidence"))})
            detection["stable_object_id"] = match.track_id
            active.append(match)
            used_track_ids.add(match.track_id)
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp_sec - track.last_seen_sec <= self.max_missing_sec
        }
        return active

    def _match(self, label: str, bbox: BBox, timestamp_sec: float, used_track_ids: set[str]) -> Optional[ObjectTrack]:
        best: Optional[ObjectTrack] = None
        best_score = -1.0
        for track in self._tracks.values():
            if track.track_id in used_track_ids or track.label != label:
                continue
            if timestamp_sec - track.last_seen_sec > self.max_missing_sec:
                continue
            overlap = _iou(track.bbox, bbox)
            distance = _distance(track.bbox, bbox)
            if overlap < self.iou_threshold and distance > self.max_center_distance:
                continue
            score = overlap + max(0.0, 1.0 - distance / self.max_center_distance)
            if score > best_score:
                best = track
                best_score = score
        return best


class SemanticEventDetector:
    CONTAINER_LABELS = {"beaker", "bottle", "tube", "flask", "container", "vial", "cylinder"}
    CAP_LABELS = {"cap", "lid", "stopper", "cover"}
    HAND_LABELS = {"hand", "glove", "left_hand", "right_hand"}
    REAGENT_LABELS = {"label", "reagent_label", "bottle_label"}
    THRESHOLDS = {
        "object_move_displacement_px": 15.0,
        "hand_contact_iou": 0.02,
        "hand_contact_distance_px": 40.0,
        "hand_contact_depth_delta": 0.08,
        "container_cap_distance_px": 70.0,
        "liquid_level_delta": 0.08,
        "reagent_label_min_chars": 2,
    }

    def __init__(self) -> None:
        self.tracker = StableObjectTracker()
        self._prev_tracks: Dict[str, ObjectTrack] = {}
        self._container_state: Dict[str, str] = {}
        self._liquid_level: Dict[str, float] = {}
        self._label_state: Dict[str, str] = {}

    def update(
        self,
        timestamp_sec: float,
        detections: Sequence[Dict[str, Any]],
        frame_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        metadata = dict(frame_metadata or {})
        normalized = [dict(item) for item in detections if isinstance(item, dict)]
        tracks = self.tracker.update(normalized, timestamp_sec)
        by_track = {track.track_id: track for track in tracks}
        events: List[Dict[str, Any]] = []
        events.extend(self._detect_motion(timestamp_sec, tracks, metadata))
        events.extend(self._detect_hand_contact(timestamp_sec, normalized, metadata))
        events.extend(self._detect_container_state(timestamp_sec, tracks, metadata))
        events.extend(self._detect_liquid_level(timestamp_sec, normalized, by_track, metadata))
        events.extend(self._detect_reagent_label(timestamp_sec, normalized, by_track, metadata))
        self._prev_tracks = {track.track_id: ObjectTrack(**track.__dict__) for track in tracks}
        return events

    def _event(self, event_type: str, timestamp_sec: float, confidence: float, metadata: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_type": event_type,
            "timestamp_sec": round(float(timestamp_sec), 3),
            "confidence": round(max(0.0, min(1.0, float(confidence))), 4),
            "metadata": metadata,
        }

    def _detect_motion(self, timestamp_sec: float, tracks: Sequence[ObjectTrack], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for track in tracks:
            previous = self._prev_tracks.get(track.track_id)
            if previous is None:
                continue
            displacement = _distance(previous.bbox, track.bbox)
            threshold = self.THRESHOLDS["object_move_displacement_px"]
            if displacement >= threshold:
                events.append(
                    self._event(
                        "object_move",
                        timestamp_sec,
                        min(0.95, displacement / 80.0),
                        {
                            **metadata,
                            "stable_object_id": track.track_id,
                            "label": track.label,
                            "displacement_px": round(displacement, 3),
                            "threshold_px": threshold,
                            "rule": "bbox_center_displacement",
                        },
                    )
                )
        return events

    def _detect_hand_contact(self, timestamp_sec: float, detections: Sequence[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        hands = [det for det in detections if self._canonical_label(det) in self.HAND_LABELS and _bbox_from_detection(det)]
        objects = [det for det in detections if self._canonical_label(det) not in self.HAND_LABELS and _bbox_from_detection(det)]
        events: List[Dict[str, Any]] = []
        for hand in hands:
            hand_bbox = _bbox_from_detection(hand)
            if hand_bbox is None:
                continue
            for obj in objects:
                obj_bbox = _bbox_from_detection(obj)
                if obj_bbox is None:
                    continue
                overlap = _iou(hand_bbox, obj_bbox)
                distance = _distance(hand_bbox, obj_bbox)
                iou_threshold = self.THRESHOLDS["hand_contact_iou"]
                distance_threshold = self.THRESHOLDS["hand_contact_distance_px"]
                hand_points = self._hand_keypoints(hand)
                keypoint_hits = [point for point in hand_points if self._point_in_bbox(point, obj_bbox)]
                depth_delta = self._depth_delta(hand, obj)
                depth_threshold = self.THRESHOLDS["hand_contact_depth_delta"]
                depth_verified = depth_delta is not None and depth_delta <= depth_threshold
                contact_verified = (
                    overlap >= iou_threshold
                    or distance <= distance_threshold
                    or bool(keypoint_hits)
                    or depth_verified
                )
                if contact_verified:
                    confidence = max(
                        overlap,
                        max(0.0, 1.0 - distance / 80.0),
                        0.85 if keypoint_hits else 0.0,
                        0.8 if depth_verified else 0.0,
                    )
                    events.append(
                        self._event(
                            "hand_contact_geometry",
                            timestamp_sec,
                            confidence,
                            {
                                **metadata,
                                "hand_label": self._canonical_label(hand),
                                "object_label": self._canonical_label(obj),
                                "hand_id": hand.get("stable_object_id"),
                                "object_id": obj.get("stable_object_id"),
                                "iou": round(overlap, 4),
                                "distance_px": round(distance, 3),
                                "iou_threshold": iou_threshold,
                                "distance_threshold_px": distance_threshold,
                                "bbox_contact_verified": overlap >= iou_threshold or distance <= distance_threshold,
                                "keypoint_contact_verified": bool(keypoint_hits),
                                "keypoint_hit_count": len(keypoint_hits),
                                "depth_contact_verified": depth_verified,
                                "depth_delta": round(depth_delta, 4) if depth_delta is not None else None,
                                "depth_threshold": depth_threshold,
                                "requires_depth_for_3d": not depth_verified,
                                "rule": "bbox_or_keypoint_or_depth_geometry",
                            },
                        )
                    )
        return events

    def _detect_container_state(self, timestamp_sec: float, tracks: Sequence[ObjectTrack], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        containers = [track for track in tracks if self._label_key(track.label) in self.CONTAINER_LABELS]
        caps = [track for track in tracks if self._label_key(track.label) in self.CAP_LABELS]
        events: List[Dict[str, Any]] = []
        for container in containers:
            distance_threshold = self.THRESHOLDS["container_cap_distance_px"]
            closed = any(_distance(container.bbox, cap.bbox) <= distance_threshold for cap in caps)
            state = "closed" if closed else "open"
            previous = self._container_state.get(container.track_id)
            if previous and previous != state:
                events.append(
                    self._event(
                        "container_closed" if state == "closed" else "container_opened",
                        timestamp_sec,
                        0.72,
                        {
                            **metadata,
                            "stable_object_id": container.track_id,
                            "container_label": container.label,
                            "previous_state": previous,
                            "state": state,
                            "cap_distance_threshold_px": distance_threshold,
                            "rule": "cap_container_center_distance",
                        },
                    )
                )
            self._container_state[container.track_id] = state
        return events

    def _detect_liquid_level(
        self,
        timestamp_sec: float,
        detections: Sequence[Dict[str, Any]],
        by_track: Dict[str, ObjectTrack],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for det in detections:
            object_id = det.get("stable_object_id")
            if not object_id:
                continue
            level = self._as_float(det.get("liquid_level") or det.get("liquid_level_ratio") or (det.get("metadata") or {}).get("liquid_level"))
            if level is None and self._label_key(self._canonical_label(det)) == "liquid":
                bbox = _bbox_from_detection(det)
                track = by_track.get(str(object_id))
                if bbox and track:
                    level = max(0.0, min(1.0, (bbox[3] - bbox[1]) / max(1.0, track.bbox[3] - track.bbox[1])))
            if level is None:
                continue
            previous = self._liquid_level.get(str(object_id))
            threshold = self.THRESHOLDS["liquid_level_delta"]
            if previous is not None and abs(level - previous) >= threshold:
                events.append(
                    self._event(
                        "liquid_level_change",
                        timestamp_sec,
                        min(0.9, abs(level - previous) * 4.0),
                        {
                            **metadata,
                            "stable_object_id": object_id,
                            "previous_level": round(previous, 4),
                            "current_level": round(level, 4),
                            "delta": round(abs(level - previous), 4),
                            "threshold": threshold,
                            "rule": "liquid_level_ratio_delta",
                        },
                    )
                )
            self._liquid_level[str(object_id)] = float(level)
        return events

    def _detect_reagent_label(
        self,
        timestamp_sec: float,
        detections: Sequence[Dict[str, Any]],
        by_track: Dict[str, ObjectTrack],
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for det in detections:
            label_key = self._label_key(self._canonical_label(det))
            text = str(det.get("ocr_text") or det.get("text") or (det.get("metadata") or {}).get("ocr_text") or "").strip()
            if label_key not in self.REAGENT_LABELS and not text:
                continue
            object_id = str(det.get("stable_object_id") or "")
            min_chars = int(self.THRESHOLDS["reagent_label_min_chars"])
            state = "verified" if len(text) >= min_chars else "unreadable"
            previous = self._label_state.get(object_id)
            if previous != state:
                events.append(
                    self._event(
                        "reagent_label_state",
                        timestamp_sec,
                        0.85 if state == "verified" else 0.45,
                        {
                            **metadata,
                            "stable_object_id": object_id,
                            "state": state,
                            "ocr_text": text,
                            "min_chars": min_chars,
                            "rule": "ocr_text_length",
                        },
                    )
                )
            self._label_state[object_id] = state
        return events

    @staticmethod
    def _canonical_label(detection: Dict[str, Any]) -> str:
        return str(detection.get("label") or detection.get("object_type") or detection.get("name") or "object").lower()

    @staticmethod
    def _label_key(label: str) -> str:
        return str(label).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _hand_keypoints(detection: Dict[str, Any]) -> List[tuple[float, float]]:
        raw = detection.get("hand_keypoints") or detection.get("keypoints") or (detection.get("metadata") or {}).get("hand_keypoints")
        points: List[tuple[float, float]] = []
        if not isinstance(raw, (list, tuple)):
            return points
        for item in raw:
            if isinstance(item, dict):
                x = item.get("x")
                y = item.get("y")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                x, y = item[0], item[1]
            else:
                continue
            try:
                points.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue
        return points

    @staticmethod
    def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
        return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

    @classmethod
    def _depth_delta(cls, hand: Dict[str, Any], obj: Dict[str, Any]) -> Optional[float]:
        hand_depth = cls._as_float(hand.get("depth") or hand.get("median_depth") or (hand.get("metadata") or {}).get("depth"))
        object_depth = cls._as_float(obj.get("depth") or obj.get("median_depth") or (obj.get("metadata") or {}).get("depth"))
        if hand_depth is None or object_depth is None:
            return None
        return abs(hand_depth - object_depth)
