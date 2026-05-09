from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Tuple

from labsopguard.event_preprocessing.schemas import DetectionFrame, Tracklet

from labsopguard.event_preprocessing.class_roles import (
    is_container_label,
    is_hand_label,
    is_lid_label,
    is_panel_label,
    is_tool_label,
)
from .multi_object_tracker import center, iou
from .schemas import TrackRelation, TrackedObject


class TrackStreamBuilder:
    def build(
        self,
        *,
        experiment_id: str,
        source_video_id: str,
        tracklets: List[Tracklet],
    ) -> List[TrackedObject]:
        objects: List[TrackedObject] = []
        for track in tracklets:
            centroids = [center(bbox) for bbox in track.bboxes]
            velocities = []
            for prev, cur in zip(centroids, centroids[1:]):
                velocities.append(math.dist(prev, cur))
            confidences = [track.mean_confidence]
            expected_samples = self._expected_samples(track.start_frame_idx, track.end_frame_idx, track.frame_indices)
            observed_samples = len(set(track.frame_indices))
            missing_samples = max(0, expected_samples - observed_samples)
            occlusion_ratio = round(missing_samples / expected_samples, 4) if expected_samples else 0.0
            fragment_count = max(1, int(getattr(track, "fragment_count", 1) or 1), self._fragment_count(track.frame_indices))
            recovered = bool(getattr(track, "recovered_from_fragment", False) or fragment_count > 1)
            id_switch_risk = self._id_switch_risk(track, occlusion_ratio, fragment_count)
            track_confidence = self._track_confidence(track.mean_confidence, occlusion_ratio, id_switch_risk, fragment_count)
            objects.append(
                TrackedObject(
                    track_id=track.track_id,
                    experiment_id=experiment_id,
                    source_video_id=source_video_id,
                    class_name=track.class_name,
                    display_name=track.class_name,
                    start_time_sec=round(track.start_time_sec, 3),
                    end_time_sec=round(track.end_time_sec, 3),
                    frame_indices=list(track.frame_indices),
                    bboxes=list(track.bboxes),
                    centroids=centroids,
                    confidence_stats={
                        "mean": round(track.mean_confidence, 4),
                        "min": round(min(confidences), 4),
                        "max": round(max(confidences), 4),
                    },
                    velocity_stats={
                        "mean_px_per_sample": round(sum(velocities) / len(velocities), 3) if velocities else 0.0,
                        "max_px_per_sample": round(max(velocities), 3) if velocities else 0.0,
                        "displacement_px": round(track.displacement_px, 3),
                    },
                    track_confidence=track_confidence,
                    id_switch_risk=id_switch_risk,
                    occlusion_ratio=occlusion_ratio,
                    fragment_count=fragment_count,
                    recovered_from_fragment=recovered,
                    tracking_backend=track.tracking_backend,
                    tracking_backend_version=track.tracking_backend_version,
                    state_labels=self._state_labels(track.class_name, track.displacement_px),
                )
            )
        return objects

    @staticmethod
    def _expected_samples(start_frame_idx: int, end_frame_idx: int, frame_indices: List[int]) -> int:
        if not frame_indices:
            return 0
        if len(frame_indices) <= 1:
            return 1
        diffs = [b - a for a, b in zip(sorted(frame_indices), sorted(frame_indices)[1:]) if b > a]
        step = max(1, int(round(sum(diffs) / len(diffs)))) if diffs else 1
        return max(1, int((end_frame_idx - start_frame_idx) / step) + 1)

    @staticmethod
    def _fragment_count(frame_indices: List[int]) -> int:
        if len(frame_indices) <= 2:
            return 1
        ordered = sorted(frame_indices)
        diffs = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
        if not diffs:
            return 1
        typical = max(1, int(round(sum(diffs) / len(diffs))))
        return 1 + sum(1 for diff in diffs if diff > typical * 2)

    @staticmethod
    def _id_switch_risk(track: Tracklet, occlusion_ratio: float, fragment_count: int) -> float:
        duration_samples = len(track.frame_indices)
        short_track_penalty = 0.25 if duration_samples < 3 else 0.0
        fragment_penalty = min(0.4, max(0, fragment_count - 1) * 0.18)
        backend_penalty = 0.12 if str(track.tracking_backend).startswith("iou_baseline") else 0.05
        risk = short_track_penalty + fragment_penalty + occlusion_ratio * 0.45 + backend_penalty
        return round(min(1.0, risk), 4)

    @staticmethod
    def _track_confidence(mean_confidence: float, occlusion_ratio: float, id_switch_risk: float, fragment_count: int) -> float:
        score = float(mean_confidence or 0.0) * 0.7 + (1.0 - occlusion_ratio) * 0.15 + (1.0 - id_switch_risk) * 0.15
        if fragment_count > 1:
            score -= min(0.2, 0.05 * (fragment_count - 1))
        return round(max(0.0, min(1.0, score)), 4)

    @staticmethod
    def _state_labels(class_name: str, displacement_px: float) -> List[str]:
        labels = []
        if is_hand_label(class_name):
            labels.append("actor_candidate")
        if is_container_label(class_name):
            labels.append("container_candidate")
        if is_tool_label(class_name):
            labels.append("tool_candidate")
        if is_panel_label(class_name):
            labels.append("panel_candidate")
        if is_lid_label(class_name):
            labels.append("lid_or_cap_candidate")
        labels.append("moving" if displacement_px >= 12.0 else "stationary")
        return labels


class TrackRelationGraphBuilder:
    def build(self, tracked_objects: List[TrackedObject]) -> List[TrackRelation]:
        relations: List[TrackRelation] = []
        for i, subject in enumerate(tracked_objects):
            for obj in tracked_objects[i + 1 :]:
                overlap_start = max(subject.start_time_sec, obj.start_time_sec)
                overlap_end = min(subject.end_time_sec, obj.end_time_sec)
                if overlap_end < overlap_start:
                    continue
                features = self._pair_features(subject, obj)
                if not features:
                    continue
                for relation_type, confidence in self._relation_types(subject, obj, features):
                    relation_id = f"rel_{relation_type}_{subject.track_id}_{obj.track_id}_{int(overlap_start * 1000)}"
                    relations.append(
                        TrackRelation(
                            relation_id=relation_id,
                            relation_type=relation_type,
                            subject_track_id=subject.track_id,
                            object_track_id=obj.track_id,
                            start_time_sec=round(overlap_start, 3),
                            end_time_sec=round(overlap_end, 3),
                            confidence=round(confidence, 4),
                            relation_features=features,
                        )
                    )
        return relations

    @staticmethod
    def _pair_features(subject: TrackedObject, obj: TrackedObject) -> Dict[str, float]:
        if not subject.centroids or not obj.centroids:
            return {}
        distances = []
        edge_distances = []
        ious = []
        shared_frames = sorted(set(subject.frame_indices) & set(obj.frame_indices))
        subject_by_frame = dict(zip(subject.frame_indices, subject.bboxes))
        obj_by_frame = dict(zip(obj.frame_indices, obj.bboxes))
        for frame_idx in shared_frames:
            s_box = subject_by_frame.get(frame_idx)
            o_box = obj_by_frame.get(frame_idx)
            if s_box is None or o_box is None:
                continue
            distances.append(math.dist(center(s_box), center(o_box)))
            edge_distances.append(_bbox_edge_distance(s_box, o_box))
            ious.append(iou(s_box, o_box))
        if not distances:
            for s, o in zip(subject.centroids, obj.centroids):
                distances.append(math.dist(s, o))
        if not distances:
            return {}
        return {
            "min_centroid_distance_px": round(min(distances), 3),
            "mean_centroid_distance_px": round(sum(distances) / len(distances), 3),
            "min_bbox_edge_distance_px": round(min(edge_distances), 3) if edge_distances else 99999.0,
            "max_bbox_iou": round(max(ious), 4) if ious else 0.0,
            "contact_frame_count": float(sum(1 for value in edge_distances if value <= 32.0) + sum(1 for value in ious if value >= 0.01)),
            "shared_frame_count": float(len(shared_frames)),
            "subject_displacement_px": float(subject.velocity_stats.get("displacement_px") or 0.0),
            "object_displacement_px": float(obj.velocity_stats.get("displacement_px") or 0.0),
            "overlap_duration_sec": round(max(0.0, min(subject.end_time_sec, obj.end_time_sec) - max(subject.start_time_sec, obj.start_time_sec)), 3),
        }

    @staticmethod
    def _relation_types(subject: TrackedObject, obj: TrackedObject, features: Dict[str, float]) -> List[Tuple[str, float]]:
        out: List[Tuple[str, float]] = []
        min_dist = float(features.get("min_centroid_distance_px") or 99999.0)
        mean_dist = float(features.get("mean_centroid_distance_px") or 99999.0)
        edge_dist = float(features.get("min_bbox_edge_distance_px") or 99999.0)
        max_iou = float(features.get("max_bbox_iou") or 0.0)
        contact_frames = float(features.get("contact_frame_count") or 0.0)
        subject_is_hand = is_hand_label(subject.class_name)
        object_is_hand = is_hand_label(obj.class_name)
        subject_is_panel = is_panel_label(subject.class_name)
        object_is_panel = is_panel_label(obj.class_name)
        subject_is_container = is_container_label(subject.class_name)
        object_is_container = is_container_label(obj.class_name)
        subject_is_tool = is_tool_label(subject.class_name)
        object_is_tool = is_tool_label(obj.class_name)
        subject_is_lid = is_lid_label(subject.class_name)
        object_is_lid = is_lid_label(obj.class_name)
        moving = max(float(features.get("subject_displacement_px") or 0.0), float(features.get("object_displacement_px") or 0.0))

        geometric_proximity = min_dist <= 80 or edge_dist <= 50 or max_iou >= 0.01
        geometric_contact = min_dist <= 45 or edge_dist <= 20 or max_iou >= 0.02 or contact_frames >= 3
        if geometric_proximity:
            out.append(("proximity", max(0.3, 1.0 - min(min_dist, edge_dist) / 220.0, max_iou)))
        if geometric_contact:
            out.append(("contact", max(0.35, 1.0 - min(min_dist, edge_dist) / 130.0, max_iou + 0.35)))
        if (subject_is_hand or object_is_hand) and not (subject_is_hand and object_is_hand) and geometric_contact:
            out.append(("glove_object_interaction", max(0.55, 0.6 + min(0.25, contact_frames * 0.05) + min(0.15, max_iou))))
            out.append(("object_manipulation", max(0.5, 0.58 + min(0.2, moving / 200.0))))
        if (subject_is_hand or object_is_hand) and moving >= 12 and geometric_contact:
            out.append(("carry", 0.68 if geometric_contact else 0.55))
        if (subject_is_hand or object_is_hand) and (subject_is_lid or object_is_lid or subject_is_container or object_is_container) and geometric_contact:
            out.append(("container_state_interaction", 0.62 if (subject_is_lid or object_is_lid) else 0.48))
        if subject_is_container and object_is_container and mean_dist <= 200:
            out.append(("transfer_posture", 0.55))
        if (subject_is_tool or object_is_tool) and (subject_is_container or object_is_container) and geometric_contact:
            out.append(("transfer_posture", 0.62))
        if (subject_is_hand or object_is_hand) and (subject_is_panel or object_is_panel) and geometric_contact:
            out.append(("panel_interaction", 0.7))
        return out


def _bbox_edge_distance(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)
