from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set

from .action_resolution import ActionResolver, RuleBasedActionResolver
from .class_roles import (
    CONTAINER_TERMS,
    HAND_TERMS,
    IGNORE_INTERACTION_TERMS,
    LID_TERMS,
    PANEL_TERMS,
    TOOL_TERMS,
    is_container_label,
    is_hand_label,
    is_interaction_object_label,
    is_lid_label,
    is_panel_label,
    is_tool_label,
    norm_label,
)
from .container_roles import ContainerRoleResolver
from .direction_resolution import RuleBasedDirectionResolver
from .evidence_grading import EventEvidenceGrader
from .schemas import ContainerRole, DetectionFrame, EventProposal, TrackRelation, TrackedObject, Tracklet
from .state_resolution import RuleBasedStateResolver

def _norm(value: str) -> str:
    return norm_label(value)


def _has_any(text: str, words: Iterable[str]) -> bool:
    lowered = _norm(text)
    return any(word in lowered for word in words)


class EventProposalBuilder:
    def __init__(self, min_event_duration: float = 0.8, max_gap_merge: float = 1.2, action_resolver: ActionResolver | None = None) -> None:
        self.min_event_duration = float(min_event_duration)
        self.max_gap_merge = float(max_gap_merge)
        self.container_roles = ContainerRoleResolver()
        self.action_resolver = action_resolver or RuleBasedActionResolver()
        self.direction_resolver = RuleBasedDirectionResolver()
        self.state_resolver = RuleBasedStateResolver()
        self.evidence_grader = EventEvidenceGrader()

    def build(
        self,
        frames: List[DetectionFrame],
        tracklets: Optional[List[Tracklet]] = None,
        tracked_objects: Optional[List[TrackedObject]] = None,
        track_relations: Optional[List[TrackRelation]] = None,
    ) -> List[EventProposal]:
        raw: List[EventProposal] = []
        tracklets_by_id = {track.track_id: track for track in tracklets or []}
        tracked_objects = tracked_objects or []
        track_relations = track_relations or []

        # hand_object_interaction: 按物体类别逐一建窗口，每对独立成素材
        raw.extend(self._windows_for_glove_per_object(frames, tracklets_by_id, tracked_objects, track_relations))

        for event_type in (
            "object_move",
            "liquid_transfer",
            "panel_operation",
            "container_state_change",
        ):
            raw.extend(self._windows_for_type(event_type, frames, tracklets_by_id, tracked_objects, track_relations))
        raw.extend(self._track_motion_candidate_windows(frames, tracklets_by_id, tracked_objects, raw))
        raw.sort(key=lambda item: (item.start_time_sec, item.event_type))
        return raw

    # 物体类别的中文显示名
    _OBJECT_DISPLAY_ZH: Dict[str, str] = {
        "balance":           "天平",
        "beaker":            "烧杯",
        "reagent_bottle":    "试剂瓶",
        "sample_bottle":     "样品瓶",
        "sample_bottle_blue":"蓝盖样品瓶",
        "paper":             "称量纸",
        "pipette":           "移液管",
        "spatula":           "刮刀",
        "tube":              "试管",
        "tube-cap":          "试管盖",
        "spearhead":         "枪头",
        "lab_coat":          "实验服",
    }

    def _windows_for_glove_per_object(
        self,
        frames: List[DetectionFrame],
        tracklets_by_id: Dict[str, Tracklet],
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
    ) -> List[EventProposal]:
        """为每个 gloved_hand × 非手套物体类别对，单独建时间窗口，生成独立 proposal。

        只要某帧中 gloved_hand 的 bbox 与该类别物体的 bbox 满足几何接触/接近条件
        (IoU ≥ 0.005 或边界距离 ≤ 60px)，就累积进该对的时间窗口。
        """
        # 不作为手套接触目标的类别（背景/穿戴物，不是实验操作对象）
        _SKIP_CONTACT_CLASSES = {"lab_coat", "goggles", "safety_glasses", "gloved_hand", "hand"}

        # 收集本次视频中实际出现的所有有效交互物体类别
        object_classes: Set[str] = set()
        for frame in frames:
            for det in frame.detections:
                if not is_hand_label(det.class_name) and _norm(det.class_name) not in _SKIP_CONTACT_CLASSES:
                    object_classes.add(_norm(det.class_name))

        proposals: List[EventProposal] = []
        for obj_class in sorted(object_classes):
            active: List[DetectionFrame] = []
            last_ts: Optional[float] = None
            for frame in frames:
                if self._glove_contacts_class(frame, obj_class):
                    if active and last_ts is not None and frame.timestamp_sec - last_ts > self.max_gap_merge:
                        self._append_glove_object_window(
                            active, obj_class, proposals,
                            tracklets_by_id, tracked_objects, track_relations,
                        )
                        active = []
                    active.append(frame)
                    last_ts = frame.timestamp_sec
                elif active and last_ts is not None and frame.timestamp_sec - last_ts > self.max_gap_merge:
                    self._append_glove_object_window(
                        active, obj_class, proposals,
                        tracklets_by_id, tracked_objects, track_relations,
                    )
                    active = []
                    last_ts = None
            if active:
                self._append_glove_object_window(
                    active, obj_class, proposals,
                    tracklets_by_id, tracked_objects, track_relations,
                )
        return proposals

    @staticmethod
    def _glove_contacts_class(frame: DetectionFrame, obj_class: str) -> bool:
        """Return True only when a gloved hand bbox actually overlaps the target object."""
        hands = [det for det in frame.detections if is_hand_label(det.class_name)]
        targets = [det for det in frame.detections if _norm(det.class_name) == obj_class]
        if not hands or not targets:
            return False
        for hand in hands:
            for target in targets:
                if _bbox_iou(hand.bbox, target.bbox) >= 0.02 or _bbox_edge_distance(hand.bbox, target.bbox) <= 30:
                    return True
        return False

    def _append_glove_object_window(
        self,
        active: List[DetectionFrame],
        obj_class: str,
        proposals: List[EventProposal],
        tracklets_by_id: Dict[str, Tracklet],
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
    ) -> None:
        """将一段连续帧窗口构建为 hand_object_interaction proposal。"""
        if not active:
            return
        start = active[0]
        end = active[-1]
        duration = max(0.0, end.timestamp_sec - start.timestamp_sec)
        if duration < self.min_event_duration and len(active) < 2:
            return

        # 收集 track_ids
        track_ids: List[str] = []
        for frame in active:
            for det in frame.detections:
                if det.track_id and (is_hand_label(det.class_name) or _norm(det.class_name) == obj_class):
                    track_ids.append(det.track_id)
        involved_track_ids = list(dict.fromkeys(track_ids))
        primary_track_id = self._primary_track_id(involved_track_ids, tracklets_by_id)

        # involved_objects: 手套 + 目标物体类别
        involved = list(dict.fromkeys(["gloved_hand", obj_class]))
        classes = sorted({_norm(det.class_name) for frame in active for det in frame.detections})

        action = self.action_resolver.resolve(
            event_type="hand_object_interaction",
            tracked_objects=tracked_objects,
            track_relations=track_relations,
            event_window={"start_time_sec": start.timestamp_sec, "end_time_sec": max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration)},
            semantic_context={"objects": involved, "classes": classes, "activities": []},
        )

        supporting_relation_ids = [
            rel.relation_id for rel in track_relations
            if rel.end_time_sec >= start.timestamp_sec and rel.start_time_sec <= max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration)
        ][:32]
        supporting_relations = [rel for rel in track_relations if rel.relation_id in set(supporting_relation_ids)]
        tracked_by_id = {obj.track_id: obj for obj in tracked_objects}
        motion_summary = self._track_motion_summary(involved_track_ids, tracklets_by_id)

        confidence = min(0.95, 0.50 + 0.06 * len(active) + max(
            (det.confidence for frame in active for det in frame.detections if _norm(det.class_name) == obj_class or is_hand_label(det.class_name)),
            default=0.0,
        ) * 0.25)
        confidence = self._quality_adjusted_confidence(confidence, involved_track_ids, tracked_by_id, None, None)

        evidence_grade, review_status, evidence_summary = self.evidence_grader.grade(
            event_type="hand_object_interaction",
            confidence=confidence,
            related_tracks=involved_track_ids,
            tracked_objects_by_id=tracked_by_id,
            supporting_relations=supporting_relations,
            direction_confidence=None,
            direction_status=None,
            state_confidence=None,
            missing_critical_fields=[],
        )

        # 中文显示名：「{实验名}-手套接触{物体中文名}」，由 EventSegmenter 读 involved_objects[1] 生成
        obj_zh = self._OBJECT_DISPLAY_ZH.get(obj_class, obj_class)

        key = f"hand_object_interaction:{obj_class}:{start.frame_idx}:{end.frame_idx}"
        proposal_id = "proposal_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        proposals.append(
            EventProposal(
                proposal_id=proposal_id,
                event_type="hand_object_interaction",
                start_frame_idx=start.frame_idx,
                end_frame_idx=end.frame_idx,
                start_time_sec=start.timestamp_sec,
                end_time_sec=max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration),
                evidence_frame_indices=[frame.frame_idx for frame in active],
                involved_objects=involved,
                dominant_object=obj_class,
                involved_track_ids=involved_track_ids,
                primary_track_id=primary_track_id,
                source_container=None,
                target_container=self._container_role_from_action(action.target_container),
                track_motion_summary=motion_summary,
                actor_track_id=action.actor_track_id,
                tool_track_id=None,
                related_tracks=action.related_tracks or involved_track_ids,
                transfer_mode=None,
                action_resolution_source="glove_per_object_contact",
                action_resolution_notes=f"gloved_hand↔{obj_class}({obj_zh}); frames={len(active)}",
                supporting_relation_ids=supporting_relation_ids,
                direction_confidence=None,
                direction_status=None,
                direction_evidence=[],
                state_before=None,
                state_after=None,
                state_change_type=None,
                state_confidence=None,
                state_evidence=[],
                evidence_grade=evidence_grade,
                review_status=review_status,
                evidence_summary=evidence_summary,
                related_detection_classes=classes,
                confidence=round(confidence, 4),
                notes=f"glove_contact_target={obj_class}; window_frames={len(active)}",
                contact_target_class=obj_class,
                contact_target_zh=obj_zh,
            )
        )

    def _windows_for_type(
        self,
        event_type: str,
        frames: List[DetectionFrame],
        tracklets_by_id: Dict[str, Tracklet],
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
    ) -> List[EventProposal]:
        active: List[DetectionFrame] = []
        proposals: List[EventProposal] = []
        last_ts: Optional[float] = None
        for frame in frames:
            if self._matches(event_type, frame, track_relations):
                if active and last_ts is not None and frame.timestamp_sec - last_ts > self.max_gap_merge:
                    self._append_window(event_type, active, proposals, tracklets_by_id, tracked_objects, track_relations)
                    active = []
                active.append(frame)
                last_ts = frame.timestamp_sec
            elif active and last_ts is not None and frame.timestamp_sec - last_ts > self.max_gap_merge:
                self._append_window(event_type, active, proposals, tracklets_by_id, tracked_objects, track_relations)
                active = []
                last_ts = None
        if active:
            self._append_window(event_type, active, proposals, tracklets_by_id, tracked_objects, track_relations)
        return proposals

    def _track_motion_candidate_windows(
        self,
        frames: List[DetectionFrame],
        tracklets_by_id: Dict[str, Tracklet],
        tracked_objects: List[TrackedObject],
        existing: List[EventProposal],
    ) -> List[EventProposal]:
        """Create weak object_move candidates from stable tracklets before the hard gate.

        This is a recall-only fallback: it never confirms movement.  The
        physical_event_gate still rejects bbox jitter, pseudo tracks, low
        identity, camera motion, and change_score-only cases.
        """
        used_tracks = {
            track_id
            for proposal in existing
            if proposal.event_type == "object_move"
            for track_id in proposal.involved_track_ids
        }
        frames_by_idx = {frame.frame_idx: frame for frame in frames}
        proposals: List[EventProposal] = []
        for track in tracklets_by_id.values():
            label = _norm(track.class_name)
            if track.track_id in used_tracks or is_hand_label(label) or not is_interaction_object_label(label):
                continue
            if len(track.frame_indices) < 2:
                continue
            if float(track.displacement_px or 0.0) < 6.0:
                continue
            active_frames = [frames_by_idx[idx] for idx in track.frame_indices if idx in frames_by_idx]
            if not active_frames:
                continue
            start = active_frames[0]
            end = active_frames[-1]
            if max(0.0, end.timestamp_sec - start.timestamp_sec) < self.min_event_duration and len(active_frames) < 2:
                continue
            key = f"object_move_track_candidate:{track.track_id}:{start.frame_idx}:{end.frame_idx}"
            proposal_id = "proposal_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
            tracked_by_id = {obj.track_id: obj for obj in tracked_objects}
            confidence = self._quality_adjusted_confidence(
                min(0.72, 0.35 + min(float(track.displacement_px or 0.0) / 80.0, 0.25) + float(track.mean_confidence or 0.0) * 0.15),
                [track.track_id],
                tracked_by_id,
                None,
                None,
            )
            proposals.append(
                EventProposal(
                    proposal_id=proposal_id,
                    event_type="object_move",
                    start_frame_idx=start.frame_idx,
                    end_frame_idx=end.frame_idx,
                    start_time_sec=start.timestamp_sec,
                    end_time_sec=max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration),
                    evidence_frame_indices=[frame.frame_idx for frame in active_frames],
                    involved_objects=[label],
                    dominant_object=label,
                    involved_track_ids=[track.track_id],
                    primary_track_id=track.track_id,
                    source_container=None,
                    target_container=None,
                    track_motion_summary=self._track_motion_summary([track.track_id], tracklets_by_id),
                    actor_track_id=None,
                    tool_track_id=None,
                    related_tracks=[track.track_id],
                    transfer_mode=None,
                    action_resolution_source="track_motion_candidate",
                    action_resolution_notes=f"weak object_move candidate from tracklet {track.track_id}; gate required",
                    supporting_relation_ids=[],
                    direction_confidence=None,
                    direction_status=None,
                    direction_evidence=[],
                    state_before=None,
                    state_after=None,
                    state_change_type=None,
                    state_confidence=None,
                    state_evidence=[],
                    evidence_grade="weak",
                    review_status="candidate_review",
                    evidence_summary="tracklet displacement cue; physical_event_gate required before confirmation",
                    related_detection_classes=[label],
                    confidence=round(confidence, 4),
                    proposal_source="track_motion_candidate_recall",
                    notes=f"recall_candidate_only; displacement_px={float(track.displacement_px or 0.0):.3f}",
                )
            )
        return proposals

    def _matches(self, event_type: str, frame: DetectionFrame, track_relations: List[TrackRelation]) -> bool:
        classes = {_norm(det.class_name) for det in frame.detections}
        text = " ".join([frame.scene_description, *frame.semantic_activities, *frame.semantic_objects]).lower()
        rel_types = {
            rel.relation_type
            for rel in track_relations
            if rel.start_time_sec <= frame.timestamp_sec <= rel.end_time_sec or abs(rel.start_time_sec - frame.timestamp_sec) <= 0.75
        }
        has_hand = any(_has_any(cls, HAND_TERMS) for cls in classes) or _has_any(text, HAND_TERMS)
        has_container = any(_has_any(cls, CONTAINER_TERMS) for cls in classes) or _has_any(text, CONTAINER_TERMS)
        if event_type == "hand_object_interaction":
            return (
                "glove_object_interaction" in rel_types
                or self._has_glove_object_geometry(frame)
            )
        if event_type == "object_move":
            return (
                "carry" in rel_types
                or ("object_manipulation" in rel_types and has_hand)
                or self._has_moving_glove_object_pair(frame)
            )
        if event_type == "liquid_transfer":
            return (
                "transfer_posture" in rel_types and has_hand
            ) or self._has_glove_transfer_geometry(frame)
        if event_type == "panel_operation":
            has_panel = any(_has_any(cls, PANEL_TERMS) for cls in classes) or _has_any(text, PANEL_TERMS)
            return "panel_interaction" in rel_types or self._has_glove_panel_geometry(frame) or (has_hand and has_panel)
        if event_type == "container_state_change":
            return (
                ("container_state_interaction" in rel_types and has_hand)
                or self._has_glove_container_state_geometry(frame)
            )
        return False

    def _append_window(
        self,
        event_type: str,
        active: List[DetectionFrame],
        proposals: List[EventProposal],
        tracklets_by_id: Dict[str, Tracklet],
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
    ) -> None:
        if not active:
            return
        start = active[0]
        end = active[-1]
        duration = max(0.0, end.timestamp_sec - start.timestamp_sec)
        if duration < self.min_event_duration and len(active) < 2:
            return
        classes: List[str] = []
        objects: List[str] = []
        track_ids: List[str] = []
        for frame in active:
            for det in frame.detections:
                label = _norm(det.class_name)
                if label and label not in IGNORE_INTERACTION_TERMS:
                    classes.append(label)
                if det.track_id:
                    track_ids.append(det.track_id)
            objects.extend(_norm(item) for item in frame.semantic_objects if item and _norm(item) not in IGNORE_INTERACTION_TERMS)
        counts = Counter([item for item in [*classes, *objects] if item])
        involved = [name for name, _ in counts.most_common(8)] or [event_type]
        dominant = involved[0] if involved else None
        involved_track_ids = list(dict.fromkeys(track_ids))
        primary_track_id = self._primary_track_id(involved_track_ids, tracklets_by_id)
        action = self.action_resolver.resolve(
            event_type=event_type,
            tracked_objects=tracked_objects,
            track_relations=track_relations,
            event_window={"start_time_sec": start.timestamp_sec, "end_time_sec": max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration)},
            semantic_context={
                "objects": involved,
                "classes": sorted(set(classes)),
                "activities": list(dict.fromkeys(item for frame in active for item in frame.semantic_activities)),
            },
        )
        source_container = self._container_role_from_action(action.source_container)
        target_container = self._container_role_from_action(action.target_container)
        if source_container is None and target_container is None:
            source_container, target_container = self.container_roles.resolve(
            event_type=event_type,
            involved_objects=involved,
            involved_track_ids=involved_track_ids,
            tracklets_by_id=tracklets_by_id,
            frames=active,
            )
        motion_summary = self._track_motion_summary(involved_track_ids, tracklets_by_id)
        supporting_relation_ids = [
            rel.relation_id
            for rel in track_relations
            if rel.end_time_sec >= start.timestamp_sec and rel.start_time_sec <= max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration)
        ][:32]
        supporting_relations = [rel for rel in track_relations if rel.relation_id in set(supporting_relation_ids)]
        tracked_by_id = {obj.track_id: obj for obj in tracked_objects}
        direction = self.direction_resolver.resolve(
            source_container=action.source_container,
            target_container=action.target_container,
            actor_track_id=action.actor_track_id,
            tool_track_id=action.tool_track_id,
            tracked_objects=tracked_objects,
            track_relations=supporting_relations,
            semantic_context={
                "objects": involved,
                "classes": sorted(set(classes)),
                "activities": list(dict.fromkeys(item for frame in active for item in frame.semantic_activities)),
            },
        ) if event_type == "liquid_transfer" else None
        state = None
        if event_type == "container_state_change":
            state_track_id = (action.source_container or {}).get("track_id") if action.source_container else primary_track_id
            state = self.state_resolver.resolve(
                tracked_object=tracked_by_id.get(str(state_track_id)) if state_track_id else None,
                related_track_relations=supporting_relations,
                semantic_summary={
                    "objects": involved,
                    "classes": sorted(set(classes)),
                    "activities": list(dict.fromkeys(item for frame in active for item in frame.semantic_activities)),
                    "description": " ".join(frame.scene_description for frame in active if frame.scene_description),
                },
            )
        missing = self._missing_critical_fields(event_type, action.actor_track_id, action.source_container, action.target_container, direction)
        confidence = min(0.95, 0.45 + 0.08 * len(active) + max((det.confidence for frame in active for det in frame.detections), default=0.0) * 0.25)
        confidence = max(confidence, action.confidence)
        confidence = self._quality_adjusted_confidence(confidence, action.related_tracks or involved_track_ids, tracked_by_id, direction.direction_confidence if direction else None, state.confidence if state else None)
        evidence_grade, review_status, evidence_summary = self.evidence_grader.grade(
            event_type=event_type,
            confidence=confidence,
            related_tracks=action.related_tracks or involved_track_ids,
            tracked_objects_by_id=tracked_by_id,
            supporting_relations=supporting_relations,
            direction_confidence=direction.direction_confidence if direction else None,
            direction_status=direction.direction_status if direction else None,
            state_confidence=state.confidence if state else None,
            missing_critical_fields=missing,
        )
        key = f"{event_type}:{start.frame_idx}:{end.frame_idx}:{','.join(involved[:3])}"
        proposal_id = "proposal_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        proposals.append(
            EventProposal(
                proposal_id=proposal_id,
                event_type=event_type,
                start_frame_idx=start.frame_idx,
                end_frame_idx=end.frame_idx,
                start_time_sec=start.timestamp_sec,
                end_time_sec=max(end.timestamp_sec, start.timestamp_sec + self.min_event_duration),
                evidence_frame_indices=[frame.frame_idx for frame in active],
                involved_objects=involved,
                dominant_object=dominant,
                involved_track_ids=involved_track_ids,
                primary_track_id=primary_track_id,
                source_container=source_container,
                target_container=target_container,
                track_motion_summary=motion_summary,
                actor_track_id=action.actor_track_id,
                tool_track_id=action.tool_track_id,
                related_tracks=action.related_tracks or involved_track_ids,
                transfer_mode=action.transfer_mode,
                action_resolution_source=action.action_resolution_source,
                action_resolution_notes=action.action_resolution_notes,
                supporting_relation_ids=supporting_relation_ids,
                direction_confidence=direction.direction_confidence if direction else None,
                direction_status=direction.direction_status if direction else None,
                direction_evidence=direction.direction_evidence if direction else [],
                state_before=state.state_before if state else None,
                state_after=state.state_after if state else None,
                state_change_type=state.state_change_type if state else None,
                state_confidence=state.confidence if state else None,
                state_evidence=state.evidence if state else [],
                evidence_grade=evidence_grade,
                review_status=review_status,
                evidence_summary=evidence_summary,
                related_detection_classes=sorted(set(classes)),
                confidence=round(confidence, 4),
                notes=f"window_size={len(active)}; change_max={max((frame.change_score for frame in active), default=0.0):.4f}",
            )
        )

    @staticmethod
    def _has_glove_object_geometry(frame: DetectionFrame) -> bool:
        hands = [det for det in frame.detections if is_hand_label(det.class_name)]
        objects = [det for det in frame.detections if is_interaction_object_label(det.class_name)]
        for hand in hands:
            for obj in objects:
                if _bbox_iou(hand.bbox, obj.bbox) >= 0.02 or _bbox_edge_distance(hand.bbox, obj.bbox) <= 20:
                    return True
        return False

    @staticmethod
    def _has_moving_glove_object_pair(frame: DetectionFrame) -> bool:
        if frame.change_score < 0.018:
            return False
        return EventProposalBuilder._has_glove_object_geometry(frame)

    @staticmethod
    def _has_glove_transfer_geometry(frame: DetectionFrame) -> bool:
        if not any(is_hand_label(det.class_name) for det in frame.detections):
            return False
        has_tool = any(is_tool_label(det.class_name) for det in frame.detections)
        containers = [det for det in frame.detections if is_container_label(det.class_name)]
        if has_tool and containers:
            return EventProposalBuilder._has_glove_object_geometry(frame)
        return len(containers) >= 2 and EventProposalBuilder._has_glove_object_geometry(frame)

    @staticmethod
    def _has_glove_panel_geometry(frame: DetectionFrame) -> bool:
        hands = [det for det in frame.detections if is_hand_label(det.class_name)]
        panels = [det for det in frame.detections if is_panel_label(det.class_name)]
        return any(_bbox_iou(hand.bbox, panel.bbox) >= 0.02 or _bbox_edge_distance(hand.bbox, panel.bbox) <= 25 for hand in hands for panel in panels)

    @staticmethod
    def _has_glove_container_state_geometry(frame: DetectionFrame) -> bool:
        hands = [det for det in frame.detections if is_hand_label(det.class_name)]
        targets = [det for det in frame.detections if is_lid_label(det.class_name) or is_container_label(det.class_name)]
        return any(_bbox_iou(hand.bbox, target.bbox) >= 0.02 or _bbox_edge_distance(hand.bbox, target.bbox) <= 25 for hand in hands for target in targets)

    @staticmethod
    def _missing_critical_fields(
        event_type: str,
        actor_track_id: Optional[str],
        source_container: Optional[Dict[str, object]],
        target_container: Optional[Dict[str, object]],
        direction,
    ) -> List[str]:
        missing: List[str] = []
        if event_type in {"hand_object_interaction", "object_move", "liquid_transfer", "panel_operation"} and not actor_track_id:
            missing.append("actor_track_id")
        if event_type == "liquid_transfer":
            if not source_container:
                missing.append("source_container")
            if not target_container:
                missing.append("target_container")
            if direction and direction.direction_status == "unknown":
                missing.append("direction_unknown")
        return missing

    @staticmethod
    def _quality_adjusted_confidence(
        base: float,
        related_tracks: List[str],
        tracked_by_id: Dict[str, TrackedObject],
        direction_confidence: Optional[float],
        state_confidence: Optional[float],
    ) -> float:
        tracks = [tracked_by_id[track_id] for track_id in related_tracks if track_id in tracked_by_id]
        if tracks:
            avg_quality = sum(track.track_confidence * (1.0 - track.id_switch_risk * 0.25) for track in tracks) / len(tracks)
            base = base * 0.72 + avg_quality * 0.28
        if direction_confidence is not None:
            base = base * 0.82 + direction_confidence * 0.18
        if state_confidence is not None:
            base = base * 0.86 + state_confidence * 0.14
        return round(max(0.0, min(0.95, base)), 4)

    @staticmethod
    def _container_role_from_action(payload: Optional[Dict[str, object]]) -> Optional[ContainerRole]:
        if not payload:
            return None
        bbox = payload.get("bbox") if isinstance(payload, dict) else None
        return ContainerRole(
            object_name=str(payload.get("display_name") or payload.get("class_name") or payload.get("object_name") or ""),
            track_id=str(payload.get("track_id")) if payload.get("track_id") else None,
            role_confidence=float(payload.get("confidence") or payload.get("role_confidence") or 0.0),
            role_source="action_resolver",
            bbox=tuple(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None,
        )

    @staticmethod
    def _primary_track_id(track_ids: List[str], tracklets_by_id: Dict[str, Tracklet]) -> Optional[str]:
        candidates = [tracklets_by_id[item] for item in track_ids if item in tracklets_by_id]
        if not candidates:
            return track_ids[0] if track_ids else None
        candidates.sort(key=lambda track: (track.displacement_px, len(track.frame_indices), track.mean_confidence), reverse=True)
        return candidates[0].track_id

    @staticmethod
    def _track_motion_summary(track_ids: List[str], tracklets_by_id: Dict[str, Tracklet]) -> Dict[str, object]:
        tracks = [tracklets_by_id[item] for item in track_ids if item in tracklets_by_id]
        if not tracks:
            return {"track_count": 0, "max_displacement_px": 0.0, "moving_track_ids": []}
        moving = [track.track_id for track in tracks if track.displacement_px >= 12.0]
        return {
            "track_count": len(tracks),
            "max_displacement_px": max(track.displacement_px for track in tracks),
            "moving_track_ids": moving,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "class_name": track.class_name,
                    "start_time_sec": track.start_time_sec,
                    "end_time_sec": track.end_time_sec,
                    "displacement_px": track.displacement_px,
                    "mean_confidence": track.mean_confidence,
                }
                for track in tracks[:12]
            ],
        }


def _bbox_edge_distance(a, b) -> float:
    import math

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)


def _bbox_iou(a, b) -> float:
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
    return inter / denom if denom > 0 else 0.0
