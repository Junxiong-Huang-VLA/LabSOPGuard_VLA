from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ActionResolvedEvent, ActionResolver
from labsopguard.event_preprocessing.class_roles import (
    is_container_label,
    is_hand_label,
    is_interaction_object_label,
    is_lid_label,
    is_panel_label,
    is_tool_label,
)
from labsopguard.event_preprocessing.tracking.schemas import TrackRelation, TrackedObject


class RuleBasedActionResolver(ActionResolver):
    def resolve(
        self,
        *,
        event_type: str,
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
        event_window: Dict[str, float],
        semantic_context: Optional[Dict[str, Any]] = None,
        qwen_frame_summary: Optional[Dict[str, Any]] = None,
    ) -> ActionResolvedEvent:
        window_relations = self._relations_in_window(track_relations, event_window)
        actor = self._best_actor(tracked_objects, window_relations)
        tool = self._best_tool(tracked_objects, window_relations)
        containers = self._container_candidates(tracked_objects, window_relations)
        source = None
        target = None
        transfer_mode = None

        if event_type == "liquid_transfer":
            source, target = self._resolve_transfer_containers(containers, actor, tool, window_relations)
            transfer_mode = "pipette_transfer" if tool and is_tool_label(tool.class_name) else "pour_or_manual_transfer"
        elif event_type == "hand_object_interaction":
            target_obj = self._best_interaction_target(tracked_objects, actor, window_relations)
            target = self._object_payload(target_obj) if target_obj else (self._object_payload(containers[0]) if containers else None)
        elif event_type == "object_move":
            target_obj = self._best_interaction_target(tracked_objects, actor, window_relations, require_moving=True)
            target = self._object_payload(target_obj) if target_obj else None
        elif event_type == "container_state_change":
            target_obj = self._best_lid_or_container(tracked_objects, actor, window_relations)
            source = self._object_payload(target_obj) if target_obj else (self._object_payload(containers[0]) if containers else None)
        elif event_type == "panel_operation":
            target_obj = self._best_panel(tracked_objects, window_relations)
            target = self._object_payload(target_obj) if target_obj else None

        related_tracks = self._related_track_ids(event_type, window_relations, actor, tool, source, target)
        confidence = self._confidence(actor, tool, source, target, window_relations)
        return ActionResolvedEvent(
            action_type=event_type,
            confidence=confidence,
            actor_track_id=actor.track_id if actor else None,
            tool_track_id=tool.track_id if tool else None,
            source_container=source,
            target_container=target,
            related_tracks=related_tracks,
            transfer_mode=transfer_mode,
            action_resolution_source="rule_based_glove_object_relation_resolver",
            action_resolution_notes=self._notes(event_type, actor, tool, source, target, window_relations),
        )

    @staticmethod
    def _relations_in_window(relations: List[TrackRelation], event_window: Dict[str, float]) -> List[TrackRelation]:
        start = float(event_window.get("start_time_sec") or 0.0)
        end = float(event_window.get("end_time_sec") or start)
        return [rel for rel in relations if rel.end_time_sec >= start and rel.start_time_sec <= end]

    @staticmethod
    def _best_actor(objects: List[TrackedObject], relations: List[TrackRelation]) -> Optional[TrackedObject]:
        candidates = [obj for obj in objects if is_hand_label(obj.class_name)]
        candidates.sort(
            key=lambda obj: (
                sum(1 for rel in relations if obj.track_id in {rel.subject_track_id, rel.object_track_id}),
                obj.velocity_stats.get("displacement_px", 0.0),
                obj.confidence_stats.get("mean", 0.0),
            ),
            reverse=True,
        )
        return candidates[0] if candidates else None

    @staticmethod
    def _best_tool(objects: List[TrackedObject], relations: List[TrackRelation]) -> Optional[TrackedObject]:
        candidates = [obj for obj in objects if is_tool_label(obj.class_name)]
        candidates.sort(key=lambda obj: sum(1 for rel in relations if obj.track_id in {rel.subject_track_id, rel.object_track_id}), reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def _best_panel(objects: List[TrackedObject], relations: List[TrackRelation]) -> Optional[TrackedObject]:
        candidates = [obj for obj in objects if is_panel_label(obj.class_name)]
        candidates.sort(key=lambda obj: sum(1 for rel in relations if obj.track_id in {rel.subject_track_id, rel.object_track_id}), reverse=True)
        return candidates[0] if candidates else None

    @staticmethod
    def _container_candidates(objects: List[TrackedObject], relations: List[TrackRelation]) -> List[TrackedObject]:
        candidates = [obj for obj in objects if is_container_label(obj.class_name)]
        candidates.sort(
            key=lambda obj: (
                sum(1 for rel in relations if obj.track_id in {rel.subject_track_id, rel.object_track_id}),
                obj.confidence_stats.get("mean", 0.0),
            ),
            reverse=True,
        )
        return candidates

    @staticmethod
    def _best_interaction_target(
        objects: List[TrackedObject],
        actor: Optional[TrackedObject],
        relations: List[TrackRelation],
        *,
        require_moving: bool = False,
    ) -> Optional[TrackedObject]:
        candidates = []
        for obj in objects:
            if actor and obj.track_id == actor.track_id:
                continue
            if not is_interaction_object_label(obj.class_name):
                continue
            displacement = float(obj.velocity_stats.get("displacement_px") or 0.0)
            if require_moving and displacement < 8.0:
                continue
            rel_score = sum(
                rel.confidence
                for rel in relations
                if obj.track_id in {rel.subject_track_id, rel.object_track_id}
                and (actor is None or actor.track_id in {rel.subject_track_id, rel.object_track_id})
                and rel.relation_type in {"glove_object_interaction", "object_manipulation", "contact", "carry", "proximity"}
            )
            if rel_score > 0:
                candidates.append((rel_score, displacement, obj.confidence_stats.get("mean", 0.0), obj))
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3] if candidates else None

    @staticmethod
    def _best_lid_or_container(objects: List[TrackedObject], actor: Optional[TrackedObject], relations: List[TrackRelation]) -> Optional[TrackedObject]:
        candidates = []
        for obj in objects:
            if actor and obj.track_id == actor.track_id:
                continue
            if not (is_lid_label(obj.class_name) or is_container_label(obj.class_name)):
                continue
            rel_score = sum(
                rel.confidence
                for rel in relations
                if obj.track_id in {rel.subject_track_id, rel.object_track_id}
                and rel.relation_type in {"container_state_interaction", "glove_object_interaction", "contact", "state_change_context"}
            )
            candidates.append((rel_score, 1 if is_lid_label(obj.class_name) else 0, obj.confidence_stats.get("mean", 0.0), obj))
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
        return candidates[0][3] if candidates else None

    def _resolve_transfer_containers(
        self,
        containers: List[TrackedObject],
        actor: Optional[TrackedObject],
        tool: Optional[TrackedObject],
        relations: List[TrackRelation],
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if not containers:
            return None, None
        if len(containers) == 1:
            return self._object_payload(containers[0]), None
        scored = []
        for obj in containers:
            relation_score = sum(
                rel.confidence
                for rel in relations
                if obj.track_id in {rel.subject_track_id, rel.object_track_id}
                and rel.relation_type in {"contact", "proximity", "transfer_posture", "carry", "glove_object_interaction"}
            )
            actor_score = sum(
                rel.confidence
                for rel in relations
                if actor
                and {rel.subject_track_id, rel.object_track_id} == {actor.track_id, obj.track_id}
                and rel.relation_type in {"contact", "glove_object_interaction", "carry"}
            )
            tool_score = sum(
                rel.confidence
                for rel in relations
                if tool and {rel.subject_track_id, rel.object_track_id} == {tool.track_id, obj.track_id}
            )
            x = obj.centroids[0][0] if obj.centroids else 0.0
            scored.append((actor_score + tool_score + relation_score * 0.5, x, obj))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        source_obj = scored[0][2]
        remaining = [item for item in scored if item[2].track_id != source_obj.track_id]
        remaining.sort(key=lambda item: item[1])
        target_obj = remaining[-1][2] if remaining else None
        return self._object_payload(source_obj), self._object_payload(target_obj) if target_obj else None

    @staticmethod
    def _object_payload(obj: Optional[TrackedObject]) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        return {
            "track_id": obj.track_id,
            "class_name": obj.class_name,
            "display_name": obj.display_name,
            "confidence": obj.confidence_stats.get("mean", 0.0),
        }

    @staticmethod
    def _related_track_ids(event_type: str, relations: List[TrackRelation], actor, tool, source, target) -> List[str]:
        ids = []
        for obj in (actor, tool):
            if obj and obj.track_id not in ids:
                ids.append(obj.track_id)
        for payload in (source, target):
            if payload and payload.get("track_id") and payload["track_id"] not in ids:
                ids.append(payload["track_id"])
        for rel in relations:
            if rel.relation_type in {
                "contact",
                "carry",
                "transfer_posture",
                "panel_interaction",
                "state_change_context",
                "glove_object_interaction",
                "object_manipulation",
                "container_state_interaction",
            }:
                for track_id in (rel.subject_track_id, rel.object_track_id):
                    if track_id not in ids:
                        ids.append(track_id)
        return ids[:16]

    @staticmethod
    def _confidence(actor, tool, source, target, relations: List[TrackRelation]) -> float:
        base = 0.35
        if actor:
            base += 0.15
        if tool:
            base += 0.1
        if source:
            base += 0.15
        if target:
            base += 0.15
        if any(rel.relation_type == "glove_object_interaction" for rel in relations):
            base += 0.12
        if any(rel.relation_type == "object_manipulation" for rel in relations):
            base += 0.08
        if any(rel.relation_type == "transfer_posture" for rel in relations):
            base += 0.1
        return round(min(0.95, base), 4)

    @staticmethod
    def _notes(event_type: str, actor, tool, source, target, relations: List[TrackRelation]) -> str:
        relation_counts = {}
        for rel in relations:
            relation_counts[rel.relation_type] = relation_counts.get(rel.relation_type, 0) + 1
        return f"rule_based event_type={event_type}; actor={getattr(actor, 'track_id', None)}; tool={getattr(tool, 'track_id', None)}; source={source}; target={target}; relations={relation_counts}"
