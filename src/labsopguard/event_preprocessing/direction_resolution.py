from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from labsopguard.event_preprocessing.tracking.schemas import TrackRelation, TrackedObject


@dataclass
class DirectionResolution:
    direction_confidence: float
    direction_status: str
    direction_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DirectionResolver(Protocol):
    def resolve(
        self,
        *,
        source_container: Optional[Dict[str, Any]],
        target_container: Optional[Dict[str, Any]],
        actor_track_id: Optional[str],
        tool_track_id: Optional[str],
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> DirectionResolution:
        ...


class RuleBasedDirectionResolver(DirectionResolver):
    def resolve(
        self,
        *,
        source_container: Optional[Dict[str, Any]],
        target_container: Optional[Dict[str, Any]],
        actor_track_id: Optional[str],
        tool_track_id: Optional[str],
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> DirectionResolution:
        evidence: List[str] = []
        score = 0.0
        if not source_container or not target_container:
            return DirectionResolution(0.0, "unknown", ["insufficient_visual_evidence"])
        by_id = {obj.track_id: obj for obj in tracked_objects}
        source_id = source_container.get("track_id")
        target_id = target_container.get("track_id")
        if source_id and target_id:
            evidence.append("source_target_tracks_present")
            score += 0.18
        if actor_track_id and source_id and self._relation_strength(actor_track_id, str(source_id), track_relations) > self._relation_strength(actor_track_id, str(target_id), track_relations):
            evidence.append("actor_contact_source_stronger")
            score += 0.24
        if tool_track_id and (self._relation_strength(str(tool_track_id), str(source_id), track_relations) or self._relation_strength(str(tool_track_id), str(target_id), track_relations)):
            evidence.append("tool_relation_present")
            score += 0.18
        source = by_id.get(str(source_id)) if source_id else None
        target = by_id.get(str(target_id)) if target_id else None
        if source and target and source.centroids and target.centroids:
            sx, sy = source.centroids[0]
            tx, ty = target.centroids[0]
            if ty > sy:
                evidence.append("target_below_source")
                score += 0.16
            if abs(sx - tx) > 20:
                evidence.append("spatially_distinct_containers")
                score += 0.08
        if any(rel.relation_type == "transfer_posture" and {rel.subject_track_id, rel.object_track_id} >= {str(source_id), str(target_id)} for rel in track_relations if source_id and target_id):
            evidence.append("transfer_posture_relation")
            score += 0.16
        semantic_text = " ".join(str(v) for v in (semantic_context or {}).values()).lower()
        if any(token in semantic_text for token in ["pour", "transfer", "pipette", "dispense", "倾倒", "转移"]):
            evidence.append("qwen_temporal_hint")
            score += 0.1
        if not evidence:
            evidence.append("insufficient_visual_evidence")
        confidence = round(min(1.0, score), 4)
        if confidence >= 0.72:
            status = "confirmed"
        elif confidence >= 0.38:
            status = "candidate"
        else:
            status = "unknown"
            if "insufficient_visual_evidence" not in evidence:
                evidence.append("insufficient_visual_evidence")
        return DirectionResolution(confidence, status, evidence)

    @staticmethod
    def _relation_strength(a: Optional[str], b: Optional[str], relations: List[TrackRelation]) -> float:
        if not a or not b:
            return 0.0
        return sum(rel.confidence for rel in relations if {rel.subject_track_id, rel.object_track_id} == {a, b} and rel.relation_type in {"contact", "proximity", "carry", "transfer_posture"})
