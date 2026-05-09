from __future__ import annotations

from typing import Dict, List, Optional

from .schemas import EventProposal, TrackRelation, TrackedObject


class EventEvidenceGrader:
    def grade(
        self,
        *,
        event_type: str,
        confidence: float,
        related_tracks: List[str],
        tracked_objects_by_id: Dict[str, TrackedObject],
        supporting_relations: List[TrackRelation],
        direction_confidence: Optional[float] = None,
        direction_status: Optional[str] = None,
        state_confidence: Optional[float] = None,
        missing_critical_fields: Optional[List[str]] = None,
    ) -> tuple[str, str, str]:
        missing = missing_critical_fields or []
        track_quality = self._track_quality(related_tracks, tracked_objects_by_id)
        relation_stability = self._relation_stability(supporting_relations)
        score = confidence * 0.28 + track_quality * 0.28 + relation_stability * 0.18
        if event_type == "liquid_transfer":
            if direction_status == "confirmed":
                score += 0.18
            elif direction_status == "candidate":
                score += 0.1
            else:
                score -= 0.12
                missing.append("direction_unknown")
        if event_type == "container_state_change":
            if state_confidence is not None:
                score += min(0.16, state_confidence * 0.16)
            else:
                missing.append("state_resolution_missing")
        if missing:
            score -= min(0.25, 0.08 * len(set(missing)))
        score = max(0.0, min(1.0, score))
        if score >= 0.72 and not missing:
            grade = "strong"
            review = "auto_confirmed"
        elif score >= 0.45:
            grade = "medium"
            review = "candidate_review"
        else:
            grade = "weak"
            review = "low_confidence"
        summary = (
            f"grade={grade}; score={score:.3f}; track_quality={track_quality:.3f}; "
            f"relation_stability={relation_stability:.3f}; direction={direction_status}:{direction_confidence}; "
            f"state_confidence={state_confidence}; missing={sorted(set(missing))}"
        )
        return grade, review, summary

    @staticmethod
    def _track_quality(track_ids: List[str], tracked_objects_by_id: Dict[str, TrackedObject]) -> float:
        tracks = [tracked_objects_by_id[track_id] for track_id in track_ids if track_id in tracked_objects_by_id]
        if not tracks:
            return 0.0
        return sum(float(track.track_confidence) * (1.0 - float(track.id_switch_risk) * 0.35) for track in tracks) / len(tracks)

    @staticmethod
    def _relation_stability(relations: List[TrackRelation]) -> float:
        if not relations:
            return 0.0
        weighted = [rel.confidence for rel in relations if rel.relation_type in {"contact", "carry", "transfer_posture", "panel_interaction", "state_change_context", "proximity"}]
        if not weighted:
            return 0.0
        return min(1.0, sum(weighted) / max(1, len(weighted)))
