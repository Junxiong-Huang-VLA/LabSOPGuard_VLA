from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ObjectStateResolution, ObjectStateResolver
from labsopguard.event_preprocessing.tracking.schemas import TrackRelation, TrackedObject


class RuleBasedStateResolver(ObjectStateResolver):
    def resolve(
        self,
        *,
        tracked_object: Optional[TrackedObject],
        roi_clip_path: Optional[Path] = None,
        keyframe_paths: Optional[List[Path]] = None,
        related_track_relations: Optional[List[TrackRelation]] = None,
        semantic_summary: Optional[Dict[str, Any]] = None,
    ) -> ObjectStateResolution:
        if tracked_object is None:
            return ObjectStateResolution(None, None, None, 0.0, ["missing_tracked_object"], "rule_based_state_resolver")
        evidence: List[str] = []
        text = " ".join(str(v) for v in (semantic_summary or {}).values()).lower()
        state_before = "unknown"
        state_after = "unknown"
        change_type = None
        confidence = 0.25
        if any(token in text for token in ["open", "opened", "cap removed", "lid open", "打开", "开盖"]):
            state_before = "closed_or_lidded"
            state_after = "open_candidate"
            change_type = "container_open_candidate"
            confidence += 0.3
            evidence.append("semantic_open_hint")
        elif any(token in text for token in ["close", "closed", "cap placed", "lid closed", "关闭", "盖上"]):
            state_before = "open_or_unlidded"
            state_after = "closed_candidate"
            change_type = "container_close_candidate"
            confidence += 0.3
            evidence.append("semantic_close_hint")
        if "moving" in tracked_object.state_labels:
            evidence.append("rest_to_moving_context")
            change_type = change_type or "rest_moving_context"
            state_before = state_before if state_before != "unknown" else "rest_or_stable"
            state_after = state_after if state_after != "unknown" else "moving_or_manipulated"
            confidence += 0.15
        relations = related_track_relations or []
        if any(rel.relation_type in {"contact", "carry", "state_change_context"} for rel in relations):
            evidence.append("related_contact_or_state_context")
            confidence += 0.15
        if any("lid" in str(item).lower() or "cap" in str(item).lower() for item in tracked_object.state_labels):
            evidence.append("lid_presence_change_candidate")
            change_type = change_type or "lid_presence_change_candidate"
            confidence += 0.1
        if not evidence:
            evidence.append("insufficient_state_evidence")
            change_type = "state_change_candidate"
        return ObjectStateResolution(
            state_before=state_before,
            state_after=state_after,
            state_change_type=change_type,
            confidence=round(min(0.9, confidence), 4),
            evidence=evidence,
            resolution_source="rule_based_state_resolver",
        )
