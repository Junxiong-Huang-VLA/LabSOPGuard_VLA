from __future__ import annotations

from typing import Any, Dict, List, Optional

from .action_resolution.rule_based import RuleBasedActionResolver
from .tracking.schemas import TrackRelation, TrackedObject


class SourceTargetResolver:
    """Compatibility facade for source/target resolution.

    New code should use ActionResolver directly. This facade keeps source-target
    role resolution isolated for future learned replacement.
    """

    def __init__(self) -> None:
        self.resolver = RuleBasedActionResolver()

    def resolve_liquid_transfer(
        self,
        tracked_objects: List[TrackedObject],
        track_relations: List[TrackRelation],
        event_window: Dict[str, float],
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved = self.resolver.resolve(
            event_type="liquid_transfer",
            tracked_objects=tracked_objects,
            track_relations=track_relations,
            event_window=event_window,
            semantic_context=semantic_context,
        )
        return resolved.to_dict()
