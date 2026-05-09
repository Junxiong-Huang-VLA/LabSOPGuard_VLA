from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import ActionResolvedEvent, ActionResolver
from .rule_based import RuleBasedActionResolver
from labsopguard.event_preprocessing.tracking.schemas import TrackedObject, TrackRelation


class ModelBasedActionResolver(ActionResolver):
    """Adapter shell for future dataset-trained action models.

    Keep this interface stable. A model implementation should consume the same
    TrackedObject/TrackRelation/window inputs and return ActionResolvedEvent.
    Until configured, it delegates to RuleBasedActionResolver so the main chain
    remains functional without an action dataset.
    """

    def __init__(self, model: Any = None, fallback: Optional[ActionResolver] = None) -> None:
        self.model = model
        self.fallback = fallback or RuleBasedActionResolver()

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
        if self.model is None:
            resolved = self.fallback.resolve(
                event_type=event_type,
                tracked_objects=tracked_objects,
                track_relations=track_relations,
                event_window=event_window,
                semantic_context=semantic_context,
                qwen_frame_summary=qwen_frame_summary,
            )
            resolved.action_resolution_source = "model_based_adapter_fallback_rule_based"
            return resolved
        raise NotImplementedError("Model action resolver adapter is ready; attach a trained model implementation here.")
