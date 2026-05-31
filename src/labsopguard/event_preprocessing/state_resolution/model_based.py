from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ObjectStateResolution, ObjectStateResolver
from .container_state_model import PrototypeContainerStateModel
from .rule_based import RuleBasedStateResolver
from labsopguard.event_preprocessing.tracking.schemas import TrackRelation, TrackedObject


class ModelBasedStateResolver(ObjectStateResolver):
    def __init__(self, model: Any = None, fallback: Optional[ObjectStateResolver] = None, model_path: Optional[str | Path] = None) -> None:
        resolved_path = model_path or os.getenv("LABSOPGUARD_CONTAINER_STATE_MODEL")
        self.model = model
        if self.model is None and resolved_path and Path(resolved_path).exists():
            self.model = PrototypeContainerStateModel.load(resolved_path)
        self.fallback = fallback or RuleBasedStateResolver()

    def resolve(
        self,
        *,
        tracked_object: Optional[TrackedObject],
        roi_clip_path: Optional[Path] = None,
        keyframe_paths: Optional[List[Path]] = None,
        related_track_relations: Optional[List[TrackRelation]] = None,
        semantic_summary: Optional[Dict[str, Any]] = None,
    ) -> ObjectStateResolution:
        if self.model is None:
            result = self.fallback.resolve(
                tracked_object=tracked_object,
                roi_clip_path=roi_clip_path,
                keyframe_paths=keyframe_paths,
                related_track_relations=related_track_relations,
                semantic_summary=semantic_summary,
            )
            result.resolution_source = "model_based_state_adapter_fallback_rule_based"
            return result
        if hasattr(self.model, "predict"):
            return self.model.predict(keyframe_paths=keyframe_paths, semantic_summary=semantic_summary)
        result = self.fallback.resolve(
            tracked_object=tracked_object,
            roi_clip_path=roi_clip_path,
            keyframe_paths=keyframe_paths,
            related_track_relations=related_track_relations,
            semantic_summary=semantic_summary,
        )
        result.resolution_source = "model_based_state_adapter_unknown_model_fallback_rule_based"
        return result
