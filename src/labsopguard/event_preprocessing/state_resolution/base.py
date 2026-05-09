from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from labsopguard.event_preprocessing.tracking.schemas import TrackRelation, TrackedObject


@dataclass
class ObjectStateResolution:
    state_before: Optional[str]
    state_after: Optional[str]
    state_change_type: Optional[str]
    confidence: float
    evidence: List[str] = field(default_factory=list)
    resolution_source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ObjectStateResolver(Protocol):
    def resolve(
        self,
        *,
        tracked_object: Optional[TrackedObject],
        roi_clip_path: Optional[Path] = None,
        keyframe_paths: Optional[List[Path]] = None,
        related_track_relations: Optional[List[TrackRelation]] = None,
        semantic_summary: Optional[Dict[str, Any]] = None,
    ) -> ObjectStateResolution:
        ...
