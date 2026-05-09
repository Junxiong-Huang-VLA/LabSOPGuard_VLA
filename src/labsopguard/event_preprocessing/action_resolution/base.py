from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from labsopguard.event_preprocessing.tracking.schemas import TrackedObject, TrackRelation


@dataclass
class ActionResolvedEvent:
    action_type: str
    confidence: float
    actor_track_id: Optional[str] = None
    tool_track_id: Optional[str] = None
    source_container: Optional[Dict[str, Any]] = None
    target_container: Optional[Dict[str, Any]] = None
    related_tracks: List[str] = field(default_factory=list)
    transfer_mode: Optional[str] = None
    action_resolution_source: str = "unknown"
    action_resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActionResolver(Protocol):
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
        ...
