from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

from .schemas import ContainerRole, DetectionFrame, Tracklet

CONTAINER_TERMS = {"bottle", "beaker", "vial", "tube", "cup", "jar", "container", "flask", "pipette", "dropper", "试剂瓶", "烧杯", "样品瓶"}


def _norm(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def _is_container(value: str) -> bool:
    text = _norm(value)
    return any(term in text for term in CONTAINER_TERMS)


def _bbox_center_x(bbox: Tuple[int, int, int, int]) -> float:
    return (bbox[0] + bbox[2]) / 2.0


class ContainerRoleResolver:
    """Resolves source/target container slots with a stable schema.

    This class is deliberately heuristic today. Future datasets can replace the internals
    with a learned source-target role model without changing PhysicalEvent/event.json.
    """

    def resolve(
        self,
        *,
        event_type: str,
        involved_objects: List[str],
        involved_track_ids: List[str],
        tracklets_by_id: Dict[str, Tracklet],
        frames: List[DetectionFrame],
    ) -> tuple[Optional[ContainerRole], Optional[ContainerRole]]:
        if event_type not in {"liquid_transfer", "container_state_change", "hand_object_interaction"}:
            return None, None

        candidates = self._container_candidates(involved_objects, involved_track_ids, tracklets_by_id, frames)
        if not candidates:
            return None, None
        if event_type == "hand_object_interaction":
            return None, candidates[0]
        if event_type == "container_state_change":
            return candidates[0], None
        if len(candidates) == 1:
            source = candidates[0] if event_type == "liquid_transfer" else None
            target = None if event_type == "liquid_transfer" else candidates[0]
            return source, target

        # For transfer-like events, use horizontal ordering as the initial role prior.
        ordered = sorted(candidates, key=lambda item: self._role_sort_key(item))
        return ordered[0], ordered[-1]

    def _container_candidates(
        self,
        involved_objects: List[str],
        involved_track_ids: List[str],
        tracklets_by_id: Dict[str, Tracklet],
        frames: List[DetectionFrame],
    ) -> List[ContainerRole]:
        roles: List[ContainerRole] = []
        seen = set()
        for track_id in involved_track_ids:
            track = tracklets_by_id.get(track_id)
            if not track or not _is_container(track.class_name):
                continue
            bbox = track.bboxes[0] if track.bboxes else None
            roles.append(
                ContainerRole(
                    object_name=track.class_name,
                    track_id=track.track_id,
                    role_confidence=min(0.85, max(0.35, track.mean_confidence)),
                    bbox=bbox,
                )
            )
            seen.add(_norm(track.class_name))
        for name in involved_objects:
            if not _is_container(name) or _norm(name) in seen:
                continue
            roles.append(ContainerRole(object_name=name, role_confidence=0.35, role_source="heuristic_semantic_object_order"))
            seen.add(_norm(name))
        return roles[:4]

    @staticmethod
    def _role_sort_key(role: ContainerRole) -> float:
        if role.bbox is None:
            return 0.0
        return _bbox_center_x(role.bbox)
