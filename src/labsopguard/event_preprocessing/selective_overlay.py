from __future__ import annotations

from typing import Iterable, List, Set

from .class_roles import CONTAINER_TERMS, HAND_TERMS, IGNORE_INTERACTION_TERMS, LID_TERMS, PANEL_TERMS, TOOL_TERMS
from .schemas import DetectionBox, PhysicalEvent


def _norm(value: str) -> str:
    return str(value or "").lower().replace("-", "_").replace(" ", "_")


def _contains_any(value: str, terms: Iterable[str]) -> bool:
    text = _norm(value)
    return any(term in text for term in terms)


class SelectiveOverlayPolicy:
    """Only draw bounding boxes for the two objects directly involved in the event.

    For hand_object_interaction with contact_target_class:
      → only gloved_hand + that specific target class (e.g. balance)
    For other events:
      → actor + source/target containers only
    """

    def relevant_track_ids(self, event: PhysicalEvent) -> List[str]:
        """Return only the track IDs of objects directly participating in this event."""
        ids: List[str] = []
        # actor (hand)
        if event.actor_track_id:
            ids.append(str(event.actor_track_id))
        # tool (e.g. pipette)
        if event.tool_track_id:
            ids.append(str(event.tool_track_id))
        # source / target containers
        for container in (event.source_container, event.target_container):
            if container and isinstance(container, dict):
                tid = container.get("track_id")
                if tid and str(tid) not in ids:
                    ids.append(str(tid))
        # primary track as fallback
        if not ids and event.primary_track_id:
            ids.append(str(event.primary_track_id))
        return ids

    def relevant_classes(self, event: PhysicalEvent) -> Set[str]:
        """Return only the class names that should be drawn for this event."""
        # hand_object_interaction with specific target: only hand + that target
        contact_target = getattr(event, "contact_target_class", None)
        if not contact_target:
            ed = event.event_data if hasattr(event, "event_data") and isinstance(event.event_data, dict) else {}
            contact_target = ed.get("contact_target_class")
            if not contact_target:
                notes = getattr(event, "notes", "") or ""
                if "glove_contact_target=" in notes:
                    contact_target = notes.split("glove_contact_target=")[1].split(";")[0].strip()

        if event.event_type == "hand_object_interaction" and contact_target:
            return HAND_TERMS | {_norm(contact_target)}

        # Other event types: narrow to direct participants
        direct_classes: Set[str] = set(HAND_TERMS)
        if event.event_type == "liquid_transfer":
            container_classes: Set[str] = set()
            for container in (event.source_container, event.target_container):
                if container and isinstance(container, dict):
                    cn = container.get("class_name") or container.get("object_name")
                    if cn:
                        container_classes.add(_norm(cn))
            direct_classes |= container_classes
            direct_classes |= TOOL_TERMS
            if not container_classes:
                # Fallback/material-stream events often know only semantic object
                # labels. Keep lab containers/tools visible instead of showing a
                # hand-only overlay.
                direct_classes |= CONTAINER_TERMS | self._event_object_classes(event)
        elif event.event_type == "panel_operation":
            direct_classes |= PANEL_TERMS
        elif event.event_type == "container_state_change":
            direct_classes |= LID_TERMS
            for container in (event.source_container, event.target_container):
                if container and isinstance(container, dict):
                    cn = container.get("class_name") or container.get("object_name")
                    if cn:
                        direct_classes.add(_norm(cn))
        elif event.event_type == "object_move":
            direct_classes |= self._event_object_classes(event)
        else:
            direct_classes |= self._event_object_classes(event)
        return direct_classes

    @staticmethod
    def _event_object_classes(event: PhysicalEvent) -> Set[str]:
        return {
            label
            for label in (_norm(item) for item in event.involved_objects or [])
            if label and label not in IGNORE_INTERACTION_TERMS and not _contains_any(label, HAND_TERMS)
        }

    def filter_detections(self, event: PhysicalEvent, detections: List[DetectionBox]) -> List[DetectionBox]:
        allowed = self.relevant_classes(event)
        if not allowed:
            return detections
        selected = []
        for det in detections:
            label = _norm(det.class_name)
            if label in allowed or any(label in token or token in label for token in allowed):
                selected.append(det)
        return selected[:8]
