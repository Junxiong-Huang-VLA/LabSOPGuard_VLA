from __future__ import annotations

import hashlib
import os
from typing import List

from .schemas import EventProposal, PhysicalEvent
from .semantic_naming import SemanticNamer


def _norm(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


class EventSegmenter:
    def __init__(
        self,
        pre_roll_sec: float = 0.8,
        post_roll_sec: float = 1.0,
        max_gap_merge: float = 1.5,
        max_event_duration_sec: float | None = None,
    ) -> None:
        self.pre_roll_sec = float(pre_roll_sec)
        self.post_roll_sec = float(post_roll_sec)
        self.max_gap_merge = float(max_gap_merge)
        self.max_event_duration_sec = float(
            max_event_duration_sec
            if max_event_duration_sec is not None
            else os.getenv("LABSOPGUARD_EVENT_MAX_DURATION_SEC", "12.0")
        )
        self.namer = SemanticNamer()

    def segment(
        self,
        proposals: List[EventProposal],
        *,
        experiment_id: str,
        source_video_id: str,
        video_duration_sec: float,
        experiment_name: str = "",
    ) -> List[PhysicalEvent]:
        merged = self._merge_proposals(proposals)
        events: List[PhysicalEvent] = []
        for index, proposal in enumerate(merged, 1):
            start, end = self._event_window(proposal, video_duration_sec)
            if end <= start:
                end = min(video_duration_sec or proposal.end_time_sec + 0.1, start + max(0.1, proposal.end_time_sec - proposal.start_time_sec))
            event_id = self._event_id(experiment_id, proposal.event_type, start, end, index)
            stable_name = self.namer.stable_name(experiment_name, proposal.event_type, proposal.involved_objects, start, end)
            # 细粒度手套接触事件：直接用中文物体名生成 display_name
            if proposal.contact_target_zh:
                display_name = f"{experiment_name}-手套接触{proposal.contact_target_zh}"
            else:
                display_name = self.namer.display_name(experiment_name, proposal.event_type, proposal.involved_objects)
            key_timestamps = sorted(set([round(start, 3), round((start + end) / 2.0, 3), round(end, 3)]))
            events.append(
                PhysicalEvent(
                    event_id=event_id,
                    experiment_id=experiment_id,
                    source_video_id=source_video_id,
                    event_type=proposal.event_type,
                    stable_name=stable_name,
                    display_name=display_name,
                    actor_name="operator" if "hand" in " ".join(proposal.related_detection_classes) or "glove" in " ".join(proposal.related_detection_classes) else "unknown_actor",
                    start_time_sec=round(start, 3),
                    end_time_sec=round(end, 3),
                    duration_sec=round(max(0.0, end - start), 3),
                    key_timestamps=key_timestamps,
                    involved_objects=proposal.involved_objects,
                    dominant_object=proposal.dominant_object,
                    involved_track_ids=proposal.involved_track_ids,
                    primary_track_id=proposal.primary_track_id,
                    source_container=proposal.source_container.to_dict() if proposal.source_container else None,
                    target_container=proposal.target_container.to_dict() if proposal.target_container else None,
                    track_motion_summary=proposal.track_motion_summary,
                    actor_track_id=proposal.actor_track_id,
                    tool_track_id=proposal.tool_track_id,
                    related_tracks=proposal.related_tracks,
                    transfer_mode=proposal.transfer_mode,
                    action_resolution_source=proposal.action_resolution_source,
                    action_resolution_notes=proposal.action_resolution_notes,
                    supporting_relation_ids=proposal.supporting_relation_ids,
                    direction_confidence=proposal.direction_confidence,
                    direction_status=proposal.direction_status,
                    direction_evidence=proposal.direction_evidence,
                    state_before=proposal.state_before,
                    state_after=proposal.state_after,
                    state_change_type=proposal.state_change_type,
                    state_confidence=proposal.state_confidence,
                    state_evidence=proposal.state_evidence,
                    evidence_grade=proposal.evidence_grade,
                    review_status=proposal.review_status,
                    evidence_summary=proposal.evidence_summary,
                    confidence=proposal.confidence,
                    event_status="confirmed" if proposal.review_status == "auto_confirmed" else "candidate",
                    proposal_source=proposal.proposal_source,
                    evidence_frame_indices=proposal.evidence_frame_indices,
                    related_detection_classes=proposal.related_detection_classes,
                    notes=proposal.notes,
                )
            )
        return events

    def _merge_proposals(self, proposals: List[EventProposal]) -> List[EventProposal]:
        ordered = sorted(proposals, key=lambda item: (item.event_type, item.start_time_sec, item.end_time_sec))
        merged: List[EventProposal] = []
        for proposal in ordered:
            if not merged:
                merged.append(proposal)
                continue
            prev = merged[-1]
            if (
                prev.event_type == proposal.event_type
                and self._merge_signature(prev) == self._merge_signature(proposal)
                and proposal.start_time_sec - prev.end_time_sec <= self.max_gap_merge
            ):
                prev.end_frame_idx = max(prev.end_frame_idx, proposal.end_frame_idx)
                prev.end_time_sec = max(prev.end_time_sec, proposal.end_time_sec)
                prev.evidence_frame_indices = sorted(set([*prev.evidence_frame_indices, *proposal.evidence_frame_indices]))
                prev.involved_objects = list(dict.fromkeys([*prev.involved_objects, *proposal.involved_objects]))[:12]
                prev.involved_track_ids = list(dict.fromkeys([*prev.involved_track_ids, *proposal.involved_track_ids]))[:24]
                prev.primary_track_id = prev.primary_track_id or proposal.primary_track_id
                prev.source_container = prev.source_container or proposal.source_container
                prev.target_container = prev.target_container or proposal.target_container
                prev.track_motion_summary = self._merge_motion_summary(prev.track_motion_summary, proposal.track_motion_summary)
                prev.actor_track_id = prev.actor_track_id or proposal.actor_track_id
                prev.tool_track_id = prev.tool_track_id or proposal.tool_track_id
                prev.related_tracks = list(dict.fromkeys([*prev.related_tracks, *proposal.related_tracks]))[:24]
                prev.transfer_mode = prev.transfer_mode or proposal.transfer_mode
                prev.action_resolution_notes = f"{prev.action_resolution_notes}; {proposal.action_resolution_notes}".strip("; ")
                prev.supporting_relation_ids = list(dict.fromkeys([*prev.supporting_relation_ids, *proposal.supporting_relation_ids]))[:48]
                prev.direction_confidence = max(
                    [value for value in [prev.direction_confidence, proposal.direction_confidence] if value is not None],
                    default=None,
                )
                prev.direction_status = self._merge_direction_status(prev.direction_status, proposal.direction_status)
                prev.direction_evidence = list(dict.fromkeys([*prev.direction_evidence, *proposal.direction_evidence]))
                prev.state_before = prev.state_before or proposal.state_before
                prev.state_after = proposal.state_after or prev.state_after
                prev.state_change_type = prev.state_change_type or proposal.state_change_type
                prev.state_confidence = max(
                    [value for value in [prev.state_confidence, proposal.state_confidence] if value is not None],
                    default=None,
                )
                prev.state_evidence = list(dict.fromkeys([*prev.state_evidence, *proposal.state_evidence]))
                prev.evidence_grade, prev.review_status = self._merge_grade(prev.evidence_grade, proposal.evidence_grade)
                prev.evidence_summary = f"{prev.evidence_summary}; {proposal.evidence_summary}".strip("; ")
                prev.related_detection_classes = sorted(set([*prev.related_detection_classes, *proposal.related_detection_classes]))
                prev.confidence = round(max(prev.confidence, proposal.confidence), 4)
                prev.notes = f"{prev.notes}; merged_with={proposal.proposal_id}"
            else:
                merged.append(proposal)
        merged.sort(key=lambda item: item.start_time_sec)
        return merged

    def _event_window(self, proposal: EventProposal, video_duration_sec: float) -> tuple[float, float]:
        video_end = max(video_duration_sec, proposal.end_time_sec)
        raw_start = max(0.0, float(proposal.start_time_sec))
        raw_end = min(video_end, max(raw_start + 0.1, float(proposal.end_time_sec)))
        start = max(0.0, raw_start - self.pre_roll_sec)
        end = min(video_end, raw_end + self.post_roll_sec)

        max_duration = max(0.5, float(self.max_event_duration_sec or 0.0))
        if end - start <= max_duration:
            return round(start, 3), round(end, 3)

        focus = (raw_start + raw_end) / 2.0
        half = max_duration / 2.0
        start = max(0.0, focus - half)
        end = min(video_end, focus + half)
        if end - start < max_duration:
            if start <= 0.0:
                end = min(video_end, start + max_duration)
            elif end >= video_end:
                start = max(0.0, end - max_duration)
        return round(start, 3), round(end, 3)

    @staticmethod
    def _merge_signature(proposal: EventProposal) -> tuple:
        if proposal.event_type == "hand_object_interaction":
            target = proposal.contact_target_class or proposal.dominant_object
            if not target and len(proposal.involved_objects) > 1:
                target = next(
                    (
                        _norm(item)
                        for item in proposal.involved_objects
                        if "hand" not in _norm(item) and "glove" not in _norm(item)
                    ),
                    "",
                )
            return (proposal.event_type, _norm(target))
        if proposal.event_type == "liquid_transfer":
            return (
                proposal.event_type,
                EventSegmenter._container_signature(proposal.source_container),
                EventSegmenter._container_signature(proposal.target_container),
                _norm(proposal.tool_track_id),
            )
        if proposal.event_type in {"panel_operation", "container_state_change"}:
            return (
                proposal.event_type,
                EventSegmenter._container_signature(proposal.source_container),
                EventSegmenter._container_signature(proposal.target_container),
                _norm(proposal.dominant_object),
            )
        return (proposal.event_type, _norm(proposal.dominant_object), _norm(proposal.primary_track_id))

    @staticmethod
    def _container_signature(container) -> str:
        if not container:
            return ""
        return _norm(container.track_id or container.object_name)

    @staticmethod
    def _event_id(experiment_id: str, event_type: str, start: float, end: float, index: int) -> str:
        raw = f"{experiment_id}:{event_type}:{start:.3f}:{end:.3f}:{index}"
        return "evt_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _merge_motion_summary(left: dict, right: dict) -> dict:
        left = dict(left or {})
        right = dict(right or {})
        moving = list(dict.fromkeys([*(left.get("moving_track_ids") or []), *(right.get("moving_track_ids") or [])]))
        tracks = list((left.get("tracks") or []) + (right.get("tracks") or []))
        by_id = {}
        for track in tracks:
            if isinstance(track, dict) and track.get("track_id"):
                by_id[track["track_id"]] = track
        return {
            "track_count": len(by_id) or max(int(left.get("track_count") or 0), int(right.get("track_count") or 0)),
            "max_displacement_px": max(float(left.get("max_displacement_px") or 0.0), float(right.get("max_displacement_px") or 0.0)),
            "moving_track_ids": moving,
            "tracks": list(by_id.values())[:12],
        }

    @staticmethod
    def _merge_direction_status(left: str | None, right: str | None) -> str | None:
        order = {"unknown": 0, None: 0, "candidate": 1, "confirmed": 2}
        return left if order.get(left, 0) >= order.get(right, 0) else right

    @staticmethod
    def _merge_grade(left: str, right: str) -> tuple[str, str]:
        order = {"weak": 0, "medium": 1, "strong": 2}
        grade = left if order.get(left, 0) >= order.get(right, 0) else right
        review = "auto_confirmed" if grade == "strong" else ("candidate_review" if grade == "medium" else "low_confidence")
        return grade, review
