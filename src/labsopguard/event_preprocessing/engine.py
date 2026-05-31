from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2

from labsopguard.config import RuntimeSettings
from labsopguard.video_analysis import VideoAnalysisPipeline

from .activity_presegmenter import PresegmentConfig
from .class_roles import is_container_label, is_interaction_object_label, is_panel_label, is_tool_label, norm_label
from .event_proposal import EventProposalBuilder
from .event_segmentation import EventSegmenter
from .frame_detection_stream import DetectionFrameStreamBuilder
from .key_material_extraction import KeyMaterialExtractor
from .material_index_writer import EventMaterialIndexWriter
from .physical_event_gate import (
    gate_container_state_change,
    gate_hand_object_contact,
    gate_liquid_transfer,
    gate_object_move,
    gate_panel_operation,
    summarize_gate_decisions,
)
from .schemas import METADATA_VERSION, DetectionFrame, EventAssetPack, EventProposal, PhysicalEvent, dump_json
from .tracking import TrackRelationGraphBuilder, TrackStreamBuilder
from .track_normalizer import track_evidence_from_points

logger = logging.getLogger(__name__)


class EventPreprocessingEngine:
    def __init__(self, settings: RuntimeSettings, yolo_model_path: Optional[str] = None) -> None:
        self.settings = settings
        self.pipeline = VideoAnalysisPipeline(settings=settings, yolo_model_path=yolo_model_path or settings.yolo_model_path)

        presegment_config = self._load_presegment_config(settings)
        cache_dir = (
            settings.project_root / "outputs" / "cache" / "detection"
            if settings.detection_cache_enabled
            else None
        )

        self.stream_builder = DetectionFrameStreamBuilder(
            self.pipeline,
            interval_sec=0.5,
            presegment_config=presegment_config,
            cache_dir=cache_dir,
            batch_size=settings.batch_size,
        )
        self.proposal_builder = EventProposalBuilder(min_event_duration=0.8, max_gap_merge=1.2)
        self.segmenter = EventSegmenter(pre_roll_sec=0.8, post_roll_sec=1.0, max_gap_merge=1.5)
        self.extractor = KeyMaterialExtractor(self.pipeline)
        self.track_stream_builder = TrackStreamBuilder()
        self.relation_graph_builder = TrackRelationGraphBuilder()

    @staticmethod
    def _load_presegment_config(settings: RuntimeSettings) -> PresegmentConfig:
        import yaml, os
        config_path = settings.project_root / "configs" / "model" / "detection_runtime.yaml"
        if not config_path.exists():
            return PresegmentConfig()
        try:
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            logger.warning("Failed to parse presegment config: %s", exc)
            return PresegmentConfig()
        ps = payload.get("presegment", {})
        if not ps:
            return PresegmentConfig()
        return PresegmentConfig(
            enabled=ps.get("enabled", True),
            scan_fps=float(ps.get("scan_fps", 2.0)),
            scan_resolution=tuple(ps.get("scan_resolution", [160, 120])),
            motion_threshold_mode=ps.get("motion_threshold_mode", "adaptive"),
            motion_fixed_threshold=float(ps.get("motion_fixed_threshold", 0.02)),
            min_segment_sec=float(ps.get("min_segment_sec", 3.0)),
            merge_gap_sec=float(ps.get("merge_gap_sec", 5.0)),
            padding_sec=float(ps.get("padding_sec", 2.0)),
            skip_if_video_shorter_than=float(ps.get("skip_if_video_shorter_than", 30.0)),
            forced_sample_interval_sec=float(ps.get("forced_sample_interval_sec", 60.0)),
        )

    def _adapt_to_video_duration(self, duration: float) -> None:
        """Adjust sampling density and merge gaps based on actual video length.

        For long videos (>120s) the fixed max_frames=360 causes actual frame
        intervals larger than max_gap_merge, which breaks window detection
        and produces 0 events.  We scale both parameters so that:
          - actual_interval  ≈ interval_sec  (capped at ~1.5s for long videos)
          - max_gap_merge    = actual_interval * 2.5  (always spans at least 2 frames)
        """
        import os, math
        target_interval = float(os.getenv("LABSOPGUARD_EVENT_INTERVAL_SEC", "0.5"))
        env_max = os.getenv("LABSOPGUARD_EVENT_MAX_FRAMES")
        if env_max:
            max_frames = int(env_max)
        else:
            # Aim for ~1 frame/s on very long videos, minimum 600 frames
            max_frames = max(600, int(duration / max(0.5, target_interval)))
        self.stream_builder.max_frames = max_frames
        actual_interval = duration / max_frames if max_frames > 0 else target_interval
        # gap must be at least 2.5× the actual sampling interval so consecutive
        # matching frames form a continuous window rather than isolated spikes
        self.proposal_builder.max_gap_merge = max(1.2, actual_interval * 2.5)
        self.segmenter.max_gap_merge = max(1.5, actual_interval * 3.0)

    def run(
        self,
        *,
        experiment_id: str,
        experiment_name: str,
        source_video: str | Path,
        output_dir: str | Path,
        material_index_path: str | Path,
        analysis_frames: Optional[List[Dict[str, Any]]] = None,
        material_stream: Optional[List[Dict[str, Any]]] = None,
        source_video_id: Optional[str] = None,
        max_events: int = 80,
        time_range: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        materials_root = output_path / "materials" / "events"
        materials_root.mkdir(parents=True, exist_ok=True)
        duration = self._video_duration(source_video)
        video_id = source_video_id or f"{experiment_id}:video:0"

        effective_duration = duration
        if time_range:
            effective_duration = min(duration, time_range[1]) - max(0, time_range[0])
        self._adapt_to_video_duration(effective_duration)
        detection_frames, tracklets = self.stream_builder.build(source_video, analysis_frames, material_stream, time_range=time_range)

        # Write presegment result for diagnostics
        presegment_result = self.stream_builder.last_presegment_result
        if presegment_result:
            presegment_path = output_path / "presegment_result.json"
            try:
                dump_json(presegment_path, {
                    "schema_version": "presegment.v1",
                    "experiment_id": experiment_id,
                    "segments": [seg.to_dict() for seg in presegment_result],
                    "total_active_sec": round(sum(s.duration_sec for s in presegment_result), 3),
                    "video_duration_sec": round(duration, 3),
                })
            except OSError as exc:
                logger.warning("Failed to write presegment_result.json: %s", exc)
        tracked_objects = self.track_stream_builder.build(
            experiment_id=experiment_id,
            source_video_id=video_id,
            tracklets=tracklets,
        )
        track_relations = self.relation_graph_builder.build(tracked_objects)
        proposals = self.proposal_builder.build(detection_frames, tracklets, tracked_objects, track_relations)
        raw_proposal_rows = self._raw_proposal_rows(proposals)
        proposals, gate_decisions, rejected_candidates = self._apply_physical_event_gates(
            proposals,
            detection_frames=detection_frames,
            tracklets=tracklets,
            tracked_objects=tracked_objects,
            track_relations=track_relations,
        )
        gate_summary = summarize_gate_decisions(gate_decisions)
        gate_summary.setdefault("qwen_audit_enabled", False)
        gate_summary.setdefault("qwen_audit_count", 0)
        gate_summary.setdefault("config_path_used", self._physical_gate_config_path())
        self._write_gate_artifacts(output_path, gate_decisions, rejected_candidates, gate_summary)
        trace_summary = self._write_candidate_trace_artifacts(
            output_path,
            experiment_id=experiment_id,
            video_id=video_id,
            source_video=source_video,
            duration_sec=duration,
            time_range=time_range,
            detection_frames=detection_frames,
            tracklets=tracklets,
            tracked_objects=tracked_objects,
            track_relations=track_relations,
            raw_proposal_rows=raw_proposal_rows,
            gate_decisions=gate_decisions,
        )
        confirmed_proposals = [proposal for proposal in proposals if proposal.status == "confirmed"]
        events = self.segmenter.segment(
            confirmed_proposals[: max(1, max_events * 3)],
            experiment_id=experiment_id,
            source_video_id=video_id,
            video_duration_sec=duration,
            experiment_name=experiment_name,
        )[:max_events]

        asset_packs: List[EventAssetPack] = []
        for event in events:
            asset_packs.append(self.extractor.extract_assets(source_video, event, materials_root / event.event_id, tracked_objects=tracked_objects))

        index_writer = EventMaterialIndexWriter(material_index_path)
        try:
            index_writer.reset_experiment(experiment_id)
            records = index_writer.write_events(events)
        finally:
            index_writer.close()

        physical_events_payload = {
            "schema": "physical_events.v4",
            "schema_version": "physical_events.v4",
            "metadata_version": METADATA_VERSION,
            "experiment_id": experiment_id,
            "source_video_id": video_id,
            "tracklets": [track.to_dict() for track in tracklets],
            "tracked_objects": [track.to_dict() for track in tracked_objects],
            "track_relations": [relation.to_dict() for relation in track_relations],
            "events": [event.to_dict() for event in events],
            "event_count": len(events),
            "physical_event_gate_summary": gate_summary,
            "event_candidate_trace_summary": trace_summary,
        }
        preprocessing_payload = {
            "schema_version": "preprocessing.v4",
            "metadata_version": METADATA_VERSION,
            "experiment_id": experiment_id,
            "event_preprocessing": {
                "detection_frame_count": len(detection_frames),
                "tracklet_count": len(tracklets),
                "tracked_object_count": len(tracked_objects),
                "track_relation_count": len(track_relations),
                "proposal_count": len(proposals),
                "confirmed_proposal_count": len(confirmed_proposals),
                "physical_event_gate_summary": gate_summary,
                "event_candidate_trace_summary": trace_summary,
                "physical_event_count": len(events),
                "asset_pack_count": len(asset_packs),
                "overlay_mode": "event_selective",
                "event_types": sorted(set(event.event_type for event in events)),
            },
            "detection_frames_sample": [frame.to_dict() for frame in detection_frames[:20]],
            "tracklets": [track.to_dict() for track in tracklets],
            "tracked_objects": [track.to_dict() for track in tracked_objects],
            "track_relations": [relation.to_dict() for relation in track_relations],
            "event_proposals": [proposal.to_dict() for proposal in proposals[:200]],
            "raw_event_proposals": raw_proposal_rows[:200],
            "rejected_physical_event_candidates": rejected_candidates[:200],
            "physical_events": [event.to_dict() for event in events],
            "event_asset_packs": [pack.to_dict() for pack in asset_packs],
            "indexed_material_records": [record.to_dict() for record in records],
        }
        material_event_items = [self._material_stream_event_item(event) for event in events]

        # Auto-publish best frames (one per event) to library
        self._auto_publish_best_frames(experiment_id, output_path)

        # Auto-sync to global material index
        self._sync_global_index(experiment_id, output_path)

        return {
            "physical_events_payload": physical_events_payload,
            "preprocessing_payload": preprocessing_payload,
            "material_event_items": material_event_items,
            "events": [event.to_dict() for event in events],
            "asset_packs": [pack.to_dict() for pack in asset_packs],
            "indexed_records": [record.to_dict() for record in records],
        }

    def _apply_physical_event_gates(
        self,
        proposals: List[EventProposal],
        *,
        detection_frames: Sequence[DetectionFrame],
        tracklets: Sequence[Any],
        tracked_objects: Sequence[Any],
        track_relations: Sequence[Any],
    ) -> tuple[List[EventProposal], List[Dict[str, Any]], List[Dict[str, Any]]]:
        frames_by_idx = {frame.frame_idx: frame for frame in detection_frames}
        tracklets_by_id = {track.track_id: track for track in tracklets}
        tracked_by_id = {track.track_id: track for track in tracked_objects}
        gate_decisions: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for proposal in proposals:
            frames = [frames_by_idx[idx] for idx in proposal.evidence_frame_indices if idx in frames_by_idx]
            relations = self._relations_for_proposal(proposal, track_relations)
            decision = self._gate_proposal(proposal, frames, tracklets_by_id, tracked_by_id, relations)
            decision.setdefault("candidate_id", proposal.proposal_id)
            decision.setdefault("time_start", proposal.start_time_sec)
            decision.setdefault("time_end", proposal.end_time_sec)
            decision.setdefault("actor_track_id", proposal.actor_track_id)
            decision.setdefault("object_track_ids", list(proposal.involved_track_ids or []))
            decision.setdefault("object_labels", list(proposal.involved_objects or []))
            gate_decisions.append(decision)
            self._apply_gate_to_proposal(proposal, decision)
            if decision.get("status") == "rejected":
                rejected.append(self._rejected_candidate_row(proposal, decision))
        return proposals, gate_decisions, rejected

    def _gate_proposal(
        self,
        proposal: EventProposal,
        frames: Sequence[DetectionFrame],
        tracklets_by_id: Dict[str, Any],
        tracked_by_id: Dict[str, Any],
        relations: Sequence[Any],
    ) -> Dict[str, Any]:
        candidate = {
            "candidate_id": proposal.proposal_id,
            "event_type": proposal.event_type,
            "time_start": proposal.start_time_sec,
            "time_end": proposal.end_time_sec,
            "object_labels": proposal.involved_objects,
            "object_track_ids": proposal.involved_track_ids,
            "change_score": max((frame.change_score for frame in frames), default=0.0),
        }
        if proposal.event_type == "hand_object_interaction":
            return gate_hand_object_contact(
                event_candidate=candidate,
                frame_evidence_list=self._contact_rows_for_proposal(proposal, frames),
                external_observation={
                    "hand_track_id": proposal.actor_track_id,
                    "object_track_id": proposal.primary_track_id,
                    "has_hand": any(self._is_hand_label(det.class_name) for frame in frames for det in frame.detections),
                    "has_object": bool(proposal.dominant_object),
                },
            )
        if proposal.event_type == "object_move":
            track_id = self._select_object_motion_track(proposal, tracklets_by_id)
            track = self._track_evidence(track_id, tracklets_by_id, tracked_by_id) if track_id else None
            hand_contact = {"status": "confirmed"} if self._has_relation(relations, {"carry", "object_manipulation", "glove_object_interaction"}) or proposal.actor_track_id else {"status": "candidate"}
            return gate_object_move(
                event_candidate=candidate,
                track=track,
                scene_motion={"method": "none", "limitations": ["no_scene_stabilization"]},
                hand_contact=hand_contact,
            )
        if proposal.event_type == "liquid_transfer":
            return gate_liquid_transfer(
                event_candidate=candidate,
                source_container_track={"track_id": proposal.source_container.track_id} if proposal.source_container and proposal.source_container.track_id else None,
                target_container_track={"track_id": proposal.target_container.track_id} if proposal.target_container and proposal.target_container.track_id else None,
                tool_track={"track_id": proposal.tool_track_id} if proposal.tool_track_id else None,
                liquid_observation={
                    "has_liquid_region": False,
                    "source_level_down": False,
                    "target_level_up": False,
                    "visual_change_score": 0.0,
                },
                qwen_semantics={"direction_status": proposal.direction_status} if proposal.direction_status else None,
            )
        if proposal.event_type == "panel_operation":
            panel_relation = self._has_relation(relations, {"panel_interaction"})
            return gate_panel_operation(
                event_candidate=candidate,
                hand_track={"track_id": proposal.actor_track_id} if proposal.actor_track_id else None,
                device_track={"track_id": proposal.primary_track_id} if proposal.primary_track_id else None,
                control_roi={"source": "relation_or_geometry"} if panel_relation else None,
                external_observation={
                    "hand_in_control_roi_frames": len(frames) if panel_relation else 0,
                    "contact_frames": len(frames) if panel_relation else 0,
                    "display_changed": False,
                    "button_state_changed": False,
                    "switch_state_changed": False,
                },
            )
        if proposal.event_type == "container_state_change":
            changed_fields = [proposal.state_change_type] if proposal.state_change_type else []
            return gate_container_state_change(
                event_candidate=candidate,
                container_track={"track_id": proposal.primary_track_id} if proposal.primary_track_id else None,
                pre_state={"state": proposal.state_before} if proposal.state_before else None,
                post_state={"state": proposal.state_after} if proposal.state_after else None,
                frame_pair_evidence={"changed_fields": changed_fields},
            )
        return {
            "status": "uncertain",
            "event_type": proposal.event_type,
            "confidence": 0.0,
            "hard_gate": {"passed": False, "gate_name": "unknown", "required_evidence": [], "passed_evidence": [], "failed_evidence": []},
            "evidence": {},
            "reject_reasons": ["unsupported_event_type"],
            "limitations": [],
            "audit": {},
        }

    @staticmethod
    def _apply_gate_to_proposal(proposal: EventProposal, decision: Dict[str, Any]) -> None:
        status = str(decision.get("status") or "uncertain")
        proposal.status = status
        proposal.hard_gate = dict(decision.get("hard_gate") or {})
        proposal.evidence_detail = dict(decision.get("evidence") or {})
        proposal.reject_reasons = list(decision.get("reject_reasons") or [])
        proposal.limitations = list(decision.get("limitations") or [])
        proposal.confidence = round(min(float(proposal.confidence or 0.0), float(decision.get("confidence") or proposal.confidence or 0.0)), 4)
        if status == "confirmed":
            proposal.review_status = "auto_confirmed"
            proposal.evidence_grade = "strong"
        elif status == "candidate":
            proposal.review_status = "candidate_review"
            proposal.evidence_grade = "medium"
        elif status == "rejected":
            proposal.review_status = "rejected"
            proposal.evidence_grade = "weak"
        else:
            proposal.review_status = "uncertain"
            proposal.evidence_grade = "weak"
        reasons = ", ".join(proposal.reject_reasons[:5])
        gate_name = proposal.hard_gate.get("gate_name") if isinstance(proposal.hard_gate, dict) else ""
        proposal.evidence_summary = f"{proposal.evidence_summary}; physical_event_gate={gate_name}:{status}; reject_reasons={reasons}".strip("; ")

    @staticmethod
    def _relations_for_proposal(proposal: EventProposal, relations: Sequence[Any]) -> List[Any]:
        supporting = set(proposal.supporting_relation_ids or [])
        return [
            rel
            for rel in relations
            if rel.relation_id in supporting
            or (rel.end_time_sec >= proposal.start_time_sec and rel.start_time_sec <= proposal.end_time_sec)
        ]

    @staticmethod
    def _has_relation(relations: Sequence[Any], relation_types: set[str]) -> bool:
        return any(str(rel.relation_type) in relation_types for rel in relations)

    @staticmethod
    def _select_object_motion_track(proposal: EventProposal, tracklets_by_id: Dict[str, Any]) -> Optional[str]:
        candidates = [tracklets_by_id[track_id] for track_id in proposal.involved_track_ids if track_id in tracklets_by_id]
        object_tracks = [track for track in candidates if not EventPreprocessingEngine._is_hand_label(track.class_name)]
        if not object_tracks:
            object_tracks = candidates
        if not object_tracks:
            return proposal.primary_track_id
        object_tracks.sort(key=lambda track: (float(track.displacement_px or 0.0), len(track.frame_indices), float(track.mean_confidence or 0.0)), reverse=True)
        return object_tracks[0].track_id

    @staticmethod
    def _track_evidence(track_id: str, tracklets_by_id: Dict[str, Any], tracked_by_id: Dict[str, Any]) -> Dict[str, Any] | None:
        tracklet = tracklets_by_id.get(track_id)
        if tracklet is None:
            return None
        tracked = tracked_by_id.get(track_id)
        total = max(1, len(tracklet.bboxes) - 1)
        duration = max(0.0, float(tracklet.end_time_sec) - float(tracklet.start_time_sec))
        points = [
            {
                "bbox": bbox,
                "time_sec": round(float(tracklet.start_time_sec) + duration * (idx / total), 3),
            }
            for idx, bbox in enumerate(tracklet.bboxes)
        ]
        identity = float(getattr(tracked, "track_confidence", None) or getattr(tracklet, "mean_confidence", 0.0) or 0.0)
        id_switch = float(getattr(tracked, "id_switch_risk", None) if tracked is not None else (0.12 if int(tracklet.fragment_count or 1) <= 1 else 0.4))
        return track_evidence_from_points(
            track_id=tracklet.track_id,
            object_label=tracklet.class_name,
            points=points,
            track_type="tracker_track" if tracklet.track_id else "inferred_track",
            identity_confidence=identity,
            id_switch_risk=id_switch,
            can_confirm_motion=True,
            limitations=[] if tracklet.track_id else ["inferred track without stable tracker id"],
        ).to_dict()

    @staticmethod
    def _contact_rows_for_proposal(proposal: EventProposal, frames: Sequence[DetectionFrame]) -> List[Dict[str, Any]]:
        target_label = (proposal.contact_target_class or proposal.dominant_object or "").strip().lower()
        rows: List[Dict[str, Any]] = []
        for frame in frames:
            detections = [
                {"label": det.class_name, "confidence": det.confidence, "bbox": list(det.bbox), "track_id": det.track_id}
                for det in frame.detections
            ]
            hands = [det for det in frame.detections if EventPreprocessingEngine._is_hand_label(det.class_name)]
            targets = [
                det
                for det in frame.detections
                if not EventPreprocessingEngine._is_hand_label(det.class_name)
                and (not target_label or det.class_name.strip().lower() == target_label)
            ]
            interactions = []
            for hand in hands:
                for target in targets:
                    iou = EventPreprocessingEngine._bbox_iou(hand.bbox, target.bbox)
                    coverage = EventPreprocessingEngine._object_coverage(hand.bbox, target.bbox)
                    distance = EventPreprocessingEngine._bbox_edge_distance(hand.bbox, target.bbox)
                    if iou > 0.0 or coverage > 0.0 or distance <= 30.0:
                        interactions.append(
                            {
                                "object_label": target.class_name,
                                "iou": round(iou, 4),
                                "object_coverage_by_hand": round(coverage, 4),
                                "distance_px": round(distance, 3),
                                "score": 0.8 if iou > 0.0 or coverage > 0.0 else 0.5,
                            }
                        )
            rows.append(
                {
                    "frame_index": frame.frame_idx,
                    "time_sec": frame.timestamp_sec,
                    "detections": detections,
                    "hand_object_interactions": interactions,
                }
            )
        return rows

    @staticmethod
    def _rejected_candidate_row(proposal: EventProposal, decision: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "candidate_id": proposal.proposal_id,
            "event_type": proposal.event_type,
            "status": decision.get("status"),
            "time_start": proposal.start_time_sec,
            "time_end": proposal.end_time_sec,
            "source_view": None,
            "actor_track_id": proposal.actor_track_id,
            "object_track_ids": proposal.involved_track_ids,
            "object_labels": proposal.involved_objects,
            "reject_reasons": decision.get("reject_reasons") or [],
            "evidence_detail": decision.get("evidence") or {},
            "limitations": decision.get("limitations") or [],
        }

    @staticmethod
    def _write_gate_artifacts(output_path: Path, gate_decisions: Sequence[Dict[str, Any]], rejected: Sequence[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        EventPreprocessingEngine._write_jsonl(output_path / "physical_event_gate_decisions.jsonl", gate_decisions)
        EventPreprocessingEngine._write_jsonl(output_path / "rejected_physical_event_candidates.jsonl", rejected)
        EventPreprocessingEngine._write_jsonl(output_path / "qwen_event_audits.jsonl", [])
        dump_json(output_path / "physical_event_gate_summary.json", summary)

    def _write_candidate_trace_artifacts(
        self,
        output_path: Path,
        *,
        experiment_id: str,
        video_id: str,
        source_video: str | Path,
        duration_sec: float,
        time_range: Optional[tuple],
        detection_frames: Sequence[DetectionFrame],
        tracklets: Sequence[Any],
        tracked_objects: Sequence[Any],
        track_relations: Sequence[Any],
        raw_proposal_rows: Sequence[Dict[str, Any]],
        gate_decisions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Persist the v2.2 candidate-generation funnel without changing gate behavior."""
        output_path.mkdir(parents=True, exist_ok=True)
        source_video_path = str(Path(source_video))
        detection_counts = self._detection_counts(detection_frames)
        tracklet_counts = self._tracklet_counts(tracklets)
        relation_counts = Counter(str(getattr(rel, "relation_type", "") or "unknown") for rel in track_relations)
        proposal_counts = Counter(str(row.get("event_type") or "unknown") for row in raw_proposal_rows)
        gate_counts_by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"confirmed": 0, "candidate": 0, "rejected": 0, "uncertain": 0})
        for decision in gate_decisions:
            event_type = str(decision.get("event_type") or "unknown")
            status = str(decision.get("status") or "uncertain")
            gate_counts_by_type[event_type][status] = gate_counts_by_type[event_type].get(status, 0) + 1

        first_zero_stage = self._first_zero_stage(
            frames_sampled=len(detection_frames),
            detections=int(detection_counts["detections"]),
            tracklets=len(tracklets),
            relations=len(track_relations),
            raw_proposals=len(raw_proposal_rows),
            gate_decisions=len(gate_decisions),
        )
        zero_diagnosis = self._zero_candidate_diagnosis(
            first_zero_stage=first_zero_stage,
            detection_counts=detection_counts,
            tracklets=tracklets,
            track_relations=track_relations,
            raw_proposal_rows=raw_proposal_rows,
            gate_decisions=gate_decisions,
        )
        if not (self.settings.allowed_detection_labels or []):
            zero_diagnosis.append("runtime_allowed_labels_empty")
        if not (self.pipeline.yolo_model_path or self.settings.yolo_model_path):
            zero_diagnosis.append("yolo_model_path_missing")
        zero_diagnosis = sorted(set(zero_diagnosis))

        labels = Counter()
        for frame in detection_frames:
            labels.update(norm_label(det.class_name) for det in frame.detections if det.class_name)
        top_labels = dict(labels.most_common(20))
        range_start = float(time_range[0]) if time_range else 0.0
        range_end = float(time_range[1]) if time_range else float(duration_sec or 0.0)
        trace_rows = [
            self._trace_row(video_id, "all", range_start, range_end, "frame_sampling", {"frames_sampled": len(detection_frames)}, top_labels, []),
            self._trace_row(video_id, "all", range_start, range_end, "yolo_detection", detection_counts, top_labels, self._stage_drop_reasons("yolo_detection", zero_diagnosis)),
            self._trace_row(video_id, "all", range_start, range_end, "tracking", tracklet_counts, dict(tracklet_counts.get("tracklets_by_label", {})), self._stage_drop_reasons("tracking", zero_diagnosis)),
            self._trace_row(video_id, "all", range_start, range_end, "relation", {"relations": len(track_relations)}, dict(relation_counts.most_common(20)), self._stage_drop_reasons("relation", zero_diagnosis)),
            self._trace_row(video_id, "all", range_start, range_end, "proposal", {"raw_proposals": len(raw_proposal_rows)}, dict(proposal_counts.most_common(20)), self._stage_drop_reasons("proposal", zero_diagnosis)),
            self._trace_row(video_id, "all", range_start, range_end, "gate", self._gate_count_payload(gate_decisions), {}, self._stage_drop_reasons("gate", zero_diagnosis)),
        ]

        summary = {
            "schema": "event_candidate_trace_summary.v2.2",
            "experiment_id": experiment_id,
            "video_id": video_id,
            "source_video_path": source_video_path,
            "view": self._infer_view(source_video_path),
            "duration_sec": round(float(duration_sec or 0.0), 3),
            "time_range": [round(range_start, 3), round(range_end, 3)] if time_range else None,
            "frames_total": None,
            "frames_sampled": len(detection_frames),
            "yolo_frames_with_detections": sum(1 for frame in detection_frames if frame.detections),
            "detections_by_label": top_labels,
            "tracklets_by_label": dict(Counter(norm_label(getattr(track, "class_name", "")) for track in tracklets if getattr(track, "class_name", "")).most_common(50)),
            "relations_by_type": dict(relation_counts.most_common(50)),
            "raw_proposals_by_type": dict(proposal_counts.most_common(50)),
            "gate_decisions_by_type": {event_type: dict(counts) for event_type, counts in sorted(gate_counts_by_type.items())},
            "first_zero_stage": first_zero_stage,
            "zero_candidate_diagnosis": zero_diagnosis,
            "runtime": {
                "project_root": str(self.settings.project_root),
                "yolo_model_path": self.pipeline.yolo_model_path or self.settings.yolo_model_path,
                "device": self.settings.device,
                "allowed_detection_label_count": len(self.settings.allowed_detection_labels or []),
            },
        }
        self._write_jsonl(output_path / "event_candidate_trace.jsonl", trace_rows)
        self._write_jsonl(output_path / "raw_event_proposals.jsonl", list(raw_proposal_rows))
        dump_json(output_path / "event_candidate_trace_summary.json", summary)
        return summary

    @staticmethod
    def _raw_proposal_rows(proposals: Sequence[EventProposal]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for proposal in proposals:
            rows.append(
                {
                    "proposal_id": proposal.proposal_id,
                    "event_type": proposal.event_type,
                    "time_start": proposal.start_time_sec,
                    "time_end": proposal.end_time_sec,
                    "source": proposal.proposal_source or proposal.action_resolution_source or "event_proposal_builder",
                    "actor_track_id": proposal.actor_track_id,
                    "object_track_ids": list(proposal.involved_track_ids or []),
                    "evidence": {
                        "confidence": proposal.confidence,
                        "evidence_frame_indices": list(proposal.evidence_frame_indices or []),
                        "related_detection_classes": list(proposal.related_detection_classes or []),
                        "track_motion_summary": proposal.track_motion_summary or {},
                        "supporting_relation_ids": list(proposal.supporting_relation_ids or []),
                        "action_resolution_source": proposal.action_resolution_source,
                        "action_resolution_notes": proposal.action_resolution_notes,
                    },
                    "proposal_reason": proposal.notes or proposal.evidence_summary or proposal.action_resolution_notes,
                    "will_enter_gate": True,
                    "blocked_before_gate": False,
                    "block_reason": None,
                }
            )
        return rows

    @staticmethod
    def _detection_counts(frames: Sequence[DetectionFrame]) -> Dict[str, Any]:
        labels = Counter()
        counts = {
            "frames_sampled": len(frames),
            "detections": 0,
            "hand_detections": 0,
            "object_detections": 0,
            "container_detections": 0,
            "device_detections": 0,
            "liquid_related_detections": 0,
        }
        for frame in frames:
            for det in frame.detections:
                label = norm_label(det.class_name)
                labels[label] += 1
                counts["detections"] += 1
                if EventPreprocessingEngine._is_hand_label(label):
                    counts["hand_detections"] += 1
                if is_interaction_object_label(label):
                    counts["object_detections"] += 1
                if is_container_label(label):
                    counts["container_detections"] += 1
                    counts["liquid_related_detections"] += 1
                if is_panel_label(label):
                    counts["device_detections"] += 1
                if is_tool_label(label):
                    counts["liquid_related_detections"] += 1
        counts["top_labels"] = dict(labels.most_common(20))
        return counts

    @staticmethod
    def _tracklet_counts(tracklets: Sequence[Any]) -> Dict[str, Any]:
        labels = Counter(norm_label(getattr(track, "class_name", "")) for track in tracklets if getattr(track, "class_name", ""))
        hand_tracklets = sum(1 for track in tracklets if EventPreprocessingEngine._is_hand_label(getattr(track, "class_name", "")))
        object_tracklets = sum(1 for track in tracklets if is_interaction_object_label(getattr(track, "class_name", "")))
        return {
            "tracklets": len(tracklets),
            "hand_tracklets": hand_tracklets,
            "object_tracklets": object_tracklets,
            "tracklets_by_label": dict(labels.most_common(20)),
        }

    @staticmethod
    def _gate_count_payload(gate_decisions: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        status_counts = Counter(str(item.get("status") or "uncertain") for item in gate_decisions)
        return {
            "gate_confirmed": status_counts.get("confirmed", 0),
            "gate_candidate": status_counts.get("candidate", 0),
            "gate_rejected": status_counts.get("rejected", 0),
            "gate_uncertain": status_counts.get("uncertain", 0),
        }

    @staticmethod
    def _trace_row(
        video_id: str,
        view: str,
        time_start: float,
        time_end: float,
        stage: str,
        counts: Dict[str, Any],
        top_labels: Dict[str, int],
        drop_reasons: Sequence[str],
    ) -> Dict[str, Any]:
        base_counts = {
            "frames_sampled": 0,
            "detections": 0,
            "hand_detections": 0,
            "object_detections": 0,
            "container_detections": 0,
            "device_detections": 0,
            "liquid_related_detections": 0,
            "tracklets": 0,
            "hand_tracklets": 0,
            "object_tracklets": 0,
            "relations": 0,
            "raw_proposals": 0,
            "gate_confirmed": 0,
            "gate_candidate": 0,
            "gate_rejected": 0,
            "gate_uncertain": 0,
        }
        for key, value in counts.items():
            if key in base_counts:
                base_counts[key] = int(value or 0)
        return {
            "video_id": video_id,
            "view": view,
            "window_id": "all",
            "time_start": round(float(time_start or 0.0), 3),
            "time_end": round(float(time_end or 0.0), 3),
            "stage": stage,
            "counts": base_counts,
            "top_labels": top_labels,
            "drop_reasons": list(drop_reasons),
            "notes": "",
        }

    @staticmethod
    def _first_zero_stage(*, frames_sampled: int, detections: int, tracklets: int, relations: int, raw_proposals: int, gate_decisions: int) -> str | None:
        if frames_sampled <= 0:
            return "frame_sampling"
        if detections <= 0:
            return "yolo_detection"
        if tracklets <= 0:
            return "tracking"
        if relations <= 0:
            return "relation"
        if raw_proposals <= 0:
            return "proposal"
        if gate_decisions <= 0:
            return "gate"
        return None

    @staticmethod
    def _zero_candidate_diagnosis(
        *,
        first_zero_stage: str | None,
        detection_counts: Dict[str, Any],
        tracklets: Sequence[Any],
        track_relations: Sequence[Any],
        raw_proposal_rows: Sequence[Dict[str, Any]],
        gate_decisions: Sequence[Dict[str, Any]],
    ) -> List[str]:
        reasons: List[str] = []
        if first_zero_stage:
            reasons.append(f"first_zero_stage:{first_zero_stage}")
        if int(detection_counts.get("detections") or 0) == 0:
            reasons.append("no_yolo_detections")
        elif int(detection_counts.get("hand_detections") or 0) == 0:
            reasons.append("no_hand_detections")
        if int(detection_counts.get("object_detections") or 0) == 0:
            reasons.append("no_interaction_object_detections")
        if not tracklets:
            reasons.append("tracklets_empty")
        if not track_relations:
            reasons.append("relations_empty")
        if not raw_proposal_rows:
            reasons.append("no_raw_proposals")
        if raw_proposal_rows and not gate_decisions:
            reasons.append("raw_proposals_did_not_enter_gate")
        top_labels = set((detection_counts.get("top_labels") or {}).keys())
        lab_like = any(
            EventPreprocessingEngine._is_hand_label(label)
            or is_interaction_object_label(label)
            or is_container_label(label)
            or is_panel_label(label)
            or is_tool_label(label)
            for label in top_labels
        )
        if top_labels and not lab_like:
            reasons.append("detections_are_non_lab_labels")
        return sorted(set(reasons))

    @staticmethod
    def _stage_drop_reasons(stage: str, zero_diagnosis: Sequence[str]) -> List[str]:
        marker = f"first_zero_stage:{stage}"
        if marker in zero_diagnosis:
            return list(zero_diagnosis)
        return [reason for reason in zero_diagnosis if reason.startswith(stage)]

    @staticmethod
    def _infer_view(path_text: str) -> str:
        text = path_text.lower()
        if "first" in text or "camera_00" in text:
            return "first_person"
        if "third" in text or "camera_01" in text:
            return "third_person"
        return "unknown"

    def _physical_gate_config_path(self) -> str | None:
        candidates = [
            self.settings.project_root / "configs" / "model" / "physical_event_gate.yaml",
            Path(__file__).resolve().parents[4] / "configs" / "physical_event_gate.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return None

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

    @staticmethod
    def _is_hand_label(value: str) -> bool:
        text = str(value or "").lower()
        return "hand" in text or "glove" in text

    @staticmethod
    def _bbox_iou(a: Iterable[float], b: Iterable[float]) -> float:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0

    @staticmethod
    def _object_coverage(hand_bbox: Iterable[float], object_bbox: Iterable[float]) -> float:
        hx1, hy1, hx2, hy2 = [float(v) for v in hand_bbox]
        ox1, oy1, ox2, oy2 = [float(v) for v in object_bbox]
        ix1, iy1 = max(hx1, ox1), max(hy1, oy1)
        ix2, iy2 = min(hx2, ox2), min(hy2, oy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        object_area = max(1.0, max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1))
        return inter / object_area

    @staticmethod
    def _bbox_edge_distance(a: Iterable[float], b: Iterable[float]) -> float:
        import math

        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
        dx = max(bx1 - ax2, ax1 - bx2, 0.0)
        dy = max(by1 - ay2, ay1 - by2, 0.0)
        return math.hypot(dx, dy)

    def _auto_publish_best_frames(self, experiment_id: str, experiment_dir: Path) -> None:
        """Auto-publish one best keyframe per event to the official material library."""
        try:
            from labsopguard.material_approval import MaterialApprovalGate
            library_root = self.settings.project_root / "outputs" / "material_references"
            gate = MaterialApprovalGate(experiment_dir, library_root=library_root)
            result = gate.auto_publish_best_frames()
            if result.get("published_count", 0) > 0:
                logger.info("Auto-published %d best frames for %s", result["published_count"], experiment_id)
        except Exception as exc:
            logger.warning("Auto-publish best frames failed: %s", exc)

    def _sync_global_index(self, experiment_id: str, experiment_dir: Path) -> None:
        """Sync this experiment's materials to the global cross-experiment index."""
        try:
            from labsopguard.global_index import GlobalMaterialIndex
            global_db = self.settings.project_root / "outputs" / "global_material_index.sqlite"
            gidx = GlobalMaterialIndex(global_db)
            count = gidx.sync_experiment(experiment_id, experiment_dir)
            gidx.close()
            if count > 0:
                logger.info("Synced %d materials to global index for %s", count, experiment_id)
        except Exception as exc:
            logger.warning("Failed to sync global material index: %s", exc)

    @staticmethod
    def _video_duration(video_path: str | Path) -> float:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return round(frames / fps, 3) if fps else 0.0

    @staticmethod
    def _material_stream_event_item(event: PhysicalEvent) -> Dict[str, Any]:
        asset = event.asset_pack or {}
        return {
            "schema_version": "material_stream.event.v1",
            "item_id": f"mat_{event.event_id}",
            "experiment_id": event.experiment_id,
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp_sec": event.start_time_sec,
            "time_start": event.start_time_sec,
            "time_end": event.end_time_sec,
            "duration_sec": event.duration_sec,
            "display_name": event.display_name,
            "stable_name": event.stable_name,
            "actor_name": event.actor_name,
            "actor_track_id": event.actor_track_id,
            "tool_track_id": event.tool_track_id,
            "source_container": event.source_container,
            "target_container": event.target_container,
            "transfer_mode": event.transfer_mode,
            "direction_confidence": event.direction_confidence,
            "direction_status": event.direction_status,
            "direction_evidence": event.direction_evidence,
            "state_before": event.state_before,
            "state_after": event.state_after,
            "state_change_type": event.state_change_type,
            "state_confidence": event.state_confidence,
            "state_evidence": event.state_evidence,
            "evidence_grade": event.evidence_grade,
            "review_status": event.review_status,
            "evidence_summary": event.evidence_summary,
            "related_tracks": event.related_tracks,
            "action_resolution_source": event.action_resolution_source,
            "action_resolution_notes": event.action_resolution_notes,
            "involved_track_ids": event.involved_track_ids,
            "primary_track_id": event.primary_track_id,
            "track_motion_summary": event.track_motion_summary,
            "object_labels": event.involved_objects,
            "detected_activities": [event.event_type],
            "semantic_tags": [event.event_type, *event.involved_objects],
            "clip_id": event.event_id,
            "clip_file_path": asset.get("clip_path"),
            "preview_path": asset.get("preview_path"),
            "keyframe_paths": asset.get("keyframe_paths") or [],
            "scene_description": event.display_name,
            "analysis": {"event_preprocessing": event.to_dict()},
            "provenance": {"source": "event_preprocessing", "metadata_version": METADATA_VERSION},
        }
