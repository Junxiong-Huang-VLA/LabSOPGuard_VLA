from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from labsopguard.config import RuntimeSettings
from labsopguard.video_analysis import VideoAnalysisPipeline

from .event_proposal import EventProposalBuilder
from .event_segmentation import EventSegmenter
from .frame_detection_stream import DetectionFrameStreamBuilder
from .key_material_extraction import KeyMaterialExtractor
from .material_index_writer import EventMaterialIndexWriter
from .schemas import METADATA_VERSION, DetectionFrame, EventAssetPack, EventProposal, PhysicalEvent, dump_json
from .tracking import TrackRelationGraphBuilder, TrackStreamBuilder


class EventPreprocessingEngine:
    def __init__(self, settings: RuntimeSettings, yolo_model_path: Optional[str] = None) -> None:
        self.settings = settings
        self.pipeline = VideoAnalysisPipeline(settings=settings, yolo_model_path=yolo_model_path or settings.yolo_model_path)
        self.stream_builder = DetectionFrameStreamBuilder(self.pipeline, interval_sec=0.5)
        self.proposal_builder = EventProposalBuilder(min_event_duration=0.8, max_gap_merge=1.2)
        self.segmenter = EventSegmenter(pre_roll_sec=0.8, post_roll_sec=1.0, max_gap_merge=1.5)
        self.extractor = KeyMaterialExtractor(self.pipeline)
        self.track_stream_builder = TrackStreamBuilder()
        self.relation_graph_builder = TrackRelationGraphBuilder()

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
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        materials_root = output_path / "materials" / "events"
        materials_root.mkdir(parents=True, exist_ok=True)
        duration = self._video_duration(source_video)
        video_id = source_video_id or f"{experiment_id}:video:0"

        self._adapt_to_video_duration(duration)
        detection_frames, tracklets = self.stream_builder.build(source_video, analysis_frames, material_stream)
        tracked_objects = self.track_stream_builder.build(
            experiment_id=experiment_id,
            source_video_id=video_id,
            tracklets=tracklets,
        )
        track_relations = self.relation_graph_builder.build(tracked_objects)
        proposals = self.proposal_builder.build(detection_frames, tracklets, tracked_objects, track_relations)
        events = self.segmenter.segment(
            proposals[: max(1, max_events * 3)],
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
            "schema_version": "physical_events.v4",
            "metadata_version": METADATA_VERSION,
            "experiment_id": experiment_id,
            "source_video_id": video_id,
            "tracklets": [track.to_dict() for track in tracklets],
            "tracked_objects": [track.to_dict() for track in tracked_objects],
            "track_relations": [relation.to_dict() for relation in track_relations],
            "events": [event.to_dict() for event in events],
            "event_count": len(events),
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
            "physical_events": [event.to_dict() for event in events],
            "event_asset_packs": [pack.to_dict() for pack in asset_packs],
            "indexed_material_records": [record.to_dict() for record in records],
        }
        material_event_items = [self._material_stream_event_item(event) for event in events]
        return {
            "physical_events_payload": physical_events_payload,
            "preprocessing_payload": preprocessing_payload,
            "material_event_items": material_event_items,
            "events": [event.to_dict() for event in events],
            "asset_packs": [pack.to_dict() for pack in asset_packs],
            "indexed_records": [record.to_dict() for record in records],
        }

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
