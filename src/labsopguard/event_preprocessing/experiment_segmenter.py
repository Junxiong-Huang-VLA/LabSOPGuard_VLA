"""Experiment boundary detection and automatic segmentation.

Detects boundaries between independent experiments within a long continuous
video recording, enabling automatic split into sub-experiments for individual
pipeline processing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .activity_presegmenter import ActivitySegment

logger = logging.getLogger(__name__)


@dataclass
class SegmentationConfig:
    enabled: bool = True
    min_gap_sec: float = 180.0
    min_experiment_duration_sec: float = 60.0
    max_experiments: int = 10
    object_change_threshold: float = 0.4
    use_vlm_confirmation: bool = False
    vlm_confirmation_threshold: float = 0.5
    merge_short_gaps: bool = True
    skip_if_video_shorter_than: float = 600.0
    boundary_sample_frames: int = 5
    boundary_sample_interval_sec: float = 1.0


@dataclass
class ExperimentBoundary:
    boundary_id: str
    position_sec: float
    gap_before_sec: float
    confidence: float
    signals: List[str] = field(default_factory=list)
    objects_before: List[str] = field(default_factory=list)
    objects_after: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "position_sec": round(self.position_sec, 3),
            "gap_before_sec": round(self.gap_before_sec, 3),
            "confidence": round(self.confidence, 3),
            "signals": self.signals,
            "objects_before": self.objects_before,
            "objects_after": self.objects_after,
        }


@dataclass
class ExperimentSegment:
    segment_id: str
    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    activity_segments: List[ActivitySegment] = field(default_factory=list)
    boundary_before: Optional[ExperimentBoundary] = None
    display_name: str = ""
    scene_summary: str = ""
    naming_confidence: float = 0.0
    naming_source: str = "fallback"
    preview_video_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "index": self.index,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "duration_sec": round(self.duration_sec, 3),
            "activity_segment_count": len(self.activity_segments),
            "boundary_before": self.boundary_before.to_dict() if self.boundary_before else None,
            "display_name": self.display_name or f"Experiment {self.index + 1}",
            "scene_summary": self.scene_summary,
            "naming_confidence": round(self.naming_confidence, 3),
            "naming_source": self.naming_source,
            "preview_video_path": self.preview_video_path,
        }


@dataclass
class ExperimentSegmentation:
    video_duration_sec: float
    total_segments: int
    segments: List[ExperimentSegment]
    boundaries: List[ExperimentBoundary]
    unassigned_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "experiment_segmentation.v1",
            "video_duration_sec": round(self.video_duration_sec, 3),
            "total_segments": self.total_segments,
            "segments": [s.to_dict() for s in self.segments],
            "boundaries": [b.to_dict() for b in self.boundaries],
            "unassigned_time_sec": round(self.unassigned_time_sec, 3),
        }


class ExperimentSegmenter:
    """Detects experiment boundaries in long video recordings."""

    def __init__(
        self,
        config: Optional[SegmentationConfig] = None,
        yolo_pipeline: Optional[Any] = None,
    ) -> None:
        self.config = config or SegmentationConfig()
        self.yolo_pipeline = yolo_pipeline

    def segment(
        self,
        activity_segments: List[ActivitySegment],
        video_path: Optional[Path] = None,
        video_duration_sec: float = 0.0,
    ) -> ExperimentSegmentation:
        """Detect experiment boundaries and split into segments."""
        if not activity_segments:
            return ExperimentSegmentation(
                video_duration_sec=video_duration_sec,
                total_segments=0,
                segments=[],
                boundaries=[],
                unassigned_time_sec=video_duration_sec,
            )

        duration = video_duration_sec or (
            max(s.end_sec for s in activity_segments) if activity_segments else 0.0
        )

        if not self.config.enabled or duration < self.config.skip_if_video_shorter_than:
            segment = ExperimentSegment(
                segment_id="seg_0",
                index=0,
                start_sec=activity_segments[0].start_sec,
                end_sec=activity_segments[-1].end_sec,
                duration_sec=activity_segments[-1].end_sec - activity_segments[0].start_sec,
                activity_segments=list(activity_segments),
            )
            return ExperimentSegmentation(
                video_duration_sec=duration,
                total_segments=1,
                segments=[segment],
                boundaries=[],
                unassigned_time_sec=duration - segment.duration_sec,
            )

        # Level 1: Find candidate boundaries (long gaps)
        candidates = self._find_gap_candidates(activity_segments)

        # Level 2: Object change confirmation (if YOLO available)
        if video_path and self.yolo_pipeline:
            candidates = self._confirm_with_object_change(candidates, activity_segments, video_path)

        # Filter by confidence and limit
        boundaries = sorted(candidates, key=lambda b: b.confidence, reverse=True)
        boundaries = boundaries[: self.config.max_experiments - 1]
        boundaries.sort(key=lambda b: b.position_sec)

        # Split activity segments by boundaries
        segments = self._split_by_boundaries(activity_segments, boundaries, duration)

        # Merge short segments
        segments = self._merge_short_segments(segments)

        total_active = sum(s.duration_sec for s in segments)
        unassigned = max(0.0, duration - total_active)

        logger.info(
            "Experiment segmentation: %.1fs video -> %d experiments (boundaries: %d)",
            duration, len(segments), len(boundaries),
        )

        return ExperimentSegmentation(
            video_duration_sec=duration,
            total_segments=len(segments),
            segments=segments,
            boundaries=boundaries,
            unassigned_time_sec=unassigned,
        )

    def _find_gap_candidates(self, activity_segments: List[ActivitySegment]) -> List[ExperimentBoundary]:
        """Level 1: Find gaps >= min_gap_sec between activity segments."""
        candidates = []
        sorted_segments = sorted(activity_segments, key=lambda s: s.start_sec)

        for i in range(1, len(sorted_segments)):
            gap = sorted_segments[i].start_sec - sorted_segments[i - 1].end_sec
            if gap >= self.config.min_gap_sec:
                confidence = min(1.0, 0.4 + (gap / self.config.min_gap_sec) * 0.3)
                candidates.append(ExperimentBoundary(
                    boundary_id=f"boundary_{i}",
                    position_sec=sorted_segments[i].start_sec,
                    gap_before_sec=gap,
                    confidence=confidence,
                    signals=["long_gap"],
                ))

        return candidates

    def _confirm_with_object_change(
        self,
        candidates: List[ExperimentBoundary],
        activity_segments: List[ActivitySegment],
        video_path: Path,
    ) -> List[ExperimentBoundary]:
        """Level 2: Check if object composition changes at boundary."""
        if not candidates:
            return candidates

        sorted_segments = sorted(activity_segments, key=lambda s: s.start_sec)

        for boundary in candidates:
            before_time = boundary.position_sec - boundary.gap_before_sec
            after_time = boundary.position_sec

            objects_before = self._detect_objects_at_time(video_path, before_time)
            objects_after = self._detect_objects_at_time(video_path, after_time)

            boundary.objects_before = list(objects_before)
            boundary.objects_after = list(objects_after)

            distance = self._jaccard_distance(objects_before, objects_after)
            if distance > self.config.object_change_threshold:
                boundary.confidence = min(1.0, boundary.confidence + 0.3)
                boundary.signals.append("object_change")

        return candidates

    def _detect_objects_at_time(self, video_path: Path, time_sec: float) -> Set[str]:
        """Run YOLO on a few frames around time_sec to get object set."""
        if not self.yolo_pipeline:
            return set()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return set()

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        objects: Set[str] = set()

        for i in range(self.config.boundary_sample_frames):
            t = time_sec + i * self.config.boundary_sample_interval_sec
            frame_idx = min(int(t * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            detections = self.yolo_pipeline._run_yolo(frame, frame_idx, t)
            for det in detections:
                objects.add(det.class_name)

        cap.release()
        return objects

    @staticmethod
    def _jaccard_distance(set_a: Set[str], set_b: Set[str]) -> float:
        """Jaccard distance: 1 - |intersection| / |union|."""
        if not set_a and not set_b:
            return 0.0
        union = set_a | set_b
        if not union:
            return 0.0
        intersection = set_a & set_b
        return 1.0 - len(intersection) / len(union)

    def _split_by_boundaries(
        self,
        activity_segments: List[ActivitySegment],
        boundaries: List[ExperimentBoundary],
        duration: float,
    ) -> List[ExperimentSegment]:
        """Split activity segments into experiment segments using boundaries."""
        if not boundaries:
            all_start = min(s.start_sec for s in activity_segments)
            all_end = max(s.end_sec for s in activity_segments)
            return [ExperimentSegment(
                segment_id="seg_0",
                index=0,
                start_sec=all_start,
                end_sec=all_end,
                duration_sec=all_end - all_start,
                activity_segments=list(activity_segments),
            )]

        sorted_segments = sorted(activity_segments, key=lambda s: s.start_sec)
        boundary_times = [b.position_sec for b in boundaries]

        experiment_groups: List[List[ActivitySegment]] = []
        current_group: List[ActivitySegment] = []

        boundary_idx = 0
        for seg in sorted_segments:
            while boundary_idx < len(boundary_times) and seg.start_sec >= boundary_times[boundary_idx]:
                if current_group:
                    experiment_groups.append(current_group)
                    current_group = []
                boundary_idx += 1
            current_group.append(seg)

        if current_group:
            experiment_groups.append(current_group)

        results: List[ExperimentSegment] = []
        for idx, group in enumerate(experiment_groups):
            if not group:
                continue
            start = min(s.start_sec for s in group)
            end = max(s.end_sec for s in group)
            boundary_before = boundaries[idx - 1] if idx > 0 and idx - 1 < len(boundaries) else None
            results.append(ExperimentSegment(
                segment_id=f"seg_{idx}",
                index=idx,
                start_sec=start,
                end_sec=end,
                duration_sec=end - start,
                activity_segments=group,
                boundary_before=boundary_before,
            ))

        return results

    def _merge_short_segments(self, segments: List[ExperimentSegment]) -> List[ExperimentSegment]:
        """Merge segments shorter than min_experiment_duration into neighbors."""
        if len(segments) <= 1:
            return segments

        merged: List[ExperimentSegment] = []
        for seg in segments:
            if seg.duration_sec < self.config.min_experiment_duration_sec and merged:
                prev = merged[-1]
                merged[-1] = ExperimentSegment(
                    segment_id=prev.segment_id,
                    index=prev.index,
                    start_sec=prev.start_sec,
                    end_sec=seg.end_sec,
                    duration_sec=seg.end_sec - prev.start_sec,
                    activity_segments=prev.activity_segments + seg.activity_segments,
                    boundary_before=prev.boundary_before,
                    display_name=prev.display_name,
                    scene_summary=prev.scene_summary,
                    naming_confidence=prev.naming_confidence,
                    naming_source=prev.naming_source,
                    preview_video_path=prev.preview_video_path,
                )
            else:
                merged.append(seg)

        # Re-index
        for i, seg in enumerate(merged):
            seg.index = i
            seg.segment_id = f"seg_{i}"

        return merged
