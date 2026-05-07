from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

from labsopguard.time_sync import SyncAnchor


@dataclass
class SemanticFrameObservation:
    camera_id: str
    view_type: str = ""
    source_group: str = ""
    local_timestamp_sec: float = 0.0
    frame_id: Optional[int] = None
    scene_description: str = ""
    object_labels: List[str] = field(default_factory=list)
    detected_activities: List[str] = field(default_factory=list)
    transcript_segment: Optional[str] = None
    confidence: float = 0.5


class MultimodalSemanticSyncEngine:
    """Deterministic semantic event matcher for cross-view experiment sync.

    The engine consumes already-produced frame semantics. It does not call a
    remote model; VLM/API output can be fed in through `scene_description`,
    `object_labels`, `detected_activities`, and transcript text.
    """

    EVENT_RULES: Dict[str, Dict[str, Any]] = {
        "experiment_start": {
            "description": "Experiment begins or operator starts setup",
            "keywords": ["start", "begin", "setup", "prepare", "prepares", "enter", "enters", "开始", "准备", "进入"],
        },
        "ppe_action": {
            "description": "Operator adjusts or wears PPE",
            "keywords": ["glove", "gloves", "lab coat", "ppe", "protective", "手套", "实验服", "防护"],
        },
        "tool_pickup": {
            "description": "Operator picks up or reaches for a tool",
            "keywords": ["pick", "pickup", "picks up", "take", "grasp", "hold", "pipette", "spatula", "拿起", "取", "抓取"],
        },
        "container_placement": {
            "description": "Container is placed or moved on the bench",
            "keywords": ["place", "placed", "put", "set down", "beaker", "tube", "container", "放置", "放到", "烧杯", "试管", "容器"],
        },
        "weighing": {
            "description": "Sample weighing or balance use",
            "keywords": ["weigh", "weighing", "balance", "scale", "称量", "天平", "质量"],
        },
        "liquid_transfer": {
            "description": "Liquid transfer or pipetting",
            "keywords": ["transfer", "liquid", "pipette", "pour", "drop", "add", "加液", "移液", "滴加", "倒入", "液体"],
        },
        "stirring": {
            "description": "Stirring or mixing",
            "keywords": ["stir", "stirring", "mix", "mixing", "shake", "搅拌", "混合", "摇匀"],
        },
        "experiment_end": {
            "description": "Experiment ends or operator finishes",
            "keywords": ["finish", "finished", "complete", "completed", "end", "cleanup", "完成", "结束", "清理"],
        },
    }

    REFERENCE_VIEW_PRIORITY = {
        "first_person": 4,
        "front": 4,
        "operator": 4,
        "third_person": 3,
        "overview": 2,
    }

    @staticmethod
    def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _text_from_values(values: Iterable[Any]) -> str:
        chunks: List[str] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, list):
                chunks.extend(str(item) for item in value if item is not None)
            elif isinstance(value, dict):
                chunks.extend(str(item) for item in value.values() if item is not None)
            else:
                chunks.append(str(value))
        return " ".join(chunks).lower()

    @classmethod
    def observation_from_item(cls, item: Any) -> Optional[SemanticFrameObservation]:
        data = item.to_dict() if hasattr(item, "to_dict") else dict(item) if isinstance(item, dict) else {}
        if not data:
            return None
        camera_id = str(data.get("camera_id") or data.get("stream_id") or "").strip()
        if not camera_id:
            return None
        local_timestamp = cls._as_float(data.get("local_timestamp_sec"), cls._as_float(data.get("timestamp_sec"), 0.0))
        if local_timestamp is None:
            return None
        return SemanticFrameObservation(
            camera_id=camera_id,
            view_type=str(data.get("view_type") or ""),
            source_group=str(data.get("source_group") or ""),
            local_timestamp_sec=float(local_timestamp),
            frame_id=data.get("local_frame_id", data.get("frame_id")),
            scene_description=str(data.get("scene_description") or data.get("description") or ""),
            object_labels=[str(item) for item in (data.get("object_labels") or []) if item is not None],
            detected_activities=[str(item) for item in (data.get("detected_activities") or []) if item is not None],
            transcript_segment=data.get("transcript_segment"),
            confidence=float(cls._as_float(data.get("confidence"), 0.5) or 0.5),
        )

    @classmethod
    def _infer_event_matches(cls, observation: SemanticFrameObservation) -> List[Dict[str, Any]]:
        text = cls._text_from_values(
            [
                observation.scene_description,
                observation.object_labels,
                observation.detected_activities,
                observation.transcript_segment,
            ]
        )
        matches: List[Dict[str, Any]] = []
        for event_type, rule in cls.EVENT_RULES.items():
            matched_keywords = [keyword for keyword in rule["keywords"] if keyword.lower() in text]
            if not matched_keywords:
                continue
            keyword_score = min(0.35, len(matched_keywords) * 0.08)
            evidence_score = min(0.2, (len(observation.object_labels) + len(observation.detected_activities)) * 0.03)
            confidence = max(0.0, min(1.0, 0.42 + keyword_score + evidence_score + observation.confidence * 0.18))
            matches.append(
                {
                    "event_type": event_type,
                    "description": rule["description"],
                    "confidence": round(confidence, 4),
                    "evidence": ", ".join(matched_keywords[:5]) or rule["description"],
                }
            )
        return matches

    @classmethod
    def _choose_reference_stream(
        cls,
        observations: Sequence[SemanticFrameObservation],
        event_candidates: Dict[str, Dict[str, Dict[str, Any]]],
        requested_reference: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        cameras = sorted({obs.camera_id for obs in observations})
        if not cameras:
            return None
        if requested_reference and requested_reference in cameras:
            selected = requested_reference
        else:
            view_by_camera: Dict[str, str] = {}
            group_by_camera: Dict[str, str] = {}
            for obs in observations:
                view_by_camera.setdefault(obs.camera_id, obs.view_type)
                group_by_camera.setdefault(obs.camera_id, obs.source_group)
            event_count_by_camera = {
                camera_id: sum(1 for per_camera in event_candidates.values() if camera_id in per_camera)
                for camera_id in cameras
            }
            selected = max(
                cameras,
                key=lambda camera_id: (
                    event_count_by_camera.get(camera_id, 0),
                    cls.REFERENCE_VIEW_PRIORITY.get(str(view_by_camera.get(camera_id) or "").lower(), 0),
                    camera_id,
                ),
            )
        selected_obs = next((obs for obs in observations if obs.camera_id == selected), None)
        return {
            "camera_id": selected,
            "reason": "chosen because it has the clearest complete experiment timeline",
            "view_type": selected_obs.view_type if selected_obs else "",
            "source_group": selected_obs.source_group if selected_obs else "",
        }

    @staticmethod
    def _offset_stats(anchors: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        by_camera: Dict[str, List[float]] = {}
        for anchor in anchors:
            try:
                offset = float(anchor["reference_time_sec"]) - float(anchor["local_time_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            by_camera.setdefault(str(anchor.get("camera_id")), []).append(offset)
        offset_by_camera = {camera: round(float(median(offsets)), 6) for camera, offsets in by_camera.items() if offsets}
        residuals: List[float] = []
        for anchor in anchors:
            camera = str(anchor.get("camera_id"))
            if camera not in offset_by_camera:
                continue
            residuals.append(abs((float(anchor["local_time_sec"]) + offset_by_camera[camera]) - float(anchor["reference_time_sec"])))
        residual = round(float(sum(residuals) / len(residuals)), 6) if residuals else 0.0
        return {
            "anchor_count": len(anchors),
            "estimated_offset_sec_by_camera": offset_by_camera,
            "estimated_residual_error_sec": residual,
            "notes": [],
        }

    @classmethod
    def build(
        cls,
        *,
        experiment_id: str,
        run_id: str,
        frame_items: Sequence[Any],
        reference_camera_id: Optional[str] = None,
        min_anchor_confidence: float = 0.55,
    ) -> Dict[str, Any]:
        observations = [obs for obs in (cls.observation_from_item(item) for item in frame_items) if obs is not None]
        event_candidates: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for observation in observations:
            for match in cls._infer_event_matches(observation):
                event_type = match["event_type"]
                per_camera = event_candidates.setdefault(event_type, {})
                current = per_camera.get(observation.camera_id)
                candidate = {
                    **match,
                    "camera_id": observation.camera_id,
                    "view_type": observation.view_type,
                    "source_group": observation.source_group,
                    "local_timestamp_sec": round(observation.local_timestamp_sec, 3),
                    "frame_id": observation.frame_id,
                }
                if current is None or (candidate["confidence"], -candidate["local_timestamp_sec"]) > (current["confidence"], -current["local_timestamp_sec"]):
                    per_camera[observation.camera_id] = candidate

        reference_stream = cls._choose_reference_stream(observations, event_candidates, reference_camera_id)
        semantic_events: List[Dict[str, Any]] = []
        sync_anchors: List[Dict[str, Any]] = []
        warnings: List[str] = []
        if reference_stream is None:
            warnings.append("no_stream_observations")
        else:
            reference_camera = reference_stream["camera_id"]
            event_index = 1
            all_cameras = sorted({obs.camera_id for obs in observations})
            for event_type, per_camera in sorted(event_candidates.items()):
                if len(per_camera) < 2:
                    continue
                event_id = f"sem_evt_{event_index:04d}"
                event_index += 1
                matched = []
                for camera_id, candidate in sorted(per_camera.items()):
                    matched.append(
                        {
                            "camera_id": camera_id,
                            "view_type": candidate.get("view_type"),
                            "local_timestamp_sec": candidate["local_timestamp_sec"],
                            "frame_id": candidate.get("frame_id"),
                            "evidence": candidate.get("evidence") or candidate.get("description"),
                            "confidence": candidate["confidence"],
                        }
                    )
                missing = [camera_id for camera_id in all_cameras if camera_id not in per_camera]
                event_confidence = round(min(item["confidence"] for item in matched), 4)
                semantic_events.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "description": cls.EVENT_RULES[event_type]["description"],
                        "confidence": event_confidence,
                        "matched_streams": matched,
                        "missing_streams": missing,
                    }
                )
                ref_candidate = per_camera.get(reference_camera)
                if not ref_candidate:
                    continue
                for camera_id, candidate in sorted(per_camera.items()):
                    if camera_id == reference_camera:
                        continue
                    confidence = round(min(candidate["confidence"], ref_candidate["confidence"]), 4)
                    if confidence < min_anchor_confidence:
                        continue
                    sync_anchors.append(
                        {
                            "camera_id": camera_id,
                            "reference_camera_id": reference_camera,
                            "local_time_sec": candidate["local_timestamp_sec"],
                            "reference_time_sec": ref_candidate["local_timestamp_sec"],
                            "method": "multimodal_semantic",
                            "confidence": confidence,
                            "event_id": event_id,
                        }
                    )

        status = "insufficient_overlap"
        if len(sync_anchors) >= 2:
            status = "calibrated"
        elif sync_anchors:
            status = "partial"
        if not sync_anchors and semantic_events:
            warnings.append("semantic_events_found_but_no_reference_overlap")
        elif not semantic_events:
            warnings.append("insufficient_overlap")

        return {
            "schema_version": "multimodal_semantic_sync.v1",
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": status,
            "reference_stream": reference_stream,
            "semantic_events": semantic_events,
            "sync_anchors": sync_anchors,
            "alignment_quality": cls._offset_stats(sync_anchors),
            "warnings": warnings,
        }

    @staticmethod
    def anchors_as_sync_anchors(result: Dict[str, Any]) -> List[SyncAnchor]:
        anchors: List[SyncAnchor] = []
        for item in result.get("sync_anchors") or []:
            try:
                anchors.append(
                    SyncAnchor(
                        camera_id=str(item["camera_id"]),
                        local_time_sec=float(item["local_time_sec"]),
                        reference_time_sec=float(item["reference_time_sec"]),
                        method=str(item.get("method") or "multimodal_semantic"),
                        confidence=float(item.get("confidence", 0.55)),
                        metadata={k: v for k, v in item.items() if k not in {"camera_id", "local_time_sec", "reference_time_sec", "method", "confidence"}},
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return anchors


def build_multimodal_semantic_sync(**kwargs: Any) -> Dict[str, Any]:
    return MultimodalSemanticSyncEngine.build(**kwargs)
