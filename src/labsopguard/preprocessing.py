from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2


@dataclass
class TimeAlignedRecord:
    timestamp_sec: float
    source_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingArtifact:
    aligned_text: List[TimeAlignedRecord]
    key_timestamps: List[float]
    video_index: List[Dict[str, Any]]
    detected_changes: List[Dict[str, Any]]
    video_streams: List[Dict[str, Any]] = field(default_factory=list)
    key_frames: List[Dict[str, Any]] = field(default_factory=list)
    key_clips: List[Dict[str, Any]] = field(default_factory=list)
    time_anchored_material_stream: List[Dict[str, Any]] = field(default_factory=list)
    alignment_summary: Dict[str, Any] = field(default_factory=dict)


class MultiModalPreprocessor:
    @staticmethod
    def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(min_value, min(max_value, value))

    @staticmethod
    def _env_float(name: str, default: float, *, min_value: float, max_value: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(min_value, min(max_value, value))

    def _build_video_streams(self, video_assets: Sequence[Any]) -> List[Dict[str, Any]]:
        streams: List[Dict[str, Any]] = []
        running_offset = 0.0
        has_explicit_offsets = any(
            getattr(asset, "metadata", {}).get("offset_source") == "explicit"
            for asset in (video_assets or [])
        )
        for index, asset in enumerate(video_assets or []):
            metadata = dict(getattr(asset, "metadata", {}) or {})
            duration = round(self._as_float(getattr(asset, "duration_sec", None), 0.0) or 0.0, 3)
            explicit_offset = self._as_float(metadata.get("start_offset_sec"))
            sync_profile = metadata.get("sync_profile") if isinstance(metadata.get("sync_profile"), dict) else {}
            clock_scale = self._as_float(metadata.get("clock_scale", sync_profile.get("clock_scale")), 1.0) or 1.0
            start_offset = round(explicit_offset if explicit_offset is not None else running_offset, 3)
            end_offset = round(start_offset + (duration * clock_scale), 3)
            streams.append(
                {
                    "video_index": index,
                    "video_asset_id": getattr(asset, "asset_id", None),
                    "filename": getattr(asset, "filename", ""),
                    "file_path": getattr(asset, "file_path", None),
                    "recorded_file_path": metadata.get("recorded_file_path"),
                    "fps": getattr(asset, "fps", None),
                    "frame_count": getattr(asset, "frame_count", None),
                    "duration_sec": duration,
                    "width": getattr(asset, "width", None),
                    "height": getattr(asset, "height", None),
                    "size_bytes": getattr(asset, "size_bytes", None),
                    "hash_sha256": getattr(asset, "hash_sha256", None),
                    "start_offset_sec": start_offset,
                    "end_offset_sec": end_offset,
                    "camera_id": metadata.get("camera_id"),
                    "sync_group": metadata.get("sync_group"),
                    "source_type": metadata.get("source_type"),
                    "ingest_mode": metadata.get("ingest_mode"),
                    "is_live_source": bool(metadata.get("is_live_source", False)),
                    "clock_scale": clock_scale,
                    "clock_drift_ppm": metadata.get("clock_drift_ppm", sync_profile.get("drift_ppm")),
                    "sync_profile": sync_profile,
                    "stream_health": metadata.get("stream_health") or {},
                    "offset_source": metadata.get("offset_source", "explicit" if explicit_offset is not None else "sequential"),
                }
            )
            running_offset = max(running_offset, end_offset) if has_explicit_offsets else end_offset
        return streams

    def align_text_records(
        self,
        duration_sec: float,
        context_text: str,
        protocol_text: str,
        context_records: Optional[Sequence[Any]] = None,
        video_streams: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[TimeAlignedRecord]:
        aligned: List[TimeAlignedRecord] = []
        records = list(context_records or [])
        streams = list(video_streams or [])
        total_duration = round(max(duration_sec, 0.0), 3)
        untimed_records: List[Any] = []

        def stream_transform(stream: Dict[str, Any]) -> tuple[float, float]:
            sync_profile = stream.get("sync_profile") if isinstance(stream.get("sync_profile"), dict) else {}
            offset = self._as_float(sync_profile.get("offset_sec"), self._as_float(stream.get("start_offset_sec"), 0.0)) or 0.0
            scale = self._as_float(sync_profile.get("clock_scale"), self._as_float(stream.get("clock_scale"), 1.0)) or 1.0
            return offset, scale

        stream_transforms_by_index = {
            stream.get("video_index"): stream_transform(stream)
            for stream in streams
        }
        stream_transforms_by_asset = {
            stream.get("video_asset_id"): stream_transform(stream)
            for stream in streams
            if stream.get("video_asset_id")
        }

        for record in records:
            metadata = dict(getattr(record, "metadata", {}) or {})
            timestamp = self._as_float(getattr(record, "timestamp_sec", None))
            start_time = self._as_float(getattr(record, "start_time_sec", None))
            local_timestamp = self._as_float(metadata.get("local_timestamp_sec"))
            if timestamp is None and start_time is not None:
                timestamp = start_time
            if timestamp is None and local_timestamp is not None:
                anchor_video_index = getattr(record, "anchor_video_index", None)
                anchor_video_asset_id = getattr(record, "anchor_video_asset_id", None)
                transform = stream_transforms_by_index.get(anchor_video_index)
                if transform is None and anchor_video_asset_id:
                    transform = stream_transforms_by_asset.get(anchor_video_asset_id)
                if transform is not None:
                    base_offset, clock_scale = transform
                    timestamp = base_offset + (local_timestamp * clock_scale)
            if timestamp is None:
                untimed_records.append(record)
                continue
            aligned.append(
                TimeAlignedRecord(
                    timestamp_sec=round(max(timestamp, 0.0), 3),
                    source_type=getattr(record, "source_type", "context"),
                    content=getattr(record, "content", ""),
                    metadata=metadata,
                )
            )

        if untimed_records:
            if total_duration <= 0 and streams:
                total_duration = round(max((stream.get("end_offset_sec", 0.0) for stream in streams), default=0.0), 3)
            step = total_duration / max(len(untimed_records), 1) if total_duration else 0.0
            for index, record in enumerate(untimed_records):
                aligned.append(
                    TimeAlignedRecord(
                        timestamp_sec=round(index * step, 3),
                        source_type=getattr(record, "source_type", "context"),
                        content=getattr(record, "content", ""),
                        metadata=dict(getattr(record, "metadata", {}) or {}),
                    )
                )

        if context_text and not aligned:
            chunks = [part.strip() for part in context_text.splitlines() if part.strip()]
            if not chunks:
                chunks = [context_text.strip()]
            step = total_duration / max(len(chunks), 1) if total_duration else 0.0
            for index, text in enumerate(chunks):
                aligned.append(
                    TimeAlignedRecord(
                        timestamp_sec=round(index * step, 3),
                        source_type="context",
                        content=text,
                    )
                )

        if protocol_text:
            aligned.append(
                TimeAlignedRecord(
                    timestamp_sec=0.0,
                    source_type="protocol",
                    content=protocol_text,
                    metadata={"anchored_to": "global_start"},
                )
            )

        aligned.sort(key=lambda item: (item.timestamp_sec, item.source_type, item.content))
        return aligned

    @staticmethod
    def _sanitize_filename(name: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(name))

    def _build_video_index(
        self,
        material_stream: Sequence[Any],
        video_streams: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        stream_by_asset = {
            stream.get("video_asset_id"): stream
            for stream in (video_streams or [])
            if stream.get("video_asset_id")
        }
        index_payload: List[Dict[str, Any]] = []
        for item in material_stream:
            stream_info = stream_by_asset.get(getattr(item, "media_asset_id", None), {})
            index_payload.append(
                {
                    "schema_version": getattr(item, "schema_version", "material_stream.v1"),
                    "item_id": getattr(item, "item_id", ""),
                    "video_asset_id": getattr(item, "media_asset_id", None),
                    "stream_id": getattr(item, "stream_id", None),
                    "camera_id": stream_info.get("camera_id"),
                    "sync_group": stream_info.get("sync_group"),
                    "stream_start_offset_sec": stream_info.get("start_offset_sec"),
                    "stream_end_offset_sec": stream_info.get("end_offset_sec"),
                    "timestamp_sec": round(self._as_float(getattr(item, "timestamp_sec", 0.0), 0.0) or 0.0, 3),
                    "local_timestamp_sec": round(self._as_float(getattr(item, "local_timestamp_sec", 0.0), 0.0) or 0.0, 3),
                    "frame_id": getattr(item, "frame_id", None),
                    "local_frame_id": getattr(item, "local_frame_id", None),
                    "frame_bgr_path": getattr(item, "frame_bgr_path", None),
                    "scene_description": getattr(item, "scene_description", ""),
                    "transcript_segment": getattr(item, "transcript_segment", None),
                    "conversation_context": getattr(item, "conversation_context", None),
                    "linked_context_event_ids": getattr(item, "linked_context_event_ids", []),
                    "detected_objects": getattr(item, "detected_objects", []),
                    "object_labels": getattr(item, "object_labels", []),
                    "detected_activities": getattr(item, "detected_activities", []),
                    "is_key_frame": bool(getattr(item, "is_key_frame", False)),
                    "key_frame_reason": getattr(item, "key_frame_reason", None),
                    "change_score": round(self._as_float(getattr(item, "change_score", 0.0), 0.0) or 0.0, 4),
                    "clip_id": getattr(item, "clip_id", None),
                    "analysis": getattr(item, "analysis", {}) or {},
                }
            )
        index_payload.sort(key=lambda entry: (entry["timestamp_sec"], entry["frame_id"] or 0))
        return index_payload

    def _build_detected_changes(self, physical_events: Sequence[Any]) -> List[Dict[str, Any]]:
        changes: List[Dict[str, Any]] = []
        for event in physical_events:
            metadata = getattr(event, "metadata", {}) or {}
            changes.append(
                {
                    "event_id": getattr(event, "event_id", ""),
                    "event_type": getattr(event, "event_type", ""),
                    "timestamp_sec": round(self._as_float(getattr(event, "timestamp_sec", 0.0), 0.0) or 0.0, 3),
                    "end_timestamp_sec": self._as_float(getattr(event, "end_timestamp_sec", None)),
                    "confidence": round(self._as_float(getattr(event, "confidence", 0.0), 0.0) or 0.0, 4),
                    "location": getattr(event, "location", None),
                    "video_asset_id": metadata.get("video_asset_id"),
                    "camera_id": metadata.get("camera_id"),
                    "stream_id": metadata.get("stream_id"),
                    "clip_id": metadata.get("clip_id"),
                    "metadata": metadata,
                }
            )
        changes.sort(key=lambda entry: (entry["timestamp_sec"], entry["event_type"]))
        return changes

    def _build_key_frames(self, video_index: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not video_index:
            return []

        def stream_key(entry: Dict[str, Any]) -> str:
            return str(entry.get("stream_id") or entry.get("video_asset_id") or entry.get("camera_id") or "default")

        def entry_key(entry: Dict[str, Any]) -> tuple:
            return (
                entry.get("video_asset_id"),
                entry.get("stream_id"),
                entry.get("frame_id"),
                entry.get("timestamp_sec"),
            )

        def rank_key(entry: Dict[str, Any]) -> tuple:
            reason = str(entry.get("key_frame_reason") or "")
            semantic_bonus = 1 if reason in {"activity_shift", "object_state_change", "visual_change"} else 0
            object_bonus = len(entry.get("object_labels") or []) + len(entry.get("detected_objects") or [])
            activity_bonus = len(entry.get("detected_activities") or [])
            return (
                self._as_float(entry.get("change_score"), 0.0) or 0.0,
                semantic_bonus,
                object_bonus,
                activity_bonus,
                self._as_float(entry.get("timestamp_sec"), 0.0) or 0.0,
            )

        def supplemental(entry: Dict[str, Any], reason: str) -> Dict[str, Any]:
            item = dict(entry)
            item["is_key_frame"] = True
            item["key_frame_reason"] = item.get("key_frame_reason") or reason
            frame_id = item.get("frame_id")
            ts = str(item.get("timestamp_sec", "0")).replace(".", "p")
            item["clip_id"] = f"{stream_key(item)}:supplement:{frame_id if frame_id is not None else ts}"
            return item

        min_per_stream = self._env_int("LABSOPGUARD_PREPROCESS_KEYFRAME_MIN_PER_STREAM", 3, min_value=0, max_value=60)
        fallback_limit = self._env_int("LABSOPGUARD_PREPROCESS_FALLBACK_KEYFRAME_LIMIT", 12, min_value=1, max_value=240)
        by_stream: Dict[str, List[Dict[str, Any]]] = {}
        for entry in video_index:
            by_stream.setdefault(stream_key(entry), []).append(entry)

        selected: List[Dict[str, Any]] = []
        selected_keys: set[tuple] = set()
        explicit_any = any(entry.get("is_key_frame") for entry in video_index)

        for entries in by_stream.values():
            explicit = [entry for entry in entries if entry.get("is_key_frame")]
            for entry in explicit:
                key = entry_key(entry)
                if key not in selected_keys:
                    selected.append(entry)
                    selected_keys.add(key)

            target = min(max(min_per_stream, len(explicit)), len(entries))
            if target <= len(explicit):
                continue
            for candidate in sorted(entries, key=rank_key, reverse=True):
                key = entry_key(candidate)
                if key in selected_keys:
                    continue
                reason = "change_peak_supplement" if (self._as_float(candidate.get("change_score"), 0.0) or 0.0) > 0 else "coverage_supplement"
                selected.append(supplemental(candidate, reason))
                selected_keys.add(key)
                if sum(1 for item in selected if stream_key(item) == stream_key(candidate)) >= target:
                    break

        if not explicit_any and len(selected) < min(fallback_limit, len(video_index)):
            for candidate in sorted(video_index, key=rank_key, reverse=True):
                key = entry_key(candidate)
                if key in selected_keys:
                    continue
                reason = "change_peak_supplement" if (self._as_float(candidate.get("change_score"), 0.0) or 0.0) > 0 else "coverage_supplement"
                selected.append(supplemental(candidate, reason))
                selected_keys.add(key)
                if len(selected) >= min(fallback_limit, len(video_index)):
                    break

        selected.sort(key=lambda entry: (self._as_float(entry.get("timestamp_sec"), 0.0) or 0.0, entry.get("frame_id") or 0))
        return selected

    @staticmethod
    def _clip_window_for_key_frame(frame: Dict[str, Any], sample_interval_sec: float) -> float:
        base = max(
            float(sample_interval_sec or 0.0),
            MultiModalPreprocessor._env_float("LABSOPGUARD_KEY_CLIP_MIN_CONTEXT_SEC", 1.5, min_value=0.1, max_value=60.0),
        )
        max_window = max(
            base,
            MultiModalPreprocessor._env_float("LABSOPGUARD_KEY_CLIP_MAX_CONTEXT_SEC", 6.0, min_value=base, max_value=300.0),
        )
        reason = str(frame.get("key_frame_reason") or "")
        change_score = float(frame.get("change_score") or 0.0)
        multiplier = 1.0
        if reason in {"activity_shift", "object_state_change", "change_peak_supplement"}:
            multiplier = max(multiplier, 1.45)
        high_change_score = MultiModalPreprocessor._env_float(
            "LABSOPGUARD_KEY_CLIP_HIGH_CHANGE_SCORE", 0.055, min_value=0.0, max_value=1.0
        )
        if reason == "visual_change" or change_score >= high_change_score:
            multiplier = max(multiplier, 1.25)
        return round(min(max_window, base * multiplier), 3)

    def _build_key_clips(
        self,
        key_frames: Sequence[Dict[str, Any]],
        video_streams: Optional[Sequence[Dict[str, Any]]] = None,
        sample_interval_sec: float = 2.0,
        clip_output_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        stream_by_asset = {
            stream.get("video_asset_id"): stream
            for stream in (video_streams or [])
            if stream.get("video_asset_id")
        }
        clips: List[Dict[str, Any]] = []
        for frame in key_frames:
            timestamp = self._as_float(frame.get("timestamp_sec"), 0.0) or 0.0
            local_timestamp = self._as_float(frame.get("local_timestamp_sec"), timestamp) or timestamp
            stream_info = stream_by_asset.get(frame.get("video_asset_id"), {})
            duration_sec = self._as_float(stream_info.get("duration_sec"), None)
            window_sec = self._clip_window_for_key_frame(frame, sample_interval_sec)
            global_start = round(max(0.0, timestamp - window_sec), 3)
            global_end = round(timestamp + window_sec, 3)
            local_start = round(max(0.0, local_timestamp - window_sec), 3)
            local_end = round(local_timestamp + window_sec, 3)
            if duration_sec is not None:
                local_end = round(min(local_end, duration_sec), 3)
                stream_end = self._as_float(stream_info.get("end_offset_sec"), global_end)
                if stream_end is not None:
                    global_end = round(min(global_end, stream_end), 3)

            clip_id = frame.get("clip_id") or f"{frame.get('video_asset_id') or 'video'}:{frame.get('frame_id')}"
            clips.append(
                {
                    "clip_id": clip_id,
                    "video_asset_id": frame.get("video_asset_id"),
                    "stream_id": frame.get("stream_id"),
                    "camera_id": stream_info.get("camera_id"),
                    "sync_group": stream_info.get("sync_group"),
                    "anchor_timestamp_sec": round(timestamp, 3),
                    "local_anchor_timestamp_sec": round(local_timestamp, 3),
                    "start_time_sec": global_start,
                    "end_time_sec": global_end,
                    "local_start_time_sec": local_start,
                    "local_end_time_sec": local_end,
                    "key_frame_id": frame.get("frame_id"),
                    "key_frame_path": frame.get("frame_bgr_path"),
                    "source_path": stream_info.get("recorded_file_path") or stream_info.get("file_path"),
                    "reason": frame.get("key_frame_reason") or ("change_peak" if frame.get("change_score", 0.0) > 0 else "anchor"),
                    "window_sec": window_sec,
                    "file_path": None,
                    "file_exists": False,
                    "render_status": "virtual",
                }
            )
            clip_record = clips[-1]
            if clip_output_dir is not None:
                rendered_path = self._materialize_clip(
                    source_path=stream_info.get("recorded_file_path") or stream_info.get("file_path"),
                    clip_id=clip_id,
                    clip_output_dir=clip_output_dir,
                    start_time_sec=local_start,
                    end_time_sec=local_end,
                    fps_hint=self._as_float(stream_info.get("fps"), None),
                )
                if rendered_path is not None:
                    clip_record["file_path"] = rendered_path
                    clip_record["file_exists"] = True
                    clip_record["render_status"] = "rendered"
        clips.sort(key=lambda clip: (clip["start_time_sec"], clip["clip_id"]))
        return clips

    def _materialize_clip(
        self,
        source_path: Optional[str],
        clip_id: str,
        clip_output_dir: Path,
        start_time_sec: float,
        end_time_sec: float,
        fps_hint: Optional[float] = None,
    ) -> Optional[str]:
        if not source_path:
            return None

        source = Path(source_path)
        if not source.exists() or not source.is_file():
            return None

        clip_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = clip_output_dir / f"{self._sanitize_filename(clip_id)}.mp4"
        if output_path.exists():
            return str(output_path)

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            return None

        writer = None
        try:
            fps = self._as_float(capture.get(cv2.CAP_PROP_FPS), fps_hint or 30.0) or fps_hint or 30.0
            start_frame = max(0, int(start_time_sec * fps))
            end_frame = max(start_frame + 1, int(end_time_sec * fps))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0:
                end_frame = min(end_frame, total_frames)

            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ok, frame = capture.read()
            if not ok or frame is None:
                return None

            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                return None

            current_frame = start_frame
            while ok and frame is not None and current_frame < end_frame:
                writer.write(frame)
                current_frame += 1
                if current_frame >= end_frame:
                    break
                ok, frame = capture.read()

            return str(output_path) if output_path.exists() else None
        finally:
            if writer is not None:
                writer.release()
            capture.release()

    def build_artifact(
        self,
        duration_sec: float,
        context_text: str,
        protocol_text: str,
        physical_events: List[Any],
        material_stream: List[Any],
        context_records: Optional[Sequence[Any]] = None,
        video_assets: Optional[Sequence[Any]] = None,
        clip_output_dir: Optional[Path] = None,
        clip_window_sec: float = 2.0,
    ) -> PreprocessingArtifact:
        video_streams = self._build_video_streams(video_assets or [])
        aligned = self.align_text_records(
            duration_sec=duration_sec,
            context_text=context_text,
            protocol_text=protocol_text,
            context_records=context_records,
            video_streams=video_streams,
        )
        video_index = self._build_video_index(material_stream, video_streams=video_streams)
        detected_changes = self._build_detected_changes(physical_events)
        key_frames = self._build_key_frames(video_index)
        key_clips = self._build_key_clips(
            key_frames,
            video_streams=video_streams,
            sample_interval_sec=clip_window_sec,
            clip_output_dir=clip_output_dir,
        )
        time_anchored_material_stream = list(video_index)
        key_timestamps = sorted(
            {
                0.0,
                round(duration_sec, 3) if duration_sec else 0.0,
                *[round(item.timestamp_sec, 3) for item in aligned],
                *[round(entry.get("timestamp_sec", 0.0), 3) for entry in video_index],
                *[round(entry.get("timestamp_sec", 0.0), 3) for entry in detected_changes],
            }
        )
        timed_context_records = sum(1 for record in aligned if record.metadata.get("anchored_to") != "global_start")
        stream_ranges = [
            {
                "video_asset_id": stream.get("video_asset_id"),
                "camera_id": stream.get("camera_id"),
                "start_offset_sec": stream.get("start_offset_sec"),
                "end_offset_sec": stream.get("end_offset_sec"),
                "offset_source": stream.get("offset_source"),
                "clock_drift_ppm": stream.get("clock_drift_ppm"),
                "sync_residual_error_sec": (stream.get("sync_profile") or {}).get("residual_error_sec"),
                "stream_health": stream.get("stream_health") or {},
            }
            for stream in video_streams
        ]

        has_calibrated = any(stream.get("offset_source") == "calibrated" for stream in video_streams)
        has_explicit = any(stream.get("offset_source") == "explicit" for stream in video_streams)
        alignment_summary = {
            "video_count": len(video_streams),
            "context_record_count": len(aligned),
            "timed_context_record_count": timed_context_records,
            "untimed_context_record_count": max(len(aligned) - timed_context_records, 0),
            "key_frame_count": len(key_frames),
            "key_clip_count": len(key_clips),
            "materialized_key_clip_count": sum(1 for clip in key_clips if clip.get("file_exists")),
            "change_event_count": len(detected_changes),
            "anchor_strategy": "calibrated" if has_calibrated else ("explicit_offsets" if has_explicit else "sequential"),
            "stream_ranges": stream_ranges,
            "max_sync_residual_error_sec": max(
                (
                    self._as_float((stream.get("sync_profile") or {}).get("residual_error_sec"), 0.0) or 0.0
                    for stream in video_streams
                ),
                default=0.0,
            ),
            "stream_health": [stream.get("stream_health") or {} for stream in video_streams],
            "global_start_sec": key_timestamps[0] if key_timestamps else 0.0,
            "global_end_sec": key_timestamps[-1] if key_timestamps else round(duration_sec, 3),
        }

        return PreprocessingArtifact(
            aligned_text=aligned,
            key_timestamps=key_timestamps,
            video_index=video_index,
            detected_changes=detected_changes,
            video_streams=video_streams,
            key_frames=key_frames,
            key_clips=key_clips,
            time_anchored_material_stream=time_anchored_material_stream,
            alignment_summary=alignment_summary,
        )
