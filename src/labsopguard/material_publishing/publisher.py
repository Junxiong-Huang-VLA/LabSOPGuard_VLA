from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .archive_planner import ArchivePlan, ArchivePlanner
from .naming import container_label, display_name as make_display_name, searchable_text, stable_name as make_stable_name
from .schemas import MATERIAL_PUBLISH_VERSION, MaterialPublishRecord, PublishedPaths, read_json, write_json
from .semantic_enhancer import DisplayNameEnhancer, QwenVlmDisplayNameEnhancer
from .upload_manifest import write_upload_manifest


class SemanticMaterialPublisher:
    """Publish canonical event asset packs into a friendly derived archive layer."""

    def __init__(
        self,
        experiment_dir: str | Path,
        *,
        experiment_id: Optional[str] = None,
        copy_strategy: str = "hardlink_or_copy",
        display_name_enhancer: Optional[DisplayNameEnhancer] = None,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.experiment_id = experiment_id or self.experiment_dir.name
        self.copy_strategy = copy_strategy
        self.planner = ArchivePlanner(self.experiment_dir)
        self.display_name_enhancer = display_name_enhancer or QwenVlmDisplayNameEnhancer()

    def publish(self) -> Dict[str, Any]:
        events = self._load_events()
        if not events:
            raise FileNotFoundError("No event assets found under materials/events")
        events = self._ensure_event_assets(events)
        experiment_name = self._experiment_name()
        records: List[MaterialPublishRecord] = []
        for event, event_json_path, payload in events:
            record = self._publish_event(event, event_json_path, payload, experiment_name)
            records.append(record)
        manifest = write_upload_manifest(self.experiment_dir, self.experiment_id, records)
        self._write_index(records)
        official_updates = self._update_official_steps(records)
        published_payload = {
            "schema_version": "published_materials.v1",
            "experiment_id": self.experiment_id,
            "total": len(records),
            "items": [record.to_dict() for record in records],
            "upload_manifest_path": str((self.experiment_dir / "upload_manifest.json").resolve()),
            "official_step_updates": official_updates,
        }
        write_json(self.experiment_dir / "published_materials.json", published_payload)
        return {"published_materials": published_payload, "upload_manifest": manifest}

    def list_published(self) -> Dict[str, Any]:
        payload = read_json(self.experiment_dir / "published_materials.json", None)
        if payload:
            return payload
        items = []
        for path in sorted((self.experiment_dir / "published_materials").glob("*/*/*/material_publish.json")):
            item = read_json(path, {})
            if item:
                items.append(item)
        return {"schema_version": "published_materials.v1", "experiment_id": self.experiment_id, "total": len(items), "items": items}

    def _load_events(self) -> List[Tuple[Dict[str, Any], Path, Dict[str, Any]]]:
        events_dir = self.experiment_dir / "materials" / "events"
        found: List[Tuple[Dict[str, Any], Path, Dict[str, Any]]] = []
        known_event_ids: set[str] = set()
        if events_dir.exists():
            for event_json_path in sorted(events_dir.glob("*/event.json")):
                payload = read_json(event_json_path, {})
                event = payload.get("event") if isinstance(payload, dict) else None
                if isinstance(event, dict):
                    event_id = str(event.get("event_id") or event_json_path.parent.name)
                    event["event_id"] = event_id
                    known_event_ids.add(event_id)
                    found.append((event, event_json_path, payload))
        physical = read_json(self.experiment_dir / "physical_events.json", {})
        if isinstance(physical, dict):
            physical_events = physical.get("events") or physical.get("physical_events") or []
        elif isinstance(physical, list):
            physical_events = physical
        else:
            physical_events = []
        seen_ids: set[str] = set(known_event_ids)
        for idx, event in enumerate(physical_events):
            if not isinstance(event, dict):
                continue
            event_id = event.get("event_id") or event.get("id")
            if event_id and str(event_id) in known_event_ids:
                continue
            if not event_id:
                start = self._event_start_time(event)
                event_id = self._safe_event_id(f"event_{start:.3f}", idx, seen_ids)
            else:
                event_id = self._safe_event_id(event_id, idx, seen_ids)
            event["event_id"] = str(event_id)
            event.setdefault("experiment_id", self.experiment_id)
            event_json_path = events_dir / str(event_id) / "event.json"
            found.append((event, event_json_path, {"event": event, "asset_pack": event.get("asset_pack") or {}}))
        if found:
            return found
        fallback = self._load_events_from_material_stream()
        if fallback:
            self._write_fallback_physical_events([event for event, _, _ in fallback])
        return fallback

    def _ensure_event_assets(self, events: List[Tuple[Dict[str, Any], Path, Dict[str, Any]]]) -> List[Tuple[Dict[str, Any], Path, Dict[str, Any]]]:
        """Repair historical/fallback event records before publishing.

        Older fallback records may have valid event metadata and keyframes but no
        clip.  A published material without a clip is not useful in the
        workspace, so cut a browser-playable event clip directly from the
        experiment source video and write it back into event.json/physical_events.
        """
        source_video = self._source_video_path()
        repaired: List[Tuple[Dict[str, Any], Path, Dict[str, Any]]] = []
        updated_events: Dict[str, Dict[str, Any]] = {}
        force_yolo_repair = os.environ.get("LABSOPGUARD_FORCE_YOLO_REPAIR") == "1"
        material_items = self._material_items_by_id()
        for event, event_json_path, payload in events:
            event_id = str(event.get("event_id") or event_json_path.parent.name)
            event_dir = event_json_path.parent
            event_dir.mkdir(parents=True, exist_ok=True)
            asset = payload.get("asset_pack") or event.get("asset_pack") or {}
            if not isinstance(asset, dict):
                asset = {}
            changed = False
            if self._upgrade_material_fallback_event(event, payload, material_items):
                changed = True

            overlay_mode = str(asset.get("overlay_mode") or event.get("overlay_mode") or "")
            needs_annotated_clip = force_yolo_repair or overlay_mode not in {"event_selective", "track_selective", "live_yolo_event_selective"}
            if source_video and (not self._asset_exists(asset.get("clip_path")) or needs_annotated_clip):
                start = self._event_start_time(event)
                end = self._event_end_time(event, start)
                if end <= start:
                    end = start + max(0.8, self._float(event.get("duration_sec"), 2.5))
                extracted = self._extract_yolo_event_assets(source_video, event, event_dir)
                if extracted:
                    asset = extracted
                    fresh_payload = read_json(event_json_path, {}) or {}
                    if isinstance(fresh_payload, dict) and fresh_payload.get("overlay_metadata"):
                        payload["overlay_metadata"] = fresh_payload.get("overlay_metadata")
                    changed = True
                else:
                    clip_path = event_dir / "clip.mp4"
                    if self._cut_video_clip(source_video, start, end, clip_path):
                        asset["clip_path"] = str(clip_path.resolve())
                        asset["asset_status"] = "ready"
                        asset["overlay_mode"] = "source_video_time_cut"
                        changed = True

            if source_video and (not self._asset_exists(asset.get("preview_path")) or not self._existing_keyframes(asset)):
                start = self._event_start_time(event)
                end = self._event_end_time(event, start + 2.5)
                written = self._write_keyframes_from_source(source_video, start, max(end, start + 0.8), event_dir)
                if written:
                    asset["keyframe_paths"] = [str(path.resolve()) for path in written]
                    asset["preview_path"] = str(written[min(1, len(written) - 1)].resolve())
                    asset["asset_status"] = "ready"
                    changed = True

            asset.setdefault("event_id", event_id)
            asset.setdefault("event_json_path", str(event_json_path.resolve()))
            event["asset_pack"] = asset
            payload["event"] = event
            payload["asset_pack"] = asset
            if changed or not event_json_path.exists():
                write_json(event_json_path, payload)
                updated_events[event_id] = event
            repaired.append((event, event_json_path, payload))

        if updated_events:
            self._sync_physical_event_assets(updated_events)
        return repaired

    def _extract_yolo_event_assets(self, source_video: Path, event: Dict[str, Any], event_dir: Path) -> Optional[Dict[str, Any]]:
        try:
            from labsopguard.config import load_runtime_settings
            from labsopguard.event_preprocessing.key_material_extraction import KeyMaterialExtractor
            from labsopguard.event_preprocessing.schemas import PhysicalEvent
            from labsopguard.video_analysis import VideoAnalysisPipeline

            project_root = self._project_root()
            settings = load_runtime_settings(project_root)
            if os.environ.get("LABSOPGUARD_FORCE_YOLO_REPAIR") == "1":
                try:
                    material_conf = float(os.environ.get("LABSOPGUARD_MATERIAL_YOLO_CONF", "0.08"))
                    settings.confidence_threshold = min(float(settings.confidence_threshold), material_conf)
                    settings.max_detections = max(int(settings.max_detections), 50)
                except Exception:
                    pass
            pipeline = VideoAnalysisPipeline(settings=settings, yolo_model_path=settings.yolo_model_path)
            physical_event = self._physical_event_from_dict(event, source_video_id=event.get("source_video_id") or f"{self.experiment_id}:video:0")
            pack = KeyMaterialExtractor(pipeline, overlay_mode="event_selective").extract_assets(
                source_video,
                physical_event,
                event_dir,
            )
            if pipeline.yolo_model is None:
                payload = read_json(event_dir / "event.json", {}) or {}
                payload.setdefault("overlay_metadata", {})
                payload["overlay_metadata"]["detector_warning"] = "YOLO26 weights were not available; clip contains timeline overlay only"
                write_json(event_dir / "event.json", payload)
            return pack.to_dict()
        except Exception as exc:
            payload = read_json(event_dir / "event.json", {}) or {"event": event}
            if isinstance(payload, dict):
                payload.setdefault("overlay_metadata", {})
                payload["overlay_metadata"]["yolo_repair_error"] = f"{type(exc).__name__}: {exc}"
                write_json(event_dir / "event.json", payload)
            return None

    def _project_root(self) -> Path:
        try:
            # .../outputs/experiments/<experiment_id> -> project root
            return self.experiment_dir.resolve().parents[2]
        except Exception:
            return Path.cwd()

    def _physical_event_from_dict(self, event: Dict[str, Any], *, source_video_id: Any):
        from labsopguard.event_preprocessing.schemas import PhysicalEvent

        start = self._event_start_time(event)
        end = self._event_end_time(event, start)
        if end <= start:
            end = start + max(0.8, self._float(event.get("duration_sec"), 2.5))
        mid = start + (end - start) / 2.0
        key_timestamps = event.get("key_timestamps") or [start, mid, max(start, end - 0.05)]
        if not isinstance(key_timestamps, list):
            key_timestamps = [start, mid, max(start, end - 0.05)]
        involved_objects = event.get("involved_objects") or event.get("object_labels") or []
        if not isinstance(involved_objects, list):
            involved_objects = [str(involved_objects)]
        return PhysicalEvent(
            event_id=str(event.get("event_id") or f"event_{start:.3f}"),
            experiment_id=str(event.get("experiment_id") or self.experiment_id),
            source_video_id=str(source_video_id),
            event_type=str(event.get("event_type") or "hand_object_interaction"),
            stable_name=str(event.get("stable_name") or ""),
            display_name=str(event.get("display_name") or f"Event @ {start:.1f}s"),
            actor_name=str(event.get("actor_name") or "operator"),
            start_time_sec=round(start, 3),
            end_time_sec=round(end, 3),
            duration_sec=round(max(0.0, end - start), 3),
            key_timestamps=[float(value) for value in key_timestamps[:3]],
            involved_objects=[str(value) for value in involved_objects if str(value).strip()],
            dominant_object=event.get("dominant_object") or (involved_objects[0] if involved_objects else None),
            involved_track_ids=[str(value) for value in (event.get("involved_track_ids") or [])],
            primary_track_id=event.get("primary_track_id"),
            source_container=event.get("source_container"),
            target_container=event.get("target_container"),
            track_motion_summary=event.get("track_motion_summary") or {},
            actor_track_id=event.get("actor_track_id"),
            tool_track_id=event.get("tool_track_id"),
            related_tracks=[str(value) for value in (event.get("related_tracks") or [])],
            transfer_mode=event.get("transfer_mode"),
            action_resolution_source=str(event.get("action_resolution_source") or event.get("proposal_source") or "material_stream_fallback"),
            action_resolution_notes=str(event.get("action_resolution_notes") or event.get("notes") or ""),
            supporting_relation_ids=[str(value) for value in (event.get("supporting_relation_ids") or [])],
            direction_confidence=event.get("direction_confidence"),
            direction_status=event.get("direction_status"),
            direction_evidence=[str(value) for value in (event.get("direction_evidence") or [])],
            state_before=event.get("state_before"),
            state_after=event.get("state_after"),
            state_change_type=event.get("state_change_type"),
            state_confidence=event.get("state_confidence"),
            state_evidence=[str(value) for value in (event.get("state_evidence") or [])],
            evidence_grade=str(event.get("evidence_grade") or "medium"),
            review_status=str(event.get("review_status") or "candidate_review"),
            evidence_summary=str(event.get("evidence_summary") or event.get("qwen_summary") or ""),
            confidence=self._float(event.get("confidence"), 0.58),
            event_status=str(event.get("event_status") or "candidate"),
            proposal_source=str(event.get("proposal_source") or "material_stream_fallback"),
            evidence_frame_indices=[int(value) for value in (event.get("evidence_frame_indices") or []) if str(value).lstrip("-").isdigit()],
            related_detection_classes=[str(value) for value in (event.get("related_detection_classes") or involved_objects)],
            notes=str(event.get("notes") or ""),
            asset_pack=event.get("asset_pack") if isinstance(event.get("asset_pack"), dict) else None,
        )

    def _source_video_path(self) -> Optional[Path]:
        experiment = read_json(self.experiment_dir / "experiment.json", {}) or {}
        output_paths = experiment.get("output_paths") or {}
        candidates: List[Any] = []
        candidates.append(output_paths.get("source_video"))
        candidates.extend(output_paths.get("source_videos") or [])
        candidates.extend(experiment.get("video_paths") or [])
        for item in experiment.get("video_inputs") or []:
            if isinstance(item, dict):
                candidates.extend([item.get("video_path"), item.get("source")])
        for item in experiment.get("video_assets") or []:
            if isinstance(item, dict):
                candidates.append(item.get("file_path"))
        for value in candidates:
            path = self._resolve_material_path(value)
            if path and path.exists() and path.is_file():
                return path
        return None

    def _asset_exists(self, value: Any) -> bool:
        path = self._resolve_material_path(value)
        return bool(path and path.exists() and path.is_file() and path.stat().st_size > 0)

    def _existing_keyframes(self, asset: Dict[str, Any]) -> List[Path]:
        paths = []
        for value in asset.get("keyframe_paths") or []:
            path = self._resolve_material_path(value)
            if path and path.exists() and path.is_file() and path.stat().st_size > 0:
                paths.append(path)
        return paths

    def _sync_physical_event_assets(self, updated_events: Dict[str, Dict[str, Any]]) -> None:
        physical_path = self.experiment_dir / "physical_events.json"
        payload = read_json(physical_path, None)
        if payload is None:
            return
        event_key = "events"
        if isinstance(payload, dict):
            if isinstance(payload.get("events"), list):
                events = payload.get("events")
            elif isinstance(payload.get("physical_events"), list):
                event_key = "physical_events"
                events = payload.get("physical_events")
            else:
                events = None
        else:
            events = payload
        if not isinstance(events, list):
            return
        changed = False
        for idx, item in enumerate(events):
            if not isinstance(item, dict):
                continue
            event_id = str(item.get("event_id") or "")
            if event_id in updated_events:
                item.update(updated_events[event_id])
                events[idx] = item
                changed = True
        if changed:
            if isinstance(payload, dict):
                payload[event_key] = events
                payload["event_count"] = len(events)
            else:
                payload = events
            write_json(physical_path, payload)

    def _material_items_by_id(self) -> Dict[str, Dict[str, Any]]:
        payload = read_json(self.experiment_dir / "material_stream.json", [])
        if isinstance(payload, dict):
            payload = payload.get("items") or payload.get("material_stream") or []
        if not isinstance(payload, list):
            return {}
        lookup: Dict[str, Dict[str, Any]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            for key in ("item_id", "material_id", "clip_id", "event_id"):
                value = item.get(key)
                if value:
                    lookup[str(value)] = item
        return lookup

    def _upgrade_material_fallback_event(
        self,
        event: Dict[str, Any],
        payload: Dict[str, Any],
        material_items: Dict[str, Dict[str, Any]],
    ) -> bool:
        """Normalize historical material-stream fallback events before publish."""
        changed = False
        start = self._event_start_time(event)
        end = self._event_end_time(event, start)
        if end <= start:
            end = start + max(0.8, self._float(event.get("duration_sec"), 2.5))
        keys = event.get("key_timestamps") if isinstance(event.get("key_timestamps"), list) else []
        clean_keys = []
        for value in keys:
            ts = self._float(value, start)
            if start <= ts <= end:
                clean_keys.append(round(ts, 3))
        if len(clean_keys) < 3:
            mid = start + (end - start) / 2.0
            clean_keys = [round(start, 3), round(mid, 3), round(max(start, end - 0.05), 3)]
            event["key_timestamps"] = clean_keys
            changed = True

        source_id = str(event.get("source_material_item_id") or event.get("material_item_id") or "")
        item = material_items.get(source_id) or material_items.get(str(event.get("event_id") or ""))
        is_fallback = (
            payload.get("fallback_source") == "material_stream"
            or event.get("proposal_source") == "material_stream_fallback"
            or bool(source_id)
        )
        if is_fallback and item:
            sections = self._scene_sections(item)
            inferred_type = self._event_type_from_material_item(item, sections)
            current_type = str(event.get("event_type") or "")
            if inferred_type and (
                not current_type
                or (
                    current_type == "liquid_transfer"
                    and not event.get("source_container")
                    and not event.get("target_container")
                    and inferred_type != current_type
                )
            ):
                event["event_type"] = inferred_type
                changed = True
            if not event.get("involved_objects") and sections["objects"]:
                event["involved_objects"] = sections["objects"]
                changed = True
            if not event.get("dominant_object") and sections["objects"]:
                event["dominant_object"] = sections["objects"][0]
                changed = True
            if not event.get("related_detection_classes") and sections["objects"]:
                event["related_detection_classes"] = sections["objects"]
                changed = True
            if not event.get("qwen_summary") and sections["description"]:
                event["qwen_summary"] = sections["description"]
                changed = True
        if not event.get("proposal_source"):
            event["proposal_source"] = "material_stream_fallback"
            changed = True
        if not event.get("action_resolution_source"):
            event["action_resolution_source"] = "material_stream_fallback"
            changed = True
        return changed

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    @classmethod
    def _cut_video_clip(cls, source: Path, start: float, end: float, output: Path) -> bool:
        output.parent.mkdir(parents=True, exist_ok=True)
        start = max(0.0, float(start))
        duration = max(0.2, float(end) - start)
        ffmpeg_exe = cls._find_ffmpeg()
        if ffmpeg_exe:
            tmp = output.with_name(f"{output.stem}.tmp{output.suffix}")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(tmp),
            ]
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
                if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                    shutil.move(str(tmp), str(output))
                    return True
            except Exception:
                pass
            tmp.unlink(missing_ok=True)

        try:
            import cv2

            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                return False
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                cap.release()
                return False
            writer = None
            for codec in ("avc1", "mp4v"):
                candidate = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
                if candidate.isOpened():
                    writer = candidate
                    break
                candidate.release()
            if writer is None:
                cap.release()
                return False
            start_frame = max(0, int(start * fps))
            end_frame = max(start_frame, int((start + duration) * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_idx = start_frame
            while frame_idx <= end_frame:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                frame_idx += 1
            cap.release()
            writer.release()
            return output.exists() and output.stat().st_size > 0
        except Exception:
            output.unlink(missing_ok=True)
            return False

    @staticmethod
    def _write_keyframes_from_source(source: Path, start: float, end: float, event_dir: Path) -> List[Path]:
        try:
            import cv2

            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                return []
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            timestamps = [start, (start + end) / 2.0, max(start, end - 0.05)]
            written: List[Path] = []
            for idx, ts in enumerate(timestamps, start=1):
                frame_idx = max(0, int(float(ts) * fps))
                if frame_count > 0:
                    frame_idx = min(frame_idx, frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                path = event_dir / f"keyframe_{idx:02d}.jpg"
                if cv2.imwrite(str(path), frame):
                    written.append(path)
            cap.release()
            return written
        except Exception:
            return []

    def _load_events_from_material_stream(self) -> List[Tuple[Dict[str, Any], Path, Dict[str, Any]]]:
        material_stream = read_json(self.experiment_dir / "material_stream.json", [])
        if isinstance(material_stream, dict):
            material_stream = material_stream.get("items") or material_stream.get("material_stream") or []
        if not isinstance(material_stream, list):
            return []

        items = [item for item in material_stream if isinstance(item, dict)]
        items.sort(key=lambda item: self._float(self._first_present(item.get("timestamp_sec"), item.get("time_start"), item.get("start_time_sec"))))
        events_dir = self.experiment_dir / "materials" / "events"
        found: List[Tuple[Dict[str, Any], Path, Dict[str, Any]]] = []
        seen_ids: set[str] = set()
        for idx, item in enumerate(items):
            keyframes = self._material_keyframes(item)
            clip = self._resolve_material_path(item.get("clip_file_path") or item.get("clip_path"))
            if not keyframes and not clip:
                continue
            event_id = self._safe_event_id(item.get("event_id") or item.get("clip_id") or item.get("item_id") or f"frame_{idx:04d}", idx, seen_ids)
            start = self._float(self._first_present(item.get("time_start"), item.get("start_time_sec"), item.get("timestamp_sec")))
            end = self._float(self._first_present(item.get("time_end"), item.get("end_time_sec"), item.get("end_timestamp_sec")), start)
            if end <= start:
                end = self._fallback_end_time(items, idx, start)
            raw_keys = item.get("key_timestamps") if isinstance(item.get("key_timestamps"), list) else []
            key_timestamps: List[float] = []
            for value in raw_keys:
                ts = self._float(value, start)
                if start <= ts <= end:
                    key_timestamps.append(round(ts, 3))
            if not key_timestamps:
                mid = start + (end - start) / 2.0
                key_timestamps = [round(start, 3), round(mid, 3), round(max(start, end - 0.05), 3)]
            key_timestamps = list(dict.fromkeys(key_timestamps))[:3]
            sections = self._scene_sections(item)
            event_type = self._event_type_from_material_item(item, sections)
            display = self._display_from_material_item(item, sections, start)
            preview = self._resolve_material_path(item.get("preview_path")) or (keyframes[0] if keyframes else None)
            asset = {
                "event_id": event_id,
                "clip_path": str(clip) if clip and clip.exists() else None,
                "preview_path": str(preview) if preview and preview.exists() else None,
                "keyframe_paths": [str(path) for path in keyframes if path.exists()],
                "event_json_path": str(events_dir / event_id / "event.json"),
                "overlay_mode": "frame_material_fallback",
                "asset_status": "ready",
            }
            event = {
                "event_id": event_id,
                "experiment_id": self.experiment_id,
                "event_type": event_type,
                "actor_name": str(item.get("actor_name") or "operator"),
                "start_time_sec": round(start, 3),
                "end_time_sec": round(end, 3),
                "duration_sec": round(max(0.0, end - start), 3),
                "key_timestamps": key_timestamps,
                "display_name": display,
                "stable_name": str(item.get("stable_name") or ""),
                "involved_objects": sections["objects"],
                "dominant_object": sections["objects"][0] if sections["objects"] else None,
                "evidence_grade": str(item.get("evidence_grade") or ("medium" if item.get("is_key_frame") else "weak")),
                "review_status": str(item.get("review_status") or "candidate_review"),
                "qwen_summary": sections["description"],
                "source_container": item.get("source_container"),
                "target_container": item.get("target_container"),
                "related_tracks": item.get("related_tracks") or item.get("linked_context_event_ids") or [],
                "source_material_item_id": item.get("item_id") or item.get("material_id"),
                "source_frame_id": item.get("frame_id"),
                "source_video_id": item.get("source_video_id") or item.get("media_asset_id"),
                "confidence": self._float(item.get("confidence"), 0.3),
                "proposal_source": "material_stream_fallback",
                "action_resolution_source": "material_stream_fallback",
                "evidence_frame_indices": [int(item.get("frame_id"))] if str(item.get("frame_id", "")).lstrip("-").isdigit() else [],
                "related_detection_classes": sections["objects"],
                "asset_pack": asset,
            }
            event_json_path = events_dir / event_id / "event.json"
            payload = {"event": event, "asset_pack": asset, "fallback_source": "material_stream"}
            write_json(event_json_path, payload)
            found.append((event, event_json_path, payload))
        return found

    def _write_fallback_physical_events(self, events: List[Dict[str, Any]]) -> None:
        if not events:
            return
        physical_path = self.experiment_dir / "physical_events.json"
        existing = read_json(physical_path, {}) or {}
        if isinstance(existing, dict):
            existing_events = existing.get("events") or existing.get("physical_events")
        else:
            existing_events = existing
        if isinstance(existing_events, list) and existing_events:
            return
        payload = {
            "schema_version": "physical_events.v4",
            "metadata_version": "material_stream_fallback.v1",
            "experiment_id": self.experiment_id,
            "source_video_id": events[0].get("source_video_id"),
            "tracklets": [],
            "tracked_objects": [],
            "track_relations": [],
            "events": events,
            "event_count": len(events),
            "fallback_source": "material_stream",
        }
        write_json(physical_path, payload)

    def _resolve_material_path(self, value: Any) -> Optional[Path]:
        if not value:
            return None
        path_text = str(value).replace("\\", "/")
        marker = f"outputs/experiments/{self.experiment_id}/"
        idx = path_text.find(marker)
        if idx >= 0:
            return self.experiment_dir / path_text[idx + len(marker):]
        generic_match = re.search(r"outputs/experiments/[^/]+/(.+)$", path_text)
        if generic_match:
            candidate = self.experiment_dir / generic_match.group(1)
            if candidate.exists():
                return candidate
        upload_marker = f"uploads/experiments/{self.experiment_id}/"
        idx = path_text.find(upload_marker)
        if idx >= 0:
            return self._project_root() / upload_marker / path_text[idx + len(upload_marker):]
        upload_match = re.search(r"uploads/experiments/[^/]+/(.+)$", path_text)
        if upload_match:
            candidate = self._project_root() / "uploads" / "experiments" / self.experiment_id / upload_match.group(1)
            if candidate.exists():
                return candidate
        path = Path(path_text)
        if path.is_absolute():
            return path
        for prefix in ("artifacts/", "materials/", "published_materials/", "clips/", "uploads/"):
            if path_text.startswith(prefix):
                if prefix == "uploads/":
                    return self._project_root() / path_text
                return self.experiment_dir / path_text
        candidate = self.experiment_dir / path_text
        if candidate.exists():
            return candidate
        return path

    def _material_keyframes(self, item: Dict[str, Any]) -> List[Path]:
        raw_paths: List[Any] = []
        for key in ("keyframe_paths", "frame_bgr_path", "frame_path", "preview_path"):
            value = item.get(key)
            if isinstance(value, list):
                raw_paths.extend(value)
            elif value:
                raw_paths.append(value)
        resolved: List[Path] = []
        seen: set[str] = set()
        for value in raw_paths:
            path = self._resolve_material_path(value)
            if path and path.exists():
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    resolved.append(path)
        return resolved[:3]

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _first_present(*values: Any) -> Any:
        for value in values:
            if value is not None and value != "":
                return value
        return None

    def _event_start_time(self, event: Dict[str, Any], default: float = 0.0) -> float:
        return self._float(
            self._first_present(
                event.get("start_time_sec"),
                event.get("time_start"),
                event.get("timestamp_sec"),
                event.get("start_timestamp_sec"),
                event.get("start_sec"),
            ),
            default,
        )

    def _event_end_time(self, event: Dict[str, Any], default: float = 0.0) -> float:
        return self._float(
            self._first_present(
                event.get("end_time_sec"),
                event.get("time_end"),
                event.get("end_timestamp_sec"),
                event.get("stop_time_sec"),
                event.get("end_sec"),
            ),
            default,
        )

    def _fallback_end_time(self, items: List[Dict[str, Any]], idx: int, start: float) -> float:
        for item in items[idx + 1:]:
            next_start = self._float(self._first_present(item.get("timestamp_sec"), item.get("time_start"), item.get("start_time_sec")))
            if next_start > start:
                return max(start + 0.5, min(next_start, start + 5.0))
        return start + 3.0

    @staticmethod
    def _safe_event_id(value: Any, idx: int, seen: set[str]) -> str:
        base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")
        base = base[:72] or f"event_{idx:04d}"
        event_id = base
        counter = 2
        while event_id in seen:
            event_id = f"{base}_{counter}"
            counter += 1
        seen.add(event_id)
        return event_id

    @staticmethod
    def _scene_sections(item: Dict[str, Any]) -> Dict[str, Any]:
        raw = item.get("scene_description") or item.get("description") or ""
        parsed: Dict[str, Any] = {}
        if isinstance(raw, str):
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").replace("json\n", "", 1).strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(cleaned[start:end + 1])
                    if isinstance(value, dict):
                        parsed = value
                except Exception:
                    parsed = {}
        activities = item.get("detected_activities") or parsed.get("detected_activities") or parsed.get("actions") or []
        objects = item.get("object_labels") or item.get("detected_objects") or parsed.get("object_labels") or parsed.get("objects") or []
        indicators = item.get("step_indicators") or parsed.get("step_indicators") or []
        return {
            "description": str(parsed.get("description") or parsed.get("scene_summary") or raw or ""),
            "activities": SemanticMaterialPublisher._list_text(activities),
            "objects": SemanticMaterialPublisher._list_text(objects),
            "indicators": SemanticMaterialPublisher._list_text(indicators),
        }

    @staticmethod
    def _list_text(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, dict):
                    text = item.get("class_name") or item.get("label") or item.get("name")
                else:
                    text = item
                text = str(text or "").strip()
                if text:
                    result.append(text)
            return result
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _event_type_from_material_item(item: Dict[str, Any], sections: Dict[str, Any]) -> str:
        explicit = item.get("event_type")
        if explicit:
            return str(explicit)
        activities_text = " ".join(sections.get("activities") or []).lower()
        indicators_text = " ".join(sections.get("indicators") or []).lower()
        description_text = str(sections.get("description") or "").lower()
        object_text = " ".join(sections.get("objects") or []).lower()
        action_text = " ".join([activities_text, indicators_text, description_text]).lower()
        static_terms = [
            "stationary", "static", "arrangement", "setup", "prepared", "preparation",
            "\u51c6\u5907", "\u5e03\u7f6e", "\u6574\u7406", "\u5c31\u4f4d",
        ]
        transfer_terms = [
            "\u79fb\u6db2", "\u8f6c\u79fb", "\u52a0\u6837", "\u52a0\u6db2",
            "\u5012\u5165", "\u6df7\u5408", "\u5206\u6db2", "pipette", "transfer",
            "pour", "dispense", "titrate", "add liquid", "mix",
        ]
        panel_terms = [
            "\u5929\u5e73", "\u79f0\u91cf", "\u8bfb\u6570", "\u6309\u94ae",
            "\u9762\u677f", "\u8bb0\u5f55", "\u4eea\u5668", "\u8bbe\u5907",
            "balance", "scale", "screen", "button", "panel", "record", "instrument",
        ]
        container_terms = [
            "\u6253\u5f00", "\u5173\u95ed", "\u74f6\u76d6", "\u76d6\u5b50",
            "\u5bb9\u5668", "open", "close", "cap", "lid", "container",
        ]
        move_terms = [
            "\u653e\u7f6e", "\u62ff\u53d6", "\u79fb\u52a8", "\u53d6\u6837",
            "move", "place", "pick", "remove", "prepare", "setup",
        ]
        if any(token in activities_text for token in static_terms) and not any(token in activities_text for token in transfer_terms):
            return "object_move"
        if any(token in action_text for token in transfer_terms):
            return "liquid_transfer"
        if any(token in action_text for token in panel_terms):
            return "panel_operation"
        if any(token in action_text for token in container_terms):
            return "container_state_change"
        if any(token in action_text for token in move_terms):
            return "object_move"
        # Object-only fallbacks are intentionally conservative. A reagent
        # bottle in a static keyframe is not enough evidence for liquid transfer.
        if any(token in object_text for token in ["balance", "scale"]):
            return "panel_operation"
        if any(token in object_text for token in ["cap", "lid"]):
            return "container_state_change"
        if object_text:
            return "object_move"
        return "hand_object_interaction"
    @staticmethod
    def _display_from_material_item(item: Dict[str, Any], sections: Dict[str, Any], start: float) -> str:
        existing = str(item.get("display_name") or "").strip()
        if existing:
            return existing
        for group in (sections.get("activities") or [], sections.get("indicators") or [], sections.get("objects") or []):
            for value in group:
                text = str(value or "").strip()
                if text:
                    return f"{text} @ {start:.1f}s"
        description = re.sub(r"\s+", " ", str(sections.get("description") or "")).strip()
        if description:
            return f"{description[:28]} @ {start:.1f}s"
        return f"Key frame @ {start:.1f}s"

    def _publish_event(self, event: Dict[str, Any], event_json_path: Path, payload: Dict[str, Any], experiment_name: str) -> MaterialPublishRecord:
        event_id = str(event.get("event_id") or event_json_path.parent.name)
        stable = str(event.get("stable_name") or make_stable_name(experiment_name, event))
        actor = str(event.get("actor_name") or "operator_unknown")
        plan = self.planner.plan(event=event, stable_name=stable, actor_name=actor)
        plan.publish_dir.mkdir(parents=True, exist_ok=True)
        asset = payload.get("asset_pack") or event.get("asset_pack") or {}
        display, display_source = self.display_name_enhancer.enhance(experiment_name, event, asset)
        warnings: List[str] = []
        clip_target = plan.clip_path
        source_clip = self._resolve_material_path(asset.get("clip_path"))
        if source_clip and source_clip.suffix.lower() in {".mp4", ".webm"}:
            clip_target = plan.clip_path.with_suffix(source_clip.suffix.lower())
        clip = self._publish_file(asset.get("clip_path"), clip_target, warnings, "missing_clip")
        preview = self._publish_file(asset.get("preview_path"), plan.preview_path, warnings, "missing_preview")
        keyframes = []
        for idx, source in enumerate(asset.get("keyframe_paths") or [], start=1):
            target = plan.publish_dir / f"keyframe_{idx:02d}.jpg"
            copied = self._publish_file(source, target, warnings, f"missing_keyframe_{idx:02d}")
            if copied:
                keyframes.append(str(copied))
        if not keyframes:
            warnings.append("no_keyframes")
        event_copy = self._publish_file(event_json_path if event_json_path.exists() else None, plan.event_json_path, warnings, "missing_event_json")
        if event.get("direction_status") in {"unknown", None} and event.get("event_type") == "liquid_transfer":
            warnings.append("direction_not_confirmed")
        if not event.get("actor_name"):
            warnings.append("actor_unknown")
        if (asset.get("overlay_mode") or event.get("overlay_mode")) not in {
            None,
            "event_selective",
            "track_selective",
            "live_yolo_event_selective",
            "source_video_time_cut",
            "frame_material_fallback",
        }:
            warnings.append("overlay_mode_not_event_selective")
        published_paths = PublishedPaths(
            clip=str(clip) if clip else None,
            preview=str(preview) if preview else None,
            keyframes=keyframes,
            event_json=str(event_copy) if event_copy else None,
            material_publish=str(plan.material_publish_path),
        ).to_dict()
        start_time = self._event_start_time(event)
        end_time = self._event_end_time(event, start_time)
        if end_time <= start_time:
            end_time = start_time + max(0.0, self._float(event.get("duration_sec"), 0.0))
        record = MaterialPublishRecord(
            schema_version=MATERIAL_PUBLISH_VERSION,
            experiment_id=self.experiment_id,
            event_id=event_id,
            material_id=f"mat_{event_id}",
            stable_name=stable,
            display_name=display,
            actor_name=actor,
            event_type=str(event.get("event_type") or "event"),
            time_start=start_time,
            time_end=end_time,
            source_container=self._container(event.get("source_container")),
            target_container=self._container(event.get("target_container")),
            evidence_grade=event.get("evidence_grade"),
            review_status=event.get("review_status"),
            canonical_event_path=str(event_json_path),
            published_paths=published_paths,
            warnings=list(dict.fromkeys(warnings)),
            extra={
                "quality_score": asset.get("quality_score"),
                "quality_grade": asset.get("quality_grade"),
                "quality_reasons": asset.get("quality_reasons") or [],
                "asset_status": asset.get("asset_status"),
                "direction_status": event.get("direction_status"),
                "direction_confidence": event.get("direction_confidence"),
                "actor_track_id": event.get("actor_track_id"),
                "tool_track_id": event.get("tool_track_id"),
                "related_tracks": event.get("related_tracks") or event.get("involved_track_ids") or [],
                "copy_strategy": self.copy_strategy,
                "publish_dir": str(plan.publish_dir),
                "display_name_source": display_source,
            },
        )
        write_json(plan.material_publish_path, record.to_dict())
        return record

    def _publish_file(self, source: Any, target: Path, warnings: List[str], missing_warning: str) -> Optional[Path]:
        if not source:
            warnings.append(missing_warning)
            return None
        source_path = self._resolve_material_path(source)
        if source_path is None:
            warnings.append(missing_warning)
            return None
        if not source_path.exists():
            warnings.append(missing_warning)
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if source_path.resolve() == target.resolve():
                    return target
            except Exception:
                pass
            target.unlink(missing_ok=True)
        if self.copy_strategy == "copy":
            shutil.copy2(source_path, target)
            return target
        try:
            os.link(source_path, target)
        except OSError:
            shutil.copy2(source_path, target)
        return target

    @staticmethod
    def _container(container: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not container:
            return None
        return {
            "track_id": container.get("track_id"),
            "class_name": container.get("class_name") or container.get("object_name"),
            "display_name": container.get("display_name") or container_label(container),
            "confidence": container.get("confidence") or container.get("role_confidence"),
        }

    def _experiment_name(self) -> str:
        experiment = read_json(self.experiment_dir / "experiment.json", {}) or {}
        return str(experiment.get("title") or experiment.get("experiment_name") or self.experiment_id)

    def _write_index(self, records: List[MaterialPublishRecord]) -> None:
        db_path = self.experiment_dir / "material_index.sqlite"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            self._ensure_index_schema(conn)
            for record in records:
                self._upsert_index(conn, record)
            conn.commit()
        finally:
            conn.close()

    def _update_official_steps(self, records: List[MaterialPublishRecord]) -> Dict[str, Any]:
        official_path = self.experiment_dir / "official_steps.json"
        if not official_path.exists():
            return {"status": "skipped", "reason": "official_steps.json missing", "updated_count": 0}
        payload = read_json(official_path, {}) or {}
        by_event = {record.event_id: record.to_dict() for record in records}
        updated = 0
        for step in payload.get("official_steps") or []:
            linked = [str(item) for item in step.get("linked_event_ids") or []]
            refs = [by_event[event_id] for event_id in linked if event_id in by_event]
            if not refs:
                continue
            bundle = step.setdefault("evidence_bundle", {})
            existing = {str(item.get("event_id")): item for item in bundle.get("published_material_refs") or [] if isinstance(item, dict)}
            for ref in refs:
                existing[ref["event_id"]] = {
                    "event_id": ref["event_id"],
                    "material_id": ref["material_id"],
                    "display_name": ref["display_name"],
                    "stable_name": ref["stable_name"],
                    "event_type": ref["event_type"],
                    "published_paths": ref["published_paths"],
                    "material_publish_path": ref["published_paths"].get("material_publish"),
                }
            bundle["published_material_refs"] = list(existing.values())
            updated += 1
        if updated:
            write_json(official_path, payload)
            self._append_publish_governance_decisions(records, payload)
        return {"status": "updated" if updated else "unchanged", "updated_count": updated}

    def _append_publish_governance_decisions(self, records: List[MaterialPublishRecord], official_payload: Dict[str, Any]) -> None:
        log_path = self.experiment_dir / "step_review_log.json"
        if not log_path.exists():
            return
        log = read_json(log_path, {}) or {}
        decisions = log.setdefault("governance_decisions", [])
        existing_ids = {str(item.get("governance_decision_id")) for item in decisions if isinstance(item, dict)}
        event_ids = {record.event_id for record in records}
        now = datetime.now(timezone.utc).isoformat()
        for step in official_payload.get("official_steps") or []:
            linked = set(str(item) for item in step.get("linked_event_ids") or [])
            matched = sorted(linked & event_ids)
            if not matched:
                continue
            raw = f"{step.get('official_step_id')}:publish_material_refs:{','.join(matched)}"
            decision_id = "stepgov_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
            if decision_id in existing_ids:
                continue
            decisions.append(
                {
                    "governance_decision_id": decision_id,
                    "official_step_id": step.get("official_step_id"),
                    "decision": "publish_material_refs_update",
                    "rationale": "Semantic material publisher linked published material references to official step evidence bundle.",
                    "operator": "system",
                    "operator_role": "system",
                    "created_at": now,
                    "metadata_version": "operational_step_governance.v1",
                    "linked_event_ids": matched,
                }
            )
        write_json(log_path, log)

    @staticmethod
    def _ensure_index_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS event_materials (
                material_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                event_id TEXT UNIQUE,
                event_type TEXT,
                display_name TEXT,
                stable_name TEXT,
                actor_name TEXT,
                clip_path TEXT,
                preview_path TEXT,
                searchable_text TEXT,
                created_at TEXT,
                metadata_version TEXT,
                payload_json TEXT
            )
            """
        )
        for statement in (
            "ALTER TABLE event_materials ADD COLUMN published_path TEXT",
            "ALTER TABLE event_materials ADD COLUMN material_publish_path TEXT",
            "ALTER TABLE event_materials ADD COLUMN source_container_class TEXT",
            "ALTER TABLE event_materials ADD COLUMN target_container_class TEXT",
            "ALTER TABLE event_materials ADD COLUMN actor_track_id TEXT",
        ):
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.execute("CREATE INDEX IF NOT EXISTS idx_event_materials_published ON event_materials(published_path)")

    @staticmethod
    def _upsert_index(conn: sqlite3.Connection, record: MaterialPublishRecord) -> None:
        payload = record.to_dict()
        paths = record.published_paths or {}
        source_class = (record.source_container or {}).get("class_name") if record.source_container else None
        target_class = (record.target_container or {}).get("class_name") if record.target_container else None
        text = searchable_text(
            [
                record.display_name,
                record.stable_name,
                record.event_type,
                record.actor_name,
                source_class,
                target_class,
                record.evidence_grade,
                record.review_status,
                " ".join(record.warnings),
            ]
        )
        row = {
            "material_id": record.material_id,
            "experiment_id": record.experiment_id,
            "event_id": record.event_id,
            "event_type": record.event_type,
            "display_name": record.display_name,
            "stable_name": record.stable_name,
            "actor_name": record.actor_name,
            "clip_path": paths.get("clip"),
            "preview_path": paths.get("preview"),
            "published_path": str(Path(paths.get("material_publish", "")).parent) if paths.get("material_publish") else None,
            "material_publish_path": paths.get("material_publish"),
            "source_container_class": source_class,
            "target_container_class": target_class,
            "actor_track_id": record.extra.get("actor_track_id"),
            "searchable_text": text,
            "payload_json": json.dumps(payload, ensure_ascii=False),
        }
        existing = conn.execute("SELECT material_id FROM event_materials WHERE event_id = ?", (record.event_id,)).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE event_materials
                SET display_name=:display_name, stable_name=:stable_name, actor_name=:actor_name,
                    clip_path=:clip_path, preview_path=:preview_path, published_path=:published_path,
                    material_publish_path=:material_publish_path, source_container_class=:source_container_class,
                    target_container_class=:target_container_class, actor_track_id=:actor_track_id,
                    searchable_text=:searchable_text, payload_json=:payload_json
                WHERE event_id=:event_id
                """,
                row,
            )
        else:
            conn.execute(
                """
                INSERT INTO event_materials
                (material_id, experiment_id, event_id, event_type, display_name, stable_name, actor_name,
                 clip_path, preview_path, published_path, material_publish_path, source_container_class,
                 target_container_class, actor_track_id, searchable_text, payload_json)
                VALUES (:material_id, :experiment_id, :event_id, :event_type, :display_name, :stable_name,
                 :actor_name, :clip_path, :preview_path, :published_path, :material_publish_path,
                 :source_container_class, :target_container_class, :actor_track_id, :searchable_text, :payload_json)
                """,
                row,
            )
