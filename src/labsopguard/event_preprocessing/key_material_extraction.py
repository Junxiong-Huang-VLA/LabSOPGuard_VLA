from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .class_roles import (
    is_container_label,
    is_hand_label,
    is_lid_label,
    is_panel_label,
    is_tool_label,
)
from .schemas import EventAssetPack, PhysicalEvent, dump_json
from .tracking.schemas import TrackedObject
from .selective_overlay import SelectiveOverlayPolicy


def _norm(value: str) -> str:
    return str(value or "").lower().replace("-", "_").replace(" ", "_")


class KeyMaterialExtractor:
    def __init__(self, pipeline: Any, overlay_mode: str = "event_selective") -> None:
        self.pipeline = pipeline
        self.overlay_mode = overlay_mode
        self.policy = SelectiveOverlayPolicy()
        self._contact_counters: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._active_pairs: set = set()

    def _reset_contact_state(self) -> None:
        self._contact_counters = {}
        self._active_pairs = set()

    def extract_assets(
        self,
        video_path: str | Path,
        event: PhysicalEvent,
        event_dir: str | Path,
        *,
        tracked_objects: Optional[List[TrackedObject]] = None,
    ) -> EventAssetPack:
        event_path = Path(event_dir)
        event_path.mkdir(parents=True, exist_ok=True)
        clip_path = event_path / "clip.mp4"
        preview_path = event_path / "preview.jpg"
        keyframes = [event_path / f"keyframe_{idx:02d}.jpg" for idx in range(1, 4)]
        event_json_path = event_path / "event.json"

        status = "ready"
        keyframe_selection: List[Dict[str, Any]] = []
        try:
            self._write_clip(video_path, event, clip_path)
            keyframe_selection = self._write_keyframes(video_path, event, preview_path, keyframes)
        except Exception as exc:
            status = f"failed:{exc}"

        keyframe_paths = [str(path) for path in keyframes if path.exists()]
        quality = self.score_asset_quality(
            event=event,
            asset_status=status,
            preview_path=preview_path,
            keyframe_paths=keyframe_paths,
            keyframe_selection=keyframe_selection,
        )
        asset_pack = EventAssetPack(
            event_id=event.event_id,
            clip_path=str(clip_path),
            preview_path=str(preview_path),
            keyframe_paths=keyframe_paths,
            event_json_path=str(event_json_path),
            overlay_mode=self.overlay_mode,
            asset_status=status,
            quality_score=quality["quality_score"],
            quality_grade=quality["quality_grade"],
            quality_reasons=quality["quality_reasons"],
        )
        event.asset_pack = asset_pack.to_dict()
        overlay_metadata = self._overlay_metadata(event, tracked_objects or [])
        if keyframe_selection:
            overlay_metadata["keyframe_selection"] = keyframe_selection
        dump_json(event_json_path, {"event": event.to_dict(), "asset_pack": asset_pack.to_dict(), "overlay_metadata": overlay_metadata})
        return asset_pack

    @staticmethod
    def score_asset_quality(
        *,
        event: PhysicalEvent,
        asset_status: str,
        preview_path: str | Path,
        keyframe_paths: List[str],
        keyframe_selection: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Score material usability without requiring live model inference."""
        score = 100.0
        reasons: List[str] = []
        duration = float(getattr(event, "duration_sec", 0.0) or 0.0)
        if duration <= 0:
            score -= 20
            reasons.append("duration_missing")
        elif duration < 0.8:
            score -= 18
            reasons.append("duration_too_short")
        elif duration < 1.5:
            score -= 8
            reasons.append("duration_short")
        elif duration > 30.0:
            score -= 18
            reasons.append("duration_too_long")
        elif duration > 18.0:
            score -= 8
            reasons.append("duration_long")
        else:
            reasons.append("duration_ok")

        status = str(asset_status or "")
        if status != "ready":
            score -= 35
            reasons.append("asset_not_ready")
        else:
            reasons.append("asset_ready")

        keyframe_count = len(keyframe_paths or [])
        if keyframe_count <= 0:
            score -= 25
            reasons.append("keyframes_missing")
        elif keyframe_count == 1:
            score -= 10
            reasons.append("keyframes_sparse")
        elif keyframe_count == 2:
            score -= 4
            reasons.append("keyframes_acceptable")
        else:
            score += 3
            reasons.append("keyframes_good")

        if Path(preview_path).exists():
            score += 4
            reasons.append("preview_present")
        else:
            score -= 14
            reasons.append("preview_missing")

        frames = keyframe_selection or []
        detection_counts: List[float] = []
        contacts = 0
        sharpness_values: List[float] = []
        area_values: List[float] = []
        for item in frames:
            detection_counts.append(KeyMaterialExtractor._safe_float(item.get("detection_count"), 0.0))
            quality = item.get("quality") if isinstance(item.get("quality"), dict) else {}
            if quality.get("contact"):
                contacts += 1
            sharpness_values.append(KeyMaterialExtractor._safe_float(quality.get("sharpness"), 0.0))
            area_values.append(KeyMaterialExtractor._safe_float(quality.get("box_area_ratio"), 0.0))

        if frames:
            avg_detections = sum(detection_counts) / max(1, len(detection_counts))
            if avg_detections <= 0:
                score -= 16
                reasons.append("detections_missing")
            elif avg_detections < 2:
                score -= 6
                reasons.append("detections_sparse")
            else:
                score += min(6.0, avg_detections)
                reasons.append("detections_good")

            if contacts > 0:
                score += min(8.0, contacts * 3.0)
                reasons.append("contact_evidence_present")
            elif str(getattr(event, "event_type", "")) in {"hand_object_interaction", "liquid_transfer", "object_move"}:
                score -= 8
                reasons.append("contact_evidence_missing")

            avg_sharpness = sum(sharpness_values) / max(1, len(sharpness_values))
            if avg_sharpness < 0.25:
                score -= 12
                reasons.append("sharpness_low")
            elif avg_sharpness < 0.45:
                score -= 5
                reasons.append("sharpness_marginal")
            else:
                score += 4
                reasons.append("sharpness_good")

            avg_area = sum(area_values) / max(1, len(area_values))
            if avg_area < 0.01:
                score -= 8
                reasons.append("box_area_tiny")
            elif avg_area > 0.75:
                score -= 5
                reasons.append("box_area_excessive")
            else:
                score += 3
                reasons.append("box_area_ok")
        else:
            score -= 10
            reasons.append("keyframe_metadata_missing")

        score = round(max(0.0, min(100.0, score)), 1)
        if score >= 85:
            grade = "excellent"
        elif score >= 70:
            grade = "good"
        elif score >= 50:
            grade = "fair"
        else:
            grade = "poor"
        return {
            "quality_score": score,
            "quality_grade": grade,
            "quality_reasons": list(dict.fromkeys(reasons)),
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ---- video / keyframe generation ------------------------------------

    def _write_clip(self, video_path: str | Path, event: PhysicalEvent, output_path: Path) -> None:
        """Step 1: ffmpeg cut raw clip. Step 2: annotate the short clip with live YOLO."""
        raw_clip = output_path.with_name(f"{output_path.stem}.raw_cut{output_path.suffix}")
        self._ffmpeg_cut(video_path, event.start_time_sec, event.end_time_sec, raw_clip)
        if not raw_clip.exists() or raw_clip.stat().st_size == 0:
            raise RuntimeError(f"Failed to cut raw clip for {event.event_id}")
        self._annotate_clip(raw_clip, event, output_path)
        raw_clip.unlink(missing_ok=True)

    def _ffmpeg_cut(self, video_path: str | Path, start_sec: float, end_sec: float, output: Path) -> None:
        start = max(0.0, start_sec)
        duration = max(0.1, end_sec - start)
        ffmpeg_exe = self._find_ffmpeg()
        if ffmpeg_exe:
            cmd = [
                ffmpeg_exe, "-y",
                "-ss", f"{start:.3f}", "-i", str(video_path),
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an", str(output),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
            if output.exists() and output.stat().st_size > 0:
                return
        # fallback: opencv copy
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sf = max(0, int(start * fps))
        ef = min(int(end_sec * fps), total - 1) if total > 0 else int(end_sec * fps)
        tmp = output.with_name(f"{output.stem}.raw_mp4v{output.suffix}")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if writer.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
            for _ in range(ef - sf + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
            writer.release()
        cap.release()
        if not self._transcode(tmp, output):
            if tmp.exists():
                shutil.move(str(tmp), str(output))

    def _annotate_clip(self, clip_path: Path, event: PhysicalEvent, output_path: Path) -> None:
        """Read the short clip, run live YOLO every frame, write annotated output."""
        self._reset_contact_state()
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            shutil.copy2(clip_path, output_path)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_output = output_path.with_name(f"{output_path.stem}.raw_mp4v{output_path.suffix}")
        writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            shutil.copy2(clip_path, output_path)
            return
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = event.start_time_sec + frame_idx / fps
            detections = self._detect_filtered(frame, frame_idx, ts, event)
            annotated = self._render_pil(frame, detections, event)
            writer.write(annotated)
            frame_idx += 1
        cap.release()
        writer.release()
        if raw_output != output_path and not self._transcode(raw_output, output_path):
            shutil.move(str(raw_output), str(output_path))

    def _write_keyframes(
        self,
        video_path: str | Path,
        event: PhysicalEvent,
        preview_path: Path,
        keyframes: List[Path],
    ) -> List[Dict[str, Any]]:
        candidates = self._select_keyframe_candidates(video_path, event, count=len(keyframes))
        if not candidates:
            raise ValueError(f"Cannot select keyframes from video: {video_path}")

        best_preview: Optional[Tuple[float, Path]] = None
        metadata: List[Dict[str, Any]] = []
        for candidate, path in zip(candidates, keyframes):
            annotated = self._render_pil(candidate["frame"], candidate["detections"], event)
            if not cv2.imwrite(str(path), annotated):
                continue
            score = float(candidate["score"])
            if best_preview is None or score > best_preview[0]:
                best_preview = (score, path)
            metadata.append(
                {
                    "path": str(path),
                    "timestamp_sec": round(float(candidate["timestamp_sec"]), 3),
                    "frame_idx": int(candidate["frame_idx"]),
                    "score": round(score, 4),
                    "detection_count": len(candidate["detections"]),
                    "quality": candidate.get("quality") or {},
                }
            )
        if best_preview is not None:
            shutil.copy2(best_preview[1], preview_path)
        elif keyframes and keyframes[0].exists():
            shutil.copy2(keyframes[0], preview_path)
        return metadata

    # ---- per-frame: live YOLO → event filter → contact filter → PIL render
    #
    # Contact thresholds are tuned for 720p/1080p lab footage. A debounce
    # window avoids single-frame flicker; keyframe selection still falls back
    # to high-quality event-relevant boxes if contact evidence is sparse.
    _CONTACT_IOU = float(os.environ.get("LABSOPGUARD_CONTACT_IOU", "0.015"))
    _CONTACT_DIST = float(os.environ.get("LABSOPGUARD_CONTACT_DIST_PX", "36"))
    _CONTACT_ENTER_FRAMES = int(os.environ.get("LABSOPGUARD_CONTACT_ENTER_FRAMES", "2"))
    _CONTACT_EXIT_FRAMES = int(os.environ.get("LABSOPGUARD_CONTACT_EXIT_FRAMES", "2"))
    _MATERIAL_CONF = float(os.environ.get("LABSOPGUARD_MATERIAL_YOLO_CONF", "0.10"))
    _MATERIAL_MAX_DET = int(os.environ.get("LABSOPGUARD_MATERIAL_MAX_DET", "96"))
    _KEYFRAME_SCAN_COUNT = int(os.environ.get("LABSOPGUARD_KEYFRAME_SCAN_COUNT", "13"))
    _MAX_RENDER_BOXES = int(os.environ.get("LABSOPGUARD_MATERIAL_MAX_RENDER_BOXES", "12"))

    def _select_keyframe_candidates(
        self,
        video_path: str | Path,
        event: PhysicalEvent,
        *,
        count: int,
    ) -> List[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        candidates: List[Dict[str, Any]] = []
        for ts in self._candidate_timestamps(event):
            frame_idx = max(0, int(float(ts) * fps))
            if total_frames > 0:
                frame_idx = min(frame_idx, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            raw_detections = self._run_material_yolo(frame, frame_idx, float(ts))
            display_detections = self._select_display_detections(event, raw_detections, temporal=False)
            score, quality = self._keyframe_score(frame, raw_detections, display_detections)
            candidates.append(
                {
                    "timestamp_sec": float(ts),
                    "frame_idx": int(frame_idx),
                    "frame": frame,
                    "detections": display_detections,
                    "score": score,
                    "quality": quality,
                }
            )
        cap.release()
        if not candidates:
            return []

        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        selected: List[Dict[str, Any]] = []
        start = max(0.0, float(event.start_time_sec))
        end = max(start + 0.1, float(event.end_time_sec))
        min_sep = max(0.2, (end - start) / max(1, count * 3))
        for candidate in candidates:
            if all(abs(float(candidate["timestamp_sec"]) - float(prev["timestamp_sec"])) >= min_sep for prev in selected):
                selected.append(candidate)
            if len(selected) >= count:
                break
        if len(selected) < count:
            selected_ids = {id(item) for item in selected}
            for candidate in candidates:
                if id(candidate) not in selected_ids:
                    selected.append(candidate)
                    selected_ids.add(id(candidate))
                if len(selected) >= count:
                    break
        selected.sort(key=lambda item: float(item["timestamp_sec"]))
        return selected[:count]

    def _candidate_timestamps(self, event: PhysicalEvent) -> List[float]:
        start = max(0.0, float(event.start_time_sec))
        end = max(start + 0.1, float(event.end_time_sec))
        safe_end = max(start, end - 0.05)
        timestamps: List[float] = []
        for value in event.key_timestamps or []:
            try:
                ts = float(value)
            except (TypeError, ValueError):
                continue
            if start <= ts <= end:
                timestamps.append(ts)
        scan_count = max(3, min(25, self._KEYFRAME_SCAN_COUNT))
        for ts in np.linspace(start, safe_end, num=scan_count):
            timestamps.append(float(ts))
        span = end - start
        timestamps.extend([start + span * 0.15, start + span * 0.5, start + span * 0.85])
        unique: List[float] = []
        seen: set[int] = set()
        for ts in sorted(max(start, min(safe_end, float(value))) for value in timestamps):
            bucket = int(round(ts * 100))
            if bucket in seen:
                continue
            seen.add(bucket)
            unique.append(ts)
        return unique

    @staticmethod
    def _keyframe_score(frame, raw_detections, display_detections) -> Tuple[float, Dict[str, Any]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 180.0)
        exposure = max(0.0, 1.0 - abs(float(gray.mean()) - 128.0) / 128.0)
        frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))
        box_area = 0.0
        for det in display_detections:
            x1, y1, x2, y2 = det.bbox
            box_area += max(0, x2 - x1) * max(0, y2 - y1)
        area_ratio = min(1.0, box_area / frame_area)
        has_hand = any(is_hand_label(det.class_name) for det in display_detections)
        has_object = any(not is_hand_label(det.class_name) for det in display_detections)
        contact = False
        hands = [det for det in display_detections if is_hand_label(det.class_name)]
        objects = [det for det in display_detections if not is_hand_label(det.class_name)]
        for hand in hands:
            for obj in objects:
                if KeyMaterialExtractor._is_contact(hand.bbox, obj.bbox):
                    contact = True
                    break
            if contact:
                break
        confidence_sum = sum(float(det.confidence) for det in display_detections)
        score = (
            min(1.6, confidence_sum)
            + min(1.2, len(display_detections) * 0.18)
            + (0.85 if contact else 0.0)
            + (0.35 if has_hand and has_object else 0.0)
            + sharpness * 0.25
            + exposure * 0.15
            + area_ratio * 0.35
            + min(0.3, len(raw_detections) * 0.03)
        )
        quality = {
            "sharpness": round(sharpness, 4),
            "exposure": round(exposure, 4),
            "box_area_ratio": round(area_ratio, 4),
            "has_hand": has_hand,
            "has_object": has_object,
            "contact": contact,
            "raw_detection_count": len(raw_detections),
        }
        return score, quality

    def _run_material_yolo(self, frame, frame_idx: int, ts: float):
        from .schemas import DetectionBox

        if getattr(self.pipeline, "yolo_model", None) is None:
            return []
        settings = getattr(self.pipeline, "settings", None)
        old_conf = getattr(settings, "confidence_threshold", None) if settings is not None else None
        old_max = getattr(settings, "max_detections", None) if settings is not None else None
        try:
            if settings is not None:
                if old_conf is not None:
                    settings.confidence_threshold = min(float(old_conf), self._MATERIAL_CONF)
                else:
                    settings.confidence_threshold = self._MATERIAL_CONF
                if old_max is not None:
                    settings.max_detections = max(int(old_max), self._MATERIAL_MAX_DET)
            raw = self.pipeline._run_yolo(frame, frame_idx, ts)
        finally:
            if settings is not None:
                if old_conf is not None:
                    settings.confidence_threshold = old_conf
                if old_max is not None:
                    settings.max_detections = old_max
        return [
            DetectionBox(
                tuple(int(v) for v in det.bbox),
                str(det.class_name),
                float(det.confidence),
            )
            for det in raw
        ]

    def _select_display_detections(self, event: PhysicalEvent, detections, *, temporal: bool):
        class_filtered = self.policy.filter_detections(event, detections)
        if not class_filtered:
            return []
        if temporal:
            interacting = self._keep_interacting(class_filtered)
        else:
            interacting = self._spatial_interacting(class_filtered)
        if interacting:
            return self._rank_detections_for_event(event, interacting)
        return self._rank_detections_for_event(event, class_filtered)

    def _spatial_interacting(self, detections):
        hands = [d for d in detections if is_hand_label(d.class_name)]
        objects = [d for d in detections if not is_hand_label(d.class_name)]
        active_h: set[int] = set()
        active_o: set[int] = set()
        for hi, hand in enumerate(hands):
            for oi, obj in enumerate(objects):
                if self._is_contact(hand.bbox, obj.bbox):
                    active_h.add(hi)
                    active_o.add(oi)
        return [hands[i] for i in active_h] + [objects[i] for i in active_o]

    def _rank_detections_for_event(self, event: PhysicalEvent, detections, limit: Optional[int] = None):
        limit = max(1, int(limit or self._MAX_RENDER_BOXES))
        ranked = sorted(detections, key=lambda det: self._detection_score(event, det), reverse=True)
        selected = []
        for det in ranked:
            label = _norm(det.class_name)
            duplicate = False
            for existing in selected:
                if label == _norm(existing.class_name) and _bbox_iou(det.bbox, existing.bbox) >= 0.75:
                    duplicate = True
                    break
            if duplicate:
                continue
            selected.append(det)
            if len(selected) >= limit:
                break
        return selected

    def _detection_score(self, event: PhysicalEvent, det) -> float:
        label = _norm(det.class_name)
        tokens = self._event_class_tokens(event)
        score = float(det.confidence)
        if is_hand_label(label):
            score += 0.25
        if any(label == token or label in token or token in label for token in tokens):
            score += 0.45
        if event.event_type == "liquid_transfer" and (is_container_label(label) or is_tool_label(label)):
            score += 0.25
        elif event.event_type == "panel_operation" and is_panel_label(label):
            score += 0.25
        elif event.event_type == "container_state_change" and (is_container_label(label) or is_lid_label(label)):
            score += 0.25
        elif event.event_type == "object_move" and not is_hand_label(label):
            score += 0.12
        x1, y1, x2, y2 = det.bbox
        area = max(0, x2 - x1) * max(0, y2 - y1)
        score += min(0.18, area / 250000.0)
        return score

    @staticmethod
    def _event_class_tokens(event: PhysicalEvent) -> set[str]:
        tokens: set[str] = set()
        for value in [
            *(event.involved_objects or []),
            *(getattr(event, "related_detection_classes", []) or []),
            event.dominant_object,
        ]:
            text = _norm(str(value or ""))
            if text:
                tokens.add(text)
        for container in (event.source_container, event.target_container):
            if container and isinstance(container, dict):
                for key in ("class_name", "object_name", "display_name"):
                    text = _norm(str(container.get(key) or ""))
                    if text:
                        tokens.add(text)
        return tokens

    def _detect_filtered(self, frame, frame_idx: int, ts: float, event: PhysicalEvent):
        """Run live YOLO and keep ranked event-relevant boxes for material clips."""
        detections = self._run_material_yolo(frame, frame_idx, ts)
        return self._select_display_detections(event, detections, temporal=True)

    @staticmethod
    def _pair_key(hand, obj) -> Tuple[str, str]:
        # Key by class-name pair only. Instances of the same class pair
        # collapse to one debounce counter, which is fine for this domain
        # (rarely more than 1-2 hands and 1-2 target objects per event) and
        # avoids the counter resetting whenever a bbox jitters across a grid.
        return (_norm(hand.class_name), _norm(obj.class_name))

    def _keep_interacting(self, detections):
        """Keep only hand/object pairs that have been in contact for N consecutive frames,
        and drop them only after being separated for N consecutive frames (hysteresis)."""
        hands = [d for d in detections if is_hand_label(d.class_name)]
        objects = [d for d in detections if not is_hand_label(d.class_name)]
        if not hands or not objects:
            # no candidates this frame: advance miss counter for all active pairs
            self._advance_misses()
            return []

        seen_pairs: set = set()
        contacts_this_frame: set = set()
        for hi, hand in enumerate(hands):
            for oi, obj in enumerate(objects):
                key = self._pair_key(hand, obj)
                seen_pairs.add(key)
                counter = self._contact_counters.setdefault(key, {"hit": 0, "miss": 0})
                if self._is_contact(hand.bbox, obj.bbox):
                    counter["hit"] += 1
                    counter["miss"] = 0
                    if counter["hit"] >= self._CONTACT_ENTER_FRAMES:
                        self._active_pairs.add(key)
                        contacts_this_frame.add((hi, oi, key))
                else:
                    counter["miss"] += 1
                    counter["hit"] = 0
                    if counter["miss"] >= self._CONTACT_EXIT_FRAMES:
                        self._active_pairs.discard(key)
                    elif key in self._active_pairs:
                        # still within exit-debounce window: keep showing
                        contacts_this_frame.add((hi, oi, key))

        # any previously-tracked pair not seen in YOLO this frame counts as a miss
        for key in list(self._contact_counters.keys()):
            if key not in seen_pairs:
                counter = self._contact_counters[key]
                counter["miss"] += 1
                counter["hit"] = 0
                if counter["miss"] >= self._CONTACT_EXIT_FRAMES:
                    self._active_pairs.discard(key)
                    self._contact_counters.pop(key, None)

        if not contacts_this_frame:
            return []
        active_hands = {hi for hi, _, _ in contacts_this_frame}
        active_objects = {oi for _, oi, _ in contacts_this_frame}
        return [hands[i] for i in active_hands] + [objects[i] for i in active_objects]

    def _advance_misses(self) -> None:
        for key in list(self._contact_counters.keys()):
            counter = self._contact_counters[key]
            counter["miss"] += 1
            counter["hit"] = 0
            if counter["miss"] >= self._CONTACT_EXIT_FRAMES:
                self._active_pairs.discard(key)
                self._contact_counters.pop(key, None)

    @classmethod
    def _is_contact(cls, a: Tuple, b: Tuple) -> bool:
        return _bbox_iou(a, b) >= cls._CONTACT_IOU or _bbox_edge_distance(a, b) <= cls._CONTACT_DIST

    def _annotate_frame(self, frame, frame_idx: int, ts: float, event: PhysicalEvent):
        # Keyframe rendering does not have a continuous temporal context;
        # use the strict spatial test only, so single-frame judgement is tight.
        detections = self._detect_filtered_strict(frame, frame_idx, ts, event)
        return self._render_pil(frame, detections, event)

    def _detect_filtered_strict(self, frame, frame_idx: int, ts: float, event: PhysicalEvent):
        """Strict spatial-only filter used for standalone keyframes (no temporal debounce)."""
        detections = self._run_material_yolo(frame, frame_idx, ts)
        return self._select_display_detections(event, detections, temporal=False)

    # ---- PIL rendering (same quality as annotated video) ----------------

    @classmethod
    @lru_cache(maxsize=1)
    def _resolve_font_path(cls) -> Optional[str]:
        env_value = os.environ.get("LABSOPGUARD_FONT_PATH")
        if env_value and Path(env_value).exists():
            return env_value
        for cand in (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        ):
            if Path(cand).exists():
                return cand
        return None

    @classmethod
    @lru_cache(maxsize=16)
    def _get_font(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        path = cls._resolve_font_path()
        if path:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    @staticmethod
    def _with_alpha(color: Tuple[int, int, int], alpha: int) -> Tuple[int, int, int, int]:
        return (color[0], color[1], color[2], alpha)

    @staticmethod
    def _class_color(class_name: str) -> Tuple[int, int, int]:
        palette = [
            (0, 191, 255),
            (76, 175, 80),
            (255, 167, 38),
            (244, 81, 30),
            (126, 87, 194),
            (38, 198, 218),
            (171, 71, 188),
            (255, 112, 67),
        ]
        digest = hashlib.md5(class_name.lower().encode("utf-8")).hexdigest()
        return palette[int(digest[:2], 16) % len(palette)]

    @classmethod
    def _display_text(cls, value: object) -> str:
        text = " ".join(str(value or "").split())
        if cls._resolve_font_path():
            return text
        return text.encode("ascii", errors="replace").decode("ascii")

    @staticmethod
    def _measure_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> Tuple[int, int]:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return (right - left, bottom - top)

    def _render_pil(self, frame, detections, event: PhysicalEvent):
        from .schemas import DetectionBox

        base = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        label_font = self._get_font(18)
        title_font = self._get_font(22)

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self._class_color(det.class_name)
            draw.rounded_rectangle(
                [x1, y1, x2, y2], radius=10,
                outline=self._with_alpha(color, 255), width=3,
            )
            label = f"{self._display_text(det.class_name)} {det.confidence:.2f}"
            lw, lh = self._measure_text(draw, label, label_font)
            lx1 = x1
            ly1 = max(4, y1 - lh - 14)
            lx2 = min(base.size[0] - 4, lx1 + lw + 16)
            ly2 = ly1 + lh + 10
            draw.rounded_rectangle(
                [lx1, ly1, lx2, ly2], radius=8,
                fill=self._with_alpha(color, 220),
            )
            draw.text((lx1 + 8, ly1 + 4), label, font=label_font, fill=(255, 255, 255, 255))

        # title banner
        display = self._display_text(getattr(event, "display_name", None) or event.event_type)
        tw, th = self._measure_text(draw, display, title_font)
        bx2 = min(base.size[0] - 12, 24 + tw + 24)
        by2 = 16 + th + 16
        draw.rounded_rectangle(
            [12, 10, bx2, by2], radius=12,
            fill=(16, 18, 24, 200), outline=(255, 255, 255, 50), width=1,
        )
        draw.text((24, 16), display, font=title_font, fill=(244, 247, 250, 255))

        # evidence grade badge
        grade = getattr(event, "evidence_grade", None) or ""
        if grade:
            grade_colors = {"A": (76, 175, 80), "B": (255, 167, 38), "C": (244, 81, 30), "D": (229, 57, 53)}
            gc = grade_colors.get(grade[0].upper(), (158, 158, 158))
            gtext = f"Grade {grade}"
            gw, gh = self._measure_text(draw, gtext, label_font)
            gx = base.size[0] - gw - 28
            draw.rounded_rectangle(
                [gx, 12, gx + gw + 20, 12 + gh + 10], radius=8,
                fill=self._with_alpha(gc, 220),
            )
            draw.text((gx + 10, 16), gtext, font=label_font, fill=(255, 255, 255, 255))

        merged = Image.alpha_composite(base, overlay).convert("RGB")
        return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)

    # ---- metadata / transcode helpers -----------------------------------

    @staticmethod
    def _overlay_metadata(event: PhysicalEvent, tracked_objects: List[TrackedObject]) -> Dict[str, Any]:
        by_id = {track.track_id: track for track in tracked_objects}
        related = event.related_tracks or event.involved_track_ids or []
        high_risk = [
            {
                "track_id": track_id,
                "id_switch_risk": by_id[track_id].id_switch_risk,
                "occlusion_ratio": by_id[track_id].occlusion_ratio,
                "fragment_count": by_id[track_id].fragment_count,
            }
            for track_id in related
            if track_id in by_id and (by_id[track_id].id_switch_risk >= 0.45 or by_id[track_id].fragment_count > 1)
        ]
        missing = [track_id for track_id in related if track_id not in by_id]
        fallback_reason = None
        if missing:
            fallback_reason = "related_track_missing_from_track_stream"
        elif not related:
            fallback_reason = "no_related_tracks_available_class_filter_fallback"
        return {
            "overlay_mode": "live_yolo_event_selective",
            "high_risk_tracks": high_risk,
            "missing_related_tracks": missing,
            "overlay_fallback_reason": fallback_reason,
            "evidence_grade": event.evidence_grade,
            "review_status": event.review_status,
        }

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

    @staticmethod
    def _transcode(source: Path, target: Path) -> bool:
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = None
        if ffmpeg_exe:
            tmp = target.with_name(f"{target.stem}.h264_tmp{target.suffix}")
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(source),
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
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
                if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                    shutil.move(str(tmp), str(target))
                    source.unlink(missing_ok=True)
                    return True
                tmp.unlink(missing_ok=True)
            except Exception:
                tmp.unlink(missing_ok=True)

        tmp = target.with_name(f"{target.stem}.h264_tmp{target.suffix}")
        cap = cv2.VideoCapture(str(source))
        writer = None
        try:
            if not cap.isOpened():
                return False
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if width <= 0 or height <= 0:
                return False
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height))
            if not writer.isOpened():
                return False
            written = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
                written += 1
            writer.release()
            writer = None
            cap.release()
            if written <= 0 or not tmp.exists() or tmp.stat().st_size <= 0:
                tmp.unlink(missing_ok=True)
                return False
            shutil.move(str(tmp), str(target))
            source.unlink(missing_ok=True)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            return False
        finally:
            if writer is not None:
                writer.release()
            cap.release()


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _bbox_edge_distance(a, b) -> float:
    import math
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)
