from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiment.vlm_client import DashScopeVLClient
from labsopguard.config import RuntimeSettings
from labsopguard.resilience import CircuitBreaker, RateLimiter, RetryConfig, resilient_call


logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    keypoints: Optional[List[Tuple[int, int]]] = None


@dataclass
class FrameAnalysis:
    frame_idx: int
    timestamp_sec: float
    detections: List[DetectionResult]
    scene_description: str
    detected_activities: List[str]
    object_labels: List[str]
    step_indicators: List[str]
    ppe_status: Dict[str, bool]
    vlm_confidence: float
    alerts: List[str]
    alert_details: List[Dict[str, object]]


def _normalize_detection_label(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _bbox_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def _clone_detection_for_frame(det: DetectionResult, frame_idx: int, timestamp_sec: float) -> DetectionResult:
    return DetectionResult(
        frame_idx=frame_idx,
        timestamp_sec=timestamp_sec,
        bbox=det.bbox,
        class_name=det.class_name,
        confidence=det.confidence,
        keypoints=det.keypoints,
    )


class TemporalDetectionSmoother:
    def __init__(self, *, min_hits: int = 3, hold_frames: int = 5, iou_threshold: float = 0.35) -> None:
        self.min_hits = max(1, int(min_hits))
        self.hold_frames = max(0, int(hold_frames))
        self.iou_threshold = float(iou_threshold)
        self._tracks: List[Dict[str, object]] = []

    def update(self, detections: List[DetectionResult], frame_idx: int, timestamp_sec: float) -> List[DetectionResult]:
        matched_tracks: set[int] = set()
        for det in detections:
            label = _normalize_detection_label(det.class_name)
            best_track = -1
            best_iou = 0.0
            for track_idx, track in enumerate(self._tracks):
                if track_idx in matched_tracks:
                    continue
                if track.get("label") != label:
                    continue
                last_det = track.get("detection")
                if not isinstance(last_det, DetectionResult):
                    continue
                score = _bbox_iou(last_det.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track = track_idx
            if best_track >= 0 and best_iou >= self.iou_threshold:
                track = self._tracks[best_track]
                track["detection"] = det
                track["hits"] = int(track.get("hits", 0)) + 1
                track["missed"] = 0
                matched_tracks.add(best_track)
            else:
                self._tracks.append({"label": label, "detection": det, "hits": 1, "missed": 0})
                matched_tracks.add(len(self._tracks) - 1)

        stable: List[DetectionResult] = []
        next_tracks: List[Dict[str, object]] = []
        for track_idx, track in enumerate(self._tracks):
            det = track.get("detection")
            if not isinstance(det, DetectionResult):
                continue
            if track_idx not in matched_tracks:
                track["missed"] = int(track.get("missed", 0)) + 1
            missed = int(track.get("missed", 0))
            hits = int(track.get("hits", 0))
            if missed <= self.hold_frames:
                next_tracks.append(track)
                if hits >= self.min_hits:
                    stable.append(_clone_detection_for_frame(det, frame_idx, timestamp_sec))
        self._tracks = next_tracks
        stable.sort(key=lambda item: item.confidence, reverse=True)
        return stable


class AdaptiveFrameSampler:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings

    def sample(self, fps: float, total_frames: int) -> List[Tuple[int, float]]:
        duration_sec = total_frames / fps if fps else 0.0
        if duration_sec <= 0:
            return []
        target_frames = int(min(self.settings.max_frames, max(self.settings.min_frames, duration_sec // max(self.settings.sample_interval_sec, 1.0))))
        target_frames = max(1, target_frames)
        if duration_sec <= 20:
            target_frames = max(target_frames, min(self.settings.max_frames, 12))
        interval = duration_sec / target_frames
        sampled: List[Tuple[int, float]] = []
        for idx in range(target_frames):
            ts = min(duration_sec, round(idx * interval, 3))
            frame_idx = min(total_frames - 1, int(ts * fps))
            sampled.append((frame_idx, ts))
        sampled.append((total_frames - 1, round(duration_sec, 3)))
        dedup = {}
        for frame_idx, ts in sampled:
            dedup[frame_idx] = ts
        return [(frame_idx, dedup[frame_idx]) for frame_idx in sorted(dedup)]


class VideoAnalysisPipeline:
    def __init__(
        self,
        settings: RuntimeSettings,
        yolo_model_path: Optional[str] = None,
        vlm_api_key: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        sample_interval: Optional[float] = None,
        max_frames: Optional[int] = None,
    ):
        self.settings = settings
        if sample_interval is not None:
            self.settings.sample_interval_sec = sample_interval
        if max_frames is not None:
            self.settings.max_frames = max_frames
        self.sampler = AdaptiveFrameSampler(self.settings)
        self.yolo_model = None
        self.yolo_model_path: Optional[str] = None
        self.allowed_detection_labels = {
            _normalize_detection_label(item)
            for item in (self.settings.allowed_detection_labels or [])
            if _normalize_detection_label(item)
        }
        model_path = yolo_model_path or self.settings.yolo_model_path
        if model_path:
            candidate_path = Path(str(model_path))
            if not candidate_path.is_absolute():
                candidate_path = self.settings.project_root / candidate_path
            if candidate_path.exists():
                model_path = str(candidate_path.resolve())
        if not model_path or not Path(str(model_path)).exists():
            try:
                from labsopguard.detectors import resolve_yolo26_weights_path

                resolved = resolve_yolo26_weights_path(str(model_path) if model_path else None, allow_fallbacks=not self.settings.strict_model)
                if resolved:
                    model_path = str(resolved)
            except Exception as exc:
                if self.settings.strict_model:
                    raise RuntimeError(f"YOLO26 model resolution failed in strict mode: {exc}") from exc
        if not model_path or not Path(str(model_path)).exists():
            if self.settings.strict_model:
                configured = yolo_model_path or self.settings.yolo_model_path or "configs/model/detection_runtime.yaml:model"
                raise FileNotFoundError(f"YOLO26 strict model path does not exist: {configured}")
        else:
            try:
                from ultralytics import YOLO

                self.yolo_model = YOLO(str(model_path))
                self.yolo_model_path = str(Path(str(model_path)).resolve())
                self.settings.yolo_model_path = self.yolo_model_path
            except Exception as exc:
                self.yolo_model = None
                if self.settings.strict_model:
                    raise RuntimeError(f"Failed to initialize YOLO26 strict model: {model_path}") from exc
        self.vlm_client = None
        if vlm_api_key:
            try:
                self.vlm_client = DashScopeVLClient(
                    api_key=vlm_api_key,
                    base_url=vlm_base_url,
                    model=os.environ.get("KEY_ACTION_VLM_MODEL")
                    or os.environ.get("QWEN_VL_MODEL")
                    or os.environ.get("VLM_MODEL")
                    or DashScopeVLClient.DEFAULT_MODEL,
                )
            except Exception as exc:
                logger.warning("VLM client disabled: %s", exc)
                self.vlm_client = None
        self._vlm_circuit_breaker = CircuitBreaker(name="vlm_api")
        self._vlm_retry_config = RetryConfig(
            max_retries=int(os.environ.get("LABSOPGUARD_VLM_MAX_RETRIES", "3")),
            backoff_factor=float(os.environ.get("LABSOPGUARD_VLM_BACKOFF_FACTOR", "2.0")),
            timeout=float(os.environ.get("LABSOPGUARD_VLM_TIMEOUT", "60")),
        )
        self._vlm_rate_limiter = RateLimiter(
            calls_per_second=float(os.environ.get("LABSOPGUARD_VLM_CALLS_PER_SECOND", "2.0"))
        )
        self._last_vlm_status = "available" if self.vlm_client is not None else "disabled"

    @staticmethod
    def _coerce_list(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(value)]

    @staticmethod
    def _extract_scene_json(value: object) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        text = str(value or "").strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = text.strip("`").replace("json\n", "", 1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _has_actor_presence(detections: List[DetectionResult], scene_payload: Optional[Dict[str, Any]] = None) -> bool:
        payload = scene_payload or {}
        text = " ".join(
            [
                str(payload.get("description") or payload.get("scene_summary") or ""),
                " ".join(VideoAnalysisPipeline._coerce_list(payload.get("detected_activities") or payload.get("actions"))),
                " ".join(VideoAnalysisPipeline._coerce_list(payload.get("object_labels") or payload.get("objects"))),
            ]
        ).lower()
        negative_actor = any(phrase in text for phrase in ["未见操作人员", "无操作人员", "未见人员", "无人", "no operator", "no person"])
        if negative_actor:
            return False
        actor_keywords = ["person", "operator", "human", "hand", "arm", "face", "人员", "操作员", "手部", "手"]
        if any(keyword in text for keyword in actor_keywords):
            return True
        return any(any(keyword in det.class_name.lower() for keyword in ["person", "human", "hand", "arm", "face"]) for det in detections)

    @staticmethod
    def _ppe_required(scene_payload: Optional[Dict[str, Any]] = None) -> bool:
        payload = scene_payload or {}
        text = " ".join(
            [
                str(payload.get("description") or payload.get("scene_summary") or ""),
                " ".join(VideoAnalysisPipeline._coerce_list(payload.get("detected_activities") or payload.get("actions"))),
                " ".join(VideoAnalysisPipeline._coerce_list(payload.get("step_indicators"))),
            ]
        ).lower()
        passive_terms = ["静置", "摆放", "无人", "未见操作人员", "no operator", "no person"]
        if any(term in text for term in passive_terms):
            return False
        active_terms = ["transfer", "pipette", "pour", "weigh", "mix", "取", "移液", "转移", "倒入", "称量", "混合", "操作"]
        return any(term in text for term in active_terms)

    def _build_alerts(
        self,
        ppe_status: Dict[str, bool],
        detections: Optional[List[DetectionResult]] = None,
        scene_payload: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        detections = detections or []
        if not self._has_actor_presence(detections, scene_payload):
            return ["ppe_not_applicable:no_actor_detected"]
        if not self._ppe_required(scene_payload):
            return ["ppe_not_applicable:scene_not_ppe_relevant"]
        alert_order = ("gloves", "goggles", "lab_coat")
        return [f"missing_{item}" for item in alert_order if not bool(ppe_status.get(item, False))]

    def _build_alert_details(
        self,
        alert_keys: List[str],
        *,
        frame_idx: Optional[int] = None,
        timestamp_sec: Optional[float] = None,
        detections: Optional[List[DetectionResult]] = None,
        rule_basis: str = "",
    ) -> List[Dict[str, object]]:
        rule_map = self.settings.alert_rules or {}
        details: List[Dict[str, object]] = []
        for alert_key in alert_keys:
            if alert_key.startswith("ppe_not_applicable"):
                reason = alert_key.split(":", 1)[1] if ":" in alert_key else "not_applicable"
                details.append(
                    {
                        "rule_id": alert_key,
                        "severity": "info",
                        "color": "#607D8B",
                        "title": "PPE 不适用",
                        "message": "当前帧未检测到需要 PPE 判定的人员或操作场景。",
                        "related_classes": [],
                        "source_frame": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "evidence_refs": [{"type": "video_frame", "frame_idx": frame_idx, "timestamp_sec": timestamp_sec}],
                        "rule_basis": rule_basis or f"person/step/scene gating returned {reason}",
                        "confidence": 1.0,
                    }
                )
                continue
            rule = rule_map.get(alert_key, {}) if isinstance(rule_map, dict) else {}
            related_classes = rule.get("related_classes", []) if isinstance(rule, dict) else []
            details.append(
                {
                    "rule_id": alert_key,
                    "severity": str(rule.get("severity", "medium")),
                    "color": str(rule.get("color", "#E53935")),
                    "title": str(rule.get("title", alert_key)),
                    "message": str(rule.get("message", alert_key)),
                    "related_classes": list(related_classes) if isinstance(related_classes, list) else [],
                    "source_frame": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "evidence_refs": [{"type": "video_frame", "frame_idx": frame_idx, "timestamp_sec": timestamp_sec}],
                    "rule_basis": rule_basis or "actor present and PPE-relevant operation detected; detector/VLM PPE evidence is missing",
                    "confidence": 0.8,
                }
            )
        return details

    def analyze_video(self, video_path: str) -> List[FrameAnalysis]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled = self.sampler.sample(fps, total_frames)

        # Smart VLM frame selection: run YOLO on all sampled frames first,
        # then pick the most informative subset for expensive VLM calls.
        yolo_results: List[Tuple[int, float, List[DetectionResult], np.ndarray]] = []
        for frame_idx, ts in sampled:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            detections = self._run_yolo(frame, frame_idx, ts)
            yolo_results.append((frame_idx, ts, detections, frame))

        vlm_indices = self._select_vlm_frames(yolo_results, self.settings.max_vlm_frames)

        results: List[FrameAnalysis] = []
        prev_vlm_payload: Dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, (frame_idx, ts, detections, frame) in enumerate(yolo_results):
                if i in vlm_indices:
                    frame_path = Path(temp_dir) / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    vlm_payload = self._run_vlm(str(frame_path), detections)
                    prev_vlm_payload = vlm_payload
                else:
                    # Reuse previous VLM result for low-change frames
                    vlm_payload = prev_vlm_payload if prev_vlm_payload else {
                        "description": "", "detected_activities": [],
                        "object_labels": sorted({d.class_name for d in detections}),
                        "step_indicators": [], "ppe_status": {}, "confidence": 0.0,
                    }
                ppe_status = self._fuse_ppe(vlm_payload.get("ppe_status", {}), detections)
                alerts = self._build_alerts(ppe_status, detections, vlm_payload)
                alert_details = self._build_alert_details(
                    alerts,
                    frame_idx=frame_idx,
                    timestamp_sec=ts,
                    detections=detections,
                    rule_basis="person_presence_gating + step_aware_gating + scene_aware_gating",
                )
                results.append(
                    FrameAnalysis(
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        detections=detections,
                        scene_description=vlm_payload.get("description", ""),
                        detected_activities=vlm_payload.get("detected_activities", []),
                        object_labels=vlm_payload.get("object_labels", []),
                        step_indicators=vlm_payload.get("step_indicators", []),
                        ppe_status=ppe_status,
                        vlm_confidence=float(vlm_payload.get("confidence", 0.0)),
                        alerts=alerts,
                        alert_details=alert_details,
                    )
                )
        cap.release()
        return results

    def _select_vlm_frames(
        self,
        yolo_results: List[Tuple[int, float, List["DetectionResult"], np.ndarray]],
        max_vlm: int,
    ) -> set:
        """Select the most informative frame indices for VLM analysis.

        Strategy: always include first and last frame, then rank by:
        1. Frames where new object classes appear
        2. Frames with highest detection count change from previous
        3. Evenly spaced coverage to fill remaining budget
        """
        if len(yolo_results) <= max_vlm:
            return set(range(len(yolo_results)))

        scores: List[float] = []
        prev_classes: set = set()
        prev_count = 0

        for i, (_, _, detections, _) in enumerate(yolo_results):
            cur_classes = {d.class_name for d in detections}
            new_classes = cur_classes - prev_classes
            count_change = abs(len(detections) - prev_count)

            score = len(new_classes) * 3.0 + count_change * 0.5
            scores.append(score)

            prev_classes = cur_classes
            prev_count = len(detections)

        selected: set = set()
        # Always include first and last
        selected.add(0)
        selected.add(len(yolo_results) - 1)

        # Add top-scoring frames
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in ranked:
            if len(selected) >= max_vlm:
                break
            selected.add(idx)

        # Fill remaining budget with evenly spaced frames
        if len(selected) < max_vlm:
            step = max(1, len(yolo_results) // (max_vlm - len(selected) + 1))
            for idx in range(0, len(yolo_results), step):
                if len(selected) >= max_vlm:
                    break
                selected.add(idx)

        return selected

    def _run_yolo(self, frame: np.ndarray, frame_idx: int, ts: float) -> List[DetectionResult]:
        if self.yolo_model is None:
            return []
        detections: List[DetectionResult] = []
        try:
            predict_kwargs: Dict[str, object] = {
                "conf": self.settings.confidence_threshold,
                "iou": self.settings.iou_threshold,
                "max_det": self.settings.max_detections,
                "device": self.settings.device,
                "verbose": False,
            }
            if self.settings.yolo_imgsz:
                predict_kwargs["imgsz"] = self.settings.yolo_imgsz
            yolo_results = self.yolo_model(frame, **predict_kwargs)
        except ValueError as exc:
            message = str(exc)
            if "Invalid CUDA" not in message and "CUDA" not in message:
                raise
            self.settings.device = "cpu"
            predict_kwargs = {
                "conf": self.settings.confidence_threshold,
                "iou": self.settings.iou_threshold,
                "max_det": self.settings.max_detections,
                "device": "cpu",
                "verbose": False,
            }
            if self.settings.yolo_imgsz:
                predict_kwargs["imgsz"] = self.settings.yolo_imgsz
            yolo_results = self.yolo_model(frame, **predict_kwargs)
        except TypeError:
            yolo_results = self.yolo_model(frame, verbose=False)
        for result in yolo_results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            names = getattr(result, "names", {}) or {}
            for idx, box in enumerate(boxes):
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                if self.allowed_detection_labels and _normalize_detection_label(class_name) not in self.allowed_detection_labels:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                keypoints = None
                if getattr(result, "keypoints", None) is not None and len(result.keypoints.xy) > idx:
                    kpts = result.keypoints.xy[idx].cpu().numpy()
                    keypoints = [(int(x), int(y)) for x, y in kpts if x > 0 and y > 0]
                detections.append(
                    DetectionResult(
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        bbox=(x1, y1, x2, y2),
                        class_name=class_name,
                        confidence=confidence,
                        keypoints=keypoints,
                    )
                )
        return detections

    def _run_yolo_batch(
        self, frames: List[np.ndarray], frame_indices: List[int], timestamps: List[float]
    ) -> List[List[DetectionResult]]:
        """Batch YOLO inference for improved GPU throughput."""
        if self.yolo_model is None or len(frames) == 0:
            return [[] for _ in frames]

        try:
            predict_kwargs: Dict[str, object] = {
                "conf": self.settings.confidence_threshold,
                "iou": self.settings.iou_threshold,
                "max_det": self.settings.max_detections,
                "device": self.settings.device,
                "verbose": False,
            }
            if self.settings.yolo_imgsz:
                predict_kwargs["imgsz"] = self.settings.yolo_imgsz
            batch_results = self.yolo_model(frames, **predict_kwargs)
        except (ValueError, RuntimeError):
            # OOM or CUDA error: fall back to sequential
            return [self._run_yolo(f, fi, ts) for f, fi, ts in zip(frames, frame_indices, timestamps)]
        except TypeError:
            return [self._run_yolo(f, fi, ts) for f, fi, ts in zip(frames, frame_indices, timestamps)]

        all_detections: List[List[DetectionResult]] = []
        for result_idx, result in enumerate(batch_results):
            frame_idx = frame_indices[result_idx] if result_idx < len(frame_indices) else 0
            ts = timestamps[result_idx] if result_idx < len(timestamps) else 0.0
            detections: List[DetectionResult] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                names = getattr(result, "names", {}) or {}
                for idx, box in enumerate(boxes):
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                    if self.allowed_detection_labels and _normalize_detection_label(class_name) not in self.allowed_detection_labels:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                    keypoints = None
                    if getattr(result, "keypoints", None) is not None and len(result.keypoints.xy) > idx:
                        kpts = result.keypoints.xy[idx].cpu().numpy()
                        keypoints = [(int(x), int(y)) for x, y in kpts if x > 0 and y > 0]
                    detections.append(
                        DetectionResult(
                            frame_idx=frame_idx,
                            timestamp_sec=ts,
                            bbox=(x1, y1, x2, y2),
                            class_name=class_name,
                            confidence=confidence,
                            keypoints=keypoints,
                        )
                    )
            all_detections.append(detections)
        return all_detections

    def _run_vlm(self, image_path: str, detections: List[DetectionResult]) -> Dict[str, object]:
        if self.vlm_client is None:
            return {
                "description": "vlm_unavailable",
                "detected_activities": [],
                "object_labels": sorted({det.class_name for det in detections}),
                "step_indicators": [],
                "ppe_status": {},
                "confidence": 0.0,
            }
        prompt = (
            "Return strict JSON for a laboratory frame. "
            "Describe current operation, objects, protocol hints, and PPE. "
            "Use keys description, detected_activities, object_labels, step_indicators, "
            "ppe_status, confidence. PPE must include gloves, goggles, lab_coat. "
            "Keep description concise and mention any visible safety risk."
        )
        description = resilient_call(
            self.vlm_client.describe_scene,
            image_path=image_path,
            prompt=prompt,
            retry_config=self._vlm_retry_config,
            circuit_breaker=self._vlm_circuit_breaker,
            rate_limiter=self._vlm_rate_limiter,
            fallback=None,
        )
        if description is None:
            self._last_vlm_status = "temporarily_unavailable"
            return {
                "description": "vlm_temporarily_unavailable",
                "detected_activities": [],
                "object_labels": sorted({det.class_name for det in detections}),
                "step_indicators": [],
                "ppe_status": {},
                "confidence": 0.0,
            }
        self._last_vlm_status = "available"
        return {
            **(
                {
                    "description": self._extract_scene_json(description.description).get("description")
                    or self._extract_scene_json(description.description).get("scene_summary")
                    or description.description,
                    "detected_activities": self._coerce_list(
                        self._extract_scene_json(description.description).get("detected_activities")
                        or self._extract_scene_json(description.description).get("actions")
                        or description.detected_activities
                    ),
                    "object_labels": self._coerce_list(
                        self._extract_scene_json(description.description).get("object_labels")
                        or self._extract_scene_json(description.description).get("objects")
                        or description.object_labels
                    ),
                    "step_indicators": self._coerce_list(
                        self._extract_scene_json(description.description).get("step_indicators")
                        or description.step_indicators
                    ),
                    "ppe_status": self._extract_scene_json(description.description).get("ppe_status") or description.ppe_status,
                    "confidence": self._extract_scene_json(description.description).get("confidence", description.confidence),
                }
            )
        }

    def _fuse_ppe(self, vlm_ppe: Dict[str, bool], detections: List[DetectionResult]) -> Dict[str, bool]:
        present_by_alias = {key: False for key in ("gloves", "goggles", "lab_coat")}
        registry = self.settings.class_registry or {}
        thresholds = self.settings.ppe_class_conf_thresholds or {}
        for det in detections:
            name = det.class_name.lower()
            for target, aliases in registry.items():
                if target not in ("glove", "goggles", "lab_coat"):
                    continue
                expected = set(alias.lower() for alias in aliases)
                if name in expected and det.confidence >= float(thresholds.get(target, self.settings.confidence_threshold)):
                    normalized = "gloves" if target == "glove" else target
                    present_by_alias[normalized] = True
        fused: Dict[str, bool] = {}
        for key in present_by_alias:
            fused[key] = bool(vlm_ppe.get(key, False) or present_by_alias[key])
        return fused

    def _is_alert_relevant_detection(self, class_name: str, alert_details: List[Dict[str, object]]) -> bool:
        normalized = class_name.lower()
        if not alert_details:
            return False
        for detail in alert_details:
            related = detail.get("related_classes", [])
            if isinstance(related, list) and normalized in {str(item).lower() for item in related}:
                return True
        return False

    @staticmethod
    def _overlay_text(value: str, limit: int = 84) -> str:
        collapsed = " ".join(str(value or "").split())
        return collapsed[:limit]

    @staticmethod
    def _severity_rank(value: str) -> int:
        ranks = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}
        return ranks.get(str(value).lower(), 0)

    @staticmethod
    def _hex_to_bgr(value: str, default: Tuple[int, int, int] = (0, 0, 255)) -> Tuple[int, int, int]:
        text = str(value or "").strip().lstrip("#")
        if len(text) != 6:
            return default
        try:
            r = int(text[0:2], 16)
            g = int(text[2:4], 16)
            b = int(text[4:6], 16)
        except ValueError:
            return default
        return (b, g, r)

    @staticmethod
    def _hex_to_rgb(value: str, default: Tuple[int, int, int] = (255, 0, 0)) -> Tuple[int, int, int]:
        b, g, r = VideoAnalysisPipeline._hex_to_bgr(value, (default[2], default[1], default[0]))
        return (r, g, b)

    @staticmethod
    def _with_alpha(color: Tuple[int, int, int], alpha: int) -> Tuple[int, int, int, int]:
        return (color[0], color[1], color[2], alpha)

    @staticmethod
    def _collapse_text(value: object) -> str:
        return " ".join(str(value or "").split())

    @classmethod
    @lru_cache(maxsize=1)
    def _resolve_font_path(cls) -> Optional[str]:
        candidates: List[str] = []
        env_value = os.environ.get("LABSOPGUARD_FONT_PATH")
        if env_value:
            candidates.append(env_value)
        candidates.extend(
            [
                "C:/Windows/Fonts/msyh.ttc",
                "C:/Windows/Fonts/msyhbd.ttc",
                "C:/Windows/Fonts/simhei.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf",
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/arphic/ukai.ttc",
            ]
        )
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return candidate
        return None

    @classmethod
    @lru_cache(maxsize=16)
    def _get_font(cls, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_path = cls._resolve_font_path()
        if font_path:
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    @classmethod
    def _supports_cjk(cls) -> bool:
        return cls._resolve_font_path() is not None

    @classmethod
    def _display_text(cls, value: object) -> str:
        text = cls._collapse_text(value)
        if cls._supports_cjk():
            return text
        return text.encode("ascii", errors="replace").decode("ascii")

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
    def _wrap_text(cls, value: object, max_chars: int, max_lines: int) -> List[str]:
        text = cls._display_text(value)
        if not text:
            return []
        lines: List[str] = []
        current = ""
        for char in text:
            candidate = f"{current}{char}"
            if len(candidate) <= max_chars:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = char
            if len(lines) >= max_lines:
                break
        if current and len(lines) < max_lines:
            lines.append(current)
        if len(lines) == max_lines and "".join(lines) != text:
            lines[-1] = lines[-1][: max(1, max_chars - 1)].rstrip() + "..."
        return lines

    @classmethod
    def _measure_text(
        cls,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ) -> Tuple[int, int]:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return (right - left, bottom - top)

    @classmethod
    def _format_ppe_status(cls, ppe_status: Dict[str, bool]) -> str:
        labels = [("手套", "gloves"), ("护目镜", "goggles"), ("实验服", "lab_coat")]
        parts = [f"{label}:{'是' if bool(ppe_status.get(key, False)) else '否'}" for label, key in labels]
        return "PPE " + "  ".join(parts)

    @classmethod
    def _format_alert_titles(cls, analysis: FrameAnalysis) -> str:
        if analysis.alert_details:
            return " | ".join(cls._display_text(detail.get("title", "")) for detail in analysis.alert_details[:3])
        if analysis.alerts:
            return " | ".join(cls._display_text(item) for item in analysis.alerts[:3])
        return "无"

    @classmethod
    def _build_panel_lines(cls, analysis: FrameAnalysis) -> List[str]:
        scene_lines = cls._wrap_text(f"场景: {analysis.scene_description or '无'}", 34, 3)
        message = ""
        if analysis.alert_details:
            message = str(analysis.alert_details[0].get("message", ""))
        message_lines = cls._wrap_text(f"提示: {message}", 34, 2) if message else []
        base = [
            f"时间: {analysis.timestamp_sec:.1f}s",
            f"活动: {' | '.join(cls._display_text(item) for item in analysis.detected_activities[:3]) or '无'}",
            f"步骤: {' | '.join(cls._display_text(item) for item in analysis.step_indicators[:3]) or '无'}",
            cls._format_ppe_status(analysis.ppe_status),
            f"告警: {cls._format_alert_titles(analysis)}",
            f"检测: {len(analysis.detections)} 个目标  VLM:{analysis.vlm_confidence:.2f}",
        ]
        return [*scene_lines, *base, *message_lines]

    @staticmethod
    def _summary_window_sec(total_duration_sec: float) -> float:
        if total_duration_sec <= 0:
            return 0.0
        return min(5.0, max(2.0, total_duration_sec * 0.08))

    @staticmethod
    def _merge_object_labels(base_labels: List[str], detections: List[DetectionResult]) -> List[str]:
        merged: List[str] = []
        seen = set()
        for label in [*base_labels, *(det.class_name for det in detections)]:
            normalized = str(label).strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(str(label))
        return merged

    def _compose_overlay_analysis(
        self,
        semantic_snapshot: Optional[FrameAnalysis],
        frame_idx: int,
        timestamp_sec: float,
        detections: List[DetectionResult],
    ) -> Optional[FrameAnalysis]:
        if semantic_snapshot is None and not detections:
            return None

        base_ppe = semantic_snapshot.ppe_status if semantic_snapshot is not None else {}
        fused_ppe = self._fuse_ppe(base_ppe, detections)
        scene_payload = {
            "description": semantic_snapshot.scene_description if semantic_snapshot is not None else "",
            "detected_activities": list(semantic_snapshot.detected_activities) if semantic_snapshot is not None else [],
            "object_labels": list(semantic_snapshot.object_labels) if semantic_snapshot is not None else [],
            "step_indicators": list(semantic_snapshot.step_indicators) if semantic_snapshot is not None else [],
        }
        alerts = self._build_alerts(fused_ppe, detections, scene_payload)
        alert_details = self._build_alert_details(
            alerts,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            detections=detections,
            rule_basis="overlay frame evaluated with nearest semantic snapshot and live detector output",
        )
        object_labels = self._merge_object_labels(
            semantic_snapshot.object_labels if semantic_snapshot is not None else [],
            detections,
        )

        return FrameAnalysis(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            detections=detections,
            scene_description=semantic_snapshot.scene_description if semantic_snapshot is not None else "",
            detected_activities=list(semantic_snapshot.detected_activities) if semantic_snapshot is not None else [],
            object_labels=object_labels,
            step_indicators=list(semantic_snapshot.step_indicators) if semantic_snapshot is not None else [],
            ppe_status=fused_ppe,
            vlm_confidence=float(semantic_snapshot.vlm_confidence) if semantic_snapshot is not None else 0.0,
            alerts=alerts,
            alert_details=alert_details,
        )

    def create_annotated_video(self, video_path: str, analyses: List[FrameAnalysis], output_path: str) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_duration_sec = total_frames / fps if fps else 0.0
        summary_window_sec = self._summary_window_sec(total_duration_sec)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer, raw_output = self._open_browser_video_writer(output, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open annotated video writer: {output_path}")
        ordered_analyses = sorted(analyses, key=lambda item: item.frame_idx)
        snapshot_index = 0
        current_snapshot = ordered_analyses[0] if ordered_analyses else None
        frame_idx = 0
        smoother = (
            TemporalDetectionSmoother(
                min_hits=self.settings.smoothing_min_hits,
                hold_frames=self.settings.smoothing_hold_frames,
                iou_threshold=self.settings.smoothing_iou_threshold,
            )
            if self.settings.smoothing_enabled
            else None
        )
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            while (
                ordered_analyses
                and snapshot_index + 1 < len(ordered_analyses)
                and ordered_analyses[snapshot_index + 1].frame_idx <= frame_idx
            ):
                snapshot_index += 1
                current_snapshot = ordered_analyses[snapshot_index]

            timestamp_sec = frame_idx / fps if fps else 0.0
            if self.yolo_model is not None:
                raw_detections = self._run_yolo(frame, frame_idx, timestamp_sec)
                live_detections = smoother.update(raw_detections, frame_idx, timestamp_sec) if smoother else raw_detections
            else:
                live_detections = list(current_snapshot.detections) if current_snapshot is not None else []
            overlay_analysis = self._compose_overlay_analysis(
                semantic_snapshot=current_snapshot,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                detections=live_detections,
            )
            if overlay_analysis is not None:
                show_summary = timestamp_sec >= max(0.0, total_duration_sec - summary_window_sec)
                frame = self._annotate_frame(frame, overlay_analysis, show_summary=show_summary)
            writer.write(frame)
            frame_idx += 1
        cap.release()
        writer.release()
        if raw_output != output:
            if not self._transcode_to_browser_mp4(raw_output, output):
                shutil.move(str(raw_output), str(output))
        return output_path

    @staticmethod
    def _transcode_to_browser_mp4(source: Path, target: Path) -> bool:
        """Prefer H.264/yuv420p for browser preview, keep mp4v fallback if unavailable."""
        ffmpeg_exe = shutil.which("ffmpeg")
        if not ffmpeg_exe:
            try:
                import imageio_ffmpeg

                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_exe = None
        if not ffmpeg_exe:
            return False
        temp_target = target.with_name(f"{target.stem}.h264_tmp{target.suffix}")
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
            str(temp_target),
        ]
        try:
            completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=600)
            if completed.returncode != 0 or not temp_target.exists() or temp_target.stat().st_size <= 0:
                if temp_target.exists():
                    temp_target.unlink()
                return False
            shutil.move(str(temp_target), str(target))
            source.unlink(missing_ok=True)
            return True
        except Exception:
            if temp_target.exists():
                temp_target.unlink()
            return False

    @staticmethod
    def _open_browser_video_writer(output: Path, fps: float, size: Tuple[int, int]) -> Tuple[cv2.VideoWriter, Path]:
        """Use H.264 when OpenCV can provide it; otherwise fall back later."""
        for codec in ("avc1", "H264"):
            writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*codec), fps, size)
            if writer.isOpened():
                return writer, output
            writer.release()
        raw_output = output.with_name(f"{output.stem}.raw_mp4v{output.suffix}")
        return cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, size), raw_output

    def _annotate_frame(self, frame: np.ndarray, analysis: FrameAnalysis, show_summary: bool = True) -> np.ndarray:
        base = Image.fromarray(cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)).convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        alert_active = bool(analysis.alerts)
        top_alert = None
        if analysis.alert_details:
            top_alert = max(analysis.alert_details, key=lambda item: self._severity_rank(str(item.get("severity", ""))))
        top_alert_color = self._hex_to_rgb(str(top_alert.get("color", "#E53935"))) if top_alert else (229, 57, 53)

        label_font = self._get_font(18)
        body_font = self._get_font(22)
        title_font = self._get_font(24)

        for det in analysis.detections:
            x1, y1, x2, y2 = det.bbox
            is_alert_target = self._is_alert_relevant_detection(det.class_name, analysis.alert_details)
            color = top_alert_color if is_alert_target else self._class_color(det.class_name)
            width = 4 if is_alert_target else 3
            draw.rounded_rectangle(
                [x1, y1, x2, y2],
                radius=10,
                outline=self._with_alpha(color, 255),
                width=width,
            )
            label = f"{self._display_text(det.class_name)} {det.confidence:.2f}"
            label_width, label_height = self._measure_text(draw, label, label_font)
            label_x1 = x1
            label_y1 = max(10, y1 - label_height - 16)
            label_x2 = min(base.size[0] - 10, label_x1 + label_width + 18)
            label_y2 = label_y1 + label_height + 12
            draw.rounded_rectangle(
                [label_x1, label_y1, label_x2, label_y2],
                radius=10,
                fill=self._with_alpha(color, 235),
            )
            draw.text((label_x1 + 9, label_y1 + 5), label, font=label_font, fill=(255, 255, 255, 255))

        if alert_active:
            draw.rounded_rectangle(
                [8, 8, base.size[0] - 8, base.size[1] - 8],
                radius=18,
                outline=self._with_alpha(top_alert_color, 255),
                width=6,
            )
            if show_summary:
                top_alert_title = self._display_text(top_alert.get("title", "告警")) if top_alert else "告警"
                banner_text = f"{top_alert_title}  {self._format_alert_titles(analysis)}"
                banner_width, banner_height = self._measure_text(draw, banner_text, title_font)
                banner_x2 = min(base.size[0] - 16, 28 + banner_width + 28)
                banner_y2 = 22 + banner_height + 18
                draw.rounded_rectangle(
                    [16, 16, banner_x2, banner_y2],
                    radius=14,
                    fill=self._with_alpha(top_alert_color, 238),
                )
                draw.text((30, 24), banner_text, font=title_font, fill=(255, 255, 255, 255))

        if show_summary:
            panel_lines = self._build_panel_lines(analysis)
            line_height = 30
            panel_width = min(620, base.size[0] - 32)
            panel_height = 30 + len(panel_lines) * line_height
            panel_top = 76 if alert_active else 18
            panel_bottom = min(base.size[1] - 18, panel_top + panel_height)
            draw.rounded_rectangle(
                [16, panel_top, 16 + panel_width, panel_bottom],
                radius=16,
                fill=(16, 18, 24, 196),
                outline=(255, 255, 255, 42),
                width=1,
            )
            y = panel_top + 16
            for line in panel_lines:
                draw.text((30, y), line, font=body_font, fill=(244, 247, 250, 255))
                y += line_height

        merged = Image.alpha_composite(base, overlay).convert("RGB")
        return cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)

    def export_json(self, analyses: List[FrameAnalysis]) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for item in analyses:
            record = asdict(item)
            record["detections"] = [asdict(det) for det in item.detections]
            record["detector_runtime"] = {
                "weights_path": self.yolo_model_path or self.settings.yolo_model_path,
                "strict_model": self.settings.strict_model,
                "imgsz": self.settings.yolo_imgsz,
                "allowed_labels": sorted(self.allowed_detection_labels) if self.allowed_detection_labels else None,
                "smoothing_enabled": self.settings.smoothing_enabled,
            }
            payload.append(record)
        return payload
