from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from project_name.action.skeleton_sequence_classifier import SkeletonSequenceClassifier
from project_name.video.capture import FramePacket

# Force ultralytics to use a local writable config path, avoiding user-profile permission issues.
_local_cfg_dir = Path.cwd() / ".ultralytics"
_local_cfg_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_local_cfg_dir))

import logging

_log = logging.getLogger(__name__)

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    _log.warning("ultralytics not installed; YOLO backend will be unavailable.")
    YOLO = None


@dataclass
class DetectionEvent:
    frame_id: int
    timestamp_sec: float
    ppe: Dict[str, bool]
    objects: List[Dict[str, Any]]
    actions: List[str]
    confidence: float
    layer_outputs: Dict[str, Any]


def _load_runtime_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


class MultiLevelDetector:
    """Multi-layer detector with real YOLO backend (when available) + stable fallback."""

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        runtime_config_path: str = "configs/model/detection_runtime.yaml",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.runtime_cfg = _load_runtime_config(runtime_config_path)
        self.backend = str(self.runtime_cfg.get("backend", "ultralytics")).lower()
        self.model_name = str(self.runtime_cfg.get("model", "yolov8n.pt"))
        self.device = str(self.runtime_cfg.get("device", "cuda:0"))
        self.layer2_window = int(self.runtime_cfg.get("layer2_window", 30))
        self.strict_model = bool(self.runtime_cfg.get("strict_model", True))
        self.layer2_backend = str(self.runtime_cfg.get("layer2_backend", "skateformer"))
        ppe_cfg = self.runtime_cfg.get("ppe", {}) if isinstance(self.runtime_cfg, dict) else {}
        self.ppe_hold_frames = int(ppe_cfg.get("hold_frames", 20))
        raw_ppe_th = ppe_cfg.get("class_conf_thresholds", {}) if isinstance(ppe_cfg, dict) else {}
        self.ppe_class_thresholds = {
            str(k).strip().lower(): float(v) for k, v in (raw_ppe_th.items() if isinstance(raw_ppe_th, dict) else [])
        }
        self._ppe_last_seen: Dict[str, int] = {"wear_gloves": -10**9, "wear_goggles": -10**9, "wear_lab_coat": -10**9}

        self.class_registry = self.runtime_cfg.get("class_registry", {})
        self.alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.class_registry.items():
            self.alias_to_canonical[self._canonicalize_token(canonical)] = canonical
            for a in aliases:
                self.alias_to_canonical[self._canonicalize_token(str(a))] = canonical

        self._model = None
        if self.backend == "ultralytics" and YOLO is not None:
            self._model = YOLO(self.model_name)
        if self.backend == "ultralytics" and self._model is None and self.strict_model:
            raise RuntimeError(
                "Ultralytics backend is required by strict_model, but YOLO runtime/model is unavailable."
            )
        self._action_classifier = SkeletonSequenceClassifier(
            window_size=self.layer2_window,
            backend=self.layer2_backend,
        )

    def _normalize_label(self, raw_label: str) -> str:
        key = self._canonicalize_token(raw_label)
        return self.alias_to_canonical.get(key, raw_label)

    @staticmethod
    def _canonicalize_token(text: str) -> str:
        return (
            str(text)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _ppe_threshold(self, canonical_label: str) -> float:
        key = self._canonicalize_token(canonical_label)
        if key in self.ppe_class_thresholds:
            return float(self.ppe_class_thresholds[key])
        return max(0.1, float(self.confidence_threshold) * 0.5)

    def _resolve_ppe_flags(self, frame_id: int, objects: List[Dict[str, Any]]) -> Dict[str, bool]:
        seen_now = {
            "wear_gloves": False,
            "wear_goggles": False,
            "wear_lab_coat": False,
        }

        for obj in objects:
            label = self._canonicalize_token(str(obj.get("label", "")))
            score = float(obj.get("score", 0.0))
            if label in {"glove", "gloves"} and score >= self._ppe_threshold("glove"):
                seen_now["wear_gloves"] = True
            if label in {"goggle", "goggles", "safety_glasses", "protective_goggles", "eyewear"} and score >= self._ppe_threshold("goggles"):
                seen_now["wear_goggles"] = True
            if label in {"lab_coat", "labcoat", "white_coat", "coat"} and score >= self._ppe_threshold("lab_coat"):
                seen_now["wear_lab_coat"] = True

        out: Dict[str, bool] = {}
        for key, val in seen_now.items():
            if val:
                self._ppe_last_seen[key] = int(frame_id)
                out[key] = True
                continue
            out[key] = (int(frame_id) - int(self._ppe_last_seen.get(key, -10**9))) <= self.ppe_hold_frames
        return out

    def _detect_with_ultralytics(
        self, frame_bgr: np.ndarray
    ) -> tuple[List[Dict[str, Any]], Optional[List[List[float]]], int]:
        if self._model is None:
            return [], None, 0

        preds = self._model.predict(
            source=frame_bgr,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )
        objects: List[Dict[str, Any]] = []
        keypoints17: Optional[List[List[float]]] = None
        pose_instances = 0
        for result in preds:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                score = float(box.conf.item())
                xyxy = box.xyxy[0].tolist()
                label = self._normalize_label(str(names.get(cls_id, str(cls_id))))
                objects.append(
                    {
                        "label": label,
                        "bbox": [int(v) for v in xyxy],
                        "score": score,
                    }
                )
            if hasattr(result, "keypoints") and result.keypoints is not None:
                try:
                    kp_xy = result.keypoints.xy
                    kp_conf = getattr(result.keypoints, "conf", None)
                    if kp_xy is not None and len(kp_xy) > 0:
                        pose_instances = int(len(kp_xy))
                        for i in range(pose_instances):
                            xy = kp_xy[i].cpu().numpy()  # [K,2]
                            if kp_conf is not None and len(kp_conf) > i:
                                cf = kp_conf[i].cpu().numpy().reshape(-1, 1)
                            else:
                                cf = np.ones((xy.shape[0], 1), dtype=np.float32)
                            kps = np.concatenate([xy, cf], axis=1)
                            if kps.shape[0] >= 17 and keypoints17 is None:
                                # Use the first valid person pose as layer-2 input.
                                keypoints17 = kps[:17].tolist()
                except (RuntimeError, AttributeError, ValueError) as exc:
                    _log.warning("Keypoint extraction failed (frame skipped): %s", exc)
                    keypoints17 = None
        return objects, keypoints17, pose_instances

    def _detect_fallback(self, frame: FramePacket) -> List[Dict[str, Any]]:
        # Deterministic fallback for environments without realtime model runtime.
        dx = int((frame.frame_id % 20) - 10)
        dy = int(((frame.frame_id // 2) % 12) - 6)
        return [
            {
                "label": "sample_container",
                "bbox": [120 + dx, 90 + dy, 250 + dx, 260 + dy],
                "score": 0.78,
            },
            {
                "label": "pipette",
                "bbox": [270 - dx, 120 + dy, 380 - dx, 220 + dy],
                "score": 0.66,
            },
        ]

    def detect(self, frame: FramePacket) -> DetectionEvent:
        objects: List[Dict[str, Any]] = []
        keypoints17: Optional[List[List[float]]] = None
        pose_instances = 0
        backend_used = "fallback_heuristic"
        if self.backend == "ultralytics" and self._model is not None:
            objects, keypoints17, pose_instances = self._detect_with_ultralytics(
                frame.frame_bgr
            )
            backend_used = "ultralytics"
        elif self.strict_model:
            raise RuntimeError("strict_model=true but ultralytics backend is not available.")
        else:
            objects = self._detect_fallback(frame)

        ppe_flags = self._resolve_ppe_flags(frame.frame_id, objects)
        labels = {self._canonicalize_token(str(o.get("label", ""))) for o in objects}

        layer2_res = self._action_classifier.update(keypoints17)
        actions = ["verify_label"]
        if layer2_res.action not in {"unknown", "warmup", "verify_label"}:
            actions.append(layer2_res.action)
        elif "pipette" in labels or "sample_container" in labels:
            actions.append("pipette_transfer")
        confidence = float(np.mean([o.get("score", 0.0) for o in objects])) if objects else 0.3

        layer_outputs = {
            "layer1_realtime_pose": {
                "model": self.model_name if backend_used == "ultralytics" else "bootstrap_heuristic",
                "backend": backend_used,
                "objects_count": len(objects),
                "pose_keypoints_17": keypoints17,
                "pose_instances": pose_instances,
            },
            "layer2_action_analysis": {
                "model": f"{self.layer2_backend}/ST-GCN++",
                "window_size": self.layer2_window,
                "actions": actions,
                "action_confidence": layer2_res.confidence,
                "backend": layer2_res.backend,
            },
            "layer3_vlm_semantic": {
                "model": "Qwen3-VL-8B(interface)",
                "enabled": bool(self.runtime_cfg.get("enable_vlm", False)),
                "scene_tag": "lab_operation_normal" if confidence >= 0.5 else "lab_operation_uncertain",
            },
            "layer4_step_anomaly": {
                "framework": "PREGO(interface)",
                "step_state": {
                    "verify_label": "done" if "verify_label" in actions else "pending",
                    "pipette_transfer": "done" if "pipette_transfer" in actions else "pending",
                },
            },
        }

        return DetectionEvent(
            frame_id=frame.frame_id,
            timestamp_sec=frame.timestamp_sec,
            ppe={
                "wear_gloves": bool(ppe_flags.get("wear_gloves", False)),
                "wear_goggles": bool(ppe_flags.get("wear_goggles", False)),
                "wear_lab_coat": bool(ppe_flags.get("wear_lab_coat", False)),
            },
            objects=[o for o in objects if float(o.get("score", 0.0)) >= self.confidence_threshold],
            actions=actions,
            confidence=max(0.1, min(1.0, confidence)),
            layer_outputs=layer_outputs,
        )
