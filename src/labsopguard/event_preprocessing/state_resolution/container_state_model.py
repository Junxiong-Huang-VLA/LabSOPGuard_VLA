from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .base import ObjectStateResolution

STATE_MODEL_VERSION = "container_state_prototype.v1"


def extract_keyframe_features(paths: Optional[List[Path]]) -> Dict[str, float]:
    images = []
    for path in paths or []:
        if not path or not Path(path).exists():
            continue
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
    if not images:
        return {"brightness": 0.0, "edge_density": 0.0, "blue_ratio": 0.0, "sample_count": 0.0}
    brightness = []
    edge_density = []
    blue_ratio = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        brightness.append(float(gray.mean() / 255.0))
        edge_density.append(float((edges > 0).mean()))
        blue = image[:, :, 0].astype(np.float32)
        red_green = image[:, :, 1:].mean(axis=2).astype(np.float32) + 1e-6
        blue_ratio.append(float(np.clip((blue / red_green).mean() / 2.0, 0.0, 1.0)))
    return {
        "brightness": round(sum(brightness) / len(brightness), 5),
        "edge_density": round(sum(edge_density) / len(edge_density), 5),
        "blue_ratio": round(sum(blue_ratio) / len(blue_ratio), 5),
        "sample_count": float(len(images)),
    }


class PrototypeContainerStateModel:
    """Small JSON-backed state classifier for container open/closed candidates.

    The model file stores class prototypes over simple visual features. This keeps
    the production interface stable while allowing later replacement with a CNN/VLM
    model without changing the ObjectStateResolver contract.
    """

    def __init__(self, prototypes: Dict[str, Dict[str, float]], threshold: float = 0.18) -> None:
        self.prototypes = prototypes
        self.threshold = float(threshold)

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeContainerStateModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            prototypes={str(key): {k: float(v) for k, v in value.items()} for key, value in (payload.get("prototypes") or {}).items()},
            threshold=float(payload.get("threshold") or 0.18),
        )

    def predict(self, *, keyframe_paths: Optional[List[Path]] = None, semantic_summary: Optional[Dict[str, Any]] = None) -> ObjectStateResolution:
        text = " ".join(str(value) for value in (semantic_summary or {}).values()).lower()
        if any(token in text for token in ["open", "opened", "lid open", "cap removed", "打开", "开盖"]):
            return ObjectStateResolution("closed_or_lidded", "open_candidate", "container_open_candidate", 0.82, ["semantic_open_hint", "prototype_state_model"], "prototype_container_state_model")
        if any(token in text for token in ["closed", "lid closed", "cap placed", "关闭", "盖上"]):
            return ObjectStateResolution("open_or_unlidded", "closed_candidate", "container_close_candidate", 0.82, ["semantic_close_hint", "prototype_state_model"], "prototype_container_state_model")
        features = extract_keyframe_features(keyframe_paths)
        if not self.prototypes or features.get("sample_count", 0.0) <= 0:
            return ObjectStateResolution("unknown", "unknown", "state_change_candidate", 0.2, ["missing_state_model_features"], "prototype_container_state_model")
        label, distance = self._nearest(features)
        confidence = max(0.25, min(0.9, 1.0 - distance / max(self.threshold * 3.0, 1e-6)))
        if distance > self.threshold:
            return ObjectStateResolution("unknown", f"{label}_candidate", "container_state_candidate", round(confidence * 0.7, 4), ["prototype_distance_above_threshold", f"nearest_state={label}"], "prototype_container_state_model")
        if label in {"open", "open_candidate"}:
            return ObjectStateResolution("closed_or_lidded", "open_candidate", "container_open_candidate", round(confidence, 4), ["prototype_open_visual_match"], "prototype_container_state_model")
        if label in {"closed", "closed_candidate", "lidded"}:
            return ObjectStateResolution("open_or_unlidded", "closed_candidate", "container_close_candidate", round(confidence, 4), ["prototype_closed_visual_match"], "prototype_container_state_model")
        return ObjectStateResolution("unknown", label, "container_state_candidate", round(confidence, 4), [f"prototype_match={label}"], "prototype_container_state_model")

    def _nearest(self, features: Dict[str, float]) -> tuple[str, float]:
        best_label = "unknown"
        best_distance = float("inf")
        for label, proto in self.prototypes.items():
            keys = set(proto) & set(features)
            if not keys:
                continue
            distance = math.sqrt(sum((float(features[key]) - float(proto[key])) ** 2 for key in keys) / len(keys))
            if distance < best_distance:
                best_label = label
                best_distance = distance
        return best_label, best_distance


def write_default_container_state_model(path: str | Path) -> None:
    payload = {
        "schema_version": STATE_MODEL_VERSION,
        "threshold": 0.22,
        "prototypes": {
            "open": {"brightness": 0.55, "edge_density": 0.11, "blue_ratio": 0.15},
            "closed": {"brightness": 0.42, "edge_density": 0.05, "blue_ratio": 0.12},
        },
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
