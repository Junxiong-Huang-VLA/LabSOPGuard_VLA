from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict

import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuntimeSettings:
    project_root: Path
    yolo_model_path: str | None = None
    device: str = "cpu"
    strict_model: bool = True
    yolo_imgsz: int = 960
    sample_interval_sec: float = 2.0
    min_frames: int = 8
    max_frames: int = 36
    max_vlm_frames: int = 18
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 50
    allowed_detection_labels: list[str] = field(default_factory=list)
    smoothing_enabled: bool = True
    smoothing_min_hits: int = 3
    smoothing_hold_frames: int = 5
    smoothing_iou_threshold: float = 0.35
    ppe_consensus_ratio: float = 0.45
    ppe_hold_frames: int = 2
    class_registry: Dict[str, list[str]] = field(default_factory=dict)
    ppe_class_conf_thresholds: Dict[str, float] = field(default_factory=dict)
    alert_rules: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    detection_cache_enabled: bool = True
    batch_size: int = 8

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def uploads_dir(self) -> Path:
        return self.project_root / "uploads"


def _as_float(data: Dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(data: Dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(data: Dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_runtime_device(device: Any) -> str:
    requested = str(device or "auto").strip()
    normalized = requested.lower()
    if normalized in {"", "auto"}:
        try:
            import torch  # type: ignore

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception as exc:
            logger.warning("Auto device detection failed, falling back to CPU: %s", exc)
            return "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda") or normalized.isdigit():
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
                return "cpu"
            if normalized == "cuda":
                return "cuda:0"
            index_text = normalized.split(":", 1)[1] if normalized.startswith("cuda:") else normalized
            index = int(index_text.split(",", 1)[0])
            return requested if index < torch.cuda.device_count() else "cpu"
        except Exception as exc:
            logger.warning("CUDA device validation failed for %r, falling back to CPU: %s", requested, exc)
            return "cpu"
    return requested


def load_runtime_settings(project_root: Path) -> RuntimeSettings:
    config_path = project_root / "configs" / "model" / "detection_runtime.yaml"
    alert_config_path = project_root / "configs" / "alerts" / "alerting.yaml"
    payload: Dict[str, Any] = {}
    if config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    alert_payload: Dict[str, Any] = {}
    if alert_config_path.exists():
        alert_payload = yaml.safe_load(alert_config_path.read_text(encoding="utf-8")) or {}

    sampling = payload.get("sampling", {})
    detection = payload.get("detection", {})
    ppe = payload.get("ppe", {})
    smoothing = payload.get("smoothing", {})
    cache = payload.get("cache", {})
    alert_rules = alert_payload.get("video_analysis_rules", {}) or {}

    yolo_model = os.getenv("YOLO26_WEIGHTS_PATH") or os.getenv("LABSOPGUARD_YOLO_MODEL") or payload.get("model")
    device = resolve_runtime_device(os.getenv("DETECTOR_DEVICE") or os.getenv("YOLO_DEVICE") or payload.get("device", "auto"))
    strict_env = os.getenv("LABSOPGUARD_STRICT_MODEL")
    strict_model = _as_bool({"strict_model": strict_env}, "strict_model", _as_bool(payload, "strict_model", True)) if strict_env is not None else _as_bool(payload, "strict_model", True)
    yolo_imgsz = _as_int({"imgsz": os.getenv("LABSOPGUARD_YOLO_IMGSZ")}, "imgsz", _as_int(detection, "imgsz", 960))
    candidates = [yolo_model]
    if not strict_model:
        candidates.extend(payload.get("model_fallbacks") or [])
    yolo_model = None
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if not path.is_absolute():
            path = project_root / path
        if path.exists() and path.is_file():
            yolo_model = str(path.resolve())
            break

    return RuntimeSettings(
        project_root=project_root,
        yolo_model_path=yolo_model,
        device=device,
        strict_model=strict_model,
        yolo_imgsz=yolo_imgsz,
        sample_interval_sec=_as_float(sampling, "base_interval_sec", 2.0),
        min_frames=_as_int(sampling, "min_frames", 8),
        max_frames=_as_int(sampling, "max_frames", 36),
        max_vlm_frames=_as_int(sampling, "max_vlm_frames", 18),
        confidence_threshold=_as_float(detection, "confidence_threshold", 0.25),
        iou_threshold=_as_float(detection, "iou_threshold", 0.45),
        max_detections=_as_int(detection, "max_detections", 50),
        allowed_detection_labels=[str(item) for item in detection.get("allowed_labels", []) or []],
        smoothing_enabled=_as_bool(smoothing, "enabled", True),
        smoothing_min_hits=_as_int(smoothing, "min_hits", 3),
        smoothing_hold_frames=_as_int(smoothing, "hold_frames", 5),
        smoothing_iou_threshold=_as_float(smoothing, "iou_threshold", 0.35),
        ppe_consensus_ratio=_as_float(ppe, "consensus_ratio", 0.45),
        ppe_hold_frames=_as_int(ppe, "hold_frames", 2),
        class_registry=payload.get("class_registry", {}) or {},
        ppe_class_conf_thresholds=ppe.get("class_conf_thresholds", {}) or {},
        alert_rules=alert_rules,
        detection_cache_enabled=_as_bool(cache, "detection_cache_enabled", True),
        batch_size=_as_int(cache, "batch_size", 8),
    )
