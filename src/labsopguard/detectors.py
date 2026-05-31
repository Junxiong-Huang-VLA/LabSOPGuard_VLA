from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class DetectorStatus:
    provider: str
    weights_path: Optional[str]
    weights_exists: bool
    device: str
    available: bool
    error: Optional[str] = None
    imgsz: Optional[int] = None
    allowed_labels: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "weights_path": self.weights_path,
            "weights_exists": self.weights_exists,
            "device": self.device,
            "available": self.available,
            "error": self.error,
            "imgsz": self.imgsz,
            "allowed_labels": self.allowed_labels,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _yaml_model_config(root: Path) -> Tuple[Optional[str], List[str], bool]:
    """Read strict model resolution settings from detection_runtime.yaml."""
    try:
        import yaml  # type: ignore

        cfg_path = root / "configs" / "model" / "detection_runtime.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            return (
                str(cfg["model"]) if cfg.get("model") else None,
                [str(item) for item in (cfg.get("model_fallbacks") or []) if item],
                _as_bool(os.getenv("LABSOPGUARD_STRICT_MODEL"), _as_bool(cfg.get("strict_model"), True)),
            )
    except Exception:
        pass
    return None, [], _as_bool(os.getenv("LABSOPGUARD_STRICT_MODEL"), True)


def _yaml_detection_config(root: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        cfg_path = root / "configs" / "model" / "detection_runtime.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            detection = cfg.get("detection") or {}
            return detection if isinstance(detection, dict) else {}
    except Exception:
        pass
    return {}


def resolve_yolo26_weights_path(override_path: Optional[str] = None, *, allow_fallbacks: Optional[bool] = None) -> Optional[Path]:
    root = _project_root()
    yaml_model, yaml_fallbacks, strict_model = _yaml_model_config(root)
    include_fallbacks = (not strict_model) if allow_fallbacks is None else bool(allow_fallbacks)
    primary_candidates = [
        override_path,
        os.getenv("YOLO26_WEIGHTS_PATH"),
        os.getenv("LABSOPGUARD_YOLO_MODEL"),
        yaml_model,
    ]
    if strict_model and not include_fallbacks:
        for value in primary_candidates:
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = root / path
            return path.resolve() if path.exists() and path.is_file() else None
        return None
    candidates = list(primary_candidates)
    if include_fallbacks:
        candidates.extend(yaml_fallbacks)
        # Central external model store (no weights tracked in the repo).
        models_root = Path(os.environ.get("LAB_MODELS_DIR", r"D:\LabModels"))
        candidates.extend(
            str(models_root / "yolo" / view / "current" / "best.pt")
            for view in ("third_person", "first_person")
        )
    seen: set[str] = set()
    for value in candidates:
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = root / path
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def resolve_detector_device(device: Optional[str] = None) -> str:
    requested = (device or os.getenv("DETECTOR_DEVICE") or os.getenv("YOLO_DEVICE") or "auto").strip()
    normalized = requested.lower()
    if normalized in {"", "auto"}:
        try:
            import torch  # type: ignore

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized.startswith("cuda") or normalized.isdigit():
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
                logger.warning("Detector requested %s but CUDA is unavailable; falling back to CPU.", requested)
                return "cpu"
            if normalized == "cuda":
                return "cuda:0"
            index_text = normalized.split(":", 1)[1] if normalized.startswith("cuda:") else normalized
            index = int(index_text.split(",", 1)[0])
            if index < torch.cuda.device_count():
                return requested
            logger.warning("Detector requested %s but only %s CUDA device(s) are available; falling back to CPU.", requested, torch.cuda.device_count())
            return "cpu"
        except Exception:
            return "cpu"
    return requested


class YOLO26Detector:
    """Thin adapter around ultralytics YOLO with a stable project-level output schema."""

    def __init__(
        self,
        weights_path: str | Path,
        *,
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 50,
        imgsz: Optional[int] = None,
        allowed_labels: Optional[List[str]] = None,
    ) -> None:
        self.weights_path = Path(weights_path).resolve()
        self.device = resolve_detector_device(device)
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.imgsz = int(imgsz) if imgsz else None
        self.allowed_labels = {_normalize_label(item) for item in (allowed_labels or []) if _normalize_label(item)}
        self._model = None
        self._error: Optional[str] = None
        self._load()

    def _load(self) -> None:
        if not self.weights_path.exists():
            self._error = f"YOLO26 weights not found: {self.weights_path}"
            logger.warning(self._error)
            return
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(str(self.weights_path))
            logger.info("YOLO26 detector loaded: weights=%s device=%s", self.weights_path, self.device)
        except Exception as exc:
            self._error = str(exc)
            self._model = None
            logger.exception("Failed to initialize YOLO26 detector from %s", self.weights_path)

    @property
    def status(self) -> DetectorStatus:
        return DetectorStatus(
            provider="ultralytics_yolo26",
            weights_path=str(self.weights_path),
            weights_exists=self.weights_path.exists(),
            device=self.device,
            available=self._model is not None,
            error=self._error,
            imgsz=self.imgsz,
            allowed_labels=sorted(self.allowed_labels) if self.allowed_labels else None,
        )

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        if self._model is None:
            return []
        try:
            predict_kwargs: Dict[str, Any] = {
                "source": frame,
                "conf": self.confidence_threshold,
                "iou": self.iou_threshold,
                "max_det": self.max_detections,
                "device": self.device,
                "verbose": False,
            }
            if self.imgsz:
                predict_kwargs["imgsz"] = self.imgsz
            results = self._model.predict(**predict_kwargs)
        except TypeError:
            results = self._model(frame, verbose=False)

        detections: List[Dict[str, Any]] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            keypoints = getattr(result, "keypoints", None)
            for index, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, cls_id))
                if self.allowed_labels and _normalize_label(label) not in self.allowed_labels:
                    continue
                confidence = float(box.conf[0])
                bbox = [float(v) for v in box.xyxy[0].detach().cpu().numpy().tolist()]
                item: Dict[str, Any] = {
                    "label": label,
                    "object_type": label,
                    "confidence": round(confidence, 6),
                    "score": round(confidence, 6),
                    "bbox": [round(v, 3) for v in bbox],
                    "optional_ocr": None,
                    "optional_attributes": {
                        "class_id": cls_id,
                        "weights_path": str(self.weights_path),
                        "device": self.device,
                    },
                }
                if keypoints is not None and getattr(keypoints, "xy", None) is not None and len(keypoints.xy) > index:
                    points = keypoints.xy[index].detach().cpu().numpy().tolist()
                    item["keypoints"] = [[round(float(x), 3), round(float(y), 3)] for x, y in points if x > 0 and y > 0]
                detections.append(item)
        return detections

    def detect_image_path(self, image_path: str | Path) -> List[Dict[str, Any]]:
        frame = cv2.imread(str(image_path))
        if frame is None:
            try:
                data = np.fromfile(str(image_path), dtype=np.uint8)
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                frame = None
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect_frame(frame)


def build_yolo26_detector(
    *,
    weights_path: Optional[str] = None,
    device: Optional[str] = None,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 50,
    imgsz: Optional[int] = None,
    allowed_labels: Optional[List[str]] = None,
    allow_fallbacks: Optional[bool] = None,
) -> Optional[YOLO26Detector]:
    root = _project_root()
    _, _, strict_model = _yaml_model_config(root)
    resolved = resolve_yolo26_weights_path(weights_path, allow_fallbacks=allow_fallbacks)
    if resolved is None:
        message = "YOLO26 detector disabled: no weights path resolved"
        if strict_model:
            raise FileNotFoundError(message)
        logger.warning(message)
        return None
    detection_cfg = _yaml_detection_config(root)
    if imgsz is None:
        try:
            imgsz = int(detection_cfg.get("imgsz")) if detection_cfg.get("imgsz") else None
        except (TypeError, ValueError):
            imgsz = None
    if allowed_labels is None:
        labels = detection_cfg.get("allowed_labels") or []
        allowed_labels = [str(item) for item in labels if item] if isinstance(labels, list) else None
    detector = YOLO26Detector(
        resolved,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        imgsz=imgsz,
        allowed_labels=allowed_labels,
    )
    if strict_model and not detector.status.available:
        raise RuntimeError(detector.status.error or f"YOLO26 detector unavailable: {resolved}")
    return detector if detector.status.available else detector


def yolo26_diagnostics(weights_path: Optional[str] = None, device: Optional[str] = None, *, allow_fallbacks: Optional[bool] = None) -> Dict[str, Any]:
    root = _project_root()
    _, _, strict_model = _yaml_model_config(root)
    detection_cfg = _yaml_detection_config(root)
    allowed_labels = detection_cfg.get("allowed_labels") or []
    try:
        imgsz = int(detection_cfg.get("imgsz")) if detection_cfg.get("imgsz") else None
    except (TypeError, ValueError):
        imgsz = None
    resolved = resolve_yolo26_weights_path(weights_path, allow_fallbacks=allow_fallbacks)
    selected_device = resolve_detector_device(device)
    if resolved is None:
        payload = DetectorStatus(
            provider="ultralytics_yolo26",
            weights_path=None,
            weights_exists=False,
            device=selected_device,
            available=False,
            error="No YOLO26 weights resolved. Set YOLO26_WEIGHTS_PATH or LABSOPGUARD_YOLO_MODEL.",
            imgsz=imgsz,
            allowed_labels=[str(item) for item in allowed_labels] if isinstance(allowed_labels, list) else None,
        ).to_dict()
        payload["strict_model"] = strict_model
        return payload
    try:
        import ultralytics  # type: ignore

        ultralytics_available = True
        ultralytics_version = getattr(ultralytics, "__version__", None)
    except Exception as exc:
        ultralytics_available = False
        ultralytics_version = None
        payload = DetectorStatus(
            provider="ultralytics_yolo26",
            weights_path=str(resolved),
            weights_exists=resolved.exists(),
            device=selected_device,
            available=False,
            error=f"ultralytics import failed: {exc}",
            imgsz=imgsz,
            allowed_labels=[str(item) for item in allowed_labels] if isinstance(allowed_labels, list) else None,
        ).to_dict()
        payload["strict_model"] = strict_model
        return payload
    payload = DetectorStatus(
        provider="ultralytics_yolo26",
        weights_path=str(resolved),
        weights_exists=resolved.exists(),
        device=selected_device,
        available=bool(ultralytics_available and resolved.exists()),
        imgsz=imgsz,
        allowed_labels=[str(item) for item in allowed_labels] if isinstance(allowed_labels, list) else None,
    ).to_dict()
    payload["ultralytics_version"] = ultralytics_version
    payload["strict_model"] = strict_model
    return payload
