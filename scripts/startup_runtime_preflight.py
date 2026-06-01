from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.config import load_runtime_settings  # noqa: E402
from labsopguard.detectors import yolo26_diagnostics  # noqa: E402


def _is_generic_fallback(path: str | None) -> bool:
    if not path:
        return False
    name = Path(path).name.lower()
    return name in {"yolo26s.pt", "yolo26n.pt", "yolo26s-pose.pt", "yolov8n-pose.pt", "yolov8n.pt"}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one-click LabEmbodied runtime before services start.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "run_logs" / "startup_runtime.json")
    parser.add_argument("--load-model", action="store_true", help="Instantiate VideoAnalysisPipeline to catch corrupt weights early.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    settings = load_runtime_settings(project_root)
    detector_status = yolo26_diagnostics(settings.yolo_model_path)
    errors: List[str] = []
    warnings: List[str] = []

    if not settings.strict_model:
        errors.append("strict_model must be true; production startup must not silently fallback to generic weights.")
    if not settings.yolo_model_path:
        errors.append("No YOLO26 model path resolved from configs/model/detection_runtime.yaml or environment.")
    elif not Path(settings.yolo_model_path).exists():
        errors.append(f"YOLO26 model path does not exist: {settings.yolo_model_path}")
    if _is_generic_fallback(settings.yolo_model_path):
        errors.append(f"Generic fallback weights are not allowed in one-click startup: {settings.yolo_model_path}")
    if int(settings.yolo_imgsz) < 960:
        warnings.append(f"yolo_imgsz is {settings.yolo_imgsz}; 960+ is recommended for tube-cap/pipette-tip small objects.")
    if not settings.allowed_detection_labels:
        errors.append("detection.allowed_labels is empty; label whitelist is required to block COCO fallback labels.")
    if not settings.smoothing_enabled:
        warnings.append("Temporal smoothing is disabled; annotated videos may flicker.")
    if not detector_status.get("available"):
        errors.append(f"YOLO26 detector unavailable: {detector_status.get('error') or 'unknown error'}")

    loaded_model = False
    if args.load_model and not errors:
        try:
            from labsopguard.video_analysis import VideoAnalysisPipeline

            pipeline = VideoAnalysisPipeline(settings=settings)
            loaded_model = bool(pipeline.yolo_model is not None)
            if not loaded_model:
                errors.append("VideoAnalysisPipeline did not load a YOLO26 model.")
        except Exception as exc:
            errors.append(f"VideoAnalysisPipeline load failed: {type(exc).__name__}: {exc}")

    payload: Dict[str, Any] = {
        "schema_version": "startup_runtime.v1",
        "project_root": str(project_root),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "runtime": {
            "strict_model": settings.strict_model,
            "yolo_model_path": settings.yolo_model_path,
            "yolo_imgsz": settings.yolo_imgsz,
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
            "max_detections": settings.max_detections,
            "allowed_detection_labels": settings.allowed_detection_labels,
            "smoothing_enabled": settings.smoothing_enabled,
            "smoothing_min_hits": settings.smoothing_min_hits,
            "smoothing_hold_frames": settings.smoothing_hold_frames,
            "material_conf": "env LABSOPGUARD_MATERIAL_YOLO_CONF or default 0.10",
        },
        "detector_status": detector_status,
        "model_loaded": loaded_model,
    }
    _write_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
