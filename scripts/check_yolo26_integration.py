#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.detectors import build_yolo26_detector, yolo26_diagnostics


def _first_dataset_image() -> Path | None:
    for folder in [
        PROJECT_ROOT / "data" / "dataset" / "images" / "train",
        PROJECT_ROOT / "data" / "dataset" / "images" / "val",
        PROJECT_ROOT / "data" / "dataset" / "images" / "test",
    ]:
        if folder.exists():
            for image in folder.glob("*.jpg"):
                return image
    return None


def _frame_from_video(video_path: Path, output_path: Path) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read first frame from video: {video_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real YOLO26 single-frame inference probe.")
    parser.add_argument("--weights", default=None, help="YOLO26 weights path. Defaults to env/config resolution.")
    parser.add_argument("--image", default=None, help="Image path for single-frame inference.")
    parser.add_argument("--video", default=None, help="Video path; first frame is extracted and inferred.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "checks" / "yolo26_integration.json"))
    args = parser.parse_args()

    image_path: Path | None = Path(args.image) if args.image else None
    if args.video:
        image_path = _frame_from_video(Path(args.video), PROJECT_ROOT / "outputs" / "checks" / "yolo26_probe_frame.jpg")
    if image_path is None:
        image_path = _first_dataset_image()
    if image_path is None or not image_path.exists():
        raise FileNotFoundError("No image/video probe input found. Pass --image or --video.")

    detector = build_yolo26_detector(weights_path=args.weights)
    diagnostics = yolo26_diagnostics(args.weights)
    detections = detector.detect_image_path(image_path) if detector and detector.status.available else []
    payload = {
        "diagnostics": diagnostics,
        "probe_image": str(image_path),
        "detection_count": len(detections),
        "detections": detections,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if diagnostics.get("available") and len(detections) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
