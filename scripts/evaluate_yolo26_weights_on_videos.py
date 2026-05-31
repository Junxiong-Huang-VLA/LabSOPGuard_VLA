from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.config import load_runtime_settings  # noqa: E402
from labsopguard.detectors import resolve_detector_device  # noqa: E402


def _normalize_label(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _expand_paths(values: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        matches = glob.glob(value)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(value))
    return sorted({path.resolve() for path in paths})


def _parse_weight(value: str) -> Tuple[str, Path]:
    if "=" in value:
        name, raw_path = value.split("=", 1)
        return name.strip() or Path(raw_path).stem, Path(raw_path)
    path = Path(value)
    parent = path.parent.parent.name if path.parent.name == "weights" else path.parent.name
    name = parent or path.stem
    return name, path


def _iter_video_frames(video_path: Path, sample_interval_sec: float, max_frames: int) -> Iterator[Tuple[int, float, object]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * sample_interval_sec))) if sample_interval_sec > 0 else 1
    emitted = 0
    frame_idx = 0
    while total <= 0 or frame_idx < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        yield frame_idx, frame_idx / fps if fps else 0.0, frame
        emitted += 1
        if max_frames > 0 and emitted >= max_frames:
            break
        frame_idx += step
    cap.release()


def _summarize_video(model: object, video_path: Path, args: argparse.Namespace, allowed_labels: set[str]) -> Dict[str, object]:
    frames = 0
    hit_frames = 0
    total_detections = 0
    confidences: List[float] = []
    labels: Counter[str] = Counter()
    current_miss = 0
    max_miss = 0
    for frame_idx, timestamp_sec, frame in _iter_video_frames(video_path, args.sample_interval_sec, args.max_frames_per_video):
        frames += 1
        predict_kwargs = {
            "source": frame,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "device": args.device,
            "verbose": False,
        }
        if args.imgsz:
            predict_kwargs["imgsz"] = args.imgsz
        results = model.predict(**predict_kwargs)
        frame_detections = 0
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0])
                label = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
                if allowed_labels and _normalize_label(label) not in allowed_labels:
                    continue
                confidence = float(box.conf[0])
                labels[label] += 1
                confidences.append(confidence)
                frame_detections += 1
        if frame_detections > 0:
            hit_frames += 1
            current_miss = 0
        else:
            current_miss += 1
            max_miss = max(max_miss, current_miss)
        total_detections += frame_detections
    avg_confidence = statistics.fmean(confidences) if confidences else 0.0
    return {
        "video_path": str(video_path),
        "frames_sampled": frames,
        "frames_with_detections": hit_frames,
        "detection_coverage": round(hit_frames / frames, 6) if frames else 0.0,
        "total_detections": total_detections,
        "detections_per_frame": round(total_detections / frames, 6) if frames else 0.0,
        "avg_confidence": round(avg_confidence, 6),
        "max_consecutive_miss_frames": max_miss,
        "labels": dict(labels.most_common()),
    }


def _summarize_weight(weight_name: str, weight_path: Path, videos: List[Path], args: argparse.Namespace, allowed_labels: set[str]) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "name": weight_name,
        "weights_path": str(weight_path),
        "weights_exists": weight_path.exists(),
        "videos": [],
        "error": None,
    }
    if not weight_path.exists():
        payload["error"] = f"weights not found: {weight_path}"
        return payload
    try:
        from ultralytics import YOLO  # type: ignore

        model = YOLO(str(weight_path))
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload

    video_results = []
    for video_path in videos:
        try:
            video_results.append(_summarize_video(model, video_path, args, allowed_labels))
        except Exception as exc:
            video_results.append({"video_path": str(video_path), "error": f"{type(exc).__name__}: {exc}"})
    payload["videos"] = video_results
    valid = [item for item in video_results if not item.get("error")]
    frames = sum(int(item["frames_sampled"]) for item in valid)
    hit_frames = sum(int(item["frames_with_detections"]) for item in valid)
    detections = sum(int(item["total_detections"]) for item in valid)
    confidences = [float(item["avg_confidence"]) for item in valid if float(item.get("avg_confidence", 0.0)) > 0]
    max_miss = max([int(item["max_consecutive_miss_frames"]) for item in valid] or [0])
    label_counter: Counter[str] = Counter()
    for item in valid:
        label_counter.update(item.get("labels", {}))
    coverage = hit_frames / frames if frames else 0.0
    avg_confidence = statistics.fmean(confidences) if confidences else 0.0
    payload["summary"] = {
        "videos_evaluated": len(valid),
        "frames_sampled": frames,
        "frames_with_detections": hit_frames,
        "detection_coverage": round(coverage, 6),
        "total_detections": detections,
        "detections_per_frame": round(detections / frames, 6) if frames else 0.0,
        "avg_confidence": round(avg_confidence, 6),
        "max_consecutive_miss_frames": max_miss,
        "labels": dict(label_counter.most_common()),
    }
    return payload


def _write_csv(path: Path, results: List[Dict[str, object]]) -> None:
    rows = []
    for item in results:
        summary = item.get("summary") or {}
        rows.append(
            {
                "name": item.get("name"),
                "weights_path": item.get("weights_path"),
                "weights_exists": item.get("weights_exists"),
                "error": item.get("error"),
                "videos_evaluated": summary.get("videos_evaluated"),
                "frames_sampled": summary.get("frames_sampled"),
                "detection_coverage": summary.get("detection_coverage"),
                "detections_per_frame": summary.get("detections_per_frame"),
                "avg_confidence": summary.get("avg_confidence"),
                "max_consecutive_miss_frames": summary.get("max_consecutive_miss_frames"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    settings = load_runtime_settings(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description="A/B evaluate YOLO26 weights on the same real lab videos.")
    parser.add_argument("--videos", nargs="+", required=True, help="Video paths or glob patterns.")
    parser.add_argument(
        "--weights",
        nargs="+",
        default=[
            "v4_focus_auto=outputs/training/yolo26s_pose_lab_v4_focus_auto/weights/best.pt",
            "autodl_8_1_1=outputs/training/yolo26s_autodl_8_1_1/weights/best.pt",
            "allphotos_e40=outputs/training/yolo26s_allphotos_e40/weights/best.pt",
        ],
        help="Weights as path or name=path.",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "yolo26_ab_eval.json")
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--max-frames-per-video", type=int, default=240)
    parser.add_argument("--conf", type=float, default=float(settings.confidence_threshold))
    parser.add_argument("--iou", type=float, default=float(settings.iou_threshold))
    parser.add_argument("--max-det", type=int, default=int(settings.max_detections))
    parser.add_argument("--imgsz", type=int, default=int(settings.yolo_imgsz))
    parser.add_argument("--device", default=resolve_detector_device(settings.device))
    parser.add_argument("--allowed-labels", default=",".join(settings.allowed_detection_labels))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = _expand_paths(args.videos)
    missing_videos = [str(path) for path in videos if not path.exists()]
    if missing_videos:
        raise FileNotFoundError(f"Video path(s) not found: {missing_videos}")
    allowed_labels = {_normalize_label(item) for item in args.allowed_labels.split(",") if _normalize_label(item)}
    weights = [_parse_weight(item) for item in args.weights]
    results = [_summarize_weight(name, path if path.is_absolute() else (PROJECT_ROOT / path).resolve(), videos, args, allowed_labels) for name, path in weights]
    ranked = sorted(
        results,
        key=lambda item: (
            (item.get("summary") or {}).get("detection_coverage", 0.0),
            (item.get("summary") or {}).get("avg_confidence", 0.0),
            -int((item.get("summary") or {}).get("max_consecutive_miss_frames", 10**9)),
        ),
        reverse=True,
    )
    payload = {
        "videos": [str(path) for path in videos],
        "settings": {
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "imgsz": args.imgsz,
            "device": args.device,
            "allowed_labels": sorted(allowed_labels),
            "sample_interval_sec": args.sample_interval_sec,
            "max_frames_per_video": args.max_frames_per_video,
        },
        "ranking": [item["name"] for item in ranked],
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(args.out.with_suffix(".csv"), results)
    print(json.dumps({"out": str(args.out), "ranking": payload["ranking"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
