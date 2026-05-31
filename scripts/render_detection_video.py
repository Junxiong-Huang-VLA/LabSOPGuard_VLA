from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render detection video with one model.")
    parser.add_argument("--video", required=True, help="Input RGB video path.")
    parser.add_argument("--weights", required=True, help="Model weights path, e.g. best.pt")
    parser.add_argument("--out-video", default="outputs/predictions/fine_tuned_detect_60s.mp4")
    parser.add_argument("--out-json", default="outputs/predictions/fine_tuned_detect_60s.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    local_cfg = Path(".ultralytics")
    local_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cfg.resolve()))

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required for rendering") from exc

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    step = max(1, int(round(src_fps / args.target_fps)))
    max_frames = int(round(args.duration_sec * args.target_fps))

    out_path = Path(args.out_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.out_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    read_id = 0
    out_count = 0
    class_counter: Counter[str] = Counter()
    avg_conf_sum = 0.0
    avg_conf_n = 0

    while out_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read_id % step != 0:
            read_id += 1
            continue

        result = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        annotated = result.plot(conf=True, labels=True, boxes=True)

        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                class_counter[str(names.get(cls_id, cls_id))] += 1
                avg_conf_sum += conf
                avg_conf_n += 1

        if writer is None:
            h, w = annotated.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.target_fps),
                (w, h),
                True,
            )

        writer.write(annotated)
        out_count += 1
        read_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    avg_conf = (avg_conf_sum / avg_conf_n) if avg_conf_n > 0 else 0.0
    summary: Dict[str, Any] = {
        "video": str(video_path),
        "weights": str(weights_path),
        "out_video": str(out_path),
        "frames_rendered": out_count,
        "target_fps": args.target_fps,
        "duration_sec": args.duration_sec,
        "conf": args.conf,
        "avg_conf": round(avg_conf, 6),
        "class_counter": dict(class_counter),
    }
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"video saved: {out_path}")
    print(f"meta saved: {meta_path}")
    print(f"frames rendered: {out_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
