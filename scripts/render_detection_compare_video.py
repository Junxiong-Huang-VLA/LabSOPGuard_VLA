from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render before/after detection comparison video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model-before", default="yolo26s-pose.pt")
    parser.add_argument("--model-after", required=True, help="Path to fine-tuned weights, e.g. best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--out-video", default="outputs/predictions/detection_compare_60s.mp4")
    return parser.parse_args()


def _draw_boxes(frame, objs: List[Dict[str, Any]], color) -> Any:
    out = frame.copy()
    for o in objs:
        x1, y1, x2, y2 = [int(v) for v in o["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            f"{o['label']}:{o['score']:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _predict(model, frame_bgr, conf: float) -> List[Dict[str, Any]]:
    preds = model.predict(source=frame_bgr, conf=conf, verbose=False)
    objs: List[Dict[str, Any]] = []
    for r in preds:
        names = r.names
        for b in r.boxes:
            cls_id = int(b.cls.item())
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            score = float(b.conf.item())
            objs.append({"label": str(names.get(cls_id, cls_id)), "bbox": xyxy, "score": score})
    return objs


def main() -> int:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    local_cfg = Path(".ultralytics")
    local_cfg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(local_cfg.resolve()))

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        raise RuntimeError("ultralytics is required") from exc

    model_before = YOLO(args.model_before)
    model_after = YOLO(args.model_after)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_fps = src_fps if src_fps and src_fps > 0 else 30.0
    step = max(1, int(round(src_fps / args.target_fps)))
    max_frames = int(round(args.duration_sec * args.target_fps))

    writer = None
    read_id = 0
    out_count = 0
    while out_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if read_id % step != 0:
            read_id += 1
            continue

        objs_before = _predict(model_before, frame, args.conf)
        objs_after = _predict(model_after, frame, args.conf)
        left = _draw_boxes(frame, objs_before, (0, 200, 255))
        right = _draw_boxes(frame, objs_after, (0, 255, 0))
        cv2.putText(left, "Before (baseline)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.putText(right, "After (fine-tuned)", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        comp = cv2.hconcat([left, right])

        if writer is None:
            out_path = Path(args.out_video)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = comp.shape[:2]
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(args.target_fps), (w, h), True)
        writer.write(comp)

        out_count += 1
        read_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    print(f"compare video: {args.out_video}")
    print(f"frames rendered: {out_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
