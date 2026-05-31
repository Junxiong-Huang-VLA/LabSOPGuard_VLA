from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def _default_depth_path(rgb_video: Path) -> Path:
    if rgb_video.stem.lower().endswith("_rgb"):
        return rgb_video.with_name(rgb_video.stem[:-4] + "_depth" + rgb_video.suffix)
    return rgb_video.with_name(rgb_video.stem + "_depth" + rgb_video.suffix)


def _build_depth_pipeline(model_id: str, prefer_cuda: bool):
    try:
        import torch
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError(
            "Missing dependencies. Please install torch and transformers first."
        ) from exc

    use_cuda = prefer_cuda and torch.cuda.is_available()
    device = 0 if use_cuda else -1
    return pipeline(task="depth-estimation", model=model_id, device=device), torch


def _to_depth_array(pred: object, frame_hw: tuple[int, int], torch_mod) -> np.ndarray:
    h, w = frame_hw
    if isinstance(pred, dict):
        if "predicted_depth" in pred:
            d = pred["predicted_depth"]
            if hasattr(torch_mod, "Tensor") and isinstance(d, torch_mod.Tensor):
                arr = d.detach().cpu().numpy()
            else:
                arr = np.asarray(d)
        elif "depth" in pred:
            arr = np.asarray(pred["depth"])
        else:
            raise ValueError("Unexpected depth pipeline output keys.")
    else:
        arr = np.asarray(pred)

    if arr.ndim == 3:
        arr = arr.squeeze()
    if arr.shape != (h, w):
        arr = cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return arr.astype(np.float32)


def _depth_to_u8(depth: np.ndarray) -> np.ndarray:
    valid = depth[np.isfinite(depth)]
    if valid.size == 0:
        return np.zeros_like(depth, dtype=np.uint8)
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if hi <= lo:
        return np.zeros_like(depth, dtype=np.uint8)
    clipped = np.clip(depth, lo, hi)
    norm = (clipped - lo) / (hi - lo + 1e-6)
    return (norm * 255.0).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pseudo depth video from an RGB video using a depth model."
    )
    parser.add_argument("--rgb-video", required=True, help="Input RGB video path.")
    parser.add_argument(
        "--depth-video",
        default=None,
        help="Output depth video path. Default: replace *_rgb with *_depth.",
    )
    parser.add_argument(
        "--model-id",
        default="Intel/dpt-hybrid-midas",
        help="HuggingFace model id for depth estimation.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output fps. 0 means reuse source fps.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process. 0 means all frames.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rgb_video = Path(args.rgb_video)
    if not rgb_video.exists():
        raise FileNotFoundError(f"RGB video not found: {rgb_video}")

    depth_video = Path(args.depth_video) if args.depth_video else _default_depth_path(rgb_video)
    depth_video.parent.mkdir(parents=True, exist_ok=True)

    depth_pipe, torch_mod = _build_depth_pipeline(args.model_id, prefer_cuda=not args.cpu)

    cap = cv2.VideoCapture(str(rgb_video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {rgb_video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = args.fps if args.fps > 0 else (src_fps if src_fps > 0 else 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(depth_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
        True,
    )

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pred = depth_pipe(pil_img)
        depth = _to_depth_array(pred, (height, width), torch_mod)
        depth_u8 = _depth_to_u8(depth)
        depth_vis = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
        writer.write(depth_vis)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"processed frames: {frame_idx}")

    cap.release()
    writer.release()
    print(f"depth video saved: {depth_video}")
    print(f"total frames: {frame_idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
