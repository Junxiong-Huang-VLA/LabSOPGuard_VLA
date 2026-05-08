from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def get_video_duration_sec(video_path: str | Path) -> float:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required to inspect video duration") from exc

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        if fps <= 0:
            raise RuntimeError(f"Cannot determine FPS for video: {video_path}")
        return float(frames / fps)
    finally:
        cap.release()


def default_roi(width: int, height: int) -> tuple[int, int, int, int]:
    x = int(width * 0.1)
    y = int(height * 0.45)
    w = int(width * 0.8)
    h = int(height * 0.5)
    return x, y, w, h
