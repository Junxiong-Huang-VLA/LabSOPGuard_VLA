from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp_sec: float
    frame_bgr: np.ndarray
    source: str


class VideoCaptureStream:
    """Unified real-time/file video reader for SOP monitoring."""

    def __init__(self, source: str | int, target_fps: Optional[float] = None) -> None:
        self.source = source
        self.target_fps = target_fps

    def frames(self, max_frames: Optional[int] = None) -> Generator[FramePacket, None, None]:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source}")

        try:
            count = 0
            src_fps = cap.get(cv2.CAP_PROP_FPS)
            src_fps = src_fps if src_fps and src_fps > 0 else 30.0
            step = 1
            if self.target_fps and self.target_fps > 0:
                step = max(1, int(round(src_fps / self.target_fps)))

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if count % step != 0:
                    count += 1
                    continue

                packet = FramePacket(
                    frame_id=count,
                    timestamp_sec=float(count / src_fps),
                    frame_bgr=frame,
                    source=str(self.source),
                )
                yield packet

                count += 1
                if max_frames is not None and count >= max_frames:
                    break
        finally:
            # Always release the capture handle, even if the caller raises mid-iteration.
            cap.release()


def ensure_video_exists(path: str) -> bool:
    return Path(path).exists()
