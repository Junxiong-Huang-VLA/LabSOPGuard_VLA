from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np

from project_name.video.capture import FramePacket, VideoCaptureStream


@dataclass
class CameraInputConfig:
    mode: str
    source: str
    target_fps: float


class CameraInput:
    def __init__(self, cfg: CameraInputConfig) -> None:
        self.cfg = cfg

    def frames(self, max_frames: Optional[int] = None) -> Generator[FramePacket, None, None]:
        mode = self.cfg.mode.lower()
        if mode == "mock":
            count = int(max_frames or 10)
            for i in range(count):
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                yield FramePacket(
                    frame_id=i,
                    timestamp_sec=float(i / max(self.cfg.target_fps, 1e-6)),
                    frame_bgr=frame,
                    source="mock_camera",
                )
            return

        stream = VideoCaptureStream(source=self.cfg.source, target_fps=self.cfg.target_fps)
        for pkt in stream.frames(max_frames=max_frames):
            yield pkt

