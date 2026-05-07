from __future__ import annotations

import math
import time
from typing import Sequence

import cv2
import numpy as np

from .models import StreamStat, VideoFrame


def _to_display_image(frame_data: np.ndarray) -> np.ndarray:
    if frame_data.ndim == 2:
        normalized = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    return frame_data.copy()


def _resolution_label(width: int, height: int) -> str:
    tags = {
        (3840, 2160): "4K",
        (2560, 1440): "1440p",
        (1920, 1080): "1080p",
        (1280, 720): "720p",
        (640, 480): "480p",
    }
    return tags.get((int(width), int(height)), "")


class Renderer:
    def __init__(self, window_name: str, show_stats: bool) -> None:
        self._window_name = window_name
        self._show_stats = show_stats

    @property
    def window_name(self) -> str:
        return self._window_name

    def render(self, frame: VideoFrame, overlay_stat: StreamStat) -> bool:
        image = _to_display_image(frame.data)
        if self._show_stats:
            sender_id = str(overlay_stat.extra.get("sender_id", ""))
            camera_id = str(overlay_stat.extra.get("camera_id", ""))
            stream_channel_id = str(overlay_stat.extra.get("stream_channel_id", ""))
            stream_name = str(overlay_stat.extra.get("stream_name", ""))
            lines = [
                f"state={overlay_stat.state}",
                f"fps_in={overlay_stat.fps_in:.1f} fps_out={overlay_stat.fps_out:.1f} pkt={overlay_stat.packet_rate:.1f}",
                f"bitrate={overlay_stat.bitrate_kbps:.0f}kbps queue={overlay_stat.queue_depth}",
                f"latency={overlay_stat.e2e_latency_ms_avg:.1f}ms drop={overlay_stat.drop_count}",
                f"rx={overlay_stat.packets_rx} lost={overlay_stat.packets_lost}",
            ]
            if sender_id or camera_id or stream_channel_id:
                lines.insert(1, f"sender={sender_id or '-'} camera={camera_id or '-'} stream={stream_channel_id or '-'}")
            if stream_name:
                lines.insert(2, f"stream_type={stream_name}")
            megapixels = (int(frame.width) * int(frame.height)) / 1_000_000.0
            label = _resolution_label(frame.width, frame.height)
            res_text = f"res={frame.width}x{frame.height} {megapixels:.2f}MP"
            if label:
                res_text = f"res={frame.width}x{frame.height} {label} {megapixels:.2f}MP"
            lines.insert(3, res_text)
            for idx, line in enumerate(lines):
                cv2.putText(image, line, (20, 30 + idx * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(self._window_name, image)
        return self.poll_events()

    def poll_events(self) -> bool:
        key = cv2.waitKey(1)
        return key not in (27, ord("q"))

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            cv2.destroyAllWindows()


class MonitorWallRenderer:
    def __init__(self, window_name: str, show_stats: bool, tile_width: int = 640, tile_height: int = 360) -> None:
        self._window_name = window_name
        self._show_stats = show_stats
        self._tile_width = max(240, tile_width)
        self._tile_height = max(180, tile_height)

    @staticmethod
    def _screen_size() -> tuple[int, int]:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception:
            pass
        return 1920, 1080

    def render(self, tiles: Sequence[dict[str, object]]) -> bool:
        if not tiles:
            canvas = np.zeros((self._tile_height, self._tile_width, 3), dtype=np.uint8)
            cv2.putText(canvas, "Waiting for camera stream...", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.imshow(self._window_name, canvas)
            return self.poll_events()

        dynamic_width = self._tile_width
        dynamic_height = self._tile_height
        for tile in tiles:
            frame = tile.get("frame")
            if not isinstance(frame, VideoFrame):
                continue
            dynamic_width = max(dynamic_width, int(frame.width))
            dynamic_height = max(dynamic_height, int(frame.height))

        count = len(tiles)
        cols = max(1, math.ceil(math.sqrt(count)))
        rows = max(1, math.ceil(count / cols))
        screen_w, screen_h = self._screen_size()
        max_canvas_w = max(640, int(screen_w * 0.96))
        max_canvas_h = max(360, int(screen_h * 0.90))
        canvas_w = cols * dynamic_width
        canvas_h = rows * dynamic_height
        scale = min(max_canvas_w / float(canvas_w), max_canvas_h / float(canvas_h), 1.0)
        if scale < 1.0:
            dynamic_width = max(240, int(dynamic_width * scale))
            dynamic_height = max(180, int(dynamic_height * scale))
        canvas = np.zeros((rows * dynamic_height, cols * dynamic_width, 3), dtype=np.uint8)

        for idx, tile in enumerate(tiles):
            row = idx // cols
            col = idx % cols
            y0 = row * dynamic_height
            y1 = y0 + dynamic_height
            x0 = col * dynamic_width
            x1 = x0 + dynamic_width

            frame = tile.get("frame")
            if not isinstance(frame, VideoFrame):
                continue
            image = _to_display_image(frame.data)
            image = cv2.resize(image, (dynamic_width, dynamic_height), interpolation=cv2.INTER_AREA)

            camera_id = str(tile.get("camera_id", "")) or str(frame.camera_id or "-")
            sender_id = str(tile.get("sender_id", "")) or str(frame.sender_id or "-")
            stream_name = str(tile.get("stream_name", "")) or str(frame.stream_name or "-")
            stat = tile.get("stat")
            megapixels = (int(frame.width) * int(frame.height)) / 1_000_000.0
            label = _resolution_label(frame.width, frame.height)
            res_text = f"{frame.width}x{frame.height} {megapixels:.2f}MP"
            if label:
                res_text = f"{frame.width}x{frame.height} {label} {megapixels:.2f}MP"

            title = f"{camera_id} ({stream_name})"
            cv2.putText(image, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"sender={sender_id}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
            cv2.putText(image, res_text, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            if self._show_stats and isinstance(stat, StreamStat):
                cv2.putText(
                    image,
                    f"in/out={stat.fps_in:.1f}/{stat.fps_out:.1f} pkt={stat.packet_rate:.0f}",
                    (12, dynamic_height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            canvas[y0:y1, x0:x1] = image

        cv2.imshow(self._window_name, canvas)
        return self.poll_events()

    def poll_events(self) -> bool:
        key = cv2.waitKey(1)
        return key not in (27, ord("q"))

    def close(self) -> None:
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            cv2.destroyAllWindows()
