from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np


logger = logging.getLogger("usb_camera_worker")


class UsbCameraWorker:
    def __init__(
        self,
        camera_id: str,
        device_index: int,
        width: int,
        height: int,
        fps: int,
        quality: int,
    ) -> None:
        self.camera_id = camera_id
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.quality = max(1, min(100, quality))
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._online = False
        self._frame_count = 0
        self._last_error = ""
        self._last_frame_ts = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _open_capture(self) -> cv2.VideoCapture:
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(self.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self) -> None:
        frame_interval = 1.0 / self.fps
        while not self._stop.is_set():
            cap = self._open_capture()
            if not cap.isOpened():
                self._set_offline(f"failed to open device index {self.device_index}")
                cap.release()
                self._stop.wait(1.0)
                continue

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or self.width)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.height)
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
            logger.info(
                "USB worker opened %s index=%d %dx%d@%.1ffps",
                self.camera_id,
                self.device_index,
                actual_w,
                actual_h,
                actual_fps,
            )

            failures = 0
            while not self._stop.is_set():
                start = time.monotonic()
                ok, frame = cap.read()
                if ok and frame is not None:
                    with self._lock:
                        self._frame = frame
                        self._online = True
                        self._last_error = ""
                        self._frame_count += 1
                        self._last_frame_ts = time.time()
                    failures = 0
                else:
                    failures += 1
                    self._set_offline(f"read failed {failures} times")
                    if failures >= self.fps * 2:
                        break
                sleep_s = frame_interval - (time.monotonic() - start)
                if sleep_s > 0:
                    self._stop.wait(sleep_s)

            cap.release()
            self._stop.wait(0.5)

    def _set_offline(self, error: str) -> None:
        with self._lock:
            self._online = False
            self._last_error = error

    def status(self) -> dict:
        with self._lock:
            frame = self._frame
            return {
                "camera_id": self.camera_id,
                "device_index": self.device_index,
                "online": self._online,
                "width": int(frame.shape[1]) if frame is not None else 0,
                "height": int(frame.shape[0]) if frame is not None else 0,
                "fps": self.fps,
                "frame_count": self._frame_count,
                "last_frame_ts": self._last_frame_ts,
                "last_error": self._last_error,
            }

    def jpeg(self, quality: int | None = None) -> bytes | None:
        with self._lock:
            frame = self._frame.copy() if self._frame is not None else None
        if frame is None:
            return None
        q = self.quality if quality is None else max(1, min(100, quality))
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return None
        return jpeg.tobytes()


def make_handler(worker: UsbCameraWorker):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path.startswith("/status"):
                self._send_json(worker.status())
                return
            if self.path.startswith("/snapshot"):
                jpeg = worker.jpeg()
                if jpeg is None:
                    self._send_json({"detail": "frame is not ready"}, status=503)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                frame_interval = 1.0 / worker.fps
                while True:
                    start = time.monotonic()
                    jpeg = worker.jpeg()
                    if jpeg is not None:
                        try:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n"
                                b"\r\n" + jpeg + b"\r\n"
                            )
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            return
                    sleep_s = frame_interval - (time.monotonic() - start)
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                return
            self._send_json({"detail": "not found"}, status=404)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=75)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    worker = UsbCameraWorker(
        camera_id=args.camera_id,
        device_index=args.device_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        quality=args.quality,
    )
    worker.start()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(worker))
    try:
        logger.info("USB worker listening on http://%s:%d for %s", args.host, args.port, args.camera_id)
        server.serve_forever()
    finally:
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
