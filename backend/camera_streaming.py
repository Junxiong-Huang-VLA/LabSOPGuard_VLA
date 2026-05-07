from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from wvd_sdk.client import Client
from wvd_sdk.types import ReadStatus
from backend.feishu_notifier import FeishuApiError, FeishuConfigError, FeishuNotifier

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cameras" / "receiver_5cam.json"
JPEG_QUALITY = int(os.environ.get("CAMERA_JPEG_QUALITY", "70"))
TARGET_FPS = int(os.environ.get("CAMERA_TARGET_FPS", "15"))
CAPTURE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "captures"
RECORDING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "recordings"
CALIBRATION_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "calibration"

# 鈹€鈹€ Camera labels 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
CAMERA_LABELS: dict[str, str] = {}

def _load_labels():
    label_file = PROJECT_ROOT / "config" / "cameras" / "labels.json"
    if label_file.exists():
        try:
            CAMERA_LABELS.update(json.loads(label_file.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("Failed to load camera labels: %s", e)

_load_labels()

def _label(camera_id: str) -> str:
    return CAMERA_LABELS.get(camera_id, camera_id)

# 鈹€鈹€ USB camera config 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


def _load_usb_cameras() -> dict[str, int]:
    """Load local USB monitor camera mapping.

    Defaults intentionally keep PTZ's dedicated camera index out of the
    multi-camera monitor. Override with CAMERA_USB_CAMERAS='usb0:1,usb1:2'
    or per-camera CAMERA_USB_MAIN_INDEX / CAMERA_USB_SIDE_INDEX.
    """
    raw = os.environ.get("CAMERA_USB_CAMERAS", "").strip()
    if raw:
        parsed: dict[str, int] = {}
        for item in raw.split(","):
            if ":" not in item:
                logger.warning("Ignoring invalid CAMERA_USB_CAMERAS item: %r", item)
                continue
            cam_id, index_text = item.split(":", 1)
            cam_id = cam_id.strip()
            if not cam_id:
                continue
            try:
                parsed[cam_id] = int(index_text.strip())
            except ValueError:
                logger.warning("Ignoring invalid USB index for %s: %r", cam_id, index_text)
        if parsed:
            return parsed

    return {
        "usb0": _env_int("CAMERA_USB_MAIN_INDEX", 1),
        "usb1": _env_int("CAMERA_USB_SIDE_INDEX", 2),
    }


USB_CAMERAS: dict[str, int] = _load_usb_cameras()
USB_ISOLATED = os.environ.get("CAMERA_USB_ISOLATED", "1").strip().lower() not in {"0", "false", "no", "off"}
USB_WORKER_HOST = os.environ.get("CAMERA_USB_WORKER_HOST", "127.0.0.1")
USB_WORKER_BASE_PORT = _env_int("CAMERA_USB_WORKER_BASE_PORT", 8700)
USB_WORKER_WIDTH = _env_int("CAMERA_USB_WORKER_WIDTH", 640)
USB_WORKER_HEIGHT = _env_int("CAMERA_USB_WORKER_HEIGHT", 480)
USB_WORKER_FPS = _env_int("CAMERA_USB_WORKER_FPS", TARGET_FPS)
USB_WORKER_QUALITY = _env_int("CAMERA_USB_WORKER_QUALITY", JPEG_QUALITY)


def _load_usb_worker_ports() -> dict[str, int]:
    raw = os.environ.get("CAMERA_USB_WORKER_PORTS", "").strip()
    if raw:
        parsed: dict[str, int] = {}
        for item in raw.split(","):
            if ":" not in item:
                logger.warning("Ignoring invalid CAMERA_USB_WORKER_PORTS item: %r", item)
                continue
            cam_id, port_text = item.split(":", 1)
            cam_id = cam_id.strip()
            if not cam_id:
                continue
            try:
                parsed[cam_id] = int(port_text.strip())
            except ValueError:
                logger.warning("Ignoring invalid USB worker port for %s: %r", cam_id, port_text)
        if parsed:
            return parsed
    return {
        "usb0": USB_WORKER_BASE_PORT,
        "usb1": USB_WORKER_BASE_PORT + 2,
    }


USB_WORKER_PORTS: dict[str, int] = _load_usb_worker_ports()

# 鈹€鈹€ Network quality tracker 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class _NetQualityTracker:
    def __init__(self, history_size: int = 60) -> None:
        self._lock = threading.Lock()
        self._history: dict[str, deque[dict]] = {}
        self._history_size = history_size

    def push(self, camera_id: str, rx: int, lost: int, fps_in: float, fps_out: float, pkt_rate: float, latency_ms: float) -> None:
        sample = {"ts": time.time(), "rx": rx, "lost": lost, "fps_in": round(fps_in, 1), "fps_out": round(fps_out, 1), "pkt_rate": round(pkt_rate, 1), "latency_ms": round(latency_ms, 1)}
        with self._lock:
            if camera_id not in self._history:
                self._history[camera_id] = deque(maxlen=self._history_size)
            self._history[camera_id].append(sample)

    def get(self, camera_id: str) -> list[dict]:
        with self._lock:
            return list(self._history.get(camera_id, []))

    def get_all(self) -> dict[str, list[dict]]:
        with self._lock:
            return {k: list(v) for k, v in self._history.items()}

    def summary(self, camera_id: str) -> dict:
        samples = self.get(camera_id)
        if not samples:
            return {"camera_id": camera_id, "samples": 0}
        recent = samples[-10:]
        total_rx = sum(s["rx"] for s in recent)
        total_lost = sum(s["lost"] for s in recent)
        loss_rate = total_lost / max(1, total_rx + total_lost)
        avg_fps = sum(s["fps_out"] for s in recent) / len(recent)
        avg_latency = sum(s["latency_ms"] for s in recent) / len(recent)
        return {"camera_id": camera_id, "samples": len(samples), "loss_rate": round(loss_rate, 3), "avg_fps_out": round(avg_fps, 1), "avg_latency_ms": round(avg_latency, 1), "avg_pkt_rate": round(sum(s["pkt_rate"] for s in recent) / len(recent), 1)}

_net_quality = _NetQualityTracker()

# 鈹€鈹€ Adaptive quality 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_adaptive_quality: dict[str, int] = {}
_ADAPTIVE_ENABLED = os.environ.get("CAMERA_ADAPTIVE_QUALITY", "1") not in ("0", "false")

def _get_jpeg_quality(camera_id: str) -> int:
    if not _ADAPTIVE_ENABLED:
        return JPEG_QUALITY
    return _adaptive_quality.get(camera_id, JPEG_QUALITY)

def _update_adaptive_quality(camera_id: str) -> None:
    if not _ADAPTIVE_ENABLED:
        return
    summary = _net_quality.summary(camera_id)
    loss = summary.get("loss_rate", 0)
    if loss > 0.5:
        _adaptive_quality[camera_id] = max(30, JPEG_QUALITY - 30)
    elif loss > 0.2:
        _adaptive_quality[camera_id] = max(40, JPEG_QUALITY - 15)
    else:
        _adaptive_quality[camera_id] = JPEG_QUALITY

# 鈹€鈹€ USB capture thread 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class _UsbCapture:
    def __init__(self, camera_id: str, device_index: int) -> None:
        self.camera_id = camera_id
        self.device_index = device_index
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._online = False
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._frame_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self._stop.clear()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            logger.warning("USB camera %s (index %d) failed to open", self.camera_id, self.device_index)
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        with self._lock:
            self._width, self._height, self._fps, self._online = w, h, fps, True
        logger.info("USB camera %s opened: index=%d %dx%d@%.1ffps", self.camera_id, self.device_index, w, h, fps)
        frame_interval = 1.0 / min(TARGET_FPS, fps)
        while not self._stop.is_set():
            t0 = time.monotonic()
            ret, frame = cap.read()
            with self._lock:
                if ret and frame is not None:
                    self._frame = frame
                    self._online = True
                    self._frame_count += 1
                else:
                    self._online = False
            elapsed = time.monotonic() - t0
            if (s := frame_interval - elapsed) > 0:
                time.sleep(s)
        cap.release()
        with self._lock:
            self._online = False

    def get_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            return (True, self._frame.copy()) if self._frame is not None else (False, None)

    def get_info(self) -> dict:
        with self._lock:
            return {"camera_id": self.camera_id, "online": self._online, "width": self._width, "height": self._height, "fps": self._fps, "frame_count": self._frame_count}

# 鈹€鈹€ Capture service 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class _CaptureService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._interval_sec = 5.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._total_captured = 0
        self._session_dir: Optional[Path] = None
        self._per_camera: dict[str, int] = {}
        self._started_at: Optional[str] = None
        self._camera_filter: set[str] | None = None
        self._sync_ts_enabled = True
        self._alerts: deque[dict] = deque(maxlen=200)

    def start(self, interval_sec: float, camera_ids: list[str] | None = None, sync_timestamps: bool = True) -> dict:
        with self._lock:
            if self._running:
                return {"status": "already_running", "session_dir": str(self._session_dir)}
            self._interval_sec = max(0.5, interval_sec)
            self._running = True
            self._stop_event.clear()
            self._total_captured = 0
            self._per_camera = {}
            self._sync_ts_enabled = sync_timestamps
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._started_at = ts
            self._session_dir = CAPTURE_OUTPUT_DIR / ts
            self._session_dir.mkdir(parents=True, exist_ok=True)
            self._camera_filter = set(camera_ids) if camera_ids else None
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            return {"status": "started", "session_dir": str(self._session_dir), "interval_sec": self._interval_sec}

    def stop(self) -> dict:
        with self._lock:
            if not self._running:
                return {"status": "not_running"}
            self._running = False
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        with self._lock:
            if self._session_dir:
                self._write_session_meta()
            return {"status": "stopped", "total_captured": self._total_captured, "per_camera": dict(self._per_camera), "session_dir": str(self._session_dir)}

    def get_status(self) -> dict:
        with self._lock:
            return {"running": self._running, "interval_sec": self._interval_sec, "total_captured": self._total_captured, "per_camera": dict(self._per_camera), "session_dir": str(self._session_dir) if self._session_dir else None, "started_at": self._started_at}

    def get_alerts(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._alerts)[-limit:]

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            self._capture_all()
            self._stop_event.wait(timeout=self._interval_sec)

    def _capture_all(self) -> None:
        sync_ts = datetime.now()
        ts_str = sync_ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        ts_ms = int(sync_ts.timestamp() * 1000)
        captured_this_round: dict[str, bool] = {}

        if _client is not None:
            for cam in _client.list_cameras():
                if self._camera_filter and cam.camera_id not in self._camera_filter:
                    continue
                try:
                    handle = _client.open_camera(cam.camera_id)
                    status, frame, meta = handle.get_latest_rgb()
                    if status == ReadStatus.OK and frame is not None:
                        self._save_frame(cam.camera_id, ts_str, ts_ms, frame)
                        captured_this_round[cam.camera_id] = True
                    else:
                        captured_this_round[cam.camera_id] = False
                except Exception:
                    captured_this_round[cam.camera_id] = False

        for cam_id, cap in _usb_captures.items():
            if self._camera_filter and cam_id not in self._camera_filter:
                continue
            ok, frame = cap.get_frame()
            if ok and frame is not None:
                self._save_frame(cam_id, ts_str, ts_ms, frame)
                captured_this_round[cam_id] = True
            else:
                captured_this_round[cam_id] = False

        for cam_id, success in captured_this_round.items():
            if not success:
                self._add_alert(cam_id, "capture_miss", f"Failed to capture frame from {cam_id}")

    def _save_frame(self, camera_id: str, ts_str: str, ts_ms: int, frame: np.ndarray) -> None:
        cam_dir = self._session_dir / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{camera_id}_{ts_str}.jpg"
        path = cam_dir / filename
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if self._sync_ts_enabled:
            meta_path = cam_dir / f"{camera_id}_{ts_str}.json"
            meta = {"camera_id": camera_id, "label": _label(camera_id), "timestamp_ms": ts_ms, "timestamp_iso": datetime.fromtimestamp(ts_ms / 1000).isoformat(), "filename": filename, "width": frame.shape[1], "height": frame.shape[0]}
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self._total_captured += 1
            self._per_camera[camera_id] = self._per_camera.get(camera_id, 0) + 1

    def _add_alert(self, camera_id: str, alert_type: str, message: str) -> None:
        with self._lock:
            self._alerts.append({"ts": datetime.now().isoformat(), "camera_id": camera_id, "type": alert_type, "message": message})

    def _write_session_meta(self) -> None:
        meta = {"started_at": self._started_at, "stopped_at": datetime.now().strftime("%Y%m%d_%H%M%S"), "total_captured": self._total_captured, "per_camera": dict(self._per_camera), "interval_sec": self._interval_sec}
        meta_path = self._session_dir / "session_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

_capture_service = _CaptureService()

# 鈹€鈹€ Recording service 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class _RecordingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._writers: dict[str, dict] = {}

    def start(self, camera_ids: list[str], fps: float = 15.0) -> dict:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = RECORDING_OUTPUT_DIR / ts
        session_dir.mkdir(parents=True, exist_ok=True)
        started = []
        with self._lock:
            for cam_id in camera_ids:
                if cam_id in self._writers:
                    continue
                path = session_dir / f"{cam_id}_{ts}.mp4"
                self._writers[cam_id] = {"path": path, "writer": None, "fps": fps, "frames": 0, "session_dir": session_dir, "started_at": ts}
                started.append(cam_id)
        for cam_id in started:
            threading.Thread(target=self._record_loop, args=(cam_id,), daemon=True).start()
        return {"status": "started", "session_dir": str(session_dir), "cameras": started}

    def stop(self, camera_ids: list[str] | None = None) -> dict:
        stopped = {}
        with self._lock:
            targets = list(camera_ids) if camera_ids else list(self._writers.keys())
            for cam_id in targets:
                info = self._writers.pop(cam_id, None)
                if info:
                    writer = info.get("writer")
                    if writer is not None:
                        writer.release()
                    stopped[cam_id] = {"path": str(info["path"]), "frames": info["frames"]}
        return {"status": "stopped", "recordings": stopped}

    def get_status(self) -> dict:
        with self._lock:
            return {"recording": bool(self._writers), "cameras": {k: {"path": str(v["path"]), "frames": v["frames"]} for k, v in self._writers.items()}}

    def _record_loop(self, camera_id: str) -> None:
        frame_interval = 1.0 / 15
        while True:
            with self._lock:
                info = self._writers.get(camera_id)
                if info is None:
                    return
            frame = _try_read_frame(camera_id)
            if frame is None:
                time.sleep(frame_interval)
                continue
            with self._lock:
                info = self._writers.get(camera_id)
                if info is None:
                    return
                if info["writer"] is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    info["writer"] = cv2.VideoWriter(str(info["path"]), fourcc, info["fps"], (w, h))
                info["writer"].write(frame)
                info["frames"] += 1
            time.sleep(frame_interval)

_recording_service = _RecordingService()

# 鈹€鈹€ Singletons 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

_client: Optional[Client] = None
_client_lock = threading.Lock()
_usb_captures: dict[str, _UsbCapture] = {}
_usb_started = False
_usb_lock = threading.Lock()
_usb_workers: dict[str, subprocess.Popen] = {}
_usb_worker_lock = threading.Lock()

def _ptz_service_running() -> bool:
    module = sys.modules.get("backend.ptz_tracker_streaming")
    service = getattr(module, "_service", None) if module is not None else None
    return bool(getattr(service, "started", False))


def _usb_worker_port(camera_id: str) -> int:
    if camera_id in USB_WORKER_PORTS:
        return USB_WORKER_PORTS[camera_id]
    try:
        offset = list(USB_CAMERAS).index(camera_id) * 2
    except ValueError:
        offset = int(USB_CAMERAS[camera_id])
    return USB_WORKER_BASE_PORT + offset


def _usb_worker_url(camera_id: str, path: str) -> str:
    return f"http://{USB_WORKER_HOST}:{_usb_worker_port(camera_id)}{path}"


def _worker_alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _ensure_usb_worker(camera_id: str) -> None:
    if camera_id not in USB_CAMERAS:
        raise ValueError(f"USB camera {camera_id} not found")
    if not USB_ISOLATED:
        return
    with _usb_worker_lock:
        proc = _usb_workers.get(camera_id)
        if _worker_alive(proc):
            return

        port = _usb_worker_port(camera_id)
        log_dir = PROJECT_ROOT / "outputs" / "run_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / f"usb_worker_{camera_id}_{port}.out.log"
        stderr_path = log_dir / f"usb_worker_{camera_id}_{port}.err.log"
        stdout = stdout_path.open("ab")
        stderr = stderr_path.open("ab")
        args = [
            sys.executable,
            str(PROJECT_ROOT / "backend" / "usb_camera_worker.py"),
            "--camera-id",
            camera_id,
            "--device-index",
            str(USB_CAMERAS[camera_id]),
            "--host",
            USB_WORKER_HOST,
            "--port",
            str(port),
            "--width",
            str(USB_WORKER_WIDTH),
            "--height",
            str(USB_WORKER_HEIGHT),
            "--fps",
            str(USB_WORKER_FPS),
            "--quality",
            str(USB_WORKER_QUALITY),
        ]
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(PROJECT_ROOT),
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform.startswith("win") else 0,
            )
            _usb_workers[camera_id] = proc
            logger.info("Started isolated USB worker %s on %s", camera_id, _usb_worker_url(camera_id, "/stream"))
        finally:
            stdout.close()
            stderr.close()


def _usb_worker_status(camera_id: str, *, start: bool = False) -> dict:
    if start:
        _ensure_usb_worker(camera_id)
    proc = _usb_workers.get(camera_id)
    if not _worker_alive(proc):
        return {
            "camera_id": camera_id,
            "device_index": USB_CAMERAS.get(camera_id),
            "online": False,
            "width": 0,
            "height": 0,
            "fps": USB_WORKER_FPS,
            "frame_count": 0,
            "last_error": "worker is not running",
        }
    try:
        with urllib.request.urlopen(_usb_worker_url(camera_id, "/status"), timeout=1.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {
            "camera_id": camera_id,
            "device_index": USB_CAMERAS.get(camera_id),
            "online": False,
            "width": 0,
            "height": 0,
            "fps": USB_WORKER_FPS,
            "frame_count": 0,
            "last_error": str(exc),
        }


def _read_usb_worker_snapshot(camera_id: str) -> bytes | None:
    _ensure_usb_worker(camera_id)
    try:
        with urllib.request.urlopen(_usb_worker_url(camera_id, "/snapshot"), timeout=5.0) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 503:
            return None
        raise
    except Exception:
        return None


def _iter_usb_worker_stream(camera_id: str):
    while True:
        _ensure_usb_worker(camera_id)
        try:
            with urllib.request.urlopen(_usb_worker_url(camera_id, "/stream"), timeout=10.0) as resp:
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
        except Exception as exc:
            logger.warning("USB worker stream reconnecting for %s: %s", camera_id, exc)
            time.sleep(0.5)


def _stop_usb_workers(reason: str = "") -> None:
    with _usb_worker_lock:
        if reason and _usb_workers:
            logger.info("Stopping isolated USB workers: %s", reason)
        for proc in list(_usb_workers.values()):
            if _worker_alive(proc):
                proc.terminate()
        for proc in list(_usb_workers.values()):
            try:
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        _usb_workers.clear()


def stop_usb_cameras(reason: str = "", *, stop_workers: bool = False) -> None:
    global _usb_started
    with _usb_lock:
        if not _usb_captures and not _usb_started:
            if stop_workers:
                _stop_usb_workers(reason)
            return
        if reason:
            logger.info("Stopping local USB monitor cameras: %s", reason)
        for cap in list(_usb_captures.values()):
            cap.stop()
        _usb_captures.clear()
        _usb_started = False
    if stop_workers:
        _stop_usb_workers(reason)


def _start_usb_cameras() -> None:
    global _usb_started
    if USB_ISOLATED:
        return
    with _usb_lock:
        if _usb_started:
            return
        _usb_started = True
        for cam_id, dev_idx in USB_CAMERAS.items():
            cap = _UsbCapture(cam_id, dev_idx)
            cap.start()
            _usb_captures[cam_id] = cap

def get_client() -> Client:
    global _client
    with _client_lock:
        if _client is None:
            config_path = os.environ.get("CAMERA_RECEIVER_CONFIG", str(DEFAULT_CONFIG))
            logger.info("Starting wvd_sdk Client with config: %s", config_path)
            _client = Client(config_path=config_path, auto_start=True)
    _start_usb_cameras()
    return _client

def shutdown_client() -> None:
    global _client
    _capture_service.stop()
    _recording_service.stop()
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None
    stop_usb_cameras("camera streaming shutdown", stop_workers=True)

def _is_usb(camera_id: str) -> bool:
    return camera_id in USB_CAMERAS

def _try_read_frame(camera_id: str) -> Optional[np.ndarray]:
    try:
        if _is_usb(camera_id):
            if USB_ISOLATED:
                data = _read_usb_worker_snapshot(camera_id)
                if not data:
                    return None
                arr = np.frombuffer(data, dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cap = _usb_captures.get(camera_id)
            if cap:
                ok, f = cap.get_frame()
                return f if ok else None
            return None
        if _client:
            handle = _client.open_camera(camera_id)
            s, f, _ = handle.get_latest_rgb()
            return f if s == ReadStatus.OK else None
    except Exception:
        return None

# 鈹€鈹€ Quality polling thread 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _quality_poll_loop() -> None:
    while True:
        time.sleep(2)
        if _client is None:
            continue
        try:
            for cam in _client.list_cameras():
                handle = _client.open_camera(cam.camera_id)
                profiles = handle.get_stream_profiles()
                fps_out = profiles[0].fps if profiles else 0
                _net_quality.push(cam.camera_id, 0, 0, fps_out, fps_out, 0, 0)
                _update_adaptive_quality(cam.camera_id)
        except Exception:
            pass

threading.Thread(target=_quality_poll_loop, daemon=True).start()

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# API ENDPOINTS 鈥?fixed routes FIRST, then /{camera_id} routes
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

@router.get("")
def list_cameras():
    client = get_client()
    result = []
    for c in client.list_cameras():
        result.append({"camera_id": c.camera_id, "label": _label(c.camera_id), "sender_id": c.sender_id, "online": c.online, "source": "wireless"})
    if USB_ISOLATED:
        for cam_id, dev_idx in USB_CAMERAS.items():
            info = _usb_worker_status(cam_id, start=False)
            result.append({
                "camera_id": cam_id,
                "label": _label(cam_id),
                "sender_id": "usb_worker",
                "online": bool(info.get("online")),
                "source": "usb",
                "device_index": dev_idx,
                "isolated": True,
            })
    else:
        for cap in _usb_captures.values():
            info = cap.get_info()
            result.append({"camera_id": info["camera_id"], "label": _label(info["camera_id"]), "sender_id": "usb_local", "online": info["online"], "source": "usb"})
    return {"cameras": result}

@router.get("/stats")
def all_camera_stats():
    client = get_client()
    result = []
    for c in client.list_cameras():
        try:
            handle = client.open_camera(c.camera_id)
            status = handle.get_status()
            profiles = handle.get_stream_profiles()
            quality_summary = _net_quality.summary(c.camera_id)
            result.append({"camera_id": c.camera_id, "label": _label(c.camera_id), "sender_id": c.sender_id, "online": c.online, "source": "wireless", "status": status.last_status_code, "profiles": [{"stream_name": p.stream_name, "width": p.width, "height": p.height, "fps": p.fps, "pixel_format": p.pixel_format} for p in profiles], "quality": quality_summary, "jpeg_quality": _get_jpeg_quality(c.camera_id)})
        except Exception:
            result.append({"camera_id": c.camera_id, "label": _label(c.camera_id), "sender_id": c.sender_id, "online": False, "source": "wireless", "status": "error", "profiles": [], "quality": {}, "jpeg_quality": JPEG_QUALITY})
    if USB_ISOLATED:
        for cam_id, dev_idx in USB_CAMERAS.items():
            info = _usb_worker_status(cam_id, start=False)
            online = bool(info.get("online"))
            result.append({
                "camera_id": cam_id,
                "label": _label(cam_id),
                "sender_id": "usb_worker",
                "online": online,
                "source": "usb",
                "status": "ok" if online else info.get("last_error") or "offline",
                "profiles": [{"stream_name": "rgb", "width": info.get("width", 0), "height": info.get("height", 0), "fps": info.get("fps", USB_WORKER_FPS), "pixel_format": "BGR8"}] if online else [],
                "quality": {},
                "jpeg_quality": USB_WORKER_QUALITY,
                "device_index": dev_idx,
                "isolated": True,
            })
    else:
        for cap in _usb_captures.values():
            info = cap.get_info()
            result.append({"camera_id": info["camera_id"], "label": _label(info["camera_id"]), "sender_id": "usb_local", "online": info["online"], "source": "usb", "status": "ok" if info["online"] else "offline", "profiles": [{"stream_name": "rgb", "width": info["width"], "height": info["height"], "fps": info["fps"], "pixel_format": "BGR8"}] if info["online"] else [], "quality": {}, "jpeg_quality": JPEG_QUALITY})
    return {"cameras": result}

# 鈹€鈹€ Capture routes (fixed) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class CaptureStartRequest(BaseModel):
    interval_sec: float = 5.0
    camera_ids: list[str] | None = None
    sync_timestamps: bool = True

@router.post("/capture/start")
def capture_start(req: CaptureStartRequest):
    get_client()
    return _capture_service.start(req.interval_sec, req.camera_ids, req.sync_timestamps)

@router.post("/capture/stop")
def capture_stop():
    return _capture_service.stop()

@router.get("/capture/status")
def capture_status():
    return _capture_service.get_status()

@router.get("/capture/alerts")
def capture_alerts(limit: int = Query(default=50, ge=1, le=500)):
    return {"alerts": _capture_service.get_alerts(limit)}

@router.get("/capture/history")
def capture_history():
    sessions = []
    if CAPTURE_OUTPUT_DIR.exists():
        for d in sorted(CAPTURE_OUTPUT_DIR.iterdir(), reverse=True):
            if d.is_dir():
                meta_path = d / "session_meta.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                total = sum(1 for _ in d.rglob("*.jpg"))
                sessions.append({"session_id": d.name, "path": str(d), "total_frames": total, **meta})
    return {"sessions": sessions}

@router.delete("/capture/history/{session_id}")
def delete_capture_session(session_id: str):
    session_dir = CAPTURE_OUTPUT_DIR / session_id
    if not session_dir.exists():
        raise HTTPException(404, "Session not found")
    shutil.rmtree(session_dir)
    return {"status": "deleted", "session_id": session_id}

# 鈹€鈹€ Recording routes (fixed) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class RecordingStartRequest(BaseModel):
    camera_ids: list[str]
    fps: float = 15.0

@router.post("/recording/start")
def recording_start(req: RecordingStartRequest):
    get_client()
    return _recording_service.start(req.camera_ids, req.fps)

@router.post("/recording/stop")
def recording_stop(camera_ids: list[str] | None = None):
    return _recording_service.stop(camera_ids)

@router.get("/recording/status")
def recording_status():
    return _recording_service.get_status()

# 鈹€鈹€ Network quality routes (fixed) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

@router.get("/quality")
def network_quality():
    return {"cameras": {cam_id: _net_quality.summary(cam_id) for cam_id in list(_net_quality.get_all().keys())}}

@router.get("/quality/history")
def quality_history():
    return _net_quality.get_all()

# 鈹€鈹€ Calibration routes (fixed) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

class CalibrationCaptureRequest(BaseModel):
    camera_ids: list[str]
    count: int = 20
    interval_sec: float = 2.0


class FeishuSnapshotRequest(BaseModel):
    message: str | None = None


class FeishuPushRequest(BaseModel):
    camera_id: str
    message: str | None = None
    quality: int = 95


def _default_feishu_snapshot_message(label: str, camera_id: str, ts: str) -> str:
    return f"鎽勫儚澶存姄鎷峔n鎽勫儚澶? {label} ({camera_id})\n鏃堕棿: {ts}"


def _push_snapshot_to_feishu(camera_id: str, message: str | None, quality: int) -> dict:
    get_client()
    frame = _read_frame(camera_id)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, f"Cannot encode frame from {camera_id}")

    label = _label(camera_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload_message = (message or "").strip() or _default_feishu_snapshot_message(label, camera_id, ts)
    try:
        result = FeishuNotifier.from_env().send_text_and_image(
            text=payload_message,
            image_bytes=jpeg.tobytes(),
            filename=f"{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
        )
    except FeishuConfigError as exc:
        raise HTTPException(503, str(exc))
    except FeishuApiError as exc:
        raise HTTPException(502, str(exc))
    return {
        "status": "sent",
        "camera_id": camera_id,
        "label": label,
        "timestamp": ts,
        "feishu": result.to_dict(),
    }


@router.post("/calibration/capture")
def calibration_capture(req: CalibrationCaptureRequest):
    get_client()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = CALIBRATION_OUTPUT_DIR / ts
    session_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[str]] = {}
    for i in range(req.count):
        sync_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        for cam_id in req.camera_ids:
            frame = _try_read_frame(cam_id)
            if frame is not None:
                cam_dir = session_dir / cam_id
                cam_dir.mkdir(parents=True, exist_ok=True)
                filename = f"calib_{cam_id}_{sync_ts}_{i:03d}.jpg"
                cv2.imwrite(str(cam_dir / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 98])
                results.setdefault(cam_id, []).append(filename)
        if i < req.count - 1:
            time.sleep(req.interval_sec)
    meta = {"session": ts, "cameras": req.camera_ids, "count": req.count, "interval_sec": req.interval_sec, "results": results}
    (session_dir / "calibration_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "done", "session_dir": str(session_dir), "results": {k: len(v) for k, v in results.items()}}

@router.get("/calibration/history")
def calibration_history():
    sessions = []
    if CALIBRATION_OUTPUT_DIR.exists():
        for d in sorted(CALIBRATION_OUTPUT_DIR.iterdir(), reverse=True):
            if d.is_dir():
                meta_path = d / "calibration_meta.json"
                meta = {}
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                sessions.append({"session_id": d.name, "path": str(d), **meta})
    return {"sessions": sessions}

# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
# Per-camera routes (MUST be after all fixed routes)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲

@router.post("/feishu/snapshot")
def push_camera_snapshot_to_feishu(req: FeishuPushRequest):
    camera_id = str(req.camera_id or "").strip()
    if not camera_id:
        raise HTTPException(422, "camera_id is required")
    quality = int(req.quality)
    if quality < 1 or quality > 100:
        raise HTTPException(422, "quality must be between 1 and 100")
    return _push_snapshot_to_feishu(camera_id=camera_id, message=req.message, quality=quality)


@router.get("/{camera_id}/status")
def camera_status(camera_id: str):
    get_client()
    if _is_usb(camera_id):
        if USB_ISOLATED:
            info = _usb_worker_status(camera_id, start=True)
            online = bool(info.get("online"))
            return {
                "camera_id": camera_id,
                "label": _label(camera_id),
                "sender_id": "usb_worker",
                "online": online,
                "rgb_available": online,
                "depth_available": False,
                "last_status_code": "ok" if online else info.get("last_error") or "offline",
                "device_index": USB_CAMERAS[camera_id],
                "isolated": True,
            }
        cap = _usb_captures.get(camera_id)
        if cap is None:
            raise HTTPException(404, f"Camera {camera_id} not found")
        info = cap.get_info()
        return {"camera_id": info["camera_id"], "label": _label(camera_id), "sender_id": "usb_local", "online": info["online"], "rgb_available": info["online"], "depth_available": False, "last_status_code": "ok" if info["online"] else "offline"}
    client = get_client()
    try:
        handle = client.open_camera(camera_id)
    except ValueError:
        raise HTTPException(404, f"Camera {camera_id} not found")
    status = handle.get_status()
    return {"camera_id": status.camera_id, "label": _label(camera_id), "sender_id": status.sender_id, "online": status.online, "rgb_available": status.rgb_available, "depth_available": status.depth_available, "last_status_code": status.last_status_code}

@router.get("/{camera_id}/snapshot")
def camera_snapshot(camera_id: str, quality: int = Query(default=95, ge=1, le=100), save: bool = Query(default=False)):
    get_client()
    frame = _read_frame(camera_id)
    if save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        save_dir = CAPTURE_OUTPUT_DIR / "snapshots" / camera_id
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{camera_id}_{ts}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg", headers={"Cache-Control": "no-cache"})


@router.post("/{camera_id}/feishu/snapshot")
def camera_snapshot_to_feishu(
    camera_id: str,
    req: FeishuSnapshotRequest | None = None,
    quality: int = Query(default=95, ge=1, le=100),
):
    return _push_snapshot_to_feishu(
        camera_id=camera_id,
        message=req.message if req else None,
        quality=quality,
    )


def _read_frame(camera_id: str) -> np.ndarray:
    frame = _try_read_frame(camera_id)
    if frame is None:
        raise HTTPException(503, f"Cannot read frame from {camera_id}")
    return frame

# 鈹€鈹€ MJPEG streams 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def _generate_mjpeg_usb(camera_id: str):
    if USB_ISOLATED:
        yield from _iter_usb_worker_stream(camera_id)
        return
    cap = _usb_captures.get(camera_id)
    if cap is None:
        return
    frame_interval = 1.0 / TARGET_FPS
    while True:
        t0 = time.monotonic()
        q = _get_jpeg_quality(camera_id)
        ok, frame = cap.get_frame()
        if ok and frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
            data = jpeg.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
        if (s := frame_interval - (time.monotonic() - t0)) > 0:
            time.sleep(s)

def _generate_mjpeg_wireless(camera_id: str):
    client = get_client()
    try:
        handle = client.open_camera(camera_id)
    except ValueError:
        return
    frame_interval = 1.0 / TARGET_FPS
    last_jpeg = None
    while True:
        t0 = time.monotonic()
        q = _get_jpeg_quality(camera_id)
        status_code, frame, meta = handle.get_latest_rgb()
        if status_code == ReadStatus.OK and frame is not None:
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
            last_jpeg = jpeg.tobytes()
        elif last_jpeg is None:
            status_code, frame, meta = handle.read_rgb(timeout_ms=500)
            if status_code == ReadStatus.OK and frame is not None:
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
                last_jpeg = jpeg.tobytes()
        if last_jpeg is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(last_jpeg)).encode() + b"\r\n\r\n" + last_jpeg + b"\r\n"
        if (s := frame_interval - (time.monotonic() - t0)) > 0:
            time.sleep(s)

@router.get("/{camera_id}/stream")
def camera_stream(camera_id: str):
    get_client()
    gen = _generate_mjpeg_usb(camera_id) if _is_usb(camera_id) else _generate_mjpeg_wireless(camera_id)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame", headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0", "X-Accel-Buffering": "no"})
