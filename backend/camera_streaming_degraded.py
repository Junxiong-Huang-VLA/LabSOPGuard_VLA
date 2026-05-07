from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.feishu_notifier import FeishuApiError, FeishuConfigError, FeishuNotifier
from backend.wireless_video.rtp import RtpH264Depacketizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "cameras" / "receiver_5cam.json"
LABELS_FILE = PROJECT_ROOT / "config" / "cameras" / "labels.json"

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras-degraded"])


class FeishuSnapshotRequest(BaseModel):
    message: str | None = None


class FeishuPushRequest(BaseModel):
    camera_id: str
    message: str | None = None
    quality: int = 80


def _default_feishu_snapshot_message(label: str, camera_id: str, ts: str) -> str:
    return f"鎽勫儚澶存姄鎷峔n鎽勫儚澶? {label} ({camera_id})\n鏃堕棿: {ts}"


@dataclass
class _Channel:
    camera_id: str
    label: str
    listen_ip: str
    listen_port: int
    sender_id: str = "unknown_sender"
    online: bool = False
    packets_rx: int = 0
    packets_lost: int = 0
    last_seen: float = 0.0
    last_error: str = ""
    stream_name: str = "rgb"


class _FallbackReceiver:
    """UDP metadata listener used when PyAV is unavailable.

    It intentionally does not decode H264. It keeps the configured ports bound,
    tracks incoming packet metadata, and serves an MJPEG status tile so the UI
    still exposes every configured channel.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        self._channels = self._load_channels()
        self._threads: list[threading.Thread] = []

    def _load_channels(self) -> dict[str, _Channel]:
        cfg = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        labels = {}
        if LABELS_FILE.exists():
            labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
        network = cfg.get("network", {})
        listen_ip = str(network.get("listen_ip") or "0.0.0.0")
        base_port = int(network.get("listen_port") or 5600)
        port_step = int(network.get("port_step") or 2)
        channels: dict[str, _Channel] = {}
        raw_channels = cfg.get("channels") or []
        if not raw_channels:
            raw_channels = [
                {"channel_id": f"cam{index}", "listen_port": base_port + index * port_step}
                for index in range(3)
            ]
        for index, item in enumerate(raw_channels):
            channel_id = str(item.get("camera_id") or item.get("channel_id") or f"cam{index}")
            port = int(item.get("listen_port") or (base_port + index * port_step))
            channels[channel_id] = _Channel(
                camera_id=channel_id,
                label=str(labels.get(channel_id, channel_id)),
                listen_ip=listen_ip,
                listen_port=port,
            )
        return channels

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            for channel in self._channels.values():
                thread = threading.Thread(target=self._listen_loop, args=(channel,), daemon=True)
                thread.start()
                self._threads.append(thread)

    def _listen_loop(self, channel: _Channel) -> None:
        parser = RtpH264Depacketizer()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 * 1024 * 1024)
            sock.bind((channel.listen_ip, channel.listen_port))
            sock.settimeout(0.2)
        except Exception as exc:
            with self._lock:
                channel.last_error = f"bind failed: {exc}"
            try:
                sock.close()
            except Exception:
                pass
            return

        while True:
            try:
                data, _ = sock.recvfrom(65535)
                pkt = parser.parse(data)
                with self._lock:
                    channel.packets_rx += 1
                    channel.packets_lost = parser.lost_packets
                    channel.last_seen = time.monotonic()
                    channel.online = True
                    channel.last_error = "PyAV missing: receiving packets, decode disabled"
                    if pkt.sender_id:
                        channel.sender_id = pkt.sender_id
                    if pkt.camera_id:
                        channel.camera_id = pkt.camera_id
                    if pkt.stream_name:
                        channel.stream_name = pkt.stream_name
            except socket.timeout:
                with self._lock:
                    if channel.last_seen <= 0 or time.monotonic() - channel.last_seen > 3.0:
                        channel.online = False
                continue
            except Exception as exc:
                with self._lock:
                    channel.last_error = str(exc)

    def list_channels(self) -> list[_Channel]:
        self.start()
        with self._lock:
            return [channel for channel in self._channels.values()]

    def get_channel(self, camera_id: str) -> _Channel | None:
        self.start()
        with self._lock:
            for channel in self._channels.values():
                if channel.camera_id == camera_id:
                    return channel
            return self._channels.get(camera_id)


_receiver = _FallbackReceiver()


def _make_status_frame(channel: _Channel | None, *, width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (24, 24, 24)
    if channel is None:
        title = "Camera not found"
        lines = ["No configured channel matches this camera id."]
    else:
        status = "ONLINE" if channel.online else "WAITING"
        title = f"{channel.label} ({channel.camera_id})"
        lines = [
            f"Status: {status}",
            f"UDP: {channel.listen_ip}:{channel.listen_port}",
            f"Packets: {channel.packets_rx}  Lost: {channel.packets_lost}",
            f"Sender: {channel.sender_id}  Stream: {channel.stream_name}",
            "PyAV is not installed, so H264 decode is disabled.",
            "Install package 'av' to restore real video frames.",
        ]
    color = (80, 220, 120) if channel and channel.online else (80, 160, 255)
    cv2.putText(frame, title, (48, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
    y = 160
    for line in lines:
        cv2.putText(frame, line, (52, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        y += 52
    cv2.putText(
        frame,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        (52, height - 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (170, 170, 170),
        2,
        cv2.LINE_AA,
    )
    return frame


def _jpeg_bytes(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return b""
    return jpeg.tobytes()


@router.get("")
def list_cameras():
    cameras = []
    for channel in _receiver.list_channels():
        cameras.append(
            {
                "camera_id": channel.camera_id,
                "label": channel.label,
                "sender_id": channel.sender_id,
                "online": channel.online,
                "source": "wireless-degraded",
                "listen_port": channel.listen_port,
            }
        )
    return {"cameras": cameras}


@router.get("/stats")
def all_camera_stats():
    result = []
    for channel in _receiver.list_channels():
        result.append(
            {
                "camera_id": channel.camera_id,
                "label": channel.label,
                "sender_id": channel.sender_id,
                "online": channel.online,
                "source": "wireless-degraded",
                "status": channel.last_error or ("ok" if channel.online else "waiting"),
                "profiles": [
                    {
                        "stream_name": channel.stream_name,
                        "width": 1280,
                        "height": 720,
                        "fps": 0,
                        "pixel_format": "H264-not-decoded",
                    }
                ],
                "quality": {
                    "camera_id": channel.camera_id,
                    "samples": channel.packets_rx,
                    "loss_rate": 0,
                    "avg_fps_out": 0,
                    "avg_latency_ms": 0,
                    "avg_pkt_rate": 0,
                },
                "jpeg_quality": 80,
            }
        )
    return {"cameras": result}


@router.get("/capture/status")
def capture_status():
    return {"running": False, "interval_sec": 0, "total_captured": 0, "per_camera": {}, "session_dir": None, "started_at": None}


@router.get("/recording/status")
def recording_status():
    return {"recording": False, "cameras": {}}


@router.get("/capture/alerts")
def capture_alerts(limit: int = Query(default=50, ge=1, le=500)):
    return {"alerts": []}


@router.get("/capture/history")
def capture_history():
    return {"sessions": []}


@router.post("/capture/start")
def capture_start():
    return {"status": "degraded", "detail": "PyAV is required for capture"}


@router.post("/capture/stop")
def capture_stop():
    return {"status": "stopped"}


@router.post("/recording/start")
def recording_start():
    return {"status": "degraded", "detail": "PyAV is required for recording"}


@router.post("/recording/stop")
def recording_stop():
    return {"status": "stopped"}


def _push_snapshot_to_feishu(camera_id: str, message: str | None, quality: int) -> dict:
    channel = _receiver.get_channel(camera_id)
    label = channel.label if channel else camera_id
    image = _jpeg_bytes(_make_status_frame(channel), quality)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload_message = (message or "").strip() or _default_feishu_snapshot_message(label, camera_id, ts)
    try:
        result = FeishuNotifier.from_env().send_text_and_image(
            text=payload_message,
            image_bytes=image,
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
    channel = _receiver.get_channel(camera_id)
    if channel is None:
        return {"camera_id": camera_id, "online": False, "last_status_code": "not_found"}
    return {
        "camera_id": channel.camera_id,
        "label": channel.label,
        "sender_id": channel.sender_id,
        "online": channel.online,
        "rgb_available": channel.online,
        "depth_available": False,
        "last_status_code": channel.last_error or ("ok" if channel.online else "waiting"),
        "listen_port": channel.listen_port,
    }


@router.get("/{camera_id}/snapshot")
def camera_snapshot(camera_id: str, quality: int = Query(default=80, ge=1, le=100), save: bool = Query(default=False)):
    channel = _receiver.get_channel(camera_id)
    return StreamingResponse(
        iter([_jpeg_bytes(_make_status_frame(channel), quality)]),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )


@router.post("/{camera_id}/feishu/snapshot")
def camera_snapshot_to_feishu(
    camera_id: str,
    req: FeishuSnapshotRequest | None = None,
    quality: int = Query(default=80, ge=1, le=100),
):
    return _push_snapshot_to_feishu(
        camera_id=camera_id,
        message=req.message if req else None,
        quality=quality,
    )

def _generate_mjpeg(camera_id: str) -> Iterator[bytes]:
    while True:
        channel = _receiver.get_channel(camera_id)
        jpeg = _jpeg_bytes(_make_status_frame(channel), 80)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.5)


@router.get("/{camera_id}/stream")
def camera_stream(camera_id: str):
    return StreamingResponse(
        _generate_mjpeg(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )

