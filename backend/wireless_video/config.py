from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CameraConfig:
    camera_id: Optional[str] = None
    enabled: bool = True
    # Phase default: RGB-only transport; depth interfaces remain available in SDK.
    enable_depth_streams: bool = False
    width: int = 1280
    height: int = 720
    fps: int = 30
    format_priority: List[str] = field(default_factory=lambda: ["RGB", "MJPG", "YUYV", "NV12"])
    device_index: Optional[int] = None
    serial_number: Optional[str] = None
    device_uid: Optional[str] = None
    remote_port: Optional[int] = None
    auto_exposure: bool = True
    exposure: Optional[int] = None
    gain: Optional[int] = None


@dataclass
class CodecConfig:
    codec: str = "h264"
    bitrate_kbps: int = 3000
    gop: int = 30
    max_b_frames: int = 0
    preset: str = "ultrafast"
    tune: str = "zerolatency"
    thread_count: int = 0
    thread_type: str = "slice"


@dataclass
class NetworkTxConfig:
    remote_ip: str = "127.0.0.1"
    remote_port: int = 5600
    port_step: int = 2
    mtu: int = 1200
    ttl: int = 64
    socket_buffer_bytes: int = 4 * 1024 * 1024


@dataclass
class NetworkRxConfig:
    listen_ip: str = "0.0.0.0"
    listen_port: int = 5600
    port_step: int = 2
    jitter_ms: int = 80
    reorder_window_frames: int = 6
    depacketizer_timeout_ms: int = 1000
    depacketizer_max_frames: int = 256
    depacketizer_max_fus: int = 256
    socket_buffer_bytes: int = 4 * 1024 * 1024


@dataclass
class RuntimeConfig:
    queue_size: int = 4
    camera_timeout_ms: int = 100
    recv_timeout_ms: int = 100
    watchdog_interval_ms: int = 1000
    degraded_threshold: int = 10
    reconnect_backoff_ms: List[int] = field(default_factory=lambda: [1000, 2000, 5000, 10000])
    log_interval_ms: int = 2000
    drop_frame_divisor: int = 1
    adaptive_drop_enabled: bool = True
    adaptive_drop_max_divisor: int = 4
    adaptive_drop_recover_window: int = 30
    auto_discover_cameras: bool = False
    auto_discover_interval_ms: int = 2000
    auto_fallback_resolution: bool = False
    auto_recover_resolution: bool = False
    auto_recover_window_ms: int = 30000
    auto_recover_cooldown_ms: int = 60000
    sync_profile_across_channels: bool = True
    # Receiver-only: when >0 and channels is empty, auto bind
    # listen_port + i * port_step for i in [0, count).
    auto_listen_port_count: int = 0


@dataclass
class DisplayConfig:
    window_name: str = "Gemini Receiver"
    show_stats: bool = True


@dataclass
class ReceiverChannelConfig:
    channel_id: Optional[str] = None
    enabled: bool = True
    listen_port: Optional[int] = None
    window_name: Optional[str] = None
    show_stats: Optional[bool] = None
    sender_id: Optional[str] = None
    camera_id: Optional[str] = None
    stream_name: Optional[str] = None


@dataclass
class SenderConfig:
    sender_id: str = "sender0"
    camera: CameraConfig = field(default_factory=CameraConfig)
    cameras: List[CameraConfig] = field(default_factory=list)
    codec: CodecConfig = field(default_factory=CodecConfig)
    network: NetworkTxConfig = field(default_factory=NetworkTxConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


@dataclass
class ReceiverConfig:
    network: NetworkRxConfig = field(default_factory=NetworkRxConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    channels: List[ReceiverChannelConfig] = field(default_factory=list)


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _camera_config(data: dict) -> CameraConfig:
    return CameraConfig(**data)


def _camera_config_list(data: object) -> List[CameraConfig]:
    if not isinstance(data, list):
        return []
    return [CameraConfig(**item) for item in data if isinstance(item, dict)]


def _codec_config(data: dict) -> CodecConfig:
    return CodecConfig(**data)


def _network_tx_config(data: dict) -> NetworkTxConfig:
    return NetworkTxConfig(**data)


def _network_rx_config(data: dict) -> NetworkRxConfig:
    return NetworkRxConfig(**data)


def _runtime_config(data: dict) -> RuntimeConfig:
    return RuntimeConfig(**data)


def _display_config(data: dict) -> DisplayConfig:
    return DisplayConfig(**data)


def _receiver_channel_config_list(data: object) -> List[ReceiverChannelConfig]:
    if not isinstance(data, list):
        return []
    return [ReceiverChannelConfig(**item) for item in data if isinstance(item, dict)]


def load_sender_config(path: str | Path) -> SenderConfig:
    raw = _load_json(path)
    return SenderConfig(
        sender_id=raw.get("sender_id", "sender0"),
        camera=_camera_config(raw.get("camera", {})),
        cameras=_camera_config_list(raw.get("cameras", [])),
        codec=_codec_config(raw.get("codec", {})),
        network=_network_tx_config(raw.get("network", {})),
        runtime=_runtime_config(raw.get("runtime", {})),
    )


def load_receiver_config(path: str | Path) -> ReceiverConfig:
    raw = _load_json(path)
    return ReceiverConfig(
        network=_network_rx_config(raw.get("network", {})),
        runtime=_runtime_config(raw.get("runtime", {})),
        display=_display_config(raw.get("display", {})),
        channels=_receiver_channel_config_list(raw.get("channels", [])),
    )
