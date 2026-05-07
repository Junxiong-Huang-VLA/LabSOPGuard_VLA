from __future__ import annotations

import logging
import os
from typing import Any

from .models import StreamStat

try:
    from prometheus_client import Gauge, start_http_server
except Exception:  # pragma: no cover - optional dependency
    Gauge = None
    start_http_server = None


logger = logging.getLogger(__name__)


def _make_gauge(name: str, description: str, labels: list[str]):
    if Gauge is None:
        return None
    try:
        return Gauge(name, description, labels)
    except ValueError:
        return None


LABELS = ["role", "channel", "port", "stream"]
WIRELESS_FPS_IN = _make_gauge("labsopguard_wireless_video_fps_in", "Wireless video input FPS", LABELS)
WIRELESS_FPS_OUT = _make_gauge("labsopguard_wireless_video_fps_out", "Wireless video output FPS", LABELS)
WIRELESS_PACKET_RATE = _make_gauge("labsopguard_wireless_video_packet_rate", "Wireless video packet rate", LABELS)
WIRELESS_BITRATE_KBPS = _make_gauge("labsopguard_wireless_video_bitrate_kbps", "Wireless video bitrate in kbps", LABELS)
WIRELESS_QUEUE_DEPTH = _make_gauge("labsopguard_wireless_video_queue_depth", "Wireless video queue depth", LABELS)
WIRELESS_DROP_COUNT = _make_gauge("labsopguard_wireless_video_drop_count", "Wireless video local drop count", LABELS)
WIRELESS_TX_DROP_COUNT = _make_gauge("labsopguard_wireless_video_tx_drop_count", "Wireless video transport send drop count", LABELS)
WIRELESS_RECONNECT_COUNT = _make_gauge("labsopguard_wireless_video_reconnect_count", "Wireless video reconnect count", LABELS)
WIRELESS_LATENCY_MS = _make_gauge("labsopguard_wireless_video_latency_ms", "Wireless video average end-to-end latency", LABELS)
WIRELESS_PACKETS_RX = _make_gauge("labsopguard_wireless_video_packets_rx", "Wireless video received packet count", LABELS)
WIRELESS_PACKETS_LOST = _make_gauge("labsopguard_wireless_video_packets_lost", "Wireless video lost packet count", LABELS)
WIRELESS_STATE = _make_gauge(
    "labsopguard_wireless_video_state",
    "Wireless video state as one-hot labels",
    LABELS + ["state"],
)

KNOWN_STATES = ("INIT", "RUNNING", "DEGRADED", "RECONNECTING")


def _set(gauge: Any, labels: list[str], value: float) -> None:
    if gauge is not None:
        gauge.labels(*labels).set(float(value))


def emit_wireless_video_metrics(
    *,
    role: str,
    channel: str,
    port: int,
    stat: StreamStat,
    stream_name: str = "",
) -> None:
    labels = [str(role), str(channel), str(port), str(stream_name or stat.extra.get("stream_name") or "-")]
    for gauge, value in (
        (WIRELESS_FPS_IN, stat.fps_in),
        (WIRELESS_FPS_OUT, stat.fps_out),
        (WIRELESS_PACKET_RATE, stat.packet_rate),
        (WIRELESS_BITRATE_KBPS, stat.bitrate_kbps),
        (WIRELESS_QUEUE_DEPTH, stat.queue_depth),
        (WIRELESS_DROP_COUNT, stat.drop_count),
        (WIRELESS_TX_DROP_COUNT, stat.tx_drop_count),
        (WIRELESS_RECONNECT_COUNT, stat.reconnect_count),
        (WIRELESS_LATENCY_MS, stat.e2e_latency_ms_avg),
        (WIRELESS_PACKETS_RX, stat.packets_rx),
        (WIRELESS_PACKETS_LOST, stat.packets_lost),
    ):
        _set(gauge, labels, float(value or 0.0))
    if WIRELESS_STATE is not None:
        active_state = str(stat.state or "INIT").upper()
        states = set(KNOWN_STATES)
        states.add(active_state)
        for state in states:
            WIRELESS_STATE.labels(*(labels + [state])).set(1.0 if state == active_state else 0.0)


def start_metrics_server_from_env(env_name: str, *, default_port: int = 0) -> int:
    raw_port = os.getenv(env_name, str(default_port)).strip()
    try:
        port = int(raw_port)
    except ValueError:
        port = 0
    if port <= 0 or start_http_server is None:
        return 0
    start_http_server(port)
    logger.info("wireless video metrics server listening on :%d via %s", port, env_name)
    return port
