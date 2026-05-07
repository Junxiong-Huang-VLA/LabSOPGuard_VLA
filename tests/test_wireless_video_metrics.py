from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.wireless_video.metrics import emit_wireless_video_metrics, start_metrics_server_from_env
from backend.wireless_video.models import StreamStat

pytestmark = pytest.mark.unit


def test_wireless_video_metrics_emit_without_http_server():
    stat = StreamStat(
        fps_in=30.0,
        fps_out=29.5,
        packet_rate=900.0,
        queue_depth=1,
        packets_rx=100,
        packets_lost=2,
        state="RUNNING",
        extra={"stream_name": "rgb"},
    )

    emit_wireless_video_metrics(
        role="receiver",
        channel="rx0",
        port=5600,
        stat=stat,
    )


def test_wireless_video_metrics_server_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WIRELESS_VIDEO_RECEIVER_METRICS_PORT", raising=False)

    assert start_metrics_server_from_env("WIRELESS_VIDEO_RECEIVER_METRICS_PORT") == 0
