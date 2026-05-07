from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.unit


def test_wireless_video_compose_profile_and_metrics_ports():
    compose = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]

    sender = services["wireless-video-sender"]
    receiver = services["wireless-video-receiver"]

    assert sender["profiles"] == ["wireless-video"]
    assert receiver["profiles"] == ["wireless-video"]
    assert "9301:9301" in sender["ports"]
    assert "9302:9302" in receiver["ports"]
    assert "5600-5610:5600-5610/udp" in receiver["ports"]
    assert "WIRELESS_VIDEO_SENDER_METRICS_PORT=9301" in sender["environment"]
    assert "WIRELESS_VIDEO_RECEIVER_METRICS_PORT=9302" in receiver["environment"]
    assert "9301/metrics" in sender["healthcheck"]["test"][-1]
    assert "9302/metrics" in receiver["healthcheck"]["test"][-1]
    assert sender["logging"]["options"]["max-size"] == "20m"
    assert receiver["deploy"]["resources"]["limits"]["memory"] == "2G"


def test_wireless_video_sender_default_config_exists():
    payload = yaml.safe_load((PROJECT_ROOT / "config" / "cameras" / "sender_default.json").read_text(encoding="utf-8"))

    assert payload["sender_id"] == "sender0"
    assert payload["network"]["remote_port"] == 5600
    assert payload["camera"]["camera_id"] == "cam0"
