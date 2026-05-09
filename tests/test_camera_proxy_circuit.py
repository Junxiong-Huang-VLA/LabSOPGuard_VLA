from __future__ import annotations

import pytest

pytest.skip("Camera proxy circuit tests moved to D:\\MultiCameraMonitor scope.", allow_module_level=True)

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.circuit_breaker import CircuitBreaker

pytestmark = pytest.mark.unit


def test_camera_proxy_circuit_opens_after_start_failure(tmp_path, monkeypatch):
    camera_proxy = importlib.import_module("backend.camera_proxy")
    monkeypatch.setattr(
        camera_proxy,
        "_CAMERA_CIRCUIT",
        CircuitBreaker(failure_threshold=1, recovery_sec=60.0),
    )
    monkeypatch.setattr(camera_proxy, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(camera_proxy._CAMERA_PROXY, "healthcheck", lambda timeout=1.0: False)

    def _fail_popen(*args, **kwargs):
        raise RuntimeError("start failed")

    service_proxy = importlib.import_module("backend.service_proxy")
    monkeypatch.setattr(service_proxy.subprocess, "Popen", _fail_popen)

    with pytest.raises(RuntimeError, match="start failed"):
        camera_proxy.ensure_camera_service()

    assert camera_proxy.camera_circuit_snapshot()["state"] == "open"
    with pytest.raises(RuntimeError, match="circuit breaker"):
        camera_proxy.ensure_camera_service()
