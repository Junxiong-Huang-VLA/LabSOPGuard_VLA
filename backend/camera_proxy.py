from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import APIRouter, Request

from backend.service_proxy import ManagedHttpServiceProxy, env_float, env_int
from labsopguard.circuit_breaker import CircuitBreaker


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_SERVICE_HOST = os.environ.get("CAMERA_SERVICE_HOST", "127.0.0.1")
CAMERA_SERVICE_PORT = int(os.environ.get("CAMERA_SERVICE_PORT", "8101"))
CAMERA_SERVICE_BASE = f"http://{CAMERA_SERVICE_HOST}:{CAMERA_SERVICE_PORT}"

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])

_CAMERA_CIRCUIT = CircuitBreaker(
    failure_threshold=env_int("CAMERA_PROXY_CIRCUIT_FAILURES", 3),
    recovery_sec=env_float("CAMERA_PROXY_CIRCUIT_RECOVERY_SEC", 30.0),
)

_CAMERA_PROXY = ManagedHttpServiceProxy(
    service_display="Camera",
    metric_name="camera_proxy",
    base_url=CAMERA_SERVICE_BASE,
    route_base_path="/api/v1/cameras",
    startup_args=[
        sys.executable,
        "-m",
        "uvicorn",
        "backend.camera_service:app",
        "--host",
        CAMERA_SERVICE_HOST,
        "--port",
        str(CAMERA_SERVICE_PORT),
    ],
    startup_env={
        "CAMERA_USB_ISOLATED": "1",
        "CAMERA_USB_CAMERAS": "usb0:1,usb1:2",
        "CAMERA_USB_MAIN_INDEX": "1",
        "CAMERA_USB_SIDE_INDEX": "2",
        "CAMERA_USB_WORKER_PORTS": "usb0:8700,usb1:8702",
    },
    cwd=PROJECT_ROOT,
    log_dir=PROJECT_ROOT / "outputs" / "run_logs",
    log_prefix=f"camera_service_{CAMERA_SERVICE_PORT}",
    startup_timeout_sec=30.0,
    circuit=_CAMERA_CIRCUIT,
    logger=logger,
)


def _sync_proxy_state() -> None:
    _CAMERA_PROXY.circuit = _CAMERA_CIRCUIT
    _CAMERA_PROXY.cwd = PROJECT_ROOT
    _CAMERA_PROXY.log_dir = PROJECT_ROOT / "outputs" / "run_logs"


def camera_circuit_snapshot() -> dict:
    return _CAMERA_CIRCUIT.snapshot()


def _healthcheck(timeout: float = 1.0) -> bool:
    return _CAMERA_PROXY.healthcheck(timeout=timeout)


def ensure_camera_service() -> None:
    _sync_proxy_state()
    _CAMERA_PROXY.ensure_service()


def shutdown_camera_service() -> None:
    _CAMERA_PROXY.shutdown()


@router.api_route("", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_camera_request(request: Request, path: str = ""):
    _sync_proxy_state()
    return await _CAMERA_PROXY.proxy_request(request, path)
