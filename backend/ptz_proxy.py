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
PTZ_SERVICE_HOST = os.environ.get("PTZ_SERVICE_HOST", "127.0.0.1")
PTZ_SERVICE_PORT = int(os.environ.get("PTZ_SERVICE_PORT", "8201"))
PTZ_SERVICE_BASE = f"http://{PTZ_SERVICE_HOST}:{PTZ_SERVICE_PORT}"

router = APIRouter(prefix="/api/v1/ptz-tracker", tags=["ptz-tracker"])

_PTZ_CIRCUIT = CircuitBreaker(
    failure_threshold=env_int("PTZ_PROXY_CIRCUIT_FAILURES", 3),
    recovery_sec=env_float("PTZ_PROXY_CIRCUIT_RECOVERY_SEC", 30.0),
)

_PTZ_PROXY = ManagedHttpServiceProxy(
    service_display="PTZ",
    metric_name="ptz_proxy",
    base_url=PTZ_SERVICE_BASE,
    route_base_path="/api/v1/ptz-tracker",
    startup_args=[
        sys.executable,
        "-m",
        "uvicorn",
        "backend.ptz_service:app",
        "--host",
        PTZ_SERVICE_HOST,
        "--port",
        str(PTZ_SERVICE_PORT),
    ],
    startup_env={
        "PTZ_VIDEO_MODE": "opencv",
        "PTZ_CAMERA_INDEX": "0",
        "PTZ_OPENCV_BACKEND": "dshow",
        "PTZ_VIDEO_FOURCC": "MJPG",
        "PTZ_WVD_CAMERA": "",
    },
    cwd=PROJECT_ROOT,
    log_dir=PROJECT_ROOT / "outputs" / "run_logs",
    log_prefix=f"ptz_service_{PTZ_SERVICE_PORT}",
    startup_timeout_sec=60.0,
    circuit=_PTZ_CIRCUIT,
    logger=logger,
)


def _sync_proxy_state() -> None:
    _PTZ_PROXY.circuit = _PTZ_CIRCUIT
    _PTZ_PROXY.cwd = PROJECT_ROOT
    _PTZ_PROXY.log_dir = PROJECT_ROOT / "outputs" / "run_logs"


def ptz_circuit_snapshot() -> dict:
    return _PTZ_CIRCUIT.snapshot()


def _healthcheck(timeout: float = 1.0) -> bool:
    return _PTZ_PROXY.healthcheck(timeout=timeout)


def ensure_ptz_service() -> None:
    _sync_proxy_state()
    _PTZ_PROXY.ensure_service()


def shutdown_ptz_service_process() -> None:
    _PTZ_PROXY.shutdown()


@router.api_route("", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
@router.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
async def proxy_ptz_request(request: Request, path: str = ""):
    _sync_proxy_state()
    return await _PTZ_PROXY.proxy_request(request, path)
