from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.circuit_breaker import CircuitBreaker

try:
    from labsopguard.ops_metrics import set_external_service_metrics
except Exception:  # pragma: no cover - metrics are optional at runtime
    def set_external_service_metrics(
        service: str,
        snapshot: dict[str, Any] | None,
        last_request_ms: float | None = None,
    ) -> None:
        return None


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _alive(proc: subprocess.Popen | None) -> bool:
    return proc is not None and proc.poll() is None


def _copy_response_headers(resp: requests.Response) -> dict[str, str]:
    excluded = {
        "connection",
        "content-encoding",
        "content-length",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
    return {k: v for k, v in resp.headers.items() if k.lower() not in excluded}


def _iter_stream(resp: requests.Response):
    try:
        yield from resp.iter_content(chunk_size=64 * 1024)
    finally:
        resp.close()


class ManagedHttpServiceProxy:
    def __init__(
        self,
        *,
        service_display: str,
        metric_name: str,
        base_url: str,
        route_base_path: str,
        startup_args: list[str],
        startup_env: dict[str, str] | None = None,
        cwd: Path | None = None,
        log_dir: Path | None = None,
        log_prefix: str,
        startup_timeout_sec: float = 30.0,
        circuit: CircuitBreaker | None = None,
        logger: logging.Logger | None = None,
        health_path: str = "/healthz",
        stream_predicate: Callable[[Request, str], bool] | None = None,
    ) -> None:
        self.service_display = service_display
        self.metric_name = metric_name
        self.base_url = base_url.rstrip("/")
        self.route_base_path = route_base_path.rstrip("/")
        self.startup_args = list(startup_args)
        self.startup_env = dict(startup_env or {})
        self.cwd = cwd or PROJECT_ROOT
        self.log_dir = log_dir or (self.cwd / "outputs" / "run_logs")
        self.log_prefix = log_prefix
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.circuit = circuit or CircuitBreaker()
        self.logger = logger or logging.getLogger(__name__)
        self.health_path = health_path
        self.stream_predicate = stream_predicate or self._default_stream_predicate
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _default_stream_predicate(request: Request, path: str) -> bool:
        return request.method == "GET" and (path == "stream" or path.endswith("/stream"))

    def snapshot(self) -> dict[str, Any]:
        return self.circuit.snapshot()

    def emit_metrics(self, last_request_ms: float | None = None) -> None:
        set_external_service_metrics(self.metric_name, self.circuit.snapshot(), last_request_ms)

    def healthcheck(self, timeout: float = 1.0) -> bool:
        resp = None
        try:
            resp = requests.get(f"{self.base_url}{self.health_path}", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False
        finally:
            if resp is not None:
                resp.close()

    def ensure_service(self) -> None:
        if not self.circuit.allow():
            self.emit_metrics()
            raise RuntimeError(f"{self.service_display} service circuit breaker is open")

        started = time.perf_counter()
        if self.healthcheck(timeout=0.3):
            self.circuit.record_success()
            self.emit_metrics((time.perf_counter() - started) * 1000.0)
            return

        with self._lock:
            if self.healthcheck(timeout=0.3):
                self.circuit.record_success()
                self.emit_metrics((time.perf_counter() - started) * 1000.0)
                return
            if _alive(self._process):
                try:
                    self._process.terminate()
                    self._process.wait(timeout=2.0)
                except Exception:
                    try:
                        self._process.kill()
                    except Exception:
                        pass

            self.log_dir.mkdir(parents=True, exist_ok=True)
            stdout = (self.log_dir / f"{self.log_prefix}.out.log").open("ab")
            stderr = (self.log_dir / f"{self.log_prefix}.err.log").open("ab")
            env = os.environ.copy()
            env.update(self.startup_env)
            try:
                self._process = subprocess.Popen(
                    self.startup_args,
                    cwd=str(self.cwd),
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform.startswith("win") else 0,
                )
                self.logger.info("Started %s service on %s", self.service_display, self.base_url)
            except Exception as exc:
                self.circuit.record_failure(exc)
                self.emit_metrics((time.perf_counter() - started) * 1000.0)
                raise
            finally:
                stdout.close()
                stderr.close()

        deadline = time.monotonic() + self.startup_timeout_sec
        while time.monotonic() < deadline:
            if self.healthcheck(timeout=0.5):
                self.circuit.record_success()
                self.emit_metrics((time.perf_counter() - started) * 1000.0)
                return
            time.sleep(0.3)
        err = RuntimeError(f"{self.service_display} service did not start on {self.base_url}")
        self.circuit.record_failure(err)
        self.emit_metrics((time.perf_counter() - started) * 1000.0)
        raise err

    def shutdown(self) -> None:
        with self._lock:
            if not _alive(self._process):
                self._process = None
                return
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    async def proxy_request(self, request: Request, path: str = ""):
        try:
            await run_in_threadpool(self.ensure_service)
        except Exception as exc:
            raise HTTPException(502, f"{self.service_display} service unavailable: {exc}") from exc

        target_url = f"{self.base_url}{self.route_base_path}"
        if path:
            target_url += f"/{path}"
        if request.url.query:
            target_url += f"?{request.url.query}"

        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection"}
        }
        stream = self.stream_predicate(request, path)
        read_timeout = None if stream else 20.0
        if not self.circuit.allow():
            self.emit_metrics()
            raise HTTPException(502, f"{self.service_display} service circuit breaker is open")

        started = time.perf_counter()
        try:
            resp = await run_in_threadpool(
                requests.request,
                request.method,
                target_url,
                data=body or None,
                headers=headers,
                stream=stream,
                timeout=(5.0, read_timeout),
            )
        except requests.RequestException as exc:
            self.circuit.record_failure(exc)
            self.emit_metrics((time.perf_counter() - started) * 1000.0)
            raise HTTPException(502, f"{self.service_display} service request failed: {exc}") from exc

        if resp.status_code >= 500:
            self.circuit.record_failure(f"HTTP {resp.status_code}")
        else:
            self.circuit.record_success()
        self.emit_metrics((time.perf_counter() - started) * 1000.0)

        response_headers = _copy_response_headers(resp)
        media_type = resp.headers.get("content-type")
        if stream:
            return StreamingResponse(
                _iter_stream(resp),
                status_code=resp.status_code,
                media_type=media_type,
                headers=response_headers,
            )

        content = resp.content
        resp.close()
        return Response(
            content=content,
            status_code=resp.status_code,
            media_type=media_type,
            headers=response_headers,
        )
