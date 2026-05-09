from __future__ import annotations

import json
import signal
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from labsopguard.runtime_paths import RUNTIME_ROOT, ensure_runtime_dirs
from labsopguard.soak_test import SoakTestConfig, load_soak_test_config, run_soak_test


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CaptureDaemonOptions:
    restart_on_failure: bool = True
    restart_backoff_sec: float = 30.0
    max_restarts: Optional[int] = None
    status_path: str = ".runtime/capture_daemon/status.json"
    run_once: bool = False


def load_capture_daemon_options(config_path: str | Path) -> CaptureDaemonOptions:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    daemon = payload.get("daemon") or {}
    max_restarts = daemon.get("max_restarts")
    return CaptureDaemonOptions(
        restart_on_failure=bool(daemon.get("restart_on_failure", True)),
        restart_backoff_sec=float(daemon.get("restart_backoff_sec", 30.0)),
        max_restarts=int(max_restarts) if max_restarts is not None else None,
        status_path=str(daemon.get("status_path") or ".runtime/capture_daemon/status.json"),
        run_once=bool(daemon.get("run_once", False)),
    )


class CaptureDaemon:
    """Supervise the multi-camera capture runner and persist daemon state."""

    def __init__(self, config: SoakTestConfig, options: CaptureDaemonOptions) -> None:
        ensure_runtime_dirs()
        self.config = config
        self.options = options
        self.stop_requested = False
        self.restart_count = 0
        self.status_path = Path(options.status_path)
        if not self.status_path.is_absolute():
            self.status_path = RUNTIME_ROOT.parent / self.status_path
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

    def request_stop(self, *_: Any) -> None:
        self.stop_requested = True
        self._write_status({"state": "stopping"})

    def _write_status(self, payload: Dict[str, Any]) -> None:
        status = {
            "schema_version": "capture_daemon.v1",
            "updated_at": _utc_now_iso(),
            "experiment_id": self.config.experiment_id,
            "restart_count": self.restart_count,
            **payload,
        }
        self.status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> Dict[str, Any]:
        last_report: Dict[str, Any] = {}
        self._write_status({"state": "starting", "options": asdict(self.options)})
        while not self.stop_requested:
            self._write_status({"state": "running"})
            try:
                last_report = run_soak_test(self.config)
            except Exception as exc:
                last_report = {
                    "schema_version": "capture_daemon_error.v1",
                    "experiment_id": self.config.experiment_id,
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": _utc_now_iso(),
                }

            self._write_status({"state": "completed", "last_report": last_report})
            if self.options.run_once or last_report.get("status") == "passed":
                break
            if not self.options.restart_on_failure:
                break
            self.restart_count += 1
            if self.options.max_restarts is not None and self.restart_count > self.options.max_restarts:
                self._write_status({"state": "failed", "last_report": last_report, "reason": "max_restarts exceeded"})
                break
            self._write_status({"state": "backoff", "last_report": last_report})
            time.sleep(max(0.1, float(self.options.restart_backoff_sec)))
        self._write_status({"state": "stopped", "last_report": last_report})
        return last_report


def run_capture_daemon(config_path: str | Path, *, run_once: bool = False) -> Dict[str, Any]:
    config = load_soak_test_config(config_path)
    options = load_capture_daemon_options(config_path)
    if run_once:
        options.run_once = True
    daemon = CaptureDaemon(config, options)
    signal.signal(signal.SIGINT, daemon.request_stop)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, daemon.request_stop)
    return daemon.run()
