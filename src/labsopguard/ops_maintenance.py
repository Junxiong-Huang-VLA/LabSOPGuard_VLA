from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from labsopguard.ops_jobs import submit_ops_job


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class PeriodicMaintenanceScheduler:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._project_root: Optional[Path] = None
        self._started_at: Optional[str] = None
        self._last_submitted: Dict[str, str] = {}
        self._submitted_counts: Dict[str, int] = {}

    def start(self, project_root: str | Path) -> Dict[str, Any]:
        with self._lock:
            self._project_root = Path(project_root)
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop.clear()
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._thread = threading.Thread(target=self._loop, name="ops-maintenance", daemon=True)
            self._thread.start()
            return self.status()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        with self._lock:
            if self._thread is thread:
                self._thread = None
            return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "schema_version": "periodic_maintenance.v1",
                "enabled": _env_bool("LABSOPGUARD_PERIODIC_MAINTENANCE_ENABLED", False),
                "running": self._thread is not None and self._thread.is_alive(),
                "project_root": str(self._project_root) if self._project_root else None,
                "started_at": self._started_at,
                "last_submitted": dict(self._last_submitted),
                "submitted_counts": dict(self._submitted_counts),
                "intervals_sec": self._intervals(),
            }

    def _intervals(self) -> Dict[str, float]:
        return {
            "material_health": max(30.0, _env_float("LABSOPGUARD_MATERIAL_HEALTH_INTERVAL_SEC", 900.0)),
            "material_reindex": max(60.0, _env_float("LABSOPGUARD_MATERIAL_REINDEX_INTERVAL_SEC", 3600.0)),
            "published_reindex": max(60.0, _env_float("LABSOPGUARD_PUBLISHED_REINDEX_INTERVAL_SEC", 3600.0)),
            "sqlite_maintenance": max(60.0, _env_float("LABSOPGUARD_SQLITE_MAINTENANCE_INTERVAL_SEC", 1800.0)),
        }

    def _max_workers(self) -> int:
        return max(1, _env_int("LABSOPGUARD_MATERIAL_MAINTENANCE_WORKERS", min(8, os.cpu_count() or 1)))

    def _sqlite_workers(self) -> int:
        return max(1, _env_int("LABSOPGUARD_SQLITE_MAINTENANCE_WORKERS", self._max_workers()))

    def _record_submit(self, kind: str) -> None:
        with self._lock:
            self._last_submitted[kind] = datetime.now(timezone.utc).isoformat()
            self._submitted_counts[kind] = self._submitted_counts.get(kind, 0) + 1

    def _submit_due(self, kind: str, due: Dict[str, float], now: float) -> None:
        project_root = self._project_root
        if project_root is None or now < due[kind]:
            return
        from labsopguard.material_maintenance import (
            maintain_sqlite_databases,
            rebuild_workspace_material_index,
            rebuild_workspace_published_materials_index,
            scan_experiment_material_health,
        )

        experiments_root = project_root / "outputs" / "experiments"
        max_workers = self._max_workers()
        if kind == "material_health":
            submit_ops_job(kind, scan_experiment_material_health, experiments_root, max_workers=max_workers)
        elif kind == "material_reindex":
            submit_ops_job(
                kind,
                rebuild_workspace_material_index,
                experiments_root,
                project_root / "outputs" / "workspace_material_index.sqlite",
                force_experiment_indexes=False,
                max_workers=max_workers,
            )
        elif kind == "published_reindex":
            submit_ops_job(
                kind,
                rebuild_workspace_published_materials_index,
                experiments_root,
                project_root / "outputs" / "workspace_published_materials.sqlite",
                max_workers=max_workers,
            )
        elif kind == "sqlite_maintenance":
            submit_ops_job(
                kind,
                maintain_sqlite_databases,
                project_root,
                backup=_env_bool("LABSOPGUARD_SQLITE_BACKUP_ENABLED", False),
                max_workers=self._sqlite_workers(),
            )
        self._record_submit(kind)
        due[kind] = now + self._intervals()[kind]

    def _loop(self) -> None:
        intervals = self._intervals()
        now = time.monotonic()
        due = {
            "material_health": now + min(5.0, intervals["material_health"]),
            "material_reindex": now + intervals["material_reindex"],
            "published_reindex": now + intervals["published_reindex"],
            "sqlite_maintenance": now + min(10.0, intervals["sqlite_maintenance"]),
        }
        while not self._stop.wait(1.0):
            if not _env_bool("LABSOPGUARD_PERIODIC_MAINTENANCE_ENABLED", False):
                continue
            current = time.monotonic()
            for kind in ("material_health", "material_reindex", "published_reindex", "sqlite_maintenance"):
                try:
                    self._submit_due(kind, due, current)
                except Exception:
                    due[kind] = current + min(60.0, self._intervals()[kind])


_scheduler = PeriodicMaintenanceScheduler()


def start_periodic_maintenance(project_root: str | Path) -> Dict[str, Any]:
    return _scheduler.start(project_root)


def stop_periodic_maintenance() -> Dict[str, Any]:
    return _scheduler.stop()


def periodic_maintenance_status() -> Dict[str, Any]:
    return _scheduler.status()
