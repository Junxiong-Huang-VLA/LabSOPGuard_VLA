from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_MAX_WORKERS = max(1, _env_int("LABSOPGUARD_OPS_JOB_WORKERS", 2))
_MAX_HISTORY = max(10, _env_int("LABSOPGUARD_OPS_JOB_HISTORY", 100))
def _new_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="ops-job")


_executor = _new_executor()
_executor_stopped = False
_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trim_history() -> None:
    if len(_jobs) <= _MAX_HISTORY:
        return
    finished = [
        (job.get("finished_at") or job.get("submitted_at") or "", job_id)
        for job_id, job in _jobs.items()
        if job.get("status") in {"completed", "failed", "cancelled"}
    ]
    for _, job_id in sorted(finished)[: max(0, len(_jobs) - _MAX_HISTORY)]:
        _jobs.pop(job_id, None)


def _emit_metrics_locked() -> None:
    counts: Dict[str, int] = {}
    for job in _jobs.values():
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    try:
        from labsopguard.ops_metrics import set_ops_job_metrics

        set_ops_job_metrics(counts)
    except Exception:
        pass


def _record(job_id: str, **updates: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(updates)
        _trim_history()
        _emit_metrics_locked()


def submit_ops_job(kind: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    global _executor, _executor_stopped
    job_id = f"ops_{uuid.uuid4().hex[:12]}"
    submitted = _now_iso()
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "submitted_at": submitted,
            "started_at": None,
            "finished_at": None,
            "elapsed_ms": None,
            "error": None,
            "result": None,
        }
        _emit_metrics_locked()

    def run_job() -> Any:
        started = time.perf_counter()
        _record(job_id, status="running", started_at=_now_iso())
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            _record(
                job_id,
                status="failed",
                finished_at=_now_iso(),
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)},
            )
            raise
        _record(
            job_id,
            status="completed",
            finished_at=_now_iso(),
            elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
            result=result,
        )
        return result

    with _lock:
        if _executor_stopped:
            _executor = _new_executor()
            _executor_stopped = False
        executor = _executor
    future = executor.submit(run_job)
    future.add_done_callback(lambda fut: _mark_cancelled(job_id, fut))
    return get_ops_job(job_id) or {"job_id": job_id, "kind": kind, "status": "queued"}


def _mark_cancelled(job_id: str, future: Future) -> None:
    if not future.cancelled():
        return
    _record(job_id, status="cancelled", finished_at=_now_iso())


def get_ops_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def list_ops_jobs(limit: int = 50) -> Dict[str, Any]:
    limit = max(1, min(int(limit), 500))
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda job: str(job.get("submitted_at") or ""), reverse=True)
        return {
            "schema_version": "ops_jobs.v1",
            "max_workers": _MAX_WORKERS,
            "total": len(jobs),
            "jobs": [dict(job) for job in jobs[:limit]],
        }


def shutdown_ops_jobs() -> None:
    global _executor_stopped
    _executor.shutdown(wait=False, cancel_futures=True)
    with _lock:
        _executor_stopped = True
