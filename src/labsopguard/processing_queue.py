"""Processing concurrency control.

Limits simultaneous experiment processing to prevent GPU OOM and API rate-limiting.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueuedTask:
    experiment_id: str
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    status: str = "queued"


class ProcessingQueue:
    """Limits concurrent experiment processing with a semaphore."""

    def __init__(self, max_concurrent: int = 1) -> None:
        self.max_concurrent = max(1, max_concurrent)
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._tasks: Dict[str, QueuedTask] = {}
        self._processing_count = 0

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent)
        return self._semaphore

    @property
    def queue_size(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == "queued")

    @property
    def processing_count(self) -> int:
        return self._processing_count

    def enqueue(self, experiment_id: str) -> QueuedTask:
        task = QueuedTask(experiment_id=experiment_id)
        self._tasks[experiment_id] = task
        logger.info("Experiment %s queued (queue_size=%d)", experiment_id, self.queue_size)
        return task

    async def acquire(self, experiment_id: str) -> None:
        await self.semaphore.acquire()
        self._processing_count += 1
        task = self._tasks.get(experiment_id)
        if task:
            task.status = "processing"
            task.started_at = time.time()
        logger.info(
            "Experiment %s acquired processing slot (active=%d/%d)",
            experiment_id, self._processing_count, self.max_concurrent,
        )

    def release(self, experiment_id: str, success: bool = True) -> None:
        self.semaphore.release()
        self._processing_count = max(0, self._processing_count - 1)
        task = self._tasks.get(experiment_id)
        if task:
            task.status = "completed" if success else "failed"
        logger.info(
            "Experiment %s released slot (status=%s, active=%d/%d)",
            experiment_id, "ok" if success else "failed",
            self._processing_count, self.max_concurrent,
        )

    def get_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(experiment_id)
        if not task:
            return None
        wait_time = None
        if task.status == "queued":
            position = sum(
                1 for t in self._tasks.values()
                if t.status == "queued" and t.queued_at < task.queued_at
            )
            wait_time = position * 180
        return {
            "experiment_id": experiment_id,
            "status": task.status,
            "queued_at": task.queued_at,
            "started_at": task.started_at,
            "estimated_wait_sec": wait_time,
        }

    def cleanup_old(self, max_age_sec: float = 3600) -> None:
        now = time.time()
        to_remove = [
            eid for eid, task in self._tasks.items()
            if task.status in ("completed", "failed") and (now - task.queued_at) > max_age_sec
        ]
        for eid in to_remove:
            del self._tasks[eid]


_global_queue: Optional[ProcessingQueue] = None


def get_processing_queue(max_concurrent: int = 1) -> ProcessingQueue:
    global _global_queue
    if _global_queue is None:
        _global_queue = ProcessingQueue(max_concurrent=max_concurrent)
    return _global_queue
