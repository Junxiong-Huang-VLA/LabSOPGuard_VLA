"""Tests for processing queue concurrency control."""
import asyncio

import pytest

from labsopguard.processing_queue import ProcessingQueue, QueuedTask


class TestProcessingQueue:
    def test_enqueue(self):
        q = ProcessingQueue(max_concurrent=1)
        task = q.enqueue("exp-001")
        assert task.status == "queued"
        assert q.queue_size == 1

    def test_acquire_and_release(self):
        async def _test():
            q = ProcessingQueue(max_concurrent=2)
            q.enqueue("exp-001")
            await q.acquire("exp-001")
            assert q.processing_count == 1
            q.release("exp-001", success=True)
            assert q.processing_count == 0
        asyncio.run(_test())

    def test_concurrency_limit(self):
        async def _test():
            q = ProcessingQueue(max_concurrent=1)
            q.enqueue("exp-001")
            q.enqueue("exp-002")
            await q.acquire("exp-001")

            acquired = False
            try:
                await asyncio.wait_for(q.acquire("exp-002"), timeout=0.1)
                acquired = True
            except asyncio.TimeoutError:
                pass

            assert not acquired
            assert q.processing_count == 1

            q.release("exp-001")
            await asyncio.wait_for(q.acquire("exp-002"), timeout=1.0)
            assert q.processing_count == 1
            q.release("exp-002")
        asyncio.run(_test())

    def test_get_status(self):
        q = ProcessingQueue(max_concurrent=1)
        q.enqueue("exp-001")
        status = q.get_status("exp-001")
        assert status is not None
        assert status["status"] == "queued"
        assert status["estimated_wait_sec"] is not None

    def test_cleanup_old(self):
        q = ProcessingQueue(max_concurrent=1)
        task = q.enqueue("exp-old")
        task.status = "completed"
        task.queued_at = 0
        q.cleanup_old(max_age_sec=1)
        assert q.get_status("exp-old") is None
