from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.concurrency_benchmark import run_concurrency_benchmark

pytestmark = pytest.mark.unit


def test_concurrency_benchmark_completes_parallel_tasks():
    report = run_concurrency_benchmark(task_count=12, max_workers=4, sleep_ms=20)

    assert report["completed"] == 12
    assert report["errors"] == 0
    assert report["max_workers"] == 4
    assert report["active_threads"] > 1
    assert report["estimated_speedup"] > 1.0
