from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percentile))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def _sleep_task(index: int, sleep_ms: float) -> dict[str, Any]:
    started = time.perf_counter()
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    return {
        "index": index,
        "thread": threading.current_thread().name,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def run_concurrency_benchmark(
    *,
    task_count: int = 32,
    max_workers: int = 8,
    sleep_ms: float = 5.0,
) -> dict[str, Any]:
    task_count = max(0, int(task_count))
    max_workers = max(1, int(max_workers))
    sleep_ms = max(0.0, float(sleep_ms))
    if task_count == 0:
        return {
            "task_count": 0,
            "completed": 0,
            "errors": 0,
            "max_workers": max_workers,
            "active_threads": 0,
            "elapsed_ms": 0.0,
            "task_p50_ms": 0.0,
            "task_p95_ms": 0.0,
            "estimated_serial_ms": 0.0,
            "estimated_speedup": 0.0,
        }

    started = time.perf_counter()
    task_elapsed_ms: list[float] = []
    task_threads: set[str] = set()
    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="labsop-bench") as executor:
        futures = [executor.submit(_sleep_task, index, sleep_ms) for index in range(task_count)]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                errors += 1
                continue
            task_elapsed_ms.append(float(result["elapsed_ms"]))
            task_threads.add(str(result["thread"]))

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    estimated_serial_ms = sum(task_elapsed_ms)
    return {
        "task_count": task_count,
        "completed": len(task_elapsed_ms),
        "errors": errors,
        "max_workers": max_workers,
        "active_threads": len(task_threads),
        "elapsed_ms": round(elapsed_ms, 3),
        "task_p50_ms": round(_percentile(task_elapsed_ms, 0.50), 3),
        "task_p95_ms": round(_percentile(task_elapsed_ms, 0.95), 3),
        "estimated_serial_ms": round(estimated_serial_ms, 3),
        "estimated_speedup": round(estimated_serial_ms / elapsed_ms, 3) if elapsed_ms > 0 else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight LabSOPGuard thread-pool benchmark.")
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sleep-ms", type=float, default=5.0)
    args = parser.parse_args(argv)
    report = run_concurrency_benchmark(
        task_count=args.tasks,
        max_workers=args.workers,
        sleep_ms=args.sleep_ms,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
