from __future__ import annotations

import time

from labsopguard.resilience import CircuitBreaker, RateLimiter, RetryConfig, resilient_call


def test_resilient_call_times_out_to_fallback() -> None:
    def slow_call() -> str:
        time.sleep(0.2)
        return "late"

    result = resilient_call(
        slow_call,
        retry_config=RetryConfig(max_retries=0, timeout=0.05),
        fallback="fallback",
    )

    assert result == "fallback"


def test_resilient_call_respects_retry_then_success() -> None:
    attempts = {"count": 0}

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary")
        return "ok"

    result = resilient_call(
        flaky,
        retry_config=RetryConfig(max_retries=1, backoff_factor=0.01, timeout=1.0),
        fallback="fallback",
    )

    assert result == "ok"
    assert attempts["count"] == 2


def test_circuit_breaker_skips_open_calls() -> None:
    breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_sec=60)
    breaker.record_failure()

    result = resilient_call(lambda: "should-not-run", circuit_breaker=breaker, fallback="fallback")

    assert result == "fallback"


def test_rate_limiter_spaces_calls() -> None:
    limiter = RateLimiter(calls_per_second=20.0)
    start = time.monotonic()
    limiter.wait()
    limiter.wait()

    assert time.monotonic() - start >= 0.025
