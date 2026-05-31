"""Resilience utilities: retry with backoff, circuit breaker, timeout."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_factor: float = 2.0
    max_backoff: float = 30.0
    timeout: float = 60.0


class CircuitBreaker:
    """Opens after consecutive failures, auto-resets after cooldown."""

    def __init__(self, name: str = "default", failure_threshold: int = 5, cooldown_sec: float = 30.0) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_sec = cooldown_sec
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if (time.time() - self._opened_at) >= self.cooldown_sec:
            self._half_open()
            return False
        return True

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._opened_at = time.time()
            logger.warning(
                "Circuit breaker [%s] OPEN after %d consecutive failures (cooldown=%.0fs)",
                self.name, self._consecutive_failures, self.cooldown_sec,
            )

    def _half_open(self) -> None:
        logger.info("Circuit breaker [%s] half-open, allowing next attempt", self.name)
        self._consecutive_failures = self.failure_threshold - 1
        self._opened_at = None


class RateLimiter:
    """Small process-local rate limiter for slow external APIs."""

    def __init__(self, calls_per_second: float = 2.0) -> None:
        self.calls_per_second = max(float(calls_per_second or 0.0), 0.0)
        self._min_interval = 1.0 / self.calls_per_second if self.calls_per_second > 0 else 0.0
        self._last_call_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_sec = self._min_interval - (now - self._last_call_at)
            if wait_sec > 0:
                time.sleep(wait_sec)
            self._last_call_at = time.monotonic()


def _call_with_timeout(
    func: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    timeout: float,
) -> Any:
    if timeout <= 0:
        return func(*args, **kwargs)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="resilient-call")
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as exc:
        future.cancel()
        name = getattr(func, "__name__", repr(func))
        raise TimeoutError(f"{name} timed out after {timeout:.1f}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def resilient_call(
    func: Callable,
    *args: Any,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker: Optional[CircuitBreaker] = None,
    rate_limiter: Optional[RateLimiter] = None,
    fallback: Any = None,
    **kwargs: Any,
) -> Any:
    """Call func with retry, backoff, and circuit breaker protection."""
    config = retry_config or RetryConfig()

    if circuit_breaker and circuit_breaker.is_open:
        logger.warning("Circuit breaker open, skipping call to %s", func.__name__)
        return fallback

    last_exc = None
    for attempt in range(config.max_retries + 1):
        try:
            if rate_limiter:
                rate_limiter.wait()
            result = _call_with_timeout(func, args, kwargs, float(config.timeout or 0.0))
            if circuit_breaker:
                circuit_breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            if circuit_breaker:
                circuit_breaker.record_failure()

            if attempt >= config.max_retries:
                break

            retry_after = _extract_retry_after(exc)
            if retry_after:
                wait = min(retry_after, config.max_backoff)
                logger.warning(
                    "Rate limited on %s, waiting %.1fs (attempt %d/%d)",
                    func.__name__, wait, attempt + 1, config.max_retries,
                )
            else:
                wait = min(config.backoff_factor ** attempt, config.max_backoff)
                logger.warning(
                    "Call to %s failed (%s), retrying in %.1fs (attempt %d/%d)",
                    func.__name__, exc, wait, attempt + 1, config.max_retries,
                )
            time.sleep(wait)

    logger.error(
        "All %d attempts to %s failed. Last error: %s",
        config.max_retries + 1, func.__name__, last_exc,
    )
    return fallback


def _extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract Retry-After from HTTP exception responses."""
    response = getattr(exc, "response", None)
    if response is None:
        return None
    status = getattr(response, "status_code", None)
    if status != 429:
        return None
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass
    return 5.0
