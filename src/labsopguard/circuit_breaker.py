from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional


class CircuitBreaker:
    def __init__(
        self,
        *,
        failure_threshold: int = 3,
        recovery_sec: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = max(1, int(failure_threshold))
        self.recovery_sec = max(0.1, float(recovery_sec))
        self._clock = clock
        self._lock = threading.RLock()
        self._state = "closed"
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._last_error: Optional[str] = None

    def allow(self) -> bool:
        with self._lock:
            if self._state != "open":
                return True
            opened_at = self._opened_at or self._clock()
            if self._clock() - opened_at >= self.recovery_sec:
                self._state = "half_open"
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._state = "closed"
            self._failure_count = 0
            self._opened_at = None
            self._last_error = None

    def record_failure(self, error: Any) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_error = str(error)
            if self._state == "half_open" or self._failure_count >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            remaining_sec = 0.0
            if self._state == "open" and self._opened_at is not None:
                remaining_sec = max(0.0, self.recovery_sec - (self._clock() - self._opened_at))
            return {
                "state": self._state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_sec": self.recovery_sec,
                "remaining_recovery_sec": round(remaining_sec, 3),
                "last_error": self._last_error,
            }
