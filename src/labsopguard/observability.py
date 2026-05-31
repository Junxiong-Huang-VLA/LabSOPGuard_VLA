"""Observability utilities: structured logging and pipeline timing."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageTimer:
    """Records timing for each pipeline stage."""

    stages: Dict[str, float] = field(default_factory=dict)
    _current_stage: Optional[str] = field(default=None, repr=False)
    _stage_start: float = field(default=0.0, repr=False)

    def start(self, stage_name: str) -> None:
        if self._current_stage:
            self.stop()
        self._current_stage = stage_name
        self._stage_start = time.perf_counter()

    def stop(self) -> float:
        if not self._current_stage:
            return 0.0
        elapsed = time.perf_counter() - self._stage_start
        self.stages[self._current_stage] = round(elapsed, 3)
        logger.info("Stage [%s] completed in %.3fs", self._current_stage, elapsed)
        self._current_stage = None
        return elapsed

    @contextmanager
    def measure(self, stage_name: str):
        self.start(stage_name)
        try:
            yield
        finally:
            self.stop()

    @property
    def total_sec(self) -> float:
        return round(sum(self.stages.values()), 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stages": dict(self.stages),
            "total_sec": self.total_sec,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Timing saved to %s (total=%.1fs)", path, self.total_sec)


class StructuredLogger:
    """JSON structured logging adapter."""

    def __init__(self, name: str, experiment_id: Optional[str] = None) -> None:
        self._logger = logging.getLogger(name)
        self.experiment_id = experiment_id
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs: Any) -> None:
        self.context.update(kwargs)

    def _format(self, level: str, message: str, **extra: Any) -> str:
        record = {
            "level": level,
            "message": message,
            "experiment_id": self.experiment_id,
            **self.context,
            **extra,
        }
        return json.dumps({k: v for k, v in record.items() if v is not None}, ensure_ascii=False)

    def info(self, message: str, **extra: Any) -> None:
        self._logger.info(self._format("INFO", message, **extra))

    def warning(self, message: str, **extra: Any) -> None:
        self._logger.warning(self._format("WARN", message, **extra))

    def error(self, message: str, **extra: Any) -> None:
        self._logger.error(self._format("ERROR", message, **extra))


def setup_json_logging(level: int = logging.INFO) -> None:
    """Configure root logger with JSON format for production."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_entry = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_entry, ensure_ascii=False)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
