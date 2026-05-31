"""Tests for observability module."""
import json
import time
from pathlib import Path

from labsopguard.observability import StageTimer, StructuredLogger


class TestStageTimer:
    def test_measure_stage(self):
        timer = StageTimer()
        with timer.measure("test_stage"):
            time.sleep(0.05)
        assert "test_stage" in timer.stages
        assert timer.stages["test_stage"] >= 0.04

    def test_multiple_stages(self):
        timer = StageTimer()
        with timer.measure("stage_a"):
            time.sleep(0.01)
        with timer.measure("stage_b"):
            time.sleep(0.01)
        assert len(timer.stages) == 2
        assert timer.total_sec > 0

    def test_save(self, tmp_path: Path):
        timer = StageTimer()
        timer.stages = {"ingestion": 2.1, "detection": 45.3}
        path = tmp_path / "timing.json"
        timer.save(path)
        data = json.loads(path.read_text())
        assert data["stages"]["ingestion"] == 2.1
        assert data["total_sec"] == 47.4

    def test_to_dict(self):
        timer = StageTimer()
        timer.stages = {"a": 1.0, "b": 2.0}
        d = timer.to_dict()
        assert d["total_sec"] == 3.0


class TestStructuredLogger:
    def test_format_includes_experiment_id(self):
        slog = StructuredLogger("test", experiment_id="exp-001")
        slog.info("test message", extra_field="value")
        slog.warning("warning msg")
        slog.error("error msg")

    def test_set_context(self):
        slog = StructuredLogger("test")
        slog.set_context(user="admin", stage="ingestion")
        assert slog.context["user"] == "admin"
