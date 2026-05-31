from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_backend_main_project_root():
    """Keep tests that monkeypatch backend.main.PROJECT_ROOT from leaking state."""
    yield
    module = sys.modules.get("backend.main") or sys.modules.get("LabSOPGuard.backend.main")
    if module is None:
        return
    module.PROJECT_ROOT = PROJECT_ROOT
    if hasattr(module, "_EXPERIMENTS"):
        module._EXPERIMENTS.clear()
    if hasattr(module, "_EXPERIMENT_DETAIL_RESPONSE_CACHE"):
        module._EXPERIMENT_DETAIL_RESPONSE_CACHE.clear()
    task_store = getattr(module, "EXPERIMENT_TASK_STORE", None)
    if task_store is not None:
        task_store.base_dir = PROJECT_ROOT / "outputs" / "experiments" / "tasks"
        task_store.base_dir.mkdir(parents=True, exist_ok=True)
