from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ops_router import create_ops_router

pytestmark = pytest.mark.unit


def test_ops_router_registers_stable_paths(tmp_path: Path):
    app = FastAPI()
    app.include_router(
        create_ops_router(
            require_operator_context=lambda: {"operator_role": "admin"},
            project_root=lambda: tmp_path,
            safe_project_path=lambda path_value, default_path: default_path,
        )
    )

    paths = {route.path for route in app.routes if route.path.startswith("/api/v1/ops")}

    assert paths == {
        "/api/v1/ops/jobs",
        "/api/v1/ops/jobs/{job_id}",
        "/api/v1/ops/maintenance",
        "/api/v1/ops/maintenance/start",
        "/api/v1/ops/maintenance/stop",
        "/api/v1/ops/sqlite/maintenance",
        "/api/v1/ops/workspace-governance",
    }
