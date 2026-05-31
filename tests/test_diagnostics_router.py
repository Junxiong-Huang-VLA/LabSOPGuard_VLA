from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.diagnostics_router import create_diagnostics_router


pytestmark = pytest.mark.unit


def test_diagnostics_router_registers_runtime_contract(tmp_path: Path):
    app = FastAPI()
    app.include_router(
        create_diagnostics_router(
            require_operator_context=lambda: {"operator": "pytest"},
            project_root=lambda: tmp_path,
            model_status=lambda: {"yolo_model_exists": False},
        )
    )

    response = TestClient(app).get("/api/v1/diagnostics")

    assert response.status_code == 200
    assert response.json() == {
        "schema_version": "diagnostics.v1",
        "project_root": str(tmp_path),
        "model_status": {"yolo_model_exists": False},
    }
