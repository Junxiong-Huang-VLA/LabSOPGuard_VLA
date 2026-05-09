from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import APIRouter, Depends


def create_diagnostics_router(
    *,
    require_operator_context: Callable[..., Dict[str, Any]],
    project_root: Callable[[], Path],
    model_status: Callable[[], Dict[str, Any]],
) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["diagnostics"])

    @router.get("/diagnostics")
    async def get_runtime_diagnostics(
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        """Return runtime wiring status for ASR, embeddings, and detector integrations."""
        _ = auth_ctx
        return {
            "schema_version": "diagnostics.v1",
            "project_root": str(project_root()),
            "model_status": model_status(),
        }

    return router
