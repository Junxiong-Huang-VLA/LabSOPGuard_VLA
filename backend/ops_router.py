from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException


ProjectRootGetter = Callable[[], Path]
SafeProjectPath = Callable[[Optional[str], Path], Path]


def create_ops_router(
    *,
    require_operator_context: Callable[..., Dict[str, Any]],
    project_root: ProjectRootGetter,
    safe_project_path: SafeProjectPath,
) -> APIRouter:
    router = APIRouter(prefix="/api/v1/ops", tags=["ops"])

    @router.get("/jobs")
    async def list_ops_jobs_api(
        limit: int = 50,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.ops_jobs import list_ops_jobs

        return list_ops_jobs(limit=limit)

    @router.get("/jobs/{job_id}")
    async def get_ops_job_api(
        job_id: str,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.ops_jobs import get_ops_job

        job = get_ops_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ops job not found")
        return job

    @router.get("/maintenance")
    async def get_periodic_maintenance_api(
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.ops_maintenance import periodic_maintenance_status

        return periodic_maintenance_status()

    @router.post("/maintenance/start")
    async def start_periodic_maintenance_api(
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.ops_maintenance import start_periodic_maintenance

        return start_periodic_maintenance(project_root())

    @router.post("/maintenance/stop")
    async def stop_periodic_maintenance_api(
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.ops_maintenance import stop_periodic_maintenance

        return stop_periodic_maintenance()

    @router.post("/sqlite/maintenance")
    async def maintain_sqlite_databases_api(
        background: bool = True,
        backup: bool = False,
        max_workers: Optional[int] = None,
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.material_maintenance import maintain_sqlite_databases

        root = project_root()
        if background:
            from labsopguard.ops_jobs import submit_ops_job

            return submit_ops_job(
                "sqlite_maintenance",
                maintain_sqlite_databases,
                root,
                backup=backup,
                max_workers=max_workers,
            )
        return maintain_sqlite_databases(root, backup=backup, max_workers=max_workers)

    @router.get("/workspace-governance")
    async def get_workspace_governance(
        auth_ctx: Dict[str, Any] = Depends(require_operator_context),
    ):
        _ = auth_ctx
        from labsopguard.workspace_governance import build_workspace_governance_report

        return build_workspace_governance_report(project_root().parent)

    return router
