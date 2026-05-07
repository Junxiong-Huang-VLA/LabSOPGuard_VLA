from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.ptz_tracker_streaming import router as ptz_router
from backend.ptz_tracker_streaming import shutdown_ptz_service


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        shutdown_ptz_service()


app = FastAPI(
    title="LabSOPGuard PTZ Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    return {"ok": True, "pid": os.getpid()}


app.include_router(ptz_router)
