from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.camera_streaming import router as camera_router
from backend.camera_streaming import shutdown_client


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        yield
    finally:
        shutdown_client()


app = FastAPI(
    title="LabSOPGuard Camera Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/healthz")
def healthz():
    return {"ok": True}


app.include_router(camera_router)
