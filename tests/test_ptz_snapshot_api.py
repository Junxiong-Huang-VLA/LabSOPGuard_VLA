import pytest

pytest.skip("PTZ tracker API tests moved to D:\\PtzTracker scope.", allow_module_level=True)

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend import ptz_tracker_streaming


JPEG_BYTES = b"\xff\xd8\xff\xe0test-jpeg\xff\xd9"


class FakePtzService:
    def __init__(self, *, started: bool, frame: bytes | None = JPEG_BYTES) -> None:
        self.started = started
        self.frame = frame
        self.timestamp = 1234.5
        self.ensure_started_called = False

    def ensure_started(self) -> None:
        self.ensure_started_called = True
        self.started = True

    def latest_snapshot(self) -> tuple[bytes | None, float]:
        return self.frame, self.timestamp

    def video_source_id(self) -> str:
        return "opencv:0"


def _client(monkeypatch, service: FakePtzService) -> TestClient:
    monkeypatch.setattr(ptz_tracker_streaming, "_service", service)
    app = FastAPI()
    app.include_router(ptz_tracker_streaming.router)
    return TestClient(app)


def test_ptz_snapshot_returns_jpeg_from_running_service(monkeypatch):
    service = FakePtzService(started=True)
    client = _client(monkeypatch, service)

    response = client.get("/api/v1/ptz-tracker/snapshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-ptz-frame-timestamp"] == "1234.5"
    assert response.headers["x-ptz-video-source"] == "opencv:0"
    assert response.content == JPEG_BYTES
    assert service.ensure_started_called is False


def test_ptz_snapshot_requires_started_service_by_default(monkeypatch):
    service = FakePtzService(started=False)
    client = _client(monkeypatch, service)

    response = client.get("/api/v1/ptz-tracker/snapshot")

    assert response.status_code == 503
    assert service.ensure_started_called is False


def test_ptz_snapshot_can_auto_start(monkeypatch):
    service = FakePtzService(started=False)
    client = _client(monkeypatch, service)

    response = client.get("/api/v1/ptz-tracker/snapshot?auto_start=true")

    assert response.status_code == 200
    assert response.content == JPEG_BYTES
    assert service.ensure_started_called is True


def test_ptz_snapshot_returns_503_when_frame_is_not_ready(monkeypatch):
    service = FakePtzService(started=True, frame=None)
    client = _client(monkeypatch, service)

    response = client.get("/api/v1/ptz-tracker/snapshot?timeout_ms=0")

    assert response.status_code == 503
