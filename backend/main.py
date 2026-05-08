"""
RealityLoop OP?- FastAPI 

"""
from __future__ import annotations

import asyncio
import copy
import base64
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import socket
import shutil
import ssl
import subprocess
import sys
import time
import uuid
from urllib.parse import quote, urlsplit
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form, Request, BackgroundTasks, Header, Depends, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# ?- 
try:
    import redis
except ImportError:
    redis = None

try:
    from celery import Celery
except ImportError:
    Celery = None

try:
    from prometheus_client import make_asgi_app
except ImportError:
    make_asgi_app = None

# Python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
KEY_ACTION_INDEXER_SRC = PROJECT_ROOT.parent / "src"
if KEY_ACTION_INDEXER_SRC.exists() and str(KEY_ACTION_INDEXER_SRC) not in sys.path:
    sys.path.insert(0, str(KEY_ACTION_INDEXER_SRC))

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

CONFIG_PATH = PROJECT_ROOT / "configs"

# 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_EXPERIMENT_ID_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,127}$")
_UPLOAD_CHUNK_SIZE = 1024 * 1024


def _media_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".json": "application/json",
        ".html": "text/html; charset=utf-8",
        ".pdf": "application/pdf",
        ".sqlite": "application/vnd.sqlite3",
        ".db": "application/vnd.sqlite3",
    }
    return media_types.get(suffix, "application/octet-stream")


def _parse_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    match = re.match(r"^bytes=(\d*)-(\d*)$", range_header.strip())
    if not match or file_size <= 0:
        return None
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return None
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    else:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            return None
        start = max(0, file_size - suffix_length)
        end = file_size - 1
    if start >= file_size or end < start:
        return None
    return start, min(end, file_size - 1)


def _iter_file_range(path: Path, start: int, end: int):
    with path.open("rb") as file_obj:
        file_obj.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = file_obj.read(min(_UPLOAD_CHUNK_SIZE, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def _serve_project_file(path: Path, request: Request, *, media_type: Optional[str] = None) -> Response:
    media_type = media_type or _media_type_for_path(path)
    headers = {"Accept-Ranges": "bytes"}
    range_header = request.headers.get("range")
    if range_header and media_type.startswith(("video/", "audio/")):
        file_size = path.stat().st_size
        byte_range = _parse_byte_range(range_header, file_size)
        if byte_range is None:
            return Response(
                status_code=416,
                headers={"Content-Range": f"bytes */{file_size}", "Accept-Ranges": "bytes"},
            )
        start, end = byte_range
        return StreamingResponse(
            _iter_file_range(path, start, end),
            status_code=206,
            media_type=media_type,
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Content-Length": str(end - start + 1),
            },
        )
    return FileResponse(
        path=str(path),
        media_type=media_type,
        filename=path.name,
        content_disposition_type="inline" if media_type.startswith(("video/", "image/")) else "attachment",
        headers=headers if media_type.startswith(("video/", "audio/")) else None,
    )


_BROWSER_SAFE_MP4_FOURCCS = {"avc1", "h264"}


def _find_ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _video_fourcc_tag(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return ""
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        return "".join(chr((fourcc >> (8 * idx)) & 0xFF) for idx in range(4)).strip().lower()
    except Exception:
        return ""
    finally:
        cap.release()


def _is_material_clip_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return (
        path.name.lower() == "clip.mp4"
        and ("published_materials" in parts or ("materials" in parts and "events" in parts))
    )


def _transcode_material_clip_for_browser(source: Path, target: Path) -> Optional[Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_exe = _find_ffmpeg_exe()
    if ffmpeg_exe:
        tmp = target.with_name(f"{target.stem}.tmp_{os.getpid()}_{int(time.time() * 1000)}{target.suffix}")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(tmp),
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
            if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 0:
                shutil.move(str(tmp), str(target))
                return target
            logger.warning("ffmpeg material clip transcode failed: %s", result.stderr[-800:] if result.stderr else result.returncode)
        except Exception as exc:
            logger.warning("ffmpeg material clip transcode error: %s", exc)
        finally:
            tmp.unlink(missing_ok=True)

    tmp = target.with_name(f"{target.stem}.tmp_{os.getpid()}_{int(time.time() * 1000)}{target.suffix}")
    cap = cv2.VideoCapture(str(source))
    writer = None
    try:
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return None
        out_width = width - (width % 2)
        out_height = height - (height % 2)
        if out_width <= 0 or out_height <= 0:
            return None
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"avc1"), fps, (out_width, out_height))
        if not writer.isOpened():
            return None
        written = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[1] != out_width or frame.shape[0] != out_height:
                frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            written += 1
        writer.release()
        writer = None
        cap.release()
        if written <= 0 or not tmp.exists() or tmp.stat().st_size <= 0:
            tmp.unlink(missing_ok=True)
            return None
        if _video_fourcc_tag(tmp) not in _BROWSER_SAFE_MP4_FOURCCS:
            tmp.unlink(missing_ok=True)
            return None
        shutil.move(str(tmp), str(target))
        return target
    except Exception as exc:
        logger.warning("Failed to transcode material clip for browser playback: %s -> %s (%s)", source, target, exc)
        tmp.unlink(missing_ok=True)
        return None
    finally:
        if writer is not None:
            writer.release()
        cap.release()


def _browser_playable_material_clip(path: Path) -> Path:
    if path.suffix.lower() != ".mp4" or not _is_material_clip_path(path):
        return path
    if _video_fourcc_tag(path) in _BROWSER_SAFE_MP4_FOURCCS:
        return path
    cache = path.with_name(f"{path.stem}.browser{path.suffix}")
    try:
        source_mtime = path.stat().st_mtime
        if (
            cache.exists()
            and cache.is_file()
            and cache.stat().st_size > 0
            and cache.stat().st_mtime >= source_mtime
            and _video_fourcc_tag(cache) in _BROWSER_SAFE_MP4_FOURCCS
        ):
            return cache
    except OSError:
        return path
    transcoded = _transcode_material_clip_for_browser(path, cache)
    return transcoded or path


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except Exception:
        return default
    return parsed if parsed >= minimum else default


MAX_VIDEO_UPLOAD_BYTES = _env_int("REALITYLOOP_MAX_VIDEO_UPLOAD_BYTES", 1024 * 1024 * 1024)
MAX_AUDIO_UPLOAD_BYTES = _env_int("REALITYLOOP_MAX_AUDIO_UPLOAD_BYTES", 256 * 1024 * 1024)


def _validate_experiment_id(experiment_id: str) -> str:
    value = str(experiment_id or "")
    if not value or not _EXPERIMENT_ID_RE.fullmatch(value):
        raise HTTPException(
            status_code=400,
            detail="Invalid experiment_id format; allowed: letters, numbers, dot, underscore, hyphen",
        )
    return value


def _sanitize_upload_filename(filename: Optional[str], *, default_name: str) -> str:
    raw_name = Path(str(filename or "")).name.strip()
    if not raw_name:
        raw_name = default_name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._")
    if not cleaned:
        cleaned = default_name
    return cleaned[:128]


async def _save_upload_file(upload: UploadFile, target_path: Path, *, max_bytes: int) -> int:
    total = 0
    try:
        with target_path.open("wb") as handle:
            while True:
                chunk = await upload.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail=f"File too large; max {max_bytes} bytes")
                handle.write(chunk)
    except HTTPException:
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
    finally:
        try:
            await upload.close()
        except Exception:
            pass
    return total


def _is_blocked_callback_ip(addr: ipaddress._BaseAddress) -> bool:
    return any(
        [
            addr.is_private,
            addr.is_loopback,
            addr.is_link_local,
            addr.is_multicast,
            addr.is_reserved,
            addr.is_unspecified,
        ]
    )


def _parse_callback_allowed_hosts() -> List[str]:
    raw = os.environ.get("REALITYLOOP_CALLBACK_ALLOWED_HOSTS", "")
    return [item.strip().lower() for item in raw.split(",") if item.strip()]


def _host_matches_allowed_patterns(host: str, patterns: List[str]) -> bool:
    if not patterns:
        return True
    normalized = host.strip().lower()
    for pattern in patterns:
        if pattern.startswith("*."):
            suffix = pattern[1:]
            if normalized.endswith(suffix) and normalized != pattern[2:]:
                return True
        elif normalized == pattern:
            return True
    return False


def _parse_callback_allowed_cidrs() -> List[ipaddress._BaseNetwork]:
    raw = os.environ.get("REALITYLOOP_CALLBACK_ALLOWED_CIDRS", "")
    cidrs: List[ipaddress._BaseNetwork] = []
    for item in [token.strip() for token in raw.split(",") if token.strip()]:
        try:
            cidrs.append(ipaddress.ip_network(item, strict=False))
        except ValueError as exc:
            raise ValueError(f"invalid callback CIDR: {item}") from exc
    return cidrs


def _ip_in_allowed_cidrs(addr: ipaddress._BaseAddress, cidrs: List[ipaddress._BaseNetwork]) -> bool:
    if not cidrs:
        return False
    return any(addr in network for network in cidrs)


def _validate_callback_url(callback_url: str) -> str:
    value = str(callback_url or "").strip()
    if not value:
        raise ValueError("callback_url is empty")
    if len(value) > 2048:
        raise ValueError("callback_url is too long")
    parsed = urlsplit(value)
    scheme = (parsed.scheme or "").lower()
    allow_http = os.environ.get("REALITYLOOP_CALLBACK_ALLOW_HTTP", "false").lower() in {"1", "true", "yes"}
    allowed_schemes = {"https", "http"} if allow_http else {"https"}
    if scheme not in allowed_schemes:
        raise ValueError(f"callback_url scheme must be one of: {', '.join(sorted(allowed_schemes))}")
    if not parsed.hostname:
        raise ValueError("callback_url hostname is required")
    host = parsed.hostname.strip().lower()
    allowed_hosts = _parse_callback_allowed_hosts()
    if allowed_hosts and not _host_matches_allowed_patterns(host, allowed_hosts):
        raise ValueError("callback_url host is not in allowed host whitelist")
    if host in {"localhost", "localhost.localdomain"}:
        raise ValueError("callback_url host is not allowed")
    allow_private = os.environ.get("REALITYLOOP_CALLBACK_ALLOW_PRIVATE", "false").lower() in {"1", "true", "yes"}
    allowed_cidrs = _parse_callback_allowed_cidrs()
    try:
        direct_ip = ipaddress.ip_address(host)
    except ValueError:
        direct_ip = None
    if direct_ip is not None:
        allowed_by_cidr = _ip_in_allowed_cidrs(direct_ip, allowed_cidrs)
        if allowed_cidrs and not allowed_by_cidr:
            raise ValueError("callback_url IP is not in allowed CIDR whitelist")
        if _is_blocked_callback_ip(direct_ip) and not allow_private and not allowed_by_cidr:
            raise ValueError("callback_url points to a private or local IP")
        return value
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        resolved = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except Exception as exc:
        raise ValueError(f"callback_url host resolve failed: {exc}") from exc
    if not resolved:
        raise ValueError("callback_url host resolve returned no addresses")
    resolved_addrs: List[ipaddress._BaseAddress] = []
    for item in resolved:
        sockaddr = item[4]
        if not sockaddr:
            continue
        ip_value = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_value)
        except ValueError:
            continue
        resolved_addrs.append(addr)
    if not resolved_addrs:
        raise ValueError("callback_url resolve yielded no valid IP addresses")
    if allowed_cidrs and not any(_ip_in_allowed_cidrs(addr, allowed_cidrs) for addr in resolved_addrs):
        raise ValueError("callback_url resolved IPs are not in allowed CIDR whitelist")
    for addr in resolved_addrs:
        allowed_by_cidr = _ip_in_allowed_cidrs(addr, allowed_cidrs)
        if _is_blocked_callback_ip(addr) and not allow_private and not allowed_by_cidr:
            raise ValueError("callback_url resolves to private/local address")
    return value


def _require_operator_context_from_request(request: Request) -> Dict[str, Any]:
    return _require_operator_context(
        authorization=request.headers.get("Authorization"),
        x_operator=request.headers.get("X-Operator"),
        x_operator_role=request.headers.get("X-Operator-Role"),
        x_allowed_experiments=request.headers.get("X-Allowed-Experiments"),
        x_actor_scope=request.headers.get("X-Actor-Scope"),
    )


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    outputs_dir = PROJECT_ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    auth_refresh_task = None
    if os.environ.get("REALITYLOOP_AUTH_CACHE_REFRESH_ENABLED", "true").lower() not in {"0", "false", "no"}:
        auth_refresh_task = asyncio.create_task(_auth_cache_refresh_loop())
    rules_path = CONFIG_PATH / "sop" / "rules.yaml"
    if not rules_path.exists():
        logger.warning(f"SOP? {rules_path}")
    model_cfg = CONFIG_PATH / "model" / "detection_runtime.yaml"
    if not model_cfg.exists():
        logger.warning(f"? {model_cfg}")
    _recover_orphaned_tasks()
    try:
        yield
    finally:
        if auth_refresh_task is not None:
            auth_refresh_task.cancel()


def _recover_orphaned_tasks() -> None:
    """On startup, mark any tasks that are still 'running' or 'queued' as failed.

    These are tasks that were in-flight when the server last stopped.  Without
    this, the frontend shows a permanent spinner and the experiment can never be
    re-processed.
    """
    task_dir = PROJECT_ROOT / "outputs" / "experiments" / "tasks"
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    recovered = 0
    if task_dir.exists():
        for task_file in task_dir.glob("*.json"):
            try:
                task = json.loads(task_file.read_text(encoding="utf-8-sig"))
                if task.get("status") in {"running", "queued"}:
                    task["status"] = "failed"
                    task["error_type"] = "server_restart"
                    task["error_message"] = "Task was interrupted by server restart. Re-run analysis to continue."
                    task_file.write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
                    # Also patch the experiment.json so the UI reflects the failure immediately
                    eid = task.get("experiment_id")
                    if eid:
                        exp_file = exp_dir / eid / "experiment.json"
                        if exp_file.exists():
                            exp = json.loads(exp_file.read_text(encoding="utf-8"))
                            if exp.get("status") in {"running", "analyzing", "queued"}:
                                exp["status"] = "failed"
                                exp["processing_stage"] = "failed"
                                exp["processing_error"] = "Task interrupted by server restart. Click 'Run analysis' to retry."
                                exp_file.write_text(json.dumps(exp, ensure_ascii=False, indent=2), encoding="utf-8")
                    recovered += 1
            except Exception:
                pass
    if recovered:
        logger.info("Recovered %d orphaned task(s) to failed state on startup", recovered)

# FastAPI?
app = FastAPI(
    title="RealityLoop Lab SOP Situational Awareness Platform",
    description="AI",
    version="2.0.0",
    lifespan=app_lifespan,
)

if make_asgi_app is not None:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
else:
    logger.warning("prometheus_client is not installed; /metrics is disabled")

# CORS?
_allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:5174,"
        "http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:5174,"
        "http://localhost:8080,http://127.0.0.1:8080",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class SystemConfig(BaseModel):
    """."""
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    max_concurrent_streams: int = 8
    yolo_model_path: str = "yolov8n-pose.pt"
    device: str = "cuda:0"
    alert_websocket_port: int = 8001
    pdf_output_dir: str = "outputs/reports"
    video_cache_dir: str = "outputs/video_cache"


class ClipBackfillRequest(BaseModel):
    start_time_sec: float
    end_time_sec: float
    camera_id: Optional[str] = None
    clip_id: Optional[str] = None

# 
def load_system_config() -> SystemConfig:
    config_file = CONFIG_PATH / "model" / "detection_runtime.yaml"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        return SystemConfig(**yaml_config.get("system", {}))
    return SystemConfig()

SYSTEM_CONFIG = load_system_config()

# Redis (graceful fallback)
_USE_REDIS = False
redis_client = None
if redis is not None:
    try:
        redis_client = redis.Redis.from_url(SYSTEM_CONFIG.redis_url, decode_responses=True)
        redis_client.ping()
        _USE_REDIS = True
    except Exception as e:
        print(f"Redis unavailable ({e}); using in-memory storage")
        _USE_REDIS = False
        redis_client = None
else:
    print("redis ")

# Celery
celery_app = None
if Celery is not None:
    celery_app = Celery(
        "sop_monitor",
        broker=SYSTEM_CONFIG.celery_broker_url,
        backend=SYSTEM_CONFIG.celery_broker_url,
    )
    if not _USE_REDIS:
        celery_app.conf.task_always_eager = True
        celery_app.conf.task_store_eager_result = False
else:
    print("celery is not installed; async task execution is disabled")

#  (Redisallback)
_MEM_STORAGE: Dict[str, Any] = {}

def _redis_hset(key: str, mapping: dict):
    if _USE_REDIS:
        redis_client.hset(key, mapping=mapping)
    else:
        _MEM_STORAGE.setdefault(key, {}).update(mapping)

def _redis_hgetall(key: str) -> dict:
    if _USE_REDIS:
        return redis_client.hgetall(key)
    return _MEM_STORAGE.get(key, {})

def _redis_lpush(key: str, value: str):
    if _USE_REDIS:
        redis_client.lpush(key, value)
    else:
        _MEM_STORAGE.setdefault(key, []).insert(0, value)

def _redis_lrange(key: str, start: int, end: int) -> list:
    if _USE_REDIS:
        return redis_client.lrange(key, start, end)
    data = _MEM_STORAGE.get(key, [])
    if end == -1:
        return data[start:]
    return data[start:end + 1]

def _redis_ltrim(key: str, start: int, end: int):
    if _USE_REDIS:
        redis_client.ltrim(key, start, end)
    else:
        data = _MEM_STORAGE.get(key, [])
        _MEM_STORAGE[key] = data[start:] if end == -1 else data[start:end + 1]

# 
class TaskRequest(BaseModel):
    """."""
    video_path: str
    sop_rules_path: Optional[str] = None
    camera_id: str = "camera_001"
    priority: int = 1
    callback_url: Optional[str] = None

class TaskStatus(BaseModel):
    """?"""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    current_step: Optional[str] = None
    violations_count: int = 0
    created_at: str
    updated_at: str

class ViolationAlert(BaseModel):
    """."""
    alert_id: str
    task_id: str
    frame_id: int
    timestamp_sec: float
    rule_id: str
    severity: str  # Critical, Major, Minor
    message: str
    screenshot_url: Optional[str] = None
    camera_id: str

# ?
class ConnectionManager:
    """WebSocket?"""
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# ---------------------------------------------------------------------------
# [DEPRECATED]  - 
# ?src/project_name/?/api/v1/streams/* ?/api/v1/tasks/* 
# /api/v1/experiments/*
# ---------------------------------------------------------------------------
_LEGACY_AVAILABLE = False
try:
    from project_name.detection.multi_level_detector import MultiLevelDetector
    from project_name.monitoring.sop_engine import SOPComplianceEngine, ViolationEvent
    from project_name.video.capture import FramePacket
    from project_name.report.pdf_report import PDFReportGenerator
    _LEGACY_AVAILABLE = True
except Exception as _legacy_err:
    logger.warning(f"[DEPRECATED] legacy module bootstrap failed: {_legacy_err}")
    MultiLevelDetector = None
    SOPComplianceEngine = None
    ViolationEvent = None
    FramePacket = None
    PDFReportGenerator = None

# ---------------------------------------------------------------------------
# [DEPRECATED]  - ?
# ?src/experiment/ 
# ---------------------------------------------------------------------------

video_manager = None  # Deprecated - ?

# API
@app.get("/")
async def root():
    """?"""
    return {
        "system": "RealityLoop Lab SOP Situational Awareness Platform",
        "version": "1.0.0",
        "status": "running",
        "active_streams": 0,  # Deprecated - 
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/streams/start")
async def start_video_stream(camera_id: str, video_source: str, request: Request):
    """[DEPRECATED] ?"""
    _require_operator_context_from_request(request)
    if not _LEGACY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Legacy monitoring module is unavailable; this API is deprecated")
    try:
        success = await video_manager.start_stream(camera_id, video_source)
        if success:
            return {"message": f"?{camera_id} ", "camera_id": camera_id}
        else:
            raise HTTPException(status_code=400, detail="")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/streams/stop")
async def stop_video_stream(camera_id: str, request: Request):
    """[DEPRECATED] ?"""
    _require_operator_context_from_request(request)
    if video_manager is None:
        raise HTTPException(status_code=503, detail="")
    await video_manager.stop_stream(camera_id)
    return {"message": f"Video stream {camera_id} stopped"}

@app.get("/api/v1/streams/status")
async def get_streams_status(request: Request):
    """[DEPRECATED] ?"""
    _require_operator_context_from_request(request)
    if video_manager is None:
        raise HTTPException(status_code=503, detail="")
    status = {}
    for camera_id in video_manager.active_streams:
        sop_engine = video_manager.sop_engines.get(camera_id)
        status[camera_id] = {
            "active": True,
            "recording": camera_id in video_manager.recorders,
            "compliance_status": sop_engine.build_status() if sop_engine else {}
        }
    return status

@app.get("/api/v1/streams/screenshot")
async def take_stream_screenshot(camera_id: str, request: Request):
    """[DEPRECATED] """
    _require_operator_context_from_request(request)
    if video_manager is None:
        raise HTTPException(status_code=503, detail="")
    try:
        path = video_manager.take_screenshot(camera_id)
        return {"camera_id": camera_id, "path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/streams/recording/start")
async def start_stream_recording(camera_id: str, request: Request):
    """[DEPRECATED] ?"""
    _require_operator_context_from_request(request)
    if video_manager is None:
        raise HTTPException(status_code=503, detail="")
    try:
        path = video_manager.start_recording(camera_id)
        return {"camera_id": camera_id, "path": path, "recording": True}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/v1/streams/recording/stop")
async def stop_stream_recording(camera_id: str, request: Request):
    """[DEPRECATED] """
    _require_operator_context_from_request(request)
    if video_manager is None:
        raise HTTPException(status_code=503, detail="")
    path = video_manager.stop_recording(camera_id)
    return {"camera_id": camera_id, "path": path, "recording": False}

@app.post("/api/v1/tasks/analyze")
async def create_analysis_task(request: TaskRequest, http_request: Request):
    """[DEPRECATED] """
    _require_operator_context_from_request(http_request)
    if not _LEGACY_AVAILABLE or celery_app is None:
        raise HTTPException(status_code=503, detail="Legacy monitoring module or Celery is unavailable; this API is deprecated")
    task_id = str(uuid.uuid4())

    task_info = {
        "task_id": task_id,
        "status": "pending",
        "video_path": request.video_path,
        "camera_id": request.camera_id,
        "created_at": datetime.now().isoformat()
    }
    _redis_hset(f"task:{task_id}", mapping=task_info)

    task = analyze_video_task.delay(
        task_id=task_id,
        video_path=request.video_path,
        sop_rules_path=request.sop_rules_path,
        camera_id=request.camera_id,
        callback_url=request.callback_url
    )
    _redis_hset(f"task:{task_id}", {"celery_task_id": task.id})

    current = _redis_hgetall(f"task:{task_id}")
    return {"task_id": task_id, "status": current.get("status", "pending")}

@app.get("/api/v1/tasks/{task_id}/status")
async def get_task_status(task_id: str, request: Request):
    """?"""
    _require_operator_context_from_request(request)
    task_info = _redis_hgetall(f"task:{task_id}")
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")

    return task_info

@app.get("/api/v1/tasks/{task_id}/violations")
async def get_task_violations(task_id: str, request: Request):
    """."""
    _require_operator_context_from_request(request)
    violations = _redis_lrange(f"violations:{task_id}", 0, -1)
    return [json.loads(v) for v in violations]

@app.get("/api/v1/alerts/recent")
async def get_recent_alerts(camera_id: str, request: Request, limit: int = 50):
    """?"""
    _require_operator_context_from_request(request)
    alerts = _redis_lrange(f"alerts:{camera_id}", 0, limit - 1)
    return [json.loads(a) for a in alerts]

@app.post("/api/v1/reports/generate")
async def generate_stream_report(camera_id: str, request: Request, limit: int = 200):
    """[DEPRECATED] """
    _require_operator_context_from_request(request)
    if not _LEGACY_AVAILABLE:
        raise HTTPException(status_code=503, detail="Legacy monitoring module is unavailable; this API is deprecated")
    alerts = _redis_lrange(f"alerts:{camera_id}", 0, limit - 1)
    violations: List[Dict[str, Any]] = []
    for item in alerts:
        try:
            data = json.loads(item)
            violations.append(
                {
                    "frame_id": int(data.get("frame_id", 0)),
                    "timestamp_sec": float(data.get("timestamp_sec", 0.0)),
                    "rule_id": str(data.get("rule_id", "unknown")),
                    "severity": str(data.get("severity", "Minor")),
                    "message": str(data.get("message", "")),
                    "screenshot_path": data.get("screenshot_url"),
                }
            )
        except Exception:
            continue

    task_id = f"stream_report_{uuid.uuid4().hex[:8]}"
    report_path = generate_pdf_report(task_id, violations, f"stream:{camera_id}")
    return {
        "task_id": task_id,
        "camera_id": camera_id,
        "violations_count": len(violations),
        "report_path": str(report_path),
    }

@app.get("/api/v1/alerts/export")
async def export_alerts(camera_id: str, request: Request, limit: int = 500):
    _require_operator_context_from_request(request)
    alerts = _redis_lrange(f"alerts:{camera_id}", 0, limit - 1)
    payload = [json.loads(a) for a in alerts]
    out_dir = PROJECT_ROOT / "outputs" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"alerts_{camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"camera_id": camera_id, "count": len(payload), "path": str(out_path)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket"""
    try:
        query = websocket.query_params
        query_token = str(query.get("token") or "").strip()
        query_authorization = str(query.get("authorization") or "").strip()
        if not query_authorization and query_token:
            query_authorization = f"Bearer {query_token}"
        auth_ctx = _auth_context_from_headers(
            authorization=websocket.headers.get("Authorization") or query_authorization or None,
            x_operator=websocket.headers.get("X-Operator") or query.get("operator"),
            x_operator_role=websocket.headers.get("X-Operator-Role") or query.get("operator_role"),
            x_allowed_experiments=websocket.headers.get("X-Allowed-Experiments") or query.get("allowed_experiments"),
            x_actor_scope=websocket.headers.get("X-Actor-Scope") or query.get("actor_scope"),
        )
        if _auth_required_enabled() and auth_ctx.get("auth_source") == "anonymous_default":
            await websocket.close(code=1008, reason="Authentication required")
            return
    except HTTPException as exc:
        await websocket.close(code=1008, reason=str(exc.detail))
        return

    client_id = str(uuid.uuid4())
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()

            # 
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            elif data.get("type") == "subscribe_alerts":
                # 
                camera_id = data.get("camera_id")
                if camera_id:
                    await websocket.send_json({
                        "type": "subscription_confirmed",
                        "camera_id": camera_id
                    })

    except WebSocketDisconnect:
        manager.disconnect(client_id)

# [DEPRECATED] Celery
if celery_app is not None:
    @celery_app.task(bind=True)
    def analyze_video_task(self, task_id: str, video_path: str, sop_rules_path: Optional[str],
                          camera_id: str, callback_url: Optional[str]):
        """[DEPRECATED] """
        if not _LEGACY_AVAILABLE:
            raise RuntimeError("")

        try:
            # ?
            _redis_hset(f"task:{task_id}", {"status": "running"})

            # 
            detector = MultiLevelDetector()

            # SOP
            if sop_rules_path and Path(sop_rules_path).exists():
                with open(sop_rules_path, 'r', encoding='utf-8') as f:
                    sop_rules = yaml.safe_load(f) or {}
            else:
                sop_rules_path = CONFIG_PATH / "sop" / "rules.yaml"
                with open(sop_rules_path, 'r', encoding='utf-8') as f:
                    sop_rules = yaml.safe_load(f) or {}

            sop_engine = SOPComplianceEngine(rules=sop_rules)

            # 
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f": {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = 1.0 / fps

            violations = []
            frame_id = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 
                frame_packet = FramePacket(
                    frame_id=frame_id,
                    timestamp_sec=frame_id * frame_interval,
                    frame_bgr=frame,
                    source=camera_id
                )

                # ?
                detection_event = detector.detect(frame_packet)

                # SOP?
                frame_violations = sop_engine.update(detection_event)

                # 
                for violation in frame_violations:
                    violation_data = {
                        "frame_id": violation.frame_id,
                        "timestamp_sec": violation.timestamp_sec,
                        "rule_id": violation.rule_id,
                        "severity": violation.severity,
                        "message": violation.message
                    }
                    violations.append(violation_data)
                    _redis_lpush(f"violations:{task_id}", json.dumps(violation_data))

                # 
                progress = (frame_id + 1) / total_frames * 100
                _redis_hset(f"task:{task_id}", {"progress": progress})
                _redis_hset(f"task:{task_id}", {"current_frame": frame_id})

                frame_id += 1

            cap.release()

            # PDF
            report_path = generate_pdf_report(task_id, violations, video_path)

            # ?
            _redis_hset(f"task:{task_id}", {"status": "completed"})
            _redis_hset(f"task:{task_id}", {"violations_count": len(violations)})
            _redis_hset(f"task:{task_id}", {"report_path": str(report_path)})
            _redis_hset(f"task:{task_id}", {"completed_at": datetime.now().isoformat()})

            # 
            if callback_url:
                send_callback_notification(callback_url, task_id, violations)

            return {"task_id": task_id, "violations_count": len(violations), "report_path": str(report_path)}

        except Exception as e:
            logger.error(f" {task_id} : {e}")
            _redis_hset(f"task:{task_id}", {"status": "failed"})
            _redis_hset(f"task:{task_id}", {"error": str(e)})
            raise
else:
    # Celery 
    def analyze_video_task(*args, **kwargs):
        raise RuntimeError("Celery is not installed; async task execution is disabled")

def generate_pdf_report(task_id: str, violations: List[Dict], video_path: str) -> Path:
    """[DEPRECATED] PDF"""
    if not _LEGACY_AVAILABLE:
        raise RuntimeError("")
    report_generator = PDFReportGenerator()

    # 
    report_data = {
        "task_id": task_id,
        "video_path": video_path,
        "violations": violations,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_violations": len(violations),
            "critical_count": len([v for v in violations if v["severity"] == "Critical"]),
            "major_count": len([v for v in violations if v["severity"] == "Major"]),
            "minor_count": len([v for v in violations if v["severity"] == "Minor"])
        }
    }

    # 
    output_path = PROJECT_ROOT / SYSTEM_CONFIG.pdf_output_dir / f"report_{task_id}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report_generator.generate_report(report_data, str(output_path))
    return output_path

def send_callback_notification(callback_url: str, task_id: str, violations: List[Dict]):
    """."""
    payload = {
        "task_id": task_id,
        "violations_count": len(violations),
        "status": "completed",
        "timestamp": datetime.now().isoformat()
    }

    try:
        safe_callback_url = _validate_callback_url(callback_url)
        requests.post(safe_callback_url, json=payload, timeout=10, allow_redirects=False)
    except ValueError as e:
        logger.warning(f"callback url blocked: {e}")
    except Exception as e:
        logger.warning(f"? {e}")

# ---------------------------------------------------------------------------
#  API
# ---------------------------------------------------------------------------

try:
    from experiment.models import (
        Experiment, ExperimentTimeline, ExperimentStatus, StepRecord,
        StepStatus, ContextEvent, MediaAsset, ProcessStage, MediaType,
        _now_iso, _uuid,
    )
    from experiment.service import ExperimentService
    from labsopguard import FileBackedTaskStore, FormalExperimentWorkflow, VideoAnalysisPipeline, load_runtime_settings
except ImportError as e:
    logger.warning(f"Experiment service not available: {e}")
    Experiment = None
    ExperimentService = None
    FileBackedTaskStore = None
    FormalExperimentWorkflow = None
    VideoAnalysisPipeline = None
    load_runtime_settings = None

# ?+ 
_EXPERIMENTS: Dict[str, Dict[str, Any]] = {}
FORMAL_WORKFLOW = FormalExperimentWorkflow() if FormalExperimentWorkflow is not None else None
RUNTIME_SETTINGS = load_runtime_settings(PROJECT_ROOT) if load_runtime_settings is not None else None
EXPERIMENT_TASK_STORE = (
    FileBackedTaskStore(PROJECT_ROOT / "outputs" / "experiments" / "tasks")
    if FileBackedTaskStore is not None else None
)

def _save_experiment(exp_dict: Dict[str, Any]) -> None:
    exp_id = _validate_experiment_id(str(exp_dict.get("experiment_id") or ""))
    exp_dict["experiment_id"] = exp_id
    _EXPERIMENTS[exp_id] = exp_dict
    out_dir = _experiment_output_dir(exp_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment.json").write_text(
        json.dumps(exp_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _load_experiments() -> List[Dict[str, Any]]:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments"
    loaded = []
    if exp_dir.exists():
        for exp_sub in exp_dir.iterdir():
            if exp_sub.is_dir():
                json_file = exp_sub / "experiment.json"
                if json_file.exists():
                    try:
                        loaded.append(json.loads(json_file.read_text(encoding="utf-8-sig")))
                    except Exception:
                        pass
    return loaded


def _experiment_output_dir(experiment_id: str) -> Path:
    exp_id = _validate_experiment_id(experiment_id)
    base_dir = (PROJECT_ROOT / "outputs" / "experiments").resolve()
    target = (base_dir / exp_id).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Experiment path escapes workspace root") from exc
    return target


def _create_friendly_experiment_alias(experiment_id: str, title: str) -> None:
    """Create a directory junction with human-readable Chinese name pointing to the UUID dir."""
    if not title or title == experiment_id:
        return
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title.strip())[:60]
    if not safe_title:
        return
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    alias_name = f"{safe_title}_{date_str}"
    alias_path = PROJECT_ROOT / "outputs" / "experiments" / alias_name
    target = _experiment_output_dir(experiment_id)
    if alias_path.exists() or not target.exists():
        return
    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["cmd", "/c", "mklink", "/J", str(alias_path), str(target)],
                           capture_output=True, timeout=5)
        else:
            alias_path.symlink_to(target)
        logger.info("Created experiment alias: %s -> %s", alias_name, experiment_id)
    except Exception:
        pass


def _normalize_experiment_dict(exp: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(exp)
    normalized.setdefault("status", "created")
    normalized.setdefault("video_asset_id", None)
    normalized.setdefault("analysis_job_id", None)
    normalized.setdefault("analyzed_at", None)
    normalized.setdefault("started_at", None)
    normalized.setdefault("completed_at", None)
    normalized.setdefault("video_paths", [])
    normalized.setdefault("video_metadata", [])
    normalized.setdefault("video_inputs", [])
    normalized.setdefault("context_inputs", [])
    normalized.setdefault("protocol_text", None)
    normalized.setdefault("timeline", None)
    normalized.setdefault("total_steps", 0)
    normalized.setdefault("inferred_steps", 0)
    normalized.setdefault("evidence_count", 0)
    normalized.setdefault("processing_stage", ProcessStage.INGESTION.value)
    normalized.setdefault("processing_error", None)
    normalized.setdefault("output_paths", {})
    if "avg_confidence" not in normalized:
        normalized["avg_confidence"] = None
    return normalized


def _load_json_if_exists(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return None


_EXPERIMENT_DETAIL_CACHE_CONTROL = "private, max-age=15, stale-while-revalidate=60"
_EXPERIMENT_DETAIL_CACHE_TTL_SEC = 60.0
_EXPERIMENT_DETAIL_CACHE_MAX_ENTRIES = 96
_EXPERIMENT_DETAIL_VARY = "Authorization, X-Operator-Role, X-Allowed-Experiment-Ids, X-Actor-Name"
_EXPERIMENT_DETAIL_RESPONSE_CACHE: Dict[str, Dict[str, Any]] = {}


def _json_response_body(content: Any) -> bytes:
    return json.dumps(
        jsonable_encoder(content),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _etag_for_body(body: bytes) -> str:
    return f"\"{hashlib.sha256(body).hexdigest()[:24]}\""


def _etag_matches(if_none_match: Optional[str], etag: str) -> bool:
    if not if_none_match:
        return False
    candidates = [item.strip() for item in if_none_match.split(",") if item.strip()]
    if "*" in candidates:
        return True
    bare_etag = etag[2:] if etag.startswith("W/") else etag
    for candidate in candidates:
        bare_candidate = candidate[2:] if candidate.startswith("W/") else candidate
        if bare_candidate == bare_etag:
            return True
    return False


def _experiment_json_cache_headers(etag: str) -> Dict[str, str]:
    return {
        "Cache-Control": _EXPERIMENT_DETAIL_CACHE_CONTROL,
        "ETag": etag,
        "Vary": _EXPERIMENT_DETAIL_VARY,
    }


def _path_cache_token(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    try:
        stat_result = path.stat()
    except OSError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "is_dir": path.is_dir(),
        "mtime_ns": stat_result.st_mtime_ns,
        "size": stat_result.st_size,
    }


def _directory_tree_cache_token(path: Path, *, max_entries: int = 512) -> Dict[str, Any]:
    root_token = _path_cache_token(path)
    if not root_token.get("exists") or not root_token.get("is_dir"):
        return root_token
    entries: List[Any] = []
    truncated = False
    try:
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            try:
                stat_result = child.stat()
            except OSError:
                continue
            entries.append((str(child.relative_to(path)), stat_result.st_mtime_ns, stat_result.st_size))
    except OSError:
        truncated = True
    return {
        **root_token,
        "file_count_sampled": len(entries),
        "truncated": truncated,
        "fingerprint": hashlib.sha1(json.dumps(entries, separators=(",", ":")).encode("utf-8")).hexdigest(),
    }


def _experiment_state_cache_token(exp: Dict[str, Any]) -> str:
    return hashlib.sha1(
        _json_response_body(
            {
                "experiment_id": exp.get("experiment_id"),
                "status": exp.get("status"),
                "processing_stage": exp.get("processing_stage"),
                "analysis_job_id": exp.get("analysis_job_id"),
                "analyzed_at": exp.get("analyzed_at"),
                "completed_at": exp.get("completed_at"),
                "updated_at": exp.get("updated_at"),
                "output_paths": exp.get("output_paths") or {},
                "context_inputs": exp.get("context_inputs") or [],
                "video_inputs": exp.get("video_inputs") or [],
                "video_paths": exp.get("video_paths") or [],
            }
        )
    ).hexdigest()


def _experiment_detail_cache_signature(parts: Any) -> str:
    return hashlib.sha1(_json_response_body(parts)).hexdigest()


def _trim_experiment_detail_response_cache() -> None:
    if len(_EXPERIMENT_DETAIL_RESPONSE_CACHE) <= _EXPERIMENT_DETAIL_CACHE_MAX_ENTRIES:
        return
    oldest_keys = sorted(
        _EXPERIMENT_DETAIL_RESPONSE_CACHE,
        key=lambda key: float(_EXPERIMENT_DETAIL_RESPONSE_CACHE[key].get("created_at", 0.0)),
    )
    for key in oldest_keys[: max(1, len(oldest_keys) - _EXPERIMENT_DETAIL_CACHE_MAX_ENTRIES)]:
        _EXPERIMENT_DETAIL_RESPONSE_CACHE.pop(key, None)


def _cached_experiment_json_response(
    request: Request,
    *,
    cache_key: str,
    signature: str,
    build_payload: Callable[[], Any],
) -> Response:
    now = time.time()
    entry = _EXPERIMENT_DETAIL_RESPONSE_CACHE.get(cache_key)
    if (
        entry
        and entry.get("signature") == signature
        and now - float(entry.get("created_at", 0.0)) <= _EXPERIMENT_DETAIL_CACHE_TTL_SEC
    ):
        etag = str(entry["etag"])
        headers = _experiment_json_cache_headers(etag)
        if _etag_matches(request.headers.get("if-none-match"), etag):
            return Response(status_code=304, headers=headers)
        return Response(content=entry["body"], media_type="application/json", headers=headers)

    body = _json_response_body(build_payload())
    etag = _etag_for_body(body)
    _EXPERIMENT_DETAIL_RESPONSE_CACHE[cache_key] = {
        "signature": signature,
        "etag": etag,
        "body": body,
        "created_at": now,
    }
    _trim_experiment_detail_response_cache()
    headers = _experiment_json_cache_headers(etag)
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return Response(content=body, media_type="application/json", headers=headers)


def _safe_artifact_path(path_value: Optional[str], default_path: Path) -> Path:
    try:
        candidate = _safe_project_path(path_value, default_path)
    except HTTPException:
        return default_path
    return candidate if candidate.exists() else default_path


def _first_existing_project_path(candidates: List[Any], default_path: Path) -> Optional[Path]:
    for candidate in candidates:
        if not candidate:
            continue
        try:
            path = _safe_project_path(str(candidate), default_path)
        except HTTPException:
            continue
        if path.exists() and path.is_file():
            return path
    return default_path if default_path.exists() and default_path.is_file() else None


def _experiment_source_video_candidates(
    experiment_dir: Path,
    exp: Dict[str, Any],
    output_paths: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    output_paths = output_paths or (exp.get("output_paths") or {})
    source_candidates: List[Any] = []
    source_candidates.append(output_paths.get("source_video"))
    source_candidates.extend(output_paths.get("source_videos") or [])
    source_candidates.extend(exp.get("video_paths") or [])
    for item in exp.get("video_inputs") or []:
        if isinstance(item, dict):
            source_candidates.extend([item.get("video_path"), item.get("source")])
    for item in exp.get("video_assets") or []:
        if isinstance(item, dict):
            source_candidates.append(item.get("file_path"))
    source_candidates.extend(_experiment_upload_video_candidates(experiment_dir))
    return source_candidates


def _raise_if_source_video_candidates_escape_project(
    candidates: List[Any],
    detail: str = "Video path must stay inside project root",
) -> None:
    for candidate in candidates:
        if not candidate:
            continue
        try:
            _safe_project_path(str(candidate), PROJECT_ROOT)
        except HTTPException as exc:
            raise HTTPException(status_code=403, detail=detail) from exc


def _experiment_upload_video_candidates(experiment_dir: Path) -> List[Path]:
    upload_dir = experiment_dir / "uploads"
    if not upload_dir.exists():
        return []
    videos = sorted(upload_dir.glob("*.mp4"))
    preferred = [path for path in videos if path.stem.endswith(".browser_h264") and ".browser_h264.browser_h264" not in path.stem]
    preferred += [path for path in videos if ".browser_h264" not in path.stem]
    preferred += videos
    seen = set()
    result: List[Path] = []
    for path in preferred:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _experiment_output_artifact_paths(experiment_id: str, exp: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[Path]]:
    experiment_dir = _experiment_output_dir(experiment_id)
    if exp is None:
        exp = _load_json_if_exists(experiment_dir / "experiment.json") or {}
    output_paths = exp.get("output_paths") or {} if isinstance(exp, dict) else {}
    source_candidates: List[Any] = []
    if isinstance(exp, dict):
        source_candidates = _experiment_source_video_candidates(experiment_dir, exp, output_paths)
    source_video = _first_existing_project_path(source_candidates, experiment_dir / "source.mp4")

    annotated_candidates = [
        output_paths.get("annotated_video"),
        experiment_dir / "analysis" / "annotated.browser_h264.mp4",
        experiment_dir / "analysis" / "annotated.mp4",
    ]
    annotated_video = _first_existing_project_path(annotated_candidates, experiment_dir / "analysis" / "annotated.mp4")
    return {
        "experiment_json": _safe_artifact_path(output_paths.get("experiment_json"), experiment_dir / "experiment.json"),
        "timeline_json": _safe_artifact_path(output_paths.get("timeline_json"), experiment_dir / "timeline.json"),
        "steps_json": _safe_artifact_path(output_paths.get("steps_json"), experiment_dir / "steps.json"),
        "step_candidates_json": _safe_artifact_path(output_paths.get("step_candidates_json"), experiment_dir / "step_candidates.json"),
        "step_bridge_summary_json": _safe_artifact_path(output_paths.get("step_bridge_summary_json"), experiment_dir / "step_bridge_summary.json"),
        "official_steps_json": _safe_artifact_path(output_paths.get("official_steps_json"), experiment_dir / "official_steps.json"),
        "step_review_log_json": _safe_artifact_path(output_paths.get("step_review_log_json"), experiment_dir / "step_review_log.json"),
        "sop_schema_validation_json": _safe_artifact_path(output_paths.get("sop_schema_validation_json"), experiment_dir / "sop_schema_validation.json"),
        "physical_events_json": _safe_artifact_path(output_paths.get("physical_events_json"), experiment_dir / "physical_events.json"),
        "material_stream_json": _safe_artifact_path(output_paths.get("material_stream_json"), experiment_dir / "material_stream.json"),
        "material_stream_v2_jsonl": _safe_artifact_path(output_paths.get("material_stream_v2_jsonl"), experiment_dir / "material_stream.v2.jsonl"),
        "preprocessing_json": _safe_artifact_path(output_paths.get("preprocessing_json"), experiment_dir / "preprocessing.json"),
        "material_index": _safe_artifact_path(output_paths.get("material_index"), experiment_dir / "material_index.sqlite"),
        "published_materials_json": _safe_artifact_path(output_paths.get("published_materials_json"), experiment_dir / "published_materials.json"),
        "upload_manifest_json": _safe_artifact_path(output_paths.get("upload_manifest_json"), experiment_dir / "upload_manifest.json"),
        "experiment_run_manifest_json": _safe_artifact_path(output_paths.get("experiment_run_manifest_json"), experiment_dir / "experiment_run_manifest.json"),
        "stream_manifest_json": _safe_artifact_path(output_paths.get("stream_manifest_json"), experiment_dir / "stream_manifest.json"),
        "timeline_alignment_json": _safe_artifact_path(output_paths.get("timeline_alignment_json"), experiment_dir / "timeline_alignment.json"),
        "semantic_sync_anchors_json": _safe_artifact_path(
            output_paths.get("semantic_sync_anchors_json") or output_paths.get("semantic_sync_anchors"),
            experiment_dir / "semantic_sync_anchors.json",
        ),
        "transcript_segments_jsonl": _safe_artifact_path(output_paths.get("transcript_segments_jsonl"), experiment_dir / "transcript_segments.jsonl"),
        "structured_json": _safe_artifact_path(output_paths.get("structured_json"), experiment_dir / "structured.json"),
        "analysis_json": _safe_artifact_path(output_paths.get("analysis_json"), experiment_dir / "analysis" / "analysis.json"),
        "professional_report_pdf": _safe_artifact_path(
            output_paths.get("professional_report_pdf"),
            experiment_dir / "reports" / f"{_professional_report_basename()}.pdf",
        ),
        "professional_report_html": _safe_artifact_path(
            output_paths.get("professional_report_html"),
            experiment_dir / "reports" / f"{_professional_report_basename()}.html",
        ),
        "professional_report_json": _safe_artifact_path(
            output_paths.get("professional_report_json"),
            experiment_dir / "reports" / f"{_professional_report_basename()}.json",
        ),
        "professional_report_manifest_json": _safe_artifact_path(
            output_paths.get("professional_report_manifest_json"),
            experiment_dir / "reports" / "professional_report_manifest.json",
        ),
        "annotated_video": annotated_video,
        "source_video": source_video,
    }


def _split_csv_filter(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return parts or None


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode("ascii"))


_OIDC_DISCOVERY_CACHE: Dict[str, Dict[str, Any]] = {}
_JWKS_CACHE: Dict[str, Dict[str, Any]] = {}


def _auth_cache_dir() -> Path:
    raw = os.environ.get("REALITYLOOP_AUTH_CACHE_DIR")
    path = Path(raw) if raw else PROJECT_ROOT / "outputs" / "auth_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _auth_cache_path(prefix: str, key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return _auth_cache_dir() / f"{prefix}_{digest}.json"


def _read_auth_disk_cache(prefix: str, key: str) -> Optional[Dict[str, Any]]:
    path = _auth_cache_path(prefix, key)
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if float(cached.get("expires_at") or 0) <= time.time():
        return None
    payload = cached.get("payload")
    return payload if isinstance(payload, dict) else None


def _auth_redis_key(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    return f"realityloop:auth_cache:{prefix}:{digest}"


def _read_auth_shared_cache(prefix: str, key: str) -> Optional[Dict[str, Any]]:
    if not globals().get("_USE_REDIS") or globals().get("redis_client") is None:
        return None
    try:
        raw = redis_client.get(_auth_redis_key(prefix, key))
        if not raw:
            return None
        cached = json.loads(raw)
        payload = cached.get("payload")
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _write_auth_shared_cache(prefix: str, key: str, payload: Dict[str, Any], ttl_sec: int = 3600) -> None:
    if not globals().get("_USE_REDIS") or globals().get("redis_client") is None:
        return
    try:
        redis_client.setex(
            _auth_redis_key(prefix, key),
            ttl_sec,
            json.dumps(
                {
                    "schema_version": "auth_distributed_cache.v1",
                    "cache_key": key,
                    "payload": payload,
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        return


def _write_auth_disk_cache(prefix: str, key: str, payload: Dict[str, Any], ttl_sec: int = 3600) -> None:
    path = _auth_cache_path(prefix, key)
    cache_payload = {
        "schema_version": "auth_discovery_cache.v1",
        "cache_key": key,
        "cache_prefix": prefix,
        "key_hash": hashlib.sha1(key.encode("utf-8")).hexdigest()[:24],
        "expires_at": time.time() + ttl_sec,
        "payload": payload,
    }
    path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _b64url_uint(value: str) -> int:
    return int.from_bytes(_b64url_decode(value), "big")


def _oidc_discovery(issuer_url: str) -> Dict[str, Any]:
    issuer = issuer_url.rstrip("/")
    cached = _OIDC_DISCOVERY_CACHE.get(issuer)
    if cached and cached.get("expires_at", 0) > time.time():
        return cached["payload"]
    shared_cached = _read_auth_shared_cache("oidc", issuer)
    if shared_cached:
        _OIDC_DISCOVERY_CACHE[issuer] = {"payload": shared_cached, "expires_at": time.time() + 3600}
        _write_auth_disk_cache("oidc", issuer, shared_cached)
        return shared_cached
    disk_cached = _read_auth_disk_cache("oidc", issuer)
    if disk_cached:
        _OIDC_DISCOVERY_CACHE[issuer] = {"payload": disk_cached, "expires_at": time.time() + 3600}
        _write_auth_shared_cache("oidc", issuer, disk_cached)
        return disk_cached
    url = f"{issuer}/.well-known/openid-configuration"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"OIDC discovery failed: {exc}") from exc
    _OIDC_DISCOVERY_CACHE[issuer] = {"payload": payload, "expires_at": time.time() + 3600}
    _write_auth_shared_cache("oidc", issuer, payload)
    _write_auth_disk_cache("oidc", issuer, payload)
    return payload


def _jwks_payload() -> Dict[str, Any]:
    jwks_url = os.environ.get("REALITYLOOP_JWKS_URL")
    issuer = os.environ.get("REALITYLOOP_OAUTH_ISSUER_URL") or os.environ.get("REALITYLOOP_JWT_ISSUER")
    if not jwks_url and issuer:
        jwks_url = _oidc_discovery(issuer).get("jwks_uri")
    if not jwks_url:
        raise HTTPException(status_code=401, detail="JWKS URL or OAuth issuer is not configured")
    cached = _JWKS_CACHE.get(jwks_url)
    if cached and cached.get("expires_at", 0) > time.time():
        return cached["payload"]
    shared_cached = _read_auth_shared_cache("jwks", jwks_url)
    if shared_cached:
        _JWKS_CACHE[jwks_url] = {"payload": shared_cached, "expires_at": time.time() + 3600}
        _write_auth_disk_cache("jwks", jwks_url, shared_cached)
        return shared_cached
    disk_cached = _read_auth_disk_cache("jwks", jwks_url)
    if disk_cached:
        _JWKS_CACHE[jwks_url] = {"payload": disk_cached, "expires_at": time.time() + 3600}
        _write_auth_shared_cache("jwks", jwks_url, disk_cached)
        return disk_cached
    try:
        response = requests.get(jwks_url, timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise HTTPException(status_code=401, detail=f"JWKS fetch failed: {exc}") from exc
    _JWKS_CACHE[jwks_url] = {"payload": payload, "expires_at": time.time() + 3600}
    _write_auth_shared_cache("jwks", jwks_url, payload)
    _write_auth_disk_cache("jwks", jwks_url, payload)
    return payload


def _public_key_from_x5c(jwk: Dict[str, Any]):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        from cryptography.hazmat.primitives import hashes
    except Exception as exc:
        raise HTTPException(status_code=401, detail="cryptography is required for x5c JWT verification") from exc
    chain_values = jwk.get("x5c") or []
    if not chain_values:
        return None
    try:
        certs = [x509.load_der_x509_certificate(base64.b64decode(value)) for value in chain_values]
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid x5c certificate chain") from exc
    now = datetime.now(timezone.utc)
    for cert in certs:
        not_before = getattr(cert, "not_valid_before_utc", None) or cert.not_valid_before.replace(tzinfo=timezone.utc)
        not_after = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(tzinfo=timezone.utc)
        if now < not_before or now > not_after:
            raise HTTPException(status_code=401, detail="x5c certificate is not currently valid")
    for idx in range(len(certs) - 1):
        child = certs[idx]
        issuer = certs[idx + 1]
        issuer_public_key = issuer.public_key()
        if not isinstance(issuer_public_key, rsa.RSAPublicKey):
            raise HTTPException(status_code=401, detail="x5c issuer key is not RSA")
        try:
            issuer_public_key.verify(
                child.signature,
                child.tbs_certificate_bytes,
                padding.PKCS1v15(),
                child.signature_hash_algorithm,
            )
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Invalid x5c certificate chain signature") from exc
    trusted_fingerprints = [
        item.replace(":", "").lower()
        for item in (os.environ.get("REALITYLOOP_JWKS_X5C_TRUSTED_FINGERPRINTS") or "").split(",")
        if item.strip()
    ]
    if trusted_fingerprints:
        root_fp = certs[-1].fingerprint(hashes.SHA256()).hex().lower()
        if root_fp not in trusted_fingerprints:
            raise HTTPException(status_code=401, detail="x5c root fingerprint is not trusted")
    elif os.environ.get("REALITYLOOP_JWKS_X5C_SYSTEM_TRUST", "true").lower() not in {"0", "false", "no"}:
        _verify_x5c_against_system_trust(certs)
    public_key = certs[0].public_key()
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise HTTPException(status_code=401, detail="x5c leaf key is not RSA")
    return public_key


def _load_system_ca_certificates() -> List[Any]:
    try:
        from cryptography import x509
    except Exception as exc:
        raise HTTPException(status_code=401, detail="cryptography is required for system CA verification") from exc
    candidates: List[Path] = []
    verify_paths = ssl.get_default_verify_paths()
    for value in (verify_paths.cafile, os.environ.get("REQUESTS_CA_BUNDLE"), os.environ.get("SSL_CERT_FILE")):
        if value:
            candidates.append(Path(value))
    try:
        import certifi

        candidates.append(Path(certifi.where()))
    except Exception:
        pass
    certs = []
    seen: set[str] = set()
    for path in candidates:
        if not path.exists() or str(path) in seen:
            continue
        seen.add(str(path))
        try:
            pem_data = path.read_bytes()
        except Exception:
            continue
        for block in pem_data.split(b"-----END CERTIFICATE-----"):
            if b"-----BEGIN CERTIFICATE-----" not in block:
                continue
            try:
                certs.append(x509.load_pem_x509_certificate(block + b"-----END CERTIFICATE-----\n"))
            except Exception:
                continue
    return certs


def _verify_x5c_against_system_trust(certs: List[Any]) -> None:
    try:
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except Exception as exc:
        raise HTTPException(status_code=401, detail="cryptography is required for system CA verification") from exc
    if not certs:
        raise HTTPException(status_code=401, detail="x5c certificate chain is empty")
    candidate = certs[-1]
    trust_anchors = _load_system_ca_certificates()
    for anchor in trust_anchors:
        if anchor.subject != candidate.issuer:
            continue
        public_key = anchor.public_key()
        if not isinstance(public_key, rsa.RSAPublicKey):
            continue
        try:
            public_key.verify(
                candidate.signature,
                candidate.tbs_certificate_bytes,
                padding.PKCS1v15(),
                candidate.signature_hash_algorithm,
            )
            return
        except Exception:
            continue
    raise HTTPException(status_code=401, detail="x5c chain is not anchored in system CA trust store")


def _refresh_auth_cache_entry(path: Path) -> None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        prefix = cached.get("cache_prefix")
        key = cached.get("cache_key")
    except Exception:
        return
    if prefix not in {"oidc", "jwks"} or not key:
        return
    try:
        if prefix == "oidc":
            response = requests.get(f"{str(key).rstrip('/')}/.well-known/openid-configuration", timeout=5)
        else:
            response = requests.get(str(key), timeout=5)
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return
    if prefix == "oidc":
        _OIDC_DISCOVERY_CACHE[str(key).rstrip("/")] = {"payload": payload, "expires_at": time.time() + 3600}
    else:
        _JWKS_CACHE[str(key)] = {"payload": payload, "expires_at": time.time() + 3600}
    _write_auth_shared_cache(prefix, str(key), payload)
    _write_auth_disk_cache(prefix, str(key), payload)


async def _auth_cache_refresh_loop() -> None:
    interval = max(60, int(os.environ.get("REALITYLOOP_AUTH_CACHE_REFRESH_INTERVAL_SEC", "900")))
    while True:
        try:
            for path in _auth_cache_dir().glob("*.json"):
                try:
                    cached = json.loads(path.read_text(encoding="utf-8"))
                    expires_at = float(cached.get("expires_at") or 0)
                except Exception:
                    continue
                if expires_at - time.time() <= interval:
                    await asyncio.to_thread(_refresh_auth_cache_entry, path)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Auth cache background refresh failed")
        await asyncio.sleep(interval)


def _verify_rs256_jwt(header: Dict[str, Any], parts: List[str]) -> None:
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except Exception as exc:
        raise HTTPException(status_code=401, detail="cryptography is required for RS256 JWT verification") from exc
    kid = header.get("kid")
    keys = _jwks_payload().get("keys") or []
    jwk = next((key for key in keys if (not kid or key.get("kid") == kid) and key.get("kty") == "RSA"), None)
    if not jwk:
        raise HTTPException(status_code=401, detail="JWT signing key not found in JWKS")
    try:
        if jwk.get("x5c"):
            public_key = _public_key_from_x5c(jwk)
        else:
            public_key = rsa.RSAPublicNumbers(_b64url_uint(jwk["e"]), _b64url_uint(jwk["n"])).public_key()
        public_key.verify(
            _b64url_decode(parts[2]),
            f"{parts[0]}.{parts[1]}".encode("ascii"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid RS256 JWT signature") from exc


def _jwt_claims_from_authorization(authorization: Optional[str]) -> Optional[Dict[str, Any]]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Authorization must be a Bearer token")
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(status_code=401, detail="Malformed JWT")
    try:
        header = json.loads(_b64url_decode(parts[0]).decode("utf-8"))
        claims = json.loads(_b64url_decode(parts[1]).decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Malformed JWT payload") from exc
    alg = str(header.get("alg") or "")
    if alg not in {"HS256", "RS256"}:
        raise HTTPException(status_code=401, detail="Unsupported JWT algorithm")
    if alg == "HS256":
        secret = os.environ.get("REALITYLOOP_JWT_SECRET") or os.environ.get("JWT_SECRET")
        if not secret:
            raise HTTPException(status_code=401, detail="JWT secret is not configured")
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
        expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        try:
            signature = _b64url_decode(parts[2])
        except Exception as exc:
            raise HTTPException(status_code=401, detail="Malformed JWT signature") from exc
        if not hmac.compare_digest(signature, expected):
            raise HTTPException(status_code=401, detail="Invalid JWT signature")
    else:
        _verify_rs256_jwt(header, parts)
    now = int(time.time())
    if claims.get("exp") is not None and int(claims["exp"]) < now:
        raise HTTPException(status_code=401, detail="JWT expired")
    expected_issuer = os.environ.get("REALITYLOOP_JWT_ISSUER")
    if expected_issuer and claims.get("iss") != expected_issuer:
        raise HTTPException(status_code=401, detail="Invalid JWT issuer")
    expected_audience = os.environ.get("REALITYLOOP_JWT_AUDIENCE")
    if expected_audience:
        aud = claims.get("aud")
        audiences = aud if isinstance(aud, list) else [aud]
        if expected_audience not in audiences:
            raise HTTPException(status_code=401, detail="Invalid JWT audience")
    return claims


def _claims_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return _split_csv_filter(value) or []
    return []


def _oauth_scope_context(scope: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    tokens = str(scope or "").split()
    experiments: List[str] = []
    for token in tokens:
        if token.startswith("role:"):
            result["operator_role"] = token.split(":", 1)[1]
        elif token.startswith("experiment:"):
            experiments.append(token.split(":", 1)[1])
        elif token.startswith("actor:"):
            result["actor_scope"] = token.split(":", 1)[1]
    if experiments:
        result["allowed_experiment_ids"] = experiments
    return result


def _auth_context_from_headers(
    *,
    authorization: Optional[str] = None,
    x_operator: Optional[str],
    x_operator_role: Optional[str],
    x_allowed_experiments: Optional[str],
    x_actor_scope: Optional[str],
) -> Dict[str, Any]:
    claims = _jwt_claims_from_authorization(authorization)
    if claims is not None:
        scoped = _oauth_scope_context(claims.get("scope") or claims.get("scp"))
        role = str(
            claims.get("operator_role")
            or claims.get("role")
            or scoped.get("operator_role")
            or "reviewer"
        ).strip().lower()
        if role not in {"reviewer", "approver", "admin", "system"}:
            role = "reviewer"
        allowed = (
            _claims_list(claims.get("allowed_experiment_ids"))
            or _claims_list(claims.get("allowed_experiments"))
            or scoped.get("allowed_experiment_ids")
            or []
        )
        return {
            "operator": claims.get("operator") or claims.get("preferred_username") or claims.get("sub") or "jwt_user",
            "operator_role": role,
            "allowed_experiment_ids": allowed,
            "actor_scope": claims.get("actor_scope") or scoped.get("actor_scope"),
            "auth_source": "jwt",
            "jwt_subject": claims.get("sub"),
        }
    role = (x_operator_role or "reviewer").strip().lower()
    if role not in {"reviewer", "approver", "admin", "system"}:
        role = "reviewer"
    return {
        "operator": x_operator or "anonymous",
        "operator_role": role,
        "allowed_experiment_ids": _split_csv_filter(x_allowed_experiments) or [],
        "actor_scope": x_actor_scope or None,
        "auth_source": "headers" if any([x_operator, x_operator_role, x_allowed_experiments, x_actor_scope]) else "anonymous_default",
    }


def _auth_required_enabled() -> bool:
    return str(os.environ.get("REALITYLOOP_AUTH_REQUIRED", "false")).strip().lower() in {"1", "true", "yes", "on"}


def _require_operator_context(
    x_operator: Optional[str] = Header(None, alias="X-Operator"),
    x_operator_role: Optional[str] = Header(None, alias="X-Operator-Role"),
    x_allowed_experiments: Optional[str] = Header(None, alias="X-Allowed-Experiments"),
    x_actor_scope: Optional[str] = Header(None, alias="X-Actor-Scope"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> Dict[str, Any]:
    auth_ctx = _auth_context_from_headers(
        authorization=authorization,
        x_operator=x_operator,
        x_operator_role=x_operator_role,
        x_allowed_experiments=x_allowed_experiments,
        x_actor_scope=x_actor_scope,
    )
    if _auth_required_enabled() and auth_ctx.get("auth_source") == "anonymous_default":
        raise HTTPException(status_code=401, detail="Authentication required")
    return auth_ctx


def _enforce_experiment_scope(auth_ctx: Dict[str, Any], experiment_id: str) -> str:
    safe_experiment_id = _validate_experiment_id(experiment_id)
    operator_role = str(auth_ctx.get("operator_role") or "").strip().lower()
    if operator_role in {"admin", "system"}:
        return safe_experiment_id
    allowed = {str(item).strip() for item in (auth_ctx.get("allowed_experiment_ids") or []) if str(item).strip()}
    if "*" in allowed:
        return safe_experiment_id
    if allowed and safe_experiment_id not in allowed:
        raise HTTPException(status_code=403, detail="Operator is not allowed to access this experiment")
    return safe_experiment_id


def _scope_filter_experiments(auth_ctx: Dict[str, Any], experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    operator_role = str(auth_ctx.get("operator_role") or "").strip().lower()
    if operator_role in {"admin", "system"}:
        return experiments
    allowed = {str(item).strip() for item in (auth_ctx.get("allowed_experiment_ids") or []) if str(item).strip()}
    if not allowed or "*" in allowed:
        return experiments
    return [item for item in experiments if str(item.get("experiment_id") or "") in allowed]


def _ensure_material_index(experiment_id: str, artifacts: Dict[str, Optional[Path]]) -> Path:
    index_path = artifacts["material_index"]
    material_stream_path = artifacts["material_stream_json"]
    preprocessing_path = artifacts["preprocessing_json"]
    if index_path is None:
        raise HTTPException(status_code=404, detail="Material index path is unavailable")

    # Rebuild when: index file is missing; index is stale vs. material_stream.json /
    # preprocessing.json / materials/events; or index is empty. This guarantees the
    # search endpoint reflects the latest EventPreprocessingEngine run.
    def _needs_rebuild() -> bool:
        if not index_path.exists():
            return True
        try:
            index_mtime = index_path.stat().st_mtime
        except OSError:
            return True
        for candidate in (material_stream_path, preprocessing_path):
            if candidate is not None and candidate.exists():
                try:
                    if candidate.stat().st_mtime > index_mtime + 0.5:
                        return True
                except OSError:
                    pass
        events_dir = index_path.parent / "materials" / "events"
        if events_dir.exists():
            try:
                latest = max((child.stat().st_mtime for child in events_dir.iterdir()), default=0.0)
                if latest > index_mtime + 0.5:
                    return True
            except OSError:
                pass
        try:
            import sqlite3
            conn = sqlite3.connect(str(index_path))
            try:
                cur = conn.execute("SELECT COUNT(*) FROM material_items")
                if int(cur.fetchone()[0] or 0) == 0:
                    return True
            finally:
                conn.close()
        except Exception:
            return True
        return False

    if not _needs_rebuild():
        return index_path
    if material_stream_path is None or not material_stream_path.exists():
        if index_path.exists():
            return index_path
        raise HTTPException(status_code=404, detail="Material stream is not ready")

    from labsopguard.retrieval import MaterialRetrievalIndex

    material_stream = _load_json_if_exists(material_stream_path) or []
    preprocessing = _load_json_if_exists(preprocessing_path) if preprocessing_path is not None else None
    index = MaterialRetrievalIndex(index_path)
    try:
        index.reset()
        index.index_payloads(material_stream, preprocessing=preprocessing or {}, experiment_id=experiment_id)
    finally:
        index.close()
    return index_path


def _workspace_material_index_path() -> Path:
    path = PROJECT_ROOT / "outputs" / "materials" / "material_index.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _workspace_published_materials_index_path() -> Path:
    path = PROJECT_ROOT / "outputs" / "materials" / "published_materials_index.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _safe_project_path(path_value: Optional[str], default_path: Path) -> Path:
    if not path_value:
        return default_path
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if PROJECT_ROOT.resolve() not in resolved.parents and resolved != PROJECT_ROOT.resolve():
        raise HTTPException(status_code=400, detail="Path must stay inside project root")
    return resolved


_MATERIAL_LONG_CLIP_THRESHOLD_SEC = 30.0
_PLAYABLE_CLIP_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v"}


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _material_delivery_safe_name(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*\s]+', "_", value).strip("._") or "material"


def _material_delivery_date_label(value: str) -> str:
    if value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%d")
        except ValueError:
            pass
        match = re.search(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", value)
        if match:
            return "".join(match.groups())
    return datetime.now().strftime("%Y%m%d")


def _formal_material_reference_root_for_exp(exp_dir: Path) -> Path:
    local_root = exp_dir / "material_references"
    for meta_path in (local_root / "manifest.json", local_root / "素材索引.json"):
        payload = _load_json_if_exists(meta_path)
        if not isinstance(payload, dict):
            continue
        candidate = payload.get("formal_material_references") or payload.get("simplified_material_references")
        if candidate:
            return Path(str(candidate))

    exp = _load_json_if_exists(exp_dir / "experiment.json") or {}
    title = str(
        exp.get("title")
        or exp.get("experiment_title")
        or exp.get("experiment_name")
        or exp.get("name")
        or exp_dir.name
    )
    date = _material_delivery_date_label(str(exp.get("created_at") or exp.get("experiment_date") or exp.get("date") or exp_dir.name))
    label = _material_delivery_safe_name(f"{title}_{date}")
    outputs_dir = exp_dir.parent.parent if exp_dir.parent.name == "experiments" else exp_dir.parent
    return outputs_dir / "material_references" / label


def _material_reference_root_candidates(exp_dir: Path) -> List[Path]:
    formal_root = _formal_material_reference_root_for_exp(exp_dir)
    local_root = exp_dir / "material_references"
    roots: List[Path] = []
    for root in (formal_root, local_root):
        if root not in roots:
            roots.append(root)
    return roots


def _material_reference_index_exists(exp_dir: Path) -> bool:
    for ref_root in _material_reference_root_candidates(exp_dir):
        if (ref_root / "素材索引.jsonl").exists() or (ref_root / "素材索引.json").exists():
            return True
    return False


def _material_reference_root_and_rows(exp_dir: Path) -> tuple[Path, List[Dict[str, Any]]]:
    fallback_root = exp_dir / "material_references"
    local_root = exp_dir / "material_references"
    for ref_root in _material_reference_root_candidates(exp_dir):
        rows = _material_reference_rows_from_root(ref_root)
        if not rows:
            continue
        if ref_root.resolve() == local_root.resolve():
            approved_rows = []
            for row in rows:
                candidate_status = str(row.get("candidate_status") or "").lower()
                review_status = str(row.get("review_status") or "").lower()
                has_approval_trace = bool(row.get("approved_at") or row.get("approved_by"))
                if (
                    row.get("formal_material_reference")
                    or candidate_status == "approved"
                    or (review_status == "accepted" and has_approval_trace)
                ):
                    approved_rows.append(row)
            if approved_rows:
                return ref_root, approved_rows
            continue
        return ref_root, rows
    return fallback_root, []


def _material_reference_rows_from_root(ref_root: Path) -> List[Dict[str, Any]]:
    index_jsonl = ref_root / "素材索引.jsonl"
    rows = _read_jsonl_rows(index_jsonl)
    if rows:
        return rows
    payload = _load_json_if_exists(ref_root / "素材索引.json")
    records = payload.get("records") if isinstance(payload, dict) else None
    return [row for row in (records or []) if isinstance(row, dict)]


def _material_reference_rows(exp_dir: Path) -> List[Dict[str, Any]]:
    return _material_reference_root_and_rows(exp_dir)[1]


def _material_row_path(row: Dict[str, Any], root: Path) -> Optional[Path]:
    raw_path = row.get("stored_file") or row.get("stored_path") or row.get("file_path")
    if raw_path:
        path = Path(str(raw_path))
        return path if path.is_absolute() else root / path
    filename = row.get("stored_filename") or row.get("file_name")
    asset_kind = row.get("asset_kind") or row.get("material_type")
    if filename and asset_kind:
        return root / str(asset_kind) / str(filename)
    return None


def _object_labels_from_material_row(row: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    values: List[Any] = [row.get("primary_object"), row.get("object_label")]
    if isinstance(row.get("object_labels"), list):
        values.extend(row.get("object_labels") or [])
    if isinstance(action.get("objects"), list):
        values.extend(action.get("objects") or [])
    for value in values:
        text = str(value or "").strip()
        if text and text not in labels:
            labels.append(text)
    return labels


def _material_reference_items(exp_dir: Path, experiment_id: str) -> Dict[str, Any]:
    ref_root, rows = _material_reference_root_and_rows(exp_dir)
    items: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
        if asset_kind == "专业报告":
            continue
        path = _material_row_path(row, ref_root)
        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        start_sec = _float_value(row.get("time_start") or row.get("start_sec") or action.get("start_sec") or row.get("source_offset_sec"), 0.0)
        end_sec = _float_value(row.get("time_end") or row.get("end_sec") or action.get("end_sec"), start_sec)
        item_id = str(row.get("item_id") or row.get("candidate_id") or row.get("micro_segment_id") or f"material_reference_{index:04d}")
        url = _experiment_file_api_path(path, experiment_id) if path else None
        preview_url = url if asset_kind == "关键帧" else None
        clip_url = url if asset_kind == "关键片段" else None
        items.append(
            {
                **row,
                "item_id": item_id,
                "experiment_id": experiment_id,
                "event_id": row.get("event_id") or row.get("micro_segment_id") or item_id,
                "display_name": row.get("display_name") or action.get("title") or row.get("action_name") or asset_kind or item_id,
                "event_type": row.get("action_name") or action.get("title") or asset_kind,
                "timestamp_sec": start_sec,
                "time_start": start_sec,
                "time_end": end_sec,
                "duration_sec": max(0.0, end_sec - start_sec),
                "object_labels": _object_labels_from_material_row(row),
                "actions": [row.get("action_name") or action.get("title") or asset_kind],
                "review_status": row.get("review_status") or "accepted",
                "evidence_level": row.get("evidence_level") or "recovered_reference",
                "preview_url": preview_url,
                "clip_url": clip_url,
                "report_url": url if asset_kind == "专业报告" else None,
                "material_url": url,
                "frame_path": str(path) if path and asset_kind == "关键帧" else row.get("frame_path"),
                "clip_file_path": str(path) if path and asset_kind == "关键片段" else row.get("clip_file_path"),
                "published_paths": {
                    "keyframe": str(path) if path and asset_kind == "关键帧" else "",
                    "clip": str(path) if path and asset_kind == "关键片段" else "",
                    "report": str(path) if path and asset_kind == "专业报告" else "",
                },
                "payload": row,
            }
        )
    return {
        "schema_version": "published_materials.material_references_fallback.v1",
        "experiment_id": experiment_id,
        "total": len(items),
        "items": items,
        "source": str(ref_root),
    }


def _workspace_published_payload_with_media_urls(payload: Dict[str, Any]) -> Dict[str, Any]:
    items = payload.get("items") if isinstance(payload, dict) else []
    if not isinstance(items, list):
        return payload
    updated_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            updated_items.append(item)
            continue
        updated = dict(item)
        experiment_id = str(updated.get("experiment_id") or "").strip()
        if experiment_id:
            published_paths = updated.get("published_paths") if isinstance(updated.get("published_paths"), dict) else {}
            preview_source = updated.get("preview_path") or published_paths.get("preview") or published_paths.get("keyframe")
            clip_source = updated.get("clip_path") or published_paths.get("clip")
            report_source = updated.get("report_path") or published_paths.get("report")
            if not updated.get("preview_url") and preview_source:
                updated["preview_url"] = _experiment_file_api_path(Path(str(preview_source)), experiment_id)
            if not updated.get("clip_url") and clip_source:
                updated["clip_url"] = _experiment_file_api_path(Path(str(clip_source)), experiment_id)
            if not updated.get("report_url") and report_source:
                updated["report_url"] = _experiment_file_api_path(Path(str(report_source)), experiment_id)
        updated_items.append(updated)
    return {**payload, "items": updated_items}


def _rebuild_workspace_published_materials_index_quietly() -> Dict[str, Any]:
    try:
        from labsopguard.material_maintenance import rebuild_workspace_published_materials_index

        return rebuild_workspace_published_materials_index(
            PROJECT_ROOT / "outputs" / "experiments",
            _workspace_published_materials_index_path(),
        )
    except Exception as exc:
        logger.exception("Workspace published material reindex failed")
        return {"status": "failed", "error": str(exc)}


_MATERIAL_REVIEW_QUEUE_DIR_NAME = "_material_review_queue"
_LEGACY_MATERIAL_CANDIDATES_DIR_NAME = "material_candidates"
_MATERIAL_CANDIDATE_INDEX_FILENAME = "\u7d20\u6750\u5019\u9009\u7d22\u5f15.jsonl"


def _material_candidate_roots(experiment_id: str) -> List[Path]:
    exp_dir = _experiment_output_dir(experiment_id)
    return [
        exp_dir / _MATERIAL_REVIEW_QUEUE_DIR_NAME,
        exp_dir / _LEGACY_MATERIAL_CANDIDATES_DIR_NAME,
    ]


def _material_candidate_root(experiment_id: str) -> Path:
    roots = _material_candidate_roots(experiment_id)
    for root in roots:
        if (root / _MATERIAL_CANDIDATE_INDEX_FILENAME).exists():
            return root
    for root in roots:
        if root.exists():
            return root
    return roots[0]


def _material_candidate_index_path(experiment_id: str) -> Path:
    return _material_candidate_root(experiment_id) / _MATERIAL_CANDIDATE_INDEX_FILENAME


def _material_candidate_rows(experiment_id: str) -> List[Dict[str, Any]]:
    return _read_jsonl_rows(_material_candidate_index_path(experiment_id))


def _material_candidate_status(row: Dict[str, Any]) -> str:
    raw_status = row.get("candidate_status") or row.get("review_status") or "pending"
    status = str(raw_status).strip().lower()
    if not status or status in {"none", "null", "unknown"}:
        return "pending"
    if status == "accepted":
        return "approved"
    return status


def _material_candidate_group_status(rows: List[Dict[str, Any]]) -> str:
    statuses = [_material_candidate_status(row) for row in rows]
    if any(status in {"approved", "accepted"} for status in statuses):
        return "approved"
    if any(status in {"pending", "review", "needs_review", "candidate"} for status in statuses):
        return "pending"
    if statuses and all(status == "not_selected" for status in statuses):
        return "not_selected"
    if any(status in {"rejected", "failed", "blocked"} for status in statuses):
        return next(status for status in statuses if status in {"rejected", "failed", "blocked"})
    return statuses[0] if statuses else "pending"


def _first_material_candidate_value(rows: List[Dict[str, Any]], key: str, default: Any = None) -> Any:
    for row in rows:
        value = row.get(key)
        if value not in (None, "", [], {}):
            return value
    return default


def _material_candidate_file_payload(experiment_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    candidate_root = _material_candidate_root(experiment_id)
    path = _material_row_path(row, candidate_root)
    url = _experiment_file_api_path(path, experiment_id) if path else None
    asset_kind = str(row.get("asset_kind") or row.get("material_type") or "")
    candidate_status = _material_candidate_status(row)
    review_status = str(row.get("review_status") or candidate_status)
    return {
        **row,
        "candidate_status": candidate_status,
        "review_status": review_status,
        "url": url,
        "preview_url": url if asset_kind == "关键帧" else row.get("preview_url"),
        "clip_url": url if asset_kind == "关键片段" else row.get("clip_url"),
        "exists": bool(path and path.exists() and path.is_file()),
    }


def _float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _material_candidates_payload(experiment_id: str) -> Dict[str, Any]:
    rows = _material_candidate_rows(experiment_id)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("candidate_group_id") or row.get("candidate_id") or "ungrouped")
        groups.setdefault(group_id, []).append(_material_candidate_file_payload(experiment_id, row))

    items: List[Dict[str, Any]] = []
    for group_id, group_rows in sorted(groups.items()):
        keyframes = [row for row in group_rows if row.get("asset_kind") == "关键帧"]
        clips = [row for row in group_rows if row.get("asset_kind") == "关键片段"]
        first = group_rows[0] if group_rows else {}
        recommended = [row for row in group_rows if row.get("recommended") is True]
        group_status = _material_candidate_group_status(group_rows)
        items.append(
            {
                "candidate_group_id": group_id,
                "status": group_status,
                "review_status": "approved" if group_status == "approved" else str(_first_material_candidate_value(group_rows, "review_status", group_status)),
                "recommended": bool(recommended),
                "recommended_count": len(recommended),
                "quality_score": max((_float_value(row.get("quality_score"), 0.0) for row in group_rows), default=0.0),
                "pipeline_status": _first_material_candidate_value(group_rows, "pipeline_status"),
                "pipeline_stage": _first_material_candidate_value(group_rows, "pipeline_stage"),
                "pipeline_flow": _first_material_candidate_value(group_rows, "pipeline_flow", []),
                "review_gate_policy": _first_material_candidate_value(group_rows, "review_gate_policy"),
                "yolo_recheck": _first_material_candidate_value(group_rows, "yolo_recheck"),
                "vlm_semantics": _first_material_candidate_value(group_rows, "vlm_semantics"),
                "primary_object": _first_material_candidate_value(group_rows, "primary_object") or first.get("primary_object"),
                "action_name": _first_material_candidate_value(group_rows, "action_name") or first.get("action_name"),
                "micro_segment_id": _first_material_candidate_value(group_rows, "micro_segment_id") or first.get("micro_segment_id"),
                "parent_segment_id": _first_material_candidate_value(group_rows, "parent_segment_id") or first.get("parent_segment_id"),
                "keyframes": keyframes,
                "clips": clips,
                "files": group_rows,
            }
        )

    candidate_root = _material_candidate_root(experiment_id)
    manifest = _load_json_if_exists(candidate_root / "manifest.json") or {}
    return {
        "schema_version": "material_candidates.api.v1",
        "experiment_id": experiment_id,
        "total": len(items),
        "file_total": len(rows),
        "pending_total": sum(1 for row in rows if _material_candidate_status(row) == "pending"),
        "approved_total": sum(1 for row in rows if _material_candidate_status(row) in {"approved", "accepted"}),
        "items": items,
        "manifest": manifest,
        "candidate_index": str(candidate_root / _MATERIAL_CANDIDATE_INDEX_FILENAME),
    }


def _auto_publish_key_action_material_candidates(
    experiment_id: str,
    candidate_summary: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    enabled = os.environ.get("KEY_ACTION_AUTO_PUBLISH_MATERIAL_CANDIDATES", "1").strip().lower() not in {"0", "false", "no", "off"}
    result: Dict[str, Any] = {
        "schema_version": "key_action_material_auto_publish.v1",
        "enabled": enabled,
        "status": "skipped",
        "eligible_count": 0,
        "approved_count": 0,
    }
    if not enabled:
        result["reason"] = "disabled_by_env"
        return result
    if not isinstance(candidate_summary, dict) or candidate_summary.get("status") == "failed":
        result["reason"] = "candidate_generation_unavailable"
        return result

    eligible_ids: List[str] = []
    for row in _material_candidate_rows(experiment_id):
        if row.get("recommended") is not True:
            continue
        if row.get("exists") is False:
            continue
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        candidate_status = str(row.get("candidate_status") or "pending").lower()
        review_status = str(row.get("review_status") or "pending").lower()
        if candidate_status in {"rejected", "not_selected"} or review_status in {"rejected", "not_selected"}:
            continue
        yolo_recheck = row.get("yolo_recheck") if isinstance(row.get("yolo_recheck"), dict) else {}
        if str(yolo_recheck.get("status") or "").lower() != "passed":
            continue
        pipeline_status = str(row.get("pipeline_status") or "").lower()
        if pipeline_status not in {"vlm_assisted_yolo_recheck_passed", "yolo_recheck_passed_frontend_review_required"}:
            continue
        vlm_semantics = row.get("vlm_semantics") if isinstance(row.get("vlm_semantics"), dict) else {}
        if str(vlm_semantics.get("status") or "").lower() in {"error", "uncertain_vlm_review", "weak_but_yolo_preserved"}:
            continue
        eligible_ids.append(candidate_id)

    result["eligible_count"] = len(eligible_ids)
    if not eligible_ids:
        result["reason"] = "no_recommended_yolo_vlm_passed_candidates"
        return result

    try:
        from key_action_indexer.material_references import approve_material_candidates  # type: ignore

        approval = approve_material_candidates(
            _key_action_output_dir(experiment_id),
            candidate_ids=eligible_ids,
            reviewer="key_action_pipeline",
            notes="Auto-published recommended YOLO/VLM-passed material candidates after upload analysis.",
        )
    except Exception as exc:
        logger.exception("Key-action material auto-publish failed for experiment %s", experiment_id)
        result.update({"status": "failed", "error": str(exc)})
        return result

    exp_dir = _experiment_output_dir(experiment_id)
    published_materials = _sync_published_materials_from_references(exp_dir, experiment_id)
    approved_count = int(approval.get("approved_count") or 0) if isinstance(approval, dict) else 0
    workspace_reindex = _rebuild_workspace_published_materials_index_quietly() if approved_count else {"status": "skipped", "reason": "no_approved_candidates"}
    result.update(
        {
            "status": "completed",
            "approved_count": approved_count,
            "approval": approval,
            "published_materials": published_materials,
            "workspace_published_materials_reindex": workspace_reindex,
        }
    )
    return result


def _published_material_items(exp_dir: Path, experiment_id: str) -> Dict[str, Any]:
    payload = _load_json_if_exists(exp_dir / "published_materials.json")
    if isinstance(payload, dict) and payload.get("items"):
        return payload
    fallback = _material_reference_items(exp_dir, experiment_id)
    try:
        from labsopguard.material_publishing import SemanticMaterialPublisher

        listed = SemanticMaterialPublisher(exp_dir, experiment_id=experiment_id).list_published()
        if isinstance(listed, dict) and listed.get("items"):
            return listed
    except Exception:
        logger.debug("Published material listing unavailable; falling back to material_references", exc_info=True)
    if fallback.get("items"):
        return fallback
    return payload if isinstance(payload, dict) else fallback


def _sync_published_materials_from_references(exp_dir: Path, experiment_id: str) -> Dict[str, Any]:
    payload = _material_reference_items(exp_dir, experiment_id)
    payload = {
        **payload,
        "schema_version": "published_materials.approved_material_references.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(payload.get("items") or []),
        "policy": "Only frontend-approved keyframes and key clips are synchronized into the key material library. Professional PDF reports are approved into the professional report folder only.",
    }
    _write_json(exp_dir / "published_materials.json", payload)
    return payload


def _professional_report_basename() -> str:
    raw = os.environ.get("PROFESSIONAL_REPORT_BASENAME", "professional_report_qwen36max").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return cleaned or "professional_report_qwen36max"


def _professional_report_output_paths(experiment_id: str) -> Dict[str, Path]:
    reports_dir = _experiment_output_dir(experiment_id) / "reports"
    base = _professional_report_basename()
    return {
        "pdf": reports_dir / f"{base}.pdf",
        "html": reports_dir / f"{base}.html",
        "json": reports_dir / f"{base}.json",
        "manifest": reports_dir / "professional_report_manifest.json",
    }


def _professional_report_enabled() -> bool:
    return os.environ.get("EXPERIMENT_PROFESSIONAL_REPORT_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _experiment_file_api_path(path: Path, experiment_id: str) -> Optional[str]:
    resolved_path = path.resolve()
    exp_dir = _experiment_output_dir(experiment_id)
    try:
        rel = resolved_path.relative_to(exp_dir.resolve())
    except ValueError:
        pass
    else:
        return f"/api/v1/experiments/{experiment_id}/files/{quote(rel.as_posix(), safe='/')}"

    for ref_root in _material_reference_root_candidates(exp_dir):
        try:
            rel = resolved_path.relative_to(ref_root.resolve())
        except ValueError:
            continue
        return f"/api/v1/experiments/{experiment_id}/material-references/files/{quote(rel.as_posix(), safe='/')}"
    return None


def _write_professional_report_html_snapshot(*, sidecar_path: Path, html_path: Path, summary: Dict[str, Any]) -> None:
    if html_path.exists():
        return
    import html as html_lib

    sidecar = _load_json_if_exists(sidecar_path) or {}
    report = sidecar.get("report") if isinstance(sidecar, dict) else {}
    context = sidecar.get("context") if isinstance(sidecar, dict) else {}
    overview = context.get("overview") if isinstance(context, dict) else {}
    experiment = overview.get("experiment") if isinstance(overview, dict) else {}
    executive = report.get("executive_summary") if isinstance(report, dict) else {}
    findings = report.get("key_findings") if isinstance(report, dict) else []
    alerts = (report.get("risk_alerts") or {}).get("alerts", []) if isinstance(report, dict) else []

    def esc(value: Any) -> str:
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        return html_lib.escape(str(value or "-"))

    finding_items = "".join(
        f"<li><strong>{esc(item.get('finding') if isinstance(item, dict) else item)}</strong><br>{esc(item.get('evidence') if isinstance(item, dict) else '')}</li>"
        for item in (findings if isinstance(findings, list) else [])
    )
    alert_items = "".join(
        f"<li><strong>{esc(item.get('severity') if isinstance(item, dict) else '')}</strong> {esc(item.get('rule') if isinstance(item, dict) else item)}</li>"
        for item in (alerts if isinstance(alerts, list) else [])
    )
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{esc(experiment.get('experiment_name') if isinstance(experiment, dict) else summary.get('experiment_id'))} - Professional Report</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 40px; color: #172033; line-height: 1.65; }}
    h1, h2 {{ color: #0f172a; }}
    .meta {{ background: #f5f7fb; border: 1px solid #dbe3ef; padding: 16px; border-radius: 8px; }}
    li {{ margin-bottom: 10px; }}
    code {{ background: #eef2f7; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>RealityLoop Professional Experiment Analysis Report</h1>
  <div class="meta">
    <div>Experiment: {esc(experiment.get('experiment_name') if isinstance(experiment, dict) else "")}</div>
    <div>Experiment ID: <code>{esc(summary.get('experiment_id'))}</code></div>
    <div>Generated At: {esc(summary.get('generated_at'))}</div>
    <div>PDF: {esc(summary.get('pdf_path'))}</div>
  </div>
  <h2>Executive Summary</h2>
  <p>{esc(executive.get('summary') if isinstance(executive, dict) else "")}</p>
  <p><strong>Conclusion:</strong> {esc(executive.get('overall_conclusion') if isinstance(executive, dict) else "")}</p>
  <h2>Key Findings</h2>
  <ul>{finding_items or "<li>-</li>"}</ul>
  <h2>Risk Alerts</h2>
  <ul>{alert_items or "<li>-</li>"}</ul>
</body>
</html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_text, encoding="utf-8")


def _sync_professional_report_material_delivery(
    experiment_id: str,
    exp: Dict[str, Any],
    report_summary: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        from key_action_indexer.material_references import sync_professional_report_material_references

        report_payload = {
            **report_summary,
            "experiment_title": exp.get("title") or exp.get("experiment_name") or experiment_id,
            "experiment_name": exp.get("title") or exp.get("experiment_name") or experiment_id,
            "experiment_date": exp.get("created_at") or exp.get("started_at") or _now_iso(),
        }
        session_dir = _experiment_output_dir(experiment_id) / "key_action_index"
        if not session_dir.exists():
            session_dir = _experiment_output_dir(experiment_id)
        return sync_professional_report_material_references(
            session_dir,
            report_summary=report_payload,
            archive_existing=False,
        )
    except Exception as exc:
        logger.exception("Professional report material delivery sync failed for %s", experiment_id)
        return {"available": False, "error": str(exc)}


def _generate_professional_report_for_experiment(
    experiment_id: str,
    *,
    output_paths: Optional[Dict[str, Any]] = None,
    fail_pipeline: Optional[bool] = None,
) -> Dict[str, Any]:
    report_paths = _professional_report_output_paths(experiment_id)
    summary: Dict[str, Any] = {
        "schema_version": "professional_report.pipeline.v1",
        "experiment_id": experiment_id,
        "enabled": _professional_report_enabled(),
        "available": False,
        "pdf_path": str(report_paths["pdf"]),
        "html_path": str(report_paths["html"]),
        "sidecar_path": str(report_paths["json"]),
        "manifest_path": str(report_paths["manifest"]),
    }
    report_paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    if not summary["enabled"]:
        summary["status"] = "disabled"
        _write_json(report_paths["manifest"], summary)
        return summary

    should_fail_pipeline = (
        os.environ.get("EXPERIMENT_PROFESSIONAL_REPORT_FAIL_PIPELINE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if fail_pipeline is None
        else bool(fail_pipeline)
    )
    try:
        from labsopguard.professional_report import generate_professional_report_pdf

        exp_dir = _experiment_output_dir(experiment_id)
        exp = _normalize_experiment_dict(_load_json_if_exists(exp_dir / "experiment.json") or {})
        exp["output_paths"] = {
            **(exp.get("output_paths") or {}),
            **({key: str(value) for key, value in (output_paths or {}).items()}),
        }
        overview = _build_analysis_overview(exp)
        try:
            key_actions = _key_action_results_payload(experiment_id)  # type: ignore[name-defined]
        except Exception as exc:
            logger.info("Professional report key-action payload unavailable for %s: %s", experiment_id, exc)
            key_actions = None
        try:
            materials = _published_material_items(exp_dir, experiment_id)
        except Exception as exc:
            logger.info("Professional report material payload unavailable for %s: %s", experiment_id, exc)
            materials = {"schema_version": "published_materials.empty", "total": 0, "items": []}

        logo_path = PROJECT_ROOT / "docs" / "images" / "realityloop-logo-report.png"
        generated = generate_professional_report_pdf(
            overview=overview,
            key_actions=key_actions,
            materials=materials,
            output_pdf_path=report_paths["pdf"],
            logo_path=logo_path if logo_path.exists() else None,
        )
        qwen_meta = generated.get("qwen") if isinstance(generated.get("qwen"), dict) else {}
        report_model = (
            qwen_meta.get("model")
            or os.environ.get("QWEN_REPORT_MODEL")
            or "qwen3.6-max-preview"
        )
        summary.update(
            {
                **generated,
                "status": "completed",
                "available": report_paths["pdf"].exists(),
                "model": report_model,
                "qwen_model": report_model,
                "pdf_path": str(report_paths["pdf"]),
                "html_path": str(report_paths["html"]),
                "sidecar_path": str(report_paths["json"]),
                "manifest_path": str(report_paths["manifest"]),
                "pdf_url": f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_pdf",
                "html_url": f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_html",
                "sidecar_url": f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_json",
                "generated_at": _now_iso(),
            }
        )
        if report_paths["pdf"].exists():
            summary["size_bytes"] = report_paths["pdf"].stat().st_size
        _write_professional_report_html_snapshot(
            sidecar_path=report_paths["json"],
            html_path=report_paths["html"],
            summary=summary,
        )
        _write_json(report_paths["manifest"], summary)
        summary["material_delivery"] = _sync_professional_report_material_delivery(experiment_id, exp, summary)
        _write_json(report_paths["manifest"], summary)
        return summary
    except Exception as exc:
        logger.exception("Professional PDF report generation failed for experiment %s", experiment_id)
        summary.update(
            {
                "status": "failed",
                "available": False,
                "error": str(exc),
                "generated_at": _now_iso(),
            }
        )
        _write_json(report_paths["manifest"], summary)
        if should_fail_pipeline:
            raise
        return summary


def _attach_professional_report_output_paths(
    experiment_id: str,
    output_paths: Dict[str, Any],
    report_summary: Dict[str, Any],
) -> Dict[str, Any]:
    if report_summary.get("pdf_path"):
        output_paths["professional_report_pdf"] = str(report_summary["pdf_path"])
    if report_summary.get("html_path"):
        output_paths["professional_report_html"] = str(report_summary["html_path"])
    if report_summary.get("sidecar_path"):
        output_paths["professional_report_json"] = str(report_summary["sidecar_path"])
    if report_summary.get("manifest_path"):
        output_paths["professional_report_manifest_json"] = str(report_summary["manifest_path"])
    output_paths["professional_report_summary"] = report_summary
    return output_paths


def _key_action_output_dir(experiment_id: str) -> Path:
    return _experiment_output_dir(experiment_id) / "key_action_index"


def _default_key_action_yolo_model_path(view: str) -> Optional[str]:
    view_name = str(view or "").strip().lower()
    env_name = (
        "KEY_ACTION_YOLO_FIRST_PERSON_MODEL"
        if view_name == "first_person"
        else "KEY_ACTION_YOLO_THIRD_PERSON_MODEL"
    )
    env_path = os.environ.get(env_name)
    candidates = [
        Path(env_path) if env_path else None,
        PROJECT_ROOT / "models" / "yolo" / view_name / "best.pt",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def _default_key_action_yolo_device() -> str:
    configured = os.environ.get("KEY_ACTION_YOLO_DEVICE")
    if configured is not None and configured.strip():
        return configured.strip()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return os.environ.get("KEY_ACTION_YOLO_CUDA_DEVICE", "0").strip() or "0"
    except Exception:
        pass
    return "cpu"


def _with_default_key_action_yolo_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config or {})
    merged.setdefault("detector_backend", "yolo")
    merged.setdefault("yolo_scan_both_views", True)
    merged.setdefault("yolo_preferred_view", "first_person")
    merged.setdefault("yolo_device", _default_key_action_yolo_device())
    merged.setdefault("yolo_conf", float(os.environ.get("KEY_ACTION_YOLO_CONF", "0.25")))
    merged.setdefault("yolo_iou", float(os.environ.get("KEY_ACTION_YOLO_IOU", "0.45")))
    first_model = _default_key_action_yolo_model_path("first_person")
    third_model = _default_key_action_yolo_model_path("third_person")
    if first_model:
        merged.setdefault("yolo_first_person_model_path", first_model)
    if third_model:
        merged.setdefault("yolo_third_person_model_path", third_model)
    return merged


def _key_action_vlm_assist_config() -> Dict[str, Any]:
    import importlib.util

    enabled = os.environ.get("KEY_ACTION_ENABLE_VLM_ASSIST", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    try:
        max_groups = int(os.environ.get("KEY_ACTION_VLM_MAX_GROUPS", "8"))
    except Exception:
        max_groups = 8
    return {
        "enabled": enabled,
        "model": (
            os.environ.get("KEY_ACTION_VLM_MODEL")
            or os.environ.get("QWEN_VL_MODEL")
            or os.environ.get("VLM_MODEL")
            or "qwen3.6-plus"
        ),
        "max_groups": max(0, max_groups),
        "api_key_configured": bool(os.environ.get("DASHSCOPE_API_KEY")),
        "dashscope_installed": importlib.util.find_spec("dashscope") is not None,
    }


def _build_key_action_vlm_assist_client() -> tuple[Any | None, Dict[str, Any]]:
    config = _key_action_vlm_assist_config()
    if not config["enabled"]:
        return None, {**config, "configured": False, "reason": "disabled"}
    if not config["dashscope_installed"]:
        return None, {**config, "configured": False, "reason": "dashscope_not_installed"}
    if not config["api_key_configured"]:
        return None, {**config, "configured": False, "reason": "dashscope_api_key_missing"}
    try:
        from experiment.vlm_client import DashScopeVLClient

        client = DashScopeVLClient(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=os.environ.get("DASHSCOPE_BASE_URL"),
            model=str(config["model"]),
            timeout=int(os.environ.get("KEY_ACTION_VLM_TIMEOUT", "60")),
        )
        return client, {**config, "configured": True}
    except Exception as exc:
        logger.warning("Key-action VLM assist unavailable: %s", exc)
        return None, {**config, "configured": False, "error": str(exc)}


def _key_action_status_path(experiment_id: str) -> Path:
    return _key_action_output_dir(experiment_id) / "job_status.json"


def _write_key_action_status(experiment_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    status_path = _key_action_status_path(experiment_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    current = _load_json_if_exists(status_path) or {}
    if not isinstance(current, dict):
        current = {}
    current.update(payload)
    current.setdefault("experiment_id", experiment_id)
    current.setdefault("output_dir", str(_key_action_output_dir(experiment_id)))
    current["updated_at"] = _now_iso()
    _write_json(status_path, current)
    return current


def _read_key_action_status(experiment_id: str) -> Dict[str, Any]:
    payload = _load_json_if_exists(_key_action_status_path(experiment_id))
    if isinstance(payload, dict):
        return payload
    return _key_action_status_payload(experiment_id)


def _key_action_summary_for_experiment(experiment_id: str) -> Dict[str, Any]:
    status = _key_action_status_payload(experiment_id)
    summary = dict(status.get("summary") or {})
    return {
        "status": status.get("status", "not_started"),
        "progress": status.get("progress", 0.0),
        "message": status.get("message"),
        "segment_count": int(summary.get("segment_count") or 0),
        "micro_segment_count": int(summary.get("micro_segment_count") or 0),
        "interaction_count": int(summary.get("interaction_count") or 0),
        "vector_count": int(summary.get("vector_count") or 0),
        "raw_yolo_interaction_count": int(summary.get("raw_yolo_interaction_count") or 0),
        "source": summary.get("source"),
    }


def _key_action_file_url(experiment_id: str, path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = _experiment_output_dir(experiment_id) / path
    return _experiment_file_api_path(path, experiment_id)


def _key_action_status_payload(experiment_id: str) -> Dict[str, Any]:
    output_dir = _key_action_output_dir(experiment_id)
    status_payload = _load_json_if_exists(_key_action_status_path(experiment_id))
    if isinstance(status_payload, dict) and status_payload:
        return _reconcile_key_action_status_summary(experiment_id, status_payload)
    manifest = _load_json_if_exists(output_dir / "manifest.json") or _load_json_if_exists(_experiment_output_dir(experiment_id) / "manifest.json") or {}
    has_outputs = output_dir.exists() or bool(manifest)
    return {
        "experiment_id": experiment_id,
        "status": "completed" if has_outputs else "not_started",
        "progress": 1.0 if has_outputs else 0.0,
        "message": "Key-action recovery artifacts are available." if has_outputs else "Key-action index is not ready.",
        "output_dir": str(output_dir),
        "completed_at": manifest.get("generated_at") or manifest.get("created_at"),
        "summary": {
            "segment_count": len(manifest.get("actions") or []),
            "keyframe_count": manifest.get("keyframe_count", 0),
            "key_clip_count": manifest.get("key_clip_count", 0),
            "report_count": manifest.get("report_count", 0),
            "recovery_status": manifest.get("recovery_status"),
        },
    }


def _jsonl_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            count += 1
    return count


def _yolo_interaction_count_from_rows(path: Path) -> int:
    count = 0
    for row in _read_jsonl_rows(path):
        count += len(_key_action_list(row.get("hand_object_interactions")))
    return count


def _reconcile_key_action_status_summary(experiment_id: str, status_payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(status_payload)
    output_dir = _key_action_output_dir(experiment_id)
    metadata_dir = output_dir / "metadata"
    cv_dir = output_dir / "cv_outputs"
    segment_rows = _jsonl_row_count(metadata_dir / "key_action_segments.jsonl")
    micro_rows = _jsonl_row_count(metadata_dir / "micro_segments.jsonl")
    vector_rows = _jsonl_row_count(metadata_dir / "vector_metadata.jsonl")
    micro_vector_rows = _jsonl_row_count(metadata_dir / "micro_vector_metadata.jsonl")
    yolo_frame_rows = _jsonl_row_count(cv_dir / "yolo_frame_rows.jsonl")
    raw_interactions = _yolo_interaction_count_from_rows(cv_dir / "yolo_frame_rows.jsonl") if yolo_frame_rows else 0
    pipeline_summary = _load_json_if_exists(output_dir / "pipeline_summary.json") or {}
    detector_summary = pipeline_summary.get("detector_summary") if isinstance(pipeline_summary, dict) else None
    summary = dict(payload.get("summary") or {})
    if segment_rows:
        summary["segment_count"] = segment_rows
        summary["source"] = "key_action_index_metadata"
    elif isinstance(detector_summary, dict) and detector_summary.get("segment_count") is not None:
        summary["segment_count"] = detector_summary.get("segment_count")
    if micro_rows:
        summary["micro_segment_count"] = micro_rows
        summary["raw_micro_segment_count"] = micro_rows
    elif isinstance(pipeline_summary, dict):
        if pipeline_summary.get("micro_segment_count") is not None:
            summary["micro_segment_count"] = pipeline_summary.get("micro_segment_count")
        if pipeline_summary.get("raw_micro_segment_count") is not None:
            summary["raw_micro_segment_count"] = pipeline_summary.get("raw_micro_segment_count")
    if yolo_frame_rows:
        summary["yolo_frame_row_count"] = yolo_frame_rows
    if vector_rows or micro_vector_rows:
        summary["vector_count"] = vector_rows + micro_vector_rows
        summary["segment_vector_count"] = vector_rows
        summary["micro_vector_count"] = micro_vector_rows
    if raw_interactions:
        summary["interaction_count"] = raw_interactions
        summary["raw_yolo_interaction_count"] = raw_interactions
    if isinstance(detector_summary, dict):
        merged_detector = dict(detector_summary)
        if segment_rows:
            merged_detector["segment_count"] = segment_rows
        if yolo_frame_rows:
            merged_detector["frame_rows"] = yolo_frame_rows
        if raw_interactions:
            merged_detector["interaction_count"] = raw_interactions
        summary["detector_summary"] = merged_detector
    payload["summary"] = summary
    return payload


def _key_action_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _key_action_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _key_action_copy_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [copy.deepcopy(row) for row in rows if isinstance(row, dict)]


def _key_action_put_file_url(experiment_id: str, row: Dict[str, Any], path_key: str, url_key: str) -> None:
    if not isinstance(row, dict):
        return
    url = _key_action_file_url(experiment_id, row.get(path_key))
    if url:
        row[url_key] = url


def _enrich_key_action_keyframes(experiment_id: str, keyframes: Any) -> None:
    if not isinstance(keyframes, dict):
        return
    urls: List[str] = []
    key_pairs = {
        "start": "start_url",
        "middle": "middle_url",
        "end": "end_url",
        "contact": "contact_url",
        "peak": "peak_url",
        "release": "release_url",
        "contact_frame": "contact_frame_url",
        "peak_frame": "peak_frame_url",
        "release_frame": "release_frame_url",
    }
    for path_key, url_key in key_pairs.items():
        url = _key_action_file_url(experiment_id, keyframes.get(path_key))
        if url:
            keyframes[url_key] = url
            urls.append(url)
    if urls:
        keyframes["urls"] = urls


def _enrich_key_action_media_ref(experiment_id: str, ref: Any) -> None:
    if not isinstance(ref, dict):
        return
    _key_action_put_file_url(experiment_id, ref, "clip_path", "clip_url")
    _key_action_put_file_url(experiment_id, ref, "annotated_clip_path", "annotated_clip_url")
    _key_action_put_file_url(experiment_id, ref, "keyframe_path", "keyframe_url")
    if isinstance(ref.get("keyframe_paths"), list):
        urls = [_key_action_file_url(experiment_id, path) for path in ref.get("keyframe_paths") or []]
        ref["keyframe_urls"] = [url for url in urls if url]
    _enrich_key_action_keyframes(experiment_id, ref.get("keyframes"))


def _enrich_key_action_event(experiment_id: str, event: Dict[str, Any]) -> None:
    _key_action_put_file_url(experiment_id, event, "keyframe_path", "keyframe_url")
    _key_action_put_file_url(experiment_id, event, "keyframe_path", "preview_url")
    _key_action_put_file_url(experiment_id, event, "path", "url")
    _key_action_put_file_url(experiment_id, event, "path", "preview_url")


def _enrich_key_action_micro_segment(experiment_id: str, micro: Dict[str, Any]) -> None:
    _enrich_key_action_media_ref(experiment_id, micro.get("first_person"))
    _enrich_key_action_media_ref(experiment_id, micro.get("third_person"))
    for path_key, url_key in (
        ("first_person_clip", "first_person_clip_url"),
        ("third_person_clip", "third_person_clip_url"),
        ("peak_keyframe", "peak_keyframe_url"),
    ):
        _key_action_put_file_url(experiment_id, micro, path_key, url_key)
    _enrich_key_action_keyframes(experiment_id, micro.get("keyframes"))
    for binding in _key_action_list(micro.get("asset_bindings")):
        _enrich_key_action_media_ref(experiment_id, binding)


def _enrich_key_action_segment(experiment_id: str, segment: Dict[str, Any]) -> None:
    _enrich_key_action_media_ref(experiment_id, segment.get("first_person"))
    _enrich_key_action_media_ref(experiment_id, segment.get("third_person"))
    for binding in _key_action_list(segment.get("asset_bindings")):
        _enrich_key_action_media_ref(experiment_id, binding)
    for micro in _key_action_list(segment.get("micro_segments")):
        if isinstance(micro, dict):
            micro.setdefault("parent_segment_id", segment.get("segment_id"))
            _enrich_key_action_micro_segment(experiment_id, micro)
    for event in _key_action_list(segment.get("interaction_events")):
        if isinstance(event, dict):
            event.setdefault("segment_id", segment.get("segment_id"))
            _enrich_key_action_event(experiment_id, event)
    for keyframe in _key_action_list(segment.get("interaction_keyframes")):
        if isinstance(keyframe, dict):
            keyframe.setdefault("segment_id", segment.get("segment_id"))
            _enrich_key_action_event(experiment_id, keyframe)


def _key_action_segment_id_for_row(row: Dict[str, Any], segments: List[Dict[str, Any]]) -> Optional[str]:
    if not segments:
        return None
    local_time = _key_action_float(row.get("local_time_sec") or row.get("time_sec") or row.get("timestamp_sec"), -1.0)
    view = str(row.get("source_view") or row.get("view") or "").strip().lower()
    for segment in segments:
        ref = segment.get(view) if view in {"first_person", "third_person"} else None
        if not isinstance(ref, dict):
            ref = segment.get("first_person") if view.startswith("first") else segment.get("third_person")
        if not isinstance(ref, dict):
            continue
        start_sec = _key_action_float(ref.get("local_start_sec"), -1.0)
        end_sec = _key_action_float(ref.get("local_end_sec"), -1.0)
        if start_sec - 0.25 <= local_time <= end_sec + 0.25:
            return str(segment.get("segment_id") or "")
    if len(segments) == 1:
        return str(segments[0].get("segment_id") or "")
    return None


def _yolo_frame_rows_to_interaction_events(
    experiment_id: str,
    yolo_frame_rows: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for frame_row in yolo_frame_rows:
        if not isinstance(frame_row, dict):
            continue
        segment_id = _key_action_segment_id_for_row(frame_row, segments)
        for interaction in _key_action_list(frame_row.get("hand_object_interactions")):
            if not isinstance(interaction, dict):
                continue
            index = len(events) + 1
            hand_label = interaction.get("hand_label")
            object_label = interaction.get("object_label")
            event = {
                "event_id": f"yolo_interaction_{index:04d}",
                "segment_id": segment_id,
                "view": frame_row.get("source_view") or frame_row.get("view"),
                "source_view": frame_row.get("source_view") or frame_row.get("view"),
                "frame_index": frame_row.get("frame_index"),
                "local_time_sec": frame_row.get("local_time_sec") or frame_row.get("time_sec"),
                "global_time": frame_row.get("global_time") or frame_row.get("global_timestamp"),
                "interaction": f"{hand_label or 'hand'}->{object_label or 'object'}",
                "hand_label": hand_label,
                "object_label": object_label,
                "object_name": interaction.get("object_name") or object_label,
                "confidence": interaction.get("score") or interaction.get("confidence"),
                "score": interaction.get("score") or interaction.get("confidence"),
                "distance_px": interaction.get("distance_px"),
                "iou": interaction.get("iou"),
                "hand_bbox": interaction.get("hand_bbox"),
                "object_bbox": interaction.get("object_bbox"),
                "detections": frame_row.get("detections") or [],
                "source": "yolo_frame_rows",
            }
            _enrich_key_action_event(experiment_id, event)
            events.append(event)
    return events


def _legacy_key_action_manifest_segments(
    experiment_id: str,
    output_dir: Path,
    root_manifest: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions = root_manifest.get("actions") or []
    segments: List[Dict[str, Any]] = []
    for index, action in enumerate(actions if isinstance(actions, list) else []):
        if not isinstance(action, dict):
            continue
        start_sec = _key_action_float(action.get("start_sec"), 0.0)
        end_sec = _key_action_float(action.get("end_sec"), start_sec)
        segment_id = str(action.get("id") or f"seg_{index + 1:06d}")
        objects = [str(item) for item in (action.get("objects") or [])]
        segments.append(
            {
                "session_id": experiment_id,
                "segment_id": segment_id,
                "global_start_time": root_manifest.get("generated_at") or "",
                "global_end_time": root_manifest.get("generated_at") or "",
                "duration_sec": max(0.0, end_sec - start_sec),
                "third_person": {
                    "video_path": "",
                    "clip_path": "",
                    "local_start_sec": start_sec,
                    "local_end_sec": end_sec,
                },
                "first_person": None,
                "cv_detection": {
                    "avg_motion_score": 0.0,
                    "avg_active_score": 0.0,
                    "start_reason": "recovered_manifest",
                    "end_reason": "recovered_manifest",
                },
                "text_description": {
                    "action_type": action.get("title") or segment_id,
                    "summary": action.get("description") or "",
                    "tools": [],
                    "objects": objects,
                    "numbers": [],
                },
                "dialogue_context": [],
                "yolo_labels": objects,
                "visual_keywords": objects,
                "index": {
                    "embedding_id": segment_id,
                    "index_text": " ".join([str(action.get("title") or ""), str(action.get("description") or ""), *objects]),
                    "vector_store": str(output_dir / "index"),
                },
            }
        )
    return segments


def _key_action_results_payload(experiment_id: str) -> Dict[str, Any]:
    output_dir = _key_action_output_dir(experiment_id)
    metadata_dir = output_dir / "metadata"
    cv_dir = output_dir / "cv_outputs"
    exp_dir = _experiment_output_dir(experiment_id)
    root_manifest = _load_json_if_exists(exp_dir / "manifest.json") or {}
    status = _key_action_status_payload(experiment_id)

    segment_rows = _read_jsonl_rows(metadata_dir / "key_action_segments.jsonl")
    using_index_metadata = bool(segment_rows)
    if using_index_metadata:
        segments = _key_action_copy_rows(segment_rows)
        micro_segments = _key_action_copy_rows(_read_jsonl_rows(metadata_dir / "micro_segments.jsonl"))
        if not micro_segments:
            for segment in segments:
                for micro in _key_action_list(segment.get("micro_segments")):
                    if isinstance(micro, dict):
                        row = copy.deepcopy(micro)
                        row.setdefault("parent_segment_id", segment.get("segment_id"))
                        micro_segments.append(row)
        micro_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for micro in micro_segments:
            parent_id = str(micro.get("parent_segment_id") or micro.get("segment_id") or "")
            micro_by_parent.setdefault(parent_id, []).append(micro)
            _enrich_key_action_micro_segment(experiment_id, micro)
        for segment in segments:
            segment_id = str(segment.get("segment_id") or "")
            if not segment.get("micro_segments") and segment_id in micro_by_parent:
                segment["micro_segments"] = copy.deepcopy(micro_by_parent[segment_id])
            _enrich_key_action_segment(experiment_id, segment)

        interaction_keyframes: List[Dict[str, Any]] = []
        segment_interaction_events: List[Dict[str, Any]] = []
        yolo_interactions: List[Dict[str, Any]] = []
        for segment in segments:
            segment_id = segment.get("segment_id")
            for event in _key_action_list(segment.get("interaction_events")):
                if isinstance(event, dict):
                    row = copy.deepcopy(event)
                    row.setdefault("segment_id", segment_id)
                    _enrich_key_action_event(experiment_id, row)
                    segment_interaction_events.append(row)
            for keyframe in _key_action_list(segment.get("interaction_keyframes")):
                if isinstance(keyframe, dict):
                    row = copy.deepcopy(keyframe)
                    row.setdefault("segment_id", segment_id)
                    _enrich_key_action_event(experiment_id, row)
                    interaction_keyframes.append(row)
            for interaction in _key_action_list(segment.get("yolo_interactions")):
                if isinstance(interaction, dict):
                    row = copy.deepcopy(interaction)
                    row.setdefault("segment_id", segment_id)
                    yolo_interactions.append(row)

        yolo_frame_rows = _read_jsonl_rows(cv_dir / "yolo_frame_rows.jsonl")
        raw_yolo_events = _yolo_frame_rows_to_interaction_events(experiment_id, yolo_frame_rows, segments)
        interaction_events = raw_yolo_events or segment_interaction_events or yolo_interactions
        vector_metadata = _key_action_copy_rows(_read_jsonl_rows(metadata_dir / "vector_metadata.jsonl"))
        micro_vector_metadata = _key_action_copy_rows(_read_jsonl_rows(metadata_dir / "micro_vector_metadata.jsonl"))
        pipeline_summary = _load_json_if_exists(output_dir / "pipeline_summary.json") or {}
        yolo_frame_scan = _load_json_if_exists(metadata_dir / "yolo_frame_scan.json") or {}
        summary: Dict[str, Any] = {
            **(status.get("summary") if isinstance(status.get("summary"), dict) else {}),
            **(pipeline_summary.get("summary") if isinstance(pipeline_summary.get("summary"), dict) else {}),
            "source": "key_action_index_metadata",
            "segment_count": len(segments),
            "micro_segment_count": len(micro_segments),
            "interaction_event_count": len(interaction_events),
            "segment_interaction_event_count": len(segment_interaction_events),
            "interaction_keyframe_count": len(interaction_keyframes),
            "yolo_frame_row_count": len(yolo_frame_rows),
            "raw_yolo_interaction_count": len(raw_yolo_events),
            "yolo_interaction_count": len(yolo_interactions),
            "vector_metadata_count": len(vector_metadata),
        }
        detector_summary = pipeline_summary.get("detector_summary")
        if isinstance(detector_summary, dict):
            summary["detector_summary"] = detector_summary
        if not yolo_frame_scan:
            yolo_frame_scan = {
                "frame_rows": len(yolo_frame_rows),
                "interaction_count": len(raw_yolo_events),
                "source": "cv_outputs/yolo_frame_rows.jsonl",
            }
    else:
        segments = _legacy_key_action_manifest_segments(experiment_id, output_dir, root_manifest)
        micro_segments = []
        interaction_events = []
        interaction_keyframes = []
        vector_metadata = []
        micro_vector_metadata = []
        yolo_frame_scan = {}
        summary = {
            **(status.get("summary") if isinstance(status.get("summary"), dict) else {}),
            "source": "recovered_manifest" if root_manifest else "key_action_index",
            "segment_count": len(segments),
            "micro_segment_count": 0,
            "interaction_event_count": 0,
            "interaction_keyframe_count": 0,
        }

    formal_report_path = _experiment_output_artifact_paths(experiment_id).get("professional_report_pdf")
    return {
        "experiment_id": experiment_id,
        "status": status,
        "output_dir": str(output_dir),
        "summary": summary,
        "formal_report": {
            "path": str(formal_report_path) if formal_report_path else None,
            "url": f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_pdf" if formal_report_path and formal_report_path.exists() else None,
            "available": bool(formal_report_path and formal_report_path.exists()),
        },
        "formal_report_path": str(formal_report_path) if formal_report_path else None,
        "formal_report_url": f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_pdf" if formal_report_path and formal_report_path.exists() else None,
        "detected_segments": segments,
        "segments": segments,
        "vector_metadata": vector_metadata,
        "micro_vector_metadata": micro_vector_metadata,
        "micro_segments": micro_segments,
        "interaction_events": interaction_events,
        "interaction_keyframes": interaction_keyframes,
        "yolo_frame_scan": yolo_frame_scan,
        "debug": {
            "projection_source": summary.get("source"),
            "report": str(output_dir / "reports" / "mvp_validation_report.md"),
            "formal_report_path": str(formal_report_path) if formal_report_path else None,
            "formal_report_url": f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_pdf" if formal_report_path and formal_report_path.exists() else None,
        },
    }


def _key_action_review_url(experiment_id: str, path_value: Any) -> Optional[str]:
    if not path_value:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        output_candidate = _key_action_output_dir(experiment_id) / path
        exp_candidate = _experiment_output_dir(experiment_id) / path
        path = output_candidate if output_candidate.exists() else exp_candidate
    return _experiment_file_api_path(path, experiment_id)


def _enrich_key_action_review_item_urls(experiment_id: str, item: Dict[str, Any]) -> Dict[str, Any]:
    enriched = copy.deepcopy(item)
    enriched["preview_urls"] = [
        url
        for url in (_key_action_review_url(experiment_id, path) for path in enriched.get("preview_paths") or [])
        if url
    ]
    enriched["clip_urls"] = [
        url
        for url in (_key_action_review_url(experiment_id, path) for path in enriched.get("clip_paths") or [])
        if url
    ]
    return enriched


def _key_action_quality_payload(experiment_id: str) -> Dict[str, Any]:
    try:
        from key_action_indexer.review_queue import build_quality_convergence  # type: ignore

        payload = build_quality_convergence(_key_action_output_dir(experiment_id))
    except Exception as exc:
        logger.exception("Key-action quality convergence failed for %s", experiment_id)
        payload = {
            "schema_version": "key_action_quality_convergence.error",
            "experiment_id": experiment_id,
            "status": "failed",
            "error": str(exc),
        }
    payload["experiment_id"] = experiment_id
    return payload


def _key_action_review_queue_payload(experiment_id: str) -> Dict[str, Any]:
    try:
        from key_action_indexer.review_queue import build_review_queue  # type: ignore

        payload = build_review_queue(
            _key_action_output_dir(experiment_id),
            material_candidates=_material_candidates_payload(experiment_id),
        )
    except Exception as exc:
        logger.exception("Key-action review queue failed for %s", experiment_id)
        payload = {
            "schema_version": "key_action_review_queue.error",
            "experiment_id": experiment_id,
            "summary": {"total": 0, "pending": 0, "approved": 0, "rejected": 0, "needs_review": 0},
            "items": [],
            "quality": _key_action_quality_payload(experiment_id),
            "error": str(exc),
        }
    payload["experiment_id"] = experiment_id
    payload["items"] = [
        _enrich_key_action_review_item_urls(experiment_id, item)
        for item in (payload.get("items") or [])
        if isinstance(item, dict)
    ]
    return payload


def _key_action_evidence_adapter_payload(experiment_id: str) -> Dict[str, Any]:
    output_dir = _key_action_output_dir(experiment_id)
    metadata = output_dir / "metadata"
    files = {
        "object_tracks": metadata / "object_tracks.jsonl",
        "panel_ocr": metadata / "panel_ocr.jsonl",
        "equipment_panel_states": metadata / "equipment_panel_states.jsonl",
        "liquid_state": metadata / "liquid_state.jsonl",
        "liquid_segmentation": metadata / "liquid_segmentation.jsonl",
        "container_state": metadata / "container_state.jsonl",
        "container_state_events": metadata / "container_state_events.jsonl",
        "model_observation_events": metadata / "model_observation_events.jsonl",
        "advanced_vision_evidence": metadata / "advanced_vision_evidence.jsonl",
    }
    counts = {name: _jsonl_row_count(path) for name, path in files.items()}
    try:
        from key_action_indexer.evidence_adapter_validation import validate_evidence_adapters  # type: ignore

        validation = validate_evidence_adapters(output_dir)
    except Exception as exc:
        logger.exception("Key-action evidence adapter validation failed for %s", experiment_id)
        validation = {
            "schema_version": "key_action_evidence_adapter_validation.error",
            "status": "fail",
            "error": str(exc),
        }
    return {
        "schema_version": "key_action_advanced_evidence_adapters.v1",
        "experiment_id": experiment_id,
        "metadata_dir": str(metadata),
        "protocol_doc": str(PROJECT_ROOT.parent / "docs" / "advanced_evidence_input_protocol.md"),
        "input_contracts": {
            "object_tracks.jsonl": "object trajectory, bbox track points, motion and identity confidence",
            "panel_ocr.jsonl": "equipment OCR, readout, button, knob, switch or control state rows",
            "liquid_state.jsonl": "liquid level, meniscus, flow, mask, volume or stream direction rows",
            "container_state.jsonl": "open/closed, cap/lid, color, liquid-level or container state rows",
        },
        "accepted_aliases": {
            "panel_ocr.jsonl": ["equipment_panel_states.jsonl", "equipment_panel_ocr.jsonl", "equipment_control_states.jsonl"],
            "liquid_state.jsonl": ["liquid_segmentation.jsonl", "liquid_level_events.jsonl", "liquid_flow_events.jsonl"],
            "container_state.jsonl": ["container_state_events.jsonl", "container_open_close_events.jsonl", "container_color_events.jsonl"],
        },
        "counts": counts,
        "validation": validation,
        "adapters": validation.get("adapters") if isinstance(validation, dict) else {},
        "summary": validation.get("summary") if isinstance(validation, dict) else {},
        "ready": bool(counts.get("model_observation_events") or counts.get("advanced_vision_evidence")),
    }


def _infer_key_action_start_time(*names: str) -> str:
    for name in names:
        match = re.search(r"(20\d{6})[-_]?(\d{6})", str(name or ""))
        if not match:
            continue
        date_text, time_text = match.groups()
        try:
            return datetime.strptime(f"{date_text}{time_text}", "%Y%m%d%H%M%S").astimezone().isoformat()
        except Exception:
            continue
    return _now_iso()


def _infer_key_action_camera_id(path_value: Optional[str], *, view: str) -> str:
    name = Path(str(path_value or "")).name.lower()
    if any(token in name for token in ("bottom_view", "usb", "front_view", "front-view", "front view")):
        return "bottom_view"
    if any(token in name for token in ("top_view", "rgb", "top-view", "top view")):
        return "top_view"
    if "bottom" in name and "top" not in name:
        return "bottom_view"
    if "top" in name:
        return "top_view"
    if view == "third_person":
        if "overview" in name:
            return "overview"
        return "third_person"
    if "first" in name or "fpv" in name or "ego" in name:
        return "first_person"
    return "first_person"


def _run_key_action_index_task(
    experiment_id: str,
    *,
    third_person_video_path: str,
    first_person_video_path: Optional[str],
    session_start_time: str,
    detection_config: Dict[str, Any],
) -> None:
    output_dir = _key_action_output_dir(experiment_id)
    detection_config = _with_default_key_action_yolo_config(detection_config)
    _write_key_action_status(
        experiment_id,
        {
            "status": "running",
            "progress": 0.1,
            "message": "Preparing key-action indexing inputs",
            "started_at": _now_iso(),
            "third_person_video_path": third_person_video_path,
            "first_person_video_path": first_person_video_path,
        },
    )
    try:
        try:
            from key_action_indexer.pipeline import run_pipeline  # type: ignore
            from key_action_indexer.validation import validate_video_source  # type: ignore
            from key_action_indexer.yolo_analysis import (  # type: ignore
                enrich_key_action_index_with_yolo,
                run_yolo_on_experiment_focus_clips,
                run_yolo_on_segment_clips,
            )
        except Exception as exc:
            raise RuntimeError(
                "Full key-action runner is not present in the recovered workspace. "
                "Uploaded videos were saved, but automatic YOLO segment generation cannot run yet."
            ) from exc

        output_dir.mkdir(parents=True, exist_ok=True)
        third_info = validate_video_source(third_person_video_path)
        first_info = validate_video_source(first_person_video_path) if first_person_video_path else None
        third_fps = third_info.get("fps") or 30
        first_fps = (first_info or {}).get("fps") or third_fps
        manifest = {
            "session_id": experiment_id,
            "session_start_time": session_start_time,
            "videos": {
                "third_person": {
                    "path": third_person_video_path,
                    "start_time": session_start_time,
                    "fps": third_fps,
                    "offset_sec": 0,
                    "role": "third_person",
                    "camera_id": _infer_key_action_camera_id(third_person_video_path, view="third_person"),
                }
            },
            "detection_config": detection_config,
            "output_dir": str(output_dir),
        }
        if first_person_video_path:
            manifest["videos"]["first_person"] = {
                "path": first_person_video_path,
                "start_time": session_start_time,
                "fps": first_fps,
                "offset_sec": 0,
                "role": "first_person",
                "camera_id": _infer_key_action_camera_id(first_person_video_path, view="first_person"),
            }
        manifest_path = output_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_key_action_status(experiment_id, {"progress": 0.35, "message": "Running key-action detection, clipping and indexing"})
        summary = run_pipeline(manifest_path, dry_run=False)
        try:
            views = ["third_person"] + (["first_person"] if first_person_video_path else [])
            model_paths_by_view = {
                "first_person": detection_config.get("yolo_first_person_model_path"),
                "third_person": detection_config.get("yolo_third_person_model_path"),
            }
            _write_key_action_status(
                experiment_id,
                {"progress": 0.88, "message": "Rendering YOLO annotated dual-view key-action clips"},
            )
            yolo_clip_summary = run_yolo_on_segment_clips(
                output_dir,
                model_path=detection_config.get("yolo_model_path"),
                project_root=PROJECT_ROOT,
                preferred_view=str(detection_config.get("yolo_preferred_view") or "first_person"),
                views=views,
                model_paths_by_view=model_paths_by_view,
                conf=float(detection_config.get("yolo_conf", 0.25)),
                iou=float(detection_config.get("yolo_iou", 0.45)),
                device=str(detection_config.get("yolo_device") or os.environ.get("KEY_ACTION_YOLO_DEVICE", "cpu")),
                detect_fps=float(os.environ.get("KEY_ACTION_YOLO_ANNOTATION_FPS", "5.0")),
                class_thresholds=detection_config.get("yolo_class_thresholds")
                if isinstance(detection_config.get("yolo_class_thresholds"), dict)
                else None,
            )
            summary["yolo_annotated_clips"] = yolo_clip_summary
            summary["yolo_index_enrichment"] = enrich_key_action_index_with_yolo(output_dir)
        except Exception as exc:
            logger.exception("YOLO annotated key-action clip generation failed for experiment %s", experiment_id)
            summary["yolo_annotated_clips"] = {"available": False, "error": str(exc)}
        candidate_summary: Optional[Dict[str, Any]] = None
        try:
            from key_action_indexer.material_references import build_yolo_material_candidates  # type: ignore

            vlm_assist_client, vlm_assist_config = _build_key_action_vlm_assist_client()
            summary["key_action_vlm_assist"] = vlm_assist_config
            _write_key_action_status(
                experiment_id,
                {
                    "progress": 0.94,
                    "message": "Building YOLO/VLM-assisted review-gated physical-action material candidates",
                    "key_action_vlm_assist": vlm_assist_config,
                },
            )
            candidate_summary = build_yolo_material_candidates(
                output_dir,
                archive_existing=False,
                rebuild_source=True,
                vlm_client=vlm_assist_client,
                enable_vlm=bool(vlm_assist_config.get("enabled")),
                max_vlm_groups=int(vlm_assist_config.get("max_groups") or 0),
                vlm_model_name=str(vlm_assist_config.get("model") or ""),
            )
        except Exception as exc:
            logger.exception("Key-action material candidate generation failed for experiment %s", experiment_id)
            candidate_summary = {"status": "failed", "error": str(exc)}
        _write_key_action_status(
            experiment_id,
            {
                "progress": 0.965,
                "message": "Staging review-gated key material candidates",
                "material_candidates": candidate_summary,
            },
        )
        material_auto_publish = {
            "status": "skipped",
            "reason": "candidate_review_required",
            "policy": "Keyframes, key clips, and professional reports stay in the candidate queue until manually approved.",
            "approved_count": 0,
        }
        try:
            from key_action_indexer.material_references import reset_material_references_to_approved_candidates  # type: ignore

            material_auto_publish["formal_reset"] = reset_material_references_to_approved_candidates(
                output_dir,
                approved_rows=[],
                merge_existing=False,
            )
        except Exception as exc:
            logger.exception("Failed to reset formal material references for candidate-first flow: %s", experiment_id)
            material_auto_publish["formal_reset"] = {"status": "failed", "error": str(exc)}
        if isinstance(candidate_summary, dict):
            candidate_summary["auto_publish"] = material_auto_publish
        summary["material_candidates"] = candidate_summary
        summary["material_auto_publish"] = material_auto_publish
        try:
            _write_key_action_status(
                experiment_id,
                {"progress": 0.975, "message": "Rendering continuous experiment-focus dual-view YOLO clips"},
            )
            focus_clip_summary = run_yolo_on_experiment_focus_clips(
                output_dir,
                model_path=detection_config.get("yolo_model_path"),
                project_root=PROJECT_ROOT,
                preferred_view=str(detection_config.get("yolo_preferred_view") or "first_person"),
                views=["third_person"] + (["first_person"] if first_person_video_path else []),
                model_paths_by_view={
                    "first_person": detection_config.get("yolo_first_person_model_path"),
                    "third_person": detection_config.get("yolo_third_person_model_path"),
                },
                conf=float(detection_config.get("yolo_conf", 0.25)),
                iou=float(detection_config.get("yolo_iou", 0.45)),
                device=str(detection_config.get("yolo_device") or os.environ.get("KEY_ACTION_YOLO_DEVICE", "cpu")),
                detect_fps=float(os.environ.get("KEY_ACTION_YOLO_EXPERIMENT_FOCUS_FPS", os.environ.get("KEY_ACTION_YOLO_ANNOTATION_FPS", "5.0"))),
                class_thresholds=detection_config.get("yolo_class_thresholds")
                if isinstance(detection_config.get("yolo_class_thresholds"), dict)
                else None,
            )
            summary["yolo_experiment_focus_clips"] = focus_clip_summary
        except Exception as exc:
            logger.exception("YOLO experiment-focus clip generation failed for experiment %s", experiment_id)
            summary["yolo_experiment_focus_clips"] = {"available": False, "error": str(exc)}
        try:
            from key_action_indexer.quality_gate import build_quality_gate  # type: ignore
            from key_action_indexer.reviewed_dataset import freeze_reviewed_dataset  # type: ignore

            reviewed_release = freeze_reviewed_dataset(output_dir)
            summary["reviewed_dataset"] = reviewed_release
            summary["reviewed_quality_gate"] = build_quality_gate(
                output_dir,
                output_path=output_dir / "reports" / "quality_gate.json",
            )
        except Exception as exc:
            logger.exception("Reviewed release convergence failed for experiment %s", experiment_id)
            summary["reviewed_dataset"] = {"available": False, "error": str(exc)}
        _write_json(output_dir / "pipeline_summary.json", summary)
        result = _key_action_results_payload(experiment_id)
        quality_payload = _key_action_quality_payload(experiment_id)
        quality_gate = quality_payload.get("quality_gate") if isinstance(quality_payload, dict) else None
        if not isinstance(quality_gate, dict):
            quality_gate = {}
        can_mark_complete = bool(quality_gate.get("can_mark_complete"))
        final_key_action_status = "completed" if can_mark_complete else "needs_review"
        final_task_status = "completed" if can_mark_complete else "partial_failed"
        final_message = (
            "Key-action pipeline and professional report completed"
            if can_mark_complete
            else "Key-action outputs generated; quality gate requires review before completion"
        )
        summary["quality_gate"] = quality_gate
        summary["quality_convergence"] = quality_payload

        exp = _load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or {}
        if isinstance(exp, dict) and exp:
            output_paths = {
                **(exp.get("output_paths") or {}),
                "key_action_index": str(output_dir),
            }
            professional_report = _generate_professional_report_for_experiment(
                experiment_id,
                output_paths=output_paths,
            )
            output_paths = _attach_professional_report_output_paths(
                experiment_id,
                output_paths,
                professional_report,
            )
            exp.setdefault("output_paths", {})
            exp["output_paths"] = output_paths
            exp["key_action_index"] = {
                "status": final_key_action_status,
                "segment_count": len(result.get("segments") or []),
                "output_dir": str(output_dir),
                "report": str(output_dir / "reports" / "mvp_validation_report.md"),
                "material_candidates": candidate_summary,
                "quality_gate": quality_gate,
            }
            exp.setdefault("metadata", {})["professional_report"] = professional_report
            completed_at = _now_iso()
            projected_step_groups = _key_action_step_groups_for_overview(experiment_id)
            task_id = exp.get("analysis_job_id") or exp.get("run_id") or f"key_action_{experiment_id}"
            exp["status"] = ExperimentStatus.ANALYZED.value if can_mark_complete else "partial_failed"
            exp["processing_stage"] = ProcessStage.OUTPUT_GENERATION.value
            exp["completed_at"] = completed_at if can_mark_complete else None
            exp["analyzed_at"] = completed_at
            exp["analysis_job_id"] = task_id
            exp["total_steps"] = len(projected_step_groups.get("candidate") or []) + len(projected_step_groups.get("inferred") or [])
            _save_experiment(exp)
            _persist_experiment_task_state(
                experiment_id,
                {
                    "task_id": task_id,
                    "experiment_id": experiment_id,
                    "status": final_task_status,
                    "current_stage": ProcessStage.OUTPUT_GENERATION.value,
                    "progress": 1.0 if can_mark_complete else 0.98,
                    "message": final_message,
                    "video_path": third_person_video_path,
                    "started_at": _read_key_action_status(experiment_id).get("started_at"),
                    "completed_at": completed_at if can_mark_complete else None,
                    "output_paths": output_paths,
                    "professional_report": professional_report,
                    "key_action_index": exp.get("key_action_index"),
                    "quality_gate": quality_gate,
                    "error_type": None,
                    "error_message": None,
                },
            )

        _write_key_action_status(
            experiment_id,
            {
                "status": final_key_action_status,
                "progress": 1.0 if can_mark_complete else 0.98,
                "message": final_message,
                "completed_at": _now_iso() if can_mark_complete else None,
                "summary": summary,
                "quality_gate": quality_gate,
                "material_candidates": candidate_summary,
            },
        )
    except Exception as exc:
        logger.exception("Key-action pipeline failed for experiment %s", experiment_id)
        _write_key_action_status(
            experiment_id,
            {
                "status": "failed",
                "progress": 1.0,
                "message": str(exc),
                "error": str(exc),
                "completed_at": _now_iso(),
            },
        )


def _resolve_material_diagnostic_path(path_value: Any) -> Optional[Path]:
    if not path_value:
        return None
    raw_path = str(path_value)
    normalized = raw_path.replace("\\", "/")
    normalized_lower = normalized.lower()
    for root_name in ("outputs", "uploads"):
        marker = f"{root_name}/experiments/"
        marker_index = normalized_lower.find(marker)
        if marker_index < 0:
            continue
        tail = normalized[marker_index + len(marker):]
        parts = [part for part in tail.split("/") if part]
        if not parts:
            continue
        experiment_root = (PROJECT_ROOT / root_name / "experiments").resolve()
        candidate = experiment_root.joinpath(*parts).resolve()
        try:
            candidate.relative_to(experiment_root)
        except ValueError:
            continue
        return candidate
    try:
        return _safe_project_path(raw_path, PROJECT_ROOT)
    except HTTPException:
        return None


def _material_path_exists(path_value: Any) -> bool:
    path = _resolve_material_diagnostic_path(path_value)
    return bool(path and path.exists() and path.is_file())


def _material_clip_is_playable(path_value: Any) -> bool:
    path = _resolve_material_diagnostic_path(path_value)
    return bool(path and path.exists() and path.is_file() and path.suffix.lower() in _PLAYABLE_CLIP_SUFFIXES)


def _material_duration_sec(item: Dict[str, Any]) -> float:
    for key in ("duration_sec", "clip_duration_sec"):
        value = item.get(key)
        if value is not None:
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    try:
        return max(0.0, float(item.get("time_end", 0.0)) - float(item.get("time_start", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _material_published_paths(item: Dict[str, Any]) -> Dict[str, Any]:
    paths = item.get("published_paths")
    return paths if isinstance(paths, dict) else {}


def _build_material_diagnostics(experiment_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_items = payload.get("items") if isinstance(payload, dict) else []
    items = [item for item in raw_items if isinstance(item, dict)]
    warnings_by_type: Dict[str, int] = {}
    warnings_count = 0
    broken_clip_paths: List[str] = []
    clip_count = 0
    missing_clip_count = 0
    missing_preview_count = 0
    missing_keyframe_count = 0
    playable_clip_count = 0
    long_clip_count = 0
    durations: List[float] = []
    evidence_items: List[Dict[str, Any]] = []

    for item in items:
        paths = _material_published_paths(item)
        clip_path = paths.get("clip") or item.get("clip_file_path") or item.get("clip_path")
        preview_path = paths.get("preview") or item.get("preview_path")
        if not preview_path:
            preview_path = paths.get("keyframe") or item.get("frame_path")
        keyframe_paths = paths.get("keyframes") or item.get("keyframe_paths") or []
        if not isinstance(keyframe_paths, list):
            keyframe_paths = [keyframe_paths]
        if not keyframe_paths and paths.get("keyframe"):
            keyframe_paths = [paths.get("keyframe")]

        if clip_path:
            clip_count += 1
        if not _material_path_exists(clip_path):
            missing_clip_count += 1
            if len(broken_clip_paths) < 20:
                broken_clip_paths.append(str(clip_path) if clip_path else "")
        if not _material_path_exists(preview_path):
            missing_preview_count += 1
        if not keyframe_paths or not any(_material_path_exists(path) for path in keyframe_paths):
            missing_keyframe_count += 1
        if _material_clip_is_playable(clip_path):
            playable_clip_count += 1

        duration_sec = _material_duration_sec(item)
        durations.append(duration_sec)
        if duration_sec > _MATERIAL_LONG_CLIP_THRESHOLD_SEC:
            long_clip_count += 1

        warnings = item.get("warnings") or []
        if isinstance(warnings, str):
            warnings = [warnings]
        for warning in warnings:
            warning_type = str(warning)
            warnings_count += 1
            warnings_by_type[warning_type] = warnings_by_type.get(warning_type, 0) + 1

        material_path = clip_path or preview_path
        material_url = _experiment_file_api_path(Path(str(material_path)), experiment_id) if material_path else None
        yolo_recheck = item.get("yolo_recheck") if isinstance(item.get("yolo_recheck"), dict) else {}
        vlm_semantics = item.get("vlm_semantics") if isinstance(item.get("vlm_semantics"), dict) else {}
        source_candidate_file = item.get("source_candidate_file") or item.get("source_reference_file")
        evidence_items.append(
            {
                "item_id": item.get("item_id") or item.get("material_id") or item.get("candidate_id") or item.get("event_id"),
                "candidate_id": item.get("candidate_id"),
                "candidate_group_id": item.get("candidate_group_id"),
                "asset_kind": item.get("asset_kind") or item.get("material_type"),
                "display_name": item.get("display_name") or item.get("action_name") or item.get("event_type"),
                "formal_material_reference": bool(item.get("formal_material_reference")),
                "review_status": item.get("review_status"),
                "approved_by": item.get("approved_by"),
                "approved_at": item.get("approved_at"),
                "yolo_recheck_status": yolo_recheck.get("status"),
                "yolo_valid_evidence_count": yolo_recheck.get("valid_evidence_count"),
                "vlm_status": vlm_semantics.get("status"),
                "vlm_model": vlm_semantics.get("model"),
                "vlm_description": vlm_semantics.get("description"),
                "source_file": item.get("source_file"),
                "source_candidate_file": source_candidate_file,
                "stored_file": item.get("stored_file") or material_path,
                "material_path": material_path,
                "material_exists": _material_path_exists(material_path),
                "material_url": material_url,
                "url_accessible": bool(material_url and _material_path_exists(material_path)),
            }
        )

    return {
        "experiment_id": experiment_id,
        "published_total": int(payload.get("total", len(items))) if isinstance(payload, dict) else len(items),
        "clip_count": clip_count,
        "missing_clip_count": missing_clip_count,
        "missing_preview_count": missing_preview_count,
        "missing_keyframe_count": missing_keyframe_count,
        "avg_duration_sec": round(sum(durations) / len(durations), 3) if durations else 0.0,
        "long_clip_count": long_clip_count,
        "long_clip_threshold_sec": _MATERIAL_LONG_CLIP_THRESHOLD_SEC,
        "warnings_count": warnings_count,
        "warnings_by_type": warnings_by_type,
        "playable_clip_count": playable_clip_count,
        "broken_clip_paths": broken_clip_paths,
        "formal_material_reference_count": sum(1 for item in evidence_items if item.get("formal_material_reference")),
        "url_accessible_count": sum(1 for item in evidence_items if item.get("url_accessible")),
        "evidence_items": evidence_items,
    }


def _experiment_model_status() -> Dict[str, Any]:
    yolo_path = Path(RUNTIME_SETTINGS.yolo_model_path) if (RUNTIME_SETTINGS and RUNTIME_SETTINGS.yolo_model_path) else None
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    from labsopguard.asr import asr_diagnostics
    from labsopguard.embeddings import embedding_diagnostics
    from labsopguard.detectors import yolo26_diagnostics

    asr_status = asr_diagnostics()
    embedding_status = embedding_diagnostics()
    detector_status = yolo26_diagnostics(str(yolo_path) if yolo_path else None)
    return {
        "yolo_model_path": str(yolo_path) if yolo_path else None,
        "yolo_model_exists": bool(yolo_path and yolo_path.exists()),
        "yolo_model_name": yolo_path.name if yolo_path else None,
        "yolo26_status": detector_status,
        "vlm_model": os.environ.get("KEY_ACTION_VLM_MODEL")
        or os.environ.get("QWEN_VL_MODEL")
        or os.environ.get("VLM_MODEL")
        or "qwen3.6-plus",
        "vlm_enabled": bool(dashscope_key),
        "dashscope_configured": bool(dashscope_key),
        **asr_status,
        **embedding_status,
        "fallback_mode": bool(embedding_status.get("fallback_mode")),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _probe_video_metadata(video_path: Path, video_index: int) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "video_index": video_index,
        "video_path": str(video_path),
        "filename": video_path.name,
        "size_bytes": video_path.stat().st_size if video_path.exists() else None,
        "source_type": "file",
        "ingest_mode": "file",
    }
    cap = cv2.VideoCapture(str(video_path))
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_sec = round(frame_count / fps, 3) if fps > 0 else 0.0
            metadata.update(
                {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                }
            )
    finally:
        cap.release()
    return metadata


def _looks_like_live_source(source: Optional[str]) -> bool:
    if not source:
        return False
    lowered = str(source).lower()
    return lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")) or lowered.isdigit()


def _is_local_file_source(source: Optional[str]) -> bool:
    if not source:
        return False
    return Path(str(source)).exists()


def _classify_processing_error(exc: Exception) -> Dict[str, str]:
    message = str(exc)
    lowered = message.lower()
    if "upload" in lowered and "missing" in lowered:
        return {"error_type": "upload_missing", "error_message": message}
    if "video file not found" in lowered or "cannot open video" in lowered or "video not found" in lowered:
        return {"error_type": "video_not_found", "error_message": message}
    if "dashscope" in lowered or "qwen" in lowered or "openai" in lowered:
        return {"error_type": "model_call_failed", "error_message": message}
    if "write" in lowered or "permission" in lowered:
        return {"error_type": "output_write_failed", "error_message": message}
    if "analysis" in lowered or "pipeline" in lowered or "annotated" in lowered:
        return {"error_type": "pipeline_invoke_failed", "error_message": message}
    return {"error_type": "unexpected_exception", "error_message": message}


def _experiment_task_state(experiment_id: str) -> Dict[str, Any]:
    if EXPERIMENT_TASK_STORE is None:
        return {}
    task_file = EXPERIMENT_TASK_STORE.base_dir / f"{experiment_id}.json"
    if not task_file.exists():
        return {}
    try:
        return json.loads(task_file.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring invalid experiment task state %s: %s", task_file, exc)
        return {}


def _merge_experiment_with_task(exp: Dict[str, Any]) -> Dict[str, Any]:
    merged = _normalize_experiment_dict(exp)
    task_state = _experiment_task_state(merged["experiment_id"])
    if task_state:
        merged["task"] = task_state
        merged["analysis_job_id"] = task_state.get("task_id", merged.get("analysis_job_id"))
        merged["processing_stage"] = task_state.get("current_stage", merged.get("processing_stage"))
        merged["processing_error"] = task_state.get("error_message", merged.get("processing_error"))
        merged["output_paths"] = task_state.get("output_paths", merged.get("output_paths", {}))
        task_status = task_state.get("status")
        if task_status:
            merged["status"] = (
                ExperimentStatus.ANALYZED.value
                if task_status == "completed"
                else task_status
            )
        merged["started_at"] = task_state.get("started_at", merged.get("started_at"))
        merged["completed_at"] = task_state.get("completed_at", merged.get("completed_at"))
    return merged


def _build_experiment_task_payload(
    task_id: str,
    experiment_id: str,
    *,
    status: str,
    current_stage: str,
    progress: float,
    video_path: str,
    output_paths: Optional[Dict[str, str]] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "task_id": task_id,
        "experiment_id": experiment_id,
        "status": status,
        "current_stage": current_stage,
        "progress": progress,
        "video_path": video_path,
        "error_type": error_type,
        "error_message": error_message,
        "started_at": _now_iso() if status in {"running", "completed", "failed"} else None,
        "completed_at": _now_iso() if status in {"completed", "failed"} else None,
        "output_paths": output_paths or {},
    }
    return payload


def _persist_experiment_task_state(experiment_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if EXPERIMENT_TASK_STORE is None:
        return payload
    existing = _experiment_task_state(experiment_id)
    if existing:
        update_fields = dict(payload)
        current = EXPERIMENT_TASK_STORE.update(experiment_id, **update_fields)
    else:
        current = EXPERIMENT_TASK_STORE.create(experiment_id, payload)
    return current


EXPECTED_KEY_ACTION_DUAL_VIEW_SOURCES = [
    {"camera_id": "top_view", "source_group": "key_action_dual_view", "source_type": "file", "view_type": "third_person"},
    {"camera_id": "bottom_view", "source_group": "key_action_dual_view", "source_type": "file", "view_type": "first_person"},
]


def _ensure_experiment_run_metadata(exp: Dict[str, Any]) -> Dict[str, Any]:
    exp.setdefault("metadata", {})
    if not exp.get("run_id"):
        exp["run_id"] = f"run_{uuid.uuid4().hex[:12]}"
    if not exp["metadata"].get("global_t0_wall_time"):
        exp["metadata"]["global_t0_wall_time"] = exp.get("created_at") or _now_iso()
    if not exp["metadata"].get("global_timeline_schema"):
        exp["metadata"]["global_timeline_schema"] = "experiment_timeline.global.v1"
    return exp


def _infer_stream_contract(descriptor: Dict[str, Any], index: int) -> Dict[str, Any]:
    camera_id = str(descriptor.get("camera_id") or f"camera_{index:02d}")
    source_type = str(descriptor.get("source_type") or descriptor.get("ingest_mode") or "file").lower()
    view_type = str(descriptor.get("view_type") or descriptor.get("role") or "").strip().lower()
    camera_key = camera_id.lower()
    if not view_type:
        view_type = "third_person" if camera_key in {"top", "top_view", "overview", "third_person"} else "first_person"
    source_group = str(descriptor.get("source_group") or "").strip().lower()
    if not source_group:
        source_group = "key_action_dual_view"
    normalized_source_type = "file" if source_type in {"usb", "wired"} else source_type
    return {
        "camera_id": camera_id,
        "source_group": source_group,
        "source_type": normalized_source_type,
        "view_type": view_type,
        "role": descriptor.get("role") or view_type,
    }


def _experiment_stream_descriptors(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    descriptors = list(exp.get("video_inputs") or exp.get("video_metadata") or [])
    if not descriptors and exp.get("video_paths"):
        descriptors = [
            {
                "video_index": index,
                "video_path": path,
                "source": path,
                "source_type": "file",
                "camera_id": f"camera_{index:02d}",
                "start_offset_sec": 0.0,
            }
            for index, path in enumerate(exp.get("video_paths") or [])
        ]
    return [dict(item) for item in descriptors if isinstance(item, dict)]


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_confidence(value: Any, default: Optional[float] = None) -> Optional[float]:
    numeric = _optional_float(value)
    if numeric is None:
        return default
    return max(0.0, min(1.0, numeric))


def _is_semantic_sync_method(value: Any) -> bool:
    method = str(value or "").strip().lower()
    return method in {"multimodal_semantic", "semantic", "semantic_sync"} or "semantic" in method


def _semantic_sync_anchors_from_descriptor(descriptor: Dict[str, Any], profile: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    sync_profile = descriptor.get("sync_profile") if isinstance(descriptor.get("sync_profile"), dict) else {}
    anchors = descriptor.get("sync_anchors") if isinstance(descriptor.get("sync_anchors"), list) else []
    profile = profile or {}
    semantic_method = _is_semantic_sync_method(descriptor.get("sync_method")) or _is_semantic_sync_method(profile.get("alignment_method")) or _is_semantic_sync_method(sync_profile.get("method"))
    semantic_anchors: List[Dict[str, Any]] = []
    for index, anchor in enumerate(anchors):
        if not isinstance(anchor, dict):
            continue
        anchor_method = anchor.get("method") or anchor.get("anchor_type") or anchor.get("type")
        if semantic_method or _is_semantic_sync_method(anchor_method):
            semantic_anchors.append(
                {
                    **anchor,
                    "anchor_index": anchor.get("anchor_index", index),
                    "method": anchor.get("method") or "multimodal_semantic",
                }
            )
    return semantic_anchors


def _sync_profile_from_descriptor(descriptor: Dict[str, Any], *, processed: bool = False) -> Dict[str, Any]:
    sync_profile = descriptor.get("sync_profile") if isinstance(descriptor.get("sync_profile"), dict) else {}
    sync_method = str(descriptor.get("sync_method") or descriptor.get("alignment_method") or sync_profile.get("method") or "").strip()
    sync_method_key = sync_method.lower()
    sync_anchors = descriptor.get("sync_anchors") if isinstance(descriptor.get("sync_anchors"), list) else []
    offset_source = str(descriptor.get("offset_source") or sync_profile.get("method") or "").strip().lower()
    clock_scale = _optional_float(descriptor.get("clock_scale", sync_profile.get("clock_scale")))
    clock_drift_ppm = _optional_float(descriptor.get("clock_drift_ppm", sync_profile.get("drift_ppm")))
    start_offset = _optional_float(descriptor.get("start_offset_sec", sync_profile.get("offset_sec")))
    confidence = _bounded_confidence(
        descriptor.get("sync_confidence", descriptor.get("alignment_confidence", sync_profile.get("confidence")))
    )
    explicit_status = str(descriptor.get("alignment_status") or "").strip().lower()
    if explicit_status in {"pending", "explicit", "shared_recording", "calibrated"}:
        status = explicit_status
    elif offset_source in {"explicit", "shared_recording", "shared_recording_session", "calibrated"}:
        status = "shared_recording" if "shared_recording" in offset_source else offset_source
    elif sync_method_key in {"shared_recording_session", "shared_recording", "recording_session"} or descriptor.get("recording_frames") is not None:
        status = "shared_recording"
    elif (
        sync_method_key in {"calibrated", "visual_calibration", "hardware_timecode", "sync_board"}
        or (_is_semantic_sync_method(sync_method_key) and bool(sync_anchors or sync_profile.get("anchor_count")))
        or sync_method_key.startswith("calibrated")
        or sync_method_key.startswith("auto_visual")
        or "drift_corrected" in sync_method_key
        or descriptor.get("hardware_timecode_start_sec") is not None
        or descriptor.get("sync_board_offset_sec") is not None
    ):
        status = "calibrated"
    elif sync_anchors or sync_method or (start_offset not in (None, 0.0)) or confidence is not None or processed:
        status = "explicit"
    else:
        status = "pending"

    if not sync_method:
        if status == "shared_recording":
            sync_method = "shared_recording_session"
        elif status == "calibrated":
            sync_method = "calibrated"
        elif status == "explicit":
            sync_method = "explicit_offset"
        else:
            sync_method = "pending"

    if confidence is None:
        confidence = {
            "pending": 0.0,
            "explicit": 0.7,
            "shared_recording": 0.85,
            "calibrated": 0.95,
        }.get(status, 0.0)

    semantic_anchors = _semantic_sync_anchors_from_descriptor(descriptor, {"alignment_method": sync_method})
    semantic_anchor_count = len(semantic_anchors)
    if semantic_anchor_count == 0 and _is_semantic_sync_method(sync_method):
        semantic_anchor_count = int(sync_profile.get("anchor_count") or len(sync_anchors) or 0)
    return {
        "alignment_status": status,
        "alignment_method": sync_method,
        "alignment_confidence": confidence,
        "sync_confidence": confidence,
        "sync_anchors": sync_anchors,
        "sync_anchor_count": int(sync_profile.get("anchor_count") or len(sync_anchors) or 0),
        "semantic_sync_anchor_count": semantic_anchor_count,
        "has_semantic_sync": bool(semantic_anchor_count or _is_semantic_sync_method(sync_method)),
        "local_zero_global_sec": start_offset if start_offset is not None else 0.0,
        "clock_scale": clock_scale if clock_scale is not None else 1.0,
        "clock_drift_ppm": clock_drift_ppm,
        "sync_group": descriptor.get("sync_group"),
        "offset_source": offset_source or None,
        "sync_profile": sync_profile,
        "residual_error_sec": _optional_float(sync_profile.get("residual_error_sec")),
        "pending_reason": "no_sync_anchors_or_offsets" if status == "pending" else None,
    }


def _aggregate_alignment_status(items: List[Dict[str, Any]]) -> str:
    statuses = {str(item.get("alignment_status") or "pending") for item in items}
    if not statuses:
        return "pending"
    if len(statuses) == 1:
        return next(iter(statuses))
    return "mixed"


def _semantic_sync_payload(
    *,
    experiment_id: str,
    run_id: str,
    streams: List[Dict[str, Any]],
    descriptors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    anchors: List[Dict[str, Any]] = []
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            continue
        profile = _sync_profile_from_descriptor(descriptor)
        contract = _infer_stream_contract(descriptor, index)
        stream_id = descriptor.get("asset_id") or descriptor.get("media_asset_id") or descriptor.get("stream_id") or contract["camera_id"]
        for anchor in _semantic_sync_anchors_from_descriptor(descriptor, profile):
            anchors.append(
                {
                    "anchor_id": anchor.get("anchor_id") or f"{contract['camera_id']}:semantic:{len(anchors):04d}",
                    "experiment_id": experiment_id,
                    "run_id": run_id,
                    "camera_id": contract["camera_id"],
                    "stream_id": stream_id,
                    "video_index": descriptor.get("video_index", index),
                    "source_group": contract["source_group"],
                    "view_type": contract["view_type"],
                    "method": anchor.get("method") or profile["alignment_method"],
                    "local_time_sec": anchor.get("local_time_sec"),
                    "reference_time_sec": anchor.get("reference_time_sec"),
                    "global_time_sec": anchor.get("reference_time_sec", anchor.get("global_time_sec")),
                    "confidence": _bounded_confidence(anchor.get("confidence"), profile["alignment_confidence"]),
                    "semantic_label": anchor.get("semantic_label") or anchor.get("label") or anchor.get("event_type"),
                    "description": anchor.get("description") or anchor.get("text"),
                    "evidence_refs": anchor.get("evidence_refs") or [],
                }
            )
    return {
        "schema_version": "semantic_sync_anchors.v1",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "method": "multimodal_semantic",
        "total": len(anchors),
        "semantic_events": [],
        "sync_anchors": anchors,
        "alignment_streams": [
            {
                "camera_id": item.get("camera_id"),
                "video_index": item.get("video_index"),
                "alignment_method": item.get("alignment_method"),
                "alignment_status": item.get("alignment_status"),
                "semantic_sync_anchor_count": item.get("semantic_sync_anchor_count", 0),
                "has_semantic_sync": bool(item.get("has_semantic_sync")),
            }
            for item in streams
        ],
        "anchors": anchors,
    }


def _semantic_sync_artifact_payload(
    *,
    experiment_id: str,
    run_id: str,
    streams: List[Dict[str, Any]],
    descriptors: List[Dict[str, Any]],
    semantic_sync: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(semantic_sync, dict) and semantic_sync:
        payload = dict(semantic_sync)
        payload.setdefault("schema_version", "multimodal_semantic_sync.v1")
        payload.setdefault("experiment_id", experiment_id)
        payload.setdefault("run_id", run_id)
        payload.setdefault("method", "multimodal_semantic")
        anchors = payload.get("sync_anchors")
        if not isinstance(anchors, list):
            anchors = payload.get("anchors") if isinstance(payload.get("anchors"), list) else []
            payload["sync_anchors"] = anchors
        payload.setdefault("anchors", anchors)
        payload.setdefault("total", len(anchors))
        payload.setdefault("semantic_events", [])
        payload.setdefault(
            "alignment_streams",
            [
                {
                    "camera_id": item.get("camera_id"),
                    "video_index": item.get("video_index"),
                    "alignment_method": item.get("alignment_method"),
                    "alignment_status": item.get("alignment_status"),
                    "semantic_sync_anchor_count": item.get("semantic_sync_anchor_count", 0),
                    "has_semantic_sync": bool(item.get("has_semantic_sync")),
                }
                for item in streams
            ],
        )
        return payload
    return _semantic_sync_payload(
        experiment_id=experiment_id,
        run_id=run_id,
        streams=streams,
        descriptors=descriptors,
    )


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _timeline_to_dict(timeline: Any) -> Dict[str, Any]:
    if timeline is None:
        return {}
    if isinstance(timeline, dict):
        return timeline
    if hasattr(timeline, "to_dict"):
        return timeline.to_dict()
    return {}


def _stream_infos_from_timeline(timeline: Any) -> List[Dict[str, Any]]:
    timeline_dict = _timeline_to_dict(timeline)
    metadata = timeline_dict.get("metadata") if isinstance(timeline_dict, dict) else {}
    streams = (metadata.get("video_streams") if isinstance(metadata, dict) else []) or []
    return [dict(item) for item in streams if isinstance(item, dict)]


def _build_transcript_segments(exp: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    stream_profiles: List[Dict[str, Any]] = []
    for descriptor in _experiment_stream_descriptors(exp):
        if not isinstance(descriptor, dict):
            continue
        stream_profiles.append({**descriptor, "_alignment_profile": _sync_profile_from_descriptor(descriptor, processed=False)})
    for index, item in enumerate(exp.get("context_inputs") or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        kind = str(item.get("kind") or item.get("source_type") or "conversation").lower()
        timestamp = item.get("timestamp_sec", item.get("start_time_sec"))
        local_timestamp = item.get("local_timestamp_sec")
        alignment_source = item
        if timestamp is None and local_timestamp is not None:
            for descriptor in stream_profiles:
                if item.get("video_index") is not None and str(descriptor.get("video_index")) == str(item.get("video_index")):
                    alignment_source = descriptor
                    break
                if item.get("video_asset_id") and str(descriptor.get("asset_id") or descriptor.get("media_asset_id") or "") == str(item.get("video_asset_id")):
                    alignment_source = descriptor
                    break
                if item.get("camera_id") and str(descriptor.get("camera_id") or "") == str(item.get("camera_id")):
                    alignment_source = descriptor
                    break
        alignment_profile = (
            alignment_source.get("_alignment_profile")
            if isinstance(alignment_source, dict) and isinstance(alignment_source.get("_alignment_profile"), dict)
            else _sync_profile_from_descriptor(alignment_source, processed=timestamp is not None)
        )
        if timestamp is None and local_timestamp is not None:
            local_float = _optional_float(local_timestamp)
            if local_float is not None:
                timestamp = round(local_float * float(alignment_profile["clock_scale"]) + float(alignment_profile["local_zero_global_sec"]), 3)
        segments.append(
            {
                "schema_version": "transcript_segment.v1",
                "segment_id": item.get("segment_id") or item.get("event_id") or f"ctx_{index:04d}",
                "experiment_id": exp.get("experiment_id"),
                "run_id": exp.get("run_id"),
                "kind": kind,
                "text": text,
                "global_start_sec": timestamp,
                "global_end_sec": item.get("end_time_sec"),
                "local_timestamp_sec": local_timestamp,
                "camera_id": item.get("camera_id"),
                "video_index": item.get("video_index"),
                "video_asset_id": item.get("video_asset_id"),
                "source_file": item.get("source_file") or item.get("audio_path"),
                "alignment_status": item.get("alignment_status") or alignment_profile["alignment_status"],
                "alignment_method": item.get("alignment_method") or alignment_profile["alignment_method"],
                "alignment_confidence": item.get("alignment_confidence") or alignment_profile["alignment_confidence"],
                "sync_confidence": item.get("sync_confidence") or alignment_profile["sync_confidence"],
                "local_zero_global_sec": alignment_profile["local_zero_global_sec"],
                "clock_scale": alignment_profile["clock_scale"],
                "clock_drift_ppm": alignment_profile["clock_drift_ppm"],
                "offset_source": alignment_profile["offset_source"],
                "residual_error_sec": alignment_profile["residual_error_sec"],
                "sync_group": alignment_profile["sync_group"],
            }
        )
    return segments


def _material_stream_v2_rows(
    *,
    experiment_id: str,
    run_id: str,
    material_stream: List[Any],
    exp: Dict[str, Any],
    timeline: Any,
) -> List[Dict[str, Any]]:
    timeline_dict = _timeline_to_dict(timeline)
    steps = (timeline_dict.get("steps", []) if isinstance(timeline_dict, dict) else []) or []
    asset_metadata: Dict[str, Dict[str, Any]] = {}
    for asset in exp.get("video_assets") or []:
        if isinstance(asset, dict) and asset.get("asset_id"):
            asset_metadata[str(asset["asset_id"])] = dict(asset.get("metadata") or {})
            asset_metadata[str(asset["asset_id"])]["file_path"] = asset.get("file_path")
    for descriptor in _experiment_stream_descriptors(exp):
        keys = [
            descriptor.get("asset_id"),
            descriptor.get("media_asset_id"),
            descriptor.get("stream_id"),
            descriptor.get("camera_id"),
        ]
        if descriptor.get("video_index") is not None:
            keys.append(str(descriptor.get("video_index")))
        for key in keys:
            if key is not None:
                asset_metadata.setdefault(str(key), {}).update(descriptor)
    for stream in _stream_infos_from_timeline(timeline):
        asset_id = stream.get("asset_id")
        if asset_id:
            asset_metadata.setdefault(str(asset_id), {}).update(stream)

    rows: List[Dict[str, Any]] = []
    for item in material_stream:
        data = item.to_dict() if hasattr(item, "to_dict") else dict(item) if isinstance(item, dict) else {}
        if not data:
            continue
        global_time = data.get("global_timestamp_sec", data.get("global_time_sec", data.get("timestamp_sec")))
        local_time = data.get("local_timestamp_sec")
        asset_meta = asset_metadata.get(str(data.get("media_asset_id") or data.get("stream_id") or ""), {})
        camera_id = data.get("camera_id") or asset_meta.get("camera_id")
        if not asset_meta and camera_id:
            asset_meta = asset_metadata.get(str(camera_id), {})
        step_refs = []
        try:
            global_float = float(global_time)
        except (TypeError, ValueError):
            global_float = None
        if global_float is not None:
            for step in steps:
                if not isinstance(step, dict):
                    continue
                start = step.get("start_time_sec")
                end = step.get("end_time_sec", start)
                try:
                    if float(start) <= global_float <= float(end if end is not None else start):
                        step_refs.append(step.get("step_id") or step.get("id"))
                except (TypeError, ValueError):
                    continue
        sync_profile = asset_meta.get("sync_profile") if isinstance(asset_meta.get("sync_profile"), dict) else {}
        alignment_profile = _sync_profile_from_descriptor(asset_meta, processed=bool(sync_profile))
        alignment_confidence = (
            data.get("alignment_confidence")
            or asset_meta.get("alignment_confidence")
            or sync_profile.get("confidence")
            or alignment_profile["alignment_confidence"]
        )
        rows.append(
            {
                "schema_version": "material_stream.v2",
                "item_id": data.get("item_id"),
                "experiment_id": experiment_id,
                "run_id": run_id,
                "global_time_sec": global_time,
                "local_time_sec": local_time,
                "camera_id": camera_id,
                "view_type": data.get("view_type") or asset_meta.get("view_type"),
                "source_group": data.get("source_group") or asset_meta.get("source_group"),
                "source_type": data.get("source_type") or asset_meta.get("source_type"),
                "source_uri": data.get("frame_bgr_path") or asset_meta.get("file_path") or asset_meta.get("video_path"),
                "media_asset_id": data.get("media_asset_id"),
                "stream_id": data.get("stream_id"),
                "frame_index": data.get("frame_id"),
                "local_frame_index": data.get("local_frame_id"),
                "alignment_status": data.get("alignment_status") or alignment_profile["alignment_status"],
                "alignment_method": data.get("alignment_method") or alignment_profile["alignment_method"],
                "alignment_confidence": alignment_confidence,
                "sync_confidence": data.get("sync_confidence") or alignment_profile["sync_confidence"],
                "sync_group": data.get("sync_group") or alignment_profile.get("sync_group") or asset_meta.get("sync_group"),
                "sync_anchor_count": alignment_profile["sync_anchor_count"],
                "semantic_sync_anchor_count": alignment_profile["semantic_sync_anchor_count"],
                "has_semantic_sync": alignment_profile["has_semantic_sync"],
                "local_zero_global_sec": alignment_profile["local_zero_global_sec"],
                "clock_scale": alignment_profile["clock_scale"],
                "clock_drift_ppm": alignment_profile["clock_drift_ppm"],
                "offset_source": alignment_profile["offset_source"],
                "residual_error_sec": alignment_profile["residual_error_sec"],
                "detected_objects": data.get("detected_objects") or [],
                "object_labels": data.get("object_labels") or [],
                "detected_activities": data.get("detected_activities") or [],
                "scene_description": data.get("scene_description"),
                "is_key_frame": bool(data.get("is_key_frame")),
                "key_frame_reason": data.get("key_frame_reason"),
                "change_score": data.get("change_score"),
                "transcript_refs": data.get("linked_context_event_ids") or [],
                "transcript_segment": data.get("transcript_segment"),
                "step_refs": [ref for ref in step_refs if ref],
                "analysis": data.get("analysis") or {},
            }
        )
    return rows


def _write_experiment_run_artifacts(
    experiment_id: str,
    exp: Dict[str, Any],
    *,
    material_stream: Optional[List[Any]] = None,
    timeline: Any = None,
    semantic_sync: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    exp = _ensure_experiment_run_metadata(exp)
    run_id = str(exp["run_id"])
    exp_dir = _experiment_output_dir(experiment_id)
    exp_dir.mkdir(parents=True, exist_ok=True)
    descriptors = _experiment_stream_descriptors(exp)
    processed_streams = _stream_infos_from_timeline(timeline)
    registered_streams = []
    source_streams = processed_streams or descriptors
    for index, descriptor in enumerate(source_streams):
        if not isinstance(descriptor, dict):
            continue
        contract = _infer_stream_contract(descriptor, index)
        alignment_profile = _sync_profile_from_descriptor(descriptor, processed=bool(processed_streams))
        registered_streams.append(
            {
                **contract,
                "video_index": descriptor.get("video_index", index),
                "media_asset_id": descriptor.get("asset_id") or descriptor.get("media_asset_id"),
                "video_path": descriptor.get("video_path") or descriptor.get("source"),
                "filename": descriptor.get("filename"),
                "fps": descriptor.get("fps"),
                "duration_sec": descriptor.get("duration_sec"),
                "start_offset_sec": alignment_profile["local_zero_global_sec"],
                "clock_scale": alignment_profile["clock_scale"],
                "clock_drift_ppm": alignment_profile["clock_drift_ppm"],
                "sync_group": alignment_profile["sync_group"],
                "sync_anchors": alignment_profile["sync_anchors"],
                "sync_anchor_count": alignment_profile["sync_anchor_count"],
                "sync_confidence": alignment_profile["sync_confidence"],
                "alignment_status": alignment_profile["alignment_status"],
                "alignment_method": alignment_profile["alignment_method"],
                "alignment_confidence": alignment_profile["alignment_confidence"],
                "semantic_sync_anchor_count": alignment_profile["semantic_sync_anchor_count"],
                "has_semantic_sync": alignment_profile["has_semantic_sync"],
                "offset_source": alignment_profile["offset_source"],
                "residual_error_sec": alignment_profile["residual_error_sec"],
                "sync_profile": alignment_profile["sync_profile"],
                "is_live_source": descriptor.get("is_live_source", False),
            }
        )

    run_manifest = {
        "schema_version": "experiment_run.v1",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "global_t0_wall_time": exp.get("metadata", {}).get("global_t0_wall_time"),
        "status": exp.get("status"),
        "source_contract": {
            "key_action_dual_view": {
                "description": "Dual-view key-action inputs only; PTZ and multi-camera transport live outside this project.",
                "expected_sources": EXPECTED_KEY_ACTION_DUAL_VIEW_SOURCES,
            },
        },
        "artifact_paths": {
            "stream_manifest": "stream_manifest.json",
            "timeline_alignment": "timeline_alignment.json",
            "semantic_sync_anchors": "semantic_sync_anchors.json",
            "material_stream_v2": "material_stream.v2.jsonl",
            "transcript_segments": "transcript_segments.jsonl",
        },
    }
    stream_manifest = {
        "schema_version": "stream_manifest.v1",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "alignment_status": _aggregate_alignment_status(registered_streams),
        "alignment_counts": {
            status: sum(1 for item in registered_streams if item.get("alignment_status") == status)
            for status in ("pending", "explicit", "shared_recording", "calibrated")
        },
        "expected_key_action_dual_view_count": 2,
        "expected_sources": [*EXPECTED_KEY_ACTION_DUAL_VIEW_SOURCES],
        "registered_streams": registered_streams,
    }
    alignment_streams = []
    for index, descriptor in enumerate(source_streams):
        if not isinstance(descriptor, dict):
            continue
        contract = _infer_stream_contract(descriptor, index)
        alignment_profile = _sync_profile_from_descriptor(descriptor, processed=bool(processed_streams))
        alignment_streams.append(
            {
                **contract,
                "video_index": descriptor.get("video_index", index),
                "media_asset_id": descriptor.get("asset_id") or descriptor.get("media_asset_id"),
                **alignment_profile,
            }
        )
    timeline_alignment = {
        "schema_version": "timeline_alignment.v1",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "alignment_status": _aggregate_alignment_status(alignment_streams),
        "alignment_counts": {
            status: sum(1 for item in alignment_streams if item.get("alignment_status") == status)
            for status in ("pending", "explicit", "shared_recording", "calibrated")
        },
        "global_t0_wall_time": exp.get("metadata", {}).get("global_t0_wall_time"),
        "time_base": "experiment_global_seconds",
        "streams": alignment_streams,
    }

    paths = {
        "experiment_run_manifest_json": str(exp_dir / "experiment_run_manifest.json"),
        "stream_manifest_json": str(exp_dir / "stream_manifest.json"),
        "timeline_alignment_json": str(exp_dir / "timeline_alignment.json"),
        "semantic_sync_anchors_json": str(exp_dir / "semantic_sync_anchors.json"),
        "transcript_segments_jsonl": str(exp_dir / "transcript_segments.jsonl"),
        "material_stream_v2_jsonl": str(exp_dir / "material_stream.v2.jsonl"),
    }
    if semantic_sync is None:
        existing_semantic_sync = _load_json_if_exists(Path(paths["semantic_sync_anchors_json"]))
        if isinstance(existing_semantic_sync, dict) and existing_semantic_sync.get("schema_version") == "multimodal_semantic_sync.v1":
            semantic_sync = existing_semantic_sync
    _write_json(Path(paths["experiment_run_manifest_json"]), run_manifest)
    _write_json(Path(paths["stream_manifest_json"]), stream_manifest)
    _write_json(Path(paths["timeline_alignment_json"]), timeline_alignment)
    _write_json(
        Path(paths["semantic_sync_anchors_json"]),
        _semantic_sync_artifact_payload(
            experiment_id=experiment_id,
            run_id=run_id,
            streams=alignment_streams,
            descriptors=source_streams,
            semantic_sync=semantic_sync,
        ),
    )
    _write_jsonl(Path(paths["transcript_segments_jsonl"]), _build_transcript_segments(exp))
    if material_stream is not None:
        _write_jsonl(
            Path(paths["material_stream_v2_jsonl"]),
            _material_stream_v2_rows(
                experiment_id=experiment_id,
                run_id=run_id,
                material_stream=material_stream,
                exp=exp,
                timeline=timeline,
            ),
        )
    elif not Path(paths["material_stream_v2_jsonl"]).exists():
        _write_jsonl(Path(paths["material_stream_v2_jsonl"]), [])
    return paths


def _initialize_waiting_analysis_task(experiment_id: str, exp: Dict[str, Any]) -> Dict[str, Any]:
    exp = _ensure_experiment_run_metadata(exp)
    state = {
        "task_id": exp["run_id"],
        "experiment_id": experiment_id,
        "status": "waiting_for_sources",
        "current_stage": "waiting_for_sources",
        "progress": 0.0,
        "message": "Experiment run created; waiting for multi-monitor sources or uploaded video.",
        "video_path": None,
        "started_at": None,
        "completed_at": None,
        "output_paths": exp.get("output_paths", {}),
        "error_type": None,
        "error_message": None,
        "created_at": exp.get("created_at") or _now_iso(),
    }
    return _persist_experiment_task_state(experiment_id, state)


def _queue_experiment_auto_analysis(
    *,
    experiment_id: str,
    background_tasks: BackgroundTasks,
    source_ref: Optional[str] = None,
    source_type: Optional[str] = None,
    sample_interval: float = 3.0,
    max_frames: int = 10,
    trigger: str = "auto",
    force_service_only: bool = False,
) -> Dict[str, Any]:
    if ExperimentService is None or EXPERIMENT_TASK_STORE is None:
        return {"status": "skipped", "reason": "experiment service not available"}

    exp = _normalize_experiment_dict(_load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or {})
    if not exp:
        raise RuntimeError(f"Experiment not found: {experiment_id}")

    existing_task = _experiment_task_state(experiment_id)
    if existing_task and existing_task.get("status") in {"queued", "running", "analyzing", "generating_outputs", "writing_back"}:
        return existing_task

    descriptors = _experiment_stream_descriptors(exp)
    primary_input = descriptors[0] if descriptors else None
    selected_source = source_ref or (exp.get("video_paths", [None])[0] if exp.get("video_paths") else None)
    if not selected_source and primary_input:
        selected_source = primary_input.get("video_path") or primary_input.get("source")
    if not selected_source:
        exp["status"] = "waiting_for_sources"
        exp["processing_stage"] = "waiting_for_sources"
        exp["processing_error"] = None
        exp["output_paths"] = {
            **(exp.get("output_paths") or {}),
            **_write_experiment_run_artifacts(experiment_id, exp, material_stream=[]),
        }
        _save_experiment(exp)
        return _initialize_waiting_analysis_task(experiment_id, exp)

    selected_source_type = (source_type or (primary_input.get("source_type") if primary_input else None) or ("file" if _is_local_file_source(selected_source) else "rtsp")).lower()
    is_local_file = _is_local_file_source(selected_source)
    is_live_source = (not is_local_file) and (_looks_like_live_source(selected_source) or selected_source_type in {"rtsp", "usb", "udp", "rtmp", "http"})
    if not is_local_file and not is_live_source:
        raise RuntimeError(f"video or source not found: {selected_source}")

    task_id = str(uuid.uuid4())
    task_state = _persist_experiment_task_state(
        experiment_id,
        {
            "task_id": task_id,
            "experiment_id": experiment_id,
            "status": "queued",
            "current_stage": ProcessStage.INGESTION.value,
            "progress": 0.0,
            "message": f"Queued automatically by {trigger}",
            "video_path": str(selected_source),
            "started_at": None,
            "completed_at": None,
            "output_paths": exp.get("output_paths", {}),
            "error_type": None,
            "error_message": None,
            "created_at": _now_iso(),
            "trigger": trigger,
        },
    )
    exp["analysis_job_id"] = task_id
    exp["status"] = "queued"
    exp["processing_stage"] = ProcessStage.INGESTION.value
    exp["processing_error"] = None
    exp["started_at"] = None
    exp["completed_at"] = None
    exp["output_paths"] = {
        **(exp.get("output_paths") or {}),
        **_write_experiment_run_artifacts(experiment_id, exp),
    }
    _save_experiment(exp)

    if is_local_file and FORMAL_WORKFLOW is not None and selected_source_type == "file" and not force_service_only:
        background_tasks.add_task(
            _run_experiment_pipeline,
            experiment_id=experiment_id,
            task_id=task_id,
            video_file=Path(str(selected_source)),
            sample_interval=sample_interval,
            max_frames=max_frames,
            qwen_writeback_config=_qwen_writeback_config_from_request(None),
        )
    else:
        background_tasks.add_task(
            _run_experiment_service_only,
            experiment_id=experiment_id,
            source_ref=str(selected_source),
            sample_interval=sample_interval,
            max_frames=max_frames,
            qwen_writeback_config=_qwen_writeback_config_from_request(None),
            task_id=task_id,
        )
    return task_state


@app.get("/api/v1/diagnostics", tags=["diagnostics"])
async def get_runtime_diagnostics(
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Return runtime wiring status for Qwen ASR, embeddings, and YOLO26 detector."""
    _ = auth_ctx
    return {
        "schema_version": "diagnostics.v1",
        "project_root": str(PROJECT_ROOT),
        "model_status": _experiment_model_status(),
    }


def _update_experiment_record(
    experiment_id: str,
    **fields: Any,
) -> Dict[str, Any]:
    exp = _normalize_experiment_dict(_load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or {})
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    exp.update(fields)
    _save_experiment(_normalize_experiment_dict(exp))
    return exp


def _run_video_analysis_pipeline(
    *,
    task_id: str,
    experiment_id: str,
    video_path: str,
    output_dir: Path,
    sample_interval: float,
    max_frames: int,
) -> Dict[str, Any]:
    settings = RUNTIME_SETTINGS
    yolo_path = _resolve_video_analysis_model_path()

    _persist_experiment_task_state(experiment_id, {
        "status": "running", "current_stage": "yolo_detection",
        "progress": 0.05, "message": "YOLO 检测模型加载中 Loading YOLO model...",
    })

    pipeline = VideoAnalysisPipeline(
        settings=settings,
        yolo_model_path=yolo_path,
        vlm_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        vlm_base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        sample_interval=sample_interval,
        max_frames=max_frames,
    )

    _persist_experiment_task_state(experiment_id, {
        "status": "running", "current_stage": "frame_sampling",
        "progress": 0.08, "message": "视频抽帧与检测中 Sampling & detecting frames...",
    })

    analyses = pipeline.analyze_video(video_path)

    _persist_experiment_task_state(experiment_id, {
        "status": "running", "current_stage": "analysis_json",
        "progress": 0.18, "message": "帧分析完成，写入结果 Writing analysis JSON...",
    })

    analysis_json_path = output_dir / "analysis.json"
    annotated_video_path = output_dir / "annotated.mp4"
    _write_json(analysis_json_path, pipeline.export_json(analyses))

    _persist_experiment_task_state(experiment_id, {
        "status": "running", "current_stage": "annotated_video",
        "progress": 0.22, "message": "标注视频生成中 Encoding annotated video...",
    })

    pipeline.create_annotated_video(video_path, analyses, str(annotated_video_path))

    _persist_experiment_task_state(experiment_id, {
        "status": "running", "current_stage": "video_analysis_done",
        "progress": 0.38, "message": "视频分析完成 Video analysis done",
    })

    return {
        "analyses": analyses,
        "analysis_json_path": str(analysis_json_path),
        "annotated_video_path": str(annotated_video_path),
    }


def _write_experiment_processing_outputs(
    *,
    experiment_id: str,
    experiment_record: Dict[str, Any],
    experiment_result: Dict[str, Any],
    task_state: Dict[str, Any],
    video_analysis_artifacts: Dict[str, str],
) -> Dict[str, str]:
    output_dir = _experiment_output_dir(experiment_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_obj = experiment_result["experiment"]
    timeline_obj = experiment_result["timeline"]
    physical_events = experiment_result.get("physical_events", [])
    material_stream = experiment_result.get("material_stream", [])
    semantic_sync = experiment_result.get("semantic_sync") if isinstance(experiment_result.get("semantic_sync"), dict) else None

    exp_out = {**experiment_record, **experiment_obj.to_dict()}
    exp_out["status"] = ExperimentStatus.ANALYZED.value
    exp_out["analysis_job_id"] = task_state["task_id"]
    exp_out["processing_stage"] = ProcessStage.OUTPUT_GENERATION.value
    exp_out["processing_error"] = None
    exp_out["started_at"] = task_state.get("started_at")
    exp_out["completed_at"] = task_state.get("completed_at")
    exp_out["analyzed_at"] = task_state.get("completed_at")
    exp_out["avg_confidence"] = timeline_obj.avg_confidence

    (output_dir / "timeline.json").write_text(timeline_obj.to_json(), encoding="utf-8")
    _write_json(output_dir / "steps.json", [s.to_dict() for s in timeline_obj.steps])
    _write_json(output_dir / "physical_events.json", [pe.to_dict() for pe in physical_events])
    _write_json(output_dir / "material_stream.json", [item.to_dict() for item in material_stream])

    structured_payload: Dict[str, Any]
    if FORMAL_WORKFLOW is not None:
        structured_payload = FORMAL_WORKFLOW.build_structured_output(exp_out, experiment_result)
    else:
        timeline_dict = timeline_obj.to_dict()
        structured_payload = {
            "experiment_id": experiment_id,
            "title": exp_out.get("title", experiment_id),
            "status": ExperimentStatus.ANALYZED.value,
            "steps": [s.to_dict() for s in timeline_obj.steps],
            "timeline": timeline_dict.get("steps", []),
            "evidence": [],
            "protocol": exp_out.get("protocol_text"),
            "sop": None,
            "analysis": {
                "job_id": task_state.get("task_id"),
                "analyzed_at": task_state.get("completed_at"),
                "available": True,
            },
            "statistics": {
                "total_steps": timeline_dict.get("total_steps", 0),
                "confirmed_count": timeline_dict.get("confirmed_steps", 0),
                "inferred_count": timeline_dict.get("inferred_steps", 0),
                "average_confidence": timeline_dict.get("avg_confidence"),
            },
        }
    _write_json(output_dir / "structured.json", structured_payload)
    _write_json(output_dir / "preprocessing.json", structured_payload.get("preprocessing_layer", {}))

    output_paths = {
        "experiment_json": str(output_dir / "experiment.json"),
        "timeline_json": str(output_dir / "timeline.json"),
        "steps_json": str(output_dir / "steps.json"),
        "step_candidates_json": str(output_dir / "step_candidates.json"),
        "step_bridge_summary_json": str(output_dir / "step_bridge_summary.json"),
        "official_steps_json": str(output_dir / "official_steps.json"),
        "step_review_log_json": str(output_dir / "step_review_log.json"),
        "sop_schema_validation_json": str(output_dir / "sop_schema_validation.json"),
        "physical_events_json": str(output_dir / "physical_events.json"),
        "material_stream_json": str(output_dir / "material_stream.json"),
        "material_stream_v2_jsonl": str(output_dir / "material_stream.v2.jsonl"),
        "preprocessing_json": str(output_dir / "preprocessing.json"),
        "material_index": str(output_dir / "material_index.sqlite"),
        "published_materials_json": str(output_dir / "published_materials.json"),
        "upload_manifest_json": str(output_dir / "upload_manifest.json"),
        "experiment_run_manifest_json": str(output_dir / "experiment_run_manifest.json"),
        "stream_manifest_json": str(output_dir / "stream_manifest.json"),
        "timeline_alignment_json": str(output_dir / "timeline_alignment.json"),
        "semantic_sync_anchors_json": str(output_dir / "semantic_sync_anchors.json"),
        "transcript_segments_jsonl": str(output_dir / "transcript_segments.jsonl"),
        "structured_json": str(output_dir / "structured.json"),
        "analysis_json": video_analysis_artifacts["analysis_json_path"],
        "annotated_video": video_analysis_artifacts["annotated_video_path"],
        "source_video": (
            experiment_record.get("video_paths", [None])[0]
            if experiment_record.get("video_paths")
            else ((experiment_record.get("video_inputs") or [{}])[0].get("video_path", ""))
        ),
        "source_videos": experiment_record.get("video_paths", []),
    }
    output_paths.update(
        _write_experiment_run_artifacts(
            experiment_id,
            exp_out,
            material_stream=material_stream,
            timeline=timeline_obj,
            semantic_sync=semantic_sync,
        )
    )
    event_preprocessing = _run_event_preprocessing_for_output(
        experiment_id=experiment_id,
        experiment_record=exp_out,
        output_dir=output_dir,
        output_paths=output_paths,
        material_stream=[item.to_dict() for item in material_stream],
    )
    if event_preprocessing:
        exp_out.setdefault("metadata", {})["event_preprocessing"] = {
            "metadata_version": event_preprocessing.get("preprocessing_payload", {}).get("metadata_version"),
            "physical_event_count": event_preprocessing.get("preprocessing_payload", {})
            .get("event_preprocessing", {})
            .get("physical_event_count", 0),
        }
    exp_out["output_paths"] = output_paths
    _save_experiment(_normalize_experiment_dict(exp_out))
    _create_friendly_experiment_alias(experiment_id, exp_out.get("title", ""))
    return output_paths


def _run_event_preprocessing_for_output(
    *,
    experiment_id: str,
    experiment_record: Dict[str, Any],
    output_dir: Path,
    output_paths: Dict[str, str],
    material_stream: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    source_video = output_paths.get("source_video")
    if not source_video or not Path(source_video).exists():
        logger.warning("Event preprocessing skipped for %s: source video missing", experiment_id)
        return None
    try:
        from labsopguard.event_preprocessing import EventPreprocessingEngine

        analysis_frames = _load_json_if_exists(Path(output_paths["analysis_json"])) or []
        engine = EventPreprocessingEngine(settings=RUNTIME_SETTINGS, yolo_model_path=RUNTIME_SETTINGS.yolo_model_path)
        result = engine.run(
            experiment_id=experiment_id,
            experiment_name=experiment_record.get("title") or experiment_id,
            source_video=source_video,
            output_dir=output_dir,
            material_index_path=output_paths["material_index"],
            analysis_frames=analysis_frames,
            material_stream=material_stream,
            source_video_id=experiment_record.get("video_asset_id") or f"{experiment_id}:video:0",
        )
        _write_json(output_dir / "physical_events.json", result["physical_events_payload"])
        merged_material_stream = [*material_stream, *result["material_event_items"]]
        _write_json(output_dir / "material_stream.json", merged_material_stream)
        _backlink_event_preprocessing_to_steps(output_dir, result["events"])

        existing_preprocessing = _load_json_if_exists(output_dir / "preprocessing.json") or {}
        preprocessing_payload = {
            **existing_preprocessing,
            **result["preprocessing_payload"],
            "legacy_preprocessing": existing_preprocessing,
        }
        _write_json(output_dir / "preprocessing.json", preprocessing_payload)
        _run_step_bridge_for_output(
            experiment_id=experiment_id,
            output_dir=output_dir,
            physical_events_payload=result["physical_events_payload"],
            preprocessing_payload=preprocessing_payload,
        )
        return result
    except Exception as exc:
        logger.exception("Event preprocessing failed for %s", experiment_id)
        existing_preprocessing = _load_json_if_exists(output_dir / "preprocessing.json") or {}
        existing_preprocessing["event_preprocessing_error"] = str(exc)
        existing_preprocessing["metadata_version"] = "event_preprocessing.v1"
        _write_json(output_dir / "preprocessing.json", existing_preprocessing)
        return None


_SOP_DEFAULT_STEPS = [
    {
        "protocol_step_id": "step_ppe",
        "protocol_step_name": "穿戴防护装备",
        "step_index": 0,
        "required_event_types": ["hand_object_interaction"],
        "optional_event_types": ["object_move"],
        "critical_fields": ["actor_track_id"],
        "event_reuse_policy": "prefer_unique",
    },
    {
        "protocol_step_id": "step_prepare",
        "protocol_step_name": "准备实验器材",
        "step_index": 1,
        "required_event_types": ["object_move"],
        "optional_event_types": ["hand_object_interaction", "container_state_change"],
        "critical_fields": ["actor_track_id"],
        "event_reuse_policy": "prefer_unique",
    },
    {
        "protocol_step_id": "step_open_container",
        "protocol_step_name": "打开容器盖",
        "step_index": 2,
        "required_event_types": ["container_state_change"],
        "optional_event_types": ["hand_object_interaction"],
        "critical_fields": ["state_change_type", "state_confidence"],
        "event_reuse_policy": "prefer_unique",
    },
    {
        "protocol_step_id": "step_transfer",
        "protocol_step_name": "液体转移操作",
        "step_index": 3,
        "required_event_types": ["liquid_transfer"],
        "optional_event_types": ["hand_object_interaction", "object_move"],
        "critical_fields": ["source_container", "target_container", "direction_status"],
        "event_reuse_policy": "prefer_unique",
    },
    {
        "protocol_step_id": "step_panel",
        "protocol_step_name": "设备面板操作",
        "step_index": 4,
        "required_event_types": ["panel_operation"],
        "optional_event_types": ["hand_object_interaction"],
        "critical_fields": ["actor_track_id"],
        "event_reuse_policy": "allow_reuse",
    },
    {
        "protocol_step_id": "step_close_container",
        "protocol_step_name": "关闭容器盖",
        "step_index": 5,
        "required_event_types": ["container_state_change"],
        "optional_event_types": ["hand_object_interaction"],
        "critical_fields": ["state_change_type", "state_confidence"],
        "event_reuse_policy": "prefer_unique",
    },
]


def _enrich_steps_for_bridge(steps: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure every step has required_event_types.  When steps.json has only generic
    placeholder steps (no event type constraints), substitute with SOP default steps
    whose event types actually appear in the physical events stream."""
    if not steps or not events:
        return steps
    # Check if steps already have explicit event type bindings
    has_bindings = any(
        step.get("required_event_types") or step.get("optional_event_types")
        for step in steps
        if isinstance(step, dict)
    )
    if has_bindings:
        return steps
    # Collect event types actually present
    observed_types = {str(e.get("event_type")) for e in events if e.get("event_type")}
    # Only keep SOP default steps whose required type is observed
    enriched = [
        s for s in _SOP_DEFAULT_STEPS
        if any(t in observed_types for t in (s.get("required_event_types") or []))
    ]
    if not enriched:
        # Fallback: use all default steps as optional
        enriched = [
            {**s, "required_event_types": [], "optional_event_types": s["required_event_types"] + s.get("optional_event_types", [])}
            for s in _SOP_DEFAULT_STEPS
        ]
    return enriched


_EVENT_TYPE_STEP_LABELS = {
    "hand_object_interaction": "手部接触与操作",
    "object_move": "物体移动与摆放",
    "liquid_transfer": "液体转移",
    "panel_operation": "设备/称量操作",
    "container_state_change": "容器状态变化",
}


def _is_single_generic_step(steps: List[Dict[str, Any]]) -> bool:
    if len(steps) != 1 or not isinstance(steps[0], dict):
        return False
    step = steps[0]
    name = str(step.get("step_name") or step.get("protocol_step_name") or "").strip().lower()
    description = str(step.get("step_description") or "")
    if name in {"实验操作", "experiment operation", "operation", "实验步骤"}:
        return True
    return description.strip().startswith("```json") and float(step.get("start_time_sec") or 0.0) <= 0.1


def _event_time(event: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(event.get(key) if event.get(key) is not None else default)
    except (TypeError, ValueError):
        return default


def _dominant_event_type(events: List[Dict[str, Any]]) -> str:
    counts: Dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type") or "hand_object_interaction")
        counts[event_type] = counts.get(event_type, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0] if counts else "hand_object_interaction"


def _confidence_from_events(events: List[Dict[str, Any]]) -> float:
    grade_scores = {"strong": 0.9, "medium": 0.68, "weak": 0.42}
    values = [grade_scores.get(str(event.get("evidence_grade")), 0.58) for event in events]
    return round(sum(values) / len(values), 4) if values else 0.0


def _build_aligned_steps_from_events(experiment_id: str, output_dir: Path, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_events = sorted((event for event in events if isinstance(event, dict)), key=lambda event: _event_time(event, "start_time_sec"))
    if not sorted_events:
        return []
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_type = ""
    current_start = 0.0
    for event in sorted_events:
        start = _event_time(event, "start_time_sec")
        event_type = str(event.get("event_type") or "hand_object_interaction")
        should_split = False
        if current:
            elapsed = start - current_start
            should_split = elapsed >= 12.0 or (event_type != current_type and elapsed >= 4.0)
        if should_split:
            groups.append(current)
            current = []
        if not current:
            current_start = start
            current_type = event_type
        current.append(event)
    if current:
        groups.append(current)

    now = datetime.now(timezone.utc).isoformat()
    steps: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups):
        event_type = _dominant_event_type(group)
        start = min(_event_time(event, "start_time_sec") for event in group)
        end = max(_event_time(event, "end_time_sec", _event_time(event, "start_time_sec")) for event in group)
        if end <= start:
            end = start + 1.0
        label = _EVENT_TYPE_STEP_LABELS.get(event_type, event_type)
        first_display = next((str(event.get("display_name")) for event in group if event.get("display_name")), "")
        evidence_refs = [
            {
                "evidence_id": f"{str(event.get('event_id') or idx)}:evidence",
                "evidence_type": "physical_event",
                "source": "material_stream",
                "timestamp_sec": _event_time(event, "start_time_sec"),
                "confidence": _confidence_from_events([event]),
                "description": event.get("display_name") or event.get("event_type"),
            }
            for event in group[:5]
        ]
        confidence = _confidence_from_events(group)
        steps.append(
            {
                "step_id": f"aligned_step_{idx:03d}",
                "protocol_step_id": f"aligned_step_{idx:03d}",
                "experiment_id": experiment_id,
                "step_index": idx,
                "step_name": label,
                "protocol_step_name": label,
                "step_description": first_display or label,
                "status": "candidate",
                "start_time_sec": round(start, 3),
                "end_time_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "confidence": confidence,
                "step_confidence": "high" if confidence >= 0.75 else ("medium" if confidence >= 0.5 else "low"),
                "completed_by_inference": False,
                "inference_method": "material_stream_event_alignment",
                "inference_model": "material_stream_fallback",
                "required_event_types": [event_type],
                "optional_event_types": [],
                "critical_fields": [],
                "event_reuse_policy": "allow_reuse",
                "evidence_refs": evidence_refs,
                "parameters": [],
                "linked_context_events": [],
                "linked_physical_events": [str(event.get("event_id")) for event in group if event.get("event_id")],
                "metadata": {"source": "material_stream_event_alignment", "event_count": len(group)},
                "created_at": now,
                "updated_at": now,
            }
        )
    timeline_path = output_dir / "timeline.json"
    existing_timeline = _load_json_if_exists(timeline_path) or {}
    timeline = {
        **existing_timeline,
        "timeline_id": existing_timeline.get("timeline_id") or f"timeline_{experiment_id}",
        "experiment_id": experiment_id,
        "title": existing_timeline.get("title") or experiment_id,
        "steps": steps,
        "total_steps": len(steps),
        "confirmed_steps": 0,
        "candidate_steps": len(steps),
        "inferred_steps": 0,
        "skipped_steps": 0,
        "avg_confidence": round(sum(float(step.get("confidence") or 0.0) for step in steps) / len(steps), 4) if steps else 0.0,
        "start_time_sec": steps[0]["start_time_sec"] if steps else 0.0,
        "end_time_sec": steps[-1]["end_time_sec"] if steps else 0.0,
        "total_duration_sec": round((steps[-1]["end_time_sec"] - steps[0]["start_time_sec"]) if steps else 0.0, 3),
        "updated_at": now,
    }
    _write_json(timeline_path, timeline)
    return steps


def _run_step_bridge_for_output(
    *,
    experiment_id: str,
    output_dir: Path,
    physical_events_payload: Dict[str, Any],
    preprocessing_payload: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    steps = _load_json_if_exists(output_dir / "steps.json") or []
    if not isinstance(steps, list) or not steps:
        logger.warning("Step bridge skipped for %s: steps.json missing or empty", experiment_id)
        return None
    events = physical_events_payload.get("events") or []
    if isinstance(events, dict):
        events = list(events.values())
    if _is_single_generic_step(steps) and events:
        aligned_steps = _build_aligned_steps_from_events(experiment_id, output_dir, events)
        if aligned_steps:
            steps = aligned_steps
        else:
            steps = _enrich_steps_for_bridge(steps, events)
    else:
        steps = _enrich_steps_for_bridge(steps, events)
    # Write enriched steps back so the frontend can display them
    _write_json(output_dir / "steps.json", steps)
    try:
        from labsopguard.step_bridge import StepBridgeEngine
        from labsopguard.step_review import StepReviewStore

        result = StepBridgeEngine().run(
            experiment_id=experiment_id,
            output_dir=output_dir,
            steps=steps,
            physical_events_payload=physical_events_payload,
            preprocessing_payload=preprocessing_payload,
        )
        StepReviewStore(experiment_id, output_dir).ensure_outputs()
        return result
    except Exception as exc:
        logger.exception("Step bridge failed for %s", experiment_id)
        _write_json(
            output_dir / "step_bridge_summary.json",
            {
                "schema_version": "step_bridge_summary.v1",
                "experiment_id": experiment_id,
                "error": str(exc),
                "confirmed_steps": [],
                "candidate_steps": [],
                "inferred_steps": [],
                "needs_review_steps": [],
                "missing_steps": [],
                "out_of_order_steps": [],
                "blocked_steps": [],
            },
        )
        return None


def _backlink_event_preprocessing_to_steps(output_dir: Path, events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    steps_path = output_dir / "steps.json"
    steps = _load_json_if_exists(steps_path) or []
    if not isinstance(steps, list):
        return
    changed = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        step_start = float(step.get("start_time_sec") or 0.0)
        step_end = float(step.get("end_time_sec") or (step_start + 10.0))
        linked = list(step.get("linked_physical_events") or [])
        for event in events:
            event_start = float(event.get("start_time_sec") or 0.0)
            event_end = float(event.get("end_time_sec") or event_start)
            if event_end < step_start or event_start > step_end:
                continue
            event_id = event.get("event_id")
            if event_id and event_id not in linked:
                linked.append(event_id)
                changed = True
        step["linked_physical_events"] = linked
    if changed:
        _write_json(steps_path, steps)
        timeline_path = output_dir / "timeline.json"
        timeline = _load_json_if_exists(timeline_path) or {}
        if isinstance(timeline, dict) and isinstance(timeline.get("steps"), list):
            by_id = {step.get("step_id"): step for step in steps if isinstance(step, dict)}
            for step in timeline["steps"]:
                if isinstance(step, dict) and step.get("step_id") in by_id:
                    step["linked_physical_events"] = by_id[step["step_id"]].get("linked_physical_events", [])
            _write_json(timeline_path, timeline)


def _collect_experiment_inputs(exp: Dict[str, Any]) -> tuple[str, str]:
    context_parts: List[str] = []
    for ctx in exp.get("context_inputs", []):
        if isinstance(ctx, dict):
            text = ctx.get("text")
            if text:
                context_parts.append(str(text))
        elif ctx:
            context_parts.append(str(ctx))
    return "\n".join(context_parts).strip(), str(exp.get("protocol_text") or "")


def _invoke_experiment_service(
    *,
    experiment_id: str,
    experiment_record: Dict[str, Any],
    source_ref: str,
    sample_interval: float,
    max_frames: int,
) -> Dict[str, Any]:
    if ExperimentService is None:
        raise RuntimeError("Experiment service not available")

    context_text, protocol_text = _collect_experiment_inputs(experiment_record)
    service = ExperimentService(
        vlm_api_key=os.environ.get("DASHSCOPE_API_KEY"),
        vlm_base_url=os.environ.get("DASHSCOPE_BASE_URL"),
        frame_sample_interval=sample_interval,
        max_frames=max_frames,
        yolo26_weights_path=_resolve_video_analysis_model_path(),
        detector_device=os.environ.get("DETECTOR_DEVICE") or (RUNTIME_SETTINGS.device if RUNTIME_SETTINGS else None),
    )
    video_inputs = experiment_record.get("video_inputs") or experiment_record.get("video_metadata") or []
    if hasattr(service, "set_video_inputs") and video_inputs:
        service.set_video_inputs(video_inputs)
    elif hasattr(service, "set_videos") and experiment_record.get("video_paths"):
        service.set_videos(experiment_record.get("video_paths", []))
    else:
        service.set_video(str(source_ref))
    service.set_context(context_text)
    if hasattr(service, "set_context_inputs"):
        service.set_context_inputs(experiment_record.get("context_inputs", []))
    service.set_protocol(protocol_text)
    return service.process(
        experiment_id=experiment_id,
        experiment_title=experiment_record.get("title", experiment_id),
    )


def _qwen_writeback_config_from_request(req: Optional["ProcessExperimentRequest"] = None):
    from labsopguard.qwen_writeback import QwenFrameWritebackConfig

    config = QwenFrameWritebackConfig.from_env()
    if req is not None:
        if req.qwen_frame_writeback_enabled is not None:
            config.enabled = bool(req.qwen_frame_writeback_enabled)
        if req.qwen_frame_writeback_limit is not None:
            config.limit = max(0, int(req.qwen_frame_writeback_limit))
        if req.qwen_frame_writeback_force_live is not None:
            config.force_live = bool(req.qwen_frame_writeback_force_live)
    return config


def _run_qwen_frame_writeback_if_enabled(
    *,
    experiment_id: str,
    output_paths: Dict[str, str],
    config: Any,
) -> Optional[Dict[str, Any]]:
    if not getattr(config, "enabled", False):
        return None
    from labsopguard.qwen_writeback import writeback_qwen_frame_analysis

    exp_dir = _experiment_output_dir(experiment_id)
    report = writeback_qwen_frame_analysis(exp_dir, exp_id=experiment_id, config=config)
    output_paths["qwen_frame_writeback_json"] = str(report.get("report_path") or (exp_dir / "qwen_frame_writeback.json"))
    output_paths["material_index"] = str(exp_dir / "material_index.sqlite")
    return report


def _attach_video_analysis_artifacts_to_experiment(
    *,
    experiment_id: str,
    task_id: str,
    source_video: Path,
    analysis_json: Path,
    annotated_video: Path,
    sample_interval: float,
    max_frames: int,
    run_experiment_outputs: bool = True,
    qwen_writeback_config: Any = None,
) -> Dict[str, Any]:
    safe_experiment_id = _validate_experiment_id(experiment_id)
    exp = _normalize_experiment_dict(
        _load_json_if_exists(_experiment_output_dir(safe_experiment_id) / "experiment.json") or {}
    )
    if not exp:
        raise RuntimeError(f"Experiment not found: {safe_experiment_id}")
    if not source_video.exists():
        raise RuntimeError(f"Video analysis source video not found: {source_video}")
    if not analysis_json.exists():
        raise RuntimeError(f"Video analysis JSON not found: {analysis_json}")
    if not annotated_video.exists():
        raise RuntimeError(f"Annotated video not found: {annotated_video}")

    exp_dir = _experiment_output_dir(safe_experiment_id)
    analysis_dir = exp_dir / "analysis"
    upload_dir = PROJECT_ROOT / "uploads" / "experiments" / safe_experiment_id / "videos"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    attached_source = upload_dir / source_video.name
    attached_analysis = analysis_dir / "analysis.json"
    attached_annotated = analysis_dir / "annotated.mp4"
    if source_video.resolve() != attached_source.resolve():
        shutil.copy2(source_video, attached_source)
    shutil.copy2(analysis_json, attached_analysis)
    shutil.copy2(annotated_video, attached_annotated)

    exp.setdefault("video_paths", [])
    if str(attached_source) not in exp["video_paths"]:
        exp["video_paths"].append(str(attached_source))
    if not exp.get("video_inputs"):
        video_metadata = _probe_video_metadata(attached_source, 0)
        try:
            from labsopguard.video_input_schema import VideoInputValidationError, normalize_video_input

            video_metadata, _ = normalize_video_input(video_metadata, index=0, strict=True)
        except Exception:
            video_metadata = {
                "video_index": 0,
                "video_path": str(attached_source),
                "source": str(attached_source),
                "source_type": "file",
                "ingest_mode": "file",
                "camera_id": "camera_0",
                "start_offset_sec": 0.0,
            }
        exp.setdefault("video_inputs", []).append(video_metadata)
        exp.setdefault("video_metadata", []).append(video_metadata)
    exp["video_asset_id"] = exp.get("video_asset_id") or f"{safe_experiment_id}:video:0"
    exp["analysis_job_id"] = task_id
    exp["status"] = "running" if run_experiment_outputs else ExperimentStatus.ANALYZED.value
    exp["processing_stage"] = ProcessStage.VIDEO_UNDERSTANDING.value
    exp["processing_error"] = None
    exp["output_paths"] = {
        **(exp.get("output_paths") or {}),
        "source_video": str(attached_source),
        "analysis_json": str(attached_analysis),
        "annotated_video": str(attached_annotated),
    }
    _save_experiment(exp)

    if not run_experiment_outputs:
        completed_at = _now_iso()
        exp["status"] = ExperimentStatus.ANALYZED.value
        exp["processing_stage"] = ProcessStage.OUTPUT_GENERATION.value
        exp["completed_at"] = completed_at
        exp["analyzed_at"] = completed_at
        _save_experiment(exp)
        return exp["output_paths"]

    result = _invoke_experiment_service(
        experiment_id=safe_experiment_id,
        experiment_record=exp,
        source_ref=str(attached_source),
        sample_interval=sample_interval,
        max_frames=max_frames,
    )
    completed_at = _now_iso()
    video_analysis_artifacts = {
        "analysis_json_path": str(attached_analysis),
        "annotated_video_path": str(attached_annotated),
    }
    output_paths = _write_experiment_processing_outputs(
        experiment_id=safe_experiment_id,
        experiment_record=exp,
        experiment_result=result,
        task_state={
            "task_id": task_id,
            "status": "completed",
            "started_at": exp.get("started_at") or completed_at,
            "completed_at": completed_at,
        },
        video_analysis_artifacts=video_analysis_artifacts,
    )
    qwen_report = None
    try:
        qwen_report = _run_qwen_frame_writeback_if_enabled(
            experiment_id=safe_experiment_id,
            output_paths=output_paths,
            config=qwen_writeback_config or _qwen_writeback_config_from_request(None),
        )
    except Exception as exc:
        logger.exception("Qwen frame writeback failed while attaching video analysis for %s", safe_experiment_id)
        if getattr(qwen_writeback_config, "fail_pipeline", False):
            raise
        output_paths["qwen_frame_writeback_error"] = str(exc)

    _persist_experiment_task_state(
        safe_experiment_id,
        {
            "status": "running",
            "current_stage": "professional_report",
            "progress": 0.95,
            "message": "Generating professional PDF report",
            "output_paths": output_paths,
        },
    )
    professional_report = _generate_professional_report_for_experiment(
        safe_experiment_id,
        output_paths=output_paths,
    )
    output_paths = _attach_professional_report_output_paths(
        safe_experiment_id,
        output_paths,
        professional_report,
    )

    final_state = _persist_experiment_task_state(
        safe_experiment_id,
        {
            "task_id": task_id,
            "experiment_id": safe_experiment_id,
            "status": "completed",
            "current_stage": ProcessStage.OUTPUT_GENERATION.value,
            "progress": 1.0,
            "video_path": str(attached_source),
            "started_at": exp.get("started_at") or completed_at,
            "completed_at": completed_at,
            "output_paths": output_paths,
            "qwen_frame_writeback": qwen_report,
            "professional_report": professional_report,
            "error_type": None,
            "error_message": None,
        },
    )
    exp = _normalize_experiment_dict(_load_json_if_exists(exp_dir / "experiment.json") or exp)
    exp["status"] = ExperimentStatus.ANALYZED.value
    exp["processing_stage"] = ProcessStage.OUTPUT_GENERATION.value
    exp["processing_error"] = None
    exp["completed_at"] = final_state.get("completed_at")
    exp["analyzed_at"] = final_state.get("completed_at")
    exp["analysis_job_id"] = task_id
    exp["output_paths"] = output_paths
    if qwen_report is not None:
        exp.setdefault("metadata", {})["qwen_frame_writeback"] = qwen_report
    exp.setdefault("metadata", {})["professional_report"] = professional_report
    _save_experiment(exp)
    return output_paths


def _bootstrap_empty_video_analysis_artifacts(output_dir: Path) -> Dict[str, Any]:
    analysis_json_path = output_dir / "analysis.json"
    annotated_video_path = output_dir / "annotated.mp4"
    _write_json(analysis_json_path, [])
    return {
        "analyses": [],
        "analysis_json_path": str(analysis_json_path),
        "annotated_video_path": str(annotated_video_path),
    }


def _run_experiment_service_only(
    *,
    experiment_id: str,
    source_ref: str,
    sample_interval: float,
    max_frames: int,
    qwen_writeback_config: Any = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    exp = _normalize_experiment_dict(
        _load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or {}
    )
    if not exp:
        raise RuntimeError("Experiment not found")

    task_id = task_id or str(uuid.uuid4())
    started_at = _now_iso()
    analysis_dir = _experiment_output_dir(experiment_id) / "analysis"
    output_paths: Dict[str, str] = {}

    try:
        exp["analysis_job_id"] = task_id
        exp["status"] = ExperimentStatus.ANALYZING.value
        exp["processing_stage"] = ProcessStage.CONTEXT_INTEGRATION.value
        exp["processing_error"] = None
        exp["started_at"] = started_at
        exp["completed_at"] = None
        _save_experiment(exp)
        _persist_experiment_task_state(
            experiment_id,
            {
                "task_id": task_id,
                "experiment_id": experiment_id,
                "status": "running",
                "current_stage": ProcessStage.VIDEO_UNDERSTANDING.value,
                "progress": 0.1,
                "message": "Experiment analysis running",
                "video_path": str(source_ref),
                "started_at": started_at,
                "completed_at": None,
                "output_paths": output_paths,
                "error_type": None,
                "error_message": None,
            },
        )

        result = _invoke_experiment_service(
            experiment_id=experiment_id,
            experiment_record=exp,
            source_ref=source_ref,
            sample_interval=sample_interval,
            max_frames=max_frames,
        )
        video_analysis_artifacts = _bootstrap_empty_video_analysis_artifacts(analysis_dir)
        completed_at = _now_iso()
        output_paths = _write_experiment_processing_outputs(
            experiment_id=experiment_id,
            experiment_record=exp,
            experiment_result=result,
            task_state={
                "task_id": task_id,
                "status": ExperimentStatus.ANALYZED.value,
                "started_at": started_at,
                "completed_at": completed_at,
            },
            video_analysis_artifacts=video_analysis_artifacts,
        )
        qwen_writeback_report = None
        try:
            qwen_writeback_report = _run_qwen_frame_writeback_if_enabled(
                experiment_id=experiment_id,
                output_paths=output_paths,
                config=qwen_writeback_config or _qwen_writeback_config_from_request(None),
            )
        except Exception as exc:
            logger.exception("Qwen frame writeback failed for %s", experiment_id)
            if getattr(qwen_writeback_config, "fail_pipeline", False):
                raise
            output_paths["qwen_frame_writeback_error"] = str(exc)
        _persist_experiment_task_state(
            experiment_id,
            {
                "status": "running",
                "current_stage": "professional_report",
                "progress": 0.95,
                "message": "Generating professional PDF report",
                "output_paths": output_paths,
            },
        )
        professional_report = _generate_professional_report_for_experiment(
            experiment_id,
            output_paths=output_paths,
        )
        output_paths = _attach_professional_report_output_paths(
            experiment_id,
            output_paths,
            professional_report,
        )
        final_state = _persist_experiment_task_state(
            experiment_id,
            {
                "task_id": task_id,
                "experiment_id": experiment_id,
                "status": ExperimentStatus.ANALYZED.value,
                "current_stage": ProcessStage.OUTPUT_GENERATION.value,
                "progress": 1.0,
                "video_path": str(source_ref),
                "started_at": started_at,
                "completed_at": completed_at,
                "output_paths": output_paths,
                "qwen_frame_writeback": qwen_writeback_report,
                "professional_report": professional_report,
                "error_type": None,
                "error_message": None,
            },
        )
        exp = _normalize_experiment_dict(
            _load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or exp
        )
        exp["status"] = ExperimentStatus.ANALYZED.value
        exp["processing_stage"] = ProcessStage.OUTPUT_GENERATION.value
        exp["processing_error"] = None
        exp["completed_at"] = completed_at
        exp["analysis_job_id"] = task_id
        exp["output_paths"] = output_paths
        if qwen_writeback_report is not None:
            exp.setdefault("metadata", {})["qwen_frame_writeback"] = qwen_writeback_report
        exp.setdefault("metadata", {})["professional_report"] = professional_report
        _save_experiment(exp)
        return final_state
    except Exception as exc:
        error_payload = _classify_processing_error(exc)
        failed_at = _now_iso()
        failed_state = _persist_experiment_task_state(
            experiment_id,
            {
                "task_id": task_id,
                "experiment_id": experiment_id,
                "status": "failed",
                "current_stage": exp.get("processing_stage", ProcessStage.INGESTION.value),
                "progress": 1.0,
                "video_path": str(source_ref),
                "started_at": started_at,
                "completed_at": failed_at,
                "output_paths": output_paths,
                **error_payload,
            },
        )
        exp["status"] = "failed"
        exp["processing_error"] = error_payload["error_message"]
        exp["processing_stage"] = exp.get("processing_stage", ProcessStage.INGESTION.value)
        exp["completed_at"] = failed_at
        exp["analysis_job_id"] = task_id
        exp["output_paths"] = output_paths
        _save_experiment(exp)
        logger.exception("Inline experiment processing failed for %s", experiment_id)
        return failed_state


def _empty_timeline_payload(exp: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.now().isoformat()
    return {
        "timeline_id": f"{exp['experiment_id']}_timeline",
        "experiment_id": exp["experiment_id"],
        "title": exp.get("title", ""),
        "steps": [],
        "total_steps": 0,
        "confirmed_steps": 0,
        "candidate_steps": 0,
        "inferred_steps": 0,
        "skipped_steps": 0,
        "avg_confidence": None,
        "start_time_sec": 0.0,
        "end_time_sec": 0.0,
        "total_duration_sec": 0.0,
        "video_asset_id": exp.get("video_asset_id"),
        "video_duration_sec": None,
        "video_coverage_ratio": 0.0,
        "context_summary": None,
        "protocol_id": exp.get("protocol_id"),
        "protocol_name": None,
        "protocol_text": exp.get("protocol_text"),
        "processing_stage": exp.get("processing_stage", ProcessStage.INGESTION.value),
        "models_used": exp.get("models_used", []),
        "inference_count": 0,
        "media_assets": [],
        "context_events": [],
        "metadata": {
            "analysis_available": False,
            "analysis_status": exp.get("status", ExperimentStatus.CREATED.value),
        },
        "created_at": exp.get("created_at", now),
        "updated_at": exp.get("analyzed_at") or exp.get("created_at", now),
    }


def _empty_structured_payload(exp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "experiment_id": exp["experiment_id"],
        "title": exp.get("title", ""),
        "status": exp.get("status", ExperimentStatus.CREATED.value),
        "steps": [],
        "timeline": [],
        "evidence": [],
        "protocol": None,
        "sop": None,
        "analysis": None,
        "input_layer": {
            "context_summary": None,
            "protocol_text": exp.get("protocol_text"),
            "video_inputs": exp.get("video_metadata", []),
        },
        "preprocessing_layer": {
            "aligned_text": [],
            "key_timestamps": [],
            "video_index": [],
            "detected_changes": [],
            "video_streams": exp.get("video_metadata", []),
            "key_frames": [],
            "key_clips": [],
            "time_anchored_material_stream": [],
            "alignment_summary": {"video_count": len(exp.get("video_paths", []))},
        },
        "statistics": {
            "total_steps": 0,
            "confirmed_count": 0,
            "inferred_count": 0,
            "average_confidence": None,
        },
    }

# API 
class CreateExperimentRequest(BaseModel):
    title: str
    description: str = ""
    context_text: Optional[str] = None
    protocol_text: Optional[str] = None


class ProcessExperimentRequest(BaseModel):
    video_path: Optional[str] = None
    source_type: Optional[str] = None
    sample_interval: float = 3.0
    max_frames: int = 10
    qwen_frame_writeback_enabled: Optional[bool] = None
    qwen_frame_writeback_limit: Optional[int] = None
    qwen_frame_writeback_force_live: Optional[bool] = None


class UploadStreamRequest(BaseModel):
    source: str
    source_type: str = "rtsp"
    camera_id: Optional[str] = None
    view_type: Optional[str] = None
    role: Optional[str] = None
    source_group: Optional[str] = None
    auto_analyze: bool = False
    sync_group: Optional[str] = None
    start_offset_sec: Optional[float] = 0.0
    capture_duration_sec: Optional[float] = 15.0
    sync_method: Optional[str] = None
    sync_anchors: Optional[List[Dict[str, Any]]] = None
    hardware_timecode_start_sec: Optional[float] = None
    sync_board_offset_sec: Optional[float] = None
    clock_scale: Optional[float] = None
    clock_drift_ppm: Optional[float] = None
    sync_confidence: Optional[float] = None


class TimelineAlignmentUpdateRequest(BaseModel):
    streams: Optional[List[Dict[str, Any]]] = None
    rebuild_only: bool = False


class StepReviewRequest(BaseModel):
    step_candidate_id: str
    rationale: str = ""
    operator: str = "system"
    operator_role: str = "approver"


class StepEditAndApproveRequest(StepReviewRequest):
    edits: Optional[Dict[str, Any]] = None


class StepLockRequest(BaseModel):
    operator: str = "system"
    operator_role: str = "admin"
    rationale: str = "lock official step"


class StepSupersedeRequest(BaseModel):
    operator: str = "system"
    operator_role: str = "admin"
    rationale: str
    replacement_payload: Optional[Dict[str, Any]] = None


class MaterialUploadRequest(BaseModel):
    provider: str = "local"
    destination_root: Optional[str] = None
    clock_scale: Optional[float] = None
    clock_drift_ppm: Optional[float] = None


class MaterialCandidateApprovalRequest(BaseModel):
    reviewer: Optional[str] = None
    notes: Optional[str] = None
    candidate_ids: Optional[List[str]] = None
    selected_keyframe_ids: Optional[List[str]] = None
    selected_clip_ids: Optional[List[str]] = None


class KeyActionReviewDecisionRequest(BaseModel):
    decision: str
    reviewer: Optional[str] = "frontend_reviewer"
    note: Optional[str] = ""
    boundary_start_sec: Optional[float] = None
    boundary_end_sec: Optional[float] = None


class KeyActionReviewBulkRequest(BaseModel):
    item_ids: Optional[List[str]] = None
    decision: str
    reviewer: Optional[str] = "frontend_reviewer"
    note: Optional[str] = ""


class ExperimentRecordingStartRequest(BaseModel):
    camera_ids: Optional[List[str]] = None
    fps: float = 15.0


class ExperimentRecordingStopRequest(BaseModel):
    camera_ids: Optional[List[str]] = None
    sample_interval: float = 3.0
    max_frames: int = 30
    auto_analyze: bool = True


# API 

@app.get("/api/v1/experiments", tags=["experiments"])
async def list_experiments(
    limit: int = 50,
    offset: int = 0,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    all_exps = [_merge_experiment_with_task(exp) for exp in _load_experiments()]
    all_exps = _scope_filter_experiments(auth_ctx, all_exps)
    for exp in all_exps:
        experiment_id = str(exp.get("experiment_id") or "")
        if experiment_id:
            exp["key_action_summary"] = _key_action_summary_for_experiment(experiment_id)
    all_exps.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {
        "total": len(all_exps),
        "experiments": all_exps[offset:offset + limit],
    }


@app.post("/api/v1/experiments", tags=["experiments"])
async def create_experiment(
    req: CreateExperimentRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    if Experiment is None:
        raise HTTPException(status_code=503, detail="Experiment service not available")

    exp = Experiment(
        title=req.title,
        description=req.description,
        context_inputs=[{"text": req.context_text, "uploaded_at": datetime.now().isoformat()}] if req.context_text else [],
        protocol_text=req.protocol_text,
    )
    exp.status = ExperimentStatus.CREATED
    exp.avg_confidence = None
    exp_dict = _normalize_experiment_dict(exp.to_dict())
    _ensure_experiment_run_metadata(exp_dict)
    exp_dict["output_paths"] = {
        **(exp_dict.get("output_paths") or {}),
        **_write_experiment_run_artifacts(exp_dict["experiment_id"], exp_dict, material_stream=[]),
    }
    _save_experiment(exp_dict)
    _initialize_waiting_analysis_task(exp_dict["experiment_id"], exp_dict)
    return {"experiment_id": exp.experiment_id, "experiment": exp_dict}


@app.delete("/api/v1/experiments/{experiment_id}", tags=["experiments"])
async def delete_experiment(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp_dir = _experiment_output_dir(safe_experiment_id)
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"Experiment {safe_experiment_id} not found")
    import shutil
    shutil.rmtree(exp_dir, ignore_errors=True)
    _EXPERIMENTS.pop(safe_experiment_id, None)
    alias_dir = PROJECT_ROOT / "outputs" / "experiments"
    if alias_dir.exists():
        for entry in alias_dir.iterdir():
            if entry.is_junction() or entry.is_symlink():
                try:
                    if entry.resolve() == exp_dir or not entry.exists():
                        entry.unlink()
                except Exception:
                    pass
    return {"deleted": safe_experiment_id}


@app.get("/api/v1/experiments/{experiment_id}", tags=["experiments"])
async def get_experiment(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    return _merge_experiment_with_task(await get_experiment_dict(safe_experiment_id))


@app.get("/api/v1/experiments/{experiment_id}/status", tags=["experiments"])
async def get_experiment_status(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    task_state = _experiment_task_state(safe_experiment_id)
    if task_state:
        return task_state
    exp = _merge_experiment_with_task(await get_experiment_dict(safe_experiment_id))
    primary_source = exp.get("video_paths", [None])[0] if exp.get("video_paths") else None
    if not primary_source and exp.get("video_inputs"):
        primary_source = exp["video_inputs"][0].get("video_path")
    return {
        "task_id": exp.get("analysis_job_id"),
        "experiment_id": safe_experiment_id,
        "status": exp.get("status", "created"),
        "current_stage": exp.get("processing_stage", ProcessStage.INGESTION.value),
        "progress": 0.0,
        "video_path": primary_source,
        "error_type": None,
        "error_message": exp.get("processing_error"),
        "started_at": exp.get("started_at"),
        "completed_at": exp.get("completed_at"),
        "output_paths": exp.get("output_paths", {}),
    }


@app.post("/api/v1/experiments/{experiment_id}/upload/video", tags=["experiments"])
async def upload_video(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    upload_dir = PROJECT_ROOT / "uploads" / "experiments" / safe_experiment_id / "videos"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_upload_filename(file.filename, default_name="video_upload.mp4")
    file_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    file_size = await _save_upload_file(file, file_path, max_bytes=MAX_VIDEO_UPLOAD_BYTES)

    exp.setdefault("video_paths", [])
    exp.setdefault("video_metadata", [])
    exp.setdefault("video_inputs", [])
    exp["video_paths"].append(str(file_path))
    video_index = len(exp["video_paths"]) - 1
    video_metadata = _probe_video_metadata(file_path, video_index)
    from labsopguard.video_input_schema import VideoInputValidationError, normalize_video_input
    try:
        video_metadata, _ = normalize_video_input(video_metadata, index=video_index, strict=True)
    except VideoInputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    exp["video_metadata"].append(video_metadata)
    exp["video_inputs"].append(video_metadata)
    exp["video_asset_id"] = exp.get("video_asset_id") or f"{safe_experiment_id}:video:0"
    if exp.get("status") in {"created", ExperimentStatus.CREATED.value, ExperimentStatus.DRAFT.value}:
        exp["status"] = ExperimentStatus.VIDEO_UPLOADED.value
    _ensure_experiment_run_metadata(exp)
    exp["output_paths"] = {
        **(exp.get("output_paths") or {}),
        **_write_experiment_run_artifacts(safe_experiment_id, exp),
    }
    _save_experiment(exp)
    analysis_task = _queue_experiment_auto_analysis(
        experiment_id=safe_experiment_id,
        background_tasks=background_tasks,
        source_ref=str(file_path),
        source_type="file",
        trigger="video_upload",
    )

    return {
        "experiment_id": safe_experiment_id,
        "video_path": str(file_path),
        "video_asset_id": exp["video_asset_id"],
        "status": exp["status"],
        "size": file_size,
        "video_metadata": video_metadata,
        "analysis_task": analysis_task,
    }


@app.post("/api/v1/experiments/{experiment_id}/key-actions/upload-and-run", tags=["experiments"])
async def upload_and_run_key_actions(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    first_person_video: Optional[UploadFile] = File(None),
    third_person_video: Optional[UploadFile] = File(None),
    top_video: Optional[UploadFile] = File(None),
    bottom_video: Optional[UploadFile] = File(None),
    session_start_time: Optional[str] = Form(None),
    sample_fps: float = Form(2.0),
    start_threshold: float = Form(0.18),
    end_threshold: float = Form(0.08),
    start_min_duration_sec: float = Form(1.0),
    end_min_duration_sec: float = Form(2.0),
    merge_gap_sec: float = Form(3.0),
    min_segment_duration_sec: float = Form(2.0),
    buffer_sec: float = Form(1.0),
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Upload dual-view videos and run key-action indexing plus report delivery."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)

    def has_upload(upload: Optional[UploadFile]) -> bool:
        return bool(upload and upload.filename)

    third_upload = (
        third_person_video
        if has_upload(third_person_video)
        else bottom_video
        if has_upload(bottom_video)
        else None
    )
    first_upload = (
        first_person_video
        if has_upload(first_person_video)
        else top_video
        if has_upload(top_video)
        else None
    )
    if first_upload is third_upload:
        first_upload = None
    if third_upload is None:
        raise HTTPException(status_code=400, detail="A third_person_video/bottom_video file is required")

    upload_dir = _experiment_output_dir(safe_experiment_id) / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)

    third_name = _sanitize_upload_filename(third_upload.filename, default_name="third_person.mp4")
    third_path = upload_dir / f"third_{uuid.uuid4().hex[:8]}_{third_name}"
    third_size = await _save_upload_file(third_upload, third_path, max_bytes=MAX_VIDEO_UPLOAD_BYTES)

    first_path: Optional[Path] = None
    first_size: Optional[int] = None
    if first_upload is not None:
        first_name = _sanitize_upload_filename(first_upload.filename, default_name="first_person.mp4")
        first_path = upload_dir / f"first_{uuid.uuid4().hex[:8]}_{first_name}"
        first_size = await _save_upload_file(first_upload, first_path, max_bytes=MAX_VIDEO_UPLOAD_BYTES)

    def register_video(path: Path, *, view_type: str) -> Dict[str, Any]:
        exp.setdefault("video_paths", [])
        exp.setdefault("video_metadata", [])
        exp.setdefault("video_inputs", [])
        exp["video_paths"].append(str(path))
        video_index = len(exp["video_inputs"])
        metadata = _probe_video_metadata(path, video_index)
        metadata["view_type"] = view_type
        metadata["role"] = view_type
        from labsopguard.video_input_schema import VideoInputValidationError, normalize_video_input

        try:
            normalized, _ = normalize_video_input(metadata, index=video_index, strict=True)
        except VideoInputValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        exp["video_metadata"].append(normalized)
        exp["video_inputs"].append(normalized)
        return normalized

    third_metadata = register_video(third_path, view_type="third_person")
    first_metadata = register_video(first_path, view_type="first_person") if first_path else None
    inferred_start = session_start_time or _infer_key_action_start_time(
        third_upload.filename or "",
        first_upload.filename if first_upload else "",
    )
    detection_config = {
        "sample_fps": float(sample_fps),
        "start_threshold": float(start_threshold),
        "end_threshold": float(end_threshold),
        "start_min_duration_sec": float(start_min_duration_sec),
        "end_min_duration_sec": float(end_min_duration_sec),
        "merge_gap_sec": float(merge_gap_sec),
        "min_segment_duration_sec": float(min_segment_duration_sec),
        "buffer_sec": float(buffer_sec),
        "motion_normalization": "adaptive",
        "roi_mode": "manifest_or_default",
    }
    detection_config = _with_default_key_action_yolo_config(detection_config)

    exp["video_asset_id"] = exp.get("video_asset_id") or f"{safe_experiment_id}:key-action-video:0"
    task_id = exp.get("analysis_job_id") or exp.get("run_id") or f"key_action_{safe_experiment_id}"
    exp["analysis_job_id"] = task_id
    if exp.get("status") in {"created", ExperimentStatus.CREATED.value, ExperimentStatus.DRAFT.value}:
        exp["status"] = ExperimentStatus.VIDEO_UPLOADED.value
    _ensure_experiment_run_metadata(exp)
    exp["output_paths"] = {
        **(exp.get("output_paths") or {}),
        **_write_experiment_run_artifacts(safe_experiment_id, exp),
        "key_action_index": str(_key_action_output_dir(safe_experiment_id)),
    }
    exp["key_action_index"] = {
        "status": "queued",
        "third_person_video_path": str(third_path),
        "first_person_video_path": str(first_path) if first_path else None,
        "session_start_time": inferred_start,
        "output_dir": str(_key_action_output_dir(safe_experiment_id)),
    }
    _save_experiment(exp)

    status = _write_key_action_status(
        safe_experiment_id,
        {
            "status": "queued",
            "progress": 0.0,
            "message": "Videos uploaded; key-action pipeline queued",
            "third_person_video_path": str(third_path),
            "first_person_video_path": str(first_path) if first_path else None,
            "session_start_time": inferred_start,
            "detection_config": detection_config,
            "output_dir": str(_key_action_output_dir(safe_experiment_id)),
        },
    )
    _persist_experiment_task_state(
        safe_experiment_id,
        {
            "task_id": task_id,
            "experiment_id": safe_experiment_id,
            "status": "queued",
            "current_stage": "key_action_index",
            "progress": 0.0,
            "message": "Videos uploaded; key-action pipeline queued",
            "video_path": str(third_path),
            "started_at": None,
            "completed_at": None,
            "output_paths": exp.get("output_paths", {}),
            "error_type": None,
            "error_message": None,
        },
    )
    background_tasks.add_task(
        _run_key_action_index_task,
        safe_experiment_id,
        third_person_video_path=str(third_path),
        first_person_video_path=str(first_path) if first_path else None,
        session_start_time=inferred_start,
        detection_config=detection_config,
    )
    return {
        "experiment_id": safe_experiment_id,
        "status": status,
        "third_person_video_path": str(third_path),
        "first_person_video_path": str(first_path) if first_path else None,
        "third_size": third_size,
        "first_size": first_size,
        "third_person_metadata": third_metadata,
        "first_person_metadata": first_metadata,
    }


@app.post("/api/v1/experiments/{experiment_id}/upload/stream", tags=["experiments"])
async def upload_stream(
    experiment_id: str,
    req: UploadStreamRequest,
    background_tasks: BackgroundTasks,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Register a live or network stream source for an experiment."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    if req.sync_confidence is not None and not 0.0 <= float(req.sync_confidence) <= 1.0:
        raise HTTPException(status_code=400, detail="sync_confidence must be between 0 and 1")
    exp = await get_experiment_dict(safe_experiment_id)
    exp.setdefault("video_metadata", [])
    exp.setdefault("video_inputs", [])
    stream_index = len(exp["video_inputs"])
    provided_fields = set(getattr(req, "model_fields_set", None) or getattr(req, "__fields_set__", set()) or set())
    raw_descriptor = {
        "video_index": stream_index,
        "video_path": req.source,
        "source": req.source,
        "source_type": req.source_type,
        "ingest_mode": req.source_type,
        "camera_id": req.camera_id,
        "view_type": req.view_type,
        "role": req.role or req.view_type,
        "source_group": req.source_group,
        "sync_group": req.sync_group,
        "start_offset_sec": req.start_offset_sec,
        "offset_source": "explicit" if "start_offset_sec" in provided_fields else None,
        "capture_duration_sec": req.capture_duration_sec,
        "is_live_source": req.source_type in {"rtsp", "usb"},
    }
    for field in (
        "sync_method",
        "sync_anchors",
        "hardware_timecode_start_sec",
        "sync_board_offset_sec",
        "clock_scale",
        "clock_drift_ppm",
        "sync_confidence",
    ):
        value = getattr(req, field, None)
        if value is not None:
            raw_descriptor[field] = value
    from labsopguard.video_input_schema import VideoInputValidationError, normalize_video_input
    try:
        stream_descriptor, validation_warnings = normalize_video_input(raw_descriptor, index=stream_index, strict=True)
    except VideoInputValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    exp["video_inputs"].append(stream_descriptor)
    exp["video_metadata"].append(dict(stream_descriptor))
    exp["video_asset_id"] = exp.get("video_asset_id") or f"{safe_experiment_id}:video:0"
    if exp.get("status") in {"created", ExperimentStatus.CREATED.value, ExperimentStatus.DRAFT.value}:
        exp["status"] = ExperimentStatus.VIDEO_UPLOADED.value
    _ensure_experiment_run_metadata(exp)
    exp["output_paths"] = {
        **(exp.get("output_paths") or {}),
        **_write_experiment_run_artifacts(safe_experiment_id, exp),
    }
    _save_experiment(exp)
    analysis_task = None
    if req.auto_analyze:
        analysis_task = _queue_experiment_auto_analysis(
            experiment_id=safe_experiment_id,
            background_tasks=background_tasks,
            source_ref=req.source,
            source_type=req.source_type,
            trigger="stream_upload",
        )
    return {
        "experiment_id": safe_experiment_id,
        "stream": stream_descriptor,
        "status": exp["status"],
        "validation_warnings": validation_warnings,
        "analysis_task": analysis_task,
    }


@app.post("/api/v1/experiments/{experiment_id}/timeline-alignment", tags=["experiments"])
async def update_experiment_timeline_alignment(
    experiment_id: str,
    req: TimelineAlignmentUpdateRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Update per-stream time alignment metadata and rebuild alignment artifacts."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    exp.setdefault("video_inputs", [])
    allowed_fields = {
        "sync_method",
        "sync_anchors",
        "sync_group",
        "start_offset_sec",
        "hardware_timecode_start_sec",
        "sync_board_offset_sec",
        "clock_scale",
        "clock_drift_ppm",
        "sync_confidence",
        "alignment_status",
        "offset_source",
    }
    allowed_statuses = {"pending", "explicit", "shared_recording", "calibrated"}
    updated_streams: List[Dict[str, Any]] = []
    streams = req.streams or []
    if streams and req.rebuild_only:
        raise HTTPException(status_code=400, detail="rebuild_only cannot include stream updates")
    for update in streams:
        if not isinstance(update, dict):
            raise HTTPException(status_code=400, detail="stream update must be an object")
        unknown_fields = sorted(set(update) - allowed_fields - {"video_index", "media_asset_id", "asset_id", "camera_id"})
        if unknown_fields:
            raise HTTPException(status_code=400, detail=f"Unsupported alignment fields: {unknown_fields}")
        if "alignment_status" in update and str(update["alignment_status"]).lower() not in allowed_statuses:
            raise HTTPException(status_code=400, detail=f"alignment_status must be one of {sorted(allowed_statuses)}")
        if "sync_confidence" in update and _bounded_confidence(update.get("sync_confidence")) is None:
            raise HTTPException(status_code=400, detail="sync_confidence must be numeric")
        if "sync_confidence" in update and not 0.0 <= float(update["sync_confidence"]) <= 1.0:
            raise HTTPException(status_code=400, detail="sync_confidence must be between 0 and 1")
        if "clock_scale" in update:
            clock_scale = _optional_float(update.get("clock_scale"))
            if clock_scale is None or clock_scale <= 0:
                raise HTTPException(status_code=400, detail="clock_scale must be positive")

        target = None
        for descriptor in exp.get("video_inputs") or []:
            if not isinstance(descriptor, dict):
                continue
            if update.get("video_index") is not None:
                try:
                    if int(descriptor.get("video_index", -1)) == int(update["video_index"]):
                        target = descriptor
                        break
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail="video_index must be an integer") from exc
            for key in ("media_asset_id", "asset_id", "camera_id"):
                if update.get(key) is not None and str(descriptor.get(key) or "") == str(update[key]):
                    target = descriptor
                    break
            if target is not None:
                break
        if target is None:
            raise HTTPException(status_code=404, detail="Stream descriptor not found")
        for field in allowed_fields:
            if field in update:
                target[field] = update[field]
        if "start_offset_sec" in update and "offset_source" not in update:
            target["offset_source"] = "explicit"
        if "sync_confidence" in target:
            target["sync_confidence"] = _bounded_confidence(target["sync_confidence"])
        updated_streams.append(dict(target))

    exp["video_metadata"] = [dict(item) for item in exp.get("video_inputs") or [] if isinstance(item, dict)]
    _ensure_experiment_run_metadata(exp)
    exp["output_paths"] = {
        **(exp.get("output_paths") or {}),
        **_write_experiment_run_artifacts(safe_experiment_id, exp),
    }
    _save_experiment(exp)
    alignment_path = _experiment_output_artifact_paths(safe_experiment_id, exp)["timeline_alignment_json"]
    return {
        "experiment_id": safe_experiment_id,
        "updated": len(updated_streams),
        "rebuild_only": req.rebuild_only,
        "timeline_alignment": _load_json_if_exists(alignment_path) if alignment_path else None,
        "output_paths": exp.get("output_paths", {}),
    }


@app.get("/api/v1/experiments/{experiment_id}/video", tags=["experiments"])
async def get_experiment_video(
    experiment_id: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Stream the primary uploaded experiment video."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    experiment_dir = _experiment_output_dir(safe_experiment_id)
    _raise_if_source_video_candidates_escape_project(
        _experiment_source_video_candidates(experiment_dir, exp),
        "Video path must stay inside project root",
    )
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    file_path = artifacts.get("source_video")
    if file_path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    file_path = file_path.resolve()
    try:
        file_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Video path must stay inside project root") from exc
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return _serve_project_file(file_path, request, media_type="video/mp4")


@app.get("/api/v1/experiments/{experiment_id}/artifacts/{artifact_name}", tags=["experiments"])
async def get_experiment_artifact(
    experiment_id: str,
    artifact_name: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    if artifact_name == "source_video":
        experiment_dir = _experiment_output_dir(safe_experiment_id)
        _raise_if_source_video_candidates_escape_project(
            _experiment_source_video_candidates(experiment_dir, exp),
            "Artifact path must stay inside project root",
        )
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    if artifact_name not in artifacts:
        raise HTTPException(status_code=404, detail="Artifact not found")
    artifact_path = artifacts[artifact_name]
    if artifact_path is None or not artifact_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not ready: {artifact_name}")
    artifact_path = artifact_path.resolve()
    try:
        artifact_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Artifact path must stay inside project root") from exc

    return _serve_project_file(artifact_path, request)


@app.get("/api/v1/experiments/{experiment_id}/files/{file_path:path}", tags=["experiments"])
async def serve_experiment_file(
    experiment_id: str,
    file_path: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Serve any file under an experiment's output directory (clips, keyframes, etc)."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp_dir = _experiment_output_dir(safe_experiment_id)
    target = (exp_dir / file_path).resolve()
    try:
        target.relative_to(exp_dir.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return _serve_project_file(_browser_playable_material_clip(target), request)


@app.get("/api/v1/experiments/{experiment_id}/material-references/files/{file_path:path}", tags=["experiments"])
async def serve_experiment_material_reference_file(
    experiment_id: str,
    file_path: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Serve approved material-reference files from the experiment's formal delivery folder."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    rel = Path(file_path)
    if rel.is_absolute():
        raise HTTPException(status_code=403, detail="Path traversal not allowed")
    checked_inside_delivery = False
    for root in _material_reference_root_candidates(_experiment_output_dir(safe_experiment_id)):
        root_resolved = root.resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            continue
        checked_inside_delivery = True
        if target.exists() and target.is_file():
            return _serve_project_file(_browser_playable_material_clip(target), request)
    if checked_inside_delivery:
        raise HTTPException(status_code=404, detail="File not found")
    raise HTTPException(status_code=403, detail="Path traversal not allowed")


@app.get("/api/v1/experiments/{experiment_id}/materials/search", tags=["experiments"])
async def search_experiment_materials(
    experiment_id: str,
    objects: Optional[str] = None,
    actions: Optional[str] = None,
    event_type: Optional[str] = None,
    actor_name: Optional[str] = None,
    display_name: Optional[str] = None,
    source_container_class: Optional[str] = None,
    target_container_class: Optional[str] = None,
    start_time_sec: Optional[float] = None,
    end_time_sec: Optional[float] = None,
    camera_id: Optional[str] = None,
    stream_id: Optional[str] = None,
    has_clip: Optional[bool] = None,
    clip_exists: Optional[bool] = None,
    text: Optional[str] = None,
    embedding_text: Optional[str] = None,
    limit: int = 50,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Search time-anchored material stream items by object/action/time/camera/clip/text."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    index_path = _ensure_material_index(safe_experiment_id, artifacts)

    from labsopguard.retrieval import MaterialQuery, MaterialRetrievalIndex
    from labsopguard.event_preprocessing.material_index_writer import EventMaterialIndexWriter

    index = MaterialRetrievalIndex(index_path)
    try:
        items = index.query(
            MaterialQuery(
                objects=_split_csv_filter(objects),
                actions=_split_csv_filter(actions),
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                camera_id=camera_id,
                stream_id=stream_id,
                has_clip=has_clip,
                clip_exists=clip_exists,
                text=text,
                embedding_text=embedding_text,
                limit=min(max(int(limit), 1), 500),
            )
        )
        embedding_mode = index.embedding_provider.mode
    finally:
        index.close()

    event_items = []
    event_filter_active = bool(event_type or actor_name or display_name or source_container_class or target_container_class)
    if event_filter_active:
        event_index = EventMaterialIndexWriter(index_path)
        try:
            event_items = event_index.query_events(
                experiment_id=safe_experiment_id,
                event_type=event_type,
                actor_name=actor_name,
                display_name=display_name,
                source_container_class=source_container_class,
                target_container_class=target_container_class,
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                text=text,
                limit=min(max(int(limit), 1), 500),
            )
        finally:
            event_index.close()

    for item in items:
        item.pop("embedding_json", None)
    if event_filter_active:
        items = event_items
    return {
        "experiment_id": safe_experiment_id,
        "total": len(items),
        "items": items,
        "index_path": str(index_path),
        "embedding_mode": embedding_mode if embedding_text else None,
    }


@app.get("/api/v1/experiments/{experiment_id}/materials/health", tags=["experiments"])
async def get_experiment_material_index_health(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Return material-index health, including broken clip references."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    index_path = _ensure_material_index(safe_experiment_id, artifacts)

    from labsopguard.retrieval import MaterialRetrievalIndex

    index = MaterialRetrievalIndex(index_path)
    try:
        health = index.health_check()
    finally:
        index.close()
    from labsopguard.ops_metrics import set_material_health_metrics

    set_material_health_metrics(safe_experiment_id, health)
    return {
        "experiment_id": safe_experiment_id,
        "index_path": str(index_path),
        **health,
    }


@app.post("/api/v1/experiments/{experiment_id}/materials/publish", tags=["experiments"])
async def publish_experiment_materials(
    experiment_id: str,
    force_rebuild_events: bool = False,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Publish canonical event asset packs into friendly browsable material archives.

    If materials/events/ is absent but a source video is known, EventPreprocessingEngine
    is run automatically before publishing.
    """
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    experiment_record = await get_experiment_dict(safe_experiment_id)
    exp_dir = _experiment_output_dir(safe_experiment_id)
    events_dir = exp_dir / "materials" / "events"

    # Auto-trigger when directory is absent OR empty (empty dir = engine ran but produced nothing)
    events_missing = force_rebuild_events or not events_dir.exists() or not any(events_dir.iterdir())
    if events_missing:
        physical_events_path = exp_dir / "physical_events.json"
        source_video = (
            (experiment_record.get("video_inputs") or [{}])[0].get("video_path", "")
            or (experiment_record.get("video_paths") or [None])[0]
        )
        if source_video and Path(source_video).exists():
            if force_rebuild_events and events_dir.exists():
                shutil.rmtree(events_dir)
            logger.info(
                "Auto-triggering EventPreprocessingEngine for %s (force_rebuild_events=%s)",
                safe_experiment_id,
                force_rebuild_events,
            )
            material_stream_raw = _load_json_if_exists(exp_dir / "material_stream.json") or []
            material_stream = material_stream_raw if isinstance(material_stream_raw, list) else []
            if force_rebuild_events:
                material_stream = [
                    item
                    for item in material_stream
                    if not (isinstance(item, dict) and item.get("schema_version") == "material_stream.event.v1")
                ]
            output_paths = {
                "source_video": str(source_video),
                "analysis_json": str(exp_dir / "analysis.json"),
                "material_index": str(exp_dir / "material_index.sqlite"),
            }
            _run_event_preprocessing_for_output(
                experiment_id=safe_experiment_id,
                experiment_record=experiment_record,
                output_dir=exp_dir,
                output_paths=output_paths,
                material_stream=material_stream,
            )
        elif not physical_events_path.exists():
            raise HTTPException(status_code=404, detail="Event assets are not ready; no source video or physical_events.json found")

    from labsopguard.material_publishing import SemanticMaterialPublisher

    try:
        result = SemanticMaterialPublisher(exp_dir, experiment_id=safe_experiment_id).publish()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Re-run step bridge with freshly generated events
    physical_events_payload = _load_json_if_exists(exp_dir / "physical_events.json") or {}
    _run_step_bridge_for_output(
        experiment_id=safe_experiment_id,
        output_dir=exp_dir,
        physical_events_payload=physical_events_payload,
    )

    return {
        "experiment_id": safe_experiment_id,
        "published_total": result["published_materials"]["total"],
        "event_preprocessing_rebuilt": bool(force_rebuild_events),
        "published_materials": result["published_materials"],
        "upload_manifest": result["upload_manifest"],
    }


@app.get("/api/v1/experiments/{experiment_id}/materials/published", tags=["experiments"])
async def get_published_experiment_materials(
    experiment_id: str,
    request: Request,
    limit: Optional[int] = None,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    exp_dir = _experiment_output_dir(safe_experiment_id)
    effective_limit = min(limit, 500) if limit is not None and limit > 0 else None
    published_json = exp_dir / "published_materials.json"
    signature_parts: Dict[str, Any] = {
        "experiment": _experiment_state_cache_token(exp),
        "published_json": _path_cache_token(published_json),
        "material_reference_roots": [
            _directory_tree_cache_token(root, max_entries=128)
            for root in _material_reference_root_candidates(exp_dir)
        ],
        "limit": effective_limit,
    }
    if not published_json.exists():
        signature_parts["materials_events"] = _directory_tree_cache_token(exp_dir / "materials" / "events", max_entries=512)
        signature_parts["physical_events"] = _path_cache_token(exp_dir / "physical_events.json")

    def build_payload() -> Dict[str, Any]:
        payload = _published_material_items(exp_dir, safe_experiment_id)
        if effective_limit is not None:
            items = payload.get("items") if isinstance(payload, dict) else None
            if isinstance(items, list):
                limited_items = items[:effective_limit]
                payload = {**payload, "items": limited_items, "returned": len(limited_items)}
        return payload

    return _cached_experiment_json_response(
        request,
        cache_key=f"experiment:{safe_experiment_id}:materials:published:{effective_limit or 'all'}",
        signature=_experiment_detail_cache_signature(signature_parts),
        build_payload=build_payload,
    )


@app.get("/api/v1/experiments/{experiment_id}/materials/candidates", tags=["experiments"])
async def get_experiment_material_candidates(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    return _material_candidates_payload(safe_experiment_id)


@app.post("/api/v1/experiments/{experiment_id}/materials/candidates/{candidate_group_id}/approve", tags=["experiments"])
async def approve_experiment_material_candidate(
    experiment_id: str,
    candidate_group_id: str,
    request: MaterialCandidateApprovalRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    candidate_ids = []
    for values in (request.candidate_ids, request.selected_keyframe_ids, request.selected_clip_ids):
        candidate_ids.extend([str(item) for item in (values or []) if str(item).strip()])
    try:
        from key_action_indexer.material_references import approve_material_candidates

        approval = approve_material_candidates(
            _key_action_output_dir(safe_experiment_id),
            candidate_group_id=candidate_group_id,
            candidate_ids=candidate_ids or None,
            reviewer=request.reviewer,
            notes=request.notes,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Material candidate approval failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    exp_dir = _experiment_output_dir(safe_experiment_id)
    published_materials = _sync_published_materials_from_references(exp_dir, safe_experiment_id)
    workspace_reindex = _rebuild_workspace_published_materials_index_quietly()
    return {
        "experiment_id": safe_experiment_id,
        "candidate_group_id": candidate_group_id,
        "approval": approval,
        "candidates": _material_candidates_payload(safe_experiment_id),
        "published_materials": published_materials,
        "workspace_published_materials_reindex": workspace_reindex,
    }


@app.get("/api/v1/experiments/{experiment_id}/materials/diagnostics", tags=["experiments"])
async def get_experiment_material_diagnostics(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    exp_dir = _experiment_output_dir(safe_experiment_id)
    payload = _published_material_items(exp_dir, safe_experiment_id)
    return _build_material_diagnostics(safe_experiment_id, payload)


@app.get("/api/v1/experiments/{experiment_id}/materials/upload-manifest", tags=["experiments"])
async def get_experiment_material_upload_manifest(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    manifest_path = _experiment_output_dir(safe_experiment_id) / "upload_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Upload manifest is not ready; run materials/publish first")
    return _load_json_if_exists(manifest_path) or {}


@app.post("/api/v1/experiments/{experiment_id}/materials/upload", tags=["experiments"])
async def upload_experiment_published_materials(
    experiment_id: str,
    request: MaterialUploadRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    manifest_path = _experiment_output_dir(safe_experiment_id) / "upload_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Upload manifest is not ready; run materials/publish first")
    from labsopguard.material_publishing import uploader_for

    uploader = uploader_for(request.provider)
    try:
        if request.provider.lower() in {"local", "nas"}:
            destination = request.destination_root or str(PROJECT_ROOT / "outputs" / "published_uploads" / safe_experiment_id)
            result = uploader.upload_manifest(manifest_path, destination_root=destination)
        else:
            result = uploader.upload_manifest(manifest_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"experiment_id": safe_experiment_id, **result.to_dict()}


@app.post("/api/v1/experiments/{experiment_id}/materials/reindex", tags=["experiments"])
async def rebuild_experiment_material_index_api(
    experiment_id: str,
    force: bool = True,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    from labsopguard.material_maintenance import rebuild_experiment_material_index

    return rebuild_experiment_material_index(_experiment_output_dir(safe_experiment_id), force=force)


@app.post("/api/v1/materials/reindex", tags=["materials"])
async def rebuild_workspace_material_index_api(
    force_experiment_indexes: bool = False,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import rebuild_workspace_material_index

    return rebuild_workspace_material_index(
        PROJECT_ROOT / "outputs" / "experiments",
        _workspace_material_index_path(),
        force_experiment_indexes=force_experiment_indexes,
    )


@app.get("/api/v1/materials/health", tags=["materials"])
async def get_workspace_material_health(
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import scan_experiment_material_health

    report = scan_experiment_material_health(PROJECT_ROOT / "outputs" / "experiments")
    from labsopguard.ops_metrics import set_material_health_metrics

    set_material_health_metrics(
        "workspace",
        {
            "total_items": report.get("total_items", 0),
            "broken_clip_reference_count": report.get("total_broken_clip_references", 0),
        },
    )
    return report


@app.post("/api/v1/materials/published/reindex", tags=["materials"])
async def rebuild_workspace_published_materials_api(
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import rebuild_workspace_published_materials_index

    return rebuild_workspace_published_materials_index(
        PROJECT_ROOT / "outputs" / "experiments",
        _workspace_published_materials_index_path(),
    )


@app.get("/api/v1/materials/published/health", tags=["materials"])
async def get_workspace_published_materials_health(
    auto_rebuild: bool = True,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import check_workspace_published_materials_lifecycle

    return check_workspace_published_materials_lifecycle(
        PROJECT_ROOT / "outputs" / "experiments",
        _workspace_published_materials_index_path(),
        auto_rebuild=auto_rebuild,
    )


@app.get("/api/v1/materials/published", tags=["materials"])
async def get_workspace_published_materials(
    text: Optional[str] = None,
    event_type: Optional[str] = None,
    actor_name: Optional[str] = None,
    limit: int = 100,
    cursor: Optional[str] = None,
    sort_by: str = "time_start",
    sort_order: str = "asc",
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import check_workspace_published_materials_lifecycle, query_workspace_published_materials

    index_path = _workspace_published_materials_index_path()
    index_lifecycle = check_workspace_published_materials_lifecycle(
        PROJECT_ROOT / "outputs" / "experiments",
        index_path,
        auto_rebuild=True,
    )
    actor_filter = actor_name or auth_ctx.get("actor_scope")
    payload = query_workspace_published_materials(
        index_path,
        text=text,
        event_type=event_type,
        actor_name=actor_filter,
        limit=min(max(int(limit), 1), 500),
        cursor=cursor,
        sort_by=("relevance" if text and sort_by == "relevance" else sort_by),
        sort_order=sort_order,
        operator_role=auth_ctx["operator_role"],
        allowed_experiment_ids=auth_ctx["allowed_experiment_ids"],
    )
    payload["index_lifecycle"] = index_lifecycle
    return _workspace_published_payload_with_media_urls(payload)


@app.post("/api/v1/materials/published/usage/click", tags=["materials"])
async def record_workspace_published_material_click_api(
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import record_workspace_published_material_click, rebuild_workspace_published_materials_index

    payload = await request.json()
    material_id = str(payload.get("material_id") or payload.get("event_id") or "")
    if not material_id:
        raise HTTPException(status_code=400, detail="material_id or event_id is required")
    index_path = _workspace_published_materials_index_path()
    if not index_path.exists():
        rebuild_workspace_published_materials_index(PROJECT_ROOT / "outputs" / "experiments", index_path)
    operator = str(payload.get("operator") or request.headers.get("X-Operator") or "anonymous")
    result = record_workspace_published_material_click(
        index_path,
        material_id,
        experiments_root=PROJECT_ROOT / "outputs" / "experiments",
        operator=operator,
        payload={"usage_type": "click", "operator": operator},
    )
    if not result.get("updated"):
        raise HTTPException(status_code=404, detail=result)
    return result


@app.get("/api/v1/materials/search", tags=["materials"])
async def search_workspace_materials(
    objects: Optional[str] = None,
    actions: Optional[str] = None,
    start_time_sec: Optional[float] = None,
    end_time_sec: Optional[float] = None,
    camera_id: Optional[str] = None,
    stream_id: Optional[str] = None,
    has_clip: Optional[bool] = None,
    clip_exists: Optional[bool] = None,
    text: Optional[str] = None,
    embedding_text: Optional[str] = None,
    limit: int = 50,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.material_maintenance import rebuild_workspace_material_index
    from labsopguard.retrieval import MaterialQuery, MaterialRetrievalIndex

    index_path = _workspace_material_index_path()
    if not index_path.exists():
        rebuild_workspace_material_index(PROJECT_ROOT / "outputs" / "experiments", index_path)
    index = MaterialRetrievalIndex(index_path)
    try:
        items = index.query(
            MaterialQuery(
                objects=_split_csv_filter(objects),
                actions=_split_csv_filter(actions),
                start_time_sec=start_time_sec,
                end_time_sec=end_time_sec,
                camera_id=camera_id,
                stream_id=stream_id,
                has_clip=has_clip,
                clip_exists=clip_exists,
                text=text,
                embedding_text=embedding_text,
                limit=min(max(int(limit), 1), 500),
            )
        )
        embedding_mode = index.embedding_provider.mode
    finally:
        index.close()
    for item in items:
        item.pop("embedding_json", None)
    return {
        "total": len(items),
        "items": items,
        "index_path": str(index_path),
        "embedding_mode": embedding_mode if embedding_text else None,
    }


@app.post("/api/v1/ops/capture/preflight", tags=["ops"])
async def preflight_capture_sources(
    config_path: Optional[str] = None,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.soak_test import load_soak_test_config, preflight_soak_test_sources

    path = _safe_project_path(config_path, PROJECT_ROOT / "configs" / "runtime" / "multicam_soak.yaml")
    config = load_soak_test_config(path)
    return preflight_soak_test_sources(config)


@app.get("/api/v1/ops/workspace-governance", tags=["ops"])
async def get_workspace_governance(
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    from labsopguard.workspace_governance import build_workspace_governance_report

    return build_workspace_governance_report(PROJECT_ROOT.parent)


@app.post("/api/v1/experiments/{experiment_id}/materials/backfill-clip", tags=["experiments"])
async def backfill_experiment_material_clip(
    experiment_id: str,
    request: ClipBackfillRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Cut a historical clip from material_stream segment files."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    material_stream_path = artifacts["material_stream_json"]
    if material_stream_path is None or not material_stream_path.exists():
        raise HTTPException(status_code=404, detail="Material stream is not ready")

    from labsopguard.clip_backfill import backfill_clip_from_material_stream

    clip_id = request.clip_id or f"backfill_{request.camera_id or 'all'}_{request.start_time_sec:.3f}_{request.end_time_sec:.3f}".replace(".", "p")
    output_dir = _experiment_output_dir(safe_experiment_id) / "clips"
    output_path = output_dir / f"{clip_id}.mp4"
    try:
        clip = backfill_clip_from_material_stream(
            material_stream_path,
            start_time_sec=request.start_time_sec,
            end_time_sec=request.end_time_sec,
            camera_id=request.camera_id,
            clip_id=clip_id,
            output_path=output_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Clip backfill failed: {exc}") from exc
    return {
        "experiment_id": safe_experiment_id,
        "clip": clip.to_dict(),
    }


@app.get("/api/v1/experiments/{experiment_id}/materials/timeline", tags=["experiments"])
async def get_experiment_material_timeline(
    experiment_id: str,
    request: Request,
    limit: Optional[int] = None,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Return the unified material timeline for visualization/report pages."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    effective_limit = min(limit, 1000) if limit is not None and limit > 0 else None
    signature = _experiment_detail_cache_signature(
        {
            "experiment": _experiment_state_cache_token(exp),
            "material_stream": _path_cache_token(artifacts.get("material_stream_json")),
            "preprocessing": _path_cache_token(artifacts.get("preprocessing_json")),
            "limit": effective_limit,
        }
    )

    def build_payload() -> Dict[str, Any]:
        material_stream = _load_json_if_exists(artifacts["material_stream_json"]) if artifacts["material_stream_json"] else []
        preprocessing = _load_json_if_exists(artifacts["preprocessing_json"]) if artifacts["preprocessing_json"] else {}
        material_stream = material_stream or []
        preprocessing = preprocessing or {}
        events = preprocessing.get("detected_changes", []) or []
        key_clips = preprocessing.get("key_clips", []) or []
        context_inputs = exp.get("context_inputs", []) or []
        timeline_items: List[Dict[str, Any]] = []
        for item in material_stream:
            timeline_items.append({
                "type": "material",
                "timestamp_sec": item.get("timestamp_sec", item.get("local_timestamp_sec", 0.0)),
                "camera_id": item.get("camera_id"),
                "item": item,
            })
        for event in events:
            timeline_items.append({
                "type": "event",
                "timestamp_sec": event.get("timestamp_sec", 0.0),
                "camera_id": (event.get("metadata") or {}).get("camera_id"),
                "item": event,
            })
        for context in context_inputs:
            timestamp = context.get("timestamp_sec") or context.get("start_time_sec")
            if timestamp is not None:
                timeline_items.append({
                    "type": "context",
                    "timestamp_sec": timestamp,
                    "camera_id": None,
                    "item": context,
                })
        timeline_items.sort(key=lambda item: float(item.get("timestamp_sec") or 0.0))
        total_items = len(timeline_items)
        returned_items = timeline_items[:effective_limit] if effective_limit is not None else timeline_items
        return {
            "experiment_id": safe_experiment_id,
            "schema_version": "material_timeline.v1",
            "material_count": len(material_stream),
            "event_count": len(events),
            "key_clip_count": len(key_clips),
            "context_count": len(context_inputs),
            "camera_ids": sorted({item.get("camera_id") for item in material_stream if item.get("camera_id")}),
            "key_clips": key_clips,
            "total_items": total_items,
            "returned": len(returned_items),
            "items": returned_items,
        }

    return _cached_experiment_json_response(
        request,
        cache_key=f"experiment:{safe_experiment_id}:materials:timeline:{effective_limit or 'all'}",
        signature=signature,
        build_payload=build_payload,
    )



def _contract_status(exp_status: Any, stage: Any, task_status: Any = None) -> str:
    raw_status = str(task_status or exp_status or "").lower()
    raw_stage = str(stage or "").lower()
    if raw_status == "failed" or raw_stage == "failed":
        return "failed"
    if raw_status == "waiting_for_sources" or raw_stage == "waiting_for_sources":
        return "waiting_for_sources"
    if raw_status == "queued":
        return "queued"
    if raw_status == "running":
        if "write" in raw_stage:
            return "writing_back"
        if "output" in raw_stage or "generation" in raw_stage or "render" in raw_stage:
            return "generating_outputs"
        return "analyzing"
    if raw_stage in {"writing_back", "qwen_frame_writeback"}:
        return "writing_back"
    if "output" in raw_stage or "generation" in raw_stage:
        return "generating_outputs"
    if raw_status in {"completed", "analyzed"}:
        return "completed"
    if raw_status in {"uploaded", "video_uploaded"}:
        return "uploaded"
    if raw_status in {"created", ""}:
        return "uploaded" if raw_stage in {"ingestion", "video_uploaded"} else "uploaded"
    return raw_status if raw_status in {"uploaded", "waiting_for_sources", "queued", "analyzing", "generating_outputs", "writing_back", "completed", "failed", "partial_failed"} else "partial_failed"


def _overview_result_version(exp: Dict[str, Any], paths: List[Optional[Path]]) -> str:
    source = {
        "experiment_id": exp.get("experiment_id"),
        "analysis_job_id": exp.get("analysis_job_id"),
        "updated": [str(path.stat().st_mtime_ns) for path in paths if path and path.exists()],
    }
    return hashlib.sha1(json.dumps(source, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def _step_groups(steps: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    official = [s for s in steps if str(s.get("status")) == "confirmed"]
    candidate = [s for s in steps if str(s.get("status")) in {"candidate", "needs_review"}]
    inferred = [s for s in steps if str(s.get("status")) == "inferred"]

    # Steps from _SOP_DEFAULT_STEPS / _enrich_steps_for_bridge have no status field.
    # Treat them as candidate steps so the frontend displays them.
    unclassified = [s for s in steps if not s.get("status")]
    if unclassified and not official and not candidate:
        candidate = [{**s, "status": "candidate"} for s in unclassified]

    if not official and not candidate and inferred:
        candidate = list(inferred)
    return {"official": official, "candidate": candidate, "inferred": inferred}


def _official_step_records_for_overview(experiment_id: str) -> List[Dict[str, Any]]:
    payload = _load_json_if_exists(_experiment_output_dir(experiment_id) / "official_steps.json") or {}
    records = payload.get("official_steps") if isinstance(payload, dict) else []
    active_status = {"approved", "locked", "reopened", "proposed"}
    result = []
    for item in records or []:
        lifecycle = str(item.get("lifecycle_status") or item.get("status") or "")
        if lifecycle not in active_status:
            continue
        result.append({
            "step_id": item.get("protocol_step_id"),
            "step_name": item.get("protocol_step_name"),
            "status": "confirmed" if lifecycle in {"approved", "locked"} else lifecycle,
            "lifecycle_status": lifecycle,
            "official_step_id": item.get("official_step_id"),
            "source_step_candidate_id": item.get("source_step_candidate_id"),
            "linked_event_ids": item.get("linked_event_ids") or [],
            "evidence_bundle": item.get("evidence_bundle") or {},
            "version": item.get("version"),
            "locked": bool(item.get("locked")),
            "metadata_version": item.get("metadata_version"),
        })
    return result


def _candidate_step_records_for_overview(experiment_id: str, official_records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    payload = _load_json_if_exists(_experiment_output_dir(experiment_id) / "step_candidates.json") or {}
    official_sources = {str(item.get("source_step_candidate_id")) for item in official_records if item.get("source_step_candidate_id")}
    candidate = []
    inferred = []
    for item in payload.get("step_candidates") or []:
        if str(item.get("step_candidate_id")) in official_sources:
            continue
        row = {
            "step_id": item.get("protocol_step_id"),
            "step_name": item.get("protocol_step_name"),
            "status": item.get("candidate_status"),
            "step_candidate_id": item.get("step_candidate_id"),
            "matched_event_ids": item.get("matched_event_ids") or [],
            "evidence_grade": item.get("evidence_grade"),
            "review_status": item.get("review_status"),
            "confidence": item.get("candidate_score"),
            "metadata_version": item.get("metadata_version"),
        }
        if item.get("candidate_status") == "inferred":
            inferred.append(row)
        elif item.get("candidate_status") in {"candidate", "needs_review", "confirmed"}:
            candidate.append(row)
    return {"candidate": candidate, "inferred": inferred}


def _extract_scene_sections(frame: Dict[str, Any]) -> Dict[str, Any]:
    raw_description = frame.get("scene_description") or frame.get("description") or ""
    parsed: Dict[str, Any] = {}
    if isinstance(raw_description, str):
        cleaned = raw_description.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json\n", "", 1).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(cleaned[start:end + 1])
                if isinstance(value, dict):
                    parsed = value
            except Exception:
                parsed = {}
    activities = frame.get("detected_activities") or parsed.get("detected_activities") or parsed.get("actions") or []
    objects = frame.get("object_labels") or parsed.get("object_labels") or parsed.get("objects") or []
    step_indicators = frame.get("step_indicators") or parsed.get("step_indicators") or []
    if isinstance(activities, str):
        activities = [activities]
    if isinstance(objects, str):
        objects = [objects]
    if isinstance(step_indicators, str):
        step_indicators = [step_indicators]
    return {
        "description": parsed.get("description") or parsed.get("scene_summary") or raw_description,
        "activities": activities,
        "objects": objects,
        "step_indicators": step_indicators,
        "ppe_assessment": frame.get("ppe_status") or parsed.get("ppe_status") or {},
        "alerts": frame.get("alert_details") or frame.get("alerts") or [],
        "detections": frame.get("detections") or [],
        "raw": parsed or raw_description,
    }


def _parse_datetime_value(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _key_action_session_start_dt(experiment_id: str) -> Optional[datetime]:
    output_dir = _key_action_output_dir(experiment_id)
    manifest = _load_json_if_exists(output_dir / "manifest.json") or {}
    status_payload = _load_json_if_exists(_key_action_status_path(experiment_id)) or {}
    candidates = [
        manifest.get("session_start_time") if isinstance(manifest, dict) else None,
        status_payload.get("session_start_time") if isinstance(status_payload, dict) else None,
    ]
    videos = manifest.get("videos") if isinstance(manifest, dict) else {}
    if isinstance(videos, dict):
        for video in videos.values():
            if isinstance(video, dict):
                candidates.append(video.get("start_time"))
    for candidate in candidates:
        parsed = _parse_datetime_value(candidate)
        if parsed is not None:
            return parsed
    return None


def _seconds_from_key_action_start(experiment_id: str, value: Any) -> Optional[float]:
    parsed = _parse_datetime_value(value)
    start = _key_action_session_start_dt(experiment_id)
    if parsed is None or start is None:
        return None
    if parsed.tzinfo is not None and start.tzinfo is None:
        start = start.replace(tzinfo=parsed.tzinfo)
    if parsed.tzinfo is None and start.tzinfo is not None:
        parsed = parsed.replace(tzinfo=start.tzinfo)
    try:
        return max(0.0, (parsed - start).total_seconds())
    except Exception:
        return None


def _enrich_overview_evidence_refs(experiment_id: str, refs: Any) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for ref in refs or []:
        if not isinstance(ref, dict):
            continue
        row = copy.deepcopy(ref)
        path_value = row.get("path") or row.get("clip_path") or row.get("keyframe_path")
        if path_value:
            url = _key_action_file_url(experiment_id, path_value)
            if url:
                row["url"] = url
        enriched.append(row)
    return enriched


def _process_step_status_for_overview(raw_status: Any) -> str:
    status = str(raw_status or "").strip().lower()
    if status in {"not_observed", "inferred_missing", "missing", "skipped", "branch_not_taken"}:
        return "inferred"
    return "candidate"


def _key_action_step_groups_for_overview(experiment_id: str) -> Dict[str, List[Dict[str, Any]]]:
    output_dir = _key_action_output_dir(experiment_id)
    timeline_rows = _read_jsonl_rows(output_dir / "metadata" / "experiment_process_timeline.jsonl")
    steps: List[Dict[str, Any]] = []
    for index, item in enumerate(timeline_rows, start=1):
        if str(item.get("event_type") or "") != "experiment_step":
            continue
        start_sec = item.get("start_time_sec")
        if start_sec is None:
            start_sec = _seconds_from_key_action_start(experiment_id, item.get("global_time"))
        end_sec = item.get("end_time_sec")
        if end_sec is None:
            end_sec = _seconds_from_key_action_start(experiment_id, item.get("global_end_time"))
        status = _process_step_status_for_overview(item.get("status"))
        step_id = str(item.get("step_id") or item.get("timeline_event_id") or f"key_action_step_{index:03d}")
        row = {
            "step_id": step_id,
            "experiment_id": experiment_id,
            "step_index": item.get("order") or index,
            "step_name": item.get("text") or item.get("name") or step_id,
            "step_description": item.get("reasoning") or "",
            "status": status,
            "source_status": item.get("status"),
            "start_time_sec": start_sec,
            "end_time_sec": end_sec,
            "duration_sec": (float(end_sec) - float(start_sec)) if start_sec is not None and end_sec is not None else None,
            "confidence": item.get("confidence"),
            "step_confidence": "high" if _float_value(item.get("confidence"), 0.0) >= 0.75 else "medium",
            "completed_by_inference": True,
            "requires_human_confirmation": bool(item.get("requires_human_confirmation")),
            "evidence_refs": _enrich_overview_evidence_refs(experiment_id, item.get("evidence_refs")),
            "parameters": [],
            "created_at": item.get("global_time"),
            "updated_at": item.get("global_end_time") or item.get("global_time"),
            "metadata_version": "key_action_process_projection.v1",
        }
        steps.append(row)
    if not steps:
        process = _load_json_if_exists(output_dir / "metadata" / "experiment_process.json") or {}
        for index, item in enumerate(process.get("steps") or [], start=1):
            if not isinstance(item, dict):
                continue
            start_sec = _seconds_from_key_action_start(experiment_id, item.get("global_start_time"))
            end_sec = _seconds_from_key_action_start(experiment_id, item.get("global_end_time"))
            status = _process_step_status_for_overview(item.get("status"))
            step_id = str(item.get("step_id") or f"key_action_step_{index:03d}")
            steps.append(
                {
                    "step_id": step_id,
                    "experiment_id": experiment_id,
                    "step_index": item.get("order") or index,
                    "step_name": item.get("name") or step_id,
                    "step_description": "; ".join(item.get("confidence_reasons") or []),
                    "status": status,
                    "source_status": item.get("status"),
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "duration_sec": (float(end_sec) - float(start_sec)) if start_sec is not None and end_sec is not None else None,
                    "confidence": item.get("confidence"),
                    "step_confidence": "high" if _float_value(item.get("confidence"), 0.0) >= 0.75 else "medium",
                    "completed_by_inference": True,
                    "requires_human_confirmation": bool(item.get("requires_human_confirmation")),
                    "evidence_refs": _enrich_overview_evidence_refs(experiment_id, item.get("evidence_refs")),
                    "parameters": item.get("parameters") or [],
                    "metadata_version": "key_action_process_projection.v1",
                }
            )
    candidate = [step for step in steps if step.get("status") != "inferred"]
    inferred = [step for step in steps if step.get("status") == "inferred"]
    return {"official": [], "candidate": candidate, "inferred": inferred}


def _key_action_overview_counts(experiment_id: str) -> Dict[str, Any]:
    output_dir = _key_action_output_dir(experiment_id)
    metadata_dir = output_dir / "metadata"
    cv_dir = output_dir / "cv_outputs"
    yolo_clip_summary = _load_json_if_exists(metadata_dir / "yolo_clip_summary.json") or {}
    video_understanding_summary = _load_json_if_exists(metadata_dir / "video_understanding_summary.json") or {}
    pipeline_summary = _load_json_if_exists(output_dir / "pipeline_summary.json") or {}
    if not yolo_clip_summary and isinstance(pipeline_summary, dict):
        yolo_clip_summary = pipeline_summary.get("yolo_annotated_clips") or {}
    return {
        "segment_count": _jsonl_row_count(metadata_dir / "key_action_segments.jsonl"),
        "micro_segment_count": _jsonl_row_count(metadata_dir / "micro_segments.jsonl"),
        "yolo_frame_row_count": _jsonl_row_count(cv_dir / "yolo_frame_rows.jsonl"),
        "yolo_annotated_clip_count": int((yolo_clip_summary or {}).get("clips") or 0),
        "yolo_detection_count": int((yolo_clip_summary or {}).get("detections") or 0),
        "video_event_count": int((video_understanding_summary or {}).get("video_event_count") or 0),
        "normalized_object_counts": (video_understanding_summary or {}).get("normalized_object_counts") or {},
        "event_type_counts": (video_understanding_summary or {}).get("event_type_counts") or {},
        "yolo_clip_summary": yolo_clip_summary or {},
        "video_understanding_summary": video_understanding_summary or {},
    }


def _key_action_scene_summary_for_overview(experiment_id: str, counts: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = _key_action_output_dir(experiment_id)
    if not output_dir.exists():
        return {}
    vlm_summary = _load_json_if_exists(output_dir / "metadata" / "scene_vlm_summary.json") or {}
    if isinstance(vlm_summary, dict) and vlm_summary.get("description"):
        return {
            "description": vlm_summary.get("description"),
            "activities": vlm_summary.get("activities") or vlm_summary.get("detected_activities") or [],
            "objects": vlm_summary.get("objects") or vlm_summary.get("object_labels") or [],
            "visible_lab_objects": vlm_summary.get("visible_lab_objects") or vlm_summary.get("object_labels") or [],
            "uncertain_objects": vlm_summary.get("uncertain_objects") or [],
            "step_indicators": vlm_summary.get("step_indicators") or [],
            "ppe_assessment": vlm_summary.get("ppe_assessment") or vlm_summary.get("ppe_status") or {},
            "alerts": vlm_summary.get("alerts") or [],
            "detections": vlm_summary.get("detections") or [],
            "evidence_source": vlm_summary.get("evidence_source") or "qwen_vl_scene_summary",
            "raw": vlm_summary,
        }
    objects = [
        key
        for key, _value in sorted(
            (counts.get("normalized_object_counts") or {}).items(),
            key=lambda item: int(item[1] or 0),
            reverse=True,
        )
    ][:12]
    activities = [
        key
        for key, _value in sorted(
            (counts.get("event_type_counts") or {}).items(),
            key=lambda item: int(item[1] or 0),
            reverse=True,
        )
    ][:10]
    pipeline_summary = _load_json_if_exists(output_dir / "pipeline_summary.json") or {}
    vlm_assist = pipeline_summary.get("key_action_vlm_assist") if isinstance(pipeline_summary, dict) else {}
    vlm_reason = str((vlm_assist or {}).get("reason") or (vlm_assist or {}).get("error") or "")
    process = _load_json_if_exists(output_dir / "metadata" / "experiment_process.json") or {}
    step_indicators = [
        f"{step.get('name') or step.get('step_id')}: {step.get('status')}"
        for step in (process.get("steps") or [])
        if isinstance(step, dict)
    ][:8]
    segment_count = int(counts.get("segment_count") or 0)
    micro_count = int(counts.get("micro_segment_count") or 0)
    detection_count = int(counts.get("yolo_detection_count") or 0)
    event_count = int(counts.get("video_event_count") or 0)
    if not bool((vlm_assist or {}).get("configured")):
        degrade = vlm_reason or "dashscope_api_key_missing"
        description = (
            f"Qwen-VL 场景语义复核未启用（{degrade}）。当前仅展示 YOLO/微片段证据："
            f"{segment_count} 个 episode、{micro_count} 个微片段、{detection_count} 个检测框；"
            f"主要对象包括：{', '.join(objects[:8]) if objects else '暂无对象统计'}。"
        )
        evidence_source = "yolo_micro_evidence_qwen_vl_unavailable"
    else:
        description = (
            "Qwen-VL 已配置，但本次运行没有生成独立场景摘要；"
            f"当前只展示候选素材级 VLM/YOLO 复核证据和 {micro_count} 个微片段统计。"
        )
        evidence_source = "yolo_micro_evidence_scene_summary_not_generated"
    return {
        "description": description,
        "activities": activities,
        "objects": objects,
        "visible_lab_objects": objects,
        "uncertain_objects": [],
        "step_indicators": step_indicators,
        "ppe_assessment": {},
        "alerts": [],
        "detections": [
            {"class_name": key, "count": value}
            for key, value in (counts.get("normalized_object_counts") or {}).items()
        ],
        "evidence_source": evidence_source,
        "raw": counts.get("video_understanding_summary") or {},
    }


def _key_action_video_path(experiment_id: str, view: str) -> Optional[Path]:
    output_dir = _key_action_output_dir(experiment_id)
    manifest = _load_json_if_exists(output_dir / "manifest.json") or {}
    videos = manifest.get("videos") if isinstance(manifest, dict) else {}
    if isinstance(videos, dict):
        row = videos.get(view)
        if isinstance(row, dict) and row.get("path"):
            return Path(str(row.get("path")))
    status_payload = _load_json_if_exists(_key_action_status_path(experiment_id)) or {}
    key = f"{view}_video_path"
    if isinstance(status_payload, dict) and status_payload.get(key):
        return Path(str(status_payload.get(key)))
    return None


def _artifact_metadata_for_experiment_path(
    name: str,
    experiment_id: str,
    path: Optional[Path],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    public_url = _experiment_file_api_path(path, experiment_id) if path else None
    payload = _artifact_metadata(name, path, public_url)
    if extra:
        payload.update(extra)
    return payload


def _key_action_media_artifacts_for_overview(experiment_id: str) -> Dict[str, Dict[str, Any]]:
    output_dir = _key_action_output_dir(experiment_id)
    segment_rows = _read_jsonl_rows(output_dir / "metadata" / "key_action_segments.jsonl")
    first_segment = segment_rows[0] if segment_rows else {}
    focus_window = _load_json_if_exists(output_dir / "metadata" / "experiment_focus_window.json") or {}
    focus_extra = {
        "time_start_sec": focus_window.get("start_sec"),
        "true_start_sec": focus_window.get("true_start_sec"),
        "time_end_sec": focus_window.get("end_sec"),
        "duration_sec": focus_window.get("duration_sec"),
        "global_start_time": focus_window.get("global_start_time"),
        "global_end_time": focus_window.get("global_end_time"),
        "focus_source": focus_window.get("source"),
        "focus_anchor": focus_window.get("anchor"),
    } if isinstance(focus_window, dict) and focus_window.get("start_sec") is not None else {}

    def clip_path(view: str, annotated: bool) -> Optional[Path]:
        ref = first_segment.get(view) if isinstance(first_segment, dict) else None
        if not isinstance(ref, dict):
            return None
        value = ref.get("annotated_clip_path") if annotated else ref.get("clip_path")
        return Path(str(value)) if value else None

    def focus_path(view: str, annotated: bool) -> Optional[Path]:
        suffix = f"{view}_yolo_annotated.mp4" if annotated else f"{view}.mp4"
        return output_dir / "clips" / "experiment_focus" / suffix

    def prefer_existing(*paths: Optional[Path]) -> Optional[Path]:
        for path in paths:
            if path and path.exists():
                return path
        for path in paths:
            if path:
                return path
        return None

    first_video = _key_action_video_path(experiment_id, "first_person")
    third_video = _key_action_video_path(experiment_id, "third_person")
    first_focus_annotated = focus_path("first_person", annotated=True)
    third_focus_annotated = focus_path("third_person", annotated=True)
    first_focus_clip = focus_path("first_person", annotated=False)
    third_focus_clip = focus_path("third_person", annotated=False)
    artifacts = {
        "first_person_video": _artifact_metadata_for_experiment_path("first_person_video", experiment_id, first_video),
        "third_person_video": _artifact_metadata_for_experiment_path("third_person_video", experiment_id, third_video),
        "first_person_experiment_focus_annotated_video": _artifact_metadata_for_experiment_path(
            "first_person_experiment_focus_annotated_video",
            experiment_id,
            first_focus_annotated,
            focus_extra,
        ),
        "third_person_experiment_focus_annotated_video": _artifact_metadata_for_experiment_path(
            "third_person_experiment_focus_annotated_video",
            experiment_id,
            third_focus_annotated,
            focus_extra,
        ),
        "first_person_experiment_focus_video": _artifact_metadata_for_experiment_path(
            "first_person_experiment_focus_video",
            experiment_id,
            first_focus_clip,
            focus_extra,
        ),
        "third_person_experiment_focus_video": _artifact_metadata_for_experiment_path(
            "third_person_experiment_focus_video",
            experiment_id,
            third_focus_clip,
            focus_extra,
        ),
        "first_person_annotated_video": _artifact_metadata_for_experiment_path(
            "first_person_annotated_video",
            experiment_id,
            prefer_existing(first_focus_annotated, clip_path("first_person", annotated=True)),
            focus_extra,
        ),
        "third_person_annotated_video": _artifact_metadata_for_experiment_path(
            "third_person_annotated_video",
            experiment_id,
            prefer_existing(third_focus_annotated, clip_path("third_person", annotated=True)),
            focus_extra,
        ),
        "first_person_key_action_clip": _artifact_metadata_for_experiment_path(
            "first_person_key_action_clip",
            experiment_id,
            prefer_existing(first_focus_clip, clip_path("first_person", annotated=False)),
            focus_extra,
        ),
        "third_person_key_action_clip": _artifact_metadata_for_experiment_path(
            "third_person_key_action_clip",
            experiment_id,
            prefer_existing(third_focus_clip, clip_path("third_person", annotated=False)),
            focus_extra,
        ),
    }
    return artifacts


def _contract_alerts(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alerts: List[Dict[str, Any]] = []
    for frame in frames:
        frame_idx = frame.get("frame_idx")
        timestamp = frame.get("timestamp_sec")
        camera_id = frame.get("camera_id")
        detections = frame.get("detections") or []
        related_objects = sorted({str(det.get("class_name") or det.get("label") or "") for det in detections if isinstance(det, dict) and (det.get("class_name") or det.get("label"))})
        details = frame.get("alert_details") or []
        if not details and frame.get("alerts"):
            details = [{"rule_id": key, "title": key, "message": key, "severity": "medium", "related_classes": []} for key in frame.get("alerts") or []]
        for detail in details:
            rule_id = str(detail.get("rule_id") or detail.get("alert_id") or "alert")
            alert_id = f"{rule_id}:frame:{frame_idx}:t:{timestamp}"
            alerts.append({
                "alert_id": alert_id,
                "rule_name": detail.get("title") or rule_id,
                "rule_id": rule_id,
                "severity": detail.get("severity") or "medium",
                "source_frame": frame_idx,
                "timestamp_sec": timestamp,
                "camera_id": camera_id,
                "event_id": detail.get("event_id"),
                "evidence_refs": detail.get("evidence_refs") or [{"type": "video_frame", "frame_idx": frame_idx, "timestamp_sec": timestamp}],
                "rule_basis": detail.get("rule_basis") or "PPE rule evaluated against actor/scene gating and detector/VLM evidence.",
                "related_objects": detail.get("related_classes") or related_objects,
                "confidence": detail.get("confidence", frame.get("vlm_confidence", 0.0)),
                "message": detail.get("message") or rule_id,
            })
    return alerts


def _artifact_metadata(name: str, path: Optional[Path], public_url: Optional[str] = None) -> Dict[str, Any]:
    exists = bool(path and path.exists())
    return {
        "name": name,
        "ready": exists,
        "kind": path.suffix.lstrip(".") if path else None,
        "size_bytes": path.stat().st_size if exists and path else 0,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists and path else None,
        "url": public_url,
    }


def _build_analysis_overview(exp: Dict[str, Any]) -> Dict[str, Any]:
    experiment_id = exp["experiment_id"]
    artifacts = _experiment_output_artifact_paths(experiment_id, exp)
    task_state = _experiment_task_state(experiment_id)
    key_action_available = _key_action_output_dir(experiment_id).exists() or _key_action_status_path(experiment_id).exists()
    key_action_state = _read_key_action_status(experiment_id) if key_action_available else {}
    if key_action_state.get("status") == "not_started":
        key_action_state = {}
    effective_task_state = key_action_state or task_state
    key_action_counts = _key_action_overview_counts(experiment_id) if key_action_available else {}
    key_action_media_artifacts = _key_action_media_artifacts_for_overview(experiment_id) if key_action_available else {}
    timeline = _load_json_if_exists(artifacts["timeline_json"]) or _empty_timeline_payload(exp)
    steps = _load_json_if_exists(artifacts["steps_json"]) or timeline.get("steps", []) or []
    # Merge status/time from step_candidates.json into steps when available
    step_candidates_data = _load_json_if_exists(_experiment_output_dir(experiment_id) / "step_candidates.json") or {}
    sc_list = step_candidates_data.get("step_candidates") or []
    if sc_list:
        sc_by_id = {str(c.get("protocol_step_id")): c for c in sc_list if isinstance(c, dict)}
        for step in steps:
            sid = str(step.get("protocol_step_id") or step.get("step_id") or "")
            cand = sc_by_id.get(sid)
            if cand:
                step.setdefault("step_id", sid)
                step.setdefault("step_name", cand.get("protocol_step_name") or step.get("protocol_step_name", ""))
                step.setdefault("status", cand.get("candidate_status", "candidate"))
                step.setdefault("confidence", cand.get("candidate_score", 0.0))
                # Time from matched events
                matched_ids = cand.get("matched_event_ids") or []
                pe_data = _load_json_if_exists(_experiment_output_dir(experiment_id) / "physical_events.json") or {}
                pe_events = pe_data.get("events", pe_data) if isinstance(pe_data, dict) else pe_data
                pe_events = pe_events if isinstance(pe_events, list) else []
                matched_events = [e for e in pe_events if str(e.get("event_id")) in set(matched_ids)]
                if matched_events:
                    step.setdefault("start_time_sec", min(float(e.get("start_time_sec") or 0) for e in matched_events))
                    step.setdefault("end_time_sec", max(float(e.get("end_time_sec") or 0) for e in matched_events))
    official_records = _official_step_records_for_overview(experiment_id)
    if official_records:
        candidate_records = _candidate_step_records_for_overview(experiment_id, official_records)
        groups = {"official": official_records, "candidate": candidate_records["candidate"], "inferred": candidate_records["inferred"]}
        display_steps = official_records + candidate_records["candidate"] + candidate_records["inferred"]
    else:
        groups = _step_groups(steps)
        display_steps = steps
        if key_action_available and not (groups["official"] or groups["candidate"] or groups["inferred"]):
            groups = _key_action_step_groups_for_overview(experiment_id)
            display_steps = groups["official"] + groups["candidate"] + groups["inferred"]
    frames = _load_json_if_exists(artifacts["analysis_json"]) or []
    material_stream = _load_json_if_exists(artifacts["material_stream_json"]) or []
    writeback = _load_json_if_exists(_experiment_output_dir(experiment_id) / "qwen_frame_writeback.json") or {}
    status = _contract_status(
        exp.get("status"),
        effective_task_state.get("current_stage")
        or effective_task_state.get("stage")
        or effective_task_state.get("message")
        or exp.get("processing_stage"),
        effective_task_state.get("status"),
    )
    has_writeback_report = bool(writeback and not writeback.get("failures"))
    has_material_writeback = bool(material_stream and exp.get("output_paths"))
    key_action_has_summary = bool(key_action_counts.get("segment_count") or key_action_counts.get("video_event_count"))
    key_action_has_steps = bool(groups["official"] or groups["candidate"] or groups["inferred"])
    key_action_has_artifacts = bool(key_action_counts.get("segment_count") or _material_reference_items(_experiment_output_dir(experiment_id), experiment_id).get("items"))
    key_action_has_annotated = any(
        item.get("ready")
        for name, item in key_action_media_artifacts.items()
        if "annotated" in name
    )
    key_action_completed = str(key_action_state.get("status") or "").lower() == "completed"
    readiness = {
        "summary_ready": bool(frames or material_stream or display_steps or key_action_has_summary),
        "steps_ready": bool(key_action_has_steps),
        "alerts_ready": bool(frames or material_stream or key_action_has_summary),
        "artifacts_ready": bool((artifacts["material_stream_json"].exists() if artifacts.get("material_stream_json") else False) or key_action_has_artifacts),
        "annotated_video_ready": bool((artifacts["annotated_video"] and artifacts["annotated_video"].exists()) or key_action_has_annotated),
        "writeback_ready": bool(has_writeback_report or has_material_writeback or key_action_completed or artifacts.get("professional_report_pdf")),
    }

    if status == "completed" and key_action_completed and key_action_has_summary:
        status = "completed" if readiness["summary_ready"] and readiness["steps_ready"] and readiness["artifacts_ready"] else "partial_failed"
    elif exp.get("status") in {"completed", "analyzed"} and not all(readiness.values()):
        status = "partial_failed"
    elif exp.get("status") in {"completed", "analyzed"} and all(readiness.values()):
        status = "completed"
    elif status == "completed" and not all(readiness.values()):
        status = "partial_failed" if effective_task_state.get("status") == "completed" else status
    total_detections = sum(len(f.get("detections", [])) for f in frames if isinstance(f, dict))
    if not total_detections:
        total_detections = int(key_action_counts.get("yolo_detection_count") or 0)
    alerts = _contract_alerts([f for f in frames if isinstance(f, dict)])
    candidate_count = len(groups["candidate"])
    official_count = len(groups["official"])
    inferred_count = len(groups["inferred"])
    avg_values = [float(s.get("confidence")) for s in display_steps if s.get("confidence") is not None]
    result_version = _overview_result_version(
        exp,
        [
            artifacts.get("official_steps_json"),
            artifacts.get("step_candidates_json"),
            artifacts.get("steps_json"),
            artifacts.get("analysis_json"),
            artifacts.get("material_stream_json"),
            artifacts.get("semantic_sync_anchors_json"),
            _experiment_output_dir(experiment_id) / "qwen_frame_writeback.json",
            _key_action_output_dir(experiment_id) / "pipeline_summary.json",
            _key_action_output_dir(experiment_id) / "metadata" / "experiment_process_timeline.jsonl",
            _key_action_output_dir(experiment_id) / "metadata" / "video_understanding_summary.json",
            _key_action_output_dir(experiment_id) / "metadata" / "yolo_clip_summary.json",
        ],
    )
    run_id = str(effective_task_state.get("task_id") or exp.get("analysis_job_id") or f"{experiment_id}:{result_version}")
    scene_frame = next((f for f in frames if isinstance(f, dict) and (f.get("scene_description") or f.get("detections") or f.get("alerts"))), frames[0] if frames else {})
    scene_summary = _extract_scene_sections(scene_frame if isinstance(scene_frame, dict) else {})
    if key_action_available and not str(scene_summary.get("description") or "").strip():
        scene_summary = _key_action_scene_summary_for_overview(experiment_id, key_action_counts)
    return {
        "schema_version": "analysis_overview.v1",
        "experiment": {
            "experiment_id": experiment_id,
            "experiment_name": exp.get("title") or experiment_id,
            "description": exp.get("description") or "",
        },
        "run": {
            "run_id": run_id,
            "result_version": result_version,
            "status": status,
            "stage": effective_task_state.get("current_stage") or effective_task_state.get("stage") or exp.get("processing_stage") or status,
            "progress": float(effective_task_state.get("progress") or (1.0 if status in {"completed", "partial_failed"} else 0.0)),
            "message": effective_task_state.get("message") or "",
            "updated_at": effective_task_state.get("updated_at") or effective_task_state.get("completed_at") or exp.get("completed_at") or exp.get("analyzed_at") or exp.get("created_at"),
            "trace_id": f"{experiment_id}:{run_id}:{result_version}",
        },
        "readiness": readiness,
        "summary": {
            "frame_count": len(frames) if frames else len(material_stream) or int(key_action_counts.get("yolo_frame_row_count") or key_action_counts.get("video_event_count") or 0),
            "detection_count": total_detections or sum(len(i.get("detected_objects", [])) for i in material_stream if isinstance(i, dict)),
            "alert_count": len(alerts),
            "official_step_count": official_count,
            "candidate_step_count": candidate_count,
            "confirmed_step_count": official_count,
            "inferred_step_count": inferred_count,
            "avg_confidence": (sum(avg_values) / len(avg_values)) if avg_values else timeline.get("avg_confidence"),
            "model_name": (_experiment_model_status().get("yolo_model_name") or "unknown"),
        },
        "steps": groups,
        "scene_summary": scene_summary,
        "alerts": alerts,
        "markers": {
            "steps": [{"id": s.get("step_id"), "label": s.get("step_name"), "timestamp_sec": s.get("start_time_sec"), "kind": "step"} for s in display_steps],
            "alerts": [{"id": a.get("alert_id"), "label": a.get("rule_name"), "timestamp_sec": a.get("timestamp_sec"), "kind": "alert", "severity": a.get("severity")} for a in alerts],
            "evidence": [{"id": ref.get("evidence_id"), "timestamp_sec": ref.get("timestamp_sec"), "kind": "evidence"} for s in steps for ref in (s.get("evidence_refs") or [])],
        },
        "artifacts": {
            "source_video": _artifact_metadata("source_video", artifacts.get("source_video"), f"/api/v1/experiments/{experiment_id}/video" if artifacts.get("source_video") else None),
            "annotated_video": _artifact_metadata("annotated_video", artifacts.get("annotated_video"), f"/api/v1/experiments/{experiment_id}/artifacts/annotated_video"),
            "analysis_json": _artifact_metadata("analysis_json", artifacts.get("analysis_json"), f"/api/v1/experiments/{experiment_id}/artifacts/analysis_json"),
            "material_stream": _artifact_metadata("material_stream", artifacts.get("material_stream_json"), f"/api/v1/experiments/{experiment_id}/artifacts/material_stream_json"),
            "material_stream_v2": _artifact_metadata("material_stream_v2", artifacts.get("material_stream_v2_jsonl"), f"/api/v1/experiments/{experiment_id}/artifacts/material_stream_v2_jsonl"),
            "experiment_run_manifest": _artifact_metadata("experiment_run_manifest", artifacts.get("experiment_run_manifest_json"), f"/api/v1/experiments/{experiment_id}/artifacts/experiment_run_manifest_json"),
            "stream_manifest": _artifact_metadata("stream_manifest", artifacts.get("stream_manifest_json"), f"/api/v1/experiments/{experiment_id}/artifacts/stream_manifest_json"),
            "timeline_alignment": _artifact_metadata("timeline_alignment", artifacts.get("timeline_alignment_json"), f"/api/v1/experiments/{experiment_id}/artifacts/timeline_alignment_json"),
            "semantic_sync_anchors": _artifact_metadata("semantic_sync_anchors", artifacts.get("semantic_sync_anchors_json"), f"/api/v1/experiments/{experiment_id}/artifacts/semantic_sync_anchors_json"),
            "transcript_segments": _artifact_metadata("transcript_segments", artifacts.get("transcript_segments_jsonl"), f"/api/v1/experiments/{experiment_id}/artifacts/transcript_segments_jsonl"),
            "material_index": _artifact_metadata("material_index", artifacts.get("material_index"), None),
            "official_steps": _artifact_metadata("official_steps", artifacts.get("official_steps_json"), f"/api/v1/experiments/{experiment_id}/artifacts/official_steps_json"),
            "step_review_log": _artifact_metadata("step_review_log", artifacts.get("step_review_log_json"), f"/api/v1/experiments/{experiment_id}/artifacts/step_review_log_json"),
            "professional_report_pdf": _artifact_metadata(
                "professional_report_pdf",
                artifacts.get("professional_report_pdf"),
                f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_pdf",
            ),
            "professional_report_html": _artifact_metadata(
                "professional_report_html",
                artifacts.get("professional_report_html"),
                f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_html",
            ),
            "professional_report_json": _artifact_metadata(
                "professional_report_json",
                artifacts.get("professional_report_json"),
                f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_json",
            ),
            "professional_report_manifest": _artifact_metadata(
                "professional_report_manifest",
                artifacts.get("professional_report_manifest_json"),
                f"/api/v1/experiments/{experiment_id}/artifacts/professional_report_manifest_json",
            ),
            **key_action_media_artifacts,
        },
        "debug": {
            "local_output_dir": str(_experiment_output_dir(experiment_id)),
            "raw_paths": {k: str(v) if v else None for k, v in artifacts.items()},
            "task_state": effective_task_state,
            "legacy_task_state": task_state,
            "key_action_state": key_action_state,
            "key_action_counts": key_action_counts,
            "legacy_status": exp.get("status"),
            "legacy_processing_stage": exp.get("processing_stage"),
        },
    }


@app.get("/api/v1/experiments/{experiment_id}/physical-events", tags=["experiments"])
async def get_experiment_physical_events(
    experiment_id: str,
    event_type: Optional[str] = None,
    actor_name: Optional[str] = None,
    start_time_sec: Optional[float] = None,
    end_time_sec: Optional[float] = None,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    payload = _load_json_if_exists(_experiment_output_dir(safe_experiment_id) / "physical_events.json") or {}
    events = payload.get("events") if isinstance(payload, dict) else payload
    events = events or []
    filtered = []
    for event in events:
        if event_type and event.get("event_type") != event_type:
            continue
        if actor_name and event.get("actor_name") != actor_name:
            continue
        if start_time_sec is not None and float(event.get("end_time_sec") or 0.0) < float(start_time_sec):
            continue
        if end_time_sec is not None and float(event.get("start_time_sec") or 0.0) > float(end_time_sec):
            continue
        filtered.append(event)
    return {
        "experiment_id": safe_experiment_id,
        "schema_version": payload.get("schema_version", "physical_events.legacy") if isinstance(payload, dict) else "physical_events.legacy",
        "total": len(filtered),
        "events": filtered,
    }


@app.get("/api/v1/experiments/{experiment_id}/materials/events/{event_id}", tags=["experiments"])
async def get_experiment_event_material(
    experiment_id: str,
    event_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    event_json = _experiment_output_dir(safe_experiment_id) / "materials" / "events" / event_id / "event.json"
    if not event_json.exists():
        raise HTTPException(status_code=404, detail="Event material not found")
    return _load_json_if_exists(event_json) or {}


@app.get("/api/v1/experiments/{experiment_id}/analysis-overview", tags=["experiments"])
@app.get("/api/experiments/{experiment_id}/analysis-overview", tags=["experiments"])
async def get_experiment_analysis_overview(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    return _build_analysis_overview(exp)


@app.get("/api/v1/experiments/{experiment_id}/key-actions/status", tags=["experiments"])
async def get_key_action_status(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    return _key_action_status_payload(safe_experiment_id)


@app.get("/api/v1/experiments/{experiment_id}/key-actions/results", tags=["experiments"])
async def get_key_action_results(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    return _key_action_results_payload(safe_experiment_id)


@app.get("/api/v1/experiments/{experiment_id}/key-actions/quality", tags=["experiments"])
async def get_key_action_quality(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    return _key_action_quality_payload(safe_experiment_id)


@app.get("/api/v1/experiments/{experiment_id}/key-actions/review-queue", tags=["experiments"])
async def get_key_action_review_queue(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    return _key_action_review_queue_payload(safe_experiment_id)


@app.post("/api/v1/experiments/{experiment_id}/key-actions/review/items/{item_id}/decision", tags=["experiments"])
async def decide_key_action_review_item(
    experiment_id: str,
    item_id: str,
    request: KeyActionReviewDecisionRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        from key_action_indexer.review_queue import apply_review_decision  # type: ignore

        decision = apply_review_decision(
            _key_action_output_dir(safe_experiment_id),
            item_id=item_id,
            decision=request.decision,
            reviewer=request.reviewer or "frontend_reviewer",
            note=request.note or "",
            boundary_start_sec=request.boundary_start_sec,
            boundary_end_sec=request.boundary_end_sec,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Key-action review decision failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "experiment_id": safe_experiment_id,
        "item_id": item_id,
        "decision": decision,
        "queue": _key_action_review_queue_payload(safe_experiment_id),
    }


@app.post("/api/v1/experiments/{experiment_id}/key-actions/review/bulk", tags=["experiments"])
async def bulk_decide_key_action_review_items(
    experiment_id: str,
    request: KeyActionReviewBulkRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    queue = _key_action_review_queue_payload(safe_experiment_id)
    item_ids = request.item_ids or [
        str(item.get("item_id"))
        for item in queue.get("items", [])
        if isinstance(item, dict) and str(item.get("review_status") or "pending") == "pending"
    ]
    if not item_ids:
        raise HTTPException(status_code=400, detail="No review items selected")
    try:
        from key_action_indexer.review_queue import apply_review_decision  # type: ignore

        decisions = [
            apply_review_decision(
                _key_action_output_dir(safe_experiment_id),
                item_id=item_id,
                decision=request.decision,
                reviewer=request.reviewer or "frontend_reviewer",
                note=request.note or "",
            )
            for item_id in item_ids
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Key-action bulk review decision failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "experiment_id": safe_experiment_id,
        "decision_count": len(decisions),
        "decisions": decisions,
        "queue": _key_action_review_queue_payload(safe_experiment_id),
    }


@app.get("/api/v1/experiments/{experiment_id}/key-actions/review/export", tags=["experiments"])
async def export_key_action_review_queue(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        from key_action_indexer.review_queue import export_review_queue  # type: ignore

        queue = _key_action_review_queue_payload(safe_experiment_id)
        return export_review_queue(_key_action_output_dir(safe_experiment_id), queue=queue)
    except Exception as exc:
        logger.exception("Key-action review export failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/experiments/{experiment_id}/key-actions/review/freeze", tags=["experiments"])
async def freeze_key_action_reviewed_dataset(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        from key_action_indexer.reviewed_dataset import freeze_reviewed_dataset, load_reviewed_export  # type: ignore

        manifest = freeze_reviewed_dataset(_key_action_output_dir(safe_experiment_id))
        return {
            "experiment_id": safe_experiment_id,
            "manifest": manifest,
            "release": manifest.get("release") if isinstance(manifest, dict) else None,
            "reviewed_export": load_reviewed_export(_key_action_output_dir(safe_experiment_id)),
            "queue": _key_action_review_queue_payload(safe_experiment_id),
        }
    except Exception as exc:
        logger.exception("Key-action reviewed dataset freeze failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/experiments/{experiment_id}/key-actions/review/rollback", tags=["experiments"])
async def rollback_key_action_reviewed_release(
    experiment_id: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    payload = await request.json() if request.headers.get("content-length") not in {None, "0"} else {}
    version = str(payload.get("version") or "").strip() or None
    try:
        from key_action_indexer.reviewed_dataset import load_reviewed_export, rollback_reviewed_release  # type: ignore

        rollback = rollback_reviewed_release(_key_action_output_dir(safe_experiment_id), version=version)
        return {
            "experiment_id": safe_experiment_id,
            "rollback": rollback,
            "reviewed_export": load_reviewed_export(_key_action_output_dir(safe_experiment_id)),
            "queue": _key_action_review_queue_payload(safe_experiment_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Key-action reviewed release rollback failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/experiments/{experiment_id}/key-actions/review/promote", tags=["experiments"])
async def promote_key_action_reviewed_release(
    experiment_id: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    payload = await request.json() if request.headers.get("content-length") not in {None, "0"} else {}
    version = str(payload.get("version") or "").strip() or None
    reviewer = str(payload.get("reviewer") or auth_ctx.get("operator") or "").strip()
    if not reviewer or reviewer in {"anonymous", "frontend_reviewer"}:
        raise HTTPException(status_code=400, detail="Promotion requires an explicit reviewer identity")
    note = str(payload.get("note") or "")
    try:
        query_count = max(1, min(50, int(payload.get("query_count") or 50)))
    except (TypeError, ValueError):
        query_count = 50
    try:
        from key_action_indexer.reviewed_dataset import load_reviewed_export, promote_reviewed_release  # type: ignore

        promotion = promote_reviewed_release(
            _key_action_output_dir(safe_experiment_id),
            version=version,
            reviewer=reviewer,
            note=note,
            query_count=query_count,
        )
        return {
            "experiment_id": safe_experiment_id,
            "promotion": promotion,
            "reviewed_export": load_reviewed_export(_key_action_output_dir(safe_experiment_id)),
            "queue": _key_action_review_queue_payload(safe_experiment_id),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Key-action reviewed release promotion failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/experiments/{experiment_id}/key-actions/evidence/adapters", tags=["experiments"])
async def get_key_action_evidence_adapters(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    return _key_action_evidence_adapter_payload(safe_experiment_id)


@app.post("/api/v1/experiments/{experiment_id}/key-actions/retrieval/evaluate", tags=["experiments"])
async def evaluate_key_action_retrieval(
    experiment_id: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    payload = await request.json() if request.headers.get("content-length") not in {None, "0"} else {}
    try:
        query_count = max(20, min(50, int(payload.get("query_count") or 50)))
    except (TypeError, ValueError):
        query_count = 50
    try:
        from key_action_indexer.retrieval_eval import run_default_chinese_query_eval  # type: ignore

        return {
            "experiment_id": safe_experiment_id,
            "evaluation": run_default_chinese_query_eval(_key_action_output_dir(safe_experiment_id), query_count=query_count),
        }
    except Exception as exc:
        logger.exception("Key-action retrieval evaluation failed for %s", safe_experiment_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/experiments/{experiment_id}/key-actions/query", tags=["experiments"])
async def query_key_action_index(
    experiment_id: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    payload = await request.json()
    query = str(payload.get("query") or "").strip()
    try:
        top_k = max(1, int(payload.get("top_k") or 5))
    except (TypeError, ValueError):
        top_k = 5
    filters: Dict[str, Any] = {}
    index_level = str(payload.get("index_level") or "").strip()
    if index_level and index_level != "all":
        filters["index_level"] = index_level
    primary_object = payload.get("primary_object") or payload.get("object")
    if primary_object:
        filters["object"] = primary_object
    interaction_type = payload.get("interaction_type") or payload.get("action_type") or payload.get("action")
    if interaction_type:
        filters["action"] = interaction_type
    if payload.get("asset_type"):
        filters["asset_type"] = payload.get("asset_type")
    if payload.get("start_time") is not None:
        filters["start_time"] = payload.get("start_time")
    if payload.get("end_time") is not None:
        filters["end_time"] = payload.get("end_time")
    if query:
        try:
            from key_action_indexer.query_validation import query_session_index  # type: ignore

            query_payload = query_session_index(
                _key_action_output_dir(safe_experiment_id),
                [query],
                top_k=top_k,
                filters=filters,
            )
            query_rows = query_payload.get("queries") if isinstance(query_payload, dict) else []
            first_query = query_rows[0] if query_rows else {}
            results = first_query.get("results") if isinstance(first_query, dict) else []
            return {
                "experiment_id": safe_experiment_id,
                "query": query,
                "results": results or [],
                "validation_summary": {
                    "source": "vector_index",
                    "result_count": len(results or []),
                    "filters": filters,
                    "index_dir": query_payload.get("index_dir") if isinstance(query_payload, dict) else None,
                },
            }
        except Exception as exc:
            logger.warning("Key-action vector query fallback for %s: %s", safe_experiment_id, exc)
    normalized_query = query.lower()
    results = []
    for segment in _key_action_results_payload(safe_experiment_id).get("segments", []):
        text = json.dumps(segment, ensure_ascii=False).lower()
        if not normalized_query or normalized_query in text:
            results.append(segment)
    return {
        "experiment_id": safe_experiment_id,
        "query": query,
        "results": results[:top_k],
        "validation_summary": {
            "source": "segment_substring_fallback",
            "result_count": min(len(results), top_k),
            "filters": filters,
        },
    }


@app.get("/api/v1/experiments/{experiment_id}/analysis", tags=["experiments"])
async def get_experiment_analysis(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    artifacts = _experiment_output_artifact_paths(safe_experiment_id, exp)
    analysis_path = artifacts["analysis_json"]
    analysis_payload = _load_json_if_exists(analysis_path) if analysis_path is not None else None
    frames = analysis_payload or []
    total_detections = sum(len(frame.get("detections", [])) for frame in frames if isinstance(frame, dict))
    total_alerts = sum(len(frame.get("alerts", [])) for frame in frames if isinstance(frame, dict))
    frames_with_detections = sum(1 for frame in frames if isinstance(frame, dict) and frame.get("detections"))
    frames_with_alerts = sum(1 for frame in frames if isinstance(frame, dict) and frame.get("alerts"))
    return {
        "experiment_id": safe_experiment_id,
        "analysis_ready": bool(analysis_payload),
        "model_status": _experiment_model_status(),
        "summary": {
            "frame_count": len(frames),
            "total_detections": total_detections,
            "total_alerts": total_alerts,
            "frames_with_detections": frames_with_detections,
            "frames_with_alerts": frames_with_alerts,
        },
        "artifacts": {
            "source_video_url": f"/api/v1/experiments/{safe_experiment_id}/video",
            "annotated_video_url": f"/api/v1/experiments/{safe_experiment_id}/artifacts/annotated_video",
            "analysis_json_url": f"/api/v1/experiments/{safe_experiment_id}/artifacts/analysis_json",
        },
        "analysis": analysis_payload,
    }


@app.post("/api/v1/experiments/{experiment_id}/upload/context", tags=["experiments"])
async def upload_context(
    experiment_id: str,
    context_text: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    exp.setdefault("context_inputs", [])
    exp["context_inputs"].append({"text": context_text, "uploaded_at": datetime.now().isoformat()})
    _save_experiment(exp)
    return {"experiment_id": safe_experiment_id, "context_count": len(exp["context_inputs"])}


def _asr_job_store(experiment_id: str):
    from labsopguard.asr_jobs import ASRJobStore

    return ASRJobStore(_experiment_output_dir(experiment_id) / "asr_jobs")


def _append_asr_context_records(
    exp: Dict[str, Any],
    *,
    audio_path: Path,
    transcript: Any,
) -> List[Dict[str, Any]]:
    exp.setdefault("context_inputs", [])
    context_records = []
    for index, segment in enumerate(transcript.segments):
        record = segment.to_context_input()
        record["source_file"] = str(audio_path)
        record["segment_index"] = index
        record["asr_provider"] = transcript.provider
        record["asr_model"] = transcript.model
        record["uploaded_at"] = datetime.now().isoformat()
        context_records.append(record)
    exp["context_inputs"].extend(context_records)
    return context_records


def _run_asr_background_job(
    experiment_id: str,
    job_id: str,
    audio_path: str,
    language: Optional[str],
    prompt: Optional[str],
    chunk_duration_sec: float,
    force_chunk: bool,
) -> None:
    from labsopguard.asr_jobs import transcribe_audio_in_chunks

    store = _asr_job_store(experiment_id)
    job = store.load(job_id)
    try:
        job.status = "running"
        store.save(job)
        transcript, chunk_count = transcribe_audio_in_chunks(
            audio_path,
            _experiment_output_dir(experiment_id) / "asr_jobs" / job_id,
            language=language,
            prompt=prompt,
            chunk_duration_sec=chunk_duration_sec,
            force_chunk=force_chunk,
        )
        exp = _normalize_experiment_dict(_load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or {})
        records = _append_asr_context_records(exp, audio_path=Path(audio_path), transcript=transcript)
        _save_experiment(exp)
        job.status = "completed"
        job.provider = transcript.provider
        job.model = transcript.model
        job.text = transcript.text
        job.segment_count = len(records)
        job.context_count = len(exp.get("context_inputs", []))
        job.chunk_count = chunk_count
        job.completed_at = datetime.now().isoformat()
        store.save(job)
    except Exception as exc:
        job.retry_count += 1
        job.status = "failed"
        job.error = str(exc)
        store.save(job)


@app.post("/api/v1/experiments/{experiment_id}/upload/asr", tags=["experiments"])
async def upload_asr_transcript(
    experiment_id: str,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Transcribe uploaded audio and append transcript segments to context_inputs."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    upload_dir = PROJECT_ROOT / "uploads" / "experiments" / safe_experiment_id / "audio"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_upload_filename(audio.filename, default_name="audio_input.wav")
    audio_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    await _save_upload_file(audio, audio_path, max_bytes=MAX_AUDIO_UPLOAD_BYTES)

    from labsopguard.asr import ASRUnavailableError, transcribe_audio_file

    try:
        transcript = transcribe_audio_file(audio_path, language=language, prompt=prompt)
    except ASRUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ASR transcription failed: {exc}") from exc

    context_records = _append_asr_context_records(exp, audio_path=audio_path, transcript=transcript)
    _save_experiment(exp)
    return {
        "experiment_id": safe_experiment_id,
        "provider": transcript.provider,
        "model": transcript.model,
        "language": transcript.language,
        "text": transcript.text,
        "segment_count": len(context_records),
        "context_count": len(exp["context_inputs"]),
        "segments": context_records,
    }


@app.post("/api/v1/experiments/{experiment_id}/upload/asr/jobs", tags=["experiments"])
async def submit_asr_transcription_job(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    chunk_duration_sec: float = Form(60.0),
    force_chunk: bool = Form(False),
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Queue long ASR transcription and append transcript segments when finished."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    upload_dir = PROJECT_ROOT / "uploads" / "experiments" / safe_experiment_id / "audio"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_upload_filename(audio.filename, default_name="audio_input.wav")
    audio_path = upload_dir / f"{uuid.uuid4().hex[:8]}_{safe_name}"
    await _save_upload_file(audio, audio_path, max_bytes=MAX_AUDIO_UPLOAD_BYTES)

    store = _asr_job_store(safe_experiment_id)
    job = store.create(safe_experiment_id, str(audio_path), language=language, prompt=prompt)
    background_tasks.add_task(
        _run_asr_background_job,
        safe_experiment_id,
        job.job_id,
        str(audio_path),
        language,
        prompt,
        float(chunk_duration_sec),
        bool(force_chunk),
    )
    return {"experiment_id": safe_experiment_id, "job": job.to_dict()}


@app.get("/api/v1/experiments/{experiment_id}/upload/asr/jobs/{job_id}", tags=["experiments"])
async def get_asr_transcription_job(
    experiment_id: str,
    job_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    store = _asr_job_store(safe_experiment_id)
    try:
        job = store.load(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="ASR job not found") from exc
    return {"experiment_id": safe_experiment_id, "job": job.to_dict()}


@app.post("/api/v1/experiments/{experiment_id}/upload/protocol", tags=["experiments"])
async def upload_protocol(
    experiment_id: str,
    protocol_text: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """ protocol"""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    exp["protocol_text"] = protocol_text
    _save_experiment(exp)
    return {"experiment_id": safe_experiment_id, "protocol_length": len(protocol_text)}


def _run_experiment_pipeline(
    *,
    experiment_id: str,
    task_id: str,
    video_file: Path,
    sample_interval: float,
    max_frames: int,
    qwen_writeback_config: Any = None,
) -> None:
    if ExperimentService is None:
        raise RuntimeError("Experiment service not available")
    exp = _normalize_experiment_dict(_load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or {})
    if not exp:
        raise RuntimeError("Experiment not found")

    analysis_dir = _experiment_output_dir(experiment_id) / "analysis"
    output_paths: Dict[str, str] = {}

    try:
        started_at = _now_iso()
        _persist_experiment_task_state(
            experiment_id,
            _build_experiment_task_payload(
                task_id,
                experiment_id,
                status="running",
                current_stage="video_analysis",
                progress=0.1,
                video_path=str(video_file),
                output_paths=output_paths,
            ),
        )
        exp["analysis_job_id"] = task_id
        exp["status"] = "running"
        exp["processing_stage"] = "video_analysis"
        exp["processing_error"] = None
        exp["started_at"] = started_at
        _save_experiment(exp)

        video_analysis_artifacts = _run_video_analysis_pipeline(
            task_id=task_id,
            experiment_id=experiment_id,
            video_path=str(video_file),
            output_dir=analysis_dir,
            sample_interval=sample_interval,
            max_frames=max_frames,
        )
        output_paths.update({
            "analysis_json": video_analysis_artifacts["analysis_json_path"],
            "annotated_video": video_analysis_artifacts["annotated_video_path"],
        })
        _persist_experiment_task_state(
            experiment_id,
            {
                "status": "running",
                "current_stage": ProcessStage.VIDEO_UNDERSTANDING.value,
                "progress": 0.45,
                "message": "VLM 场景理解中 Running VLM scene understanding...",
                "video_path": str(video_file),
                "output_paths": output_paths,
                "started_at": started_at,
            },
        )
        exp["processing_stage"] = ProcessStage.VIDEO_UNDERSTANDING.value
        _save_experiment(exp)

        result = _invoke_experiment_service(
            experiment_id=experiment_id,
            experiment_record=exp,
            source_ref=str(video_file),
            sample_interval=sample_interval,
            max_frames=max_frames,
        )

        _persist_experiment_task_state(experiment_id, {
            "status": "running", "current_stage": "event_detection",
            "progress": 0.65, "message": "事件检测与步骤匹配中 Event detection & step matching...",
        })

        output_paths = _write_experiment_processing_outputs(
            experiment_id=experiment_id,
            experiment_record=exp,
            experiment_result=result,
            task_state={
                "task_id": task_id,
                "status": "completed",
                "started_at": started_at,
                "completed_at": _now_iso(),
            },
            video_analysis_artifacts=video_analysis_artifacts,
        )

        _persist_experiment_task_state(experiment_id, {
            "status": "running", "current_stage": "writeback",
            "progress": 0.85, "message": "输出写回与索引构建 Writing results & building index...",
        })

        qwen_writeback_report = None
        try:
            qwen_writeback_report = _run_qwen_frame_writeback_if_enabled(
                experiment_id=experiment_id,
                output_paths=output_paths,
                config=qwen_writeback_config or _qwen_writeback_config_from_request(None),
            )
        except Exception as exc:
            logger.exception("Qwen frame writeback failed for %s", experiment_id)
            if getattr(qwen_writeback_config, "fail_pipeline", False):
                raise
            output_paths["qwen_frame_writeback_error"] = str(exc)
        _persist_experiment_task_state(experiment_id, {
            "status": "running",
            "current_stage": "professional_report",
            "progress": 0.95,
            "message": "Generating professional PDF report",
            "output_paths": output_paths,
        })
        professional_report = _generate_professional_report_for_experiment(
            experiment_id,
            output_paths=output_paths,
        )
        output_paths = _attach_professional_report_output_paths(
            experiment_id,
            output_paths,
            professional_report,
        )
        final_state = _persist_experiment_task_state(
            experiment_id,
            {
                "task_id": task_id,
                "experiment_id": experiment_id,
                "status": "completed",
                "current_stage": ProcessStage.OUTPUT_GENERATION.value,
                "progress": 1.0,
                "message": "处理完成 Completed",
                "video_path": str(video_file),
                "started_at": started_at,
                "completed_at": _now_iso(),
                "output_paths": output_paths,
                "qwen_frame_writeback": qwen_writeback_report,
                "professional_report": professional_report,
                "error_type": None,
                "error_message": None,
            },
        )
        exp = _normalize_experiment_dict(_load_json_if_exists(_experiment_output_dir(experiment_id) / "experiment.json") or exp)
        exp["status"] = ExperimentStatus.ANALYZED.value
        exp["processing_stage"] = ProcessStage.OUTPUT_GENERATION.value
        exp["processing_error"] = None
        exp["completed_at"] = final_state.get("completed_at")
        exp["analysis_job_id"] = task_id
        exp["output_paths"] = output_paths
        if qwen_writeback_report is not None:
            exp.setdefault("metadata", {})["qwen_frame_writeback"] = qwen_writeback_report
        exp.setdefault("metadata", {})["professional_report"] = professional_report
        _save_experiment(exp)
    except Exception as exc:
        error_payload = _classify_processing_error(exc)
        failed_at = _now_iso()
        _persist_experiment_task_state(
            experiment_id,
            {
                "task_id": task_id,
                "experiment_id": experiment_id,
                "status": "failed",
                "current_stage": exp.get("processing_stage", ProcessStage.INGESTION.value),
                "progress": 1.0,
                "video_path": str(video_file),
                "started_at": exp.get("started_at"),
                "completed_at": failed_at,
                "output_paths": output_paths,
                **error_payload,
            },
        )
        exp["status"] = "failed"
        exp["processing_error"] = error_payload["error_message"]
        exp["processing_stage"] = exp.get("processing_stage", ProcessStage.INGESTION.value)
        exp["completed_at"] = failed_at
        exp["analysis_job_id"] = task_id
        exp["output_paths"] = output_paths
        _save_experiment(exp)
        logger.exception("Experiment processing failed for %s", experiment_id)


@app.post("/api/v1/experiments/{experiment_id}/process", tags=["experiments"])
async def process_experiment(
    experiment_id: str,
    background_tasks: BackgroundTasks,
    req: Optional[ProcessExperimentRequest] = None,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """Queue experiment processing and return a formal task record."""
    if ExperimentService is None or EXPERIMENT_TASK_STORE is None:
        raise HTTPException(status_code=503, detail="Experiment service not available")
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)

    try:
        exp = await get_experiment_dict(safe_experiment_id)
    except HTTPException:
        raise HTTPException(status_code=404, detail="Experiment not found")

    existing_task = _experiment_task_state(safe_experiment_id)
    if existing_task and existing_task.get("status") in {"queued", "running"}:
        return existing_task

    def _persist_preflight_failure(error_type: str, message: str, status_code: int) -> None:
        primary_source = (req.video_path if req else None) or (exp.get("video_paths", [None])[0] if exp.get("video_paths") else None)
        if not primary_source and exp.get("video_inputs"):
            primary_source = exp["video_inputs"][0].get("video_path")
        task_id = str(uuid.uuid4())
        failed_state = _persist_experiment_task_state(
            safe_experiment_id,
            {
                "task_id": task_id,
                "experiment_id": safe_experiment_id,
                "status": "failed",
                "current_stage": ProcessStage.INGESTION.value,
                "progress": 1.0,
                "video_path": primary_source,
                "started_at": None,
                "completed_at": _now_iso(),
                "output_paths": {},
                "error_type": error_type,
                "error_message": message,
            },
        )
        exp["analysis_job_id"] = failed_state.get("task_id")
        exp["status"] = "failed"
        exp["processing_stage"] = ProcessStage.INGESTION.value
        exp["processing_error"] = message
        exp["completed_at"] = failed_state.get("completed_at")
        exp["output_paths"] = {}
        _save_experiment(exp)
        raise HTTPException(status_code=status_code, detail=message)

    runtime_video_inputs = list(exp.get("video_inputs") or [])
    requested_source = req.video_path if req else None
    requested_source_type_hint = req.source_type if req else None
    if requested_source and (requested_source_type_hint or not _is_local_file_source(requested_source)):
        requested_source_type = (req.source_type if req and req.source_type else None) or ("rtsp" if _looks_like_live_source(requested_source) else "file")
        runtime_video_inputs = [
            {
                "video_index": 0,
                "video_path": requested_source,
                "source": requested_source,
                "source_type": requested_source_type,
                "ingest_mode": requested_source_type,
                "capture_duration_sec": max(req.sample_interval * max(req.max_frames - 1, 1), req.sample_interval),
            }
        ] + runtime_video_inputs

    primary_input = runtime_video_inputs[0] if runtime_video_inputs else None
    source_ref = requested_source or (exp.get("video_paths", [None])[0] if exp.get("video_paths") else None)
    if not source_ref and primary_input:
        source_ref = primary_input.get("video_path")
    if not source_ref:
        _persist_preflight_failure(
            "upload_missing",
            "upload missing: no video or stream source provided. Upload video first, register a stream, or pass video_path.",
            400,
        )

    source_type = (req.source_type if req and req.source_type else None) or (primary_input.get("source_type") if primary_input else None)
    source_type = (source_type or ("file" if _is_local_file_source(source_ref) else "rtsp")).lower()
    is_local_file = _is_local_file_source(source_ref)
    is_live_source = (not is_local_file) and (bool(primary_input) or _looks_like_live_source(source_ref) or source_type in {"rtsp", "usb", "udp", "rtmp"})
    if not is_local_file and not is_live_source:
        _persist_preflight_failure("video_not_found", f"video or source not found: {source_ref}", 404)

    if runtime_video_inputs and runtime_video_inputs != exp.get("video_inputs", []):
        exp["video_inputs"] = runtime_video_inputs
        exp["video_metadata"] = exp.get("video_metadata") or list(runtime_video_inputs)

    sample_interval = req.sample_interval if req else 3.0
    max_frames = req.max_frames if req else 10
    qwen_writeback_config = _qwen_writeback_config_from_request(req)
    video_file = Path(source_ref) if is_local_file else None

    if FORMAL_WORKFLOW is None or not is_local_file or source_type != "file":
        inline_state = _run_experiment_service_only(
            experiment_id=safe_experiment_id,
            source_ref=str(source_ref),
            sample_interval=sample_interval,
            max_frames=max_frames,
            qwen_writeback_config=qwen_writeback_config,
        )
        if inline_state.get("status") == "failed":
            raise HTTPException(
                status_code=500,
                detail=inline_state.get("error_message") or "Experiment processing failed",
            )
        return inline_state

    task_id = str(uuid.uuid4())
    created_at = _now_iso()
    task_state = _persist_experiment_task_state(
        safe_experiment_id,
        {
            "task_id": task_id,
            "experiment_id": safe_experiment_id,
            "status": "queued",
            "current_stage": ProcessStage.INGESTION.value,
            "progress": 0.0,
            "video_path": str(source_ref),
            "started_at": None,
            "completed_at": None,
            "output_paths": {},
            "error_type": None,
            "error_message": None,
            "created_at": created_at,
        },
    )
    exp["analysis_job_id"] = task_id
    exp["status"] = "queued"
    exp["processing_stage"] = ProcessStage.INGESTION.value
    exp["processing_error"] = None
    exp["started_at"] = None
    exp["completed_at"] = None
    exp["output_paths"] = {}
    _save_experiment(exp)

    background_tasks.add_task(
        _run_experiment_pipeline,
        experiment_id=safe_experiment_id,
        task_id=task_id,
        video_file=video_file,
        sample_interval=sample_interval,
        max_frames=max_frames,
        qwen_writeback_config=qwen_writeback_config,
    )
    return task_state


@app.get("/api/v1/experiments/{experiment_id}/timeline", tags=["experiments"])
async def get_experiment_timeline(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    timeline = _load_json_if_exists(_experiment_output_dir(safe_experiment_id) / "timeline.json")
    return timeline if timeline is not None else _empty_timeline_payload(exp)


@app.get("/api/v1/experiments/{experiment_id}/steps", tags=["experiments"])
async def get_experiment_steps(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    official_records = _official_step_records_for_overview(safe_experiment_id)
    if official_records:
        candidate_records = _candidate_step_records_for_overview(safe_experiment_id, official_records)
        return {
            "schema_version": "official_first_steps.v1",
            "experiment_id": safe_experiment_id,
            "official_steps": official_records,
            "candidate_steps": candidate_records["candidate"],
            "inferred_steps": candidate_records["inferred"],
        }
    return _load_json_if_exists(_experiment_output_dir(safe_experiment_id) / "steps.json") or []


@app.get("/api/v1/experiments/{experiment_id}/steps/{step_id}", tags=["experiments"])
async def get_step_detail(
    experiment_id: str,
    step_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    steps_file = _experiment_output_dir(safe_experiment_id) / "steps.json"
    if not steps_file.exists():
        raise HTTPException(status_code=404, detail="Step not found")
    steps = json.loads(steps_file.read_text(encoding="utf-8"))
    for step in steps:
        if step.get("step_id") == step_id:
            return step
    raise HTTPException(status_code=404, detail=f"Step {step_id} not found")


def _step_review_store(experiment_id: str):
    from labsopguard.step_review import StepReviewStore

    store = StepReviewStore(experiment_id, _experiment_output_dir(experiment_id))
    store.ensure_outputs()
    return store


@app.post("/api/v1/experiments/{experiment_id}/steps/review/approve", tags=["experiments"])
async def approve_step_candidate(
    experiment_id: str,
    request: StepReviewRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).approve(
            step_candidate_id=request.step_candidate_id,
            decision="approve",
            rationale=request.rationale,
            operator=request.operator,
            operator_role=request.operator_role,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409 if "locked" in str(exc) else 404, detail=str(exc))


@app.post("/api/v1/experiments/{experiment_id}/steps/review/reject", tags=["experiments"])
async def reject_step_candidate(
    experiment_id: str,
    request: StepReviewRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).reject(
            step_candidate_id=request.step_candidate_id,
            rationale=request.rationale,
            operator=request.operator,
            operator_role=request.operator_role,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/v1/experiments/{experiment_id}/steps/review/edit-and-approve", tags=["experiments"])
async def edit_and_approve_step_candidate(
    experiment_id: str,
    request: StepEditAndApproveRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).approve(
            step_candidate_id=request.step_candidate_id,
            decision="edit_and_approve",
            rationale=request.rationale,
            operator=request.operator,
            operator_role=request.operator_role,
            edits=request.edits or {},
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409 if "locked" in str(exc) else 404, detail=str(exc))


@app.post("/api/v1/experiments/{experiment_id}/steps/review/defer", tags=["experiments"])
async def defer_step_candidate(
    experiment_id: str,
    request: StepReviewRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).defer(
            step_candidate_id=request.step_candidate_id,
            rationale=request.rationale,
            operator=request.operator,
            operator_role=request.operator_role,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/v1/experiments/{experiment_id}/steps/{official_step_id}/lock", tags=["experiments"])
async def lock_official_step(
    experiment_id: str,
    official_step_id: str,
    request: StepLockRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).lock(
            official_step_id=official_step_id,
            operator=request.operator,
            operator_role=request.operator_role,
            rationale=request.rationale,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/v1/experiments/{experiment_id}/steps/{official_step_id}/reopen", tags=["experiments"])
async def reopen_official_step(
    experiment_id: str,
    official_step_id: str,
    request: StepLockRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).reopen(
            official_step_id=official_step_id,
            operator=request.operator,
            operator_role=request.operator_role,
            rationale=request.rationale,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/v1/experiments/{experiment_id}/steps/{official_step_id}/supersede", tags=["experiments"])
async def supersede_official_step(
    experiment_id: str,
    official_step_id: str,
    request: StepSupersedeRequest,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    try:
        return _step_review_store(safe_experiment_id).supersede(
            official_step_id=official_step_id,
            operator=request.operator,
            operator_role=request.operator_role,
            rationale=request.rationale,
            replacement_payload=request.replacement_payload or {},
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/v1/experiments/{experiment_id}/evidence", tags=["experiments"])
async def get_experiment_evidence(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    await get_experiment_dict(safe_experiment_id)
    steps = _load_json_if_exists(_experiment_output_dir(safe_experiment_id) / "steps.json") or []

    all_evidence = []
    seen_ids = set()
    for step in steps:
        for ref in step.get("evidence_refs", []):
            if ref.get("evidence_id") not in seen_ids:
                all_evidence.append({**ref, "step_id": step.get("step_id")})
                seen_ids.add(ref.get("evidence_id"))

    return {"total": len(all_evidence), "evidence": all_evidence}


@app.get("/api/v1/experiments/{experiment_id}/structured", tags=["experiments"])
async def get_structured_json(
    experiment_id: str,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """."""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp = await get_experiment_dict(safe_experiment_id)
    structured_file = _experiment_output_dir(safe_experiment_id) / "structured.json"
    if structured_file.exists():
        return json.loads(structured_file.read_text(encoding="utf-8"))

    timeline_file = _experiment_output_dir(safe_experiment_id) / "timeline.json"
    if timeline_file.exists():
        timeline = json.loads(timeline_file.read_text(encoding="utf-8"))
        evidence_payload = await get_experiment_evidence(safe_experiment_id, auth_ctx)
        return {
            "experiment_id": safe_experiment_id,
            "title": exp.get("title", ""),
            "status": exp.get("status"),
            "steps": timeline.get("steps", []),
            "timeline": timeline.get("steps", []),
            "evidence": evidence_payload.get("evidence", []),
            "protocol": exp.get("protocol_text"),
            "sop": None,
            "analysis": {
                "job_id": exp.get("analysis_job_id"),
                "analyzed_at": exp.get("analyzed_at"),
                "available": True,
            },
            "statistics": {
                "total_steps": timeline.get("total_steps", 0),
                "confirmed_count": timeline.get("confirmed_steps", 0),
                "inferred_count": timeline.get("inferred_steps", 0),
                "average_confidence": timeline.get("avg_confidence"),
            },
        }

    return _empty_structured_payload(exp)


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------

async def get_experiment_dict(experiment_id: str) -> Dict[str, Any]:
    safe_experiment_id = _validate_experiment_id(experiment_id)
    if safe_experiment_id in _EXPERIMENTS:
        return _normalize_experiment_dict(_EXPERIMENTS[safe_experiment_id])
    exp_file = _experiment_output_dir(safe_experiment_id) / "experiment.json"
    if exp_file.exists():
        exp = _normalize_experiment_dict(json.loads(exp_file.read_text(encoding="utf-8-sig")))
        _EXPERIMENTS[safe_experiment_id] = exp
        return exp
    raise HTTPException(status_code=404, detail="Experiment not found")


# ---------------------------------------------------------------------------
# ?
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

@app.patch("/api/v1/experiments/{experiment_id}/steps/{step_id}", tags=["experiments"])
async def update_step(
    experiment_id: str,
    step_id: str,
    request: Request,
    auth_ctx: Dict[str, Any] = Depends(_require_operator_context),
):
    """?reviewed_steps.json ?"""
    safe_experiment_id = _enforce_experiment_scope(auth_ctx, experiment_id)
    exp_dir = _experiment_output_dir(safe_experiment_id)
    steps_file = exp_dir / "steps.json"
    if not steps_file.exists():
        raise HTTPException(status_code=404, detail="Steps not found.")

    steps = json.loads(steps_file.read_text(encoding="utf-8"))
    update_data = await request.json()
    reviewer = update_data.pop("reviewer", "unknown")
    reason = update_data.pop("reason", "")

    step_found = False
    original_step = None
    updated_step = None
    changes = []

    for i, step in enumerate(steps):
        if step.get("step_id") == step_id:
            step_found = True
            original_step = dict(step)  # ?

            # ?
            for field_name in ["step_name", "step_description", "status", "start_time_sec",
                               "end_time_sec", "duration_sec", "notes", "reviewer_note"]:
                if field_name in update_data:
                    old_val = step.get(field_name)
                    new_val = update_data[field_name]
                    if old_val != new_val:
                        changes.append({
                            "step_id": step_id,
                            "field": field_name,
                            "old_value": old_val,
                            "new_value": new_val,
                            "reason": reason,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })

                    if field_name == "end_time_sec":
                        start = update_data.get("start_time_sec", step.get("start_time_sec"))
                        if start is not None:
                            step["duration_sec"] = new_val - start
                    step[field_name] = new_val

            # Update timestamp
            step["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated_step = step
            break

    if not step_found:
        raise HTTPException(status_code=404, detail=f"Step {step_id} not found")

    # ?steps.json
    steps_file.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")

    #  reviewed_steps.json
    reviewed_dir = exp_dir
    reviewed_file = reviewed_dir / "reviewed_steps.json"

    existing_reviewed = {}
    if reviewed_file.exists():
        try:
            existing_reviewed = json.loads(reviewed_file.read_text(encoding="utf-8"))
        except Exception:
            existing_reviewed = {}

    reviewed_data = {
        "experiment_id": safe_experiment_id,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reviewer": reviewer,
        "original_steps": existing_reviewed.get("reviewed_steps", []),
        "reviewed_steps": steps,
        "changes": existing_reviewed.get("changes", []) + changes,
    }

    reviewed_file.write_text(
        json.dumps(reviewed_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    #  timeline
    timeline_file = exp_dir / "timeline.json"
    if timeline_file.exists():
        timeline = json.loads(timeline_file.read_text(encoding="utf-8"))
        for i, tl_step in enumerate(timeline.get("steps", [])):
            if tl_step.get("step_id") == step_id:
                timeline["steps"][i] = updated_step
                break
        timeline_file.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "step": updated_step,
        "changes_count": len(changes),
        "reviewed_steps_path": str(reviewed_file),
    }


# ---------------------------------------------------------------------------
#  API
# ---------------------------------------------------------------------------

def _resolve_video_analysis_model_path(override_path: Optional[str] = None) -> Optional[str]:
    candidates: List[str] = []
    if override_path:
        candidate_path = Path(override_path)
        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / override_path
        if candidate_path.exists():
            return str(candidate_path.resolve())
        if RUNTIME_SETTINGS is not None and getattr(RUNTIME_SETTINGS, "strict_model", False):
            raise RuntimeError(f"YOLO26 override path does not exist in strict mode: {override_path}")
        candidates.append(override_path)
    env_override = os.environ.get("LABSOPGUARD_YOLO_MODEL")
    if env_override:
        candidates.append(env_override)
    if RUNTIME_SETTINGS is not None and RUNTIME_SETTINGS.yolo_model_path:
        candidates.append(RUNTIME_SETTINGS.yolo_model_path)

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate
        if candidate_path.exists():
            return str(candidate_path.resolve())
    return None


@app.api_route(
    "/api/v1/video-analysis/{removed_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    include_in_schema=False,
)
async def removed_standalone_video_analysis(removed_path: str):
    raise HTTPException(
        status_code=404,
        detail="Standalone video analysis has been removed; use experiment analysis instead",
    )


# ── SPA static file serving (production) ─────────────────────────
_FRONTEND_DIST = PROJECT_ROOT / "frontend-app" / "dist"
if _FRONTEND_DIST.exists():
    from fastapi.staticfiles import StaticFiles

    _ASSETS_DIR = _FRONTEND_DIST / "assets"
    if _ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="frontend-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        file_path = _FRONTEND_DIST / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_FRONTEND_DIST / "index.html"))

