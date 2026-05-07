from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
RUNTIME_ROOT = PROJECT_ROOT / ".runtime"
ULTRALYTICS_ROOT = PROJECT_ROOT / ".ultralytics"
MEMORY_ROOT = PROJECT_ROOT / "memory"
STREAM_BUFFER_ROOT = RUNTIME_ROOT / "stream_buffers"


def ensure_runtime_dirs() -> None:
    for path in (RUNTIME_ROOT, ULTRALYTICS_ROOT, MEMORY_ROOT, STREAM_BUFFER_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def configure_runtime_environment() -> None:
    ensure_runtime_dirs()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_ROOT))
