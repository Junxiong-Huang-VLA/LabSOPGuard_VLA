from __future__ import annotations

from pathlib import Path

from .manifest import build_manifest, discover_camera_runs

__all__ = ["build_manifest", "discover_camera_runs"]


def normalize_root(path: str | Path) -> Path:
    return Path(str(path))

