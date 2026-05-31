from __future__ import annotations

from pathlib import Path
from typing import Iterable


SESSION_LAYOUT_VERSION = "key_action_session_layout.v1"

STANDARD_SESSION_DIRS: tuple[str, ...] = (
    "raw",
    "transcript",
    "uploads",
    "cv_outputs",
    "clips",
    "keyframes",
    "metadata",
    "index",
    "debug",
    "evaluation",
    "reports",
    "exports",
)


def initialize_session_dir(session_dir: str | Path, extra_dirs: Iterable[str] | None = None) -> dict[str, Path]:
    root = Path(session_dir)
    names = [*STANDARD_SESSION_DIRS, *(extra_dirs or [])]
    paths = {name: root / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def session_layout_summary(session_dir: str | Path) -> dict[str, object]:
    root = Path(session_dir)
    return {
        "schema_version": SESSION_LAYOUT_VERSION,
        "session_dir": str(root),
        "directories": {
            name: {
                "path": str(root / name),
                "exists": (root / name).exists(),
            }
            for name in STANDARD_SESSION_DIRS
        },
    }
