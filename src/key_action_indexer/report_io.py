"""Shared report I/O helper (P4 consolidation).

Several validation/accuracy modules each had a private ``_write_json`` that
serialized a report dict with identical options. This consolidates them into a
single helper so the JSON formatting (indent, unicode handling) is guaranteed
consistent across every report the system writes.

Behavior is intentionally identical to the previous private helpers:
``json.dumps(payload, ensure_ascii=False, indent=2)`` plus parent-dir creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def write_json_report(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write ``payload`` as pretty UTF-8 JSON, creating parent dirs.

    Returns the path written (as a Path) for convenient chaining.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
