from __future__ import annotations

import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VENDOR_PY = _PROJECT_ROOT / "vendor_py"

if _VENDOR_PY.exists():
    vendor_path = str(_VENDOR_PY)
    if vendor_path not in sys.path:
        # Keep installed environment packages ahead of vendored fallbacks.
        sys.path.append(vendor_path)
