from __future__ import annotations

from typing import Any, Dict, List


def build_recovery_plan(error_code: str) -> List[Dict[str, Any]]:
    if error_code == "missing_target_xyz":
        return [
            {"type": "logic", "command": "pause"},
            {"type": "logic", "command": "request_relocalization"},
            {"type": "logic", "command": "notify_operator"},
        ]
    return [
        {"type": "logic", "command": "pause"},
        {"type": "cartesian_move", "command": "retreat_safe_pose"},
        {"type": "logic", "command": "notify_operator"},
    ]

