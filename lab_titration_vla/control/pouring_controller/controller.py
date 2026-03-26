from __future__ import annotations

from typing import Any, Dict, List


def build_pouring_sequence(volume_ml: float = 1.0, duration_s: float = 2.0) -> List[Dict[str, Any]]:
    return [
        {"type": "cartesian_move", "command": "align_above_target", "speed_scale": 0.15},
        {"type": "wrist_rotate", "angle_deg": 25, "duration_s": duration_s / 2.0},
        {"type": "hold", "duration_s": duration_s / 2.0, "meta": {"volume_ml": volume_ml}},
        {"type": "wrist_rotate", "angle_deg": -25, "duration_s": duration_s / 2.0},
    ]

