from __future__ import annotations

from typing import Any, Dict, List, Optional


def detect_endpoint_frame(events: List[Dict[str, Any]]) -> Optional[int]:
    """
    Minimal endpoint detector for titration workflow.
    Returns frame_id when 'pipette_transfer' appears stable.
    """
    recent = 0
    for e in events:
        acts = e.get("actions") or e.get("detection", {}).get("actions", [])
        if "pipette_transfer" in acts:
            recent += 1
            if recent >= 5:
                return int(e.get("frame_id", -1))
        else:
            recent = 0
    return None

