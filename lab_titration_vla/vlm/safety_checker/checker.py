from __future__ import annotations

from typing import Any, Dict, List


def check_safety(plan: Dict[str, Any], ppe_state: Dict[str, bool]) -> Dict[str, Any]:
    violations: List[str] = []
    if not bool(ppe_state.get("wear_gloves", False)):
        violations.append("no_gloves")
    if not bool(ppe_state.get("wear_goggles", False)):
        violations.append("no_goggles")
    return {
        "safe_to_execute": len(violations) == 0,
        "violations": violations,
        "recommendation": "pause_and_alert" if violations else "proceed",
        "plan": plan,
    }

