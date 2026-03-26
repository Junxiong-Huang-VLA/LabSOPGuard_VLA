from __future__ import annotations

from typing import Dict, List


def compute_policy_metrics(success_flags: List[bool], task_time_s: List[float]) -> Dict[str, float]:
    n = len(success_flags)
    if n == 0:
        return {"success_rate": 0.0, "avg_task_time_s": 0.0}
    success_rate = float(sum(1 for v in success_flags if v) / n)
    avg_time = float(sum(task_time_s) / max(1, len(task_time_s)))
    return {"success_rate": success_rate, "avg_task_time_s": avg_time}

