from __future__ import annotations

from typing import Dict, List


def action_exact_match(pred: List[str], gt: List[str]) -> float:
    return 1.0 if pred == gt else 0.0


def compute_metrics(action_scores: List[float], perception_scores: List[float]) -> Dict[str, float]:
    if not action_scores:
        return {"action_exact_match": 0.0, "perception_confidence_mean": 0.0}

    return {
        "action_exact_match": sum(action_scores) / len(action_scores),
        "perception_confidence_mean": sum(perception_scores) / len(perception_scores) if perception_scores else 0.0,
    }
