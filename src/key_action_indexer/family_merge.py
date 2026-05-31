from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import read_jsonl


OBJECT_FAMILIES: dict[str, list[str]] = {
    "bottle_family": ["bottle", "sample_bottle", "sample_bottle_blue", "reagent_bottle"],
    "pipette_family": ["pipette", "pipette_tip"],
    "spatula_family": ["spatula", "spoon"],
    "tube_family": ["tube", "test_tube"],
    "balance_family": ["balance"],
    "paper_family": ["paper", "weighing_paper", "filter_paper"],
    "equipment_family": ["magnetic_stirrer", "magnetic_stir_bar"],
}


def _primary(row: dict[str, Any]) -> str:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
    return str(row.get("primary_object") or interaction.get("primary_object") or "")


def object_family(label: str) -> str | None:
    normalized = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    for family, labels in OBJECT_FAMILIES.items():
        if normalized in labels:
            return family
    return None


def _gap(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(right.get("start_sec", 0.0) or 0.0) - float(left.get("end_sec", 0.0) or 0.0)


def analyze_object_family_merge(
    session_dir: str | Path,
    *,
    micro_segments_path: str | Path | None = None,
    max_gap_sec: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    source = Path(micro_segments_path) if micro_segments_path else session / "metadata" / "micro_segments.jsonl"
    rows = read_jsonl(source) if source.exists() else []
    ordered = sorted(
        rows,
        key=lambda row: (str(row.get("parent_segment_id") or ""), float(row.get("start_sec", 0.0) or 0.0)),
    )
    candidates: list[dict[str, Any]] = []
    risky: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        if str(left.get("parent_segment_id") or "") != str(right.get("parent_segment_id") or ""):
            continue
        left_obj = _primary(left)
        right_obj = _primary(right)
        if left_obj == right_obj:
            continue
        family = object_family(left_obj)
        if not family or family != object_family(right_obj):
            continue
        gap = _gap(left, right)
        if gap < -0.05 or gap > max_gap_sec:
            continue
        pair = {
            "left_micro_segment_id": left.get("micro_segment_id"),
            "right_micro_segment_id": right.get("micro_segment_id"),
            "left_primary_object": left_obj,
            "right_primary_object": right_obj,
            "family": family,
            "gap_sec": gap,
        }
        candidates.append(pair)
        left_level = str(left.get("evidence_level") or (left.get("evidence") or {}).get("evidence_level") or "")
        right_level = str(right.get("evidence_level") or (right.get("evidence") or {}).get("evidence_level") or "")
        if left_level != right_level and (left_level.startswith("insufficient") or right_level.startswith("insufficient")):
            risky.append({**pair, "reason": "evidence_level_conflict"})
    estimated_count = max(0, len(rows) - len(candidates))
    recommendation = "analysis_only"
    if candidates and len(risky) / max(1, len(candidates)) < 0.25:
        recommendation = "consider_family_merge_after_gt_review"
    result = {
        "session_dir": str(session),
        "source": str(source),
        "object_families": OBJECT_FAMILIES,
        "adjacent_same_family_micro_count": len(candidates),
        "candidate_family_merge_pairs": candidates,
        "estimated_micro_count_after_family_merge": estimated_count,
        "risky_merge_pairs": risky,
        "recommendation": recommendation,
    }
    target = Path(output_path) if output_path else session / "evaluation" / "object_family_merge_analysis.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
