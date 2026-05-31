from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import load_micro_eval_config, load_manual_micro_segments


ALLOWED_PRIMARY_OBJECTS = {
    "balance",
    "bottle",
    "reagent_bottle",
    "sample_bottle",
    "sample_bottle_blue",
    "spatula",
    "pipette",
    "pipette_tip",
    "tube",
    "paper",
    "hand",
    "other",
}


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(0.0, min(float(a["end_sec"]), float(b["end_sec"])) - max(float(a["start_sec"]), float(b["start_sec"])))


def _inside_windows(row: dict[str, Any], windows: list[dict[str, Any]]) -> bool:
    if not windows:
        return True
    return any(_overlap(row, window) > 0 for window in windows)


def _labeled_duration(windows: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (float(window.get("start_sec", 0.0) or 0.0), float(window.get("end_sec", 0.0) or 0.0))
        for window in windows
        if float(window.get("end_sec", 0.0) or 0.0) > float(window.get("start_sec", 0.0) or 0.0)
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def validate_micro_gt(
    ground_truth_path: str | Path,
    eval_config_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = load_manual_micro_segments(ground_truth_path)
    eval_config = load_micro_eval_config(eval_config_path) or {}
    windows = list(eval_config.get("labeled_windows") or [])
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_rows = []
    source = Path(ground_truth_path)
    if source.exists():
        raw_rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    raw_by_id = {str(row.get("micro_segment_id") or ""): row for row in raw_rows if isinstance(row, dict)}

    for row in rows:
        row_id = row.get("micro_segment_id")
        raw = raw_by_id.get(str(row_id), {})
        if float(row.get("start_sec", 0.0)) >= float(row.get("end_sec", 0.0)):
            errors.append({"type": "invalid_time_range", "micro_segment_id": row_id})
        if windows and not _inside_windows(row, windows):
            warnings.append({"type": "gt_outside_labeled_windows", "micro_segment_id": row_id})
        primary = str(row.get("primary_object") or "")
        if primary and primary not in ALLOWED_PRIMARY_OBJECTS:
            warnings.append({"type": "unknown_primary_object", "micro_segment_id": row_id, "primary_object": primary})
        if not str(row.get("action_type") or "").strip():
            warnings.append({"type": "missing_action_type", "micro_segment_id": row_id})
        manual_status = raw.get("manual_review_status")
        if raw.get("needs_manual_label") is True or (
            manual_status is not None and str(manual_status or "").lower() in {"", "unlabeled", "needs_review"}
        ):
            warnings.append({"type": "manual_labeling_required", "micro_segment_id": row_id})

    ordered = sorted(rows, key=lambda item: (float(item.get("start_sec", 0.0)), float(item.get("end_sec", 0.0))))
    for left, right in zip(ordered, ordered[1:]):
        overlap = _overlap(left, right)
        shorter = min(float(left["end_sec"]) - float(left["start_sec"]), float(right["end_sec"]) - float(right["start_sec"]))
        if shorter > 0 and overlap / shorter >= 0.7:
            warnings.append(
                {
                    "type": "severe_gt_overlap",
                    "left": left.get("micro_segment_id"),
                    "right": right.get("micro_segment_id"),
                    "overlap_sec": overlap,
                }
            )

    completeness = str(eval_config.get("gt_completeness") or "unknown")
    labeled_duration = _labeled_duration(windows)
    precision_is_formal = bool(completeness == "complete" and windows)
    metric_mode = "formal" if precision_is_formal else "debugging"
    if completeness == "complete" and labeled_duration > 30.0 and len(rows) <= 3:
        warnings.append(
            {
                "type": "complete window has very few GT segments; verify annotation completeness",
                "gt_count": len(rows),
                "labeled_duration_sec": labeled_duration,
            }
        )
    if completeness == "complete" and not windows:
        errors.append({"type": "complete_gt_requires_labeled_windows"})

    result = {
        "ground_truth": str(ground_truth_path),
        "eval_config": str(eval_config_path),
        "gt_count": len(rows),
        "gt_completeness": completeness,
        "labeled_window_count": len(windows),
        "labeled_duration_sec": labeled_duration,
        "metric_mode": metric_mode,
        "precision_is_formal": precision_is_formal,
        "formal_metric_warning": None
        if precision_is_formal
        else "GT coverage is partial, unknown, or unlabeled; precision/recall are debugging metrics only.",
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
