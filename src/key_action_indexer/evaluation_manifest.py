from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluation import load_micro_eval_config
from .schemas import SessionManifest, read_jsonl, write_jsonl


EVALUATION_MANIFEST_SCHEMA_VERSION = "key_action_evaluation_manifest.v1"
MICRO_GT_TEMPLATE_SCHEMA_VERSION = "key_action_micro_gt_template.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path_status(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    source = Path(path)
    return {
        "path": str(source),
        "exists": source.exists(),
        "size_bytes": int(source.stat().st_size) if source.exists() and source.is_file() else 0,
    }


def _nested(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _read_jsonl_if_exists(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    return read_jsonl(source) if source.exists() else []


def _row_primary_object(row: dict[str, Any]) -> str:
    return _safe_str(row.get("primary_object") or _nested(row, "interaction").get("primary_object"))


def _row_interaction_type(row: dict[str, Any]) -> str:
    return _safe_str(row.get("interaction_type") or _nested(row, "interaction").get("interaction_type"))


def _row_action_type(row: dict[str, Any]) -> str:
    text = _nested(row, "text_description")
    return _safe_str(row.get("action_type") or row.get("label") or text.get("action_type"))


def _row_summary(row: dict[str, Any]) -> str:
    text = _nested(row, "text_description")
    return _safe_str(row.get("summary") or text.get("summary") or text.get("index_text"))


def _row_segment_id(row: dict[str, Any]) -> str:
    return _safe_str(row.get("segment_id") or row.get("parent_segment_id"))


def _row_start(row: dict[str, Any]) -> float:
    return _safe_float(row.get("start_sec", row.get("session_start_sec", row.get("local_start_sec"))))


def _row_end(row: dict[str, Any]) -> float:
    return _safe_float(row.get("end_sec", row.get("session_end_sec", row.get("local_end_sec"))))


def _merged_duration(windows: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (_safe_float(window.get("start_sec")), _safe_float(window.get("end_sec")))
        for window in windows
        if _safe_float(window.get("end_sec")) > _safe_float(window.get("start_sec"))
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return round(sum(end - start for start, end in merged), 6)


def _labeled_windows_from_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments, start=1):
        start = _row_start(segment)
        end = _row_end(segment)
        if end <= start:
            continue
        segment_id = _row_segment_id(segment) or f"segment_{index:06d}"
        windows.append(
            {
                "window_id": f"win_{index:03d}",
                "segment_id": segment_id,
                "start_sec": start,
                "end_sec": end,
                "description": _row_summary(segment) or _row_action_type(segment) or "key action segment labeled window",
            }
        )
    return windows


def _labeled_windows_from_micro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        segment_id = _row_segment_id(row) or "unassigned_segment"
        grouped.setdefault(segment_id, []).append(row)
    windows: list[dict[str, Any]] = []
    for index, (segment_id, segment_rows) in enumerate(sorted(grouped.items()), start=1):
        start = min((_row_start(row) for row in segment_rows), default=0.0)
        end = max((_row_end(row) for row in segment_rows), default=0.0)
        if end <= start:
            continue
        windows.append(
            {
                "window_id": f"win_{index:03d}",
                "segment_id": segment_id,
                "start_sec": start,
                "end_sec": end,
                "description": "derived from predicted micro-segment span",
            }
        )
    return windows


def _template_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    micro_id = _safe_str(row.get("micro_segment_id")) or f"pred_micro_{index:06d}"
    segment_id = _row_segment_id(row)
    return {
        "segment_id": segment_id,
        "parent_segment_id": segment_id,
        "micro_segment_id": micro_id,
        "source_prediction_id": micro_id,
        "start_sec": _row_start(row),
        "end_sec": _row_end(row),
        "primary_object": _row_primary_object(row),
        "action_type": _row_action_type(row),
        "interaction_type": _row_interaction_type(row),
        "prediction_summary": _row_summary(row),
        "needs_manual_label": True,
        "manual_review_status": "unlabeled",
        "gt_keep": "",
        "manual_fields_required": [
            "manual_review_status",
            "gt_keep",
            "start_sec",
            "end_sec",
            "primary_object",
            "action_type",
            "interaction_type",
            "annotator_notes",
            "missed_micro_segments_in_same_window",
        ],
        "annotator_notes": "",
        "missed_micro_segments_in_same_window": "",
    }


def build_micro_gt_template_manifest(
    session_dir: str | Path | None = None,
    *,
    micro_segments_path: str | Path | None = None,
    key_action_segments_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    gt_completeness: str = "unknown",
) -> dict[str, Any]:
    session = Path(session_dir) if session_dir is not None else None
    if micro_segments_path is None:
        if session is None:
            raise ValueError("session_dir or micro_segments_path is required")
        micro_segments_path = session / "metadata" / "micro_segments.jsonl"
    if key_action_segments_path is None and session is not None:
        key_action_segments_path = session / "metadata" / "key_action_segments.jsonl"
    target_dir = Path(output_dir) if output_dir is not None else (session / "annotation" / "micro_gt" if session else Path(micro_segments_path).parent)
    target_dir.mkdir(parents=True, exist_ok=True)

    micro_rows = _read_jsonl_if_exists(micro_segments_path)
    key_segments = _read_jsonl_if_exists(key_action_segments_path)
    windows = _labeled_windows_from_segments(key_segments) or _labeled_windows_from_micro_rows(micro_rows)
    completeness = str(gt_completeness or "unknown")
    precision_is_formal = bool(completeness == "complete" and windows)
    metric_mode = "formal_after_manual_validation" if precision_is_formal else "debugging_until_complete_gt"

    template_path = target_dir / "manual_micro_gt.template.jsonl"
    eval_config_path = target_dir / "eval_config.json"
    manifest_path = target_dir / "micro_gt_manifest.json"
    template_rows = [_template_row(row, index) for index, row in enumerate(micro_rows, start=1)]
    write_jsonl(template_path, template_rows)

    eval_config = {
        "schema_version": MICRO_GT_TEMPLATE_SCHEMA_VERSION,
        "generated_at": _now(),
        "session_dir": str(session) if session is not None else None,
        "source_micro_segments": str(micro_segments_path),
        "source_key_action_segments": str(key_action_segments_path) if key_action_segments_path is not None else None,
        "manual_micro_gt_template": str(template_path),
        "gt_completeness": completeness,
        "labeled_windows": windows,
        "labeled_window_count": len(windows),
        "labeled_duration_sec": _merged_duration(windows),
        "metric_mode": metric_mode,
        "precision_is_formal": precision_is_formal,
        "manual_labeling_required": True,
        "formal_metric_warning": None
        if precision_is_formal
        else "Template rows are prediction-seeded and require human completion before precision/recall are formal.",
        "annotation_rules": [
            "Label every visible hand-object micro interaction inside each labeled window.",
            "Edit prediction-seeded start/end/object/action fields instead of accepting them blindly.",
            "Set gt_keep=false or delete rows for false-positive predictions.",
            "Add rows for missed micro-interactions in the same windows.",
        ],
    }
    eval_config_path.write_text(json.dumps(eval_config, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "schema_version": MICRO_GT_TEMPLATE_SCHEMA_VERSION,
        "generated_at": _now(),
        "session_dir": str(session) if session is not None else None,
        "micro_segment_count": len(micro_rows),
        "key_action_segment_count": len(key_segments),
        "manual_micro_gt_template": str(template_path),
        "eval_config": str(eval_config_path),
        "gt_completeness": completeness,
        "labeled_window_count": len(windows),
        "labeled_duration_sec": eval_config["labeled_duration_sec"],
        "metric_mode": metric_mode,
        "precision_is_formal": precision_is_formal,
    }
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    return summary


def build_evaluation_manifest(
    manifest_path: str | Path,
    *,
    output_path: str | Path | None = None,
    dialogue_path: str | Path | None = None,
    sop_path: str | Path | None = None,
    segment_labels_path: str | Path | None = None,
    micro_labels_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    expected_output_path: str | Path | None = None,
    time_alignment_anchors_path: str | Path | None = None,
    version: int = 1,
) -> dict[str, Any]:
    manifest = SessionManifest.load(manifest_path)
    session_dir = Path(manifest.output_dir)
    dialogue_source = dialogue_path or (manifest.transcript.path if manifest.transcript else None)
    target = Path(output_path) if output_path else session_dir / "evaluation" / "evaluation_manifest.json"
    third_person = _path_status(manifest.videos.third_person.path)
    first_person = _path_status(manifest.videos.first_person.path if manifest.videos.first_person else None)
    dialogue = _path_status(dialogue_source)
    sop = _path_status(sop_path)
    segment_labels = _path_status(segment_labels_path)
    micro_labels = _path_status(micro_labels_path)
    eval_config_status = _path_status(eval_config_path)
    expected_output = _path_status(expected_output_path)
    time_alignment_anchors = _path_status(time_alignment_anchors_path)
    human_labels_available = segment_labels["exists"] and micro_labels["exists"]
    eval_config = load_micro_eval_config(eval_config_path) if eval_config_path else None
    labeled_windows = list((eval_config or {}).get("labeled_windows") or [])
    gt_completeness = str((eval_config or {}).get("gt_completeness") or ("partial" if micro_labels["exists"] else "unknown"))
    precision_is_formal = bool(micro_labels["exists"] and gt_completeness == "complete" and labeled_windows)
    metric_mode = "formal" if precision_is_formal else "debugging"
    coverage = {
        "real_dual_view_video": bool(third_person["exists"] and first_person["exists"]),
        "dialogue": bool(dialogue["exists"]),
        "sop": bool(sop["exists"]),
        "human_labels": bool(human_labels_available),
        "expected_output": bool(expected_output["exists"]),
        "time_alignment_anchors": bool(time_alignment_anchors["exists"]),
        "complete_micro_gt": precision_is_formal,
    }
    required_coverage_keys = [key for key in coverage if key != "complete_micro_gt"]
    audit = [
        {
            "timestamp": _now(),
            "actor": "key_action_indexer",
            "action": "evaluation_manifest_created",
            "source_session_id": manifest.session_id,
            "details": {"manifest_path": str(manifest_path), "output_path": str(target)},
        }
    ]
    result = {
        "schema_version": EVALUATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": _now(),
        "source_session_id": manifest.session_id,
        "version": int(version),
        "valid": all(coverage[key] for key in required_coverage_keys),
        "gt_completeness": gt_completeness,
        "labeled_window_count": len(labeled_windows),
        "metric_mode": metric_mode,
        "precision_recall_are_formal": precision_is_formal,
        "formal_metric_warning": None
        if precision_is_formal
        else "Complete labeled windows and manual micro GT are required before precision/recall are formal.",
        "coverage": coverage,
        "missing_requirements": [key for key in required_coverage_keys if not coverage[key]],
        "audit_trail": audit,
        "assets": {
            "manifest": _path_status(manifest_path),
            "videos": {
                "third_person": third_person,
                "first_person": first_person,
            },
            "dialogue": dialogue,
            "sop": sop,
            "human_labels": {
                "segment_labels": segment_labels,
                "micro_labels": micro_labels,
                "eval_config": eval_config_status,
            },
            "expected_output": expected_output,
            "time_alignment_anchors": time_alignment_anchors,
        },
        "ground_truth_coverage": {
            "gt_completeness": gt_completeness,
            "labeled_window_count": len(labeled_windows),
            "labeled_duration_sec": _merged_duration(labeled_windows),
            "metric_mode": metric_mode,
            "precision_recall_are_formal": precision_is_formal,
        },
        "expected_metrics": {
            "time_alignment": ["mae_sec", "max_residual_sec", "anchor_coverage_rate", "drift_error_sec"],
            "retrieval": ["top1_hit_rate", "topk_hit_rate", "expected_object_hit_rate"],
            "micro_segments": ["precision", "recall", "f1", "mean_iou", "primary_object_accuracy"],
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["manifest_path"] = str(target)
    return result


__all__ = [
    "EVALUATION_MANIFEST_SCHEMA_VERSION",
    "MICRO_GT_TEMPLATE_SCHEMA_VERSION",
    "build_evaluation_manifest",
    "build_micro_gt_template_manifest",
]
