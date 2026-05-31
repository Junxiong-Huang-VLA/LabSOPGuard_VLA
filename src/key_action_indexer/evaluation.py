from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .family_merge import object_family
from .schemas import read_jsonl


def load_manual_segments(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for row in read_jsonl(source):
        if "start_sec" not in row or "end_sec" not in row:
            continue
        rows.append(
            {
                "segment_id": str(row.get("segment_id", f"gt_{len(rows) + 1:06d}")),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": str(row.get("label", "")),
            }
        )
    return rows


def load_manual_micro_segments(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for row in read_jsonl(source):
        if "start_sec" not in row or "end_sec" not in row:
            continue
        rows.append(
            {
                "micro_segment_id": str(row.get("micro_segment_id", f"gt_micro_{len(rows) + 1:06d}")),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "primary_object": str(row.get("primary_object", "")),
                "interaction_type": str(row.get("interaction_type", "")),
                "action_type": str(row.get("action_type", "")),
            }
        )
    return rows


def load_manual_micro_segments(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    rows = []
    for row in read_jsonl(source):
        if "start_sec" not in row or "end_sec" not in row:
            continue
        rows.append(
            {
                "micro_segment_id": str(row.get("micro_segment_id", f"gt_micro_{len(rows) + 1:06d}")),
                "parent_segment_id": str(row.get("parent_segment_id", "")),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": str(row.get("label") or row.get("action_type") or ""),
                "action_type": str(row.get("action_type") or row.get("label") or ""),
                "primary_object": str(row.get("primary_object") or ""),
                "interaction_type": str(row.get("interaction_type") or ""),
            }
        )
    return rows


def compute_temporal_iou(segment_a: dict[str, Any], segment_b: dict[str, Any]) -> float:
    a_start = float(segment_a["start_sec"])
    a_end = float(segment_a["end_sec"])
    b_start = float(segment_b["start_sec"])
    b_end = float(segment_b["end_sec"])
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return float(intersection / union)


def _normalize_predicted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows, start=1):
        if "start_sec" not in row or "end_sec" not in row:
            continue
        normalized.append(
            {
                "segment_id": str(row.get("segment_id", f"pred_{idx:06d}")),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
            }
        )
    return normalized


def _normalize_micro(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows, start=1):
        if "start_sec" not in row or "end_sec" not in row:
            continue
        interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
        text = row.get("text_description") if isinstance(row.get("text_description"), dict) else {}
        normalized.append(
            {
                "micro_segment_id": str(row.get("micro_segment_id", f"pred_micro_{idx:06d}")),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "primary_object": str(row.get("primary_object") or interaction.get("primary_object") or ""),
                "interaction_type": str(row.get("interaction_type") or interaction.get("interaction_type") or ""),
                "action_type": str(row.get("action_type") or text.get("action_type") or ""),
            }
        )
    return normalized


def _normalize_micro(rows: list[dict[str, Any]], *, default_prefix: str = "micro") -> list[dict[str, Any]]:
    normalized = []
    for idx, row in enumerate(rows, start=1):
        if "start_sec" not in row or "end_sec" not in row:
            continue
        interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
        text_description = row.get("text_description") if isinstance(row.get("text_description"), dict) else {}
        detected_objects = row.get("detected_objects") or interaction.get("detected_objects") or []
        if not isinstance(detected_objects, list):
            detected_objects = []
        normalized.append(
            {
                "micro_segment_id": str(row.get("micro_segment_id", f"{default_prefix}_{idx:06d}")),
                "parent_segment_id": str(row.get("parent_segment_id") or row.get("segment_id") or ""),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label": str(row.get("label") or text_description.get("action_type") or row.get("action_type") or ""),
                "action_type": str(text_description.get("action_type") or row.get("action_type") or row.get("label") or ""),
                "primary_object": str(interaction.get("primary_object") or row.get("primary_object") or ""),
                "interaction_type": str(interaction.get("interaction_type") or row.get("interaction_type") or ""),
                "detected_objects": [str(item) for item in detected_objects],
            }
        )
    return normalized


def _normalized_object_set(row: dict[str, Any]) -> set[str]:
    values = {str(row.get("primary_object") or "")}
    values.update(str(item) for item in row.get("detected_objects") or [])
    return {value.strip().lower().replace("-", "_").replace(" ", "_") for value in values if value}


def match_predicted_to_ground_truth(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    def row_id(row: dict[str, Any]) -> Any:
        return row.get("segment_id") or row.get("micro_segment_id")

    matches: list[dict[str, Any]] = []
    used_predicted: set[int] = set()
    used_gt: set[int] = set()
    candidates = []
    for pred_idx, pred in enumerate(predicted):
        for gt_idx, gt in enumerate(ground_truth):
            candidates.append((compute_temporal_iou(pred, gt), pred_idx, gt_idx))
    for iou, pred_idx, gt_idx in sorted(candidates, key=lambda item: item[0], reverse=True):
        if iou < iou_threshold:
            break
        if pred_idx in used_predicted or gt_idx in used_gt:
            continue
        used_predicted.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append(
            {
                "predicted_segment_id": predicted[pred_idx].get("segment_id"),
                "ground_truth_segment_id": ground_truth[gt_idx].get("segment_id"),
                "predicted_micro_segment_id": predicted[pred_idx].get("micro_segment_id"),
                "ground_truth_micro_segment_id": ground_truth[gt_idx].get("micro_segment_id"),
                "iou": float(iou),
            }
        )
    return {
        "matches": matches,
        "unmatched_predicted": [row_id(predicted[idx]) for idx in range(len(predicted)) if idx not in used_predicted],
        "unmatched_ground_truth": [row_id(ground_truth[idx]) for idx in range(len(ground_truth)) if idx not in used_gt],
    }


def match_micro_predicted_to_ground_truth(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    used_predicted: set[int] = set()
    used_gt: set[int] = set()
    candidates: list[tuple[float, int, int, int]] = []
    for pred_idx, pred in enumerate(predicted):
        for gt_idx, gt in enumerate(ground_truth):
            if pred.get("parent_segment_id") and gt.get("parent_segment_id") and pred.get("parent_segment_id") != gt.get("parent_segment_id"):
                continue
            iou = compute_temporal_iou(pred, gt)
            if iou < iou_threshold:
                continue
            pred_obj = str(pred.get("primary_object") or "")
            gt_obj = str(gt.get("primary_object") or "")
            if gt_obj and pred_obj == gt_obj:
                stage = 2
            elif gt_obj and pred_obj and object_family(pred_obj) and object_family(pred_obj) == object_family(gt_obj):
                stage = 1
            else:
                stage = 0
            candidates.append((stage, iou, pred_idx, gt_idx))
    for stage, iou, pred_idx, gt_idx in sorted(candidates, key=lambda item: (item[0], item[1]), reverse=True):
        if iou < iou_threshold:
            break
        if pred_idx in used_predicted or gt_idx in used_gt:
            continue
        used_predicted.add(pred_idx)
        used_gt.add(gt_idx)
        pred = predicted[pred_idx]
        gt = ground_truth[gt_idx]
        primary_object_match = bool(gt.get("primary_object")) and pred.get("primary_object") == gt.get("primary_object")
        pred_family = object_family(str(pred.get("primary_object") or ""))
        gt_family = object_family(str(gt.get("primary_object") or ""))
        primary_object_family_match = bool(primary_object_match or (gt_family and pred_family == gt_family))
        primary_object_detected_match = bool(gt.get("primary_object")) and str(gt.get("primary_object")) in _normalized_object_set(pred)
        interaction_type_match = bool(gt.get("interaction_type")) and pred.get("interaction_type") == gt.get("interaction_type")
        action_type_match = bool(gt.get("action_type")) and pred.get("action_type") == gt.get("action_type")
        matches.append(
            {
                "predicted_micro_segment_id": pred.get("micro_segment_id"),
                "ground_truth_micro_segment_id": gt.get("micro_segment_id"),
                "parent_segment_id": pred.get("parent_segment_id") or gt.get("parent_segment_id"),
                "iou": float(iou),
                "primary_object_match": primary_object_match,
                "primary_object_family_match": primary_object_family_match,
                "primary_object_detected_match": primary_object_detected_match,
                "object_match_stage": "exact" if stage == 2 else "family" if stage == 1 else "temporal",
                "interaction_type_match": interaction_type_match,
                "action_type_match": action_type_match,
                "predicted_primary_object": pred.get("primary_object"),
                "ground_truth_primary_object": gt.get("primary_object"),
                "predicted_detected_objects": sorted(_normalized_object_set(pred)),
                "predicted_object_family": pred_family,
                "ground_truth_object_family": gt_family,
                "predicted_interaction_type": pred.get("interaction_type"),
                "ground_truth_interaction_type": gt.get("interaction_type"),
                "predicted_action_type": pred.get("action_type"),
                "ground_truth_action_type": gt.get("action_type"),
            }
        )
    return {
        "matches": matches,
        "unmatched_predicted": [predicted[idx].get("micro_segment_id") for idx in range(len(predicted)) if idx not in used_predicted],
        "unmatched_ground_truth": [ground_truth[idx].get("micro_segment_id") for idx in range(len(ground_truth)) if idx not in used_gt],
    }


def evaluate_segments(
    predicted_segments_path: str | Path,
    manual_segments_path: str | Path,
    iou_threshold: float = 0.3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    predicted = _normalize_predicted(read_jsonl(predicted_segments_path)) if Path(predicted_segments_path).exists() else []
    ground_truth = load_manual_segments(manual_segments_path)
    result = evaluate_segment_rows(predicted, ground_truth, iou_threshold=iou_threshold)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_micro_segments(
    predicted_micro_segments_path: str | Path,
    manual_micro_segments_path: str | Path,
    iou_threshold: float = 0.3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    predicted = _normalize_micro(read_jsonl(predicted_micro_segments_path), default_prefix="pred_micro") if Path(predicted_micro_segments_path).exists() else []
    ground_truth = _normalize_micro(load_manual_micro_segments(manual_micro_segments_path), default_prefix="gt_micro")
    result = evaluate_micro_segment_rows(predicted, ground_truth, iou_threshold=iou_threshold)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_segment_rows(
    predicted_segments: list[dict[str, Any]],
    ground_truth_segments: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    predicted = _normalize_predicted(predicted_segments)
    ground_truth = _normalize_predicted(ground_truth_segments)
    match_result = match_predicted_to_ground_truth(predicted, ground_truth, iou_threshold=iou_threshold)
    tp = len(match_result["matches"])
    fp = max(0, len(predicted) - tp)
    fn = max(0, len(ground_truth) - tp)
    precision = float(tp / len(predicted)) if predicted else 0.0
    recall = float(tp / len(ground_truth)) if ground_truth else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    mean_iou = float(sum(item["iou"] for item in match_result["matches"]) / tp) if tp else 0.0
    result: dict[str, Any] = {
        "num_predicted": len(predicted),
        "num_ground_truth": len(ground_truth),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "iou_threshold": float(iou_threshold),
        "matches": match_result["matches"],
        "unmatched_predicted": match_result["unmatched_predicted"],
        "unmatched_ground_truth": match_result["unmatched_ground_truth"],
    }
    return result


def evaluate_micro_segment_rows(
    predicted_segments: list[dict[str, Any]],
    ground_truth_segments: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    predicted = _normalize_micro(predicted_segments)
    ground_truth = _normalize_micro(ground_truth_segments)
    match_result = match_predicted_to_ground_truth(predicted, ground_truth, iou_threshold=iou_threshold)
    matches = match_result["matches"]
    pred_by_id = {item["micro_segment_id"]: item for item in predicted}
    gt_by_id = {item["micro_segment_id"]: item for item in ground_truth}
    object_hits = 0
    object_family_hits = 0
    object_presence_hits = 0
    interaction_hits = 0
    action_hits = 0
    detailed_matches = []
    for match in matches:
        pred_id = match.get("predicted_micro_segment_id") or match.get("predicted_segment_id")
        gt_id = match.get("ground_truth_micro_segment_id") or match.get("ground_truth_segment_id")
        pred = pred_by_id.get(str(pred_id)) or {}
        gt = gt_by_id.get(str(gt_id)) or {}
        object_ok = bool(pred.get("primary_object")) and pred.get("primary_object") == gt.get("primary_object")
        object_family_ok = bool(match.get("primary_object_family_match"))
        object_presence_ok = bool(match.get("primary_object_detected_match"))
        interaction_ok = bool(pred.get("interaction_type")) and pred.get("interaction_type") == gt.get("interaction_type")
        action_ok = bool(pred.get("action_type")) and pred.get("action_type") == gt.get("action_type")
        object_hits += int(object_ok)
        object_family_hits += int(object_family_ok)
        object_presence_hits += int(object_presence_ok)
        interaction_hits += int(interaction_ok)
        action_hits += int(action_ok)
        detailed_matches.append(
            {
                **match,
                "primary_object_match": object_ok,
                "primary_object_family_match": object_family_ok,
                "primary_object_detected_match": object_presence_ok,
                "interaction_type_match": interaction_ok,
                "action_type_match": action_ok,
            }
        )
    tp = len(matches)
    fp = max(0, len(predicted) - tp)
    fn = max(0, len(ground_truth) - tp)
    precision = float(tp / len(predicted)) if predicted else 0.0
    recall = float(tp / len(ground_truth)) if ground_truth else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    mean_iou = float(sum(item["iou"] for item in matches) / tp) if tp else 0.0
    return {
        "num_predicted": len(predicted),
        "num_ground_truth": len(ground_truth),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "primary_object_accuracy": float(object_hits / tp) if tp else 0.0,
        "object_family_accuracy": float(object_family_hits / tp) if tp else 0.0,
        "object_presence_accuracy": float(object_presence_hits / tp) if tp else 0.0,
        "interaction_type_accuracy": float(interaction_hits / tp) if tp else 0.0,
        "action_type_accuracy": float(action_hits / tp) if tp else 0.0,
        "iou_threshold": float(iou_threshold),
        "matches": detailed_matches,
        "unmatched_predicted": match_result["unmatched_predicted"],
        "unmatched_ground_truth": match_result["unmatched_ground_truth"],
    }


def compute_micro_quality_stats(
    micro_segments_path: str | Path,
    output_path: str | Path | None = None,
    parent_segments_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(micro_segments_path) if Path(micro_segments_path).exists() else []
    confidence_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    object_counts: dict[str, int] = {}
    object_family_counts: dict[str, int] = {}
    durations: list[float] = []
    missing_peak_keyframe = 0
    missing_any_keyframe = 0
    dialogue_context_available = 0
    manual_corrected = 0
    evidence_counts: dict[str, int] = {}
    sorted_rows = sorted(rows, key=lambda item: (str(item.get("parent_segment_id") or ""), float(item.get("start_sec", 0.0) or 0.0)))
    same_object_adjacent = 0
    for previous, current in zip(sorted_rows, sorted_rows[1:]):
        prev_interaction = previous.get("interaction") if isinstance(previous.get("interaction"), dict) else {}
        curr_interaction = current.get("interaction") if isinstance(current.get("interaction"), dict) else {}
        if previous.get("parent_segment_id") == current.get("parent_segment_id") and (
            prev_interaction.get("primary_object") or previous.get("primary_object")
        ) == (curr_interaction.get("primary_object") or current.get("primary_object")):
            same_object_adjacent += 1

    for row in rows:
        quality = row.get("quality") if isinstance(row.get("quality"), dict) else {}
        interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
        keyframes = row.get("keyframes") if isinstance(row.get("keyframes"), dict) else {}
        confidence = str(quality.get("confidence") or "unknown")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        for warning in quality.get("warnings") or []:
            key = str(warning)
            warning_counts[key] = warning_counts.get(key, 0) + 1
        primary_object = str(interaction.get("primary_object") or row.get("primary_object") or "unknown")
        object_counts[primary_object] = object_counts.get(primary_object, 0) + 1
        primary_family = str(interaction.get("primary_object_family") or row.get("primary_object_family") or object_family(primary_object) or "unknown")
        object_family_counts[primary_family] = object_family_counts.get(primary_family, 0) + 1
        durations.append(float(row.get("duration_sec", max(0.0, float(row.get("end_sec", 0.0)) - float(row.get("start_sec", 0.0))) or 0.0)))
        if not keyframes.get("peak_frame"):
            missing_peak_keyframe += 1
        if not any(keyframes.get(name) for name in ("contact_frame", "peak_frame", "release_frame")):
            missing_any_keyframe += 1
        if row.get("dialogue_context_available") or row.get("dialogue_context"):
            dialogue_context_available += 1
        if row.get("manual_corrected"):
            manual_corrected += 1
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        evidence_level = str(row.get("evidence_level") or evidence.get("evidence_level") or "unknown")
        evidence_counts[evidence_level] = evidence_counts.get(evidence_level, 0) + 1

    total = len(rows)
    duration_sum = sum(durations)
    low_confidence_count = confidence_counts.get("low", 0)
    single_frame_count = warning_counts.get("only_single_frame_evidence", 0)
    parent_duration_sec = 0.0
    parent_segment_count = 0
    parent_with_micro_count = 0
    parent_without_micro_segment_ids: list[str] = []
    parent_path = Path(parent_segments_path) if parent_segments_path is not None else Path(micro_segments_path).with_name("key_action_segments.jsonl")
    if parent_path.exists():
        parent_ids = {str(row.get("parent_segment_id") or "") for row in rows if row.get("parent_segment_id")}
        parents = read_jsonl(parent_path)
        parent_segment_count = len(parents)
        parent_with_micro_count = len(parent_ids)
        for parent in parents:
            parent_id = str(parent.get("segment_id") or parent.get("parent_segment_id") or "")
            if parent_id in parent_ids:
                parent_duration_sec += float(parent.get("duration_sec", 0.0) or 0.0)
            elif parent_id:
                parent_without_micro_segment_ids.append(parent_id)
    if parent_duration_sec <= 0.0 and rows:
        parent_start = min(float(row.get("start_sec", 0.0) or 0.0) for row in rows)
        parent_end = max(float(row.get("end_sec", row.get("start_sec", 0.0)) or 0.0) for row in rows)
        parent_duration_sec = max(0.0, parent_end - parent_start)
    micro_per_minute = float(total / parent_duration_sec * 60.0) if parent_duration_sec > 0 else 0.0
    avg_duration = float(duration_sum / total) if total else 0.0
    median_duration = float(statistics.median(durations)) if durations else 0.0
    low_confidence_ratio = float(low_confidence_count / total) if total else 0.0
    single_frame_evidence_ratio = float(single_frame_count / total) if total else 0.0
    warnings_list: list[str] = []
    if micro_per_minute > 20.0:
        warnings_list.append("high_micro_per_minute")
    if low_confidence_ratio > 0.4:
        warnings_list.append("many_low_confidence_micro_segments")
    if same_object_adjacent >= max(3, total // 5 if total else 3):
        warnings_list.append("adjacent_same_object_micro_segments")
    if median_duration < 0.75 and total:
        warnings_list.append("very_short_median_micro_duration")
    result = {
        "micro_segment_count": total,
        "num_micro_segments": total,
        "parent_duration_sec": parent_duration_sec,
        "micro_per_minute": micro_per_minute,
        "avg_micro_duration_sec": avg_duration,
        "median_micro_duration_sec": median_duration,
        "low_confidence_ratio": low_confidence_ratio,
        "single_frame_evidence_ratio": single_frame_evidence_ratio,
        "same_object_adjacent_micro_count": same_object_adjacent,
        "parent_segment_count": parent_segment_count,
        "parent_with_micro_count": parent_with_micro_count,
        "parent_without_micro_count": len(parent_without_micro_segment_ids),
        "parent_micro_coverage_ratio": float(parent_with_micro_count / parent_segment_count) if parent_segment_count else 0.0,
        "parent_without_micro_segment_ids": parent_without_micro_segment_ids,
        "possible_over_segmentation": bool(warnings_list),
        "warnings": warnings_list,
        "confidence_counts": confidence_counts,
        "evidence_level_counts": evidence_counts,
        "warning_counts": warning_counts,
        "primary_object_counts": object_counts,
        "primary_object_family_counts": object_family_counts,
        "duration": {
            "min": min(durations) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
            "mean": duration_sum / total if total else 0.0,
            "total": duration_sum,
        },
        "missing_peak_keyframe_count": missing_peak_keyframe,
        "missing_any_keyframe_count": missing_any_keyframe,
        "dialogue_context_available_count": dialogue_context_available,
        "manual_corrected_count": manual_corrected,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_micro_segments(
    predicted_segments_path: str | Path,
    manual_segments_path: str | Path,
    iou_threshold: float = 0.3,
    output_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
) -> dict[str, Any]:
    predicted = read_jsonl(predicted_segments_path) if Path(predicted_segments_path).exists() else []
    ground_truth = load_manual_micro_segments(manual_segments_path)
    eval_config = load_micro_eval_config(eval_config_path) if eval_config_path else None
    result = evaluate_micro_segment_rows(predicted, ground_truth, iou_threshold=iou_threshold, eval_config=eval_config)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_micro_segment_rows(
    predicted_micro_segments: list[dict[str, Any]],
    ground_truth_micro_segments: list[dict[str, Any]],
    iou_threshold: float = 0.3,
    eval_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    predicted = _normalize_micro(predicted_micro_segments, default_prefix="pred_micro")
    ground_truth = _normalize_micro(ground_truth_micro_segments, default_prefix="gt_micro")
    coverage = micro_eval_coverage(predicted, ground_truth, eval_config)
    predicted_for_eval = coverage["predictions_inside_labeled_window_rows"]
    ground_truth_for_eval = coverage["ground_truth_inside_labeled_window_rows"]
    match_result = match_micro_predicted_to_ground_truth(predicted, ground_truth, iou_threshold=iou_threshold)
    if eval_config:
        match_result = match_micro_predicted_to_ground_truth(predicted_for_eval, ground_truth_for_eval, iou_threshold=iou_threshold)
    tp = len(match_result["matches"])
    precision = float(tp / len(predicted_for_eval)) if predicted_for_eval else 0.0
    recall = float(tp / len(ground_truth_for_eval)) if ground_truth_for_eval else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    mean_iou = float(sum(item["iou"] for item in match_result["matches"]) / tp) if tp else 0.0
    object_eval_matches = [item for item in match_result["matches"] if item.get("ground_truth_primary_object")]
    interaction_eval_matches = [item for item in match_result["matches"] if item.get("ground_truth_interaction_type")]
    action_eval_matches = [item for item in match_result["matches"] if item.get("ground_truth_action_type")]
    primary_object_accuracy = (
        float(sum(1 for item in object_eval_matches if item["primary_object_match"]) / len(object_eval_matches))
        if object_eval_matches
        else 0.0
    )
    object_family_accuracy = (
        float(sum(1 for item in object_eval_matches if item.get("primary_object_family_match")) / len(object_eval_matches))
        if object_eval_matches
        else 0.0
    )
    object_presence_accuracy = (
        float(sum(1 for item in object_eval_matches if item.get("primary_object_detected_match")) / len(object_eval_matches))
        if object_eval_matches
        else 0.0
    )
    confusion: dict[str, dict[str, int]] = {}
    for item in object_eval_matches:
        gt_obj = str(item.get("ground_truth_primary_object") or "unknown")
        pred_obj = str(item.get("predicted_primary_object") or "unknown")
        confusion.setdefault(gt_obj, {})[pred_obj] = confusion.setdefault(gt_obj, {}).get(pred_obj, 0) + 1
    action_type_accuracy = (
        float(sum(1 for item in action_eval_matches if item["action_type_match"]) / len(action_eval_matches))
        if action_eval_matches
        else 0.0
    )
    interaction_type_accuracy = (
        float(sum(1 for item in interaction_eval_matches if item["interaction_type_match"]) / len(interaction_eval_matches))
        if interaction_eval_matches
        else 0.0
    )
    return {
        "num_predicted": len(predicted_for_eval),
        "num_ground_truth": len(ground_truth_for_eval),
        "num_predicted_total": len(predicted),
        "num_ground_truth_total": len(ground_truth),
        "true_positive": tp,
        "false_positive": max(0, len(predicted_for_eval) - tp),
        "false_negative": max(0, len(ground_truth_for_eval) - tp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "primary_object_accuracy": primary_object_accuracy,
        "object_family_accuracy": object_family_accuracy,
        "object_presence_accuracy": object_presence_accuracy,
        "interaction_type_accuracy": interaction_type_accuracy,
        "action_type_accuracy": action_type_accuracy,
        "object_confusion_matrix": confusion,
        "object_aware_matching": True,
        "iou_threshold": float(iou_threshold),
        "matches": match_result["matches"],
        "unmatched_predicted": match_result["unmatched_predicted"],
        "unmatched_ground_truth": match_result["unmatched_ground_truth"],
        **{key: value for key, value in coverage.items() if not key.endswith("_rows")},
        "metric_mode": "formal" if coverage.get("precision_is_formal") else "debugging",
        "formal_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "primary_object_accuracy": primary_object_accuracy,
            "object_family_accuracy": object_family_accuracy,
            "object_presence_accuracy": object_presence_accuracy,
            "interaction_type_accuracy": interaction_type_accuracy,
            "action_type_accuracy": action_type_accuracy,
        }
        if coverage.get("precision_is_formal")
        else None,
        "debugging_metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "primary_object_accuracy": primary_object_accuracy,
            "object_family_accuracy": object_family_accuracy,
            "object_presence_accuracy": object_presence_accuracy,
            "interaction_type_accuracy": interaction_type_accuracy,
            "action_type_accuracy": action_type_accuracy,
        }
        if not coverage.get("precision_is_formal")
        else None,
        "note": None
        if coverage.get("precision_is_formal")
        else "GT coverage is partial or unknown; precision is for debugging only.",
    }


def load_micro_eval_config(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8-sig"))


def _window_overlap(row: dict[str, Any], window: dict[str, Any]) -> float:
    start = float(row.get("start_sec", 0.0) or 0.0)
    end = float(row.get("end_sec", start) or start)
    w_start = float(window.get("start_sec", 0.0) or 0.0)
    w_end = float(window.get("end_sec", w_start) or w_start)
    return max(0.0, min(end, w_end) - max(start, w_start))


def _inside_windows(row: dict[str, Any], windows: list[dict[str, Any]]) -> bool:
    if not windows:
        return True
    return any(_window_overlap(row, window) > 0 for window in windows)


def _labeled_duration(windows: list[dict[str, Any]]) -> float:
    if not windows:
        return 0.0
    intervals = sorted(
        (float(window.get("start_sec", 0.0) or 0.0), float(window.get("end_sec", 0.0) or 0.0))
        for window in windows
    )
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def micro_eval_coverage(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    eval_config: dict[str, Any] | None,
) -> dict[str, Any]:
    windows = list((eval_config or {}).get("labeled_windows") or [])
    gt_completeness = str((eval_config or {}).get("gt_completeness") or "unknown")
    if windows:
        predicted_inside = [row for row in predicted if _inside_windows(row, windows)]
        predicted_outside = [row for row in predicted if not _inside_windows(row, windows)]
        gt_inside = [row for row in ground_truth if _inside_windows(row, windows)]
    else:
        predicted_inside = list(predicted)
        predicted_outside = []
        gt_inside = list(ground_truth)
    precision_is_formal = bool(windows and gt_completeness == "complete")
    return {
        "gt_completeness": gt_completeness,
        "labeled_duration_sec": _labeled_duration(windows),
        "labeled_window_count": len(windows),
        "predictions_inside_labeled_windows": len(predicted_inside),
        "predictions_outside_labeled_windows": len(predicted_outside),
        "precision_is_formal": precision_is_formal,
        "precision_note": "formal within complete labeled windows" if precision_is_formal else "GT coverage unknown or partial; precision is for debugging only.",
        "predictions_inside_labeled_window_rows": predicted_inside,
        "predictions_outside_labeled_window_rows": predicted_outside,
        "ground_truth_inside_labeled_window_rows": gt_inside,
    }


def _read_json_if_exists(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def _write_json_if_requested(output_path: str | Path | None, data: Mapping[str, Any]) -> None:
    if output_path is None:
        return
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return float(2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _row_start(row: Mapping[str, Any]) -> float | None:
    for key in ("start_sec", "local_start_sec"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _row_end(row: Mapping[str, Any]) -> float | None:
    for key in ("end_sec", "local_end_sec"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def _time_overlap_sec(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_start = _row_start(left)
    left_end = _row_end(left)
    right_start = _row_start(right)
    right_end = _row_end(right)
    if None in (left_start, left_end, right_start, right_end):
        return 0.0
    return max(0.0, min(float(left_end), float(right_end)) - max(float(left_start), float(right_start)))


def _load_jsonl_if_exists(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    return read_jsonl(source) if source.exists() else []


def _keyframe_coverage_from_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    required = ("contact_frame", "peak_frame", "release_frame")
    total = 0
    present = 0
    by_role = {role: {"required": 0, "present": 0} for role in required}
    missing: list[dict[str, Any]] = []
    for row in rows:
        row_id = row.get("micro_segment_id") or row.get("segment_id")
        keyframes = _as_mapping(row.get("keyframes"))
        if keyframes:
            for role in required:
                total += 1
                by_role[role]["required"] += 1
                if keyframes.get(role):
                    present += 1
                    by_role[role]["present"] += 1
                else:
                    missing.append({"id": row_id, "role": role})
            continue
        interaction_keyframes = _as_list(row.get("interaction_keyframes"))
        total += 1
        if interaction_keyframes:
            present += 1
        else:
            missing.append({"id": row_id, "role": "interaction_keyframe"})
    return {
        "required_keyframes": total,
        "present_keyframes": present,
        "coverage": _safe_div(present, total),
        "by_role": {
            role: {
                **counts,
                "coverage": _safe_div(float(counts["present"]), float(counts["required"])),
            }
            for role, counts in by_role.items()
        },
        "missing": missing[:100],
    }


def evaluate_key_segments_and_keyframes(
    session_dir: str | Path | None = None,
    *,
    predicted_segments_path: str | Path | None = None,
    ground_truth_segments_path: str | Path | None = None,
    key_action_segments_path: str | Path | None = None,
    micro_segments_path: str | Path | None = None,
    iou_threshold: float = 0.3,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate key segment extraction plus keyframe coverage.

    When GT is unavailable, temporal precision/recall are explicitly marked as
    coverage-only so dry-run sessions can still produce a QA artifact.
    """

    session = Path(session_dir) if session_dir is not None else None
    if session is not None:
        predicted_segments_path = predicted_segments_path or session / "cv_outputs" / "detected_segments.jsonl"
        key_action_segments_path = key_action_segments_path or session / "metadata" / "key_action_segments.jsonl"
        micro_segments_path = micro_segments_path or session / "metadata" / "micro_segments.jsonl"
        candidate_gt = session / "evaluation" / "manual_segments.jsonl"
        ground_truth_segments_path = ground_truth_segments_path or (candidate_gt if candidate_gt.exists() else None)
        output_path = output_path or session / "evaluation" / "segment_keyframe_eval.json"

    predicted = _normalize_predicted(_load_jsonl_if_exists(predicted_segments_path))
    ground_truth = load_manual_segments(ground_truth_segments_path) if ground_truth_segments_path and Path(ground_truth_segments_path).exists() else []
    segment_metrics = evaluate_segment_rows(predicted, ground_truth, iou_threshold=iou_threshold) if ground_truth else {
        "num_predicted": len(predicted),
        "num_ground_truth": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mean_iou": 0.0,
        "iou_threshold": float(iou_threshold),
        "matches": [],
        "unmatched_predicted": [row.get("segment_id") for row in predicted],
        "unmatched_ground_truth": [],
        "metric_mode": "coverage_only_no_ground_truth",
    }
    boundary_errors = []
    gt_by_id = {row.get("segment_id"): row for row in ground_truth}
    pred_by_id = {row.get("segment_id"): row for row in predicted}
    for match in segment_metrics.get("matches") or []:
        pred = pred_by_id.get(match.get("predicted_segment_id")) or {}
        gt = gt_by_id.get(match.get("ground_truth_segment_id")) or {}
        if pred and gt:
            boundary_errors.append(
                {
                    "predicted_segment_id": pred.get("segment_id"),
                    "ground_truth_segment_id": gt.get("segment_id"),
                    "start_error_sec": abs(float(pred.get("start_sec", 0.0)) - float(gt.get("start_sec", 0.0))),
                    "end_error_sec": abs(float(pred.get("end_sec", 0.0)) - float(gt.get("end_sec", 0.0))),
                    "iou": match.get("iou"),
                }
            )
    key_action_rows = _load_jsonl_if_exists(key_action_segments_path)
    micro_rows = _load_jsonl_if_exists(micro_segments_path)
    keyframe_rows = micro_rows if micro_rows else key_action_rows
    keyframe_coverage = _keyframe_coverage_from_rows(keyframe_rows)
    result = {
        "schema_version": "key_action_segment_keyframe_eval.v1",
        "segment_metrics": segment_metrics,
        "keyframe_coverage": keyframe_coverage,
        "boundary_error": {
            "count": len(boundary_errors),
            "mean_start_error_sec": statistics.mean([item["start_error_sec"] for item in boundary_errors]) if boundary_errors else 0.0,
            "mean_end_error_sec": statistics.mean([item["end_error_sec"] for item in boundary_errors]) if boundary_errors else 0.0,
            "max_boundary_error_sec": max(
                [max(item["start_error_sec"], item["end_error_sec"]) for item in boundary_errors],
                default=0.0,
            ),
            "examples": boundary_errors[:50],
        },
        "summary_score": round(
            (float(segment_metrics.get("f1") or 0.0) if ground_truth else 0.5) * 0.6
            + float(keyframe_coverage.get("coverage") or 0.0) * 0.4,
            4,
        ),
    }
    _write_json_if_requested(output_path, result)
    return result


evaluate_segment_keyframes = evaluate_key_segments_and_keyframes


def _event_labels(row: Mapping[str, Any], *, include_actions: bool = True, include_states: bool = True) -> list[str]:
    labels: list[str] = []
    event_type = str(row.get("event_type") or "")
    action_type = str(row.get("action_type") or "")
    if include_actions and action_type:
        labels.append(f"action:{action_type}")
    if include_actions and event_type == "experiment_action_classification" and action_type:
        labels.append(f"action_event:{action_type}")
    if include_states:
        state_values = [event_type, *[str(item) for item in _as_list(row.get("state_change_types"))]]
        for value in state_values:
            if not value:
                continue
            lowered = value.lower()
            if any(token in lowered for token in ("state", "liquid", "equipment", "container", "movement", "track", "contact")):
                labels.append(f"state:{value}")
    seen: set[str] = set()
    output: list[str] = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            output.append(label)
    return output


def _match_labeled_events(
    predicted: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    min_overlap_sec: float = 0.0,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    candidates: list[tuple[float, int, int, str]] = []
    for pred_idx, pred in enumerate(predicted):
        pred_labels = set(_event_labels(pred))
        if not pred_labels:
            continue
        for gt_idx, gt in enumerate(ground_truth):
            labels = pred_labels & set(_event_labels(gt))
            if not labels:
                continue
            overlap = _time_overlap_sec(pred, gt)
            has_time = _row_start(pred) is not None and _row_start(gt) is not None
            if has_time and overlap <= min_overlap_sec:
                continue
            candidates.append((overlap if has_time else 1.0, pred_idx, gt_idx, sorted(labels)[0]))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    for overlap, pred_idx, gt_idx, label in sorted(candidates, key=lambda item: item[0], reverse=True):
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append(
            {
                "predicted_event_id": predicted[pred_idx].get("video_event_id") or predicted[pred_idx].get("event_id"),
                "ground_truth_event_id": ground_truth[gt_idx].get("video_event_id") or ground_truth[gt_idx].get("event_id"),
                "label": label,
                "overlap_sec": overlap,
            }
        )
    return matches, used_pred, used_gt


def evaluate_action_state_recognition(
    predicted_events_path: str | Path,
    ground_truth_events_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    predicted = _load_jsonl_if_exists(predicted_events_path)
    ground_truth = _load_jsonl_if_exists(ground_truth_events_path)
    pred_label_counts = Counter(label for row in predicted for label in _event_labels(row))
    gt_label_counts = Counter(label for row in ground_truth for label in _event_labels(row))
    matches, used_pred, used_gt = _match_labeled_events(predicted, ground_truth) if ground_truth else ([], set(), set())
    match_counts = Counter(match["label"] for match in matches)
    labels = sorted(set(pred_label_counts) | set(gt_label_counts))
    per_class = {}
    for label in labels:
        tp = int(match_counts.get(label, 0))
        fp = max(0, int(pred_label_counts.get(label, 0)) - tp)
        fn = max(0, int(gt_label_counts.get(label, 0)) - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        per_class[label] = {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
        }
    macro_f1 = statistics.mean([item["f1"] for item in per_class.values()]) if per_class else 0.0
    result = {
        "schema_version": "key_action_action_state_eval.v1",
        "num_predicted": len(predicted),
        "num_ground_truth": len(ground_truth),
        "metric_mode": "formal" if ground_truth else "coverage_only_no_ground_truth",
        "matches": matches,
        "unmatched_predicted": [
            row.get("video_event_id") or row.get("event_id") for idx, row in enumerate(predicted) if idx not in used_pred
        ],
        "unmatched_ground_truth": [
            row.get("video_event_id") or row.get("event_id") for idx, row in enumerate(ground_truth) if idx not in used_gt
        ],
        "per_class": per_class,
        "macro_f1": macro_f1,
        "action_label_counts": {
            "predicted": {key: value for key, value in sorted(pred_label_counts.items()) if key.startswith("action")},
            "ground_truth": {key: value for key, value in sorted(gt_label_counts.items()) if key.startswith("action")},
        },
        "state_label_counts": {
            "predicted": {key: value for key, value in sorted(pred_label_counts.items()) if key.startswith("state")},
            "ground_truth": {key: value for key, value in sorted(gt_label_counts.items()) if key.startswith("state")},
        },
        "summary_score": round(macro_f1 if ground_truth else min(1.0, len(predicted) / 10.0), 4),
    }
    _write_json_if_requested(output_path, result)
    return result


def _process_steps(process: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = process.get("steps")
    return [dict(step) for step in steps if isinstance(step, Mapping)] if isinstance(steps, list) else []


def _step_key(step: Mapping[str, Any]) -> str:
    return str(step.get("step_id") or step.get("expected_action") or step.get("name") or "")


def _match_steps(system_steps: list[dict[str, Any]], gt_steps: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_id = {_step_key(step): step for step in system_steps if _step_key(step)}
    matched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[int] = set()
    for gt in gt_steps:
        key = _step_key(gt)
        if key in by_id:
            sys = by_id[key]
            matched.append((sys, gt))
            for idx, row in enumerate(system_steps):
                if row is sys:
                    used.add(idx)
                    break
            continue
        expected = str(gt.get("expected_action") or "")
        for idx, sys in enumerate(system_steps):
            if idx in used:
                continue
            if expected and expected == str(sys.get("expected_action") or ""):
                matched.append((sys, gt))
                used.add(idx)
                break
    return matched


def evaluate_step_reasoning(
    system_process_path: str | Path,
    ground_truth_process_path: str | Path | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    system = _read_json_if_exists(system_process_path)
    ground_truth = _read_json_if_exists(ground_truth_process_path) if ground_truth_process_path else {}
    system_steps = _process_steps(system)
    gt_steps = _process_steps(ground_truth)
    matches = _match_steps(system_steps, gt_steps) if gt_steps else []
    status_hits = 0
    observed_hits = 0
    inferred_hits = 0
    completed_hits = 0
    match_rows = []
    for sys, gt in matches:
        status_hit = str(sys.get("status") or "") == str(gt.get("status") or "")
        observed_hit = bool(sys.get("observed")) == bool(gt.get("observed"))
        inferred_hit = bool(sys.get("inferred")) == bool(gt.get("inferred"))
        completed_hit = bool(sys.get("completed")) == bool(gt.get("completed"))
        status_hits += int(status_hit)
        observed_hits += int(observed_hit)
        inferred_hits += int(inferred_hit)
        completed_hits += int(completed_hit)
        match_rows.append(
            {
                "system_step_id": sys.get("step_id"),
                "ground_truth_step_id": gt.get("step_id"),
                "status_match": status_hit,
                "observed_match": observed_hit,
                "inferred_match": inferred_hit,
                "completed_match": completed_hit,
            }
        )
    total = len(matches)
    next_step_accuracy = 0.0
    if ground_truth:
        next_step_accuracy = 1.0 if system.get("next_step_id") == ground_truth.get("next_step_id") else 0.0
    inferred_steps = [step for step in system_steps if step.get("inferred")]
    inference_quality = statistics.mean([float(step.get("confidence") or 0.0) for step in inferred_steps]) if inferred_steps else 1.0
    result = {
        "schema_version": "key_action_step_reasoning_eval.v1",
        "metric_mode": "formal" if gt_steps else "coverage_only_no_ground_truth",
        "system_step_count": len(system_steps),
        "ground_truth_step_count": len(gt_steps),
        "matched_step_count": total,
        "step_accuracy": _safe_div(status_hits + completed_hits + observed_hits, total * 3),
        "status_accuracy": _safe_div(status_hits, total),
        "observed_accuracy": _safe_div(observed_hits, total),
        "completed_accuracy": _safe_div(completed_hits, total),
        "inferred_accuracy": _safe_div(inferred_hits, total),
        "next_step_accuracy": next_step_accuracy,
        "inference_quality": inference_quality,
        "matches": match_rows,
        "summary_score": round(
            (_safe_div(status_hits + completed_hits + observed_hits, total * 3) if total else min(1.0, len(system_steps) / 5.0)) * 0.65
            + next_step_accuracy * 0.2
            + inference_quality * 0.15,
            4,
        ),
    }
    _write_json_if_requested(output_path, result)
    return result


def evaluate_evidence_chain(
    process_path: str | Path,
    *,
    assets_path: str | Path | None = None,
    timeline_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    process = _read_json_if_exists(process_path)
    session = Path(process_path).parent.parent if Path(process_path).parent.name == "metadata" else Path(process_path).parent
    assets = _load_jsonl_if_exists(assets_path or session / "metadata" / "material_asset_catalog.jsonl")
    timeline = _load_jsonl_if_exists(timeline_path or session / "metadata" / "unified_multimodal_timeline.jsonl")
    asset_ids = {str(row.get("asset_id")) for row in assets if row.get("asset_id")}
    timeline_ids = {str(row.get("timeline_event_id") or row.get("event_id") or "") for row in timeline}
    steps = _process_steps(process)
    step_rows = []
    complete = 0
    visual_or_inference = 0
    for step in steps:
        refs = [ref for ref in step.get("evidence_refs") or [] if isinstance(ref, Mapping)]
        has_asset = any(str(ref.get("asset_id") or "") in asset_ids or ref.get("type") == "asset" for ref in refs)
        has_video = any(ref.get("type") == "video_event" or ref.get("video_event_id") for ref in refs)
        has_state = any(ref.get("type") == "state_change" or ref.get("state_change_id") for ref in refs)
        has_timeline = any(str(ref.get("timeline_event_id") or "") in timeline_ids for ref in refs)
        has_inference = bool(step.get("inferred") and (step.get("history_basis") or step.get("confidence_reasons")))
        is_complete = bool(refs and (has_asset or has_video or has_state or has_timeline or has_inference))
        complete += int(is_complete)
        visual_or_inference += int(has_asset or has_video or has_inference)
        step_rows.append(
            {
                "step_id": step.get("step_id"),
                "evidence_ref_count": len(refs),
                "has_visual_evidence": bool(has_asset or has_video),
                "has_state_evidence": bool(has_state),
                "has_timeline_evidence": bool(has_timeline),
                "has_inference_evidence": bool(has_inference),
                "complete": is_complete,
            }
        )
    result = {
        "schema_version": "key_action_evidence_chain_eval.v1",
        "step_count": len(steps),
        "complete_step_count": complete,
        "visual_or_inference_step_count": visual_or_inference,
        "evidence_completeness_score": _safe_div(complete, len(steps)),
        "visual_or_inference_score": _safe_div(visual_or_inference, len(steps)),
        "reverse_index_entries": len(_as_mapping(process.get("evidence_index"))),
        "missing_steps": [row for row in step_rows if not row["complete"]],
        "steps": step_rows,
        "summary_score": round(_safe_div(complete, len(steps)), 4),
    }
    _write_json_if_requested(output_path, result)
    return result


def build_pipeline_evaluation_report(
    session_dir: str | Path,
    *,
    ground_truth_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    session = Path(session_dir)
    evaluation_dir = session / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    gt_root = Path(ground_truth_dir) if ground_truth_dir is not None else evaluation_dir
    target = Path(output_path) if output_path is not None else evaluation_dir / "pipeline_evaluation_report.json"

    segment_eval = evaluate_key_segments_and_keyframes(
        session,
        ground_truth_segments_path=gt_root / "manual_segments.jsonl" if (gt_root / "manual_segments.jsonl").exists() else None,
        iou_threshold=iou_threshold,
        output_path=evaluation_dir / "segment_keyframe_eval.json",
    )
    event_eval = evaluate_action_state_recognition(
        session / "metadata" / "video_understanding.jsonl",
        gt_root / "manual_events.jsonl" if (gt_root / "manual_events.jsonl").exists() else None,
        output_path=evaluation_dir / "action_state_eval.json",
    )
    process_gt = gt_root / "manual_process.json"
    if not process_gt.exists():
        process_gt = gt_root / "ground_truth_process.json"
    step_eval = evaluate_step_reasoning(
        session / "metadata" / "experiment_process.json",
        process_gt if process_gt.exists() else None,
        output_path=evaluation_dir / "step_reasoning_eval.json",
    )
    evidence_eval = evaluate_evidence_chain(
        session / "metadata" / "experiment_process.json",
        output_path=evaluation_dir / "evidence_chain_eval.json",
    )
    try:
        from .artifact_schema import validate_session_artifacts

        schema_validation = validate_session_artifacts(session, output_path=evaluation_dir / "artifact_validation_report.json")
    except Exception as exc:
        schema_validation = {"valid": False, "error": str(exc)}
    scores = {
        "segments": float(segment_eval.get("summary_score") or 0.0),
        "actions_and_states": float(event_eval.get("summary_score") or 0.0),
        "step_reasoning": float(step_eval.get("summary_score") or 0.0),
        "evidence_chain": float(evidence_eval.get("summary_score") or 0.0),
        "json_schema": 1.0 if schema_validation.get("valid") else 0.0,
    }
    report = {
        "schema_version": "key_action_pipeline_evaluation.v1",
        "session_dir": str(session),
        "ground_truth_dir": str(gt_root),
        "scores": scores,
        "overall_score": round(statistics.mean(scores.values()) if scores else 0.0, 4),
        "segment_keyframe_eval": segment_eval,
        "action_state_eval": event_eval,
        "step_reasoning_eval": step_eval,
        "evidence_chain_eval": evidence_eval,
        "schema_validation": schema_validation,
    }
    _write_json_if_requested(target, report)
    return report
