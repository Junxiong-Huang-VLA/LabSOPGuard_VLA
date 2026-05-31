from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .action_detector import build_segments_from_scores, detect_key_action_segments
from .config import DetectorConfig
from .evaluation import evaluate_micro_segment_rows, evaluate_segment_rows, load_manual_micro_segments, load_manual_segments
from .schemas import FrameScore, SessionManifest, read_jsonl, to_json_dict
from .schemas import write_jsonl


TUNABLE_FIELDS = (
    "start_threshold",
    "end_threshold",
    "merge_gap_sec",
    "min_segment_duration_sec",
)


def parse_float_list(value: str | float | int | list[float] | tuple[float, ...] | None, default: float) -> list[float]:
    if value is None:
        return [float(default)]
    if isinstance(value, (float, int)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        parsed = [float(item) for item in value]
    else:
        parsed = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed:
        raise ValueError("parameter list must contain at least one numeric value")
    return parsed


def _frame_score_from_row(row: dict[str, Any]) -> FrameScore:
    time_sec = float(row.get("time_sec", row.get("local_time_sec", 0.0)))
    return FrameScore(
        time_sec=time_sec,
        frame_index=int(row.get("frame_index", 0)),
        local_time_sec=float(row["local_time_sec"]) if row.get("local_time_sec") is not None else time_sec,
        global_time=str(row["global_time"]) if row.get("global_time") is not None else None,
        motion_score=float(row.get("motion_score", 0.0)),
        active_score=float(row.get("active_score", row.get("motion_score", 0.0))),
        roi=row.get("roi"),
        is_active=bool(row.get("is_active", False)),
    )


def _load_frame_scores(path: str | Path) -> list[FrameScore]:
    return [_frame_score_from_row(row) for row in read_jsonl(path)]


def _detector_config_with_overrides(base_config: DetectorConfig, overrides: dict[str, float]) -> DetectorConfig:
    values = asdict(base_config)
    values.update(overrides)
    return DetectorConfig.from_dict(values)


def _segment_rows(segments: list[Any]) -> list[dict[str, Any]]:
    return [to_json_dict(segment) for segment in segments]


def _total_duration(segments: list[Any]) -> float:
    return float(sum(float(getattr(segment, "duration_sec", float(segment.end_sec) - float(segment.start_sec))) for segment in segments))


def _best_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            float(item["f1"]),
            float(item["mean_iou"]),
            float(item["recall"]),
            float(item["precision"]),
            -int(item["segment_count"]),
            -float(item["total_duration"]),
        ),
    )


def _best_micro_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            float(item["f1"]),
            float(item["mean_iou"]),
            float(item.get("primary_object_accuracy", 0.0)),
            float(item["recall"]),
            float(item["precision"]),
            -int(item["micro_segment_count"]),
        ),
    )


def _best_micro_precision_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            float(item["precision"]),
            float(item["f1"]),
            float(item["mean_iou"]),
            float(item.get("primary_object_accuracy", 0.0)),
            -int(item["micro_segment_count"]),
        ),
    )


def _best_micro_recall_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not results:
        return None
    return max(
        results,
        key=lambda item: (
            float(item["recall"]),
            float(item["f1"]),
            float(item["mean_iou"]),
            float(item.get("primary_object_accuracy", 0.0)),
            -int(item["micro_segment_count"]),
        ),
    )


def tune_detector(
    manifest_path: str | Path,
    ground_truth_path: str | Path,
    start_thresholds: list[float] | None = None,
    end_thresholds: list[float] | None = None,
    merge_gap_secs: list[float] | None = None,
    min_segment_duration_secs: list[float] | None = None,
    iou_threshold: float = 0.3,
    dry_run: bool = False,
    run_detection: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = SessionManifest.load(manifest_path)
    session_dir = Path(manifest.output_dir)
    frame_scores_path = session_dir / "cv_outputs" / "frame_scores.jsonl"
    target = Path(output_path) if output_path is not None else session_dir / "evaluation" / "tuning_results.json"

    ground_truth = load_manual_segments(ground_truth_path)
    if not ground_truth:
        raise ValueError(
            f"No manual labels found in ground truth file: {ground_truth_path}. "
            "Provide JSONL rows with start_sec and end_sec before running tuning."
        )

    base_config = manifest.detection_config
    grid = {
        "start_threshold": start_thresholds or [float(base_config.start_threshold)],
        "end_threshold": end_thresholds or [float(base_config.end_threshold)],
        "merge_gap_sec": merge_gap_secs or [float(base_config.merge_gap_sec)],
        "min_segment_duration_sec": min_segment_duration_secs or [float(base_config.min_segment_duration_sec)],
    }

    frame_score_source = "existing"
    if run_detection or not frame_scores_path.exists():
        frame_score_source = "detection_only"
        _, frame_scores = detect_key_action_segments(
            manifest.videos.third_person,
            roi=manifest.workbench_roi,
            config=base_config,
            dry_run=dry_run,
            frame_scores_output_path=frame_scores_path,
        )
    else:
        frame_scores = _load_frame_scores(frame_scores_path)

    if not frame_scores:
        raise ValueError(f"No frame scores available for tuning: {frame_scores_path}")

    duration_sec = float(max(score.time_sec for score in frame_scores))
    results: list[dict[str, Any]] = []
    for values in itertools.product(*(grid[field] for field in TUNABLE_FIELDS)):
        config_values = {field: float(value) for field, value in zip(TUNABLE_FIELDS, values)}
        config = _detector_config_with_overrides(base_config, config_values)
        segments = build_segments_from_scores(
            frame_scores,
            video_source=manifest.videos.third_person,
            duration_sec=duration_sec,
            config=config,
        )
        metrics = evaluate_segment_rows(_segment_rows(segments), ground_truth, iou_threshold=iou_threshold)
        results.append(
            {
                **config_values,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "mean_iou": float(metrics["mean_iou"]),
                "segment_count": len(segments),
                "total_duration": _total_duration(segments),
                "true_positive": int(metrics["true_positive"]),
                "false_positive": int(metrics["false_positive"]),
                "false_negative": int(metrics["false_negative"]),
            }
        )

    best = _best_result(results)
    output = {
        "manifest": str(manifest_path),
        "ground_truth": str(ground_truth_path),
        "frame_scores": str(frame_scores_path),
        "frame_score_source": frame_score_source,
        "iou_threshold": float(iou_threshold),
        "scanned_parameters": grid,
        "results": results,
        "best_config": best,
        "output_path": str(target),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _micro_time(row: dict[str, Any]) -> float:
    for key in ("time_sec", "local_time_sec", "timestamp_sec", "sec", "t"):
        if row.get(key) is not None:
            return float(row[key])
    return 0.0


def _micro_parent(row: dict[str, Any]) -> str:
    return str(row.get("parent_segment_id") or row.get("segment_id") or "segment")


def _micro_object(row: dict[str, Any]) -> str:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
    return str(
        row.get("primary_object")
        or row.get("object_label")
        or interaction.get("primary_object")
        or interaction.get("object_label")
        or "object"
    )


def _micro_score(row: dict[str, Any]) -> float:
    interaction = row.get("interaction") if isinstance(row.get("interaction"), dict) else {}
    value = row.get(
        "interaction_score",
        row.get(
            "score",
            interaction.get(
                "score",
                interaction.get("max_interaction_score", row.get("active_score", 0.0)),
            ),
        ),
    )
    return float(value or 0.0)


def _micro_active(row: dict[str, Any], threshold: float) -> bool:
    if row.get("hand_detected") is False:
        return False
    return _micro_score(row) >= threshold


def _expand_micro_interaction_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        interactions = row.get("hand_object_interactions")
        if isinstance(interactions, list) and interactions:
            for interaction in interactions:
                if not isinstance(interaction, dict):
                    continue
                item = dict(row)
                item["interaction"] = interaction
                item["primary_object"] = interaction.get("object_label") or interaction.get("primary_object")
                item["object_label"] = item["primary_object"]
                item["interaction_score"] = float(interaction.get("score", row.get("interaction_score", 0.0)) or 0.0)
                item["hand_detected"] = True
                expanded.append(item)
        else:
            detections = row.get("detections") if isinstance(row.get("detections"), list) else []
            item = dict(row)
            if "hand_detected" not in item:
                item["hand_detected"] = any(
                    "hand" in str(det.get("label", "")).lower()
                    for det in detections
                    if isinstance(det, dict)
                )
            expanded.append(item)
    return expanded


def _micro_segments_from_rows(
    rows: list[dict[str, Any]],
    *,
    interaction_threshold: float,
    merge_gap_sec: float,
    min_duration_sec: float,
) -> list[dict[str, Any]]:
    active_rows = [row for row in _expand_micro_interaction_rows(rows) if _micro_active(row, interaction_threshold)]
    active_rows.sort(key=lambda row: (_micro_parent(row), _micro_object(row), _micro_time(row)))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in active_rows:
        grouped.setdefault((_micro_parent(row), _micro_object(row)), []).append(row)

    predicted: list[dict[str, Any]] = []
    sequence_by_parent: dict[str, int] = {}
    for (parent_id, primary_object), group in grouped.items():
        current: list[dict[str, Any]] = []
        for row in group:
            if not current:
                current = [row]
                continue
            if _micro_time(row) - _micro_time(current[-1]) > merge_gap_sec:
                predicted.extend(_rows_to_micro(parent_id, primary_object, current, min_duration_sec, sequence_by_parent))
                current = [row]
            else:
                current.append(row)
        predicted.extend(_rows_to_micro(parent_id, primary_object, current, min_duration_sec, sequence_by_parent))
    return sorted(predicted, key=lambda row: (float(row["start_sec"]), str(row["micro_segment_id"])))


def _rows_to_micro(
    parent_id: str,
    primary_object: str,
    rows: list[dict[str, Any]],
    min_duration_sec: float,
    sequence_by_parent: dict[str, int],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    times = sorted(_micro_time(row) for row in rows)
    deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
    sample_period = min(deltas) if deltas else 0.5
    start_sec = times[0]
    end_sec = times[-1] + sample_period
    duration = max(0.0, end_sec - start_sec)
    if duration < min_duration_sec:
        return []
    sequence_by_parent[parent_id] = sequence_by_parent.get(parent_id, 0) + 1
    micro_id = f"{parent_id}_micro_sweep_{sequence_by_parent[parent_id]:03d}"
    scores = [_micro_score(row) for row in rows]
    return [
        {
            "micro_segment_id": micro_id,
            "parent_segment_id": parent_id,
            "start_sec": round(start_sec, 6),
            "end_sec": round(end_sec, 6),
            "duration_sec": round(duration, 6),
            "interaction": {
                "primary_object": primary_object,
                "interaction_type": f"hand_{primary_object}_contact",
                "max_interaction_score": max(scores) if scores else 0.0,
                "avg_interaction_score": sum(scores) / len(scores) if scores else 0.0,
            },
            "text_description": {
                "action_type": str(rows[0].get("action_type") or ""),
            },
        }
    ]


def tune_micro_thresholds(
    session_dir: str | Path,
    ground_truth_path: str | Path,
    *,
    frame_rows_path: str | Path | None = None,
    interaction_thresholds: list[float] | None = None,
    merge_gap_secs: list[float] | None = None,
    min_duration_secs: list[float] | None = None,
    iou_threshold: float = 0.3,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    rows_path = Path(frame_rows_path) if frame_rows_path else session / "cv_outputs" / "yolo_micro_frame_rows.jsonl"
    if not rows_path.exists():
        raise FileNotFoundError(f"yolo micro frame rows not found: {rows_path}")
    frame_rows = read_jsonl(rows_path)
    ground_truth = load_manual_micro_segments(ground_truth_path)
    if not ground_truth:
        raise ValueError(f"No manual micro labels found in ground truth file: {ground_truth_path}")

    thresholds = interaction_thresholds or [0.35, 0.45, 0.55]
    gaps = merge_gap_secs or [1.0, 1.5, 2.0]
    durations = min_duration_secs or [0.5, 1.0]
    target_dir = Path(output_dir) if output_dir else session / "evaluation"
    target_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = target_dir / "micro_threshold_sweep.jsonl"
    best_path = target_dir / "micro_threshold_sweep_best.json"

    results: list[dict[str, Any]] = []
    for interaction_threshold, merge_gap_sec, min_duration_sec in itertools.product(thresholds, gaps, durations):
        predicted = _micro_segments_from_rows(
            frame_rows,
            interaction_threshold=float(interaction_threshold),
            merge_gap_sec=float(merge_gap_sec),
            min_duration_sec=float(min_duration_sec),
        )
        metrics = evaluate_micro_segment_rows(predicted, ground_truth, iou_threshold=iou_threshold)
        results.append(
            {
                "interaction_threshold": float(interaction_threshold),
                "merge_gap_sec": float(merge_gap_sec),
                "min_duration_sec": float(min_duration_sec),
                "micro_segment_count": len(predicted),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "mean_iou": float(metrics["mean_iou"]),
                "primary_object_accuracy": float(metrics.get("primary_object_accuracy", 0.0)),
                "interaction_type_accuracy": float(metrics.get("interaction_type_accuracy", 0.0)),
                "action_type_accuracy": float(metrics.get("action_type_accuracy", 0.0)),
                "true_positive": int(metrics["true_positive"]),
                "false_positive": int(metrics["false_positive"]),
                "false_negative": int(metrics["false_negative"]),
            }
        )
    best_by_f1 = _best_micro_result(results)
    best_by_precision = _best_micro_precision_result(results)
    best_by_recall = _best_micro_recall_result(results)
    write_jsonl(sweep_path, results)
    best_payload = {
        "session_dir": str(session),
        "frame_rows": str(rows_path),
        "ground_truth": str(ground_truth_path),
        "iou_threshold": float(iou_threshold),
        "scanned_parameters": {
            "interaction_threshold": thresholds,
            "merge_gap_sec": gaps,
            "min_duration_sec": durations,
        },
        "best_by_f1": best_by_f1,
        "best_by_precision": best_by_precision,
        "best_by_recall": best_by_recall,
        "recommended_config": best_by_f1,
        "best_config": best_by_f1,
        "sweep_path": str(sweep_path),
    }
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**best_payload, "best_path": str(best_path), "result_count": len(results)}
