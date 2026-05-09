from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from labsopguard.datasets.action_dataset import ActionDataset, ActionEventAnnotation, load_action_dataset, validate_action_dataset


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def load_predicted_events(path: str | Path) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return list(payload.get("events") or payload.get("physical_events") or [])


def evaluate_event_predictions(
    predictions: List[Dict[str, Any]],
    annotations: List[ActionEventAnnotation],
    *,
    iou_threshold: float = 0.35,
) -> Dict[str, Any]:
    matched_pred: set[int] = set()
    matches = []
    for ann in annotations:
        best_idx: Optional[int] = None
        best_iou = 0.0
        for idx, pred in enumerate(predictions):
            if idx in matched_pred:
                continue
            if pred.get("event_type") != ann.event_type:
                continue
            score = temporal_iou(
                float(pred.get("start_time_sec") or pred.get("time_start") or 0.0),
                float(pred.get("end_time_sec") or pred.get("time_end") or 0.0),
                ann.start_time_sec,
                ann.end_time_sec,
            )
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_pred.add(best_idx)
            matches.append({"annotation_event_id": ann.event_id, "predicted_event_id": predictions[best_idx].get("event_id"), "event_type": ann.event_type, "temporal_iou": round(best_iou, 4)})
    tp = len(matches)
    fp = max(0, len(predictions) - len(matched_pred))
    fn = max(0, len(annotations) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "schema_version": "event_regression_metrics.v1",
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matches": matches,
    }


def evaluate_dataset_outputs(dataset_path: str | Path, outputs_root: str | Path, *, iou_threshold: float = 0.35) -> Dict[str, Any]:
    dataset = load_action_dataset(dataset_path)
    validation = validate_action_dataset(dataset, require_video_files=False, root=Path(dataset_path).parent)
    per_video = []
    totals = {"true_positive": 0, "false_positive": 0, "false_negative": 0}
    for record in dataset.records:
        candidates = [
            Path(outputs_root) / record.video_id / "physical_events.json",
            Path(outputs_root) / (record.experiment_id or record.video_id) / "physical_events.json",
        ]
        pred_path = next((path for path in candidates if path.exists()), None)
        predictions = load_predicted_events(pred_path) if pred_path else []
        metrics = evaluate_event_predictions(predictions, record.annotations, iou_threshold=iou_threshold)
        for key in totals:
            totals[key] += int(metrics[key])
        per_video.append({"video_id": record.video_id, "experiment_id": record.experiment_id, "prediction_path": str(pred_path) if pred_path else None, **metrics})
    precision = totals["true_positive"] / (totals["true_positive"] + totals["false_positive"]) if totals["true_positive"] + totals["false_positive"] else 0.0
    recall = totals["true_positive"] / (totals["true_positive"] + totals["false_negative"]) if totals["true_positive"] + totals["false_negative"] else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "schema_version": "event_regression_report.v1",
        "dataset": validation,
        "iou_threshold": iou_threshold,
        "summary": {**totals, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)},
        "videos": per_video,
    }
