from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from dateutil.parser import isoparse
except Exception:  # pragma: no cover - stdlib fallback covers normal ISO inputs.
    isoparse = None

from .schemas import DetectedSegment, SessionManifest, TranscriptSource, TranscriptUtterance, VideoSource, write_jsonl
from .frame_time_map import capture_sec_to_video_sec


TIME_ALIGNMENT_EVAL_SCHEMA_VERSION = "key_action_time_alignment_eval.v1"


def parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value)
    if isoparse is not None:
        return isoparse(text)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def local_sec_to_global_time(video_source: VideoSource, local_sec: float) -> datetime:
    start = parse_time(video_source.start_time)
    return start + timedelta(seconds=float(video_source.offset_sec) + float(local_sec))


def global_time_to_local_sec(video_source: VideoSource, global_time: str | datetime) -> float:
    start = parse_time(video_source.start_time)
    current = parse_time(global_time)
    return (current - start).total_seconds() - float(video_source.offset_sec)


def global_time_to_video_sec(video_source: VideoSource, global_time: str | datetime) -> float:
    capture_sec = max(0.0, global_time_to_local_sec(video_source, global_time))
    return max(0.0, capture_sec_to_video_sec(video_source, capture_sec, use_frame_time_map="auto"))


def strict_common_overlap_from_view_intervals(
    view_intervals: Mapping[str, Mapping[str, Any]],
    *,
    source: str,
    requested_overlap: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the exact intersection of per-view global timeline intervals."""
    views: dict[str, dict[str, Any]] = {}
    for view_id, interval in view_intervals.items():
        if not isinstance(interval, Mapping):
            continue
        start = _first_float(interval, ("global_start_sec", "global_start"))
        end = _first_float(interval, ("global_end_sec", "global_end"))
        if start is None or end is None or end <= start:
            continue
        row = dict(interval)
        row["global_start_sec"] = round(start, 6)
        row["global_end_sec"] = round(end, 6)
        row["global_start"] = round(start, 6)
        row["global_end"] = round(end, 6)
        row["duration_sec"] = round(end - start, 6)
        views[str(view_id)] = row
    if len(views) < 2:
        return None

    overlap_start = max(float(item["global_start_sec"]) for item in views.values())
    overlap_end = min(float(item["global_end_sec"]) for item in views.values())
    if overlap_end <= overlap_start:
        return {
            "available": False,
            "source": source,
            "reason": "view_global_intervals_do_not_overlap",
            "views": views,
        }

    result: dict[str, Any] = {
        "available": True,
        "source": source,
        "global_start_sec": round(overlap_start, 6),
        "global_end_sec": round(overlap_end, 6),
        "global_start": round(overlap_start, 6),
        "global_end": round(overlap_end, 6),
        "duration_sec": round(overlap_end - overlap_start, 6),
        "views": views,
    }
    requested = _requested_common_overlap(requested_overlap)
    if requested is not None:
        result["requested_common_overlap"] = requested
        result["requested_common_overlap_clamped"] = bool(
            abs(float(requested["global_start_sec"]) - overlap_start) > 1e-6
            or abs(float(requested["global_end_sec"]) - overlap_end) > 1e-6
        )
    return result


def transcript_sec_to_global_time(transcript_source: TranscriptSource, local_sec: float) -> datetime:
    start = parse_time(transcript_source.start_time)
    return start + timedelta(seconds=float(transcript_source.offset_sec) + float(local_sec))


def align_transcript_to_global_time(transcript_path: str | Path, transcript_source: TranscriptSource) -> list[TranscriptUtterance]:
    rows: list[TranscriptUtterance] = []
    path = Path(transcript_path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            global_start = transcript_sec_to_global_time(transcript_source, float(raw["start_sec"]))
            global_end = transcript_sec_to_global_time(transcript_source, float(raw["end_sec"]))
            rows.append(TranscriptUtterance.from_raw(raw, global_start, global_end))
    return rows


def find_dialogue_for_segment(
    global_start_time: str | datetime,
    global_end_time: str | datetime,
    utterances: Iterable[TranscriptUtterance],
    window_sec: float = 3.0,
) -> list[TranscriptUtterance]:
    start = parse_time(global_start_time) - timedelta(seconds=window_sec)
    end = parse_time(global_end_time) + timedelta(seconds=window_sec)
    matched: list[TranscriptUtterance] = []
    for utterance in utterances:
        utt_start = parse_time(utterance.global_start_time)
        utt_end = parse_time(utterance.global_end_time)
        if utt_start <= end and utt_end >= start:
            matched.append(utterance)
    return matched


def evaluate_time_alignment(
    anchors: Iterable[Mapping[str, Any]] | str | Path,
    predictions: Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | Path | None = None,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    anchor_rows = _load_rows_or_iterable(anchors)
    prediction_by_id = _prediction_lookup(predictions)
    residual_rows: list[dict[str, Any]] = []
    required_count = 0
    covered_count = 0
    for index, anchor in enumerate(anchor_rows, start=1):
        anchor_id = str(anchor.get("anchor_id") or anchor.get("id") or anchor.get("event_id") or f"anchor_{index:06d}")
        required = anchor.get("required", True) is not False
        expected = _first_time(anchor, ("expected_global_time", "global_time", "global_start_time", "expected_time"))
        predicted = _first_time(anchor, ("predicted_global_time", "predicted_time", "aligned_global_time", "estimated_global_time"))
        if predicted is None and anchor_id in prediction_by_id:
            predicted = _first_time(
                prediction_by_id[anchor_id],
                ("predicted_global_time", "predicted_time", "aligned_global_time", "estimated_global_time", "global_time"),
            )
        if required and expected is not None:
            required_count += 1
        if expected is None or predicted is None:
            residual_rows.append(
                {
                    "anchor_id": anchor_id,
                    "source": anchor.get("source"),
                    "required": required,
                    "covered": False,
                    "expected_global_time": expected.isoformat() if expected is not None else None,
                    "predicted_global_time": predicted.isoformat() if predicted is not None else None,
                    "residual_sec": None,
                    "abs_residual_sec": None,
                }
            )
            continue
        residual = (predicted - expected).total_seconds()
        if required:
            covered_count += 1
        covered_row = {
            "anchor_id": anchor_id,
            "source": anchor.get("source"),
            "required": required,
            "expected_global_time": expected.isoformat(),
            "predicted_global_time": predicted.isoformat(),
            "residual_sec": round(residual, 6),
            "abs_residual_sec": round(abs(residual), 6),
            "residual_sign": "positive" if residual > 0 else "negative" if residual < 0 else "zero",
        }
        residual_rows.append(
            {
                "anchor_id": anchor_id,
                "source": anchor.get("source"),
                "required": required,
                "covered": True,
                "expected_global_time": expected.isoformat(),
                "predicted_global_time": predicted.isoformat(),
                "expected_timestamp_sec": expected.timestamp(),
                "residual_sec": round(residual, 6),
                "abs_residual_sec": round(abs(residual), 6),
            }
        )
        residual_rows[-1].update(covered_row)

    covered = [row for row in residual_rows if row.get("covered")]
    abs_residuals = [float(row["abs_residual_sec"]) for row in covered if row.get("abs_residual_sec") is not None]
    residual_values = [float(row.get("residual_sec") or 0.0) for row in covered]
    offset_history = [
        {
            "anchor_id": str(row.get("anchor_id")),
            "source": row.get("source"),
            "required": bool(row.get("required")),
            "offset_sec": float(row.get("residual_sec") or 0.0),
            "expected_global_time": row.get("expected_global_time"),
            "predicted_global_time": row.get("predicted_global_time"),
        }
        for row in covered
        if row.get("residual_sec") is not None
    ]
    jitter = _offset_jitter(residual_values)
    drift = _drift_error(covered)
    metrics = {
        "anchor_count": len(anchor_rows),
        "required_anchor_count": required_count,
        "evaluated_anchor_count": len(covered),
        "covered_required_anchor_count": covered_count,
        "anchor_coverage_rate": round(covered_count / required_count, 6) if required_count else 0.0,
        "mean_offset_sec": round(sum(residual_values) / len(residual_values), 6) if residual_values else 0.0,
        "mae_sec": round(sum(abs_residuals) / len(abs_residuals), 6) if abs_residuals else 0.0,
        "max_residual_sec": round(max(abs_residuals), 6) if abs_residuals else 0.0,
        "jitter_sec": jitter["jitter_sec"],
        "jitter_p95_sec": jitter["jitter_p95_sec"],
        "drift_error_sec": drift["drift_error_sec"],
        "drift_error_per_min": drift["drift_error_per_min"],
        "drift_span_sec": drift["drift_span_sec"],
    }
    alert_reasons: list[str] = []
    if required_count and (covered_count / required_count) < 0.8:
        alert_reasons.append("low_coverage")
    if metrics["mae_sec"] > 1.0:
        alert_reasons.append("high_mae")
    if metrics["max_residual_sec"] > 2.5:
        alert_reasons.append("large_residual")
    if drift["drift_error_per_min"] > 1.0:
        alert_reasons.append("high_drift_rate")
    if drift["drift_span_sec"] > 0 and metrics["max_residual_sec"] > 1.0 and metrics["jitter_sec"] > 0.6:
        alert_reasons.append("unstable_alignment")
    alignment_alert = bool(alert_reasons)
    result = {
        "schema_version": TIME_ALIGNMENT_EVAL_SCHEMA_VERSION,
        "metrics": metrics,
        "residuals": residual_rows,
        "metric_labels": {
            "mae_sec": "MAE",
            "max_residual_sec": "maximum absolute residual",
            "anchor_coverage_rate": "anchor coverage rate",
            "drift_error_sec": "end-to-end drift error",
            "jitter_sec": "alignment jitter",
        },
        "offset_history": offset_history,
        "alignment_alert": alignment_alert,
        "alignment_alert_reason": alert_reasons,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _load_rows_or_iterable(value: Iterable[Mapping[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if not path.exists():
            return []
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        row = json.loads(text)
                        if isinstance(row, Mapping):
                            rows.append(dict(row))
            return rows
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, Mapping):
            for key in ("anchors", "time_alignment_anchors", "rows"):
                if isinstance(data.get(key), list):
                    return [dict(row) for row in data[key] if isinstance(row, Mapping)]
            return [dict(data)]
        if isinstance(data, list):
            return [dict(row) for row in data if isinstance(row, Mapping)]
        return []
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _prediction_lookup(predictions: Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | Path | None) -> dict[str, dict[str, Any]]:
    if predictions is None:
        return {}
    if isinstance(predictions, Mapping):
        if any(isinstance(value, Mapping) for value in predictions.values()):
            return {str(key): dict(value) for key, value in predictions.items() if isinstance(value, Mapping)}
        rows = _load_rows_or_iterable([predictions])
    else:
        rows = _load_rows_or_iterable(predictions)
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        anchor_id = str(row.get("anchor_id") or row.get("segment_id") or row.get("id") or row.get("event_id") or f"anchor_{index:06d}")
        result[anchor_id] = row
    return result


def _first_time(row: Mapping[str, Any], keys: tuple[str, ...]) -> datetime | None:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            return parse_time(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_float(row: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if number == float("inf") or number == float("-inf") or number != number:
            continue
        return number
    return None


def _requested_common_overlap(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    common = payload.get("common_overlap") if isinstance(payload.get("common_overlap"), Mapping) else {}
    start = _first_float(common, ("global_start_sec", "global_start"))
    end = _first_float(common, ("global_end_sec", "global_end"))
    if start is None:
        start = _first_float(payload, ("common_overlap_start_sec", "global_start_sec", "global_start"))
    if end is None:
        end = _first_float(payload, ("common_overlap_end_sec", "global_end_sec", "global_end"))
    if start is None or end is None or end <= start:
        return None
    return {
        "global_start_sec": round(start, 6),
        "global_end_sec": round(end, 6),
        "global_start": round(start, 6),
        "global_end": round(end, 6),
        "duration_sec": round(end - start, 6),
    }


def _drift_error(covered: list[Mapping[str, Any]]) -> dict[str, float]:
    sortable = [
        row
        for row in covered
        if row.get("expected_timestamp_sec") is not None and row.get("residual_sec") is not None
    ]
    sortable.sort(key=lambda row: float(row.get("expected_timestamp_sec") or 0.0))
    if len(sortable) < 2:
        return {"drift_error_sec": 0.0, "drift_error_per_min": 0.0, "drift_span_sec": 0.0}
    first = sortable[0]
    last = sortable[-1]
    span = float(last["expected_timestamp_sec"]) - float(first["expected_timestamp_sec"])
    drift = abs(float(last["residual_sec"]) - float(first["residual_sec"]))
    return {
        "drift_error_sec": round(drift, 6),
        "drift_error_per_min": round(drift / (span / 60.0), 6) if span > 0 else 0.0,
        "drift_span_sec": round(max(0.0, span), 6),
    }


def _offset_jitter(offsets: list[float]) -> dict[str, float]:
    if len(offsets) < 3:
        return {"jitter_sec": 0.0, "jitter_p95_sec": 0.0}
    ordered = sorted(float(item) for item in offsets)
    if len(ordered) < 2:
        return {"jitter_sec": 0.0, "jitter_p95_sec": 0.0}
    diffs = [abs(b - a) for a, b in zip(ordered, ordered[1:]) if b > a or b < a]
    if not diffs:
        return {"jitter_sec": 0.0, "jitter_p95_sec": 0.0}
    diffs.sort()
    p95_index = max(0, min(len(diffs) - 1, int(round(0.95 * (len(diffs) - 1)))))
    return {
        "jitter_sec": round(sum(diffs) / len(diffs), 6),
        "jitter_p95_sec": round(diffs[p95_index], 6),
    }


def generate_multimodal_alignment(
    manifest: SessionManifest,
    segments: list[DetectedSegment],
    utterances: list[TranscriptUtterance],
    output_path: str | Path,
    dialogue_window_sec: float = 3.0,
) -> list[dict]:
    rows: list[dict] = []
    for segment in segments:
        segment_start = parse_time(segment.global_start_time)
        segment_end = parse_time(segment.global_end_time)
        streams = {
            view_id: {
                "video_path": source.path,
                "local_start_sec": global_time_to_video_sec(source, segment_start),
                "local_end_sec": global_time_to_video_sec(source, segment_end),
                "fps": source.fps,
                "offset_sec": source.offset_sec,
            }
            for view_id, source in manifest.videos.all_sources().items()
        }
        dialogue = find_dialogue_for_segment(segment_start, segment_end, utterances, window_sec=dialogue_window_sec)
        rows.append(
            {
                "session_id": manifest.session_id,
                "segment_id": segment.segment_id,
                "global_start_time": segment.global_start_time,
                "global_end_time": segment.global_end_time,
                "streams": streams,
                "transcript_refs": [item.utterance_id for item in dialogue],
                "transcript_text": [item.text for item in dialogue],
            }
        )
    write_jsonl(output_path, rows)
    return rows


def apply_alignment_correction(
    segments: list[dict[str, Any]],
    drift_result: dict[str, Any],
    *,
    degradation_factor: float = 0.85,
) -> list[dict[str, Any]]:
    """Apply alignment correction or degradation to segment list.

    If alignment is healthy and a stable window exists, correct timestamps.
    If alignment has drift alerts, degrade confidence instead.
    """
    summary = drift_result.get("summary") if isinstance(drift_result.get("summary"), dict) else {}
    status = str(summary.get("status", "no_data"))
    stable_window = drift_result.get("stable_window") if isinstance(drift_result.get("stable_window"), dict) else None

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        if status == "drift_alert":
            raw_score = float(segment.get("final_score", segment.get("score", 0.0)) or 0.0)
            segment["final_score"] = round(raw_score * degradation_factor, 6)
            segment["alignment_report"] = {
                "degraded": True,
                "degradation_factor": degradation_factor,
                "reason": "drift_alert_confidence_downgrade",
                "original_score": raw_score,
                "drift_status": status,
                "drift_events": summary.get("drift_events", 0),
            }
        elif status == "healthy" and stable_window is not None:
            correction_sec = float(stable_window.get("mean_offset_sec", 0.0))
            if abs(correction_sec) > 0.01:
                segment["alignment_report"] = {
                    "degraded": False,
                    "corrected": True,
                    "correction_sec": round(correction_sec, 6),
                    "drift_status": status,
                }
            else:
                segment["alignment_report"] = {"degraded": False, "corrected": False, "drift_status": status}
        else:
            segment["alignment_report"] = {"degraded": False, "corrected": False, "drift_status": status}
    return segments


def estimate_sliding_window_drift(
    offset_history: list[dict[str, Any]],
    *,
    window_size: int = 5,
    alert_threshold_sec: float = 1.5,
    smoothing_alpha: float = 0.3,
) -> dict[str, Any]:
    if not offset_history:
        return {
            "smoothed_offsets": [],
            "drift_windows": [],
            "alerts": [],
            "stable_window": None,
            "summary": {
                "mean_offset_ms": 0.0,
                "jitter_ms": 0.0,
                "drift_events": 0,
                "max_drift_sec": 0.0,
                "status": "no_data",
                "window_size": window_size,
                "smoothing_alpha": smoothing_alpha,
                "alert_threshold_sec": alert_threshold_sec,
            },
        }

    offsets = [float(item.get("offset_sec", 0.0)) for item in offset_history]

    smoothed: list[float] = []
    ema = offsets[0]
    for value in offsets:
        ema = smoothing_alpha * value + (1.0 - smoothing_alpha) * ema
        smoothed.append(round(ema, 6))

    smoothed_rows = [
        {
            "index": i,
            "raw_offset_sec": round(offsets[i], 6),
            "smoothed_offset_sec": smoothed[i],
            "anchor_id": offset_history[i].get("anchor_id"),
        }
        for i in range(len(offsets))
    ]

    drift_windows: list[dict[str, Any]] = []
    for i in range(len(smoothed) - window_size + 1):
        window = smoothed[i : i + window_size]
        window_drift = abs(window[-1] - window[0])
        window_mean = sum(window) / len(window)
        window_jitter = sum(abs(b - a) for a, b in zip(window, window[1:])) / max(1, len(window) - 1)
        drift_windows.append({
            "window_start": i,
            "window_end": i + window_size - 1,
            "drift_sec": round(window_drift, 6),
            "mean_offset_sec": round(window_mean, 6),
            "jitter_sec": round(window_jitter, 6),
            "exceeds_threshold": bool(window_drift > alert_threshold_sec),
        })

    alerts: list[dict[str, Any]] = []
    for window in drift_windows:
        if window["exceeds_threshold"]:
            alerts.append({
                "type": "drift_threshold_exceeded",
                "window_start": window["window_start"],
                "window_end": window["window_end"],
                "drift_sec": window["drift_sec"],
                "threshold_sec": alert_threshold_sec,
            })

    stable_window = None
    if drift_windows:
        stable_window = min(drift_windows, key=lambda w: w["jitter_sec"])

    mean_offset_ms = round(sum(offsets) / len(offsets) * 1000.0, 3) if offsets else 0.0
    diffs = [abs(b - a) for a, b in zip(smoothed, smoothed[1:])]
    jitter_ms = round(sum(diffs) / len(diffs) * 1000.0, 3) if diffs else 0.0
    max_drift = max((w["drift_sec"] for w in drift_windows), default=0.0)

    status = "healthy"
    if alerts:
        status = "drift_alert"
    elif max_drift > alert_threshold_sec * 0.7:
        status = "warning"

    return {
        "smoothed_offsets": smoothed_rows,
        "drift_windows": drift_windows,
        "alerts": alerts,
        "stable_window": stable_window,
        "summary": {
            "mean_offset_ms": mean_offset_ms,
            "jitter_ms": jitter_ms,
            "drift_events": len(alerts),
            "max_drift_sec": round(max_drift, 6),
            "status": status,
            "window_size": window_size,
            "smoothing_alpha": smoothing_alpha,
            "alert_threshold_sec": alert_threshold_sec,
        },
    }
