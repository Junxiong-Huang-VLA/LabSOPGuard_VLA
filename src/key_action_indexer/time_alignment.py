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

    covered = [row for row in residual_rows if row.get("covered")]
    abs_residuals = [float(row["abs_residual_sec"]) for row in covered if row.get("abs_residual_sec") is not None]
    drift = _drift_error(covered)
    metrics = {
        "anchor_count": len(anchor_rows),
        "required_anchor_count": required_count,
        "evaluated_anchor_count": len(covered),
        "covered_required_anchor_count": covered_count,
        "anchor_coverage_rate": round(covered_count / required_count, 6) if required_count else 0.0,
        "mae_sec": round(sum(abs_residuals) / len(abs_residuals), 6) if abs_residuals else 0.0,
        "max_residual_sec": round(max(abs_residuals), 6) if abs_residuals else 0.0,
        "drift_error_sec": drift["drift_error_sec"],
        "drift_error_per_min": drift["drift_error_per_min"],
        "drift_span_sec": drift["drift_span_sec"],
    }
    result = {
        "schema_version": TIME_ALIGNMENT_EVAL_SCHEMA_VERSION,
        "metrics": metrics,
        "residuals": residual_rows,
        "metric_labels": {
            "mae_sec": "MAE",
            "max_residual_sec": "maximum absolute residual",
            "anchor_coverage_rate": "anchor coverage rate",
            "drift_error_sec": "end-to-end drift error",
        },
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
        anchor_id = str(row.get("anchor_id") or row.get("id") or row.get("event_id") or f"anchor_{index:06d}")
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
                "local_start_sec": global_time_to_local_sec(source, segment_start),
                "local_end_sec": global_time_to_local_sec(source, segment_end),
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
