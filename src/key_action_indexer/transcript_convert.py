from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .schemas import write_jsonl


_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_srt_timestamp(value: str) -> float:
    text = value.strip().replace(",", ".")
    hours, minutes, seconds = text.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_time_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("s") and re.fullmatch(r"-?\d+(?:\.\d+)?s", text):
        text = text[:-1]
    if ":" in text:
        parts = text.replace(",", ".").split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _text_from_row(row: dict[str, Any]) -> str:
    value = _first_present(row, ("text", "transcript", "utterance", "sentence", "word", "content"))
    if value is None and isinstance(row.get("alternatives"), list) and row["alternatives"]:
        alt = row["alternatives"][0]
        if isinstance(alt, dict):
            value = _first_present(alt, ("transcript", "text", "content"))
    return str(value or "").strip()


def _time_pair_from_row(row: dict[str, Any]) -> tuple[float | None, float | None]:
    timestamp = row.get("timestamp")
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        return _parse_time_value(timestamp[0]), _parse_time_value(timestamp[1])
    start = _parse_time_value(
        _first_present(row, ("start_sec", "start", "start_time", "startTime", "begin", "offset_sec", "offset"))
    )
    end = _parse_time_value(_first_present(row, ("end_sec", "end", "end_time", "endTime", "finish", "stop")))
    if start is not None and end is None:
        duration = _parse_time_value(_first_present(row, ("duration_sec", "duration", "duration_s")))
        if duration is not None:
            end = start + duration
    return start, end


def _looks_like_timed_row(row: dict[str, Any]) -> bool:
    start, end = _time_pair_from_row(row)
    return start is not None and end is not None and bool(_text_from_row(row))


def _candidate_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    if _looks_like_timed_row(payload):
        return [payload]
    for key in ("segments", "utterances", "chunks", "entries", "items", "words"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    results = payload.get("results")
    if isinstance(results, dict):
        for key in ("segments", "utterances", "items", "words"):
            value = results.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    if isinstance(results, list):
        return [row for row in results if isinstance(row, dict)]
    return []


def _normalize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    normalized: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        start, end = _time_pair_from_row(row)
        text = _text_from_row(row)
        if start is None or end is None or end <= start or not text:
            skipped += 1
            continue
        utterance_id = _first_present(row, ("utterance_id", "id", "segment_id"))
        normalized.append(
            {
                "utterance_id": str(utterance_id) if utterance_id is not None else "",
                "start_sec": float(start),
                "end_sec": float(end),
                "text": text,
            }
        )
    normalized.sort(key=lambda item: (item["start_sec"], item["end_sec"], item["text"]))
    for index, row in enumerate(normalized, start=1):
        if not row["utterance_id"]:
            row["utterance_id"] = f"utt_{index:03d}"
    return normalized, skipped


def srt_to_transcript_rows(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", text.strip())
    rows: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        time_line_index = next((idx for idx, line in enumerate(lines) if "-->" in line), -1)
        if time_line_index < 0:
            continue
        match = _TIMESTAMP_RE.search(lines[time_line_index])
        if not match:
            continue
        utterance_text = " ".join(lines[time_line_index + 1 :]).strip()
        if not utterance_text:
            continue
        rows.append(
            {
                "utterance_id": f"utt_{len(rows) + 1:03d}",
                "start_sec": parse_srt_timestamp(match.group("start")),
                "end_sec": parse_srt_timestamp(match.group("end")),
                "text": utterance_text,
            }
        )
    return rows


def _load_json_or_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        return [row for row in rows if isinstance(row, dict)], "jsonl"
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    return _candidate_rows(payload), "json"


def _merged_intervals(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals = sorted((float(row["start_sec"]), float(row["end_sec"])) for row in rows if row["end_sec"] > row["start_sec"])
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def transcript_coverage_summary(
    rows: list[dict[str, Any]],
    *,
    duration_sec: float | None = None,
    skipped_row_count: int = 0,
) -> dict[str, Any]:
    intervals = _merged_intervals(rows)
    spoken_duration = sum(end - start for start, end in intervals)
    start_sec = intervals[0][0] if intervals else None
    end_sec = intervals[-1][1] if intervals else None
    transcript_span = float(end_sec - start_sec) if start_sec is not None and end_sec is not None else 0.0
    denominator = float(duration_sec) if duration_sec and duration_sec > 0 else transcript_span
    return {
        "utterance_count": len(rows),
        "skipped_row_count": int(skipped_row_count),
        "coverage_start_sec": start_sec,
        "coverage_end_sec": end_sec,
        "transcript_span_sec": round(transcript_span, 6),
        "spoken_duration_sec": round(spoken_duration, 6),
        "coverage_duration_sec": round(denominator, 6) if denominator else 0.0,
        "coverage_ratio": round(spoken_duration / denominator, 6) if denominator else 0.0,
        "gap_count": max(0, len(intervals) - 1),
    }


def transcript_to_rows(path: str | Path) -> tuple[list[dict[str, Any]], str, int]:
    source = Path(path)
    if source.suffix.lower() == ".srt":
        return srt_to_transcript_rows(source), "srt", 0
    rows, input_format = _load_json_or_jsonl(source)
    normalized, skipped = _normalize_rows(rows)
    return normalized, input_format, skipped


def convert_transcript_to_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    duration_sec: float | None = None,
    summary_output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows, input_format, skipped = transcript_to_rows(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(target, rows)
    coverage = transcript_coverage_summary(rows, duration_sec=duration_sec, skipped_row_count=skipped)
    summary = {
        "input": str(input_path),
        "output": str(target),
        "input_format": input_format,
        "utterance_count": len(rows),
        "coverage": coverage,
    }
    if summary_output_path is not None:
        summary_target = Path(summary_output_path)
        summary_target.parent.mkdir(parents=True, exist_ok=True)
        summary_target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_output"] = str(summary_target)
    return summary


def convert_srt_to_jsonl(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    return convert_transcript_to_jsonl(input_path, output_path)
