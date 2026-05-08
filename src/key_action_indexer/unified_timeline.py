from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schemas import SessionManifest, read_jsonl, write_jsonl
from .time_alignment import parse_time


ABSOLUTE_TIME_KEYS = ("timestamp", "global_time", "time")
TIMESTAMP_MS_KEYS = ("timestamp_ms",)
SESSION_SEC_KEYS = ("session_sec", "session_time_sec")
SOURCE_SEC_KEYS = ("source_time_sec", "source_sec", "local_sec", "local_time_sec", "time_sec", "start_sec")
SESSION_FALLBACK_SEC_KEYS = ("time_sec", "local_time_sec", "start_sec")
TEXT_KEYS = ("text", "content", "message", "caption", "description", "label", "response", "reply")
PATH_KEYS = ("path", "media_path", "image_path", "file_path")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class TimelineCalibration:
    slope: float = 1.0
    intercept_sec: float = 0.0
    anchor_count: int = 0
    method: str = "none"
    fixed_offset_sec: float = 0.0
    latency_sec: float = 0.0
    source_offset_sec: float = 0.0
    global_origin_time: str | None = None
    source_start_time: str | None = None
    global_seconds_basis: str = "session"
    residual_mean_abs_sec: float = 0.0
    residual_max_abs_sec: float = 0.0

    def map_source_sec(self, source_sec: float) -> float:
        mapped = self.slope * float(source_sec) + self.intercept_sec
        if self.method in {"none", "fixed_offset"}:
            mapped += self.source_offset_sec
        return mapped + self.fixed_offset_sec - self.latency_sec

    def has_clock_model(self) -> bool:
        return self.method in {"linear_drift", "offset", "fixed_offset"}

    def summary(self) -> dict[str, Any]:
        offset_sec = self.intercept_sec + self.fixed_offset_sec - self.latency_sec + self.source_offset_sec
        drift_rate = self.slope - 1.0
        return {
            "method": self.method,
            "anchor_count": self.anchor_count,
            "slope": self.slope,
            "drift_rate": drift_rate,
            "drift_sec_per_hour": drift_rate * 3600.0,
            "intercept_sec": self.intercept_sec,
            "offset_sec": offset_sec,
            "fixed_offset_sec": self.fixed_offset_sec,
            "latency_sec": self.latency_sec,
            "correction_sec": self.fixed_offset_sec - self.latency_sec,
            "source_offset_sec": self.source_offset_sec,
            "global_origin_time": self.global_origin_time,
            "source_start_time": self.source_start_time,
            "global_seconds_basis": self.global_seconds_basis,
            "residual_mean_abs_sec": self.residual_mean_abs_sec,
            "residual_max_abs_sec": self.residual_max_abs_sec,
        }


def _get(data: Any, key: str, default: Any = None) -> Any:
    if isinstance(data, Mapping):
        return data.get(key, default)
    return getattr(data, key, default)


def _has_value(row: Mapping[str, Any], key: str) -> bool:
    return key in row and row[key] is not None and row[key] != ""


def _first_value(row: Mapping[str, Any], keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        if _has_value(row, key):
            return key, row[key]
    return None, None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_number(row: Mapping[str, Any], keys: Iterable[str]) -> tuple[str | None, float | None]:
    for key in keys:
        if _has_value(row, key):
            number = _as_float(row[key])
            if number is not None:
                return key, number
    return None, None


def _manifest_session_start(manifest: Any) -> datetime | None:
    value = _get(manifest, "session_start_time")
    if value is None:
        return None
    return _parse_datetime(value)


def _manifest_session_id(manifest: Any) -> str | None:
    value = _get(manifest, "session_id")
    return str(value) if value is not None else None


def _parse_datetime(value: Any, origin: datetime | None = None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        if isinstance(value, (int, float)):
            seconds = _epoch_seconds_from_numeric(float(value))
            if seconds is None:
                return None
            parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
        else:
            text = str(value).strip()
            number = _as_float(text)
            if number is not None:
                seconds = _epoch_seconds_from_numeric(number)
                if seconds is None:
                    return None
                parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
            else:
                try:
                    parsed = parse_time(text)
                except (TypeError, ValueError):
                    return None
    return _align_timezone(parsed, origin)


def _epoch_seconds_from_numeric(value: float) -> float | None:
    magnitude = abs(value)
    if magnitude >= 1_000_000_000_000:
        return value / 1000.0
    if magnitude >= 1_000_000_000:
        return value
    return None


def _align_timezone(value: datetime, origin: datetime | None) -> datetime:
    if origin is None or origin.tzinfo is None:
        return value
    if value.tzinfo is None:
        return value.replace(tzinfo=origin.tzinfo)
    return value.astimezone(origin.tzinfo)


def _seconds_between(value: datetime, origin: datetime) -> float:
    return (_align_timezone(value, origin) - origin).total_seconds()


def _datetime_from_global_sec(global_sec: float, calibration: TimelineCalibration, origin: datetime | None) -> datetime | None:
    if calibration.global_seconds_basis == "epoch":
        value = datetime.fromtimestamp(global_sec, tz=timezone.utc)
        return _align_timezone(value, origin)
    if origin is None:
        return None
    return origin + timedelta(seconds=global_sec)


def _resolve_origin(manifest: Any = None, global_origin_time: str | datetime | None = None) -> datetime | None:
    if global_origin_time is not None:
        return _parse_datetime(global_origin_time)
    return _manifest_session_start(manifest)


def _anchor_global_sec(row: Mapping[str, Any], origin: datetime | None) -> tuple[float | None, str]:
    number = _as_float(row.get("global_sec"))
    if number is not None:
        return number, "global_sec"
    _, value = _first_value(row, ("global_time", "timestamp", "time"))
    parsed = _parse_datetime(value, origin)
    if parsed is None:
        return None, ""
    if origin is None:
        return parsed.timestamp(), "global_time_epoch"
    return _seconds_between(parsed, origin), "global_time_session"


def fit_timeline_calibration(
    anchors: Iterable[Mapping[str, Any]] | None = None,
    *,
    manifest: Any = None,
    global_origin_time: str | datetime | None = None,
    fixed_offset_sec: float = 0.0,
    latency_sec: float = 0.0,
    fixed_latency_sec: float | None = None,
    source_start_time: str | datetime | None = None,
    source_offset_sec: float = 0.0,
) -> TimelineCalibration:
    if fixed_latency_sec is not None:
        latency_sec = fixed_latency_sec
    origin = _resolve_origin(manifest, global_origin_time)
    basis = "session" if origin is not None else "epoch"
    source_start = _parse_datetime(source_start_time, origin) if source_start_time is not None else None
    points: list[tuple[float, float]] = []
    basis_seen = basis
    for anchor in anchors or []:
        _, source_sec = _first_number(anchor, SOURCE_SEC_KEYS)
        if source_sec is None:
            continue
        global_sec, global_basis = _anchor_global_sec(anchor, origin)
        if global_sec is None:
            continue
        if global_basis == "global_time_epoch":
            basis_seen = "epoch"
        points.append((source_sec, global_sec))

    if len(points) >= 2:
        slope, intercept = _fit_line(points)
        method = "linear_drift"
    elif len(points) == 1:
        slope = 1.0
        intercept = points[0][1] - points[0][0]
        method = "offset"
    else:
        slope = 1.0
        intercept = 0.0
        method = "fixed_offset" if fixed_offset_sec or latency_sec or source_offset_sec else "none"

    residuals = [abs((slope * source_sec + intercept) - global_sec) for source_sec, global_sec in points]
    return TimelineCalibration(
        slope=slope,
        intercept_sec=intercept,
        anchor_count=len(points),
        method=method,
        fixed_offset_sec=float(fixed_offset_sec),
        latency_sec=float(latency_sec),
        source_offset_sec=float(source_offset_sec),
        global_origin_time=origin.isoformat() if origin is not None else None,
        source_start_time=source_start.isoformat() if source_start is not None else None,
        global_seconds_basis=basis_seen,
        residual_mean_abs_sec=sum(residuals) / len(residuals) if residuals else 0.0,
        residual_max_abs_sec=max(residuals) if residuals else 0.0,
    )


def calibrate_timeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return fit_timeline_calibration(*args, **kwargs).summary()


def fit_drift_calibration(*args: Any, **kwargs: Any) -> TimelineCalibration:
    return fit_timeline_calibration(*args, **kwargs)


def _fit_line(points: list[tuple[float, float]]) -> tuple[float, float]:
    count = float(len(points))
    sum_x = sum(point[0] for point in points)
    sum_y = sum(point[1] for point in points)
    sum_xx = sum(point[0] * point[0] for point in points)
    sum_xy = sum(point[0] * point[1] for point in points)
    denominator = count * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-12:
        return 1.0, (sum_y - sum_x) / count
    slope = (count * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / count
    return slope, intercept


def _coerce_calibration(
    calibration: TimelineCalibration | Mapping[str, Any] | None,
    *,
    manifest: Any = None,
    fixed_offset_sec: float = 0.0,
    latency_sec: float = 0.0,
    fixed_latency_sec: float | None = None,
    source_start_time: str | datetime | None = None,
    source_offset_sec: float = 0.0,
) -> TimelineCalibration:
    if fixed_latency_sec is not None:
        latency_sec = fixed_latency_sec
    if calibration is None:
        return fit_timeline_calibration(
            manifest=manifest,
            fixed_offset_sec=fixed_offset_sec,
            latency_sec=latency_sec,
            source_start_time=source_start_time,
            source_offset_sec=source_offset_sec,
        )
    if isinstance(calibration, TimelineCalibration):
        updated = calibration
    else:
        fixed_value = calibration.get("fixed_offset_sec")
        latency_value = calibration.get("latency_sec")
        if fixed_value is None and latency_value is None and "correction_sec" in calibration:
            fixed_value = calibration.get("correction_sec")
            latency_value = 0.0
        updated = TimelineCalibration(
            slope=float(calibration.get("slope", 1.0)),
            intercept_sec=float(calibration.get("intercept_sec", 0.0)),
            anchor_count=int(calibration.get("anchor_count", 0)),
            method=str(calibration.get("method", "none")),
            fixed_offset_sec=float(fixed_value or 0.0),
            latency_sec=float(latency_value or 0.0),
            source_offset_sec=float(calibration.get("source_offset_sec", 0.0)),
            global_origin_time=calibration.get("global_origin_time"),
            source_start_time=calibration.get("source_start_time"),
            global_seconds_basis=str(calibration.get("global_seconds_basis", "session")),
            residual_mean_abs_sec=float(calibration.get("residual_mean_abs_sec", 0.0)),
            residual_max_abs_sec=float(calibration.get("residual_max_abs_sec", 0.0)),
        )
    if fixed_offset_sec or latency_sec or fixed_latency_sec is not None:
        updated = replace(
            updated,
            fixed_offset_sec=updated.fixed_offset_sec + float(fixed_offset_sec),
            latency_sec=updated.latency_sec + float(latency_sec),
        )
    if source_start_time is not None:
        source_start = _parse_datetime(source_start_time, _parse_datetime(updated.global_origin_time))
        updated = replace(updated, source_start_time=source_start.isoformat() if source_start is not None else None)
    if source_offset_sec:
        updated = replace(updated, source_offset_sec=updated.source_offset_sec + float(source_offset_sec))
    return updated


def read_event_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def read_events_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return read_event_jsonl(path)


def load_event_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_event_jsonl(path)


def _absolute_datetime_from_row(row: Mapping[str, Any], origin: datetime | None) -> tuple[datetime | None, str]:
    key, value = _first_value(row, ABSOLUTE_TIME_KEYS)
    parsed = _parse_datetime(value, origin)
    if parsed is not None:
        return parsed, "absolute_time"
    key, value = _first_value(row, TIMESTAMP_MS_KEYS)
    millis = _as_float(value)
    if key is not None and millis is not None and abs(millis) >= 1_000_000_000_000:
        return _align_timezone(datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc), origin), "absolute_timestamp_ms"
    return None, ""


def _apply_clock_correction(value: datetime, calibration: TimelineCalibration) -> datetime:
    return value + timedelta(seconds=calibration.fixed_offset_sec - calibration.latency_sec)


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in TEXT_KEYS + ("value",):
            text = _extract_text(value.get(key))
            if text:
                return text
        return ""
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return " ".join(part for part in parts if part)
    return str(value)


def _row_text(row: Mapping[str, Any]) -> str:
    for key in TEXT_KEYS:
        if _has_value(row, key):
            text = _extract_text(row[key])
            if text:
                return text
    description = row.get("text_description")
    if isinstance(description, Mapping):
        for key in ("index_text", "summary"):
            text = _extract_text(description.get(key))
            if text:
                return text
    transcript_text = row.get("transcript_text")
    if isinstance(transcript_text, list):
        return " ".join(str(item) for item in transcript_text if item)
    for key in ("index_text", "summary", "action_type"):
        if _has_value(row, key):
            return str(row[key])
    return ""


def _collect_links(row: Mapping[str, Any]) -> list[dict[str, str]]:
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(rel: str, value: Any) -> None:
        if not value:
            return
        path = str(value)
        token = f"{rel}:{path}"
        if token not in seen:
            seen.add(token)
            links.append({"rel": rel, "path": path})

    for key in PATH_KEYS + ("clip_path", "annotated_clip_path", "source_image_path", "video_path"):
        if _has_value(row, key):
            add(key, row[key])
    for view_key in ("first_person", "third_person"):
        value = row.get(view_key)
        if isinstance(value, Mapping):
            for key in ("clip_path", "annotated_clip_path", "video_path"):
                add(f"{view_key}.{key}", value.get(key))
    streams = row.get("streams")
    if isinstance(streams, Mapping):
        for stream_name, stream in streams.items():
            if isinstance(stream, Mapping):
                add(f"streams.{stream_name}.video_path", stream.get("video_path"))
    keyframes = row.get("keyframes")
    if isinstance(keyframes, Mapping):
        for key, value in keyframes.items():
            add(f"keyframes.{key}", value)
    elif isinstance(keyframes, list):
        for index, value in enumerate(keyframes, start=1):
            if isinstance(value, Mapping):
                add(f"keyframes.{index}", value.get("path"))
            else:
                add(f"keyframes.{index}", value)
    return links


def _infer_event_type(row: Mapping[str, Any], default: str = "event") -> str:
    role = str(row.get("role") or row.get("speaker") or row.get("author") or "").lower()
    if role in {"assistant", "ai", "model"}:
        return "ai_reply"
    if role in {"user", "human"}:
        return "user_text"
    if _has_value(row, "upload_type"):
        return "upload"
    key, value = _first_value(row, ("event_type", "type", "kind"))
    if key is not None:
        return str(value)
    if any(_has_value(row, key) for key in PATH_KEYS):
        return "upload"
    return default


def _infer_modality(row: Mapping[str, Any], event_type: str) -> str:
    if _has_value(row, "modality"):
        return str(row["modality"])
    upload_type = str(row.get("upload_type") or "").lower()
    if upload_type:
        if upload_type in {"image", "photo", "figure", "screenshot"}:
            return "image"
        if upload_type in {"text", "note"}:
            return "text"
        return upload_type
    for key in PATH_KEYS:
        if _has_value(row, key) and Path(str(row[key])).suffix.lower() in IMAGE_EXTENSIONS:
            return "image"
    if event_type in {"segment", "micro_segment"}:
        return "video"
    if event_type == "alignment":
        return "multimodal"
    if event_type in {"user_text", "ai_reply", "transcript"} or _row_text(row):
        return "text"
    return "unknown"


def _source_name(row: Mapping[str, Any], default: str = "event_jsonl") -> str:
    for key in ("source", "source_name", "stream"):
        if _has_value(row, key):
            return str(row[key])
    return default


def _duration_sec(row: Mapping[str, Any]) -> float:
    _, duration = _first_number(row, ("duration_sec", "duration"))
    if duration is not None:
        return max(0.0, duration)
    _, start = _first_number(row, ("start_sec", "local_start_sec"))
    _, end = _first_number(row, ("end_sec", "local_end_sec"))
    if start is not None and end is not None and end >= start:
        return end - start
    start_time = _parse_datetime(row.get("global_start_time") or row.get("global_time"))
    end_time = _parse_datetime(row.get("global_end_time"))
    if start_time is not None and end_time is not None:
        return max(0.0, (end_time - start_time).total_seconds())
    return 0.0


def _session_time_sec(global_time: datetime | None, origin: datetime | None) -> float | None:
    if global_time is None or origin is None:
        return None
    return _seconds_between(global_time, origin)


def _source_time_sec_from_row(row: Mapping[str, Any]) -> float | None:
    _, value = _first_number(row, SOURCE_SEC_KEYS + SESSION_SEC_KEYS)
    if value is not None:
        return value
    for view_key in ("first_person", "third_person"):
        view = row.get(view_key)
        if isinstance(view, Mapping):
            _, value = _first_number(view, ("local_start_sec", "local_time_sec", "start_sec"))
            if value is not None:
                return value
    streams = row.get("streams")
    if isinstance(streams, Mapping):
        for stream in streams.values():
            if isinstance(stream, Mapping):
                _, value = _first_number(stream, ("local_start_sec", "local_time_sec", "start_sec"))
                if value is not None:
                    return value
    return None


def _confidence_for_strategy(strategy: str) -> float:
    if strategy in {"absolute_time", "absolute_timestamp_ms", "existing_global_time"}:
        return 1.0
    if strategy == "session_time":
        return 0.9
    if strategy == "drift_linear":
        return 0.85
    if strategy in {"drift_offset", "source_start_offset", "fixed_offset"}:
        return 0.75
    if strategy == "fallback_session_time":
        return 0.7
    return 0.0


def build_timeline_event(
    row: Mapping[str, Any],
    *,
    manifest: Any = None,
    calibration: TimelineCalibration | Mapping[str, Any] | None = None,
    session_id: str | None = None,
    source: str | None = None,
    index: int | None = None,
    source_start_time: str | datetime | None = None,
    offset_sec: float = 0.0,
    fixed_offset_sec: float = 0.0,
    latency_sec: float = 0.0,
    fixed_latency_sec: float | None = None,
) -> dict[str, Any]:
    origin = _manifest_session_start(manifest)
    cal = _coerce_calibration(
        calibration,
        manifest=manifest,
        fixed_offset_sec=fixed_offset_sec,
        latency_sec=latency_sec,
        fixed_latency_sec=fixed_latency_sec,
        source_start_time=source_start_time,
        source_offset_sec=offset_sec,
    )
    if origin is None:
        origin = _parse_datetime(cal.global_origin_time)
    event_source = _source_name(row, default=source or "event_jsonl")
    raw_id = row.get("event_id") or row.get("id")
    timeline_event_id = str(raw_id) if raw_id is not None else f"{event_source}_{index or 1:06d}"

    global_dt, strategy = _absolute_datetime_from_row(row, origin)
    if global_dt is not None:
        global_dt = _apply_clock_correction(global_dt, cal)
    else:
        _, session_sec = _first_number(row, SESSION_SEC_KEYS)
        if session_sec is not None and origin is not None:
            global_dt = _apply_clock_correction(origin + timedelta(seconds=session_sec), cal)
            strategy = "session_time"
        else:
            _, timestamp_ms = _first_number(row, TIMESTAMP_MS_KEYS)
            relative_timestamp_sec = timestamp_ms / 1000.0 if timestamp_ms is not None else None
            if relative_timestamp_sec is not None and origin is not None:
                global_dt = _apply_clock_correction(origin + timedelta(seconds=relative_timestamp_sec), cal)
                strategy = "session_time"
            _, source_sec = _first_number(row, SOURCE_SEC_KEYS)
            if source_sec is None:
                source_sec = relative_timestamp_sec
            can_map_calibrated = source_sec is not None and cal.has_clock_model() and (
                origin is not None or (cal.global_seconds_basis == "epoch" and cal.anchor_count > 0)
            )
            if global_dt is None and can_map_calibrated:
                mapped_sec = cal.map_source_sec(source_sec)
                global_dt = _datetime_from_global_sec(mapped_sec, cal, origin)
                if global_dt is not None:
                    if cal.method == "linear_drift":
                        strategy = "drift_linear"
                    elif cal.method == "offset":
                        strategy = "drift_offset"
                    else:
                        strategy = "fixed_offset"
            if global_dt is None and source_sec is not None:
                source_origin = _parse_datetime(row.get("source_start_time"), origin)
                if source_origin is None:
                    source_origin = _parse_datetime(source_start_time or cal.source_start_time, origin)
                if source_origin is not None:
                    row_offset = _as_float(row.get("offset_sec"))
                    source_offset = row_offset if row_offset is not None else cal.source_offset_sec
                    global_dt = _apply_clock_correction(source_origin + timedelta(seconds=source_offset + source_sec), cal)
                    strategy = "source_start_offset"
            if global_dt is None and origin is not None:
                _, fallback_sec = _first_number(row, SESSION_FALLBACK_SEC_KEYS)
                if fallback_sec is not None:
                    global_dt = _apply_clock_correction(origin + timedelta(seconds=fallback_sec), cal)
                    strategy = "fallback_session_time"
    if not strategy:
        strategy = "unanchored"

    event_type = _infer_event_type(row)
    return {
        "timeline_event_id": timeline_event_id,
        "session_id": session_id or str(row.get("session_id") or _manifest_session_id(manifest) or ""),
        "event_type": event_type,
        "modality": _infer_modality(row, event_type),
        "source": event_source,
        "source_type": event_source,
        "global_time": global_dt.isoformat() if global_dt is not None else None,
        "session_time_sec": _session_time_sec(global_dt, origin),
        "source_time_sec": _source_time_sec_from_row(row),
        "duration_sec": _duration_sec(row),
        "anchor_confidence": _confidence_for_strategy(strategy),
        "anchor_strategy": strategy,
        "payload": dict(row),
        "links": _collect_links(row),
        "text": _row_text(row),
    }


def build_event_anchors(
    rows: Iterable[Mapping[str, Any]],
    *,
    manifest: Any = None,
    calibration: TimelineCalibration | Mapping[str, Any] | None = None,
    session_id: str | None = None,
    source: str | None = None,
    source_start_time: str | datetime | None = None,
    offset_sec: float = 0.0,
    fixed_offset_sec: float = 0.0,
    latency_sec: float = 0.0,
    fixed_latency_sec: float | None = None,
) -> list[dict[str, Any]]:
    return [
        build_timeline_event(
            row,
            manifest=manifest,
            calibration=calibration,
            session_id=session_id,
            source=source,
            index=index,
            source_start_time=source_start_time,
            offset_sec=offset_sec,
            fixed_offset_sec=fixed_offset_sec,
            latency_sec=latency_sec,
            fixed_latency_sec=fixed_latency_sec,
        )
        for index, row in enumerate(rows, start=1)
    ]


def build_timeline_events(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return build_event_anchors(*args, **kwargs)


def _existing_event_type(row: Mapping[str, Any], source: str | None) -> str:
    if _has_value(row, "event_type"):
        return str(row["event_type"])
    if _has_value(row, "micro_segment_id"):
        return "micro_segment"
    if _has_value(row, "utterance_id"):
        return "transcript"
    if _has_value(row, "segment_id") and _has_value(row, "streams"):
        return "alignment"
    if _has_value(row, "segment_id"):
        return "segment"
    if source:
        return source.rstrip("s")
    return _infer_event_type(row)


def _existing_event_id(row: Mapping[str, Any], source: str | None, index: int) -> str:
    for key in ("timeline_event_id", "micro_segment_id", "segment_id", "utterance_id", "event_id", "id"):
        if _has_value(row, key):
            return str(row[key])
    return f"{source or 'timeline'}_{index:06d}"


def _existing_global_time(row: Mapping[str, Any], origin: datetime | None) -> tuple[datetime | None, str]:
    for key in ("global_time", "global_start_time", "global_start", "start_global_time", "timestamp"):
        if _has_value(row, key):
            parsed = _parse_datetime(row[key], origin)
            if parsed is not None:
                return parsed, "existing_global_time"
    if origin is not None:
        _, seconds = _first_number(row, ("session_sec", "start_sec", "time_sec", "local_time_sec"))
        if seconds is not None:
            return origin + timedelta(seconds=seconds), "fallback_session_time"
    return None, "unanchored"


def normalize_timeline_row(
    row: Mapping[str, Any],
    *,
    manifest: Any = None,
    source: str | None = None,
    index: int = 1,
) -> dict[str, Any]:
    if "timeline_event_id" in row and "payload" in row:
        result = dict(row)
        result.setdefault("source", source or str(row.get("source") or "timeline"))
        result.setdefault("source_type", result.get("source"))
        result.setdefault("source_time_sec", _source_time_sec_from_row(row))
        result.setdefault("links", _collect_links(row))
        result.setdefault("text", _row_text(row))
        return result

    origin = _manifest_session_start(manifest)
    global_dt, strategy = _existing_global_time(row, origin)
    event_type = _existing_event_type(row, source)
    return {
        "timeline_event_id": _existing_event_id(row, source, index),
        "session_id": str(row.get("session_id") or _manifest_session_id(manifest) or ""),
        "event_type": event_type,
        "modality": _infer_modality(row, event_type),
        "source": source or _source_name(row, default=event_type),
        "source_type": source or _source_name(row, default=event_type),
        "global_time": global_dt.isoformat() if global_dt is not None else None,
        "session_time_sec": _session_time_sec(global_dt, origin),
        "source_time_sec": _source_time_sec_from_row(row),
        "duration_sec": _duration_sec(row),
        "anchor_confidence": _confidence_for_strategy(strategy),
        "anchor_strategy": strategy,
        "payload": dict(row),
        "links": _collect_links(row),
        "text": _row_text(row),
    }


def _iter_existing_rows(existing_rows: Any) -> Iterable[tuple[str | None, Mapping[str, Any]]]:
    if existing_rows is None:
        return []
    if isinstance(existing_rows, Mapping):
        items: list[tuple[str | None, Mapping[str, Any]]] = []
        for source, rows in existing_rows.items():
            for row in rows or []:
                items.append((str(source), row))
        return items
    return [(None, row) for row in existing_rows]


def _timeline_sort_key(row: Mapping[str, Any]) -> tuple[int, float, str]:
    parsed = _parse_datetime(row.get("global_time"))
    if parsed is None:
        return (1, 0.0, str(row.get("timeline_event_id") or ""))
    if parsed.tzinfo is None:
        timestamp = parsed.replace(tzinfo=timezone.utc).timestamp()
    else:
        timestamp = parsed.timestamp()
    return (0, timestamp, str(row.get("timeline_event_id") or ""))


def build_unified_timeline(
    *,
    existing_rows: Any = None,
    event_rows: Iterable[Mapping[str, Any]] | None = None,
    manifest: Any = None,
    calibration: TimelineCalibration | Mapping[str, Any] | None = None,
    session_id: str | None = None,
    source_start_time: str | datetime | None = None,
    offset_sec: float = 0.0,
    fixed_offset_sec: float = 0.0,
    latency_sec: float = 0.0,
    fixed_latency_sec: float | None = None,
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for index, (source, row) in enumerate(_iter_existing_rows(existing_rows), start=1):
        timeline.append(normalize_timeline_row(row, manifest=manifest, source=source, index=index))
    timeline.extend(
        build_event_anchors(
            event_rows or [],
            manifest=manifest,
            calibration=calibration,
            session_id=session_id,
            source_start_time=source_start_time,
            offset_sec=offset_sec,
            fixed_offset_sec=fixed_offset_sec,
            latency_sec=latency_sec,
            fixed_latency_sec=fixed_latency_sec,
        )
    )
    return sorted(timeline, key=_timeline_sort_key)


def _read_rows_if_present(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path)
    if not source.exists():
        return []
    return read_jsonl(source)


def _read_json_if_present(path: str | Path | None) -> Any:
    if path is None:
        return None
    source = Path(path)
    if not source.exists():
        return None
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    return json.loads(source.read_text(encoding="utf-8"))


def _source_calibration_config(config: Any, source: str) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    if any(key in config for key in ("anchors", "latency_sec", "fixed_latency_sec", "fixed_offset_sec", "offset_sec", "clock_offset_sec")):
        return config
    aliases = {
        "user_text": ("user_text", "user_events", "user"),
        "ai_reply": ("ai_reply", "ai_events", "assistant", "ai"),
        "upload": ("upload", "uploads", "upload_events"),
    }
    for key in aliases.get(source, (source,)):
        value = config.get(key)
        if isinstance(value, Mapping):
            return value
    default = config.get("default")
    return default if isinstance(default, Mapping) else {}


def _calibration_from_config(config: Any, source: str, manifest: Any) -> TimelineCalibration:
    spec = _source_calibration_config(config, source)
    offset = spec.get("fixed_offset_sec", spec.get("offset_sec", spec.get("clock_offset_sec", 0.0)))
    latency = spec.get("latency_sec", spec.get("fixed_latency_sec", 0.0))
    return fit_timeline_calibration(
        spec.get("anchors") or [],
        manifest=manifest,
        fixed_offset_sec=float(offset or 0.0),
        latency_sec=float(latency or 0.0),
        source_start_time=spec.get("source_start_time"),
        source_offset_sec=float(spec.get("source_offset_sec", 0.0) or 0.0),
    )


def _event_rows_from_path(path: str | Path | None, *, source: str, event_type: str, modality: str | None = None) -> list[dict[str, Any]]:
    rows = _read_rows_if_present(path)
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.setdefault("source", source)
        item.setdefault("event_type", event_type)
        if modality is not None:
            item.setdefault("modality", modality)
        prepared.append(item)
    return prepared


def _interaction_event_rows(segment_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in segment_rows:
        interactions = segment.get("interaction_events")
        if not isinstance(interactions, list):
            continue
        for index, interaction in enumerate(interactions, start=1):
            if not isinstance(interaction, Mapping):
                continue
            event_id = interaction.get("event_id") or f"{segment.get('segment_id', 'segment')}_interaction_{index:03d}"
            rows.append(
                {
                    "event_id": event_id,
                    "session_id": segment.get("session_id"),
                    "segment_id": segment.get("segment_id"),
                    "event_type": "yolo_interaction",
                    "modality": "video",
                    "source": "yolo_interaction",
                    "global_time": interaction.get("global_time"),
                    "local_time_sec": interaction.get("local_time_sec"),
                    "text": interaction.get("interaction") or interaction.get("object_name") or "",
                    "path": interaction.get("keyframe_path"),
                    "confidence": interaction.get("confidence"),
                    "payload": dict(interaction),
                }
            )
    return rows


def _micro_anchor_rows(micro_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    phase_fields = (
        ("contact", "contact_start_sec"),
        ("peak", "peak_interaction_sec"),
        ("release", "contact_end_sec"),
    )
    for micro in micro_rows:
        interaction = micro.get("interaction") if isinstance(micro.get("interaction"), Mapping) else {}
        keyframes = micro.get("keyframes") if isinstance(micro.get("keyframes"), Mapping) else {}
        micro_id = str(micro.get("micro_segment_id") or "")
        for phase, field in phase_fields:
            if field not in interaction:
                continue
            keyframe_key = "release_frame" if phase == "release" else f"{phase}_frame"
            rows.append(
                {
                    "event_id": f"{micro_id}_{phase}",
                    "session_id": micro.get("session_id"),
                    "segment_id": micro.get("parent_segment_id"),
                    "micro_segment_id": micro_id,
                    "event_type": f"micro_{phase}_anchor",
                    "modality": "video",
                    "source": "micro_anchor",
                    "session_sec": interaction.get(field),
                    "text": f"{interaction.get('interaction_type', 'interaction')} {phase}".strip(),
                    "path": keyframes.get(keyframe_key),
                    "primary_object": interaction.get("primary_object"),
                    "payload": {
                        "phase": phase,
                        "interaction": dict(interaction),
                    },
                }
            )
    return rows


def _as_path_list(paths: str | Path | Iterable[str | Path] | None) -> list[str | Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        return [paths]
    return list(paths)


def build_unified_timeline_from_jsonl(
    *,
    manifest: Any = None,
    segment_path: str | Path | None = None,
    micro_path: str | Path | None = None,
    alignment_path: str | Path | None = None,
    transcript_path: str | Path | None = None,
    event_paths: str | Path | Iterable[str | Path] | None = None,
    timeline_paths: Mapping[str, str | Path] | None = None,
    output_path: str | Path | None = None,
    calibration: TimelineCalibration | Mapping[str, Any] | None = None,
    source_start_time: str | datetime | None = None,
    offset_sec: float = 0.0,
    fixed_offset_sec: float = 0.0,
    latency_sec: float = 0.0,
    fixed_latency_sec: float | None = None,
) -> list[dict[str, Any]]:
    existing: dict[str, list[dict[str, Any]]] = {
        "segment": _read_rows_if_present(segment_path),
        "micro_segment": _read_rows_if_present(micro_path),
        "alignment": _read_rows_if_present(alignment_path),
        "transcript": _read_rows_if_present(transcript_path),
    }
    for source, path in (timeline_paths or {}).items():
        existing[str(source)] = _read_rows_if_present(path)
    event_rows: list[dict[str, Any]] = []
    for path in _as_path_list(event_paths):
        event_rows.extend(_read_rows_if_present(path))
    timeline = build_unified_timeline(
        existing_rows=existing,
        event_rows=event_rows,
        manifest=manifest,
        calibration=calibration,
        source_start_time=source_start_time,
        offset_sec=offset_sec,
        fixed_offset_sec=fixed_offset_sec,
        latency_sec=latency_sec,
        fixed_latency_sec=fixed_latency_sec,
    )
    if output_path is not None:
        write_jsonl(output_path, timeline)
    return timeline


def write_unified_timeline(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl(path, list(rows))


def merge_unified_timeline(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    return build_unified_timeline(*args, **kwargs)


def build_time_anchors(
    timeline_rows: Iterable[Mapping[str, Any]],
    source_reports: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    reports = source_reports or {}
    anchors: list[dict[str, Any]] = []
    for index, row in enumerate(timeline_rows, start=1):
        source = str(row.get("source") or row.get("source_type") or "timeline")
        report = reports.get(source) if isinstance(reports.get(source), Mapping) else {}
        anchors.append(
            {
                "anchor_id": f"anchor_{index:06d}",
                "source_event": {
                    "event_id": row.get("timeline_event_id"),
                    "source": source,
                    "event_type": row.get("event_type"),
                    "source_time_sec": row.get("source_time_sec"),
                },
                "target_event": {
                    "global_time": row.get("global_time"),
                    "session_time_sec": row.get("session_time_sec"),
                    "duration_sec": row.get("duration_sec"),
                },
                "confidence": float(row.get("anchor_confidence", 0.0) or 0.0),
                "reason": str(row.get("anchor_strategy") or "unanchored"),
                "calibration": {
                    "method": report.get("method"),
                    "offset_sec": report.get("offset_sec"),
                    "drift_rate": report.get("drift_rate"),
                    "drift_sec_per_hour": report.get("drift_sec_per_hour"),
                    "anchor_count": report.get("anchor_count"),
                    "residual_mean_abs_sec": report.get("residual_mean_abs_sec"),
                    "residual_max_abs_sec": report.get("residual_max_abs_sec"),
                },
                "links": row.get("links", []),
            }
        )
    return anchors


def _manifest_time_source_reports(manifest: SessionManifest, transcript_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for view_id, source in manifest.videos.all_sources().items():
        calibration = fit_timeline_calibration(
            manifest=manifest,
            source_start_time=source.start_time,
            source_offset_sec=float(source.offset_sec or 0.0),
        )
        reports[f"video:{view_id}"] = {
            **calibration.summary(),
            "input_path": source.path,
            "input_event_count": 0,
            "view": view_id,
            "source_role": "video",
        }
    if manifest.transcript is not None:
        calibration = fit_timeline_calibration(
            manifest=manifest,
            source_start_time=manifest.transcript.start_time,
            source_offset_sec=float(manifest.transcript.offset_sec or 0.0),
        )
        reports["transcript"] = {
            **calibration.summary(),
            "input_path": manifest.transcript.path,
            "input_event_count": len(transcript_rows),
            "source_role": "transcript",
        }
    return reports


def generate_unified_timeline(
    manifest_path: str | Path,
    output_dir: str | Path | None = None,
    user_events_path: str | Path | None = None,
    ai_events_path: str | Path | None = None,
    uploads_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = SessionManifest.load(manifest_path)
    session_dir = Path(manifest.output_dir)
    metadata_dir = session_dir / "metadata"
    target_dir = Path(output_dir) if output_dir is not None else metadata_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    segment_rows = _read_rows_if_present(metadata_dir / "key_action_segments.jsonl")
    micro_rows = _read_rows_if_present(metadata_dir / "micro_segments.jsonl")
    alignment_rows = _read_rows_if_present(metadata_dir / "multimodal_alignment.jsonl")
    transcript_rows = _read_rows_if_present(session_dir / "transcript" / "aligned_transcript.jsonl")

    existing_rows = {
        "segment": segment_rows,
        "micro_segment": micro_rows,
        "alignment": alignment_rows,
        "transcript": transcript_rows,
        "yolo_interaction": _interaction_event_rows(segment_rows),
        "micro_anchor": _micro_anchor_rows(micro_rows),
    }
    timeline = build_unified_timeline(existing_rows=existing_rows, manifest=manifest)

    calibration_config = _read_json_if_present(calibration_path)
    external_specs = [
        ("session_context", "session_context", "text", metadata_dir / "session_context_events.jsonl"),
        ("user_text", "user_text", "text", user_events_path or metadata_dir / "user_text_events.jsonl"),
        ("ai_reply", "ai_reply", "text", ai_events_path or metadata_dir / "ai_reply_events.jsonl"),
        ("upload", "upload", None, uploads_path or metadata_dir / "upload_events.jsonl"),
    ]
    calibration_report: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "session_id": manifest.session_id,
        "calibration_path": str(calibration_path) if calibration_path else None,
        "dry_run": dry_run,
        "sources": _manifest_time_source_reports(manifest, transcript_rows),
    }
    for source, event_type, modality, path in external_specs:
        rows = _event_rows_from_path(path, source=source, event_type=event_type, modality=modality)
        calibration = _calibration_from_config(calibration_config, source, manifest)
        calibration_report["sources"][source] = {
            **calibration.summary(),
            "input_path": str(path) if path else None,
            "input_event_count": len(rows),
        }
        timeline.extend(
            build_event_anchors(
                rows,
                manifest=manifest,
                calibration=calibration,
                session_id=manifest.session_id,
                source=source,
            )
        )

    timeline = sorted(timeline, key=_timeline_sort_key)
    timeline_path = target_dir / "unified_multimodal_timeline.jsonl"
    anchors_path = target_dir / "time_anchors.jsonl"
    report_path = target_dir / "time_calibration_report.json"
    time_anchors = build_time_anchors(timeline, calibration_report["sources"])
    write_jsonl(timeline_path, timeline)
    write_jsonl(anchors_path, time_anchors)
    calibration_report.update(
        {
            "timeline_path": str(timeline_path),
            "time_anchors_path": str(anchors_path),
            "calibration_report": str(report_path),
            "event_count": len(timeline),
            "time_anchor_count": len(time_anchors),
            "artifact_counts": {
                "segments": len(segment_rows),
                "micro_segments": len(micro_rows),
                "alignment_rows": len(alignment_rows),
                "transcript_rows": len(transcript_rows),
                "yolo_interaction_events": len(existing_rows["yolo_interaction"]),
                "micro_anchor_events": len(existing_rows["micro_anchor"]),
            },
        }
    )
    report_path.write_text(json.dumps(calibration_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return calibration_report


def generate_unified_multimodal_timeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return generate_unified_timeline(*args, **kwargs)


def run_unified_timeline(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return generate_unified_timeline(*args, **kwargs)
