from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TIMESTAMP_COL = "packet_system_timestamp_us"


class ConfigError(ValueError):
    """Raised when a sync config cannot be loaded or validated."""


@dataclass(frozen=True)
class TimestampConfig:
    preferred_cols: tuple[str, ...] = (DEFAULT_TIMESTAMP_COL,)

    @classmethod
    def from_mapping(cls, payload: Any) -> "TimestampConfig":
        if payload is None:
            return cls()
        if isinstance(payload, str):
            return cls((payload,))
        if not isinstance(payload, dict):
            raise ConfigError("timestamp must be a mapping or a timestamp column string")
        cols: list[str] = []
        explicit = payload.get("field") or payload.get("col") or payload.get("column")
        if explicit:
            cols.append(str(explicit))
        preferred = payload.get("preferred_cols")
        if preferred is None:
            preferred = payload.get("preferred_columns")
        if isinstance(preferred, str):
            cols.append(preferred)
        elif isinstance(preferred, list):
            cols.extend(str(item) for item in preferred if str(item).strip())
        elif preferred is not None:
            raise ConfigError("timestamp.preferred_cols must be a list of column names")
        if not cols:
            cols.append(DEFAULT_TIMESTAMP_COL)
        return cls(tuple(dict.fromkeys(col.strip() for col in cols if col.strip())))


@dataclass(frozen=True)
class StreamConfig:
    camera_id: str
    stream_path: Path
    frames_csv: str = "frames.csv"
    filters: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Any) -> "StreamConfig":
        if not isinstance(payload, dict):
            raise ConfigError("each streams item must be a mapping")
        camera_id = str(payload.get("camera_id") or payload.get("id") or "").strip()
        stream_path_raw = payload.get("stream_path") or payload.get("path") or payload.get("camera_dir")
        if not camera_id:
            raise ConfigError("streams item is missing camera_id")
        if not stream_path_raw:
            raise ConfigError(f"stream {camera_id} is missing stream_path")
        return cls(
            camera_id=camera_id,
            stream_path=Path(str(stream_path_raw)),
            frames_csv=str(payload.get("frames_csv") or "frames.csv"),
            filters=_parse_filters(payload),
        )

    @property
    def frames_path(self) -> Path:
        return self.stream_path / self.frames_csv


@dataclass(frozen=True)
class QualityGateConfig:
    max_abs_time_diff_us_warning: int | None = None
    unmatched_count_warning: int | None = None
    unmatched_rate_warning: float | None = None

    @classmethod
    def from_mapping(cls, payload: Any) -> "QualityGateConfig":
        if payload is None:
            return cls()
        if not isinstance(payload, dict):
            raise ConfigError("quality_gate must be a mapping")
        return cls(
            max_abs_time_diff_us_warning=_parse_optional_int(
                _first_present(payload, "max_abs_time_diff_us_warning", "max_abs_time_diff_us")
            ),
            unmatched_count_warning=_parse_optional_int(
                _first_present(payload, "unmatched_count_warning", "unmatched_frames_warning", "unmatched_frames")
            ),
            unmatched_rate_warning=_parse_optional_float(
                _first_present(payload, "unmatched_rate_warning", "unmatched_ratio_warning")
            ),
        )


@dataclass(frozen=True)
class SyncConfig:
    run_id: str = "default_run"
    experiment_name: str = ""
    batch_id: str = ""
    reference_camera: str = "cam01"
    output_dir: Path = Path("outputs/realityloop_sync/default_run")
    timestamp: TimestampConfig = field(default_factory=TimestampConfig)
    tolerance_us: int | None = None
    quality_gate: QualityGateConfig = field(default_factory=QualityGateConfig)
    streams: tuple[StreamConfig, ...] = ()
    manifest_csv: Path | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, base_dir: Path | None = None) -> "SyncConfig":
        if not isinstance(payload, dict):
            raise ConfigError("config root must be a mapping")
        base_dir = base_dir or Path.cwd()
        run_id = str(payload.get("run_id") or "default_run")
        output_dir = _resolve_path(payload.get("output_dir") or f"outputs/realityloop_sync/{run_id}", base_dir)
        tolerance_raw = payload.get("tolerance_us")
        tolerance_us = None if tolerance_raw in (None, "") else int(tolerance_raw)
        manifest_raw = payload.get("manifest_csv")
        streams_raw = payload.get("streams") or []
        if not isinstance(streams_raw, list):
            raise ConfigError("streams must be a list")
        streams = tuple(StreamConfig.from_mapping(row) for row in streams_raw)
        return cls(
            run_id=run_id,
            experiment_name=str(payload.get("experiment_name") or ""),
            batch_id=str(payload.get("batch_id") or ""),
            reference_camera=str(payload.get("reference_camera") or "cam01"),
            output_dir=output_dir,
            timestamp=TimestampConfig.from_mapping(payload.get("timestamp")),
            tolerance_us=tolerance_us,
            quality_gate=QualityGateConfig.from_mapping(payload.get("quality_gate")),
            streams=streams,
            manifest_csv=_resolve_path(manifest_raw, base_dir) if manifest_raw else None,
        )


def load_config(path: str | Path) -> SyncConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return SyncConfig.from_mapping(payload, base_dir=config_path.parent)


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute() or _looks_like_unc(str(value)):
        return path
    return (base_dir / path).resolve()


def _looks_like_unc(value: str) -> bool:
    return value.startswith("\\\\") or value.startswith("//")


def _parse_filters(payload: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    raw = payload.get("filters")
    if raw is None:
        raw = payload.get("filter")
    if raw is None:
        raw = payload.get("where")
    if raw is None and payload.get("stream_type"):
        raw = {"stream_type": payload.get("stream_type")}
    if raw in (None, ""):
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("stream filters must be a mapping, for example filters: {stream_type: rgb}")
    filters: dict[str, tuple[str, ...]] = {}
    for key, value in raw.items():
        col = str(key).strip()
        if not col:
            continue
        if isinstance(value, (list, tuple, set)):
            values = tuple(str(item).strip() for item in value if str(item).strip())
        else:
            values = (str(value).strip(),)
        if values:
            filters[col] = values
    return filters


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
