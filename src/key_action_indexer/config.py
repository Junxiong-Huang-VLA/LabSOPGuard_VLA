from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .schemas import DetectionConfig, MicroSegmentConfig


DetectorConfig = DetectionConfig


@dataclass
class PipelineConfig:
    dry_run: bool = False
    detector: DetectorConfig = field(default_factory=DetectorConfig)


@dataclass
class DryRunConfig:
    enabled: bool = False
    synthesize_missing_inputs: bool = True
    synthetic_duration_sec: float = 960.0
    write_placeholder_media: bool = True


@dataclass
class TimeAlignmentConfig:
    default_latency_sec: float = 0.0
    default_fixed_offset_sec: float = 0.0
    source_calibration_path: str | None = None
    anchor_confidence_floor: float = 0.5


@dataclass
class RetrievalConfig:
    index_levels: list[str] = field(default_factory=lambda: ["segment", "micro_segment"])
    top_k_default: int = 5
    include_evidence_refs: bool = True


@dataclass
class InputIngestionConfig:
    user_text_events_path: str | None = None
    ai_reply_events_path: str | None = None
    upload_events_path: str | None = None
    calibration_path: str | None = None
    normalize_to_metadata: bool = True


@dataclass
class KeyActionConfig:
    schema_version: str = "key_action_config.v1"
    dry_run: DryRunConfig = field(default_factory=DryRunConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    micro_segments: MicroSegmentConfig = field(default_factory=MicroSegmentConfig)
    time_alignment: TimeAlignmentConfig = field(default_factory=TimeAlignmentConfig)
    inputs: InputIngestionConfig = field(default_factory=InputIngestionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "KeyActionConfig":
        if not data:
            return cls()
        return cls(
            schema_version=str(data.get("schema_version") or "key_action_config.v1"),
            dry_run=_dataclass_from_dict(DryRunConfig, data.get("dry_run")),
            detection=DetectionConfig.from_dict(data.get("detection") or data.get("detection_config")),
            micro_segments=MicroSegmentConfig.from_dict(data.get("micro_segments") or data.get("micro_segment_config")),
            time_alignment=_dataclass_from_dict(TimeAlignmentConfig, data.get("time_alignment")),
            inputs=_dataclass_from_dict(InputIngestionConfig, data.get("inputs") or data.get("input_ingestion")),
            retrieval=_dataclass_from_dict(RetrievalConfig, data.get("retrieval")),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_dict(cls: type, data: dict[str, Any] | None):
    values = asdict(cls())
    if data:
        for key in values:
            if key in data and data[key] is not None:
                values[key] = data[key]
    return cls(**values)


def default_config_dict() -> dict[str, Any]:
    return KeyActionConfig().to_json_dict()


def load_key_action_config(path: str | Path | None = None, data: dict[str, Any] | None = None) -> KeyActionConfig:
    if data is not None:
        return KeyActionConfig.from_dict(data)
    if path is None:
        return KeyActionConfig()
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Config file does not exist: {source}")
    return KeyActionConfig.from_dict(json.loads(source.read_text(encoding="utf-8-sig")))


def write_default_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = default_config_dict()
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
