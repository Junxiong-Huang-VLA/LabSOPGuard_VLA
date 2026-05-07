from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

MATERIAL_PUBLISH_VERSION = "material_publish.v1"
UPLOAD_MANIFEST_VERSION = "material_upload_manifest.v1"


@dataclass
class PublishedPaths:
    clip: Optional[str]
    preview: Optional[str]
    keyframes: List[str] = field(default_factory=list)
    event_json: Optional[str] = None
    material_publish: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MaterialPublishRecord:
    schema_version: str
    experiment_id: str
    event_id: str
    material_id: str
    stable_name: str
    display_name: str
    actor_name: str
    event_type: str
    time_start: float
    time_end: float
    source_container: Optional[Dict[str, Any]]
    target_container: Optional[Dict[str, Any]]
    evidence_grade: Optional[str]
    review_status: Optional[str]
    canonical_event_path: str
    published_paths: Dict[str, Any]
    warnings: List[str]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UploadManifestItem:
    event_id: str
    material_id: str
    display_name: str
    stable_name: str
    asset_type: str
    event_type: str
    local_clip_path: Optional[str]
    local_metadata_path: str
    recommended_remote_path: str
    evidence_grade: Optional[str]
    review_status: Optional[str]
    remote_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def read_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
