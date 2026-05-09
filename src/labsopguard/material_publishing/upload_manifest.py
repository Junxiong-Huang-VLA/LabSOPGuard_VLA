from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .naming import slugify
from .schemas import UPLOAD_MANIFEST_VERSION, MaterialPublishRecord, UploadManifestItem, write_json


def remote_path_for(record: MaterialPublishRecord) -> str:
    actor = slugify(record.actor_name or "operator_unknown", fallback="operator_unknown")
    event_type = slugify(record.event_type, fallback="event")
    return f"{slugify(record.experiment_id, fallback='experiment')}/{actor}/{event_type}/{record.stable_name}/"


def build_upload_manifest(experiment_id: str, records: List[MaterialPublishRecord]) -> Dict[str, Any]:
    items = []
    for record in records:
        paths = record.published_paths or {}
        items.append(
            UploadManifestItem(
                event_id=record.event_id,
                material_id=record.material_id,
                display_name=record.display_name,
                stable_name=record.stable_name,
                asset_type="event_clip",
                event_type=record.event_type,
                local_clip_path=paths.get("clip"),
                local_metadata_path=str(paths.get("material_publish") or ""),
                recommended_remote_path=remote_path_for(record),
                evidence_grade=record.evidence_grade,
                review_status=record.review_status,
                remote_url=None,
            ).to_dict()
        )
    return {
        "schema_version": UPLOAD_MANIFEST_VERSION,
        "experiment_id": experiment_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": items,
    }


def write_upload_manifest(experiment_dir: str | Path, experiment_id: str, records: List[MaterialPublishRecord]) -> Dict[str, Any]:
    payload = build_upload_manifest(experiment_id, records)
    write_json(Path(experiment_dir) / "upload_manifest.json", payload)
    return payload
