from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import read_json, write_json


@dataclass
class UploadResult:
    provider: str
    uploaded_count: int
    skipped_count: int
    items: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "material_upload_result.v1",
            "provider": self.provider,
            "uploaded_count": self.uploaded_count,
            "skipped_count": self.skipped_count,
            "items": self.items,
            "warnings": self.warnings,
        }


class MaterialUploader:
    provider = "base"

    def upload_manifest(self, manifest_path: str | Path, **kwargs: Any) -> UploadResult:
        raise NotImplementedError


class LocalUploader(MaterialUploader):
    provider = "local"

    def upload_manifest(self, manifest_path: str | Path, *, destination_root: str | Path, **_: Any) -> UploadResult:
        manifest_path = Path(manifest_path)
        manifest = read_json(manifest_path, {})
        destination_root = Path(destination_root)
        items: List[Dict[str, Any]] = []
        warnings: List[str] = []
        uploaded = 0
        skipped = 0
        for item in manifest.get("items") or []:
            remote_base = destination_root / str(item.get("recommended_remote_path") or item.get("event_id") or "material")
            remote_base.mkdir(parents=True, exist_ok=True)
            copied: Dict[str, Optional[str]] = {}
            for key, source_key, target_name in (
                ("clip", "local_clip_path", "clip.mp4"),
                ("metadata", "local_metadata_path", "material_publish.json"),
            ):
                source = item.get(source_key)
                if not source or not Path(source).exists():
                    warnings.append(f"missing_{key}:{item.get('event_id')}")
                    copied[key] = None
                    skipped += 1
                    continue
                target = remote_base / target_name
                if not target.exists():
                    shutil.copy2(source, target)
                    uploaded += 1
                copied[key] = str(target)
            item["remote_url"] = str(remote_base)
            items.append({**item, "uploaded_paths": copied})
        result = UploadResult(self.provider, uploaded, skipped, items, warnings)
        manifest["items"] = [dict(item) for item in manifest.get("items") or []]
        write_json(manifest_path, manifest)
        write_json(manifest_path.parent / "upload_result.json", result.to_dict())
        return result


class NasUploader(LocalUploader):
    provider = "nas"


class StubRemoteUploader(MaterialUploader):
    def __init__(self, provider: str) -> None:
        self.provider = provider

    def upload_manifest(self, manifest_path: str | Path, **_: Any) -> UploadResult:
        manifest = read_json(manifest_path, {})
        return UploadResult(
            provider=self.provider,
            uploaded_count=0,
            skipped_count=len(manifest.get("items") or []),
            warnings=[f"{self.provider}_uploader_not_configured"],
        )


def uploader_for(provider: str) -> MaterialUploader:
    name = str(provider or "local").lower()
    if name == "local":
        return LocalUploader()
    if name == "nas":
        return NasUploader()
    if name in {"s3", "minio", "oss"}:
        return StubRemoteUploader(name)
    raise ValueError(f"Unsupported uploader provider: {provider}")
