from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def normalize_sha256(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    if _SHA256_RE.fullmatch(text):
        return text
    return None


def _normalized_view(value: Any) -> Optional[str]:
    aliases = {
        "first": "first_person",
        "first_person": "first_person",
        "operator": "first_person",
        "operator_view": "first_person",
        "ego": "first_person",
        "fpv": "first_person",
        "third": "third_person",
        "third_person": "third_person",
        "overview": "third_person",
        "table": "third_person",
        "top": "third_person",
        "top_view": "third_person",
        "bottom": "third_person",
    }
    return aliases.get(str(value or "").strip().lower())


def _iter_mapping_values(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        for item in value.values():
            if isinstance(item, Mapping):
                yield item
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                yield item


def _view_matches(payload: Mapping[str, Any], view_type: Optional[str]) -> bool:
    if not view_type:
        return True
    wanted = _normalized_view(view_type) or str(view_type)
    for key in ("view_type", "view", "role", "camera_id"):
        current = _normalized_view(payload.get(key)) or str(payload.get(key) or "").strip()
        if current == wanted:
            return True
    return False


def _candidate_payloads(manifest: Mapping[str, Any], view_type: Optional[str]) -> Iterable[Mapping[str, Any]]:
    for key in ("video_ref", "video", "source", "storage", "ref"):
        value = manifest.get(key)
        if isinstance(value, Mapping):
            yield value

    videos = manifest.get("videos")
    if isinstance(videos, Mapping):
        view_key = _normalized_view(view_type) if view_type else None
        if view_key and isinstance(videos.get(view_key), Mapping):
            yield videos[view_key]  # type: ignore[index]
        for payload in _iter_mapping_values(videos):
            yield payload

    for key in ("video_store_refs", "refs", "sources", "items", "files"):
        yield from _iter_mapping_values(manifest.get(key))

    yield manifest


def extract_video_store_ref(manifest: Mapping[str, Any], *, view_type: Optional[str] = None) -> Dict[str, Any]:
    """Extract a single video-store reference from a sidecar or upload manifest."""
    selected: Optional[Mapping[str, Any]] = None
    fallback: Optional[Mapping[str, Any]] = None
    for candidate in _candidate_payloads(manifest, view_type):
        if fallback is None:
            fallback = candidate
        if _view_matches(candidate, view_type):
            selected = candidate
            break
    payload = selected or fallback or manifest

    sha256 = normalize_sha256(
        payload.get("sha256")
        or payload.get("hash")
        or payload.get("digest")
        or payload.get("content_sha256")
        or manifest.get("sha256")
    )
    path_value = (
        payload.get("absolute_path")
        or payload.get("video_path")
        or payload.get("path")
        or payload.get("file_path")
        or manifest.get("absolute_path")
        or manifest.get("video_path")
        or manifest.get("path")
    )
    size_value = payload.get("size_bytes") or payload.get("size") or manifest.get("size_bytes") or manifest.get("size")
    original_filename = (
        payload.get("original_filename")
        or payload.get("filename")
        or manifest.get("original_filename")
        or (Path(str(path_value)).name if path_value else None)
    )
    return {
        "sha256": sha256,
        "path": str(path_value) if path_value else None,
        "absolute_path": str(payload.get("absolute_path") or path_value) if path_value else None,
        "size_bytes": size_value,
        "original_filename": original_filename,
        "safe_filename": payload.get("safe_filename") or original_filename,
        "content_type": payload.get("content_type") or manifest.get("content_type"),
        "metadata_path": payload.get("metadata_path") or manifest.get("metadata_path"),
        "video_store_root": payload.get("video_store_root") or manifest.get("video_store_root"),
        "deduplicated": payload.get("deduplicated"),
    }
