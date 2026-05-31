"""Detection result cache to avoid re-running YOLO on unchanged videos."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .schemas import DetectionBox, DetectionFrame, Tracklet

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION = "detection_cache.v2"


def compute_cache_key(
    video_path: str | Path,
    weights_path: str | None,
    imgsz: int,
    confidence_threshold: float,
    interval_sec: float,
    extra_components: Any | None = None,
) -> str:
    video_path = Path(video_path)
    try:
        mtime = str(video_path.stat().st_mtime)
        size = str(video_path.stat().st_size)
    except OSError:
        mtime = "unknown"
        size = "unknown"

    components_list = [
        str(video_path.resolve()),
        mtime,
        size,
        str(weights_path or "none"),
        str(imgsz),
        f"{confidence_threshold:.3f}",
        f"{interval_sec:.2f}",
    ]
    if extra_components is not None:
        components_list.append(
            json.dumps(extra_components, sort_keys=True, default=str, ensure_ascii=True)
        )
    components = "|".join(components_list)
    return hashlib.sha256(components.encode()).hexdigest()[:24]


class DetectionCache:
    """Persistent cache for DetectionFrame results."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.manifest.json"

    def _data_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.frames.json"

    def has(self, cache_key: str) -> bool:
        return self._manifest_path(cache_key).exists() and self._data_path(cache_key).exists()

    def load(self, cache_key: str) -> Optional[List[DetectionFrame]]:
        if not self.has(cache_key):
            return None
        try:
            manifest = json.loads(self._manifest_path(cache_key).read_text(encoding="utf-8"))
            if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
                logger.info("Cache schema mismatch, invalidating: %s", cache_key)
                self.invalidate(cache_key)
                return None

            data = json.loads(self._data_path(cache_key).read_text(encoding="utf-8"))
            frames = []
            for item in data:
                detections = [
                    DetectionBox(
                        bbox=tuple(d["bbox"]),
                        class_name=d["class_name"],
                        confidence=d["confidence"],
                        track_id=d.get("track_id"),
                    )
                    for d in item.get("detections", [])
                ]
                frames.append(DetectionFrame(
                    frame_idx=item["frame_idx"],
                    timestamp_sec=item["timestamp_sec"],
                    detections=detections,
                    semantic_activities=item.get("semantic_activities", []),
                    semantic_objects=item.get("semantic_objects", []),
                    scene_description=item.get("scene_description", ""),
                    change_score=item.get("change_score", 0.0),
                ))
            logger.info("Detection cache hit: %s (%d frames)", cache_key, len(frames))
            return frames
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Cache corrupted (%s), invalidating: %s", exc, cache_key)
            self.invalidate(cache_key)
            return None

    def save(
        self,
        cache_key: str,
        frames: List[DetectionFrame],
        metadata: Optional[dict] = None,
    ) -> None:
        manifest = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": cache_key,
            "frame_count": len(frames),
            **(metadata or {}),
        }
        data = [frame.to_dict() for frame in frames]

        try:
            self._manifest_path(cache_key).write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._data_path(cache_key).write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Detection cache saved: %s (%d frames)", cache_key, len(frames))
        except OSError as exc:
            logger.warning("Failed to write detection cache: %s", exc)

    def invalidate(self, cache_key: str) -> None:
        for path in [self._manifest_path(cache_key), self._data_path(cache_key)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
