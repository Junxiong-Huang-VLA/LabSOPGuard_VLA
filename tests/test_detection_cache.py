"""Tests for the detection result cache."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from labsopguard.event_preprocessing.detection_cache import (
    CACHE_SCHEMA_VERSION,
    DetectionCache,
    compute_cache_key,
)
from labsopguard.event_preprocessing.schemas import DetectionBox, DetectionFrame


def _make_frames(n: int = 5) -> list:
    frames = []
    for i in range(n):
        frames.append(DetectionFrame(
            frame_idx=i * 15,
            timestamp_sec=i * 0.5,
            detections=[
                DetectionBox(bbox=(10, 20, 50, 60), class_name="bottle", confidence=0.85),
                DetectionBox(bbox=(100, 100, 200, 200), class_name="glove", confidence=0.72),
            ],
            change_score=0.03 * i,
        ))
    return frames


class TestComputeCacheKey:
    def test_deterministic(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")
        key1 = compute_cache_key(video, "model.pt", 960, 0.25, 0.5)
        key2 = compute_cache_key(video, "model.pt", 960, 0.25, 0.5)
        assert key1 == key2

    def test_different_weights_different_key(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")
        key1 = compute_cache_key(video, "model_a.pt", 960, 0.25, 0.5)
        key2 = compute_cache_key(video, "model_b.pt", 960, 0.25, 0.5)
        assert key1 != key2

    def test_different_imgsz_different_key(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")
        key1 = compute_cache_key(video, "model.pt", 960, 0.25, 0.5)
        key2 = compute_cache_key(video, "model.pt", 640, 0.25, 0.5)
        assert key1 != key2

    def test_extra_components_change_key(self, tmp_path: Path) -> None:
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video data")
        key1 = compute_cache_key(
            video,
            "model.pt",
            960,
            0.25,
            0.5,
            extra_components={"time_range": (0.0, 10.0)},
        )
        key2 = compute_cache_key(
            video,
            "model.pt",
            960,
            0.25,
            0.5,
            extra_components={"time_range": (10.0, 20.0)},
        )
        assert key1 != key2


class TestDetectionCache:
    def test_save_and_load(self, tmp_path: Path) -> None:
        cache = DetectionCache(tmp_path / "cache")
        frames = _make_frames(5)
        cache.save("test_key", frames)

        assert cache.has("test_key")
        loaded = cache.load("test_key")
        assert loaded is not None
        assert len(loaded) == 5
        assert loaded[0].frame_idx == 0
        assert loaded[0].timestamp_sec == 0.0
        assert len(loaded[0].detections) == 2
        assert loaded[0].detections[0].class_name == "bottle"
        assert loaded[0].detections[0].confidence == 0.85

    def test_cache_miss(self, tmp_path: Path) -> None:
        cache = DetectionCache(tmp_path / "cache")
        assert not cache.has("missing_key")
        assert cache.load("missing_key") is None

    def test_invalidate(self, tmp_path: Path) -> None:
        cache = DetectionCache(tmp_path / "cache")
        frames = _make_frames(3)
        cache.save("key_to_delete", frames)
        assert cache.has("key_to_delete")

        cache.invalidate("key_to_delete")
        assert not cache.has("key_to_delete")
        assert cache.load("key_to_delete") is None

    def test_corrupted_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Write corrupted data
        (cache_dir / "bad.manifest.json").write_text('{"schema_version": "detection_cache.v1"}')
        (cache_dir / "bad.frames.json").write_text("not valid json!!!")

        cache = DetectionCache(cache_dir)
        result = cache.load("bad")
        assert result is None
        # Should have been cleaned up
        assert not cache.has("bad")

    def test_schema_version_mismatch(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        (cache_dir / "old.manifest.json").write_text(json.dumps({
            "schema_version": "detection_cache.v0_old",
            "frame_count": 1,
        }))
        (cache_dir / "old.frames.json").write_text("[]")

        cache = DetectionCache(cache_dir)
        result = cache.load("old")
        assert result is None


def test_cache_runtime_settings_are_loaded(tmp_path: Path) -> None:
    from labsopguard.config import load_runtime_settings

    config_dir = tmp_path / "configs" / "model"
    config_dir.mkdir(parents=True)
    (config_dir / "detection_runtime.yaml").write_text(
        "\n".join([
            "cache:",
            "  detection_cache_enabled: false",
            "  batch_size: 3",
        ]),
        encoding="utf-8",
    )

    settings = load_runtime_settings(tmp_path)

    assert settings.detection_cache_enabled is False
    assert settings.batch_size == 3
