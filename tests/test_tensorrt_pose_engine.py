from __future__ import annotations

import pytest

from lab_vla.perception.tensorrt_pose_engine import resolve_int8_calibration_cache


pytestmark = pytest.mark.unit


def test_int8_cache_validation_is_noop_when_int8_disabled(tmp_path):
    missing_cache = tmp_path / "missing.cache"

    assert resolve_int8_calibration_cache(False, str(missing_cache)) is None


def test_int8_cache_validation_requires_existing_cache(tmp_path):
    with pytest.raises(RuntimeError, match="requires an existing calibration cache"):
        resolve_int8_calibration_cache(True, None)

    with pytest.raises(RuntimeError, match="calibration cache not found"):
        resolve_int8_calibration_cache(True, str(tmp_path / "missing.cache"))


def test_int8_cache_validation_returns_existing_cache(tmp_path):
    cache = tmp_path / "calibration.cache"
    cache.write_bytes(b"calibration")

    assert resolve_int8_calibration_cache(True, str(cache)) == cache
