from __future__ import annotations

import pytest

from key_action_indexer.yolo_analysis import (
    TemporalDetectionSmoother,
    _tools_objects_from_labels,
    _visual_keywords,
    filter_detections_by_allowed_labels,
    filter_detections_by_class_threshold,
    parse_class_thresholds,
)
from key_action_indexer.schemas import DetectionConfig
from key_action_indexer.yolo_detector import _resolve_yolo_device, mock_yolo_frame_rows, scan_yolo_video


class _FakeCuda:
    def __init__(self, available: bool, count: int) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _FakeTorch:
    def __init__(self, available: bool, count: int) -> None:
        self.cuda = _FakeCuda(available, count)


def test_parse_class_thresholds_accepts_json_and_key_value_pairs() -> None:
    from_json = parse_class_thresholds('{"container": 0.4}')
    from_pairs = parse_class_thresholds("container=0.4, pipette=0.5")

    assert from_json["container"] == 0.4
    assert from_pairs["container"] == 0.4
    assert from_pairs["pipette"] == 0.5


def test_filter_detections_by_threshold_and_allowed_labels() -> None:
    detections = [
        {"label": "container", "confidence": 0.45},
        {"label": "pipette", "confidence": 0.2},
    ]

    filtered = filter_detections_by_class_threshold(detections, {"container": 0.4, "pipette": 0.3})
    allowed = filter_detections_by_allowed_labels(filtered, ["container"])

    assert filtered == [{"label": "container", "confidence": 0.45}]
    assert allowed == filtered


def test_analysis_index_exposes_magnetic_stir_bar_for_retrieval() -> None:
    from collections import Counter

    counts = Counter({"magnetic_stir_bar": 2, "reagent_bottle_open": 1})

    tools, objects = _tools_objects_from_labels(counts)
    keywords = _visual_keywords(counts)

    assert "magnetic_stir_bar" in tools
    assert "magnetic_stir_bar" in objects
    assert "磁力搅拌子" in keywords
    assert "reagent_bottle_open" in objects


def test_temporal_detection_smoother_keeps_recent_confirmed_label() -> None:
    smoother = TemporalDetectionSmoother(hold_frames=2, min_hits=2)

    first = smoother.update([{"label": "container", "confidence": 0.8, "bbox": [0, 0, 10, 10]}])
    second = smoother.update([{"label": "container", "confidence": 0.82, "bbox": [0, 0, 10, 10]}])
    third = smoother.update([])

    assert first == []
    assert second and second[0]["label"] == "container"
    assert third and third[0]["label"] == "container"


def test_yolo_dry_run_rows_use_scorer_active_threshold(tmp_path) -> None:
    rows = mock_yolo_frame_rows(
        duration_sec=1.0,
        sample_fps=1.0,
        video_path=tmp_path / "dry.mp4",
        active_windows=[(0.0, 0.0)],
        active_threshold=0.2,
    )

    assert len(rows) == 2
    assert "is_experiment_active" in rows[0]
    assert rows[0]["is_experiment_active"] is True


def test_scan_yolo_video_dry_run_does_not_require_active_threshold_in_row_scope(tmp_path) -> None:
    rows = scan_yolo_video(
        video_path=tmp_path / "missing.mp4",
        dry_run=True,
        sample_fps=1.0,
        mock_duration_sec=1.0,
        active_threshold=0.2,
    )

    assert rows
    assert all("is_experiment_active" in row for row in rows)


def test_yolo_auto_device_can_fall_back_but_explicit_cuda_is_strict() -> None:
    assert DetectionConfig().yolo_device == "auto"
    assert _resolve_yolo_device("auto", torch_module=_FakeTorch(False, 0)) == "cpu"
    with pytest.raises(RuntimeError, match="explicit CUDA YOLO device"):
        _resolve_yolo_device("0", torch_module=_FakeTorch(False, 0))
    with pytest.raises(RuntimeError, match="explicit CUDA YOLO device"):
        _resolve_yolo_device("cuda:0", torch_module=_FakeTorch(False, 0))
    assert _resolve_yolo_device("0", torch_module=_FakeTorch(True, 1)) == "0"
    with pytest.raises(RuntimeError, match="out of range"):
        _resolve_yolo_device("1", torch_module=_FakeTorch(True, 1))
    assert _resolve_yolo_device("auto", torch_module=_FakeTorch(True, 2)) == "0"
    assert _resolve_yolo_device(None, torch_module=_FakeTorch(True, 2)) == "0"
    assert _resolve_yolo_device("", torch_module=_FakeTorch(True, 2)) == "0"
