"""Tests for frontend/API contract validators (P3, §10 & §17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer import frontend_contract as fc


def _good_window():
    return {
        "window_id": "w1",
        "third_view_realtime_preview_url": "file:///w1/third.mp4",
        "first_view_realtime_preview_url": "file:///w1/first.mp4",
        "side_by_side_realtime_preview_url": "file:///w1/sbs.mp4",
        "experiment_window_duration_s": 180.0,
        "preview_duration_s": 30.0,
        "preview_mode": "fast_preview",
        "fast_preview_url": "file:///w1/fast.mp4",
    }


def test_good_window_passes():
    r = fc.validate_window_view_contract(_good_window())
    assert r["ok"], r["issues"]


def test_third_equals_side_by_side_is_flagged():
    w = _good_window()
    w["third_view_realtime_preview_url"] = w["side_by_side_realtime_preview_url"]
    r = fc.validate_window_view_contract(w)
    assert not r["ok"]
    assert any("third_view" in i for i in r["issues"])


def test_first_equals_side_by_side_is_flagged():
    w = _good_window()
    w["first_view_realtime_preview_url"] = w["side_by_side_realtime_preview_url"]
    r = fc.validate_window_view_contract(w)
    assert not r["ok"]
    assert any("first_view" in i for i in r["issues"])


def test_missing_single_view_as_null_is_ok():
    # This mirrors the real backend: missing single-view -> null (frontend shows
    # 待生成), NOT a side-by-side substitution. null must be acceptable.
    w = _good_window()
    w["third_view_realtime_preview_url"] = None
    r = fc.validate_window_view_contract(w)
    assert r["ok"], r["issues"]
    assert r["third_preview_present"] is False


def test_missing_duration_fields_flagged():
    w = _good_window()
    w["experiment_window_duration_s"] = None
    r = fc.validate_window_view_contract(w)
    assert not r["ok"]
    assert any("experiment_window_duration_s" in i for i in r["issues"])


def test_fast_preview_without_label_flagged():
    w = _good_window()
    w["preview_mode"] = None
    r = fc.validate_window_view_contract(w)
    assert not r["ok"]
    assert any("preview_mode" in i for i in r["issues"])


def test_preview_display_report_lists_missing():
    windows = [
        _good_window(),
        {**_good_window(), "window_id": "w2", "first_view_realtime_preview_url": None},
    ]
    rep = fc.build_preview_display_report(windows)
    assert rep["missing_first_view_preview"] == ["w2"]
    assert rep["missing_third_view_preview"] == []
    assert rep["fast_preview_label"] == "快速预览，不代表完整实验时长"


def test_runtime_summary_value_without_source_flagged():
    summary = {"stages": {
        "总耗时": {"value": 123.4, "source_file": None, "source_field": None},
    }}
    r = fc.validate_runtime_summary(summary)
    assert not r["ok"]
    assert any("总耗时" in i for i in r["issues"])


def test_runtime_summary_missing_value_shows_weiji_lu():
    summary = {"stages": {
        "视频时间戳对齐": {"value": None},
    }}
    r = fc.validate_runtime_summary(summary)
    # missing value with no fake display is OK and renders 未记录
    stage = [s for s in r["stages"] if s["stage"] == "视频时间戳对齐"][0]
    assert stage["displayed"] == "未记录"


def test_runtime_summary_good():
    summary = {"stages": {
        s: {"value": 1.0, "source_file": "f.json", "source_field": "x", "displayed": "1.0s"}
        for s in fc.RUNTIME_STAGES
    }}
    r = fc.validate_runtime_summary(summary)
    assert r["ok"], r["issues"]


def test_write_reports(tmp_path):
    windows = [_good_window()]
    summary = {"stages": {s: {"value": None} for s in fc.RUNTIME_STAGES}}
    out = fc.write_frontend_contract_reports(
        tmp_path, window_items=windows, runtime_summary=summary)
    for key in ("window_view_artifact_contract_report",
                "frontend_preview_display_validation_report",
                "runtime_summary_data_contract_report"):
        assert Path(out[key]).exists()
    data = json.loads(Path(out["window_view_artifact_contract_report"]).read_text(encoding="utf-8"))
    assert data["ok"]
