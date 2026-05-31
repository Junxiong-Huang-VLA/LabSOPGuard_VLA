"""Tests for the benchmark/calibration accuracy harness (AGENTS.md §1.2).

These tests enforce the hard contract: expected files never influence detection,
metrics are correct, and the validation_mode/accuracy_validated stamps are right.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer import benchmark_accuracy as ba


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def _formal_windows(windows):
    return {"schema_version": "formal_experiment_windows.v1", "windows": windows}


def _win(wid, start_us, end_us):
    return {
        "experiment_window_id": wid,
        "start_global_timestamp_us": start_us,
        "end_global_timestamp_us": end_us,
        "status": "formal",
    }


def _mat(mid, action_type, objs, start_us, end_us, window_id="w1"):
    return {
        "material_id": mid,
        "action_type": action_type,
        "object_refs": objs,
        "start_global_timestamp_us": start_us,
        "end_global_timestamp_us": end_us,
        "experiment_window_id": window_id,
        "official_status": "needs_review",
    }


def _det_action(mid, action_type, objs, start_us, end_us, window_id="w1"):
    """Normalized detected-action dict, as load_detected_actions emits."""
    return {
        "material_id": mid,
        "action_type": action_type,
        "object_refs": [str(o).lower() for o in objs],
        "start_us": start_us,
        "end_us": end_us,
        "window_id": window_id,
        "official_status": "needs_review",
    }


# --------------------------------------------------------------------------
# Contract: no expected files -> production_self_check, not validated, templates
# --------------------------------------------------------------------------


def test_production_self_check_when_no_expected_files(tmp_path):
    meta = tmp_path / "metadata"
    reports = tmp_path / "reports"
    fw = tmp_path / "formal_experiment_windows.json"
    ms = tmp_path / "material_stream.jsonl"
    _write_json(fw, _formal_windows([_win("w1", 0, 120_000_000)]))
    ms.write_text(
        json.dumps(_mat("m1", "hand_object_contact", ["container"], 1_000_000, 2_000_000)) + "\n",
        encoding="utf-8",
    )

    result = ba.run_benchmark_evaluation(
        formal_windows_path=fw,
        material_stream_path=ms,
        metadata_dir=meta,
        reports_dir=reports,
    )

    assert result["validation_mode"] == ba.VALIDATION_MODE_PRODUCTION_SELF_CHECK
    assert result["accuracy_validated"] is False
    # templates written, not filled
    assert (meta / "expected_windows.template.json").exists()
    assert (meta / "expected_actions.template.json").exists()
    win = json.loads((reports / "window_accuracy_report.json").read_text(encoding="utf-8"))
    assert win["status"] == "not_validated_without_expected_windows"
    assert win["accuracy_validated"] is False
    act = json.loads((reports / "action_accuracy_report.json").read_text(encoding="utf-8"))
    assert act["status"] == "not_validated_without_expected_actions"


# --------------------------------------------------------------------------
# Contract: expected files present -> benchmark mode, validated
# --------------------------------------------------------------------------


def test_benchmark_mode_when_expected_present(tmp_path):
    meta = tmp_path / "metadata"
    reports = tmp_path / "reports"
    fw = tmp_path / "formal_experiment_windows.json"
    ms = tmp_path / "material_stream.jsonl"
    _write_json(fw, _formal_windows([_win("w1", 0, 120_000_000)]))
    ms.write_text("", encoding="utf-8")

    _write_json(meta / "expected_windows.json", {
        "expected_windows": [
            {"expected_window_id": "e1",
             "start_global_timestamp_us": 0,
             "end_global_timestamp_us": 120_000_000},
        ]
    })
    _write_json(meta / "expected_actions.json", {"expected_actions": []})

    result = ba.run_benchmark_evaluation(
        formal_windows_path=fw, material_stream_path=ms,
        metadata_dir=meta, reports_dir=reports,
    )
    assert result["validation_mode"] == ba.VALIDATION_MODE_BENCHMARK
    assert result["accuracy_validated"] is True
    win = json.loads((reports / "window_accuracy_report.json").read_text(encoding="utf-8"))
    assert win["matched_window_count"] == 1
    assert win["recall"] == 1.0


# --------------------------------------------------------------------------
# Metrics correctness: windows
# --------------------------------------------------------------------------


def test_window_scoring_tp_fp_fn():
    detected = [
        {"window_id": "d1", "start_us": 0, "end_us": 100_000_000},      # matches e1
        {"window_id": "d2", "start_us": 500_000_000, "end_us": 600_000_000},  # FP
    ]
    expected = [
        {"expected_window_id": "e1",
         "start_global_timestamp_us": 5_000_000,
         "end_global_timestamp_us": 95_000_000},
        {"expected_window_id": "e2",  # missed - nothing near it
         "start_global_timestamp_us": 900_000_000,
         "end_global_timestamp_us": 1_000_000_000},
    ]
    rep = ba.score_windows(detected, expected)
    assert rep["matched_window_count"] == 1
    assert rep["false_positive_window_count"] == 1
    assert rep["missed_window_count"] == 1
    assert rep["false_positive_window_ids"] == ["d2"]
    assert rep["missed_window_ids"] == ["e2"]
    assert rep["start_boundary_error_s_avg"] is not None


# --------------------------------------------------------------------------
# Metrics correctness: actions (missed + false positive + classification)
# --------------------------------------------------------------------------


def test_action_scoring_counts():
    detected = [
        _det_action("m1", "hand_object_contact", ["reagent_bottle"], 10_000_000, 11_000_000),
        _det_action("m2", "hand_object_contact", ["reagent_bottle"], 12_000_000, 13_000_000),
        _det_action("m3", "device_panel_interaction", ["balance"], 20_000_000, 21_000_000),
    ]
    expected = [
        {"expected_action_id": "ea1", "action_type": "hand_object_contact",
         "object_type": "reagent_bottle", "expected_count": 3,  # expect 3, only 2 detected -> 1 missed
         "rough_start_global_timestamp_us": 0,
         "rough_end_global_timestamp_us": 30_000_000},
    ]
    res = ba.score_actions(detected, expected)
    s = res["summary"]
    assert s["matched_action_total"] == 2
    assert s["missed_action_total"] == 1
    # m3 (device_panel) was never expected -> false positive
    assert s["false_positive_action_total"] == 1
    assert res["missed_events"][0]["missed_count"] == 1


def test_classification_error_detected():
    # object matches an expectation, but action_type differs -> classification err
    detected = [
        _det_action("m1", "object_move", ["reagent_bottle"], 10_000_000, 11_000_000),
    ]
    expected = [
        {"expected_action_id": "ea1", "action_type": "hand_object_contact",
         "object_type": "reagent_bottle", "expected_count": 1,
         "rough_start_global_timestamp_us": 0,
         "rough_end_global_timestamp_us": 30_000_000},
    ]
    res = ba.score_actions(detected, expected)
    assert res["summary"]["classification_error_total"] == 1
    assert res["classification_errors"][0]["detected_action_type"] == "object_move"
    assert res["classification_errors"][0]["expected_action_type"] == "hand_object_contact"


# --------------------------------------------------------------------------
# Hard contract: expected files do not change detected inputs
# --------------------------------------------------------------------------


def test_expected_files_do_not_mutate_detection(tmp_path):
    fw = tmp_path / "formal_experiment_windows.json"
    ms = tmp_path / "material_stream.jsonl"
    _write_json(fw, _formal_windows([_win("w1", 0, 120_000_000)]))
    ms.write_text(
        json.dumps(_mat("m1", "hand_object_contact", ["container"], 1_000_000, 2_000_000)) + "\n",
        encoding="utf-8",
    )
    before_fw = fw.read_text(encoding="utf-8")
    before_ms = ms.read_text(encoding="utf-8")

    meta = tmp_path / "metadata"
    _write_json(meta / "expected_windows.json", {
        "expected_windows": [
            {"expected_window_id": "e1",
             "start_global_timestamp_us": 0,
             "end_global_timestamp_us": 120_000_000}]})
    _write_json(meta / "expected_actions.json", {
        "expected_actions": [
            {"expected_action_id": "ea1", "action_type": "hand_object_contact",
             "object_type": "container", "expected_count": 99}]})  # absurd count

    ba.run_benchmark_evaluation(
        formal_windows_path=fw, material_stream_path=ms,
        metadata_dir=meta, reports_dir=tmp_path / "reports",
    )
    # Detection inputs are byte-identical: the harness never writes back.
    assert fw.read_text(encoding="utf-8") == before_fw
    assert ms.read_text(encoding="utf-8") == before_ms


def test_templates_not_overwritten_when_human_files_present(tmp_path):
    meta = tmp_path / "metadata"
    _write_json(meta / "expected_windows.json", {"expected_windows": []})
    _write_json(meta / "expected_actions.json", {"expected_actions": []})
    ctx = ba.resolve_validation_context(meta)
    assert ctx.has_expected_windows and ctx.has_expected_actions
    assert ctx.validation_mode == ba.VALIDATION_MODE_BENCHMARK
