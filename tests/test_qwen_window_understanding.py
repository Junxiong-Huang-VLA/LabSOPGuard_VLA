"""Tests for Qwen per-window understanding (P2, §14).

Enforces the honesty contract: missing config -> no fabricated observed_facts;
raw long video never used; hallucinated evidence_refs flagged as unsupported.
No network/GPU calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer import qwen_window_understanding as qu


def _window():
    return {
        "experiment_window_id": "w1",
        "third_view_realtime_preview": "windows/w1/third.mp4",
        "first_view_realtime_preview": "windows/w1/first.mp4",
        "sample_grid": "windows/w1/grid.jpg",
    }


def _materials():
    return [
        {"material_id": "m1", "action_type": "hand_object_contact",
         "evidence_bundle_id": "eb1", "first_keyframe": "m1/f.jpg",
         "third_keyframe": "m1/t.jpg", "first_keyclip": "m1/fc.mp4",
         "peak_global_timestamp_us": 1000},
        {"material_id": "m2", "action_type": "object_move",
         "evidence_bundle_id": "eb2", "peak_global_timestamp_us": 2000},
    ]


# --- integration status -------------------------------------------------


def test_offline_mode_status():
    s = qu.detect_integration_status(vlm_mode=qu.VLM_MODE_OFFLINE)
    assert s.vlm_status == qu.STATUS_OFFLINE_MODE
    assert not s.ready


def test_real_mode_without_client_is_missing_config():
    s = qu.detect_integration_status(
        vlm_mode=qu.VLM_MODE_REAL_QWEN_ASYNC, model="qwen3.5-flash", has_client=False)
    assert s.vlm_status == qu.STATUS_MISSING_CONFIG
    assert not s.ready


def test_unsupported_model_flagged():
    s = qu.detect_integration_status(
        vlm_mode=qu.VLM_MODE_REAL_QWEN_ASYNC, model="gpt-4o", has_client=True)
    assert s.vlm_status == qu.STATUS_UNSUPPORTED_MODEL


def test_ready_when_client_and_supported_model():
    s = qu.detect_integration_status(
        vlm_mode=qu.VLM_MODE_REAL_QWEN_ASYNC, model="qwen3.5-plus", has_client=True)
    assert s.ready


def test_write_integration_status_file(tmp_path):
    qu.write_integration_status(tmp_path, vlm_mode=qu.VLM_MODE_OFFLINE)
    data = json.loads((tmp_path / "qwen_vlm_integration_status.json").read_text(encoding="utf-8"))
    assert data["vlm_status"] == qu.STATUS_OFFLINE_MODE
    assert data["schema_version"] == qu.INTEGRATION_STATUS_SCHEMA


# --- input assembly never uses raw long video ---------------------------


def test_input_assembly_no_raw_video():
    inp = qu.assemble_window_vlm_input(window=_window(), materials=_materials())
    assert inp["raw_long_video_used"] is False
    assert inp["third_view_realtime_preview"] == "windows/w1/third.mp4"
    assert len(inp["evidence_refs"]) == 2
    assert "m1/f.jpg" in inp["source_keyframes"]


# --- honesty gate: missing config -> no fabricated facts ----------------


def test_missing_config_produces_no_fabricated_facts():
    status = qu.detect_integration_status(
        vlm_mode=qu.VLM_MODE_REAL_QWEN_ASYNC, model="qwen3.5-flash", has_client=False)
    rec = qu.build_window_understanding(
        window=_window(), materials=_materials(), vlm_raw=None, status=status)
    assert rec["understanding_generated"] is False
    assert rec["observed_facts"] == []
    assert rec["inferred_steps"] == []
    assert rec["uncertainties"]  # explains why not generated
    assert rec["raw_long_video_used"] is False


# --- field mapping + evidence ref resolution ----------------------------


def test_field_mapping_from_vlm_raw():
    status = qu.detect_integration_status(
        vlm_mode=qu.VLM_MODE_REAL_QWEN_ASYNC, model="qwen3.5-plus", has_client=True)
    vlm_raw = {
        "strong_facts": ["hand touches reagent bottle"],
        "weak_inferences": ["likely weighing step"],
        "unresolved_questions": ["which reagent?"],
        "visible_objects": ["reagent_bottle"],
        "evidence_refs": [{"material_id": "m1"}],
    }
    rec = qu.build_window_understanding(
        window=_window(), materials=_materials(), vlm_raw=vlm_raw, status=status)
    assert rec["understanding_generated"] is True
    assert rec["observed_facts"] == ["hand touches reagent bottle"]
    assert rec["inferred_steps"] == ["likely weighing step"]
    assert rec["uncertainties"] == ["which reagent?"]
    assert "reagent_bottle" in rec["involved_objects"]


def test_hallucinated_evidence_ref_flagged_unsupported():
    status = qu.detect_integration_status(
        vlm_mode=qu.VLM_MODE_REAL_QWEN_ASYNC, model="qwen3.5-plus", has_client=True)
    vlm_raw = {
        "strong_facts": ["x"],
        "evidence_refs": [{"material_id": "m1"}, {"material_id": "GHOST_999"}],
    }
    rec = qu.build_window_understanding(
        window=_window(), materials=_materials(), vlm_raw=vlm_raw, status=status)
    # GHOST_999 does not resolve to a real material -> unsupported
    flagged = [c for c in rec["unsupported_claims"]
               if isinstance(c, dict) and "unresolved_evidence_ref" in c]
    assert flagged
    assert flagged[0]["unresolved_evidence_ref"]["material_id"] == "GHOST_999"


def test_write_and_report(tmp_path):
    status = qu.detect_integration_status(vlm_mode=qu.VLM_MODE_OFFLINE)
    rec = qu.build_window_understanding(
        window=_window(), materials=_materials(), vlm_raw=None, status=status)
    out = qu.write_window_understanding(tmp_path / "w1", rec)
    assert Path(out["json"]).exists() and Path(out["md"]).exists()
    md = Path(out["md"]).read_text(encoding="utf-8")
    assert "observed_facts" in md
    report = qu.build_understanding_report([rec], status)
    assert report["window_count"] == 1
    assert report["understanding_generated_count"] == 0
    assert report["raw_long_video_used_anywhere"] is False
