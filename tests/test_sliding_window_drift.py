from __future__ import annotations

from key_action_indexer.time_alignment import apply_alignment_correction, estimate_sliding_window_drift


def test_empty_input():
    result = estimate_sliding_window_drift([])
    assert result["summary"]["status"] == "no_data"
    assert result["smoothed_offsets"] == []
    assert result["drift_windows"] == []
    assert result["alerts"] == []


def test_single_point():
    result = estimate_sliding_window_drift([{"offset_sec": 0.5, "anchor_id": "a1"}])
    assert result["summary"]["status"] == "healthy"
    assert len(result["smoothed_offsets"]) == 1


def test_healthy_stable_offsets():
    offsets = [{"offset_sec": 0.1 + i * 0.01, "anchor_id": f"a{i}"} for i in range(10)]
    result = estimate_sliding_window_drift(offsets, window_size=5, alert_threshold_sec=1.0)
    assert result["summary"]["status"] == "healthy"
    assert result["summary"]["drift_events"] == 0
    assert len(result["drift_windows"]) == 6


def test_drift_alert_triggered():
    offsets = [
        {"offset_sec": 0.1, "anchor_id": "a0"},
        {"offset_sec": 0.2, "anchor_id": "a1"},
        {"offset_sec": 0.3, "anchor_id": "a2"},
        {"offset_sec": 2.0, "anchor_id": "a3"},
        {"offset_sec": 3.5, "anchor_id": "a4"},
    ]
    result = estimate_sliding_window_drift(offsets, window_size=3, alert_threshold_sec=0.5)
    assert result["summary"]["status"] == "drift_alert"
    assert result["summary"]["drift_events"] > 0
    assert len(result["alerts"]) > 0


def test_ema_smoothing_effect():
    offsets = [{"offset_sec": float(i % 2), "anchor_id": f"a{i}"} for i in range(10)]
    result = estimate_sliding_window_drift(offsets, smoothing_alpha=0.1)
    smoothed_values = [r["smoothed_offset_sec"] for r in result["smoothed_offsets"]]
    raw_values = [r["raw_offset_sec"] for r in result["smoothed_offsets"]]
    smoothed_range = max(smoothed_values) - min(smoothed_values)
    raw_range = max(raw_values) - min(raw_values)
    assert smoothed_range < raw_range


def test_stable_window_found():
    offsets = [{"offset_sec": 0.5, "anchor_id": f"a{i}"} for i in range(10)]
    result = estimate_sliding_window_drift(offsets, window_size=3)
    assert result["stable_window"] is not None
    assert result["stable_window"]["jitter_sec"] == 0.0


def test_summary_fields():
    offsets = [{"offset_sec": i * 0.1, "anchor_id": f"a{i}"} for i in range(6)]
    result = estimate_sliding_window_drift(offsets, window_size=3, alert_threshold_sec=2.0, smoothing_alpha=0.5)
    summary = result["summary"]
    assert "mean_offset_ms" in summary
    assert "jitter_ms" in summary
    assert "drift_events" in summary
    assert "max_drift_sec" in summary
    assert "window_size" in summary
    assert summary["window_size"] == 3
    assert summary["smoothing_alpha"] == 0.5
    assert summary["alert_threshold_sec"] == 2.0


def test_apply_alignment_correction_drift_alert():
    segments = [{"segment_id": "s1", "final_score": 0.8}]
    drift_result = {
        "summary": {"status": "drift_alert", "drift_events": 2},
        "stable_window": None,
    }
    corrected = apply_alignment_correction(segments, drift_result, degradation_factor=0.85)
    assert corrected[0]["final_score"] == round(0.8 * 0.85, 6)
    assert corrected[0]["alignment_report"]["degraded"] is True


def test_apply_alignment_correction_healthy():
    segments = [{"segment_id": "s1", "final_score": 0.8}]
    drift_result = {
        "summary": {"status": "healthy"},
        "stable_window": {"mean_offset_sec": 0.05, "jitter_sec": 0.01},
    }
    corrected = apply_alignment_correction(segments, drift_result)
    assert corrected[0]["final_score"] == 0.8
    assert corrected[0]["alignment_report"]["degraded"] is False
    assert corrected[0]["alignment_report"]["corrected"] is True
