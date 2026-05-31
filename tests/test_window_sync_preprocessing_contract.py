from __future__ import annotations

from key_action_indexer.experiment_window_state import (
    _preview_rows_from_window_sync,
    _realtime_rows_from_window_sync,
    _window_start_boundary_metadata,
    _window_sync_duration_s,
)
from key_action_indexer.material_references import (
    KEY_CLIP_DIR_NAME,
    KEYFRAME_DIR_NAME,
    _dual_view_material_pairing_row,
    _keyclip_evidence_status,
    _window_sync_repeat_count,
)


def _clip_row(view: str, *, start: int = 10, end: int = 20) -> dict:
    return {
        "view": view,
        "asset_kind": KEY_CLIP_DIR_NAME,
        "stored_file": __file__,
        "start_sec": 1.0,
        "end_sec": 3.0,
        "source_window_sync_index": __file__,
        "start_window_sync_index": start,
        "end_window_sync_index": end,
    }


def _keyframe_row(view: str, *, peak: int = 15) -> dict:
    return {
        "view": view,
        "asset_kind": KEYFRAME_DIR_NAME,
        "stored_file": __file__,
        "source_window_sync_index": __file__,
        "peak_window_sync_index": peak,
        "global_timestamp_us": 1_000_000,
    }


def test_keyclip_evidence_requires_window_sync_row_range() -> None:
    valid = _keyclip_evidence_status([_clip_row("first_person")])
    assert valid["valid"] is True

    missing_range = _clip_row("first_person")
    missing_range.pop("start_window_sync_index")
    invalid = _keyclip_evidence_status([missing_range])
    assert invalid["valid"] is False
    assert "missing_or_invalid_window_sync_row_range" in invalid["issues"]


def test_dual_view_pairing_requires_same_sync_row_range() -> None:
    rows = [
        _keyframe_row("first_person", peak=15),
        _keyframe_row("third_person", peak=15),
        _clip_row("first_person", start=10, end=20),
        _clip_row("third_person", start=10, end=20),
    ]
    paired = _dual_view_material_pairing_row("bundle", "hand_object_contact", rows, "dual_view_valid", "ok")
    assert paired["pairing_status"] == "dual_view_valid"
    assert paired["row_range_match"] is True

    rows[-1] = _clip_row("third_person", start=11, end=21)
    mismatched = _dual_view_material_pairing_row("bundle", "hand_object_contact", rows, "dual_view_valid", "ok")
    assert mismatched["pairing_status"] == "suspicious_needs_review"
    assert mismatched["reason"] == "first_third_keyclip_row_range_mismatch"


def test_final_preview_sampling_is_bounded_and_not_preflight_fps_driven() -> None:
    rows = [{"global_timestamp_us": str(1_000_000 + i * 200_000)} for i in range(2000)]
    selected = _preview_rows_from_window_sync(rows, output_fps=15.0, max_preview_sec=10.0)
    assert len(selected) == 150
    assert selected[0] == rows[0]
    assert selected[-1] == rows[-1]


def test_realtime_preview_rows_preserve_timestamp_duration() -> None:
    rows = [{"global_timestamp_us": str(1_000_000 + i * 500_000)} for i in range(21)]
    selected = _realtime_rows_from_window_sync(rows, output_fps=15.0)
    real_duration = _window_sync_duration_s(rows)
    encoded_duration = len(selected) / 15.0

    assert real_duration == 10.0
    assert len(selected) > len(rows)
    assert abs(encoded_duration - real_duration) < 0.1


def test_keyclip_repeat_counts_preserve_sparse_sync_duration() -> None:
    rows = [{"global_timestamp_us": str(1_000_000 + i * 1_000_000)} for i in range(5)]
    encoded_frames = sum(_window_sync_repeat_count(rows, index, output_fps=15.0) for index in range(len(rows)))
    encoded_duration = encoded_frames / 15.0

    assert abs(encoded_duration - 4.0) < 0.2


def test_actual_experiment_start_does_not_overwrite_raw_window_start() -> None:
    sync_rows = [{"global_timestamp_us": str(10_000_000 + i * 1_000_000)} for i in range(101)]
    row = {
        "start_sec": 100.0,
        "end_sec": 200.0,
        "start_global_timestamp_us": 10_000_000,
        "end_global_timestamp_us": 110_000_000,
        "supporting_third_segments": [{"start_sec": 120.0, "end_sec": 122.0}],
    }

    metadata = _window_start_boundary_metadata(row, sync_rows, [])

    assert metadata["raw_window_start_global_timestamp_us"] == 10_000_000
    assert metadata["actual_experiment_start_global_timestamp_us"] != metadata["raw_window_start_global_timestamp_us"]
    assert metadata["focus_preview_start_global_timestamp_us"] == metadata["actual_experiment_start_global_timestamp_us"]
    assert metadata["actual_start_status"] == "detected"
