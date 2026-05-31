from __future__ import annotations

from key_action_indexer.config import DetectorConfig
from key_action_indexer.pipeline import (
    _fast_locate_max_total_fine_scan_sec,
    _fast_locate_fine_window_segments_from_yolo_rows,
    _filter_short_weak_experiment_episodes,
    _split_and_limit_fast_locate_windows,
    _yolo_planned_batch_size,
)
from key_action_indexer.schemas import DetectedSegment, SessionManifest


def test_fast_locate_budget_limit_is_default(monkeypatch):
    monkeypatch.delenv("KEY_ACTION_FAST_LOCATE_KEEP_ALL_FINE_WINDOWS", raising=False)
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_FINE_WINDOW_SEC", "60")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC", "60")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_BALANCED_FINE_WINDOW_SELECTION", "0")

    windows = [
        {"start_sec": 0, "end_sec": 60, "score": 1.0},
        {"start_sec": 120, "end_sec": 180, "score": 0.9},
        {"start_sec": 240, "end_sec": 300, "score": 0.8},
    ]

    selected = _split_and_limit_fast_locate_windows(windows)

    assert len(selected) == 1
    assert selected[0]["start_sec"] == 0


def test_fast_locate_keep_all_windows_is_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_KEEP_ALL_FINE_WINDOWS", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_FINE_WINDOW_SEC", "60")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC", "60")

    windows = [
        {"start_sec": 0, "end_sec": 60, "score": 1.0},
        {"start_sec": 120, "end_sec": 180, "score": 0.9},
    ]

    selected = _split_and_limit_fast_locate_windows(windows)

    assert [(item["start_sec"], item["end_sec"]) for item in selected] == [(0, 60), (120, 180)]


def test_fast_locate_adaptive_budget_scales_from_activity_window_count(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC", "60")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_ADAPTIVE_TOTAL_FINE_SCAN_BUDGET", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_FINE_WINDOW_SEC", "30")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FINE_SCAN_BUDGET_PER_WINDOW_SEC", "30")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_TOTAL_FINE_SCAN_SEC", "60")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_ADAPTIVE_TOTAL_FINE_SCAN_SEC", "300")

    assert _fast_locate_max_total_fine_scan_sec(2) == 60
    assert _fast_locate_max_total_fine_scan_sec(5) == 150


def test_yolo_planned_batch_size_uses_scan_role_overrides(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_YOLO_BATCH_SIZE", "16")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_YOLO_BATCH_SIZE", "4")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FINE_YOLO_BATCH_SIZE", "8")

    assert _yolo_planned_batch_size("long_video_coarse") == 4
    assert _yolo_planned_batch_size("micro_refine") == 8


def test_proxy_fine_windows_prioritize_pre_action_lookback(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FINE_SEED_REQUIRE_INTERACTION", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PROXY_FINE_SEED_REQUIRE_INTERACTION", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PROXY_FINE_SEED_CLUSTER_GAP_SEC", "45")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PROXY_FINE_WINDOW_PRE_PAD_SEC", "120")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PROXY_FINE_WINDOW_POST_PAD_SEC", "45")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_FINE_WINDOW_SEC", "90")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MAX_TOTAL_FINE_SCAN_SEC", "90")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_BALANCED_FINE_WINDOW_SELECTION", "1")

    manifest = SessionManifest.from_dict(
        {
            "session_id": "proxy_window_test",
            "session_start_time": "2026-05-20T00:00:00+08:00",
            "videos": {
                "third_person": {
                    "view": "third_person",
                    "path": "third.mp4",
                    "start_time": "2026-05-20T00:00:00+08:00",
                }
            },
            "output_dir": "out",
        }
    )
    rows = [
        {
            "source_view": "third_person",
            "video_path": r"D:\run\analysis_proxy\third_person\proxy.mp4",
            "source_fps": 0.1,
            "local_time_sec": 2910.0,
            "active_score": 1.0,
            "detections": [{"label": "paper"}, {"label": "balance"}],
            "hand_object_interactions": [],
        },
        {
            "source_view": "third_person",
            "video_path": r"D:\run\analysis_proxy\third_person\proxy.mp4",
            "source_fps": 0.1,
            "local_time_sec": 2940.0,
            "active_score": 1.0,
            "detections": [{"label": "gloved_hand"}, {"label": "paper"}],
            "hand_object_interactions": [{"hand_label": "gloved_hand", "object_label": "paper", "score": 0.4}],
        },
    ]

    selected = _fast_locate_fine_window_segments_from_yolo_rows(
        manifest,
        [],
        rows,
        DetectorConfig(long_video_two_stage_sampling=True, long_video_stage1_sample_fps=1 / 60),
    )

    assert [(round(segment.start_sec, 3), round(segment.end_sec, 3)) for segment in selected] == [(2820.0, 2910.0)]
    assert "proxy_lookback_applied=True" in selected[0].decision_trace


def _episode_manifest(*, dual_view: bool = True) -> SessionManifest:
    videos = {
        "third_person": {
            "view": "third_person",
            "path": "third.mp4",
            "start_time": "2026-05-20T00:00:00+08:00",
        }
    }
    if dual_view:
        videos["first_person"] = {
            "view": "first_person",
            "path": "first.mp4",
            "start_time": "2026-05-20T00:00:00+08:00",
        }
    return SessionManifest.from_dict(
        {
            "session_id": "episode_filter_test",
            "session_start_time": "2026-05-20T00:00:00+08:00",
            "videos": videos,
            "output_dir": "out",
        }
    )


def _detected(
    segment_id: str,
    start: float,
    end: float,
    *,
    interactions: int,
    support: int,
    confidence: float,
    source_views: list[str] | None = None,
) -> DetectedSegment:
    views = source_views or ["first_person", "third_person"]
    return DetectedSegment(
        segment_id=segment_id,
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        global_start_time="2026-05-20T00:00:00+08:00",
        global_end_time="2026-05-20T00:00:00+08:00",
        avg_motion_score=confidence,
        avg_active_score=confidence,
        start_reason="test",
        end_reason="test",
        detector_backend="yolo_interaction",
        detector_source_view="global_multiview" if len(views) > 1 else views[0],
        yolo_interaction_count=interactions,
        boundary_support_count=support,
        boundary_confidence=confidence,
        final_score=confidence,
        decision_trace=[f"source_views={','.join(views)}"],
    )


def test_short_weak_episode_fragment_merges_before_filter(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FILTER_SHORT_WEAK_EPISODES", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_EXPERIMENT_EPISODE_SEC", "20")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_INTERACTIONS", "8")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_SUPPORT_COUNT", "3")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_CONFIDENCE", "0.65")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_SHORT_WEAK_MERGE_GAP_SEC", "10")

    kept, summary = _filter_short_weak_experiment_episodes(
        _episode_manifest(),
        [
            _detected("strong_a", 0, 40, interactions=12, support=12, confidence=0.9),
            _detected("weak_bridge", 44, 49, interactions=1, support=1, confidence=0.2),
            _detected("strong_b", 53, 90, interactions=11, support=11, confidence=0.88),
        ],
        expected_count=None,
        dry_run=False,
    )

    assert len(kept) == 1
    assert kept[0].start_sec == 0
    assert kept[0].end_sec == 90
    assert summary["short_weak_merge"]["merged_count"] == 2
    assert summary["short_weak_merge"]["weak_fragment_segment_ids"] == ["weak_bridge"]
    assert summary["removed_count"] == 0


def test_high_support_short_episode_becomes_candidate_action_window(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FILTER_SHORT_WEAK_EPISODES", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_EXPERIMENT_EPISODE_SEC", "20")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_INTERACTIONS", "8")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_SUPPORT_COUNT", "3")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_SHORT_EPISODE_CONFIDENCE", "0.65")

    kept, summary = _filter_short_weak_experiment_episodes(
        _episode_manifest(),
        [_detected("supported_short", 100, 110, interactions=1, support=5, confidence=0.9)],
        expected_count=None,
        dry_run=False,
    )

    assert kept == []
    assert summary["removed_count"] == 1
    assert summary["candidate_action_window_count"] == 1
    assert summary["candidate_action_windows"][0]["candidate_status"] == "candidate_action_window"
    assert "too_short_action_window" in summary["candidate_action_windows"][0]["candidate_reasons"]


def test_adjacent_action_windows_merge_into_long_official_episode(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FILTER_SHORT_WEAK_EPISODES", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_EXPERIMENT_EPISODE_SEC", "20")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_SHORT_WEAK_MERGE_GAP_SEC", "10")

    kept, summary = _filter_short_weak_experiment_episodes(
        _episode_manifest(),
        [
            _detected("frag_a", 0, 12, interactions=5, support=5, confidence=0.9),
            _detected("frag_b", 15, 30, interactions=5, support=5, confidence=0.9),
        ],
        expected_count=None,
        dry_run=False,
    )

    assert len(kept) == 1
    assert kept[0].start_sec == 0
    assert kept[0].end_sec == 30
    assert summary["official_episode_count"] == 1
    assert summary["candidate_action_window_count"] == 0
    assert summary["action_window_gap_merge"]["merged_count"] == 1


def test_single_view_long_segment_is_candidate_not_official(monkeypatch):
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_FILTER_SHORT_WEAK_EPISODES", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MIN_EXPERIMENT_EPISODE_SEC", "20")

    kept, summary = _filter_short_weak_experiment_episodes(
        _episode_manifest(),
        [
            _detected(
                "single_view_long",
                100,
                145,
                interactions=12,
                support=12,
                confidence=0.92,
                source_views=["third_person"],
            )
        ],
        expected_count=None,
        dry_run=False,
    )

    assert kept == []
    assert summary["formal_results_allowed"] is False
    assert summary["candidate_action_window_count"] == 1
    assert summary["candidate_action_windows"][0]["single_view_candidate"] is True
    assert "single_view_candidate" in summary["candidate_action_windows"][0]["candidate_reasons"]
