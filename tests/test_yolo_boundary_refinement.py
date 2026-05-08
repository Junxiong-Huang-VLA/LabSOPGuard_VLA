from __future__ import annotations

import pytest

from key_action_indexer.config import DetectorConfig
from key_action_indexer.pipeline import _refine_yolo_detected_segments
from key_action_indexer.schemas import DetectedSegment, SessionManifest, SessionVideos, VideoSource


def _manifest() -> SessionManifest:
    third = VideoSource(
        name="third_person",
        path="third.mp4",
        start_time="2026-04-24T16:57:18+08:00",
        fps=15.0,
    )
    first = VideoSource(
        name="first_person",
        path="first.mp4",
        start_time="2026-04-24T16:57:18+08:00",
        fps=15.0,
    )
    return SessionManifest(
        session_id="weighing_session",
        session_start_time="2026-04-24T16:57:18+08:00",
        videos=SessionVideos(third_person=third, first_person=first),
        config={"experiment_title": "solid weighing experiment"},
        output_dir="out",
    )


def _segment() -> DetectedSegment:
    return DetectedSegment(
        segment_id="seg_000001",
        start_sec=0.0,
        end_sec=79.967,
        duration_sec=79.967,
        global_start_time="2026-04-24T16:57:18+08:00",
        global_end_time="2026-04-24T16:58:37.967000+08:00",
        avg_motion_score=0.1,
        avg_active_score=0.44,
        start_reason="active_score_above_threshold",
        end_reason="active_score_below_threshold",
    )


def test_yolo_boundary_refinement_ignores_static_and_transfer_noise_for_weighing() -> None:
    config = DetectorConfig(
        sample_fps=2.0,
        parent_sample_fps=2.0,
        buffer_sec=1.0,
        min_segment_duration_sec=2.0,
    )
    rows = [
        {"local_time_sec": 0.0, "label_counts": {"paper": 2, "lab_coat": 1}, "active_score": 0.25, "motion_score": 0.0},
        {
            "local_time_sec": 13.333333,
            "label_counts": {"pipette": 5, "gloved_hand": 2},
            "hand_object_interactions": [{"object_label": "pipette", "score": 0.9}],
            "active_score": 0.8,
            "motion_score": 0.9,
        },
        {"local_time_sec": 16.0, "label_counts": {"paper": 2, "lab_coat": 2}, "active_score": 0.25, "motion_score": 0.0},
        {
            "local_time_sec": 16.533333,
            "label_counts": {"spatula": 1, "balance": 1, "sample_bottle": 1},
            "active_score": 0.62,
            "motion_score": 0.0,
        },
        {
            "local_time_sec": 17.6,
            "label_counts": {"gloved_hand": 2, "spatula": 1},
            "hand_object_interactions": [{"object_label": "spatula", "score": 1.0}],
            "active_score": 1.0,
            "motion_score": 1.0,
        },
        {
            "local_time_sec": 74.666667,
            "label_counts": {"gloved_hand": 2, "balance": 1},
            "hand_object_interactions": [{"object_label": "balance", "score": 0.42}],
            "active_score": 0.48,
            "motion_score": 0.42,
        },
        {"local_time_sec": 79.466667, "label_counts": {"paper": 1, "lab_coat": 1}, "active_score": 0.2, "motion_score": 0.0},
    ]

    refined = _refine_yolo_detected_segments(_manifest(), [_segment()], rows, config, duration_sec=79.967)
    segment = refined[0]

    assert segment.start_sec == pytest.approx(16.0, abs=0.001)
    assert segment.end_sec == pytest.approx(18.6, abs=0.001)
    assert segment.global_start_time == "2026-04-24T16:57:34+08:00"
    assert segment.boundary_source == "yolo_physical_evidence_window"
    assert segment.boundary_confidence > 0.6


def test_yolo_boundary_refinement_splits_parent_window_by_physical_evidence_clusters() -> None:
    config = DetectorConfig(
        sample_fps=2.0,
        parent_sample_fps=2.0,
        buffer_sec=1.0,
        merge_gap_sec=5.0,
        min_segment_duration_sec=2.0,
    )
    rows = [
        {"local_time_sec": 16.0, "label_counts": {"paper": 2, "lab_coat": 2}, "active_score": 0.25, "motion_score": 0.0},
        {
            "local_time_sec": 16.533333,
            "label_counts": {"spatula": 1, "balance": 1, "sample_bottle": 1},
            "active_score": 0.62,
            "motion_score": 0.0,
        },
        {
            "local_time_sec": 17.6,
            "label_counts": {"gloved_hand": 2, "spatula": 1},
            "hand_object_interactions": [{"object_label": "spatula", "score": 1.0}],
            "active_score": 1.0,
            "motion_score": 1.0,
        },
        {"local_time_sec": 30.4, "label_counts": {"spatula": 1, "beaker": 1}, "active_score": 0.58, "motion_score": 0.2},
        {"local_time_sec": 32.0, "label_counts": {"spatula": 1, "beaker": 1}, "active_score": 0.58, "motion_score": 0.2},
        {
            "local_time_sec": 44.8,
            "label_counts": {"gloved_hand": 1, "spatula": 1, "beaker": 1},
            "hand_object_interactions": [{"object_label": "spatula", "score": 0.8}],
            "active_score": 0.9,
            "motion_score": 0.4,
        },
        {
            "local_time_sec": 46.933333,
            "label_counts": {"gloved_hand": 1, "spatula": 1, "beaker": 1},
            "hand_object_interactions": [{"object_label": "beaker", "score": 0.7}],
            "active_score": 0.8,
            "motion_score": 0.3,
        },
    ]

    refined = _refine_yolo_detected_segments(_manifest(), [_segment()], rows, config, duration_sec=79.967)

    assert [segment.segment_id for segment in refined] == ["seg_000001", "seg_000002", "seg_000003"]
    assert [(round(segment.start_sec, 3), round(segment.end_sec, 3)) for segment in refined] == [
        (16.0, 18.6),
        (29.867, 33.0),
        (44.267, 47.933),
    ]
    assert all(segment.boundary_source == "yolo_physical_evidence_cluster" for segment in refined)
    assert [segment.boundary_support_count for segment in refined] == [2, 2, 2]
