from __future__ import annotations

from key_action_indexer.tracklet_annotations import (
    build_tracklet_annotations,
    prepare_tracklet_render_plan,
    select_tracklet_annotations,
)


def test_tracklet_annotations_link_by_view_label_and_interpolate_short_gap() -> None:
    rows = [
        {
            "source_view": "first_person",
            "alignment_time_sec": 1.0,
            "frame_index": 10,
            "global_time": "2026-05-14T09:00:01+08:00",
            "detections": [{"label": "sample_bottle", "confidence": 0.9, "bbox": [10, 10, 30, 30]}],
        },
        {
            "source_view": "first_person",
            "alignment_time_sec": 2.0,
            "frame_index": 20,
            "global_time": "2026-05-14T09:00:02+08:00",
            "detections": [{"label": "pipette", "confidence": 0.8, "bbox": [120, 20, 150, 50]}],
        },
        {
            "source_view": "first_person",
            "alignment_time_sec": 3.0,
            "frame_index": 30,
            "global_time": "2026-05-14T09:00:03+08:00",
            "detections": [{"label": "sample_bottle", "confidence": 0.7, "bbox": [30, 10, 50, 30]}],
        },
        {
            "source_view": "third_person",
            "alignment_time_sec": 1.0,
            "frame_index": 10,
            "detections": [{"label": "sample_bottle", "confidence": 0.88, "bbox": [300, 10, 320, 30]}],
        },
    ]

    annotations = build_tracklet_annotations(rows, labels=["sample_bottle"], max_missing_frames=1)

    first_person = [row for row in annotations if row["view"] == "first_person"]
    assert [row["source"] for row in first_person] == ["detected", "interpolated", "detected"]
    assert len({row["tracklet_id"] for row in first_person}) == 1
    assert all(row["object_track_id"] == row["tracklet_id"] for row in first_person)
    assert first_person[1]["frame_index"] == 20
    assert first_person[1]["bbox"] == [20.0, 10.0, 40.0, 30.0]
    assert first_person[1]["confidence"] == 0.595
    assert first_person[1]["interpolation"]["previous_frame_index"] == 10
    assert first_person[1]["interpolation"]["next_frame_index"] == 30

    third_person = [row for row in annotations if row["view"] == "third_person"]
    assert len(third_person) == 1
    assert third_person[0]["tracklet_id"] != first_person[0]["tracklet_id"]


def test_tracklet_annotations_split_when_iou_and_center_distance_fail() -> None:
    rows = [
        {
            "view": "first_person",
            "time_sec": 1.0,
            "frame_index": 1,
            "detections": [{"label": "tube", "confidence": 0.91, "bbox": [0, 0, 20, 20]}],
        },
        {
            "view": "first_person",
            "time_sec": 2.0,
            "frame_index": 2,
            "detections": [{"label": "tube", "confidence": 0.89, "bbox": [200, 200, 220, 220]}],
        },
    ]

    annotations = build_tracklet_annotations(
        rows,
        max_center_distance_px=25.0,
        min_iou=0.2,
        max_missing_frames=0,
    )

    assert len(annotations) == 2
    assert len({row["tracklet_id"] for row in annotations}) == 2
    assert all(row["source"] == "detected" for row in annotations)


def test_tracklet_select_and_render_plan_prepare_frame_overlays() -> None:
    rows = [
        {
            "source_view": "first_person",
            "alignment_time_sec": 4.0,
            "frame_index": 40,
            "detections": [{"label": "beaker", "confidence": 0.8, "bbox": [10, 20, 50, 80]}],
        },
        {
            "source_view": "first_person",
            "alignment_time_sec": 5.0,
            "frame_index": 50,
            "detections": [],
        },
        {
            "source_view": "first_person",
            "alignment_time_sec": 6.0,
            "frame_index": 60,
            "detections": [{"label": "beaker", "confidence": 0.82, "bbox": [20, 20, 60, 80]}],
        },
    ]
    annotations = build_tracklet_annotations(rows, max_missing_frames=1)

    selected = select_tracklet_annotations(
        annotations,
        view="first_person",
        labels=["beaker"],
        time_sec=5.0,
        time_tolerance_sec=0.01,
    )
    assert len(selected) == 1
    assert selected[0]["source"] == "interpolated"

    without_interpolated = select_tracklet_annotations(
        annotations,
        view="first_person",
        labels=["beaker"],
        time_sec=5.0,
        time_tolerance_sec=0.01,
        include_interpolated=False,
    )
    assert without_interpolated == []

    plan = prepare_tracklet_render_plan(
        annotations,
        view="first_person",
        labels=["beaker"],
        time_sec=5.0,
        time_tolerance_sec=0.01,
    )

    assert len(plan) == 1
    assert plan[0]["source"] == "tracklet_annotations"
    assert plan[0]["frame_index"] == 50
    assert len(plan[0]["annotations"]) == 1
    overlay = plan[0]["annotations"][0]
    assert overlay["source"] == "interpolated"
    assert overlay["bbox"] == [15.0, 20.0, 55.0, 80.0]
    assert overlay["style"]["dash"] == [4, 3]


def test_tracklet_summary_stabilizes_single_frame_bbox_jump() -> None:
    rows = [
        {
            "view": "first_person",
            "time_sec": 1.0,
            "detections": [{"label": "sample_bottle", "confidence": 0.9, "bbox": [10, 10, 40, 70]}],
        },
        {
            "view": "first_person",
            "time_sec": 1.2,
            "detections": [{"label": "sample_bottle", "confidence": 0.88, "bbox": [300, 220, 345, 295]}],
        },
        {
            "view": "first_person",
            "time_sec": 1.4,
            "detections": [{"label": "sample_bottle", "confidence": 0.91, "bbox": [14, 12, 44, 72]}],
        },
    ]

    summary = build_tracklet_annotations(rows, target_labels=["sample_bottle"], include_hands=False)

    track = summary["tracklets"][0]
    assert track["stabilized_outlier_count"] == 1
    repaired = next(point for point in track["points"] if point.get("stabilization"))
    assert repaired["source"] == "stabilized_outlier"
    assert repaired["bbox"] == [12.0, 11.0, 42.0, 71.0]
    assert repaired["raw_bbox"] == [300.0, 220.0, 345.0, 295.0]
    assert track["quality"]["stabilized_outlier_count"] == 1
