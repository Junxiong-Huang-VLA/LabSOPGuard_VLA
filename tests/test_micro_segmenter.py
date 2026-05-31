from __future__ import annotations

from key_action_indexer.clip_extractor import _row_view as _clip_row_view
from key_action_indexer.micro_quality_enrichment import enrich_micro_row_quality
from key_action_indexer.micro_segmenter import (
    InteractionState,
    _draw_evidence_boxes,
    _max_micro_segments_per_parent,
    _micro_asset_worker_count,
    _micro_keyframe_box_mode,
    _micro_keyframe_draw_boxes_enabled,
    _micro_keyframe_roles,
    _micro_clip_views,
    _row_view,
    choose_primary_interaction,
    generate_micro_segments,
)
from key_action_indexer.schemas import (
    CVDetectionSummary,
    ClipReference,
    KeyActionSegment,
    MicroSegmentConfig,
    SegmentIndexInfo,
    SessionManifest,
    TextDescription,
)


def test_choose_primary_interaction_prefers_stronger_hand_object_candidate() -> None:
    candidates = [
        {"object_label": "balance", "score": 0.2, "overlap_ratio": 0.1, "distance_score": 0.2},
        {"object_label": "container", "score": 0.75, "overlap_ratio": 0.5, "distance_score": 0.8},
    ]

    result = choose_primary_interaction(candidates)

    assert result is not None
    assert result["object_label"] == "container"


def test_row_view_keeps_top_as_third_person_and_bottom_as_first_person() -> None:
    assert _row_view({"camera": "top"}) == "third_person"
    assert _row_view({"camera": "top_view"}) == "third_person"
    assert _row_view({"camera": "bottom"}) == "first_person"
    assert _row_view({"camera": "bottom_view"}) == "first_person"
    assert _clip_row_view({"camera": "top_view"}) == "third_person"
    assert _clip_row_view({"camera": "bottom_view"}) == "first_person"


def test_fast_deferred_asset_mode_uses_dual_view_clip_and_worker_defaults(monkeypatch) -> None:
    for name in (
        "KEY_ACTION_FAST_LOCATE_ONLY",
        "KEY_ACTION_DEFER_SEGMENT_ASSETS",
        "KEY_ACTION_MICRO_ASSET_WORKERS",
        "KEY_ACTION_FAST_LOCATE_MICRO_ASSET_WORKERS",
        "KEY_ACTION_MICRO_CLIP_VIEWS",
        "KEY_ACTION_FAST_LOCATE_MICRO_CLIP_VIEWS",
        "KEY_ACTION_MAX_MICROS_PER_SEGMENT",
        "KEY_ACTION_FAST_LOCATE_MAX_MICROS_PER_SEGMENT",
        "KEY_ACTION_MICRO_KEYFRAME_ROLES",
        "KEY_ACTION_FAST_LOCATE_KEYFRAME_ROLES",
        "KEY_ACTION_MICRO_KEYFRAME_BOX_MODE",
        "KEY_ACTION_FAST_LOCATE_KEYFRAME_BOX_MODE",
        "KEY_ACTION_MICRO_KEYFRAME_DRAW_BOXES",
        "KEY_ACTION_FAST_LOCATE_KEYFRAME_DRAW_BOXES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEY_ACTION_DEFER_SEGMENT_ASSETS", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_MICRO_ASSET_WORKERS", "8")

    assert _micro_asset_worker_count() == 8
    assert _micro_clip_views() == {"third_person", "first_person"}
    assert _micro_keyframe_roles() == {"peak"}
    assert _micro_keyframe_box_mode() == "strict"
    assert _micro_keyframe_draw_boxes_enabled() is True
    assert _max_micro_segments_per_parent() == 6


def test_strict_keyframe_boxes_scale_to_original_frame() -> None:
    import numpy as np

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = InteractionState(
        frame_index=360,
        local_time_sec=12.0,
        global_time="2026-04-29T17:25:12+08:00",
        source_view="third_person",
        hand_detected=True,
        objects_near_hand=["paper"],
        primary_object="paper",
        interaction_type="hand_paper_contact",
        interaction_score=1.0,
        hand_object_distance=0.0,
        bbox_overlap=0.1,
        detected_objects=["gloved_hand", "paper"],
        raw_row={
            "frame_width": 800,
            "frame_height": 450,
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "paper",
                    "score": 1.0,
                    "hand_bbox": [100, 50, 150, 100],
                    "object_bbox": [200, 100, 300, 160],
                }
            ],
        },
    )

    rendered = _draw_evidence_boxes(
        frame.copy(),
        state,
        "paper",
        source_view="third_person",
        target_local_time_sec=12.0,
        strict=True,
    )
    # 800x450 evidence coordinates should scale to 1280x720 display pixels.
    assert rendered[160, 320, 2] > 100
    assert rendered[80, 160, 1] > 100

    wrong_view = _draw_evidence_boxes(
        frame.copy(),
        state,
        "paper",
        source_view="first_person",
        target_local_time_sec=12.0,
        strict=True,
    )
    assert int(wrong_view.sum()) == 0


def test_generate_micro_segments_writes_per_view_micro_assets(tmp_path, monkeypatch) -> None:
    for name in (
        "KEY_ACTION_MICRO_CLIP_VIEWS",
        "KEY_ACTION_MICRO_KEYFRAME_ROLES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KEY_ACTION_MICRO_CLIP_VIEWS", "all")
    monkeypatch.setenv("KEY_ACTION_MICRO_KEYFRAME_ROLES", "peak")

    manifest = SessionManifest.from_dict(
        {
            "session_id": "s1",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {"path": str(tmp_path / "third.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
                "first_person": {"path": str(tmp_path / "first.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
            },
            "output_dir": str(tmp_path),
        }
    )
    parent = KeyActionSegment(
        session_id="s1",
        segment_id="seg_000020",
        global_start_time="2026-04-29T17:25:00+08:00",
        global_end_time="2026-04-29T17:25:05+08:00",
        duration_sec=5.0,
        third_person=ClipReference("third.mp4", "third_clip.mp4", 0.0, 5.0),
        first_person=ClipReference("first.mp4", "first_clip.mp4", 0.0, 5.0),
        cv_detection=CVDetectionSummary(0.8, 0.8, "start", "end"),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo("", "", ""),
    )
    rows = []
    for view in ("third_person", "first_person"):
        for index, local_time in enumerate((1.0, 1.5, 2.0), start=1):
            rows.append(
                {
                    "source_view": view,
                    "local_time_sec": local_time,
                    "frame_index": int(local_time * 30),
                    "frame_width": 640,
                    "frame_height": 480,
                    "detections": [
                        {"label": "gloved_hand", "confidence": 0.9, "bbox": [10 + index, 10, 40 + index, 40]},
                        {"label": "paper", "confidence": 0.86, "bbox": [35 + index, 20, 95 + index, 55]},
                    ],
                    "hand_object_interactions": [
                        {
                            "hand_label": "gloved_hand",
                            "object_label": "paper",
                            "score": 0.76,
                            "iou": 0.08,
                            "hand_bbox": [10 + index, 10, 40 + index, 40],
                            "object_bbox": [35 + index, 20, 95 + index, 55],
                        }
                    ],
                }
            )

    micros, _dedup_log = generate_micro_segments(
        manifest=manifest,
        key_segments=[parent],
        yolo_frame_rows=rows,
        utterances=[],
        clips_dir=tmp_path / "clips",
        keyframes_dir=tmp_path / "keyframes",
        config=MicroSegmentConfig(),
        dry_run=True,
    )

    assert len(micros) == 1
    micro = micros[0]
    assert micro.first_person is not None
    assert micro.first_person.clip_path and micro.first_person.clip_path.endswith("_first_person.mp4")
    assert micro.third_person.clip_path and micro.third_person.clip_path.endswith("_third_person.mp4")
    bindings = {binding["view"]: binding for binding in micro.asset_bindings}
    assert set(bindings) == {"first_person", "third_person"}
    assert "first_person" in bindings["first_person"]["keyframe_path"]
    assert "third_person" in bindings["third_person"]["keyframe_path"]
    assert parent.micro_segments[0]["first_person_clip"]
    assert parent.micro_segments[0]["third_person_clip"]


def test_generate_micro_segments_records_secondary_object_audit(tmp_path) -> None:
    manifest = SessionManifest.from_dict(
        {
            "session_id": "s1",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {"path": str(tmp_path / "third.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
            },
            "output_dir": str(tmp_path),
        }
    )
    parent = KeyActionSegment(
        session_id="s1",
        segment_id="seg_000010",
        global_start_time="2026-04-29T17:25:00+08:00",
        global_end_time="2026-04-29T17:25:05+08:00",
        duration_sec=5.0,
        third_person=ClipReference("third.mp4", "third_clip.mp4", 0.0, 5.0),
        first_person=None,
        cv_detection=CVDetectionSummary(0.8, 0.8, "start", "end"),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo("", "", ""),
    )
    rows = []
    for index, local_time in enumerate([1.0, 1.5, 2.0], start=1):
        hand_bbox = [10 + index, 10, 40 + index, 40]
        paper_bbox = [35 + index, 20, 95 + index, 55]
        balance_bbox = [90, 35, 170, 95]
        rows.append(
            {
                "source_view": "third_person",
                "local_time_sec": local_time,
                "frame_index": int(local_time * 30),
                "frame_width": 640,
                "frame_height": 480,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.9, "bbox": hand_bbox},
                    {"label": "paper", "confidence": 0.86, "bbox": paper_bbox},
                    {"label": "balance", "confidence": 0.8, "bbox": balance_bbox},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": "paper",
                        "score": 0.76,
                        "iou": 0.08,
                        "hand_bbox": hand_bbox,
                        "object_bbox": paper_bbox,
                    },
                    {
                        "hand_label": "gloved_hand",
                        "object_label": "balance",
                        "score": 0.68,
                        "iou": 0.04,
                        "hand_bbox": hand_bbox,
                        "object_bbox": balance_bbox,
                    },
                ],
            }
        )

    micros, _dedup_log = generate_micro_segments(
        manifest=manifest,
        key_segments=[parent],
        yolo_frame_rows=rows,
        utterances=[],
        clips_dir=tmp_path / "clips",
        keyframes_dir=tmp_path / "keyframes",
        config=MicroSegmentConfig(),
        dry_run=True,
    )

    assert len(micros) == 1
    micro = micros[0]
    assert micro.interaction.primary_object == "paper"
    assert micro.interaction.secondary_objects == ["balance"]
    assert "hand-paper+balance" in micro.interaction.secondary_actions
    assert micro.window_audit["interaction_frame_count"] == 3
    assert micro.window_audit["target_object_support"]["interaction_frame_count"] == 3
    assert micro.window_audit["secondary_object_support"][0]["object"] == "balance"
    assert micro.window_audit["secondary_object_support"][0]["interaction_frame_count"] == 3
    assert parent.micro_segments[0]["secondary_objects"] == ["balance"]


def test_generate_micro_segments_backfills_low_signal_yolo_parent_coverage(tmp_path) -> None:
    manifest = SessionManifest.from_dict(
        {
            "session_id": "s1",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {"path": str(tmp_path / "third.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
                "first_person": {"path": str(tmp_path / "first.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
            },
            "output_dir": str(tmp_path),
        }
    )
    parent = KeyActionSegment(
        session_id="s1",
        segment_id="seg_000001",
        global_start_time="2026-04-29T17:25:00+08:00",
        global_end_time="2026-04-29T17:25:05+08:00",
        duration_sec=5.0,
        third_person=ClipReference("third.mp4", "third_clip.mp4", 0.0, 5.0),
        first_person=ClipReference("first.mp4", "first_clip.mp4", 0.0, 5.0),
        cv_detection=CVDetectionSummary(0.8, 0.8, "start", "end"),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo("", "", ""),
    )
    rows = [
        {
            "source_view": "first_person",
            "local_time_sec": 2.0,
            "frame_index": 60,
            "detections": [
                {"label": "gloved_hand", "confidence": 0.9, "bbox": [5, 5, 25, 25]},
                {"label": "balance", "confidence": 0.9, "bbox": [20, 20, 80, 80]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "balance",
                    "score": 0.42,
                    "iou": 0.02,
                    "hand_bbox": [5, 5, 25, 25],
                    "object_bbox": [20, 20, 80, 80],
                }
            ],
        }
    ]

    micros, _dedup_log = generate_micro_segments(
        manifest=manifest,
        key_segments=[parent],
        yolo_frame_rows=rows,
        utterances=[],
        clips_dir=tmp_path / "clips",
        keyframes_dir=tmp_path / "keyframes",
        config=MicroSegmentConfig(),
        dry_run=True,
    )

    assert len(micros) == 1
    assert micros[0].interaction.primary_object == "balance"
    assert "coverage_backfill_candidate" in micros[0].quality.warnings
    assert "single_frame_coverage_candidate" in micros[0].quality.warnings
    assert micros[0].manual_correction_note == "auto_coverage_backfill_from_yolo_parent_rows"
    assert micros[0].evidence["coverage_backfill"] is True
    assert micros[0].evidence["coverage_signal_grade"] == "single_frame_yolo_candidate"
    assert micros[0].evidence["coverage_evidence_frame_count"] == 1
    assert micros[0].evidence["coverage_bbox_frame_count"] == 1
    assert parent.micro_segments[0]["micro_segment_id"] == micros[0].micro_segment_id


def test_generate_micro_segments_backfills_uncovered_object_when_parent_has_micro(tmp_path) -> None:
    manifest = SessionManifest.from_dict(
        {
            "session_id": "s1",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {"path": str(tmp_path / "third.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
            },
            "output_dir": str(tmp_path),
        }
    )
    parent = KeyActionSegment(
        session_id="s1",
        segment_id="seg_000002",
        global_start_time="2026-04-29T17:25:00+08:00",
        global_end_time="2026-04-29T17:25:08+08:00",
        duration_sec=8.0,
        third_person=ClipReference("third.mp4", "third_clip.mp4", 0.0, 8.0),
        first_person=None,
        cv_detection=CVDetectionSummary(0.8, 0.8, "start", "end"),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo("", "", ""),
    )
    rows = []
    for local_time in (1.0, 1.5, 2.0):
        rows.append(
            {
                "source_view": "third_person",
                "local_time_sec": local_time,
                "frame_index": int(local_time * 30),
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.9, "bbox": [10, 10, 45, 45]},
                    {"label": "paper", "confidence": 0.86, "bbox": [38, 18, 95, 58]},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": "paper",
                        "score": 0.76,
                        "iou": 0.08,
                        "hand_bbox": [10, 10, 45, 45],
                        "object_bbox": [38, 18, 95, 58],
                    }
                ],
            }
        )
    rows.append(
        {
            "source_view": "third_person",
            "local_time_sec": 5.0,
            "frame_index": 150,
            "frame_width": 640,
            "frame_height": 480,
            "detections": [
                {"label": "gloved_hand", "confidence": 0.83, "bbox": [430, 106, 554, 214]},
                {"label": "beaker", "confidence": 0.43, "bbox": [383, 200, 502, 338]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "beaker",
                    "score": 0.36,
                    "iou": 0.02,
                    "hand_bbox": [430, 106, 554, 214],
                    "object_bbox": [383, 200, 502, 338],
                }
            ],
        }
    )

    micros, _dedup_log = generate_micro_segments(
        manifest=manifest,
        key_segments=[parent],
        yolo_frame_rows=rows,
        utterances=[],
        clips_dir=tmp_path / "clips",
        keyframes_dir=tmp_path / "keyframes",
        config=MicroSegmentConfig(),
        dry_run=True,
    )

    assert [micro.interaction.primary_object for micro in micros] == ["paper", "beaker"]
    assert "coverage_backfill_candidate" not in micros[0].quality.warnings
    assert "coverage_backfill_candidate" in micros[1].quality.warnings
    assert micros[1].evidence["coverage_backfill"] is True
    assert micros[1].yolo_evidence[0]["frame_width"] == 640
    assert micros[1].yolo_evidence[0]["frame_height"] == 480
    assert micros[1].yolo_evidence[0]["time_sec"] == 5.0


def test_enrich_micro_row_quality_marks_continuity_and_clears_low_signal() -> None:
    row = {
        "micro_segment_id": "m1",
        "interaction": {
            "primary_object": "beaker",
            "max_interaction_score": 0.7,
            "avg_interaction_score": 0.65,
            "evidence_frame_indices": [1, 2],
        },
        "quality": {"warnings": ["coverage_backfill_candidate", "low_signal_yolo_candidate"]},
        "keyframes": {
            "contact_frame": "contact.jpg",
            "peak_frame": "peak.jpg",
            "release_frame": "release.jpg",
        },
        "evidence": {"limitations": []},
        "yolo_evidence": [
            {
                "interaction_score": 0.6,
                "hand_object_interactions": [
                    {"object_label": "beaker", "hand_bbox": [0, 0, 10, 10], "object_bbox": [9, 9, 20, 20], "score": 0.6}
                ],
            },
            {
                "interaction_score": 0.7,
                "hand_object_interactions": [
                    {"object_label": "beaker", "hand_bbox": [1, 1, 10, 10], "object_bbox": [8, 8, 20, 20], "score": 0.7}
                ],
            },
        ],
    }

    enriched, changed = enrich_micro_row_quality(row)

    assert changed is True
    assert enriched["evidence"]["coverage_signal_grade"] == "physical_continuity_candidate"
    assert enriched["evidence"]["process_evidence_role"] == "strong_process_evidence"
    assert enriched["evidence"]["process_eligible"] is True
    assert enriched["evidence"]["retrieval_priority_bucket"] == "high_physical_continuity"
    assert enriched["evidence"]["coverage_bbox_frame_count"] == 2
    assert "low_signal_yolo_candidate" not in enriched["quality"]["warnings"]
    assert enriched["evidence"]["keyframe_selection_basis"] == "contact_peak_release_from_physical_yolo_frames"


def test_enrich_micro_row_quality_marks_single_frame_as_retrieval_candidate() -> None:
    row = {
        "micro_segment_id": "m1",
        "interaction": {
            "primary_object": "balance",
            "max_interaction_score": 0.7,
            "avg_interaction_score": 0.7,
            "evidence_frame_indices": [10],
        },
        "quality": {"warnings": []},
        "evidence": {"limitations": []},
        "yolo_evidence": [
            {
                "interaction_score": 0.7,
                "hand_object_interactions": [
                    {"object_label": "balance", "hand_bbox": [0, 0, 10, 10], "object_bbox": [9, 9, 20, 20], "score": 0.7}
                ],
            }
        ],
    }

    enriched, changed = enrich_micro_row_quality(row)

    assert changed is True
    assert enriched["evidence"]["coverage_signal_grade"] == "single_frame_yolo_candidate"
    assert enriched["evidence"]["process_evidence_role"] == "retrieval_candidate"
    assert enriched["evidence"]["process_eligible"] is False
    assert "retrieval_candidate_only" in enriched["quality"]["warnings"]
    assert any("not eligible for strong process claims" in item for item in enriched["evidence"]["limitations"])
