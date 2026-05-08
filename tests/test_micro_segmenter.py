from __future__ import annotations

from key_action_indexer.clip_extractor import _row_view as _clip_row_view
from key_action_indexer.micro_quality_enrichment import enrich_micro_row_quality
from key_action_indexer.micro_segmenter import _row_view, choose_primary_interaction, generate_micro_segments
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

    micros = generate_micro_segments(
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
