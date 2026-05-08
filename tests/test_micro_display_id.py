from __future__ import annotations

from pathlib import Path

from key_action_indexer.micro_segmenter import generate_micro_segments
from key_action_indexer.schemas import CVDetectionSummary, ClipReference, KeyActionSegment, MicroSegmentConfig, SegmentIndexInfo, SessionManifest, TextDescription


def _parent() -> KeyActionSegment:
    return KeyActionSegment(
        session_id="s1",
        segment_id="seg_000001",
        global_start_time="2026-04-29T17:25:00+08:00",
        global_end_time="2026-04-29T17:25:20+08:00",
        duration_sec=20.0,
        third_person=ClipReference("third.mp4", "third_clip.mp4", 0.0, 20.0),
        first_person=ClipReference("first.mp4", "first_clip.mp4", 0.0, 20.0),
        cv_detection=CVDetectionSummary(0.8, 0.8, "start", "end"),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo("", "", ""),
    )


def test_micro_display_ids_are_contiguous_and_do_not_replace_unique_ids(tmp_path: Path) -> None:
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
    rows = [
        {"source_view": "first_person", "local_time_sec": 1.0, "frame_index": 30, "hand_object_interactions": [{"hand_label": "hand", "object_label": "balance", "score": 0.9}]},
        {"source_view": "first_person", "local_time_sec": 2.0, "frame_index": 60, "hand_object_interactions": [{"hand_label": "hand", "object_label": "balance", "score": 0.9}]},
        {"source_view": "first_person", "local_time_sec": 5.0, "frame_index": 150, "hand_object_interactions": [{"hand_label": "hand", "object_label": "reagent_bottle", "score": 0.9}]},
        {"source_view": "first_person", "local_time_sec": 6.0, "frame_index": 180, "hand_object_interactions": [{"hand_label": "hand", "object_label": "reagent_bottle", "score": 0.9}]},
    ]
    micros = generate_micro_segments(
        manifest=manifest,
        key_segments=[_parent()],
        yolo_frame_rows=rows,
        utterances=[],
        clips_dir=tmp_path / "clips",
        keyframes_dir=tmp_path / "keyframes",
        config=MicroSegmentConfig(default_interaction_threshold=0.2, default_min_duration_sec=0.5),
        dry_run=True,
    )
    assert [item.display_order for item in micros] == [1, 2]
    assert [item.display_id for item in micros] == ["micro_001", "micro_002"]
    assert all(item.micro_segment_id.startswith("seg_000001_micro_") for item in micros)

