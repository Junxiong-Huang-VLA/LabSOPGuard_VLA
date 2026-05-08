from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.clip_extractor import extract_multiview_clips
from key_action_indexer.pipeline import run_pipeline
from key_action_indexer.schemas import DetectedSegment, SessionManifest, read_jsonl, write_jsonl


def _interaction_row(segment_id: str = "seg_000001", local_time_sec: float = 15.0) -> dict:
    return {
        "segment_id": segment_id,
        "view": "third_person",
        "local_time_sec": local_time_sec,
        "detections": [
            {"label": "hand", "confidence": 0.91, "bbox": [10, 10, 60, 70]},
            {"label": "sample_bottle", "confidence": 0.82, "bbox": [55, 20, 100, 85]},
        ],
    }


def test_extract_multiview_clips_writes_interaction_keyframe_in_dry_run(tmp_path: Path) -> None:
    manifest = SessionManifest.from_dict(
        {
            "session_id": "s1",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                }
            },
            "output_dir": str(tmp_path / "session"),
        }
    )
    segment = DetectedSegment(
        segment_id="seg_000001",
        start_sec=10.0,
        end_sec=20.0,
        duration_sec=10.0,
        global_start_time="2026-04-29T17:25:10+08:00",
        global_end_time="2026-04-29T17:25:20+08:00",
        avg_motion_score=0.8,
        avg_active_score=0.8,
        start_reason="active",
        end_reason="inactive",
    )

    key_segment = extract_multiview_clips(
        manifest,
        segment,
        tmp_path / "clips",
        tmp_path / "keyframes",
        yolo_frame_rows=[_interaction_row()],
        dry_run=True,
    )

    assert (tmp_path / "keyframes" / "seg_000001" / "third_person_start.jpg").exists()
    assert (tmp_path / "keyframes" / "seg_000001" / "interaction_001.jpg").exists()
    assert key_segment.interaction_events[0].interaction == "手与瓶子交互"
    assert key_segment.yolo_interactions[0].object_label == "sample_bottle"


def test_pipeline_adds_interaction_metadata_to_segments_and_vectors(tmp_path: Path) -> None:
    output_dir = tmp_path / "session"
    (output_dir / "cv_outputs").mkdir(parents=True)
    write_jsonl(output_dir / "cv_outputs" / "yolo_frame_rows.jsonl", [_interaction_row("seg_000002", 620.5)])
    manifest = {
        "session_id": "test_session",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {
                "path": str(output_dir / "raw" / "third_person.mp4"),
                "start_time": "2026-04-29T17:25:00+08:00",
                "fps": 30,
                "offset_sec": 0,
            }
        },
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)

    assert summary["segment_count"] >= 3
    segments = read_jsonl(output_dir / "metadata" / "key_action_segments.jsonl")
    seg_2 = next(item for item in segments if item["segment_id"] == "seg_000002")
    assert seg_2["interaction_keyframes"][0]["path"].endswith("interaction_001.jpg")
    assert seg_2["interaction_events"][0]["interaction"] == "手与瓶子交互"
    assert seg_2["yolo_interactions"][0]["object_label"] == "sample_bottle"
    assert "手与瓶子交互" in seg_2["index"]["index_text"]

    vectors = read_jsonl(output_dir / "metadata" / "vector_metadata.jsonl")
    vector_2 = next(item for item in vectors if item["segment_id"] == "seg_000002")
    assert "手与瓶子交互" in vector_2["index_text"]
    assert "手与瓶子交互" in vector_2["visual_keywords"]
    assert "sample_bottle" in vector_2["visual_keywords"]
    assert vector_2["yolo_evidence"][0]["detections"]
