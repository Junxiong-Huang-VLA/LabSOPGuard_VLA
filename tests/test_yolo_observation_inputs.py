from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.advanced_vision_evidence import build_advanced_vision_evidence
from key_action_indexer.cli import main
from key_action_indexer.model_observations import build_model_observation_events
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.video_understanding import build_video_understanding
from key_action_indexer.yolo_observation_inputs import build_yolo_observation_inputs


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_yolo_observation_inputs_build_object_tracks_and_downstream_events(tmp_path: Path, capsys) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 10.0,
                "end_sec": 12.0,
                "global_start_time": "2026-05-03T09:00:10+08:00",
                "global_end_time": "2026-05-03T09:00:12+08:00",
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    write_jsonl(
        cv_outputs / "yolo_micro_frame_rows.jsonl",
        [
            {
                "source_view": "first_person",
                "alignment_time_sec": 10.0,
                "global_time": "2026-05-03T09:00:10+08:00",
                "frame_index": 300,
                "detections": [
                    {"label": "sample_bottle", "confidence": 0.9, "bbox": [10, 10, 30, 30]},
                    {"label": "paper", "confidence": 0.8, "bbox": [100, 100, 120, 120]},
                ],
            },
            {
                "source_view": "first_person",
                "alignment_time_sec": 11.0,
                "global_time": "2026-05-03T09:00:11+08:00",
                "frame_index": 330,
                "detections": [
                    {"label": "sample_bottle", "confidence": 0.86, "bbox": [30, 10, 50, 30]},
                    {"label": "paper", "confidence": 0.8, "bbox": [100, 100, 120, 120]},
                ],
            },
        ],
    )
    (metadata / "yolo_micro_scan_summary.json").write_text(
        json.dumps({"model_path": "LabSOPGuard/outputs/training/model/weights/best.pt"}),
        encoding="utf-8",
    )

    summary = build_yolo_observation_inputs(session, min_motion_px=5.0)
    tracks = _rows(metadata / "object_tracks.jsonl")
    gate_rows = _rows(metadata / "physical_event_gate_decisions.jsonl")
    rejected_rows = _rows(metadata / "rejected_physical_event_candidates.jsonl")

    assert summary["generated_track_count"] == 1
    assert summary["physical_event_gate_summary"]["rejected"] == 1
    assert gate_rows[0]["status"] == "rejected"
    assert rejected_rows[0]["event_type"] == "object_move"
    assert tracks[0]["object_label"] == "sample_bottle"
    assert tracks[0]["event_type"] == "object_track_observed"
    assert tracks[0]["motion_state"] == "tracked_static"
    assert tracks[0]["status"] == "rejected"
    assert tracks[0]["track_type"] == "label_level_pseudo_track"
    assert "label_level_pseudo_track" in tracks[0]["reject_reasons"]
    assert tracks[0]["displacement_px"] == 20.0

    observation_summary = build_model_observation_events(session)
    video_summary = build_video_understanding(session)
    video_rows = _rows(metadata / "video_understanding.jsonl")

    assert observation_summary["event_count"] == 1
    assert video_summary["input_counts"]["model_observation_events"] == 1
    assert any(row["event_type"] == "object_track_observed" for row in video_rows)
    assert not any(row["event_type"] == "object_movement_detected" for row in video_rows)

    assert main(["yolo-observation-inputs", "--session-dir", str(session), "--min-motion-px", "5"]) == 0
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["track_count"] == 1


def test_yolo_observation_inputs_emits_static_tracks_without_claiming_movement(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 2.0,
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_micro_frame_rows.jsonl",
        [
            {"source_view": "first_person", "alignment_time_sec": 1.0, "detections": [{"label": "balance", "confidence": 0.7, "bbox": [10, 10, 30, 30]}]},
            {"source_view": "first_person", "alignment_time_sec": 2.0, "detections": [{"label": "balance", "confidence": 0.7, "bbox": [10, 10, 30, 30]}]},
        ],
    )

    build_yolo_observation_inputs(session, min_motion_px=5.0)
    build_model_observation_events(session)
    build_advanced_vision_evidence(session)
    build_video_understanding(session)
    observations = _rows(metadata / "model_observation_events.jsonl")
    video_rows = _rows(metadata / "video_understanding.jsonl")

    assert observations[0]["event_type"] == "object_track_observed"
    assert any(row["event_type"] == "object_track_observed" for row in video_rows)
    assert not any(row["event_type"] == "object_movement_detected" for row in video_rows)
