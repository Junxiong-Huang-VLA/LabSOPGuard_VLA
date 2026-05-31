from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.advanced_vision_evidence import build_advanced_vision_evidence
from key_action_indexer.cli import main
from key_action_indexer.model_observations import build_model_observation_events, load_model_observation_events
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.video_understanding import build_video_understanding


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_model_inputs(session: Path) -> None:
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 10.0,
                "end_sec": 15.0,
                "global_start_time": "2026-05-03T09:00:10+08:00",
                "global_end_time": "2026-05-03T09:00:15+08:00",
                "interaction": {
                    "interaction_type": "hand_pipette_contact",
                    "primary_object": "pipette",
                },
                "text_description": {"action_type": "pipetting", "summary": "pipette liquid"},
                "evidence": {"evidence_level": "visual_confirmed"},
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    write_jsonl(
        metadata / "liquid_segmentation.jsonl",
        [
            {
                "event_id": "liq_001",
                "session_id": "s1",
                "micro_segment_id": "micro_001",
                "start_sec": 10.5,
                "end_sec": 12.0,
                "object_label": "beaker",
                "volume_ml": 1.2,
                "mask_path": "masks/liq_001.png",
                "confidence": 0.91,
                "model_name": "liquid-seg-v1",
            }
        ],
    )
    write_jsonl(
        metadata / "equipment_panel_states.jsonl",
        [
            {
                "event_id": "panel_001",
                "session_id": "s1",
                "micro_segment_id": "micro_001",
                "start_sec": 12.0,
                "end_sec": 13.0,
                "equipment_label": "balance_panel",
                "display_text": "1.20 g",
                "button_state": "tare_released",
                "confidence": 0.88,
            }
        ],
    )
    write_jsonl(
        metadata / "container_state_events.jsonl",
        [
            {
                "event_id": "container_001",
                "session_id": "s1",
                "micro_segment_id": "micro_001",
                "start_sec": 13.0,
                "end_sec": 14.0,
                "container_label": "tube_A",
                "before_state": "closed",
                "after_state": "open",
                "confidence": 0.86,
            }
        ],
    )
    write_jsonl(
        metadata / "object_tracks.jsonl",
        [
            {
                "event_id": "track_001",
                "session_id": "s1",
                "micro_segment_id": "micro_001",
                "start_sec": 10.0,
                "end_sec": 15.0,
                "object_label": "pipette",
                "track_id": "pipette_track_001",
                "points": [
                    {"time_sec": 10.0, "bbox": [0, 0, 10, 10], "confidence": 0.9},
                    {"time_sec": 15.0, "bbox": [20, 0, 30, 10], "confidence": 0.9},
                ],
                "confidence": 0.92,
            }
        ],
    )


def test_model_observation_events_normalize_external_outputs_and_cli(tmp_path: Path, capsys) -> None:
    session = tmp_path / "session"
    _write_model_inputs(session)

    summary = build_model_observation_events(session)
    rows = load_model_observation_events(session / "metadata" / "model_observation_events.jsonl")

    assert summary["event_count"] == 4
    assert summary["input_counts"] == {
        "liquid_segmentation": 1,
        "equipment_panel_state": 1,
        "container_state": 1,
        "object_track": 1,
    }
    assert {row["source_type"] for row in rows} == {
        "liquid_segmentation",
        "equipment_panel_state",
        "container_state",
        "object_track",
    }
    assert {row["micro_segment_id"] for row in rows} == {"micro_001"}
    track_row = next(row for row in rows if row["source_type"] == "object_track")
    assert track_row["measurement"]["displacement_px"] == 20.0
    assert track_row["metrics"]["track_point_count"] == 2

    output_path = session / "metadata" / "custom_model_observation_events.jsonl"
    assert main(["model-observations", "--session-dir", str(session), "--output", str(output_path)]) == 0
    cli_summary = json.loads(capsys.readouterr().out)

    assert output_path.exists()
    assert cli_summary["event_count"] == 4
    assert cli_summary["model_observation_events"] == str(output_path)
    assert cli_summary["summary_path"] == str(output_path.with_name("custom_model_observation_events_summary.json"))
    assert output_path.with_name("custom_model_observation_events_summary.json").exists()


def test_model_observations_bind_to_advanced_evidence_and_video_understanding(tmp_path: Path) -> None:
    session = tmp_path / "session"
    _write_model_inputs(session)

    advanced_summary = build_advanced_vision_evidence(session)
    video_summary = build_video_understanding(session)
    advanced_rows = _read_jsonl(session / "metadata" / "advanced_vision_evidence.jsonl")
    video_rows = _read_jsonl(session / "metadata" / "video_understanding.jsonl")

    assert advanced_summary["input_counts"]["model_observation_events"] == 4
    assert video_summary["input_counts"]["model_observation_events"] == 4
    assert {row["evidence_type"] for row in advanced_rows if row["payload"].get("source") == "model_observation_events"} >= {
        "liquid_level_change",
        "equipment_control_change",
        "container_open_close",
        "object_trajectory_movement",
    }
    assert {row["event_type"] for row in video_rows} >= {
        "liquid_level_change_detected",
        "equipment_panel_operation_candidate",
        "container_state_change_candidate",
        "object_movement_detected",
    }
