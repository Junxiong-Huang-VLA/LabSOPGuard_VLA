from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.advanced_vision_evidence import build_advanced_vision_evidence
from key_action_indexer.cli import main
from key_action_indexer.lab_model_signal_inputs import build_lab_model_signal_inputs
from key_action_indexer.model_observations import build_model_observation_events
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.video_understanding import build_video_understanding


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_lab_model_signal_inputs_bridge_yolo_rows_into_state_candidates(tmp_path: Path, capsys) -> None:
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
    write_jsonl(
        cv_outputs / "yolo_micro_frame_rows.jsonl",
        [
            {
                "source_view": "first_person",
                "alignment_time_sec": 10.5,
                "global_time": "2026-05-03T09:00:10.500000+08:00",
                "frame_index": 315,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.9, "bbox": [8, 8, 34, 34]},
                    {"label": "balance", "confidence": 0.82, "bbox": [18, 12, 58, 52]},
                    {"label": "pipette", "confidence": 0.87, "bbox": [100, 20, 116, 78]},
                    {"label": "tube", "confidence": 0.84, "bbox": [112, 70, 152, 140]},
                    {"label": "tube-cap", "confidence": 0.8, "bbox": [126, 54, 146, 74]},
                ],
                "hand_object_interactions": [
                    {"hand_label": "gloved_hand", "object_label": "balance", "score": 0.74},
                    {"hand_label": "gloved_hand", "object_label": "pipette", "score": 0.69},
                ],
            }
        ],
    )

    summary = build_lab_model_signal_inputs(session)

    assert summary["generated_candidate_count"] == 3
    assert summary["candidate_counts"] == {
        "container_state_candidate": 1,
        "equipment_panel_candidate": 1,
        "liquid_transfer_candidate": 1,
    }
    assert _rows(metadata / "equipment_panel_states.jsonl")[0]["event_type"] == "equipment_panel_interaction_candidate"
    assert _rows(metadata / "container_state_events.jsonl")[0]["state"] == "cap_visible_near_container"
    assert _rows(metadata / "liquid_segmentation.jsonl")[0]["event_type"] == "liquid_flow_candidate"

    observation_summary = build_model_observation_events(session)
    advanced_summary = build_advanced_vision_evidence(session)
    video_summary = build_video_understanding(session)
    video_rows = _rows(metadata / "video_understanding.jsonl")

    assert observation_summary["event_count"] == 3
    assert advanced_summary["input_counts"]["model_observation_events"] == 3
    assert video_summary["input_counts"]["model_observation_events"] == 3
    assert {row["event_type"] for row in video_rows} >= {
        "equipment_panel_operation_candidate",
        "container_state_change_candidate",
        "liquid_flow_candidate_visual",
    }
    assert not any(row["event_type"] == "equipment_panel_operation_detected" for row in video_rows)

    assert main(["lab-model-signal-inputs", "--session-dir", str(session)]) == 0
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["generated_candidate_count"] == 3


def test_lab_model_signal_inputs_preserve_external_rows(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir()
    write_jsonl(metadata / "micro_segments.jsonl", [])
    write_jsonl(
        metadata / "equipment_panel_states.jsonl",
        [
            {
                "event_id": "external_panel_001",
                "session_id": "s1",
                "event_type": "equipment_panel_state_measured",
                "equipment_label": "balance_panel",
                "display_text": "1.20 g",
                "confidence": 0.9,
                "payload": {"source": "external_ocr_model"},
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_micro_frame_rows.jsonl",
        [
            {
                "source_view": "first_person",
                "alignment_time_sec": 1.0,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.8, "bbox": [0, 0, 20, 20]},
                    {"label": "balance", "confidence": 0.8, "bbox": [5, 5, 45, 45]},
                ],
                "hand_object_interactions": [{"hand_label": "gloved_hand", "object_label": "balance", "score": 0.7}],
            }
        ],
    )

    build_lab_model_signal_inputs(session)
    rows = _rows(metadata / "equipment_panel_states.jsonl")

    assert len(rows) == 2
    assert any(row["event_id"] == "external_panel_001" for row in rows)
    assert any(row["payload"]["source"] == "lab_model_signal_inputs" for row in rows)
