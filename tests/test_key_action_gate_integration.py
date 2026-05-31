from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.schemas import write_jsonl
from key_action_indexer.advanced_vision_evidence import _hand_object_contact_evidence
from key_action_indexer.yolo_observation_inputs import build_yolo_observation_inputs


def test_key_action_label_level_track_writes_gate_rejection_artifacts(tmp_path: Path) -> None:
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
                "end_sec": 3.0,
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_micro_frame_rows.jsonl",
        [
            {"alignment_time_sec": 1.0, "detections": [{"label": "reagent_bottle", "confidence": 0.9, "bbox": [10, 10, 40, 40]}]},
            {"alignment_time_sec": 2.0, "detections": [{"label": "reagent_bottle", "confidence": 0.9, "bbox": [60, 10, 90, 40]}]},
            {"alignment_time_sec": 3.0, "detections": [{"label": "reagent_bottle", "confidence": 0.9, "bbox": [110, 10, 140, 40]}]},
        ],
    )

    summary = build_yolo_observation_inputs(session)
    gate_rows = _rows(metadata / "physical_event_gate_decisions.jsonl")
    rejected_rows = _rows(metadata / "rejected_physical_event_candidates.jsonl")
    tracks = _rows(metadata / "object_tracks.jsonl")

    assert summary["physical_event_gate_summary"]["rejected"] == 1
    assert gate_rows[0]["status"] == "rejected"
    assert "label_level_pseudo_track" in gate_rows[0]["reject_reasons"]
    assert rejected_rows[0]["event_type"] == "object_move"
    assert tracks[0]["event_type"] == "object_track_observed"
    assert tracks[0]["can_confirm_motion"] is False


def test_advanced_vision_hand_contact_confirmed_uses_physical_gate() -> None:
    micro = {
        "session_id": "s1",
        "micro_segment_id": "micro_001",
        "global_start_time": 1.0,
        "global_end_time": 2.0,
        "interaction": {"primary_object": "pipette", "interaction_type": "hand_pipette_contact"},
    }
    hand_bbox = [180, 90, 250, 170]
    pipette_bbox = [238, 112, 340, 126]
    rows = [
        {
            "frame_id": f"frame_{idx}",
            "local_time_sec": 1.0 + idx * 0.2,
            "view": "front",
            "detections": [
                {"label": "gloved_hand", "confidence": 0.91, "bbox": hand_bbox},
                {"label": "pipette", "confidence": 0.93, "bbox": pipette_bbox},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "pipette",
                    "score": 0.9,
                    "hand_bbox": hand_bbox,
                    "object_bbox": pipette_bbox,
                    "object_coverage_by_hand": 0.12,
                }
            ],
        }
        for idx in range(2)
    ]

    evidence = _hand_object_contact_evidence(micro, rows, [])

    assert evidence[0]["status"] == "confirmed"
    assert evidence[0]["hard_gate"]["passed"] is True
    assert evidence[0]["physical_event_gate"]["hard_gate"]["gate_name"] == "gate_hand_object_contact"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
