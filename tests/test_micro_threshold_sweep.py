from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.schemas import read_jsonl
from key_action_indexer.tuning import tune_micro_thresholds


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_tune_micro_thresholds_uses_existing_frame_rows_and_gt(tmp_path: Path) -> None:
    session = tmp_path / "session"
    frame_rows = session / "cv_outputs" / "yolo_micro_frame_rows.jsonl"
    gt = tmp_path / "manual_micro_segments.jsonl"
    _write_jsonl(
        frame_rows,
        [
            {"parent_segment_id": "seg_001", "time_sec": 1.0, "primary_object": "pipette", "interaction_score": 0.7, "hand_detected": True, "action_type": "pipetting"},
            {"parent_segment_id": "seg_001", "time_sec": 1.5, "primary_object": "pipette", "interaction_score": 0.8, "hand_detected": True, "action_type": "pipetting"},
            {"parent_segment_id": "seg_001", "time_sec": 5.0, "primary_object": "tube", "interaction_score": 0.2, "hand_detected": True, "action_type": "tube_interaction"},
        ],
    )
    _write_jsonl(
        gt,
        [
            {
                "micro_segment_id": "gt_001",
                "parent_segment_id": "seg_001",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "primary_object": "pipette",
                "action_type": "pipetting",
                "interaction_type": "hand_pipette_contact",
            }
        ],
    )

    result = tune_micro_thresholds(
        session,
        gt,
        interaction_thresholds=[0.3, 0.75],
        merge_gap_secs=[1.0],
        min_duration_secs=[0.5],
        iou_threshold=0.3,
    )

    sweep_path = session / "evaluation" / "micro_threshold_sweep.jsonl"
    best_path = session / "evaluation" / "micro_threshold_sweep_best.json"
    rows = read_jsonl(sweep_path)
    assert result["result_count"] == 2
    assert sweep_path.exists()
    assert best_path.exists()
    assert rows[0]["interaction_threshold"] == 0.3
    assert result["best_config"]["interaction_threshold"] == 0.3
    assert result["best_config"]["true_positive"] == 1
    assert "frame_rows" in json.loads(best_path.read_text(encoding="utf-8"))
