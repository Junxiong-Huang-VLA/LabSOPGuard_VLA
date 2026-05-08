from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.micro_tuning import apply_micro_tuning, rerun_micro_with_config
from key_action_indexer.schemas import write_jsonl


def _raw_micro(micro_id: str, start: float, end: float) -> dict:
    return {
        "micro_segment_id": micro_id,
        "parent_segment_id": "seg_000001",
        "session_id": "s1",
        "start_sec": start,
        "end_sec": end,
        "duration_sec": end - start,
        "global_start_time": f"t{start}",
        "global_end_time": f"t{end}",
        "first_person": {"clip_path": f"{micro_id}_first.mp4", "local_start_sec": start, "local_end_sec": end},
        "third_person": {"clip_path": f"{micro_id}_third.mp4", "local_start_sec": start, "local_end_sec": end},
        "interaction": {
            "primary_object": "sample_bottle",
            "interaction_type": "hand_sample_bottle_contact",
            "detected_objects": ["hand", "sample_bottle"],
            "avg_interaction_score": 0.8,
            "max_interaction_score": 0.8,
            "contact_start_sec": start,
            "peak_interaction_sec": start,
            "contact_end_sec": end,
            "evidence_frame_indices": [1],
        },
        "keyframes": {"peak_frame": f"{micro_id}_peak.jpg"},
        "quality": {"confidence": "high", "warnings": []},
        "dialogue_context": [],
        "text_description": {"action_type": "bottle_interaction", "index_text": f"sample_bottle {micro_id}"},
        "index": {"index_level": "micro_segment", "embedding_id": f"emb_{micro_id}"},
    }


def test_apply_micro_tuning_exports_recommended_config_and_rerun_reuses_raw(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    (session / "evaluation").mkdir()
    write_jsonl(session / "metadata" / "micro_segments_raw.jsonl", [_raw_micro("m1", 1.0, 2.0), _raw_micro("m2", 2.4, 3.0)])
    write_jsonl(
        session / "metadata" / "vector_metadata.jsonl",
        [
            {
                "index_level": "segment",
                "segment_id": "seg_000001",
                "session_id": "s1",
                "index_text": "parent segment sample bottle",
            }
        ],
    )
    sweep_best = session / "evaluation" / "micro_threshold_sweep_best.json"
    sweep_best.write_text(
        json.dumps(
            {"recommended_config": {"interaction_threshold": 0.4, "min_duration_sec": 0.5, "merge_gap_sec": 1.0}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    applied = apply_micro_tuning(session, sweep_best)
    output_config = Path(applied["output_config"])
    result = rerun_micro_with_config(session, output_config)

    assert output_config.exists()
    assert result["reran_yolo"] is False
    assert Path(result["micro_segments"]).exists()
    assert result["raw_micro_count"] == 2
    assert result["merged_micro_count"] == 1
    assert (session / "metadata" / "micro_vector_metadata.jsonl").exists()
