from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.schemas import write_jsonl
from key_action_indexer.yolo_relabel import export_yolo_relabel_pack


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_yolo_relabel_pack_generates_missing_class_candidates_without_video(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (session / "manifest.json").write_text(json.dumps({"session_id": "s1", "videos": {"third_person": {"path": str(session / "raw" / "missing.mp4"), "fps": 30}}}), encoding="utf-8")
    ground_truth = metadata / "micro_gt.jsonl"
    write_jsonl(
        ground_truth,
        [
            {"micro_segment_id": "micro_liquid", "start_sec": 1.0, "end_sec": 2.0, "primary_object": "pipette", "interaction_type": "liquid transfer pour stream meniscus level"},
            {"micro_segment_id": "micro_control", "start_sec": 3.0, "end_sec": 4.0, "primary_object": "balance", "interaction_type": "press button adjust knob read display"},
        ],
    )

    output_dir = session / "annotation" / "missing_pack"
    summary = export_yolo_relabel_pack(
        session,
        ground_truth_path=ground_truth,
        output_dir=output_dir,
        missing_classes=["liquid_stream", "meniscus", "button", "knob", "display"],
        views=["third_person"],
        samples_per_segment=1,
        max_candidates_per_class=1,
    )

    assert summary["image_count"] == 0
    assert summary["candidate_count"] == len(summary["target_classes"])
    assert summary["training_ready"] is False
    rows = _jsonl_rows(output_dir / "candidate_manifest.jsonl")
    assert {"liquid_stream", "meniscus_line", "button", "knob", "display"}.issubset({row["target_class"] for row in rows})
    assert all(row["candidate_status"] == "candidate_unreviewed" for row in rows)
