from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.missing_step_recovery import build_missing_step_recovery_plan
from key_action_indexer.schemas import write_jsonl


def test_missing_step_recovery_plan_finds_video_transcript_and_asset_candidates(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    transcript = tmp_path / "transcript"
    metadata.mkdir()
    transcript.mkdir()
    (metadata / "experiment_process.json").write_text(
        json.dumps(
            {
                "session_id": "recover_session",
                "steps": [
                    {"step_id": "weigh", "status": "completed", "observed": True, "global_end_time": "2026-05-03T10:00:05+08:00"},
                    {
                        "step_id": "record",
                        "name": "Record balance readout",
                        "expected_action": "recording",
                        "status": "not_observed",
                        "confidence": 0.0,
                        "requires_human_confirmation": True,
                    },
                    {"step_id": "pipette", "status": "completed", "observed": True, "global_start_time": "2026-05-03T10:00:20+08:00"},
                ],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [
            {
                "video_event_id": "vu_record",
                "event_type": "experiment_action_classification",
                "action_type": "recording",
                "primary_object": "balance",
                "confidence": 0.72,
                "global_start_time": "2026-05-03T10:00:12+08:00",
                "global_end_time": "2026-05-03T10:00:13+08:00",
                "segment_id": "seg_record",
                "micro_segment_id": "micro_record",
                "text": "record balance display readout",
            }
        ],
    )
    write_jsonl(transcript / "aligned_transcript.jsonl", [{"global_start_time": "2026-05-03T10:00:11+08:00", "text": "record the balance readout"}])
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [
            {
                "asset_id": "asset_record",
                "asset_type": "keyframe",
                "path": "keyframes/readout.jpg",
                "global_start_time": "2026-05-03T10:00:12+08:00",
                "objects": ["balance"],
                "actions": ["recording"],
                "search_text": "balance display readout recording",
            }
        ],
    )

    plan = build_missing_step_recovery_plan(tmp_path, window_padding_sec=2.0)
    record = plan["steps"][0]

    assert plan["target_step_count"] == 1
    assert record["step_id"] == "record"
    assert record["candidate_evidence"]["counts"] == {"video_events": 1, "transcript_utterances": 1, "assets": 1}
    assert record["candidate_evidence"]["video_events"][0]["video_event_id"] == "vu_record"
    assert (metadata / "missing_step_recovery_plan.json").exists()
