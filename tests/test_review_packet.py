from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.review_packet import build_recovery_review_packet


def test_recovery_review_packet_lists_candidates_and_decision_template(tmp_path: Path) -> None:
    plan = {
        "schema_version": "missing_step_recovery_plan/v1",
        "session_id": "session_a",
        "source_paths": {
            "experiment_process": {
                "path": str(tmp_path / "session_a" / "metadata" / "experiment_process.json"),
                "exists": True,
            }
        },
        "steps": [
            {
                "step_id": "step_001",
                "name": "Weighing",
                "expected_action": "weighing",
                "status": "not_observed",
                "confidence": 0.2,
                "recovery_reason": "step is not observed",
                "recovery_window": {
                    "global_start_time": "2026-05-08T10:00:00+08:00",
                    "global_end_time": "2026-05-08T10:00:10+08:00",
                },
                "candidate_evidence": {
                    "counts": {"video_events": 1, "transcript_utterances": 0, "assets": 1},
                    "video_events": [
                        {
                            "video_event_id": "evt_1",
                            "event_type": "hand_object_contact",
                            "primary_object": "balance",
                            "confidence": 0.81,
                            "match_score": 0.9,
                            "segment_id": "seg_001",
                            "micro_segment_id": "seg_001_micro_001",
                            "global_start_time": "2026-05-08T10:00:01+08:00",
                            "global_end_time": "2026-05-08T10:00:04+08:00",
                            "asset_refs": [{"path": "clips/micro.mp4"}, {"path": "keyframes/peak.jpg"}],
                        }
                    ],
                    "assets": [
                        {
                            "asset_id": "asset_1",
                            "asset_type": "keyframe",
                            "source_type": "micro_keyframe",
                            "path": "keyframes/peak.jpg",
                            "match_score": 0.7,
                        }
                    ],
                },
                "human_confirmation_suggestion": {
                    "decision_hint": "review_for_approval",
                    "rationale": "candidate overlaps the recovery window",
                    "note_template": "step_001: visual_match=; decision=",
                    "candidate_strength": {
                        "best_video_match_score": 0.9,
                        "best_asset_match_score": 0.7,
                        "transcript_candidate_count": 0,
                    },
                },
            }
        ],
    }
    plan_path = tmp_path / "plan.json"
    packet_path = tmp_path / "packet.md"
    decisions_path = tmp_path / "decisions.template.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    summary = build_recovery_review_packet(plan_path, packet_path, output_decisions=decisions_path)
    packet = packet_path.read_text(encoding="utf-8")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))

    assert summary["step_count"] == 1
    assert "clips/micro.mp4" in packet
    assert "keyframes/peak.jpg" in packet
    assert decisions["decisions"][0]["confirmation_id"] == "session_a:step_001"
    assert decisions["decisions"][0]["decision"] == ""
