from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.query_validation import query_index, validate_queries
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.vector_index import VectorIndex


def _build_index(session: Path) -> None:
    metadata = [
        {
            "index_level": "micro_segment",
            "segment_id": "seg_000001",
            "micro_segment_id": "seg_000001_micro_001",
            "primary_object": "sample_bottle",
            "detected_objects": ["sample_bottle", "pipette"],
            "interaction_type": "hand_sample_bottle_contact",
            "action_type": "bottle_interaction",
            "global_start_time": "2026-04-29T17:25:01+08:00",
            "global_end_time": "2026-04-29T17:25:03+08:00",
            "index_text": "手 碰 接触 瓶子 sample_bottle hand_sample_bottle_contact",
            "keyframes": ["peak.jpg"],
        },
        {
            "index_level": "segment",
            "segment_id": "seg_000001",
            "action_type": "weighing",
            "index_text": "加样 移液 微升 parent fallback segment",
            "limitations": [
                "missing pipette or tube visual evidence",
                "missing transcript evidence",
                "returned parent segment because no trustworthy sample_adding micro was found",
            ],
            "rerank_reasons": ["fallback_parent_segment_for_insufficient_pipette_or_asr"],
        },
    ]
    index = VectorIndex()
    index.build([row["index_text"] for row in metadata], metadata)
    index.save(session / "index")
    write_jsonl(session / "metadata" / "vector_metadata.jsonl", metadata)


def test_validate_queries_counts_expected_hits_and_valid_fallback(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    _build_index(session)
    config = tmp_path / "query_validation.json"
    config.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": "手碰瓶子",
                        "expected_objects": ["sample_bottle"],
                        "expected_index_level": "micro_segment",
                        "top_k": 2,
                    },
                    {
                        "query": "加样",
                        "expected_objects": ["pipette", "pipette_tip", "tube"],
                        "expected_index_level": "micro_segment",
                        "top_k": 2,
                        "allow_parent_fallback_when_insufficient_evidence": True,
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_queries(session, config)

    assert result["query_count"] == 2
    assert result["status"] == "pass"
    assert result["acceptance_hit_rate"] == 1.0
    assert result["queries"][0]["expected_object_hit"] is True
    assert result["queries"][1]["fallback_reason_valid"] is True
    assert result["query_hit_rate"] == 1.0


def test_validate_queries_checks_detected_objects_actions_time_and_traceability(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    _build_index(session)
    config = tmp_path / "query_validation.json"
    config.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "query": "pipette sample_bottle",
                        "expected_objects": ["pipette"],
                        "expected_actions": ["bottle_interaction", "hand_sample_bottle_contact"],
                        "expected_index_level": "micro_segment",
                        "expected_time_window": {
                            "start": "2026-04-29T17:25:00+08:00",
                            "end": "2026-04-29T17:25:04+08:00",
                        },
                        "require_traceability": True,
                        "top_k": 2,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_queries(session, config)
    row = result["queries"][0]

    assert row["expected_object_hit"] is True
    assert row["expected_action_hit"] is True
    assert row["expected_time_window_hit"] is True
    assert row["traceability_hit"] is True
    assert row["acceptance_hit"] is True
    assert result["traceability_hit_rate"] == 1.0


def test_validate_queries_can_fail_low_quality_acceptance(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    metadata = [
        {
            "index_level": "micro_segment",
            "segment_id": "seg_000001",
            "micro_segment_id": "seg_000001_micro_001",
            "primary_object": "balance",
            "detected_objects": ["balance", "gloved_hand"],
            "interaction_type": "hand_balance_contact",
            "action_type": "weighing",
            "global_start_time": "2026-04-29T17:25:01+08:00",
            "global_end_time": "2026-04-29T17:25:03+08:00",
            "index_text": "balance weighing hand contact",
            "keyframes": ["peak.jpg"],
            "quality_warnings": ["very_low_signal_yolo_candidate"],
            "evidence": {
                "evidence_level": "visual_confirmed",
                "coverage_signal_grade": "very_low_signal_yolo_candidate",
                "coverage_backfill": True,
                "limitations": ["coverage backfill candidate"],
            },
        }
    ]
    index = VectorIndex()
    index.build([row["index_text"] for row in metadata], metadata)
    index.save(session / "index")
    write_jsonl(session / "metadata" / "vector_metadata.jsonl", metadata)
    config = tmp_path / "query_validation.json"
    config.write_text(
        json.dumps(
            {
                "thresholds": {"min_acceptance_hit_rate": 1.0},
                "quality_policy": {
                    "allow_very_low_signal": False,
                    "allow_coverage_backfill": False,
                },
                "queries": [
                    {
                        "query": "balance weighing",
                        "expected_objects": ["balance"],
                        "expected_actions": ["weighing"],
                        "expected_index_level": "micro_segment",
                        "require_traceability": True,
                        "top_k": 1,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_queries(session, config)
    row = result["queries"][0]

    assert result["status"] == "fail"
    assert result["query_hit_rate"] == 1.0
    assert result["acceptance_hit_rate"] == 0.0
    assert row["topk_hit"] is True
    assert row["quality_hit"] is False
    assert "very_low_signal_not_allowed" in row["top1_quality_failures"]
    assert any(item["metric"] == "acceptance_hit_rate" for item in result["threshold_failures"])


def test_cli_query_runs_direct_search(tmp_path: Path, capsys) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)
    _build_index(session)

    exit_code = main(["query", "--session-dir", str(session), "--query", "sample_bottle", "--top-k", "1"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["query_count"] == 1
    assert payload["queries"][0]["result_count"] == 1
    assert payload["queries"][0]["results"][0]["segment_id"] == "seg_000001"


def test_query_index_flattens_nested_view_clips_for_traceability(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    metadata = [
        {
            "index_level": "segment",
            "segment_id": "reviewed_seg_001",
            "index_text": "reviewed balance weighing",
            "third_person": {"clip_path": "clips/reviewed_third.mp4"},
            "first_person": {"clip_path": "clips/reviewed_first.mp4"},
            "keyframes": ["peak.jpg"],
        }
    ]
    index = VectorIndex()
    index.build([metadata[0]["index_text"]], metadata)
    index.save(index_dir)

    payload = query_index(index_dir, ["balance"], top_k=1)
    result = payload["queries"][0]["results"][0]

    assert result["third_person_clip"] == "clips/reviewed_third.mp4"
    assert result["first_person_clip"] == "clips/reviewed_first.mp4"
