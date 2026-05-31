from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer.retrieval_eval import (
    FIXED_CHINESE_QUERY_BENCHMARK,
    build_default_chinese_query_eval_config,
    build_gold_query_benchmark,
    confirm_gold_query_benchmark,
    run_default_chinese_query_eval,
)
from key_action_indexer.schemas import write_jsonl
from key_action_indexer.vector_index import VectorIndex


def test_default_chinese_eval_builds_fixed_50_bound_queries(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    row = {
        "index_level": "micro_segment",
        "segment_id": "seg_1",
        "micro_segment_id": "micro_1",
        "primary_object": "balance",
        "action_type": "weighing",
        "index_text": "balance weighing sample keyframe",
        "keyframes": ["peak.jpg"],
    }
    write_jsonl(metadata / "micro_segments.jsonl", [row])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [row])
    index = VectorIndex()
    index.build([row["index_text"]], [row])
    index.save(session / "index")

    config = build_default_chinese_query_eval_config(session, query_count=50)

    assert len(FIXED_CHINESE_QUERY_BENCHMARK) == 50
    assert len(config["queries"]) == 50
    assert all(query.get("expected_segment_ids") or query.get("expected_micro_segment_ids") for query in config["queries"])
    assert (metadata / "gold_query_benchmark.json").exists()
    assert config["benchmark_binding_mode"] == "bootstrap_pending_human_verification"


def test_gold_query_benchmark_marks_bootstrap_as_non_final_gt(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    row = {"index_level": "micro_segment", "segment_id": "seg_1", "micro_segment_id": "micro_1", "primary_object": "balance", "action_type": "weighing", "index_text": "balance weighing"}
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [row])

    gold = build_gold_query_benchmark(session, query_count=50, overwrite=True)

    assert gold["query_count"] == 50
    assert gold["human_verified_query_count"] == 0
    assert gold["queries"][0]["binding_source"] == "bootstrap_auto"
    assert gold["queries"][0]["human_verified"] is False


def test_default_chinese_eval_writes_trend(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    row = {
        "index_level": "micro_segment",
        "segment_id": "seg_1",
        "micro_segment_id": "micro_1",
        "primary_object": "balance",
        "detected_objects": ["balance", "sample"],
        "action_type": "weighing",
        "global_start_time": "2026-05-08T10:00:00+08:00",
        "global_end_time": "2026-05-08T10:00:02+08:00",
        "index_text": "balance weighing sample keyframe 称量 样品 天平",
        "keyframes": ["peak.jpg"],
    }
    write_jsonl(metadata / "micro_segments.jsonl", [row])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [row])
    index = VectorIndex()
    index.build([row["index_text"]], [row])
    index.save(session / "index")

    result = run_default_chinese_query_eval(session, query_count=20)

    assert result["query_count"] == 20
    assert result["category_summary"]
    assert (session / "evaluation" / "retrieval_eval_trend.jsonl").exists()
    assert (session / "evaluation" / "retrieval_eval_trend.md").exists()


def test_confirm_gold_query_benchmark_requires_human_decision_file(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / "metadata").mkdir(parents=True)

    with pytest.raises(ValueError, match="requires a human decision file"):
        confirm_gold_query_benchmark(session, query_count=1)


def test_confirm_gold_query_benchmark_locks_expected_ids_for_hard_metrics(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    row = {
        "index_level": "micro_segment",
        "segment_id": "seg_1",
        "micro_segment_id": "micro_1",
        "primary_object": "balance",
        "detected_objects": ["balance", "sample"],
        "action_type": "weighing",
        "evidence_level": "visual_confirmed",
        "index_text": "balance weighing sample keyframe",
        "keyframes": ["peak.jpg"],
    }
    write_jsonl(metadata / "micro_segments.jsonl", [row])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [row])
    index = VectorIndex()
    index.build([row["index_text"]], [row])
    index.save(session / "index")

    decisions = {
        "decisions": [
            {
                "query_id": f"gold_cn_{index:03d}",
                "decision": "approved",
                "expected_segment_ids": ["seg_1"],
                "expected_micro_segment_ids": ["micro_1"],
                "expected_index_level": "micro_segment",
                "reviewer": "tester",
            }
            for index in range(1, 4)
        ]
    }
    decisions_path = metadata / "gold_query_decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    gold = confirm_gold_query_benchmark(session, query_count=3, reviewer="tester", decisions_path=decisions_path)
    result = run_default_chinese_query_eval(session, query_count=3)

    assert gold["human_verified_query_count"] == 3
    assert all(query["human_verified"] is True for query in gold["queries"])
    assert all(query["id_authoritative"] is True for query in gold["queries"])
    assert result["human_verified_query_count"] == 3
    assert result["expected_id_hit_rate"] == 1.0
    assert result["top1_hit_rate"] == 1.0


def test_not_applicable_gold_queries_are_excluded_from_eval_denominator(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    row = {
        "index_level": "micro_segment",
        "segment_id": "seg_1",
        "micro_segment_id": "micro_1",
        "primary_object": "balance",
        "detected_objects": ["balance", "sample"],
        "action_type": "weighing",
        "evidence_level": "visual_confirmed",
        "index_text": "balance weighing sample keyframe 称量 样品 天平",
        "keyframes": ["peak.jpg"],
    }
    write_jsonl(metadata / "micro_segments.jsonl", [row])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [row])
    index = VectorIndex()
    index.build([row["index_text"]], [row])
    index.save(session / "index")

    decisions = {
        "decisions": [
            {
                "query_id": "gold_cn_001",
                "decision": "approved",
                "expected_segment_ids": ["seg_1"],
                "expected_micro_segment_ids": ["micro_1"],
                "expected_index_level": "micro_segment",
                "reviewer": "tester",
            },
            {"query_id": "gold_cn_002", "decision": "not_applicable", "reviewer": "tester", "note": "no tube workflow"},
            {"query_id": "gold_cn_003", "decision": "out_of_scope", "reviewer": "tester", "note": "no plate workflow"},
        ]
    }
    decisions_path = metadata / "gold_query_decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    gold = confirm_gold_query_benchmark(session, query_count=3, reviewer="tester", decisions_path=decisions_path)
    result = run_default_chinese_query_eval(session, query_count=3)

    assert gold["query_count"] == 3
    assert gold["applicable_query_count"] == 1
    assert gold["excluded_query_count"] == 2
    assert gold["human_verified_query_count"] == 1
    assert gold["human_reviewed_query_count"] == 3
    assert gold["binding_mode"] == "human_verified_review_file"
    assert result["status"] == "pass"
    assert result["total_query_count"] == 3
    assert result["query_count"] == 1
    assert result["applicable_query_count"] == 1
    assert result["excluded_query_count"] == 2
    assert result["failed_query_count"] == 0
    assert len([row for row in result["queries"] if row.get("excluded_from_evaluation")]) == 2


def test_confirm_gold_query_benchmark_clears_stale_ids_when_reconfirmed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    row = {
        "index_level": "micro_segment",
        "segment_id": "seg_1",
        "micro_segment_id": "micro_1",
        "primary_object": "balance",
        "detected_objects": ["balance", "sample"],
        "action_type": "weighing",
        "evidence_level": "visual_confirmed",
        "index_text": "balance weighing sample keyframe 绉伴噺 鏍峰搧 澶╁钩",
        "keyframes": ["peak.jpg"],
    }
    write_jsonl(metadata / "micro_segments.jsonl", [row])
    write_jsonl(metadata / "micro_vector_metadata.jsonl", [row])
    index = VectorIndex()
    index.build([row["index_text"]], [row])
    index.save(session / "index")
    gold_path = metadata / "gold_query_benchmark.json"
    gold = build_gold_query_benchmark(session, query_count=2, overwrite=True)
    for query in gold["queries"]:
        query["expected_segment_ids"] = ["stale_segment"]
        query["expected_micro_segment_ids"] = ["stale_micro"]
        query["verified_top_result"] = {"segment_id": "stale_segment", "micro_segment_id": "stale_micro"}
    gold_path.write_text(json.dumps(gold), encoding="utf-8")

    decisions = {
        "decisions": [
            {
                "query_id": "gold_cn_001",
                "decision": "approved",
                "expected_micro_segment_ids": ["micro_1"],
                "expected_index_level": "micro_segment",
                "reviewer": "tester",
            },
            {"query_id": "gold_cn_002", "decision": "not_applicable", "reviewer": "tester", "note": "no tube workflow"},
        ]
    }
    decisions_path = metadata / "gold_query_decisions.json"
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    confirmed = confirm_gold_query_benchmark(session, query_count=2, reviewer="tester", decisions_path=decisions_path)
    result = run_default_chinese_query_eval(session, query_count=2)

    approved, excluded = confirmed["queries"]
    assert "expected_segment_ids" not in approved
    assert approved["expected_micro_segment_ids"] == ["micro_1"]
    assert "verified_top_result" not in approved
    assert "expected_segment_ids" not in excluded
    assert "expected_micro_segment_ids" not in excluded
    assert "verified_top_result" not in excluded
    assert result["status"] == "pass"
    assert result["expected_id_hit_rate"] == 1.0
