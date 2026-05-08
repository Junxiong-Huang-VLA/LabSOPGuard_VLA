from __future__ import annotations

from pathlib import Path

from key_action_indexer.retrieval_eval import (
    FIXED_CHINESE_QUERY_BENCHMARK,
    build_default_chinese_query_eval_config,
    build_gold_query_benchmark,
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
