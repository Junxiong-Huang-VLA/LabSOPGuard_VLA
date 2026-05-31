from __future__ import annotations

from key_action_indexer.vector_index import rerank_results


def test_empty_input():
    assert rerank_results([]) == []


def test_basic_ranking():
    results = [
        {"segment_id": "s1", "score": 0.5, "third_person_clip": "a.mp4"},
        {"segment_id": "s2", "score": 0.9, "first_person_clip": "b.mp4", "third_person_clip": "c.mp4", "evidence": {"evidence_level": "strong"}, "quality": "high"},
    ]
    reranked = rerank_results(results)
    assert reranked[0]["segment_id"] == "s2"
    assert "retrieval_boost_factors" in reranked[0]
    assert "fusion_score" in reranked[0]


def test_dedup_same_segment():
    results = [
        {"segment_id": "s1", "micro_segment_id": "m1", "score": 0.9, "third_person_clip": "x.mp4"},
        {"segment_id": "s1", "micro_segment_id": "m2", "score": 0.8, "third_person_clip": "y.mp4"},
        {"segment_id": "s2", "score": 0.7, "third_person_clip": "z.mp4"},
    ]
    reranked = rerank_results(results, dedup_same_segment=True)
    micro_ids = [r.get("micro_segment_id") for r in reranked if r.get("micro_segment_id")]
    assert "m1" in micro_ids
    assert "m2" in micro_ids
    assert len(reranked) == 3


def test_dedup_removes_duplicate_segment_id():
    results = [
        {"segment_id": "s1", "score": 0.9, "third_person_clip": "x.mp4"},
        {"segment_id": "s1", "score": 0.5, "third_person_clip": "y.mp4"},
    ]
    reranked = rerank_results(results, dedup_same_segment=True)
    assert len(reranked) == 1
    assert reranked[0]["score"] == 0.9


def test_custom_fusion_weights():
    results = [
        {"segment_id": "s1", "score": 0.3, "first_person_clip": "a.mp4", "third_person_clip": "b.mp4", "evidence": {"evidence_level": "strong"}, "quality": "high"},
        {"segment_id": "s2", "score": 0.9, "third_person_clip": "c.mp4", "evidence": {"evidence_level": "weak"}},
    ]
    reranked = rerank_results(results, fusion_weights={"text_similarity": 0.1, "evidence_strength": 0.9})
    assert reranked[0]["segment_id"] == "s1"


def test_max_results():
    results = [{"segment_id": f"s{i}", "score": float(i) / 10, "third_person_clip": "x.mp4"} for i in range(10)]
    reranked = rerank_results(results, max_results=3)
    assert len(reranked) == 3


def test_boost_factors_contain_all_keys():
    results = [{"segment_id": "s1", "score": 0.7, "third_person_clip": "x.mp4"}]
    reranked = rerank_results(results)
    factors = reranked[0]["retrieval_boost_factors"]
    assert "text_similarity" in factors
    assert "time_proximity" in factors
    assert "dual_view_consistency" in factors
    assert "evidence_strength" in factors
    assert "fusion_score" in factors
