"""End-to-end test: fusion_weights flow through VectorIndex.query()."""
from __future__ import annotations

from key_action_indexer.vector_index import VectorIndex, rerank_results


def test_fusion_weights_through_query():
    index = VectorIndex()
    texts = [
        "操作员使用移液枪吸取200微升溶液",
        "天平称量样品 spatula 取样",
        "记录实验数据 观察样品颜色",
    ]
    metadata = [
        {
            "segment_id": "s1",
            "index_text": texts[0],
            "first_person_clip": "fp1.mp4",
            "third_person_clip": "tp1.mp4",
            "evidence": {"evidence_level": "strong"},
            "quality": "high",
        },
        {
            "segment_id": "s2",
            "index_text": texts[1],
            "third_person_clip": "tp2.mp4",
            "evidence": {"evidence_level": "weak"},
            "quality": "low",
        },
        {
            "segment_id": "s3",
            "index_text": texts[2],
            "first_person_clip": "fp3.mp4",
            "third_person_clip": "tp3.mp4",
            "evidence": {"evidence_level": "moderate"},
            "quality": "medium",
        },
    ]
    index.build(texts, metadata)

    results_no_fusion = index.query("移液枪操作", top_k=3)
    results_with_fusion = index.query(
        "移液枪操作",
        top_k=3,
        fusion_weights={"text_similarity": 0.3, "evidence_strength": 0.7},
    )

    assert len(results_with_fusion) > 0
    for result in results_with_fusion:
        assert "retrieval_boost_factors" in result
        assert "fusion_score" in result
        factors = result["retrieval_boost_factors"]
        assert "text_similarity" in factors
        assert "evidence_strength" in factors


def test_fusion_weights_affects_ranking():
    index = VectorIndex()
    texts = ["pipette transfer liquid", "balance weighing"]
    metadata = [
        {
            "segment_id": "s1",
            "index_text": texts[0],
            "third_person_clip": "tp.mp4",
            "evidence": {"evidence_level": "weak"},
            "quality": "low",
            "score": 0.9,
        },
        {
            "segment_id": "s2",
            "index_text": texts[1],
            "first_person_clip": "fp.mp4",
            "third_person_clip": "tp.mp4",
            "evidence": {"evidence_level": "strong"},
            "quality": "high",
            "score": 0.3,
        },
    ]
    index.build(texts, metadata)

    results = index.query("pipette", top_k=2, fusion_weights={"text_similarity": 0.1, "evidence_strength": 0.9})
    assert len(results) == 2
    assert all("fusion_score" in r for r in results)
