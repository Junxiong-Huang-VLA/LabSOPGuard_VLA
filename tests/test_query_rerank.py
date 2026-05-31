from __future__ import annotations

from key_action_indexer.vector_index import EmbeddingBackend, VectorIndex


def _index(rows: list[dict]) -> VectorIndex:
    return VectorIndex(EmbeddingBackend(kind="hashing")).build([row["index_text"] for row in rows], rows)


def test_bottle_micro_gets_boost_for_hand_touch_bottle_query() -> None:
    rows = [
        {
            "segment_id": "seg_1",
            "micro_segment_id": "m1",
            "index_level": "micro_segment",
            "primary_object": "reagent_bottle",
            "interaction_type": "hand_reagent_bottle_contact",
            "detected_objects": ["hand", "reagent_bottle"],
            "action_type": "reagent_bottle_interaction",
            "keyframes": ["peak.jpg"],
            "class_threshold": {"query_boost": 1.25},
            "index_text": "hand reagent_bottle contact",
        }
    ]
    result = _index(rows).query("\u624b\u78b0\u74f6\u5b50", top_k=1)[0]
    assert result["micro_segment_id"] == "m1"
    assert result["rerank_score"] > 0
    assert any("matched_primary_object" in reason for reason in result["rerank_reasons"])


def test_balance_micro_gets_boost_for_weighing_query() -> None:
    rows = [
        {
            "segment_id": "seg_1",
            "micro_segment_id": "m1",
            "index_level": "micro_segment",
            "primary_object": "balance",
            "interaction_type": "hand_balance_contact",
            "detected_objects": ["balance"],
            "action_type": "weighing",
            "keyframes": ["peak.jpg"],
            "class_threshold": {"query_boost": 1.2},
            "index_text": "balance weighing",
        }
    ]
    result = _index(rows).query("\u79f0\u91cf", top_k=1)[0]
    assert result["primary_object"] == "balance"
    assert result["rerank_score"] > 0


def test_spatula_micro_gets_class_specific_boost() -> None:
    rows = [
        {
            "segment_id": "seg_1",
            "micro_segment_id": "m1",
            "index_level": "micro_segment",
            "primary_object": "spatula",
            "interaction_type": "hand_spatula_contact",
            "detected_objects": ["spatula"],
            "action_type": "spatula_interaction",
            "keyframes": ["peak.jpg"],
            "class_threshold": {"query_boost": 1.5},
            "index_text": "spatula hand contact \u4f7f\u7528\u522e\u52fa",
        }
    ]
    result = _index(rows).query("\u4f7f\u7528\u522e\u52fa", top_k=1)[0]
    assert result["primary_object"] == "spatula"
    assert any("class_specific_query_boost" in reason for reason in result["rerank_reasons"])
    assert any("index_level_boost" in reason for reason in result["rerank_reasons"])


def test_pipetting_without_pipette_or_asr_is_not_forced_to_micro() -> None:
    rows = [
        {
            "segment_id": "seg_parent",
            "index_level": "segment",
            "action_type": "weighing",
            "index_text": "parent segment with reagent bottle and balance",
        },
        {
            "segment_id": "seg_1",
            "micro_segment_id": "m1",
            "index_level": "micro_segment",
            "primary_object": "reagent_bottle",
            "interaction_type": "hand_reagent_bottle_contact",
            "detected_objects": ["reagent_bottle"],
            "action_type": "reagent_bottle_interaction",
            "index_text": "reagent_bottle contact",
        },
    ]
    result = _index(rows).query("\u52a0\u6837", top_k=1)[0]
    assert result["index_level"] == "segment"
    assert result.get("micro_segment_id") is None

