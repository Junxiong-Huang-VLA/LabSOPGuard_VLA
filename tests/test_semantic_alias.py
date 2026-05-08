from __future__ import annotations

from key_action_indexer.semantic_alias import expand_query, infer_action_type_from_metadata, score_query_metadata_match


def test_expand_query_maps_bottle_alias() -> None:
    expanded = expand_query("\u624b\u78b0\u74f6\u5b50")
    assert "reagent_bottle" in expanded["target_objects"]
    assert "bottle" in expanded["target_objects"]


def test_expand_query_maps_weighing_alias() -> None:
    expanded = expand_query("\u79f0\u91cf")
    assert "balance" in expanded["target_objects"]


def test_expand_query_maps_spatula_alias() -> None:
    expanded = expand_query("\u4f7f\u7528\u522e\u52fa")
    assert expanded["canonical_action"] == "\u4f7f\u7528\u522e\u52fa"
    assert "spatula" in expanded["target_objects"]


def test_expand_query_maps_pipetting_alias() -> None:
    expanded = expand_query("\u52a0\u6837")
    assert "pipette" in expanded["target_objects"]
    assert "pipette_tip" in expanded["target_objects"]
    assert "tube" in expanded["target_objects"]


def test_infer_action_type_from_metadata_uses_visual_and_dialogue() -> None:
    assert infer_action_type_from_metadata({"primary_object": "balance"}) == "weighing"
    assert infer_action_type_from_metadata({"primary_object": "spatula"}) == "spatula_interaction"
    assert infer_action_type_from_metadata({"primary_object": "reagent_bottle"}, "\u52a0\u6837 200 \u5fae\u5347") == "pipetting"


def test_score_query_metadata_match_protects_pipetting_without_evidence() -> None:
    result = score_query_metadata_match(
        "\u52a0\u6837",
        {
            "index_level": "micro_segment",
            "primary_object": "reagent_bottle",
            "interaction_type": "hand_reagent_bottle_contact",
            "detected_objects": ["reagent_bottle"],
            "related_dialogue": [],
        },
    )
    assert result["rerank_score"] == 0.0
    assert "insufficient_pipette_or_dialogue_evidence" in result["rerank_reasons"]

