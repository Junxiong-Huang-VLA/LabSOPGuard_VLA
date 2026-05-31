from __future__ import annotations

from key_action_indexer.semantic_alias import expand_query, infer_action_type_from_metadata, score_query_metadata_match


def test_expand_query_maps_bottle_alias() -> None:
    expanded = expand_query("\u624b\u78b0\u74f6\u5b50")
    assert "reagent_bottle" in expanded["target_objects"]
    assert "reagent_bottle_open" in expanded["target_objects"]
    assert "bottle_cap" in expanded["target_objects"]
    assert "bottle" in expanded["target_objects"]


def test_expand_query_maps_reagent_bottle_business_action() -> None:
    expanded = expand_query("\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c")
    assert expanded["canonical_action"] == "reagent_bottle_interaction"
    assert {"reagent_bottle", "reagent_bottle_open", "bottle_cap", "sample_bottle"}.issubset(set(expanded["target_objects"]))


def test_expand_query_maps_paper_and_balance_panel_business_actions() -> None:
    paper = expand_query("\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c")
    panel = expand_query("\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c")

    assert paper["canonical_action"] == "weighing_paper_operation"
    assert {"paper", "weighing_paper"}.issubset(set(paper["target_objects"]))
    assert panel["canonical_action"] == "equipment_panel_operation"
    assert {"balance", "panel"}.issubset(set(panel["target_objects"]))


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


def test_expand_query_separates_chinese_recording_and_sample_handling() -> None:
    assert expand_query("\u67e5\u627e\u8bb0\u5f55\u8bfb\u6570\u7247\u6bb5")["canonical_action"] == "recording"
    assert expand_query("\u67e5\u627e\u6837\u54c1\u5904\u7406\u7247\u6bb5")["canonical_action"] == "sample_handling"


def test_infer_action_type_from_metadata_uses_visual_and_dialogue() -> None:
    assert infer_action_type_from_metadata({"primary_object": "balance"}) == "equipment_panel_operation"
    assert infer_action_type_from_metadata({"primary_object": "paper"}) == "weighing_paper_operation"
    assert infer_action_type_from_metadata({"primary_object": "spatula"}) == "spatula_interaction"
    assert infer_action_type_from_metadata({"primary_object": "reagent_bottle"}, "\u52a0\u6837 200 \u5fae\u5347") == "pipetting"
    assert infer_action_type_from_metadata({"primary_object": "sample_bottle"}) == "reagent_bottle_interaction"


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


def test_score_query_metadata_match_prioritizes_distinct_semantic_targets() -> None:
    sample = score_query_metadata_match(
        "sample handling",
        {
            "index_level": "segment",
            "primary_object": "reagent_bottle",
            "action_type": "reagent_bottle_interaction",
            "interaction_type": "hand_reagent_bottle_contact",
            "detected_objects": ["reagent_bottle"],
        },
    )
    paper = score_query_metadata_match(
        "sample handling",
        {
            "index_level": "segment",
            "primary_object": "paper",
            "action_type": "hand_object_interaction",
            "interaction_type": "hand_paper_contact",
            "detected_objects": ["paper"],
        },
    )
    recording = score_query_metadata_match(
        "recording balance readout",
        {
            "index_level": "segment",
            "primary_object": "paper",
            "action_type": "hand_object_interaction",
            "interaction_type": "hand_paper_contact",
            "detected_objects": ["paper", "balance"],
            "start_sec": 52.0,
            "duration_sec": 12.0,
        },
    )

    assert sample["rerank_score"] > paper["rerank_score"]
    assert "sample_handling_object_priority" in sample["rerank_reasons"]
    assert "recording_late_window_candidate" in recording["rerank_reasons"]

