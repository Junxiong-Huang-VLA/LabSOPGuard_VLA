from __future__ import annotations

from copy import deepcopy

from key_action_indexer.semantic_actions import WEIGHING_OPERATION, enrich_semantic_action


def _weighing_frames(object_label: str = "paper") -> list[dict]:
    frames = []
    for frame_index in range(3):
        hand_box = [92 + frame_index, 82, 130 + frame_index, 132]
        object_box = [125, 95, 188, 138]
        balance_box = [80, 70, 240, 170]
        frames.append(
            {
                "frame_index": frame_index,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.91, "bbox": hand_box},
                    {"label": object_label, "confidence": 0.88, "bbox": object_box},
                    {"label": "balance", "confidence": 0.94, "bbox": balance_box},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": object_label,
                        "score": 0.76,
                        "hand_bbox": hand_box,
                        "object_bbox": object_box,
                    }
                ],
            }
        )
    return frames


def test_enrich_semantic_action_promotes_paper_on_stable_balance_to_weighing() -> None:
    candidate = {
        "micro_segment_id": "micro_001",
        "primary_object": "paper",
        "interaction_type": "hand_paper_contact",
    }

    enriched = enrich_semantic_action(candidate, yolo_window_evidence={"frames": _weighing_frames("paper")})

    assert enriched["primary_object"] == "paper"
    assert enriched["raw_primary_object"] == "paper"
    assert enriched["manipulated_object"] == "paper"
    assert enriched["instrument_context"] == "balance"
    assert enriched["semantic_action"] == WEIGHING_OPERATION
    assert enriched["display_title"] == "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c"
    assert enriched["semantic_evidence"]["balance_stable"] is True
    assert enriched["semantic_evidence"]["balance_region_interaction"] is True
    assert "weighing_operation_priority_rule" in enriched["semantic_reasons"]


def test_enrich_semantic_action_separates_raw_balance_from_manipulated_sample_bottle() -> None:
    candidate = {
        "micro_segment_id": "micro_002",
        "primary_object": "balance",
        "interaction": {"primary_object": "balance", "interaction_type": "hand_balance_contact"},
    }

    enriched = enrich_semantic_action(candidate, yolo_window_evidence={"frames": _weighing_frames("sample_bottle")})

    assert enriched["primary_object"] == "balance"
    assert enriched["raw_primary_object"] == "balance"
    assert enriched["manipulated_object"] == "sample_bottle"
    assert enriched["instrument_context"] == "balance"
    assert enriched["semantic_action"] == WEIGHING_OPERATION
    assert enriched["display_title"] == "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c"
    assert enriched["semantic_evidence"]["manipulated_object_candidates"] == ["sample_bottle"]


def test_enrich_semantic_action_does_not_promote_outside_balance_region() -> None:
    frames = _weighing_frames("sample_bottle")
    for frame in frames:
        for detection in frame["detections"]:
            if detection["label"] == "sample_bottle":
                detection["bbox"] = [400, 400, 460, 470]
        frame["hand_object_interactions"][0]["object_bbox"] = [400, 400, 460, 470]
    candidate = {"primary_object": "sample_bottle", "interaction_type": "hand_sample_bottle_contact"}

    enriched = enrich_semantic_action(candidate, yolo_window_evidence={"frames": frames})

    assert enriched["raw_primary_object"] == "sample_bottle"
    assert enriched["manipulated_object"] == "sample_bottle"
    assert enriched["instrument_context"] is None
    assert enriched["semantic_action"] == "hand_object_interaction"
    assert enriched["semantic_evidence"]["balance_stable"] is True
    assert enriched["semantic_evidence"]["balance_region_interaction"] is False


def test_enrich_semantic_action_keeps_raw_beaker_as_manipulated_object() -> None:
    candidate = {
        "primary_object": "beaker",
        "secondary_objects": ["sample_bottle"],
        "interaction_type": "hand_beaker_contact",
    }
    frames = [
        {
            "detections": [
                {"label": "gloved_hand", "confidence": 0.86, "bbox": [90, 80, 160, 150]},
                {"label": "beaker", "confidence": 0.72, "bbox": [130, 120, 240, 240]},
                {"label": "sample_bottle", "confidence": 0.88, "bbox": [20, 120, 70, 210]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "beaker",
                    "score": 0.7,
                    "hand_bbox": [90, 80, 160, 150],
                    "object_bbox": [130, 120, 240, 240],
                }
            ],
        }
    ]

    enriched = enrich_semantic_action(candidate, yolo_window_evidence={"frames": frames})

    assert enriched["raw_primary_object"] == "beaker"
    assert enriched["manipulated_object"] == "beaker"
    assert enriched["instrument_context"] is None
    assert enriched["semantic_action"] == "hand_object_interaction"


def test_reagent_bottle_variants_share_user_visible_semantic_title() -> None:
    for label in ("reagent_bottle", "reagent_bottle_open", "bottle_cap", "sample_bottle"):
        enriched = enrich_semantic_action({"primary_object": label})

        assert enriched["raw_primary_object"] == label
        assert enriched["manipulated_object"] == label
        assert enriched["display_title"] == "手部与试剂瓶操作"


def test_business_display_titles_do_not_coarsen_internal_detection_labels() -> None:
    cases = {
        "weighing_paper": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
        "paper": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
        "balance": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
        "panel": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    }
    for label, title in cases.items():
        enriched = enrich_semantic_action({"primary_object": label})

        assert enriched["raw_primary_object"] == label
        assert enriched["manipulated_object"] == label
        assert enriched["display_title"] == title


def test_enrich_semantic_action_does_not_promote_beaker_or_reagent_bottle_to_weighing() -> None:
    for label in ("beaker", "reagent_bottle"):
        candidate = {
            "micro_segment_id": f"micro_{label}",
            "primary_object": label,
            "interaction_type": f"hand_{label}_contact",
        }

        enriched = enrich_semantic_action(candidate, yolo_window_evidence={"frames": _weighing_frames(label)})

        assert enriched["raw_primary_object"] == label
        assert enriched["manipulated_object"] == label
        assert enriched["instrument_context"] is None
        assert enriched["semantic_action"] == "hand_object_interaction"
        assert enriched["semantic_evidence"]["balance_stable"] is True
        assert enriched["semantic_evidence"]["balance_region_interaction"] is False


def test_enrich_semantic_action_accepts_explicit_region_evidence_and_does_not_mutate_input() -> None:
    candidate = {
        "primary_object": "spatula",
        "interaction_type": "hand_spatula_contact",
        "yolo_evidence": [
            {"frame_index": 1, "detections": [{"label": "balance", "bbox": [0, 0, 100, 100]}]},
            {"frame_index": 2, "detections": [{"label": "balance", "bbox": [0, 0, 100, 100]}]},
        ],
    }
    original = deepcopy(candidate)
    yolo_window = {
        "hand_object_interactions": [
            {
                "frame_index": 2,
                "hand_label": "hand",
                "object_label": "spatula",
                "in_balance_region": True,
                "score": 0.81,
            }
        ]
    }

    enriched = enrich_semantic_action(candidate, yolo_window_evidence=yolo_window)

    assert candidate == original
    assert enriched["semantic_action"] == WEIGHING_OPERATION
    assert enriched["manipulated_object"] == "spatula"
    assert enriched["instrument_context"] == "balance"
    assert enriched["display_title"] == "\u5929\u5e73\u79f0\u91cf-\u624b\u4e0e\u836f\u5319"
