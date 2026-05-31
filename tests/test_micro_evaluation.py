from __future__ import annotations

from key_action_indexer.evaluation import compute_temporal_iou, evaluate_micro_segment_rows


def test_micro_temporal_iou() -> None:
    assert compute_temporal_iou({"start_sec": 0, "end_sec": 10}, {"start_sec": 5, "end_sec": 15}) == 5 / 15


def test_micro_evaluation_reports_object_accuracy() -> None:
    predicted = [
        {
            "micro_segment_id": "pred_1",
            "start_sec": 0,
            "end_sec": 10,
            "interaction": {"primary_object": "balance", "interaction_type": "hand_balance_contact"},
            "text_description": {"action_type": "weighing"},
        }
    ]
    gt = [
        {
            "micro_segment_id": "gt_1",
            "start_sec": 1,
            "end_sec": 9,
            "primary_object": "balance",
            "interaction_type": "hand_balance_contact",
            "action_type": "weighing",
        }
    ]
    result = evaluate_micro_segment_rows(predicted, gt, iou_threshold=0.3)
    assert result["true_positive"] == 1
    assert result["primary_object_accuracy"] == 1.0
    assert result["interaction_type_accuracy"] == 1.0
    assert result["action_type_accuracy"] == 1.0


def test_micro_evaluation_empty_inputs() -> None:
    result = evaluate_micro_segment_rows([], [], iou_threshold=0.3)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0

