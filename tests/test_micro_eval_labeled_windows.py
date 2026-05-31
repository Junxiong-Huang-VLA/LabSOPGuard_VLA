from __future__ import annotations

from key_action_indexer.evaluation import evaluate_micro_segment_rows


def test_labeled_windows_exclude_unlabeled_false_positives() -> None:
    predicted = [
        {"micro_segment_id": "pred_1", "start_sec": 1.0, "end_sec": 3.0, "primary_object": "bottle"},
        {"micro_segment_id": "pred_2", "start_sec": 5.0, "end_sec": 6.0, "primary_object": "spatula"},
        {"micro_segment_id": "pred_outside", "start_sec": 20.0, "end_sec": 22.0, "primary_object": "balance"},
    ]
    ground_truth = [
        {"micro_segment_id": "gt_1", "start_sec": 1.0, "end_sec": 3.0, "primary_object": "bottle"},
    ]
    result = evaluate_micro_segment_rows(
        predicted,
        ground_truth,
        eval_config={
            "labeled_windows": [{"window_id": "win_1", "start_sec": 0.0, "end_sec": 10.0}],
            "gt_completeness": "partial",
        },
    )

    assert result["num_predicted"] == 2
    assert result["predictions_outside_labeled_windows"] == 1
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["precision"] == 0.5
    assert result["precision_is_formal"] is False


def test_complete_labeled_window_marks_precision_formal() -> None:
    result = evaluate_micro_segment_rows(
        [{"micro_segment_id": "pred_1", "start_sec": 1.0, "end_sec": 3.0}],
        [{"micro_segment_id": "gt_1", "start_sec": 1.0, "end_sec": 3.0}],
        eval_config={
            "labeled_windows": [{"window_id": "win_1", "start_sec": 0.0, "end_sec": 10.0}],
            "gt_completeness": "complete",
        },
    )

    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["precision_is_formal"] is True
