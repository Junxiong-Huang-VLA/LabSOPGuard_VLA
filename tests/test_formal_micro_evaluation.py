from __future__ import annotations

from key_action_indexer.evaluation import evaluate_micro_segment_rows


def test_complete_gt_outputs_formal_metrics() -> None:
    result = evaluate_micro_segment_rows(
        [{"micro_segment_id": "pred_1", "start_sec": 1.0, "end_sec": 3.0, "primary_object": "balance"}],
        [{"micro_segment_id": "gt_1", "start_sec": 1.0, "end_sec": 3.0, "primary_object": "balance"}],
        eval_config={
            "labeled_windows": [{"window_id": "win_1", "start_sec": 0.0, "end_sec": 10.0}],
            "gt_completeness": "complete",
        },
    )

    assert result["precision_is_formal"] is True
    assert result["metric_mode"] == "formal"
    assert result["formal_metrics"]["precision"] == 1.0
    assert result["debugging_metrics"] is None


def test_partial_or_missing_eval_config_outputs_debugging_metrics() -> None:
    partial = evaluate_micro_segment_rows(
        [{"micro_segment_id": "pred_1", "start_sec": 1.0, "end_sec": 3.0}],
        [{"micro_segment_id": "gt_1", "start_sec": 1.0, "end_sec": 3.0}],
        eval_config={
            "labeled_windows": [{"window_id": "win_1", "start_sec": 0.0, "end_sec": 10.0}],
            "gt_completeness": "partial",
        },
    )
    missing = evaluate_micro_segment_rows(
        [{"micro_segment_id": "pred_1", "start_sec": 1.0, "end_sec": 3.0}],
        [{"micro_segment_id": "gt_1", "start_sec": 1.0, "end_sec": 3.0}],
    )

    assert partial["metric_mode"] == "debugging"
    assert partial["precision_is_formal"] is False
    assert missing["metric_mode"] == "debugging"
    assert missing["note"] == "GT coverage is partial or unknown; precision is for debugging only."
