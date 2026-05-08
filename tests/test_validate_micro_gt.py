from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.micro_gt_validation import validate_micro_gt
from key_action_indexer.schemas import write_jsonl


def test_validate_micro_gt_warns_for_outside_window_overlap_and_too_few_complete_gt(tmp_path: Path) -> None:
    gt = tmp_path / "manual_micro.jsonl"
    config = tmp_path / "eval_config.json"
    write_jsonl(
        gt,
        [
            {"micro_segment_id": "gt_001", "start_sec": 1.0, "end_sec": 5.0, "primary_object": "sample_bottle", "interaction_type": "hand_contact", "action_type": "bottle_interaction"},
            {"micro_segment_id": "gt_002", "start_sec": 2.0, "end_sec": 5.5, "primary_object": "unknown_tool", "interaction_type": "hand_contact", "action_type": ""},
            {"micro_segment_id": "gt_003", "start_sec": 80.0, "end_sec": 82.0, "primary_object": "balance", "interaction_type": "hand_balance_contact", "action_type": "weighing"},
        ],
    )
    config.write_text(
        json.dumps(
            {
                "labeled_windows": [{"window_id": "win_001", "start_sec": 0.0, "end_sec": 68.7}],
                "gt_completeness": "complete",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_micro_gt(gt, config, output_path=tmp_path / "micro_gt_validation.json")
    warning_types = {item["type"] for item in result["warnings"]}

    assert result["valid"] is True
    assert "gt_outside_labeled_windows" in warning_types
    assert "unknown_primary_object" in warning_types
    assert "missing_action_type" in warning_types
    assert "severe_gt_overlap" in warning_types
    assert "complete window has very few GT segments; verify annotation completeness" in warning_types


def test_validate_micro_gt_marks_template_rows_as_non_formal(tmp_path: Path) -> None:
    gt = tmp_path / "manual_micro_gt.template.jsonl"
    config = tmp_path / "eval_config.json"
    write_jsonl(
        gt,
        [
            {
                "micro_segment_id": "pred_001",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "primary_object": "balance",
                "interaction_type": "hand_balance_contact",
                "action_type": "weighing",
                "needs_manual_label": True,
                "manual_review_status": "unlabeled",
            }
        ],
    )
    config.write_text(
        json.dumps(
            {
                "labeled_windows": [{"window_id": "win_001", "start_sec": 0.0, "end_sec": 5.0}],
                "gt_completeness": "unknown",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = validate_micro_gt(gt, config)
    warning_types = {item["type"] for item in result["warnings"]}

    assert result["metric_mode"] == "debugging"
    assert result["precision_is_formal"] is False
    assert "manual_labeling_required" in warning_types
