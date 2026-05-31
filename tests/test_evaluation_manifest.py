from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.evaluation_manifest import build_evaluation_manifest, build_micro_gt_template_manifest
from key_action_indexer.schemas import read_jsonl, write_jsonl


def _write_manifest(path: Path, output_dir: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "session_id": "eval_manifest_session",
                "session_start_time": "2026-04-29T17:25:00+08:00",
                "videos": {
                    "third_person": {
                        "path": str(output_dir / "raw" / "third.mp4"),
                        "start_time": "2026-04-29T17:25:00+08:00",
                        "fps": 30,
                    }
                },
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_build_micro_gt_template_manifest_from_segments(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    micro_path = metadata / "micro_segments.jsonl"
    key_path = metadata / "key_action_segments.jsonl"
    write_jsonl(
        key_path,
        [
            {
                "segment_id": "seg_001",
                "start_sec": 10.0,
                "end_sec": 20.0,
                "text_description": {"action_type": "weighing", "summary": "weigh sample"},
            }
        ],
    )
    write_jsonl(
        micro_path,
        [
            {
                "micro_segment_id": "seg_001_micro_001",
                "parent_segment_id": "seg_001",
                "start_sec": 11.0,
                "end_sec": 12.5,
                "interaction": {"primary_object": "balance", "interaction_type": "hand_balance_contact"},
                "text_description": {"action_type": "weighing", "summary": "hand on balance"},
            }
        ],
    )

    summary = build_micro_gt_template_manifest(session)
    template = read_jsonl(session / "annotation" / "micro_gt" / "manual_micro_gt.template.jsonl")
    eval_config = json.loads((session / "annotation" / "micro_gt" / "eval_config.json").read_text(encoding="utf-8"))

    assert summary["micro_segment_count"] == 1
    assert template[0]["segment_id"] == "seg_001"
    assert template[0]["micro_segment_id"] == "seg_001_micro_001"
    assert template[0]["primary_object"] == "balance"
    assert template[0]["action_type"] == "weighing"
    assert template[0]["needs_manual_label"] is True
    assert "manual_fields_required" in template[0]
    assert eval_config["gt_completeness"] == "unknown"
    assert eval_config["labeled_window_count"] == 1
    assert eval_config["metric_mode"] == "debugging_until_complete_gt"
    assert eval_config["precision_is_formal"] is False


def test_micro_gt_template_cli_runs_without_video_or_ffmpeg(tmp_path: Path) -> None:
    session = tmp_path / "session"
    write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [{"micro_segment_id": "m1", "parent_segment_id": "s1", "start_sec": 1.0, "end_sec": 2.0}],
    )
    write_jsonl(
        session / "metadata" / "key_action_segments.jsonl",
        [{"segment_id": "s1", "start_sec": 0.0, "end_sec": 3.0}],
    )

    exit_code = main(["micro-gt-template", "--session-dir", str(session)])

    assert exit_code == 0
    assert (session / "annotation" / "micro_gt" / "manual_micro_gt.template.jsonl").exists()
    assert (session / "annotation" / "micro_gt" / "eval_config.json").exists()


def test_evaluation_manifest_marks_missing_complete_gt_as_debugging(tmp_path: Path) -> None:
    session = tmp_path / "session"
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, session)
    micro_gt = tmp_path / "manual_micro_gt.jsonl"
    eval_config = tmp_path / "eval_config.json"
    write_jsonl(micro_gt, [{"micro_segment_id": "gt1", "start_sec": 1.0, "end_sec": 2.0, "primary_object": "balance"}])
    eval_config.write_text(
        json.dumps(
            {
                "gt_completeness": "partial",
                "labeled_windows": [{"window_id": "w1", "start_sec": 0.0, "end_sec": 5.0}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_evaluation_manifest(manifest_path, micro_labels_path=micro_gt, eval_config_path=eval_config)

    assert result["gt_completeness"] == "partial"
    assert result["labeled_window_count"] == 1
    assert result["metric_mode"] == "debugging"
    assert result["precision_recall_are_formal"] is False
    assert result["ground_truth_coverage"]["metric_mode"] == "debugging"
