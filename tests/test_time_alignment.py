from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.schemas import TranscriptSource, VideoSource
from key_action_indexer.evaluation_manifest import build_evaluation_manifest
from key_action_indexer.time_alignment import (
    align_transcript_to_global_time,
    evaluate_time_alignment,
    find_dialogue_for_segment,
    global_time_to_local_sec,
    local_sec_to_global_time,
    parse_time,
)


def test_video_time_mapping() -> None:
    third = VideoSource(
        name="third_person",
        path="third.mp4",
        start_time="2026-04-29T17:25:00+08:00",
        fps=30,
        offset_sec=0,
    )
    assert local_sec_to_global_time(third, 600).isoformat() == "2026-04-29T17:35:00+08:00"

    first = VideoSource(
        name="first_person",
        path="first.mp4",
        start_time="2026-04-29T17:25:02+08:00",
        fps=30,
        offset_sec=0,
    )
    assert global_time_to_local_sec(first, parse_time("2026-04-29T17:35:00+08:00")) == 598


def test_transcript_offset_and_dialogue_matching(tmp_path: Path) -> None:
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_1", "start_sec": 10, "end_sec": 12, "text": "开始移液"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    source = TranscriptSource(
        path=str(transcript_path),
        start_time="2026-04-29T17:25:00+08:00",
        offset_sec=5,
    )
    utterances = align_transcript_to_global_time(transcript_path, source)
    assert utterances[0].global_start_time == "2026-04-29T17:25:15+08:00"
    matched = find_dialogue_for_segment(
        "2026-04-29T17:25:14+08:00",
        "2026-04-29T17:25:16+08:00",
        utterances,
        window_sec=0,
    )
    assert [item.utterance_id for item in matched] == ["utt_1"]


def test_evaluate_time_alignment_outputs_required_metrics(tmp_path: Path) -> None:
    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "anchor_id": "a1",
                        "source": "first_person",
                        "expected_global_time": "2026-05-03T09:00:00+08:00",
                        "predicted_global_time": "2026-05-03T09:00:00.200000+08:00",
                    }
                ),
                json.dumps(
                    {
                        "anchor_id": "a2",
                        "source": "third_person",
                        "expected_global_time": "2026-05-03T09:01:00+08:00",
                        "predicted_global_time": "2026-05-03T09:01:00.800000+08:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = evaluate_time_alignment(anchors, output_path=tmp_path / "time_alignment_eval.json")

    assert result["metrics"]["mae_sec"] == 0.5
    assert result["metrics"]["max_residual_sec"] == 0.8
    assert result["metrics"]["anchor_coverage_rate"] == 1.0
    assert result["metrics"]["drift_error_sec"] == 0.6
    assert (tmp_path / "time_alignment_eval.json").exists()


def test_build_evaluation_manifest_requires_full_real_eval_coverage(tmp_path: Path) -> None:
    session = tmp_path / "session"
    raw = session / "raw"
    raw.mkdir(parents=True)
    third = raw / "third.mp4"
    first = raw / "first.mp4"
    dialogue = tmp_path / "dialogue.jsonl"
    sop = tmp_path / "sop.json"
    segment_labels = tmp_path / "manual_segments.jsonl"
    micro_labels = tmp_path / "manual_micro_segments.jsonl"
    expected = tmp_path / "expected_output.json"
    anchors = tmp_path / "anchors.jsonl"
    for path in (third, first, dialogue, sop, segment_labels, micro_labels, expected, anchors):
        path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "session_id": "eval_session",
                "session_start_time": "2026-05-03T09:00:00+08:00",
                "videos": {
                    "third_person": {"path": str(third), "start_time": "2026-05-03T09:00:00+08:00"},
                    "first_person": {"path": str(first), "start_time": "2026-05-03T09:00:00+08:00"},
                },
                "transcript": {"path": str(dialogue), "start_time": "2026-05-03T09:00:00+08:00"},
                "output_dir": str(session),
            }
        ),
        encoding="utf-8",
    )

    result = build_evaluation_manifest(
        manifest_path,
        sop_path=sop,
        segment_labels_path=segment_labels,
        micro_labels_path=micro_labels,
        expected_output_path=expected,
        time_alignment_anchors_path=anchors,
    )

    assert result["valid"] is True
    assert result["coverage"]["real_dual_view_video"] is True
    assert result["coverage"]["dialogue"] is True
    assert result["coverage"]["sop"] is True
    assert result["coverage"]["human_labels"] is True
    assert result["coverage"]["expected_output"] is True
    assert result["coverage"]["time_alignment_anchors"] is True
    assert Path(result["manifest_path"]).exists()
