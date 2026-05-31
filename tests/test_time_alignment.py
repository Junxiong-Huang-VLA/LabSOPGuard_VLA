from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer.schemas import DetectedSegment, SessionManifest, SessionVideos, TranscriptSource, VideoSource
from key_action_indexer.evaluation_manifest import build_evaluation_manifest
from key_action_indexer.time_alignment import (
    align_transcript_to_global_time,
    evaluate_time_alignment,
    find_dialogue_for_segment,
    generate_multimodal_alignment,
    global_time_to_video_sec,
    global_time_to_local_sec,
    local_sec_to_global_time,
    parse_time,
    strict_common_overlap_from_view_intervals,
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


def test_global_time_to_video_sec_uses_source_capture_seconds_with_frames_csv(tmp_path: Path) -> None:
    video = tmp_path / "first.mp4"
    video.write_bytes(b"fake-video")
    frames_csv = tmp_path / "frames.csv"
    frames_csv.write_text(
        "stream_type,local_time_us\n"
        "rgb,1000000\n"
        "rgb,2000000\n"
        "rgb,3000000\n"
        "rgb,4000000\n"
        "rgb,5000000\n",
        encoding="utf-8",
    )
    source = VideoSource(
        name="first_person",
        path=str(video),
        start_time="2026-05-20T09:52:25+08:00",
        fps=30,
        duration_sec=2.0,
        frames_csv_path=str(frames_csv),
        capture_start_time="2026-05-20T09:52:25+08:00",
        capture_start_source="frames_csv_first_rgb_local_time_us",
        capture_start_status="capture_metadata",
    )

    assert global_time_to_local_sec(source, "2026-05-20T09:52:27+08:00") == 2.0
    assert global_time_to_video_sec(source, "2026-05-20T09:52:27+08:00") == 2.0


def test_strict_common_overlap_clamps_stale_payload_to_view_intersection() -> None:
    overlap = strict_common_overlap_from_view_intervals(
        {
            "first_person": {"global_start_sec": 0.0, "global_end_sec": 5737.247},
            "third_person": {"global_start_sec": 0.0, "global_end_sec": 6349.512},
        },
        source="existing_timeline_alignment",
        requested_overlap={"common_overlap": {"global_start_sec": 0.0, "global_end_sec": 7501.0}},
    )

    assert overlap is not None
    assert overlap["available"] is True
    assert overlap["global_start_sec"] == pytest.approx(0.0)
    assert overlap["global_end_sec"] == pytest.approx(5737.247)
    assert overlap["duration_sec"] == pytest.approx(5737.247)
    assert overlap["requested_common_overlap"]["global_end_sec"] == pytest.approx(7501.0)
    assert overlap["requested_common_overlap_clamped"] is True
    assert overlap["views"]["first_person"]["global_end"] == pytest.approx(5737.247)


def test_generate_multimodal_alignment_uses_source_capture_seconds_with_frames_csv(tmp_path: Path) -> None:
    first_video = tmp_path / "first.mp4"
    third_video = tmp_path / "third.mp4"
    first_video.write_bytes(b"fake-video")
    third_video.write_bytes(b"fake-video")
    frames_csv = tmp_path / "frames.csv"
    frames_csv.write_text(
        "stream_type,local_time_us\n"
        "rgb,1000000\n"
        "rgb,2000000\n"
        "rgb,3000000\n"
        "rgb,4000000\n"
        "rgb,5000000\n",
        encoding="utf-8",
    )
    first = VideoSource(
        name="first_person",
        path=str(first_video),
        start_time="2026-05-20T09:52:25+08:00",
        fps=30,
        duration_sec=2.0,
        frames_csv_path=str(frames_csv),
        capture_start_time="2026-05-20T09:52:25+08:00",
        capture_start_source="frames_csv_first_rgb_local_time_us",
        capture_start_status="capture_metadata",
    )
    third = VideoSource(
        name="third_person",
        path=str(third_video),
        start_time="2026-05-20T09:52:25+08:00",
        fps=30,
        duration_sec=10.0,
    )
    manifest = SessionManifest(
        session_id="pts_alignment",
        session_start_time="2026-05-20T09:52:25+08:00",
        videos=SessionVideos(third_person=third, first_person=first),
    )
    segment = DetectedSegment(
        segment_id="seg_001",
        start_sec=0.0,
        end_sec=2.0,
        duration_sec=2.0,
        global_start_time="2026-05-20T09:52:27+08:00",
        global_end_time="2026-05-20T09:52:29+08:00",
        avg_motion_score=1.0,
        avg_active_score=1.0,
        start_reason="test",
        end_reason="test",
    )

    rows = generate_multimodal_alignment(manifest, [segment], [], tmp_path / "alignment.jsonl")

    first_stream = rows[0]["streams"]["first_person"]
    assert first_stream["local_start_sec"] == pytest.approx(2.0)
    assert first_stream["local_end_sec"] == pytest.approx(4.0)


def test_dual_view_source_windows_preserve_session_duration_with_capture_start_offset(tmp_path: Path) -> None:
    third_video = tmp_path / "third.mp4"
    first_video = tmp_path / "first.mp4"
    frames_csv = tmp_path / "frames.csv"
    third_video.write_bytes(b"fake-video")
    first_video.write_bytes(b"fake-video")
    frames_csv.write_text(
        "stream_type,local_time_us\n"
        "rgb,1000000\n"
        "rgb,1001000\n"
        "rgb,1002000\n"
        "rgb,5000000000\n",
        encoding="utf-8",
    )
    third = VideoSource(
        name="third_person",
        path=str(third_video),
        start_time="2026-05-20T09:52:24.981698+08:00",
        fps=30,
        duration_sec=5000.0,
        frames_csv_path=str(frames_csv),
    )
    first = VideoSource(
        name="first_person",
        path=str(first_video),
        start_time="2026-05-20T09:52:25.092162+08:00",
        fps=30,
        duration_sec=5000.0,
        frames_csv_path=str(frames_csv),
    )
    manifest = SessionManifest(
        session_id="source_window_alignment",
        session_start_time="2026-05-20T09:52:24.981698+08:00",
        videos=SessionVideos(third_person=third, first_person=first),
    )
    segment = DetectedSegment(
        segment_id="seg_001",
        start_sec=924.506,
        end_sec=1334.24,
        duration_sec=409.734,
        global_start_time="2026-05-20T10:07:49.487698+08:00",
        global_end_time="2026-05-20T10:14:39.221698+08:00",
        avg_motion_score=1.0,
        avg_active_score=1.0,
        start_reason="test",
        end_reason="test",
    )

    rows = generate_multimodal_alignment(manifest, [segment], [], tmp_path / "alignment.jsonl")

    streams = rows[0]["streams"]
    third_stream = streams["third_person"]
    first_stream = streams["first_person"]
    segment_duration = segment.duration_sec
    capture_start_offset = parse_time(first.start_time) - parse_time(third.start_time)
    assert third_stream["local_end_sec"] - third_stream["local_start_sec"] == pytest.approx(segment_duration)
    assert first_stream["local_end_sec"] - first_stream["local_start_sec"] == pytest.approx(segment_duration)
    assert third_stream["local_start_sec"] - first_stream["local_start_sec"] == pytest.approx(
        capture_start_offset.total_seconds()
    )


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
