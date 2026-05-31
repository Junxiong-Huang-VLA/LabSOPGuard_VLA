from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.pipeline import run_pipeline
from key_action_indexer.schemas import read_jsonl
from key_action_indexer.transcript_convert import convert_srt_to_jsonl, convert_transcript_to_jsonl, srt_to_transcript_rows


def test_srt_to_jsonl_conversion(tmp_path: Path) -> None:
    srt = tmp_path / "transcript.srt"
    srt.write_text(
        "1\n00:00:08,000 --> 00:00:10,500\n现在开始称量。\n\n"
        "2\n00:00:18,000 --> 00:00:20,000\n用刮勺取一点样品。\n",
        encoding="utf-8",
    )
    output = tmp_path / "transcript.jsonl"

    summary = convert_srt_to_jsonl(srt, output)
    rows = srt_to_transcript_rows(srt)

    assert summary["utterance_count"] == 2
    assert output.exists()
    assert rows[0]["start_sec"] == 8.0
    assert rows[0]["end_sec"] == 10.5
    assert rows[1]["text"] == "用刮勺取一点样品。"


def test_jsonl_transcript_mounts_to_dry_run_micro(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "现在开始称量。"}, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "session_id": "dry_asr",
                "session_start_time": "2026-04-29T17:25:00+08:00",
                "videos": {
                    "third_person": {
                        "path": str(output_dir / "raw" / "third.mp4"),
                        "start_time": "2026-04-29T17:25:00+08:00",
                        "fps": 30,
                    }
                },
                "transcript": {
                    "path": str(transcript),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "offset_sec": 0,
                },
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    run_pipeline(manifest, dry_run=True)
    micros = [
        json.loads(line)
        for line in (output_dir / "metadata" / "micro_segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert micros
    assert any(row.get("dialogue_context_available") for row in micros)


def test_json_transcript_segments_normalize_with_coverage_summary(tmp_path: Path) -> None:
    source = tmp_path / "whisper.json"
    source.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": 1, "start": 0.5, "end": 1.5, "text": "start weighing"},
                    {"id": 2, "start": 3.0, "end": 4.0, "text": "add sample"},
                    {"id": 3, "start": 5.0, "end": 5.0, "text": "bad timing"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "transcript.jsonl"
    summary_path = tmp_path / "coverage.json"

    summary = convert_transcript_to_jsonl(source, output, duration_sec=10.0, summary_output_path=summary_path)
    rows = read_jsonl(output)
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["input_format"] == "json"
    assert summary["utterance_count"] == 2
    assert summary["coverage"]["skipped_row_count"] == 1
    assert summary["coverage"]["coverage_ratio"] == 0.2
    assert saved_summary["coverage"]["gap_count"] == 1
    assert rows == [
        {"utterance_id": "1", "start_sec": 0.5, "end_sec": 1.5, "text": "start weighing"},
        {"utterance_id": "2", "start_sec": 3.0, "end_sec": 4.0, "text": "add sample"},
    ]


def test_jsonl_transcript_common_start_end_keys_normalize(tmp_path: Path) -> None:
    source = tmp_path / "asr.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"start_time": "00:00:02.000", "end_time": "00:00:03.250", "transcript": "pick bottle"}),
                json.dumps({"start_sec": 4.0, "duration_sec": 1.0, "text": "cap tube"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "normalized.jsonl"

    summary = convert_transcript_to_jsonl(source, output)
    rows = read_jsonl(output)

    assert summary["input_format"] == "jsonl"
    assert summary["coverage"]["utterance_count"] == 2
    assert rows[0]["start_sec"] == 2.0
    assert rows[0]["end_sec"] == 3.25
    assert rows[0]["text"] == "pick bottle"
    assert rows[1]["end_sec"] == 5.0
