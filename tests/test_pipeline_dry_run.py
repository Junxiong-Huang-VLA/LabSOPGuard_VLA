from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.pipeline import run_pipeline
from key_action_indexer.vector_index import VectorIndex


def test_dry_run_pipeline_outputs_and_query(tmp_path: Path) -> None:
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "现在开始称量样品。"}, ensure_ascii=False),
                json.dumps({"utterance_id": "utt_002", "start_sec": 620.0, "end_sec": 628.0, "text": "接下来使用移液枪加 200 微升。"}, ensure_ascii=False),
                json.dumps({"utterance_id": "utt_003", "start_sec": 900.0, "end_sec": 910.0, "text": "记录一下天平读数。"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "test_session",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {
                "path": str(output_dir / "raw" / "third_person.mp4"),
                "start_time": "2026-04-29T17:25:00+08:00",
                "fps": 30,
                "offset_sec": 0,
            },
            "first_person": {
                "path": str(output_dir / "raw" / "first_person.mp4"),
                "start_time": "2026-04-29T17:25:02+08:00",
                "fps": 30,
                "offset_sec": 0,
            },
        },
        "transcript": {
            "path": str(transcript_path),
            "start_time": "2026-04-29T17:25:00+08:00",
            "offset_sec": 0,
        },
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)
    assert summary["segment_count"] >= 3
    assert (output_dir / "metadata" / "key_action_segments.jsonl").exists()
    assert (output_dir / "uploads").exists()
    assert (output_dir / "exports").exists()
    assert (output_dir / "metadata" / "video_sources.jsonl").exists()
    assert (output_dir / "metadata" / "user_text_events.jsonl").exists()
    assert (output_dir / "metadata" / "ai_reply_events.jsonl").exists()
    assert (output_dir / "metadata" / "upload_events.jsonl").exists()
    assert (output_dir / "metadata" / "input_ingestion_summary.json").exists()
    assert (output_dir / "metadata" / "vector_metadata.jsonl").exists()

    index = VectorIndex.load(output_dir / "index")
    results = index.query("找一下使用移液枪加样的片段", top_k=1)
    assert results
    assert results[0]["action_type"] == "pipetting"
