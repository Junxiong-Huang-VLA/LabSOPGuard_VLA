from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.context_fusion import build_experiment_context, load_experiment_context
from key_action_indexer.schemas import write_jsonl


def test_build_experiment_context_from_text_video_upload_and_database(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    transcript_dir = tmp_path / "transcript"
    metadata.mkdir()
    transcript_dir.mkdir()
    write_jsonl(
        metadata / "unified_multimodal_timeline.jsonl",
        [
            {"timeline_event_id": "user_1", "event_type": "user_text", "text": "实验目的是称量样品并用移液枪加样 200 微升。"},
            {"timeline_event_id": "ai_1", "event_type": "ai_reply", "text": "下一步记录天平读数。"},
            {"timeline_event_id": "up_1", "event_type": "upload", "text": "上传了样品瓶照片。"},
        ],
    )
    write_jsonl(transcript_dir / "aligned_transcript.jsonl", [{"utterance_id": "utt_1", "text": "接下来使用移液枪加 200 微升。"}])
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [{"asset_id": "asset_1", "objects": ["sample_bottle"], "actions": ["pipetting"], "search_text": "sample bottle pipette"}],
    )
    write_jsonl(
        metadata / "video_understanding.jsonl",
        [{"video_event_id": "vu_1", "event_type": "liquid_transfer_candidate", "action_type": "pipetting", "primary_object": "pipette", "confidence": 0.6}],
    )
    database = tmp_path / "history.jsonl"
    write_jsonl(database, [{"record_id": "db_1", "text": "历史实验包含 balance 和 tube。"}])

    result = build_experiment_context(tmp_path, database_paths=[database])
    loaded = load_experiment_context(metadata / "experiment_context.json")
    actions = {item["action_type"] for item in result["procedure_candidates"]}
    materials = {item["name"] for item in result["materials"]}
    reagents = {item["name"] for item in result["reagents"]}
    equipment = {item["name"] for item in result["equipment"]}

    assert loaded["session_id"] == result["session_id"]
    assert {"weighing", "pipetting", "recording"}.issubset(actions)
    assert {"pipette", "sample_bottle", "tube", "balance"}.issubset(materials)
    assert "sample" in reagents
    assert {"balance", "pipette"}.issubset(equipment)
    assert any(param["value"] == 200.0 and "微升" in param["unit"] for param in result["parameters"])
    assert result["source_counts"]["video_events"] == 1
    assert result["source_counts"]["database_rows"] == 1
    assert result["confidence"] > 0.7


def test_context_fusion_tolerates_missing_video_understanding(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "user_1", "event_type": "user_text", "text": "称量样品。"}])

    result = build_experiment_context(tmp_path)

    assert result["source_counts"]["video_events"] == 0
    assert "missing_video_understanding" in result["gaps"]
    assert result["procedure_candidates"][0]["action_type"] == "weighing"
