from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.schemas import write_jsonl


def test_reasoning_cli_commands_build_artifacts(tmp_path: Path, capsys) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    transcript = session / "transcript"
    metadata.mkdir(parents=True)
    transcript.mkdir()
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "cli_reasoning",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "interaction": {
                    "interaction_type": "hand_pipette_contact",
                    "primary_object": "pipette",
                    "detected_objects": ["hand", "pipette"],
                    "max_interaction_score": 0.8,
                },
                "keyframes": {"peak_frame": "keyframes/peak.jpg"},
                "text_description": {"action_type": "pipetting", "summary": "pipette 200 微升"},
                "quality": {"confidence": "high"},
                "evidence": {"evidence_level": "visual_confirmed"},
            }
        ],
    )
    write_jsonl(metadata / "state_change_index.jsonl", [])
    write_jsonl(metadata / "material_asset_catalog.jsonl", [])
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [{"timeline_event_id": "user_1", "event_type": "user_text", "text": "使用移液枪加样 200 微升。"}])
    write_jsonl(transcript / "aligned_transcript.jsonl", [{"utterance_id": "utt_1", "text": "使用移液枪加样。"}])

    assert main(["advanced-vision", "--session-dir", str(session)]) == 0
    capsys.readouterr()
    assert main(["understand-video", "--session-dir", str(session)]) == 0
    capsys.readouterr()
    assert main(["context", "--session-dir", str(session)]) == 0
    capsys.readouterr()
    assert main(["process", "--session-dir", str(session)]) == 0
    output = capsys.readouterr().out
    result = json.loads(output)
    history_model = tmp_path / "history_model.json"
    assert main(["history-model", "--source", str(session), "--output", str(history_model)]) == 0
    capsys.readouterr()
    inventory_output = tmp_path / "model_inventory.json"
    assert main(["model-inventory", "--project-root", str(Path.cwd()), "--output", str(inventory_output)]) == 0
    capsys.readouterr()
    assert main(["confirmation-queue", "--session-dir", str(session)]) == 0
    queue_output = capsys.readouterr().out
    queue_summary = json.loads(queue_output)

    assert (metadata / "video_understanding.jsonl").exists()
    assert (metadata / "experiment_context.json").exists()
    assert (metadata / "experiment_process.json").exists()
    assert history_model.exists()
    assert inventory_output.exists()
    assert (metadata / "human_confirmation_queue.jsonl").exists()
    assert result["step_count"] >= 1
    assert queue_summary["item_count"] >= 0
