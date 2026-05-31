from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.cli import main
from key_action_indexer.schemas import write_jsonl


def test_assets_cli_builds_state_index_catalog_and_searches(tmp_path: Path, capsys) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    clip_path = session / "clips" / "seg_001" / "micro_001_third_person.mp4"
    keyframe_path = session / "keyframes" / "micro_001" / "peak.jpg"
    metadata.mkdir(parents=True)
    clip_path.parent.mkdir(parents=True)
    keyframe_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"DRY RUN CLIP 0.000-2.000\n")
    keyframe_path.write_bytes(b"DRY RUN MICRO KEYFRAME 1.000\n")

    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "cli_session",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 3.0,
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "third_person": {"clip_path": "clips/seg_001/micro_001_third_person.mp4"},
                "interaction": {
                    "interaction_type": "hand_pipette_contact",
                    "primary_object": "pipette",
                    "contact_start_sec": 1.0,
                    "peak_interaction_sec": 2.0,
                    "contact_end_sec": 3.0,
                },
                "keyframes": {"peak_frame": "keyframes/micro_001/peak.jpg"},
                "text_description": {
                    "action_type": "pipetting",
                    "summary": "operator uses pipette",
                    "index_text": "pipette contact material evidence",
                },
                "evidence": {"evidence_level": "strong_visual"},
            }
        ],
    )
    write_jsonl(metadata / "key_action_segments.jsonl", [{"session_id": "cli_session", "segment_id": "seg_001"}])
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [])

    assert main(["assets", "--session-dir", str(session)]) == 0
    capsys.readouterr()

    assert (metadata / "state_change_index.jsonl").exists()
    assert (metadata / "material_asset_catalog.jsonl").exists()
    assert (metadata / "material_library_summary.json").exists()

    assert (
        main(
            [
                "search-assets",
                "--session-dir",
                str(session),
                "--query",
                "pipette",
                "--asset-type",
                "video_clip",
                "--limit",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    results = json.loads(output)

    assert len(results) == 1
    assert results[0]["asset_type"] == "video_clip"
    assert results[0]["source_type"] == "micro_clip"
    assert "pipette" in results[0]["objects"]
    assert results[0]["match_reasons"]
