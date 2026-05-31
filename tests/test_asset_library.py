from __future__ import annotations

from pathlib import Path

from key_action_indexer.asset_library import build_material_asset_catalog, load_material_assets, summarize_material_assets
from key_action_indexer.schemas import write_jsonl


def test_timeline_text_upload_without_file_becomes_text_asset(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    write_jsonl(
        metadata / "unified_multimodal_timeline.jsonl",
        [
            {
                "timeline_event_id": "upload_note",
                "session_id": "s1",
                "event_type": "upload",
                "modality": "text",
                "global_time": "2026-04-29T17:25:06+08:00",
                "text": "operator uploaded a note",
            }
        ],
    )

    result = build_material_asset_catalog(session)
    rows = load_material_assets(metadata / "material_asset_catalog.jsonl")

    assert result["asset_count"] == 1
    assert rows[0]["asset_type"] == "text_asset"
    assert rows[0]["quality"]["status"] == "inline_text"
    assert summarize_material_assets(rows)["missing_count"] == 0


def test_poster_named_micro_asset_is_non_real_even_when_file_exists(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    clips = session / "clips"
    metadata.mkdir(parents=True)
    clips.mkdir(parents=True)
    poster_clip = clips / "episode_000001_poster_preview.mp4"
    poster_clip.write_bytes(b"not a real extracted evidence clip")
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "micro_segment_id": "micro_001",
                "parent_segment_id": "episode_000001",
                "global_start_time": "2026-04-29T17:25:00+08:00",
                "global_end_time": "2026-04-29T17:25:04+08:00",
                "third_person": {"clip_path": str(poster_clip)},
                "interaction": {"primary_object": "beaker", "interaction_type": "hand_object_contact"},
                "evidence": {"evidence_level": "weak_visual_evidence"},
            }
        ],
    )

    build_material_asset_catalog(session)
    rows = load_material_assets(metadata / "material_asset_catalog.jsonl")

    assert len(rows) == 1
    assert rows[0]["exists"] is True
    assert rows[0]["placeholder"] is True
    assert rows[0]["source_real"] is False
    assert rows[0]["publishable_material"] is False
    assert rows[0]["missing_reason"] == "present"
