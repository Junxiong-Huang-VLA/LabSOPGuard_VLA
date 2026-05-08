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
