from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.chinese_index import refresh_micro_row_chinese_index
from key_action_indexer.material_search import search_material_index
from key_action_indexer.vector_index import EmbeddingBackend, VectorIndex


def test_micro_chinese_index_contains_summary_objects_and_evidence() -> None:
    row = refresh_micro_row_chinese_index(
        {
            "micro_segment_id": "micro_001",
            "parent_segment_id": "seg_001",
            "session_id": "s1",
            "global_start_time": "2026-05-03T09:00:10+08:00",
            "global_end_time": "2026-05-03T09:00:12+08:00",
            "interaction": {
                "primary_object": "pipette",
                "interaction_type": "hand_pipette_contact",
                "detected_objects": ["hand", "pipette", "tube"],
                "max_interaction_score": 0.91,
            },
            "text_description": {
                "action_type": "pipetting",
                "summary": "",
                "index_text": "legacy compatible text pipette",
            },
            "evidence": {
                "evidence_level": "visual_confirmed",
                "evidence_reasons": ["hand-object contact confirmed"],
                "limitations": ["no liquid mask"],
            },
        }
    )

    assert "手与移液枪" in row["text_description"]["summary"]
    assert "中文检索索引" in row["text_description"]["index_text"]
    assert "移液/加样" in row["text_description"]["index_text"]
    assert "no liquid mask" in row["text_description"]["index_text"]


def test_search_material_index_filters_assets_and_vector_hits(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    index_dir = session / "index"
    metadata.mkdir(parents=True)
    rows = [
        {
            "asset_id": "asset_keyframe_001",
            "session_id": "s1",
            "asset_type": "keyframe",
            "path": "keyframes/micro_001/peak.jpg",
            "source_type": "micro_keyframe",
            "source_id": "micro_001",
            "segment_id": "seg_001",
            "micro_segment_id": "micro_001",
            "global_start_time": "2026-05-03T09:00:10+08:00",
            "global_end_time": "2026-05-03T09:00:12+08:00",
            "objects": ["pipette", "tube"],
            "actions": ["pipetting"],
            "state_tags": ["micro_keyframe", "liquid_transfer_candidate"],
            "search_text": "移液枪 加样 pipette liquid transfer",
            "evidence_level": "candidate",
            "quality": {"status": "missing"},
        }
    ]
    (metadata / "material_asset_catalog.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    vector_metadata = [
        {
            "embedding_id": "emb_micro_001",
            "segment_id": "seg_001",
            "micro_segment_id": "micro_001",
            "session_id": "s1",
            "index_level": "micro_segment",
            "index_text": "中文摘要: 手与移液枪接触，执行移液/加样 pipette liquid transfer",
            "global_start_time": "2026-05-03T09:00:10+08:00",
            "global_end_time": "2026-05-03T09:00:12+08:00",
            "third_person_clip": "clips/micro_001.mp4",
            "first_person_clip": None,
            "related_dialogue": [],
            "action_type": "pipetting",
            "interaction_type": "hand_pipette_contact",
            "primary_object": "pipette",
            "detected_objects": ["hand", "pipette", "tube"],
            "keyframes": ["keyframes/micro_001/peak.jpg"],
        }
    ]
    VectorIndex(EmbeddingBackend(kind="hashing")).build(
        [vector_metadata[0]["index_text"]],
        vector_metadata,
    ).save(index_dir)

    result = search_material_index(
        session,
        query="加样",
        asset_type="keyframe",
        objects="pipette",
        actions="pipetting",
        start_time="2026-05-03T09:00:09+08:00",
        end_time="2026-05-03T09:00:13+08:00",
        index_level="micro_segment",
    )

    assert result["asset_count"] == 1
    assert result["assets"][0]["asset_id"] == "asset_keyframe_001"
    assert result["segment_count"] == 1
    assert result["segments"][0]["micro_segment_id"] == "micro_001"
