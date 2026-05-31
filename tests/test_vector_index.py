from __future__ import annotations

from key_action_indexer.schemas import VectorMetadata
from key_action_indexer.vector_index import EmbeddingBackend, VectorIndex


def test_vector_index_query_returns_pipetting() -> None:
    items = [
        VectorMetadata(
            embedding_id="e1",
            segment_id="seg_1",
            session_id="s1",
            index_text="实验人员使用移液枪加样，加入 200 微升样品。",
            global_start_time="t1",
            global_end_time="t2",
            third_person_clip="third1.mp4",
            first_person_clip=None,
            related_dialogue=["使用移液枪加 200 微升。"],
            action_type="pipetting",
        ),
        VectorMetadata(
            embedding_id="e2",
            segment_id="seg_2",
            session_id="s1",
            index_text="实验人员记录天平读数。",
            global_start_time="t3",
            global_end_time="t4",
            third_person_clip="third2.mp4",
            first_person_clip=None,
            related_dialogue=["记录天平读数。"],
            action_type="reading_or_recording",
        ),
    ]
    index = VectorIndex(EmbeddingBackend(kind="hashing")).build([item.index_text for item in items], items)
    results = index.query("移液枪加样", top_k=1)
    assert results[0]["segment_id"] == "seg_1"
