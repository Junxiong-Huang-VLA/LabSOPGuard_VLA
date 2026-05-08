from __future__ import annotations

from key_action_indexer.description_builder import build_segment_description
from key_action_indexer.schemas import (
    CVDetectionSummary,
    ClipReference,
    KeyActionSegment,
    SegmentIndexInfo,
    TextDescription,
    TranscriptUtterance,
)


def test_description_builder_infers_pipetting() -> None:
    segment = KeyActionSegment(
        session_id="s1",
        segment_id="seg_000001",
        global_start_time="2026-04-29T17:35:00+08:00",
        global_end_time="2026-04-29T17:35:20+08:00",
        duration_sec=20,
        third_person=ClipReference("third.mp4", "clips/seg_000001/third_person.mp4", 600, 620),
        first_person=ClipReference("first.mp4", "clips/seg_000001/first_person.mp4", 598, 618),
        cv_detection=CVDetectionSummary(0.7, 0.8, "start", "end"),
        text_description=TextDescription(),
        dialogue_context=[],
        index=SegmentIndexInfo("", "", ""),
    )
    dialogue = [
        TranscriptUtterance(
            "utt_1",
            620,
            628,
            "接下来使用移液枪加 200 微升。",
            "2026-04-29T17:35:20+08:00",
            "2026-04-29T17:35:28+08:00",
        )
    ]
    built = build_segment_description(segment, dialogue)
    assert built.text_description.action_type == "pipetting"
    assert "移液枪" in built.index.index_text
    assert "clips/seg_000001/third_person.mp4" in built.index.index_text
