from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.family_merge import analyze_object_family_merge
from key_action_indexer.schemas import write_jsonl


def _row(micro_id: str, start: float, end: float, obj: str) -> dict:
    return {
        "micro_segment_id": micro_id,
        "parent_segment_id": "seg_000001",
        "start_sec": start,
        "end_sec": end,
        "interaction": {"primary_object": obj},
        "evidence_level": "visual_confirmed",
    }


def test_family_merge_analysis_finds_bottle_family_candidate_without_modifying_micro_segments(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    source = metadata / "micro_segments.jsonl"
    rows = [
        _row("m1", 1.0, 2.0, "bottle"),
        _row("m2", 2.4, 3.0, "sample_bottle"),
        _row("m3", 4.0, 5.0, "balance"),
        _row("m4", 5.2, 6.0, "spatula"),
    ]
    write_jsonl(source, rows)

    result = analyze_object_family_merge(session)
    after_rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert result["adjacent_same_family_micro_count"] == 1
    assert result["candidate_family_merge_pairs"][0]["family"] == "bottle_family"
    assert result["estimated_micro_count_after_family_merge"] == 3
    assert after_rows == rows
