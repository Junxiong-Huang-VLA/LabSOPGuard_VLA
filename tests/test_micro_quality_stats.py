from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.evaluation import compute_micro_quality_stats


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_compute_micro_quality_stats_writes_summary(tmp_path: Path) -> None:
    source = tmp_path / "metadata" / "micro_segments.jsonl"
    output = tmp_path / "evaluation" / "micro_quality_stats.json"
    _write_jsonl(
        source,
        [
            {
                "micro_segment_id": "m1",
                "start_sec": 1.0,
                "end_sec": 3.0,
                "duration_sec": 2.0,
                "interaction": {"primary_object": "pipette"},
                "quality": {"confidence": "high", "warnings": []},
                "keyframes": {"peak_frame": "peak.jpg"},
                "dialogue_context_available": True,
            },
            {
                "micro_segment_id": "m2",
                "start_sec": 4.0,
                "end_sec": 4.4,
                "interaction": {"primary_object": "tube"},
                "quality": {"confidence": "low", "warnings": ["very_short_micro_segment", "missing_keyframe"]},
                "keyframes": {},
                "manual_corrected": True,
            },
        ],
    )

    stats = compute_micro_quality_stats(source, output)

    assert stats["micro_segment_count"] == 2
    assert stats["confidence_counts"] == {"high": 1, "low": 1}
    assert stats["warning_counts"]["very_short_micro_segment"] == 1
    assert stats["primary_object_counts"]["pipette"] == 1
    assert stats["missing_peak_keyframe_count"] == 1
    assert stats["missing_any_keyframe_count"] == 1
    assert stats["dialogue_context_available_count"] == 1
    assert stats["manual_corrected_count"] == 1
    assert output.exists()
