from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from key_action_indexer.unified_timeline import (
    build_event_anchors,
    build_timeline_event,
    build_unified_timeline,
    fit_timeline_calibration,
    read_events_jsonl,
)


def _manifest() -> dict[str, str]:
    return {
        "session_id": "session_a",
        "session_start_time": "2026-04-29T17:25:00+08:00",
    }


def test_reads_absolute_time_event_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "user_1",
                "timestamp": "2026-04-29T17:25:10+08:00",
                "role": "user",
                "message": "start weighing",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    events = build_event_anchors(read_events_jsonl(path), manifest=_manifest())

    assert events[0]["timeline_event_id"] == "user_1"
    assert events[0]["event_type"] == "user_text"
    assert events[0]["global_time"] == "2026-04-29T17:25:10+08:00"
    assert events[0]["session_time_sec"] == 10.0
    assert events[0]["text"] == "start weighing"


def test_maps_session_sec_to_global_time() -> None:
    event = build_timeline_event(
        {"id": "ai_1", "session_sec": 12.5, "role": "assistant", "content": "use the spatula"},
        manifest=_manifest(),
    )

    assert event["event_type"] == "ai_reply"
    assert event["global_time"] == "2026-04-29T17:25:12.500000+08:00"
    assert event["anchor_strategy"] == "session_time"
    assert event["source_type"] == "event_jsonl"
    assert event["source_time_sec"] == 12.5


def test_maps_timestamp_ms_as_absolute_time() -> None:
    timestamp_ms = int(datetime(2026, 4, 29, 9, 25, 3, tzinfo=timezone.utc).timestamp() * 1000)

    event = build_timeline_event({"id": "upload_ms", "timestamp_ms": timestamp_ms}, manifest=_manifest())

    assert event["global_time"] == "2026-04-29T17:25:03+08:00"
    assert event["anchor_strategy"] == "absolute_timestamp_ms"


def test_applies_fixed_latency_correction() -> None:
    calibration = fit_timeline_calibration(manifest=_manifest(), latency_sec=2.0)

    event = build_timeline_event({"id": "late_event", "session_sec": 10}, manifest=_manifest(), calibration=calibration)

    assert calibration.summary()["correction_sec"] == -2.0
    assert event["global_time"] == "2026-04-29T17:25:08+08:00"


def test_fits_two_anchor_drift_calibration() -> None:
    calibration = fit_timeline_calibration(
        [
            {"source_time_sec": 0.0, "global_sec": 0.0},
            {"source_time_sec": 100.0, "global_sec": 110.0},
        ],
        manifest=_manifest(),
    )

    event = build_timeline_event({"id": "drifted", "local_time_sec": 50.0}, manifest=_manifest(), calibration=calibration)

    assert calibration.summary()["method"] == "linear_drift"
    assert calibration.summary()["slope"] == 1.1
    assert event["global_time"] == "2026-04-29T17:25:55+08:00"
    assert event["anchor_strategy"] == "drift_linear"


def test_builds_unified_sorted_timeline() -> None:
    timeline = build_unified_timeline(
        existing_rows={
            "segment": [
                {
                    "segment_id": "segment_2",
                    "session_id": "session_a",
                    "global_start_time": "2026-04-29T17:25:20+08:00",
                    "duration_sec": 5.0,
                }
            ],
            "transcript": [
                {
                    "utterance_id": "utt_1",
                    "global_start_time": "2026-04-29T17:25:05+08:00",
                    "global_end_time": "2026-04-29T17:25:06+08:00",
                    "text": "ready",
                }
            ],
        },
        event_rows=[{"id": "event_1", "session_sec": 10.0, "message": "middle"}],
        manifest=_manifest(),
    )

    assert [row["timeline_event_id"] for row in timeline] == ["utt_1", "event_1", "segment_2"]
    assert [row["global_time"] for row in timeline] == [
        "2026-04-29T17:25:05+08:00",
        "2026-04-29T17:25:10+08:00",
        "2026-04-29T17:25:20+08:00",
    ]


def test_event_field_compatibility_for_upload_rows() -> None:
    event = build_timeline_event(
        {
            "event_id": "upload_1",
            "time": "2026-04-29T17:25:01+08:00",
            "content": [{"type": "text", "text": "uploaded gel image"}],
            "media_path": "images/gel.png",
            "upload_type": "image",
        },
        manifest=_manifest(),
    )

    assert event["timeline_event_id"] == "upload_1"
    assert event["event_type"] == "upload"
    assert event["modality"] == "image"
    assert event["text"] == "uploaded gel image"
    assert event["links"] == [{"rel": "media_path", "path": "images/gel.png"}]
