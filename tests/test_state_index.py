from __future__ import annotations

from pathlib import Path

from key_action_indexer.schemas import write_jsonl
from key_action_indexer.state_index import build_state_change_index, load_state_changes


def _metadata_dir(session_dir: Path) -> Path:
    metadata = session_dir / "metadata"
    metadata.mkdir(parents=True)
    return metadata


def test_builds_three_phase_micro_events_and_dialogue_context(tmp_path: Path) -> None:
    metadata = _metadata_dir(tmp_path)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_001_micro_001",
                "parent_segment_id": "seg_001",
                "session_id": "session_a",
                "start_sec": 10.0,
                "end_sec": 16.0,
                "global_start_time": "2026-04-29T17:25:10+08:00",
                "global_end_time": "2026-04-29T17:25:16+08:00",
                "interaction": {
                    "interaction_type": "hand_bottle_contact",
                    "primary_object": "bottle",
                    "contact_start_sec": 11.0,
                    "peak_interaction_sec": 13.0,
                    "contact_end_sec": 15.0,
                    "max_interaction_score": 0.42,
                },
                "keyframes": {
                    "contact_frame": "contact.jpg",
                    "peak_frame": "peak.jpg",
                    "release_frame": "release.jpg",
                },
                "third_person": {"clip_path": "third_micro.mp4"},
                "text_description": {
                    "action_type": "sample_handling",
                    "summary": "hand contacts bottle",
                    "index_text": "bottle contact",
                },
                "dialogue_context": [{"utterance_id": "utt_1", "text": "operator mentions the bottle"}],
                "evidence": {
                    "evidence_level": "weak_visual_evidence",
                    "limitations": ["low or sparse visual interaction evidence"],
                },
            }
        ],
    )
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_001", "session_id": "session_a"}])
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [])

    summary = build_state_change_index(tmp_path)
    rows = load_state_changes(metadata / "state_change_index.jsonl")

    assert summary["state_change_count"] == 4
    assert summary["state_type_counts"]["contact_started"] == 1
    phase_rows = {row["state_type"]: row for row in rows if row["state_type"] != "dialogue_context_available"}
    assert set(phase_rows) == {"contact_started", "peak_interaction", "contact_released"}
    assert phase_rows["contact_started"]["global_time"] == "2026-04-29T17:25:11+08:00"
    assert phase_rows["peak_interaction"]["session_time_sec"] == 13.0
    assert phase_rows["contact_released"]["asset_refs"][0]["path"] == "release.jpg"
    assert "limitation:low or sparse visual interaction evidence" in phase_rows["contact_started"]["state_tags"]

    dialogue = next(row for row in rows if row["state_type"] == "dialogue_context_available")
    assert dialogue["text"] == "operator mentions the bottle"
    assert dialogue["primary_object"] == "bottle"


def test_adds_timeline_yolo_interaction_object_contact_and_sorts(tmp_path: Path) -> None:
    metadata = _metadata_dir(tmp_path)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_001_micro_001",
                "parent_segment_id": "seg_001",
                "session_id": "session_a",
                "start_sec": 10.0,
                "end_sec": 14.0,
                "global_start_time": "2026-04-29T17:25:10+08:00",
                "global_end_time": "2026-04-29T17:25:14+08:00",
                "interaction": {
                    "interaction_type": "hand_spatula_contact",
                    "primary_object": "spatula",
                    "contact_start_sec": 10.0,
                    "peak_interaction_sec": 12.0,
                    "contact_end_sec": 14.0,
                },
                "text_description": {"action_type": "use_spatula"},
                "evidence": {"evidence_level": "visual_confirmed", "limitations": []},
            }
        ],
    )
    write_jsonl(metadata / "key_action_segments.jsonl", [{"segment_id": "seg_001", "session_id": "session_a"}])
    write_jsonl(
        metadata / "unified_multimodal_timeline.jsonl",
        [
            {
                "timeline_event_id": "yolo_001",
                "session_id": "session_a",
                "event_type": "yolo_interaction",
                "source": "yolo_interaction",
                "global_time": "2026-04-29T17:25:11+08:00",
                "session_time_sec": 11.0,
                "text": "hand near spatula",
                "links": [{"rel": "path", "path": "interaction.jpg"}],
                "payload": {
                    "event_id": "raw_yolo_001",
                    "segment_id": "seg_001",
                    "object_label": "spatula",
                    "interaction": "hand_spatula_contact",
                    "confidence": 0.9,
                },
            }
        ],
    )

    build_state_change_index(tmp_path)
    rows = load_state_changes(metadata / "state_change_index.jsonl")

    assert [row["state_type"] for row in rows] == [
        "contact_started",
        "object_contact",
        "peak_interaction",
        "contact_released",
    ]
    yolo = rows[1]
    assert yolo["state_change_id"] == "session_a:yolo_001:object_contact"
    assert yolo["primary_object"] == "spatula"
    assert yolo["evidence_level"] == "visual_confirmed"
    assert yolo["asset_refs"] == [{"asset_type": "timeline_link", "rel": "path", "path": "interaction.jpg"}]


def test_missing_time_fields_do_not_interrupt_and_keep_stable_fields(tmp_path: Path) -> None:
    metadata = _metadata_dir(tmp_path)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "micro_missing_time",
                "parent_segment_id": "seg_missing",
                "session_id": "session_a",
                "interaction": {
                    "interaction_type": "hand_tube_contact",
                    "primary_object": "tube",
                },
                "text_description": {"action_type": "sample_adding"},
                "evidence": {
                    "evidence_level": "transcript_supported",
                    "limitations": ["missing pipette or tube visual evidence"],
                },
            }
        ],
    )
    write_jsonl(metadata / "key_action_segments.jsonl", [])
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [])

    summary = build_state_change_index(tmp_path)
    rows = load_state_changes(metadata / "state_change_index.jsonl")

    assert summary["state_change_count"] == 3
    assert [row["state_type"] for row in rows] == ["contact_started", "peak_interaction", "contact_released"]
    assert all(row["global_time"] is None for row in rows)
    assert all(row["session_time_sec"] is None for row in rows)
    expected_fields = {
        "state_change_id",
        "session_id",
        "state_type",
        "global_time",
        "session_time_sec",
        "segment_id",
        "micro_segment_id",
        "primary_object",
        "interaction_type",
        "action_type",
        "evidence_level",
        "state_tags",
        "asset_refs",
        "text",
        "payload",
    }
    assert set(rows[0]) == expected_fields
    assert "missing_global_time" in rows[0]["state_tags"]
    assert "limitation:missing pipette or tube visual evidence" in rows[0]["state_tags"]


def test_enriches_asset_refs_with_asset_ids_when_catalog_exists(tmp_path: Path) -> None:
    metadata = _metadata_dir(tmp_path)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "micro_asset_linked",
                "parent_segment_id": "seg_asset",
                "session_id": "session_a",
                "start_sec": 10.0,
                "end_sec": 12.0,
                "global_start_time": "2026-04-29T17:25:10+08:00",
                "global_end_time": "2026-04-29T17:25:12+08:00",
                "interaction": {
                    "interaction_type": "hand_vial_contact",
                    "primary_object": "vial",
                    "contact_start_sec": 10.0,
                    "peak_interaction_sec": 11.0,
                    "contact_end_sec": 12.0,
                },
                "keyframes": {"contact_frame": "keyframes/micro_asset_linked/contact.jpg"},
                "third_person": {"clip_path": "clips/micro_asset_linked.mp4"},
                "text_description": {"action_type": "sample_handling"},
                "evidence": {"evidence_level": "visual_confirmed"},
            }
        ],
    )
    write_jsonl(metadata / "key_action_segments.jsonl", [])
    write_jsonl(metadata / "unified_multimodal_timeline.jsonl", [])
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [
            {
                "asset_id": "asset_contact_001",
                "asset_type": "keyframe",
                "path": "keyframes/micro_asset_linked/contact.jpg",
                "source_type": "micro_keyframe",
            },
            {
                "asset_id": "asset_clip_001",
                "asset_type": "video_clip",
                "path": "clips/micro_asset_linked.mp4",
                "source_type": "micro_clip",
            },
        ],
    )

    summary = build_state_change_index(tmp_path)
    rows = load_state_changes(metadata / "state_change_index.jsonl")

    contact = rows[0]
    assert summary["asset_id_ref_count"] >= 2
    assert contact["asset_refs"][0]["asset_id"] == "asset_contact_001"
    assert {ref["asset_id"] for ref in contact["asset_refs"] if ref.get("asset_id")} == {
        "asset_contact_001",
        "asset_clip_001",
    }
