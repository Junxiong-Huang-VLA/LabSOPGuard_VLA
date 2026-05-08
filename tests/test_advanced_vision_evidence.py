from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from key_action_indexer.advanced_vision_evidence import build_advanced_vision_evidence, load_advanced_vision_evidence
from key_action_indexer.schemas import write_jsonl


def _write_model_inventory(metadata: Path, *, liquid_available: bool = False, cap_lid_available: bool = False) -> None:
    (metadata / "model_inventory.json").write_text(
        json.dumps(
            {
                "metadata_version": "key_action_model_inventory.v1",
                "model_count": 1 if (liquid_available or cap_lid_available) else 0,
                "dataset_count": 0,
                "primary_model": {"name": "test_model"} if (liquid_available or cap_lid_available) else None,
                "capabilities": {
                    "liquid_stream_segmentation": {
                        "available": liquid_available,
                        "classes": ["liquid_stream", "meniscus"] if liquid_available else [],
                    },
                    "cap_lid_detection": {
                        "available": cap_lid_available,
                        "classes": ["tube_cap"] if cap_lid_available else [],
                    },
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _save_level_image(path: Path, *, boundary_y: int) -> None:
    image = Image.new("RGB", (120, 80), (224, 224, 228))
    liquid_band = Image.new("RGB", (120, 80 - boundary_y), (40, 90, 190))
    image.paste(liquid_band, (0, boundary_y))
    image.save(path)


def test_advanced_vision_evidence_tracks_motion_and_liquid_candidate(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    cv_outputs = tmp_path / "cv_outputs"
    keyframes = tmp_path / "keyframes" / "micro_001"
    metadata.mkdir()
    cv_outputs.mkdir()
    keyframes.mkdir(parents=True)
    _write_model_inventory(metadata, liquid_available=False, cap_lid_available=True)
    Image.new("RGB", (120, 80), (40, 40, 160)).save(keyframes / "contact.jpg")
    Image.new("RGB", (120, 80), (180, 180, 220)).save(keyframes / "release.jpg")
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 3.0,
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "interaction": {
                    "interaction_type": "hand_pipette_contact",
                    "primary_object": "pipette",
                    "detected_objects": ["hand", "pipette", "tube"],
                },
                "keyframes": {
                    "contact_frame": "keyframes/micro_001/contact.jpg",
                    "release_frame": "keyframes/micro_001/release.jpg",
                },
                "text_description": {"action_type": "pipetting"},
            },
            {
                "session_id": "s1",
                "parent_segment_id": "seg_002",
                "micro_segment_id": "micro_002",
                "start_sec": 5.0,
                "end_sec": 6.0,
                "global_start_time": "2026-04-29T17:25:05+08:00",
                "global_end_time": "2026-04-29T17:25:06+08:00",
                "interaction": {
                    "interaction_type": "hand_container_contact",
                    "primary_object": "sample_bottle",
                    "detected_objects": ["gloved_hand", "sample_bottle", "tube-cap"],
                },
                "keyframes": {},
                "text_description": {"action_type": "sample_handling"},
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            {"local_time_sec": 1.0, "detections": [{"label": "pipette", "bbox": [10, 10, 30, 30], "confidence": 0.9}]},
            {"local_time_sec": 2.0, "detections": [{"label": "pipette", "bbox": [35, 15, 55, 35], "confidence": 0.9}]},
            {"local_time_sec": 3.0, "detections": [{"label": "pipette", "bbox": [70, 20, 90, 40], "confidence": 0.9}]},
        ],
    )
    write_jsonl(
        metadata / "material_asset_catalog.jsonl",
        [
            {
                "asset_id": "asset_contact",
                "source_id": "micro_001",
                "asset_type": "keyframe",
                "path": "keyframes/micro_001/contact.jpg",
                "quality": {"status": "present"},
            }
        ],
    )

    summary = build_advanced_vision_evidence(tmp_path)
    rows = load_advanced_vision_evidence(metadata / "advanced_vision_evidence.jsonl")
    types = {row["evidence_type"] for row in rows}
    hand_contact = next(row for row in rows if row["evidence_type"] == "hand_object_contact" and row["micro_segment_id"] == "micro_001")
    movement = next(row for row in rows if row["evidence_type"] == "object_trajectory_movement")
    liquid = next(row for row in rows if row["evidence_type"] == "liquid_flow_candidate_visual")
    container = next(row for row in rows if row["evidence_type"] == "container_open_close" and row["micro_segment_id"] == "micro_002")

    assert summary["evidence_count"] == len(rows)
    assert {"hand_object_contact", "object_trajectory_movement", "liquid_flow_candidate_visual"}.issubset(types)
    assert summary["real_model_inventory"]["model_count"] >= 0
    assert hand_contact["confirmation_level"] in {"candidate", "confirmed"}
    assert hand_contact["confidence_reasons"]
    assert hand_contact["multiview_consistency"]["status"] in {"single_view_only", "not_available"}
    assert hand_contact["evidence_refs"]
    assert movement["visual_confirmation_level"] == "trajectory_confirmed"
    assert movement["confirmation_level"] == "measured"
    assert movement["confidence_reasons"] == movement["evidence_reasons"]
    assert movement["confidence"] > 0.7
    assert movement["metrics"]["track_id"] == "inferred_0001"
    assert movement["metrics"]["track_source"] == "bbox_temporal_association"
    assert movement["metrics"]["identity_confidence"] >= 0.65
    assert movement["metrics"]["normalized_displacement"] > 1.0
    assert liquid["requires_human_confirmation"] is True
    assert liquid["confirmation_level"] == "candidate"
    assert liquid["visual_confirmation_level"] == "liquid_flow_candidate"
    assert liquid["metrics"]["model_capability"]["available"] is False
    assert liquid["metrics"]["keyframe_visual_indicators"][0]["color_profile"]["dominant_color_family"] == "blue_or_purple"
    assert "does not confirm visible liquid stream without a trained fluid/level model" in liquid["limitations"]
    assert container["visual_confirmation_level"] == "container_open_close_confirmed"
    assert container["metrics"]["model_capability"]["available"] is True
    assert json.loads((metadata / "advanced_vision_evidence_summary.json").read_text(encoding="utf-8"))["evidence_type_counts"]["object_trajectory_movement"] == 1


def test_advanced_vision_evidence_confirms_liquid_level_and_container_with_inventory(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    keyframes = tmp_path / "keyframes" / "micro_001"
    metadata.mkdir()
    keyframes.mkdir(parents=True)
    _write_model_inventory(metadata, liquid_available=True, cap_lid_available=True)
    _save_level_image(keyframes / "contact.jpg", boundary_y=18)
    _save_level_image(keyframes / "release.jpg", boundary_y=58)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 3.0,
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "interaction": {
                    "interaction_type": "hand_tube_contact",
                    "primary_object": "tube",
                    "detected_objects": ["hand", "tube", "tube_cap"],
                },
                "keyframes": {
                    "contact_frame": "keyframes/micro_001/contact.jpg",
                    "release_frame": "keyframes/micro_001/release.jpg",
                },
                "text_description": {"action_type": "pipetting"},
            }
        ],
    )

    build_advanced_vision_evidence(tmp_path)
    rows = load_advanced_vision_evidence(metadata / "advanced_vision_evidence.jsonl")
    liquid_flow = next(row for row in rows if row["evidence_type"] == "liquid_flow_candidate_visual")
    liquid_level = next(row for row in rows if row["evidence_type"] == "liquid_level_change")
    container = next(row for row in rows if row["evidence_type"] == "container_open_close")

    assert liquid_flow["visual_confirmation_level"] == "liquid_flow_confirmed"
    assert liquid_flow["requires_human_confirmation"] is False
    assert liquid_flow["metrics"]["model_capability"]["classes"] == ["liquid_stream", "meniscus"]
    assert liquid_level["visual_confirmation_level"] == "liquid_level_change_confirmed"
    assert liquid_level["metrics"]["liquid_level_delta"] >= 0.4
    assert liquid_level["metrics"]["level_indicators"][0]["status"] == "detected"
    assert liquid_level["metrics"]["image_metrics"][0]["frame_quality"]["status"] == "usable"
    assert container["visual_confirmation_level"] == "container_open_close_confirmed"
    assert container["metrics"]["container_state_indicators"]["cap_lid_detection_available"] is True


def test_advanced_vision_evidence_flags_same_class_identity_risk(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    cv_outputs = tmp_path / "cv_outputs"
    metadata.mkdir()
    cv_outputs.mkdir()
    _write_model_inventory(metadata)
    write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "session_id": "s1",
                "parent_segment_id": "seg_001",
                "micro_segment_id": "micro_001",
                "start_sec": 1.0,
                "end_sec": 3.0,
                "global_start_time": "2026-04-29T17:25:01+08:00",
                "global_end_time": "2026-04-29T17:25:03+08:00",
                "interaction": {
                    "interaction_type": "hand_tube_contact",
                    "primary_object": "tube",
                    "detected_objects": ["hand", "tube"],
                },
                "keyframes": {},
                "text_description": {"action_type": "sample_handling"},
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            {
                "local_time_sec": 1.0,
                "detections": [
                    {"label": "tube", "bbox": [10, 10, 30, 30], "confidence": 0.9},
                    {"label": "tube", "bbox": [100, 10, 120, 30], "confidence": 0.9},
                ],
            },
            {
                "local_time_sec": 2.0,
                "detections": [
                    {"label": "tube", "bbox": [40, 10, 60, 30], "confidence": 0.9},
                    {"label": "tube", "bbox": [100, 40, 120, 60], "confidence": 0.9},
                ],
            },
            {
                "local_time_sec": 3.0,
                "detections": [
                    {"label": "tube", "bbox": [70, 10, 90, 30], "confidence": 0.9},
                    {"label": "tube", "bbox": [100, 70, 120, 90], "confidence": 0.9},
                ],
            },
        ],
    )

    build_advanced_vision_evidence(tmp_path)
    rows = load_advanced_vision_evidence(metadata / "advanced_vision_evidence.jsonl")
    movement = next(row for row in rows if row["evidence_type"] == "object_trajectory_movement")

    assert movement["visual_confirmation_level"] == "trajectory_candidate_identity_risk"
    assert movement["requires_human_confirmation"] is True
    assert movement["metrics"]["track_source"] == "bbox_temporal_association"
    assert movement["metrics"]["identity_confidence"] < 0.65
    assert movement["metrics"]["same_class_track_count"] == 2
    assert movement["metrics"]["max_same_class_per_frame"] == 2
    assert "same-class multi-target identity risk" in movement["limitations"][1]
