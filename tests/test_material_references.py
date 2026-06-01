from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from key_action_indexer import material_references
from key_action_indexer.material_references import (
    MATERIAL_CANDIDATE_INDEX_BASENAME,
    KEYFRAME_DIR_NAME,
    KEY_CLIP_DIR_NAME,
    MATERIAL_INDEX_BASENAME,
    approve_material_candidates,
    build_yolo_material_candidates,
    build_yolo_material_references,
    frontend_material_references_root,
    formal_material_references_root,
    material_candidates_root,
    material_references_root,
    reset_material_references_to_approved_candidates,
)


@pytest.fixture(autouse=True)
def _isolate_material_reference_env(monkeypatch):
    monkeypatch.delenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", raising=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_material_file_is_real_reads_header_without_loading_whole_file(tmp_path: Path, monkeypatch) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 1024)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(MemoryError("full file read")),
    )
    monkeypatch.setattr(material_references, "_material_visual_content_is_real", lambda _path: True)

    assert material_references._material_file_is_real(clip) is True


def _valid_evidence(view: str, *, primary_object: str = "balance", score: float = 0.78) -> list[dict]:
    times = [46.433667, 47.2, 49.133666]
    evidence: list[dict] = []
    for index, local_time in enumerate(times):
        shift = index * 3
        hand_bbox = [110 + shift, 120, 190 + shift, 230]
        object_bbox = [165 + shift, 155, 335 + shift, 315]
        evidence.append(
            {
                "view": view,
                "local_time_sec": local_time,
                "interaction_score": score,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.74, "bbox": hand_bbox},
                    {"label": primary_object, "confidence": 0.72, "bbox": object_bbox},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": primary_object,
                        "score": score,
                        "hand_bbox": hand_bbox,
                        "object_bbox": object_bbox,
                        "iou": 0.08,
                        "distance_px": 24.0,
                    }
                ],
            }
        )
    return evidence


def _paper_balance_evidence(view: str) -> list[dict]:
    rows = _valid_evidence(view, primary_object="paper", score=0.78)
    for row in rows:
        balance_bbox = [360, 180, 520, 340]
        hand_bbox = row["hand_object_interactions"][0]["hand_bbox"]
        row["detections"].append({"label": "balance", "confidence": 0.81, "bbox": balance_bbox})
        row["hand_object_interactions"].append(
            {
                "hand_label": "gloved_hand",
                "object_label": "balance",
                "score": 0.66,
                "hand_bbox": hand_bbox,
                "object_bbox": balance_bbox,
                "iou": 0.04,
            }
        )
    return rows


def _false_balance_evidence() -> list[dict]:
    return [
        {
            "view": "first_person",
            "local_time_sec": 61.866667 + index * 0.2,
            "interaction_score": 0.70,
            "detections": [
                {"label": "gloved_hand", "confidence": 0.62, "bbox": [366, 73, 542, 140]},
                {"label": "balance", "confidence": 0.31, "bbox": [342, 82, 562, 369]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "balance",
                    "score": 0.70,
                    "hand_bbox": [366, 73, 542, 140],
                    "object_bbox": [342, 82, 562, 369],
                }
            ],
        }
        for index in range(3)
    ]


def _evidence_views(rows: list[dict]) -> set[str]:
    return {
        str(row.get("view") or row.get("source_view") or row.get("requested_view") or "").strip()
        for row in rows
        if isinstance(row, dict)
    }


def _write_formal_dual_event(metadata: Path, *, micro_segment_id: str = "seg_000001_micro_004") -> None:
    _write_jsonl(
        metadata / "dual_view_action_events.jsonl",
        [
            {
                "dual_event_id": "dual_event_000123",
                "status": "matched_dual_view",
                "formal_event_promoted": True,
                "micro_segment_ids": [micro_segment_id],
                "first_evidence_id": "first_evidence_001",
                "third_evidence_id": "third_evidence_001",
                "views": {
                    "first_person": {"evidence_id": "first_evidence_001", "view": "first_person", "frame_count": 2},
                    "third_person": {"evidence_id": "third_evidence_001", "view": "third_person", "frame_count": 2},
                },
            }
        ],
    )


def test_material_taxonomy_collapses_business_action_names_but_keeps_labels() -> None:
    cases = {
        "reagent_bottle_open": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
        "bottle_cap": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
        "sample_bottle": "\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c",
        "paper": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
        "weighing_paper": "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c",
        "balance": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
        "panel": "\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c",
    }
    for label, expected in cases.items():
        row = {"primary_object": label, "raw_primary_object": label, "manipulated_object": label}

        assert material_references._approved_material_chinese_action_name(row) == expected
        assert row["raw_primary_object"] == label

    taxonomy = material_references._canonical_action_fields("balance")
    assert taxonomy["canonical_action_type"] == "equipment_panel_operation"
    assert taxonomy["canonical_object"] == "panel"
    assert material_references.CHINESE_OBJECT_NAMES["magnetic_stir_bar"] == "\u78c1\u529b\u6405\u62cc\u5b50"


def test_publish_filter_rejects_keyclip_pts_beyond_source_duration(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    stored = tmp_path / "refs" / KEY_CLIP_DIR_NAME / "clip.mp4"
    source.write_bytes(b"source")
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(b"stored")
    monkeypatch.setattr(material_references, "_material_file_is_real", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(material_references, "_source_video_duration_sec", lambda _path: 10.0)

    rows, suppressed = material_references._filter_publishable_material_rows(
        [
            {
                "asset_kind": KEY_CLIP_DIR_NAME,
                "stored_file": str(stored),
                "source_clip_path": str(source),
                "source_offset_sec": 9.8,
                "source_duration_sec": 1.0,
                "micro_segment_id": "micro-pts",
                "view": "third_person",
            }
        ],
        tmp_path / "refs",
    )

    assert rows == []
    assert suppressed[0]["suppression_reason"] == "source_pts_out_of_range"
    assert suppressed[0]["source_pts_gate"]["reason"] == "source_pts_exceeds_source_duration"


def test_publish_filter_accepts_keyframe_pts_inside_source_duration(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    stored = tmp_path / "refs" / KEYFRAME_DIR_NAME / "frame.jpg"
    source.write_bytes(b"source")
    stored.parent.mkdir(parents=True, exist_ok=True)
    stored.write_bytes(b"stored")
    monkeypatch.setattr(material_references, "_material_file_is_real", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(material_references, "_source_video_duration_sec", lambda _path: 10.0)

    rows, suppressed = material_references._filter_publishable_material_rows(
        [
            {
                "asset_kind": KEYFRAME_DIR_NAME,
                "stored_file": str(stored),
                "source_clip_path": str(source),
                "source_offset_sec": 9.5,
                "micro_segment_id": "micro-pts",
                "view": "first_person",
            }
        ],
        tmp_path / "refs",
    )

    assert suppressed == []
    assert rows[0]["source_pts_gate"]["status"] == "passed"


def test_publish_filter_blocks_formal_materials_for_unreliable_time_axis(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "time_axis_health.json").write_text(
        json.dumps(
            {
                "status": "time_axis_unreliable",
                "time_axis_unreliable": True,
                "can_publish_formal_materials": False,
            }
        ),
        encoding="utf-8",
    )
    stored = session / "material_references" / KEYFRAME_DIR_NAME / "frame.jpg"
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"stored")
    monkeypatch.setattr(material_references, "_material_file_is_real", lambda *_args, **_kwargs: True)

    rows, suppressed = material_references._filter_publishable_material_rows(
        [
            {
                "asset_kind": KEYFRAME_DIR_NAME,
                "stored_file": str(stored),
                "micro_segment_id": "micro-pts",
                "view": "third_person",
            }
        ],
        session / "material_references",
        session_root=session,
    )

    assert rows == []
    assert suppressed[0]["reason"] == "formal_material_publish_gate"
    assert suppressed[0]["suppression_reason"] == "time_axis_unreliable"


def test_filtered_interaction_boxes_prefer_evidence_interaction_pairs() -> None:
    evidence_row = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.99, "bbox": [300, 10, 380, 90]},
            {"label": "paper", "confidence": 0.98, "bbox": [410, 20, 500, 100]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.91,
                "hand_bbox": [20, 30, 80, 120],
                "object_bbox": [90, 40, 170, 130],
            }
        ],
    }

    detections = material_references._filtered_interaction_detections(evidence_row, "paper")

    boxes = {tuple(item[0]["bbox"]) for item in detections}
    assert (90.0, 40.0, 170.0, 130.0) in boxes
    assert (20.0, 30.0, 80.0, 120.0) in boxes
    assert (410, 20, 500, 100) not in boxes


def test_interaction_target_labels_split_composite_labels() -> None:
    assert material_references._interaction_target_labels("paper,balance") == {"paper", "balance"}


def test_scale_evidence_row_to_render_frame_scales_detections_and_interactions() -> None:
    evidence_row = {
        "frame_width": 1920,
        "frame_height": 1080,
        "detections": [
            {"label": "gloved_hand", "bbox": [480, 270, 960, 540]},
            {"label": "paper", "bbox": [960, 540, 1440, 810]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "hand_bbox": [480, 270, 960, 540],
                "object_bbox": [960, 540, 1440, 810],
            }
        ],
    }

    scaled = material_references._scale_evidence_row_to_frame_size(evidence_row, width=960, height=540)

    assert scaled["frame_width"] == 960
    assert scaled["frame_height"] == 540
    assert scaled["detections"][0]["bbox"] == [240.0, 135.0, 480.0, 270.0]
    assert scaled["detections"][1]["bbox"] == [480.0, 270.0, 720.0, 405.0]
    assert scaled["hand_object_interactions"][0]["hand_bbox"] == [240.0, 135.0, 480.0, 270.0]
    assert scaled["hand_object_interactions"][0]["object_bbox"] == [480.0, 270.0, 720.0, 405.0]


def test_scale_evidence_row_infers_missing_source_size_from_large_bboxes() -> None:
    evidence_row = {
        "detections": [
            {"label": "gloved_hand", "bbox": [480, 270, 960, 540]},
            {"label": "paper", "bbox": [960, 540, 1440, 810]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "hand_bbox": [480, 270, 960, 540],
                "object_bbox": [960, 540, 1440, 810],
            }
        ],
    }

    scaled = material_references._scale_evidence_row_to_frame_size(evidence_row, width=960, height=540)

    assert scaled["bbox_source_frame"]["frame_width"] == 1920
    assert scaled["bbox_source_frame"]["frame_height"] == 1080
    assert scaled["detections"][0]["bbox"] == [240.0, 135.0, 480.0, 270.0]
    assert scaled["hand_object_interactions"][0]["object_bbox"] == [480.0, 270.0, 720.0, 405.0]


def test_filtered_interaction_detections_refines_paper_to_visible_sheet() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (132, 130, 126)
    frame[54:124, 52:104] = (210, 96, 42)
    frame[58:120:8, 56:100] = (235, 118, 54)
    frame[76:116, 98:152] = (246, 248, 250)
    frame[122:190, 170:250] = (238, 242, 244)
    evidence_row = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.92, "bbox": [40, 40, 110, 130]},
            {"label": "paper", "confidence": 0.88, "bbox": [90, 60, 250, 180]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.94,
                "hand_bbox": [40, 40, 110, 130],
                "object_bbox": [90, 60, 250, 180],
            }
        ],
    }

    filtered = material_references._filtered_interaction_detections(evidence_row, "paper", frame=frame)

    paper = next(item for item, _color in filtered if item["label"] == "paper")
    assert paper["bbox"][0] < 102
    assert paper["bbox"][1] < 80
    assert paper["bbox"][2] < 158
    assert paper["bbox"][3] < 122
    assert paper["raw_yolo_bbox"] == [90.0, 60.0, 250.0, 180.0]


def test_filtered_interaction_detections_rejects_unrefinable_paper_background() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (132, 130, 126)
    frame[54:124, 52:104] = (210, 96, 42)
    frame[58:120:8, 56:100] = (235, 118, 54)
    frame[122:190, 170:250] = (238, 242, 244)
    evidence_row = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.92, "bbox": [40, 40, 110, 130]},
            {"label": "paper", "confidence": 0.88, "bbox": [90, 60, 250, 180]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.94,
                "hand_bbox": [40, 40, 110, 130],
                "object_bbox": [90, 60, 250, 180],
            }
        ],
    }

    assert material_references._filtered_interaction_detections(evidence_row, "paper", frame=frame) == []


def test_filtered_interaction_detections_rejects_neutral_balance_as_hand() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (132, 130, 126)
    frame[40:130, 40:110] = (238, 242, 244)
    frame[76:116, 98:152] = (246, 248, 250)
    evidence_row = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.94, "bbox": [40, 40, 110, 130]},
            {"label": "paper", "confidence": 0.88, "bbox": [90, 60, 160, 130]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.94,
                "hand_bbox": [40, 40, 110, 130],
                "object_bbox": [90, 60, 160, 130],
            }
        ],
    }

    assert material_references._filtered_interaction_detections(evidence_row, "paper", frame=frame) == []


def test_candidate_rebuild_preserves_manual_correction_fields() -> None:
    candidate = {
        "candidate_id": "material_candidate_001",
        "action_name": "手与试剂瓶操作",
        "primary_object": "reagent_bottle",
        "canonical_action_type": "hand-bottle",
    }
    previous = {
        "candidate_id": "material_candidate_001",
        "candidate_status": "pending",
        "review_status": "pending",
        "manual_correction": {"source": "frontend_comment", "note": "should be hand-paper"},
        "action_name": "手与paper操作",
        "primary_object": "paper",
        "canonical_action_type": "hand-paper",
        "canonical_object": "paper",
        "sop_phase": "weighing-paper-prep",
        "interaction_family": "hand-object",
    }

    material_references._preserve_candidate_review_state(candidate, previous)

    assert candidate["manual_correction"]["source"] == "frontend_comment"
    assert candidate["action_name"] == "手与paper操作"
    assert candidate["primary_object"] == "paper"
    assert candidate["canonical_action_type"] == "hand-paper"


def _interaction_proxy_evidence(view: str, *, primary_object: str = "reagent_bottle", score: float = 0.92) -> list[dict]:
    times = [46.433667, 47.2, 49.133666]
    evidence: list[dict] = []
    for index, local_time in enumerate(times):
        shift = index * 3
        hand_bbox = [595 + shift, 178, 682 + shift, 292]
        object_bbox = [603 + shift, 228, 675 + shift, 274]
        evidence.append(
            {
                "view": view,
                "local_time_sec": local_time,
                "interaction_score": score,
                "detections": [
                    {"label": "gloved_hand", "confidence": 0.86, "bbox": hand_bbox},
                    {"label": "pipette", "confidence": 0.99, "bbox": [753, 347, 842, 438]},
                    {"label": "balance", "confidence": 0.98, "bbox": [393, 146, 566, 366]},
                ],
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": primary_object,
                        "score": score,
                        "hand_bbox": hand_bbox,
                        "object_bbox": object_bbox,
                        "iou": 0.39,
                        "distance_px": 22.0,
                    }
                ],
            }
        )
    return evidence


def _session_with_one_yolo_micro(
    tmp_path: Path,
    *,
    create_source: bool = True,
    evidence: list[dict] | None = None,
    primary_object: str = "balance",
) -> Path:
    session = tmp_path / "experiment" / "key_action_index"
    metadata = session / "metadata"
    session.parent.mkdir(parents=True, exist_ok=True)
    (session.parent / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "titration_title_case_20260506_abcd1234",
                "title": "\u6ef4\u5b9a\u5b9e\u9a8cA",
                "created_at": "2026-05-06T12:34:56+08:00",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (session.parent / "timeline_alignment.json").write_text(
        json.dumps(
            {
                "schema_version": "timeline_alignment.v1",
                "alignment_status": "shared_recording",
                "streams": [
                    {"role": "first_person", "alignment_status": "shared_recording"},
                    {"role": "third_person", "alignment_status": "shared_recording"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    annotated_first = session / "clips" / "seg_000001" / "first_person_yolo_annotated.mp4"
    annotated_third = session / "clips" / "seg_000001" / "third_person_yolo_annotated.mp4"
    if create_source:
        annotated_first.parent.mkdir(parents=True, exist_ok=True)
        annotated_first.write_bytes(b"first-yolo")
        annotated_third.write_bytes(b"third-yolo")
    material_evidence = (
        evidence
        if evidence is not None
        else _valid_evidence("third_person", primary_object=primary_object)
        + _valid_evidence("first_person", primary_object=primary_object)
    )
    _write_jsonl(
        metadata / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_000001",
                "first_person": {
                    "local_start_sec": 16.0,
                    "annotated_clip_path": str(annotated_first),
                },
                "third_person": {
                    "local_start_sec": 16.0,
                    "annotated_clip_path": str(annotated_third),
                },
            }
        ],
    )
    _write_jsonl(
        metadata / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_004",
                "parent_segment_id": "seg_000001",
                "start_sec": 46.433667,
                "end_sec": 49.133666,
                "global_start_time": "2026-04-24T16:58:04.433667+08:00",
                "global_end_time": "2026-04-24T16:58:07.133666+08:00",
                "interaction": {"primary_object": primary_object},
                "yolo_evidence": material_evidence,
            },
            {
                "micro_segment_id": "seg_000001_part02_micro_001",
                "parent_segment_id": "seg_000001_part02",
                "start_sec": 1.0,
                "end_sec": 2.0,
                "interaction": {"primary_object": "balance"},
                "yolo_evidence": _valid_evidence("third_person"),
            },
            {
                "micro_segment_id": "seg_000001_micro_no_yolo",
                "parent_segment_id": "seg_000001",
                "start_sec": 50.0,
                "end_sec": 51.0,
                "interaction": {"primary_object": "spatula"},
                "yolo_evidence": [],
            },
        ],
    )
    if {"first_person", "third_person"}.issubset(_evidence_views(material_evidence)):
        _write_formal_dual_event(metadata)
    return session


def test_formal_material_references_root_strips_duplicate_title_date(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    session.mkdir(parents=True)
    (session.parent / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "solid-weighing-fast-rerun-20260513",
                "title": "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_\u5feb\u6d41\u7a0b\u4f18\u5316\u590d\u6d4b_20260513",
                "created_at": "2026-05-13T12:34:56+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert (
        formal_material_references_root(session).name
        == "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_\u5feb\u6d41\u7a0b\u4f18\u5316\u590d\u6d4b_20260513"
    )


def test_formal_material_references_root_uses_domain_title_for_technical_rerun(tmp_path: Path) -> None:
    session = tmp_path / "5f767710-255d-4029-95e4-8c3ed4f0b3fc" / "key_action_index"
    session.mkdir(parents=True)
    (session.parent / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "5f767710-255d-4029-95e4-8c3ed4f0b3fc",
                "title": "VLM enabled key-action rerun 20260514",
                "created_at": "2026-05-14T06:05:08.594137+00:00",
                "description": "Re-run the solid weighing dual-view key-action pipeline.",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (session.parent / "stream_manifest.json").write_text(
        json.dumps(
            {
                "registered_videos": [
                    {
                        "view": "third_person",
                        "video_path": r"C:\Users\Xx7\Desktop\test_long_0512_20260512_124743_785\third.mp4",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (session / "manifest.json").write_text(
        json.dumps({"session_start_time": "2026-05-12T04:47:43.785+00:00"}, ensure_ascii=False),
        encoding="utf-8",
    )

    assert formal_material_references_root(session).name == "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_20260514"


def test_formal_material_references_root_prefers_created_at_over_experiment_id_date(tmp_path: Path) -> None:
    session = tmp_path / "solid-weighing-dual-view-20260508-153648" / "key_action_index"
    session.mkdir(parents=True)
    (session.parent / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "solid-weighing-dual-view-20260508-153648",
                "title": "\u56fa\u4f53\u79f0\u91cf\u53cc\u89c6\u89d2\u5b9e\u9a8c-5.8",
                "created_at": "2026-05-11T10:00:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert formal_material_references_root(session).name == "\u56fa\u4f53\u79f0\u91cf\u53cc\u89c6\u89d2\u5b9e\u9a8c-5.8_20260511"


def test_formal_material_references_root_replaces_title_date_with_created_at(tmp_path: Path) -> None:
    session = tmp_path / "pipetting-demo" / "key_action_index"
    session.mkdir(parents=True)
    (session.parent / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "pipetting-demo",
                "title": "\u79f0\u91cf\u79fb\u6db2\u5b9e\u9a8c 2026-05-22",
                "created_at": "2026-05-31T09:48:00+08:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert formal_material_references_root(session).name == "\u79f0\u91cf\u79fb\u6db2\u5b9e\u9a8c_20260531"


def test_core_v1_physical_action_scope_keeps_only_fast_quality_targets(monkeypatch) -> None:
    monkeypatch.delenv("KEY_ACTION_PHYSICAL_ACTION_TYPES", raising=False)
    monkeypatch.delenv("KEY_ACTION_EVENT_BACKED_CANDIDATE_TYPES", raising=False)
    monkeypatch.setenv("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "core_v1")

    assert material_references.active_physical_action_types() == {
        "hand_object_contact",
        "object_movement",
        "equipment_panel_operation",
    }
    assert material_references._event_physical_action_type("liquid_transfer_candidate") == ""
    assert material_references._event_physical_action_type("container_state_change_candidate") == ""
    assert material_references._event_physical_action_type("equipment_panel_operation_candidate") == "equipment_panel_operation"


def test_core_v1_micro_materials_map_physical_action_types(monkeypatch) -> None:
    monkeypatch.delenv("KEY_ACTION_PHYSICAL_ACTION_TYPES", raising=False)
    monkeypatch.setenv("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "core_v1")

    assert (
        material_references._core_material_physical_action_type(
            {
                "canonical_action_type": "hand-balance",
                "canonical_object": "balance",
                "sop_phase": "balance-weighing",
                "interaction_family": "hand-object",
            },
            {},
            "balance",
            [],
            ["hand-balance"],
        )
        == "equipment_panel_operation"
    )
    assert (
        material_references._core_material_physical_action_type(
            {
                "canonical_action_type": "hand-bottle",
                "canonical_object": "bottle",
                "sop_phase": "reagent-bottle-handling",
                "interaction_family": "hand-object",
            },
            {},
            "sample_bottle",
            [],
            ["hand-bottle"],
        )
        == "hand_object_contact"
    )
    assert (
        material_references._core_material_physical_action_type(
            {"canonical_action_type": "object-movement", "canonical_object": "container", "sop_phase": "object-movement", "interaction_family": "motion"},
            {},
            "container",
            [],
            ["object-movement"],
        )
        == "object_movement"
    )


def test_approved_material_target_name_uses_experiment_type_action_date() -> None:
    used_names: set[str] = set()
    experiment = {"date": "20260518", "title": "\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c"}

    bottle_name = material_references._approved_material_target_name(
        {
            "asset_kind": KEY_CLIP_DIR_NAME,
            "canonical_action_type": "hand-bottle",
            "canonical_object": "bottle",
            "primary_object": "sample_bottle",
            "interaction_family": "hand-object",
            "view": "third_person",
            "start_sec": 1800.0,
            "end_sec": 1801.5,
        },
        Path("bottle_interaction.mp4"),
        experiment,
        used_names,
    )
    assert bottle_name.startswith("\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c_20260518")
    assert "bottle" not in bottle_name

    balance_name = material_references._approved_material_target_name(
        {
            "asset_kind": KEYFRAME_DIR_NAME,
            "frame_type": "peak",
            "canonical_action_type": "hand-balance",
            "canonical_object": "balance",
            "primary_object": "balance",
            "interaction_family": "hand-object",
            "view": "first_person",
            "start_sec": 46.4,
        },
        Path("balance.jpg"),
        experiment,
        used_names,
    )
    assert balance_name.startswith("\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c_20260518")
    assert "balance" not in balance_name

    weighing_contact_name = material_references._approved_material_target_name(
        {
            "asset_kind": KEYFRAME_DIR_NAME,
            "frame_type": "peak",
            "canonical_action_type": "hand-bottle",
            "canonical_object": "bottle",
            "primary_object": "sample_bottle",
            "instrument_context": "balance",
            "interaction_family": "hand-object",
            "view": "first_person",
            "start_sec": 47.2,
        },
        Path("bottle.jpg"),
        experiment,
        used_names,
    )
    assert weighing_contact_name.startswith("\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c_20260518")

    inferred_experiment_name = material_references._approved_material_target_name(
        {
            "asset_kind": KEYFRAME_DIR_NAME,
            "canonical_action_type": "hand-bottle",
            "canonical_object": "bottle",
            "primary_object": "sample_bottle",
            "instrument_context": "balance",
            "interaction_family": "hand-object",
            "view": "first_person",
            "start_sec": 47.2,
        },
        Path("bottle.jpg"),
        {"date": "20260518", "title": "stitched-real-rgb"},
        set(),
    )
    assert inferred_experiment_name.startswith("\u56fa\u4f53\u79f0\u91cf\u5b9e\u9a8c_\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c_20260518")


def test_external_material_library_keeps_frontend_mirror(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    external_root = tmp_path / "D_drive_library"
    monkeypatch.setenv("LAB_MATERIAL_LIBRARY_ROOT", str(external_root))

    label = "滴定实验A_20260506"
    assert formal_material_references_root(session) == external_root / "material_references" / label
    assert frontend_material_references_root(session) == session.parent.parent / "material_references" / label
    assert material_references_root(session) == session.parent / "material_references"


def test_build_yolo_material_references_creates_yolo_only_named_files(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)

    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)

    def fake_ffmpeg(args: list[str]) -> None:
        target = Path(args[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"generated-yolo-material")

    monkeypatch.setattr(material_references, "_run_ffmpeg", fake_ffmpeg)

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    keyframes = sorted((ref_root / KEYFRAME_DIR_NAME).glob("*"))
    clips = sorted((ref_root / KEY_CLIP_DIR_NAME).glob("*"))
    assert summary["file_count"] == 10
    assert summary["experiment_id"] == "titration_title_case_20260506_abcd1234"
    assert summary["experiment_title"] == "\u6ef4\u5b9a\u5b9e\u9a8cA"
    assert summary["experiment_date"] == "20260506"
    assert Path(summary["formal_material_references"]).name == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert Path(summary["simplified_material_references"]).name == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert formal_material_references_root(session).exists()
    assert len(keyframes) == 6
    assert len(clips) == 4
    assert all("\u5929\u5e73\u8bbe\u5907\u9762\u677f\u64cd\u4f5c" in item.name for item in keyframes + clips)
    assert all("20260424" in item.name for item in keyframes + clips)
    assert not any("YOLO" in item.name for item in keyframes + clips)
    assert not any("46.4-49.1s" in item.name or "\u7b2c\u4e00\u4eba\u79f0" in item.name or "\u7b2c\u4e09\u4eba\u79f0" in item.name for item in keyframes + clips)
    assert not any("\u63a5\u89e6\u5e27" in item.name or "\u5cf0\u503c\u5e27" in item.name or "\u91ca\u653e\u5e27" in item.name for item in keyframes)
    assert not any("part02" in item.name or "\u5b8c\u6574\u5b9e\u9a8c\u7247\u6bb5" in item.name for item in keyframes + clips)

    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(index_rows) == 10
    assert all(row["yolo_box_required"] is True for row in index_rows)
    assert all(row["box_filter"] == "hand_and_primary_object_only" for row in index_rows)
    assert {row["frame_role"] for row in index_rows if row["asset_kind"] == KEYFRAME_DIR_NAME} == {"contact", "peak", "release"}
    assert len({row["evidence_group_id"] for row in index_rows}) == 1
    assert len({row["dual_event_id"] for row in index_rows}) == 1
    assert all(row["formal_publish_gate"]["status"] == "passed" for row in index_rows)
    assert all(row["evidence_chain"]["dual_event_id"] == row["dual_event_id"] for row in index_rows)


def test_build_yolo_material_references_blocks_formal_publish_when_action_alignment_blocked(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    metadata = session / "metadata"
    (metadata / "dual_view_action_alignment_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "dual_view_action_alignment.v1",
                "dual_view_action_event_count": 1,
                "formal_event_count": 0,
                "formal_results_allowed": False,
                "decision": "no_formal_dual_view_action_events",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"should-not-publish"))

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    formal_root = formal_material_references_root(session)
    assert summary["status"] == "blocked"
    assert summary["formal_publish_blocked"] is True
    assert summary["blocked_reason"] == "formal_results_not_allowed"
    assert summary["file_count"] == 0
    assert summary["formal_published_file_count"] == 0
    assert (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8") == ""
    assert (formal_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8") == ""
    assert not list((formal_root / KEYFRAME_DIR_NAME).glob("*"))
    assert not list((formal_root / KEY_CLIP_DIR_NAME).glob("*"))


def test_build_yolo_material_references_parallelizes_generation_and_keeps_evidence_group(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    thread_names: set[str] = set()
    lock = threading.Lock()

    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_material_reference_worker_count", lambda: 4)

    def fake_clip_result(**kwargs):
        time.sleep(0.02)
        target = Path(kwargs["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"parallel-clip-material")
        with lock:
            thread_names.add(threading.current_thread().name)
        return True, None

    def fake_frame_result(**kwargs):
        time.sleep(0.02)
        target = Path(kwargs["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"parallel-frame-material")
        with lock:
            thread_names.add(threading.current_thread().name)
        return True, None

    monkeypatch.setattr(material_references, "_render_material_clip_result", fake_clip_result)
    monkeypatch.setattr(material_references, "_render_material_keyframe_result", fake_frame_result)

    summary = build_yolo_material_references(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_references_root(session) / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["material_generation_task_count"] == 10
    assert summary["parallel_workers"] == 4
    assert len(thread_names) > 1
    assert len({row["evidence_group_id"] for row in rows}) == 1
    assert {
        (row["view"], row["asset_kind"])
        for row in rows
        if row["asset_kind"] in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    } == {
        ("first_person", KEYFRAME_DIR_NAME),
        ("third_person", KEYFRAME_DIR_NAME),
        ("first_person", KEY_CLIP_DIR_NAME),
        ("third_person", KEY_CLIP_DIR_NAME),
    }


def test_publish_filter_rejects_decodable_black_or_white_screen_material(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "material_references"
    keyframe_dir = root / KEYFRAME_DIR_NAME
    keyframe_dir.mkdir(parents=True)
    real_frame = keyframe_dir / "real_lab_action.jpg"
    black_frame = keyframe_dir / "black_screen.jpg"
    image = Image.new("RGB", (32, 32), (80, 120, 160))
    for x in range(16, 32):
        for y in range(16, 32):
            image.putpixel((x, y), (220, 80, 60))
    image.save(real_frame)
    Image.new("RGB", (32, 32), (0, 0, 0)).save(black_frame)

    kept, suppressed = material_references._filter_publishable_material_rows(
        [
            {"asset_kind": KEYFRAME_DIR_NAME, "stored_file": str(real_frame), "source_real": True, "placeholder": False},
            {"asset_kind": KEYFRAME_DIR_NAME, "stored_file": str(black_frame), "source_real": True, "placeholder": False},
        ],
        root,
    )

    assert [Path(row["stored_file"]).name for row in kept] == [real_frame.name]
    assert suppressed[0]["suppression_reason"] == "stored_file_not_real_video_material"


def test_build_yolo_material_references_clears_generated_files_without_archive(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    ref_root = material_references_root(session)
    stale_keyframe = ref_root / KEYFRAME_DIR_NAME / "stale.jpg"
    stale_clip = ref_root / KEY_CLIP_DIR_NAME / "stale.mp4"
    stale_keyframe.parent.mkdir(parents=True, exist_ok=True)
    stale_clip.parent.mkdir(parents=True, exist_ok=True)
    stale_keyframe.write_bytes(b"stale-keyframe")
    stale_clip.write_bytes(b"stale-clip")

    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session, archive_existing=False)

    assert summary["archived_count"] == 0
    assert not stale_keyframe.exists()
    assert not stale_clip.exists()
    assert not any(path.name.startswith("stale") for path in (ref_root / KEYFRAME_DIR_NAME).glob("*"))
    assert not any(path.name.startswith("stale") for path in (ref_root / KEY_CLIP_DIR_NAME).glob("*"))


def test_build_yolo_material_references_requires_valid_evidence_per_view(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, evidence=_valid_evidence("third_person"))
    monkeypatch.setenv("KEY_ACTION_ALLOW_PAIRED_VIEW_CONTEXT_MATERIAL", "0")
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["file_count"] == 5
    assert index_rows == []
    assert summary["published_real_file_count"] == 0
    assert any(
        item.get("reason") == "formal_material_publish_gate"
        and item.get("suppression_reason") == "missing_explicit_dual_view_action_event"
        for item in summary["skipped"]
    )


def test_build_yolo_material_references_can_include_paired_context_when_enabled(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, evidence=_valid_evidence("third_person"))
    monkeypatch.setenv("KEY_ACTION_ALLOW_PAIRED_VIEW_CONTEXT_MATERIAL", "1")
    monkeypatch.setenv("KEY_ACTION_REQUIRE_RELIABLE_DUAL_VIEW_ALIGNMENT", "0")
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["file_count"] == 10
    assert index_rows == []
    assert any(
        item.get("reason") == "formal_material_publish_gate"
        and item.get("suppression_reason") == "missing_explicit_dual_view_action_event"
        for item in summary["skipped"]
    )


def test_build_yolo_material_references_rejects_false_physical_evidence(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, evidence=_false_balance_evidence())
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    assert summary["file_count"] == 0
    assert summary["planned_file_count"] == 0
    assert any(item.get("reason") == "no_usable_yolo_hand_object_evidence" for item in summary["skipped"])
    skipped = next(item for item in summary["skipped"] if item.get("reason") == "no_usable_yolo_hand_object_evidence")
    assert "primary_object_confidence_below_threshold" in skipped["diagnostics"]["invalid_reason_counts"]


def test_build_yolo_material_references_accepts_stable_interaction_object_proxy(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(
        tmp_path,
        evidence=_interaction_proxy_evidence("third_person") + _interaction_proxy_evidence("first_person"),
        primary_object="reagent_bottle",
    )
    monkeypatch.setenv("KEY_ACTION_ALLOW_PAIRED_VIEW_CONTEXT_MATERIAL", "0")
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))

    summary = build_yolo_material_references(session)

    ref_root = material_references_root(session)
    index_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["file_count"] == 10
    assert {row["primary_object"] for row in index_rows} == {"reagent_bottle"}
    assert {row["raw_primary_object"] for row in index_rows} == {"reagent_bottle"}
    assert {row["manipulated_object"] for row in index_rows} == {"reagent_bottle"}
    assert {row.get("instrument_context") for row in index_rows} == {None}
    assert {row["canonical_action_type"] for row in index_rows} == {"hand-bottle"}
    assert all("\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c" in row["display_title"] for row in index_rows)
    assert all("\u624b\u90e8\u4e0e\u8bd5\u5242\u74f6\u64cd\u4f5c" in row["stored_filename"] for row in index_rows)
    assert all(row["evidence"]["raw_labels"] for row in index_rows)
    assert all(row["dual_event_id"] for row in index_rows)


def test_filtered_interaction_detections_keep_only_hand_and_primary_object() -> None:
    evidence = _valid_evidence("third_person")[0]
    evidence["detections"].append({"label": "beaker", "confidence": 0.99, "bbox": [1, 2, 30, 40]})

    filtered = material_references._filtered_interaction_detections(evidence, "balance")

    labels = {item["label"] for item, _color in filtered}
    assert labels == {"gloved_hand", "balance"}


def test_filtered_interaction_detections_recomputes_for_corrected_target() -> None:
    evidence = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.95, "bbox": [100, 100, 190, 190]},
            {"label": "paper", "confidence": 0.93, "bbox": [115, 115, 205, 205]},
            {"label": "spatula", "confidence": 0.88, "bbox": [150, 140, 265, 168]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.96,
                "hand_bbox": [100, 100, 190, 190],
                "object_bbox": [115, 115, 205, 205],
            }
        ],
    }

    filtered = material_references._filtered_interaction_detections(evidence, "spatula")

    labels = {item["label"] for item, _color in filtered}
    boxes = {tuple(item["bbox"]) for item, _color in filtered}
    assert labels == {"gloved_hand", "spatula"}
    assert (150.0, 140.0, 265.0, 168.0) in boxes
    assert (115, 115, 205, 205) not in boxes


def test_filtered_interaction_detections_does_not_fallback_to_wrong_cached_target() -> None:
    evidence = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.95, "bbox": [100, 100, 190, 190]},
            {"label": "paper", "confidence": 0.93, "bbox": [115, 115, 205, 205]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.96,
                "hand_bbox": [100, 100, 190, 190],
                "object_bbox": [115, 115, 205, 205],
            }
        ],
    }

    filtered = material_references._filtered_interaction_detections(evidence, "spatula")

    assert filtered == []


def test_corrected_candidate_evidence_selection_prefers_target_object_row() -> None:
    paper_row = {
        "view": "third_person",
        "local_time_sec": 10.0,
        "interaction_score": 0.96,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.95, "bbox": [100, 100, 190, 190]},
            {"label": "paper", "confidence": 0.93, "bbox": [115, 115, 205, 205]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.96,
                "hand_bbox": [100, 100, 190, 190],
                "object_bbox": [115, 115, 205, 205],
            }
        ],
    }
    spatula_row = {
        "view": "third_person",
        "local_time_sec": 10.2,
        "interaction_score": 0.82,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.95, "bbox": [100, 100, 190, 190]},
            {"label": "spatula", "confidence": 0.88, "bbox": [150, 140, 265, 168]},
        ],
        "hand_object_interactions": [],
    }
    candidate = {
        "frame_type": "contact",
        "canonical_object": "spatula",
        "primary_object": "spatula",
        "secondary_objects": [],
    }

    selected = material_references._select_corrected_candidate_evidence_row(
        candidate,
        [paper_row, spatula_row],
        start_sec=10.0,
        end_sec=10.5,
        target_labels={"spatula"},
    )

    assert selected is spatula_row


def test_filtered_interaction_detections_recomputes_cached_false_hand_interaction() -> None:
    import numpy as np

    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    evidence = {
        "view": "third_person",
        "local_time_sec": 44.533333,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.51, "bbox": [2, 212, 151, 354]},
            {"label": "container", "confidence": 0.76, "bbox": [0, 154, 90, 285]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "container",
                "score": 0.94,
                "hand_bbox": [2, 212, 151, 354],
                "object_bbox": [0, 154, 90, 285],
            }
        ],
    }

    filtered = material_references._filtered_interaction_detections(evidence, "container", frame=frame)

    assert filtered == []


def test_tracklet_annotation_detections_are_quality_filtered_before_rendering() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)
    detections = [
        {
            "label": "gloved_hand",
            "confidence": 0.94,
            "bbox": [40, 50, 142, 118],
            "tracklet_id": "trk_hand",
            "tracklet_source": "detected",
            "view": "third_person",
        },
        {
            "label": "paper",
            "confidence": 0.91,
            "bbox": [104, 56, 178, 122],
            "tracklet_id": "trk_paper",
            "tracklet_source": "detected",
            "view": "third_person",
        },
    ]

    filtered = material_references._filtered_tracklet_annotation_detections(
        detections,
        "paper",
        frame=frame,
        source_view="third_person",
    )

    assert filtered == []


def test_tracklet_annotation_detections_require_hand_and_target_pair() -> None:
    import numpy as np

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :] = (185, 135, 45)
    frame[70:122, 86:162] = (242, 244, 246)

    filtered = material_references._filtered_tracklet_annotation_detections(
        [{"label": "paper", "confidence": 0.91, "bbox": [86, 70, 162, 122], "tracklet_id": "trk_paper"}],
        "paper",
        frame=frame,
        source_view="third_person",
    )

    assert filtered == []


def test_annotation_palette_distinguishes_paper_and_balance() -> None:
    assert material_references._annotation_color_for_label("paper") != material_references._annotation_color_for_label("balance")
    assert material_references._annotation_color_for_label("paper") != material_references._annotation_color_for_label("gloved_hand")
    assert material_references._annotation_text_color(material_references._annotation_color_for_label("balance"))


def test_annotation_target_excludes_instrument_context() -> None:
    semantic_fields = {
        "manipulated_object": "paper",
        "instrument_context": "balance",
        "corrected_primary_object": "balance",
    }

    assert material_references._annotation_target_query("paper", semantic_fields) == "paper"
    assert material_references._candidate_target_labels(
        {
            "manipulated_object": "paper",
            "primary_object": "paper",
            "instrument_context": "balance",
            "secondary_objects": ["balance"],
            "corrected_primary_object": "balance",
        }
    ) == {"paper"}


def test_tracklet_rendering_keeps_only_active_hand_object_instance() -> None:
    evidence_row = {
        "view": "third_person",
        "detections": [
            {"label": "gloved_hand", "confidence": 0.93, "bbox": [250, 80, 330, 210]},
            {"label": "paper", "confidence": 0.91, "bbox": [300, 100, 440, 220]},
            {"label": "paper", "confidence": 0.85, "bbox": [20, 150, 150, 260]},
            {"label": "balance", "confidence": 0.96, "bbox": [190, 210, 500, 500]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.94,
                "hand_bbox": [250, 80, 330, 210],
                "object_bbox": [300, 100, 440, 220],
            }
        ],
    }
    tracklet_detections = [
        {"label": "gloved_hand", "confidence": 0.92, "bbox": [252, 82, 332, 212], "tracklet_id": "trk_hand"},
        {"label": "paper", "confidence": 0.88, "bbox": [302, 102, 442, 222], "tracklet_id": "trk_active_paper"},
        {"label": "paper", "confidence": 0.86, "bbox": [20, 150, 150, 260], "tracklet_id": "trk_package_paper"},
        {"label": "balance", "confidence": 0.97, "bbox": [190, 210, 500, 500], "tracklet_id": "trk_balance"},
    ]

    selected = material_references._tracklet_detections_for_active_interaction(
        tracklet_detections,
        evidence_row,
        "paper",
    )

    assert {item["tracklet_id"] for item in selected} == {"trk_hand", "trk_active_paper"}


def test_unboxed_annotation_fallback_is_not_recommended_candidate() -> None:
    rows = [
        {
            "candidate_group_id": "group-1",
            "asset_kind": KEYFRAME_DIR_NAME,
            "exists": True,
            "quality_score": 0.95,
            "yolo_annotation_rendered": False,
            "error": "annotation_fallback_unboxed:no_hand_target_interaction_boxes:paper",
        },
        {
            "candidate_group_id": "group-1",
            "asset_kind": KEY_CLIP_DIR_NAME,
            "exists": True,
            "quality_score": 0.8,
            "yolo_annotation_rendered": True,
        },
    ]

    material_references._mark_recommended_candidates(rows)

    assert rows[0]["recommended"] is False
    assert rows[1]["recommended"] is True


def test_publish_filter_suppresses_non_real_material_rows(tmp_path: Path) -> None:
    root = tmp_path / "material_references"
    keyframe_dir = root / KEYFRAME_DIR_NAME
    keyframe_dir.mkdir(parents=True)
    real_frame = keyframe_dir / "real.jpg"
    placeholder_frame = keyframe_dir / "poster_placeholder.jpg"
    real_frame.write_bytes(b"real")
    placeholder_frame.write_bytes(b"DRY RUN PLACEHOLDER")
    rows = [
        {
            "asset_kind": KEYFRAME_DIR_NAME,
            "stored_file": str(real_frame),
            "source_real": True,
            "placeholder": False,
            "view": "third_person",
        },
        {
            "asset_kind": KEYFRAME_DIR_NAME,
            "stored_file": str(placeholder_frame),
            "source_real": False,
            "placeholder": True,
            "view": "first_person",
        },
    ]

    kept, suppressed = material_references._filter_publishable_material_rows(rows, root)

    assert len(kept) == 1
    assert kept[0]["source_real"] is True
    assert kept[0]["placeholder"] is False
    assert suppressed[0]["reason"] == "non_real_material_suppressed"


def test_stored_path_from_row_keeps_existing_workspace_relative_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("outputs") / "material.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"material")

    resolved = material_references._stored_path_from_row(
        {"stored_file": str(source)},
        tmp_path / "material_references",
    )

    assert resolved == source


def test_source_clip_for_view_prefers_unannotated_segment_clip(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    raw_clip = session / "clips" / "seg_000001" / "third_person.mp4"
    annotated_clip = session / "clips" / "seg_000001" / "third_person_yolo_annotated.mp4"
    raw_clip.parent.mkdir(parents=True, exist_ok=True)
    raw_clip.write_bytes(b"raw")
    annotated_clip.write_bytes(b"annotated")
    segment = {
        "segment_id": "seg_000001",
        "third_person": {
            "clip_path": str(raw_clip),
            "annotated_clip_path": str(annotated_clip),
        },
    }

    source = material_references._source_clip_for_view(
        session,
        {},
        segment,
        {"parent_segment_id": "seg_000001"},
        "third_person",
    )

    assert source == raw_clip


def test_build_yolo_material_references_dry_run_without_video_or_ffmpeg(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path, create_source=False)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: False)

    summary = build_yolo_material_references(session, dry_run=True)

    ref_root = material_references_root(session)
    assert (ref_root / KEYFRAME_DIR_NAME).exists()
    assert (ref_root / KEY_CLIP_DIR_NAME).exists()
    assert summary["file_count"] == 0
    assert summary["planned_file_count"] == 10
    assert summary["experiment_label"] == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert summary["dry_run"] is True
    assert summary["ffmpeg_available"] is False


def test_build_yolo_material_references_propagates_secondary_taxonomy_and_audit(tmp_path: Path, monkeypatch) -> None:
    evidence = _paper_balance_evidence("third_person") + _paper_balance_evidence("first_person")
    session = _session_with_one_yolo_micro(
        tmp_path,
        create_source=False,
        evidence=evidence,
        primary_object="paper",
    )
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: False)

    build_yolo_material_references(session, dry_run=True)

    ref_root = material_references_root(session)
    rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    row = rows[0]
    assert row["canonical_action_type"] == "hand-paper"
    assert row["raw_primary_object"] == "paper"
    assert row["manipulated_object"] == "paper"
    assert row["instrument_context"] == "balance"
    assert row["display_title"] == "\u624b\u90e8\u4e0e\u79f0\u91cf\u7eb8\u64cd\u4f5c"
    assert row["secondary_objects"] == ["balance"]
    assert "hand-paper+panel" in row["secondary_actions"]
    assert "hand-paper+panel" in row["actions"]
    assert "weighing_operation" in row["actions"]
    assert row["window_audit"]["target_object_support"]["interaction_frame_count"] == 6
    assert row["window_audit"]["secondary_object_support"][0]["object"] == "balance"
    assert row["evidence_chain"]["secondary_actions"] == row["secondary_actions"]


def test_build_yolo_material_candidates_marks_best_files_for_review(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
    monkeypatch.setattr(material_references, "_render_filtered_interaction_clip", lambda _source, _offset, _duration, target, _rows, _primary, _segment_start: Path(target).write_bytes(b"annotated-clip"))
    monkeypatch.setattr(material_references, "_extract_filtered_interaction_frame", lambda _source, _offset, target, _row, _primary, **_kwargs: Path(target).write_bytes(b"annotated-frame"))
    build_yolo_material_references(session)

    summary = build_yolo_material_candidates(session, archive_existing=False)

    candidate_root = material_candidates_root(session)
    rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["candidate_count"] == 10
    assert len(rows) == 10
    assert all(row["candidate_status"] == "pending" for row in rows)
    assert all(row["review_required"] is True for row in rows)
    assert all(row["box_filter"] == "hand_and_primary_object_only" for row in rows)
    assert {row["asset_kind"] for row in rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    group_ids = {row["candidate_group_id"] for row in rows}
    assert len(group_ids) == 1
    for group_id in group_ids:
        group_rows = [row for row in rows if row["candidate_group_id"] == group_id]
        assert sum(1 for row in group_rows if row["recommended"] and row["asset_kind"] == KEYFRAME_DIR_NAME) == 2
        assert sum(1 for row in group_rows if row["recommended"] and row["asset_kind"] == KEY_CLIP_DIR_NAME) == 2
        assert {row["view"] for row in group_rows if row["recommended"]} == {"first_person", "third_person"}


def test_candidate_recommendation_excludes_low_quality_and_weak_hand_evidence() -> None:
    base_row = {
        "exists": True,
        "asset_kind": KEYFRAME_DIR_NAME,
        "quality_score": 0.9,
        "yolo_annotation_rendered": True,
        "vlm_semantics": {
            "evidence_packet": {
                "top_detections": [
                    {"label": "paper", "confidence": 0.94, "bbox": [100, 100, 180, 160]},
                    {"label": "gloved_hand", "confidence": 0.93, "bbox": [80, 90, 130, 150]},
                ]
            }
        },
    }
    assert material_references._candidate_recommendation_eligible(base_row) is True

    low_quality_row = dict(base_row, quality_bucket="low_quality")
    assert material_references._candidate_recommendation_eligible(low_quality_row) is False

    weak_hand_row = {
        **base_row,
        "vlm_semantics": {
            "evidence_packet": {
                "top_detections": [
                    {"label": "paper", "confidence": 0.94, "bbox": [100, 100, 180, 160]},
                    {"label": "gloved_hand", "confidence": 0.61, "bbox": [0, 90, 90, 200]},
                ]
            }
        },
    }
    assert "low_confidence_hand_evidence" in material_references._candidate_quality_reasons(weak_hand_row)
    assert material_references._candidate_quality_bucket(weak_hand_row) == "low_quality"
    assert material_references._candidate_recommendation_eligible(weak_hand_row) is False


def test_candidate_recommendation_rechecks_declared_stored_file(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")
    black_frame = tmp_path / "black_screen.jpg"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(black_frame)

    row = {
        "exists": True,
        "asset_kind": KEYFRAME_DIR_NAME,
        "quality_score": 0.9,
        "source_real": True,
        "placeholder": False,
        "stored_file": str(black_frame),
        "yolo_annotation_rendered": True,
        "vlm_semantics": {
            "evidence_packet": {
                "top_detections": [
                    {"label": "paper", "confidence": 0.94, "bbox": [100, 100, 180, 160]},
                    {"label": "gloved_hand", "confidence": 0.93, "bbox": [80, 90, 130, 150]},
                ]
            }
        },
    }

    assert material_references._candidate_recommendation_eligible(row) is False


def test_sparse_paper_interaction_rejects_low_res_static_edge_paper_with_fake_hand() -> None:
    row = {
        "view": "first_person",
        "frame_width": 640,
        "frame_height": 480,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.74, "bbox": [242.0, 242.0, 490.0, 440.0]},
            {"label": "paper", "confidence": 0.69, "bbox": [431.0, 125.0, 640.0, 281.0]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.32,
                "hand_bbox": [242.0, 242.0, 490.0, 440.0],
                "object_bbox": [431.0, 125.0, 640.0, 281.0],
            }
        ],
    }

    selected, mode, valid_count = material_references._select_material_evidence_rows([row], "paper")

    assert selected == []
    assert mode == ""
    assert valid_count == 0


def test_micro_material_evidence_prefers_supplemental_frame_size_for_sparse_gate() -> None:
    compact_row = {
        "view": "first_person",
        "local_time_sec": 77.666667,
        "frame_index": 2330,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.74, "bbox": [242.0, 242.0, 490.0, 440.0]},
            {"label": "paper", "confidence": 0.69, "bbox": [431.0, 125.0, 640.0, 281.0]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "paper",
                "score": 0.32,
                "hand_bbox": [242.0, 242.0, 490.0, 440.0],
                "object_bbox": [431.0, 125.0, 640.0, 281.0],
            }
        ],
    }
    full_row = {
        **compact_row,
        "time_sec": 77.5,
        "local_time_sec": 77.5,
        "frame_index": 2325,
        "frame_width": 640,
        "frame_height": 480,
    }

    rows = material_references._micro_material_evidence_rows(
        {"yolo_evidence": [compact_row]},
        [full_row],
        primary_object="paper",
        start_sec=77.1,
        end_sec=79.1,
    )
    selected, mode, valid_count = material_references._select_material_evidence_rows(rows, "paper")

    assert any(row.get("frame_index") == 2330 and row.get("frame_width") == 640 for row in rows)
    assert selected == []
    assert mode == ""
    assert valid_count == 0


def test_filtered_interaction_detections_prefers_supported_high_confidence_hand_box() -> None:
    row = {
        "detections": [
            {"label": "gloved_hand", "confidence": 0.39, "bbox": [0, 20, 80, 160]},
            {"label": "gloved_hand", "confidence": 0.93, "bbox": [90, 80, 180, 170]},
            {"label": "reagent_bottle", "confidence": 0.92, "bbox": [40, 100, 120, 220]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "reagent_bottle",
                "score": 1.0,
                "hand_bbox": [0, 20, 80, 160],
                "object_bbox": [40, 100, 120, 220],
            },
            {
                "hand_label": "gloved_hand",
                "object_label": "reagent_bottle",
                "score": 0.91,
                "hand_bbox": [90, 80, 180, 170],
                "object_bbox": [40, 100, 120, 220],
            },
        ],
    }

    detections = material_references._filtered_interaction_detections(row, "reagent_bottle")
    hand = next(item for item, _color in detections if item["label"] == "gloved_hand")

    assert hand["bbox"] == [90.0, 80.0, 180.0, 170.0]
    assert hand["confidence"] == 0.93


def test_approve_material_candidates_promotes_recommended_files_only(tmp_path: Path, monkeypatch) -> None:
    session = _session_with_one_yolo_micro(tmp_path)
    external_root = tmp_path / "D_drive_library"
    monkeypatch.setenv("LAB_MATERIAL_LIBRARY_ROOT", str(external_root))
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
    monkeypatch.setattr(material_references, "_render_filtered_interaction_clip", lambda _source, _offset, _duration, target, _rows, _primary, _segment_start: Path(target).write_bytes(b"annotated-clip"))
    monkeypatch.setattr(material_references, "_extract_filtered_interaction_frame", lambda _source, _offset, target, _row, _primary, **_kwargs: Path(target).write_bytes(b"annotated-frame"))
    build_yolo_material_references(session)
    build_yolo_material_candidates(session, archive_existing=False)
    reset_material_references_to_approved_candidates(session, approved_rows=[], merge_existing=False)

    candidate_root = material_candidates_root(session)
    candidate_rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_id = candidate_rows[0]["candidate_group_id"]

    approval = approve_material_candidates(session, candidate_group_id=group_id, reviewer="tester")

    ref_root = material_references_root(session)
    reference_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert approval["approved_count"] == 4
    assert len(reference_rows) == 4
    assert {row["asset_kind"] for row in reference_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert all(row["review_status"] == "accepted" for row in reference_rows)
    assert all(row["taxonomy_schema_version"] == "material_action_taxonomy.v1" for row in reference_rows)
    assert all(row["physical_action_type"] in {"hand_object_contact", "equipment_panel_operation"} for row in reference_rows)
    assert all(row["candidate_disposition_schema_version"] == "material_candidate_disposition.v1" for row in reference_rows)
    assert all(row["evidence_chain"]["schema_version"] == "material_reference_trace.v1" for row in reference_rows)
    assert all(row["evidence_chain"]["candidate_disposition"] == "approved" for row in reference_rows)
    assert all(row["source_clip"] for row in reference_rows)
    assert all(Path(row["stored_file"]).exists() for row in reference_rows)
    simplified_root = Path(approval["material_references_summary"]["simplified_material_references"])
    formal_root = formal_material_references_root(session)
    frontend_root = frontend_material_references_root(session)
    simplified_rows = [
        json.loads(line)
        for line in (simplified_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert simplified_root.name == "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert formal_root == external_root / "material_references" / "\u6ef4\u5b9a\u5b9e\u9a8cA_20260506"
    assert Path(approval["material_references_summary"]["material_references"]) == formal_root
    assert Path(approval["material_references_summary"]["local_material_references_mirror"]) == ref_root
    assert Path(approval["material_references_summary"]["frontend_material_references"]) == frontend_root
    assert (frontend_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").exists()
    assert {row["asset_kind"] for row in simplified_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert len(simplified_rows) == 4
    assert all(formal_root in Path(row["stored_file"]).parents for row in simplified_rows)
    assert len(list((simplified_root / KEYFRAME_DIR_NAME).glob("*"))) == 2
    assert len(list((simplified_root / KEY_CLIP_DIR_NAME).glob("*"))) == 2

    updated_candidates = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    approved_ids = set(approval["approved_candidate_ids"])
    assert {row["candidate_status"] for row in updated_candidates if row["candidate_id"] in approved_ids} == {"approved"}
    assert {row["approval_reason_code"] for row in updated_candidates if row["candidate_id"] in approved_ids} == {"representative_yolo_hand_object_evidence"}
    assert {row["candidate_disposition_schema_version"] for row in updated_candidates if row["candidate_id"] in approved_ids} == {"material_candidate_disposition.v1"}
    assert "not_selected" in {row["candidate_status"] for row in updated_candidates if row["candidate_group_id"] == group_id and row["candidate_id"] not in approved_ids}
    assert all(
        row.get("disposition") == "not_selected_after_group_approval"
        for row in updated_candidates
        if row["candidate_group_id"] == group_id and row["candidate_id"] not in approved_ids
    )
    candidate_manifest = json.loads((candidate_root / "manifest.json").read_text(encoding="utf-8"))
    candidate_summary = json.loads((candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.json").read_text(encoding="utf-8"))
    expected_pending = sum(1 for row in updated_candidates if row["candidate_status"] == "pending")
    expected_pending_groups = {
        row["candidate_group_id"]
        for row in updated_candidates
        if row["candidate_status"] == "pending"
    }
    assert candidate_manifest["pending_total"] == expected_pending
    assert candidate_manifest["approved_total"] == 4
    assert candidate_manifest["not_selected_total"] >= 1
    assert candidate_manifest["pipeline_summary"]["groups_waiting_frontend_review"] == len(expected_pending_groups)
    assert candidate_summary["pending_total"] == candidate_manifest["pending_total"]
