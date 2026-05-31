from __future__ import annotations

import json
from pathlib import Path

import pytest

from key_action_indexer import material_references
from key_action_indexer.material_references import (
    KEYFRAME_DIR_NAME,
    KEY_CLIP_DIR_NAME,
    MATERIAL_CANDIDATE_INDEX_BASENAME,
    MATERIAL_INDEX_BASENAME,
    PAIRED_VIEW_CONTEXT_MODE,
    approve_material_candidates,
    build_yolo_material_candidates,
    build_yolo_material_references,
    confirm_material_candidates,
    complete_dual_view_material_group_ids,
    filter_complete_dual_view_material_rows,
    material_candidates_root,
    material_references_root,
    paired_view_context_scene_gate_passed,
    rename_material_candidates,
)
from key_action_indexer.evidence_package import EVIDENCE_PACKAGE_MANIFEST, PHYSICAL_CHANGE_LOG_JSONL, TIME_ALIGNMENT_JSON


@pytest.fixture(autouse=True)
def _isolate_material_candidate_env(monkeypatch):
    monkeypatch.delenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", raising=False)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _evidence(view: str) -> list[dict]:
    return [
        {
            "view": view,
            "time_sec": 46.4 + index,
            "local_time_sec": 46.4 + index,
            "interaction_score": 0.75,
            "primary_object": "balance",
            "object_box": [160, 140, 320, 300],
            "hand_box": [120, 120, 190, 230],
            "hand_label": "gloved_hand",
            "detections": [
                {"label": "gloved_hand", "confidence": 0.8, "bbox": [120, 120, 190, 230]},
                {"label": "balance", "confidence": 0.8, "bbox": [160, 140, 320, 300]},
            ],
            "hand_object_interactions": [
                {
                    "hand_label": "gloved_hand",
                    "object_label": "balance",
                    "score": 0.75,
                    "hand_bbox": [120, 120, 190, 230],
                    "object_bbox": [160, 140, 320, 300],
                    "iou": 0.08,
                }
            ],
        }
        for index in range(3)
    ]


def _write_formal_dual_event(metadata: Path, *, micro_segment_id: str = "seg_000001_micro_001") -> None:
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


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "experiment" / "key_action_index"
    first_clip = session / "clips" / "first.mp4"
    third_clip = session / "clips" / "third.mp4"
    first_clip.parent.mkdir(parents=True, exist_ok=True)
    first_clip.write_bytes(b"first")
    third_clip.write_bytes(b"third")
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
    _write_jsonl(
        session / "metadata" / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_000001",
                "start_sec": 40.0,
                "first_person_annotated_clip": str(first_clip),
                "third_person_annotated_clip": str(third_clip),
                "view_start_sec": {"first_person": 40.0, "third_person": 40.0},
            }
        ],
    )
    _write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_001",
                "parent_segment_id": "seg_000001",
                "start_sec": 46.4,
                "end_sec": 49.4,
                "primary_object": "balance",
                "interaction": {"primary_object": "balance"},
                "yolo_evidence": _evidence("first_person") + _evidence("third_person"),
            }
        ],
    )
    _write_formal_dual_event(session / "metadata")
    return session


def test_paired_view_context_scene_gate_rejects_non_lab_context(tmp_path: Path, monkeypatch) -> None:
    Image = pytest.importorskip("PIL.Image")
    ImageDraw = pytest.importorskip("PIL.ImageDraw")
    monkeypatch.setenv("KEY_ACTION_PAIRED_VIEW_CONTEXT_SCENE_GATE", "1")

    lab_frame = tmp_path / "lab_context.jpg"
    lab_image = Image.new("RGB", (160, 90), (36, 93, 140))
    draw = ImageDraw.Draw(lab_image)
    draw.rectangle((12, 42, 52, 72), fill=(235, 70, 35))
    draw.rectangle((80, 30, 124, 68), fill=(240, 170, 35))
    lab_image.save(lab_frame)

    gray_frame = tmp_path / "non_lab_context.jpg"
    Image.new("RGB", (160, 90), (205, 205, 205)).save(gray_frame)

    base_row = {
        "physical_evidence_mode": PAIRED_VIEW_CONTEXT_MODE,
        "candidate_source": "paired_view_micro_segment_key_asset_reference",
        "box_filter": "paired_view_time_alignment_asset_reference",
    }

    assert paired_view_context_scene_gate_passed({**base_row, "source_file": str(lab_frame)}) is True
    assert paired_view_context_scene_gate_passed({**base_row, "source_file": str(gray_frame)}) is False


def test_complete_dual_view_material_filter_requires_both_views_and_asset_types(monkeypatch) -> None:
    rows = [
        {"candidate_id": "complete-third-clip", "candidate_group_id": "complete", "view": "third_person", "asset_kind": KEY_CLIP_DIR_NAME},
        {"candidate_id": "complete-third-frame", "candidate_group_id": "complete", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "complete-first-clip", "candidate_group_id": "complete", "view": "first_person", "asset_kind": KEY_CLIP_DIR_NAME},
        {"candidate_id": "complete-first-frame", "candidate_group_id": "complete", "view": "first_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "single-third-clip", "candidate_group_id": "single", "view": "third_person", "asset_kind": KEY_CLIP_DIR_NAME},
        {"candidate_id": "single-third-frame", "candidate_group_id": "single", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME},
    ]
    assert complete_dual_view_material_group_ids(rows) == {"complete"}

    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", "1")
    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_KEY_CLIPS", "1")
    filtered = filter_complete_dual_view_material_rows(rows)
    assert {row["candidate_id"] for row in filtered} == {
        "complete-third-clip",
        "complete-third-frame",
        "complete-first-clip",
        "complete-first-frame",
    }


def test_complete_dual_view_material_filter_uses_micro_segment_across_view_local_candidate_groups(monkeypatch) -> None:
    rows = [
        {"candidate_id": "third-clip", "candidate_group_id": "third-only-candidate", "micro_segment_id": "micro-1", "view": "third_person", "asset_kind": KEY_CLIP_DIR_NAME},
        {"candidate_id": "third-frame", "candidate_group_id": "third-only-candidate", "micro_segment_id": "micro-1", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "first-clip", "candidate_group_id": "first-only-candidate", "micro_segment_id": "micro-1", "view": "first_person", "asset_kind": KEY_CLIP_DIR_NAME},
        {"candidate_id": "first-frame", "candidate_group_id": "first-only-candidate", "micro_segment_id": "micro-1", "view": "first_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "first-other", "candidate_group_id": "first-other", "micro_segment_id": "micro-2", "view": "first_person", "asset_kind": KEYFRAME_DIR_NAME},
    ]

    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", "1")
    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_KEY_CLIPS", "1")
    filtered = filter_complete_dual_view_material_rows(rows)

    assert {row["candidate_id"] for row in filtered} == {
        "third-clip",
        "third-frame",
        "first-clip",
        "first-frame",
    }


def test_complete_dual_view_material_filter_accepts_nested_payload_rows(monkeypatch) -> None:
    rows = [
        {"candidate_id": "third-clip", "payload": {"micro_segment_id": "micro-1", "view": "third_person", "asset_kind": KEY_CLIP_DIR_NAME}},
        {"candidate_id": "third-frame", "payload": {"micro_segment_id": "micro-1", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME}},
        {"candidate_id": "first-clip", "payload": {"micro_segment_id": "micro-1", "view": "first_person", "asset_kind": KEY_CLIP_DIR_NAME}},
        {"candidate_id": "first-frame", "payload": {"micro_segment_id": "micro-1", "view": "first_person", "asset_kind": KEYFRAME_DIR_NAME}},
    ]

    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", "1")
    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_KEY_CLIPS", "1")
    filtered = filter_complete_dual_view_material_rows(rows)

    assert len(filtered) == 4


def test_complete_dual_view_material_filter_requires_key_clips_by_default(monkeypatch) -> None:
    rows = [
        {"candidate_id": "third-frame", "micro_segment_id": "micro-1", "canonical_action_type": "hand-paper", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "first-frame", "micro_segment_id": "micro-1", "canonical_action_type": "hand-paper", "view": "first_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "third-other-frame", "micro_segment_id": "micro-2", "canonical_action_type": "hand-bottle", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME},
    ]

    monkeypatch.delenv("KEY_ACTION_REQUIRE_DUAL_VIEW_KEY_CLIPS", raising=False)
    monkeypatch.delenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", raising=False)
    filtered = filter_complete_dual_view_material_rows(rows)

    assert filtered == []


def test_complete_dual_view_material_filter_rejects_action_mismatched_same_micro(monkeypatch) -> None:
    rows = [
        {"candidate_id": "third-frame", "micro_segment_id": "micro-1", "canonical_action_type": "hand-bottle", "view": "third_person", "asset_kind": KEYFRAME_DIR_NAME},
        {"candidate_id": "first-frame", "micro_segment_id": "micro-1", "canonical_action_type": "hand-balance", "view": "first_person", "asset_kind": KEYFRAME_DIR_NAME},
    ]

    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", "1")
    filtered = filter_complete_dual_view_material_rows(rows)

    assert filtered == []


def test_complete_dual_view_material_filter_rejects_object_mismatched_same_action(monkeypatch) -> None:
    rows = [
        {
            "candidate_id": "third-frame",
            "micro_segment_id": "micro-1",
            "canonical_action_type": "hand-bottle",
            "primary_object": "reagent_bottle",
            "view": "third_person",
            "asset_kind": KEYFRAME_DIR_NAME,
        },
        {
            "candidate_id": "first-frame",
            "micro_segment_id": "micro-1",
            "canonical_action_type": "hand-bottle",
            "primary_object": "sample_bottle",
            "view": "first_person",
            "asset_kind": KEYFRAME_DIR_NAME,
        },
    ]

    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_COMPLETE_MATERIAL_GROUPS", "1")
    monkeypatch.setenv("KEY_ACTION_REQUIRE_DUAL_VIEW_KEY_CLIPS", "0")
    filtered = filter_complete_dual_view_material_rows(rows)

    assert filtered == []


def test_material_candidates_require_review_before_publish(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
    build_yolo_material_references(session, archive_existing=False)

    summary = build_yolo_material_candidates(session, archive_existing=False)

    candidate_root = material_candidates_root(session)
    rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["candidate_count"] == 10
    assert all(row["candidate_status"] == "pending" for row in rows)
    assert all(row["review_required"] is True for row in rows)
    assert all(row["pipeline_stage"] == "frontend_review_gate" for row in rows)
    assert {row["asset_kind"] for row in rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert summary["pipeline_summary"]["groups_waiting_frontend_review"] == 1


def test_segment_candidates_are_used_when_micro_segments_are_missing(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    clip = session / "clips" / "seg_000001" / "third_person_yolo_annotated.mp4"
    middle = session / "keyframes" / "seg_000001" / "third_person_middle.jpg"
    start = session / "keyframes" / "seg_000001" / "third_person_start.jpg"
    clip.parent.mkdir(parents=True, exist_ok=True)
    middle.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"clip")
    middle.write_bytes(b"middle")
    start.write_bytes(b"start")
    _write_jsonl(
        session / "metadata" / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_000001",
                "text_description": {"action_type": "pipetting"},
                "third_person": {
                    "annotated_clip_path": str(clip),
                    "local_start_sec": 3.5,
                    "local_end_sec": 16.5,
                    "yolo_detection_count": 42,
                    "yolo_label_counts": {"sample_bottle": 20, "pipette": 18},
                },
                "asset_bindings": [
                    {
                        "level": "segment",
                        "segment_id": "seg_000001",
                        "view": "third_person",
                        "clip_path": str(clip),
                        "keyframe_path": str(middle),
                        "keyframe_paths": [str(start), str(middle)],
                        "local_start_sec": 3.5,
                        "local_end_sec": 16.5,
                        "yolo_detection_count": 42,
                    }
                ],
            }
        ],
    )
    report = material_candidates_root(session) / "专业报告" / "report.pdf"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_bytes(b"report")
    _write_jsonl(
        material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl",
        [
            {
                "asset_kind": "专业报告",
                "candidate_id": "report_candidate",
                "candidate_group_id": "report_group",
                "candidate_status": "pending",
                "review_status": "pending",
                "stored_file": str(report),
                "stored_filename": report.name,
                "file_name": report.name,
            }
        ],
    )

    summary = build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    physical_rows = [row for row in rows if row["asset_kind"] in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}]

    assert summary["segment_level_fallback_used"] is True
    assert summary["preserved_candidate_count"] == 1
    assert {row["asset_kind"] for row in physical_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert all(row["candidate_status"] == "pending" for row in physical_rows)
    assert all(row["pipeline_status"] == "segment_level_key_action_review_required" for row in physical_rows)
    assert all(row["candidate_source"] == "segment_level_key_action" for row in physical_rows)
    assert any(row["candidate_id"] == "report_candidate" for row in rows)


def test_sparse_micro_evidence_stays_micro_level_human_review_candidate(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    clip = session / "clips" / "first.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"first")
    sparse_evidence = {
        "view": "first_person",
        "time_sec": 12.0,
        "local_time_sec": 12.0,
        "frame_index": 360,
        "interaction_score": 0.25,
        "detections": [
            {"label": "gloved_hand", "confidence": 0.82, "bbox": [20, 20, 90, 110]},
            {"label": "reagent_bottle", "confidence": 0.45, "bbox": [88, 28, 150, 140]},
        ],
        "hand_object_interactions": [
            {
                "hand_label": "gloved_hand",
                "object_label": "reagent_bottle",
                "score": 0.25,
                "hand_bbox": [20, 20, 90, 110],
                "object_bbox": [88, 28, 150, 140],
            }
        ],
    }
    _write_jsonl(
        session / "metadata" / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_000001",
                "start_sec": 10.0,
                "first_person": {"clip_path": str(clip), "local_start_sec": 10.0, "local_end_sec": 15.0},
            }
        ],
    )
    _write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_001",
                "parent_segment_id": "seg_000001",
                "start_sec": 11.5,
                "end_sec": 13.0,
                "primary_object": "reagent_bottle",
                "interaction": {"primary_object": "reagent_bottle"},
                "yolo_evidence": [sparse_evidence],
            }
        ],
    )
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_render_filtered_interaction_clip", lambda _source, _offset, _duration, target, _rows, _primary, _segment_start: Path(target).write_bytes(b"annotated-clip"))
    monkeypatch.setattr(material_references, "_extract_filtered_interaction_frame", lambda _source, _offset, target, _row, _primary, **_kwargs: Path(target).write_bytes(b"annotated-frame"))

    summary = build_yolo_material_candidates(session, archive_existing=False, rebuild_source=True)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    physical_rows = [row for row in rows if row["asset_kind"] in {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}]

    assert summary["segment_level_fallback_used"] is False
    assert physical_rows == []
    source_summary = json.loads(
        (material_references_root(session) / f"{MATERIAL_INDEX_BASENAME}.json").read_text(encoding="utf-8")
    )
    assert any(
        item.get("reason") == "formal_material_publish_gate"
        and item.get("suppression_reason") in {"single_view_material_rejected", "missing_reliable_dual_view_alignment"}
        for item in source_summary["skipped"]
    )


def test_video_understanding_events_surface_as_review_candidates(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    clip = session / "clips" / "micro" / "seg_000001_micro_001_third_person.mp4"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    clip.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    clip.write_bytes(b"clip")
    _write_jsonl(
        session / "metadata" / "video_understanding.jsonl",
        [
            {
                "video_event_id": "seg_000001_micro_001:liquid_transfer_candidate",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "liquid_transfer_candidate",
                "primary_object": "beaker",
                "confidence": 0.62,
                "confidence_reasons": ["candidate generated from object/action/state heuristics"],
                "confirmation_level": "visual_confirmed",
                "asset_refs": [
                    {"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe), "asset_id": "frame_asset"},
                    {"asset_type": "clip", "rel": "third_person.clip_path", "path": str(clip), "asset_id": "clip_asset"},
                ],
            }
        ],
    )

    summary = build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [row for row in rows if row.get("candidate_source") == "video_understanding_event"]

    assert summary["pipeline_summary"]["event_backed_candidates"]["candidate_count"] == 2
    assert {row["asset_kind"] for row in event_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert {row["event_type"] for row in event_rows} == {"liquid_transfer_candidate"}
    assert all(row["review_route"] == "vlm_review" for row in event_rows)
    assert all(row["quality_bucket"] == "review_candidate" for row in event_rows)
    assert all(row["physical_action_type"] == "liquid_movement" for row in event_rows)
    assert all(row["canonical_action_type"] == "liquid_movement" for row in event_rows)
    assert all(row["recommended"] is False for row in event_rows)
    assert all(row["yolo_annotated_required"] is False for row in event_rows)
    assert all(row["pipeline_status"] == "event_backed_review_required" for row in event_rows)


def test_view_action_evidence_surfaces_needs_review_candidates(tmp_path: Path, monkeypatch) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    metadata = session / "metadata"
    first_video = session / "source" / "first.mp4"
    third_video = session / "source" / "third.mp4"
    first_video.parent.mkdir(parents=True, exist_ok=True)
    first_video.write_bytes(b"first-video")
    third_video.write_bytes(b"third-video")
    window_sync = session.parent / "windows" / "formal_window_001" / "window_sync_index.csv"
    window_sync.parent.mkdir(parents=True, exist_ok=True)
    window_sync.write_text(
        "window_sync_index,global_timestamp_us,reference_camera,first_frame_index,third_frame_index,source_sync_index\n"
        "0,1000000,third_person,1,1,100\n",
        encoding="utf-8",
    )
    material_references_root(session).mkdir(parents=True, exist_ok=True)
    (material_references_root(session) / f"{MATERIAL_INDEX_BASENAME}.jsonl").write_text("{}\n", encoding="utf-8")
    _write_jsonl(
        metadata / "video_sources.jsonl",
        [
            {"view_id": "first_person", "role": "first_person", "absolute_path": str(first_video), "fps": 30.0},
            {"view_id": "third_person", "role": "third_person", "absolute_path": str(third_video), "fps": 20.0},
        ],
    )
    (metadata / "formal_experiment_windows.json").write_text(
        json.dumps(
            {
                "windows": [
                    {
                        "experiment_window_id": "formal_window_001",
                        "unit_id": "combined",
                        "start_sec": 10.0,
                        "end_sec": 20.0,
                        "start_global_timestamp_us": 1_000_000,
                        "start_sync_index": 100,
                        "end_sync_index": 200,
                        "source_window_sync_index": str(window_sync),
                        "status": "formal_window_needs_human_review",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        metadata / "view_action_evidence.jsonl",
        [
            {
                "evidence_id": "first_evidence",
                "view": "first_person",
                "peak_sec": 12.0,
                "start_sec": 11.8,
                "end_sec": 12.2,
                "action_type": "container_operation",
                "canonical_action_type": "hand-container",
                "primary_object": "container",
                "interaction_type": "hand_object_contact",
                "interaction_frame_count": 2,
                "row_count": 3,
                "confidence": 0.83,
            },
            {
                "evidence_id": "third_evidence",
                "view": "third_person",
                "peak_sec": 12.1,
                "start_sec": 11.9,
                "end_sec": 12.3,
                "action_type": "container_operation",
                "canonical_action_type": "hand-container",
                "primary_object": "container",
                "interaction_type": "hand_object_contact",
                "interaction_frame_count": 2,
                "row_count": 3,
                "confidence": 0.84,
            },
        ],
    )

    def _fake_run_ffmpeg(args: list[str]) -> None:
        target = Path(args[-1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"real-generated-material")

    monkeypatch.setattr(material_references, "_run_ffmpeg", _fake_run_ffmpeg)
    external_root = tmp_path / "LabMaterialLibrary"
    monkeypatch.setenv("KEY_ACTION_MATERIAL_LIBRARY_ROOT", str(external_root))

    summary = build_yolo_material_candidates(session, archive_existing=False)
    candidate_root = material_candidates_root(session)
    rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    view_action_rows = [row for row in rows if row.get("candidate_source") == "view_action_evidence_needs_review"]

    assert summary["pipeline_summary"]["view_action_review_candidates"]["candidate_count"] == 4
    assert len(view_action_rows) == 4
    assert {row["view"] for row in view_action_rows} == {"first_person", "third_person"}
    assert {row["asset_kind"] for row in view_action_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert all(row["candidate_status"] == "needs_review" for row in view_action_rows)
    assert all(row["review_status"] == "needs_review" for row in view_action_rows)
    assert all(row["memory_write_allowed"] is False for row in view_action_rows)
    assert all(row["official_material"] is False for row in view_action_rows)
    assert all(row["source_window_sync_index"] == str(window_sync) for row in view_action_rows)
    keyframe_rows = [row for row in view_action_rows if row["asset_kind"] == KEYFRAME_DIR_NAME]
    assert keyframe_rows
    assert all(row["selected_keyframe_score"] is not None for row in keyframe_rows)
    assert all(row["selected_keyframe_reason"] for row in keyframe_rows)
    assert (metadata / "keyframe_quality_report.json").exists()
    stream_rows = [
        json.loads(line)
        for line in (candidate_root / "material_stream.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert stream_rows
    assert all(row["source_window_sync_index"] == str(window_sync) for row in stream_rows)
    assert all(row["official_status"] == "needs_review" for row in stream_rows)
    assert (candidate_root / "material_stream.jsonl").exists()
    assert (metadata / "review_candidate_materials.jsonl").exists()
    assert (metadata / "high_confidence_materials.jsonl").exists()
    assert (metadata / "material_candidate_review_manifest.json").exists()
    assert (metadata / "dual_view_action_phase_report.json").exists()
    assert (metadata / "action_candidate_rows.jsonl").exists()
    assert (metadata / "keyclip_quality_report.json").exists()
    assert (metadata / "material_self_validation_report.json").exists()
    phase_report = json.loads((metadata / "dual_view_action_phase_report.json").read_text(encoding="utf-8"))
    assert phase_report["status_counts"]["suspicious_needs_review"] == 1
    external_experiment_root = external_root / "experiment"
    assert (external_experiment_root / "review_candidate_materials.jsonl").exists()
    assert (external_experiment_root / "material_stream.jsonl").exists()
    assert (external_experiment_root / "cli_ready_report.json").exists()


def test_weak_event_contact_does_not_resurface_as_default_review_candidate(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_001",
                "parent_segment_id": "episode_000001",
                "interaction": {"primary_object": "paper"},
                "yolo_evidence": [
                    {
                        "view": "first_person",
                        "time_sec": 1.0,
                        "interaction_score": 0.12,
                        "detections": [
                            {"label": "gloved_hand", "confidence": 0.5, "bbox": [1, 1, 20, 20]},
                            {"label": "paper", "confidence": 0.55, "bbox": [80, 80, 160, 160]},
                        ],
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        session / "metadata" / "video_understanding.jsonl",
        [
            {
                "video_event_id": "seg_000001_micro_001:hand_object_contact",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "hand_object_contact",
                "primary_object": "paper",
                "confidence": 0.8,
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            },
            {
                "video_event_id": "seg_000001_micro_001:object_state_change",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "object_state_change",
                "primary_object": "paper",
                "confidence": 0.88,
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [row for row in rows if row.get("candidate_source") == "video_understanding_event"]

    assert {row["event_type"] for row in event_rows} == {"hand_object_contact"}
    assert all(row["quality_bucket"] == "low_quality" for row in event_rows)
    assert all(row["pipeline_status"] == "event_backed_low_quality_evidence" for row in event_rows)
    assert all(row["physical_evidence_mode"] == "event_backed_gate_failed" for row in event_rows)
    assert all(row["physical_action_type"] == "hand_object_contact" for row in event_rows)
    assert all(row["review_route"] == "human_review" for row in event_rows)
    assert all("no_valid_yolo_contact_evidence" in row["review_reason_codes"] for row in event_rows)
    assert not any(row.get("event_type") == "object_state_change" for row in rows)


def test_sparse_beaker_contact_routes_to_human_review_not_default_ready(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "seg_000001_micro_001",
                "parent_segment_id": "episode_000001",
                "interaction": {"primary_object": "beaker"},
                "yolo_evidence": [
                    {
                        "view": "third_person",
                        "time_sec": 1.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "interaction_score": 0.36,
                        "detections": [
                            {"label": "gloved_hand", "confidence": 0.83, "bbox": [430, 106, 554, 214]},
                            {"label": "beaker", "confidence": 0.43, "bbox": [383, 200, 502, 338]},
                        ],
                        "hand_object_interactions": [
                            {
                                "hand_label": "gloved_hand",
                                "object_label": "beaker",
                                "score": 0.36,
                                "hand_bbox": [430, 106, 554, 214],
                                "object_bbox": [383, 200, 502, 338],
                            }
                        ],
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        session / "metadata" / "video_understanding.jsonl",
        [
            {
                "video_event_id": "seg_000001_micro_001:hand_object_contact:beaker",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "hand_object_contact",
                "primary_object": "beaker",
                "confidence": 0.84,
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_row = next(row for row in rows if row.get("event_type") == "hand_object_contact")

    assert event_row["quality_bucket"] == "review_candidate"
    assert event_row["review_route"] == "human_review"
    assert event_row["physical_evidence_mode"] == "event_backed_sparse_contact_review"
    assert event_row["usable_contact_yolo_evidence_count"] == 1
    assert event_row["valid_contact_yolo_evidence_count"] == 0
    assert event_row["contact_peak_score"] == 0.36
    assert "sparse_contact_evidence_needs_human_review" in event_row["review_reason_codes"]
    assert "valid_contact_frame_count_below_auto_ready" in event_row["review_reason_codes"]


def test_event_contact_reconnects_stale_micro_id_by_parent_segment_and_object(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "episode_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [
            {
                "micro_segment_id": "episode_000001_micro_001",
                "parent_segment_id": "episode_000001",
                "interaction": {"primary_object": "beaker", "max_interaction_score": 0.36},
                "keyframes": {"peak_frame": str(keyframe)},
                "yolo_evidence": [
                    {
                        "view": "third_person",
                        "time_sec": 1.0,
                        "frame_width": 640,
                        "frame_height": 480,
                        "interaction_score": 0.36,
                        "detections": [
                            {"label": "gloved_hand", "confidence": 0.83, "bbox": [430, 106, 554, 214]},
                            {"label": "beaker", "confidence": 0.43, "bbox": [383, 200, 502, 338]},
                        ],
                        "hand_object_interactions": [
                            {
                                "hand_label": "gloved_hand",
                                "object_label": "beaker",
                                "score": 0.36,
                                "hand_bbox": [430, 106, 554, 214],
                                "object_bbox": [383, 200, 502, 338],
                            }
                        ],
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        session / "metadata" / "video_understanding.jsonl",
        [
            {
                "video_event_id": "seg_000001_micro_001:hand_object_contact:beaker",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "hand_object_contact",
                "primary_object": "beaker",
                "confidence": 0.84,
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_row = next(row for row in rows if row.get("event_type") == "hand_object_contact")

    assert event_row["quality_bucket"] == "review_candidate"
    assert event_row["physical_evidence_mode"] == "event_backed_sparse_contact_review"
    assert event_row["usable_contact_yolo_evidence_count"] == 1
    assert event_row["contact_peak_score"] == 0.36


def test_event_backed_candidates_dedupe_same_micro_action_across_sources(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "micro_segments.jsonl",
        [{"micro_segment_id": "seg_000001_micro_001", "parent_segment_id": "episode_000001"}],
    )
    _write_jsonl(
        session / "metadata" / "video_understanding.jsonl",
        [
            {
                "video_event_id": "seg_000001_micro_001:object_movement_detected:sample_bottle",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "object_movement_detected",
                "primary_object": "sample_bottle",
                "confidence": 0.7,
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )
    _write_jsonl(
        session / "metadata" / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "seg_000001_micro_001:object_trajectory_movement:sample_bottle",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "evidence_type": "object_trajectory_movement",
                "object_label": "sample_bottle",
                "confidence": 0.65,
                "metrics": {"measurement": {"point_count": 4, "identity_confidence": 0.8, "displacement_px": 18}},
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    summary = build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_rows = [row for row in rows if row.get("physical_action_type") == "object_movement"]

    assert summary["pipeline_summary"]["event_backed_candidates"]["event_groups_considered"] == 1
    assert len(event_rows) == 1
    assert event_rows[0]["event_type"] == "object_trajectory_movement"
    assert event_rows[0]["candidate_source"] == "advanced_vision_evidence"


def test_label_level_pseudotrack_object_movement_is_suppressed(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "seg_000001_micro_001:object_trajectory_movement:reagent_bottle",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "evidence_type": "object_trajectory_movement",
                "object_label": "reagent_bottle",
                "confidence": 0.93,
                "confirmation_level": "measured",
                "confidence_reasons": [
                    "YOLO detection rows converted to standard object track observation",
                    "source_mode=yolo_frame_rows",
                    "motion_px=48.0",
                ],
                "limitations": ["label-level pseudo-track; no external re-identification tracker id"],
                "metrics": {
                    "measurement": {
                        "track_id": "yolo_track:seg_000001_micro_001:first_person:reagent_bottle",
                        "point_count": 15,
                        "identity_confidence": 0.72,
                        "displacement_px": 48.0,
                        "path_length_px": 64.0,
                        "source_mode": "yolo_frame_rows",
                    }
                },
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    summary = build_yolo_material_candidates(session, archive_existing=False)
    candidate_rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    event_summary = summary["pipeline_summary"]["event_backed_candidates"]
    assert candidate_rows == []
    assert event_summary["event_groups_considered"] == 1
    assert event_summary["surfaced_event_groups"] == 0
    assert event_summary["suppressed_event_groups"] == 1
    assert event_summary["suppressed_reason_counts"]["event_backed_gate_blocked"] == 1


def test_gate_rejected_object_movement_is_suppressed(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "seg_000001_micro_001:object_trajectory_movement:sample_bottle",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "evidence_type": "object_trajectory_movement",
                "object_label": "sample_bottle",
                "confidence": 0.86,
                "confirmation_level": "rejected",
                "visual_confirmation_level": "trajectory_rejected_by_gate",
                "metrics": {
                    "measurement": {
                        "point_count": 8,
                        "identity_confidence": 0.86,
                        "displacement_px": 42.0,
                        "path_length_px": 55.0,
                    }
                },
                "physical_event_gate": {
                    "status": "rejected",
                    "reject_reasons": ["bbox_jitter_or_static_object"],
                },
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    summary = build_yolo_material_candidates(session, archive_existing=False)
    candidate_rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    event_summary = summary["pipeline_summary"]["event_backed_candidates"]
    assert candidate_rows == []
    assert event_summary["event_groups_considered"] == 1
    assert event_summary["surfaced_event_groups"] == 0
    assert event_summary["suppressed_reason_counts"]["event_backed_gate_blocked"] == 1


def test_event_backed_review_candidates_are_limited_per_action_and_object(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "shared_peak.jpg"
    clip = session / "clips" / "micro" / "shared_clip.mp4"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    clip.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    clip.write_bytes(b"clip")
    rows = []
    for index in range(8):
        rows.append(
            {
                "evidence_id": f"seg_{index:06d}:object_trajectory_movement:sample_bottle",
                "segment_id": f"episode_{index:06d}",
                "micro_segment_id": f"seg_{index:06d}_micro_001",
                "evidence_type": "object_trajectory_movement",
                "object_label": "sample_bottle",
                "confidence": 0.9 - index * 0.01,
                "metrics": {"measurement": {"point_count": 5, "identity_confidence": 0.8, "displacement_px": 30}},
                "asset_refs": [
                    {"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)},
                    {"asset_type": "clip", "rel": "third_person.clip_path", "path": str(clip)},
                ],
            }
        )
    _write_jsonl(session / "metadata" / "advanced_vision_evidence.jsonl", rows)

    summary = build_yolo_material_candidates(session, archive_existing=False)
    candidate_rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    event_summary = summary["pipeline_summary"]["event_backed_candidates"]
    assert event_summary["event_groups_considered"] == 8
    assert event_summary["surfaced_event_groups"] == 5
    assert event_summary["suppressed_event_groups"] == 3
    assert event_summary["suppressed_reason_counts"]["event_backed_review_object_limit_exceeded"] == 3
    assert len(candidate_rows) == 10
    assert {row["quality_bucket"] for row in candidate_rows} == {"review_candidate"}


def test_event_backed_low_quality_candidates_are_limited_to_diagnostics_sample(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "shared_peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    rows = []
    for index in range(6):
        rows.append(
            {
                "evidence_id": f"seg_{index:06d}:equipment_control_change:reagent_bottle",
                "segment_id": f"episode_{index:06d}",
                "micro_segment_id": f"seg_{index:06d}_micro_001",
                "evidence_type": "equipment_control_change",
                "object_label": "reagent_bottle",
                "confidence": 0.55,
                "confirmation_level": "candidate",
                "visual_confirmation_level": "candidate_requires_panel_ocr_or_control_detector",
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        )
    _write_jsonl(session / "metadata" / "advanced_vision_evidence.jsonl", rows)

    summary = build_yolo_material_candidates(session, archive_existing=False)
    candidate_rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    event_summary = summary["pipeline_summary"]["event_backed_candidates"]
    assert event_summary["event_groups_considered"] == 6
    assert event_summary["surfaced_event_groups"] == 2
    assert event_summary["suppressed_event_groups"] == 4
    assert event_summary["suppressed_reason_counts"]["event_backed_low_quality_object_limit_exceeded"] == 4
    assert len(candidate_rows) == 2
    assert {row["quality_bucket"] for row in candidate_rows} == {"low_quality"}


def test_equipment_panel_candidate_requires_panel_evidence_not_context_object(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "advanced_vision_evidence.jsonl",
        [
            {
                "evidence_id": "seg_000001_micro_001:equipment_control_change:reagent_bottle",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "evidence_type": "equipment_control_change",
                "object_label": "reagent_bottle",
                "confidence": 0.55,
                "confirmation_level": "candidate",
                "visual_confirmation_level": "candidate_requires_panel_ocr_or_control_detector",
                "limitations": ["no OCR text confirmed; button/knob state requires panel detector"],
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_row = next(row for row in rows if row.get("event_type") == "equipment_control_change")

    assert event_row["physical_action_type"] == "equipment_panel_operation"
    assert event_row["quality_bucket"] == "low_quality"
    assert event_row["review_route"] == "human_review"
    assert "equipment_panel_operation_not_confirmed" in event_row["review_reason_codes"]


def test_liquid_transfer_candidate_requires_visual_liquid_confirmation(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    keyframe = session / "keyframes" / "micro" / "seg_000001_micro_001" / "peak.jpg"
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    keyframe.write_bytes(b"frame")
    _write_jsonl(
        session / "metadata" / "video_understanding.jsonl",
        [
            {
                "video_event_id": "seg_000001_micro_001:liquid_transfer_candidate:beaker",
                "segment_id": "episode_000001",
                "micro_segment_id": "seg_000001_micro_001",
                "event_type": "liquid_transfer_candidate",
                "primary_object": "beaker",
                "confidence": 0.62,
                "conclusion_status": "candidate",
                "anomaly_flags": [
                    "not_visual_liquid_flow_confirmed",
                    "visual_confirmation_limited",
                    "candidate_weak_bundle_rollup",
                ],
                "asset_refs": [{"asset_type": "keyframe", "rel": "peak_frame", "path": str(keyframe)}],
            }
        ],
    )

    build_yolo_material_candidates(session, archive_existing=False)
    rows = [
        json.loads(line)
        for line in (material_candidates_root(session) / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_row = next(row for row in rows if row.get("event_type") == "liquid_transfer_candidate")

    assert event_row["physical_action_type"] == "liquid_movement"
    assert event_row["quality_bucket"] == "low_quality"
    assert event_row["review_route"] == "human_review"
    assert "liquid_movement_not_visually_supported" in event_row["review_reason_codes"]


def test_approved_candidates_promote_only_selected_files(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setenv("KEY_ACTION_BUILD_LOCAL_EVIDENCE_PACKAGE", "1")
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
    monkeypatch.setattr(material_references, "_render_filtered_interaction_clip", lambda _source, _offset, _duration, target, _rows, _primary, _segment_start: Path(target).write_bytes(b"annotated-clip"))
    monkeypatch.setattr(material_references, "_extract_filtered_interaction_frame", lambda _source, _offset, target, _row, _primary, **_kwargs: Path(target).write_bytes(b"annotated-frame"))
    build_yolo_material_references(session, archive_existing=False)
    build_yolo_material_candidates(session, archive_existing=False)

    candidate_root = material_candidates_root(session)
    candidate_rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_id = candidate_rows[0]["candidate_group_id"]
    approval = approve_material_candidates(session, candidate_group_id=group_id, reviewer="tester")

    ref_root = material_references_root(session)
    promoted_rows = [
        json.loads(line)
        for line in (ref_root / f"{MATERIAL_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert approval["approved_count"] == 4
    approved_rows = [row for row in promoted_rows if row.get("review_status") == "accepted"]
    assert len(approved_rows) == 4
    assert {row["asset_kind"] for row in approved_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert all(Path(row["stored_file"]).exists() for row in approved_rows)
    assert (ref_root / EVIDENCE_PACKAGE_MANIFEST).exists()
    assert (ref_root / "key_material_references.jsonl").exists()
    assert (ref_root / PHYSICAL_CHANGE_LOG_JSONL).exists()
    assert (ref_root / TIME_ALIGNMENT_JSON).exists()
    assert approval["material_references_summary"]["local_openclaw_evidence_package"]["reference_count"] == len(promoted_rows)
    stored_files = [
        item
        for folder in (ref_root / KEYFRAME_DIR_NAME, ref_root / KEY_CLIP_DIR_NAME)
        for item in folder.iterdir()
        if item.is_file()
    ]
    assert len(stored_files) >= 4
    assert len({Path(row["stored_file"]).name for row in approved_rows}) == 4


def test_review_candidates_can_be_confirmed_and_renamed_without_publishing(tmp_path: Path) -> None:
    session = tmp_path / "experiment" / "key_action_index"
    candidate_root = material_candidates_root(session)
    frame = candidate_root / KEYFRAME_DIR_NAME / "candidate.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")
    rows = [
        {
            "candidate_id": "candidate_frame",
            "candidate_group_id": "candidate_group",
            "asset_kind": KEYFRAME_DIR_NAME,
            "material_type": KEYFRAME_DIR_NAME,
            "stored_file": str(frame),
            "stored_filename": frame.name,
            "candidate_status": "pending",
            "review_status": "pending",
            "display_title": "原始候选",
        }
    ]
    _write_jsonl(candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl", rows)

    confirmation = confirm_material_candidates(
        session,
        candidate_group_id="candidate_group",
        candidate_ids=["candidate_frame"],
        reviewer="tester",
        notes="人工确认候选",
    )
    confirmed_rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert confirmation["decision"] == "confirmed"
    assert confirmed_rows[0]["candidate_status"] == "confirmed"
    assert confirmed_rows[0]["review_status"] == "confirmed"
    assert confirmed_rows[0]["memory_eligible"] is False
    formal_index = material_references_root(session) / f"{MATERIAL_INDEX_BASENAME}.jsonl"
    formal_rows = [
        json.loads(line)
        for line in formal_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if formal_index.exists() else []
    assert formal_rows == []

    rename = rename_material_candidates(
        session,
        candidate_group_id="candidate_group",
        display_title="手部与称量纸操作",
        reviewer="tester",
    )
    renamed_rows = [
        json.loads(line)
        for line in (candidate_root / f"{MATERIAL_CANDIDATE_INDEX_BASENAME}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rename["decision"] == "rename"
    assert renamed_rows[0]["display_title"] == "手部与称量纸操作"
    assert renamed_rows[0]["human_display_title"] == "手部与称量纸操作"
    assert renamed_rows[0]["rename_scope"] == "display_only"
