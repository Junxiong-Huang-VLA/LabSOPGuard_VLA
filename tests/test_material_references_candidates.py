from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer import material_references
from key_action_indexer.material_references import (
    KEYFRAME_DIR_NAME,
    KEY_CLIP_DIR_NAME,
    MATERIAL_CANDIDATE_INDEX_BASENAME,
    MATERIAL_INDEX_BASENAME,
    approve_material_candidates,
    build_yolo_material_candidates,
    build_yolo_material_references,
    material_candidates_root,
    material_references_root,
)


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
            "interaction_score": 0.75,
            "primary_object": "balance",
            "object_box": [160, 140, 320, 300],
            "hand_box": [120, 120, 190, 230],
            "hand_label": "gloved_hand",
            "detections": [
                {"label": "gloved_hand", "confidence": 0.8, "bbox": [120, 120, 190, 230]},
                {"label": "balance", "confidence": 0.8, "bbox": [160, 140, 320, 300]},
            ],
        }
        for index in range(3)
    ]


def _session(tmp_path: Path) -> Path:
    session = tmp_path / "experiment" / "key_action_index"
    first_clip = session / "clips" / "first.mp4"
    third_clip = session / "clips" / "third.mp4"
    first_clip.parent.mkdir(parents=True, exist_ok=True)
    first_clip.write_bytes(b"first")
    third_clip.write_bytes(b"third")
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
    return session


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
    assert summary["candidate_count"] == 8
    assert all(row["candidate_status"] == "pending" for row in rows)
    assert all(row["review_required"] is True for row in rows)
    assert all(row["pipeline_stage"] == "frontend_review_gate" for row in rows)
    assert {row["asset_kind"] for row in rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert summary["pipeline_summary"]["groups_waiting_frontend_review"] == 2


def test_approved_candidates_promote_only_selected_files(tmp_path: Path, monkeypatch) -> None:
    session = _session(tmp_path)
    monkeypatch.setattr(material_references, "_ffmpeg_available", lambda _path: True)
    monkeypatch.setattr(material_references, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"material"))
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
    assert approval["approved_count"] == 2
    assert len(promoted_rows) == 2
    assert {row["asset_kind"] for row in promoted_rows} == {KEYFRAME_DIR_NAME, KEY_CLIP_DIR_NAME}
    assert all(row["review_status"] == "accepted" for row in promoted_rows)
    assert all(Path(row["stored_file"]).exists() for row in promoted_rows)
    stored_files = [
        item
        for folder in (ref_root / KEYFRAME_DIR_NAME, ref_root / KEY_CLIP_DIR_NAME)
        for item in folder.iterdir()
        if item.is_file()
    ]
    assert len(stored_files) == 2
