import json
import sqlite3
from pathlib import Path

import pytest

from key_action_indexer.material_reference_index import (
    build_key_material_reference_index,
    load_material_reference_rows,
    query_key_material_reference_index,
)


KEYFRAME_KIND = "\u5173\u952e\u5e27"
KEY_CLIP_KIND = "\u5173\u952e\u7247\u6bb5"
REPORT_KIND = "\u4e13\u4e1a\u62a5\u544a"
MATERIAL_INDEX_JSON = "\u7d20\u6750\u7d22\u5f15.json"
MATERIAL_INDEX_JSONL = "\u7d20\u6750\u7d22\u5f15.jsonl"


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_build_key_material_reference_index_from_jsonl(tmp_path):
    root = tmp_path / "formal_materials"
    keyframe_dir = root / KEYFRAME_KIND
    clip_dir = root / KEY_CLIP_KIND
    keyframe_dir.mkdir(parents=True)
    clip_dir.mkdir(parents=True)
    keyframe = keyframe_dir / "hand_paper_20260508.jpg"
    clip = clip_dir / "hand_paper_20260508.mp4"
    keyframe.write_bytes(b"fake image")
    clip.write_bytes(b"fake video")
    rows = [
        {
            "asset_kind": KEYFRAME_KIND,
            "material_type": KEYFRAME_KIND,
            "session_id": "session_20260508",
            "date": "2026-05-08",
            "action_name": "hand paper operation",
            "stored_file": str(keyframe),
            "file_name": keyframe.name,
            "segment_id": "seg_000001",
            "micro_segment_id": "seg_000001_micro_001",
            "primary_object": "paper",
            "secondary_objects": ["balance"],
            "secondary_actions": ["hand-paper+balance"],
            "review_status": "accepted",
            "quality_score": 0.9,
            "window_audit": {
                "interaction_frame_count": 3,
                "target_object_support": {"object": "paper", "interaction_frame_count": 3},
                "secondary_object_support": [{"object": "balance", "interaction_frame_count": 3}],
                "uncertainty_reasons": [],
            },
        },
        {
            "asset_kind": KEY_CLIP_KIND,
            "material_type": KEY_CLIP_KIND,
            "session_id": "session_20260508",
            "date": "2026-05-08",
            "action_name": "hand paper operation",
            "stored_file": str(clip),
            "file_name": clip.name,
            "segment_id": "seg_000001",
            "micro_segment_id": "seg_000001_micro_001",
            "primary_object": "paper",
            "secondary_objects": ["balance"],
            "secondary_actions": ["hand-paper+balance"],
            "review_status": "accepted",
        },
    ]
    _write_jsonl(root / MATERIAL_INDEX_JSONL, rows)

    summary = build_key_material_reference_index(root)

    assert summary["indexed_count"] == 2
    assert summary["asset_type_counts"] == {"keyframe": 1, "video_clip": 1}
    assert (root / "key_material_references.sqlite").exists()
    assert (root / "key_material_references.jsonl").exists()
    references = [
        json.loads(line)
        for line in (root / "key_material_references.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert not Path(references[0]["stored_file"]).is_absolute()
    assert references[0]["stored_file"].replace("\\", "/") == f"{KEYFRAME_KIND}/hand_paper_20260508.jpg"
    assert references[0]["path_mode"] == "relative_to_material_root"
    assert references[0]["session_id"] == "session_20260508"
    assert references[0]["date"] == "2026-05-08"
    assert references[0]["secondary_objects"] == ["balance"]
    assert "hand-paper+balance" in references[0]["secondary_actions"]
    assert "hand-paper+balance" in references[0]["searchable_text"]
    hits = query_key_material_reference_index(root / "key_material_references.sqlite", text="paper", limit=5)
    assert {hit["asset_type"] for hit in hits} == {"keyframe", "video_clip"}
    assert all(not Path(hit["stored_file"]).is_absolute() for hit in hits)
    compound_hits = query_key_material_reference_index(
        root / "key_material_references.sqlite",
        text="hand-paper+balance",
        action="hand-paper+balance",
        primary_object="balance",
        limit=5,
    )
    assert {hit["asset_type"] for hit in compound_hits} == {"keyframe", "video_clip"}
    assert all("hand-paper+balance" in hit["secondary_actions"] for hit in compound_hits)
    dated_hits = query_key_material_reference_index(
        root / "key_material_references.sqlite",
        action="hand paper",
        session_id="session_20260508",
        date="2026-05-08",
        limit=5,
    )
    assert {hit["asset_type"] for hit in dated_hits} == {"keyframe", "video_clip"}

    conn = sqlite3.connect(str(root / "key_material_references.sqlite"))
    try:
        count = conn.execute("SELECT COUNT(*) FROM key_material_refs").fetchone()[0]
    finally:
        conn.close()
    assert count == 2


def test_load_material_reference_rows_falls_back_to_json_records(tmp_path):
    root = tmp_path / "formal_materials"
    root.mkdir()
    payload = {
        "records": [
            {
                "asset_kind": KEYFRAME_KIND,
                "file_name": "frame.jpg",
                "primary_object": "beaker",
            }
        ]
    }
    (root / MATERIAL_INDEX_JSON).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    rows = load_material_reference_rows(root)

    assert len(rows) == 1
    assert rows[0]["primary_object"] == "beaker"


def test_reports_are_skipped_unless_requested(tmp_path):
    root = tmp_path / "formal_materials"
    report_dir = root / REPORT_KIND
    report_dir.mkdir(parents=True)
    report = report_dir / "report.pdf"
    report.write_bytes(b"%PDF")
    _write_jsonl(
        root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": REPORT_KIND,
                "material_type": REPORT_KIND,
                "stored_file": str(report),
                "file_name": report.name,
                "action_name": "formal report",
            }
        ],
    )

    skipped = build_key_material_reference_index(root)
    included = build_key_material_reference_index(root, include_reports=True, sqlite_path=root / "with_reports.sqlite")

    assert skipped["indexed_count"] == 0
    assert skipped["skipped"][0]["reason"] == "report_not_indexed"
    assert included["indexed_count"] == 1
    assert included["asset_type_counts"] == {"report": 1}


def test_placeholder_material_rows_are_not_indexed(tmp_path):
    root = tmp_path / "formal_materials"
    keyframe_dir = root / KEYFRAME_KIND
    keyframe_dir.mkdir(parents=True)
    real_frame = keyframe_dir / "real_hand_paper.jpg"
    placeholder_frame = keyframe_dir / "placeholder_hand_paper.jpg"
    real_frame.write_bytes(b"real frame bytes")
    placeholder_frame.write_bytes(b"DRY RUN PLACEHOLDER")
    _write_jsonl(
        root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "material_type": KEYFRAME_KIND,
                "stored_file": str(real_frame),
                "file_name": real_frame.name,
                "action_name": "hand paper operation",
                "source_real": True,
                "placeholder": False,
            },
            {
                "asset_kind": KEYFRAME_KIND,
                "material_type": KEYFRAME_KIND,
                "stored_file": str(placeholder_frame),
                "file_name": placeholder_frame.name,
                "action_name": "hand paper placeholder",
                "source_real": False,
                "placeholder": True,
            },
        ],
    )

    summary = build_key_material_reference_index(root)
    references = [
        json.loads(line)
        for line in (root / "key_material_references.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["indexed_count"] == 1
    assert references[0]["file_name"] == real_frame.name
    assert references[0]["source_real"] is True
    assert all(item["reason"] != "report_not_indexed" for item in summary["skipped"])


def test_decodable_blank_screen_material_rows_are_not_indexed(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    root = tmp_path / "formal_materials"
    keyframe_dir = root / KEYFRAME_KIND
    keyframe_dir.mkdir(parents=True)
    real_frame = keyframe_dir / "real_hand_paper.jpg"
    blank_frame = keyframe_dir / "white_screen.jpg"
    real = Image.new("RGB", (32, 32), (60, 100, 150))
    for x in range(16, 32):
        for y in range(16, 32):
            real.putpixel((x, y), (230, 80, 55))
    real.save(real_frame)
    Image.new("RGB", (32, 32), (255, 255, 255)).save(blank_frame)
    _write_jsonl(
        root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "material_type": KEYFRAME_KIND,
                "stored_file": str(real_frame),
                "file_name": real_frame.name,
                "action_name": "hand paper operation",
                "source_real": True,
                "placeholder": False,
                "evidence_group_id": "evidence_group_real",
            },
            {
                "asset_kind": KEYFRAME_KIND,
                "material_type": KEYFRAME_KIND,
                "stored_file": str(blank_frame),
                "file_name": blank_frame.name,
                "action_name": "hand paper blank",
                "source_real": True,
                "placeholder": False,
                "evidence_group_id": "evidence_group_blank",
            },
        ],
    )

    summary = build_key_material_reference_index(root)
    references = [
        json.loads(line)
        for line in (root / "key_material_references.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert summary["indexed_count"] == 1
    assert references[0]["file_name"] == real_frame.name
    assert references[0]["evidence_group_id"] == "evidence_group_real"
