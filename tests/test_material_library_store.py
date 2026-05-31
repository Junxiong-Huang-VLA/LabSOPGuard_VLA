import json
import sqlite3
from pathlib import Path

import pytest

from key_action_indexer.material_library_store import (
    global_material_library_db_path,
    query_material_library,
    resolve_material_file,
    sync_material_library,
    sync_material_library_package,
)
from key_action_indexer.material_reference_index import KEYFRAME_KIND, KEY_CLIP_KIND, MATERIAL_INDEX_JSONL


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_sync_material_library_builds_global_query_index(tmp_path):
    library_root = tmp_path / "LabMaterialLibrary"
    package_root = library_root / "material_references" / "experiment_package"
    keyframe_dir = package_root / KEYFRAME_KIND
    clip_dir = package_root / KEY_CLIP_KIND
    keyframe_dir.mkdir(parents=True)
    clip_dir.mkdir(parents=True)
    frame_path = keyframe_dir / "hand_paper_first.jpg"
    clip_path = clip_dir / "hand_paper_first.mp4"
    frame_path.write_bytes(b"fake image")
    clip_path.write_bytes(b"fake video")
    (package_root / "manifest.json").write_text(
        json.dumps({"experiment_id": "exp_001", "experiment_title": "dual view package"}, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "experiment_id": "exp_001",
                "asset_kind": KEYFRAME_KIND,
                "material_type": KEYFRAME_KIND,
                "action_name": "hand paper operation",
                "display_name": "hand paper frame",
                "stored_file": f"{KEYFRAME_KIND}/{frame_path.name}",
                "file_name": frame_path.name,
                "view": "first_person",
                "time_start": 12.0,
                "time_end": 13.0,
                "primary_object": "paper",
                "secondary_objects": ["balance"],
            },
            {
                "experiment_id": "exp_001",
                "asset_kind": KEY_CLIP_KIND,
                "material_type": KEY_CLIP_KIND,
                "action_name": "hand paper operation",
                "display_name": "hand paper clip",
                "stored_file": f"{KEY_CLIP_KIND}/{clip_path.name}",
                "file_name": clip_path.name,
                "view": "first_person",
                "time_start": 12.0,
                "time_end": 15.0,
                "primary_object": "paper",
                "secondary_objects": ["balance"],
            },
        ],
    )

    summary = sync_material_library(library_root, rebuild=True)
    hits = query_material_library(library_root=library_root, text="paper", limit=10)
    multi_token_hits = query_material_library(library_root=library_root, text="hand paper", limit=10)

    assert summary["indexed_count"] == 2
    assert summary["asset_type_counts"] == {"keyframe": 1, "video_clip": 1}
    assert {hit["asset_type"] for hit in hits} == {"keyframe", "video_clip"}
    assert {hit["asset_type"] for hit in multi_token_hits} == {"keyframe", "video_clip"}
    assert all(hit["package_name"] == "experiment_package" for hit in hits)
    assert all(hit["experiment_id"] == "exp_001" for hit in hits)
    assert resolve_material_file(hits[0]["material_id"], library_root=library_root).exists()

    conn = sqlite3.connect(str(global_material_library_db_path(library_root)))
    try:
        assert conn.execute("SELECT COUNT(*) FROM material_refs").fetchone()[0] == 2
    finally:
        conn.close()


def test_query_material_library_filters_by_asset_and_view(tmp_path):
    library_root = tmp_path / "LabMaterialLibrary"
    package_root = library_root / "material_references" / "experiment_package"
    keyframe_dir = package_root / KEYFRAME_KIND
    keyframe_dir.mkdir(parents=True)
    first_frame = keyframe_dir / "first.jpg"
    third_frame = keyframe_dir / "third.jpg"
    first_frame.write_bytes(b"first")
    third_frame.write_bytes(b"third")
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{first_frame.name}",
                "file_name": first_frame.name,
                "view": "first_person",
                "action_name": "bottle move",
                "primary_object": "bottle",
            },
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{third_frame.name}",
                "file_name": third_frame.name,
                "view": "third_person",
                "action_name": "balance panel operation",
                "primary_object": "balance",
            },
        ],
    )
    sync_material_library(library_root, rebuild=True)

    hits = query_material_library(
        library_root=library_root,
        asset_type="keyframe",
        view="third_person",
        action="balance",
        limit=5,
    )

    assert len(hits) == 1
    assert hits[0]["view"] == "third_person"
    assert hits[0]["primary_object"] == "balance"


def test_query_material_library_filters_by_package_action_object_date_and_keeps_session(tmp_path):
    library_root = tmp_path / "LabMaterialLibrary"
    package_root = library_root / "material_references" / "package_20260525"
    keyframe_dir = package_root / KEYFRAME_KIND
    clip_dir = package_root / KEY_CLIP_KIND
    keyframe_dir.mkdir(parents=True)
    clip_dir.mkdir(parents=True)
    frame_path = keyframe_dir / "balance_paper.jpg"
    clip_path = clip_dir / "balance_paper.mp4"
    frame_path.write_bytes(b"frame")
    clip_path.write_bytes(b"clip")
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "experiment_id": "exp_keep",
                "session_id": "session_keep",
                "date": "2026-05-25",
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{frame_path.name}",
                "file_name": frame_path.name,
                "action_name": "weigh paper on balance",
                "primary_object": "paper",
                "secondary_objects": ["balance"],
            },
            {
                "experiment_id": "exp_keep",
                "session_id": "session_keep",
                "date": "2026-05-25",
                "asset_kind": KEY_CLIP_KIND,
                "stored_file": f"{KEY_CLIP_KIND}/{clip_path.name}",
                "file_name": clip_path.name,
                "action_name": "weigh paper on balance",
                "primary_object": "paper",
                "secondary_objects": ["balance"],
            },
        ],
    )
    sync_material_library(library_root, rebuild=True)

    hits = query_material_library(
        library_root=library_root,
        package_name="package_20260525",
        action="weigh",
        primary_object="paper",
        date="2026-05-25",
        session_id="session_keep",
        limit=10,
    )

    assert {hit["asset_type"] for hit in hits} == {"keyframe", "video_clip"}
    assert {hit["stored_file"] for hit in hits} == {
        f"{KEYFRAME_KIND}/{frame_path.name}",
        f"{KEY_CLIP_KIND}/{clip_path.name}",
    }
    assert all(hit["experiment_id"] == "exp_keep" for hit in hits)
    assert all(hit["session_id"] == "session_keep" for hit in hits)
    assert all(hit["date"] == "2026-05-25" for hit in hits)


def test_sync_material_library_package_replaces_one_package(tmp_path):
    library_root = tmp_path / "LabMaterialLibrary"
    package_root = library_root / "material_references" / "experiment_package"
    keyframe_dir = package_root / KEYFRAME_KIND
    keyframe_dir.mkdir(parents=True)
    first_frame = keyframe_dir / "first.jpg"
    second_frame = keyframe_dir / "second.jpg"
    first_frame.write_bytes(b"first")
    second_frame.write_bytes(b"second")
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{first_frame.name}",
                "file_name": first_frame.name,
                "view": "first_person",
                "action_name": "first action",
                "primary_object": "paper",
            }
        ],
    )
    first = sync_material_library_package(package_root, library_root=library_root)
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{second_frame.name}",
                "file_name": second_frame.name,
                "view": "third_person",
                "action_name": "second action",
                "primary_object": "balance",
            }
        ],
    )
    (package_root / "key_material_references.jsonl").unlink(missing_ok=True)
    second = sync_material_library_package(package_root, library_root=library_root)
    hits = query_material_library(library_root=library_root, limit=10)

    assert first["indexed_count"] == 1
    assert second["indexed_count"] == 1
    assert len(hits) == 1
    assert hits[0]["file_name"] == second_frame.name


def test_sync_material_library_skips_placeholder_package_rows(tmp_path):
    library_root = tmp_path / "LabMaterialLibrary"
    package_root = library_root / "material_references" / "experiment_package"
    keyframe_dir = package_root / KEYFRAME_KIND
    keyframe_dir.mkdir(parents=True)
    real_frame = keyframe_dir / "real.jpg"
    placeholder_frame = keyframe_dir / "poster_placeholder.jpg"
    real_frame.write_bytes(b"real")
    placeholder_frame.write_bytes(b"DRY RUN PLACEHOLDER")
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{real_frame.name}",
                "file_name": real_frame.name,
                "view": "third_person",
                "action_name": "real action",
                "primary_object": "paper",
                "source_real": True,
                "placeholder": False,
            },
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{placeholder_frame.name}",
                "file_name": placeholder_frame.name,
                "view": "first_person",
                "action_name": "placeholder action",
                "primary_object": "paper",
                "source_real": False,
                "placeholder": True,
            },
        ],
    )

    summary = sync_material_library(library_root, rebuild=True)
    hits = query_material_library(library_root=library_root, limit=10)

    assert summary["indexed_count"] == 1
    assert len(hits) == 1
    assert hits[0]["file_name"] == real_frame.name
    assert summary["skipped"]


def test_sync_material_library_skips_decodable_blank_screen_rows(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    library_root = tmp_path / "LabMaterialLibrary"
    package_root = library_root / "material_references" / "experiment_package"
    keyframe_dir = package_root / KEYFRAME_KIND
    keyframe_dir.mkdir(parents=True)
    real_frame = keyframe_dir / "real.jpg"
    blank_frame = keyframe_dir / "black_screen.jpg"
    real = Image.new("RGB", (32, 32), (80, 110, 150))
    for x in range(16, 32):
        for y in range(16, 32):
            real.putpixel((x, y), (220, 70, 60))
    real.save(real_frame)
    Image.new("RGB", (32, 32), (0, 0, 0)).save(blank_frame)
    _write_jsonl(
        package_root / MATERIAL_INDEX_JSONL,
        [
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{real_frame.name}",
                "file_name": real_frame.name,
                "view": "third_person",
                "action_name": "real action",
                "primary_object": "paper",
                "source_real": True,
                "placeholder": False,
                "evidence_group_id": "evidence_group_real",
            },
            {
                "asset_kind": KEYFRAME_KIND,
                "stored_file": f"{KEYFRAME_KIND}/{blank_frame.name}",
                "file_name": blank_frame.name,
                "view": "first_person",
                "action_name": "blank action",
                "primary_object": "paper",
                "source_real": True,
                "placeholder": False,
                "evidence_group_id": "evidence_group_blank",
            },
        ],
    )

    summary = sync_material_library(library_root, rebuild=True)
    hits = query_material_library(library_root=library_root, limit=10)

    assert summary["indexed_count"] == 1
    assert hits[0]["file_name"] == real_frame.name
    assert hits[0]["evidence_group_id"] == "evidence_group_real"
