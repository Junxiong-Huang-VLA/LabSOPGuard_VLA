from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.material_search import load_material_catalog, search_material_assets


def _catalog_rows() -> list[dict]:
    return [
        {
            "asset_id": "clip_pipette",
            "asset_type": "clip",
            "path": "clips/pipette_001.mp4",
            "source_type": "micro",
            "source_id": "micro_001",
            "global_start_time": "2026-04-29T17:25:10+08:00",
            "global_end_time": "2026-04-29T17:25:15+08:00",
            "objects": ["pipette", "tube"],
            "actions": ["pipetting"],
            "state_tags": ["contact_started", "liquid_transfer"],
            "search_text": "实验人员使用移液枪向离心管加样 pipette transfer",
            "evidence_level": "high",
            "quality": {"exists": True},
        },
        {
            "asset_id": "frame_balance",
            "asset_type": "keyframe",
            "path": "keyframes/balance_peak.jpg",
            "source_type": "micro",
            "source_id": "micro_002",
            "global_start_time": "2026-04-29T17:25:30+08:00",
            "global_end_time": "2026-04-29T17:25:31+08:00",
            "objects": ["balance", "sample"],
            "actions": ["weighing"],
            "state_tags": ["peak_interaction"],
            "search_text": "天平称量样品 balance weighing",
            "evidence_level": "visual",
            "quality": {"exists": True},
        },
        {
            "asset_id": "clip_bottle",
            "asset_type": "clip",
            "path": "clips/bottle_release.mp4",
            "source_type": "segment",
            "source_id": "seg_003",
            "global_start_time": "2026-04-29T17:26:00+08:00",
            "global_end_time": "2026-04-29T17:26:04+08:00",
            "objects": ["bottle"],
            "actions": ["opening"],
            "state_tags": ["contact_released"],
            "search_text": "打开试剂瓶 bottle cap",
            "evidence_level": "medium",
            "quality": {"exists": True},
        },
        {
            "asset_id": "clip_late_pipette",
            "asset_type": "clip",
            "path": "clips/pipette_late.mp4",
            "source_type": "micro",
            "source_id": "micro_004",
            "global_start_time": "2026-04-29T17:27:00+08:00",
            "global_end_time": "2026-04-29T17:27:05+08:00",
            "objects": ["pipette"],
            "actions": ["pipetting"],
            "state_tags": ["object_contact"],
            "search_text": "pipette rinse after transfer",
            "evidence_level": "low",
            "quality": {"exists": True},
        },
    ]


def _write_catalog(tmp_path: Path) -> tuple[Path, Path]:
    session_dir = tmp_path / "session"
    metadata_dir = session_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    catalog_path = metadata_dir / "material_asset_catalog.jsonl"
    catalog_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in _catalog_rows()) + "\n",
        encoding="utf-8",
    )
    return session_dir, catalog_path


def test_load_material_catalog_from_session_dir(tmp_path: Path) -> None:
    session_dir, _ = _write_catalog(tmp_path)

    rows = load_material_catalog(session_dir)

    assert len(rows) == 4
    assert rows[0]["asset_id"] == "clip_pipette"


def test_keyword_search_matches_chinese_substring_and_adds_reasons(tmp_path: Path) -> None:
    _, catalog_path = _write_catalog(tmp_path)

    results = search_material_assets(catalog_path, query="移液枪")

    assert [row["asset_id"] for row in results] == ["clip_pipette"]
    assert results[0]["score"] > 0
    assert any("query:search_text" in reason for reason in results[0]["match_reasons"])


def test_object_and_asset_type_filters(tmp_path: Path) -> None:
    session_dir, _ = _write_catalog(tmp_path)

    results = search_material_assets(session_dir, asset_type="keyframe", objects="balance")

    assert [row["asset_id"] for row in results] == ["frame_balance"]
    assert "objects=balance" in results[0]["match_reasons"]


def test_action_filter_returns_matching_assets_in_time_order(tmp_path: Path) -> None:
    session_dir, _ = _write_catalog(tmp_path)

    results = search_material_assets(session_dir, actions=["pipetting"])

    assert [row["asset_id"] for row in results] == ["clip_pipette", "clip_late_pipette"]


def test_state_tag_filter(tmp_path: Path) -> None:
    session_dir, _ = _write_catalog(tmp_path)

    results = search_material_assets(session_dir, state_tags="contact_released")

    assert [row["asset_id"] for row in results] == ["clip_bottle"]


def test_time_range_filter_uses_global_time_overlap(tmp_path: Path) -> None:
    session_dir, _ = _write_catalog(tmp_path)

    results = search_material_assets(
        session_dir,
        start_time="2026-04-29T17:25:12+08:00",
        end_time="2026-04-29T17:25:32+08:00",
    )

    assert [row["asset_id"] for row in results] == ["clip_pipette", "frame_balance"]
    assert all("time_range_overlap" in row["match_reasons"] for row in results)


def test_limit_truncates_sorted_results(tmp_path: Path) -> None:
    session_dir, _ = _write_catalog(tmp_path)

    results = search_material_assets(session_dir, query="pipette", limit=1)

    assert [row["asset_id"] for row in results] == ["clip_pipette"]
