"""Tests for material database consistency validation (P1).

Builds tiny synthetic material roots (JSONL + SQLite) and asserts the
bidirectional checks fire correctly. No GPU, no real data.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from key_action_indexer.material_db_consistency import (
    validate_material_database,
    write_consistency_report,
)


def _make_sqlite(db_path: Path, rows: list[dict]) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute(
        "create table materials (material_id text, official_status text, "
        "memory_eligible integer, keyframe_paths text, keyclip_paths text)"
    )
    for r in rows:
        con.execute(
            "insert into materials values (?,?,?,?,?)",
            (r["material_id"], r.get("official_status"),
             1 if r.get("memory_eligible") else 0, "[]", "[]"),
        )
    con.commit()
    con.close()


def _write_stream(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "material_stream.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _mat(mid, status="needs_review", **extra):
    base = {
        "material_id": mid,
        "official_status": status,
        "window_id": "w1",
        "experiment_window_id": "w1",
        "source_window_sync_index": "windows/w1/window_sync_index.csv",
    }
    base.update(extra)
    return base


def test_consistent_db_passes(tmp_path):
    rows = [_mat("m1"), _mat("m2")]
    _write_stream(tmp_path, rows)
    _make_sqlite(tmp_path / "material_index.sqlite", rows)
    res = validate_material_database(tmp_path)
    assert res.ok
    assert res.report["counts"]["material_stream_count"] == 2
    assert res.report["counts"]["sqlite_material_count"] == 2


def test_row_count_mismatch_fails(tmp_path):
    _write_stream(tmp_path, [_mat("m1"), _mat("m2")])
    _make_sqlite(tmp_path / "material_index.sqlite", [_mat("m1")])
    res = validate_material_database(tmp_path)
    assert not res.ok
    assert res.report["checks"]["row_count_mismatch"] is True
    assert res.report["checks"]["jsonl_only_ids"] == ["m2"]


def test_status_mismatch_fails(tmp_path):
    _write_stream(tmp_path, [_mat("m1", status="needs_review")])
    _make_sqlite(tmp_path / "material_index.sqlite", [_mat("m1", status="official")])
    res = validate_material_database(tmp_path)
    assert not res.ok
    mm = res.report["checks"]["status_mismatches"]
    assert mm and mm[0]["material_id"] == "m1"
    assert mm[0]["jsonl_status"] == "needs_review"
    assert mm[0]["sqlite_status"] == "official"


def test_memory_policy_violation_fails(tmp_path):
    # non-official but memory_eligible -> violation
    rows = [_mat("m1", status="needs_review", memory_eligible=True)]
    _write_stream(tmp_path, rows)
    _make_sqlite(tmp_path / "material_index.sqlite", rows)
    res = validate_material_database(tmp_path)
    assert not res.ok
    assert "m1" in res.report["checks"]["memory_policy_violations"]


def test_orphan_material_counted(tmp_path):
    rows = [_mat("m1", source_window_sync_index="", window_id="", experiment_window_id="")]
    _write_stream(tmp_path, rows)
    _make_sqlite(tmp_path / "material_index.sqlite", rows)
    res = validate_material_database(tmp_path)
    assert res.report["orphan_material_count"] == 1
    assert "m1" in res.report["orphan_material_ids"]


def test_missing_asset_detected(tmp_path):
    rows = [_mat("m1", first_keyframe=str(tmp_path / "nope.jpg"))]
    _write_stream(tmp_path, rows)
    _make_sqlite(tmp_path / "material_index.sqlite", rows)
    res = validate_material_database(tmp_path)
    assert not res.ok
    assert res.report["checks"]["missing_assets"][0]["material_id"] == "m1"


def test_missing_asset_with_reason_is_ok(tmp_path):
    rows = [_mat("m1", first_keyframe=str(tmp_path / "nope.jpg"),
                 missing_reason="non_real_or_missing_asset")]
    _write_stream(tmp_path, rows)
    _make_sqlite(tmp_path / "material_index.sqlite", rows)
    res = validate_material_database(tmp_path)
    # missing file is acceptable when a missing_reason is recorded
    assert res.report["checks"]["missing_assets"] == []


def test_write_report_rewrites_from_live_data(tmp_path):
    rows = [_mat("m1"), _mat("m2")]
    _write_stream(tmp_path, rows)
    _make_sqlite(tmp_path / "material_index.sqlite", rows)
    # plant a stale report claiming a different state
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "database_consistency_validation_report.json").write_text(
        json.dumps({"status": "fail", "stale": True}), encoding="utf-8")

    res = write_consistency_report(tmp_path)
    assert res.ok
    on_disk = json.loads(
        (reports / "database_consistency_validation_report.json").read_text(encoding="utf-8"))
    assert on_disk["status"] == "pass"
    assert "stale" not in on_disk
    assert on_disk["schema_version"].startswith("database_consistency_validation_report")
