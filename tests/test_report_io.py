"""Characterization test for the shared report-writer helper (P4).

Locks the exact serialization so the consolidation is provably behavior-
preserving: same bytes as the previous private _write_json helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.report_io import write_json_report


def test_write_json_report_byte_identical_to_legacy(tmp_path):
    payload = {"b": 2, "a": 1, "zh": "称量实验", "nested": {"x": [1, 2, 3]}}

    # legacy inline behavior that the modules used
    legacy = json.dumps(payload, ensure_ascii=False, indent=2)

    out = write_json_report(tmp_path / "sub" / "r.json", payload)
    assert out.exists()
    assert out.read_text(encoding="utf-8") == legacy


def test_write_json_report_creates_parent_dirs(tmp_path):
    out = write_json_report(tmp_path / "a" / "b" / "c.json", {"ok": True})
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


def test_write_json_report_preserves_unicode(tmp_path):
    out = write_json_report(tmp_path / "u.json", {"label": "手部与试剂瓶接触"})
    text = out.read_text(encoding="utf-8")
    assert "手部与试剂瓶接触" in text  # not \uXXXX escaped
