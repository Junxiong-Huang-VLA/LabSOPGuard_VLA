from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer import query_evidence_package


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_query_evidence_package_return_position_judgement_is_portable(tmp_path: Path) -> None:
    package_dir = tmp_path / "solid_weighing_package"
    (package_dir / "关键片段").mkdir(parents=True)
    (package_dir / "关键片段" / "return_bottle.mp4").write_bytes(b"fake clip")
    _write_json(
        package_dir / "evidence_package_manifest.json",
        {
            "schema_version": "evidence_package_manifest.v1",
            "package_id": "pkg_solid_001",
            "path_mode": "relative_to_package_root",
            "portable": True,
            "entrypoints": {
                "key_material_references_jsonl": "key_material_references.jsonl",
                "physical_change_log_jsonl": "physical_change_log.jsonl",
                "time_alignment_json": "time_alignment.json",
            },
        },
    )
    _write_json(
        package_dir / "time_alignment.json",
        {
            "schema_version": "time_alignment.v1",
            "session_start_at": "2026-05-12T14:00:00+08:00",
            "video_streams": [{"duration_sec": 300.0, "offset_sec": 0.0, "clock_scale": 1.0}],
            "message_alignment_policy": {"default_window_before_sec": 30.0, "default_window_after_sec": 30.0},
        },
    )
    _write_jsonl(
        package_dir / "key_material_references.jsonl",
        [
            {
                "schema_version": "key_material_reference.v1",
                "material_id": "mat_return_bottle",
                "event_type": "object_move",
                "asset_type": "event_clip",
                "step_name": "试剂瓶归位",
                "start_sec": 116.0,
                "end_sec": 122.0,
                "objects": ["gloved_hand", "reagent_bottle"],
                "actions": ["object_move", "hand_object_interaction"],
                "before_state": {"zone": "A", "centroid": [100, 100]},
                "after_state": {"zone": "B", "centroid": [260, 100]},
                "formal_clip_path": "关键片段/return_bottle.mp4",
                "confidence": 0.82,
                "searchable_text": "试剂瓶 归位 reagent_bottle object_move",
            }
        ],
    )
    _write_jsonl(
        package_dir / "physical_change_log.jsonl",
        [
            {
                "schema_version": "physical_change.v1",
                "change_id": "chg_return_bottle",
                "event_type": "object_move",
                "start_sec": 116.0,
                "end_sec": 122.0,
                "before": {"zone": "A", "centroid": [100, 100]},
                "after": {"zone": "B", "centroid": [260, 100]},
                "evidence_material_ids": ["mat_return_bottle"],
            }
        ],
    )

    result = query_evidence_package(
        package_dir,
        query_text="检查试剂瓶归位是否正确",
        message_sent_at="2026-05-12T14:02:00+08:00",
    )

    assert result["time_context"]["message_video_time_sec"] == 120.0
    assert result["judgement"]["status"] == "incorrect"
    assert result["evidence_bundles"][0]["clip"]["relative_path"] == "关键片段/return_bottle.mp4"
    assert result["evidence_bundles"][0]["clip"]["package_uri"] == "package://pkg_solid_001/关键片段/return_bottle.mp4"
