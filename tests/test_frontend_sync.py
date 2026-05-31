from __future__ import annotations

import inspect
import json
from pathlib import Path

from key_action_indexer.frontend_sync import (
    KEY_MATERIAL_REFERENCES_JSONL,
    sync_frontend_artifacts,
    validate_frontend_artifact_sync,
)
from key_action_indexer.evidence_package import EVIDENCE_PACKAGE_MANIFEST, PHYSICAL_CHANGE_LOG_JSONL, TIME_ALIGNMENT_JSON
from key_action_indexer.schemas import read_jsonl, write_jsonl
from key_action_indexer.yolo_analysis import (
    annotate_clip_with_yolo,
    run_yolo_on_experiment_focus_clips,
    run_yolo_on_keyframes,
    run_yolo_on_segment_clips,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_frontend_sync_makes_material_rows_portable_and_approved(tmp_path: Path) -> None:
    source_session = tmp_path / "source_session"
    source_key = source_session / "key_action_index"
    source_material = source_session / "material_references"
    target_exp = tmp_path / "frontend_exp" / "exp-001"
    target_raw = target_exp / "raw"

    (source_key / "metadata").mkdir(parents=True)
    (source_key / "cv_outputs").mkdir(parents=True)
    (source_material / "clips").mkdir(parents=True)
    target_raw.mkdir(parents=True)
    (source_material / "clips" / "item.mp4").write_bytes(b"clip")
    (target_raw / "third_camera_00.mp4").write_bytes(b"third")
    (target_raw / "first_camera_01.mp4").write_bytes(b"first")

    _write_json(
        source_key / "manifest.json",
        {
            "session_id": "source-session",
            "session_start_time": "2026-05-08T15:36:48+08:00",
            "videos": {
                "third_person": {"path": str(source_session / "old_third.mp4"), "start_time": "2026-05-08T15:36:48+08:00"},
                "first_person": {"path": str(source_session / "old_first.mp4"), "start_time": "2026-05-08T15:36:48+08:00"},
            },
            "output_dir": str(source_key),
        },
    )
    write_jsonl(source_key / "metadata" / "key_action_segments.jsonl", [{"segment_id": "seg_001"}])
    write_jsonl(source_key / "metadata" / "micro_segments.jsonl", [{"micro_segment_id": "micro_001", "parent_segment_id": "seg_001"}])
    write_jsonl(source_key / "metadata" / "vector_metadata.jsonl", [{"id": "seg_001"}])
    write_jsonl(source_key / "cv_outputs" / "yolo_frame_rows.jsonl", [{"hand_object_interactions": [{"score": 1.0}]}])

    row = {
        "material_id": "mat_001",
        "asset_type": "video_clip",
        "asset_kind": "key_clip",
        "stored_file": str(source_material / "clips" / "item.mp4"),
        "source_file": str(source_key / "clips" / "episodes" / "seg_001" / "third_person.mp4"),
        "payload_json": json.dumps(
            {
                "stored_file": str(source_material / "clips" / "item.mp4"),
                "source_file": str(source_key / "clips" / "episodes" / "seg_001" / "third_person.mp4"),
            },
            ensure_ascii=False,
        ),
    }
    write_jsonl(source_material / KEY_MATERIAL_REFERENCES_JSONL, [row])

    summary = sync_frontend_artifacts(
        target_experiment_dir=target_exp,
        source_session_dir=source_session,
        experiment_id="exp-001",
        experiment_title="sync test",
        archive_existing=False,
        hardlink_media=False,
        refresh_focus=False,
        require_yolo_overlay=False,
    )

    target_manifest = json.loads((target_exp / "key_action_index" / "manifest.json").read_text(encoding="utf-8"))
    synced_rows = read_jsonl(target_exp / "material_references" / KEY_MATERIAL_REFERENCES_JSONL)
    synced_row = synced_rows[0]
    payload = json.loads(synced_row["payload_json"])

    assert summary["material_counts"]["approved_count"] == 1
    assert target_manifest["output_dir"] == str(target_exp / "key_action_index")
    assert target_manifest["videos"]["third_person"]["path"] == str(target_raw / "third_camera_00.mp4")
    assert synced_row["candidate_status"] == "approved"
    assert synced_row["formal_material_reference"] is True
    assert synced_row["stored_file"] == "clips/item.mp4"
    assert synced_row["path_mode"] == "relative_to_material_root"
    assert synced_row["source_file"] == "third_person.mp4"
    assert payload["stored_file"] == "clips/item.mp4"
    assert payload["source_file"] == "third_person.mp4"
    assert summary["evidence_package_summary"]["reference_count"] == 1
    assert (target_exp / "material_references" / EVIDENCE_PACKAGE_MANIFEST).exists()
    assert (target_exp / "material_references" / PHYSICAL_CHANGE_LOG_JSONL).exists()
    assert (target_exp / "material_references" / TIME_ALIGNMENT_JSON).exists()


def test_frontend_sync_accepts_flat_fast_detect_outputs(tmp_path: Path) -> None:
    source_session = tmp_path / "fast_detect_output"
    source_material = source_session / "material_references"
    target_exp = tmp_path / "frontend_exp" / "fast-001"
    raw_third = tmp_path / "raw" / "third.mp4"
    raw_first = tmp_path / "raw" / "first.mp4"

    (source_session / "metadata").mkdir(parents=True)
    (source_session / "cv_outputs").mkdir(parents=True)
    (source_material / "clips").mkdir(parents=True)
    raw_third.parent.mkdir(parents=True)
    raw_third.write_bytes(b"third")
    raw_first.write_bytes(b"first")
    (source_material / "clips" / "item.mp4").write_bytes(b"clip")

    _write_json(
        source_session / "session_manifest.json",
        {
            "session_id": "fast-detect-output",
            "session_start_time": "2026-05-20T00:00:00+08:00",
            "videos": {
                "third_person": {"path": str(raw_third), "start_time": "2026-05-20T00:00:00+08:00"},
                "first_person": {"path": str(raw_first), "start_time": "2026-05-20T00:00:00+08:00"},
            },
            "output_dir": str(source_session),
        },
    )
    write_jsonl(
        source_session / "metadata" / "key_action_segments.jsonl",
        [
            {
                "segment_id": "seg_001",
                "third_person": {"video_path": str(raw_third), "clip_path": str(raw_third), "local_start_sec": 10.0, "local_end_sec": 16.0},
                "first_person": {"video_path": str(raw_first), "clip_path": str(raw_first), "local_start_sec": 10.0, "local_end_sec": 16.0},
            }
        ],
    )
    write_jsonl(source_session / "metadata" / "micro_segments.jsonl", [{"micro_segment_id": "micro_001", "parent_segment_id": "seg_001"}])
    write_jsonl(source_session / "metadata" / "vector_metadata.jsonl", [{"id": "seg_001"}])
    write_jsonl(source_session / "cv_outputs" / "yolo_frame_rows.jsonl", [{"hand_object_interactions": [{"score": 1.0}]}])
    write_jsonl(
        source_material / KEY_MATERIAL_REFERENCES_JSONL,
        [
            {
                "material_id": "mat_001",
                "asset_kind": "key_clip",
                "stored_file": str(source_material / "clips" / "item.mp4"),
                "source_file": str(raw_third),
            }
        ],
    )

    summary = sync_frontend_artifacts(
        target_experiment_dir=target_exp,
        source_session_dir=source_session,
        experiment_id="fast-001",
        experiment_title="flat fast detect",
        archive_existing=False,
        hardlink_media=False,
        refresh_focus=False,
        require_yolo_overlay=False,
    )

    target_manifest = target_exp / "key_action_index" / "manifest.json"
    error_codes = {item["code"] for item in summary["errors"]}
    warning_codes = {item["code"] for item in summary["warnings"]}

    assert target_manifest.exists()
    assert "missing_source_key_action_index" not in error_codes
    assert "missing_key_action_manifest" not in error_codes
    assert "missing_experiment_focus_window" not in error_codes
    assert "missing_experiment_focus_window" in warning_codes
    assert summary["source_key_action_index_dir"] == str(source_session.resolve())


def test_frontend_sync_validation_requires_full_focus_and_overlay(tmp_path: Path) -> None:
    target_exp = tmp_path / "exp-001"
    key_dir = target_exp / "key_action_index"
    material_root = target_exp / "material_references"
    (key_dir / "metadata").mkdir(parents=True)
    (key_dir / "cv_outputs").mkdir(parents=True)
    (key_dir / "clips" / "experiment_focus").mkdir(parents=True)
    material_root.mkdir(parents=True)

    _write_json(key_dir / "manifest.json", {"session_id": "exp-001"})
    write_jsonl(key_dir / "metadata" / "key_action_segments.jsonl", [{"segment_id": "seg_001"}])
    write_jsonl(key_dir / "metadata" / "micro_segments.jsonl", [{"micro_segment_id": "micro_001"}])
    write_jsonl(key_dir / "metadata" / "vector_metadata.jsonl", [{"id": "seg_001"}])
    write_jsonl(key_dir / "cv_outputs" / "yolo_frame_rows.jsonl", [{"hand_object_interactions": []}])
    _write_json(
        key_dir / "metadata" / "experiment_focus_window.json",
        {"source": "first_true_experiment_episode", "duration_sec": 7.0},
    )
    (key_dir / "clips" / "experiment_focus" / "third_person.mp4").write_bytes(b"clip")
    (key_dir / "clips" / "experiment_focus" / "first_person.mp4").write_bytes(b"clip")
    write_jsonl(material_root / KEY_MATERIAL_REFERENCES_JSONL, [{"material_id": "mat_001", "stored_file": "clip.mp4"}])
    (material_root / "clip.mp4").write_bytes(b"clip")

    result = validate_frontend_artifact_sync(target_exp, require_yolo_overlay=True, min_focus_duration_sec=30.0)
    error_codes = {item["code"] for item in result["errors"]}

    assert result["status"] == "failed"
    assert "short_experiment_focus_window" in error_codes
    assert "missing_focus_annotated_clip" in error_codes


def test_yolo_annotation_helpers_default_to_auto_device() -> None:
    helpers = [
        annotate_clip_with_yolo,
        run_yolo_on_keyframes,
        run_yolo_on_segment_clips,
        run_yolo_on_experiment_focus_clips,
    ]

    for helper in helpers:
        assert inspect.signature(helper).parameters["device"].default == "auto"
