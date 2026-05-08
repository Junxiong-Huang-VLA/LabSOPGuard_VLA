from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.health_report import build_run_health_report
from key_action_indexer.schemas import read_jsonl
from key_action_indexer.session_context_seed import seed_session_context


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _session_with_metadata(tmp_path: Path) -> Path:
    session = tmp_path / "session"
    _write_json(
        session / "manifest.json",
        {
            "session_id": "ctx_seed",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(session / "raw" / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "camera_id": "top_view",
                },
                "first_person": {
                    "path": str(session / "raw" / "first.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "camera_id": "bottom_view",
                },
            },
            "detection_config": {"detector_backend": "yolo_interaction"},
            "output_dir": str(session),
        },
    )
    _write_json(
        session / "video_info.json",
        {
            "video_sources": {
                "third_person": {"duration_sec": 79.667, "fps": 15.0, "width": 960, "height": 540, "exists": True},
                "first_person": {"duration_sec": 79.667, "fps": 15.0, "width": 960, "height": 540, "exists": True},
            }
        },
    )
    _write_json(session / "pipeline_summary.json", {"session_id": "ctx_seed", "segment_count": 6, "total_action_duration_sec": 31.064})
    return session


def _minimal_health_session(tmp_path: Path) -> Path:
    session = _session_with_metadata(tmp_path)
    clip = session / "clips" / "seg_000001" / "first_person.mp4"
    keyframe = session / "keyframes" / "seg_000001" / "peak.jpg"
    clip.parent.mkdir(parents=True, exist_ok=True)
    keyframe.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"clip")
    keyframe.write_bytes(b"jpg")
    segment = {
        "segment_id": "seg_000001",
        "duration_sec": 5.0,
        "boundary_confidence": 0.82,
        "first_person": {"clip_path": str(clip)},
        "keyframes": {"peak": str(keyframe)},
    }
    micro = {
        "micro_segment_id": "seg_000001_micro_001",
        "parent_segment_id": "seg_000001",
        "duration_sec": 1.0,
        "first_person_clip": str(clip),
        "peak_keyframe": str(keyframe),
    }
    _write_jsonl(session / "cv_outputs" / "detected_segments.jsonl", [segment])
    _write_jsonl(session / "metadata" / "key_action_segments.jsonl", [segment])
    _write_jsonl(session / "metadata" / "micro_segments.jsonl", [micro])
    _write_jsonl(session / "metadata" / "vector_metadata.jsonl", [{"segment_id": "seg_000001", "index_text": "balance weighing"}])
    _write_jsonl(session / "metadata" / "micro_vector_metadata.jsonl", [{"micro_segment_id": "seg_000001_micro_001", "index_text": "hand object"}])
    _write_jsonl(session / "metadata" / "human_confirmation_queue.jsonl", [])
    _write_jsonl(session / "metadata" / "user_text_events.jsonl", [])
    _write_jsonl(session / "metadata" / "ai_reply_events.jsonl", [])
    _write_jsonl(session / "metadata" / "upload_events.jsonl", [])
    _write_jsonl(session / "metadata" / "database_records.jsonl", [])
    _write_jsonl(session / "metadata" / "sop_records.jsonl", [])
    (session / "index").mkdir(parents=True, exist_ok=True)
    (session / "index" / "fallback_index.pkl").write_bytes(b"index")
    return session


def test_seed_session_context_writes_non_label_event(tmp_path: Path) -> None:
    session = _session_with_metadata(tmp_path)

    summary = seed_session_context(session)
    rows = read_jsonl(session / "metadata" / "session_context_events.jsonl")

    assert summary["written"] is True
    assert summary["row_count"] == 1
    assert rows[0]["event_type"] == "session_context"
    assert rows[0]["non_label_context"] is True
    assert rows[0]["payload"]["evidence_policy"]["is_manual_label"] is False
    assert "no human annotation" in rows[0]["text"]


def test_seed_session_context_is_idempotent_without_force(tmp_path: Path) -> None:
    session = _session_with_metadata(tmp_path)

    seed_session_context(session)
    summary = seed_session_context(session)

    assert summary["skipped"] is True
    assert len(read_jsonl(session / "metadata" / "session_context_events.jsonl")) == 1


def test_session_context_seed_clears_thin_context_health_warning(tmp_path: Path) -> None:
    session = _minimal_health_session(tmp_path)

    before = build_run_health_report(session)
    seed_session_context(session)
    after = build_run_health_report(session)

    assert any(issue["code"] == "thin_multimodal_context" for issue in before["warnings"])
    assert not any(issue["code"] == "thin_multimodal_context" for issue in after["warnings"])
    assert after["metrics"]["context"]["counts"]["session_context_events"] == 1
