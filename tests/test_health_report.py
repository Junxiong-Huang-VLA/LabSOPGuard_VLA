from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.health_report import build_run_health_report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _minimal_session(tmp_path: Path) -> Path:
    session = tmp_path / "session"
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
        "boundary_support_count": 3,
        "boundary_source": "yolo_physical_evidence_cluster",
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
    _write_json(session / "video_info.json", {"video_sources": {"first_person": {"duration_sec": 20.0}}})
    (session / "index").mkdir(parents=True, exist_ok=True)
    (session / "index" / "fallback_index.pkl").write_bytes(b"index")
    return session


def test_health_report_passes_minimal_no_label_session(tmp_path: Path) -> None:
    session = _minimal_session(tmp_path)

    report = build_run_health_report(session)

    assert report["gate_status"] == "pass"
    assert report["metrics"]["segment_count"] == 1
    assert report["metrics"]["paths"]["missing_path_count"] == 0


def test_health_report_fails_missing_index(tmp_path: Path) -> None:
    session = _minimal_session(tmp_path)
    (session / "index" / "fallback_index.pkl").unlink()

    report = build_run_health_report(session)

    assert report["gate_status"] == "fail"
    assert any(issue["code"] == "missing_required_artifact" for issue in report["errors"])


def test_health_report_warns_on_coarse_coverage(tmp_path: Path) -> None:
    session = _minimal_session(tmp_path)
    row = json.loads((session / "cv_outputs" / "detected_segments.jsonl").read_text(encoding="utf-8").splitlines()[0])
    row["duration_sec"] = 18.0
    _write_jsonl(session / "cv_outputs" / "detected_segments.jsonl", [row])

    report = build_run_health_report(session, max_total_coverage_ratio=0.5)

    assert report["status"] == "warn"
    assert any(issue["code"] == "high_total_action_coverage" for issue in report["warnings"])


def test_health_report_fails_failed_query_validation_artifact(tmp_path: Path) -> None:
    session = _minimal_session(tmp_path)
    _write_json(
        session / "evaluation" / "query_validation.json",
        {
            "status": "fail",
            "query_count": 1,
            "acceptance_hit_rate": 0.0,
            "query_hit_rate": 1.0,
            "quality_hit_rate": 0.0,
            "failed_query_count": 1,
            "threshold_failures": [{"metric": "acceptance_hit_rate", "actual": 0.0, "minimum": 1.0}],
        },
    )

    report = build_run_health_report(session)

    assert report["gate_status"] == "fail"
    assert any(issue["code"] == "query_validation_failed" for issue in report["errors"])
    assert report["metrics"]["query_validation"]["failed_artifact_count"] == 1
