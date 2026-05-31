from __future__ import annotations

import json
from pathlib import Path

from key_action_indexer.experiment_focus import extract_experiment_focus_clips, select_experiment_focus_window
from key_action_indexer.schemas import write_jsonl


def test_experiment_focus_window_spans_all_true_episodes(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "focus-test",
                "session_start_time": "2026-05-08T15:36:48+08:00",
                "videos": {
                    "third_person": {
                        "path": str(session / "third.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    },
                    "first_person": {
                        "path": str(session / "first.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    },
                },
                "output_dir": str(session),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "experiment_episodes.jsonl",
        [
            {
                "episode_id": "episode_000001",
                "segment_id": "episode_000001",
                "session_start_sec": 38.5,
                "session_end_sec": 45.5,
                "true_start_sec": 40.5,
                "true_end_sec": 42.5,
                "primary_objects": {"spatula": 1},
            },
            {
                "episode_id": "episode_000002",
                "segment_id": "episode_000002",
                "session_start_sec": 146.5,
                "session_end_sec": 177.5,
                "true_start_sec": 148.5,
                "true_end_sec": 174.5,
                "primary_objects": {"reagent_bottle": 2},
            },
        ],
    )

    window = select_experiment_focus_window(session)

    assert window["source"] == "all_true_experiment_episodes"
    assert window["start_sec"] == 40.5
    assert window["end_sec"] == 174.5
    assert window["duration_sec"] == 134.0
    assert window["included_segment_ids"] == ["episode_000001", "episode_000002"]
    assert window["segment_count"] == 2


def test_experiment_focus_window_uses_yolo_coverage_when_no_segments(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "focus-yolo-test",
                "session_start_time": "2026-05-08T15:36:48+08:00",
                "videos": {
                    "third_person": {
                        "path": str(session / "third.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    }
                },
                "output_dir": str(session),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "key_action_segments.jsonl",
        [],
    )
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            {
                "source_view": "third_person",
                "time_sec": 0.0,
                "global_time": "2026-05-08T15:36:48+08:00",
                "label_counts": {"beaker": 1},
            },
            {
                "source_view": "third_person",
                "time_sec": 20.0,
                "global_time": "2026-05-08T15:37:08+08:00",
                "label_counts": {"gloved_hand": 1, "beaker": 1},
            },
        ],
    )

    window = select_experiment_focus_window(session)

    assert window["source"] == "yolo_scan_coverage_fallback"
    assert window["start_sec"] == 0.0
    assert window["end_sec"] > 20.0
    assert window["duration_sec"] > 10.0


def test_experiment_focus_window_extends_episode_start_to_early_yolo_activity(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "focus-episode-yolo-test",
                "session_start_time": "2026-05-08T15:36:48+08:00",
                "videos": {
                    "third_person": {
                        "path": str(session / "third.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    }
                },
                "output_dir": str(session),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "experiment_episodes.jsonl",
        [
            {
                "episode_id": "episode_000001",
                "segment_id": "episode_000001",
                "session_start_sec": 12.0,
                "session_end_sec": 18.0,
                "true_start_sec": 13.0,
                "true_end_sec": 16.0,
                "primary_objects": {"beaker": 1},
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            {
                "source_view": "third_person",
                "time_sec": 1.5,
                "global_time": "2026-05-08T15:36:49.500000+08:00",
                "label_counts": {"gloved_hand": 1, "beaker": 1},
                "active_score": 0.8,
            }
        ],
    )

    window = select_experiment_focus_window(session)

    assert window["source"] == "all_true_experiment_episodes"
    assert window["start_sec"] == 1.5
    assert window["start_boundary_status"] == "left_censored_no_confirmed_ppe_entry"
    assert window["end_sec"] >= 16.0


def test_experiment_focus_window_uses_ppe_lifecycle_boundaries(tmp_path: Path) -> None:
    session = tmp_path / "session"
    metadata = session / "metadata"
    cv_outputs = session / "cv_outputs"
    metadata.mkdir(parents=True)
    cv_outputs.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "focus-lifecycle-test",
                "session_start_time": "2026-05-08T15:36:48+08:00",
                "videos": {
                    "third_person": {
                        "path": str(session / "third.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    },
                    "first_person": {
                        "path": str(session / "first.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    },
                },
                "output_dir": str(session),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "experiment_episodes.jsonl",
        [
            {
                "episode_id": "episode_000001",
                "segment_id": "episode_000001",
                "session_start_sec": 50.0,
                "session_end_sec": 60.0,
                "true_start_sec": 50.0,
                "true_end_sec": 60.0,
                "primary_objects": {"paper": 1},
            }
        ],
    )
    write_jsonl(
        cv_outputs / "yolo_frame_rows.jsonl",
        [
            {
                "source_view": "third_person",
                "alignment_time_sec": 10.0,
                "label_counts": {"hand": 1, "PPE_Storage": 1, "lab_coat": 1},
                "source_duration_sec": 220.0,
            },
            {
                "source_view": "third_person",
                "alignment_time_sec": 55.0,
                "label_counts": {"gloved_hand": 1, "paper": 1},
                "hand_object_interactions": [
                    {
                        "hand_label": "gloved_hand",
                        "object_label": "paper",
                        "score": 0.82,
                        "object_overlap_ratio": 0.36,
                    }
                ],
                "source_duration_sec": 220.0,
            },
            {
                "source_view": "first_person",
                "alignment_time_sec": 180.0,
                "label_counts": {"gloved_hand": 1, "lab_coat": 1},
                "source_duration_sec": 220.0,
            },
        ],
    )

    window = select_experiment_focus_window(session)

    assert window["start_sec"] == 10.0
    assert window["end_sec"] == 220.0
    assert window["start_boundary_status"] == "confirmed_ppe_preparation"
    assert window["end_boundary_status"] == "open_until_video_end_or_last_ppe_evidence"
    assert window["lifecycle_boundary"]["start_boundary_confirmed"] is True
    assert window["lifecycle_boundary"]["end_boundary_confirmed"] is False
    assert window["lifecycle_boundary"]["start_reason"] == "ppe_prep_before_first_core_action"
    assert window["lifecycle_boundary"]["end_reason"] == "no_ppe_exit_seen_extend_to_video_end"


def test_long_experiment_focus_clips_default_to_source_reference(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_EXPERIMENT_FOCUS_CLIP_MAX_EXTRACT_SEC", "60")
    session = tmp_path / "session"
    metadata = session / "metadata"
    metadata.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps(
            {
                "session_id": "focus-long-source-reference",
                "session_start_time": "2026-05-08T15:36:48+08:00",
                "videos": {
                    "third_person": {
                        "path": str(session / "third.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    },
                    "first_person": {
                        "path": str(session / "first.mp4"),
                        "start_time": "2026-05-08T15:36:48+08:00",
                        "fps": 30,
                    },
                },
                "output_dir": str(session),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        metadata / "experiment_episodes.jsonl",
        [
            {
                "episode_id": "episode_000001",
                "segment_id": "episode_000001",
                "session_start_sec": 10.0,
                "session_end_sec": 220.0,
                "true_start_sec": 10.0,
                "true_end_sec": 220.0,
                "primary_objects": {"paper": 1},
            }
        ],
    )

    summary = extract_experiment_focus_clips(session, dry_run=False)

    assert summary["available"] is True
    assert summary["extracted_clip_count"] == 0
    assert summary["source_reference_count"] == 2
    assert {row["clip_file_status"] for row in summary["clips"]} == {"source_reference_only"}
    assert all(row["clip_path"] is None for row in summary["clips"])
