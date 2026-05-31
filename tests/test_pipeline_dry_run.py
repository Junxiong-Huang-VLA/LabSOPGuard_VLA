from __future__ import annotations

import json
from pathlib import Path

import pytest

import key_action_indexer.pipeline as pipeline
from key_action_indexer.config import DetectorConfig
from key_action_indexer.pipeline import run_detection_only, run_pipeline
from key_action_indexer.schemas import SessionManifest
from key_action_indexer.vector_index import VectorIndex


def test_yolo_timing_summary_marks_numeric_cuda_and_batch_stats(tmp_path: Path) -> None:
    paths = {"metadata": tmp_path / "metadata"}

    summary = pipeline._append_yolo_timing_rows(
        paths,
        [
            {
                "stage": "yolo_scan",
                "pipeline_stage": "coarse_segment_scan",
                "sampled_frames": 32,
                "read_frames": 32,
                "decode_sec": 1.5,
                "inference_sec": 2.5,
                "postprocess_sec": 0.25,
                "wall_sec": 4.5,
                "batch_size": 16,
                "actual_batch_sizes": [16, 16],
                "batch_count": 2,
                "yolo_predict_call_count": 2,
                "yolo_batch_predict_attempts": 2,
                "yolo_batch_predict_calls": 2,
                "requested_device": "0",
                "actual_device": "0",
                "scan_backend": "ffmpeg_sparse_pipe_chunks",
            }
        ],
    )

    stage = summary["by_stage"]["coarse_segment_scan"]
    assert stage["gpu_device_observed"] is True
    assert stage["batch_enabled"] is True
    assert stage["actual_batch_sizes"] == [16]
    assert stage["actual_batch_size_counts"] == {"16": 2}
    assert stage["yolo_predict_call_count"] == 2
    assert stage["yolo_batch_predict_calls"] == 2
    assert stage["decode_sec"] == 1.5
    assert stage["inference_sec"] == 2.5


def test_formal_output_gate_blocks_time_axis_and_action_alignment_failures(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    paths = {"metadata": metadata}

    action_blocked_gate = pipeline._formal_output_gate_status(
        paths,
        action_summary={
            "formal_results_allowed": False,
            "formal_event_count": 0,
            "dual_view_action_event_count": 4,
        },
        require_action_alignment=True,
    )

    assert action_blocked_gate["status"] == "blocked"
    assert action_blocked_gate["formal_results_allowed"] is False
    assert action_blocked_gate["video_memory_allowed"] is False
    assert action_blocked_gate["blocked_reason"] == "formal_results_not_allowed"
    assert action_blocked_gate["dual_view_action_gate"]["formal_event_count"] == 0

    pipeline._write_formal_output_gate(paths, action_blocked_gate)
    persisted = json.loads((metadata / "formal_output_gate.json").read_text(encoding="utf-8"))
    assert persisted["blocked_reason"] == "formal_results_not_allowed"

    (metadata / "time_axis_health.json").write_text(
        json.dumps({"time_axis_unreliable": True, "video_memory_allowed": False}),
        encoding="utf-8",
    )
    time_blocked_gate = pipeline._formal_output_gate_status(
        paths,
        action_summary={"formal_results_allowed": True, "formal_event_count": 1},
        require_action_alignment=True,
    )

    assert time_blocked_gate["status"] == "blocked"
    assert time_blocked_gate["formal_results_allowed"] is False
    assert time_blocked_gate["video_memory_allowed"] is False
    assert time_blocked_gate["blocked_reason"] == "time_axis_unreliable"


def test_formal_output_gate_requires_window_visual_review_before_action_pass(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    paths = {"metadata": metadata}
    action_summary = {
        "formal_results_allowed": True,
        "formal_event_count": 1,
        "dual_view_action_event_count": 1,
    }
    (metadata / "formal_experiment_windows.json").write_text(
        json.dumps({"window_count": 1, "windows": [{"experiment_window_id": "formal_window_001"}]}),
        encoding="utf-8",
    )

    missing_review_gate = pipeline._formal_output_gate_status(
        paths,
        action_summary=action_summary,
        require_action_alignment=True,
    )

    assert missing_review_gate["status"] == "blocked"
    assert missing_review_gate["blocked_reason"] == "formal_window_visual_review_missing"
    assert missing_review_gate["formal_results_allowed"] is False

    (metadata / "formal_window_human_review_manifest.json").write_text(
        json.dumps(
            {
                "total_formal_windows": 1,
                "passed_visual_review_count": 0,
                "recommended_reject_window_ids": ["formal_window_001"],
                "suspicious_window_ids": [],
            }
        ),
        encoding="utf-8",
    )
    rejected_review_gate = pipeline._formal_output_gate_status(
        paths,
        action_summary=action_summary,
        require_action_alignment=True,
    )

    assert rejected_review_gate["blocked_reason"] == "formal_window_visual_review_failed"
    pipeline._write_phase_consistency_from_formal_gate(paths, rejected_review_gate)
    phase = json.loads((metadata / "phase_consistency_report.json").read_text(encoding="utf-8"))
    assert phase["status"] == "action_phase_rejected"
    assert phase["visual_alignment_verified"] is False
    assert phase["action_phase_verified"] is False

    (metadata / "formal_window_human_review_manifest.json").write_text(
        json.dumps(
            {
                "total_formal_windows": 1,
                "passed_visual_review_count": 1,
                "recommended_reject_window_ids": [],
                "suspicious_window_ids": [],
                "pending_visual_review_window_ids": [],
            }
        ),
        encoding="utf-8",
    )
    passed_review_gate = pipeline._formal_output_gate_status(
        paths,
        action_summary=action_summary,
        require_action_alignment=True,
    )

    assert passed_review_gate["status"] == "passed"
    assert passed_review_gate["formal_results_allowed"] is True


def test_paired_micro_scan_windows_map_primary_evidence_to_aligned_view(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_SCAN_PAD_SEC", "0")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_MIN_WINDOW_SEC", "2")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_PAIRED_MICRO_MAX_WINDOW_SEC", "8")
    manifest = SessionManifest.from_dict(
        {
            "session_id": "paired_micro_scan",
            "session_start_time": "2026-04-29T10:00:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third.mp4"),
                    "start_time": "2026-04-29T10:00:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
                "first_person": {
                    "path": str(tmp_path / "first.mp4"),
                    "start_time": "2026-04-29T10:00:02+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
            },
            "output_dir": str(tmp_path / "session"),
        }
    )
    windows = pipeline._micro_pair_scan_windows_from_yolo_rows(
        manifest,
        [
            {
                "source_view": "third_person",
                "alignment_time_sec": 10.0,
                "active_score": 1.0,
                "interaction_score": 1.0,
                "hand_object_interactions": [{"object": "paper"}],
            }
        ],
        "first_person",
        DetectorConfig(start_threshold=0.6, end_threshold=0.3, merge_gap_sec=2.0),
    )

    assert len(windows) == 1
    assert windows[0]["start_sec"] == pytest.approx(7.0)
    assert windows[0]["end_sec"] == pytest.approx(9.0)
    assert windows[0]["source_role"] == "paired_micro_support"


def test_fast_detection_defaults_to_parallel_dual_view_coarse_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_DEFER_SEGMENT_ASSETS", "1")
    monkeypatch.setenv("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "")
    monkeypatch.delenv("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_BOTH_VIEWS", raising=False)
    manifest = SessionManifest.from_dict(
        {
            "session_id": "dual_view_fast_default",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
                "first_person": {
                    "path": str(tmp_path / "first.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
            },
            "output_dir": str(tmp_path / "session"),
        }
    )
    pipeline._ensure_fast_material_preprocess_defaults()
    views, scan_both, _fallback_view = pipeline._resolve_yolo_coarse_scan_views(
        manifest,
        DetectorConfig(detector_backend="yolo", yolo_scan_both_views=False),
    )

    assert scan_both is True
    assert views == ["first_person", "third_person"]
    fast_config = DetectorConfig(detector_backend="yolo", yolo_scan_both_views=False)
    assert pipeline._coarse_yolo_sample_fps(fast_config) == pytest.approx(0.2)
    assert pipeline._refined_yolo_sample_fps(fast_config) == pytest.approx(1.5)


def test_fast_coarse_scan_splits_dual_view_video_into_parallel_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_CHUNKED", "1")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_CHUNK_SEC", "300")
    monkeypatch.setenv("KEY_ACTION_FAST_LOCATE_COARSE_SCAN_WORKERS", "8")
    monkeypatch.setenv("KEY_ACTION_DEFER_SEGMENT_ASSETS", "1")
    monkeypatch.setenv("KEY_ACTION_DRY_RUN_DURATION_SEC", "960")
    manifest = SessionManifest.from_dict(
        {
            "session_id": "dual_view_chunked_coarse",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
                "first_person": {
                    "path": str(tmp_path / "first.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
            },
            "output_dir": str(tmp_path / "session"),
        }
    )
    config = DetectorConfig(
        detector_backend="yolo",
        yolo_scan_both_views=True,
        long_video_two_stage_sampling=True,
        long_video_stage1_sample_fps=1 / 30,
    )

    tasks = pipeline._coarse_scan_tasks(
        manifest,
        ["first_person", "third_person"],
        config,
        sample_fps=1 / 30,
        dry_run=True,
    )
    workers = pipeline._resolve_yolo_coarse_scan_workers(len(tasks), default_workers=8)

    assert len(tasks) == 8
    assert workers == 8
    assert {task["view"] for task in tasks} == {"first_person", "third_person"}
    assert all(task["chunked"] is True for task in tasks)
    assert tasks[1]["scan_start_sec"] < tasks[1]["chunk_start_sec"]
    assert tasks[-1]["scan_end_sec"] == pytest.approx(960.0)


def test_coarse_scan_runtime_env_prefers_ffmpeg_chunks_for_sparse_long_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "KEY_ACTION_DEFER_SEGMENT_ASSETS",
        "KEY_ACTION_FAST_LOCATE_ONLY",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_SPARSE_MODE",
        "KEY_ACTION_YOLO_COARSE_FFMPEG_SPARSE_MODE",
        "KEY_ACTION_FAST_LOCATE_COARSE_FFMPEG_CHUNK_SEC",
        "KEY_ACTION_YOLO_COARSE_FFMPEG_CHUNK_SEC",
    ):
        monkeypatch.delenv(name, raising=False)

    overrides, plan = pipeline._coarse_scan_runtime_env_overrides(
        DetectorConfig(detector_backend="yolo", long_video_chunk_sec=900.0),
        0.2,
        [{"view": "third_person", "scan_start_sec": 0.0, "scan_end_sec": 900.0}],
        dry_run=False,
    )

    assert overrides["KEY_ACTION_YOLO_COARSE_FFMPEG_SPARSE_MODE"] == "chunks"
    assert overrides["KEY_ACTION_YOLO_COARSE_FFMPEG_CHUNK_SEC"] == "900"
    assert "auto_selected_ffmpeg_chunks_to_avoid_per_frame_seek" in plan["reasons"]


def test_coarse_scan_io_limit_caps_outer_workers_against_ffmpeg_workers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KEY_ACTION_DEFER_SEGMENT_ASSETS", raising=False)
    monkeypatch.delenv("KEY_ACTION_FAST_LOCATE_ONLY", raising=False)
    monkeypatch.setenv("KEY_ACTION_YOLO_COARSE_FFMPEG_SPARSE_MODE", "chunks")
    monkeypatch.setenv("KEY_ACTION_YOLO_COARSE_FFMPEG_WORKERS", "4")
    monkeypatch.setenv("KEY_ACTION_YOLO_COARSE_SCAN_IO_MAX_WORKERS", "8")

    workers, plan = pipeline._resolve_yolo_coarse_scan_io_limited_workers(
        8,
        8,
        sample_fps=0.2,
        scan_tasks=[
            {"view": "third_person", "scan_start_sec": float(index * 900), "scan_end_sec": float((index + 1) * 900)}
            for index in range(8)
        ],
    )

    assert workers == 2
    assert plan["cap_reason"] == "coarse_scan_io_limit"
    assert plan["expected_concurrent_extractors"] == 8
    assert plan["sparse_mode"] == "chunks"


def test_yolo_view_alignment_uses_shared_recording_timeline(tmp_path: Path) -> None:
    output_dir = tmp_path / "experiment" / "key_action_index"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    (tmp_path / "experiment" / "timeline_alignment.json").write_text(
        json.dumps(
            {
                "alignment_status": "shared_recording",
                "alignment_reliable_for_dual_view_pairing": True,
                "streams": [
                    {"role": "first_person", "alignment_status": "shared_recording"},
                    {"role": "third_person", "alignment_status": "shared_recording"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = SessionManifest.from_dict(
        {
            "session_id": "shared_recording_alignment",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third_rgb.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                    "role": "third_person",
                    "camera_id": "third_person",
                },
                "first_person": {
                    "path": str(tmp_path / "first_rgb.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                    "role": "first_person",
                    "camera_id": "first_person",
                },
            },
            "output_dir": str(output_dir),
        }
    )

    summary = pipeline._apply_view_alignment_from_yolo(
        manifest,
        [
            {"source_view": "first_person", "local_time_sec": 10.0, "is_experiment_active": True},
            {"source_view": "third_person", "local_time_sec": 10.0, "is_experiment_active": True},
        ],
        {"metadata": metadata_dir},
    )

    assert summary["alignment_status"] == "aligned"
    assert summary["alignment_reliable_for_dual_view_pairing"] is True
    assert summary["time_ranges_by_view"]["first_person"]["camera_id"] == "first_person"


def test_yolo_view_alignment_trusts_matching_manifest_offsets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_TRUST_MANIFEST_DUAL_VIEW_ALIGNMENT", "1")
    output_dir = tmp_path / "experiment" / "key_action_index"
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True)
    manifest = SessionManifest.from_dict(
        {
            "session_id": "manifest_alignment",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third_rgb.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                    "role": "third_person",
                },
                "first_person": {
                    "path": str(tmp_path / "first_rgb.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                    "role": "first_person",
                },
            },
            "output_dir": str(output_dir),
        }
    )

    summary = pipeline._apply_view_alignment_from_yolo(
        manifest,
        [
            {"source_view": "first_person", "local_time_sec": 2819.0, "is_experiment_active": True},
            {"source_view": "third_person", "local_time_sec": 2819.0, "is_experiment_active": True},
        ],
        {"metadata": metadata_dir},
    )

    assert summary["alignment_status"] == "aligned"
    assert summary["alignment_reliable_for_dual_view_pairing"] is True
    assert summary["manifest_alignment_reliable"] is True
    assert summary["timeline_alignment_status"] == "manifest_shared_recording"


def test_pre_coarse_timeline_alignment_drives_scan_bounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_DRY_RUN_DURATION_SEC", "960")
    output_dir = tmp_path / "session"
    manifest = SessionManifest.from_dict(
        {
            "session_id": "pre_coarse_alignment",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {
                    "path": str(tmp_path / "third.mp4"),
                    "start_time": "2026-04-29T17:25:00+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
                "first_person": {
                    "path": str(tmp_path / "first.mp4"),
                    "start_time": "2026-04-29T17:25:02+08:00",
                    "fps": 30,
                    "offset_sec": 0,
                },
            },
            "output_dir": str(output_dir),
        }
    )
    paths = pipeline._mkdirs(output_dir)

    alignment = pipeline._ensure_pre_coarse_timeline_alignment(manifest, paths, dry_run=True)
    tasks = pipeline._coarse_scan_tasks(
        manifest,
        ["first_person", "third_person"],
        DetectorConfig(detector_backend="yolo", yolo_scan_both_views=True),
        sample_fps=0.5,
        dry_run=True,
    )

    assert (output_dir / "metadata" / "pre_coarse_timeline_alignment.json").exists()
    assert alignment["execution_order"][:2] == ["time_alignment_preflight", "coarse_seek_scan"]
    assert manifest.config["timeline_alignment"]["common_overlap_start_sec"] == pytest.approx(2.0)
    starts_by_view = {task["view"]: task["scan_start_sec"] for task in tasks}
    assert starts_by_view["first_person"] == pytest.approx(0.0)
    assert starts_by_view["third_person"] == pytest.approx(2.0)


def test_alignment_health_exposes_quality_fields_for_understanding(tmp_path: Path) -> None:
    output_dir = tmp_path / "session"
    metadata = output_dir / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "time_anchors.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "anchor_id": "a1",
                        "source": "first_person",
                        "expected_global_time": "2026-05-03T09:00:00+08:00",
                        "predicted_global_time": "2026-05-03T09:00:02.200000+08:00",
                    }
                ),
                json.dumps(
                    {
                        "anchor_id": "a2",
                        "source": "third_person",
                        "expected_global_time": "2026-05-03T09:01:00+08:00",
                        "predicted_global_time": "2026-05-03T09:01:03.400000+08:00",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = SessionManifest.from_dict(
        {
            "session_id": "alignment_quality",
            "session_start_time": "2026-05-03T09:00:00+08:00",
            "videos": {
                "third_person": {"path": str(tmp_path / "third.mp4"), "start_time": "2026-05-03T09:00:00+08:00"},
                "first_person": {"path": str(tmp_path / "first.mp4"), "start_time": "2026-05-03T09:00:00+08:00"},
            },
            "output_dir": str(output_dir),
        }
    )

    health, _drift, _degradation = pipeline._compute_alignment_health(manifest, {"metadata": metadata}, None)

    quality = health["time_alignment_quality"]
    assert quality["schema_version"] == "key_action_time_alignment_quality.v1"
    assert quality["status"] == "warning"
    assert quality["mae_sec"] == pytest.approx(2.8)
    assert "high_mae" in quality["alert_reasons"]
    assert (metadata / "alignment_report.json").exists()


def test_dry_run_pipeline_outputs_and_query(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "")
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        "\n".join(
            [
                json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "现在开始称量样品。"}, ensure_ascii=False),
                json.dumps({"utterance_id": "utt_002", "start_sec": 620.0, "end_sec": 628.0, "text": "接下来使用移液枪加 200 微升。"}, ensure_ascii=False),
                json.dumps({"utterance_id": "utt_003", "start_sec": 900.0, "end_sec": 910.0, "text": "记录一下天平读数。"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "test_session",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {
                "path": str(output_dir / "raw" / "third_person.mp4"),
                "start_time": "2026-04-29T17:25:00+08:00",
                "fps": 30,
                "offset_sec": 0,
            },
            "first_person": {
                "path": str(output_dir / "raw" / "first_person.mp4"),
                "start_time": "2026-04-29T17:25:02+08:00",
                "fps": 30,
                "offset_sec": 0,
            },
        },
        "transcript": {
            "path": str(transcript_path),
            "start_time": "2026-04-29T17:25:00+08:00",
            "offset_sec": 0,
        },
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)
    # Adjacent dry-run micro actions may be grouped into one experiment episode;
    # the pipeline should still preserve the underlying three micro events.
    assert summary["segment_count"] >= 2
    assert summary["micro_segment_count"] >= 3
    assert (output_dir / "metadata" / "key_action_segments.jsonl").exists()
    assert (output_dir / "uploads").exists()
    assert (output_dir / "exports").exists()
    assert (output_dir / "metadata" / "video_sources.jsonl").exists()
    assert (output_dir / "metadata" / "user_text_events.jsonl").exists()
    assert (output_dir / "metadata" / "ai_reply_events.jsonl").exists()
    assert (output_dir / "metadata" / "upload_events.jsonl").exists()
    assert (output_dir / "metadata" / "input_ingestion_summary.json").exists()
    assert (output_dir / "metadata" / "vector_metadata.jsonl").exists()
    assert (output_dir / "metadata" / "validation.json").exists()
    assert (output_dir / "metadata" / "detector_config.json").exists()
    assert (output_dir / "metadata" / "model_inventory.json").exists()
    assert (output_dir / "metadata" / "capability_gap_report.json").exists()
    assert (output_dir / "video_info.json").exists()
    assert (output_dir / "transcript" / "aligned_transcript.jsonl").exists()
    assert (output_dir / "metadata" / "session_context_events.jsonl").exists()
    assert (output_dir / "metadata" / "record_ingestion_summary.json").exists()
    assert (output_dir / "metadata" / "sop_records.jsonl").exists()
    assert (output_dir / "metadata" / "database_records.jsonl").exists()
    assert (output_dir / "metadata" / "history_model.json").exists()
    assert (output_dir / "cv_outputs" / "detected_segments.jsonl").exists()
    assert (output_dir / "metadata" / "experiment_episodes.jsonl").exists()
    assert (output_dir / "metadata" / "multimodal_alignment.jsonl").exists()
    assert (output_dir / "debug" / "frame_scores.png").exists()
    assert (output_dir / "metadata" / "micro_dedup_log.jsonl").exists()
    assert (output_dir / "metadata" / "micro_segments_raw.jsonl").exists()
    assert (output_dir / "metadata" / "micro_segments.jsonl").exists()
    assert (output_dir / "metadata" / "micro_vector_metadata.jsonl").exists()
    assert (output_dir / "evaluation" / "micro_quality_stats.json").exists()
    assert (output_dir / "evaluation" / "micro_merge_stats.json").exists()
    segment_rows = [
        json.loads(line)
        for line in (output_dir / "metadata" / "key_action_segments.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert segment_rows
    assert all(row.get("run_manifest_id") == summary["run_id"] for row in segment_rows)
    assert all(row.get("detector_backend") for row in segment_rows)
    query_smoke_report = json.loads((output_dir / "reports" / "query_smoke_test.json").read_text(encoding="utf-8"))
    assert query_smoke_report["query"]
    assert isinstance(query_smoke_report["results"], list)

    index = VectorIndex.load(output_dir / "index")
    results = index.query("找一下使用移液枪加样的片段", top_k=1)
    assert results
    assert results[0]["action_type"] == "pipetting"


def test_run_pipeline_uses_current_detection_dispatcher(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "")
    called = {"current_dispatcher": 0}
    original_detect_with_config = pipeline._detect_with_config

    def legacy_dispatcher_should_not_run(*args, **kwargs):
        raise AssertionError("run_pipeline should not call legacy _detect_segments")

    def wrapped_detect_with_config(*args, **kwargs):
        called["current_dispatcher"] += 1
        return original_detect_with_config(*args, **kwargs)

    monkeypatch.setattr(pipeline, "_detect_segments", legacy_dispatcher_should_not_run)
    monkeypatch.setattr(pipeline, "_detect_with_config", wrapped_detect_with_config)

    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "test"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "dispatcher_test",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30, "offset_sec": 0},
            "first_person": {"path": str(output_dir / "raw" / "fp.mp4"), "start_time": "2026-04-29T17:25:02+08:00", "fps": 30, "offset_sec": 0},
        },
        "transcript": {"path": str(transcript_path), "start_time": "2026-04-29T17:25:00+08:00", "offset_sec": 0},
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_pipeline(manifest_path, dry_run=True)

    assert called["current_dispatcher"] == 1
    assert summary["segment_count"] >= 1


def test_detection_only_dry_run_summary_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KEY_ACTION_PHYSICAL_ACTION_SCOPE", "")
    transcript_path = tmp_path / "dialogue.jsonl"
    transcript_path.write_text(
        json.dumps({"utterance_id": "utt_001", "start_sec": 600.0, "end_sec": 605.0, "text": "test"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "session"
    manifest = {
        "session_id": "detection_only_test",
        "session_start_time": "2026-04-29T17:25:00+08:00",
        "videos": {
            "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30, "offset_sec": 0},
            "first_person": {"path": str(output_dir / "raw" / "fp.mp4"), "start_time": "2026-04-29T17:25:02+08:00", "fps": 30, "offset_sec": 0},
        },
        "transcript": {"path": str(transcript_path), "start_time": "2026-04-29T17:25:00+08:00", "offset_sec": 0},
        "output_dir": str(output_dir),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

    summary = run_detection_only(manifest_path, dry_run=True)

    assert summary["session_id"] == "detection_only_test"
    assert summary["segment_count"] >= 1
    assert set(summary["model_inventory_summary"]) == {"primary_model", "model_count", "dataset_count"}
    assert Path(summary["detected_segments"]).exists()
    assert Path(summary["frame_scores"]).exists()
    assert Path(summary["frame_score_plot"]).exists()


def test_fast_detection_raises_instead_of_motion_fallback_on_yolo_failure(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "session"
    manifest = SessionManifest.from_dict(
        {
            "session_id": "fast_yolo_failure",
            "session_start_time": "2026-04-29T17:25:00+08:00",
            "videos": {
                "third_person": {"path": str(output_dir / "raw" / "tp.mp4"), "start_time": "2026-04-29T17:25:00+08:00", "fps": 30},
            },
            "output_dir": str(output_dir),
        }
    )
    paths = pipeline._mkdirs(output_dir)
    config = DetectorConfig(detector_backend="yolo", yolo_fallback_to_motion=True)

    def fail_yolo(*_args, **_kwargs):
        raise RuntimeError("cuda unavailable")

    def fail_motion(*_args, **_kwargs):
        raise AssertionError("fast locate must not run motion fallback after YOLO failure")

    monkeypatch.setenv("KEY_ACTION_DEFER_SEGMENT_ASSETS", "1")
    monkeypatch.setattr(pipeline, "_run_yolo_segment_detection", fail_yolo)
    monkeypatch.setattr(pipeline, "detect_key_action_segments", fail_motion)

    with pytest.raises(RuntimeError, match="cuda unavailable"):
        pipeline._detect_with_config(manifest, paths, config, dry_run=False)

    yolo_summary = json.loads((paths["metadata"] / "yolo_scan_summary.json").read_text(encoding="utf-8"))
    assert yolo_summary["fallback"] == "disabled_fast_locate"
