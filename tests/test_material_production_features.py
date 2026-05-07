from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment.service import ExperimentService
from labsopguard.retrieval import MaterialQuery, MaterialRetrievalIndex
from labsopguard.semantic_events import SemanticEventDetector
from labsopguard.stream_buffer import RingSegmentRecorder
from labsopguard.time_sync import SyncAnchor, TimeSyncCalibrator
from labsopguard.video_input_schema import VideoInputValidationError, normalize_video_input
from labsopguard.asr import TranscriptSegment


def _build_test_video(path: Path) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 72))
    for idx in range(24):
        frame = np.full((72, 96, 3), min(255, 20 + idx * 8), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_time_sync_calibrator_fits_offset_and_drift():
    anchors = [
        SyncAnchor(camera_id="cam_b", local_time_sec=0.0, reference_time_sec=2.0, method="sync_board"),
        SyncAnchor(camera_id="cam_b", local_time_sec=100.0, reference_time_sec=102.1, method="sync_board"),
    ]

    profile = TimeSyncCalibrator.fit_profile_from_anchors("cam_b", anchors, reference_camera_id="cam_a")

    assert profile.offset_sec == 2.0
    assert 999.0 <= profile.drift_ppm <= 1001.0
    assert abs(profile.local_to_global(50.0) - 52.05) < 1e-6


def test_video_input_schema_normalizes_and_rejects_invalid_sources():
    normalized, warnings = normalize_video_input(
        {
            "source": "rtsp://10.0.0.2/live",
            "source_type": "rtsp",
            "capture_duration_sec": 30,
            "sync_anchors": [{"local_time_sec": "0", "reference_time_sec": "1.2", "method": "audio_flash"}],
        },
        index=2,
        strict=True,
    )

    assert normalized["schema_version"] == "video_input.v1"
    assert normalized["camera_id"] == "camera_02"
    assert normalized["is_live_source"] is True
    assert normalized["offset_source"] == "default_zero"
    assert normalized["sync_anchors"][0]["reference_time_sec"] == 1.2
    assert warnings == ["camera_id was generated from video_index"]

    explicit, _ = normalize_video_input(
        {"source": "rtsp://10.0.0.2/live", "source_type": "rtsp", "start_offset_sec": 0.0},
        index=0,
        strict=True,
    )
    assert explicit["offset_source"] == "explicit"

    with pytest.raises(VideoInputValidationError):
        normalize_video_input({"source": "bad", "source_type": "bad_type"}, strict=True)


def test_transcript_segment_maps_to_context_input():
    segment = TranscriptSegment(text="开始移液", start_time_sec=1.25, end_time_sec=2.5, speaker="operator")

    payload = segment.to_context_input()

    assert payload["kind"] == "transcript"
    assert payload["source_type"] == "asr"
    assert payload["timestamp_sec"] == 1.25
    assert payload["end_time_sec"] == 2.5
    assert payload["speaker"] == "operator"


def test_ring_segment_recorder_backfills_clip(tmp_path: Path):
    recorder = RingSegmentRecorder(
        camera_id="cam_a",
        source_id="test",
        output_dir=tmp_path / "buffer",
        segment_duration_sec=0.5,
        retention_sec=5.0,
        fps=5.0,
    )
    for idx in range(10):
        frame = np.full((48, 64, 3), idx * 20, dtype=np.uint8)
        recorder.append_frame(frame, idx / 5.0)
    recorder.close()

    clip_path = recorder.cut_clip(0.2, 1.2, tmp_path / "clips" / "history.mp4")

    assert clip_path is not None
    assert Path(clip_path).exists()
    assert len(recorder.segments_for_range(0.2, 1.2)) >= 1


def test_semantic_event_detector_tracks_contacts_and_state_changes():
    detector = SemanticEventDetector()
    first = detector.update(
        0.0,
        [
            {"label": "beaker", "bbox": [10, 10, 60, 80]},
            {"label": "cap", "bbox": [20, 8, 50, 20]},
            {"label": "liquid", "bbox": [15, 40, 55, 78], "liquid_level": 0.35},
            {"label": "reagent_label", "bbox": [70, 10, 95, 40], "ocr_text": "NaCl"},
        ],
        frame_metadata={"camera_id": "cam_a"},
    )
    second = detector.update(
        1.0,
        [
            {"label": "beaker", "bbox": [25, 10, 75, 80]},
            {"label": "hand", "bbox": [20, 15, 70, 90]},
            {"label": "liquid", "bbox": [30, 25, 70, 78], "liquid_level": 0.55},
            {"label": "reagent_label", "bbox": [70, 10, 95, 40], "ocr_text": "NaCl"},
        ],
        frame_metadata={"camera_id": "cam_a"},
    )

    event_types = {event["event_type"] for event in first + second}
    assert "reagent_label_state" in event_types
    assert "container_opened" in event_types
    assert "hand_contact_geometry" in event_types
    assert "liquid_level_change" in event_types


def test_material_retrieval_filters_and_embedding(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MATERIAL_EMBEDDING_PROVIDER", "hash")
    clip_file = tmp_path / "clip.mp4"
    clip_file.write_bytes(b"fake")
    index = MaterialRetrievalIndex(tmp_path / "material.sqlite")
    try:
        index.reset()
        index.index_payloads(
            [
                {
                    "item_id": "item_1",
                    "experiment_id": "exp",
                    "timestamp_sec": 3.2,
                    "local_timestamp_sec": 1.2,
                    "camera_id": "cam_a",
                    "stream_id": "stream_a",
                    "video_asset_id": "asset_a",
                    "frame_id": 10,
                    "frame_bgr_path": "frame.jpg",
                    "clip_id": "clip_1",
                    "object_labels": ["pipette", "tube"],
                    "detected_activities": ["transfer"],
                    "scene_description": "pipette transfers liquid into tube",
                },
                {
                    "item_id": "item_2",
                    "experiment_id": "exp",
                    "timestamp_sec": 6.0,
                    "camera_id": "cam_b",
                    "clip_id": "clip_2",
                    "object_labels": ["beaker"],
                    "detected_activities": ["inspect"],
                    "scene_description": "beaker inspection",
                },
            ],
            preprocessing={
                "key_clips": [
                    {
                        "clip_id": "clip_1",
                        "file_path": str(clip_file),
                        "file_exists": True,
                        "reason": "visual_change",
                    },
                    {
                        "clip_id": "clip_2",
                        "file_path": str(tmp_path / "missing.mp4"),
                        "file_exists": True,
                        "reason": "missing_file",
                    }
                ]
            },
            experiment_id="exp",
        )

        rows = index.query(
            MaterialQuery(
                objects=["pipette"],
                actions=["transfer"],
                start_time_sec=0.0,
                end_time_sec=5.0,
                camera_id="cam_a",
                clip_exists=True,
                embedding_text="liquid transfer",
            )
        )

        assert len(rows) == 1
        assert rows[0]["clip_file_path"] == str(clip_file)
        assert rows[0]["embedding_score"] > 0
        health = index.health_check()
        assert health["materialized_clip_count"] == 1
        assert health["broken_clip_reference_count"] == 1
        assert health["embedding_mode"] == "hash_embedding_v1"
    finally:
        index.close()


def test_experiment_service_saves_material_index_and_sync_metadata(tmp_path: Path):
    video_path = tmp_path / "stream_source.mp4"
    _build_test_video(video_path)
    service = ExperimentService(frame_sample_interval=0.5, max_frames=6)
    service.set_video_inputs(
        [
            {
                "video_index": 0,
                "video_path": str(video_path),
                "source_type": "rtsp",
                "camera_id": "cam_sync",
                "capture_duration_sec": 2.0,
                "sync_anchors": [
                    {"local_time_sec": 0.0, "reference_time_sec": 1.0, "method": "audio_flash"},
                    {"local_time_sec": 2.0, "reference_time_sec": 3.01, "method": "audio_flash"},
                ],
            }
        ]
    )
    service.set_context("operator transfers liquid")
    service.set_protocol("1. Transfer liquid")
    service.process(experiment_id="prod_features_exp")

    paths = service.save_outputs(output_dir=str(tmp_path / "outputs"))
    preprocessing = json.loads(Path(paths["preprocessing"]).read_text(encoding="utf-8"))

    assert Path(paths["material_index"]).exists()
    assert preprocessing["schema_version"] == "preprocessing.v1"
    assert preprocessing["alignment_summary"]["anchor_strategy"] == "calibrated"
    assert preprocessing["alignment_summary"]["stream_health"][0]["frames_sampled"] > 0
    assert preprocessing["video_streams"][0]["recorded_file_path"]
    assert preprocessing["video_streams"][0]["stream_health"]["frames_recorded"] > 0
    assert preprocessing["material_index_health"]["total_items"] > 0
    assert Path(preprocessing["video_streams"][0]["recorded_file_path"]).exists()
