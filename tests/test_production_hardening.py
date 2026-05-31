from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.asr_jobs import split_audio_for_asr
from labsopguard.clip_backfill import backfill_clip_from_material_stream
from labsopguard.device_registry import video_inputs_from_registry
from labsopguard.material_maintenance import rebuild_workspace_material_index, scan_experiment_material_health
from labsopguard.semantic_events import SemanticEventDetector
from labsopguard.stream_buffer import RingSegmentRecorder
from labsopguard.sync_calibration import build_sync_calibration_report
from labsopguard.workspace_governance import build_workspace_governance_report


def _write_test_wav(path: Path, seconds: float = 1.0, sample_rate: int = 8000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        for i in range(int(sample_rate * seconds)):
            value = int(1200 * math.sin(2 * math.pi * 440 * i / sample_rate))
            wav.writeframes(struct.pack("<h", value))


def test_asr_wav_split_for_long_jobs(tmp_path: Path):
    wav_path = tmp_path / "input.wav"
    _write_test_wav(wav_path, seconds=1.2)

    chunks = split_audio_for_asr(wav_path, tmp_path / "chunks", chunk_duration_sec=0.4, force_chunk=True)

    assert len(chunks) >= 3
    assert all(chunk.exists() for chunk in chunks)


def test_clip_backfill_from_material_stream_segments(tmp_path: Path):
    recorder = RingSegmentRecorder(
        camera_id="cam_a",
        source_id="prod",
        output_dir=tmp_path / "segments" / "cam_a",
        segment_duration_sec=0.5,
        retention_sec=10,
        fps=5,
    )
    for idx in range(12):
        recorder.append_frame(np.full((48, 64, 3), idx * 10, dtype=np.uint8), idx / 5.0)
    recorder.close()
    material_stream = [
        {
            "item_id": segment.segment_id,
            "camera_id": "cam_a",
            "timestamp_sec": segment.start_time_sec,
            "end_time_sec": segment.end_time_sec,
            "recorded_file_path": segment.file_path,
            "fps": segment.fps,
        }
        for segment in recorder.segments
    ]
    stream_path = tmp_path / "material_stream.json"
    stream_path.write_text(json.dumps(material_stream), encoding="utf-8")

    clip = backfill_clip_from_material_stream(
        stream_path,
        camera_id="cam_a",
        start_time_sec=0.4,
        end_time_sec=1.4,
        output_path=tmp_path / "clips" / "backfill.mp4",
    )

    assert clip.file_exists
    assert clip.source_segment_count >= 1


def test_workspace_material_index_rebuild_and_health_scan(tmp_path: Path):
    exp_dir = tmp_path / "experiments" / "exp_a"
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.json").write_text(json.dumps({"experiment_id": "exp_a"}), encoding="utf-8")
    (exp_dir / "material_stream.json").write_text(
        json.dumps(
            [
                {
                    "item_id": "item_a",
                    "experiment_id": "exp_a",
                    "timestamp_sec": 1.0,
                    "camera_id": "cam_a",
                    "object_labels": ["pipette"],
                    "detected_activities": ["transfer"],
                    "scene_description": "pipette transfer",
                }
            ]
        ),
        encoding="utf-8",
    )
    (exp_dir / "preprocessing.json").write_text(json.dumps({"key_clips": []}), encoding="utf-8")

    report = rebuild_workspace_material_index(tmp_path / "experiments", tmp_path / "workspace.sqlite")
    health = scan_experiment_material_health(tmp_path / "experiments")

    assert report["health"]["total_items"] == 1
    assert health["total_items"] == 1


def test_sync_calibration_report_uses_hardware_and_drift_fields():
    report = build_sync_calibration_report(
        [
            {
                "camera_id": "cam_a",
                "hardware_timecode_start_sec": 1.0,
                "clock_drift_ppm": 1000,
                "capture_duration_sec": 100,
            }
        ]
    )

    profile = report["profiles"][0]
    assert report["anchor_count"] == 2
    assert profile["method"].startswith("calibrated")
    assert profile["drift_ppm"] > 0


def test_semantic_event_detector_uses_keypoints_and_depth_for_contact():
    detector = SemanticEventDetector()
    events = detector.update(
        1.0,
        [
            {"label": "hand", "bbox": [0, 0, 10, 10], "hand_keypoints": [[42, 42]], "depth": 0.5},
            {"label": "tube", "bbox": [35, 35, 55, 55], "depth": 0.54},
        ],
        frame_metadata={"camera_id": "cam_a"},
    )

    contact = [event for event in events if event["event_type"] == "hand_contact_geometry"]
    assert contact
    assert contact[0]["metadata"]["keypoint_contact_verified"] is True
    assert contact[0]["metadata"]["depth_contact_verified"] is True


def test_device_registry_builds_video_inputs():
    video_inputs = video_inputs_from_registry(
        {
            "cameras": [
                {
                    "camera_id": "cam_a",
                    "enabled": True,
                    "source_type": "rtsp",
                    "source": "rtsp://example.invalid/live",
                    "sync_group": "main",
                    "hardware_timecode_start_sec": 10,
                }
            ]
        }
    )

    assert video_inputs[0]["camera_id"] == "cam_a"
    assert video_inputs[0]["source_type"] == "rtsp"
    assert video_inputs[0]["hardware_timecode_start_sec"] == 10


def test_workspace_governance_report_classifies_roots(tmp_path: Path):
    (tmp_path / "LabSOPGuard").mkdir()
    (tmp_path / "lab_preprocessing").mkdir()
    report = build_workspace_governance_report(tmp_path)

    roles = {entry["role"] for entry in report["entries"]}
    assert "primary_project" in roles
    assert "legacy_or_upstream_preprocessing_project" in roles
