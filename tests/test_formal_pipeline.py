from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiment.models import Experiment, MediaAsset, PhysicalEvent
from experiment.service import ExperimentService
from labsopguard.input_layer import TimeAnchoredText
from labsopguard.preprocessing import MultiModalPreprocessor
from labsopguard.reasoning import StepGraphReasoner
from labsopguard.workflow import FormalExperimentWorkflow


def _build_test_video(path: Path, brightness_offset: int = 0) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (160, 120))
    for idx in range(30):
        frame = np.full((120, 160, 3), min(255, brightness_offset + idx * 4), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_material_change_score_catches_localized_motion(tmp_path: Path):
    import cv2
    import numpy as np

    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    base = np.full((120, 160, 3), 80, dtype=np.uint8)
    moved = base.copy()
    cv2.rectangle(moved, (70, 45), (92, 67), (230, 230, 230), -1)
    cv2.imwrite(str(first), base)
    cv2.imwrite(str(second), moved)

    score = ExperimentService._frame_change_score(str(second), str(first))

    assert score >= ExperimentService._keyframe_base_change_threshold()


def test_preprocessor_supplements_sparse_keyframes_and_expands_clips():
    material_stream = [
        SimpleNamespace(
            item_id=f"ms_{idx}",
            timestamp_sec=float(idx),
            local_timestamp_sec=float(idx),
            media_asset_id="asset_a",
            stream_id="stream_a",
            frame_id=idx,
            frame_bgr_path=f"frame_{idx}.jpg",
            scene_description="localized operation",
            object_labels=["pipette"] if idx == 2 else ["bench"],
            detected_activities=["transfer"] if idx == 2 else ["observe"],
            is_key_frame=(idx == 0),
            key_frame_reason="stream_start" if idx == 0 else None,
            change_score=0.09 if idx == 2 else 0.0,
            clip_id="stream_a:clip:1" if idx == 0 else None,
        )
        for idx in range(4)
    ]

    artifact = MultiModalPreprocessor().build_artifact(
        duration_sec=4.0,
        context_text="",
        protocol_text="",
        physical_events=[],
        material_stream=material_stream,
        video_assets=[],
        clip_window_sec=1.0,
    )

    reasons = {frame.get("key_frame_reason") for frame in artifact.key_frames}
    assert "change_peak_supplement" in reasons
    assert len(artifact.key_frames) >= 3
    assert any(clip.get("window_sec", 0.0) > 1.0 for clip in artifact.key_clips)


def test_integrated_keyframe_extractor_uses_adaptive_local_motion(tmp_path: Path):
    import cv2
    import numpy as np

    from integrated_system.keyframe_ai import extract_keyframes_by_diff

    video_path = tmp_path / "localized_motion.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for idx in range(18):
        frame = np.full((120, 160, 3), 80, dtype=np.uint8)
        if 7 <= idx <= 10:
            cv2.rectangle(frame, (68 + idx, 46), (90 + idx, 68), (230, 230, 230), -1)
        writer.write(frame)
    writer.release()

    meta, paths = extract_keyframes_by_diff(
        str(video_path),
        tmp_path / "keyframes",
        diff_threshold=18.0,
        min_interval_sec=0.2,
        max_keyframes=5,
    )

    assert len(paths) >= 2
    assert any(item.get("reason") in {"adaptive_motion", "fallback_top_motion"} for item in meta)
    assert any(0.6 <= float(item["timestamp"]) <= 1.2 for item in meta)


def test_time_alignment_generates_anchor_points():
    artifact = MultiModalPreprocessor().build_artifact(
        duration_sec=18.0,
        context_text='prepare sample\ntransfer liquid\nrecord notes',
        protocol_text='1. Prepare\n2. Transfer\n3. Record',
        physical_events=[],
        material_stream=[],
    )
    assert len(artifact.aligned_text) >= 3
    assert artifact.key_timestamps[0] == 0.0
    assert artifact.key_timestamps[-1] == 18.0


def test_step_graph_reasoner_parses_protocol_and_matches_stage():
    reasoner = StepGraphReasoner()
    nodes = reasoner.parse_protocol('1. Prepare sample\n2. Transfer liquid\n3. Observe result')
    assert [node.stage_name for node in nodes] == ['preparation', 'operation', 'observation']


def test_experiment_service_persists_frame_artifacts(tmp_path: Path):
    video_path = tmp_path / 'demo.mp4'
    _build_test_video(video_path)

    service = ExperimentService(frame_sample_interval=1.0, max_frames=8)
    service.set_video(str(video_path))
    service.set_context('operator prepares the sample and transfers liquid')
    service.set_protocol('1. Prepare sample\n2. Transfer liquid')

    result = service.process(experiment_id='formal_test_exp', experiment_title='Formal Test')
    assert result['timeline'].total_steps >= 1
    frame_paths = [Path(frame['path']) for frame in service._video_frames if frame.get('path')]
    assert frame_paths
    assert all(path.exists() for path in frame_paths)


def test_structured_output_contains_required_fields(tmp_path: Path):
    video_path = tmp_path / 'demo.mp4'
    _build_test_video(video_path)

    service = ExperimentService(frame_sample_interval=1.0, max_frames=8)
    service.set_video(str(video_path))
    service.set_context('operator prepares the sample and transfers liquid')
    service.set_protocol('1. Prepare sample\n2. Transfer liquid')
    result = service.process(experiment_id='formal_structured_exp', experiment_title='Formal Structured')

    workflow = FormalExperimentWorkflow()
    structured = workflow.build_structured_output(
        {
            'experiment_id': 'formal_structured_exp',
            'title': 'Formal Structured',
            'video_paths': [str(video_path)],
            'context_inputs': [{'text': 'operator prepares the sample and transfers liquid'}],
            'protocol_text': '1. Prepare sample\n2. Transfer liquid',
        },
        result,
    )
    assert 'steps' in structured
    assert structured['steps']
    assert 'preprocessing_layer' in structured
    assert 'video_streams' in structured['preprocessing_layer']
    assert 'time_anchored_material_stream' in structured['preprocessing_layer']
    first_step = structured['steps'][0]
    for key in ['step_id', 'step_name', 'stage_name', 'start_time', 'status', 'completion_type', 'confidence', 'provenance']:
        assert key in first_step

    dumped = json.dumps(structured, ensure_ascii=False)
    assert 'formal_structured_exp' in dumped


def test_preprocessor_builds_multi_video_time_anchored_artifact():
    asset_a = MediaAsset(
        experiment_id="exp_multi",
        filename="cam_a.mp4",
        duration_sec=6.0,
        frame_count=60,
        fps=10.0,
        metadata={"video_index": 0, "start_offset_sec": 0.0, "offset_source": "explicit", "camera_id": "cam_a"},
    )
    asset_b = MediaAsset(
        experiment_id="exp_multi",
        filename="cam_b.mp4",
        duration_sec=6.0,
        frame_count=60,
        fps=10.0,
        metadata={"video_index": 1, "start_offset_sec": 2.5, "offset_source": "explicit", "camera_id": "cam_b"},
    )
    context_records = [
        TimeAnchoredText(source_type="transcript", content="operator starts setup", timestamp_sec=0.4),
        TimeAnchoredText(
            source_type="transcript",
            content="liquid transferred in camera B",
            anchor_video_index=1,
            metadata={"local_timestamp_sec": 1.0},
        ),
    ]
    material_stream = [
        SimpleNamespace(
            item_id="ms_0",
            timestamp_sec=0.5,
            local_timestamp_sec=0.5,
            media_asset_id="asset_a",
            stream_id="asset_a",
            frame_id=0,
            local_frame_id=0,
            frame_bgr_path="frame_0.jpg",
            scene_description="setup bench",
            object_labels=["beaker"],
            detected_activities=["setup"],
            is_key_frame=True,
            key_frame_reason="stream_start",
            change_score=0.01,
            clip_id="asset_a:clip:1",
        ),
        SimpleNamespace(
            item_id="ms_1",
            timestamp_sec=3.5,
            local_timestamp_sec=1.0,
            media_asset_id="asset_b",
            stream_id="asset_b",
            frame_id=1,
            local_frame_id=10,
            frame_bgr_path="frame_1.jpg",
            scene_description="transfer liquid",
            object_labels=["pipette", "tube"],
            detected_activities=["transfer"],
            is_key_frame=True,
            key_frame_reason="visual_change",
            change_score=0.22,
            clip_id="asset_b:clip:1",
        ),
    ]
    physical_events = [
        PhysicalEvent(
            experiment_id="exp_multi",
            event_type="scene_change",
            timestamp_sec=3.5,
            confidence=0.82,
            metadata={"video_asset_id": "asset_b"},
        )
    ]

    artifact = MultiModalPreprocessor().build_artifact(
        duration_sec=8.5,
        context_text="operator starts setup\nliquid transferred in camera B",
        protocol_text="1. Setup\n2. Transfer",
        physical_events=physical_events,
        material_stream=material_stream,
        context_records=context_records,
        video_assets=[asset_a, asset_b],
    )

    assert len(artifact.video_streams) == 2
    assert artifact.alignment_summary["anchor_strategy"] == "explicit_offsets"
    assert artifact.aligned_text[0].timestamp_sec == 0.0
    assert any(abs(item.timestamp_sec - 3.5) < 1e-6 for item in artifact.aligned_text)
    assert len(artifact.key_frames) >= 2
    assert len(artifact.key_clips) >= 2
    assert artifact.time_anchored_material_stream[1]["video_asset_id"] == "asset_b"


def test_preprocessor_applies_clock_scale_to_local_text_and_stream_bounds():
    asset = MediaAsset(
        experiment_id="exp_drift",
        asset_id="asset_drift",
        filename="cam_drift.mp4",
        duration_sec=10.0,
        frame_count=100,
        fps=10.0,
        metadata={
            "video_index": 0,
            "camera_id": "cam_drift",
            "start_offset_sec": 2.0,
            "clock_scale": 1.01,
            "clock_drift_ppm": 10000.0,
            "offset_source": "explicit",
            "sync_profile": {
                "offset_sec": 2.0,
                "clock_scale": 1.01,
                "drift_ppm": 10000.0,
                "method": "calibrated:sync_board",
                "confidence": 0.95,
            },
        },
    )
    context_records = [
        TimeAnchoredText(
            source_type="transcript",
            content="local transcript",
            anchor_video_index=0,
            metadata={"local_timestamp_sec": 10.0},
        )
    ]

    artifact = MultiModalPreprocessor().build_artifact(
        duration_sec=12.1,
        context_text="",
        protocol_text="",
        physical_events=[],
        material_stream=[],
        context_records=context_records,
        video_assets=[asset],
    )

    assert artifact.video_streams[0]["end_offset_sec"] == 12.1
    assert any(abs(item.timestamp_sec - 12.1) < 1e-6 for item in artifact.aligned_text)


def test_experiment_service_processes_multiple_videos(tmp_path: Path):
    video_a = tmp_path / "cam_a.mp4"
    video_b = tmp_path / "cam_b.mp4"
    _build_test_video(video_a, brightness_offset=10)
    _build_test_video(video_b, brightness_offset=120)

    service = ExperimentService(frame_sample_interval=1.0, max_frames=6)
    service.set_videos([str(video_a), str(video_b)])
    service.set_context_inputs(
        [
            {"text": "operator starts setup", "kind": "transcript", "timestamp_sec": 0.5},
            {"text": "camera B sees transfer", "kind": "transcript", "local_timestamp_sec": 1.0, "video_index": 1},
        ]
    )
    service.set_context("operator starts setup\ncamera B sees transfer")
    service.set_protocol("1. Setup\n2. Transfer")

    result = service.process(experiment_id="formal_multi_exp", experiment_title="Formal Multi")
    experiment = result["experiment"]
    material_stream = result["material_stream"]

    assert len(experiment.video_assets) == 2
    assert experiment.metadata["video_stream_count"] == 2
    assert len(material_stream) >= 2
    assert len({item.media_asset_id for item in material_stream}) == 2
    timestamps = [item.timestamp_sec for item in material_stream]
    assert timestamps == sorted(timestamps)
    assert all(item.local_timestamp_sec is not None for item in material_stream)
    assert any(item.is_key_frame for item in material_stream)


def test_experiment_service_auto_aligns_visual_flash_between_streams(tmp_path: Path, monkeypatch):
    import cv2
    import numpy as np

    def build_flash_video(path: Path, flash_frame: int) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (160, 120))
        for idx in range(30):
            value = 235 if idx == flash_frame else 35
            frame = np.full((120, 160, 3), value, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    video_a = tmp_path / "cam_a_flash.mp4"
    video_b = tmp_path / "cam_b_flash.mp4"
    build_flash_video(video_a, flash_frame=10)
    build_flash_video(video_b, flash_frame=5)

    service = ExperimentService(frame_sample_interval=0.5, max_frames=12)
    monkeypatch.setattr(service._step_reasoner, "analyze_frame", lambda path: {"scene_description": "sync flash", "object_labels": []})
    service.set_video_inputs(
        [
            {"video_index": 0, "video_path": str(video_a), "source_type": "file", "camera_id": "cam_a", "sync_group": "run_flash"},
            {"video_index": 1, "video_path": str(video_b), "source_type": "file", "camera_id": "cam_b", "sync_group": "run_flash"},
        ]
    )

    result = service.process(experiment_id="formal_auto_sync_exp", experiment_title="Auto Sync")
    streams = result["timeline"].metadata["video_streams"]
    by_camera = {stream["camera_id"]: stream for stream in streams}

    assert result["experiment"].metadata["time_alignment_mode"] == "calibrated"
    assert by_camera["cam_a"]["alignment_method"] == "auto_visual_reference"
    assert by_camera["cam_b"]["alignment_method"].startswith("auto_visual")
    assert by_camera["cam_b"]["start_offset_sec"] == 0.5


def test_experiment_service_applies_multimodal_semantic_sync(monkeypatch):
    monkeypatch.setenv("LABSOPGUARD_MULTIMODAL_SEMANTIC_SYNC", "1")
    monkeypatch.setenv("LABSOPGUARD_MULTIMODAL_SEMANTIC_SYNC_APPLY", "1")

    service = ExperimentService(frame_sample_interval=1.0, max_frames=4)
    service._video_info = {
        "streams": [
            {
                "camera_id": "usbmain",
                "asset_id": "asset_usbmain",
                "view_type": "first_person",
                "source_group": "multi_monitor",
                "duration_sec": 30.0,
                "alignment_method": "shared_recording_session",
                "alignment_confidence": 0.85,
            },
            {
                "camera_id": "wireless_1",
                "asset_id": "asset_wireless_1",
                "view_type": "third_person",
                "source_group": "multi_monitor",
                "duration_sec": 30.0,
                "alignment_method": "pending",
                "alignment_confidence": 0.0,
            },
        ]
    }
    service._video_frames = [
        {"camera_id": "usbmain", "view_type": "first_person", "source_group": "multi_monitor", "local_timestamp_sec": 10.0, "timestamp_sec": 10.0, "frame_id": 1},
        {"camera_id": "wireless_1", "view_type": "third_person", "source_group": "multi_monitor", "local_timestamp_sec": 7.5, "timestamp_sec": 7.5, "frame_id": 2},
        {"camera_id": "usbmain", "view_type": "first_person", "source_group": "multi_monitor", "local_timestamp_sec": 20.0, "timestamp_sec": 20.0, "frame_id": 3},
        {"camera_id": "wireless_1", "view_type": "third_person", "source_group": "multi_monitor", "local_timestamp_sec": 17.5, "timestamp_sec": 17.5, "frame_id": 4},
    ]
    service._frame_analyses = [
        {"scene_description": "operator starts setup and prepares the sample", "object_labels": ["gloves"], "confidence": 0.9},
        {"scene_description": "operator starts setup at the bench", "object_labels": ["person"], "confidence": 0.88},
        {"scene_description": "operator uses pipette to transfer liquid into tube", "object_labels": ["pipette", "tube"], "detected_activities": ["liquid transfer"], "confidence": 0.92},
        {"scene_description": "liquid transfer with pipette is visible", "object_labels": ["pipette"], "detected_activities": ["transfer"], "confidence": 0.86},
    ]
    experiment = Experiment(experiment_id="semantic_service_exp", title="Semantic Service")

    result = service._run_multimodal_semantic_sync(experiment)

    assert result["status"] == "calibrated"
    assert result["reference_stream"]["camera_id"] == "usbmain"
    assert experiment.metadata["semantic_sync"]["applied"] is True
    assert experiment.metadata["time_alignment_mode"] == "calibrated"

    wireless_stream = next(stream for stream in service._video_info["streams"] if stream["camera_id"] == "wireless_1")
    assert wireless_stream["alignment_method"] == "multimodal_semantic"
    assert wireless_stream["start_offset_sec"] == 2.5
    assert wireless_stream["semantic_sync_anchor_count"] >= 2

    wireless_frames = [frame for frame in service._video_frames if frame["camera_id"] == "wireless_1"]
    assert [frame["timestamp_sec"] for frame in wireless_frames] == [10.0, 20.0]
    assert all(frame["alignment_method"] == "multimodal_semantic" for frame in wireless_frames)


def test_experiment_service_materializes_key_clips_on_save(tmp_path: Path):
    video_path = tmp_path / "demo.mp4"
    _build_test_video(video_path)

    service = ExperimentService(frame_sample_interval=1.0, max_frames=8)
    service.set_video(str(video_path))
    service.set_context("operator prepares sample")
    service.set_protocol("1. Prepare sample")
    service.process(experiment_id="formal_clip_exp", experiment_title="Formal Clip")

    output_dir = tmp_path / "outputs"
    paths = service.save_outputs(output_dir=str(output_dir))
    payload = json.loads(Path(paths["preprocessing"]).read_text(encoding="utf-8"))

    assert payload["key_clips"]
    rendered = [clip for clip in payload["key_clips"] if clip.get("file_exists")]
    assert rendered
    assert all(Path(clip["file_path"]).exists() for clip in rendered)
    assert payload["alignment_summary"]["materialized_key_clip_count"] >= 1


def test_experiment_service_accepts_rtsp_style_stream_source(tmp_path: Path):
    video_path = tmp_path / "stream_source.mp4"
    _build_test_video(video_path, brightness_offset=40)

    service = ExperimentService(frame_sample_interval=1.0, max_frames=5)
    service.set_video_inputs(
        [
            {
                "video_index": 0,
                "video_path": str(video_path),
                "source_type": "rtsp",
                "camera_id": "camera_stream",
                "capture_duration_sec": 4.0,
                "start_offset_sec": 0.0,
            }
        ]
    )
    service.set_context_inputs(
        [
            {"text": "stream transcript event", "kind": "transcript", "timestamp_sec": 1.0},
        ]
    )
    service.set_context("stream transcript event")
    service.set_protocol("1. Observe stream")

    result = service.process(experiment_id="formal_stream_exp", experiment_title="Formal Stream")
    experiment = result["experiment"]
    material_stream = result["material_stream"]

    assert len(experiment.video_assets) == 1
    assert experiment.video_assets[0].metadata["source_type"] == "rtsp"
    assert experiment.video_assets[0].metadata["is_live_source"] is True
    assert len(material_stream) >= 1
    assert all(item.media_asset_id == experiment.video_assets[0].asset_id for item in material_stream)
