from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as backend_main


def setup_isolated_project_root(tmp_path: Path) -> None:
    backend_main.PROJECT_ROOT = tmp_path
    backend_main._EXPERIMENTS.clear()
    (tmp_path / "outputs" / "experiments").mkdir(parents=True, exist_ok=True)
    (tmp_path / "uploads" / "experiments").mkdir(parents=True, exist_ok=True)


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_created_experiment_alignment_artifacts_stay_pending(tmp_path: Path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    created = client.post("/api/v1/experiments", json={"title": "Alignment pending"}).json()
    experiment_id = created["experiment_id"]

    detail = client.get(f"/api/v1/experiments/{experiment_id}").json()
    assert detail["status"] == "waiting_for_sources"

    exp_dir = backend_main._experiment_output_dir(experiment_id)
    timeline_alignment = json.loads((exp_dir / "timeline_alignment.json").read_text(encoding="utf-8"))
    stream_manifest = json.loads((exp_dir / "stream_manifest.json").read_text(encoding="utf-8"))

    assert timeline_alignment["alignment_status"] == "pending"
    assert timeline_alignment["streams"] == []
    assert stream_manifest["alignment_status"] == "pending"
    assert stream_manifest["registered_streams"] == []
    assert (exp_dir / "material_stream.v2.jsonl").exists()
    assert (exp_dir / "semantic_sync_anchors.json").exists()


def test_stream_upload_persists_explicit_alignment_fields(tmp_path: Path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)
    experiment_id = client.post("/api/v1/experiments", json={"title": "Explicit alignment"}).json()["experiment_id"]

    response = client.post(
        f"/api/v1/experiments/{experiment_id}/upload/stream",
        json={
            "source": str(tmp_path / "cam_a.mp4"),
            "source_type": "rtsp",
            "camera_id": "cam_a",
            "sync_method": "audio_flash",
            "sync_anchors": [{"local_time_sec": 0.0, "reference_time_sec": 1.25, "method": "audio_flash", "confidence": 0.92}],
            "clock_scale": 1.000012,
            "clock_drift_ppm": 12.0,
            "sync_confidence": 0.91,
        },
    )

    assert response.status_code == 200
    stream = response.json()["stream"]
    assert stream["sync_confidence"] == 0.91
    assert stream["clock_scale"] == 1.000012

    exp_dir = backend_main._experiment_output_dir(experiment_id)
    alignment = json.loads((exp_dir / "timeline_alignment.json").read_text(encoding="utf-8"))
    manifest = json.loads((exp_dir / "stream_manifest.json").read_text(encoding="utf-8"))
    alignment_stream = alignment["streams"][0]
    manifest_stream = manifest["registered_streams"][0]

    assert alignment["alignment_status"] == "explicit"
    assert manifest["alignment_status"] == "explicit"
    assert alignment_stream["alignment_status"] == "explicit"
    assert alignment_stream["alignment_method"] == "audio_flash"
    assert alignment_stream["sync_confidence"] == 0.91
    assert alignment_stream["sync_anchor_count"] == 1
    assert alignment_stream["clock_drift_ppm"] == 12.0
    assert manifest_stream["alignment_status"] == "explicit"
    assert manifest_stream["sync_anchors"][0]["reference_time_sec"] == 1.25


def test_alignment_update_rebuilds_calibrated_transcript_and_material_artifacts(tmp_path: Path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)
    experiment_id = client.post("/api/v1/experiments", json={"title": "Calibrated alignment"}).json()["experiment_id"]
    client.post(
        f"/api/v1/experiments/{experiment_id}/upload/stream",
        json={
            "source": str(tmp_path / "cam_b.mp4"),
            "source_type": "rtsp",
            "camera_id": "cam_b",
        },
    )

    update = client.post(
        f"/api/v1/experiments/{experiment_id}/timeline-alignment",
        json={
            "streams": [
                {
                    "camera_id": "cam_b",
                    "alignment_status": "calibrated",
                    "sync_method": "visual_calibration",
                    "sync_confidence": 0.97,
                    "clock_scale": 1.000001,
                    "clock_drift_ppm": 1.0,
                    "sync_anchors": [{"local_time_sec": 0, "reference_time_sec": 2, "method": "flash"}],
                }
            ]
        },
    )

    assert update.status_code == 200
    alignment_stream = update.json()["timeline_alignment"]["streams"][0]
    assert alignment_stream["alignment_status"] == "calibrated"
    assert alignment_stream["alignment_confidence"] == 0.97

    exp_path = backend_main._experiment_output_dir(experiment_id) / "experiment.json"
    exp = backend_main._load_json_if_exists(exp_path)
    exp["context_inputs"] = [
        {
            "text": "operator says start",
            "timestamp_sec": 2.1,
            "local_timestamp_sec": 0.1,
            "camera_id": "cam_b",
            "sync_method": "visual_calibration",
            "sync_confidence": 0.97,
        }
    ]
    backend_main._save_experiment(exp)
    output_paths = backend_main._write_experiment_run_artifacts(
        experiment_id,
        exp,
        material_stream=[
            {
                "item_id": "mat_1",
                "timestamp_sec": 2.2,
                "local_timestamp_sec": 0.2,
                "camera_id": "cam_b",
                "object_labels": ["beaker"],
            }
        ],
    )

    transcript_rows = _jsonl_rows(Path(output_paths["transcript_segments_jsonl"]))
    material_rows = _jsonl_rows(Path(output_paths["material_stream_v2_jsonl"]))
    assert transcript_rows[0]["alignment_status"] == "calibrated"
    assert transcript_rows[0]["sync_confidence"] == 0.97
    assert material_rows[0]["alignment_status"] == "calibrated"
    assert material_rows[0]["alignment_method"] == "visual_calibration"
    assert material_rows[0]["clock_scale"] == 1.000001


def test_semantic_sync_anchors_artifact_and_overview_contract(tmp_path: Path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)
    experiment_id = client.post("/api/v1/experiments", json={"title": "Semantic sync"}).json()["experiment_id"]

    response = client.post(
        f"/api/v1/experiments/{experiment_id}/upload/stream",
        json={
            "source": str(tmp_path / "semantic_cam.mp4"),
            "source_type": "rtsp",
            "camera_id": "semantic_cam",
            "sync_method": "multimodal_semantic",
            "sync_confidence": 0.88,
            "sync_anchors": [
                {
                    "local_time_sec": 3.0,
                    "reference_time_sec": 9.5,
                    "method": "multimodal_semantic",
                    "confidence": 0.88,
                    "semantic_label": "operator picks up beaker",
                    "description": "vision and transcript both mention beaker pickup",
                }
            ],
        },
    )
    assert response.status_code == 200

    exp_dir = backend_main._experiment_output_dir(experiment_id)
    alignment = json.loads((exp_dir / "timeline_alignment.json").read_text(encoding="utf-8"))
    alignment_stream = alignment["streams"][0]
    assert alignment_stream["alignment_method"] == "multimodal_semantic"
    assert alignment_stream["has_semantic_sync"] is True
    assert alignment_stream["semantic_sync_anchor_count"] == 1

    artifact_response = client.get(f"/api/v1/experiments/{experiment_id}/artifacts/semantic_sync_anchors_json")
    assert artifact_response.status_code == 200
    semantic_payload = artifact_response.json()
    assert semantic_payload["schema_version"] == "semantic_sync_anchors.v1"
    assert semantic_payload["total"] == 1
    assert semantic_payload["anchors"][0]["method"] == "multimodal_semantic"
    assert semantic_payload["anchors"][0]["semantic_label"] == "operator picks up beaker"

    overview = client.get(f"/api/v1/experiments/{experiment_id}/analysis-overview").json()
    semantic_artifact = overview["artifacts"]["semantic_sync_anchors"]
    assert semantic_artifact["ready"] is True
    assert semantic_artifact["url"].endswith("/artifacts/semantic_sync_anchors_json")


def test_run_artifacts_preserve_full_multimodal_semantic_sync_payload(tmp_path: Path):
    setup_isolated_project_root(tmp_path)
    experiment_id = "exp_semantic_full"
    exp = {
        "experiment_id": experiment_id,
        "title": "Semantic sync full payload",
        "status": "analyzed",
        "run_id": "run_semantic_full",
        "video_inputs": [
            {
                "camera_id": "usbmain",
                "video_index": 0,
                "view_type": "first_person",
                "alignment_method": "shared_recording_session",
            },
            {
                "camera_id": "wireless_1",
                "video_index": 1,
                "view_type": "third_person",
                "alignment_method": "multimodal_semantic",
                "sync_profile": {
                    "method": "calibrated:multimodal_semantic",
                    "offset_sec": 2.5,
                    "clock_scale": 1.0,
                    "confidence": 0.86,
                    "anchor_count": 2,
                },
            },
        ],
    }
    semantic_sync = {
        "schema_version": "multimodal_semantic_sync.v1",
        "experiment_id": experiment_id,
        "run_id": "run_semantic_full",
        "status": "calibrated",
        "reference_stream": {"camera_id": "usbmain", "view_type": "first_person"},
        "semantic_events": [
            {
                "event_id": "sem_evt_0001",
                "event_type": "liquid_transfer",
                "matched_streams": [{"camera_id": "usbmain"}, {"camera_id": "wireless_1"}],
            }
        ],
        "sync_anchors": [
            {
                "camera_id": "wireless_1",
                "reference_camera_id": "usbmain",
                "local_time_sec": 7.5,
                "reference_time_sec": 10.0,
                "method": "multimodal_semantic",
                "confidence": 0.86,
                "event_id": "sem_evt_0001",
            }
        ],
    }

    output_paths = backend_main._write_experiment_run_artifacts(
        experiment_id,
        exp,
        semantic_sync=semantic_sync,
    )

    payload = json.loads(Path(output_paths["semantic_sync_anchors_json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == "multimodal_semantic_sync.v1"
    assert payload["semantic_events"][0]["event_type"] == "liquid_transfer"
    assert payload["sync_anchors"][0]["method"] == "multimodal_semantic"
    assert payload["anchors"] == payload["sync_anchors"]

    alignment = json.loads(Path(output_paths["timeline_alignment_json"]).read_text(encoding="utf-8"))
    third_person = next(stream for stream in alignment["streams"] if stream["camera_id"] == "wireless_1")
    assert third_person["alignment_status"] == "calibrated"
    assert third_person["has_semantic_sync"] is True
    assert third_person["semantic_sync_anchor_count"] == 2
