from __future__ import annotations

import json
import hashlib
import os
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import backend.main as backend_main
from experiment.models import (
    Experiment,
    ExperimentStatus,
    ExperimentTimeline,
    ProcessStage,
    ProvenanceInfo,
    StepConfidence,
    StepRecord,
    StepStatus,
)


def test_formal_key_action_runtime_rejects_cpu_fallback():
    config = {
        'detector_backend': 'yolo',
        'yolo_device': 'cpu',
        'require_cuda': True,
    }

    with pytest.raises(backend_main.KeyActionCudaRequiredError):
        backend_main._validate_key_action_yolo_runtime(
            config,
            runtime_info={
                'cuda_available': False,
                'cuda_device_count': 0,
                'python_executable': 'python',
                'torch_version': 'cpu',
                'torch_cuda': None,
            },
        )


def test_key_action_timing_display_buckets_include_formal_delivery_stages():
    payload = backend_main._finalize_key_action_timing_payload(
        {
            'elapsed_sec': 12.0,
            'upload_save_sec': 1.0,
            'stages': [
                {'stage': 'time_alignment_preflight', 'duration_sec': 0.2},
                {'stage': 'yolo_fast_locate_wall', 'duration_sec': 2.0},
                {'stage': 'fine_scan_dispatch', 'duration_sec': 3.0},
                {'stage': 'segment_projection', 'duration_sec': 0.4},
                {'stage': 'micro_clip_keyframe_generation', 'duration_sec': 0.5},
                {'stage': 'quality_gate', 'duration_sec': 0.1},
                {'stage': 'material_candidate_generation', 'duration_sec': 0.6},
                {'stage': 'material_auto_publish', 'duration_sec': 0.7},
                {'stage': 'video_memory_write', 'duration_sec': 0.8},
            ],
        }
    )

    labels = {row['stage']: row['label_zh'] for row in payload['display_stages']}
    assert labels['action_alignment'] == '动作对齐'
    assert labels['micro_gate'] == 'Micro/Gate'
    assert labels['material_generation'] == '素材生成'
    assert labels['material_publish'] == '素材发布'
    assert labels['memory_write'] == 'Memory 写入'
    assert payload['timing_buckets']['memory_write'] == 0.8
    assert payload['timing_buckets']['micro_gate'] == 0.6


class FakeExperimentService:
    def __init__(self, *args, **kwargs):
        self.video = None
        self.video_inputs = None
        self.context = None
        self.context_inputs = None
        self.protocol = None

    def set_video(self, video: str):
        self.video = video

    def set_video_inputs(self, video_inputs):
        self.video_inputs = video_inputs

    def set_context(self, context: str):
        self.context = context

    def set_context_inputs(self, context_inputs):
        self.context_inputs = context_inputs

    def set_protocol(self, protocol: str):
        self.protocol = protocol

    def process(self, experiment_id: str, experiment_title: str):
        step = StepRecord(
            experiment_id=experiment_id,
            step_index=1,
            step_name='Observed transfer',
            step_description='Observed transfer event',
            status=StepStatus.CONFIRMED,
            start_time_sec=0.0,
            end_time_sec=4.0,
            duration_sec=4.0,
            confidence=0.95,
            step_confidence=StepConfidence.HIGH,
            completed_by_inference=False,
            evidence_refs=[],
            parameters=[],
            provenance=ProvenanceInfo(source='video', confidence=0.95, is_inferred=False),
        )
        timeline = ExperimentTimeline(
            experiment_id=experiment_id,
            title=experiment_title,
            steps=[step],
            processing_stage=ProcessStage.OUTPUT_GENERATION,
        )
        timeline.compute_stats()
        experiment = Experiment(
            experiment_id=experiment_id,
            title=experiment_title,
            status=ExperimentStatus.ANALYZED,
            timeline=timeline,
        )
        experiment.sync_stats()
        return {
            'experiment': experiment,
            'timeline': timeline,
            'steps': [step],
            'physical_events': [],
            'material_stream': [],
        }


def _build_local_test_video(path: Path) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (120, 90))
    for idx in range(20):
        frame = np.full((90, 120, 3), idx * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def setup_isolated_project_root(tmp_path: Path):
    backend_main.PROJECT_ROOT = tmp_path
    backend_main.VIDEO_STORE_ROOT = tmp_path / 'LabVideoStore' / 'raw_uploads'
    os.environ['LAB_VIDEO_STORE_ENABLE_BUILTIN_LEGACY_ROOT'] = '0'
    backend_main._EXPERIMENTS.clear()
    backend_main._KEY_ACTION_RAW_UPLOAD_LOCKS.clear()
    (tmp_path / 'outputs' / 'experiments').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'uploads' / 'experiments').mkdir(parents=True, exist_ok=True)


def test_created_experiment_returns_empty_analysis_contract(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    create_resp = client.post('/api/v1/experiments', json={'title': 'Empty experiment'})
    assert create_resp.status_code == 200
    payload = create_resp.json()
    experiment_id = payload['experiment_id']

    detail_resp = client.get(f'/api/v1/experiments/{experiment_id}')
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail['status'] == 'waiting_for_sources'
    assert detail['task']['status'] == 'waiting_for_sources'
    assert detail['total_steps'] == 0
    assert detail['avg_confidence'] is None

    overview_resp = client.get(f'/api/v1/experiments/{experiment_id}/analysis-overview')
    assert overview_resp.status_code == 200
    overview = overview_resp.json()
    assert overview['run']['status'] == 'waiting_for_sources'
    assert overview['run']['stage'] == 'waiting_for_sources'

    timeline_resp = client.get(f'/api/v1/experiments/{experiment_id}/timeline')
    assert timeline_resp.status_code == 200
    timeline = timeline_resp.json()
    assert timeline['steps'] == []
    assert timeline['total_steps'] == 0
    assert timeline['inferred_steps'] == 0
    assert timeline['avg_confidence'] is None
    assert 'fallback_activity_clustering' not in timeline_resp.text
    assert '"steps":[]' in timeline_resp.text
    assert '"avg_confidence":null' in timeline_resp.text

    structured_resp = client.get(f'/api/v1/experiments/{experiment_id}/structured')
    assert structured_resp.status_code == 200
    structured = structured_resp.json()
    assert structured['status'] == 'created'
    assert structured['steps'] == []
    assert structured['timeline'] == []
    assert structured['analysis'] is None


def test_uploaded_video_keeps_empty_steps_until_analysis(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Upload only'}).json()['experiment_id']
    upload_resp = client.post(
        f'/api/v1/experiments/{experiment_id}/upload/video',
        files={'file': ('demo.mp4', b'fake-video-bytes', 'video/mp4')},
    )
    assert upload_resp.status_code == 200
    assert upload_resp.json()['status'] == 'video_uploaded'
    assert upload_resp.json()['analysis_task']['status'] == 'queued'
    assert upload_resp.json()['video_store_ingest_sec'] >= 0

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    assert detail['status'] == 'failed'
    assert detail['task']['error_type'] == 'video_not_found'
    assert detail['video_asset_id'] is not None

    timeline = client.get(f'/api/v1/experiments/{experiment_id}/timeline').json()
    assert timeline['steps'] == []
    assert timeline['total_steps'] == 0

    video_resp = client.get(f'/api/v1/experiments/{experiment_id}/video')
    assert video_resp.status_code == 200


def test_register_existing_experiment_video_by_hash_returns_ingest_timing(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    content = b'existing-video-bytes'
    digest = hashlib.sha256(content).hexdigest()
    store_root = backend_main._video_store_root()
    final_dir = store_root / 'by_hash' / digest[:2] / digest[2:4] / digest
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f'{digest}.mp4'
    final_path.write_bytes(content)
    backend_main._append_video_store_hash_index(
        {
            'sha256': digest,
            'path': str(final_path),
            'absolute_path': str(final_path),
            'size_bytes': len(content),
            'original_filename': 'existing.mp4',
            'index_source': 'test_fixture',
        }
    )

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Register existing'}).json()['experiment_id']
    register_resp = client.post(
        f'/api/v1/experiments/{experiment_id}/upload/video/register',
        json={'sha256': digest},
    )
    assert register_resp.status_code == 200
    payload = register_resp.json()
    assert payload['video_path'] == str(final_path)
    assert payload['sha256'] == digest
    assert payload['deduplicated'] is True
    assert payload['copied'] is False
    assert payload['registered_via'] == 'hash_manifest'
    assert payload['video_store_ingest_sec'] >= 0

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    assert detail['video_inputs'][0]['sha256'] == digest


def test_key_action_raw_upload_registers_dual_view_sources_for_run(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setattr(backend_main, '_validate_key_action_yolo_runtime', lambda *args, **kwargs: {'available': True})

    def fake_key_action_task(experiment_id: str, **kwargs):
        backend_main._write_key_action_status(
            experiment_id,
            {
                'status': 'completed',
                'progress': 1.0,
                'third_person_video_path': kwargs.get('third_person_video_path'),
                'first_person_video_path': kwargs.get('first_person_video_path'),
                'detection_config': kwargs.get('detection_config') or {},
            },
        )

    monkeypatch.setattr(backend_main, '_run_key_action_index_task', fake_key_action_task)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Raw dual view'}).json()['experiment_id']
    first_resp = client.put(
        f'/api/v1/experiments/{experiment_id}/key-actions/upload-raw/first_person',
        content=b'first-person-video-bytes',
        headers={'Content-Type': 'video/mp4', 'X-Filename': 'first_person_rgb.mp4'},
    )
    third_resp = client.put(
        f'/api/v1/experiments/{experiment_id}/key-actions/upload-raw/third_person',
        content=b'third-person-video-bytes',
        headers={'Content-Type': 'video/mp4', 'X-Filename': 'third_person_rgb.mp4'},
    )
    assert first_resp.status_code == 200
    assert third_resp.status_code == 200
    assert first_resp.json()['view'] == 'first_person'
    assert third_resp.json()['view'] == 'third_person'
    assert first_resp.json()['metadata']['camera_id'] == 'first_person'
    assert third_resp.json()['metadata']['camera_id'] == 'third_person'

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    third_path, first_path = backend_main._registered_key_action_video_paths(detail)
    assert third_path and Path(third_path).exists()
    assert first_path and Path(first_path).exists()
    alignment = json.loads((tmp_path / 'outputs' / 'experiments' / experiment_id / 'timeline_alignment.json').read_text(encoding='utf-8'))
    assert alignment['alignment_status'] == 'shared_recording'
    assert alignment['alignment_reliable_for_dual_view_pairing'] is True
    assert {stream['role'] for stream in alignment['streams']} == {'first_person', 'third_person'}

    run_resp = client.post(
        f'/api/v1/experiments/{experiment_id}/key-actions/run',
        json={'force': True, 'session_start_time': '2026-05-20T00:00:00+08:00'},
    )
    assert run_resp.status_code == 200
    status = client.get(f'/api/v1/experiments/{experiment_id}/key-actions/status').json()
    assert status['third_person_video_path'] == third_path
    assert status['first_person_video_path'] == first_path


def test_key_action_register_raw_by_manifest_uses_existing_hash_without_copy(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    content = b'third-person-video-bytes'
    digest = hashlib.sha256(content).hexdigest()
    store_root = backend_main._video_store_root()
    final_dir = store_root / 'by_hash' / digest[:2] / digest[2:4] / digest
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f'{digest}.mp4'
    final_path.write_bytes(content)
    manifest_path = store_root / 'existing_manifest.json'
    manifest_path.write_text(
        json.dumps(
            {
                'videos': {
                    'third_person': {
                        'sha256': digest,
                        'path': str(final_path),
                        'size_bytes': len(content),
                        'original_filename': 'third_person_rgb.mp4',
                    }
                }
            }
        ),
        encoding='utf-8',
    )
    backend_main._append_video_store_hash_index(
        {
            'sha256': digest,
            'path': str(final_path),
            'absolute_path': str(final_path),
            'size_bytes': len(content),
            'original_filename': 'third_person_rgb.mp4',
            'index_source': 'test_fixture',
        }
    )

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Manifest register'}).json()['experiment_id']
    register_resp = client.post(
        f'/api/v1/experiments/{experiment_id}/key-actions/register-raw/third_person',
        json={'manifest_path': str(manifest_path)},
    )
    assert register_resp.status_code == 200
    payload = register_resp.json()
    assert payload['video_path'] == str(final_path)
    assert payload['sha256'] == digest
    assert payload['deduplicated'] is True
    assert payload['copied'] is False
    assert payload['registered_via'] == 'hash_manifest'
    assert payload['video_store_ingest_sec'] >= 0

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    third_path, first_path = backend_main._registered_key_action_video_paths(detail)
    assert third_path == str(final_path)
    assert first_path is None
    assert detail['key_action_index']['third_person_sha256'] == digest
    assert detail['key_action_index']['third_person_deduplicated'] is True
    assert detail['key_action_index']['upload_mode'] == 'hash_manifest_register'


def test_key_action_analysis_proxy_upload_registers_cache_for_raw_source(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Proxy sidecar'}).json()['experiment_id']
    raw_resp = client.put(
        f'/api/v1/experiments/{experiment_id}/key-actions/upload-raw/third_person',
        content=b'third-person-video-bytes',
        headers={'Content-Type': 'video/mp4', 'X-Filename': 'third_person_rgb.mp4'},
    )
    assert raw_resp.status_code == 200

    proxy_resp = client.put(
        f'/api/v1/experiments/{experiment_id}/key-actions/upload-analysis-proxy/third_person',
        content=b'lightweight-analysis-proxy',
        headers={'Content-Type': 'video/mp4', 'X-Filename': 'third_person_proxy.mp4'},
    )
    assert proxy_resp.status_code == 200
    payload = proxy_resp.json()
    assert payload['view'] == 'third_person'
    proxy_path = Path(payload['proxy_path'])
    meta_path = Path(payload['metadata_path'])
    assert proxy_path.exists()
    assert proxy_path.read_bytes() == b'lightweight-analysis-proxy'
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    assert meta['status'] == 'registered'
    assert meta['proxy_used'] is True

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    proxy_state = detail['key_action_index']['analysis_proxy']['third_person']
    assert proxy_state['proxy_path'] == str(proxy_path)
    assert proxy_state['cache_key'] == payload['cache_key']


def test_key_action_manifest_camera_id_respects_explicit_view_for_rgb_names():
    assert backend_main._infer_key_action_camera_id('first_abc_rgb.mp4', view='first_person') == 'first_person'
    assert backend_main._infer_key_action_camera_id('third_abc_rgb.mp4', view='third_person') == 'third_person'


def test_key_action_default_yolo_model_config_supports_third_person_20_class_config(tmp_path):
    setup_isolated_project_root(tmp_path)
    model_path = tmp_path / 'external_models' / 'yolo26s_third_20_v1' / 'best.pt'
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b'fake-weights')
    config_dir = tmp_path / 'models' / 'yolo' / 'third_person'
    config_dir.mkdir(parents=True)
    (config_dir / 'model_config.yaml').write_text(
        f'model_path: {model_path.as_posix()}\nrecommended_imgsz: 640\n',
        encoding='utf-8',
    )

    assert backend_main._default_key_action_yolo_model_path('third_person') == str(model_path)
    assert backend_main._default_key_action_yolo_imgsz('third_person') == 640


def test_analysis_results_appear_only_after_processing(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setattr(backend_main, 'ExperimentService', FakeExperimentService)
    monkeypatch.setattr(backend_main, 'FORMAL_WORKFLOW', None)
    monkeypatch.setenv('EXPERIMENT_PROFESSIONAL_REPORT_ENABLED', '0')
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Needs analysis'}).json()['experiment_id']
    client.post(
        f'/api/v1/experiments/{experiment_id}/upload/video',
        files={'file': ('demo.mp4', b'fake-video-bytes', 'video/mp4')},
    )

    process_resp = client.post(f'/api/v1/experiments/{experiment_id}/process', json={})
    assert process_resp.status_code == 200
    assert process_resp.json()['status'] == 'analyzed'

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    assert detail['status'] == 'analyzed'
    assert detail['total_steps'] == 1
    assert detail['avg_confidence'] == 0.95

    timeline = client.get(f'/api/v1/experiments/{experiment_id}/timeline').json()
    assert len(timeline['steps']) == 1
    assert timeline['steps'][0]['step_name'] == 'Observed transfer'

    evidence = client.get(f'/api/v1/experiments/{experiment_id}/evidence').json()
    assert evidence['evidence'] == []

    structured = client.get(f'/api/v1/experiments/{experiment_id}/structured').json()
    assert structured['analysis'] is not None
    assert structured['statistics']['total_steps'] == 1


def test_registered_stream_source_can_be_processed(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setattr(backend_main, 'ExperimentService', FakeExperimentService)
    monkeypatch.setattr(backend_main, 'FORMAL_WORKFLOW', None)
    monkeypatch.setenv('EXPERIMENT_PROFESSIONAL_REPORT_ENABLED', '0')
    client = TestClient(backend_main.app)

    stream_file = tmp_path / 'stream_demo.mp4'
    _build_local_test_video(stream_file)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Stream source'}).json()['experiment_id']
    stream_resp = client.post(
        f'/api/v1/experiments/{experiment_id}/upload/stream',
        json={
            'source': str(stream_file),
            'source_type': 'rtsp',
            'camera_id': 'cam_stream',
            'capture_duration_sec': 4.0,
            'sync_method': 'audio_flash',
            'sync_anchors': [{'local_time_sec': 0.0, 'reference_time_sec': 1.0, 'method': 'audio_flash'}],
            'clock_drift_ppm': 12.5,
        },
    )
    assert stream_resp.status_code == 200
    assert stream_resp.json()['stream']['source_type'] == 'rtsp'
    assert stream_resp.json()['stream']['sync_anchors'][0]['method'] == 'audio_flash'
    assert stream_resp.json()['stream']['clock_drift_ppm'] == 12.5

    process_resp = client.post(f'/api/v1/experiments/{experiment_id}/process', json={'sample_interval': 1.0, 'max_frames': 5})
    assert process_resp.status_code == 200
    assert process_resp.json()['status'] == 'analyzed'

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    assert detail['status'] == 'analyzed'
    assert detail['output_paths']['source_video'] == str(stream_file)


def test_get_experiment_video_blocks_paths_outside_project_root(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Path boundary'}).json()['experiment_id']
    outside_video = tmp_path.parent / f'{tmp_path.name}_outside.mp4'
    outside_video.write_bytes(b'not-a-real-video')

    exp = backend_main._normalize_experiment_dict(
        backend_main._load_json_if_exists(backend_main._experiment_output_dir(experiment_id) / 'experiment.json') or {}
    )
    exp['video_paths'] = [str(outside_video)]
    backend_main._save_experiment(exp)

    response = client.get(f'/api/v1/experiments/{experiment_id}/video')
    assert response.status_code == 403
    assert 'project root' in response.text.lower()


def test_get_experiment_artifact_blocks_source_video_outside_project_root(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Artifact boundary'}).json()['experiment_id']
    outside_video = tmp_path.parent / f'{tmp_path.name}_outside_artifact.mp4'
    outside_video.write_bytes(b'not-a-real-video')

    exp = backend_main._normalize_experiment_dict(
        backend_main._load_json_if_exists(backend_main._experiment_output_dir(experiment_id) / 'experiment.json') or {}
    )
    exp['video_paths'] = [str(outside_video)]
    backend_main._save_experiment(exp)

    response = client.get(f'/api/v1/experiments/{experiment_id}/artifacts/source_video')
    assert response.status_code == 403
    assert 'project root' in response.text.lower()


def test_professional_report_generation_stages_material_candidate(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('EXPERIMENT_PROFESSIONAL_REPORT_ENABLED', '1')

    from key_action_indexer.material_references import REPORT_DIR_NAME
    from labsopguard import professional_report as report_module

    def fake_generate_professional_report_pdf(*, overview, key_actions, materials, output_pdf_path, logo_path=None):
        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        output_pdf_path.write_bytes(b'%PDF-1.4\n% fake professional report\n')
        sidecar = output_pdf_path.with_suffix('.json')
        sidecar.write_text(
            json.dumps(
                {
                    'schema_version': 'professional_experiment_report.v1',
                    'context': {'overview': overview, 'key_actions': key_actions, 'materials': materials},
                    'report': {
                        'executive_summary': {
                            'summary': 'Recovered professional report summary',
                            'overall_conclusion': 'Ready for review',
                        },
                        'key_findings': [{'finding': 'Balance interaction', 'evidence': 'YOLO-backed frame'}],
                        'risk_alerts': {'alerts': []},
                    },
                },
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        return {
            'schema_version': 'professional_experiment_report.v1',
            'pdf_path': str(output_pdf_path),
            'sidecar_path': str(sidecar),
        }

    monkeypatch.setattr(report_module, 'generate_professional_report_pdf', fake_generate_professional_report_pdf)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Solid weighing'}).json()['experiment_id']
    summary = backend_main._generate_professional_report_for_experiment(experiment_id, output_paths={})
    exp = backend_main._normalize_experiment_dict(
        backend_main._load_json_if_exists(backend_main._experiment_output_dir(experiment_id) / 'experiment.json') or {}
    )
    exp['output_paths'] = backend_main._attach_professional_report_output_paths(experiment_id, exp.get('output_paths') or {}, summary)
    backend_main._save_experiment(exp)

    reports_dir = tmp_path / 'outputs' / 'experiments' / experiment_id / 'reports'
    assert (reports_dir / 'professional_report_qwen36max.pdf').exists()
    assert (reports_dir / 'professional_report_qwen36max.html').exists()
    assert (reports_dir / 'professional_report_qwen36max.json').exists()
    assert (reports_dir / 'professional_report_manifest.json').exists()

    overview = client.get(f'/api/v1/experiments/{experiment_id}/analysis-overview').json()
    assert overview['artifacts']['professional_report_pdf']['ready'] is True
    assert overview['artifacts']['professional_report_html']['ready'] is True

    delivery = summary['material_delivery']
    assert delivery['status'] == 'candidate_staged'
    assert delivery['report_count'] == 1

    candidate_root = Path(delivery['path']).parent
    report_candidate = candidate_root / REPORT_DIR_NAME / 'professional_report_qwen36max.pdf'
    assert report_candidate.exists()
    assert not list((tmp_path / 'outputs' / 'material_references').glob(f'*/{REPORT_DIR_NAME}/professional_report_qwen36max.pdf'))

    candidate_manifest = json.loads((candidate_root / 'manifest.json').read_text(encoding='utf-8'))
    candidate_readme = (candidate_root / 'README.md').read_text(encoding='utf-8')
    candidate_rows = [
        json.loads(line)
        for line in (candidate_root / '素材候选索引.jsonl').read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    assert candidate_manifest['candidate_count'] == 1
    assert 'candidates: 1' in candidate_readme
    assert candidate_rows[0]['asset_kind'] == REPORT_DIR_NAME
    assert candidate_rows[0]['candidate_status'] == 'pending'
    assert candidate_rows[0]['delivery_scope'] == 'professional_report_candidate'


def test_material_search_rebuilds_index_and_filters(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Material search'}).json()['experiment_id']
    exp_dir = tmp_path / 'outputs' / 'experiments' / experiment_id
    clip_file = exp_dir / 'clips' / 'clip_1.mp4'
    clip_file.parent.mkdir(parents=True, exist_ok=True)
    clip_file.write_bytes(b'fake-clip')

    (exp_dir / 'material_stream.json').write_text(
        json.dumps(
            [
                {
                    'item_id': 'item_1',
                    'experiment_id': experiment_id,
                    'timestamp_sec': 2.5,
                    'local_timestamp_sec': 2.0,
                    'camera_id': 'cam_a',
                    'stream_id': 'stream_a',
                    'video_asset_id': 'asset_a',
                    'frame_id': 'frame_10',
                    'frame_bgr_path': 'frame_10.jpg',
                    'clip_id': 'clip_1',
                    'object_labels': ['pipette', 'tube'],
                    'detected_activities': ['transfer'],
                    'scene_description': 'pipette transfers liquid into tube',
                }
            ]
        ),
        encoding='utf-8',
    )
    (exp_dir / 'preprocessing.json').write_text(
        json.dumps(
            {
                'key_clips': [
                    {
                        'clip_id': 'clip_1',
                        'file_path': str(clip_file),
                        'file_exists': True,
                        'reason': 'visual_change',
                    }
                ],
                'detected_changes': [
                    {
                        'event_type': 'hand_contact_geometry',
                        'metadata': {'material_item_id': 'item_1'},
                    }
                ],
            }
        ),
        encoding='utf-8',
    )

    response = client.get(
        f'/api/v1/experiments/{experiment_id}/materials/search',
        params={
            'objects': 'pipette',
            'actions': 'transfer',
            'start_time_sec': 0,
            'end_time_sec': 4,
            'camera_id': 'cam_a',
            'clip_exists': True,
            'text': 'liquid',
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload['total'] == 1
    assert payload['items'][0]['clip_file_path'] == str(clip_file)
    assert payload['items'][0]['event_types'] == ['hand_contact_geometry']
    assert (exp_dir / 'material_index.sqlite').exists()


def test_invalid_experiment_id_is_rejected(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    response = client.get('/api/v1/experiments/bad$id')
    assert response.status_code == 400
    assert 'Invalid experiment_id format' in response.text


def test_serve_experiment_file_blocks_path_traversal(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Path traversal'}).json()['experiment_id']
    outside = tmp_path / 'outputs' / 'outside.txt'
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text('secret', encoding='utf-8')

    response = client.get(f'/api/v1/experiments/{experiment_id}/files/..%2F..%2Foutside.txt')
    assert response.status_code == 403


def test_serve_experiment_file_supports_video_range_requests(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Range video'}).json()['experiment_id']
    clip = backend_main.PROJECT_ROOT / 'outputs' / 'experiments' / experiment_id / 'published_materials' / 'operator' / 'range.mp4'
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b'0123456789')

    response = client.get(
        f'/api/v1/experiments/{experiment_id}/files/published_materials/operator/range.mp4',
        headers={'Range': 'bytes=2-5'},
    )

    assert response.status_code == 206
    assert response.headers['content-range'] == 'bytes 2-5/10'
    assert response.headers['accept-ranges'] == 'bytes'
    assert response.content == b'2345'


def test_upload_video_sanitizes_filename(tmp_path):
    setup_isolated_project_root(tmp_path)
    client = TestClient(backend_main.app)

    experiment_id = client.post('/api/v1/experiments', json={'title': 'Filename sanitize'}).json()['experiment_id']
    upload_resp = client.post(
        f'/api/v1/experiments/{experiment_id}/upload/video',
        files={'file': ('../evil?.mp4', b'fake-video-bytes', 'video/mp4')},
    )
    assert upload_resp.status_code == 200
    video_path = Path(upload_resp.json()['video_path'])
    assert '..' not in video_path.name
    assert '?' not in video_path.name
    assert '/' not in video_path.name
    assert '\\' not in video_path.name
    assert video_path.exists()


def test_send_callback_notification_blocks_local_and_allows_public(monkeypatch):
    called: list[str] = []

    def fake_post(url, json=None, timeout=None, allow_redirects=None):
        called.append(url)
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(backend_main.requests, 'post', fake_post)
    monkeypatch.delenv('REALITYLOOP_CALLBACK_ALLOW_PRIVATE', raising=False)

    backend_main.send_callback_notification('https://127.0.0.1:8443/callback', 'task-local', [])
    assert called == []

    backend_main.send_callback_notification('https://8.8.8.8/callback', 'task-public', [])
    assert called == ['https://8.8.8.8/callback']


def test_send_callback_notification_respects_allowed_host_whitelist(monkeypatch):
    called: list[str] = []

    def fake_post(url, json=None, timeout=None, allow_redirects=None):
        called.append(url)
        return types.SimpleNamespace(status_code=200)

    def fake_getaddrinfo(host, port, proto=0):
        return [(None, None, None, None, ('93.184.216.34', port))]

    monkeypatch.setattr(backend_main.requests, 'post', fake_post)
    monkeypatch.setattr(backend_main.socket, 'getaddrinfo', fake_getaddrinfo)
    monkeypatch.setenv('REALITYLOOP_CALLBACK_ALLOWED_HOSTS', 'callbacks.example.com,*.trusted.example')
    monkeypatch.delenv('REALITYLOOP_CALLBACK_ALLOW_PRIVATE', raising=False)
    monkeypatch.delenv('REALITYLOOP_CALLBACK_ALLOWED_CIDRS', raising=False)

    backend_main.send_callback_notification('https://callbacks.example.com/hook', 'task-1', [])
    backend_main.send_callback_notification('https://sub.trusted.example/hook', 'task-2', [])
    backend_main.send_callback_notification('https://evil.example/hook', 'task-3', [])

    assert called == [
        'https://callbacks.example.com/hook',
        'https://sub.trusted.example/hook',
    ]


def test_send_callback_notification_respects_allowed_cidrs(monkeypatch):
    called: list[str] = []

    def fake_post(url, json=None, timeout=None, allow_redirects=None):
        called.append(url)
        return types.SimpleNamespace(status_code=200)

    monkeypatch.setattr(backend_main.requests, 'post', fake_post)
    monkeypatch.delenv('REALITYLOOP_CALLBACK_ALLOWED_HOSTS', raising=False)
    monkeypatch.setenv('REALITYLOOP_CALLBACK_ALLOWED_CIDRS', '8.8.8.0/24')
    monkeypatch.delenv('REALITYLOOP_CALLBACK_ALLOW_PRIVATE', raising=False)

    backend_main.send_callback_notification('https://8.8.8.8/hook', 'task-1', [])
    backend_main.send_callback_notification('https://1.1.1.1/hook', 'task-2', [])

    assert called == ['https://8.8.8.8/hook']


def test_auth_required_rejects_anonymous_create_experiment(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    response = client.post('/api/v1/experiments', json={'title': 'Auth required'})
    assert response.status_code == 401
    assert 'Authentication required' in response.text


def test_auth_required_allows_header_auth_and_enforces_scope(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    headers = {'X-Operator': 'auditor', 'X-Operator-Role': 'reviewer'}
    create_resp = client.post('/api/v1/experiments', json={'title': 'Auth scope'}, headers=headers)
    assert create_resp.status_code == 200
    experiment_id = create_resp.json()['experiment_id']

    denied_headers = {**headers, 'X-Allowed-Experiments': 'other-exp'}
    denied_upload = client.post(
        f'/api/v1/experiments/{experiment_id}/upload/video',
        files={'file': ('demo.mp4', b'fake-video-bytes', 'video/mp4')},
        headers=denied_headers,
    )
    assert denied_upload.status_code == 403

    allowed_headers = {**headers, 'X-Allowed-Experiments': experiment_id}
    allowed_upload = client.post(
        f'/api/v1/experiments/{experiment_id}/upload/video',
        files={'file': ('demo.mp4', b'fake-video-bytes', 'video/mp4')},
        headers=allowed_headers,
    )
    assert allowed_upload.status_code == 200


def test_auth_required_blocks_workspace_reindex_endpoint(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    response = client.post('/api/v1/materials/reindex')
    assert response.status_code == 401
    assert 'Authentication required' in response.text


def test_auth_required_enforces_scope_on_experiment_reads(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    base_headers = {'X-Operator': 'auditor', 'X-Operator-Role': 'reviewer'}
    exp_1 = client.post('/api/v1/experiments', json={'title': 'Scoped read 1'}, headers=base_headers).json()['experiment_id']
    exp_2 = client.post('/api/v1/experiments', json={'title': 'Scoped read 2'}, headers=base_headers).json()['experiment_id']

    no_auth_resp = client.get(f'/api/v1/experiments/{exp_1}')
    assert no_auth_resp.status_code == 401

    wrong_scope_headers = {**base_headers, 'X-Allowed-Experiments': exp_2}
    denied_resp = client.get(f'/api/v1/experiments/{exp_1}', headers=wrong_scope_headers)
    assert denied_resp.status_code == 403

    allowed_scope_headers = {**base_headers, 'X-Allowed-Experiments': exp_1}
    allowed_resp = client.get(f'/api/v1/experiments/{exp_1}', headers=allowed_scope_headers)
    assert allowed_resp.status_code == 200
    assert allowed_resp.json()['experiment_id'] == exp_1

    scoped_list = client.get('/api/v1/experiments', headers=allowed_scope_headers)
    assert scoped_list.status_code == 200
    payload = scoped_list.json()
    assert payload['total'] == 1
    assert payload['experiments'][0]['experiment_id'] == exp_1


def test_auth_required_blocks_workspace_material_search_read(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    response = client.get('/api/v1/materials/search')
    assert response.status_code == 401
    assert 'Authentication required' in response.text


def test_standalone_video_analysis_status_route_removed(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    response = client.get('/api/v1/video-analysis/status/nonexistent-task')
    assert response.status_code == 404


def test_auth_required_blocks_diagnostics_read(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    response = client.get('/api/v1/diagnostics')
    assert response.status_code == 401
    assert 'Authentication required' in response.text


def test_auth_required_blocks_anonymous_websocket(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect('/ws'):
            pass
    assert exc.value.code == 1008


def test_auth_required_allows_websocket_query_operator(tmp_path, monkeypatch):
    setup_isolated_project_root(tmp_path)
    monkeypatch.setenv('REALITYLOOP_AUTH_REQUIRED', 'true')
    client = TestClient(backend_main.app)

    with client.websocket_connect('/ws?operator=auditor&operator_role=reviewer') as ws:
        ws.send_json({'type': 'ping'})
        payload = ws.receive_json()
        assert payload['type'] == 'pong'
