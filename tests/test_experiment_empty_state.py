from __future__ import annotations

import json
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
    backend_main._EXPERIMENTS.clear()
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

    detail = client.get(f'/api/v1/experiments/{experiment_id}').json()
    assert detail['status'] == 'failed'
    assert detail['task']['error_type'] == 'video_not_found'
    assert detail['video_asset_id'] is not None

    timeline = client.get(f'/api/v1/experiments/{experiment_id}/timeline').json()
    assert timeline['steps'] == []
    assert timeline['total_steps'] == 0

    video_resp = client.get(f'/api/v1/experiments/{experiment_id}/video')
    assert video_resp.status_code == 200


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
