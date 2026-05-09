# Integration Report

Generated: 2026-04-20

## Frontend/Backend Integration

- Backend API routes required by this pass are present and importable.
- Frontend app is located at `frontend-app` and uses `baseURL: /api/v1` in `frontend-app/src/api.ts`.
- Frontend pages/routes verified:
  - `/experiments`: experiment list
  - `/experiments/:id/workspace`: experiment detail, upload/process/status/analysis view, key frame panel
  - `/experiments/:id/materials`: material search, ASR upload entry, embedding-text query
  - `/experiments/:id/materials/timeline`: unified material timeline, event list, key clips, clip backfill
  - `/experiments/:id/steps/:stepId`: step detail/evidence view
- Build command passed: `cd frontend-app; npm run build`.
- No fake frontend data was added. Existing pages call real backend APIs through `experimentApi`.

## Qwen API Integration

- ASR reads:
  - `DASHSCOPE_API_KEY`
  - `ASR_PROVIDER`, default `qwen`
  - `ASR_MODEL`, default `qwen3-asr-flash`
- Embedding reads:
  - `DASHSCOPE_API_KEY`
  - `MATERIAL_EMBEDDING_PROVIDER`, default `qwen`
  - `MATERIAL_EMBEDDING_MODEL` or `EMBEDDING_MODEL`, default `text-embedding-v4`
- `/api/v1/experiments/{experiment_id}/upload/asr` calls `transcribe_audio_file()` and appends transcript segments into `context_inputs`, which become `ContextEvent` during ExperimentService processing.
- `materials/search?embedding_text=...` uses `MaterialRetrievalIndex`, which calls the configured embedding provider and ranks by cosine similarity.
- Current environment result from `scripts/check_qwen_integration.py --live-embedding` after installing `dashscope`:
  - `qwen_asr_status`: `configured`
  - `qwen_embedding_status`: `configured`
  - `current_asr_model`: `qwen3-asr-flash`
  - `current_embedding_model`: `text-embedding-v4`
  - `fallback_mode`: `false`
  - live embedding probe: `qwen_dashscope:text-embedding-v4`, dimension 1024
- ASR live probe used `outputs/checks/qwen_asr_probe.wav` and returned provider `qwen_dashscope`, model `qwen3-asr-flash`, text `0.`, segment_count 1.

## YOLO26 Integration

- Detector adapter: `src/labsopguard/detectors.py`.
- Default weights resolved in this environment:
  - `outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`
- Real inference command passed:
  - `python scripts/check_yolo26_integration.py`
- Probe result:
  - detector provider: `ultralytics_yolo26`
  - device: `cpu`
  - ultralytics version: `8.4.33`
  - detections: 5 objects, including `reagent_bottle`, `sample_bottle`, `sample_bottle_blue`, `gloved_hand`, `balance`
- Main-chain connection:
  - ExperimentService calls detector per extracted frame.
  - Detector output is merged into `analysis.detected_objects` and `analysis.object_labels`.
  - Material stream writes `detected_objects` and `object_labels`.
  - PhysicalEvent generation consumes detected objects through semantic/object-diff logic.
  - Material index stores object labels and search text.

## Multisource Time Alignment

- Acceptance script: `scripts/demo_multisource_run.py`.
- Inputs:
  - local video: `outputs/demo_inputs/final_acceptance_e2e/local_camera.mp4`
  - RTSP simulation by local file: `outputs/demo_inputs/final_acceptance_e2e/rtsp_sim_camera.mp4`
  - timestamped transcript inputs at 0.2s and 1.8s
- `experiment.json.video_inputs` contains 2 inputs.
- `preprocessing.json.alignment_summary` reports:
  - `video_count`: 2
  - `anchor_strategy`: `calibrated`
  - `key_frame_count`: 8
  - `key_clip_count`: 8
  - `change_event_count`: 23
- `material_stream.json` contains both `timestamp_sec` and `local_timestamp_sec`.
- Material item backlink verified:
  - `linked_context_event_ids` non-empty
  - `transcript_segment` present
- `material_index.sqlite` contains 12 rows and is queryable.
- Acceptance material index health reports `embedding_mode=qwen_dashscope:text-embedding-v4`.

## Key Source Files

- `backend/main.py`
- `src/experiment/service.py`
- `src/labsopguard/detectors.py`
- `src/labsopguard/asr.py`
- `src/labsopguard/embeddings.py`
- `src/labsopguard/retrieval.py`
- `src/labsopguard/preprocessing.py`
- `src/labsopguard/config.py`
- `src/labsopguard/video_analysis.py`
- `frontend-app/src/api.ts`
- `frontend-app/src/App.tsx`
- `frontend-app/src/pages/ExperimentWorkspace.tsx`
- `frontend-app/src/pages/MaterialSearch.tsx`
- `frontend-app/src/pages/MaterialTimelineView.tsx`

## Configuration

- Qwen:
  - `DASHSCOPE_API_KEY`
  - `DASHSCOPE_BASE_URL`
  - `ASR_PROVIDER=qwen`
  - `ASR_MODEL=qwen3-asr-flash`
  - `MATERIAL_EMBEDDING_PROVIDER=qwen`
  - `MATERIAL_EMBEDDING_MODEL=text-embedding-v4`
  - `MATERIAL_EMBEDDING_DIMENSION=1024`
- YOLO26:
  - `YOLO26_WEIGHTS_PATH`
  - `LABSOPGUARD_YOLO_MODEL`
  - `DETECTOR_DEVICE=auto|cpu|cuda:0`
  - `YOLO26_CONFIDENCE_THRESHOLD=0.25`
  - `YOLO26_IOU_THRESHOLD=0.45`
  - `YOLO26_MAX_DETECTIONS=50`
- Defaults are documented in `.env.example`.

## Acceptance Output

- Output directory: `outputs/experiments/final_acceptance_e2e`
- Required files generated and non-empty:
  - `experiment.json`
  - `preprocessing.json`
  - `material_stream.json`
  - `physical_events.json`
  - `material_index.sqlite`
- Validation command passed:
  - `python scripts/check_pipeline_e2e.py --exp-id final_acceptance_e2e`


