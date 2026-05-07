# Audit Report

Generated: 2026-04-20

## Confirmed End-to-End Links

- Backend API routes exist and are importable:
  - `/api/v1/experiments/{experiment_id}/upload/video`
  - `/api/v1/experiments/{experiment_id}/upload/stream`
  - `/api/v1/experiments/{experiment_id}/upload/asr`
  - `/api/v1/experiments/{experiment_id}/materials/search`
  - `/api/v1/diagnostics`
- Frontend app exists and builds successfully with `npm run build` in `frontend-app`.
- Frontend routes verified in code:
  - `/experiments`: experiment list
  - `/experiments/:id/workspace`: experiment detail/workspace
  - `/experiments/:id/materials`: material search
  - `/experiments/:id/materials/timeline`: material stream timeline, event list, key clip backfill entry
  - `/experiments/:id/steps/:stepId`: step/detail evidence view
- Qwen configuration is read from environment:
  - `DASHSCOPE_API_KEY`
  - `ASR_MODEL`, default `qwen3-asr-flash`
  - `MATERIAL_EMBEDDING_MODEL` or `EMBEDDING_MODEL`, default `text-embedding-v4`
- Qwen diagnostics are exposed by `/api/v1/diagnostics` and scripts/check_qwen_integration.py.
- YOLO26 weights are resolved and real ultralytics inference runs:
  - weights: `outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`
  - `scripts/check_yolo26_integration.py` detected 5 objects on a dataset frame.
- YOLO26 detector output is now injected into ExperimentService frame analysis:
  - `analysis.detected_objects`
  - `analysis.object_labels`
  - `MultimodalMaterialStreamItem.detected_objects`
  - `MultimodalMaterialStreamItem.object_labels`
  - `PhysicalEvent` generation through semantic/object-diff rules
  - `material_index.sqlite` indexing through object labels and text blob
- Multisource acceptance demo generated and checked:
  - output dir: `outputs/experiments/final_acceptance_e2e`
  - `experiment.json`: 53182 bytes
  - `preprocessing.json`: 71760 bytes
  - `material_stream.json`: 50706 bytes
  - `physical_events.json`: 31928 bytes
  - `material_index.sqlite`: 344064 bytes
- The acceptance output contains:
  - 2 `experiment.json.video_inputs`
  - 12 material stream items
  - 23 physical events
  - 23 detected changes
  - 8 key frames
  - 8 key clips
  - 12 SQLite material index rows using `qwen_dashscope:text-embedding-v4` embeddings
  - transcript/context backlink on material items

## Design-Only Or Not Fully Live-Run Links

- Real RTSP camera ingestion was not live-run in this environment. The acceptance demo used a second local video as an RTSP simulation source, with `ingest_mode=rtsp_simulated_by_local_file`.
- Qwen ASR live transcription was run after installing `dashscope`: `outputs/checks/qwen_asr_probe.wav` returned provider `qwen_dashscope`, model `qwen3-asr-flash`, text `0.`, segment_count 1.
- Qwen embedding live API call was run after installing `dashscope`: `text-embedding-v4` returned a 1024-dimensional vector with fallback_mode=false.
- Frontend was build-verified, and route/API wiring was inspected, but browser screenshot capture was not performed in this terminal-only run.

## Problems Found

- ExperimentService previously did not have a stable YOLO26 detector adapter in the main material-stream path. YOLO existed in the standalone video-analysis path, but ExperimentService material stream relied on VLM/fallback labels.
- Qwen embedding fallback was previously silent. It now logs fallback and diagnostics expose `fallback_mode`.
- Runtime diagnostics did not expose the requested `qwen_asr_status`, `qwen_embedding_status`, current model names, or fallback mode.
- OpenCV `cv2.imread` is unreliable on Windows for several non-ASCII dataset paths. Demo/image detector reads now fall back to `np.fromfile + cv2.imdecode`.
- Current environment has `DASHSCOPE_API_KEY` configured and `dashscope` was installed during this pass; Qwen ASR and embedding probes now execute.
- Redis and seaborn are not available; backend import falls back to in-memory Redis and warns on legacy module bootstrap. These are not blockers for the checked backend loop.

## Fixes Implemented

- Added `src/labsopguard/detectors.py` with a real ultralytics YOLO26 adapter:
  - `weights_path` resolution
  - CPU/GPU auto device selection
  - single-frame inference
  - unified output schema: `label`, `confidence`, `bbox`, `optional_ocr`, `optional_attributes`
- Wired YOLO26 detector into `src/experiment/service.py` frame analysis and material-stream generation.
- Added Qwen ASR diagnostics in `src/labsopguard/asr.py`.
- Added embedding diagnostics and explicit fallback logging in `src/labsopguard/embeddings.py`.
- Added `/api/v1/diagnostics` and extended model status in `backend/main.py`.
- Added env/config support for YOLO26 runtime settings.
- Added executable acceptance scripts:
  - `scripts/check_qwen_integration.py`
  - `scripts/check_yolo26_integration.py`
  - `scripts/demo_multisource_run.py`
  - `scripts/check_pipeline_e2e.py`
- Generated `outputs/experiments/final_acceptance_e2e` and validated required files.

## Unresolved Items

- Keep `dashscope>=1.20.0` installed in the runtime image/environment; it was installed locally during this pass.
- Run a real RTSP source when hardware/network stream is available.
- Capture frontend screenshots from a browser session if visual proof is required.
- Existing repository worktree contains many unrelated dirty/untracked dataset and docs changes; they were not reverted or normalized in this acceptance pass.


