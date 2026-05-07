# RealityLoop Development Handoff - 2026-04-20

This document records the current implementation state after the latest backend/frontend closure work. It is intended as the first document to read in the next development session.

## 1. Current Rating

Current state: B - locally usable / internal demo ready for the completed demo run.

The system now has a real end-to-end demo path:

- Upload or run video analysis.
- YOLO26 runs on GPU and renders live per-frame detection boxes into annotated video.
- The video-analysis result can be written back to an experiment run.
- The experiment workspace reads one unified `analysis-overview` contract.
- The workspace can directly preview the annotated video artifact.
- Material stream, physical events, steps, material index, and Qwen frame analysis can be produced for the same experiment output directory.

It is not yet production-grade. Remaining work is listed below.

## 2. Most Important Demo Run

Use this run instead of the older `final_acceptance_e2e` run.

Experiment ID:

```text
c404e890-4e3d-4ba1-8860-bd40c7f81a37
```

Workspace URL:

```text
http://127.0.0.1:5173/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/workspace
```

Material search URL:

```text
http://127.0.0.1:5173/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/materials
```

Expected overview status for this run:

```text
status=completed
summary_ready=true
steps_ready=true
alerts_ready=true
artifacts_ready=true
annotated_video_ready=true
writeback_ready=true
```

Key output files:

```text
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/experiment.json
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/preprocessing.json
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/material_stream.json
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/physical_events.json
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/material_index.sqlite
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/analysis/analysis.json
outputs/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/analysis/annotated.mp4
```

The annotated video was verified as browser-friendly H.264:

```text
Video: h264 (avc1), yuv420p, 640x480, 30 fps
```

## 3. One-Command Startup

Use the full-stack script:

```powershell
cd D:\LabEmbodiedVLA\LabSOPGuard
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_full_stack.ps1
```

Force restart all local services:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_full_stack.ps1 -Restart
```

No browser auto-open:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_full_stack.ps1 -NoOpen
```

The script manages:

- Redis on `127.0.0.1:6379`
- FastAPI backend on `127.0.0.1:8000`
- Vite frontend on `127.0.0.1:5173`

Health checks performed by the script:

- Frontend root `/`
- `GET /api/v1/experiments/final_acceptance_e2e/analysis-overview`
- `GET /api/v1/experiments/final_acceptance_e2e/materials/search?...`

Logs:

```text
outputs/run_logs/backend_8000.err.log
outputs/run_logs/frontend_5173.err.log
outputs/run_logs/redis_6379.err.log
```

## 4. What Was Implemented

### 4.1 Unified experiment workspace contract

Added:

```text
GET /api/v1/experiments/{experiment_id}/analysis-overview
GET /api/experiments/{experiment_id}/analysis-overview
```

Implemented in:

```text
backend/main.py
```

The contract returns:

- `experiment`
- `run`
- `readiness`
- `summary`
- `steps`
- `scene_summary`
- `alerts`
- `markers`
- `artifacts`
- `debug`

Important behavior:

- `completed` is only returned when readiness gates are all true.
- `progress=100%` no longer means results are display-ready.
- Step counts are computed from the same step arrays rendered by the frontend.

### 4.2 Frontend workspace now uses one contract

Updated:

```text
frontend-app/src/pages/ExperimentWorkspace.tsx
frontend-app/src/types.ts
frontend-app/src/api.ts
```

Fixed:

- Header/run/status consistency.
- Summary and steps same source.
- Candidate steps render when official steps are empty.
- First visible step auto-selects.
- Raw scene JSON no longer leaks into main UI.
- Alert evidence fields are displayed.
- Raw video / annotated video toggle.
- Step/alert/evidence marker jump support.
- Artifacts metadata separated from debug paths.

### 4.3 `/video-analysis` can write back to experiment runs

Updated:

```text
backend/main.py
frontend-app/src/pages/VideoAnalysis.tsx
frontend-app/src/api.ts
frontend-app/src/types.ts
```

New behavior:

- `POST /api/v1/video-analysis/analyze` accepts optional form field `experiment_id`.
- If `experiment_id` is supplied, video-analysis artifacts are copied/written to the experiment output directory.
- The experiment run is then processed into material stream, physical events, steps, index, and overview artifacts.
- The frontend video-analysis page has an optional experiment ID field and a Qwen writeback cost-control checkbox.

Also added:

```text
POST /api/v1/experiments/{experiment_id}/video-analysis/attach/{task_id}
```

This attaches an already completed video-analysis task to an experiment.

### 4.4 YOLO26 GPU and live annotated boxes

Updated:

```text
src/labsopguard/video_analysis.py
```

Current behavior:

- YOLO26 weights are loaded from the configured path.
- Device is `cuda:0` when CUDA is available.
- Annotated video now runs YOLO per frame for overlay, instead of reusing fixed sampled boxes.

Validated environment:

```text
torch 2.6.0+cu124
CUDA 12.4
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

### 4.5 H.264 browser playback fallback

Updated:

```text
src/labsopguard/video_analysis.py
```

Current behavior:

- OpenCV writes an intermediate MP4.
- The system then tries to transcode to H.264/yuv420p using `ffmpeg`.
- If system ffmpeg is not installed, it uses `imageio_ffmpeg` if available.
- If transcode fails, it falls back to the OpenCV MP4 output.

The completed demo run has H.264 output and is browser-previewable.

### 4.6 Qwen frame analysis writeback

Validated on the completed demo run:

```text
python scripts/writeback_qwen_frame_analysis.py --exp-id c404e890-4e3d-4ba1-8860-bd40c7f81a37 --limit 2 --force-live --no-eval-cache
```

Result:

```text
processed_items=2
flash_written=2
plus_keyframe_reviews_written=2
failures=[]
embedding_mode=qwen_dashscope:text-embedding-v4
```

Material search verified with:

```text
embedding_text=sample bottle gloved hand
```

Top results contain:

- `qwen3_6_flash_frame`
- `qwen3_6_plus_keyframe_review`

### 4.7 PPE alert gating

Updated:

```text
src/labsopguard/video_analysis.py
```

Fixed:

- No-person/no-actor scenes no longer directly trigger `missing_goggles` or `missing_lab_coat`.
- PPE rules now use person-aware, step-aware, and scene-aware gating.
- Alert details include evidence-oriented fields.

### 4.8 UI mojibake repairs

Updated fixed text in:

```text
frontend-app/src/pages/ExperimentList.tsx
frontend-app/src/pages/Upload.tsx
frontend-app/src/pages/MaterialSearch.tsx
frontend-app/src/pages/ExperimentWorkspace.tsx
frontend-app/src/components/Layout.tsx
```

Most fixed UI strings were changed to ASCII/English to avoid repeat encoding damage on Windows/PowerShell.

## 5. Validation Commands

Backend compile:

```powershell
python -m py_compile backend\main.py src\labsopguard\video_analysis.py
```

Backend tests:

```powershell
pytest tests/test_analysis_overview_contract.py tests/test_ppe_gating.py -q
```

Frontend build:

```powershell
cd frontend-app
npm run build
```

Frontend component test:

```powershell
npm run test -- ExperimentWorkspace
```

Manual overview check:

```powershell
python -c "import requests; exp='c404e890-4e3d-4ba1-8860-bd40c7f81a37'; d=requests.get(f'http://127.0.0.1:8000/api/v1/experiments/{exp}/analysis-overview').json(); print(d['run']); print(d['readiness']); print(d['summary'])"
```

Manual material search check:

```powershell
python -c "import requests; exp='c404e890-4e3d-4ba1-8860-bd40c7f81a37'; r=requests.get(f'http://127.0.0.1:8000/api/v1/experiments/{exp}/materials/search', params={'embedding_text':'sample bottle gloved hand','limit':3}); print(r.status_code); print(r.json().get('embedding_mode'))"
```

## 6. What Is Not Done Yet

### 6.1 Production-grade job state machine

Current task state is file-backed and good enough for local demo. It is not a robust distributed job system.

Still needed:

- Stronger queued/running/completed transitions.
- Worker crash recovery.
- Timeout and retry policy per stage.
- Clear separation between video-analysis completion and experiment writeback completion.

A small fix was added so `/video-analysis/status` does not mark a task completed early when `experiment_id` exists but experiment output paths are not ready. This should still be hardened further.

### 6.2 Official step semantics are still basic

The demo run has one confirmed step, but this is not yet a mature SOP confirmation workflow.

Still needed:

- Better official/candidate/inferred promotion rules.
- Human review and locking.
- Versioned step edits.
- Richer step evidence reasoning.

### 6.3 Alert rules are still shallow

PPE gating is fixed, but risk logic is still basic.

Still needed:

- Reagent mismatch rules.
- Step-order violation rules.
- Liquid/container state-change rules.
- Threshold/quantity anomaly rules.
- Rule configuration UI or rule registry.

### 6.4 Multi-source / RTSP long-run reliability not yet proven

The data model supports multi-source inputs and time alignment, but this latest completed demo is a short local video run.

Still needed:

- Real RTSP soak test.
- Multi-camera clock drift test.
- Dropout/reconnect behavior.
- Cross-camera material stream consistency checks.

### 6.5 Qwen video understanding remains weaker than frame understanding

Frame-level Qwen writeback works. Short video understanding still needs more robust upload/retry/compression evaluation.

Still needed:

- Batch test with multiple clips.
- Timeout/retry matrix.
- Empty-response retention and diagnostics.
- Cost/latency control for video-level calls.

### 6.6 Old artifacts and filenames still contain historical mojibake

New UI labels are fixed. Existing uploaded filenames and historical JSON/task records may still contain mojibake.

Still needed:

- Optional migration script to normalize filenames/display names.
- Safer display-title layer separate from raw file paths.

### 6.7 Some legacy pages are not fully contract-driven

Fixed main workspace and critical pages. Some older pages may still read legacy endpoints.

Review later:

```text
frontend-app/src/pages/ExperimentTimeline.tsx
frontend-app/src/pages/StepDetail.tsx
frontend-app/src/pages/JsonViewer.tsx
frontend-app/src/pages/MaterialTimelineView.tsx
```

## 7. Known Good URLs

```text
http://127.0.0.1:5173/
http://127.0.0.1:5173/experiments
http://127.0.0.1:5173/upload
http://127.0.0.1:5173/video-analysis
http://127.0.0.1:5173/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/workspace
http://127.0.0.1:5173/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/materials
```

Backend API examples:

```text
http://127.0.0.1:8000/api/v1/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/analysis-overview
http://127.0.0.1:8000/api/v1/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/artifacts/annotated_video
http://127.0.0.1:8000/api/v1/experiments/c404e890-4e3d-4ba1-8860-bd40c7f81a37/materials/search?embedding_text=sample%20bottle%20gloved%20hand&limit=3
```

## 8. Important Files Changed Recently

```text
backend/main.py
src/labsopguard/video_analysis.py
frontend-app/src/pages/ExperimentWorkspace.tsx
frontend-app/src/pages/VideoAnalysis.tsx
frontend-app/src/pages/ExperimentList.tsx
frontend-app/src/pages/Upload.tsx
frontend-app/src/pages/MaterialSearch.tsx
frontend-app/src/components/Layout.tsx
frontend-app/src/api.ts
frontend-app/src/types.ts
frontend-app/src/pages/__tests__/ExperimentWorkspace.test.tsx
tests/test_analysis_overview_contract.py
tests/test_ppe_gating.py
scripts/start_full_stack.ps1
requirements.gpu-cu124.txt
```

## 9. Environment Notes

Redis:

```text
tools/redis/Redis-8.6.2-Windows-x64-msys2/redis-server.exe
```

GPU PyTorch:

```text
torch==2.6.0+cu124
torchvision==0.21.0+cu124
```

The current Python is 3.13. The CUDA wheel resolved to PyTorch `2.6.0+cu124`, not the earlier CPU-only package. GPU functionality was verified.

Qwen/DashScope environment:

```text
DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL
ASR_MODEL=qwen3-asr-flash
MATERIAL_EMBEDDING_MODEL=text-embedding-v4
QWEN_FRAME_WRITEBACK_ENABLED=false by default
```

YOLO weights:

```text
outputs/training/yolo26s_autodl_8_1_1/weights/best.pt
```

## 10. Recommended Next Development Order

1. Harden job state transitions and make video-analysis writeback a first-class task stage.
2. Add a small UI indicator for experiment writeback progress after `/video-analysis` upload.
3. Improve official step promotion and manual review workflow.
4. Add stronger alert rules beyond PPE.
5. Run a real multi-camera/RTSP soak test.
6. Add filename/display-name migration for historical mojibake artifacts.
7. Convert remaining legacy frontend pages to `analysis-overview` or explicit contracts.

## 11. Do Not Repeat These Mistakes

- Do not use `progress=100%` as proof that workspace artifacts are ready.
- Do not mark an experiment completed unless all readiness gates are true.
- Do not render raw JSON strings in the main UI.
- Do not trigger PPE missing alerts without actor/scene/step gating.
- Do not assume browser video playback works unless the MP4 is H.264/yuv420p or the browser has been tested.
- Do not rely on `final_acceptance_e2e` for the main demo; use the completed demo run above.
