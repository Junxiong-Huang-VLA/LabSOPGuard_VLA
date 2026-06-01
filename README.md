# LabEmbodied

LabEmbodied is the cleaned, flattened workspace for backend APIs, frontend UI, data-processing pipelines, indexing, retrieval, reports, tests, and configuration.

Large or sensitive runtime assets stay outside the repo:

| Asset | Default location | Notes |
| --- | --- | --- |
| YOLO/model weights | `D:\LabModels` (`LAB_MODELS_DIR`) | Referenced by `configs/model/detection_runtime.yaml` or env overrides. |
| Raw/aligned videos | `D:\LabVideo\raw_uploads` (`LAB_VIDEO_STORE_ROOT`) | Experiment folders keep `.video_ref.json` pointers instead of copied videos. |
| Approved material library | `D:\LabMaterialLibrary` (`LAB_MATERIAL_LIBRARY_ROOT`) | Backend can also read project-local `outputs/material_references` for dev runs. |
| Secrets | `.env` | Copy from `.env.example`; never commit real keys. |

## Repository Layout

```text
backend/      FastAPI service, port 8000
frontend/     Vite + React + TypeScript UI, port 5173
src/          key_action_indexer, labsopguard pipeline modules, experiment services
configs/      Runtime, model, scoring, reporting, SOP, and monitoring config
scripts/      Operational checks, rebuild tools, evaluation helpers
tests/        Pytest coverage
docs/         Specs, runbooks, and historical planning notes
```

## Quick Start

```powershell
cd D:\LabEmbodied
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
Copy-Item .env.example .env
```

Edit `.env` with `DASHSCOPE_API_KEY` and any local overrides for `LAB_MODELS_DIR`, `LAB_VIDEO_STORE_ROOT`, or `LAB_MATERIAL_LIBRARY_ROOT`.

Start both services:

```powershell
.\scripts\start_full_stack.ps1 -SkipRedis
```

Manual startup:

```powershell
$env:PYTHONPATH = "src"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

cd frontend
npm ci
npm run dev
```

## Validation

```powershell
python scripts/check_env.py --project-name LabEmbodied
python scripts/startup_runtime_preflight.py
python -m pytest -q
cd frontend
npm run build
```

For every development session, read `LabSOPGuard.md` first. It is still the project guardrail for YOLO weight resolution, DashScope/Qwen API use, the experiment processing chain, and evidence-pipeline scope.
