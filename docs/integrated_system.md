# Integrated System Mainline (`integrated_system`)

This is the only web mainline for the experimental video auto-analysis demo.

## Start

```powershell
python integrated_system/app_integrated.py
```

Open:

- `http://localhost:5001`

## Environment Variables

- `DOUBAO_API_KEY` or `ARK_API_KEY` or `OPENAI_API_KEY`: AI analysis key
- `OPENAI_BASE_URL` (optional, default: `https://ark.cn-beijing.volces.com/api/v3`)
- `INTEGRATED_PORT` (optional, default: `5001`)
- `INTEGRATED_ENABLE_HAND_DETECTION` (`1`/`0`)
- `INTEGRATED_ENABLE_AI_ANALYSIS` (`1`/`0`)
- `INTEGRATED_ENABLE_PDF` (`1`/`0`)
- `INTEGRATED_ENABLE_STEP_CHECK` (`1`/`0`)
- `INTEGRATED_ENABLE_VIDEO_EXPORT` (`1`/`0`)
- `INTEGRATED_KEYFRAME_DIFF_THRESHOLD` (default `18.0`)
- `INTEGRATED_KEYFRAME_MIN_INTERVAL_SEC` (default `1.2`)
- `INTEGRATED_MAX_KEYFRAMES` (default `12`)

## API

- `GET /` integrated web page
- `POST /api/analyze` upload video and start task
- `GET /api/status/<task_id>` query task status
- `GET /api/progress?task_id=<task_id>` SSE progress stream
- `GET /api/download/<task_id>/<file_type>` download outputs
- `GET /api/health` health check

Task status payload keys:

- `task_id`
- `status` (`pending` / `running` / `completed` / `failed`)
- `progress`
- `current_stage`
- `message`
- `outputs`

## Output Directory

All outputs go to:

- `integrated_system/outputs/<timestamp>_<task_id>/`

Typical files:

- `hand_annotated.mp4`
- `hand_detection.json`
- `keyframe_*.jpg` (under `keyframes/`)
- `part1_keyframes.json`
- `keyframe_ai_analysis.json`
- `overall_summary.txt`
- `alarm_log.json` (if step check enabled)
- `integrated_analysis_report.pdf`
- `task_result.json`

## Minimal Verification

1. Start service with `python integrated_system/app_integrated.py`
2. Open `http://localhost:5001`
3. Upload a video and click analyze
4. Watch progress from UI (SSE + polling)
5. Download annotated video / PDF / JSON outputs
