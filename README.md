# LabSOPGuard_VLA

Engineering template for laboratory SOP compliance monitoring with video understanding, event structuring, alerting, and report export.

Repository: https://github.com/Junxiong-Huang-VLA/LabSOPGuard_VLA.git

## What This Project Does

- Real-time or offline video monitoring
- Multi-level AI detection pipeline
- SOP violation detection and structured event output
- Batch inference and batch monitoring
- Per-session report export (PDF or TXT fallback)

## Project Layout

```text
LabSOPGuard/
  configs/                 # data, SOP rules, alerting, report settings
  data/
    raw/                   # raw input data (never overwrite)
    interim/               # manifests, extracted frames, temp artifacts
    processed/             # processed datasets
    splits/                # train/val/test splits
  docs/                    # technical docs
  scripts/                 # setup, scan, extraction, inference, monitoring
  src/project_name/        # core modules
  outputs/                 # predictions and reports
  web/                     # static project page
```

## Environment Rules

- Single environment for this project: `LabSOPGuard`
- Conda is the primary environment manager
- Reuse existing `LabSOPGuard` if it already exists

```powershell
conda create -n LabSOPGuard python=3.10 -y
conda activate LabSOPGuard
pip install -r requirements.txt
```

## Stable Workflow

### 1) Scan RGB/Depth video pairs and extract frames

```powershell
python .\scripts\scan_and_extract_frames.py `
  --dataset-root D:\labdata `
  --recursive `
  --max-frames-per-video 5 `
  --interval-sec 1.0 `
  --manifest-csv data\interim\video_manifest.csv `
  --report-json outputs\reports\video_scan_report.json `
  --frames-root data\interim\frames `
  --verbose
```

### 2) Batch inference (config-driven)

If `configs/data/dataset.yaml` contains:

```yaml
dataset:
  manifest_csv: data/interim/video_manifest.csv
```

Run:

```powershell
python .\scripts\infer.py --valid-only --max-frames 120 --target-fps 10
```

Or explicitly pass a manifest:

```powershell
python .\scripts\infer.py `
  --manifest-csv data\interim\video_manifest.csv `
  --valid-only `
  --max-frames 120 `
  --target-fps 10 `
  --batch-output-dir outputs\predictions\batch_infer
```

### 3) Batch monitor and report export (config-driven)

```powershell
python .\scripts\run_monitor.py --valid-only --max-frames 120 --target-fps 10
```

Or explicitly:

```powershell
python .\scripts\run_monitor.py `
  --manifest-csv data\interim\video_manifest.csv `
  --valid-only `
  --max-frames 120 `
  --target-fps 10 `
  --batch-output-dir outputs\predictions\batch_monitor
```

### 4) Batch export summary/events (config-driven)

```powershell
python .\scripts\export_results.py --valid-only --max-frames 120 --target-fps 10
```

## Main Outputs

- `outputs/predictions/batch_infer*/`
- `outputs/predictions/batch_monitor*/`
- `outputs/predictions/export_summary*.json`
- `outputs/reports/export_summary*.csv`
- `all_events.jsonl` and `all_events.csv`
- `*.report.pdf` (or TXT fallback)

## Vision MVP Inference Flow

The existing `scripts/infer.py` entry now supports a visual MVP path:

- YOLO26s-pose 2D detections/keypoints
- RGB-D fusion and robust depth sampling
- 3D keypoints in camera frame
- Optional transform to robot base frame (eye-on-base extrinsics)
- Structured exports for downstream control modules

Run single-source vision MVP:

```powershell
python .\scripts\infer.py `
  --video "D:\labdata\discription_pdf\first_person_移液与稀释_normal_correct_001_rgb.mp4" `
  --depth-path "D:\labdata\discription_pdf\first_person_移液与稀释_normal_correct_001_depth.mp4" `
  --sample-id vla_vision_mvp `
  --enable-pose `
  --export-3d `
  --export-base-frame `
  --debug-overlay
```

New outputs:

- `outputs/predictions/infer_events_pose.jsonl`
- `outputs/reports/infer_events_pose.csv`
- `outputs/predictions/debug_overlay/*` (if enabled)

Config file:

- `configs/vision_pose.yaml`
- Detailed guide: `docs/vision_mvp.md`

For a single-model pipeline (`yolo26s-pose` only), build pose dataset scaffold then train:

```powershell
python .\scripts\build_pose_dataset.py --clean
python .\scripts\train_yolo_lab.py `
  --dataset-yaml data\processed\yolo_pose_dataset\dataset.yaml `
  --model yolo26s-pose.pt `
  --name yolo26s_pose_lab_v1
```

Standardized one-command pipeline (build dataset + formal train + boxed compare video):

```powershell
python .\scripts\run_pose_training_pipeline.py `
  --epochs 30 `
  --run-name yolo26s_pose_lab_v1
```

## Git Quick Commands

```powershell
git add .
git commit -m "feat: update batch workflows and docs"
git push origin main
```

## Integrated System Mainline

Mainline entry is now `integrated_system`.

- Start: `python integrated_system/app_integrated.py`
- Web: `http://localhost:5001`
- Docs: `docs/integrated_system.md`

Core APIs:

- `POST /api/analyze`
- `GET /api/status/<task_id>`
- `GET /api/progress`
- `GET /api/download/<task_id>/<file_type>`
- `GET /api/health`

## 0-to-1 Documentation

- Project implementation doc: `docs/project_0to1_implementation.md`
- Developer execution playbook: `docs/developer_0to1_playbook.md`
- Unified pipeline entry: `python scripts/run_0to1_pipeline.py`
- Profile compare report: `python scripts/run_0to1_pipeline.py --from-stage analyze --to-stage analyze --compare-profiles`
