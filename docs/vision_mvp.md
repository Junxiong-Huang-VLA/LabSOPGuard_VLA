# Vision MVP (Pose + Depth + 3D)

## Scope

This MVP adds a visual chain for lab manipulation support:

1. RGB input
2. YOLO26s-pose inference
3. RGB-D depth fusion with robust window sampling
4. Camera-frame 3D recovery
5. Optional camera->robot-base transform with hand-eye extrinsics
6. Structured export for downstream controller consumption

This module is visual only and does not execute robot motion.

## Inputs

- RGB source: image or video (`--video` in `scripts/infer.py`)
- Optional depth source: image/video (`--depth-path`)
- Camera intrinsics: `configs/vision_pose.yaml` or `--camera-info`
- Extrinsics (eye-on-base): `configs/vision_pose.yaml` or `--extrinsics`

## Outputs

- Standard infer outputs remain unchanged:
  - `outputs/predictions/infer_events.jsonl`
  - `outputs/reports/infer_events.csv`
- Pose/3D exports (new):
  - `outputs/predictions/infer_events_pose.jsonl`
  - `outputs/reports/infer_events_pose.csv`
- Optional debug overlays:
  - `outputs/predictions/debug_overlay/...`

Each pose record includes:

- `sample_id`
- `video_path` / `image_path`
- `detections_2d`
- `keypoints_3d_camera`
- `keypoints_3d_base` (if enabled)
- `grasp_target`
- `pour_target`
- `vision_quality`
- `warnings`
- `timestamp`

## Run

Single video:

```powershell
python .\scripts\infer.py `
  --video "D:\labdata\discription_pdf\first_person_移液与稀释_normal_correct_001_rgb.mp4" `
  --depth-path "D:\labdata\discription_pdf\first_person_移液与稀释_normal_correct_001_depth.mp4" `
  --sample-id dilution_demo `
  --enable-pose `
  --export-3d `
  --export-base-frame `
  --debug-overlay
```

## Train YOLO26s-pose Only (Current Project)

Build a pose dataset scaffold from your existing detection labels:

```powershell
python .\scripts\build_pose_dataset.py `
  --src-root data\processed\yolo_dataset `
  --out-root data\processed\yolo_pose_dataset `
  --schema configs\data\pose_keypoints_schema.yaml `
  --clean
```

Train pose model:

```powershell
python .\scripts\train_yolo_lab.py `
  --dataset-yaml data\processed\yolo_pose_dataset\dataset.yaml `
  --model yolo26s-pose.pt `
  --epochs 100 `
  --imgsz 960 `
  --batch 8 `
  --workers 0 `
  --name yolo26s_pose_lab_v1
```

Important:

- The scaffolded pose labels currently use placeholder keypoints `(0,0,0)`.
- For real keypoint performance, you must replace placeholders with manual keypoint annotations.
- This still keeps a single-model route (`yolo26s-pose`) and no parallel detector model is required.

Batch manifest:

```powershell
python .\scripts\infer.py `
  --manifest-csv data\interim\video_manifest.csv `
  --valid-only `
  --enable-pose `
  --export-3d `
  --export-base-frame `
  --batch-output-dir outputs\predictions\batch_infer_pose
```

## Depth and Geometry Notes

- Depth fusion uses a 5x5 window by default (`depth_window_size`).
- Invalid values (`0`, `NaN`, out-of-range) are filtered.
- Median depth is used for robustness.
- 3D conversion:
  - `x = (u - cx) * z / fx`
  - `y = (v - cy) * z / fy`
  - `z = depth(u, v)`
- Per-point quality fields:
  - `valid`
  - `valid_depth_ratio`
  - `depth_std`
  - `source`

## Known Limitations

- Pose keypoint names are config-driven; if model keypoint schema differs, names may be generic (`kp_i`).
- Depth video alignment is currently frame-index based (MVP).
- Orientation output is a placeholder hint for downstream replacement by dedicated pose estimators.
- This module does not perform robot control execution.
