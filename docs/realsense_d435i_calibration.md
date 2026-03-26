# RealSense D435i + Dobot Hand-Eye Calibration

This project now supports ROS-style hand-eye extrinsics in config:

- `configs/robot/bridge.yaml`
- `lab_titration_vla/deploy/calibration/camera_handeye.yaml`

Supported extrinsics formats:

1. Matrix form
- `rotation: [r00..r22]`
- `translation: [tx, ty, tz]`

2. Quaternion form (recommended for ROS)
- `transform.translation.{x,y,z}`
- `transform.rotation.{x,y,z,w}`

Current configured calibration:

- calibration name: `dobot_global_calibration`
- type: `eye_on_base`
- robot base frame: `dummy_link`
- robot effector frame: `Link6`
- camera frame: `global_color_optical_frame`
- marker frame: `global_aruco_marker`

## Quick verification command

```powershell
cd D:\LabEmbodiedVLA\LabSOPGuard
python .\scripts\calibration_check.py `
  --robot-config configs/robot/bridge.yaml `
  --u 640 --v 360 --depth-m 0.55 `
  --out-json outputs/reports/calibration_check.json
```

Or use a detection bbox center:

```powershell
python .\scripts\calibration_check.py `
  --robot-config configs/robot/bridge.yaml `
  --bbox 520,280,760,520 --depth-m 0.55
```

## Batch error evaluation (with measured robot points)

Prepare a CSV:

- columns: `name,u,v,depth_m,measured_x,measured_y,measured_z`

Run:

```powershell
python .\scripts\calibration_error_eval.py `
  --robot-config configs/robot/bridge.yaml `
  --samples-csv D:\labdata\calib_points.csv `
  --out-json outputs/reports/calibration_error_report.json `
  --out-csv outputs/reports/calibration_error_points.csv
```

Outputs:

- per-point residuals (`err_x, err_y, err_z, err_dist`)
- aggregate metrics (`MAE/RMSE`), used for calibration quality gating
