# LabSOPGuard 开发者文档

## 1. 目标读者

- 新加入项目的算法工程师/后端工程师
- 需要在公司与家里双机协作开发的成员
- 需要复现实验、训练、推理与报告流程的开发者

---

## 2. 目录速览（开发关注）

- `integrated_system/`：Web 主线（Flask + 任务流水线）
- `scripts/`：数据、训练、推理、可视化、同步脚本
- `configs/`：数据、规则、报告、模型配置
- `data/`：原始/中间/处理后数据
- `outputs/`：训练输出、预测结果、报告与可视化
- `docs/`：项目文档与规范

---

## 3. 环境准备

推荐环境：`conda`，环境名固定 `LabSOPGuard`

```powershell
conda create -n LabSOPGuard python=3.10 -y
conda activate LabSOPGuard
pip install -r requirements.txt
```

环境检查：

```powershell
python .\14_check_environment.py --project-name LabSOPGuard
```

---

## 4. 本地预览（Web 主线）

前台稳定启动（推荐排障）：

```powershell
$env:MPLCONFIGDIR="D:\LabEmbodiedVLA\LabSOPGuard\.matplotlib"
$env:INTEGRATED_HOST="127.0.0.1"
$env:INTEGRATED_PORT="5001"
python .\integrated_system\app_integrated.py
```

浏览器：

- `http://127.0.0.1:5001`
- `http://127.0.0.1:5001/api/health`

脚本启动/停止：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_preview.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\stop_preview.ps1
```

---

## 5. 数据标准化与审计

标准化构建（8:2）：

```powershell
python .\scripts\standardize_dataset_8_2.py `
  --src-root data\interim\frames `
  --class-yaml data\processed\yolo_dataset\dataset.yaml `
  --out-root data\processed\yolo_dataset_std_80_20 `
  --clean
```

Pose 数据集构建：

```powershell
python .\scripts\build_pose_dataset.py `
  --src-root data\processed\yolo_dataset_std_80_20 `
  --out-root data\processed\yolo_pose_dataset_std_80_20 `
  --schema configs\data\pose_keypoints_schema.yaml `
  --clean
```

数据审计：

```powershell
python .\scripts\audit_pose_dataset.py --dataset-yaml data\processed\yolo_pose_dataset_std_80_20\dataset.yaml
```

---

## 6. 训练与评估

训练：

```powershell
python .\scripts\train_yolo_lab.py `
  --dataset-yaml data\processed\yolo_pose_dataset_std_80_20\dataset.yaml `
  --model yolo26s-pose.pt `
  --epochs 30 `
  --imgsz 960 `
  --batch 8 `
  --device 0 `
  --workers 0 `
  --name yolo26s_pose_lab_vX
```

误检/漏检分析：

```powershell
python .\scripts\analyze_detection_errors.py `
  --dataset-yaml data\processed\yolo_pose_dataset_std_80_20\dataset.yaml `
  --weights outputs\training\yolo26s_pose_lab_vX\weights\best.pt `
  --conf 0.30 `
  --out-json outputs\reports\detection_error_report_vX.json `
  --out-csv outputs\reports\detection_error_report_vX.csv
```

---

## 7. 可视化与验收

单模型检测视频（演示）：

```powershell
python .\scripts\render_detection_video.py `
  --video "D:\labdata\discription_pdf\first_person_复杂长操作_normal_correct_001_rgb.mp4" `
  --weights outputs\training\yolo26s_pose_lab_vX\weights\best.pt `
  --conf 0.30 `
  --target-fps 10 `
  --duration-sec 60 `
  --out-video outputs\predictions\fine_tuned_detect_vX_60s.mp4 `
  --out-json outputs\predictions\fine_tuned_detect_vX_60s.json
```

---

## 8. 常见故障处理

### 8.1 `mediapipe has no attribute 'solutions'`

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_mediapipe_compat.ps1 -EnvName LabSOPGuard
```

### 8.2 5001 端口连不上

1. 先前台启动确认服务是否正常
2. 检查是否有旧 PID 文件但进程不存在
3. 使用 `stop_preview.ps1` + `start_preview.ps1` 重启

### 8.3 Git 凭据失效

```powershell
git config --global credential.helper manager
git push origin main
```

---

## 9. 双机协作（公司/家里）

开始前：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action pull
```

结束后：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\sync_dual_pc.ps1 -Action save -Message "feat: update"
```

双远端同步：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\push_both_remotes.ps1 -ForceMirror
```

---

## 10. 开发约定（建议）

- 先跑通最小链路，再加复杂模块
- 每次训练迭代至少产出：
  - `best.pt`
  - 误检/漏检报告（json/csv）
  - 60 秒可视化视频
- 关键配置走 `configs/`，避免硬编码
- 新脚本必须可 CLI 运行并有基本日志

---

## 11. 建议的近期里程碑

1. 固化手部检测兼容版本与回归测试
2. 困难类别专项增广训练（lab_coat/gloved_hand/spatula）
3. 集成统一“训练完成自动出报告+视频”流水线
4. 推进多摄像头输入与队列化任务执行
