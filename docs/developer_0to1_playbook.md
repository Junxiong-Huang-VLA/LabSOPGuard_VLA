# LabSOPGuard 开发者文档（0-1 执行版）

## 1. 适用范围

该文档面向直接落地项目的开发者，目标是“拿到仓库后尽快跑通并可持续迭代”。

## 2. 必备前置

- Windows PowerShell
- Conda
- Python 3.10
- 可选：NVIDIA GPU（用于加速推理/训练）

## 3. 一次性初始化

```powershell
python .\00_setup_environment.py --project-name LabSOPGuard --python-version 3.10
python .\14_check_environment.py --project-name LabSOPGuard
```

检查点：

- `outputs/reports/environment_check_report.json`
- 若 `status=fail`，先修环境再继续

## 4. 推荐日常命令

### 4.1 全链路执行（推荐）

```powershell
python .\scripts\run_0to1_pipeline.py --dataset-root D:\labdata --recursive --valid-only
```

### 4.1.1 SOP 规则档位切换（strict / mvp）

```powershell
# 严格档（默认）
python .\scripts\run_0to1_pipeline.py --sop-profile strict --valid-only

# MVP 档（放宽 PPE，默认不要求 lab_coat）
python .\scripts\run_0to1_pipeline.py --sop-profile mvp --valid-only
```

### 4.1.2 一键对比 strict vs mvp

```powershell
python .\scripts\run_0to1_pipeline.py --from-stage analyze --to-stage analyze --valid-only --compare-profiles
```

会生成：

- `outputs/reports/profile_compare/compare_profiles.json`
- `outputs/reports/profile_compare/compare_profiles.md`

### 4.1.3 自动导出 PPE 硬样本包（用于补标/再训练）

```powershell
python .\scripts\run_0to1_pipeline.py --from-stage analyze --to-stage analyze --build-hardcase-pack
```

会生成：

- `data/interim/hardcases/ppe_missing/`（触发帧及上下文截图）
- `data/interim/hardcases/ppe_missing_manifest.csv`

### 4.2 分阶段执行

```powershell
python .\scripts\run_0to1_pipeline.py --from-stage infer --to-stage analyze --valid-only
```

### 4.3 预演（不执行）

```powershell
python .\scripts\run_0to1_pipeline.py --dry-run --dataset-root D:\labdata
```

## 5. 核心脚本职责

- `scripts/scan_and_extract_frames.py`: 扫描 RGB/Depth 配对并抽帧
- `scripts/infer.py`: 事件推理与结构化输出
- `scripts/run_monitor.py`: 规则监控、违规检测、报告构建
- `scripts/export_results.py`: 批量导出 summary 与事件
- `scripts/build_audit_assets.py`: 审计资产构建
- `scripts/run_0to1_pipeline.py`: 统一编排入口

## 6. 关键约定

- 视频命名约定：`*_rgb.*` 与 `*_depth.*`
- 推荐默认清单路径：`data/interim/video_manifest.csv`
- 统一在同一环境中执行全流程，避免解释器漂移
- 每次正式运行保留：`outputs/reports/run_0to1_meta.json`
- 告警去抖默认读取 `configs/alerts/alerting.yaml` 的 `alerting.cooldown_seconds`

## 7. 常见问题

### 7.1 扫描有样本但后续没有跑

优先检查：

- `video_manifest.csv` 中 `valid_status` 是否为 `valid`
- 是否开启了 `--valid-only`
- 运行元数据中的 `manifest_csv` 路径是否正确

### 7.2 只想重跑导出

```powershell
python .\scripts\run_0to1_pipeline.py --from-stage export --to-stage analyze
```

会自动生成：

- `outputs/reports/violation_analysis.json`
- `outputs/reports/violation_analysis.md`
- `outputs/reports/violation_diagnostics.json`
- `outputs/reports/violation_diagnostics.md`

如需覆盖告警配置，可在底层脚本传入：

```powershell
python .\scripts\run_monitor.py --alert-config configs/alerts/alerting.yaml
python .\scripts\infer.py --alert-config configs/alerts/alerting.yaml
python .\scripts\export_results.py --alert-config configs/alerts/alerting.yaml
```

### 7.3 产物回溯某次执行

查看：

- `outputs/reports/run_0to1_meta.json`

重点字段：

- `executed_stages`
- `stage_records`
- `manifest_csv`
- `valid_only`

## 8. 开发流程建议

1. 先 `--dry-run` 检查参数
2. 小样本验证（低帧数）
3. 全量执行并记录元数据
4. 对失败阶段单独重跑
5. 合并结果并更新文档

## 9. 提交前检查

```powershell
python -m py_compile scripts\run_0to1_pipeline.py
python .\scripts\run_0to1_pipeline.py --dry-run
conda run -n LabSOPGuard python -m unittest discover -s tests -p "test_sop_engine.py" -v
```

## 10. 扩展方向

- 为每阶段增加自动回归测试
- 统一错误码与失败原因分类
- 增加运行配置快照（规则/模型版本）
