# LabSOPGuard 0-1 项目实施文档

## 1. 项目目标

在单机（Windows + Conda）环境下，完成 LabSOPGuard 从数据扫描到违规检测、报告导出的可复现主链路，确保新成员可在 1 天内跑通最小闭环。

交付目标：

- 可执行主链路：`scan -> infer -> monitor -> export -> analyze -> audit`
- 可追踪输出：每个阶段均产出结构化文件
- 可复现实验：同一输入数据可重复运行并对比结果
- 可维护工程：具备统一入口脚本、阶段执行与运行元数据

## 2. 当前基线（已具备）

仓库已存在以下核心能力：

- 数据扫描与抽帧：`scripts/scan_and_extract_frames.py`
- 批量推理：`scripts/infer.py`
- 监控与报告：`scripts/run_monitor.py`
- 批量导出：`scripts/export_results.py`
- 审计资产构建：`scripts/build_audit_assets.py`
- 集成入口：`scripts/run_0to1_pipeline.py`

本轮补齐内容：

- 修复 0-1 编排中的 `manifest` 贯穿问题
- 增强阶段控制与 dry-run
- 增加运行元数据输出，便于回溯与审计

## 3. 0-1 分阶段实施

### 阶段 A：环境与配置基线

目标：统一依赖、避免脚本在不同机器上行为不一致。

执行：

1. 创建或复用环境 `LabSOPGuard`
2. 安装依赖并运行 `14_check_environment.py`
3. 检查 `configs/data/dataset.yaml` 的 `manifest_csv` 路径

验收标准：

- `outputs/reports/environment_check_report.json` 状态为 `pass`
- Python 可执行 `scripts/infer.py --help`

### 阶段 B：数据接入与可用性校验

目标：保证 RGB/Depth 数据对齐与可读。

执行：

1. 扫描数据集并生成 `video_manifest.csv`
2. 进行抽帧和扫描报告导出

验收标准：

- `data/interim/video_manifest.csv` 生成成功
- 报告中 `valid_samples > 0`

### 阶段 C：推理与 SOP 监控

目标：产出可分析的事件与违规结构化数据。

执行：

1. 批量推理生成事件
2. 批量监控生成违规列表与运行状态

验收标准：

- `outputs/predictions/batch_infer/summary.json` 存在
- `outputs/predictions/batch_monitor/summary.json` 存在
- 每条样本均可追溯到对应输出 JSON

### 阶段 D：导出与审计

目标：形成审计可交付产物。

执行：

1. 导出 summary + events（JSON/CSV）
2. 生成审计资产（截图、聚合文件）

验收标准：

- `outputs/predictions/export_summary.json` 存在
- `outputs/reports/export_summary.csv` 存在
- `outputs/reports/audit_assets/` 有内容

## 4. 统一执行入口（推荐）

使用新版编排脚本：

```powershell
python .\scripts\run_0to1_pipeline.py `
  --dataset-root D:\labdata `
  --recursive `
  --valid-only
```

可选能力：

- 仅打印命令不执行：`--dry-run`
- 只跑部分阶段：`--from-stage infer --to-stage analyze`
- 跳过某阶段：`--skip-audit`
- 全量样本（含 invalid）：`--all-samples`
- 规则配置切换：
  - 严格模式：`--sop-profile strict`
  - MVP 模式：`--sop-profile mvp`
- 一键对比 strict vs mvp：`--compare-profiles`

运行后会输出元数据：

- `outputs/reports/run_0to1_meta.json`
- `outputs/reports/violation_analysis.json`
- `outputs/reports/violation_analysis.md`
- `outputs/reports/violation_diagnostics.json`
- `outputs/reports/violation_diagnostics.md`
- `outputs/reports/profile_compare/compare_profiles.json`
- `outputs/reports/profile_compare/compare_profiles.md`

## 5. 产物清单（审计视角）

关键目录：

- `data/interim/video_manifest.csv`
- `outputs/predictions/batch_infer/`
- `outputs/predictions/batch_monitor/`
- `outputs/predictions/export_summary.json`
- `outputs/reports/export_summary.csv`
- `outputs/reports/run_0to1_meta.json`

## 6. 风险与应对

已识别风险：

- `mediapipe` API 差异导致手部检测能力降级
- 数据路径/命名不规范导致 `manifest` 无法对齐
- GPU 环境不一致导致推理性能或结果波动

应对策略：

- 先跑 `14_check_environment.py` 再跑主链路
- 严格使用 `_rgb/_depth` 后缀命名
- 对关键运行保留 `run_0to1_meta.json` 与配置快照

## 7. 里程碑建议

- M1（D+1）：跑通单机最小链路并产出报告
- M2（D+3）：完成数据质量回归脚本与失败样本归因
- M3（D+7）：固定阶段验收模板并形成周度复盘
