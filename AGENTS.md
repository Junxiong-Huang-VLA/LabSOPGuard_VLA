# AGENTS.md

## 目标

本仓库的正式版主线是 LabSOPGuard 视频分析与实验过程理解系统。任何修改都必须优先保护以下资产：

- `src/experiment/` 已跑通的实验处理链路
- `video_analysis_pipeline.py` 对应的 YOLO + Qwen VL 视频分析链路
- `backend/main.py` 中现有 `/api/v1/experiments/*` 与 `/api/v1/video-analysis/*` 接口
- `frontend-app/src/pages/Upload.tsx` 所在的上传验证流程

## 主线规则

- 不推倒重做
- 不破坏 DashScope / Qwen VL 接入
- 新的正式能力优先落到 `src/labsopguard/`
- 历史报告、手工调试脚本、临时输出物放到 `archive/` 或 `docs/archive/`
- 运行产物只允许落到 `outputs/` / `uploads/`

## 目录判定

### 正式主线

- `backend/`
- `frontend-app/`
- `src/experiment/`
- `src/labsopguard/`
- `configs/`
- `tests/`
- `docs/`

### 兼容或历史区域

- `src/project_name/`
- `integrated_system/`
- `frontend/`
- `demo/`

这些区域可以保留，但不是正式版继续扩展的首选位置。

## 修改要求

- 结构化输出变化必须同步测试和文档
- API 兼容变化必须保留旧路径
- 如果清理文件，优先归档而不是盲删
