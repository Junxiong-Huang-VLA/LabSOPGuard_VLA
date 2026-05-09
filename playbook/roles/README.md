# 角色文档索引

本目录定义了 LabSOPGuard 项目的 12 种协作角色，各司其职，覆盖从数据标注到产品决策的完整链路。

## 角色总览

| # | 角色 | 核心职责 | 主要代码区域 |
|---|------|---------|------------|
| 01 | [检测工程师](01_detection_engineer.md) | YOLO 训练/评估/权重管理 | `data/dataset/`、`configs/model/`、`tools/` |
| 02 | [事件引擎开发者](02_event_engine_developer.md) | 五类物理事件检测 | `src/labsopguard/event_preprocessing/` |
| 03 | [步骤推理开发者](03_step_bridge_developer.md) | SOP 步骤匹配与合规判断 | `src/labsopguard/step_bridge/` |
| 04 | [VLM 集成开发者](04_vlm_integration_developer.md) | Qwen/DashScope 调用、语义增强 | `src/labsopguard/video_analysis.py`、`reasoning.py` |
| 05 | [后端 API 开发者](05_backend_api_developer.md) | FastAPI 路由、任务队列、实验生命周期 | `backend/`、`src/labsopguard/workflow.py` |
| 06 | [前端开发者](06_frontend_developer.md) | 工作区 UI、视频播放器、素材时间轴 | `frontend-app/src/` |
| 07 | [数据标注员](07_data_annotator.md) | 13 类目标标注、质量控制 | Roboflow 云端、`data/dataset/` |
| 08 | [DevOps 工程师](08_devops_engineer.md) | 环境搭建、服务管理、AutoDL 训练服务器 | `scripts/`、`environment.yml` |
| 09 | [测试工程师](09_qa_engineer.md) | 单元/集成/E2E 测试、回归验收 | `tests/` |
| 10 | [SOP 领域专家](10_sop_domain_expert.md) | 实验步骤定义、合规标准、steps.json | `data/raw/sop_docs/` |
| 11 | [素材发布工程师](11_material_publisher.md) | clip/keyframe 生成、素材命名与搜索 | `src/labsopguard/material_publishing/` |
| 12 | [产品负责人](12_product_owner.md) | 优先级决策、验收标准、约束文档审批 | `LabSOPGuard.md`、`CLAUDE.md` |

## Skills（Claude Code 快捷命令）

位于 `.claude/commands/`，在对话中用 `/命令名` 调用：

| 命令 | 功能 |
|------|------|
| `/dev-check` | 开发前检查：权重/API Key/测试 |
| `/train-yolo` | YOLO 训练助手：数据统计→训练→权重切换 |
| `/val-per-class` | 各类最佳检测样图生成与展示 |
| `/run-pipeline` | 触发/调试实验完整处理链路 |
| `/stack-start` | 一键启动全栈服务并健康检查 |
| `/dataset-add` | 合并新标注数据到主数据集 |
| `/debug-event` | 事件检测链路诊断 |

## 当前紧急任务（P0）

1. **数据标注员**：补充 `tube`/`tube-cap`/`spearhead`/`pipette` 标注（≥300条/类）
2. **步骤推理开发者**：提升 step_bridge 置信度至 ≥ 0.5
