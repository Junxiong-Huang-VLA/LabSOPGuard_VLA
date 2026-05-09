# 角色十二：产品负责人（Product Owner）

## 职责定位

把控系统整体方向与优先级，定义验收标准，协调各角色工作，决策技术取舍，是唯一有权修改核心约束文档（`LabSOPGuard.md`、`CLAUDE.md`）的角色。

## 核心工作内容

### 当前系统状态（2026-04-22）
- **评级：B** — 本地可用，内部 Demo 就绪
- 完整端到端链路已验证（基准实验 `c404e890`）
- YOLO 权重：`yolo26s_autodl_8_1_1`（mAP50=0.977，测试集）
- 尚未 production-grade：`tube/tube-cap/spearhead/pipette` 四类无训练数据

### 优先级队列（由高到低）

| 优先级 | 任务 | 负责角色 |
|--------|------|---------|
| P0 | 补充 tube/pipette/spearhead 标注数据（≥300条/类） | 数据标注员 |
| P0 | step_bridge 置信度提升（目标 ≥ 0.5） | 步骤推理开发者 |
| P1 | goggles 检测能力（目前依赖 Qwen 语义） | 检测工程师 |
| P1 | lab_coat mAP50 从 0.940 提升至 ≥ 0.96 | 检测工程师 |
| P2 | 生产环境部署（Docker + nginx） | DevOps 工程师 |
| P2 | 前端 UI 打磨（移动端适配） | 前端开发者 |
| P3 | 多摄像头同步支持 | 事件引擎开发者 |

### 核心约束文档变更审批
修改以下文件前必须经过 Product Owner 审批：
- `LabSOPGuard.md` — 开发约束文档
- `CLAUDE.md` — Claude Code 约束
- `configs/model/detection_runtime.yaml` — 权重配置
- `src/labsopguard/semantic_events.py` — 事件类型枚举

### 验收标准（升级到 A 级所需）
- [ ] tube/tube-cap/spearhead/pipette 四类 mAP50 ≥ 0.95
- [ ] 全类 mAP50 ≥ 0.98
- [ ] step_bridge 在标准视频中 `compliant` 步骤 ≥ 70%
- [ ] 后端 API 可用性 ≥ 99.9%（连续 7 天）
- [ ] 视频处理延迟 ≤ 3 分钟/小时视频

### 已知技术债务
| 债务 | 风险 | 计划处理时间 |
|------|------|------------|
| `lab_preprocessing/` 孤立子项目 | 低（无依赖） | 确认无依赖后删除 |
| `start_frontend_*.py` 多个遗留启动文件 | 低 | 统一清理 |
| detect/segment 混合数据集警告 | 中 | 数据集重构时处理 |

### 沟通规范
- 与 Claude Code 协作时，每次 session 开始前：先 `/dev-check`
- 新功能需求：先描述业务目标，再讨论技术实现
- 权重切换：只需告知 run_name，Claude Code 自动更新 yaml

## 禁止事项（对所有角色）
- 任何角色不得在未经 Product Owner 审批的情况下修改核心约束文档
- 不得在生产环境直接测试未经 QA 验收的权重
