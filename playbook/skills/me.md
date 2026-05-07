# /me — 加载项目主理人上下文

调用此 skill 后，Claude Code 将完整加载你的身份、项目状态和协作约定，无需每次重新说明背景。

## 执行步骤

1. 读取 `playbook/roles/00_project_owner.md`，加载身份定位和工作约定
2. 读取 `LabSOPGuard.md`，确认当前核心约束
3. 读取 `configs/model/detection_runtime.yaml`，确认当前权重
4. 输出一段简短的上下文确认，格式如下：

---

**已加载主理人上下文**

- 身份：独立全栈负责人（研究 + 工程双轨）
- 当前权重：`<model 字段值>`
- P0 任务：tube/tube-cap/spearhead/pipette 标注补充
- 协作模式：你决策，我执行

准备好了，说吧。

---

## 附：常用 skill 速查

| 命令 | 用途 |
|------|------|
| `/dev-check` | 开发前检查 |
| `/train-yolo` | 训练新模型 |
| `/val-per-class` | 查看检测效果 |
| `/run-pipeline` | 触发实验链路 |
| `/stack-start` | 启动全栈服务 |
| `/dataset-add` | 合并新标注数据 |
| `/debug-event` | 事件链路诊断 |
