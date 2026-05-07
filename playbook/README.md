# LabSOPGuard Playbook

本目录是项目的统一知识库，包含所有角色文档和 Claude Code Skills。

```
playbook/
├── README.md          ← 你在这里
├── roles/             ← 12 个角色文档
│   ├── 01_detection_engineer.md
│   ├── 02_event_engine_developer.md
│   ├── 03_step_bridge_developer.md
│   ├── 04_vlm_integration_developer.md
│   ├── 05_backend_api_developer.md
│   ├── 06_frontend_developer.md
│   ├── 07_data_annotator.md
│   ├── 08_devops_engineer.md
│   ├── 09_qa_engineer.md
│   ├── 10_sop_domain_expert.md
│   ├── 11_material_publisher.md
│   └── 12_product_owner.md
└── skills/            ← 7 个 Claude Code Skills（归档副本）
    ├── dev-check.md
    ├── train-yolo.md
    ├── val-per-class.md
    ├── run-pipeline.md
    ├── stack-start.md
    ├── dataset-add.md
    └── debug-event.md
```

---

## 如何调用 Skills

Skills 的**可执行副本**在 `.claude/commands/`，Claude Code 从那里加载。
`playbook/skills/` 是归档/阅读副本，两者内容一致。

在 Claude Code 对话框中直接输入：

| 命令 | 触发时机 |
|------|---------|
| `/me` | **每次新对话开始时**，加载你的完整身份和项目状态 |
| `/dev-check` | 每次开始开发前 |
| `/train-yolo` | 需要训练新模型时 |
| `/val-per-class` | 需要查看各类检测效果时 |
| `/run-pipeline` | 触发或调试实验处理链路时 |
| `/stack-start` | 启动或重启全栈服务时 |
| `/dataset-add` | 合并新标注数据时 |
| `/debug-event` | 事件/clip/步骤推理异常时 |

---

## 如何使用角色文档

### 方式一：让 Claude Code 扮演角色
在对话中直接说：
> "你现在作为检测工程师，帮我评估当前数据集质量"
> "以步骤推理开发者的视角，检查这段代码"

### 方式二：开始新任务时提供上下文
> "参考 playbook/roles/01_detection_engineer.md，帮我准备下一次训练"

### 方式三：作为新成员入职手册
新加入的协作者（人类或 AI）阅读对应角色文档，了解职责边界和禁止事项。

---

## 权威来源说明

| 文件类型 | 权威路径 | 说明 |
|---------|---------|------|
| Skills（可执行） | `.claude/commands/*.md` | Claude Code 实际读取位置 |
| Skills（归档） | `playbook/skills/*.md` | 阅读/备份，不自动执行 |
| 角色文档（权威） | `playbook/roles/*.md` | 统一维护位置 |
| 角色文档（旧位置） | `docs/roles/*.md` | 已同步，后续以 playbook 为准 |
| 开发约束 | `LabSOPGuard.md` | 所有角色必须遵守 |
