# OpenClaw 只读证据包使用文档

本文档说明 OpenClaw 如何在不连接 LabSOPGuard 后端的情况下，读取、验证、检索和评估 `key_action_indexer` 生成的可索引化实验素材证据包。

## 目标

- 证据包是离线只读目录，不需要 FastAPI、数据库服务、前端页面或原始工作站路径。
- OpenClaw 只读取包内相对路径入口：`evidence_package_manifest.json`、`key_material_references.jsonl`、`physical_change_log.jsonl`、`time_alignment.json` 和可选 SQLite 索引。
- 所有可解引用素材路径必须相对证据包根目录，跨电脑复制后仍可验证和查询。
- 动作判断只基于包内证据，包括 YOLO 关键动作片段、手物交互、before/after 物理状态、时间对齐和检索分数。

## 推荐运行环境

在本机真实 YOLO 推理时优先使用 CUDA 版 `LabSOPGuard` conda 环境：

```powershell
$env:PYTHONPATH="D:\LabCapability\src"
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli --help
```

`DetectionConfig.yolo_device` 默认值为 `auto`：有 CUDA 时传 `device=0` 给 ultralytics，没有 CUDA 时自动退回 CPU。若使用 `D:\anaconda\python.exe`，当前环境是 CPU 版 torch，不适合真实视频批量 YOLO 推理。

## 证据包目录

典型目录如下：

```text
material_references/
  evidence_package_manifest.json
  manifest.json
  素材索引.jsonl
  key_material_references.jsonl
  key_material_references.sqlite
  physical_change_log.jsonl
  time_alignment.json
  关键帧/
  关键片段/
```

OpenClaw 应先读取 `evidence_package_manifest.json` 的 `entrypoints`，不要硬编码文件名。`package://{package_id}/...` URI 是逻辑引用，实际读取时映射到证据包根目录下的相对路径。

## 构建

从正式素材包构建只读证据包：

```powershell
$env:PYTHONPATH="D:\LabCapability\src"
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli evidence-package-build `
  --package-root "D:\LabCapability\data\sessions\<实验>\material_references" `
  --key-action-index-dir "D:\LabCapability\data\sessions\<实验>\key_action_index" `
  --source-manifest "D:\LabCapability\data\sessions\<实验>\key_action_index\manifest.json" `
  --package-id "<package_id>" `
  --experiment-id "<experiment_id>"
```

构建过程会生成：

- `evidence_package_manifest.json`：跨电脑入口清单、校验和、来源说明。
- `key_material_references.jsonl`：标准化检索行。
- `key_material_references.sqlite`：本地 FTS 检索索引。
- `physical_change_log.jsonl`：物理变化事件；优先使用显式 `before_state/after_state`，缺失时从 YOLO bbox 时序推导 before/after。
- `time_alignment.json`：消息时间到视频时间的映射策略。

## 验证

普通验证：

```powershell
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli evidence-package-validate `
  --package-root "D:\LabCapability\data\sessions\<实验>\material_references"
```

严格验证：

```powershell
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli evidence-package-validate `
  --package-root "D:\LabCapability\data\sessions\<实验>\material_references" `
  --strict
```

验证重点：

- `backend_required` 必须为 `false`。
- 主入口和素材路径不能是绝对路径。
- 主素材文件、clip、preview、keyframe 必须存在。
- `physical_change_log` 不能引用不存在的 `material_id`。
- 跨电脑复制后再次运行验证应保持通过。

## 同步到 LabSOPGuard 前端

OpenClaw 的主方案仍然是只读证据包，不依赖 LabSOPGuard 后端。如果需要把一次离线 GPU 实跑结果展示到既有 LabSOPGuard 前端实验页，可以使用独立的 `frontend-sync` 收口命令。它只按路径同步文件，不导入后端模块。

```powershell
$env:PYTHONPATH="D:\LabCapability\src"
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli frontend-sync `
  --target-experiment-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>" `
  --source-session-dir "D:\LabCapability\data\sessions\<实验>" `
  --third-person-video "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>\raw\third.mp4" `
  --first-person-video "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>\raw\first.mp4" `
  --run-yolo-overlay `
  --require-yolo-overlay `
  --yolo-device auto `
  --output-summary "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>\reports\frontend_sync_summary.json"
```

只做完整性校验：

```powershell
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli frontend-sync `
  --target-experiment-dir "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>" `
  --validate-only `
  --require-yolo-overlay `
  --output-summary "D:\LabCapability\LabSOPGuard\outputs\experiments\<experiment_id>\reports\frontend_sync_validation.json"
```

`frontend-sync` 会校验这些前端展示闭环条件：

- `experiment_focus_window.json` 必须覆盖所有真实实验 episode，不能只取第一个 7 秒片段。
- 双视角 `first_person_yolo_annotated.mp4` 和 `third_person_yolo_annotated.mp4` 必须存在，且时长接近实验窗口。
- `key_material_references.jsonl` 路径必须可跨电脑解引用，素材行应处于可发布/可检索状态。
- `evidence_package_manifest.json`、`time_alignment.json`、`physical_change_log.jsonl` 必须通过严格验证。
- `job_status.json` 和 `experiment.json` 会被更新为前端可见的 completed/analyzed 状态。

## 查询

单条查询：

```powershell
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli evidence-package-query `
  --package-root "D:\LabCapability\data\sessions\<实验>\material_references" `
  --query "试剂瓶有没有归位" `
  --message-time "2026-05-08T15:38:20+08:00" `
  --limit 8
```

返回结果包含：

- `intent`：动作判断意图，如 `return_position_check`、`object_move_check`、`hand_object_check`。
- `time_context`：消息时间映射到视频秒数，以及检索窗口。
- `judgement`：`correct`、`incorrect` 或 `insufficient`。
- `evidence_bundles`：候选素材、clip/keyframe 相对路径、关联物理变化、YOLO 决策链和检索分数。

## 批量评估

准备查询集：

```json
{
  "queries": [
    {
      "query": "试剂瓶有没有归位",
      "expected_intent": "return_position_check"
    },
    {
      "query": "有没有手和天平交互",
      "expected_intent": "hand_object_check",
      "expected_event_type": "hand_object_interaction"
    }
  ]
}
```

运行评估：

```powershell
& "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe" -m key_action_indexer.cli evidence-package-eval `
  --package-root "D:\LabCapability\data\sessions\<实验>\material_references" `
  --queries "D:\LabCapability\data\sessions\<实验>\reports\evidence_package_eval_queries.json" `
  --output "D:\LabCapability\data\sessions\<实验>\reports\evidence_package_eval.json" `
  --limit 8
```

`evidence-package-eval` 会输出 `evidence_package_eval.v1`，对存在期望值的字段计算：

- `intent_match_rate`
- `status_match_rate`
- `label_match_rate`
- `material_top1_match_rate`
- `material_topk_match_rate`
- `event_type_match_rate`
- `insufficient_rate`

没有期望值的查询仍会执行并保留真实返回，可用于探索性问题清单。

## Python API

```python
from key_action_indexer import EvidencePackage, evaluate_evidence_package_queries

package = EvidencePackage.load(r"D:\LabCapability\data\sessions\<实验>\material_references")
result = package.query("称量纸有没有移动", limit=8)

eval_result = evaluate_evidence_package_queries(
    package.root,
    r"D:\LabCapability\data\sessions\<实验>\reports\evidence_package_eval_queries.json",
    output_path=r"D:\LabCapability\data\sessions\<实验>\reports\evidence_package_eval.json",
)
```

## OpenClaw 接入约定

- OpenClaw 不写回证据包，不启动 LabSOPGuard 后端，不依赖原始视频绝对路径。
- 先读 `evidence_package_manifest.json`，再按 `entrypoints` 加载数据。
- 展示素材时优先使用 `clip.relative_path`、`preview.relative_path`、`key_frames[].relative_path`。
- 判断动作时优先看 `judgement` 和 `physical_changes`，不要只看 YOLO 框图。
- 时间问题优先传 `message_sent_at`，让 `time_alignment.json` 控制窗口。
- 若 `judgement.status=insufficient`，OpenClaw 应把原因暴露给用户，而不是补充外部臆测。

## 常见问题

- 查询没有结果：先验证 `key_material_references.jsonl` 是否为空，再检查 `素材索引.jsonl` 是否生成。
- 归位判断总是不充分：检查 `physical_change_log.jsonl` 中是否有 `before` 和 `after`。新逻辑可从 YOLO bbox 推导，但需要素材行带有 `yolo_evidence`、`yolo_interactions` 或 `interaction_events`。
- 跨电脑路径失效：运行 `evidence-package-validate --strict`，修复主字段里的绝对路径；历史 payload 中的绝对路径不作为读取入口。
- GPU 未使用：确认运行的是 `C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe`，并检查 `torch.cuda.is_available()`。
