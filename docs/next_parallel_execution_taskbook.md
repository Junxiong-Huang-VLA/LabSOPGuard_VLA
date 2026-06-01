# 下一轮并行执行任务书

## 总目标

把当前 `needs_review` 能力缺口收敛为可接入、可验证、可导出、可审计的生产后端能力。范围限定在 `src/key_action_indexer`，保持与 LabSOPGuard 应用独立；不扩展复杂前端、摄像头编排或云端基础设施；dry-run 必须无真实视频和 ffmpeg 也可运行。

## 并行执行结果

状态：已完成本轮并行执行。

执行方式：
- P-01、P-02、P-03、P-04/P-05 由 4 个并行 worker 同时推进。
- 主线程负责补齐跨模块集成、冲突修复、统一验收。

最终验收：
- `python -m compileall -q src tests` 通过。
- `python -m pytest -q` 通过，158 passed。
- `npm run build` 在 `frontend` 通过。
- `python -m key_action_indexer.cli run --manifest .runtime\timeline_demo_manifest.json --dry-run` 通过。
- strict artifact validation 通过：6/6 artifacts valid，128 records，0 errors。
- strict artifact export 通过：6/6 artifacts exported，128 records。
- demo QA 输出：overall_score=0.8433，overall_status=needs_review。

## P-01 标准模型输出接入

状态：已完成。

完成内容：
- 新增 `src/key_action_indexer/model_observations.py`。
- 支持读取：
  - `metadata/liquid_segmentation.jsonl`
  - `metadata/equipment_panel_states.jsonl`
  - `metadata/container_state_events.jsonl`
  - `metadata/object_tracks.jsonl`
- 统一输出：
  - `metadata/model_observation_events.jsonl`
  - `metadata/model_observation_events_summary.json`
- 接入 `advanced_vision_evidence`，把真实模型的 confirmed/measured 输出升级为 evidence。
- 接入 `video_understanding`，把模型观测转成结构化视频事件，并避免与 advanced evidence 重复。
- 接入 CLI：
  - `model-observations`
  - `model-observation-events`
- 接入 artifact schema/export，成为可校验、可导出的标准 artifact。
- 接入 pipeline summary。

验收：
- `tests/test_model_observations.py` 覆盖液体、设备面板、容器状态、物体轨迹 4 类输入。
- dry-run 无模型输入时输出空 `model_observation_events.jsonl`，不失败。
- 有模型输入时可绑定 `micro_segment_id`，并进入 advanced evidence 和 video understanding。

## P-02 标签与能力缺口审计

状态：已完成。

完成内容：
- 新增 `src/key_action_indexer/capability_gap_report.py`。
- 扫描 model inventory、dataset yaml、class schema、YOLO label 文件。
- 输出 `metadata/capability_gap_report.json`。
- 覆盖 liquid、stream、meniscus、button、knob、display、open、closed 等能力的标签基础、样本数、匹配类和推荐新增类。
- 接入 pipeline summary。
- 接入 CLI：
  - `capability-gap-report`
  - `capability-gap`

验收：
- `tests/test_capability_gap_report.py` 通过。
- demo QA 能读取该报告，并把推荐标签注入 `required_inputs`。

## P-03 QA 闭环任务映射

状态：已完成。

完成内容：
- `process_quality_report` 每个 check 增加：
  - `blocking_tasks`
  - `suggested_commands`
  - `required_inputs`
- 读取 `capability_gap_report.json` 和 `human_confirmation_queue.jsonl`。
- 新增 `next_round_scheduler`，可作为下一轮自动排程输入。
- `needs_review/fail` 可回指到 P-xx/T-xx 任务。

验收：
- `tests/test_quality_assurance.py` 通过。
- demo QA 当前生成 4 个 blocking tasks：
  - `P-01/T-STANDARD-STATE-EVIDENCE`
  - `P-02/T-CAPABILITY-GAP-AUDIT`
  - `P-04/T-HUMAN-CONFIRMATION-BATCH`
  - `P-05/T-MISSING-STEP-RECOVERY`

## P-04 人工确认批量决策

状态：已完成。

完成内容：
- `confirmation_loop.py` 支持 JSON/JSONL 批量确认决策。
- 输出 `metadata/human_confirmation_batch_result.json`。
- 写回 `experiment_process.json`。
- 保留 reviewer、note、before_state、after_state 和 audit trail。
- 接入 CLI：
  - `confirmation-batch`

验收：
- `tests/test_confirmation_loop.py` 通过。
- 当前实现为逐条处理：有效决策会先写入，后续无效行记录失败，不做事务回滚。

## P-05 未观察步骤补证计划

状态：已完成。

完成内容：
- 新增 `src/key_action_indexer/missing_step_recovery.py`。
- 针对 `not_observed`、低置信 `inferred_missing`、`skipped_or_unobserved` 步骤生成补证计划。
- 输出 `metadata/missing_step_recovery_plan.json`。
- 每个目标步骤包含：
  - recovery window
  - candidate video events
  - candidate transcript utterances
  - candidate material assets
  - search conditions
  - human confirmation suggestion
- 接入 CLI：
  - `missing-step-recovery`

验收：
- `tests/test_missing_step_recovery.py` 通过。
- demo 中 `step_004 / Record readout` 已生成补证窗口和候选证据。

## 当前仍需真实数据增强的限制

这些不是代码框架缺口，而是需要真实模型输出或人工决策输入继续推进：

- demo 未提供 `liquid_segmentation.jsonl`，所以液体流动/液位确认仍是候选级。
- demo 未提供 `equipment_panel_states.jsonl`，所以设备面板 OCR、按钮/旋钮状态仍不能强确认。
- demo 未提供 `container_state_events.jsonl`，所以容器开盖/关盖只能候选确认。
- demo 未提供 `object_tracks.jsonl`，所以真实物体轨迹目前没有外部 track 模型增强。
- demo 仍有 4 个 pending human confirmations，需要 `confirmation-batch` 输入人工决策文件。
- `step_004` 当前仍为 `not_observed`，补证计划已生成，但不会自动补抽真实视频或替人工确认。
