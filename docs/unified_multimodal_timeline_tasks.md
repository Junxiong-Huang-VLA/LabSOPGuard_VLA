# 统一多模态实验时间线任务书

## 范围

本任务只在 `src/key_action_indexer` 主线内实现统一实验时间轴能力，保持它独立于 LabSOPGuard。目标是把视频片段、ASR 转写、用户文本、AI 回复、上传图文、YOLO hand-object interaction、segment 和 micro-segment 统一映射到同一 `global_time`，并输出可检索、可审计的时间线 JSONL。

## 任务 1：用户文本时间对齐

- 输入：JSONL 用户文本事件，兼容 `event_id/id`、`timestamp/global_time/time`、`timestamp_ms`、`session_sec/time_sec/local_time_sec/start_sec`、`text/content/message`。
- 处理：绝对时间优先；否则使用 `session_start_time + session_sec`；支持固定 `offset_sec/latency_sec` 修正。
- 输出：`event_type=user_text`、`modality=text` 的 timeline event。
- 验收：无真实视频、无 ffmpeg 时也能从 JSONL 生成按 `global_time` 排序的用户文本事件。

## 任务 2：AI 回复时间对齐

- 输入：JSONL AI 回复事件，兼容用户文本同类时间字段和 `response/reply/content/message`。
- 处理：统一转换到 `global_time`；保留 prompt/response 等原始 payload。
- 输出：`event_type=ai_reply`、`modality=text` 的 timeline event。
- 验收：AI 回复可与同一时间窗口的视频 segment、ASR 和用户文本并列排序。

## 任务 3：上传图文时间对齐

- 输入：JSONL 上传事件，兼容 `upload_type`、`path/media_path/image_path/file_path`、`text/caption/message`、常见时间字段。
- 处理：图片、文本、混合上传都映射到 `global_time`；保留文件路径和说明文本。
- 输出：`event_type=upload`、`modality=image/text/multimodal` 的 timeline event。
- 验收：上传图片/文本事件能出现在统一时间线，且 payload 中可追溯原文件路径。

## 任务 4：时间漂移和延迟校准

- 输入：固定 `offset_sec/latency_sec`，以及可选校准锚点 `{source_time_sec, global_time|global_sec}`。
- 处理：锚点不少于 2 个时拟合线性 `global_sec = slope * source_sec + intercept`；锚点不足时退化为固定 offset；输出校准摘要。
- 输出：`time_calibration_report.json`。
- 验收：测试覆盖固定延迟、两锚点漂移、锚点不足退化。

## 任务 5：事件级时间锚点

- 输入：视频 segment、micro-segment、ASR utterance、用户文本、AI 回复、上传图文。
- 处理：每条事件生成统一锚点字段：`timeline_event_id`、`session_id`、`event_type`、`modality`、`source`、`global_time`、`session_time_sec`、`duration_sec`、`anchor_confidence`、`anchor_strategy`、`payload`、`links`、`text`。
- 输出：事件级 timeline rows。
- 验收：segment/micro/ASR/用户文本/AI/上传事件均可用同一字段集合消费。

## 任务 6：统一多模态时间线输出

- 输入：pipeline 既有 artifacts 以及新增事件源 JSONL。
- 处理：归一化所有事件，按 `global_time`、`event_type`、`timeline_event_id` 稳定排序。
- 输出：`metadata/unified_multimodal_timeline.jsonl` 和 `metadata/time_calibration_report.json`。
- 验收：CLI 可单独生成统一时间线；pipeline dry-run 可输出统一时间线 artifact；现有 `pytest -q`、Python 编译、segment-level retrieval 不回退。
