# 关键素材库与检索接口任务书

## 范围

本任务只补齐 `src/key_action_indexer` 主线能力，不扩展 LabSOPGuard 前端、云端服务、五路相机编排或复杂基础设施。所有能力必须保留 dry-run 可运行，不依赖真实视频或 ffmpeg。目标是在现有 segment/micro-segment、YOLO hand-object interaction、统一时间线和向量索引基础上，补齐关键素材资产化、状态变化索引和素材检索接口。

## 任务 1：真实切片与关键帧质量记录

- 输入：pipeline 生成的 clip、keyframe、metadata。
- 处理：为每个素材记录文件存在性、大小、dry-run 标记、素材类型、时间范围、来源 segment/micro/event。
- 输出：素材资产 catalog 中的 `quality` 和 `exists/size_bytes` 字段。
- 验收：dry-run 产物也能生成完整质量记录；真实文件缺失时不会中断 pipeline，而是在 catalog 中标记。

## 任务 2：关键素材统一资产 ID 与引用表

- 输入：`key_action_segments.jsonl`、`micro_segments.jsonl`、`vector_metadata.jsonl`、`unified_multimodal_timeline.jsonl`。
- 处理：把 segment clip、micro clip、keyframe、interaction keyframe、上传图文链接统一转为资产行；生成稳定 `asset_id`。
- 输出：`metadata/material_asset_catalog.jsonl`。
- 验收：每个 asset 行至少包含 `asset_id`、`asset_type`、`path`、`source_type`、`source_id`、`global_start_time/global_end_time`、`objects/actions/state_tags/search_text/quality`。

## 任务 3：状态变化索引

- 输入：micro-segment、YOLO interaction、统一时间线事件。
- 处理：生成实验状态变化事件，例如 `contact_started`、`peak_interaction`、`contact_released`、`object_contact`、`dialogue_context_available`、`evidence_level_changed`。
- 输出：`metadata/state_change_index.jsonl`。
- 验收：每个 micro 至少能产生 contact/peak/release 状态事件；事件含 `state_change_id`、`state_type`、`global_time`、`session_time_sec`、`primary_object`、`interaction_type`、`evidence_level`、`asset_refs`。

## 任务 4：图像和视频素材库 MVP

- 输入：资产 catalog、状态变化索引、统一时间线。
- 处理：形成可被检索和审计的离线素材库，不引入数据库服务。
- 输出：`metadata/material_asset_catalog.jsonl`、`metadata/state_change_index.jsonl`、`metadata/material_library_summary.json`。
- 验收：summary 统计素材类型、对象、动作、状态事件、缺失文件数量、dry-run 文件数量。

## 任务 5：素材检索接口

- 输入：资产 catalog JSONL。
- 处理：提供 Python API 和 CLI 命令，支持关键词、资产类型、对象、动作、状态标签、时间范围、limit 过滤。
- 输出：CLI JSON 结果。
- 验收：`python -m key_action_indexer.cli assets --session-dir <dir>` 可生成素材库；`python -m key_action_indexer.cli search-assets --session-dir <dir> --query <text>` 可返回素材路径和引用上下文。

## 任务 6：pipeline 自动集成

- 输入：完整 pipeline 产物。
- 处理：在 run pipeline 末尾自动生成素材库和状态变化索引。
- 输出：pipeline summary artifacts 增加 `material_asset_catalog`、`state_change_index`、`material_library_summary`。
- 验收：`pytest -q`、Python 编译、dry-run pipeline、segment-level retrieval、前端 build 均不回退。
