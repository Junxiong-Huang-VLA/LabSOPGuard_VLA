# 事件级素材锚点与索引质量任务书

## 范围

继续限定在 `src/key_action_indexer` 主线：统一时间线、状态变化、素材资产、检索。目标是让事件、时间点和素材资产可以互相追踪，同时清理结构化索引字段，避免把长摘要误当成动作标签。

## 任务 1：素材路径归一化

- 输入：`unified_multimodal_timeline.jsonl`、`micro_segments.jsonl`、`material_asset_catalog.jsonl` 中的素材路径。
- 处理：支持已经带 session 目录前缀的相对路径，避免重复拼接 session 根目录。
- 输出：catalog 中正确的 `quality.resolved_path`、`exists`、`dry_run_placeholder`。
- 验收：dry-run 生成的 keyframe/clip 不再被错误标记为 missing。

## 任务 2：时间线链接素材类型纠正

- 输入：timeline links 的 `rel/path/modality/event_type`。
- 处理：优先按文件后缀和 role 判断素材类型；`.jpg/.png` 关键帧不能因为 event modality 是 video 而被归为 `video_clip`。
- 输出：正确的 `asset_type`，包括 `keyframe`、`image`、`video_clip`、`text_asset`。
- 验收：micro keyframe、YOLO interaction keyframe 能被检索为 `keyframe`。

## 任务 3：timeline 对象与动作继承

- 输入：timeline row、payload、nested payload、interaction、text_description。
- 处理：把 `primary_object/detected_objects/interaction_type/action_type` 继承到素材 catalog。
- 输出：asset 行的 `objects/actions` 可用于结构化检索。
- 验收：`search-assets --objects sample_bottle --actions hand_sample_bottle_contact` 能命中关键帧。

## 任务 4：动作字段去污染

- 输入：timeline `text` 和 payload 长文本。
- 处理：长摘要、多行 index_text 只进入 `search_text`，不进入 `actions`。
- 输出：`material_library_summary.action_counts` 不再出现长段文本。
- 验收：`action_counts` 中没有多行 action key。

## 任务 5：状态变化事件反链 asset_id

- 输入：`state_change_index.jsonl` 的 `asset_refs.path` 与 `material_asset_catalog.jsonl`。
- 处理：构建路径查找表，把匹配到的 `asset_id/source_type` 写回 `asset_refs`。
- 输出：状态变化事件可以直接定位素材资产。
- 验收：示例 dry-run 中 `state_changes=27`、`asset_refs=90`、`asset_id_refs=90`。

## 任务 6：pipeline/CLI 顺序调整

- 输入：pipeline 与 `assets` CLI。
- 处理：先生成素材 catalog，再生成状态变化索引，让状态索引天然具备 asset_id 反链。
- 输出：`pipeline_summary.json` 和 CLI JSON 中保留三个 artifact。
- 验收：dry-run pipeline、`pytest -q`、Python 编译、前端 build 不回退。
