# LabSOPGuard 多源预处理与时间对齐验收文档

**日期：** 2026-04-19  
**范围：** 视频信息接入、对话/视频时间对齐、关键帧/关键片段提取、素材索引、物理变化检测、时间锚点素材流组织  
**结论：** 当前主链路已经从“单视频分析 demo”升级为“多源输入 -> 时间对齐 -> 素材资产化 -> 可检索输出”的工程链路。剩余主要是硬件级同步、真实长时流稳定性和更高精度视觉语义模型。

---

## 一、清单总览

| 项目 | 当前判断 | 主要产物 | 状态 |
|---|---|---|---|
| 多路视频采集与接入方案梳理 | 已有正式上传/流注册接口，多路输入模型和同步字段已进入后端 | `video_inputs`、`video_metadata`、流注册 API | 已完成 |
| 接入视频信息数据 | 本地视频上传和流注册都能写入实验记录，包含机位、来源、offset、同步字段 | `experiment.json`、`video_inputs`、`video_metadata` | 可验收 |
| 对话时间戳和视频时间戳对齐 | 支持 `ContextEvent.timestamp_sec` 与视频 global timestamp 近邻关联；支持多机位 offset/drift 校准 | `linked_context_event_ids`、`transcript_segment` | 可验收，硬件同步待增强 |
| 连续视频流关键片段/关键帧/关键时间戳提取 | 支持抽帧、关键帧、关键 clip 记录和实际 MP4 materialization | `key_frames`、`key_clips`、`clips/*.mp4` | 可验收 |
| 视频素材可索引化整理与关键素材引用保存 | 已有 SQLite + FTS5 + 可选 embedding 检索，支持联合过滤 | `material_index.sqlite`、`materials/search` | 可验收 |
| 显著物理世界变化检测和记录 | 已有 scene/object/hand/liquid 基线，并新增稳定 ID、容器开合、标签状态等语义事件 | `physical_events.json`、`detected_changes` | 可验收，模型精度待数据集增强 |
| 带时间锚点的多视频素材流 | 主输出已组织为 `time_anchored_material_stream`，并在正式输出层暴露 | `material_stream.json`、`preprocessing.json` | 可验收 |

---

## 二、接入视频信息数据

### 2.1 当前数据入口

| 入口 | 路径 | 写入内容 |
|---|---|---|
| 本地视频上传 | `POST /api/v1/experiments/{id}/upload/video` | 文件路径、文件大小、fps、帧数、宽高、时长、`source_type=file` |
| 实时流注册 | `POST /api/v1/experiments/{id}/upload/stream` | `source`、`source_type`、`camera_id`、`sync_group`、`start_offset_sec`、`capture_duration_sec` |
| 同步增强字段 | 同一流注册接口 | `sync_method`、`sync_anchors`、`hardware_timecode_start_sec`、`sync_board_offset_sec`、`clock_scale`、`clock_drift_ppm` |
| 后端持久化 | `outputs/experiments/{id}/experiment.json` | `video_inputs`、`video_metadata`、`video_paths`、`video_asset_id` |

### 2.2 数据结构

```json
{
  "video_index": 0,
  "video_path": "rtsp://192.168.1.41/stream1",
  "source": "rtsp://192.168.1.41/stream1",
  "source_type": "rtsp",
  "ingest_mode": "rtsp",
  "camera_id": "cam_front",
  "sync_group": "bench_01",
  "start_offset_sec": 0.0,
  "capture_duration_sec": 120.0,
  "sync_method": "audio_flash",
  "sync_anchors": [
    {"local_time_sec": 0.0, "reference_time_sec": 1.0, "method": "audio_flash"}
  ],
  "clock_drift_ppm": 80.0,
  "is_live_source": true
}
```

### 2.3 验收标准

| 验收点 | 判定方式 |
|---|---|
| 上传视频能被记录 | `experiment.json.video_paths` 和 `video_metadata` 出现文件信息 |
| 注册实时流能被记录 | `experiment.json.video_inputs` 出现 `source/camera_id/source_type` |
| 同步字段不丢失 | `sync_anchors/clock_drift_ppm/sync_method` 原样进入 `video_inputs` |
| 后续处理可读取 | `process` 阶段调用 `ExperimentService.set_video_inputs()` |

---

## 三、对话时间戳和视频时间戳对齐

### 3.1 当前时间轴模型

```text
视频帧 local_timestamp_sec
  -> CameraSyncProfile.local_to_global()
  -> frame.timestamp_sec
  -> MultimodalMaterialStreamItem.timestamp_sec

对话/文本 ContextEvent.timestamp_sec
  -> 与 frame.timestamp_sec 近邻匹配
  -> 写入 linked_context_event_ids / transcript_segment
```

### 3.2 支持的对齐输入

| 输入字段 | 作用 | 适用场景 |
|---|---|---|
| `timestamp_sec` | 对话或上下文事件的全局时间 | 已有统一时间轴 |
| `start_time_sec` | 上下文事件开始时间 | 片段型文本/记录 |
| `local_timestamp_sec` | 视频局部时间 | 多机位同步前的原始时间 |
| `start_offset_sec` | 视频局部时间到全局时间的平移 | 简单多机位 |
| `sync_anchors` | 多个锚点拟合 offset + drift | 高精度多机位 |
| `clock_drift_ppm` | 长视频时钟漂移修正 | 长时采集 |

### 3.3 当前关联策略

| 步骤 | 说明 |
|---|---|
| 1 | 视频帧先经过多机位同步，生成全局 `timestamp_sec` |
| 2 | 对话输入转为 `ContextEvent`，保留 `timestamp_sec/start_time_sec/end_time_sec` |
| 3 | 生成素材流时查找与视频帧时间接近的对话事件 |
| 4 | 命中的对话写入 `transcript_segment` 和 `linked_context_event_ids` |
| 5 | 推理步骤时，步骤时间段内的 `ContextEvent` 会参与命名和证据关联 |

### 3.4 现存差距

| 问题 | 当前影响 | 后续增强 |
|---|---|---|
| 对话转写本身未内置 ASR | 需要外部输入 transcript 或 context | 接入 Whisper/本地 ASR |
| 声画自动锚点仍是基础实现 | 闪光/声音锚点需要更稳健检测 | 增加音频峰值和画面亮度联合检测 |
| 硬件时间码未接设备 SDK | 字段和校准模型已准备，但不能直接读取硬件 | 按相机/采集卡补 adapter |
| 长视频 drift 依赖锚点质量 | 锚点少时只能估 offset | 固定周期触发锚点并输出残差报告 |

---

## 四、连续视频流关键片段、关键帧和关键时间戳提取

### 4.1 当前处理链路

```text
连续视频流
  -> 抽帧 sample_interval
  -> 每帧生成 frame_id / timestamp_sec / local_timestamp_sec
  -> 视觉分析生成 object_labels / detected_activities / scene_description
  -> is_key_frame 判定
  -> LabPreprocessor._build_key_frames()
  -> LabPreprocessor._build_key_clips()
  -> clips/*.mp4 实体文件落盘
```

### 4.2 关键帧输出

| 字段 | 说明 |
|---|---|
| `frame_id` | 全局帧 ID |
| `timestamp_sec` | 同步后的全局时间 |
| `local_timestamp_sec` | 当前机位局部时间 |
| `camera_id` | 机位 |
| `stream_id` | 流 ID |
| `frame_bgr_path` | 帧图片路径 |
| `object_labels` | 物体标签 |
| `detected_activities` | 动作/活动标签 |

### 4.3 关键片段输出

| 字段 | 说明 |
|---|---|
| `clip_id` | 片段唯一 ID |
| `start_time_sec` / `end_time_sec` | 全局时间窗口 |
| `source_path` | 原视频或录制文件 |
| `file_path` | 生成后的 clip 文件 |
| `file_exists` | clip 实体是否已落盘 |
| `camera_id` / `stream_id` | 来源机位和流 |
| `reason` | 生成原因，如 key frame、change event |

### 4.4 验收标准

| 验收点 | 判定方式 |
|---|---|
| 关键时间戳存在 | `preprocessing.json.key_frames[*].timestamp_sec` 不为空 |
| 关键片段记录存在 | `preprocessing.json.key_clips` 有记录 |
| 实体 clip 存在 | `file_exists=true` 且 `clips/*.mp4` 可打开 |
| 连续流可回切 | 实时流处理后 `recorded_file_path` 存在，clip 从录制文件生成 |

---

## 五、视频素材可索引化整理与关键素材引用保存

### 5.1 索引设计

当前索引层使用 SQLite 主表 + FTS5 全文索引 + 可选 hash embedding 分数。

```text
material_stream.json
  + preprocessing.json.key_clips
  + preprocessing.json.detected_changes
  -> MaterialRetrievalIndex.index_payloads()
  -> material_index.sqlite
  -> /materials/search
```

### 5.2 支持的联合查询条件

| 查询维度 | 参数 | 说明 |
|---|---|---|
| 物体 | `objects=pipette,tube` | 匹配 `object_labels` 和文本 blob |
| 动作 | `actions=transfer` | 匹配 `detected_activities` 和文本 blob |
| 时间段 | `start_time_sec/end_time_sec` | 按全局时间过滤 |
| 机位 | `camera_id=cam_front` | 只查某个摄像头 |
| 流 | `stream_id=stream_0` | 只查某一路流 |
| clip | `has_clip=true`、`clip_exists=true` | 只返回有片段或实体片段 |
| 全文 | `text=liquid` | FTS5 或 LIKE fallback |
| embedding | `embedding_text=liquid transfer` | 返回 `embedding_score` |

### 5.3 查询示例

```text
GET /api/v1/experiments/{id}/materials/search
  ?objects=pipette,tube
  &actions=transfer
  &start_time_sec=30
  &end_time_sec=90
  &camera_id=cam_front
  &clip_exists=true
  &text=liquid
```

### 5.4 验收标准

| 验收点 | 判定方式 |
|---|---|
| 索引文件存在 | `outputs/experiments/{id}/material_index.sqlite` |
| 索引缺失可重建 | 查询 API 在 `material_stream.json` 存在时自动重建 |
| 联合过滤有效 | 物体/动作/时间/机位/clip 同时过滤仍返回正确结果 |
| 关键素材可追溯 | 查询结果包含 `frame_path`、`clip_file_path`、`payload` |

---

## 六、显著物理世界变化检测和记录

### 6.1 当前检测类型

| 事件类型 | 当前来源 | 说明 |
|---|---|---|
| `scene_change` | 帧描述变化 | 场景或操作变化 |
| `object_move` | 物体集合/位置变化 | 物体移动或出现/消失 |
| `hand_contact` | hand + object 标签共现 | 基础接触判断 |
| `hand_contact_geometry` | bbox 几何关系 | 更精确的手-物接触 |
| `container_opened` / `container_closed` | 容器和盖子几何状态 | 容器开合 |
| `liquid_level_change` | 液体区域/液位变化 | 液位升降 |
| `reagent_label_state` | 标签/ocr 状态 | 试剂标签可见或文本状态 |
| `liquid_transfer` | transfer/pour/pipette 活动 | 液体转移动作 |

### 6.2 语义增强链路

```text
frame analysis detected_objects
  -> StableObjectTracker
      -> object_id 稳定跟踪
  -> SemanticEventDetector
      -> hand-object overlap / distance
      -> cap-container relation
      -> liquid level delta
      -> reagent label text/status
  -> PhysicalEvent
  -> preprocessing.detected_changes
  -> material_index.event_types
```

### 6.3 验收标准

| 验收点 | 判定方式 |
|---|---|
| 事件被记录 | `physical_events.json` 不为空 |
| 事件进入预处理 | `preprocessing.json.detected_changes` 包含事件 |
| 事件可追溯到素材 | 事件 metadata 包含 `material_item_id/frame_id/camera_id` |
| 事件可检索 | `material_index.sqlite` 中 `event_types` 可随素材返回 |

### 6.4 现存差距

| 问题 | 当前影响 | 后续增强 |
|---|---|---|
| 语义检测仍以规则和模型输出为基础 | 对遮挡、反光、微小液位变化不够稳 | 训练实验室专用 detector/segmenter |
| OCR 标签状态依赖上游检测 | 标签文字不清时会漏 | 接入专门 OCR pipeline |
| 物体 ID 是短时稳定 | 长时遮挡后可能换 ID | 引入 ReID 特征和轨迹记忆 |
| 接触判断是 bbox 几何 | 无深度时容易误判 | 融合深度/手关键点 |

---

## 七、带时间锚点的多视频素材流

### 7.1 输出结构

`MultimodalMaterialStreamItem` 是当前主素材单元，核心字段包括：

| 字段 | 说明 |
|---|---|
| `item_id` | 素材项 ID |
| `experiment_id` | 实验 ID |
| `timestamp_sec` | 全局时间锚点 |
| `local_timestamp_sec` | 机位局部时间 |
| `media_asset_id` | 对应媒体资产 |
| `stream_id` / `camera_id` | 来源流和机位 |
| `frame_id` | 视频帧 |
| `clip_id` | 关联片段 |
| `object_labels` | 物体 |
| `detected_activities` | 动作 |
| `scene_description` | 场景描述 |
| `transcript_segment` | 对齐到的对话片段 |
| `linked_context_event_ids` | 关联上下文事件 |

### 7.2 组织方式

```text
多路视频帧
  -> 统一 global timestamp 排序
  -> 每帧生成 MaterialStreamItem
  -> 关键帧写 media asset
  -> clip_id 关联关键片段
  -> context/physical events 按时间范围回链
  -> 输出 material_stream.json
  -> preprocessing.time_anchored_material_stream
```

### 7.3 与实验步骤的关系

| 关系 | 当前实现 |
|---|---|
| 素材流到步骤 | 步骤时间段内的素材 item 写入 `linked_media_assets` |
| 物理事件到步骤 | 步骤时间段内的 `PhysicalEvent` 写入 `linked_physical_events` |
| 上下文到步骤 | 时间段内 `ContextEvent` 写入 `linked_context_events` |
| 证据引用到素材 | 根据 `frame_id` 或最近时间戳补 `media_asset_id` |

### 7.4 验收标准

| 验收点 | 判定方式 |
|---|---|
| 主素材流存在 | `material_stream.json` 非空 |
| 预处理输出包含素材流 | `preprocessing.json.time_anchored_material_stream` 非空 |
| 多机位字段存在 | item 中有 `camera_id/stream_id/local_timestamp_sec` |
| 时间顺序正确 | item 按 `timestamp_sec` 可排序复盘 |
| 输出层暴露 | 正式输出包含 `time_anchored_material_stream` |

---

## 八、当前可交付物清单

| 文件 | 内容 |
|---|---|
| `experiment.json` | 实验基础信息、视频输入、处理状态 |
| `timeline.json` | 步骤、上下文、物理事件、媒体资产 |
| `steps.json` | 推理出的实验步骤 |
| `physical_events.json` | 显著物理变化事件 |
| `material_stream.json` | 带时间锚点的多视频素材流 |
| `preprocessing.json` | 视频流、关键帧、关键片段、检测变化、同步摘要 |
| `material_index.sqlite` | 可检索素材索引 |
| `clips/*.mp4` | 实体关键片段文件 |
| `recordings/*.mp4` | 实时流录制文件 |

---

## 九、代码位置

| 模块 | 作用 |
|---|---|
| `backend/main.py` | 视频上传、流注册、素材检索 API |
| `src/experiment/service.py` | 主实验处理链路，多源输入、时间同步、物理事件、素材流输出 |
| `src/experiment/models.py` | `ContextEvent`、`PhysicalEvent`、`MultimodalMaterialStreamItem` 等数据结构 |
| `src/labsopguard/input_layer.py` | 时间锚点文本和实验输入 bundle |
| `src/labsopguard/time_sync.py` | offset、drift、anchor 校准 |
| `src/labsopguard/preprocessing.py` | 关键帧、关键片段、同步摘要、预处理产物 |
| `src/labsopguard/retrieval.py` | SQLite/FTS/embedding 检索 |
| `src/labsopguard/semantic_events.py` | 稳定物体 ID 和语义事件检测 |
| `src/labsopguard/stream_buffer.py` | 环形缓存和历史片段回切 |

---

## 十、建议下一步

| 优先级 | 任务 | 目标 |
|---|---|---|
| P0 | 用真实 2-4 路摄像头跑 30 分钟连续采集 | 验证流稳定性和 clip 回切 |
| P0 | 固化每路 `camera_id/sync_group/source_type` 命名规范 | 避免后续素材混乱 |
| P1 | 接入 ASR 或标准 transcript 输入格式 | 提升对话-视频对齐质量 |
| P1 | 接入相机 SDK/同步板 adapter | 从字段级同步升级到硬件级同步 |
| P1 | 增加 stream health 指标 | FPS、丢帧、重连、延迟可观测 |
| P2 | 基于实验室数据训练专用 detector/segmenter/OCR | 提升物体、液位、标签、接触语义精度 |

