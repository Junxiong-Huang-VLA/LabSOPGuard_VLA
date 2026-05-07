# 时间锚点多视频素材流主线

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 已作为单实验内统一素材主线输出 |
| 所属链路 | 多源预处理与时间对齐的骨干数据层 |
| 一句话结论 | `time_anchored_material_stream` 是连接多视频、对话、物理事件、关键片段和检索层的统一素材主线。 |

`07` 是整组文档的骨干。前面的 `01-06` 分别解决输入、元数据、时间、片段、索引和事件；最终都应收束到这条带时间锚点的多视频素材流。

## 二、本文目标与边界

本文聚焦“统一素材主线”的结构和作用。它不是单独的检测模块，也不是检索层，而是把多路视频、上下文文本、物理事件和关键片段组织为同一条可追溯的实验素材时间线。

| 连接对象 | 连接方式 |
|---|---|
| 多路视频 | `camera_id`、`stream_id`、`media_asset_id` |
| 时间同步 | `timestamp_sec`、`local_timestamp_sec` |
| 对话/上下文 | `transcript_segment`、`linked_context_event_ids` |
| 关键片段 | `clip_id`、`key_frame_reason` |
| 物理事件 | `material_item_id` 回链 |
| 检索层 | `material_index.sqlite` 以素材流为主数据 |

## 三、核心输入

| 输入 | 来源 | 用途 |
|---|---|---|
| 多路视频帧 | `01/02` 视频接入和元数据 | 生成素材项 |
| 全局时间戳 | `03` 时间同步 | 形成统一时间轴 |
| 对话事件 | `03` `ContextEvent` | 写入 transcript 和上下文回链 |
| ASR transcript | `03` `/upload/asr` | 自动生成带时间锚点的对话事件 |
| 关键帧/clip | `04` 关键片段生成 | 写入 `clip_id` 和关键帧字段 |
| 物理事件 | `06` 事件语义层 | 通过 `material_item_id` 回链 |
| 检索索引 | `05` 检索层 | 以素材流为主数据构建索引 |

## 四、处理流程

```text
多路视频帧
  -> 时间同步
  -> 按 timestamp_sec 排序
  -> 每帧生成 MultimodalMaterialStreamItem
  -> 关键帧/clip/事件/对话回链
  -> material_stream.json
  -> preprocessing.time_anchored_material_stream
  -> material_index.sqlite
```

统一主线的核心规则：

| 规则 | 说明 |
|---|---|
| 以 `timestamp_sec` 为主时间轴 | 所有素材项按全局时间复盘 |
| 保留 `local_timestamp_sec` | 支持回查原视频局部时间 |
| 每个 item 可追溯机位 | 通过 `camera_id/stream_id/media_asset_id` |
| 关键素材带 `clip_id` | 下游可回放实体片段 |
| 上下文通过 ID 回链 | 避免只保存不可追溯文本 |

## 五、关键接口与数据结构

### 1. 素材流核心数据结构

```python
@dataclass
class MultimodalMaterialStreamItem:
    item_id: str = field(default_factory=_uuid)
    experiment_id: str = ""
    timestamp_sec: float = 0.0
    local_timestamp_sec: Optional[float] = None
    media_asset_id: Optional[str] = None
    stream_id: Optional[str] = None
    frame_id: Optional[int] = None
    clip_id: Optional[str] = None
    object_labels: List[str] = field(default_factory=list)
    detected_activities: List[str] = field(default_factory=list)
    transcript_segment: Optional[str] = None
    conversation_context: Optional[str] = None
    linked_context_event_ids: List[str] = field(default_factory=list)
```

### 2. 每帧生成一个时间锚点素材项

```python
item = MultimodalMaterialStreamItem(
    experiment_id=experiment.experiment_id,
    timestamp_sec=frame.get("timestamp_sec", 0.0),
    local_timestamp_sec=frame.get("local_timestamp_sec"),
    media_asset_id=frame.get("media_asset_id"),
    stream_id=stream_id,
    frame_id=frame.get("frame_id"),
    clip_id=clip_id,
    object_labels=object_labels,
    detected_activities=detected_activities,
    transcript_segment=transcript_segment,
    conversation_context=self._context_text[:200] if self._context_text else None,
    linked_context_event_ids=linked_context_ids,
)
```

### 3. 输出为正式预处理产物

```python
preprocessing_payload = {
    "video_streams": artifact.video_streams,
    "key_frames": artifact.key_frames,
    "key_clips": artifact.key_clips,
    "detected_changes": artifact.detected_changes,
    "alignment_summary": artifact.alignment_summary,
    "time_anchored_material_stream": artifact.time_anchored_material_stream,
}
```

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 |
|---|---|
| `material_stream.json` | 主素材流文件 |
| `schema_version=material_stream.v1` | 素材流 item 版本 |
| `preprocessing.json.time_anchored_material_stream` | 预处理输出中的素材流副本 |
| `timeline.metadata.material_stream_ids` | 时间线关联的素材项 ID |
| `timeline.metadata.material_stream_count` | 素材流数量 |
| `material_index.sqlite` | 可检索素材索引 |
| `steps.json[*].linked_media_assets` | 步骤关联的素材 item |

完整 item 示例：

```json
{
  "item_id": "item_420",
  "experiment_id": "exp_001",
  "timestamp_sec": 42.0,
  "local_timestamp_sec": 41.75,
  "media_asset_id": "video_asset_0",
  "stream_id": "stream_0",
  "camera_id": "cam_front",
  "frame_id": 420,
  "local_frame_id": 418,
  "frame_bgr_path": "outputs/experiments/exp_001/frames/frame_0420.jpg",
  "clip_id": "stream_0:clip:3",
  "object_labels": ["pipette", "tube"],
  "detected_activities": ["transfer"],
  "scene_description": "operator transfers liquid into tube",
  "transcript_segment": "操作者开始吸取 200 微升样品",
  "linked_context_event_ids": ["ctx_001"],
  "is_key_frame": true,
  "key_frame_reason": "activity_change",
  "change_score": 0.82
}
```

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 主素材流存在 | `material_stream.json` 非空 |
| 素材流版本存在 | 每个 item 带 `schema_version=material_stream.v1` |
| 预处理结果包含素材流 | `preprocessing.json.time_anchored_material_stream` 非空 |
| 每个 item 有时间锚点 | `timestamp_sec` 存在 |
| 多机位信息保留 | `camera_id/stream_id/local_timestamp_sec` 存在 |
| 素材可回链视频 | `frame_id` 或 `media_asset_id` 存在 |
| 素材可回链 clip | 关键素材包含 `clip_id` |
| 素材可索引 | `material_index.sqlite` 构建成功 |

### 验证命令

```powershell
$expId = "<experiment_id>"
$stream = Get-Content "outputs/experiments/$expId/material_stream.json" -Raw | ConvertFrom-Json
$stream.Count
$stream | Select-Object -First 5 item_id,timestamp_sec,local_timestamp_sec,camera_id,stream_id,clip_id,is_key_frame

$timeline = Get-Content "outputs/experiments/$expId/timeline.json" -Raw | ConvertFrom-Json
$timeline.metadata.material_stream_count
$timeline.metadata.material_stream_stream_ids
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| 默认报告展示还没有完全围绕素材流组织 | 用户侧感知仍偏步骤结果 |
| 素材流 schema 已有 v1 标记但未冻结兼容策略 | 前端、检索、报告需要稳定字段契约 |
| 实时增量素材流尚未产品化 | 当前主要在实验处理后输出 |
| 跨实验素材库尚未统一 | 当前索引以单实验为单位 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 冻结素材流 schema v1 兼容策略 | 前后端和报告统一字段 |
| P0 | 在报告中优先引用素材流 item | 让证据链更清晰 |
| P1 | 增加实时增量 material stream writer | 支持边采集边检索 |
| P1 | 增加跨实验素材库索引 | 支持历史实验复用和查询 |
| P2 | 增加素材流可视化时间轴 | 支持按机位、动作、物体复盘 |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `src/experiment/models.py` | `MultimodalMaterialStreamItem`，素材流核心数据结构 |
| `src/experiment/service.py` | `_generate_material_stream()`，生成时间锚点素材项 |
| `src/experiment/service.py` | `_link_material_stream()`，把素材项回链到 timeline 和 step |
| `src/experiment/service.py` | `save_outputs()`，输出 `material_stream.json` 和 `preprocessing.json` |
| `src/labsopguard/output_layer.py` | 正式输出层暴露 `time_anchored_material_stream` |

与其他文档的关系：`07` 是整组文档的统一素材主线；`01-04` 产生输入和片段，`06` 产生事件语义，`05` 依赖本主线构建检索层。
