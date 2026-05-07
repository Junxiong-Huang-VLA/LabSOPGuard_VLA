# 多源时间轴统一与对话回链

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 已支持视频局部时间、全局时间和带时间戳对话事件的统一组织 |
| 所属链路 | 多源预处理与时间对齐 |
| 一句话结论 | 本文定义 `local_timestamp_sec` 到 `timestamp_sec` 的转换方式，并说明 `ContextEvent` 如何按时间锚点回链到视频素材流。 |

`03` 是时间轴文档：它连接 `01/02` 的多视频输入和 `07` 的统一素材主线。

## 二、本文目标与边界

本文聚焦“局部时间到全局时间轴的统一”。视频帧先保留本机位局部时间，再通过 offset、anchor 和 drift 校准生成全局时间；对话或 transcript 通过 `ContextEvent.timestamp_sec` 接入同一时间轴。

| 覆盖内容 | 不覆盖内容 |
|---|---|
| `local_timestamp_sec` / `timestamp_sec` | ASR 模型本身 |
| `CameraSyncProfile.local_to_global()` | 相机 SDK 时间码读取实现 |
| `sync_anchors` / `clock_drift_ppm` | 声音/闪光检测的完整算法评测 |
| `ContextEvent` 回链素材项 | 复杂语义级文本对齐 |

## 三、核心输入

| 输入 | 说明 |
|---|---|
| `local_timestamp_sec` | 视频帧在单个视频源内部的局部时间 |
| `timestamp_sec` | 统一后的全局时间，是素材流主时间轴 |
| `start_offset_sec` | 简单平移同步参数 |
| `sync_anchors` | 多锚点同步输入，支持 offset 和 drift 拟合 |
| `clock_scale` / `clock_drift_ppm` | 长视频时钟漂移参数 |
| `ContextEvent.timestamp_sec` | 对话、transcript 或上下文事件的时间锚点 |
| `ContextEvent.content` | 对话或上下文文本内容 |
| ASR transcript | `/upload/asr` 转写生成的分段文本，写入 `context_inputs` |

## 四、处理流程

```text
对话 / transcript / context input
  -> ContextEvent(timestamp_sec)
  -> experiment.context_events

音频文件
  -> /upload/asr
  -> ASR API transcript segments
  -> context_inputs(kind=transcript, source_type=asr)
  -> ContextEvent(timestamp_sec)

视频帧
  -> local_timestamp_sec
  -> TimeSyncCalibrator
  -> timestamp_sec

素材流生成
  -> 查找时间接近的 ContextEvent
  -> 写入 linked_context_event_ids
  -> 写入 transcript_segment
```

### 对话回链机制

对话回链发生在素材流生成阶段。系统遍历带时间戳的 `ContextEvent`，如果事件时间与当前视频帧全局时间足够接近，则把该事件 ID 和文本片段写入当前素材项。

| 字段 | 写入位置 | 用途 |
|---|---|---|
| `linked_context_event_ids` | `material_stream.json` | 素材项关联到对话事件 |
| `transcript_segment` | `material_stream.json` | 保存命中的对话片段 |
| `linked_context_events` | `steps.json` | 步骤关联上下文事件 |

### ASR transcript 接入

ASR 入口接收音频或带音轨的视频文件，默认调用 Qwen/DashScope `qwen3-asr-flash`，并把识别文本转成 `context_inputs`。后续 `process` 阶段会按既有逻辑生成 `ContextEvent`。

```text
POST /api/v1/experiments/{experiment_id}/upload/asr
FormData:
  audio=<audio file>
  language=zh
  prompt=实验室操作对话
```

## 五、关键接口与数据结构

### 1. 相机局部时间转全局时间

```python
@dataclass
class CameraSyncProfile:
    camera_id: str
    offset_sec: float = 0.0
    clock_scale: float = 1.0
    drift_ppm: float = 0.0
    method: str = "sequential"

    def local_to_global(self, local_time_sec: float) -> float:
        return self.offset_sec + float(local_time_sec) * self.clock_scale
```

### 2. 多锚点拟合 offset 和 drift

```python
anchors = [
    SyncAnchor(camera_id="cam_b", local_time_sec=0.0, reference_time_sec=1.0, method="audio_flash"),
    SyncAnchor(camera_id="cam_b", local_time_sec=60.0, reference_time_sec=61.018, method="audio_flash"),
]

profile = TimeSyncCalibrator.fit_profile_from_anchors(
    camera_id="cam_b",
    anchors=anchors,
    reference_camera_id="cam_a",
)
```

### 3. 对话事件与视频素材近邻关联

```python
linked_context_ids = []
transcript_segment = None

for event in conversation_events:
    if event.timestamp_sec is None:
        continue
    if abs(event.timestamp_sec - frame.get("timestamp_sec", 0.0)) <= max(sample_interval * 1.5, 2.0):
        linked_context_ids.append(event.event_id)
        transcript_segment = event.content[:200]
        break
```

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 |
|---|---|
| `timeline.json.context_events` | 带时间戳的上下文/对话事件 |
| `material_stream.json[*].timestamp_sec` | 视频素材项全局时间 |
| `material_stream.json[*].local_timestamp_sec` | 视频素材项局部时间 |
| `material_stream.json[*].transcript_segment` | 对齐到视频帧附近的对话片段 |
| `material_stream.json[*].linked_context_event_ids` | 素材项关联的上下文事件 |
| `preprocessing.json.alignment_summary` | 多视频同步摘要 |
| `alignment_summary.max_sync_residual_error_sec` | 多锚点拟合残差摘要 |

样例输出：

```json
{
  "timestamp_sec": 12.5,
  "local_timestamp_sec": 11.5,
  "camera_id": "cam_front",
  "transcript_segment": "操作者开始吸取 200 微升样品",
  "linked_context_event_ids": ["ctx_001"]
}
```

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 对话能进入时间轴 | `timeline.json.context_events` 有 `timestamp_sec` |
| 视频帧有全局时间 | `material_stream.json[*].timestamp_sec` 存在 |
| 视频帧保留局部时间 | `local_timestamp_sec` 存在 |
| 对话能关联素材 | 素材项有 `linked_context_event_ids` 或 `transcript_segment` |
| 多机位同步有摘要 | `alignment_summary.anchor_strategy` 不是空值 |
| drift 残差可检查 | `max_sync_residual_error_sec` 可用于判断同步质量 |

### 验证命令

```powershell
$expId = "<experiment_id>"
$stream = Get-Content "outputs/experiments/$expId/material_stream.json" -Raw | ConvertFrom-Json
$stream | Where-Object { $_.transcript_segment } | Select-Object timestamp_sec,camera_id,transcript_segment,linked_context_event_ids

$prep = Get-Content "outputs/experiments/$expId/preprocessing.json" -Raw | ConvertFrom-Json
$prep.alignment_summary
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| ASR 依赖 Qwen/DashScope API 配置 | 未配置 `DASHSCOPE_API_KEY/ASR_MODEL` 时无法自动转写 |
| 对齐策略以时间邻近为主 | 未覆盖复杂语义对齐和说话人分离 |
| 声画/闪光自动检测仍基础 | 已输出残差摘要，但锚点质量仍依赖输入质量 |
| 硬件时间码字段已支持但未接 SDK | 无法直接从相机读取硬件时间 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 固化 transcript 输入格式 | 统一 `timestamp_sec/start_time_sec/end_time_sec` |
| P1 | 增加说话人分离 | 区分操作者、监督者和系统提示音 |
| P1 | 增强闪光/声音锚点自动检测 | 降低人工同步成本 |
| P1 | 输出同步残差报告 | 判断 anchor 质量和 drift 拟合质量 |
| P2 | 接入硬件时间码 adapter | 实现硬件级全局时间轴 |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `src/labsopguard/time_sync.py` | `CameraSyncProfile`、`TimeSyncCalibrator`，负责 offset 和 drift 校准 |
| `src/experiment/service.py` | `_resolve_sync_profile()`，从视频输入描述生成同步 profile |
| `src/experiment/service.py` | `_generate_material_stream()`，把对话片段回链到素材项 |
| `src/experiment/models.py` | `ContextEvent`，对话/上下文事件结构 |
| `src/labsopguard/input_layer.py` | `TimeAnchoredText`，带时间锚点文本输入结构 |

与其他文档的关系：`04` 使用全局时间生成关键帧和 clip，`07` 使用全局时间组织统一素材主线。
