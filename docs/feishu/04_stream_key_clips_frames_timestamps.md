# 关键帧、关键时间戳与关键片段生成

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 已支持关键帧、关键时间戳、clip 记录和实体 MP4 物化 |
| 所属链路 | 多源预处理与时间对齐 |
| 一句话结论 | 本文说明连续视频流如何生成关键帧、关键时间戳和 clip，并解释 `recorded_file_path`、`source_path` 与实体 clip 文件之间的关系。 |

`04` 是素材 clip 生成节点，上游依赖 `03` 的全局时间轴，下游服务于 `05` 检索层和 `07` 统一素材主线。

## 二、本文目标与边界

本文聚焦关键素材生成机制：哪些帧被选为关键帧、关键时间戳如何沉淀、clip 如何从源视频或边录视频中切出。

| 覆盖内容 | 不覆盖内容 |
|---|---|
| 关键帧生成 | 视觉模型精度评估 |
| 关键时间戳记录 | 硬件时间码读取 |
| clip 窗口规则 | 长期存储清理策略 |
| `clips/*.mp4` 实体化 | 前端播放器实现 |

## 三、核心输入

| 输入 | 说明 |
|---|---|
| `video_index` | 从多路视频抽帧得到的素材项列表 |
| `is_key_frame` | 显式关键帧标记 |
| `change_score` | 无显式关键帧时的排序依据 |
| `timestamp_sec` | 关键帧全局时间 |
| `local_timestamp_sec` | 关键帧在源视频内部的局部时间 |
| `recorded_file_path` | 实时流边录文件路径，优先用于切 clip |
| `file_path` | 原始上传视频路径，作为 fallback clip 来源 |

## 四、处理流程

```text
连续视频流
  -> 按 sample_interval 抽帧
  -> frame_id + timestamp_sec
  -> 视觉分析
  -> is_key_frame 判定
  -> key_frames
  -> key_clips
  -> clip 文件 clips/*.mp4
```

### Clip 生成条件

| 条件 | 行为 |
|---|---|
| 有 `is_key_frame=true` 的素材项 | 直接作为关键帧 |
| 没有显式关键帧 | 按 `change_score` 选择 Top N |
| 有 `recorded_file_path` | 优先从边录文件切 clip |
| 无 `recorded_file_path` 但有 `file_path` | 从原视频切 clip |
| 源文件不存在或无法打开 | 保留 clip 记录，`file_exists=false` |

### `recorded_file_path` 与 `source_path`

`source_path` 是实际切片来源。实时流场景优先使用 `recorded_file_path`，文件上传场景通常使用原始 `file_path`。

```text
source_path = recorded_file_path or file_path
```

## 五、关键接口与数据结构

### 1. 连续流抽帧记录

```python
frames.append(
    {
        "frame_id": frame_index,
        "timestamp_sec": round(timestamp, 3),
        "width": width,
        "height": height,
        "path": str(frame_path),
        "source_type": source_type,
    }
)
```

### 2. 实时流边录文件

```python
recorded_path = Path(record_output_path) if record_output_path else None
if recorded_path is not None:
    recorded_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(recorded_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps or 10.0,
        (width, height),
    )
```

### 3. 关键 clip 物化为 MP4

```python
clip_record = {
    "clip_id": clip_id,
    "start_time_sec": start_time,
    "end_time_sec": end_time,
    "source_path": stream_info.get("recorded_file_path") or stream_info.get("file_path"),
    "file_path": None,
    "file_exists": False,
    "camera_id": item.get("camera_id"),
    "stream_id": item.get("stream_id"),
}

if clip_output_dir is not None:
    file_path = self._materialize_clip(...)
    if file_path:
        clip_record["file_path"] = str(file_path)
        clip_record["file_exists"] = True
```

### Clip 窗口规则

当前默认以关键帧时间为中心，窗口大小来自 `sample_interval_sec`：

```text
global_start = max(0, timestamp_sec - sample_interval_sec)
global_end   = timestamp_sec + sample_interval_sec
local_start  = max(0, local_timestamp_sec - sample_interval_sec)
local_end    = local_timestamp_sec + sample_interval_sec
```

如果采样间隔是 2 秒，默认 clip 是关键帧前后各 2 秒。

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 |
|---|---|
| `frame_bgr_path` | 抽取出的帧图片 |
| `key_frames` | 关键帧列表，包含时间戳、机位、物体、动作 |
| `key_clips` | 关键 clip 列表，包含起止时间、clip ID、来源 |
| `clips/*.mp4` | 真实视频 clip 文件 |
| `recordings/*.mp4` | 实时流边录文件 |
| `alignment_summary.key_timestamp_count` | 关键时间戳数量 |

样例 clip：

```json
{
  "clip_id": "stream_0:clip:3",
  "video_asset_id": "video_asset_0",
  "stream_id": "stream_0",
  "camera_id": "cam_front",
  "anchor_timestamp_sec": 42.0,
  "local_anchor_timestamp_sec": 41.75,
  "start_time_sec": 40.0,
  "end_time_sec": 44.0,
  "source_path": "outputs/experiments/exp/recordings/stream_00_cam_front.mp4",
  "file_path": "outputs/experiments/exp/clips/stream_0_clip_3.mp4",
  "file_exists": true,
  "render_status": "rendered"
}
```

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 有关键帧 | `preprocessing.json.key_frames` 非空 |
| 有关键时间戳 | `alignment_summary.key_timestamps_sec` 非空 |
| 有关键 clip 记录 | `preprocessing.json.key_clips` 非空 |
| clip 文件真实存在 | `key_clips[*].file_exists=true` 且文件可打开 |
| clip 可追溯来源 | `key_clips[*].source_path` 指向原视频或录制文件 |
| clip 保留机位信息 | `camera_id/stream_id` 存在 |

### 验证命令

```powershell
$expId = "<experiment_id>"
$prep = Get-Content "outputs/experiments/$expId/preprocessing.json" -Raw | ConvertFrom-Json
$prep.key_frames | Select-Object -First 5 frame_id,timestamp_sec,camera_id,stream_id
$prep.key_clips | Select-Object clip_id,start_time_sec,end_time_sec,file_path,file_exists,render_status
Test-Path "outputs/experiments/$expId/clips"
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| 关键 clip 选择偏规则化 | 需要更多实验语义参与 clip 选择 |
| 环形缓存未默认常驻 | 模块已具备，但默认处理链路不是长期采集服务 |
| 长视频高并发压测不足 | 多路长时间运行下 clip 生成成功率需验证 |
| `file_exists=false` 时只有记录没有实体文件 | 需要检查 `source_path` 是否存在并可读 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 用真实流验证 30 分钟 clip 回切 | 确认实体 clip 稳定生成 |
| P0 | 固定 clip 命名和保存规范 | 便于报告引用和人工复查 |
| P1 | 将 `RingSegmentRecorder` 挂入常驻采集服务 | 支持实时历史回切 |
| P1 | 增加事件前后窗口配置 | 统一关键 clip 上下文长度 |
| P2 | 用模型评分优化关键 clip 选择 | 减少无意义 clip |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `src/experiment/service.py` | `VideoFrameExtractor.extract_stream_frames()`，连续流抽帧和边录 |
| `src/experiment/service.py` | `_generate_material_stream()`，判定关键帧并生成 `clip_id` |
| `src/labsopguard/preprocessing.py` | `_build_key_frames()`，生成关键帧列表 |
| `src/labsopguard/preprocessing.py` | `_build_key_clips()`，生成关键 clip 记录 |
| `src/labsopguard/preprocessing.py` | `_materialize_clip()`，从原视频或录制文件切出 MP4 |
| `src/labsopguard/stream_buffer.py` | `RingSegmentRecorder.cut_clip()`，环形缓存历史片段回切 |

与其他文档的关系：`05` 依赖 `key_clips` 做 clip 检索，`07` 将关键帧和 clip 回链进统一素材主线。
