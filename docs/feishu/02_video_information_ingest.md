# 视频输入元数据规范

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 文件输入和流输入的元数据已进入实验记录 |
| 所属链路 | 多源预处理与时间对齐 |
| 一句话结论 | 本文定义 `video_inputs` 与 `video_metadata` 的职责边界，并说明文件输入、实时流输入在注册阶段和处理阶段分别能获得哪些字段。 |

`02` 只讨论视频输入元数据规范，不展开多机位接入策略；多机位接入见 `01_multi_camera_capture_access.md`。

## 二、本文目标与边界

本文聚焦“视频输入信息如何被规范化保存”。核心问题是区分两类数据：

| 数据结构 | 定位 |
|---|---|
| `video_inputs` | 后续处理链路直接读取的规范化输入描述 |
| `video_metadata` | 上传或注册阶段采集/记录的视频信息，偏审计和展示 |

文件输入和流输入的字段可得性不同：

| 输入类型 | 注册阶段可得 | 处理阶段确认 |
|---|---|---|
| 文件输入 | 文件路径、文件名、大小、fps、帧数、宽高、时长 | 抽帧结果、素材项、关键帧 |
| 流输入 | 来源地址、来源类型、机位、同步组、时间偏移、采集时长 | 实际帧率、宽高、录制文件、帧级时间戳 |

## 三、核心输入

| 数据类型 | 字段 |
|---|---|
| 文件信息 | `filename`、`size_bytes`、`video_path` |
| 视频属性 | `fps`、`frame_count`、`width`、`height`、`duration_sec` |
| 来源信息 | `source`、`source_type`、`ingest_mode` |
| 机位信息 | `camera_id`、`sync_group`、`video_index` |
| 同步信息 | `start_offset_sec`、`sync_method`、`sync_anchors` |
| 长视频信息 | `clock_scale`、`clock_drift_ppm` |
| 实时采集信息 | `capture_duration_sec`、`is_live_source` |

## 四、处理流程

```text
上传视频
  -> _probe_video_metadata()
  -> video_metadata
  -> video_inputs
  -> experiment.json

注册实时流
  -> UploadStreamRequest
  -> stream_descriptor
  -> video_inputs
  -> video_metadata
  -> experiment.json
```

`video_inputs` 是处理链路的输入契约；`video_metadata` 是对输入源的记录和补充说明。二者当前会共享部分字段，但用途不同。

## 五、关键接口与数据结构

### 1. 上传视频写入 `video_inputs`

```python
exp.setdefault("video_paths", [])
exp.setdefault("video_metadata", [])
exp.setdefault("video_inputs", [])

exp["video_paths"].append(str(file_path))
video_index = len(exp["video_paths"]) - 1
video_metadata = _probe_video_metadata(file_path, video_index)

exp["video_metadata"].append(video_metadata)
exp["video_inputs"].append(video_metadata)
```

### 2. 视频文件元数据探测

```python
metadata = {
    "video_index": video_index,
    "video_path": str(video_path),
    "filename": video_path.name,
    "size_bytes": video_path.stat().st_size if video_path.exists() else None,
    "source_type": "file",
    "ingest_mode": "file",
}

cap = cv2.VideoCapture(str(video_path))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
duration_sec = round(frame_count / fps, 3) if fps > 0 else 0.0
```

### 3. 实时流描述写入实验记录

```python
stream_descriptor = {
    "video_index": stream_index,
    "video_path": req.source,
    "source": req.source,
    "source_type": req.source_type,
    "ingest_mode": req.source_type,
    "camera_id": req.camera_id,
    "sync_group": req.sync_group,
    "start_offset_sec": req.start_offset_sec,
    "capture_duration_sec": req.capture_duration_sec,
    "is_live_source": req.source_type in {"rtsp", "usb"},
}

exp["video_inputs"].append(stream_descriptor)
exp["video_metadata"].append(dict(stream_descriptor))
```

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 |
|---|---|
| `experiment.json.video_paths` | 上传到本地的视频文件路径 |
| `experiment.json.video_metadata` | 视频文件或流的 metadata 记录 |
| `experiment.json.video_inputs` | 后续处理使用的规范化视频输入 |
| `schema_version=video_input.v1` | 视频输入 schema 版本 |
| `video_asset_id` | 实验视频资产 ID |
| `output_paths.source_video` | 处理任务中选用的主视频源 |

样例流输入输出：

```json
{
  "video_index": 0,
  "video_path": "rtsp://10.20.30.41/live/main",
  "source": "rtsp://10.20.30.41/live/main",
  "source_type": "rtsp",
  "ingest_mode": "rtsp",
  "camera_id": "cam_wireless_front",
  "sync_group": "bench_01",
  "start_offset_sec": 0.0,
  "capture_duration_sec": 300.0,
  "is_live_source": true
}
```

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 上传文件写入记录 | `video_paths` 有文件路径，`video_metadata` 有文件信息 |
| 流注册写入记录 | `video_inputs` 有 `source/source_type/camera_id` |
| 元数据可用于处理 | `process` 阶段能从 `video_inputs` 读取输入 |
| 同步字段保留 | `sync_anchors/clock_drift_ppm/sync_method` 不丢失 |
| 多路输入不覆盖 | 多次注册后 `video_inputs` 按顺序追加 |

### 验证命令

```powershell
$expId = "<experiment_id>"
$exp = Get-Content "outputs/experiments/$expId/experiment.json" -Raw | ConvertFrom-Json
$exp.video_metadata | ConvertTo-Json -Depth 8
$exp.video_inputs | ConvertTo-Json -Depth 8
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| RTSP 流注册时不能提前确认真实 fps/分辨率 | 只有处理阶段读取时才能确认 |
| 硬件 metadata 未直接读取 | 无法获得曝光、硬件帧号、硬件时间戳 |
| 设备身份与实验机位未单独建表 | 长期运行时设备管理能力不足 |
| `video_inputs` 和 `video_metadata` 字段仍有重叠 | 已有 `video_input.v1`，但还需要收敛展示字段 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 扩大视频输入 schema 校验覆盖面 | 防止历史入口绕过 `camera_id/source_type` |
| P0 | 增加流注册后的轻量 probe | 提前发现不可访问视频源 |
| P1 | 建立 camera registry | 固化设备、机位、分辨率、同步方式 |
| P1 | 接入硬件 metadata adapter | 读取硬件时间戳、帧号、曝光参数 |
| P2 | 增加输入配置导入导出 | 支持实验台一键恢复多机位配置 |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `backend/main.py` | `_probe_video_metadata()`，探测本地视频 fps、帧数、宽高、时长 |
| `backend/main.py` | `upload_video()`，上传视频并写入 `video_inputs/video_metadata` |
| `backend/main.py` | `upload_stream()`，注册实时流并写入 `video_inputs/video_metadata` |
| `src/experiment/service.py` | `set_video_inputs()`，进入主处理链路前规范化视频输入 |

与其他文档的关系：`01` 定义接入，`03` 使用同步字段，`07` 将视频输入转换为统一素材主线。
