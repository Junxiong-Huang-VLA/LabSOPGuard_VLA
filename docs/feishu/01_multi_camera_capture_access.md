# 多机位视频源接入架构

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 主链路已支持多机位视频源接入 |
| 所属链路 | 多源预处理与时间对齐 |
| 一句话结论 | 本文定义 LabSOPGuard 如何把本地视频、RTSP、USB、HTTP、RTMP、UDP 等视频源接入统一实验处理链路，并保留 `camera_id`、`sync_group`、`video_index` 等多机位标识。 |

`01` 只讨论视频源如何接入主链路，不展开视频元数据 schema，也不覆盖 24 小时常驻采集 daemon 或相机 SDK 驱动层。

## 二、本文目标与边界

本文聚焦“多机位视频源接入能力”。核心目标是让多个视频源在同一个实验下被注册、编号、携带机位信息，并交给后续时间同步、素材流生成和检索层处理。

| 范围 | 是否覆盖 |
|---|---|
| 多路视频源注册 | 覆盖 |
| `source_type`、`camera_id`、`sync_group`、`video_index` | 覆盖 |
| 进入 `ExperimentService.set_video_inputs()` 主链路 | 覆盖 |
| 视频元数据字段规范 | 只保留必要引用，详见 `02_video_information_ingest.md` |
| 24 小时常驻采集 daemon | 不覆盖，当前不是完整常驻采集服务 |
| 相机 SDK / 硬件驱动层 | 不覆盖，仅预留字段 |

## 三、核心输入

| 输入 | 说明 |
|---|---|
| 本地视频文件 | 通过 `upload/video` 上传，适合离线实验和回归测试 |
| 实时视频流 | 通过 `upload/stream` 注册，适合 RTSP、USB、HTTP、RTMP、UDP 等来源 |
| `camera_id` | 机位唯一标识，例如 `cam_front`、`cam_top` |
| `sync_group` | 同步组，例如 `bench_01` |
| `video_index` | 同一实验内的视频输入顺序 |
| `start_offset_sec` | 简单时间偏移，用于初级多机位对齐 |
| `sync_method` / `sync_anchors` | 高精度同步输入，详见 `03_dialogue_video_time_alignment.md` |

支持的 `source_type`：

| 类型 | 用途 |
|---|---|
| `file` | 本地视频文件 |
| `rtsp` | IP 摄像头或编码器 |
| `usb` | 本机 USB 摄像头或采集卡 |
| `http` | HTTP 视频源或网关转发 |
| `rtmp` | 推流输入 |
| `udp` | 低延迟受控网络输入 |

## 四、处理流程

```text
视频文件 / RTSP / USB / HTTP / RTMP / UDP
  -> upload/video 或 upload/stream
  -> experiment.json.video_inputs
  -> ExperimentService.set_video_inputs()
  -> _extract_video()
  -> 多路抽帧 + 机位 metadata
  -> 时间同步
  -> material_stream.json
  -> preprocessing.json
  -> material_index.sqlite
```

与整条流水线的关系：

| 上游 | 本文处理 | 下游 |
|---|---|---|
| 用户上传文件或注册流 | 统一形成 `video_inputs` | `02` 规范视频元数据 |
| 摄像头/流地址 | 绑定 `camera_id/sync_group/video_index` | `03` 做时间同步 |
| 多路视频输入 | 进入实验处理主链路 | `07` 汇总为统一素材主线 |

## 五、关键接口与数据结构

### 1. 实时流注册请求

```json
POST /api/v1/experiments/{experiment_id}/upload/stream
{
  "source": "rtsp://192.168.1.41/stream1",
  "source_type": "rtsp",
  "camera_id": "cam_front",
  "sync_group": "bench_01",
  "start_offset_sec": 0.0,
  "capture_duration_sec": 120.0,
  "sync_method": "audio_flash",
  "sync_anchors": [
    {"local_time_sec": 0.0, "reference_time_sec": 1.0, "method": "audio_flash"}
  ],
  "clock_drift_ppm": 80.0
}
```

### 2. 后端流输入模型

```python
class UploadStreamRequest(BaseModel):
    source: str
    source_type: str = "rtsp"
    camera_id: Optional[str] = None
    sync_group: Optional[str] = None
    start_offset_sec: Optional[float] = 0.0
    capture_duration_sec: Optional[float] = 15.0
    sync_method: Optional[str] = None
    sync_anchors: Optional[List[Dict[str, Any]]] = None
    hardware_timecode_start_sec: Optional[float] = None
    sync_board_offset_sec: Optional[float] = None
    clock_scale: Optional[float] = None
    clock_drift_ppm: Optional[float] = None
```

### 3. 多路输入进入实验处理链路

```python
video_inputs = experiment_record.get("video_inputs") or experiment_record.get("video_metadata") or []
if hasattr(service, "set_video_inputs") and video_inputs:
    service.set_video_inputs(video_inputs)
```

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 | 下游使用 |
|---|---|---|
| `experiment.json.video_inputs` | 规范化视频输入列表 | 处理链路读取 |
| `experiment.json.video_metadata` | 输入注册或上传阶段产生的视频信息 | `02` 进一步说明 |
| `recordings/*.mp4` | 实时流边录文件 | `04` 生成实体 clip |
| `stream_health` | 每路流的读帧、重连、采样、录制指标 | 生产运行排障 |
| `material_stream.json` | 多源视频素材流 | `07` 统一素材主线 |
| `preprocessing.json` | 预处理结果 | `04/06/07` 使用 |
| `material_index.sqlite` | 检索索引 | `05` 检索层使用 |

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 多路视频源注册 | 同一实验下 `video_inputs` 至少包含 2 路以上输入 |
| 机位唯一性 | 每个输入都有稳定的 `camera_id` |
| 流来源追溯 | 每个输入都有 `source_type` 和 `source/video_path` |
| 同步参数保留 | `start_offset_sec`、`sync_anchors` 写入 `experiment.json` |
| 流健康指标 | `preprocessing.json.alignment_summary.stream_health` 有采样和录制统计 |
| 主链路处理 | `process` 阶段能读取 `video_inputs` 并生成素材流 |

### 验证命令

```powershell
$expId = "<experiment_id>"
$exp = Get-Content "outputs/experiments/$expId/experiment.json" -Raw | ConvertFrom-Json
$exp.video_inputs | Format-Table video_index,camera_id,source_type,start_offset_sec,sync_method
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| 默认 demo 仍偏单视频 | 展示层对多机位常态支持不充分 |
| 非 24 小时常驻采集 daemon | 长期监控场景还需要 stream daemon |
| 未接具体相机 SDK | 硬件帧号、曝光、硬件时间码无法直接读取 |
| 断流恢复仍是基础策略 | 已记录读帧失败和重连次数，但还缺少长期运行告警 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 固化 `camera_id` 命名规范 | 避免素材流混机位 |
| P0 | 用真实 2-4 路摄像头做 30 分钟采集测试 | 验证多路稳定性 |
| P1 | 将实时流采集封装为 stream daemon | 支持长期实验监控 |
| P1 | 增加长期 stream health dashboard | 生产可观测 |
| P2 | 按相机型号接入 SDK adapter | 获取硬件时间码和帧级元数据 |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `backend/main.py` | `UploadStreamRequest`、`upload_stream()`，正式流注册入口 |
| `backend/main.py` | `_run_formal_experiment_pipeline()`，把 `video_inputs` 交给实验服务 |
| `src/experiment/service.py` | `set_video_inputs()`，规范化多路视频输入 |
| `src/experiment/service.py` | `_extract_video()`，多路抽帧、录制和时间同步 |
| `src/labsopguard/time_sync.py` | 多机位 offset、anchor、drift 校准 |

与其他文档的关系：`02` 细化输入元数据，`03` 负责时间统一，`07` 把多机位素材收束为统一素材主线。
