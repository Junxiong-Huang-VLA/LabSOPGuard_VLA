# LabSOPGuard 多路摄像实时采集方案报告

**日期：** 2026-04-19  
**范围：** 多机位视频接入、边录边索引、时间对齐、关键帧/关键片段落盘  
**结论：** 主链路已经具备工程化接入能力；真实生产部署还需要按相机型号补充设备 SDK、PTP/同步板驱动和长期稳定性压测。

---

## 一、当前目标与整体判断

多路摄像实时采集的目标不是单纯“能打开多个视频源”，而是把多个机位统一组织成可追溯、可检索、可回切的时间锚点素材流。

| 能力项 | 当前实现 | 状态 |
|---|---|---|
| 多视频源注册 | 后端支持上传文件和注册 RTSP/USB/HTTP 等流输入，保存 `camera_id`、`source_type`、`start_offset_sec` 等字段 | 已具备 |
| 实时流采样 | `ExperimentService` 可按 `sample_interval` 从多路源抽帧，并写入素材流 | 已具备 |
| 边录边索引 | 非文件流可写入 `recordings/*.mp4`，关键片段优先从录制文件回切 | 已接入 |
| 环形缓存 | `RingSegmentRecorder` 支持分段录制、保留窗口、历史片段回切 | 已具备模块 |
| 多机位同步 | 支持显式 offset、硬件时间码、同步板 offset、声画/闪光 anchor、clock drift ppm | 已接入模型 |
| 素材资产化 | 输出 `material_stream.json`、`preprocessing.json`、`material_index.sqlite`、关键 clip 文件 | 已具备 |
| 生产级硬件适配 | 相机 SDK、同步板 GPIO/串口、PTP 时间戳读取需要按设备补 adapter | 待接入 |

---

## 二、采集链路架构

### 2.1 主链路

```text
多路视频源
  -> /api/v1/experiments/{id}/upload/video 或 upload/stream
  -> video_inputs / video_metadata
  -> ExperimentService.set_video_inputs()
  -> VideoFrameExtractor.extract_stream_frames()
      -> 抽帧
      -> 可选录制 recorded_file_path
      -> 帧级 timestamp / local_timestamp
  -> TimeSyncCalibrator
      -> local time 转 global time
      -> offset + clock_scale + drift ppm
  -> time_anchored_material_stream
  -> preprocessing.json / material_stream.json / clips/*.mp4
  -> material_index.sqlite
  -> materials/search 联合检索
```

### 2.2 输入源类型

| 输入类型 | 示例 | 适用场景 | 处理方式 |
|---|---|---|---|
| 本地文件 | `D:\videos\cam_a.mp4` | 离线复盘、标注、回归测试 | 直接抽帧，按文件时间轴处理 |
| RTSP | `rtsp://camera-a/stream1` | IP 摄像头、工业相机网关 | 抽帧 + 可选录制到 `recordings` |
| HTTP/HLS | `http://.../stream` | 网关转发、边缘盒子输出 | 作为网络流读取 |
| USB/采集卡 | `0` / `1` | 本机直连摄像头 | 按 OpenCV capture id 读取 |
| RTMP/UDP | `rtmp://...` / `udp://...` | 直播推流、低延迟链路 | 作为连续流入口读取 |

### 2.3 流注册字段

| 字段 | 说明 | 推荐要求 |
|---|---|---|
| `source` | 视频源地址或设备编号 | 必填 |
| `source_type` | `rtsp`、`usb`、`file`、`http` 等 | 必填 |
| `camera_id` | 机位唯一 ID，如 `cam_front`、`cam_top` | 必填，禁止复用 |
| `sync_group` | 同步组，如 `bench_01` | 多机位建议填写 |
| `start_offset_sec` | 显式起始偏移 | 无硬件同步时使用 |
| `capture_duration_sec` | 单次采集时长 | demo/测试建议 15-60s |
| `sync_method` | `hardware_timecode`、`sync_board`、`audio_flash`、`manual` | 生产环境建议明确 |
| `sync_anchors` | 声画/闪光/同步板锚点列表 | 高精度同步建议提供 |
| `hardware_timecode_start_sec` | 硬件时间码起点 | 支持时间码相机时提供 |
| `sync_board_offset_sec` | 同步板测得 offset | 支持同步板时提供 |
| `clock_drift_ppm` | 长视频时钟漂移 | 长视频必须记录 |

### 2.4 API 示例

```json
POST /api/v1/experiments/{experiment_id}/upload/stream
{
  "source": "rtsp://192.168.1.41/stream1",
  "source_type": "rtsp",
  "camera_id": "cam_front",
  "sync_group": "bench_01",
  "capture_duration_sec": 120.0,
  "sync_method": "audio_flash",
  "sync_anchors": [
    {"local_time_sec": 0.0, "reference_time_sec": 1.000, "method": "audio_flash"},
    {"local_time_sec": 60.0, "reference_time_sec": 61.018, "method": "audio_flash"}
  ],
  "clock_drift_ppm": 300.0
}
```

---

## 三、实时采集与边录边索引

### 3.1 边录边索引策略

实时流不能只保存抽帧记录，否则后续无法稳定回切历史片段。当前策略是：

```text
实时流读取
  -> 抽关键帧进入 material_stream
  -> 同步写 recorded_file_path
  -> preprocessing 根据关键时间窗口生成 key_clips
  -> clip 文件保存到 outputs/experiments/{id}/clips
  -> material_index.sqlite 建索引
```

### 3.2 环形缓存策略

| 参数 | 建议值 | 说明 |
|---|---|---|
| `segment_duration_sec` | 5-30s | 分段越短，回切越灵活；分段越长，文件数越少 |
| `retention_sec` | 300-1800s | 生产建议至少保留 5-30 分钟 |
| `fps` | 10-30 | 取决于动作速度和带宽 |
| `codec` | H.264/H.265 | H.265 省带宽，H.264 兼容性更好 |
| `clip_window` | 事件前 1-3s，事件后 2-5s | 便于保留上下文 |

### 3.3 输出物

| 文件/字段 | 内容 | 用途 |
|---|---|---|
| `recordings/*.mp4` | 连续流录制文件 | 后续回切 clip、人工复查 |
| `clips/*.mp4` | 关键事件片段 | 证据链、报告、检索结果回放 |
| `material_stream.json` | 带时间锚点的多视频素材流 | 统一素材主线 |
| `preprocessing.json` | 关键帧、关键片段、同步摘要、变化事件 | 预处理结果 |
| `material_index.sqlite` | SQLite + FTS 索引 | 按物体/动作/时间/机位/clip 查询 |

---

## 四、多机位同步方案

### 4.1 同步优先级

| 优先级 | 方法 | 输入 | 适用场景 | 精度判断 |
|---|---|---|---|---|
| 1 | 硬件时间码 | `hardware_timecode_start_sec` | 支持时间码的工业相机/采集卡 | 最高，依赖硬件 |
| 2 | 同步板 | `sync_board_offset_sec`、GPIO/串口事件 | 多相机实验台固定部署 | 高，推荐生产使用 |
| 3 | 声画/闪光锚点 | `sync_anchors` | 普通 IP 摄像头、手机、USB 摄像头 | 中高，依赖锚点质量 |
| 4 | 显式 offset | `start_offset_sec` | demo、半自动校准 | 中，需人工确认 |
| 5 | 顺序时间轴 | 无 | 单机位或粗略多机位 | 低，不建议生产使用 |

### 4.2 长视频漂移修正

普通摄像头长时间录制会出现时钟漂移。当前同步模型使用：

```text
global_time = offset_sec + local_time * clock_scale
clock_drift_ppm = (clock_scale - 1.0) * 1,000,000
```

如果只有一个锚点，只能校准 offset；如果有两个或更多锚点，可以同时拟合 offset 和 drift。

### 4.3 锚点设计建议

| 锚点类型 | 操作方式 | 推荐频率 | 注意事项 |
|---|---|---|---|
| 闪光锚点 | 所有机位同时看到 LED 闪光 | 开始、结束各一次；长视频每 5-10 分钟一次 | 避免画面过曝，保证所有机位可见 |
| 声音锚点 | 拍手、蜂鸣器、同步声源 | 开始、结束各一次 | 麦克风延迟需校准 |
| 同步板脉冲 | GPIO/TTL 触发 | 固定周期或事件触发 | 需要设备 adapter |
| 硬件时间码 | 相机或采集卡提供 | 全程 | 需要读取 SDK 元数据 |

---

## 五、关键指标与验收标准

| 指标 | demo 标准 | 生产标准 |
|---|---|---|
| 多路接入数量 | 2-4 路 | 4-12 路，视带宽和 GPU 而定 |
| 单路端到端采集延迟 | < 2s | < 500ms-1s，按实验需求定义 |
| 关键帧落盘成功率 | > 95% | > 99% |
| clip 回切成功率 | > 90% | > 99% |
| 多机位同步误差 | < 500ms | < 20-100ms，取决于硬件 |
| 长视频漂移修正 | 支持手动参数 | 自动 anchor 拟合 + 残差报告 |
| 断流恢复 | 手动重试 | 自动重连 + 缺口标记 |

---

## 六、现存差距

| 问题 | 影响 | 建议 |
|---|---|---|
| 未接具体相机 SDK | 无法读取硬件时间码、曝光、帧号等底层元数据 | 按相机品牌补 `CaptureAdapter` |
| 同步板尚无设备驱动 | 只能用字段表达同步结果，不能直接采集脉冲 | 接串口/GPIO/网络同步板 |
| 网络流断流恢复仍基础 | 长时间运行可能出现帧缺口 | 增加重连、心跳、丢帧统计 |
| 环形缓存尚未挂成常驻服务 | 模块可用，但默认实验 pipeline 不是常驻采集服务 | 增加 stream daemon |
| GPU/CPU 调度未按多路负载优化 | 多路高分辨率时可能掉帧 | 增加采样队列、背压和降帧策略 |

---

## 七、落地实施清单

| 阶段 | 任务 | 产物 |
|---|---|---|
| P0 | 统一所有新开发到 `LabSOPGuard` | 单一主项目 |
| P0 | 为每路摄像头定义 `camera_id`、`sync_group`、`source_type` | 机位清单 |
| P0 | 启用 `recorded_file_path` 和 clip materialization | 可回切证据片段 |
| P1 | 接入同步板/硬件时间码 adapter | 高精度同步元数据 |
| P1 | 将 `RingSegmentRecorder` 挂为常驻 stream daemon | 环形缓存服务 |
| P1 | 增加断流重连、帧率监控、丢帧告警 | 生产稳定性 |
| P2 | 增加多路 GPU/CPU 调度策略 | 高并发性能 |
| P2 | 建立长视频 drift 自动残差报告 | 同步质量可观测 |

---

## 八、相关文件

| 文件 | 作用 |
|---|---|
| `backend/main.py` | 视频上传、流注册、素材检索 API |
| `src/experiment/service.py` | 多路视频输入、同步、抽帧、边录、输出落盘 |
| `src/labsopguard/time_sync.py` | offset、drift、anchor 校准 |
| `src/labsopguard/stream_buffer.py` | 环形分段缓存和历史 clip 回切 |
| `src/labsopguard/preprocessing.py` | 关键帧、关键片段、同步摘要、素材流输出 |
| `src/labsopguard/retrieval.py` | SQLite/FTS/embedding 素材检索 |
| `src/labsopguard/semantic_events.py` | 稳定物体跟踪和语义事件检测 |

