# 多路视频采集与接入方案

## 当前落地范围

- 多视频文件接入：实验记录可保存多条 `video_paths`，处理链路会按上传顺序或显式 `start_offset_sec` 一起处理。
- 流源接入：实验记录现在也可保存 `rtsp/file/usb` 风格的 `video_inputs`，正式实验链路会直接采样，不要求先落整段视频。
- 视频信息接入：上传时自动探测 `fps / frame_count / duration / width / height / size_bytes` 并保存到 `video_metadata` / `video_inputs`。
- 时间对齐：支持两种模式。
  - `sequential`：未提供显式偏移时，按视频顺序拼接到一条全局时间轴。
  - `explicit_offsets`：为每路视频输入 `start_offset_sec` 时，按全局偏移对齐。
- 文本/对话对齐：`context_inputs` 可带 `timestamp_sec`，或带 `local_timestamp_sec + video_index/video_asset_id` 映射到全局时间轴。

## 数据约定

- 视频输入记录：
  - `video_path`
  - `video_index`
  - `source_type`，如 `file / rtsp / usb`
  - `start_offset_sec` 可选
  - `camera_id` 可选
  - `sync_group` 可选
  - `capture_duration_sec` 可选，实时源采样窗口
- 上下文输入记录：
  - `text`
  - `kind` 可选，如 `transcript`
  - `timestamp_sec` 或 `start_time_sec`
  - `local_timestamp_sec + video_index/video_asset_id`

## 预处理产物

- `preprocessing.json`
  - `aligned_text`
  - `key_timestamps`
  - `video_index`
  - `detected_changes`
  - `video_streams`
  - `key_frames`
  - `key_clips`
  - `time_anchored_material_stream`
  - `alignment_summary`

## API 用法

- 本地视频上传：
  - `POST /api/v1/experiments/{experiment_id}/upload/video`
- 注册实时或网络流：
  - `POST /api/v1/experiments/{experiment_id}/upload/stream`
  - 请求体示例：

```json
{
  "source": "rtsp://camera-host/live",
  "source_type": "rtsp",
  "camera_id": "camera_top",
  "sync_group": "lab_demo",
  "start_offset_sec": 0.0,
  "capture_duration_sec": 15.0
}
```

- 触发处理：
  - `POST /api/v1/experiments/{experiment_id}/process`
  - 当输入是流源时，会走正式实验服务主链路并直接输出 `material_stream / preprocessing / structured`。

## 实时采集接入建议

- 采集侧统一转成 `RTSP/SRT/WebRTC` 之一，不直接把原始帧推给主服务。
- 每路流入库时都补 `camera_id + start_offset_sec/sync timestamp`。
- 当前主链路已能消费多视频文件和时间锚点；实时无线采集可继续在接入层把流录制切片后复用同一套预处理链路。
