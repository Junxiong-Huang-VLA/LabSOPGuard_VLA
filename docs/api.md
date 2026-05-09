# API 文档

## 1. 总览

后端默认基地址：

- `http://127.0.0.1:8000/api/v1`

交互式接口文档：

- `http://127.0.0.1:8000/docs`

当前正式支持两组接口：

- 实验理解接口：`/experiments/*`
- 视频分析接口：`/video-analysis/*`

## 2. 视频分析接口

### 2.1 提交视频分析任务

`POST /api/v1/video-analysis/analyze`

请求类型：

- `multipart/form-data`

表单字段：

- `video`：必填，视频文件
- `sample_interval`：可选，抽帧间隔，默认 `3.0`
- `max_frames`：可选，最大抽帧数，默认 `10`
- `yolo_model`：可选，自定义 YOLO 权重路径

返回示例：

```json
{
  "task_id": "cad5c35b-7d45-4b81-bc60-f3ed502df849",
  "status": "queued",
  "message": "Video analysis started in background"
}
```

### 2.2 查询任务状态

`GET /api/v1/video-analysis/status/{task_id}`

返回字段重点：

- `status`：`queued` / `running` / `completed` / `failed`
- `progress`
- `current_stage`
- `video_ready`
- `json_ready`
- `error_message`

返回示例：

```json
{
  "task_id": "cad5c35b-7d45-4b81-bc60-f3ed502df849",
  "status": "completed",
  "progress": 1.0,
  "current_stage": "artifacts_ready",
  "video_ready": true,
  "json_ready": true
}
```

### 2.3 下载标注视频

`GET /api/v1/video-analysis/download/{task_id}/video`

返回：

- `video/mp4`

### 2.4 下载分析 JSON

`GET /api/v1/video-analysis/download/{task_id}/json`

返回：

- `application/json; charset=utf-8`

JSON 结构示例：

```json
[
  {
    "frame_idx": 0,
    "timestamp_sec": 0.0,
    "detections": [],
    "scene_description": "实验室工作台上放置一台电子天平……",
    "detected_activities": ["称量样品", "准备实验材料"],
    "object_labels": ["balance", "paper", "reagent_bottle"],
    "step_indicators": ["称量", "样品准备"],
    "ppe_status": {
      "gloves": true,
      "goggles": false,
      "lab_coat": false
    },
    "vlm_confidence": 0.87,
    "alerts": ["missing_goggles", "missing_lab_coat"],
    "alert_details": [
      {
        "rule_id": "missing_goggles",
        "severity": "high",
        "color": "#FB8C00",
        "title": "护目镜缺失",
        "message": "未检测到护目镜，请确认实验人员已佩戴眼部防护。",
        "related_classes": ["glasses", "goggles"]
      }
    ]
  }
]
```

## 3. 实验理解接口

### 3.1 创建实验

`POST /api/v1/experiments`

### 3.2 查询实验列表

`GET /api/v1/experiments`

### 3.3 查询实验详情

`GET /api/v1/experiments/{experiment_id}`

### 3.4 查询实验状态

`GET /api/v1/experiments/{experiment_id}/status`

### 3.5 上传实验视频

`POST /api/v1/experiments/{experiment_id}/upload/video`

### 3.6 上传上下文

`POST /api/v1/experiments/{experiment_id}/upload/context`

### 3.7 上传协议文本

`POST /api/v1/experiments/{experiment_id}/upload/protocol`

### 3.8 启动处理

`POST /api/v1/experiments/{experiment_id}/process`

### 3.9 查询分析结果

- `GET /api/v1/experiments/{experiment_id}/analysis`
- `GET /api/v1/experiments/{experiment_id}/timeline`
- `GET /api/v1/experiments/{experiment_id}/steps`
- `GET /api/v1/experiments/{experiment_id}/steps/{step_id}`
- `GET /api/v1/experiments/{experiment_id}/evidence`
- `GET /api/v1/experiments/{experiment_id}/structured`

### 3.10 修订步骤

`PATCH /api/v1/experiments/{experiment_id}/steps/{step_id}`

## 4. 错误处理约定

常见错误码：

- `400`：参数错误
- `404`：任务或资源不存在
- `422`：请求字段校验失败
- `500`：后端内部异常
- `503`：任务存储或依赖服务不可用

## 5. 联调建议

前端和测试脚本优先通过这两条链路联调：

1. `POST /api/v1/video-analysis/analyze`
2. `GET /api/v1/video-analysis/status/{task_id}`

原因：

- 输出产物直观
- 能快速验证模型、Qwen、下载、标注视频、JSON 编码是否正常

## PTZ snapshot API

Use this endpoint when another backend needs one still image from the pan-tilt camera. It returns raw JPEG bytes directly, which is simpler and smaller than putting base64 image data into JSON.

Current PTZ source is `opencv:0` by default, not `cam0`. Do not call `/api/v1/cameras/cam0/snapshot` for the PTZ camera unless `PTZ_VIDEO_MODE` has explicitly been changed to a shared/WVD camera source.

### PTZ tracker current frame

`GET /api/v1/ptz-tracker/snapshot`

Response:
- `200 image/jpeg`: latest frame from the PTZ tracker service, including the tracker overlay.
- `503`: the PTZ tracker service is stopped, or no frame is ready yet.

Response headers:
- `X-PTZ-Video-Source`: actual PTZ source, for example `opencv:0`.
- `X-PTZ-Frame-Timestamp`: Unix timestamp of the frame cached by the PTZ service.

Query parameters:
- `auto_start=false`: start the PTZ tracker service automatically if it is stopped. Default is `false` to avoid implicitly starting camera/model/MQTT work from a read request.
- `timeout_ms=1000`: max wait time for the first frame, from `0` to `5000` milliseconds.

Example:
```bash
curl -o ptz_snapshot.jpg "http://127.0.0.1:8000/api/v1/ptz-tracker/snapshot"
```

If the caller is allowed to start the PTZ service:
```bash
curl -o ptz_snapshot.jpg "http://127.0.0.1:8000/api/v1/ptz-tracker/snapshot?auto_start=true&timeout_ms=3000"
```

### Raw camera frame

If the caller needs a raw multi-camera frame rather than the PTZ tracker overlay, use:

`GET /api/v1/cameras/{camera_id}/snapshot`

Example:
```bash
curl -o raw_camera.jpg "http://127.0.0.1:8000/api/v1/cameras/{camera_id}/snapshot?quality=95"
```
