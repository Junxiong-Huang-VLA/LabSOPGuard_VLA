# 配置说明

## 1. 配置层次

当前项目主要有四类配置：

1. 环境变量
2. 模型运行配置
3. 报警规则配置
4. 前端代理与联调配置

## 2. 环境变量

### 2.1 必填变量

| 变量名 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | DashScope / Qwen 调用鉴权 |
| `DASHSCOPE_BASE_URL` | DashScope 兼容模式地址 |

推荐值：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2.2 可选变量

| 变量名 | 说明 |
| --- | --- |
| `LABSOPGUARD_YOLO_MODEL` | 运行时覆盖默认权重路径 |
| `LABSOPGUARD_FONT_PATH` | 覆盖视频中文字渲染字体 |

## 3. 模型运行配置

文件：

- `configs/model/detection_runtime.yaml`

当前关键配置：

- `model: outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`
- `device: cuda:0`

主要字段说明：

| 字段 | 说明 |
| --- | --- |
| `model` | 默认 YOLO 权重路径 |
| `device` | 推理设备 |
| `class_registry` | 类别别名归一化 |
| `ppe.class_conf_thresholds` | PPE 识别阈值 |
| `sampling.base_interval_sec` | 抽帧基础间隔 |
| `sampling.max_vlm_frames` | Qwen 最大分析帧数 |
| `detection.confidence_threshold` | YOLO 置信度阈值 |
| `detection.iou_threshold` | YOLO NMS IoU 阈值 |
| `detection.max_detections` | 单帧最大框数 |

## 4. 报警规则配置

文件：

- `configs/alerts/alerting.yaml`

当前 `video_analysis_rules` 已配置：

- `missing_gloves`
- `missing_goggles`
- `missing_lab_coat`

每条规则包含：

| 字段 | 说明 |
| --- | --- |
| `severity` | 严重级别 |
| `color` | 颜色 |
| `title` | 标题 |
| `message` | 文案 |
| `related_classes` | 关联类别 |

## 5. 前端配置

前端目录：

- `frontend-app/`

当前 API 基地址：

- `/api/v1`

Vite 代理到本地后端：

- `http://127.0.0.1:8000`

## 6. 运行时覆盖优先级

视频分析时，YOLO 模型路径按以下优先级解析：

1. 接口表单 `yolo_model`
2. 环境变量 `LABSOPGUARD_YOLO_MODEL`
3. `detection_runtime.yaml` 中的 `model`

## 7. 推荐配置方式

开发环境建议：

1. `.env` 中存放 DashScope 配置
2. `detection_runtime.yaml` 中配置默认权重
3. 仅在调试或对比实验时使用 `yolo_model` 覆盖

## 8. 常见配置问题

### 8.1 权重路径不存在

现象：

- 后端日志提示模型不存在
- 视频分析只有语义结果或完全失败

处理：

1. 检查 `configs/model/detection_runtime.yaml`
2. 检查 `LABSOPGUARD_YOLO_MODEL`
3. 确认本地权重文件是否存在

### 8.2 Qwen 不生效

现象：

- `scene_description` 为空或为 `vlm_unavailable`

处理：

1. 检查 `DASHSCOPE_API_KEY`
2. 检查 `DASHSCOPE_BASE_URL`
3. 确认网络可访问 DashScope

### 8.3 中文叠字异常

处理：

1. 检查系统是否存在中文字体
2. 用 `LABSOPGUARD_FONT_PATH` 指定字体文件
3. 优先使用 Windows `msyh.ttc` 或 `simhei.ttf`
