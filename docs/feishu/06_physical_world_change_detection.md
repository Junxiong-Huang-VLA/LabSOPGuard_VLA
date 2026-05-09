# 物理世界变化事件语义层

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 已支持基线物理事件和若干实验语义事件 |
| 所属链路 | 多源预处理与时间对齐中的事件语义层 |
| 一句话结论 | 本文定义 LabSOPGuard 如何把视觉检测结果转为可追溯的 `PhysicalEvent`，并将事件类型回链到素材索引。 |

`06` 不是泛泛的视频变化检测说明，而是面向实验过程的物理事件语义层。事件会进入 `physical_events.json`、`preprocessing.detected_changes`，并被 `05` 检索层作为 `event_types` 使用。

## 二、本文目标与边界

本文聚焦“显著物理世界变化如何被记录为结构化事件”。核心包括事件类型、规则来源、置信度、来源追溯字段和阈值依据。

| 覆盖内容 | 不覆盖内容 |
|---|---|
| `scene_change`、`object_move`、`hand_contact` | 通用视频监控告警系统 |
| 容器开合、液位变化、试剂标签状态 | 专用视觉模型训练细节 |
| 事件 metadata 来源追溯 | 前端事件可视化 |
| 阈值规则 | 工业级精度承诺 |

## 三、核心输入

| 输入 | 说明 |
|---|---|
| `frame` | 视频帧信息，包含 `timestamp_sec`、`frame_id`、`camera_id` |
| `analysis.detected_objects` | 上游视觉检测对象，通常含 label、bbox、OCR 或液位信息 |
| `analysis.object_labels` | 帧级物体标签集合 |
| `analysis.detected_activities` | 帧级活动/动作标签 |
| `material_item_id` | 当前事件关联的素材项 ID |
| `camera_id` / `frame_id` | 事件追溯字段 |

## 四、处理流程

```text
视频帧
  -> 视觉分析
  -> detected_objects / object_labels / activities
  -> StableObjectTracker
  -> SemanticEventDetector
  -> PhysicalEvent
  -> physical_events.json
  -> preprocessing.detected_changes
  -> material_index.event_types
```

事件生成分两层：

| 层级 | 说明 |
|---|---|
| 基线事件 | 根据场景描述、物体集合、活动标签生成 `scene_change/object_move/hand_contact/liquid_transfer` |
| 语义事件 | 根据 bbox 几何、稳定 ID、液位、OCR 状态生成 `hand_contact_geometry/container_opened/liquid_level_change/reagent_label_state` |

## 五、关键接口与数据结构

### 1. 语义事件检测入口

```python
semantic_detector = SemanticEventDetector()

for frame, analysis, item in zip(self._video_frames, self._frame_analyses, self._material_stream):
    detected_objects = analysis.get("detected_objects", [])
    semantic_events = semantic_detector.update(
        timestamp_sec=frame.get("timestamp_sec", 0.0),
        detections=detected_objects,
        frame_metadata={
            "camera_id": frame.get("camera_id"),
            "frame_id": frame.get("frame_id"),
            "material_item_id": item.item_id,
        },
    )
```

### 2. 稳定物体 ID 跟踪

```python
class StableObjectTracker:
    def update(self, detections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 通过 IoU、中心点距离和类别一致性，把当前帧 detection 绑定到稳定 object_id。
        tracked_objects = []
        for detection in detections:
            matched_track = self._match_existing_track(detection)
            object_id = matched_track.object_id if matched_track else self._new_object_id(detection)
            tracked_objects.append({**detection, "object_id": object_id})
        return tracked_objects
```

### 3. 物理事件写入统一事件结构

```python
event = PhysicalEvent(
    experiment_id=experiment.experiment_id,
    event_type=str(semantic_event.get("event_type", "semantic_event")),
    timestamp_sec=float(semantic_event.get("timestamp_sec", ts)),
    duration_sec=self._frame_extractor.sample_interval_sec,
    location=location,
    description=str(semantic_event.get("description", "")),
    confidence=float(semantic_event.get("confidence", 0.6)),
    metadata={
        **base_metadata,
        **semantic_event.get("metadata", {}),
    },
)
```

### 当前规则阈值

| 事件 | 当前阈值/规则 | 说明 |
|---|---|---|
| `object_move` | 同一稳定物体 bbox 中心位移 `>= 15px` | 记录物体移动 |
| `hand_contact_geometry` | hand 与 object 的 IoU `>= 0.02` 或中心距离 `<= 40px` | bbox 几何接触 |
| `container_opened/closed` | cap/lid 与 container 中心距离 `<= 70px` 判为 closed，否则 open | 状态变化时才发事件 |
| `liquid_level_change` | 液位比例变化 `>= 0.08` | 记录液位升降 |
| `reagent_label_state` | OCR 文本长度 `>= 2` 判为 verified，否则 unreadable | 记录标签可读状态 |

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 |
|---|---|
| `physical_events.json` | 显著物理变化事件列表 |
| `preprocessing.json.detected_changes` | 预处理中的变化事件 |
| `event_types` | 索引中可随素材返回的事件类型 |
| `metadata.material_item_id` | 事件关联的素材项 |
| `metadata.camera_id` | 事件来源机位 |
| `metadata.frame_id` | 事件来源帧 |
| `metadata.rule/threshold` | 事件规则和阈值依据 |

样例事件：

```json
{
  "event_type": "hand_contact_geometry",
  "timestamp_sec": 42.0,
  "confidence": 0.73,
  "metadata": {
    "camera_id": "cam_front",
    "frame_id": 420,
    "material_item_id": "item_420",
    "hand_label": "hand",
    "object_label": "pipette",
    "iou": 0.04,
    "distance_px": 18.2
  }
}
```

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 有物理事件输出 | `physical_events.json` 存在且可解析 |
| 事件进入预处理 | `preprocessing.json.detected_changes` 包含事件 |
| 事件有时间戳 | 每个事件有 `timestamp_sec` |
| 事件有来源 | metadata 包含 `frame_id/camera_id/material_item_id` 中至少一类 |
| 事件可随素材检索 | 查询结果返回 `event_types` |
| 手-物接触有几何依据 | `hand_contact_geometry` 事件包含接触对象和几何信息 |
| 阈值依据可追溯 | 事件 metadata 包含 `rule`、`threshold` 或几何阈值字段 |

### 验证命令

```powershell
$expId = "<experiment_id>"
$events = Get-Content "outputs/experiments/$expId/physical_events.json" -Raw | ConvertFrom-Json
$events | Group-Object event_type | Select-Object Name,Count

$prep = Get-Content "outputs/experiments/$expId/preprocessing.json" -Raw | ConvertFrom-Json
$prep.detected_changes | Select-Object -First 10 event_type,timestamp_sec,confidence
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| 语义仍依赖通用检测输出和规则 | 阈值已记录，但对细粒度实验器材仍不够稳 |
| 液位变化缺少专用分割模型 | 透明容器和反光场景容易误判 |
| 试剂标签依赖 OCR 上游质量 | 小字、遮挡、反光会影响识别 |
| 物体 ID 是短时跟踪 | 长时间遮挡后可能切换 ID |
| 接触几何主要基于 bbox | 无深度时可能误判前后关系 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 用真实实验视频回放检查事件质量 | 找出误报和漏报类型 |
| P1 | 增加实验器材专用 detector | 提升 pipette、tube、beaker、cap 等识别 |
| P1 | 接入 OCR 标签识别 pipeline | 提升试剂标签状态准确率 |
| P1 | 增加深度或手关键点信息 | 提升手-物接触几何判断 |
| P2 | 增加 ReID 轨迹记忆 | 稳定长时物体 ID |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `src/labsopguard/semantic_events.py` | `StableObjectTracker`，稳定物体 ID 跟踪 |
| `src/labsopguard/semantic_events.py` | `SemanticEventDetector`，语义事件检测 |
| `src/experiment/service.py` | `_generate_physical_events()`，把语义事件转为 `PhysicalEvent` |
| `src/labsopguard/preprocessing.py` | `_build_detected_changes()`，把物理事件写入预处理结果 |
| `src/labsopguard/retrieval.py` | `event_types`，事件类型进入索引返回 |

与其他文档的关系：`06` 产出的 `detected_changes/event_types` 被 `05` 检索层使用，也会回链到 `07` 统一素材主线。
