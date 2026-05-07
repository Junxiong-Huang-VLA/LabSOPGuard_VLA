# 角色二：事件引擎开发者（Event Engine Developer）

## 职责定位

负责 `EventPreprocessingEngine` 及五类核心物理事件检测逻辑的开发与维护。将 YOLO 检测帧流转化为结构化物理事件。

## 核心工作内容

### 主要代码区域
```
src/labsopguard/event_preprocessing/
├── engine.py                  # 主引擎入口
├── event_proposal.py          # 五类事件提案构建
├── tracking/                  # 多目标跟踪
├── frame_detection_stream.py  # YOLO 帧检测流
├── key_material_extraction.py # clip + keyframe 生成
├── evidence_grading.py        # 事件置信度评分
├── selective_overlay.py       # 检测框叠加策略
├── action_resolution/         # 动作语义解析
└── state_resolution/          # 容器状态解析
```

### 五类事件维护
| 事件类型 | 核心逻辑位置 | 触发条件 |
|---------|------------|---------|
| `hand_object_interaction` | `event_proposal.py` | gloved_hand IoU/距离 |
| `object_move` | `event_proposal.py` | 接触 + 位移 > 阈值 |
| `liquid_transfer` | `source_target_resolution.py` | 倾倒姿态检测 |
| `panel_operation` | `event_proposal.py` | 接近 balance/panel |
| `container_state_change` | `state_resolution/` | lid IoU 变化 |

### 新增事件类型规范
新增事件时必须同时：
1. 在 `event_proposal.py` 实现检测逻辑
2. 在 `semantic_events.py` 注册 `VALID_EVENT_TYPES`
3. 在 `LabSOPGuard.md` 的 `_SOP_DEFAULT_STEPS` 中补充默认步骤映射
4. 在 `step_bridge/protocol_graph.py` 更新协议图

### 关键参数（可覆盖）
```bash
LABSOPGUARD_EVENT_MAX_FRAMES=36       # 最大采样帧数
LABSOPGUARD_EVENT_INTERVAL_SEC=2.0    # 基础采样间隔（秒）
```

### 调试工具
```bash
# 查看某实验的事件分布
python -c "
import json
events = json.load(open('outputs/experiments/<id>/physical_events.json'))
from collections import Counter
print(Counter(e['event_type'] for e in events))
"
# 使用 debug-event skill
/debug-event <id>
```

## 关注质量指标
- 每个实验事件数 ≥ 5（否则 step_bridge 无法有效匹配）
- `liquid_transfer` 事件 F1 ≥ 0.80（当前最难检测的事件类型）
- clip 生成成功率 100%（依赖 imageio_ffmpeg）

## 禁止事项
- 不得修改事件 schema 的 `event_type` 字符串（下游 step_bridge 直接依赖）
- 新事件不得绕过 `EventProposalBuilder` 直接写入 `physical_events.json`
