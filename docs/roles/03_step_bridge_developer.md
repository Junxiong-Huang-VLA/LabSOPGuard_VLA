# 角色三：步骤推理开发者（Step Bridge Developer）

## 职责定位

负责 `StepBridgeEngine` 的开发与维护，将物理事件序列与 SOP 步骤进行匹配、评分与升降级决策。是合规判断的核心逻辑层。

## 核心工作内容

### 主要代码区域
```
src/labsopguard/step_bridge/
├── engine.py           # 步骤匹配主引擎
├── protocol_graph.py   # SOP 协议有向图
└── schemas.py          # 步骤推理结果 schema
```

### 工作流
1. 接收 `physical_events.json`（五类事件列表）
2. 加载实验 `steps.json`（若无 `required_event_types`，由 `_enrich_steps_for_bridge` 自动注入）
3. 按协议图顺序匹配每个步骤
4. 输出每步的 `confidence`、`grade`（`compliant`/`needs_review`/`non_compliant`）、`evidence_events`

### SOP 默认步骤（自动注入）
| 步骤 ID | required_event_types |
|--------|---------------------|
| `step_ppe` | `hand_object_interaction` |
| `step_prepare` | `object_move` |
| `step_open_container` | `container_state_change` |
| `step_transfer` | `liquid_transfer` |
| `step_panel` | `panel_operation` |
| `step_close_container` | `container_state_change` |

### 当前已知问题
- 步骤得分偏低（`needs_review`，置信度 ~0.18）
- 根因：测试视频中真实操作事件偏少，步骤证据不充分
- 不需要修改算法，真实视频跑出高质量事件后自然提升

### 调试方法
```python
# 手动运行 step_bridge
from labsopguard.step_bridge.engine import StepBridgeEngine
import json

events = json.load(open('outputs/experiments/<id>/physical_events.json'))
steps = json.load(open('path/to/steps.json'))
engine = StepBridgeEngine()
result = engine.run(events=events, steps=steps)
print(json.dumps(result, ensure_ascii=False, indent=2))
```

### 评估指标
- 步骤匹配率：`compliant` 步骤占比
- 误报率：标注为 `non_compliant` 但实际合规的比例
- 目标：在标准实验视频中 `compliant` ≥ 70%

## 关联约束
- 所有步骤 `grade` 枚举值只能是 `compliant`/`needs_review`/`non_compliant`
- 禁止在 step_bridge 内直接调用 Qwen API（语义增强由 `semantic_enhancer.py` 负责）
