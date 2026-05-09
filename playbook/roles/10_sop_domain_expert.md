# 角色十：实验室 SOP 领域专家（Lab SOP Domain Expert）

## 职责定位

负责提供实验室操作规范（SOP）的领域知识，定义实验步骤、合规标准与 PPE 要求，并将这些知识翻译为系统可消费的 `steps.json` 结构。

## 核心工作内容

### steps.json 编写规范

每个实验对应一个 `steps.json`，结构示例：
```json
[
  {
    "step_id": "step_ppe",
    "name": "穿戴防护装备",
    "description": "操作前必须佩戴手套和护目镜，穿白大褂",
    "required_event_types": ["hand_object_interaction"],
    "ppe_requirements": ["glove", "goggles", "lab_coat"],
    "order": 1,
    "mandatory": true
  },
  {
    "step_id": "step_prepare",
    "name": "准备实验器材",
    "description": "将所需仪器和试剂摆放至实验台",
    "required_event_types": ["object_move"],
    "order": 2,
    "mandatory": true
  }
]
```

### 可用事件类型（五类）
| 事件类型 | 对应操作 |
|---------|---------|
| `hand_object_interaction` | 拿取、放置、接触器具 |
| `object_move` | 器具位置移动 |
| `liquid_transfer` | 液体转移、倾倒、移液 |
| `panel_operation` | 操作天平、设备面板 |
| `container_state_change` | 开盖、关盖、换盖 |

### 实验类型模板

#### 称量实验
```
step_ppe → step_prepare → step_panel（天平调零）→ step_open_container → 
step_transfer（取样品）→ step_panel（读数）→ step_close_container
```

#### 移液与稀释实验
```
step_ppe → step_prepare → step_open_container → step_transfer（移液）→ 
step_transfer（稀释）→ step_close_container
```

#### 滴定实验
```
step_ppe → step_prepare → step_open_container → step_transfer（装液）→ 
step_panel（调零）→ step_transfer（滴定）→ step_panel（读数）
```

### PPE 合规标准
| 操作类型 | 必须 | 建议 |
|---------|------|------|
| 所有实验 | glove, lab_coat | goggles |
| 腐蚀性试剂 | glove, lab_coat, goggles | face_shield |
| 高温操作 | glove（耐热）, lab_coat | goggles |

### 与系统的接口
- `glove` → YOLO `gloved_hand` 类
- `lab_coat` → YOLO `lab_coat` 类
- `goggles` → **由 Qwen 语义判断**（当前数据集无 goggles 类）
- 不要期望 `goggles` 来自纯 YOLO 检测

### 文档维护
- 每个实验 SOP 对应 `data/raw/sop_docs/` 下一个文件
- steps.json 更新后通知后端开发者同步刷新实验配置
