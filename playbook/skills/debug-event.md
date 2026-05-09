# debug-event — 事件检测链路调试

诊断某实验的事件检测问题：事件为空、步骤推理跳过、clip 未生成等。

## 用法
```
/debug-event [experiment_id]
```

## 执行步骤

### 1. 定位实验目录
```bash
ls outputs/experiments/<id>/
```
检查关键文件是否存在：`physical_events.json`、`material_stream.json`、`materials/events/`

### 2. 检查视频文件
```bash
python -c "
import json
with open('outputs/experiments/<id>/experiment.json') as f:
    d = json.load(f)
print(d.get('source_video'))
"
```
确认 `source_video` 路径存在。

### 3. 检查事件数量
```bash
python -c "
import json
with open('outputs/experiments/<id>/physical_events.json') as f:
    events = json.load(f)
from collections import Counter
c = Counter(e['event_type'] for e in events)
print(c)
"
```

### 4. 检查步骤推理
读取 `outputs/experiments/<id>/steps_bridge_result.json`（若存在），查看各步骤 confidence 和 grade。

### 5. 检查 steps.json
确认实验的 `steps.json` 是否包含 `required_event_types`；若无，`_enrich_steps_for_bridge` 应自动注入。

### 6. 重新触发链路
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/experiments/<id>/materials/publish
```

### 7. 输出诊断报告
列出：
- 事件类型分布
- 哪些步骤被匹配/跳过
- 错误原因（若有）
- 建议修复方案
