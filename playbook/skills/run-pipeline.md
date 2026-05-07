# run-pipeline — 触发实验完整处理链路

触发或调试单个实验的完整分析链路：YOLO检测 → 事件提案 → clip生成 → 步骤推理。

## 用法
```
/run-pipeline [experiment_id]
```
不传 experiment_id 时，列出最近 5 个实验供选择。

## 执行步骤

### 1. 确认后端运行
```bash
curl -s http://127.0.0.1:8000/ | python -m json.tool
```
若未运行，提示用户执行：`.\scripts\start_full_stack.ps1 -Restart -SkipRedis`

### 2. 获取实验列表（无 id 时）
```bash
curl -s http://127.0.0.1:8000/api/v1/experiments | python -m json.tool
```

### 3. 触发完整处理
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/experiments/<id>/process | python -m json.tool
```

### 4. 补发 clip（旧实验缺少素材时）
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/experiments/<id>/materials/publish | python -m json.tool
```

### 5. 检查输出
检查 `outputs/experiments/<id>/` 下关键文件是否生成：
- `physical_events.json` — 事件数量
- `material_stream.json` — 素材数量
- `analysis/annotated.mp4` — 标注视频

### 6. 诊断链路问题
若链路失败，读取日志并定位：
- YOLO 权重问题 → 检查 `detection_runtime.yaml`
- 事件为空 → 检查视频文件是否存在
- 步骤推理跳过 → 检查 `steps.json` 是否有 `required_event_types`（自动注入机制已启用）
