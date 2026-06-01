# LabSOPGuard 集成缺口分析与修复方案

> 日期：2026-05-11 | 基于代码审计：已实现但未接入的模块

---

## 一、核心问题

上一轮已实现了 ExperimentSegmenter、ProcessingQueue、StageTimer、Resilience 等模块，但它们**只是独立代码，没有接入主 pipeline**。相当于造好了零件但没装到车上。

| 模块 | 代码 | 接入状态 |
|------|------|---------|
| ExperimentSegmenter（实验边界检测） | ✓ 完整 | **未接入** pipeline |
| ProcessingQueue（并发控制） | ✓ 完整 | **未接入** backend |
| StageTimer（耗时追踪） | ✓ 完整 | **未接入** service |
| ASR 语音转写 | ✓ 已有模块 | **未接入** pipeline |
| 双流合并分割 | ✗ 缺失 | 未实现 |
| 前端分段 UI | ✗ 缺失 | 未实现 |
| 手动边界修正 API | ✗ 缺失 | 未实现 |

---

## 二、修复方案（7 个集成任务）

### 任务 1: ExperimentSegmenter 接入主 pipeline

**问题**：`experiment_segmenter.py` 已实现三级边界检测，但 `ExperimentService.process()` 从未调用它。

**修复**：在 `process()` 的 Stage 1（帧提取）之后、Stage 2（检测）之前，插入实验分割判断。

```python
# src/experiment/service.py process() 方法中:

# Stage 1: 视频帧提取
self._extract_video(experiment)

# Stage 1.5: 实验边界检测（新增）
segmentation = self._detect_experiment_boundaries(experiment)
if segmentation.total_segments > 1:
    return self._process_sub_experiments(experiment, segmentation)

# Stage 2: 继续单实验处理...
```

**涉及文件**：
- `src/experiment/service.py`：添加 `_detect_experiment_boundaries()` 和 `_process_sub_experiments()` 方法

**预估工时**：3h

---

### 任务 2: ProcessingQueue 接入后端任务调度

**问题**：`processing_queue.py` 有完整的 semaphore 队列，但 `backend/main.py` 的 `/process` 接口直接用 `background_tasks.add_task()`，无并发限制。

**修复**：在 process 路由中包裹 ProcessingQueue：

```python
# backend/main.py 中 POST /api/v1/experiments/{id}/process:

from labsopguard.processing_queue import get_processing_queue

queue = get_processing_queue(max_concurrent=1)

async def _process_with_queue(experiment_id, ...):
    queue.enqueue(experiment_id)
    await queue.acquire(experiment_id)
    try:
        result = _run_experiment_pipeline(experiment_id, ...)
    finally:
        queue.release(experiment_id)
    return result
```

**涉及文件**：
- `backend/main.py`：修改 process 路由
- 新增 `GET /api/v1/processing/queue-status`：返回队列状态

**预估工时**：2h

---

### 任务 3: StageTimer 接入 pipeline 并输出 timing JSON

**问题**：`observability.py` 的 StageTimer 完整但从未使用，处理完看不到各阶段耗时。

**修复**：在 `ExperimentService.process()` 每个 Stage 包裹 `timer.measure()`：

```python
def process(self, ...):
    from labsopguard.observability import StageTimer
    timer = StageTimer()
    
    with timer.measure("ingestion"):
        self._extract_video(experiment)
    
    with timer.measure("video_understanding"):
        self._analyze_frames(experiment)
    
    with timer.measure("physical_events"):
        self._generate_physical_events(experiment)
    
    # ... 每个 stage 都包裹
    
    # 保存 timing
    timer.save(output_dir / "processing_timing.json")
```

**涉及文件**：
- `src/experiment/service.py`

**预估工时**：1h

---

### 任务 4: ASR 语音转写接入 pipeline

**问题**：项目已有 `asr.py` + Qwen ASR 能力，但主 pipeline 完全不调用。实验中的口头描述（如"现在开始称量 2.5 克氯化钠"）是极强的语义信号，可辅助步骤推理和实验分割。

**修复**：在 Stage 2（视频理解）中增加音频提取 + ASR：

```python
# Stage 2 追加:
with timer.measure("audio_transcription"):
    transcript = self._transcribe_audio(experiment)
    experiment.transcript_segments = transcript
```

**流程**：
```
视频文件 → ffmpeg 提取音频 → Qwen ASR 转写 → transcript_segments
    → 辅助步骤推理（"现在开始..." → step indicator）
    → 辅助实验分割（长静默 = 实验间空隙）
```

**涉及文件**：
- `src/experiment/service.py`：添加 `_transcribe_audio()` 方法
- 复用已有 `src/labsopguard/asr.py`

**预估工时**：4h

---

### 任务 5: 双流预分割合并

**问题**：双视角视频（top_view + bottom_view）各自独立做预分割，但没有合并两流的活跃段再进行实验分割。可能一个流静止但另一个流在操作。

**修复**：合并两流的 ActivitySegment 为统一时间线后再做实验边界检测：

```python
def _merge_dual_stream_segments(self, segments_top, segments_bottom):
    """取两流活跃段的并集作为统一活跃时间线"""
    all_segments = sorted(segments_top + segments_bottom, key=lambda s: s.start_sec)
    # 合并重叠段
    merged = merge_overlapping(all_segments)
    return merged
```

**涉及文件**：
- `src/experiment/service.py`：双流合并逻辑
- `src/labsopguard/event_preprocessing/experiment_segmenter.py`：接收合并后结果

**预估工时**：2h

---

### 任务 6: 前端实验分段 UI

**问题**：当检测到多段实验时，前端无法展示分段信息、无法让用户确认/修正边界。

**修复**：
1. 实验 workspace 增加"分段时间轴"组件
2. 显示检测到的实验段及其时间范围
3. 支持手动拖拽调整边界
4. "确认分割"按钮触发子实验处理

**涉及文件**：
- `frontend/src/components/ExperimentSegmentTimeline.tsx`（新建）
- `frontend/src/pages/ExperimentWorkspace.tsx`（集成）

**预估工时**：6h

---

### 任务 7: 实验分割手动修正 API

**问题**：自动分割可能误判（如操作员接电话 5 分钟被拆成两个实验）。需要 API 让用户合并/拆分/调整边界。

**修复**：

```
POST /api/v1/experiments/{id}/segmentation/override
Body: {
  "segments": [
    {"start_sec": 0, "end_sec": 2700, "label": "实验1-固体称量"},
    {"start_sec": 4200, "end_sec": 7200, "label": "实验2-移液操作"}
  ]
}
```

**涉及文件**：
- `backend/main.py`：新增路由
- `src/experiment/service.py`：支持手动分段后重跑

**预估工时**：3h

---

## 三、实施优先级

```
P0（核心集成，使已有代码生效）:
├─ 任务 1: ExperimentSegmenter 接入 pipeline     (3h)
├─ 任务 2: ProcessingQueue 接入 backend          (2h)
├─ 任务 3: StageTimer 接入 + timing 输出         (1h)
└─ 任务 5: 双流预分割合并                        (2h)
                                                  ── 8h

P1（增强能力）:
├─ 任务 4: ASR 语音转写接入                      (4h)
└─ 任务 7: 手动修正 API                          (3h)
                                                  ── 7h

P2（用户体验）:
└─ 任务 6: 前端分段 UI                           (6h)
                                                  ── 6h

总计: ~21h
```

---

## 四、集成后的完整 pipeline 效果

```
POST /api/v1/experiments/{id}/process
    │
    ▼ ProcessingQueue.acquire() ← 并发控制生效
    │
    ├─ timer.measure("ingestion")
    │   └─ 视频帧提取
    │
    ├─ timer.measure("experiment_segmentation") ← 新增
    │   ├─ 预分割（双流各跑一次）
    │   ├─ 双流合并为统一时间线
    │   ├─ ExperimentSegmenter.segment() 检测边界
    │   └─ if N>=2: 拆分为子实验分别处理
    │
    ├─ timer.measure("audio_transcription") ← 新增
    │   └─ ffmpeg 提取音频 → Qwen ASR → transcript
    │
    ├─ timer.measure("video_understanding")
    │   └─ YOLO + VLM（带重试/熔断）
    │
    ├─ timer.measure("physical_events")
    │   └─ EventPreprocessingEngine（带缓存）
    │
    ├─ timer.measure("step_reasoning")
    │   └─ 步骤推理（结合 transcript）
    │
    ├─ timer.measure("evidence_linking")
    │   └─ 证据关联 + 全局索引同步
    │
    └─ timer.save("processing_timing.json") ← 新增
    │
    ▼ ProcessingQueue.release()
```

---

## 五、验收标准

| 场景 | 预期行为 |
|------|---------|
| 上传 3h 双视角视频 | 自动拆出 2-3 段实验，各自独立分析 |
| 两个用户同时提交分析 | 第二个排队，API 返回队列位置和预估时间 |
| 分析完成后查看 timing | `processing_timing.json` 记录各阶段秒数 |
| Qwen API 超时 | 重试 3 次后降级，pipeline 继续，不阻塞 |
| 语义搜索"移液操作" | 跨所有实验返回匹配素材，按语义相似度排序 |
| 前端展示分段 | 时间轴标注实验边界，可手动调整 |
