# LabSOPGuard 最终缺口分析与修复方案

> 日期：2026-05-11 | 第三轮审计（集成后）

---

## 一、当前状态

后端 pipeline 已完整：预分割 → 实验分割 → YOLO(GPU) → VLM(带重试) → 事件检测 → 索引 → 全局搜索。但**前端展示和用户体验层**有明显空洞，且 ASR 模块虽然存在但未接入。

---

## 二、剩余缺口（按优先级排序）

### HIGH: 功能断链

| # | 问题 | 影响 |
|---|------|------|
| 1 | ASR 未接入 pipeline | 语音中的关键信息（如"现在称量 2.5 克"）完全丢失 |
| 2 | 前端无子实验展示 | 后端已能拆分多段实验，但用户看不到 |
| 3 | 前端无处理进度展示 | 3-15 分钟分析期间用户只能盲等 |

### MEDIUM: 体验缺失

| # | 问题 | 影响 |
|---|------|------|
| 4 | 无 timing API 端点 | processing_timing.json 已生成但前端无法读取 |
| 5 | 无 YOLO 检测回放 | 用户无法可视化验证检测结果 |
| 6 | 无导出/下载 | 无法打包 clip+keyframes+报告一键下载 |
| 7 | 处理完成无通知 | 长视频分析完用户不知道 |

### LOW: 架构限制

| # | 问题 | 影响 |
|---|------|------|
| 8 | 无多租户隔离 | 目前单用户可用，多用户需扩展 |
| 9 | 飞书通知已有模块未接入 | 已有 notifier 但未在 pipeline 完成时调用 |

---

## 三、修复方案

### 任务 1: ASR 语音转写接入 pipeline（P0）

**现状**：`src/labsopguard/asr.py` 完整实现了 Qwen ASR（qwen3-asr-flash），但 `process()` 从未调用。

**方案**：

```python
# service.py process() 中 Stage 2 之前插入:
with timer.measure("audio_transcription"):
    transcript = self._transcribe_audio(experiment)
    experiment.metadata["transcript_segments"] = transcript

def _transcribe_audio(self, experiment):
    """Extract audio → Qwen ASR → transcript segments."""
    from labsopguard.asr import transcribe_video_audio
    video_paths = [...]  # 同 _detect_experiment_boundaries 中的逻辑
    if not video_paths:
        return []
    return transcribe_video_audio(str(video_paths[0]), timeout=120)
```

**transcript 用途**：
- 辅助步骤推理（"现在开始称量" → step indicator）
- 辅助实验分割（长静默 = 实验间隔）
- 写入素材索引提升语义搜索质量

**工时**：3h

---

### 任务 2: 前端子实验展示组件（P0）

**方案**：新建 `ExperimentSegments.tsx`

```tsx
// 展示逻辑:
// 1. 调用 GET /api/v1/experiments/{id}/sub-experiments
// 2. 如果 total > 0, 展示分段时间轴
// 3. 每段可点击进入对应子实验 workspace
// 4. 显示各子实验事件数和处理状态
```

**UI 模型**：
```
┌─────────────────────────────────────────────┐
│  实验分段 (自动检测到 3 段)                    │
│                                               │
│  ■■■■    ·····    ■■■■■■    ····    ■■■■   │
│  实验1            实验2             实验3    │
│  0:00-0:45        1:10-1:55        2:20-3:00│
│  6 events         8 events         5 events │
│  [查看]           [查看]           [查看]    │
└─────────────────────────────────────────────┘
```

**涉及文件**：
- `frontend/src/components/ExperimentSegments.tsx`（新建）
- `frontend/src/pages/ExperimentWorkspace.tsx`（集成）

**工时**：4h

---

### 任务 3: 前端处理进度展示（P0）

**方案**：新建 `ProcessingProgress.tsx`

```tsx
// 轮询 GET /api/v1/experiments/{id}/timing
// 展示各 Stage 进度条和耗时
// 显示队列位置（如果在排队）
```

**UI 模型**：
```
┌─────────────────────────────────────────────┐
│  处理进度                                     │
│                                               │
│  ✓ 视频提取          2.1s                    │
│  ✓ 实验分割          6.2s                    │
│  ✓ 视频理解          45.3s                   │
│  ◎ 事件检测          处理中... 60%           │
│  ○ 步骤推理          等待中                   │
│  ○ 证据关联          等待中                   │
│                                               │
│  预计剩余: ~45s                               │
└─────────────────────────────────────────────┘
```

**涉及文件**：
- `frontend/src/components/ProcessingProgress.tsx`（新建）
- `frontend/src/pages/ExperimentWorkspace.tsx`（集成）

**工时**：3h

---

### 任务 4: Timing API 端点（P1）

**方案**：

```python
@app.get("/api/v1/experiments/{experiment_id}/timing", tags=["experiments"])
async def get_experiment_timing(experiment_id: str):
    timing_path = PROJECT_ROOT / "outputs" / "experiments" / experiment_id / "artifacts" / "processing_timing.json"
    if not timing_path.exists():
        return {"status": "not_available", "stages": {}}
    return json.loads(timing_path.read_text(encoding="utf-8"))
```

**工时**：30min

---

### 任务 5: 素材打包下载（P1）

**方案**：

```python
@app.get("/api/v1/experiments/{experiment_id}/export", tags=["experiments"])
async def export_experiment_materials(experiment_id: str):
    """Package clips + keyframes + report into downloadable zip."""
    # 1. 收集 materials/events/ 下所有 clip + keyframes
    # 2. 收集 reports/ 下的 PDF/HTML
    # 3. 打包为 zip (streaming response)
    # 4. 返回 StreamingResponse(content_type="application/zip")
```

**工时**：2h

---

### 任务 6: YOLO 检测回放组件（P2）

**方案**：基于 `<video>` + `<canvas>` overlay

```tsx
// VideoDetectionPlayer.tsx
// 1. 播放原始视频
// 2. 读取 detection_frames 数据
// 3. 按当前时间匹配对应帧的检测框
// 4. Canvas 叠加绘制 bbox + label
// 5. 支持暂停/跳转/高亮特定事件时间段
```

**数据源**：`GET /api/v1/experiments/{id}/materials/search?start_time_sec=X&end_time_sec=Y`

**工时**：6h

---

### 任务 7: 处理完成通知（P2）

**方案**：在 pipeline 完成时调用已有的飞书 notifier + WebSocket 推送

```python
# _run_experiment_pipeline 完成后:
from labsopguard.notifications import send_completion_notification
send_completion_notification(
    experiment_id=experiment_id,
    event_count=len(events),
    duration_sec=timer.total_sec,
    channel="feishu",  # 或 "websocket"
)
```

**前端**：WebSocket 连接监听完成事件，自动刷新页面。

**工时**：3h

---

## 四、实施路线图

```
Week 1 (P0 集成):
├─ 任务 1: ASR 接入 pipeline                    (3h)
├─ 任务 4: Timing API 端点                      (0.5h)
├─ 任务 2: 前端子实验展示                       (4h)
└─ 任务 3: 前端处理进度                         (3h)
                                                 ── 10.5h

Week 2 (P1 体验增强):
├─ 任务 5: 素材打包下载                         (2h)
├─ 任务 7: 完成通知                             (3h)
└─ 任务 6: YOLO 检测回放（前端重活）             (6h)
                                                 ── 11h

总计: ~21.5h
```

---

## 五、任务依赖

```
任务 4 (Timing API) ← 任务 3 (进度 UI) 需要此接口
任务 1 (ASR) ← 独立，无依赖
任务 2 (子实验 UI) ← 依赖已有的 /sub-experiments API
任务 5 (导出) ← 独立
任务 6 (检测回放) ← 独立（最重前端工作）
任务 7 (通知) ← 依赖已有 notifier 模块
```

---

## 六、做完后的最终状态

| 维度 | 现在 | 做完后 |
|------|------|--------|
| 语音信息 | 完全丢失 | ASR 转写 → 辅助步骤推理+语义搜索 |
| 长视频拆分 | 后端完成，前端看不到 | 时间轴展示+子实验卡片 |
| 处理进度 | 盲等 | 各阶段实时进度+预估时间 |
| 素材导出 | 不支持 | 一键下载 zip（clip+帧+报告） |
| 检测验证 | 只能看静态图 | 视频回放+bbox 叠加 |
| 完成通知 | 无 | 飞书/WebSocket 推送 |
| Timing 数据 | 文件存在但无API | 端点+可视化 |
