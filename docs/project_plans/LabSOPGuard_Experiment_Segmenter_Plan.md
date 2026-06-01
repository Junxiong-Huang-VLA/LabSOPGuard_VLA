# 实验边界检测与自动拆分方案

> 版本：v1.0 | 日期：2026-05-11

---

## 一、问题定义

### 当前限制

用户上传一条 3 小时双视角连续视频，中间包含 2-3 段独立实验。当前系统：

- 预分割（Layer 0）只能区分"有动作 vs 无动作"
- 无法识别"实验 A 结束"和"实验 B 开始"的边界
- 所有事件混在一个实验里，无法分组
- 报告无法按独立实验分别生成

### 目标

```
上传 3h 双视角视频
    ↓
自动检测出 2-3 段实验边界
    ↓
每段独立拆分为子实验
    ↓
每个子实验独立走完整 pipeline（YOLO + 事件 + clip + 报告）
    ↓
前端展示：一次上传 → 2-3 个实验卡片
```

---

## 二、实验边界检测策略（多信号融合）

### 2.1 信号源分析

| 信号 | 可靠度 | 成本 | 说明 |
|------|--------|------|------|
| 长静止段（>3min） | 高 | 极低 | 两段实验之间通常有较长的空闲 |
| 场景物品组合变化 | 中高 | 低 | 不同实验用不同器材，YOLO 检测物体集合明显改变 |
| 视频画面大幅变化 | 中 | 极低 | 摄像头重新调整、台面清理 |
| 人员离开/返回 | 中 | 低 | 人体检测信号中断较长时间 |
| VLM 单帧确认 | 高 | 中 | 对候选边界帧问"是否是新实验开始" |

### 2.2 核心算法：三级判定

```
Level 1: 时序间隔判定（必须满足）
    活跃段 A.end → 活跃段 B.start 间隔 >= min_gap_sec（默认 180s）
    
Level 2: 场景变化确认（增强置信度）
    比较间隔前后各 5 帧的 YOLO 检测物品集合
    Jaccard 距离 > 0.4 → 物品组合明显变化 → 边界置信度 +0.3
    
Level 3: VLM 语义确认（可选，对低置信度边界）
    对候选边界后的首帧调 VLM：
    "这一帧是实验操作的开始还是中间步骤？桌面上可见哪些实验器材？"
    回答包含"新实验/开始/准备" → 确认边界
```

### 2.3 输出数据结构

```python
@dataclass
class ExperimentBoundary:
    boundary_id: str
    start_sec: float              # 本段实验开始时间
    end_sec: float                # 本段实验结束时间
    gap_before_sec: float         # 与上一段之间的空闲时长
    confidence: float             # 边界置信度 0-1
    signals: List[str]            # 触发信号: ["long_gap", "object_change", "vlm_confirmed"]
    object_summary_before: List[str]  # 上一段结尾的物品
    object_summary_after: List[str]   # 本段开头的物品

@dataclass 
class ExperimentSegmentation:
    video_duration_sec: float
    total_segments: int
    segments: List[ExperimentBoundary]
    unassigned_time_sec: float    # 未被分配到任何实验的时间（空闲段）
```

---

## 三、完整流程架构

### 3.1 改造后的处理链路

```
POST /api/v1/experiments/upload  (上传 3h 视频)
    │
    ▼
Stage 0: 预分割（已有）
    │ → 输出活跃段列表 (ActivitySegment[])
    ▼
Stage 0.5: 实验边界检测（新增）
    │ → ExperimentSegmenter.segment()
    │ → 输出: ExperimentSegmentation (N 段实验)
    ▼
    ┌─── 如果 N == 1 ───→ 正常走单实验 pipeline
    │
    └─── 如果 N >= 2 ───→ 自动拆分
              │
              ├─ 创建父实验记录 (parent_experiment)
              │
              ├─ 对每段创建子实验:
              │   ├─ sub_experiment_1 (0:00 - 0:45)
              │   ├─ sub_experiment_2 (1:10 - 1:55)
              │   └─ sub_experiment_3 (2:20 - 3:00)
              │
              └─ 每个子实验独立走完整 pipeline:
                  ├─ YOLO 检测（仅本段时间范围）
                  ├─ 事件提取 + clip
                  ├─ 步骤推理
                  └─ 报告生成
```

### 3.2 视频切分策略

**不做物理切割**（避免耗时的视频重编码），而是用时间范围过滤：

```python
# EventPreprocessingEngine 已支持时间范围:
# DetectionFrameStreamBuilder 的 frame_indices 基于时间段采样
# 只需传入 (start_sec, end_sec) 限定分析范围

engine.run(
    source_video=original_3h_video,
    time_range=(start_sec, end_sec),  # 新增参数
    ...
)
```

好处：
- 无需切出多个大文件（3h 视频可能 10GB+）
- 原始视频保持完整，子实验引用同一文件
- clip 提取时直接从原视频 seek 到对应时间点

---

## 四、配置设计

### detection_runtime.yaml 新增段

```yaml
experiment_segmentation:
  enabled: true
  min_gap_sec: 180              # 两段实验间最小空闲时长（秒）
  min_experiment_duration_sec: 60  # 最短实验时长（短于此的活跃段合并到相邻实验）
  max_experiments: 10           # 单视频最大拆分数
  object_change_threshold: 0.4  # 物品 Jaccard 距离阈值
  use_vlm_confirmation: false   # 是否对低置信度边界调 VLM 确认
  vlm_confirmation_threshold: 0.5  # 置信度低于此值时才调 VLM
  merge_short_gaps: true        # 是否合并短间隔活跃段到同一实验
  skip_if_video_shorter_than: 600  # 视频短于 10min 不做拆分
```

---

## 五、性能预估

### 3 小时双视角视频

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 预分割 (Layer 0) | ~90s | 两流各 ~45s |
| 实验边界检测 | ~5s | 基于已有预分割结果 + 少量 YOLO 采样 |
| 边界帧 YOLO 物品确认 | ~2s | 仅对 2-3 个候选边界各取 5 帧 |
| VLM 边界确认（可选） | ~10s | 仅 2-3 帧 |
| 子实验1 全流程 | ~3-4min | 与当前单实验相同 |
| 子实验2 全流程 | ~3-4min | |
| 子实验3 全流程 | ~3-4min | |
| **总计** | **~12-15min** | 3 段实验并行可降至 ~5-6min |

### 对比无拆分

| 方案 | 耗时 | 效果 |
|------|------|------|
| 无拆分（全量当单实验） | ~15-20min | 事件混杂，无法区分实验 |
| 有拆分（串行处理） | ~12-15min | 每段独立，报告清晰 |
| 有拆分（并行处理） | ~5-6min | GPU 串行 + VLM 并行 |

---

## 六、实施任务书

### Phase 1: ExperimentSegmenter 核心实现

| # | 任务 | 文件 | 验收标准 | 预估工时 |
|---|------|------|---------|---------|
| 1.1 | 定义 ExperimentBoundary / ExperimentSegmentation 数据类 | `src/labsopguard/event_preprocessing/experiment_segmenter.py`（新建） | dataclass 完整定义 + to_dict() | 20min |
| 1.2 | 实现 ExperimentSegmenter.segment() 主方法 | 同上 | 接收 ActivitySegment[] → 输出 ExperimentSegmentation | 2h |
| 1.3 | Level 1: 长间隔检测逻辑 | 同上 | 活跃段间隔 >= min_gap_sec 标记为候选边界 | 含在 1.2 |
| 1.4 | Level 2: 物品变化检测 | 同上 | 对候选边界前后各取 5 帧 YOLO，计算 Jaccard 距离 | 1h |
| 1.5 | Level 3: VLM 边界确认（可选） | 同上 | 对低置信度边界调 VLM 确认，可通过配置关闭 | 30min |
| 1.6 | 短实验段合并逻辑 | 同上 | 短于 min_experiment_duration 的段合并到相邻实验 | 30min |
| 1.7 | 添加 experiment_segmentation 配置段 | `configs/model/detection_runtime.yaml` | 新增配置块 | 10min |
| 1.8 | 单元测试 | `tests/test_experiment_segmenter.py`（新建） | 覆盖：单实验视频/多实验视频/短视频跳过/边界合并 | 1.5h |

**Phase 1 产出**：独立可运行的实验分割模块，不影响现有流程。

---

### Phase 2: 时间范围过滤集成

| # | 任务 | 文件 | 验收标准 | 预估工时 |
|---|------|------|---------|---------|
| 2.1 | DetectionFrameStreamBuilder 支持 time_range 参数 | `src/labsopguard/event_preprocessing/frame_detection_stream.py` | `build(video_path, time_range=(start, end))` 仅处理该时间范围 | 1h |
| 2.2 | EventPreprocessingEngine.run() 支持 time_range | `src/labsopguard/event_preprocessing/engine.py` | 传递 time_range 到 stream_builder | 30min |
| 2.3 | KeyMaterialExtractor 时间偏移修正 | `src/labsopguard/event_preprocessing/key_material_extraction.py` | clip 时间基于原视频绝对时间（不是子实验相对时间） | 30min |
| 2.4 | 集成测试 | `tests/test_time_range_processing.py`（新建） | 用真实视频验证时间范围过滤正确性 | 1h |

---

### Phase 3: 子实验自动创建与编排

| # | 任务 | 文件 | 验收标准 | 预估工时 |
|---|------|------|---------|---------|
| 3.1 | 父子实验数据模型 | `src/experiment/models.py` | Experiment 增加 parent_id, sub_experiments[] 字段 | 30min |
| 3.2 | 子实验自动创建逻辑 | `src/experiment/service.py` | process() 开始时检测边界，N>=2 时创建子实验 | 2h |
| 3.3 | 子实验独立 pipeline 执行 | 同上 | 每个子实验调用完整 pipeline 并传入 time_range | 1h |
| 3.4 | 父实验汇总 | 同上 | 父实验记录汇总信息（总事件数、各子实验摘要） | 30min |
| 3.5 | 输出目录结构设计 | 同上 | `outputs/experiments/{parent_id}/sub_{n}/materials/events/...` | 30min |

---

### Phase 4: API 与前端适配

| # | 任务 | 文件 | 验收标准 | 预估工时 |
|---|------|------|---------|---------|
| 4.1 | API: 获取子实验列表 | `backend/main.py` | `GET /api/v1/experiments/{id}/sub-experiments` 返回子实验列表 | 1h |
| 4.2 | API: 处理进度细化 | 同上 | 返回当前处理哪个子实验、进度百分比 | 30min |
| 4.3 | 前端: 子实验卡片展示 | `frontend/src/` | 父实验 workspace 展示子实验列表，点击进入各自 workspace | 2h |
| 4.4 | 前端: 时间轴总览 | 同上 | 父实验展示 3h 完整时间轴，标注各实验段范围 | 1.5h |

---

### Phase 5: 验收与优化

| # | 任务 | 文件 | 验收标准 | 预估工时 |
|---|------|------|---------|---------|
| 5.1 | 端到端测试：上传 3h 视频 | 手动 | 自动拆出 2-3 个实验，各自有事件 + clip + 报告 | 1h |
| 5.2 | 边界检测准确性验证 | 手动 | 人工确认边界时间点偏差 < 10s | 30min |
| 5.3 | 无拆分场景兼容性 | 自动测试 | 单实验短视频仍正常处理，无影响 | 30min |
| 5.4 | 性能优化：子实验并行 | `src/experiment/service.py` | VLM/报告生成可并行，YOLO 串行（GPU 独占） | 1h |

---

## 七、实施顺序与依赖

```
Phase 1 (核心模块，独立开发)
    │
    └──→ Phase 2 (时间范围过滤，依赖 Phase 1 输出)
            │
            └──→ Phase 3 (子实验编排，依赖 Phase 2)
                    │
                    └──→ Phase 4 (API + 前端)
                            │
                            └──→ Phase 5 (验收)
```

---

## 八、文件变更清单

| 操作 | 文件路径 | 说明 |
|------|---------|------|
| 新建 | `src/labsopguard/event_preprocessing/experiment_segmenter.py` | 实验边界检测核心 |
| 新建 | `tests/test_experiment_segmenter.py` | 单元测试 |
| 新建 | `tests/test_time_range_processing.py` | 时间范围集成测试 |
| 修改 | `src/labsopguard/event_preprocessing/frame_detection_stream.py` | 增加 time_range 支持 |
| 修改 | `src/labsopguard/event_preprocessing/engine.py` | 增加 time_range 传递 |
| 修改 | `src/experiment/service.py` | 子实验编排逻辑 |
| 修改 | `src/experiment/models.py` | 父子实验模型 |
| 修改 | `backend/main.py` | 子实验 API |
| 修改 | `configs/model/detection_runtime.yaml` | 新增 experiment_segmentation 配置 |
| 修改 | `frontend/src/` | 子实验 UI 展示 |

---

## 九、总工时预估

| Phase | 工时 | 优先级 |
|-------|------|--------|
| Phase 1: 核心分割模块 | 6.5h | P0 |
| Phase 2: 时间范围过滤 | 3h | P0 |
| Phase 3: 子实验编排 | 4.5h | P0 |
| Phase 4: API + 前端 | 5h | P1 |
| Phase 5: 验收优化 | 3h | P0 |
| **合计** | **~22h** | |

---

## 十、风险与降级

| 风险 | 影响 | 降级方案 |
|------|------|---------|
| 间隔不足 3min 的连续实验无法区分 | 相邻实验被合并 | 降低 min_gap_sec 到 60s + 启用物品变化检测 |
| 同一实验中间有长休息（如接电话 5min） | 被误拆为两个实验 | 物品 Jaccard 距离过滤（相同物品 = 同一实验） |
| 3h 视频预分割耗时长 | 用户等待 | 预分割完成后立即返回拆分预览，让用户确认后再跑全量 pipeline |
| 子实验数量过多（误拆） | 资源浪费 | max_experiments=10 硬上限 + 前端允许手动合并 |

---

## 十一、用户交互流程（前端）

```
┌─────────────────────────────────────────────────┐
│  上传视频                                         │
│  [选择文件: 3h_dual_view.mp4]  [开始分析]         │
└─────────────────────────────────────────────────┘
         │
         ▼ (30s 后)
┌─────────────────────────────────────────────────┐
│  检测到 3 段独立实验                               │
│                                                   │
│  ┌─────────────────────────────────────────┐     │
│  │ ■■■■■   ·····   ■■■■■■   ···   ■■■■■  │     │
│  │ 实验1      空闲    实验2    空闲   实验3  │     │
│  │ 0:00-0:45       1:10-1:55      2:20-3:00│     │
│  └─────────────────────────────────────────┘     │
│                                                   │
│  [确认拆分并分析]    [当作整体分析]                  │
└─────────────────────────────────────────────────┘
         │
         ▼ (确认后)
┌─────────────────────────────────────────────────┐
│  处理中... (预计 12-15 分钟)                       │
│                                                   │
│  ✓ 实验1: 固体称量 (已完成, 6 个事件)              │
│  ◎ 实验2: 移液操作 (处理中... 60%)                │
│  ○ 实验3: 加热回流 (等待中)                       │
└─────────────────────────────────────────────────┘
         │
         ▼ (完成后)
┌─────────────────────────────────────────────────┐
│  实验总览                                         │
│                                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐         │
│  │ 实验1    │  │ 实验2    │  │ 实验3    │         │
│  │ 固体称量 │  │ 移液操作 │  │ 加热回流 │         │
│  │ 6 事件   │  │ 8 事件   │  │ 5 事件   │         │
│  │ [查看]   │  │ [查看]   │  │ [查看]   │         │
│  └─────────┘  └─────────┘  └─────────┘         │
└─────────────────────────────────────────────────┘
```

---

## 十二、核心算法代码

```python
class ExperimentSegmenter:
    def segment(self, activity_segments: List[ActivitySegment], video_path: Path) -> ExperimentSegmentation:
        # 1. 计算活跃段之间的间隔
        gaps = []
        for i in range(1, len(activity_segments)):
            gap = activity_segments[i].start_sec - activity_segments[i-1].end_sec
            gaps.append((i, gap))
        
        # 2. Level 1: 标记长间隔为候选边界
        candidate_boundaries = [
            (i, gap) for i, gap in gaps 
            if gap >= self.config.min_gap_sec
        ]
        
        # 3. Level 2: 对候选边界做物品变化确认
        confirmed_boundaries = []
        for boundary_idx, gap_sec in candidate_boundaries:
            confidence = 0.5  # 基础置信度（满足长间隔）
            
            # 取边界前后各 5 帧做 YOLO
            objects_before = self._detect_objects_at(video_path, activity_segments[boundary_idx-1].end_sec)
            objects_after = self._detect_objects_at(video_path, activity_segments[boundary_idx].start_sec)
            
            jaccard = self._jaccard_distance(objects_before, objects_after)
            if jaccard > self.config.object_change_threshold:
                confidence += 0.3
                signals.append("object_change")
            
            # 4. Level 3: VLM 确认（可选）
            if confidence < self.config.vlm_confirmation_threshold and self.config.use_vlm_confirmation:
                if self._vlm_confirms_new_experiment(video_path, activity_segments[boundary_idx].start_sec):
                    confidence += 0.2
                    signals.append("vlm_confirmed")
            
            confirmed_boundaries.append(ExperimentBoundary(...))
        
        # 5. 基于确认的边界拆分为实验段
        experiments = self._split_by_boundaries(activity_segments, confirmed_boundaries)
        
        # 6. 合并过短的段
        experiments = self._merge_short_segments(experiments)
        
        return ExperimentSegmentation(segments=experiments, ...)
```
