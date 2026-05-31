# LabSOPGuard 全流程 Pipeline 完整技术文档

> 版本：v1.0 | 日期：2026-05-11 | 项目路径：`D:\LabCapability\LabSOPGuard`

---

## 目录

1. [系统概述](#1-系统概述)
2. [硬件与环境配置](#2-硬件与环境配置)
3. [全流程 Pipeline 架构](#3-全流程-pipeline-架构)
4. [三层漏斗预分割架构（性能优化）](#4-三层漏斗预分割架构性能优化)
5. [各阶段详细流程](#5-各阶段详细流程)
6. [YOLO 目标检测层](#6-yolo-目标检测层)
7. [关键帧与关键片段提取机制](#7-关键帧与关键片段提取机制)
8. [数据库与索引架构](#8-数据库与索引架构)
9. [素材发布与归档机制](#9-素材发布与归档机制)
10. [文件落盘结构](#10-文件落盘结构)
11. [性能基准与优化成果](#11-性能基准与优化成果)
12. [配置参考](#12-配置参考)
13. [API 入口与触发方式](#13-api-入口与触发方式)
14. [关键代码文件索引](#14-关键代码文件索引)

---

## 1. 系统概述

LabSOPGuard 是一个实验室视频态势感知分析系统，核心能力是：

- 从连续实验视频中**自动识别关键操作事件**（如手套接触称量纸、物体移动、面板操作等）
- 为每个事件**提取关键帧和关键片段（clip）**
- 将素材**结构化索引**，支持多维度检索（时间、物体、动作、语义）
- 生成**专业实验报告**

### 核心处理链路

```
上传视频 → 预分割(Layer 0) → YOLO检测(Layer 1) → 动作分类(Layer 2) → VLM语义(Layer 3)
         → 事件提案 → 时序分割 → clip/关键帧提取 → 索引入库 → 报告生成
```

### 典型处理时间

| 视频规模 | 处理时间 |
|---------|---------|
| 3 分钟单流 | ~30s |
| 186s 双流（固体称量实验） | ~3-4 分钟 |
| 30 分钟单流 | ~5-8 分钟（预估） |
| 3 小时单流 | ~15-20 分钟（预估） |

---

## 2. 硬件与环境配置

### 硬件

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM) |
| CUDA Driver | 591.44 |
| CUDA Version | 13.1 |

### 软件环境

| 组件 | 版本 |
|------|------|
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| Ultralytics (YOLO) | 8.4.48 |
| FastAPI | 最新 |
| Uvicorn | 最新 |
| SQLite | 内置 |
| Node.js (前端) | Vite 5.4.21 |

### Conda 环境

```
环境名称: LabSOPGuard
路径: C:\Users\Xx7\.conda\envs\LabSOPGuard
Python: C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe
```

### 启动命令

```powershell
cd D:\LabCapability\LabSOPGuard
.\scripts\start_full_stack.ps1 -SkipRedis -PythonExe "C:\Users\Xx7\.conda\envs\LabSOPGuard\python.exe"
```

### 服务端口

| 服务 | 端口 | 地址 |
|------|------|------|
| FastAPI 后端 | 8000 | http://127.0.0.1:8000 |
| Vite 前端 | 5173 | http://127.0.0.1:5173 |
| API 文档 | 8000 | http://127.0.0.1:8000/docs |

---

## 3. 全流程 Pipeline 架构

### 编排入口

**文件**: `src/experiment/service.py` → `ExperimentService.process()`

**触发方式**: `POST /api/v1/experiments/{id}/process`

### 7 阶段处理流程

```
ExperimentService.process()
│
├─ Stage 1: INGESTION（视频帧提取）
│     └─ VideoFrameExtractor.extract_frames()
│     └─ 对每个视频流提取帧，保存到临时目录
│
├─ Stage 2: VIDEO_UNDERSTANDING（视频理解）
│     └─ VideoAnalysisPipeline.analyze_video()
│           ├─ YOLO 检测 (_run_yolo / _run_yolo_batch)
│           ├─ VLM 帧分析 (_run_vlm) → Qwen-VL API
│           └─ PPE 融合 (_fuse_ppe)
│     └─ _run_multimodal_semantic_sync() → 多模态语义同步
│
├─ Stage 2.5: MATERIAL_STREAM_GENERATION（多模态物料流生成）
│     └─ _generate_material_stream()
│     └─ 为每帧生成 MultimodalMaterialStreamItem
│
├─ Stage 2.6: PHYSICAL_EVENTS_GENERATION（物理事件生成）
│     └─ EventPreprocessingEngine.run()
│           ├─ ActivityPreSegmenter.segment() ← Layer 0 预分割
│           ├─ DetectionFrameStreamBuilder.build() → 检测帧 + tracklets
│           ├─ TrackStreamBuilder.build() → 多目标跟踪
│           ├─ TrackRelationGraphBuilder.build() → 轨迹关系图
│           ├─ EventProposalBuilder.build() → 5类事件提案
│           ├─ EventSegmenter.segment() → PhysicalEvent 对象
│           ├─ KeyMaterialExtractor.extract_assets() → clip + 关键帧
│           └─ EventMaterialIndexWriter.write_events() → 写入SQLite索引
│
├─ Stage 3: CONTEXT_INTEGRATION（上下文整合）
│     └─ _integrate_context()
│
├─ Stage 4: STEP_REASONING（步骤推理）
│     └─ StepReasoner.infer_steps_from_frames()
│
├─ Stage 5: EVIDENCE_LINKING（证据关联）
│     ├─ StepBridgeEngine.run() → 事件匹配步骤 + 升降级决策
│     ├─ _link_physical_events() → 物理事件关联步骤
│     └─ _link_material_stream() → 物料流关联 + Timeline 生成
│
└─ Stage 6: OUTPUT_GENERATION（输出生成）
      └─ 生成实验报告、更新状态
```

---

## 4. 三层漏斗预分割架构（性能优化）

### 设计理念

**问题**：传统做法是对整段视频均匀采样全量跑 YOLO + VLM，相当于用最贵的工具做粗筛。

**解决**：漏斗式逐层收窄，用最便宜的方法先粗筛。

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 0: 轻量预分割（ActivityPreSegmenter）                    │
│ 成本: ~6s / 186s视频                                         │
│ 方法: 帧差分 + HSV直方图变化（160x120 低分辨率，2fps 扫描）     │
│ 效果: 去除67.6%的静止段                                       │
├─────────────────────────────────────────────────────────────┤
│ Layer 1: YOLO 目标检测（仅对活跃帧）                           │
│ 成本: ~10-20ms/帧（GPU）                                      │
│ 输入: 活跃片段内的采样帧（减少84%帧数）                         │
│ 输出: DetectionFrame + change_score + tracking                │
├─────────────────────────────────────────────────────────────┤
│ Layer 2: 动作序列分类（SkateFormer）                           │
│ 成本: 极低（复用 Layer 1 的 pose keypoints）                   │
│ 方法: 30帧滑动窗口 skeleton 序列分类                           │
│ 输出: 动作类别标签                                            │
├─────────────────────────────────────────────────────────────┤
│ Layer 3: VLM 语义理解（仅对关键事件帧）                        │
│ 成本: ~2-5s/帧（Qwen API 调用）                               │
│ 输入: 智能选帧（物体变化帧 + 均匀覆盖帧）                      │
│ 输出: 语义描述、步骤推理                                       │
└─────────────────────────────────────────────────────────────┘
```

### Layer 0 预分割算法细节

**文件**: `src/labsopguard/event_preprocessing/activity_presegmenter.py`

**算法流程**:

1. 以 2fps 顺序读取视频帧（grab/retrieve 模式避免 seek 开销）
2. 将每帧缩放到 160x120（极低分辨率）
3. 双通道检测：
   - **帧差分**: `cv2.absdiff(prev_gray, gray).mean() / 255.0` → motion_score
   - **HSV 直方图相关性**: `1.0 - cv2.compareHist(prev_hist, hist, HISTCMP_CORREL)` → hist_score
4. 综合得分: `combined = max(motion_score, hist_score * 0.7)`
5. 滑动窗口平滑（window=5）降噪
6. 自适应阈值: `threshold = max(fixed_min, mean + 1.0 * std)`
7. 连续活跃帧合并为 ActivitySegment
8. 后处理: 最小段 3s、间隔 <5s 合并、前后 padding 2s
9. 兜底: 每 60s 强制保留至少一帧（防止极慢操作被遗漏）

**输出数据结构**:

```python
@dataclass
class ActivitySegment:
    start_sec: float          # 活跃段起始时间
    end_sec: float            # 活跃段结束时间
    peak_score: float         # 片段内最大活跃度
    avg_score: float          # 片段内平均活跃度
    trigger: str              # "motion" | "histogram" | "combined" | "short_video_bypass"
    stream_id: str            # 对应视频流 ID
```

### 检测结果缓存

**文件**: `src/labsopguard/event_preprocessing/detection_cache.py`

**缓存键计算**:
```
cache_key = sha256(video_path + file_mtime + file_size + weights_path + imgsz + conf_threshold + interval_sec)[:24]
```

**缓存文件**:
```
outputs/cache/detection/
├── {cache_key}.manifest.json    # 元数据（schema版本、帧数）
└── {cache_key}.frames.json      # 序列化的 DetectionFrame 列表
```

**失效条件**: 视频文件变更（mtime/size）、模型权重变更、配置参数变更时自动失效。

### 批量推理

**文件**: `src/labsopguard/video_analysis.py` → `_run_yolo_batch()`

- 支持 batch_size=8（默认）
- GPU 显存不足时自动降级到逐帧推理
- RTX 3060 6GB 在 imgsz=960 下实测 batch_size=4-8 可用

### 智能 VLM 帧选择

**文件**: `src/labsopguard/video_analysis.py` → `_select_vlm_frames()`

**选帧策略**（非均匀采样）:
1. 必选首帧和末帧
2. 优先选择**新物体首次出现的帧**（权重 3.0）
3. 其次选择**检测数量变化大的帧**（权重 0.5）
4. 剩余预算用均匀间隔补充覆盖
5. 低变化帧复用前一帧 VLM 结果（避免冗余 API 调用）

---

## 5. 各阶段详细流程

### Stage 1: 视频帧提取

**类**: `VideoFrameExtractor`（`src/experiment/service.py:119-181`）

- 自适应采样: `adaptive_target = max(6, min(max_frames, duration / sample_interval + 2))`
- 支持双流同步（top_view + bottom_view）
- 输出帧保存为 JPEG 临时文件

### Stage 2: 视频理解

**类**: `VideoAnalysisPipeline`（`src/labsopguard/video_analysis.py`）

- **AdaptiveFrameSampler**: 根据视频时长自适应采样密度
  - base_interval_sec=2.0, min_frames=8, max_frames=36, max_vlm_frames=18
  - 短视频(<=20s): 至少 12 帧
- **YOLO 检测**: 对所有采样帧运行目标检测
- **VLM 分析**: 仅对智能选出的帧调用 Qwen-VL API
- **PPE 融合**: 合并 VLM 和 YOLO 的个人防护装备检测结果

### Stage 2.6: 物理事件生成

**类**: `EventPreprocessingEngine`（`src/labsopguard/event_preprocessing/engine.py`）

这是最核心的事件提取阶段，子流程:

```
EventPreprocessingEngine.run()
├─ _adapt_to_video_duration()       # 长视频自适应参数
├─ ActivityPreSegmenter.segment()   # 预分割找活跃段
├─ DetectionFrameStreamBuilder.build()
│   ├─ 检测缓存查找
│   ├─ 仅在活跃段内采样帧
│   ├─ 批量 YOLO 推理
│   └─ 计算 change_score
├─ TrackStreamBuilder.build()       # 多目标跟踪
├─ TrackRelationGraphBuilder.build() # 轨迹关系（接触、包含、并行）
├─ EventProposalBuilder.build()     # 5类事件提案
├─ EventSegmenter.segment()         # 时序合并 → PhysicalEvent
├─ KeyMaterialExtractor.extract_assets() # 生成 clip + keyframes
└─ EventMaterialIndexWriter.write_events() # 写入 SQLite
```

### 五类核心物理事件

| 事件类型 | 说明 | 示例 |
|---------|------|------|
| hand_object_interaction | 手-物体交互 | 手套接触称量纸 |
| object_move | 物体位移 | 移动烧杯 |
| liquid_transfer | 液体转移 | 移液 |
| panel_operation | 面板/设备操作 | 操作天平面板 |
| container_state_change | 容器状态变化 | 打开/关闭瓶盖 |

---

## 6. YOLO 目标检测层

### 当前模型

| 参数 | 值 |
|------|---|
| 模型 | `yolo26s_autodl_8_1_1` (2026-04-18 训练) |
| 权重路径 | `outputs/training/yolo26s_autodl_8_1_1/weights/best.pt` |
| 大小 | 20MB |
| 架构 | YOLOv26s (Ultralytics) |
| 推理设备 | cuda:0 (RTX 3060) |
| 输入尺寸 | 960x960 |
| 置信度阈值 | 0.25 |
| IoU 阈值 | 0.45 |
| 最大检测数 | 50 |

### 可检测类别（22类实验室物体）

```
balance, beaker, bottle, gloved_hand, glove, gloves, goggles, hand,
lab_coat, labcoat, paper, pipette, pipette_tip, reagent_bottle,
sample_bottle, sample_bottle_blue, safety_glasses, spatula, spearhead,
tube, tube_cap, vial
```

### 标签归一化（class_registry）

```yaml
sample_container: [bottle, jar, vial, cup]
pipette: [pipette, dropper, pen]
glove: [glove, gloves, hand, gloved_hand, gloved-hand]
goggles: [goggle, goggles, glasses, safety_glasses, protective_goggles, eyewear]
lab_coat: [lab_coat, labcoat, white_coat, coat]
```

### Fallback 权重链

```
yolo26s_autodl_8_1_1 (主)
  → yolo26s_pose_lab_v4_focus_auto (fallback 1)
  → yolo26s_allphotos_e40 (fallback 2)
  → yolo26s_allphotos_aug_ms_ft (fallback 3)
  → yolo26s_lab_sync_v2 (fallback 4)
```

`strict_model: true` 时不走 fallback，主权重不存在直接报错。

### 检测平滑

```yaml
smoothing:
  enabled: true
  min_hits: 3        # 最少连续命中次数
  hold_frames: 5     # 丢失后保持帧数
  iou_threshold: 0.35 # 匹配IoU阈值
```

---

## 7. 关键帧与关键片段提取机制

### 提取入口

**文件**: `src/labsopguard/event_preprocessing/key_material_extraction.py`
**类**: `KeyMaterialExtractor`

### 提取流程

对每个 `PhysicalEvent`:

```python
def extract_assets(video_path, event, output_dir, tracked_objects):
    # 1. 切出 clip（事件时间段的视频片段）
    _write_clip(video_path, event, clip_path)
    
    # 2. 提取 3 个关键帧
    _write_keyframes(video_path, event, preview_path, keyframe_paths)
    
    # 3. 生成预览缩略图
    # preview.jpg = keyframe_02（中间帧）
    
    # 4. 写入事件元数据
    # event.json = 完整事件信息
```

### Clip 生成逻辑

- **时间范围**: `event.start_time_sec` ~ `event.end_time_sec`
- **前后扩展**: pre_roll=0.8s, post_roll=1.0s
- **编码**: OpenCV VideoWriter → H.264 MP4
- **分辨率**: 与原视频相同

### 关键帧选取策略

每个事件固定提取 **3 帧**:

| 帧 | 选取逻辑 | 对应文件 |
|----|---------|---------|
| keyframe_01 | 事件起始时间 `start_time_sec` | `keyframe_01.jpg` |
| keyframe_02 | 事件中间时间 `(start + end) / 2` | `keyframe_02.jpg`，同时作为 `preview.jpg` |
| keyframe_03 | 事件结束时间 `end_time_sec` | `keyframe_03.jpg` |

### 质量评估

每个 EventAssetPack 会被评分:

```python
quality_score: float        # 0.0 ~ 1.0
quality_grade: str          # "excellent" | "good" | "fair" | "poor"
quality_reasons: List[str]  # 如 ["keyframes_good", "preview_present", "clip_duration_ok"]
```

评分因素:
- 关键帧数量（3帧=good, 2帧=acceptable, 1帧=sparse, 0=missing）
- 预览图是否存在
- Clip 时长是否合理

### 输出结构

```
materials/events/evt_1b956aba0df87dbc/
├── clip.mp4           # 关键片段（2-4秒视频）
├── keyframe_01.jpg    # 事件起始帧
├── keyframe_02.jpg    # 事件中间帧
├── keyframe_03.jpg    # 事件结束帧
├── preview.jpg        # 缩略图（= keyframe_02）
└── event.json         # 完整事件元数据
```

---

## 8. 数据库与索引架构

### 存储引擎选择

**SQLite** — 嵌入式单文件数据库，零部署，随实验目录一起落盘。

不依赖 Redis/PostgreSQL/MongoDB 等外部服务（Redis 仅用于可选的任务队列，非必需）。

### 数据库 1: 事件素材索引

**文件**: `{experiment_dir}/material_index.sqlite`
**写入器**: `EventMaterialIndexWriter`（`src/labsopguard/event_preprocessing/material_index_writer.py`）
**写入时机**: `EventPreprocessingEngine.run()` 完成后自动写入

#### 表结构: `event_materials`

| 字段 | 类型 | 说明 |
|------|------|------|
| material_id | TEXT PK | 主键（`mat_evt_xxx`） |
| experiment_id | TEXT | 所属实验 ID |
| event_id | TEXT UNIQUE | 对应事件 ID |
| event_type | TEXT | 事件类型 |
| display_name | TEXT | 中文语义名称 |
| stable_name | TEXT | 稳定标识名（不变） |
| actor_name | TEXT | 操作者名称 |
| source_container_json | TEXT | 源容器信息 JSON |
| target_container_json | TEXT | 目标容器信息 JSON |
| source_container_class | TEXT | 源容器类别 |
| target_container_class | TEXT | 目标容器类别 |
| actor_track_id | TEXT | 操作者轨迹 ID |
| tool_track_id | TEXT | 工具轨迹 ID |
| transfer_mode | TEXT | 转移模式（pour/pipette/spatula） |
| direction_confidence | REAL | 方向置信度 |
| direction_status | TEXT | 方向状态 |
| evidence_grade | TEXT | 证据等级（high/medium/low） |
| review_status | TEXT | 审核状态 |
| time_start | REAL | 开始时间（秒） |
| time_end | REAL | 结束时间（秒） |
| duration_sec | REAL | 持续时长（秒） |
| semantic_tags | TEXT | JSON 标签数组 |
| involved_objects_json | TEXT | 涉及物体 JSON |
| clip_path | TEXT | clip 文件绝对路径 |
| preview_path | TEXT | 预览图绝对路径 |
| keyframe_count | INTEGER | 关键帧数量 |
| quality_score | REAL | 质量评分 |
| quality_grade | TEXT | 质量等级 |
| searchable_text | TEXT | 全文搜索拼接文本 |
| published_path | TEXT | 发布后的归档路径 |
| payload_json | TEXT | 完整事件 payload |
| created_at | TEXT | 创建时间 ISO |

#### 索引（11个 B-tree）

```sql
idx_event_materials_experiment       ON (experiment_id)
idx_event_materials_type             ON (event_type)
idx_event_materials_time             ON (time_start, time_end)
idx_event_materials_actor            ON (actor_name)
idx_event_materials_actor_track      ON (actor_track_id)
idx_event_materials_source_container ON (source_container_class, source_container_track_id)
idx_event_materials_target_container ON (target_container_class, target_container_track_id)
idx_event_materials_evidence_grade   ON (evidence_grade, review_status)
idx_event_materials_display          ON (display_name)
idx_event_materials_published        ON (published_path)
idx_event_materials_quality          ON (quality_grade, quality_score)
```

#### 查询接口

```python
writer.query_events(
    experiment_id="solid-weighing-dual-view-20260508-153648",
    event_type="hand_object_interaction",
    actor_name="operator",
    display_name="称量",           # 模糊匹配
    source_container_class="bottle",
    start_time_sec=10.0,
    end_time_sec=150.0,
    text="手套",                   # searchable_text LIKE 搜索
    limit=50,
)
```

---

### 数据库 2: 多模态素材检索索引

**文件**: `{experiment_dir}/key_action_index/retrieval_index.sqlite`（或同目录下）
**类**: `MaterialRetrievalIndex`（`src/labsopguard/retrieval.py`）
**写入时机**: `_link_material_stream()` 阶段

#### 表结构: `material_items`

| 字段 | 类型 | 说明 |
|------|------|------|
| item_id | TEXT PK | 素材项 ID |
| experiment_id | TEXT | 实验 ID |
| timestamp_sec | REAL | 时间戳 |
| camera_id | TEXT | 摄像头 ID |
| stream_id | TEXT | 视频流 ID |
| frame_path | TEXT | 帧图片路径 |
| clip_id | TEXT | 关联的 clip ID |
| clip_file_path | TEXT | clip 文件路径 |
| clip_exists | INTEGER | clip 是否物理存在 |
| object_labels_json | TEXT | 检测到的物体列表 |
| actions_json | TEXT | 检测到的动作列表 |
| event_types_json | TEXT | 关联事件类型 |
| text_blob | TEXT | 全文搜索内容（所有语义信息拼接） |
| embedding_json | TEXT | 1024维向量 JSON |
| payload_json | TEXT | 完整原始数据 |

#### 虚拟表: `material_items_fts`（FTS5 全文索引）

```sql
CREATE VIRTUAL TABLE material_items_fts USING fts5(item_id UNINDEXED, text_blob)
```

#### 三层检索能力

| 检索方式 | 实现 | 适用场景 |
|---------|------|---------|
| **结构化查询** | SQL WHERE 条件 | 按时间/物体/摄像头/clip状态过滤 |
| **全文搜索** | SQLite FTS5 MATCH | 关键词匹配（ASCII 友好） |
| **语义向量检索** | Qwen text-embedding-v4 + cosine similarity | 自然语言描述搜索 |

#### 语义向量检索详细

**Embedding 提供者**: `QwenDashScopeEmbeddingProvider`

```
模型: text-embedding-v4
维度: 1024
API: DashScope (阿里云)
Fallback: HashEmbeddingProvider（64维，纯本地哈希）
```

**索引写入流程**:

```python
# 1. 拼接所有语义文本
blob = text_blob(
    scene_description,        # 场景描述
    transcript_segment,       # 语音转写
    objects,                  # YOLO检测物体
    actions,                  # 检测动作
    qwen_flash.scene_summary, # VLM 场景摘要
    qwen_flash.risk_flags,   # 风险标识
    state_changes,           # 状态变化
    ...
)

# 2. 调用 Qwen API 生成 embedding
embedding = embedding_provider.embed(blob)  # → List[float] * 1024

# 3. 存入 SQLite
INSERT INTO material_items (..., embedding_json, ...) VALUES (..., json.dumps(embedding), ...)
```

**检索流程**:

```python
query = MaterialQuery(
    objects=["gloved_hand", "bottle"],  # 结构化过滤
    text="称量",                        # FTS 全文搜索
    embedding_text="操作员用手套拿起称量纸放到天平上",  # 语义搜索
    has_clip=True,                      # 要求有 clip
    limit=20,
)
results = index.query(query)
# 执行过程:
# 1. SQL WHERE + FTS5 MATCH → 候选集
# 2. 对候选集计算 embedding cosine similarity
# 3. 按相似度降序返回
```

---

## 9. 素材发布与归档机制

### 自动产出（Pipeline 完成即有）

Pipeline 跑完后 `materials/events/` 目录自动填充，`material_index.sqlite` 自动写入。**无需额外操作**即可在前端浏览。

### 可选的语义增强发布

**触发**: `POST /api/v1/experiments/{id}/materials/publish`

**类**: `SemanticMaterialPublisher`（`src/labsopguard/material_publishing/publisher.py`）

**流程**:

```
SemanticMaterialPublisher.publish()
├─ _load_events()                    # 读取 materials/events/ 下所有事件
├─ _ensure_event_assets()            # 确认资产完整性
├─ QwenVlmDisplayNameEnhancer        # 调用 Qwen-VL 看图生成更好的中文名
├─ ArchivePlanner                    # 规划正式归档目录
├─ 复制/硬链接资产到归档路径
├─ write_upload_manifest()           # 写入上传清单
├─ _write_index()                    # 更新索引
├─ _update_official_steps()          # 关联到 SOP 步骤
└─ 写入 published_materials.json     # 前端可读的素材清单
```

### published_materials.json 结构

```json
{
  "schema_version": "published_materials.v1",
  "experiment_id": "solid-weighing-dual-view-20260508-153648",
  "total": 6,
  "items": [
    {
      "material_id": "mat_evt_1b956aba0df87dbc",
      "event_type": "hand_object_interaction",
      "display_name": "固体称量双视角实验-5.8-手套接触称量纸",
      "time_start": 135.7,
      "time_end": 138.5,
      "clip_path": "...",
      "preview_path": "...",
      "keyframe_paths": ["...", "...", "..."],
      "quality_grade": "good",
      "semantic_tags": ["hand_object_interaction", "gloved_hand", "paper"]
    }
  ]
}
```

---

## 10. 文件落盘结构

### 完整实验目录树

```
outputs/experiments/{实验ID}/
│
├── raw/                              # 原始视频
│   ├── top_view.browser_h264.mp4     # 俯视/第三人称视频
│   ├── bottom_view.browser_h264.mp4  # 底视/第一人称视频
│   ├── dual_view_alignment.csv       # 双流对齐信息
│   └── *.source.json                 # 视频源元数据
│
├── materials/events/                  # 关键素材（自动生成）
│   ├── evt_1b956aba.../
│   │   ├── clip.mp4                  # 关键片段（2-4秒）
│   │   ├── keyframe_01.jpg           # 事件起始帧
│   │   ├── keyframe_02.jpg           # 事件中间帧
│   │   ├── keyframe_03.jpg           # 事件结束帧
│   │   ├── preview.jpg               # 缩略图
│   │   └── event.json                # 事件完整元数据
│   └── evt_2895f2a2.../
│       └── ...
│
├── key_action_index/                  # 关键动作索引
│   ├── metadata/
│   │   └── material_library_summary.json
│   └── clips/                         # 实验焦点 clip
│       └── experiment_focus/
│           ├── first_person_yolo_annotated.mp4
│           └── third_person_yolo_annotated.mp4
│
├── reports/                           # 专业报告
│   ├── professional_report_qwen36max.pdf
│   ├── professional_report_qwen36max.html
│   └── professional_report_manifest.json
│
├── material_index.sqlite              # 事件素材 SQLite 索引
├── published_materials.json           # 发布后的素材清单
├── presegment_result.json             # 预分割诊断（活跃段列表）
├── physical_events.json               # 所有物理事件数据
├── preprocessing.json                 # 预处理完整数据
└── analysis_overview.json             # 分析概览
```

### 全局缓存目录

```
outputs/cache/detection/               # YOLO 检测结果缓存
├── {cache_key}.manifest.json
└── {cache_key}.frames.json
```

---

## 11. 性能基准与优化成果

### 固体称量实验基准（186s 双流视频，681MB）

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| PyTorch | 2.11.0+cpu | 2.6.0+cu124 | GPU 加速 |
| YOLO 设备 | CPU | CUDA:0 (RTX 3060) | ~15x 单帧加速 |
| YOLO 帧数 | 746 帧 | 122 帧 | **-84%** |
| 预分割耗时 | 0 | 6.0s | 新增（可忽略） |
| 活跃时间 | 186.5s (全量) | 60.5s | **-67.6%** |
| VLM 调用帧数 | 18 帧（均匀） | 18 帧（智能选帧） | 更精准 |
| 事件检出 | 6 个 | 6 个 | **零丢失** |
| 事件时间/置信度 | - | 完全一致 | **零偏差** |

### 端到端耗时分解

| 阶段 | 单流耗时 | 说明 |
|------|---------|------|
| 预分割 (Layer 0) | 6.0s | 160x120, 2fps 扫描 |
| VLM 帧分析 | 4.0s | 37帧 YOLO + 18帧 VLM |
| 事件检测全流程 | 53.2s | 预分割+YOLO+tracking+clip |
| 其中 clip 编码 | ~40s | 主要瓶颈（I/O 密集） |
| 报告生成 (Qwen API) | 60-120s | 取决于网络 |
| **双流总计** | **~3-4 分钟** | |

### 缓存效果

- 首次运行: 完整处理
- 二次运行（视频未变、模型未变）: 直接读取缓存，跳过 YOLO → **~5s 完成**

---

## 12. 配置参考

### 完整 detection_runtime.yaml

```yaml
backend: ultralytics
model: outputs/training/yolo26s_autodl_8_1_1/weights/best.pt
model_fallbacks:
  - outputs/training/yolo26s_pose_lab_v4_focus_auto/weights/best.pt
  - outputs/training/yolo26s_allphotos_e40/weights/best.pt
  - outputs/training/detect/outputs/training/yolo26s_allphotos_aug_ms_ft/weights/best.pt
  - outputs/training/yolo26s_lab_sync_v2/weights/best.pt
device: auto
layer2_window: 30
layer2_backend: skateformer
enable_vlm: true
strict_model: true

class_registry:
  sample_container: [bottle, jar, vial, cup]
  pipette: [pipette, dropper, pen]
  glove: [glove, gloves, hand, gloved_hand, gloved-hand]
  goggles: [goggle, goggles, glasses, safety_glasses, protective_goggles, eyewear]
  lab_coat: [lab_coat, labcoat, white_coat, coat]

ppe:
  hold_frames: 20
  consensus_ratio: 0.45
  class_conf_thresholds:
    glove: 0.20
    goggles: 0.20
    lab_coat: 0.25

sampling:
  base_interval_sec: 2.0
  min_frames: 8
  max_frames: 36
  max_vlm_frames: 18

detection:
  confidence_threshold: 0.25
  iou_threshold: 0.45
  max_detections: 50
  imgsz: 960
  allowed_labels:
    - balance
    - beaker
    - bottle
    - gloved_hand
    - glove
    - gloves
    - goggles
    - hand
    - lab_coat
    - labcoat
    - paper
    - pipette
    - pipette_tip
    - reagent_bottle
    - sample_bottle
    - sample_bottle_blue
    - safety_glasses
    - spatula
    - spearhead
    - tube
    - tube_cap
    - vial

smoothing:
  enabled: true
  min_hits: 3
  hold_frames: 5
  iou_threshold: 0.35

presegment:
  enabled: true
  scan_fps: 2.0
  scan_resolution: [160, 120]
  motion_threshold_mode: adaptive
  motion_fixed_threshold: 0.02
  min_segment_sec: 3.0
  merge_gap_sec: 5.0
  padding_sec: 2.0
  skip_if_video_shorter_than: 30
  forced_sample_interval_sec: 60.0

cache:
  detection_cache_enabled: true
  batch_size: 8
```

### 环境变量

| 变量 | 说明 |
|------|------|
| DASHSCOPE_API_KEY | Qwen/DashScope API 密钥 |
| YOLO26_WEIGHTS_PATH | 覆盖 YOLO 权重路径 |
| DETECTOR_DEVICE | 覆盖推理设备（auto/cpu/cuda:0） |
| LABSOPGUARD_YOLO_IMGSZ | 覆盖推理尺寸 |
| LABSOPGUARD_EVENT_INTERVAL_SEC | 覆盖事件检测采样间隔 |
| LABSOPGUARD_EVENT_MAX_FRAMES | 覆盖事件检测最大帧数 |
| MATERIAL_EMBEDDING_MODEL | Embedding 模型名称 |
| MATERIAL_EMBEDDING_DIMENSION | Embedding 维度 |

---

## 13. API 入口与触发方式

### 核心 API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/experiments/{id}/process` | POST | 触发完整 pipeline |
| `/api/v1/experiments/{id}/materials/publish` | POST | 语义增强发布 |
| `/api/v1/experiments/{id}/materials/published` | GET | 获取已发布素材列表 |
| `/api/v1/experiments/{id}/analysis-overview` | GET | 分析概览 |
| `/api/v1/experiments/{id}/key-actions/results` | GET | 关键动作结果 |
| `/api/v1/diagnostics` | GET | 系统诊断（含 YOLO/VLM 状态） |

### 前端入口

- 实验 workspace: `http://127.0.0.1:5173/experiments/{id}/workspace`
- 素材库: `http://127.0.0.1:5173/experiments/{id}/materials`

---

## 14. 关键代码文件索引

### Pipeline 编排

| 文件 | 说明 |
|------|------|
| `src/experiment/service.py` | 主编排器 ExperimentService.process() |
| `src/labsopguard/video_analysis.py` | VideoAnalysisPipeline（YOLO+VLM） |
| `src/labsopguard/event_preprocessing/engine.py` | EventPreprocessingEngine |

### 预分割与优化（本次新增）

| 文件 | 说明 |
|------|------|
| `src/labsopguard/event_preprocessing/activity_presegmenter.py` | Layer 0 预分割 |
| `src/labsopguard/event_preprocessing/detection_cache.py` | 检测结果缓存 |
| `src/labsopguard/event_preprocessing/frame_detection_stream.py` | 帧流构建（集成预分割+缓存+批量推理） |

### 事件提取

| 文件 | 说明 |
|------|------|
| `src/labsopguard/event_preprocessing/event_proposal.py` | EventProposalBuilder |
| `src/labsopguard/event_preprocessing/event_segmentation.py` | EventSegmenter |
| `src/labsopguard/event_preprocessing/key_material_extraction.py` | KeyMaterialExtractor |
| `src/labsopguard/event_preprocessing/material_index_writer.py` | SQLite 索引写入 |

### 检索与索引

| 文件 | 说明 |
|------|------|
| `src/labsopguard/retrieval.py` | MaterialRetrievalIndex（向量+FTS检索） |
| `src/labsopguard/embeddings.py` | Embedding 提供者（Qwen/Hash） |

### 素材发布

| 文件 | 说明 |
|------|------|
| `src/labsopguard/material_publishing/publisher.py` | SemanticMaterialPublisher |
| `src/labsopguard/material_publishing/semantic_enhancer.py` | Qwen VLM 语义增强 |
| `src/labsopguard/material_publishing/archive_planner.py` | 归档规划 |

### 配置

| 文件 | 说明 |
|------|------|
| `configs/model/detection_runtime.yaml` | YOLO/预分割/缓存 配置 |
| `src/labsopguard/config.py` | RuntimeSettings 加载逻辑 |

### 测试

| 文件 | 说明 |
|------|------|
| `tests/test_activity_presegmenter.py` | 预分割单元测试 |
| `tests/test_detection_cache.py` | 缓存单元测试 |
| `tests/test_presegment_integration.py` | 集成测试 |
| `scripts/benchmark_pipeline.py` | 性能基准脚本 |

---

## 附录：固体称量实验检出事件示例

| # | 事件类型 | 时间段 | 名称 | 物体 | 置信度 |
|---|---------|--------|------|------|--------|
| 1 | object_move | 14.7s-18.0s | 物体移动_gloved_hand | gloved_hand | 0.76 |
| 2 | object_move | 26.7s-30.5s | 物体移动_手套+天平+称量纸... | gloved_hand+天平+称量纸+试剂瓶... | 0.80 |
| 3 | hand_object_interaction | 27.7s-30.5s | 手套接触称量纸 | gloved_hand+paper | 0.77 |
| 4 | hand_object_interaction | 135.7s-138.5s | 手套接触称量纸 | gloved_hand+paper | 0.78 |
| 5 | object_move | 135.7s-138.5s | 物体移动_gloved_hand | gloved_hand | 0.79 |
| 6 | panel_operation | 138.2s-142.0s | 面板/设备操作_gloved_hand | gloved_hand | 0.79 |

每个事件均生成：1 个 clip（2-4s MP4） + 3 个关键帧（JPG） + 1 个预览缩略图 + 完整元数据 JSON。
