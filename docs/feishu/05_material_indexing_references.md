# 素材索引与关键引用检索层

## 一、摘要

| 项目 | 内容 |
|---|---|
| 状态 | 已支持 SQLite、FTS5、条件过滤，并可选接入真实文本 embedding API |
| 所属链路 | 多源预处理与时间对齐的下游检索层 |
| 一句话结论 | 本文定义素材索引层如何依赖 `material_stream`、`key_clips`、`detected_changes` 构建可查询的 `material_index.sqlite`。 |

`05` 是下游检索层，不负责生成素材。它依赖 `07` 的统一素材主线、`04` 的关键片段和 `06` 的物理事件。

## 二、本文目标与边界

本文聚焦“素材可索引化整理和关键素材引用保存”。目标是让下游能按物体、动作、时间段、机位、clip 文件、全文和轻量 embedding 分数检索素材。

| 覆盖内容 | 不覆盖内容 |
|---|---|
| SQLite 索引表 | 前端检索 UI |
| FTS5 全文检索 | 正式语义向量数据库 |
| clip/frame/payload 引用 | 跨实验全局素材库 |
| `materials/search` API | 素材生成逻辑 |

`embedding_text` 默认使用 Qwen/DashScope `text-embedding-v4`；未配置 `DASHSCOPE_API_KEY` 时才回退轻量 hash embedding。

## 三、核心输入

| 输入 | 来源 | 用途 |
|---|---|---|
| `material_stream.json` | `07` 统一素材主线 | 索引主数据 |
| `preprocessing.json.key_clips` | `04` 关键片段生成 | 补充 `clip_file_path/file_exists` |
| `preprocessing.json.detected_changes` | `06` 物理事件语义层 | 补充 `event_types` |
| `experiment_id` | 实验记录 | 查询隔离和追溯 |

## 四、处理流程

```text
material_stream.json
  + preprocessing.json.key_clips
  + preprocessing.json.detected_changes
  -> MaterialRetrievalIndex.index_payloads()
  -> material_index.sqlite
  -> /materials/search
```

索引构建逻辑：

| 步骤 | 说明 |
|---|---|
| 读取素材流 | 每个素材项形成一条索引记录 |
| 合并 clip | 通过 `clip_id` 补充实体 clip 路径 |
| 合并事件 | 通过 `material_item_id` 补充事件类型 |
| 构造文本 blob | 汇总场景、对话、物体、动作、事件 |
| 写入 SQLite/FTS | 支持结构化查询和全文检索 |

## 五、关键接口与数据结构

### 1. 素材查询参数模型

```python
@dataclass
class MaterialQuery:
    objects: Optional[List[str]] = None
    actions: Optional[List[str]] = None
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    camera_id: Optional[str] = None
    stream_id: Optional[str] = None
    has_clip: Optional[bool] = None
    clip_exists: Optional[bool] = None
    text: Optional[str] = None
    embedding_text: Optional[str] = None
    limit: int = 50
```

### 2. SQLite + FTS5 索引表

```python
self.conn.execute(
    """
    CREATE TABLE IF NOT EXISTS material_items (
        item_id TEXT PRIMARY KEY,
        experiment_id TEXT,
        timestamp_sec REAL,
        camera_id TEXT,
        stream_id TEXT,
        frame_path TEXT,
        clip_id TEXT,
        clip_file_path TEXT,
        clip_exists INTEGER DEFAULT 0,
        object_labels_json TEXT,
        actions_json TEXT,
        event_types_json TEXT,
        text_blob TEXT,
        embedding_json TEXT,
        payload_json TEXT
    )
    """
)

self.conn.execute(
    "CREATE VIRTUAL TABLE IF NOT EXISTS material_items_fts USING fts5(item_id UNINDEXED, text_blob)"
)
```

### 3. 正式检索 API 调用

```text
GET /api/v1/experiments/{experiment_id}/materials/search
  ?objects=pipette,tube
  &actions=transfer
  &start_time_sec=30
  &end_time_sec=90
  &camera_id=cam_front
  &clip_exists=true
  &text=liquid
  &embedding_text=liquid transfer
```

## 六、测试和文件清单

### 文件清单

| 文件/字段 | 内容 |
|---|---|
| `material_index.sqlite` | 素材索引数据库 |
| `material_items` | 主索引表，保存时间、机位、对象、动作、clip 引用 |
| `material_items_fts` | 全文索引表 |
| `clip_file_path` | 关键 clip 文件路径 |
| `frame_path` | 关键帧路径 |
| `payload` | 原始素材项 JSON payload |
| `material_index_health` | clip 引用、断链数量、FTS 和 embedding 模式 |

样例返回：

```json
{
  "experiment_id": "exp_001",
  "total": 1,
  "items": [
    {
      "item_id": "item_1",
      "timestamp_sec": 42.0,
      "camera_id": "cam_front",
      "stream_id": "stream_0",
      "frame_path": "frames/frame_0042.jpg",
      "clip_id": "stream_0:clip:3",
      "clip_file_path": "clips/stream_0_clip_3.mp4",
      "clip_exists": 1,
      "object_labels": ["pipette", "tube"],
      "actions": ["transfer"],
      "event_types": ["hand_contact_geometry", "liquid_level_change"],
      "embedding_score": 0.43
    }
  ]
}
```

### 测试点

| 测试项 | 测试方法 |
|---|---|
| 索引文件存在 | `outputs/experiments/{id}/material_index.sqlite` 存在 |
| 索引可自动重建 | 索引缺失时查询 API 可从 JSON 产物重建 |
| 可按物体查 | `objects=pipette,tube` 返回匹配素材 |
| 可按动作查 | `actions=transfer` 返回匹配素材 |
| 可按时间查 | `start_time_sec/end_time_sec` 有效过滤 |
| 可按机位查 | `camera_id=cam_front` 有效过滤 |
| 可按 clip 查 | `clip_exists=true` 只返回有实体 clip 的素材 |
| 可全文查 | `text=liquid` 命中描述、对话或事件 |
| 可检查索引健康 | `/materials/health` 返回断链 clip 引用 |
| 可语义查 | `embedding_text=liquid transfer` 返回 embedding 相似度排序 |

### 验证命令

```powershell
$expId = "<experiment_id>"
Test-Path "outputs/experiments/$expId/material_index.sqlite"

Invoke-RestMethod "http://localhost:8000/api/v1/experiments/$expId/materials/search?objects=pipette&actions=transfer&clip_exists=true" |
  ConvertTo-Json -Depth 8
```

## 七、当前限制与风险

| 限制 | 风险 |
|---|---|
| embedding API 依赖 Qwen/DashScope 配置 | 未配置 `DASHSCOPE_API_KEY` 时自动回退 hash embedding |
| 缺少增量索引 daemon | 当前主要在实验输出阶段构建 |
| 缺少 UI 检索页面 | API 已有，前端展示仍需补 |
| clip 文件生命周期未统一管理 | 已有断链检查，但清理策略仍需统一 |

## 八、后续演进

| 优先级 | 事项 | 目标 |
|---|---|---|
| P0 | 固化查询 API 参数说明 | 方便前端和飞书机器人调用 |
| P1 | 增加向量模型评测集 | 量化语义检索效果 |
| P1 | 增加增量索引机制 | 支持边录边索引 |
| P1 | 增加索引健康告警 | 检测片段引用失效后主动通知 |
| P2 | 做素材检索 UI | 支持人工按机位、时间、动作快速复查 |

## 九、源码定位

| 代码位置 | 作用 |
|---|---|
| `src/labsopguard/retrieval.py` | `MaterialRetrievalIndex`，SQLite/FTS/embedding 索引实现 |
| `src/labsopguard/retrieval.py` | `MaterialQuery`，查询参数模型 |
| `src/experiment/service.py` | `save_outputs()`，实验输出阶段构建 `material_index.sqlite` |
| `backend/main.py` | `search_experiment_materials()`，正式检索 API |
| `backend/main.py` | `_ensure_material_index()`，索引缺失时从 JSON 自动重建 |

与其他文档的关系：上游依赖 `04`、`06`、`07`；本文件只描述索引和检索层。
