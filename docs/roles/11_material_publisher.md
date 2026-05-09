# 角色十一：素材发布工程师（Material Publisher）

## 职责定位

负责实验素材（clip、keyframe、material_index）的生成、命名、归档与语义搜索。是从原始事件到可用素材的最后一公里。

## 核心工作内容

### 主要代码区域
```
src/labsopguard/material_publishing/
├── publisher.py          # SemanticMaterialPublisher 主入口
├── semantic_enhancer.py  # Qwen VLM display_name 增强
├── archive_planner.py    # ArchivePlanner 归档策略
├── naming.py             # 素材命名规则
├── uploaders.py          # 素材上传（本地/OSS）
└── upload_manifest.py    # 上传清单管理
```

### 素材生成链路
```
EventProposalBuilder（物理事件）
  → KeyMaterialExtractor（clip + keyframe）
    → SemanticMaterialPublisher
        ├── semantic_enhancer（display_name）
        ├── archive_planner（归档路径）
        └── material_index_writer（写 SQLite）
```

### 素材命名规则
`display_name` 优先级：
1. `event.display_name`（格式：`{实验名}_{事件类型}_{涉及物体}`）
2. Qwen live 增强（`MATERIAL_DISPLAY_NAME_QWEN_ENABLED=true`）
3. rule_based 降级

**禁止**使用 `evidence_summary` 作为 display_name（该字段含 grade=/score= 调试信息）。

### 素材目录结构
```
outputs/experiments/<id>/
├── materials/
│   ├── events/         # 事件 clip（*.mp4）
│   ├── keyframes/      # 关键帧（*.jpg）
│   └── index/          # 素材索引
├── material_stream.json
└── material_index.sqlite
```

### 手动重建素材
当素材目录损坏或缺失时：
```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/<id>/materials/publish
```
auto-trigger 条件：`materials/events/` 不存在**或**为空目录。

### 素材搜索
```bash
curl "http://127.0.0.1:8000/api/v1/experiments/<id>/materials/search?q=移液操作"
```
搜索走 embedding 向量检索（`embeddings.py` + SQLite FTS）。

### 重建全量素材索引
```bash
python tools/rebuild_material_indexes.py
```

### 关注指标
- clip 生成成功率 100%
- display_name 语义质量（无调试文字泄露）
- 素材搜索召回率（人工测试关键词）

## 禁止事项
- 禁止在 `display_name` 中包含 `grade=`、`score=`、`conf=` 等调试字段
- 禁止直接修改 `material_index.sqlite`（通过 API 操作）
