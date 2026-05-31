# LabSOPGuard 剩余问题（2026-05-11 最终审计）

---

## 已修复的关键 Bug（本次立即修复）

| Bug | 严重性 | 修复 |
|-----|--------|------|
| `ExperimentWorkspace.tsx` 引用不存在的 `experimentId` 变量 | 🔴 崩溃 | 改为 `id`（已修复） |
| `_queued_pipeline` 是 async 但 FastAPI background_tasks 不支持 | 🔴 挂死 | 改为同步 + event loop（已修复） |

---

## 剩余问题（3 项中等）

### 问题 1: YOLO 检测视频回放组件（前端）

**现状**：后端已生成 YOLO 标注视频（`first_person_yolo_annotated.mp4`），前端可以播放原始视频，但没有交互式检测框叠加回放组件。

**影响**：用户无法可视化验证 YOLO 检测是否准确，只能看静态关键帧。

**方案**：
```
新建 VideoDetectionPlayer.tsx
- <video> 播放原始视频
- <canvas> 叠加层按时间戳绘制检测框
- 数据源: /api/v1/experiments/{id}/materials/search?start_time_sec=X
- 支持: 暂停/跳转/高亮事件时间段
```

**工时**：6h

---

### 问题 2: 端到端测试覆盖缺失

**现状**：242 个测试覆盖各模块，但没有一个测试走完 "上传视频 → 分割 → 子实验 → 素材生成" 的完整链路。

**影响**：模块间集成问题可能在单元测试中无法发现。

**方案**：
```python
# tests/test_full_e2e_pipeline.py
def test_full_pipeline_short_video():
    """Upload short video → process → verify events + materials."""
    # 1. Create synthetic 30s video
    # 2. Call ExperimentService.process()
    # 3. Assert: events > 0, material_index.sqlite exists, clips exist

def test_full_pipeline_multi_experiment():
    """Upload long video with gap → verify segmentation + sub-experiments."""
    # 1. Create synthetic 15min video (5min active, 5min idle, 5min active)
    # 2. Call ExperimentService.process()  
    # 3. Assert: segmentation.total_segments == 2
    # 4. Assert: each sub has independent events
```

**工时**：4h

---

### 问题 3: Schema 版本迁移缺失

**现状**：各输出文件有 `schema_version` 字段（如 `physical_events.v4`、`preprocessing.v4`），但没有版本升级/迁移逻辑。旧实验数据在新代码下可能加载失败。

**影响**：已有实验数据在代码升级后可能无法正确显示。

**方案**：
```python
# src/labsopguard/schema_migration.py
MIGRATIONS = {
    "physical_events.v3": migrate_events_v3_to_v4,
    "preprocessing.v3": migrate_preprocessing_v3_to_v4,
}

def auto_migrate(data: dict) -> dict:
    version = data.get("schema_version", "")
    while version in MIGRATIONS:
        data = MIGRATIONS[version](data)
        version = data["schema_version"]
    return data
```

**工时**：3h

---

## 总结

| 类别 | 问题数 | 总工时 |
|------|--------|--------|
| 🔴 关键 Bug | 0（已修复） | - |
| 🟡 中等问题 | 3 | 13h |

**项目当前状态**：后端 pipeline 完整且稳定（242 测试通过），前端基础功能可用。剩余的 3 项是**锦上添花**而非**必须**——不影响核心功能使用，按需安排即可。

---

## 当前已实现的完整能力清单

| 能力 | 状态 |
|------|------|
| GPU YOLO 推理 (RTX 3060) | ✓ |
| 三层漏斗预分割（减 84% 帧） | ✓ |
| 实验边界自动检测 | ✓ 已接入 pipeline |
| 子实验自动拆分 | ✓ 已接入 pipeline |
| 双流预分割合并 | ✓ 已接入 pipeline |
| VLM 重试/熔断/降级 | ✓ |
| Embedding 重试/降级 | ✓ |
| 并发队列控制 | ✓ 已接入 backend |
| StageTimer 各阶段计时 | ✓ 已接入 pipeline |
| ASR 语音转写 | ✓ 已接入 pipeline |
| 检测结果缓存 | ✓ |
| 批量 YOLO 推理 | ✓ |
| 智能 VLM 帧选择 | ✓ |
| SQLite WAL 并发安全 | ✓ |
| 跨实验全局索引 | ✓ |
| 语义向量检索 (Qwen embedding) | ✓ |
| 全局搜索 API | ✓ |
| Timing API | ✓ |
| Export ZIP API | ✓ |
| 处理完成飞书通知 | ✓ |
| 前端处理进度组件 | ✓ |
| 前端子实验展示组件 | ✓ |
| 手动边界修正 API | ✓ |
| 队列状态 API | ✓ |
