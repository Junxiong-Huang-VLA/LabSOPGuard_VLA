# LabSOPGuard 项目不足分析与改进方案

> 日期：2026-05-11 | 基于全面代码审计

---

## 一、问题总览

| 类别 | 问题数 | 最高风险 |
|------|--------|---------|
| 可靠性 & 容错 | 5 | HIGH |
| 并发 & 数据安全 | 3 | HIGH |
| 测试覆盖 | 2 | HIGH |
| 实时能力 | 2 | MEDIUM |
| 可观测性 | 4 | MEDIUM |
| 跨实验检索 | 2 | MEDIUM |
| 视频编码 | 1 | LOW |
| 前端体验 | 3 | LOW |

---

## 二、HIGH 风险问题

### 问题 1: VLM (Qwen API) 无重试/降级/熔断

**现状**: Qwen-VL API 调用无 retry、无 timeout 控制、无 rate-limit 检测。API 一旦异常（超时/限流/宕机），整个分析流程直接失败。

**影响**: 单次 API 抖动导致 3-4 分钟的分析全部作废。

**改进方案**:

```
优先级: P0
预估工时: 4h

实施内容:
1. 指数退避重试（max_retries=3, backoff_factor=2）
2. 请求超时设置（connect=10s, read=60s）
3. Rate-limit 检测（HTTP 429 → 自动等待 Retry-After）
4. 熔断器模式（连续 5 次失败 → 临时降级 30s）
5. 降级行为：VLM 不可用时仅依赖 YOLO 检测结果,
   跳过语义描述但不阻塞 pipeline
6. VLM 可用性状态暴露到 /diagnostics

涉及文件:
- src/labsopguard/video_analysis.py (_run_vlm 方法)
- src/experiment/vlm_client.py
- 新建: src/labsopguard/resilience.py (重试+熔断通用组件)
```

---

### 问题 2: SQLite 并发写入无锁保护

**现状**: `material_index.sqlite` 无 WAL 模式、无事务隔离设置。两个实验同时处理时可能并发写入同一 SQLite 文件导致数据损坏。

**影响**: 多用户同时上传分析 → 索引数据损坏 → 素材丢失。

**改进方案**:

```
优先级: P0
预估工时: 2h

实施内容:
1. 启用 SQLite WAL 模式: PRAGMA journal_mode=WAL
2. 设置 busy_timeout: PRAGMA busy_timeout=5000
3. 所有写操作包裹在 BEGIN IMMEDIATE 事务中
4. 每个实验使用独立 SQLite 文件（已是现状，确认无跨实验共享写入）
5. 添加文件锁保护发布操作（published_materials.json 的写入）

涉及文件:
- src/labsopguard/event_preprocessing/material_index_writer.py
- src/labsopguard/retrieval.py
```

---

### 问题 3: 异常静默吞没（15+ 处）

**现状**: 代码中有 15+ 处 `except Exception: pass` 或 `except Exception: fallback`，关键错误被完全吞没，无日志无告警。

**典型案例**:
- `detectors.py`: YOLO 权重加载失败 → 静默返回 None → 后续检测全空
- `material_maintenance.py`: 素材操作失败 → 静默跳过 → 数据不一致
- `professional_report.py`: 报告生成失败 → 静默返回空结果

**改进方案**:

```
优先级: P0
预估工时: 3h

实施内容:
1. 全局审查所有 except Exception/except BaseException
2. 替换为:
   - 具体异常类型（OSError, ValueError, TimeoutError）
   - 添加 logger.warning/error 日志
   - 关键路径上改为 raise（快速失败）
3. 分类处理:
   - YOLO 加载失败: logger.error + 暴露到 /diagnostics
   - VLM 调用失败: logger.warning + 降级标记
   - 文件 I/O 失败: logger.error + 返回明确错误状态
4. 添加全局 unhandled exception handler 到 FastAPI

涉及文件:
- src/labsopguard/detectors.py (5处)
- src/labsopguard/material_maintenance.py (5处)
- src/labsopguard/professional_report.py (4处)
- src/labsopguard/event_preprocessing/engine.py (2处)
- backend/main.py (全局异常处理器)
```

---

### 问题 4: 核心模块测试覆盖不足（<30%）

**现状**: `src/labsopguard/event_preprocessing/` 下核心模块（event_proposal, event_segmentation, tracking, action_resolution, state_resolution）几乎无单元测试。当前 213 个测试主要覆盖外围逻辑。

**影响**: 事件检测核心逻辑改动无法验证正确性，回归风险高。

**改进方案**:

```
优先级: P1
预估工时: 12h

实施内容:
1. EventProposalBuilder 测试:
   - 5 类事件各构造最小测试用例
   - 边界条件: 空帧、单帧、重叠事件
   - 阈值敏感性测试

2. EventSegmenter 测试:
   - 事件合并逻辑
   - pre_roll/post_roll 边界
   - max_events 截断

3. TrackStreamBuilder + MultiObjectTracker 测试:
   - 轨迹生成基本验证
   - 片段恢复逻辑
   - ID 切换场景

4. StepBridgeEngine 测试:
   - 事件-步骤匹配评分
   - 升级/降级决策
   - 乱序检测

5. 集成测试:
   - 用固体称量实验视频做 Golden Test
   - 固定输入 → 验证输出事件不偏移

新建文件:
- tests/test_event_proposal_builder.py
- tests/test_event_segmenter.py
- tests/test_track_stream_builder.py
- tests/test_step_bridge_engine.py
- tests/test_golden_experiment.py
```

---

### 问题 5: 并发处理无限制

**现状**: 无上传并发控制。多个用户同时上传大视频时：
- GPU 显存耗尽（多个 YOLO 推理并行）
- Qwen API 同时被调用，超出 rate-limit
- 磁盘 I/O 争抢导致 clip 编码极慢

**改进方案**:

```
优先级: P1
预估工时: 4h

实施内容:
1. 添加处理队列（asyncio.Semaphore 或 Redis Queue）
   - 最大并行处理数: 1（GPU 独占）
   - 排队等待的任务返回 "queued" 状态
2. 前端展示队列位置和预估等待时间
3. GPU 显存保护: 处理前检查 torch.cuda.mem_get_info()
4. Qwen API 调用加 per-second 限流（如 2 calls/sec）

涉及文件:
- backend/main.py (任务调度)
- src/experiment/service.py (添加信号量)
- frontend-app/src/ (队列状态展示)
```

---

## 三、MEDIUM 风险问题

### 问题 6: 无实时流处理能力

**现状**: 所有分析都是"录制完 → 上传 → 批处理"。无法对实时视频流做在线事件检测。

**改进方案**:

```
优先级: P2
预估工时: 20h

实施内容:
1. RTSP/USB 视频源接入层（已有 stream_buffer.py 雏形）
2. 滑动窗口在线检测:
   - 每 N 秒取一帧跑 YOLO
   - 检测结果推入 ring buffer
   - 事件提案实时生成
3. WebSocket 推送事件到前端
4. 在线 PPE 违规告警（已有 alert_rules 配置）
5. 实时 clip 回溯截取

架构:
  RTSP/USB → FrameGrabber → RingBuffer(300s)
                                ↓ (2fps)
                          OnlineDetector → EventStream → WebSocket → 前端
                                ↓ (触发时)
                          ClipBackfill → 保存 clip
```

---

### 问题 7: 无跨实验素材检索

**现状**: 每个实验一个独立 SQLite，`MaterialRetrievalIndex.query()` 无法跨实验搜索。用户无法问"所有实验中出现过移液操作的片段"。

**改进方案**:

```
优先级: P2
预估工时: 8h

实施内容:
1. 全局素材索引层:
   - 新建 outputs/global_material_index.sqlite
   - 每个实验处理完成后自动同步到全局索引
   - 包含 experiment_id 字段用于过滤
2. API 端点: GET /api/v1/materials/search?q=...&experiments=all
3. 跨实验 embedding 检索:
   - 全局 embedding 存储（或对接向量数据库如 Qdrant）
   - 支持"找到与这个操作最相似的所有片段"
4. 素材标签体系:
   - 统一 taxonomy（已有 material_taxonomy.py）
   - 跨实验标签聚合统计

涉及文件:
- 新建: src/labsopguard/global_index.py
- 修改: src/labsopguard/retrieval.py (添加 experiment_id filter)
- 修改: backend/main.py (添加全局搜索 API)
```

---

### 问题 8: 可观测性不足

**现状**: 无结构化日志、无 APM、无 pipeline 各阶段耗时追踪。生产环境出问题无法定位。

**改进方案**:

```
优先级: P2
预估工时: 6h

实施内容:
1. 结构化日志:
   - 引入 python-json-logger
   - 每个请求带 request_id / experiment_id
   - 关键阶段输出 JSON 格式日志

2. Pipeline 耗时追踪:
   - 在 ExperimentService.process() 每个 Stage 前后记录时间
   - 写入 {experiment}/processing_timing.json
   - 示例: {"ingestion": 2.1, "video_understanding": 45.3, "events": 53.2, ...}

3. Prometheus 指标（可选，已有 /metrics 占位）:
   - pipeline_processing_seconds (histogram)
   - yolo_frames_processed_total (counter)
   - vlm_api_latency_seconds (histogram)
   - vlm_api_errors_total (counter)
   - active_experiments_processing (gauge)

4. 健康检查端点改造:
   - GET /health → 简单 200（给负载均衡器用）
   - GET /diagnostics → 详细状态（现有）
   - 增加 VLM 可达性 + 最近处理成功率

涉及文件:
- backend/main.py (日志配置 + /health)
- src/experiment/service.py (耗时追踪)
- 新建: src/labsopguard/observability.py
```

---

### 问题 9: 视频编码不稳定

**现状**: clip 写入使用 OpenCV VideoWriter，在 Windows 上 OpenH264 版本不匹配导致编码失败。虽有 ffmpeg fallback，但频繁报错影响日志质量和偶发 clip 缺失。

**改进方案**:

```
优先级: P2
预估工时: 2h

实施内容:
1. clip 编码改为直接调用 ffmpeg（跳过 OpenCV VideoWriter）:
   ffmpeg -i input.mp4 -ss {start} -to {end} -c:v libx264 -preset fast clip.mp4
2. 好处:
   - 避免 OpenH264 兼容性问题
   - 编码质量更好（libx264 vs mp4v）
   - 可选硬件加速（NVENC on RTX 3060）
3. 降级: ffmpeg 不可用时 fallback 到 cv2.VideoWriter + mp4v

涉及文件:
- src/labsopguard/event_preprocessing/key_material_extraction.py
```

---

### 问题 10: 前端 API 调用无全局错误处理

**现状**: 各组件独立 catch 错误，无统一的 API 错误拦截器、无 retry UI、超时无提示。

**改进方案**:

```
优先级: P3
预估工时: 4h

实施内容:
1. axios interceptor 全局错误处理:
   - 401 → 跳转登录
   - 429 → Toast "请求过于频繁"
   - 500 → Toast "服务器异常" + 自动重试按钮
   - timeout → Toast "请求超时" + 重试
2. 处理进度条组件:
   - 实验分析中显示各阶段进度
   - 预估剩余时间
3. 离线/网络断开检测:
   - navigator.onLine 监听
   - 断网时显示全局 banner

涉及文件:
- frontend-app/src/api.ts (interceptor)
- frontend-app/src/components/Toast.tsx (新建)
- frontend-app/src/components/ProcessingProgress.tsx (新建)
```

---

## 四、LOW 风险问题

### 问题 11: 无 API 限流

**现状**: 无 per-user/per-IP 请求速率限制。恶意或异常客户端可无限制调用。

**方案**: 引入 `slowapi`（FastAPI 限流中间件），设置合理上限（如 100 req/min/IP）。

---

### 问题 12: 前端无国际化

**现状**: UI 中中英文混合硬编码。

**方案**: 短期无需处理（用户群体固定为中文用户），长期可引入 i18next。

---

## 五、实施优先级路线图

```
Phase A: 可靠性加固（1-2周）
├─ P0: VLM 重试/降级/熔断         → 4h
├─ P0: SQLite WAL + busy_timeout   → 2h
├─ P0: 异常处理修复（15处）        → 3h
├─ P1: 并发处理队列               → 4h
└─ P1: 视频编码改 ffmpeg          → 2h
                                   ──── 15h

Phase B: 质量保障（2-3周）
├─ P1: 核心模块测试补全           → 12h
├─ P2: 可观测性 + 结构化日志      → 6h
└─ P2: 前端错误处理              → 4h
                                   ──── 22h

Phase C: 能力扩展（3-4周）
├─ P2: 跨实验素材检索             → 8h
├─ P2: 实时流处理框架             → 20h
└─ P3: API 限流                   → 2h
                                   ──── 30h
```

---

## 六、改进后目标状态

| 维度 | 当前 | 目标 |
|------|------|------|
| VLM 可用性 | 无容错，单点故障 | 3次重试 + 熔断 + 降级 |
| 并发安全 | 无保护 | 队列化 + WAL + 信号量 |
| 测试覆盖 | ~30% | >60%（核心路径 100%） |
| 错误可见性 | 15处静默吞没 | 全部有日志 + 关键路径快速失败 |
| 处理追踪 | 无 | 每阶段计时 + JSON 结构化日志 |
| 检索范围 | 单实验 | 支持跨实验语义搜索 |
| 实时能力 | 仅批处理 | 在线事件流 + WebSocket 推送 |
| clip 编码 | 不稳定（OpenH264 报错） | ffmpeg 直接编码，可选 NVENC 硬件加速 |

---

## 七、建议立即执行（今天就能做）

1. **SQLite WAL 模式** — 改 2 行代码，防并发损坏
2. **VLM 超时设置** — 加一个 `timeout=60` 参数
3. **clip 编码改 ffmpeg** — 消除日志中的 OpenH264 报错
