# LabSOPGuard — 开发约束文档

本文件是 Claude Code 的强制约束清单。每次开发前必须读取，避免反复踩同一个坑。

---

## 1. 唯一真实来源原则

### 1.1 YOLO 模型权重
- **唯一配置入口**：`configs/model/detection_runtime.yaml` 的 `model` 字段
- **禁止**在任何 Python 文件中硬编码权重路径
- 解析逻辑在 `src/labsopguard/detectors.py` → `resolve_yolo26_weights_path()`，读取顺序：
  1. 函数参数 `override_path`
  2. 环境变量 `YOLO26_WEIGHTS_PATH` 或 `LABSOPGUARD_YOLO_MODEL`
  3. yaml `model` 字段（当前：`outputs/training/yolo26s_autodl_8_1_1/weights/best.pt`）
  4. yaml `model_fallbacks` 列表（`yolo26s.pt` → `yolo26n.pt`）
- 新增训练权重时：只改 yaml 的 `model` 字段，其他地方不动

### 1.2 API 调用
- **唯一 API 提供商**：Qwen / DashScope（阿里云）
- **禁止**引入 `anthropic`、`openai`（非兼容模式）、`cohere` 等其他厂商 SDK
- **禁止**在代码中硬编码 API Key
- 所有 Key 来源：`.env` 文件（`DASHSCOPE_API_KEY`）→ 环境变量
- 文本生成接口：`openai.OpenAI(api_key=..., base_url=DASHSCOPE_BASE_URL)` + `chat.completions.create`
- 多模态接口：`dashscope.MultiModalConversation.call`
- 语义命名：`MATERIAL_DISPLAY_NAME_QWEN_ENABLED=true`，模型 `MATERIAL_DISPLAY_NAME_QWEN_MODEL=qwen3.6-flash`

### 1.3 .env 文件
- 唯一路径：`D:/LabEmbodiedVLA/LabSOPGuard/.env`
- **禁止**在 `lab_preprocessing/` 或其他子目录单独维护 `.env`
- 所有服务启动前必须 `load_dotenv('.env')`

---

## 2. 实验处理完整链路

新建或重新处理实验时，以下链路**必须自动串联执行**，禁止只跑其中某一段：

```
POST /api/v1/experiments/{id}/process
  └─> _run_experiment_pipeline (local file) 或 _run_experiment_service_only (无 FORMAL_WORKFLOW)
        └─> _invoke_experiment_service        ← Qwen VLM 帧分析
        └─> _write_experiment_processing_outputs
              └─> _run_event_preprocessing_for_output   ← YOLO26 + tracking + event detection
                    └─> EventPreprocessingEngine.run()
                          ├─> YOLO26 检测帧流
                          ├─> TrackStreamBuilder（multi-object tracking）
                          ├─> EventProposalBuilder（5类事件提案）
                          ├─> KeyMaterialExtractor（clip + keyframe 生成）
                          └─> _run_step_bridge_for_output
                                └─> _enrich_steps_for_bridge（自动补充 required_event_types）
                                └─> StepBridgeEngine.run()（步骤匹配 + 升降级决策）
```

### 触发条件检查表
| 条件 | 是否阻断链路 | 处理方式 |
|------|------------|---------|
| `source_video` 文件不存在 | 是，跳过 event_preprocessing | 警告日志，继续写其他输出 |
| `steps.json` 为空 | 是，跳过 step_bridge | 警告日志 |
| `steps.json` 无 `required_event_types` | **不再阻断** | `_enrich_steps_for_bridge` 自动注入 SOP 默认步骤 |
| YOLO 权重不存在 | 是，EventPreprocessingEngine 报错 | 检查 yaml 配置 |
| DASHSCOPE_API_KEY 未设置 | 否，降级为规则命名 | 警告日志 |

### 手动补发 clip + 步骤推理（已有实验）
对已分析过但 `materials/events/` 缺失的旧实验，调用：
```
POST /api/v1/experiments/{id}/materials/publish
```
此路由会自动检测并触发：`EventPreprocessingEngine` → clip → `StepBridgeEngine`

---

## 3. 项目结构约束

### 3.1 单一主项目
- **唯一主项目**：`D:/LabEmbodiedVLA/LabSOPGuard/`
- `lab_preprocessing/` 是已迁移的历史子项目，其能力已全部集成到 LabSOPGuard：

| lab_preprocessing 模块 | LabSOPGuard 对应实现 |
|----------------------|-------------------|
| `EventRelevanceFilter` | `event_preprocessing/selective_overlay.py` → `SelectiveOverlayPolicy` |
| `MaterialNamingService` | `material_publishing/semantic_enhancer.py` → `QwenVlmDisplayNameEnhancer` |
| `MaterialArchiver` | `material_publishing/archive_planner.py` → `ArchivePlanner` + `publisher.py` → `SemanticMaterialPublisher` |
| `EventRecorder` | `event_preprocessing/event_proposal.py` → `EventProposalBuilder` |
| `ClipGenerator` | `event_preprocessing/key_material_extraction.py` → `KeyMaterialExtractor` |

- **禁止**从 `LabSOPGuard` 代码中 `import` `lab_preprocessing` 包
- **禁止**在 `lab_preprocessing/` 中新增功能；所有新功能写在 `LabSOPGuard/src/labsopguard/` 下

### 3.2 Python 包路径
- 后端启动时：`PYTHONPATH=src`（见 `start_full_stack.ps1`）
- 包名前缀：`labsopguard.*`
- 禁止在 `backend/main.py` 中直接 `sys.path.insert`

---

## 4. 五类核心物理事件

当前系统可检测并生成以下事件，**所有步骤推理和 SOP 匹配必须基于这五类**：

| 事件类型 | 触发条件 | 关键字段 |
|---------|---------|---------|
| `hand_object_interaction` | gloved_hand bbox 与物体 IoU/边界距离满足接触阈值 | `actor_track_id` |
| `object_move` | gloved_hand 接触 + 物体中心点位移 > 阈值 | `actor_track_id` |
| `liquid_transfer` | gloved_hand + container + container 空间关系（倾倒姿态） | `source_container`, `target_container`, `direction_status` |
| `panel_operation` | gloved_hand 接近 balance/panel/screen/button | `actor_track_id` |
| `container_state_change` | gloved_hand 接近 container/lid，IoU 变化 | `state_before`, `state_after`, `state_change_type` |

**SOP 默认步骤映射**（当 `steps.json` 无 `required_event_types` 时自动注入，见 `_SOP_DEFAULT_STEPS`）：

| 步骤 ID | 步骤名 | required_event_types |
|--------|--------|---------------------|
| `step_ppe` | 穿戴防护装备 | `hand_object_interaction` |
| `step_prepare` | 准备实验器材 | `object_move` |
| `step_open_container` | 打开容器盖 | `container_state_change` |
| `step_transfer` | 液体转移操作 | `liquid_transfer` |
| `step_panel` | 设备面板操作 | `panel_operation` |
| `step_close_container` | 关闭容器盖 | `container_state_change` |

---

## 5. 前后端启动

### 一键启动
```powershell
cd D:/LabEmbodiedVLA/LabSOPGuard
.\scripts\start_full_stack.ps1 -SkipRedis
```
脚本会自动：杀掉旧进程树（含子进程）→ 清理 dist/ 缓存 → 启动后端+前端 → 健康检查 → 打开最新实验

### 端口约定
| 服务 | 默认端口 |
|------|---------|
| 后端 FastAPI | 8000 |
| 前端 Vite | 5173（strictPort，不会自动切换端口） |
| Redis | 6379 |

### 手动启动（调试）
```powershell
# 后端（必须用 labsopguard 环境的 Python）
$env:PYTHONPATH="src"
E:\conda_envs\labsopguard\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# 前端（另开终端）
cd frontend-app
npm run dev
```

### 验证端点
```
GET  http://127.0.0.1:8000/                                          # status=running
GET  http://127.0.0.1:8000/api/v1/diagnostics                        # yolo26_status.available=true
GET  http://127.0.0.1:8000/api/v1/experiments                        # 实验列表
POST http://127.0.0.1:8000/api/v1/experiments/{id}/process           # 触发完整处理链路
POST http://127.0.0.1:8000/api/v1/experiments/{id}/materials/publish # 补发 clip（旧实验）
GET  http://127.0.0.1:8000/api/v1/experiments/{id}/materials/search  # 素材语义搜索
```

---

## 6. 每次开发前的检查清单

在修改任何代码前，确认以下内容：

- [ ] YOLO 权重路径：`configs/model/detection_runtime.yaml` → `model` 字段是否指向存在的 `.pt` 文件
- [ ] API Key：`DASHSCOPE_API_KEY` 在 `.env` 中是否存在
- [ ] 引入的 SDK：是否只用了 `dashscope` 和 `openai`（兼容模式）
- [ ] 新增功能位置：是否写在 `LabSOPGuard/src/labsopguard/` 下，而不是 `lab_preprocessing/`
- [ ] 新增事件类型：是否在五类之内，或明确扩展了 `_SOP_DEFAULT_STEPS` 和 `VALID_EVENT_TYPES`
- [ ] 测试：`pytest -q tests/test_model_data_enhancements.py tests/test_material_production_features.py`

---

## 7. 前端注意事项

### 视频播放器
- `<video src>` 需要绝对 URL，`activeVideo.url` 是相对路径（`/api/v1/...`）
- 必须用 `resolveArtifactUrl()` 转换为 `http://127.0.0.1:8000${url}`，不能依赖 Vite proxy
- 前端端口可能是 5173/5174 等任意值，proxy 不可靠

### CORS
- 后端默认已包含 `5173`、`5174`；需要新端口时改 `.env` 的 `CORS_ALLOW_ORIGINS`

### 关键素材入口
- `ExperimentWorkspace` 顶部 nav 有「关键素材」「素材时间轴」「步骤时间轴」链接
- 「生成关键素材」按钮 → `POST /materials/publish` → 自动链式触发 EventPreprocessingEngine → clip → StepBridge
- **auto-trigger 条件**：`materials/events/` 不存在 **或** 为空目录（两种情况都会重新跑引擎）

### 素材命名规则
- `display_name` 来自 EventProposalBuilder 生成的 `event.display_name` 字段（格式：`{实验名}_{事件类型}_{涉及物体}`）
- `evidence_summary` 字段是内部调试文本（含 grade=/score= 等），**禁止用于显示名称**
- `semantic_enhancer.py` 优先级：`event.display_name` > Qwen live > rule_based

### 启动时 orphaned task 自动恢复
- 后端启动时 `_recover_orphaned_tasks()` 将上次 `running/queued` 的任务自动标记为 `failed`
- 前端随即显示「Run analysis」按钮，可直接重新提交，无需手动清理

---

## 8. 已知限制与待改进

| 问题 | 当前状态 | 下一步 |
|-----|---------|-------|
| clip 生成依赖 imageio_ffmpeg | 已验证可用 | 无需改动 |
| `step_bridge` 步骤得分偏低 | `needs_review`（置信度 ~0.18） | 真实视频跑出高质量事件后自然提升 |
| `MATERIAL_DISPLAY_NAME_QWEN_ENABLED` | 已在 `.env` 设为 true | 启动时确认环境变量加载 |
| EventPreprocessingEngine 采样适配 | `_adapt_to_video_duration()` 自动按视频时长调整 max_frames 和 max_gap_merge | `LABSOPGUARD_EVENT_MAX_FRAMES` / `LABSOPGUARD_EVENT_INTERVAL_SEC` 可覆盖 |
| lab_preprocessing/ 孤立子项目 | 历史遗留 | 确认无依赖后可删除 |
