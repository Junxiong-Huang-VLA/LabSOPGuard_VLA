# 角色九：测试工程师（QA Engineer）

## 职责定位

负责系统各层的测试覆盖，包括单元测试、集成测试、端到端验收测试和回归测试。保障每次迭代不引入新 bug。

## 核心工作内容

### 测试结构
```
tests/
├── test_model_data_enhancements.py    # 核心：模型数据增强测试
├── test_material_production_features.py  # 核心：素材生产流程测试
└── （其他测试文件）
```

### 必须通过的测试（每次 commit 前）
```bash
pytest -q tests/test_model_data_enhancements.py tests/test_material_production_features.py
```

### 端到端验收路径（标准实验）
使用基准实验 ID：`c404e890-4e3d-4ba1-8860-bd40c7f81a37`

验收检查清单：
```bash
# 1. 后端健康
curl http://127.0.0.1:8000/api/v1/diagnostics
# 期望：yolo26_status.available=true

# 2. 实验处理
curl -X POST http://127.0.0.1:8000/api/v1/experiments/c404e890.../process

# 3. 输出文件验收
ls outputs/experiments/c404e890.../
# 期望存在：experiment.json, preprocessing.json, material_stream.json,
#           physical_events.json, material_index.sqlite,
#           analysis/analysis.json, analysis/annotated.mp4
```

参考 `docs/test_and_acceptance.md`。

### 各层测试重点

#### 检测层
- YOLO 推理结果不为空
- 各类 conf ≥ 0.25 的检测框数量合理

#### 事件引擎层
- 五类事件均可检测出至少 1 个
- 事件 schema 字段完整（`event_type`, `actor_track_id`, `start_frame`, `end_frame`）
- clip 文件生成（`materials/events/*.mp4`）

#### 步骤推理层
- step_bridge 输出包含所有步骤的 grade
- `needs_review` 置信度字段为 float

#### API 层
- 所有核心路由返回 2xx
- 任务失败时返回正确的 `failed` 状态，不挂起
- 启动后 orphaned tasks 自动恢复为 `failed`

#### 前端层
- 视频播放器能正确加载 `annotated.mp4`
- 素材时间轴渲染完整
- 步骤颜色与 grade 一致

### 回归测试触发时机
- 修改 `event_preprocessing/` 任何文件后
- 修改 `step_bridge/` 任何文件后
- 切换 YOLO 权重后
- 修改 `.env` 中 Qwen 相关配置后

### 已知测试豁免
- `tube/tube-cap/spearhead/pipette`：数据集无标注，暂不写检测回归测试
- AutoDL 训练测试：仅在云端验证，本地不跑

## 禁止事项
- 禁止在测试中 mock YOLO 推理（需要真实权重验证）
- 禁止跳过核心两个测试文件直接 commit
