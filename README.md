# LabSOPGuard

LabSOPGuard 是一个面向实验室操作视频的正式版基础系统。当前版本保留并正式化了两条已经跑通的主链路：

- `POST /api/v1/experiments/*` 驱动的实验流程理解主链路
- `POST /api/v1/video-analysis/*` 驱动的 YOLO + Qwen VL 视频分析与标注视频主链路

本次收敛的目标不是推倒重来，而是把已有能力从实验性堆叠收拢为可维护工程：

- 核心代码与历史材料分离
- 视频分析、步骤推理、结构化输出进入正式包 `src/labsopguard/`
- 现有 FastAPI / React 流程继续可用
- 输出结果统一沉淀到 `outputs/experiments/<experiment_id>/` 与 `outputs/video_analysis/`
- 任务状态改为文件持久化 + 内存友好的降级方案

## 当前能力

- YOLO 目标检测与可视化标注视频生成
- DashScope / Qwen VL 场景理解与 PPE 判断
- 实验创建、视频上传、处理、时间线和结构化结果查询
- 步骤、时间、证据、参数、置信度的结构化输出
- 物理事件、素材流、时间对齐的基线实现

## 正式版目录

```text
LabSOPGuard/
├── backend/                  FastAPI API
├── frontend-app/             React + TypeScript 前端
├── src/
│   ├── experiment/           兼容现有实验主链路的核心模型与服务
│   ├── labsopguard/          正式版收敛包
│   │   ├── input_layer.py
│   │   ├── preprocessing.py
│   │   ├── reasoning.py
│   │   ├── output_layer.py
│   │   ├── video_analysis.py
│   │   ├── workflow.py
│   │   ├── tasking.py
│   │   └── config.py
│   ├── lab_vla/              现有 VLA 相关模块，保留
│   └── project_name/         历史兼容模块，非正式主链路
├── configs/                  统一运行配置
├── docs/                     正式文档
├── docs/archive/             历史分析和交付材料归档
├── archive/                  手工调试与运行产物归档
├── scripts/run_demo.py       正式版演示运行脚本
├── tests/                    基础回归测试
└── video_analysis_pipeline.py 兼容旧入口的正式封装
```

## 快速启动

### 1. 后端依赖

```bash
pip install -r requirements.txt
```

### 2. 前端依赖

```bash
cd frontend-app
npm install
```

### 3. 环境变量

```bash
set DASHSCOPE_API_KEY=your_key
set DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 4. 启动后端

```bash
python backend/main.py
```

默认地址：`http://127.0.0.1:8000`

### 5. 启动前端

```bash
cd frontend-app
npm run dev
```

默认地址：`http://127.0.0.1:3000`

## 主链路

### 实验理解主链路

1. `POST /api/v1/experiments`
2. `POST /api/v1/experiments/{id}/upload/video`
3. `POST /api/v1/experiments/{id}/upload/context`
4. `POST /api/v1/experiments/{id}/upload/protocol`
5. `POST /api/v1/experiments/{id}/process`
6. `GET /api/v1/experiments/{id}/timeline`
7. `GET /api/v1/experiments/{id}/structured`

### 视频分析主链路

1. `POST /api/v1/video-analysis/analyze`
2. `GET /api/v1/video-analysis/status/{task_id}`
3. `GET /api/v1/video-analysis/download/{task_id}/video`
4. `GET /api/v1/video-analysis/download/{task_id}/json`

## 输出产物

实验主链路：

- `outputs/experiments/<experiment_id>/experiment.json`
- `outputs/experiments/<experiment_id>/timeline.json`
- `outputs/experiments/<experiment_id>/steps.json`
- `outputs/experiments/<experiment_id>/physical_events.json`
- `outputs/experiments/<experiment_id>/material_stream.json`
- `outputs/experiments/<experiment_id>/structured.json`

视频分析主链路：

- `outputs/video_analysis/analysis_<task_id>.json`
- `outputs/video_analysis/annotated_<task_id>.mp4`
- `outputs/video_analysis/tasks/<task_id>.json`

## 测试

```bash
python -m pytest tests/test_formal_pipeline.py tests/test_experiment_e2e.py tests/test_api_integration.py -q
```

## 清理说明

- 历史交付报告、手工调试脚本、临时分析 JSON 已迁入 `archive/` 或 `docs/archive/`
- `outputs/`、`uploads/`、`integrated_system/outputs/` 已纳入忽略规则，不再污染主干
- 旧模块保留但不再作为正式主链路继续扩散
