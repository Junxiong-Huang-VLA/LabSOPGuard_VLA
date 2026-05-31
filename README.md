# LabEmbodied

RealityLoop 实验室双视角视频理解平台 —— 单一整合仓库。

把核心算法、数据处理 pipeline、FastAPI 后端、Vite 前端、数据库检索、Qwen VLM 理解、关键素材库、实验室日报整合在一起。原视频、模型权重、训练集等大资产**外置于仓库之外**,仓库只保留代码与管道。

## 目录结构

```
src/
  key_action_indexer/   核心算法:时间对齐、片段检测、微分段、证据、向量索引、CLI
  realityloop_sync/     多机位帧同步工具(CLI: realityloop-sync)
  labsopguard/          事件预处理、关键素材发布/检索、embeddings、专业报告、检测器
  experiment/           Qwen VLM 客户端 + 实验服务
  project_name/         检测/SOP/PDF 底层模块(backend 运行时依赖)
backend/                FastAPI 服务(main.py + routers),uvicorn 启动,端口 8000
frontend/               Vite + React + TS 前端,端口 5173,/api 代理到后端
configs/                YAML 配置(检测运行时、素材打分、报告、SOP 等)
tests/                  pytest 测试集
docs/                   核心规范文档
```

## 外置资产(不在仓库内)

| 资产 | 位置 | 说明 |
|------|------|------|
| 模型权重 | `D:\LabModels`(`LAB_MODELS_DIR`) | YOLO/pose 权重,见 `D:\LabModels\README.md` |
| 原始视频 | `LAB_VIDEO_STORE_ROOT` | 双视角原始/对齐视频 |
| 实验产物 | `outputs/` | 片段、关键帧、关键片段、素材库(运行时生成) |
| 训练集 | 外置 | YOLO 标注图像,不进仓 |
| 密钥 | `.env` | 从 `.env.example` 复制后填入 |

## 启动

```bash
# 1) 后端依赖
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .

# 2) 配置密钥
copy .env.example .env        # 填入 DASHSCOPE_API_KEY 等;按需设 LAB_MODELS_DIR / LAB_VIDEO_STORE_ROOT

# 3) 启动后端(端口 8000)
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 4) 启动前端(端口 5173,自动代理 /api -> :8000)
cd frontend
npm install
npm run dev
```

CLI:`realityloop-sync --help`,核心管道 `python -m key_action_indexer.cli --help`。

## 测试

```bash
pytest                 # 后端 + 算法
cd frontend && npm run test    # 前端单测(ExperimentWorkspace / MaterialSearch 等)
```
