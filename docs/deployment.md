# 部署说明

## 1. 适用范围

本文档面向当前 `LabSOPGuard` 正式版工程，覆盖以下两条主链路的部署与启动：

- 实验理解主链路：`/api/v1/experiments/*`
- 视频分析主链路：`/api/v1/video-analysis/*`

当前默认运行方式为：

- 后端：FastAPI，入口 `backend/main.py`
- 前端：React + Vite，目录 `frontend-app/`
- 视觉模型：YOLO，默认权重见 `configs/model/detection_runtime.yaml`
- 语义模型：DashScope / Qwen-VL

## 2. 环境要求

### 2.1 操作系统

- 本地开发：Windows 10/11
- 训练环境：Linux / AutoDL 均可

### 2.2 运行时

- Python `3.10+`
- Node.js `18+`
- npm `9+`

### 2.3 模型与 GPU

- 推荐 NVIDIA GPU
- 当前默认模型配置为 `cuda:0`
- 无 GPU 时可切到 CPU，但视频分析速度会明显下降

## 3. 关键目录

```text
LabSOPGuard/
├── backend/                     后端 API
├── frontend-app/                前端页面
├── src/labsopguard/             正式版视频分析与任务链路
├── configs/model/               模型运行配置
├── configs/alerts/              报警规则配置
├── uploads/                     上传视频
├── outputs/video_analysis/      视频分析输出
└── outputs/experiments/         实验理解输出
```

## 4. 本地开发部署

### 4.1 推荐：隔离联调环境

项目已提供最小联调环境脚本：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local_e2e_env.ps1
```

该脚本会创建：

- `.\.venv-e2e\`

### 4.2 启动后端

```powershell
.\.venv-e2e\Scripts\python.exe .\backend\main.py
```

默认地址：

- `http://127.0.0.1:8000`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`

### 4.3 启动前端

```powershell
cd .\frontend-app
npm install
npm run dev
```

默认地址：

- `http://127.0.0.1:3000`

Vite 已配置 `/api` 代理到后端 `8000` 端口。

## 5. 环境变量

至少需要以下变量：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`

可选变量：

- `LABSOPGUARD_YOLO_MODEL`
- `LABSOPGUARD_FONT_PATH`

示例：

```powershell
$env:DASHSCOPE_API_KEY="your_key"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

## 6. 生产部署建议

当前仓库已具备开发联调能力，生产部署建议采用：

1. Python 虚拟环境或 Conda 独立环境运行后端
2. 由 `uvicorn` 或进程守护器托管后端
3. Nginx 反向代理前端静态资源与 `/api`
4. 模型权重放在固定绝对路径或统一挂载目录
5. `outputs/` 与 `uploads/` 放在有足够容量的磁盘

示例后端命令：

```bash
python backend/main.py
```

如需改为 `uvicorn`，需要额外补充独立 ASGI 入口和部署脚本。

## 7. 启动后检查项

启动后至少确认以下项目：

1. `GET /docs` 可访问
2. 前端 `/video-analysis` 页面可打开
3. `configs/model/detection_runtime.yaml` 指向的权重文件存在
4. DashScope Key 已加载
5. 上传视频后能生成：
   - `outputs/video_analysis/analysis_<task_id>.json`
   - `outputs/video_analysis/annotated_<task_id>.mp4`

## 8. 已知部署注意事项

- Windows 下 OpenCV 对非 ASCII 文件路径的兼容性不稳定，建议上传或处理前尽量使用英文文件名
- 若未配置 DashScope，视频链路仍可输出 YOLO 检测框，但不会有可靠的语义分析结果
- 当前视频分析为“离线生成标注视频”，不是浏览器实时流式推理
