# Development

## 环境

- Python 3.10+
- Node.js 18+
- 可选 DashScope API Key

## 常用命令

### 后端

```bash
python backend/main.py
```

### 前端

```bash
cd frontend-app
npm run dev
```

### 运行正式演示

```bash
python scripts/run_demo.py --video path/to/video.mp4 --title demo
```

### 测试

```bash
python -m pytest tests/test_formal_pipeline.py tests/test_experiment_e2e.py tests/test_api_integration.py -q
```

## 开发约束

- 保持 `/api/v1/video-analysis/*` 兼容
- 保持 DashScope / Qwen VL 接入不被破坏
- 结构性新增优先放在 `src/labsopguard/`
- `src/project_name/` 与 `integrated_system/` 视为历史兼容区，不继续扩散新逻辑
- 运行产物只落在 `outputs/` 与 `uploads/`，并继续被 `.gitignore` 忽略

## 目录责任

- `src/experiment/`: 现有实验主链路核心实现
- `src/labsopguard/`: 正式版收敛与输出约束
- `backend/main.py`: FastAPI 编排层
- `frontend-app/src/`: 流程验证前端
- `archive/`: 手工调试、旧报告和非主线材料

## 变更建议

1. 新的正式模块优先加到 `src/labsopguard/`
2. 旧接口仅做兼容，不再扩张能力面
3. 调整 schema 时同步更新 `docs/data_model.md` 和测试
