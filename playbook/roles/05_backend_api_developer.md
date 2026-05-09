# 角色五：后端 API 开发者（Backend API Developer）

## 职责定位

负责 FastAPI 后端的路由设计、实验生命周期管理、任务队列与服务健康维护。是前后端契约的维护者。

## 核心工作内容

### 主要代码区域
```
backend/
├── main.py              # FastAPI app 入口、路由注册、启动事件
src/labsopguard/
├── tasking.py           # 异步任务队列管理
├── workflow.py          # 实验处理完整链路编排
├── input_layer.py       # 视频上传与输入处理
├── output_layer.py      # 实验输出写入
├── workspace_governance.py  # 工作区治理
└── runtime_paths.py     # 路径解析
```

### 核心路由
| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/` | 服务状态 |
| GET | `/api/v1/diagnostics` | YOLO/Redis/API Key 诊断 |
| GET | `/api/v1/experiments` | 实验列表 |
| POST | `/api/v1/experiments/{id}/process` | 触发完整处理链路 |
| POST | `/api/v1/experiments/{id}/materials/publish` | 补发 clip（旧实验） |
| GET | `/api/v1/experiments/{id}/materials/search` | 素材语义搜索 |

### 任务生命周期
- 状态：`queued` → `running` → `completed` / `failed`
- 启动恢复：`_recover_orphaned_tasks()` 自动将上次 `running/queued` 标记为 `failed`
- 任务状态持久化在 SQLite（`outputs/experiments/<id>/material_index.sqlite`）

### 启动方式
```powershell
# 一键启动
.\scripts\start_full_stack.ps1 -Restart -SkipRedis

# 手动调试
$env:PYTHONPATH="src"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### CORS 配置
默认允许 `5173`、`5174`；新增端口时修改 `.env` 的 `CORS_ALLOW_ORIGINS`。

### 开发新路由规范
1. 路由统一前缀 `/api/v1/`
2. 响应 schema 用 Pydantic model 定义
3. 任务型操作（耗时 > 1s）必须走异步任务队列，不得同步阻塞
4. 错误响应统一格式：`{"detail": "...", "error_code": "..."}`

### 关注指标
- 后端启动时间 ≤ 5s
- `/process` 接口响应（任务入队）≤ 200ms
- 任务失败率 ≤ 5%

## 禁止事项
- 禁止在 `backend/main.py` 中 `sys.path.insert`（用 `PYTHONPATH=src`）
- 禁止跨路由共享可变状态（用任务队列隔离）
