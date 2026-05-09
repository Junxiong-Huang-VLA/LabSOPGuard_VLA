# 角色八：DevOps 工程师（DevOps Engineer）

## 职责定位

负责开发环境、部署环境的搭建与维护，包括 conda 环境管理、服务启停脚本、GPU 服务器配置与 CI 流程。

## 核心工作内容

### 本地环境

#### conda 环境
- 主环境：`LabSOPGuard`（`E:/conda_envs/LabSOPGuard`）
- 备份：`E:/AI_Data/_conda_envs_backup/`
- 依赖文件：`environment.yml`（GPU）/ `environment.runtime.yml`（CPU 推理）
- 初始化：
  ```powershell
  .\scripts\init_env.ps1
  # 或
  .\scripts\setup_env.ps1
  ```

#### 服务管理
```powershell
# 一键启动（推荐）
.\scripts\start_full_stack.ps1 -Restart -SkipRedis

# 停止服务
.\scripts\stop_preview.ps1

# 查端口占用
netstat -aon | findstr "8000\|5173\|6379"
```

#### 端口配置
| 服务 | 默认端口 | 配置位置 |
|------|---------|---------|
| FastAPI | 8000 | `start_full_stack.ps1` |
| Vite | 5173 | `frontend-app/vite.config.js` |
| Redis | 6379 | 系统默认 |

### AutoDL 服务器

#### 快速接入
参考 `docs/autodl_quickstart.md`

#### 常用操作
```bash
# 上传数据集
scp -r data/dataset/ user@autodl-server:/path/to/
# 下载权重
scp user@autodl-server:/path/to/best.pt outputs/training/<run>/weights/
# 下载完整训练产物
scp -r user@autodl-server:/path/to/run/ outputs/training/<run>/
```

### 双机同步
```powershell
# 本地双 PC 同步
.\scripts\sync_dual_pc.ps1
```
参考 `docs/dual_pc_sync.md`。

### 环境变量管理
- 唯一路径：`D:/LabEmbodiedVLA/LabSOPGuard/.env`
- 不得在 `lab_preprocessing/` 或其他子目录单独维护 `.env`
- 所有服务启动前必须 `load_dotenv('.env')`
- 关键变量：
  ```bash
  DASHSCOPE_API_KEY=...
  CORS_ALLOW_ORIGINS=http://localhost:5173,http://localhost:5174
  MATERIAL_DISPLAY_NAME_QWEN_ENABLED=true
  ```

### GPU 驱动要求
- 本地：RTX 4050 Laptop（6GB VRAM）→ 仅做推理
- AutoDL：RTX 5090（32GB）→ 训练
- CUDA 版本：cu124（本地）/ cu128（AutoDL）

### CI/CD
- 提交前运行：`pytest -q tests/test_model_data_enhancements.py tests/test_material_production_features.py`
- 禁止 `--no-verify` 绕过

## 故障排查

| 症状 | 排查步骤 |
|------|---------|
| 后端 8000 端口占用 | `netstat -aon \| findstr 8000` → 杀进程 |
| YOLO 加载失败 | 检查 `configs/model/detection_runtime.yaml` 路径 |
| Redis 连接失败 | `redis-cli ping` → 若失败重启 Redis |
| 前端 5173 端口被占用 | Vite 自动 +1 到 5174，CORS 记得同步更新 |
