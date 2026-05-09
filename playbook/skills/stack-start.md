# stack-start — 一键启动/重启全栈服务

启动或重启 Redis + FastAPI 后端 + Vite 前端，并验证所有服务健康。

## 执行步骤

### 1. 启动全栈
```powershell
cd D:/LabEmbodiedVLA/LabSOPGuard
.\scripts\start_full_stack.ps1 -Restart -SkipRedis
```

### 2. 健康检查（等待约 5 秒后执行）
```bash
# 后端
curl -s http://127.0.0.1:8000/ 
# YOLO 状态
curl -s http://127.0.0.1:8000/api/v1/diagnostics
```

### 3. 验证 YOLO 权重已加载
确认 `diagnostics` 响应中 `yolo26_status.available=true`，若为 false：
- 检查 `configs/model/detection_runtime.yaml` → `model` 字段路径是否正确
- 检查 `.pt` 文件是否存在

### 4. 报告服务状态
| 服务 | 端口 | 状态 |
|------|------|------|
| Redis | 6379 | ? |
| FastAPI | 8000 | ? |
| Vite 前端 | 5173/5174 | ? |

### 5. 输出访问地址
- 前端：`http://127.0.0.1:5173`
- API 文档：`http://127.0.0.1:8000/docs`
