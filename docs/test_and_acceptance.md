# 测试与验收说明

## 1. 测试目标

确认以下能力可用：

1. 后端 API 可启动
2. 前端页面可构建和访问
3. 视频上传分析链路可跑通
4. 生成 JSON 与标注视频
5. YOLO 权重与 Qwen 可协同工作
6. 中文 JSON 编码、视频下载、结果展示正常

## 2. 自动化测试

推荐先执行基础回归：

```bash
python -m pytest tests/test_formal_pipeline.py tests/test_experiment_e2e.py tests/test_api_integration.py -q
```

可选补充：

```bash
python -m pytest tests/test_experiment_chain.py tests/test_experiment_empty_state.py -q
```

## 3. 前端构建测试

```bash
cd frontend-app
npm run build
```

验收标准：

- TypeScript 编译通过
- Vite 构建通过
- 无阻断性错误

## 4. 本地联调测试

### 4.1 启动服务

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_local_e2e_env.ps1
.\.venv-e2e\Scripts\python.exe .\backend\main.py
cd .\frontend-app
npm run dev
```

### 4.2 页面测试

访问：

- `http://127.0.0.1:3000/video-analysis`

验收项：

1. 页面可打开
2. 上传视频后任务状态可轮询
3. 任务完成后可下载视频和 JSON
4. 页面能展示分析摘要和报警信息

## 5. 视频分析验收

### 5.1 基础验收

上传任意可读视频后，应满足：

1. 生成 `analysis_<task_id>.json`
2. 生成 `annotated_<task_id>.mp4`
3. JSON 为 UTF-8 编码
4. 视频能正常播放

### 5.2 当前视觉行为验收

当前版本验收标准：

1. 全程逐帧显示目标框
2. 报警外框持续可见
3. 大段报警文字仅在视频尾段汇总显示
4. JSON 仍为抽帧语义分析结果，不要求每帧都有语义字段

### 5.3 参考产物

当前仓库内可作为参考的输出：

- `outputs/video_analysis/demo_45s_source.mp4`
- `outputs/video_analysis/demo_45s_annotated_qwen_v4.mp4`

## 6. API 验收

### 6.1 视频分析接口

验收接口：

1. `POST /api/v1/video-analysis/analyze`
2. `GET /api/v1/video-analysis/status/{task_id}`
3. `GET /api/v1/video-analysis/download/{task_id}/video`
4. `GET /api/v1/video-analysis/download/{task_id}/json`

验收要求：

- 提交后能返回 `task_id`
- 状态可从 `queued/running` 变为 `completed`
- 下载接口返回有效文件

### 6.2 实验接口

至少抽测：

1. 创建实验
2. 上传视频
3. 上传上下文
4. 上传协议
5. 启动处理
6. 拉取结构化结果

## 7. 人工验收清单

- 权重文件路径正确
- DashScope Key 配置正确
- 前端中文文案正常显示
- 分析 JSON 中文正常
- 标注视频框体稳定、无明显闪烁
- 报警文字不长期遮挡关键目标区域

## 8. 不通过条件

以下情况视为未通过：

- 后端无法启动
- 上传任务无法完成
- 生成文件为空
- JSON 下载乱码或不是分析结果
- 标注视频无法播放
- 目标框只在个别帧闪现
