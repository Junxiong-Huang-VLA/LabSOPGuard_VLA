# 运维与日常操作指南

## 1. 常用操作

### 1.1 启动后端

```powershell
.\.venv-e2e\Scripts\python.exe .\backend\main.py
```

### 1.2 启动前端

```powershell
cd .\frontend-app
npm run dev
```

### 1.3 本地命令行分析视频

```powershell
python .\video_analysis_pipeline.py --video <input.mp4> --output <annotated.mp4> --interval 4 --max-frames 12
```

## 2. 输出文件位置

视频分析产物：

- `outputs/video_analysis/analysis_<task_id>.json`
- `outputs/video_analysis/annotated_<task_id>.mp4`

实验理解产物：

- `outputs/experiments/<experiment_id>/`

## 3. 常见问题排查

### 3.1 下载到的“JSON”不是分析结果

排查：

1. 确认下载的是 `outputs/video_analysis/analysis_<task_id>.json`
2. 不要误打开仓库中的兼容模块文件
3. 优先从下载接口获取，而不是手工猜路径

### 3.2 PowerShell 里中文看起来像乱码

说明：

- 很多情况下是终端显示编码问题，不是 JSON 文件损坏

建议：

1. 用 Python 读取验证
2. 用 VS Code 以 UTF-8 打开
3. 优先通过浏览器下载而不是终端直接 `Get-Content`

### 3.3 视频里只有目标框，没有语义文字

说明：

- 当前版本设计为尾段汇总，非尾段主要保留框体与报警边框

### 3.4 视频里没有检测框

排查：

1. 检查 YOLO 权重路径
2. 检查模型是否能加载
3. 检查输入视频是否能被 OpenCV 读取
4. 检查视频中是否确实包含训练过的类别

### 3.5 Qwen 没有返回结果

排查：

1. 检查 `DASHSCOPE_API_KEY`
2. 检查网络连通性
3. 检查后端日志是否出现 `vlm_unavailable` 或 `vlm_error`

### 3.6 Windows 下中文文件名视频打不开

说明：

- OpenCV 在 Windows 下对非 ASCII 路径兼容性不稳定

建议：

1. 优先使用英文文件名
2. 通过前端上传到统一上传目录
3. 必要时先手工重命名再分析

## 4. 常见维护动作

### 4.1 替换默认模型

修改：

- `configs/model/detection_runtime.yaml`

或设置：

- `LABSOPGUARD_YOLO_MODEL`

### 4.2 调整报警规则

修改：

- `configs/alerts/alerting.yaml`

### 4.3 调整尾段汇总时长

修改代码：

- `src/labsopguard/video_analysis.py`

当前由 `_summary_window_sec()` 控制。

## 5. 推荐运维习惯

1. 所有生产权重使用固定绝对路径
2. 保留至少一条可复现实验视频作为回归样本
3. 每次改视频标注逻辑后，重新生成标准样例视频
4. 不要把 `outputs/` 当作长期权威存储
