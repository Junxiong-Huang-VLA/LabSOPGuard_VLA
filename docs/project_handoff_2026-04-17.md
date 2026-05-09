# Project Handoff 2026-04-17

## 目的

这份文档用于下次继续开发时快速恢复上下文。
重点区分三类事实：

- 已完成并已真实验证
- 已接通但效果未达标
- 尚未完成或仍有明确阻塞

本文档以当前仓库实际状态和本轮实测结果为准。

## 当前目标边界

当前项目的主目标已经从“单独跑通视频分析 demo”收口到“实验主链路正式闭环”。
正式主入口是：

- `/api/v1/experiments/*`

兼容保留但不再作为主业务入口的是：

- `/api/v1/video-analysis/*`

当前不建议优先做的方向：

- 新模型训练
- 数据增强
- Prompt 打磨
- 页面美化
- 演示包装

当前最重要的是：

- 保持实验创建 -> 上传视频 -> 触发处理 -> 状态可查 -> 结果回写 -> 前端可见 这条链路稳定
- 在此基础上继续解决“模型有效性”和“结果可视化”问题

## 本轮已完成并已验证

### 1. 实验主链路已经正式闭环

已完成：

- 实验创建后可以上传视频
- `POST /api/v1/experiments/{id}/process` 现在会创建正式任务记录
- 后台会调用视频分析主链路
- 结果会正式回写到 experiment 输出目录
- 前端可以看到正式任务状态

后端关键文件：

- `backend/main.py`
- `src/experiment/service.py`
- `src/labsopguard/video_analysis.py`
- `src/labsopguard/tasking.py`

前端关键文件：

- `frontend-app/src/pages/Upload.tsx`
- `frontend-app/src/pages/ExperimentWorkspace.tsx`
- `frontend-app/src/pages/ExperimentList.tsx`
- `frontend-app/src/api.ts`
- `frontend-app/src/types.ts`

### 2. experiments 与 video-analysis 已正式收口

当前关系：

- `video-analysis` 仍保留为底层兼容能力
- `experiments` 已成为正式业务入口
- `experiments/process` 会驱动视频分析，并把 `analysis.json`、`annotated.mp4` 等结果回写到 experiment 输出目录

### 3. 任务状态已经正式化

当前 experiment 任务状态字段：

- `task_id`
- `experiment_id`
- `status`
- `current_stage`
- `progress`
- `video_path`
- `error_type`
- `error_message`
- `started_at`
- `completed_at`
- `output_paths`

当前正式状态集合：

- `created`
- `uploaded`
- `queued`
- `running`
- `failed`
- `completed`

状态落盘位置：

- `outputs/experiments/tasks/<experiment_id>.json`

状态查询接口：

- `GET /api/v1/experiments/{id}/status`

### 4. 失败已经可见

当前失败不再是“看起来没分析，但不知道原因”。

已实现：

- 前置失败可落盘
- 后台失败可落盘
- 前端可读 failed 状态
- error type 和 error message 可查询

已覆盖的失败类型：

- `upload_missing`
- `video_not_found`
- `pipeline_invoke_failed`
- `model_call_failed`
- `output_write_failed`
- `unexpected_exception`

### 5. 实验结果已经正式回写

成功后会回写到：

- `outputs/experiments/<experiment_id>/experiment.json`
- `outputs/experiments/<experiment_id>/timeline.json`
- `outputs/experiments/<experiment_id>/steps.json`
- `outputs/experiments/<experiment_id>/structured.json`
- `outputs/experiments/<experiment_id>/physical_events.json`
- `outputs/experiments/<experiment_id>/material_stream.json`
- `outputs/experiments/<experiment_id>/analysis/analysis.json`
- `outputs/experiments/<experiment_id>/analysis/annotated.mp4`

### 6. 前端基础展示已补齐

当前 workspace 页面已支持：

- 显示 experiment 正式状态
- 轮询任务状态
- 显示失败原因
- 切换原始视频 / 标注视频
- 查看 `analysis.json`
- 显示模型状态
- 显示检测与报警摘要
- 显示 experiment 产物路径

### 7. `.env` 自动加载已补齐

后端 `backend/main.py` 现在会自动读取项目根目录 `.env`。

已知注意点：

- `.env` 必须是 UTF-8 无 BOM
- 如果用 PowerShell 写文件，默认容易写出 BOM，导致第一行 key 名污染
- 本轮已经实际踩过这个坑：`DASHSCOPE_API_KEY` 因 BOM 失效，后来已修复

推荐 `.env` 内容：

```env
DASHSCOPE_API_KEY=your_real_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 本轮真实验证结果

### 成功闭环 case

experiment id:

- `a6d50154-3b09-48f8-be68-141250aac964`

结果：

- `status = completed`
- experiment 输出目录完整生成
- `analysis.json / annotated.mp4 / structured.json / timeline.json / steps.json` 全部存在

### 失败可见 case

experiment id:

- `9160fbc7-2af2-404e-bdc3-efe503a27581`

结果：

- `status = failed`
- `error_type = upload_missing`
- `error_message = upload missing: no video provided. Upload video first or pass video_path.`

### Qwen 已启用验证 case

experiment id:

- `2a485cf9-6147-4933-95f4-e4573f9d0552`

结果：

- `vlm_enabled = true`
- `Qwen` 已被真实调用
- 但该测试视频内容不适合实验场景理解，返回了低价值描述

### 真实实验视频验证 case

视频：

- `D:\labdata\discription_pdf\first_person_复杂长操作_normal_correct_001_rgb.mp4`

experiment id:

- `7cb5e134-135f-46fd-81fb-3292616ba31f`

结果：

- `status = completed`
- `Qwen` 返回高质量实验语义描述
- 已识别：电子天平、称量纸、试剂瓶、手套盒、笔记本电脑、玻璃器皿
- 已识别活动：称量样品、准备试剂、记录数据
- `annotated.mp4` 成功生成
- `analysis.json` 成功生成

但同时要注意：

- `total_detections = 0`
- 当前这条视频上 yolo26 没有出框
- 所以标注视频中会有文字面板和报警摘要，但不会出现 bbox

## 当前已接通但效果未达标

### 1. Qwen / DashScope

状态：

- 已接通
- 已实测可用

当前问题：

- 通过 PowerShell 查看 API 返回时，中文可能显示乱码，这是终端编码问题，不是 JSON 文件本身坏了
- `analysis.json` 用 Python `utf-8` 读取时内容正常

结论：

- Qwen 调用链当前属于“可用”

### 2. yolo26 检测

状态：

- 运行时权重已切到：
  - `outputs/training/yolo26s_pose_lab_v4_focus_auto/weights/best.pt`
- 模型文件存在
- 推理代码会实际加载该权重

当前问题：

- 在已测试视频上仍然经常 `detections = 0`
- 即使 Qwen 能识别实验语义，YOLO 仍可能不出框
- 因此用户会看到：有语义分析、有报警文字，但没有检测框

结论：

- yolo26 当前是“已加载但效果不稳定”，不是“未接通”

### 3. PPE / 报警

状态：

- 当前报警基于分析结果生成
- 如果没有检测框，PPE 仍可能由默认融合逻辑给出 `missing_*` 提示

当前问题：

- 当检测为 0 时，报警更像保守提示，而不是可靠裁决
- 不能把当前 PPE 报警直接视为稳定准确的安全判断

## 当前未完成 / 明确存在的问题

### 1. 前端测试文件仍有乱码历史内容

位置：

- `frontend-app/src/pages/__tests__/ExperimentWorkspace.test.tsx`

说明：

- 运行页面不依赖这些乱码文本
- 但测试文件里还有历史脏数据，需要单独清理

### 2. 前端虽然能显示标注视频链接，但仍缺少更强的分析可视化

当前已有：

- 标注视频切换
- JSON 下载
- 摘要信息

缺口：

- 没有单帧 bbox 列表与视频时间点联动
- 没有报警时间线单独视图
- 没有按 frame 的检测面板或缩略图导航

### 3. yolo26 在真实实验视频上没有稳定出框

这是当前最重要的模型侧问题。

但注意优先级：

- 现在不是“主链路没通”
- 是“主链路已通，但检测效果不足”

### 4. 旧 PDF 能力仍受 WeasyPrint 系统依赖影响

问题：

- 本机缺失 `gobject-2.0-0`
- 旧 PDF 报告相关能力会被降级跳过

影响：

- 不影响实验主链路
- 会影响旧报告/PDF 输出能力

### 5. Redis 不可用时会退回内存存储

当前现象：

- 启动日志里会显示 Redis 不可用
- 系统会退回内存存储

影响：

- 当前实验主链路不受根本性影响
- 但在线流式/跨进程场景不适合长期依赖内存回退

## 当前建议的开发优先级

### P0

1. 保持 experiment 主链路稳定，不要再让 `experiments` 和 `video-analysis` 重新分叉
2. 把前端页面与最新 analysis/artifact 展示保持一致
3. 清理前端测试文件乱码并修复对应断言

### P1

1. 排查 yolo26 在真实实验视频上 `detections=0` 的原因
2. 用多条真实实验视频做专项验证
3. 核对模型类别定义、训练集类别、运行时 class registry 和阈值是否一致

### P2

1. 补 experiment 级 artifact 浏览器体验
2. 增加按帧分析面板
3. 增加报警时间线与关键帧视图

### 暂不建议优先做

- 再开一条新视频分析产品线
- 大规模重做前端样式
- 扩写新的演示逻辑
- 在主链路之外额外加一套任务系统

## 当前关键文件职责

### 后端

- `backend/main.py`
  - FastAPI 编排层
  - experiment 主链路入口
  - task 状态管理
  - artifact 下载接口
  - analysis 查询接口

- `src/experiment/service.py`
  - 正式 experiment 结构化处理
  - step / timeline / structured output 生成

- `src/labsopguard/video_analysis.py`
  - 采样
  - YOLO 推理
  - Qwen VLM 场景理解
  - PPE 融合
  - 标注视频导出
  - analysis JSON 导出

- `src/labsopguard/tasking.py`
  - 文件化任务状态存储

### 前端

- `frontend-app/src/pages/Upload.tsx`
  - 创建实验并触发上传/处理

- `frontend-app/src/pages/ExperimentWorkspace.tsx`
  - experiment 状态展示
  - 标注视频展示
  - analysis 摘要
  - 模型状态

- `frontend-app/src/api.ts`
  - experiments / status / analysis 等接口封装

- `frontend-app/src/types.ts`
  - experiment / task / analysis 类型定义

## 启动与验证方式

### 后端

```powershell
cd D:\LabEmbodiedVLA\LabSOPGuard
C:\Users\Win10\miniconda3\envs\LabSOPGuard\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 前端

```powershell
cd D:\LabEmbodiedVLA\LabSOPGuard\frontend-app
npm.cmd run dev -- --host 127.0.0.1 --port 3000
```

### 最小验证

1. 访问 `/upload`
2. 创建实验并选择视频
3. 进入 workspace
4. 观察 `queued -> running -> completed / failed`
5. 检查：
   - 标注视频是否可播放
   - `analysis.json` 是否可下载
   - analysis 摘要是否可见
   - `output_paths` 是否完整

### 推荐排查接口

- `GET /api/v1/experiments/{id}`
- `GET /api/v1/experiments/{id}/status`
- `GET /api/v1/experiments/{id}/analysis`
- `GET /api/v1/experiments/{id}/artifacts/annotated_video`
- `GET /api/v1/experiments/{id}/artifacts/analysis_json`

## 下次继续开发前建议先做的 5 件事

1. 打开这份文档确认当前优先级不要跑偏
2. 确认 `.env` 仍是 UTF-8 无 BOM
3. 启动后端后检查 `/analysis` 返回里 `vlm_enabled` 是否为 `true`
4. 用一条真实实验视频先跑 experiment 主链路，不要先跑旧 demo
5. 如果用户反馈“没框”，先查 `analysis.summary.total_detections`，不要先怀疑前端

## 一句话总结

当前项目已经完成了“实验主链路正式闭环”和“Qwen 实际可调用”，但 yolo26 在真实视频上的检测效果仍不稳定，因此现在的主问题已经不是链路问题，而是检测有效性和结果可视化质量问题。
