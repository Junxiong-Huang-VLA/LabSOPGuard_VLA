# integrated_system 可演示交付说明（2026-03-30）

## 目标与范围
本次仅围绕 `integrated_system` 主线修通，不做大规模重构，不新建平行实现。  
目标是把“上传视频 -> 分析执行 -> 实时进度 -> 结果展示 -> 文件下载”做成可演示、可跑通的版本。

## 原来坏的/空的/未接通点
1. 结果可视化不足：前端缺少标注视频预览、关键帧缩略图、摘要直出。
2. 状态字段不够稳定：`/api/status/<task_id>` 对前端直连展示支持不足。
3. 缺少安全预览接口：无法按任务安全读取视频/图片产物进行页面预览。
4. 密钥规则不严格：AI 密钥识别包含 `OPENAI_API_KEY`，不符合仅使用 `DOUBAO_API_KEY/ARK_API_KEY` 的约束。
5. PDF 中文字体与降级提示不够明确。

## 已修通的关键链路
1. `/api/analyze`：真正创建任务并异步执行主流程。
2. 分析流程：手部检测 -> 关键帧提取 -> AI 分析（可选）-> 步骤检查（可选）-> 标注视频导出 -> PDF（可选）-> 结果落盘。
3. `/api/status/<task_id>`：统一返回核心字段：
   - `task_id`
   - `status`
   - `progress`
   - `current_stage`
   - `message`
   - `outputs`
4. `/api/progress`：SSE 实时进度；支持 `start=recent` 减少重连历史噪声。
5. `/api/download/<task_id>/<file_type>`：可下载各产物。
6. `/api/health`：返回服务状态、关键模块可用性、API key 配置状态。
7. 新增 `/api/artifact/<task_id>?path=...`：用于任务内产物安全预览（视频/关键帧）。

## 前端可用性增强（保留原模板并增强）
文件：`integrated_system/templates/integrated_index.html`

页面现可清晰展示：
1. 上传区
2. 开关区（手检/AI/PDF/步骤检查/导出视频）
3. 开始分析按钮
4. 进度条
5. 当前阶段和状态
6. 结果展示区：
   - 标注视频预览
   - 关键帧缩略图列表
   - 文本摘要
   - PDF/alarm 生成状态
7. 下载按钮区与可下载文件提示

## 本次修改文件
1. `integrated_system/app_integrated.py`
2. `integrated_system/templates/integrated_index.html`
3. `integrated_system/integrated_pdf_generator.py`
4. `openai_wrapper.py`

## 新增文件
无（未新增第四套系统，也未新增平行实现文件）。

## 运行方式
在项目根目录执行：

```bash
python integrated_system/app_integrated.py
```

浏览器访问：

```text
http://localhost:5001
```

## 验证步骤（建议按顺序）
1. 上传实验视频。
2. 点击“开始分析”，确认任务进入 `queued/running`。
3. 观察进度条和 `current_stage` 实时变化。
4. 完成后确认页面可见：
   - 标注视频预览
   - 关键帧缩略图
   - 文本摘要
   - PDF/alarm 状态
   - 下载按钮可用
5. 下载验证：
   - `annotated_video`
   - `keyframes_json`
   - `summary_txt`
   - `alarm_log`（启用步骤检查时）
   - `pdf`（启用 PDF 时）

## 输出目录与关键产物
统一输出到：

```text
integrated_system/outputs/<timestamp>_<task_id>/
```

重点产物：
1. `hand_annotated*.mp4`
2. `keyframe_*.jpg`
3. `part1_keyframes.json`
4. `overall_summary.txt`
5. `alarm_log.json`
6. `integrated_analysis_report.pdf`（或 PDF 失败时 txt fallback）

## AI 与环境变量规则
1. AI 调用统一通过 `openai_wrapper.py`。
2. 仅读取：
   - `DOUBAO_API_KEY`
   - `ARK_API_KEY`
3. 未配置 key 时会降级跳过 AI 分析，不导致系统崩溃。

## 已完成的本地验证结论
1. 主链路可跑通：`/api/analyze` -> `/api/status` 到 `completed`。
2. 下载接口可用：`annotated_video/keyframes_json/summary_txt/alarm_log/pdf` 返回 200。
3. 服务启动可用：`python integrated_system/app_integrated.py` 后 `http://127.0.0.1:5001/api/health` 返回 200。

## 当前剩余短板
1. AI 结果质量受模型与密钥配置影响，未配置 key 时仅输出降级结果。
2. 关键帧数量受视频变化程度影响，静态视频关键帧可能较少。
3. 进度通道为 SSE + 轮询兜底（非 WebSocket），但当前演示链路已稳定可用。
