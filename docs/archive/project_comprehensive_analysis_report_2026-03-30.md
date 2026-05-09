## 📌 模块概述
- 模块名称：`LabSOPGuard_VLA`（项目级，包含 `integrated_system` Web 主线与 `src/project_name` 检测监控流水线）
- 代码语言/框架：Python 3.10；Flask；OpenCV；MediaPipe；Ultralytics YOLO；NumPy/Pandas；ReportLab；OpenAI SDK（Doubao/ARK 兼容）

## 🎯 核心功能
该项目用于实验室 SOP（标准操作流程）视频合规分析，支持从视频中抽取检测事件、判断违规、并输出结构化结果与报告。  
项目有两条主路径：一条是 `integrated_system` 的 Web 交互链路（上传、进度、结果、下载）；另一条是 `scripts + src/project_name` 的离线批处理链路（scan/infer/monitor/export/analyze/audit）。  
检测层结合实时目标/姿态信息，规则引擎根据 PPE、步骤顺序和行为语义输出违规事件。  
系统可以导出 JSON/CSV/JSONL/PDF（失败时 TXT fallback）等多种结果，适合演示和批量评估。  
在 AI key 缺失时会降级运行，保证主流程不中断。

## 📂 功能清单
- 视频上传与任务创建：输入（视频文件 + 功能开关）-> 处理逻辑（`/api/analyze` 创建任务目录与后台线程）-> 输出（`task_id`、初始状态、输出目录）
- 手部检测与标注视频：输入（原始视频）-> 处理逻辑（MediaPipe 手部关键点检测 + 可选叠加绘制）-> 输出（`hand_detection.json`、`hand_annotated.mp4`）
- 关键帧提取：输入（原始视频）-> 处理逻辑（帧差阈值 + 最小间隔 + 上限控制）-> 输出（`keyframe_*.jpg`、`part1_keyframes.json`）
- 关键帧 AI 分析：输入（关键帧图像）-> 处理逻辑（`openai_wrapper.py` 调用视觉模型；缺 key 时降级）-> 输出（`keyframe_ai_analysis.json`、`overall_summary.txt`）
- SOP 步骤检查：输入（关键帧元数据 + AI 文本 + 规则配置）-> 处理逻辑（步骤推断、顺序校验、缺失步骤与条件规则求值）-> 输出（`alarm_log.json`）
- PDF 报告生成：输入（任务摘要、手检结果、报警列表、输出清单）-> 处理逻辑（ReportLab 生成，异常时 TXT 回退）-> 输出（`integrated_analysis_report.pdf` 或 `.txt`）
- 实时状态与进度推送：输入（任务执行事件）-> 处理逻辑（内存任务表 + SSE 事件流）-> 输出（`/api/status/<task_id>` 与 `/api/progress` 实时数据）
- 结果预览与下载：输入（任务 ID + 文件类型）-> 处理逻辑（产物路径映射与安全校验）-> 输出（视频/图片预览、单文件下载、打包 ZIP 下载）
- 批处理推理（离线）：输入（manifest + rules + alert config）-> 处理逻辑（`scripts/infer.py`/`run_monitor.py`/`export_results.py` 调用 `SOPMonitorPipeline`）-> 输出（批量结果 JSON、事件 JSONL/CSV、汇总报告）
- 0-to-1 一体化编排：输入（阶段参数、配置、规则）-> 处理逻辑（`scripts/run_0to1_pipeline.py` 顺序执行 scan->infer->monitor->export->analyze->audit）-> 输出（阶段元数据、分析结果、审计资产）

## 🔗 依赖关系
- 依赖的外部模块/服务/接口：
  - Python 库：`flask`、`opencv-python`、`mediapipe`、`ultralytics`、`numpy`、`pandas`、`pyyaml`、`reportlab`、`openai`、`torch`、`transformers`
  - 外部 AI 服务：Doubao/ARK 兼容 OpenAI API（通过环境变量 `DOUBAO_API_KEY` / `ARK_API_KEY`）
  - 本地配置：`configs/sop/rules.yaml`、`configs/alerts/alerting.yaml`、`configs/data/dataset.yaml` 等
- 被哪些模块调用（如已知）：
  - `integrated_system/app_integrated.py` 调用 `hand_detection.py`、`keyframe_ai.py`、`step_checker.py`、`integrated_pdf_generator.py`
  - `keyframe_ai.py` 调用根目录 `openai_wrapper.py`
  - `scripts/infer.py`、`scripts/run_monitor.py`、`scripts/export_results.py` 调用 `src/project_name/pipelines/sop_monitor_pipeline.py`
  - `sop_monitor_pipeline.py` 调用 `MultiLevelDetector`、`SOPComplianceEngine`、`AlertNotifier`、`EventStructurer`、`ReportInputBuilder`
- 数据库/缓存/消息队列等资源使用情况：
  - 数据库：未使用
  - 缓存：未使用独立缓存组件
  - 消息队列：未使用
  - 状态存储：`integrated_system` 使用进程内内存字典（`TASKS`/`PROGRESS_EVENTS`）+ 文件落盘

##  关键逻辑与注意事项
- `integrated_system` 任务状态是进程内内存，服务重启后内存态会丢失，但产物文件仍在磁盘。
- `/api/artifact/<task_id>` 做了路径越界校验（相对输出目录），这是预览接口的关键安全点。
- 手部检测模块对 MediaPipe 版本差异做了分支兼容（`mp.solutions` 不存在时降级），否则容易直接异常。
- 关键帧提取依赖帧差阈值和时间间隔，阈值设置不当会导致关键帧过少或过多。
- `step_checker.py` 使用 AST 白名单求值规则条件，变量名必须在上下文中定义，否则会记录 `rule_eval_errors`。
- `MultiLevelDetector` 在 `strict_model=true` 且 YOLO 不可用时会直接抛错；在非严格模式下走 fallback 逻辑。
- AI 分析通过 `openai_wrapper.py` 统一封装，未配置 `DOUBAO_API_KEY/ARK_API_KEY` 时返回降级结果而非中断。
- PDF 生成依赖 ReportLab 与字体可用性，失败时回退 TXT，前端/调用方需要识别扩展名变化。

## 📊 接口/API 说明（如有）
| 接口名 | 方法 | 参数 | 返回值 | 说明 |
|--------|------|------|--------|------|
| `/api/analyze` | POST | `video` 文件；`enable_hand_detection`、`enable_ai_analysis`、`enable_pdf`、`enable_step_check`、`enable_video_export` | `task_id`、`status`、`progress`、`current_stage`、`output_dir` | 创建分析任务并异步执行 |
| `/api/status/<task_id>` | GET | `task_id` | `task_id`、`status`、`progress`、`current_stage`、`message`、`outputs`、`artifacts` | 查询任务实时状态与可视化产物信息 |
| `/api/progress` | GET (SSE) | `task_id`（可选）；`start=recent`（可选） | SSE event（`progress/current_stage/message`） | 实时推送进度与阶段 |
| `/api/artifact/<task_id>` | GET | `task_id`；query: `path` | 文件流 | 任务内产物预览接口（图片/视频等） |
| `/api/download/<task_id>/<file_type>` | GET | `task_id`；`file_type`（如 `annotated_video`、`pdf`、`keyframes_json`） | 文件下载流 | 单文件下载 |
| `/api/download_bundle/<task_id>` | GET | `task_id` | ZIP 文件下载流 | 打包下载已生成产物 |
| `/api/health` | GET | 无 | `status`、`api_key_configured`、`available_modules`、`task_count` | 服务健康与关键依赖可用性检查 |

## 🚧 已知限制或潜在风险
- 任务状态以内存为主，当前不具备多实例共享与持久化任务队列能力。
- Web API 暂无鉴权/限流，直接暴露时存在安全与滥用风险。
- 视频处理与模型推理耗时较高，默认线程池在高并发下可能积压任务。
- 不同环境下编解码器可用性差异较大（例如 `mp4v` 写入失败），会影响演示稳定性。
- 规则引擎与步骤推断对文本关键词依赖较强，复杂场景可能出现误报/漏报。
- `src/project_name` 与 `integrated_system` 存在并行功能路径，长期维护需防止行为漂移。
- 目前未使用数据库记录审计链路，历史任务检索、失败重试、可观测性能力有限。

