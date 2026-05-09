# Architecture

## 目标

正式版围绕同一份实验资产组织两类流程：

- 实验过程理解
- 视频分析与标注输出

二者共享统一配置、任务状态和结构化输出约束。

## 四层结构

### 1. Input Layer

代码位置：`src/labsopguard/input_layer.py`

职责：

- 按 `experiment_id` 组织视频、上下文文本、协议文本、历史消息、上传文档
- 统一输入对象 `ExperimentInputBundle`
- 为时间对齐与后续推理提供一致入口

### 2. Preprocessing Layer

代码位置：`src/labsopguard/preprocessing.py`

职责：

- 文本时间锚点对齐
- 关键时间戳汇总
- 视频索引与素材流整理
- 物理变化事件的基线索引

现有实现：

- `MultiModalPreprocessor.align_text_records`
- `MultiModalPreprocessor.build_artifact`
- `ExperimentService._generate_physical_events`

### 3. Reasoning Layer

代码位置：`src/labsopguard/reasoning.py` 与 `src/experiment/service.py`

职责：

- 解析 protocol step graph
- 将视频步骤与 protocol 节点做基线匹配
- 区分 `confirmed / candidate / inferred`
- 通过证据和上下文生成步骤级记录

现有实现：

- `StepReasoner`
- `StepGraphReasoner`
- `ExperimentService._reason_steps`

### 4. Output Layer

代码位置：`src/labsopguard/output_layer.py`

职责：

- 生成正式结构化 JSON
- 输出步骤、时间线、证据引用、参数、置信度、来源说明
- 将时间对齐与物理事件纳入最终结果

## 视频分析子系统

代码位置：`src/labsopguard/video_analysis.py`

职责：

- 自适应采样
- YOLO 检测阈值配置化
- VLM 场景理解
- PPE 一致性融合
- 标注视频导出
- JSON 结果导出
- 文件化任务状态记录

## API 收敛

### 保留的正式接口

- `/api/v1/experiments/*`
- `/api/v1/video-analysis/*`

### 已标记历史兼容接口

- `/api/v1/streams/*`
- `/api/v1/tasks/*`

这些旧接口仍保留，但不代表正式主链路。

## 存储策略

- 配置：`configs/model/detection_runtime.yaml`
- 实验输出：`outputs/experiments/<experiment_id>/`
- 视频分析输出：`outputs/video_analysis/`
- 视频分析任务状态：`outputs/video_analysis/tasks/<task_id>.json`

## 已修正的关键问题

- `ExperimentService` 抽帧不再写入临时目录后立即失效
- 视频分析任务状态不再依赖一次性 `status_*.txt`
- 结构化结果不再只返回时间线 JSON，而是补齐正式 `structured.json`
