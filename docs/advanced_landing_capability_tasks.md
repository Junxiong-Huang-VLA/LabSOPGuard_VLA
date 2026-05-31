# 强落地实验理解能力任务书

## 范围

本轮目标不是基础候选能力，而是把实验视频理解推进到可审计、可验证、可接生产模型的后端能力。实现仍限定在 `src/key_action_indexer` 主线，保持 dry-run 可运行，不强依赖真实视频、ffmpeg、OCR 引擎或外部数据库。

## 已升级能力

### 真实项目资产接入

- 实现：`model_inventory.py`
- 输入：`LabSOPGuard/configs/model/detection_runtime.yaml`、训练权重、`class_schema.yaml`、YOLO 数据集。
- 处理：自动发现真实 `yolo26*` 权重、ONNX/Engine 产物、类别 schema、pose schema、YOLO 图片和标签数量。
- 输出：`model_inventory.json`，并写入 pipeline summary 和 advanced vision summary。
- 当前发现的项目资产：生产候选 YOLO26 姿态模型、4634 张 YOLO 数据图片、4634 个 YOLO 标签文件、13 类实验对象标注 schema。
- 约束：仍保持 `src/key_action_indexer` 对 LabSOPGuard 的代码独立性，只读取文件资产和配置，不导入 LabSOPGuard 后端模块。

### 任务 1：真实物体轨迹移动检测

- 实现：`advanced_vision_evidence.py`
- 输入：YOLO frame rows 中的 bbox、时间戳、micro-segment 时间窗。
- 处理：按对象标签聚合跨帧 bbox，计算中心点位移、路径长度、归一化位移。
- 输出：`object_trajectory_movement`，确认级别 `trajectory_confirmed`。
- 约束：没有 bbox track id 时使用标签级轨迹聚合；更高精度仍需 ByteTrack/DeepSORT 等跟踪器。

### 任务 2：液体流动与液位变化视觉证据

- 实现：`advanced_vision_evidence.py`
- 输入：contact/peak/release keyframe。
- 处理：对关键帧做经典图像分析，估计水平边缘位置、颜色变化和液位变化。
- 输出：`liquid_level_change` 或 `liquid_flow_candidate_visual`。
- 约束：没有训练好的流体/液面分割模型时，不声明真实流动已确认；输出会标明 `candidate_requires_fluid_segmentation`。

### 任务 3：设备面板 OCR 与控制状态

- 实现：`advanced_vision_evidence.py`
- 输入：面板/天平相关关键帧。
- 处理：如果本机安装 `pytesseract`，执行 OCR；否则输出需要 OCR/面板检测器的候选事件。
- 输出：`equipment_panel_ocr` 或 `equipment_control_change`。
- 约束：按钮/旋钮状态仍需要面板检测模型或模板库。

### 任务 4：容器状态变化检测

- 实现：`advanced_vision_evidence.py`
- 输入：容器相关 keyframe、对象标签。
- 处理：检测容器交互、颜色变化、cap/lid 标签；已支持真实 YOLO 类别中的 `tube-cap/tube_cap` 作为开盖/关盖证据。
- 输出：`container_open_close`、`container_color_change`。
- 约束：若当前 micro-segment 没有 cap/lid 检测结果，仍需要 cap/lid 检测器输出或人工确认。

### 任务 5：复杂 SOP 分支、循环、冲突消解

- 实现：`process_reasoner.py`
- 输入：SOP JSON、video understanding、state changes、context。
- 处理：支持 `branch_condition`、`repeatable/min_repeats/max_repeats`、重复超限、顺序冲突、分支未命中。
- 输出：步骤字段新增 `repeat_count/conflict_flags/branch_enabled/confirmation_status`。
- 约束：复杂条件表达式和多分支优化仍可继续增强。

### 任务 6：历史实验统计学习

- 实现：`history_learning.py`
- 输入：历史 session 目录、JSON、JSONL。
- 处理：统计动作频次、转移概率、持续时间分布、材料频次，并生成推荐 SOP。
- 输出：history model JSON。
- 约束：当前是本地统计模型，不接外部数据库服务。

### 任务 7：人工确认闭环

- 实现：`confirmation_loop.py`
- 输入：`experiment_process.json`。
- 处理：生成人工确认队列；支持 approved/rejected/needs_review；回写 process step。
- 输出：`human_confirmation_queue.jsonl`、`human_confirmation_decisions.jsonl`。
- 约束：当前是后端确认闭环，未接前端审批 UI。

## 仍需模型或系统集成的事项

- 液体流动的强视觉确认：当前真实标签集中有 pipette/tube/container 类，但未发现 liquid/meniscus/stream 分割类；需要把已有或新增的液体/液面标注数据接入 `model_inventory` 并训练/配置液体分割模型。
- 设备按钮/旋钮状态确认：当前真实类别中有 balance，但未发现 button/knob/display/panel 状态类；需要把面板 OCR crop、按钮/旋钮模板或检测模型接入。
- 容器开盖/关盖确认：真实 schema 已有 `tube-cap/tube_cap`，系统已接入；如果要区分 open/closed 状态，需要进一步标注状态类或时序规则。
- 多目标身份级轨迹：当前支持真实 YOLO bbox 轨迹；若需要身份级稳定轨迹，生产级应接 ByteTrack/DeepSORT 或 ReID。
- 复杂 SOP 规则引擎：当前支持常用分支和重复约束，复杂嵌套条件可接规则 DSL。
- 历史数据库服务：当前支持本地文件统计，后续可接 SQL/vector DB/实验 LIMS。
- 前端人工确认 UI：当前有后端队列和决策写回，尚未接 LabSOPGuard UI。
