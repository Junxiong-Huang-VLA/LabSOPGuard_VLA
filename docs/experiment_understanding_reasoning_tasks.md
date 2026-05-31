# 实验理解、上下文融合与步骤推理任务书

## 范围

本任务继续限定在 `src/key_action_indexer` 主线，不扩展复杂前端、云端 PTZ、五路相机编排或基础设施。目标是在现有 YOLO key action、micro-segment、统一时间线、素材库和状态变化索引基础上，补齐结构化视频理解、实验上下文融合、步骤状态机、过程补全和证据链导出。

## 当前已具备能力

- 视频切片、关键帧、micro-segment：已具备，依赖 YOLO/motion 与 dry-run fallback。
- 手部与物体接触检测：已具备基础能力，来自 YOLO hand-object interaction 和 micro-segment。
- 实验动作分类：已具备基础能力，输出 `weighing/pipetting/sample_adding_candidate/recording` 等 action type。
- 物体状态变化记录：已具备基础能力，输出 `state_change_index.jsonl` 中的 contact/peak/release/object_contact。
- 统一时间轴：已具备，输出 `unified_multimodal_timeline.jsonl` 和 `time_calibration_report.json`。
- 素材与事件绑定：已具备，输出 `material_asset_catalog.jsonl`，状态事件 `asset_refs` 可反链 `asset_id`。
- 检索：已具备 segment/micro vector retrieval 和素材 catalog 检索。

## 未完成项任务

### 任务 1：结构化视频理解结果

- 输入：micro-segment、state-change、material asset catalog。
- 处理：把接触、状态变化、动作分类、候选液体转移、候选容器状态变化、候选设备操作统一为 video understanding event。
- 输出：`metadata/video_understanding.jsonl`、`metadata/video_understanding_summary.json`。
- 验收：每条事件包含 event type、时间、对象、动作、置信度、异常标记、素材引用和 payload。

### 任务 2：事件置信度与异常标记

- 输入：evidence level、micro quality、interaction score、asset refs、缺失素材信息。
- 处理：计算 0-1 置信度；标记 weak evidence、missing keyframe、transcript-only、missing asset、candidate-only 等异常或限制。
- 输出：video understanding event 的 `confidence/confidence_reasons/anomaly_flags`。
- 验收：低证据事件不能被误报为视觉确认；候选液体转移必须标为 candidate。

### 任务 3：实验上下文融合

- 输入：用户文本、语音转写、AI 回复、上传图文、video understanding、material catalog、可选本地数据库 JSON/JSONL。
- 处理：解析实验目的、流程、材料、参数，建立 fused context。
- 输出：`metadata/experiment_context.json`。
- 验收：可从 transcript/user/AI/upload/database/video 多源生成 `purpose/procedure_candidates/materials/parameters/gaps`。

### 任务 4：步骤状态机

- 输入：video understanding、state changes、context、可选 SOP。
- 处理：定义步骤进入条件、完成条件；判断已完成、当前、下一步；处理跳步、重复、异常、冲突。
- 输出：`metadata/experiment_process.json`、`metadata/experiment_process_timeline.jsonl`。
- 验收：步骤包含 status、confidence、evidence_refs、next_step_hint、requires_human_confirmation。

### 任务 5：过程缺失检测与补全

- 输入：SOP steps、观察到的视频事件和状态事件、上下文。
- 处理：检测缺失步骤；根据前后步骤和 SOP 生成 inferred step；区分 direct observation 与 inferred completion。
- 输出：步骤级 `observed/inferred/skipped/missing_completion_reason/confidence_reasons`。
- 验收：低置信度补全必须触发人工确认。

### 任务 6：证据链结构

- 输入：video events、state changes、asset catalog、timeline/context。
- 处理：建立 step -> video/state/asset/text 证据链，支持按步骤回溯证据和按证据查找步骤。
- 输出：步骤记录内 `evidence_refs`，以及 process timeline JSONL。
- 验收：每个直接观察步骤至少引用一个 video/state/asset/text 证据；能追到 `asset_id`。

### 任务 7：规范 JSON 输出

- 输入：全部结构化结果。
- 处理：设计稳定 JSON schema 风格字段，支持后续数据库、接口和报告系统导入。
- 输出：video understanding、experiment context、experiment process 三类 JSON/JSONL。
- 验收：dry-run pipeline 自动生成，`pytest -q`、Python 编译、前端 build 不回退。

## 待加强项

- 物体移动检测：当前只能根据接触和状态变化输出 candidate，未做真实轨迹位移判定。
- 液体转移检测：当前只能根据 pipetting/加样语义和工具对象输出 candidate，未做液面/流体视觉确认。
- 设备面板操作检测：当前依赖对象/动作关键词 candidate，未做面板 UI/OCR/按钮状态识别。
- 容器状态变化检测：当前可记录容器接触/释放 candidate，未做开盖、液位、颜色变化的视觉确认。
- SOP 复杂推理：当前目标为 deterministic MVP，复杂条件、分支 SOP、历史实验统计学习仍待加强。
- 历史过程复用：当前使用本地 JSON/JSONL 作为数据库输入，未接真实数据库服务。
- 人工确认闭环：当前输出 `requires_human_confirmation`，未接 UI 或审批工作流。
