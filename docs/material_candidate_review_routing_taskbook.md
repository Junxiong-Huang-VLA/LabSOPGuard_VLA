# 关键素材证据门控分诊 Pipeline 总方案与实施任务书

## 目标

减少关键素材人工复核工作量，但不能把质量控制交给 VLM 独自承担。系统应先由 YOLO/tracklet 产生可追溯物理证据，再由规则和 VLM 分诊，最后只把模棱两可的候选交给人工。

## 不可退让原则

- YOLO 决定可画什么、哪些帧存在物理证据；VLM 只能在这些证据范围内做语义确认和歧义解释。
- 标注框只画 active hand-object instance，不画上下文物体。
- `instrument_context` 可以参与语义判断，但不能参与标注目标。
- VLM 输出 `semantic_action` 或 `corrected_primary_object` 时，必须附带 YOLO evidence refs。
- 自动通过必须同时满足连续帧数、手部置信、目标置信、tracklet 稳定、无同类实例冲突、无语义冲突。
- `auto_ready` 和 VLM 通过结果必须保留 5%-10% 抽样人工质检。
- 每次优化必须跑 contract tests，防止已修好的称量纸、手套、试剂瓶、天平区域规则悄悄退化。

## 数据结构

候选记录新增或固化以下字段：

- `review_route`: `auto_ready | vlm_review | human_review | reject_or_low_quality`
- `review_reason_codes`: 触发该 route 的原因码数组。
- `route_confidence`: 分诊置信度。
- `manipulated_object`: 被手实际操作的对象。
- `instrument_context`: 语义上下文仪器，例如 `balance`。
- `semantic_action`: 语义动作，例如 `weighing`、`pipetting`。
- `corrected_primary_object`: VLM 或规则修正后的主对象，只能引用 YOLO 证据。
- `evidence_refs`: 支撑该候选的 YOLO frame/tracklet/micro-segment 引用。
- `vlm_review`: VLM 复核输入、输出、证据引用和失败原因。
- `human_review`: 人工决策、修改字段、审核人和时间。

## 四类分诊

| route | 进入条件 | 默认处理 |
| --- | --- | --- |
| `auto_ready` | YOLO 证据强、tracklet 稳、active interaction 唯一、语义无冲突 | 不进人工主队列，只做抽样质检 |
| `vlm_review` | 框和交互证据足够强，但语义需要确认 | VLM 在 YOLO evidence refs 内确认动作/对象 |
| `human_review` | 多实例、遮挡、弱手框、VLM 与 YOLO 冲突、VLM 证据不足 | 进入人工主队列 |
| `reject_or_low_quality` | 框漂、无手-物交互、上下文物体冒充交互对象、低置信 | 不进入推荐结果 |

## 原因码

首批原因码：

- `multiple_same_class_instances`
- `weak_hand_confidence`
- `weak_object_confidence`
- `tracklet_unstable`
- `semantic_conflict`
- `vlm_yolo_disagreement`
- `unsupported_vlm_correction`
- `context_only_object`
- `no_active_hand_object_interaction`
- `bbox_scale_or_render_mismatch`
- `insufficient_consecutive_frames`
- `instrument_context_only`

## T0-T12 实施任务

T0 命名与输出契约：正式目录统一为 `实验名称_日期`，过滤 rerun、candidate、material、UUID、raw video folder 等技术名。

T1 证据包 schema：把 `review_route`、`review_reason_codes`、`evidence_refs` 写入候选 JSON/JSONL。

T2 Route classifier：实现纯规则分诊器，先不依赖 VLM，确保 dry-run 可运行。

T3 Tracklet 质量指标：输出连续帧数、bbox 抖动、插值比例、置信均值、active instance 匹配分。

T4 Active-only 标注保护：只画手和被手操作对象；天平、桌面纸包、背景瓶不作为该候选标注目标。

T5 语义字段拆分：标题和元数据区分 `manipulated_object` 与 `instrument_context`。

T6 称量场景优先规则：当 `balance` 稳定出现，且手-纸/瓶/药匙发生于天平区域，语义归入称量操作，但标注仍只画 active object。

T7 VLM 受限复核：VLM prompt 只允许引用候选 evidence refs；无 refs 的修正一律转 `human_review`。

T8 前端队列：默认展示推荐结果和 `human_review`，低质候选单独折叠，不混入结果。

T9 抽样质检：`auto_ready` 与 VLM 通过结果保留 5%-10% 人工抽检，并记录抽检通过率。

T10 Golden contract tests：固定称量纸、手套、瓶、药匙、天平上下文的黄金样例，防止回归。

T11 批量回填：对历史实验重算候选索引和 route，不自动覆盖人工已确认结果。

T12 验收发布：输出人工工作量下降比例、抽检错误率、低质混入率、VLM 证据缺失率。

## 验收标准

- `reject_or_low_quality` 不得进入默认推荐结果。
- `auto_ready` 抽检通过率低于 95% 时，自动降级相关规则到 `human_review`。
- VLM 缺少 YOLO evidence refs 的修正不得写入 `corrected_primary_object`。
- 称量纸候选只画手上正在操作的纸，不画桌面纸包或天平。
- 天平可以作为 `instrument_context`，但不能在“手与称量纸”候选里被画成交互对象。
- 命名回归测试必须覆盖技术标题、重复日期、导入实验日期优先级。

## 建议验证命令

```powershell
python -m pytest tests/test_material_candidate_pipeline_contract.py tests/test_material_references.py tests/test_yolo_detector.py tests/test_tracklet_annotations.py -q
python -m pytest -q
npm run build
python -m compileall src
```
