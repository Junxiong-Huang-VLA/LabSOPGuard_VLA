# Dataset Specification

## 1. 数据规范

建议最小样本字段：

- `sample_id`
- `video_path`
- `sop_id`
- `instruction`

可选字段：

- `camera_id`
- `operator_id`
- `action_history`
- `expected_steps`
- `ground_truth_violations`

## 2. 标注规范

### 2.1 目标标注

- `class_name`
- `confidence`
- `bbox` 或 `region`
- `center_point`（如适用）
- `depth_info`（如适用）

### 2.2 事件标注

- `event_type`
- `sop_step`
- `violation_flag`
- `severity_level`
- `timestamp`

## 3. 事件格式规范

标准事件 JSON 字段：

- `sample_id`
- `camera_id`
- `frame_id`
- `timestamp`
- `class_name`
- `confidence`
- `bbox` / `region`
- `center_point`
- `depth_info`
- `event_type`
- `sop_step`
- `violation_flag`
- `severity_level`
- `trace`

## 4. SOP 文档规范

- 文件建议放在 `data/raw/sop_docs`
- 每个 SOP 包含：
  - `sop_id`
  - required steps
  - violation rules
  - severity mapping
- SOP 变更需同步更新 `configs/sop/rules.yaml`。

## 5. 数据版本规范

- 使用 `dataset_vMAJOR.MINOR.PATCH`
- MAJOR：字段结构或标签体系变更
- MINOR：新增场景或类别
- PATCH：错误修复
