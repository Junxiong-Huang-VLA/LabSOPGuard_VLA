# Report Specification

## 1. 报告输入规范（report_input）

报告输入为结构化 JSON，建议字段：

- `title`
- `session_id`
- `summary`
- `violations`
- `timeline`

### summary

- `total_events`
- `detection_events`
- `violation_events`
- `severity_distribution`
- `compliance_ratio`

### violations[]

- `event_type`
- `sop_step`
- `severity_level`
- `timestamp`
- `violation_message`
- `class_name`（可选）
- `bbox` / `region`（可选）

### timeline[]

- `timestamp`
- `event_type`
- `sop_step`
- `violation_flag`
- `severity_level`

## 2. 报告输出规范

默认输出路径：`outputs/reports/`

- `runtime_report.pdf`（优先）
- `runtime_report.txt`（回退）

## 3. 生成流程

1. 收集结构化事件（JSONL）
2. 构建 report_input
3. 生成 PDF（失败时 TXT）
4. 归档并记录日志

## 4. 与审计系统对接建议

- 保留 `session_id`、`sop_id`、`camera_id`
- 保留严重级别分布与违规时间线
- 保留原始事件 JSON 作为追溯依据
