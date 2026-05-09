# ![RealityLoop 公司徽](./images/realityloop-logo-report.png) RealityLoop 专业实验分析报告标准

本文档是 RealityLoop 实验分析报告的唯一标准结构。后续 Qwen 撰写、后端 PDF 渲染、前端报告预览和导出都必须遵守这份结构。

## 1. 报告定位

报告面向一次实验分析运行，目标是形成一份专业、可审计、可追溯的 PDF 文档。报告需要说明分析了什么、发现了什么证据、哪些步骤或关键动作被证据支持、哪些风险需要复核，以及当前结果有哪些局限。

报告不得把模型推断写成最终事实。凡是 `candidate`、`inferred`、低置信度或证据不足的内容，都必须明确标注为“需要人工复核”。

## 2. 语言规范

- 报告标题、章节标题、正文说明、结论、建议：使用正式中文。
- 关键动作证据中的对象名：保持英文，例如 `balance`、`gloved hand`、`sample bottle blue`、`reagent bottle`。
- 实验 ID、运行 ID、模型名、版本号、时间戳、文件名：保留原始值。
- 避免宣传式表达，使用克制、证据导向、可审计的专业写法。

## 3. PDF 可见章节结构

### 0. 封面

用途：识别报告和数据来源。

必需内容：

- 报告标题
- 实验名称
- 实验 ID
- 运行 ID
- 结果版本
- 生成时间
- 报告结构版本
- Qwen 模型名称和调用状态

### 1. 执行摘要

用途：让读者快速了解本次分析结论。

必需内容：

- 总体结论
- 200 到 350 字中文摘要
- 核心指标：视频帧数、检测数量、结构化步骤数、关键动作数、关键素材数、告警数、平均置信度
- 证据充分性说明

### 2. 范围与数据来源

用途：说明报告覆盖哪些数据和能力。

必需内容：

- 实验说明或分析范围
- 源视频和视角信息，如有
- 已启用分析能力：标准实验分析、视频标注分析、关键动作索引、关键素材索引
- 数据限制说明

### 3. 关键结论

用途：沉淀最重要的分析发现。

每条结论必须包含：

- 结论内容
- 证据依据
- 对实验过程或操作判断的影响
- 置信度或复核要求

建议数量：3 到 6 条。

### 4. 步骤执行评估

用途：评估结构化实验步骤。

必需内容：

- 步骤执行概述
- 步骤表：序号、步骤名称、状态、时间范围、置信度、证据数量、评估说明
- 对 `candidate`、`inferred` 状态必须明确标注复核要求

### 5. 关键动作证据

用途：记录由物理证据支撑的关键动作。

必需内容：

- 关键动作概述
- 关键动作表：动作 ID、动作类型、动作名称、时间范围、持续时间、英文对象名、微片段数量、证据摘要、复核状态
- 证据摘要需要说明 YOLO 检测、手部与对象交互、微片段、多视角对齐等如何支撑该动作。

注意：本章节中的对象名必须保持英文。

### 6. 风险与异常

用途：说明风险、异常和告警。

必需内容：

- 风险概述
- 告警表：风险等级、规则或风险点、时间或帧、证据、建议
- 证据不足时必须明确写明需要人工复核

### 7. 关键素材与追溯

用途：让报告结论可以追溯到素材索引。

必需内容：

- 素材概述
- 素材表：素材名称、事件类型、时间范围、证据等级、相关英文对象名
- 说明素材索引和向量检索是否可用于后续复核

### 8. 综合评估与建议

用途：给出可执行的后续建议。

必需内容：

- 综合评估
- 分优先级的建议
- 人工复核重点
- 有证据支撑时给出操作改进建议

### 9. 局限性与审计信息

用途：让报告具备审计性。

必需内容：

- 局限性说明
- 报告结构版本
- 生成时间
- Qwen 模型与调用状态
- 分析模型和版本信息
- 结果版本
- Qwen 不可用时的回退原因

### 10. 签字页

用途：作为 PDF 报告最后一页，保留正式归档前的签署信息。

必需内容：

- 系统生成标识
- 审核人
- 审核日期
- 批准人
- 批准日期
- 系统生成声明

## 4. 机器输出结构

以下 JSON 字段名是程序接口，不作为 PDF 报告可见文案。Qwen 必须返回严格 JSON，不允许返回 Markdown 或代码块。字段值应使用中文撰写；只有 `objects_en` 和技术 ID 保持原始英文或原始值。

```json
{
  "schema_version": "professional_experiment_report.v1",
  "cover": {
    "report_title": "实验分析专业报告",
    "experiment_name": "实验名称",
    "experiment_id": "实验ID",
    "run_id": "运行ID",
    "result_version": "结果版本",
    "generated_at": "生成时间",
    "qwen_model": "Qwen模型名称"
  },
  "executive_summary": {
    "overall_conclusion": "总体结论",
    "summary": "中文执行摘要",
    "evidence_sufficiency": "证据充分性说明",
    "key_metrics": [
      {"label": "视频帧数", "value": "数值"}
    ]
  },
  "scope": {
    "description": "分析范围",
    "data_sources": ["数据来源"],
    "analysis_modules": ["分析能力"],
    "limitations": ["数据限制"]
  },
  "key_findings": [
    {
      "finding": "关键结论",
      "evidence": "证据依据",
      "impact": "影响说明",
      "confidence": "置信度或复核要求"
    }
  ],
  "procedure_assessment": {
    "summary": "步骤执行概述",
    "steps": [
      {
        "index": "序号",
        "step_name": "步骤名称",
        "status": "状态",
        "time_range": "时间范围",
        "confidence": "置信度",
        "evidence_count": "证据数量",
        "assessment": "评估说明"
      }
    ]
  },
  "key_action_evidence": {
    "summary": "关键动作概述",
    "actions": [
      {
        "action_id": "动作ID",
        "action_type": "动作类型",
        "action_name": "动作名称",
        "time_range": "时间范围",
        "duration": "持续时间",
        "objects_en": ["balance", "gloved hand"],
        "micro_segments": "微片段数量",
        "evidence_summary": "证据摘要",
        "review_status": "复核状态"
      }
    ]
  },
  "risk_alerts": {
    "summary": "风险概述",
    "alerts": [
      {
        "severity": "风险等级",
        "rule": "规则或风险点",
        "time_or_frame": "时间或帧",
        "evidence": "证据",
        "recommendation": "建议"
      }
    ]
  },
  "materials_traceability": {
    "summary": "素材追溯概述",
    "materials": [
      {
        "material_name": "素材名称",
        "event_type": "事件类型",
        "time_range": "时间范围",
        "evidence_grade": "证据等级",
        "related_objects_en": ["sample bottle blue"]
      }
    ]
  },
  "overall_assessment": {
    "assessment": "综合评估",
    "recommendations": [
      {
        "priority": "high | medium | low",
        "recommendation": "建议内容",
        "basis": "建议依据"
      }
    ],
    "human_review_points": ["人工复核重点"]
  },
  "limitations_audit": {
    "limitations": ["局限性"],
    "audit_metadata": {
      "report_schema_version": "报告结构版本",
      "generated_at": "生成时间",
      "qwen_model": "Qwen模型",
      "qwen_used": "是否调用Qwen",
      "analysis_model": "分析模型",
      "result_version": "结果版本",
      "fallback_reason": "回退原因"
    }
  },
  "signature_page": {
    "system_generated_id": "系统生成标识",
    "reviewer": "审核人",
    "approver": "批准人",
    "review_date": "审核日期",
    "approval_date": "批准日期",
    "statement": "系统生成声明"
  }
}
```

## 5. 生成规则

1. 后端先收集确定性的结构化数据。
2. Qwen 只负责在标准结构内撰写专业中文表述。
3. 后端对 Qwen 输出进行校验，缺失字段使用确定性中文模板补齐。
4. 后端使用 Jinja2 模板渲染专业 HTML，再使用 WeasyPrint 输出 PDF。
5. PDF 最后一页必须固定渲染签字页，包含系统生成标识、审核人和批准人。
6. 前端只向用户暴露专业 PDF 报告生成与下载入口。

