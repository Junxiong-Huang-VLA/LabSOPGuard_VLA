"""
Enhanced Scientific-Grade PDF Report Generator for LabSOPGuard.
================================================================
9-Chapter FDA ORA-LAB / ISO 17025 / WHO LQMS Standard Report:
1. Cover Page
2. Executive Summary (compliance score gauge, key findings)
3. Experimental Timeline (Gantt chart)
4. Per-Step Detail (name, time range, status, annotated screenshot, LLM summary)
5. Violation Detail Cards (timestamp, SOP reference, severity, annotated image, description, correction)
6. Statistical Analysis (4 charts: type distribution, severity pie, step compliance, time histogram)
7. Improvement Recommendations (priority-sorted corrective actions)
8. Appendix (full frame gallery, raw detection data)
9. Signature Page

Uses reportlab for PDF generation with CJK font support.
Integrates matplotlib charts and annotated keyframe images.
Includes LLM-driven text generation hooks for natural language content.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Report Data Builder (Enhanced)
# ---------------------------------------------------------------------------

def _build_report_data(
    task_result: Dict[str, Any],
    scene_profile: Dict[str, Any],
    chemistry_analysis: Dict[str, Any],
    alarm_log: Dict[str, Any],
    keyframe_meta: List[Dict[str, Any]],
    output_dir: Path,
    chart_paths: Optional[List[str]] = None,
    annotated_frame_paths: Optional[List[str]] = None,
    llm_texts: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Aggregate all analysis results into enhanced report data structure."""
    task_id = task_result.get("task_id", "unknown")
    status = task_result.get("status", "unknown")
    video_path = task_result.get("video_path", "unknown")

    exp_type = scene_profile.get("experiment_type", "general_lab_operation")
    exp_type_zh = scene_profile.get("experiment_type_zh", "一般实验室操作")
    confidence = scene_profile.get("confidence", 0.0)

    compliance = alarm_log.get("compliance_summary", {})
    alarm_count = alarm_log.get("alarm_count", 0)
    alarms = alarm_log.get("alarms", [])
    expected_steps = alarm_log.get("expected_steps", [])
    observed_steps = alarm_log.get("observed_steps", [])

    # Compliance score calculation
    total_expected = compliance.get("total_expected", len(expected_steps))
    total_observed = compliance.get("total_observed", len(observed_steps))
    compliance_ratio = (total_observed / max(total_expected, 1)) * 100 if total_expected > 0 else 0
    passed = compliance_ratio >= 80 and compliance.get("safety_violations", 0) == 0

    kf_count = len(keyframe_meta)
    module_status = task_result.get("module_status", {})
    module_notes = task_result.get("module_notes", [])

    overall_summary = ""
    summary_path = output_dir / "overall_summary.txt"
    if summary_path.exists():
        overall_summary = summary_path.read_text(encoding="utf-8")

    # Find keyframe images (prefer annotated)
    keyframe_images = []
    if annotated_frame_paths:
        keyframe_images = annotated_frame_paths
    else:
        keyframe_dir = output_dir / "keyframes"
        if keyframe_dir.exists():
            keyframe_images = sorted(str(p) for p in keyframe_dir.glob("keyframe_*.jpg"))

    # Build step detail data
    step_details = []
    for i, step_id in enumerate(expected_steps):
        step_observed = step_id in observed_steps
        step_detail = {
            "step_id": step_id,
            "step_name": _step_id_to_name(step_id),
            "index": i + 1,
            "status": "compliant" if step_observed else "missing",
            "start_sec": i * 300,  # placeholder, real data from pipeline
            "end_sec": (i + 1) * 300,
            "frames": [],
        }
        # Find frames for this step
        for kf in keyframe_meta:
            if kf.get("event_type") == "step_transition" and kf.get("event_details", {}).get("step_id") == step_id:
                step_detail["frames"].append(kf.get("file_path", ""))
        step_details.append(step_detail)

    # Severity counts
    severity_counts = {"critical": 0, "major": 0, "minor": 0, "info": 0}
    for a in alarms:
        sev = a.get("severity", "minor").lower()
        if sev in severity_counts:
            severity_counts[sev] += 1

    # Violation type counts
    violation_types = {}
    for a in alarms:
        vtype = a.get("alarm_type", a.get("violation_type", "未知"))
        violation_types[vtype] = violation_types.get(vtype, 0) + 1

    return {
        "task_id": task_id,
        "status": status,
        "video_path": video_path,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "experiment_type": exp_type,
        "experiment_type_zh": exp_type_zh,
        "scene_confidence": confidence,
        "expected_steps": expected_steps,
        "observed_steps": observed_steps,
        "step_details": step_details,
        "compliance_summary": compliance,
        "compliance_ratio": compliance_ratio,
        "compliance_passed": passed,
        "alarm_count": alarm_count,
        "alarms": alarms,
        "severity_counts": severity_counts,
        "violation_types": violation_types,
        "keyframe_count": kf_count,
        "keyframe_images": keyframe_images,
        "overall_summary": overall_summary,
        "hand_summary": task_result.get("hand_summary", {}),
        "module_status": module_status,
        "module_notes": module_notes,
        "scene_profile": scene_profile,
        "chemistry_analysis": chemistry_analysis,
        "chart_paths": chart_paths or [],
        "llm_texts": llm_texts or {},
    }


def _step_id_to_name(step_id: str) -> str:
    """Convert step ID to human-readable name."""
    name_map = {
        "wear_ppe": "穿戴PPE装备",
        "verify_reagent_label": "验证试剂标签",
        "prepare_transfer_tool": "准备转移工具",
        "execute_transfer": "执行转移操作",
        "close_container": "关闭容器",
        "clean_workspace": "清洁工作台",
        "dispose_waste": "处置废弃物",
    }
    return name_map.get(step_id, step_id.replace("_", " ").title())


# ---------------------------------------------------------------------------
# CJK Font Registration
# ---------------------------------------------------------------------------

def _register_fonts() -> Tuple[str, str]:
    """Register CJK fonts for bilingual PDF. Returns (font_name, fallback_font)."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except Exception:
        return "Helvetica", "Helvetica"

    cjk_candidates = [
        ("C:/Windows/Fonts/msyh.ttc", "MSYH"),
        ("C:/Windows/Fonts/msyhbd.ttc", "MSYHBD"),
        ("C:/Windows/Fonts/simsun.ttc", "SIMSUN"),
        ("C:/Windows/Fonts/simhei.ttf", "SIMHEI"),
    ]

    for path, name in cjk_candidates:
        try:
            p = Path(path)
            if p.exists():
                pdfmetrics.registerFont(TTFont(name, str(p)))
                return name, name
        except Exception:
            continue

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light", "STSong-Light"
    except Exception:
        pass

    return "Helvetica", "Helvetica"


# ---------------------------------------------------------------------------
# LLM Text Generation Hooks
# ---------------------------------------------------------------------------

def generate_violation_descriptions(
    alarms: List[Dict[str, Any]],
    openai_client=None,
) -> List[Dict[str, str]]:
    """Generate natural language violation descriptions using LLM.

    Each description includes: violation content, SOP reference, risk, correction suggestion.
    Falls back to template-based generation if LLM unavailable.
    """
    descriptions = []
    for alarm in alarms:
        vtype = alarm.get("alarm_type", "未知违规")
        severity = alarm.get("severity", "minor")
        timestamp = alarm.get("timestamp", 0)
        desc_zh = alarm.get("description_zh", "")
        desc_en = alarm.get("description_en", "")

        if openai_client:
            try:
                prompt = (
                    f"实验室SOP违规分析。\n"
                    f"违规类型: {vtype}\n"
                    f"严重等级: {severity}\n"
                    f"时间: {timestamp:.1f}秒\n"
                    f"原始描述: {desc_zh}\n\n"
                    f"请生成结构化违规描述，包含：\n"
                    f"1. 具体违规内容（2-3句话）\n"
                    f"2. 对应SOP条款引用\n"
                    f"3. 潜在风险\n"
                    f"4. 纠正建议\n"
                )
                response = openai_client.chat.completions.create(
                    model="qwen-vl-plus",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                text = response.choices[0].message.content
                descriptions.append({"alarm": alarm, "llm_description": text})
                continue
            except Exception:
                pass

        # Fallback: template-based
        risk_map = {"critical": "可能导致严重安全事故", "major": "存在安全隐患需立即整改", "minor": "需注意并改进"}
        correction_map = {
            "missing_ppe": "立即佩戴相应PPE装备",
            "unsafe_transfer": "在指定安全区域进行操作",
            "container_not_closed": "操作完成后及时关闭容器",
        }
        correction = correction_map.get(vtype, "请按照SOP标准操作流程执行")

        template = (
            f"【违规内容】在实验{timestamp:.1f}秒处检测到{vtype}，严重等级为{severity}。"
            f"{desc_zh or '系统自动检测到操作不符合SOP规范。'}\n"
            f"【SOP条款】该操作违反了实验室安全操作规程中关于{vtype.replace('_', ' ')}的相关规定。\n"
            f"【潜在风险】{risk_map.get(severity, '需关注')}。\n"
            f"【纠正建议】{correction}。"
        )
        descriptions.append({"alarm": alarm, "llm_description": template})

    return descriptions


def generate_step_summaries(
    step_details: List[Dict[str, Any]],
    alarms: List[Dict[str, Any]],
    openai_client=None,
) -> Dict[int, str]:
    """Generate per-step execution summaries."""
    summaries = {}
    for step in step_details:
        idx = step["index"]
        name = step["step_name"]
        status = step["status"]
        step_alarms = [a for a in alarms if a.get("step_id") == step["step_id"]]

        if openai_client:
            try:
                prompt = (
                    f"为实验室SOP步骤生成执行摘要。\n"
                    f"步骤: {name}\n"
                    f"状态: {status}\n"
                    f"相关违规数: {len(step_alarms)}\n"
                    f"请用2-3句话总结该步骤的执行情况。"
                )
                response = openai_client.chat.completions.create(
                    model="qwen-vl-plus",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                )
                summaries[idx] = response.choices[0].message.content
                continue
            except Exception:
                pass

        # Fallback
        if status == "compliant":
            summaries[idx] = f"步骤{idx}「{name}」已按SOP规范正确执行，未检测到违规行为。"
        elif step_alarms:
            summaries[idx] = f"步骤{idx}「{name}」执行过程中检测到{len(step_alarms)}条违规，需关注改进。"
        else:
            summaries[idx] = f"步骤{idx}「{name}」未在视频中检测到，可能被跳过或未正确执行。"

    return summaries


def generate_executive_summary_text(
    report_data: Dict[str, Any],
    openai_client=None,
) -> str:
    """Generate executive summary text."""
    compliance_ratio = report_data.get("compliance_ratio", 0)
    alarm_count = report_data.get("alarm_count", 0)
    severity_counts = report_data.get("severity_counts", {})
    exp_type = report_data.get("experiment_type_zh", "实验室")

    if openai_client:
        try:
            prompt = (
                f"为实验室SOP合规分析报告撰写执行摘要。\n"
                f"实验类型: {exp_type}\n"
                f"合规得分: {compliance_ratio:.1f}%\n"
                f"违规总数: {alarm_count}\n"
                f"严重/一般/轻微: {severity_counts.get('critical', 0)}/{severity_counts.get('major', 0)}/{severity_counts.get('minor', 0)}\n"
                f"请用3-5句话总结关键发现和总体评估。"
            )
            response = openai_client.chat.completions.create(
                model="doubao-1.5-vision-pro-32k",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception:
            pass

    # Fallback
    grade = "优秀" if compliance_ratio >= 90 else "良好" if compliance_ratio >= 80 else "合格" if compliance_ratio >= 60 else "不合格"
    return (
        f"本次{exp_type}实验视频AI合规分析总体评级为「{grade}」，合规得分{compliance_ratio:.1f}%。"
        f"共检测到{alarm_count}条违规，其中严重{severity_counts.get('critical', 0)}条、"
        f"一般{severity_counts.get('major', 0)}条、轻微{severity_counts.get('minor', 0)}条。"
        f"{'所有关键步骤均按规范执行。' if compliance_ratio >= 80 else '部分步骤存在合规风险，建议重点关注。'}"
    )


def generate_recommendations(
    violation_types: Dict[str, int],
    severity_counts: Dict[str, int],
    openai_client=None,
) -> List[str]:
    """Generate priority-sorted improvement recommendations."""
    if openai_client:
        try:
            prompt = (
                f"根据以下实验室违规数据生成改进建议。\n"
                f"违规类型分布: {violation_types}\n"
                f"严重等级分布: {severity_counts}\n"
                f"请生成3-5条优先级排序的改进建议，每条包含：优先级、建议内容、预期效果。"
            )
            response = openai_client.chat.completions.create(
                model="doubao-1.5-vision-pro-32k",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            text = response.choices[0].message.content
            return [line.strip() for line in text.split("\n") if line.strip()][:5]
        except Exception:
            pass

    # Fallback: rule-based recommendations
    recs = []
    if severity_counts.get("critical", 0) > 0:
        recs.append("【紧急】加强PPE穿戴检查，建议在实验开始前增加强制PPE自检环节。")
    if "missing_ppe" in violation_types:
        recs.append("【高优先级】开展PPE正确佩戴专项培训，确保所有操作人员熟练掌握。")
    if violation_types.get("unsafe_transfer", 0) > 0:
        recs.append("【中优先级】在实验台面标注安全操作区域，明确化学品转移规范位置。")
    if violation_types.get("container_not_closed", 0) > 0:
        recs.append("【中优先级】增加容器管理提醒机制，操作间隙自动检测容器状态。")
    recs.append("【持续改进】建议每周回顾违规数据，针对性优化SOP流程和培训内容。")
    return recs[:5]


# ---------------------------------------------------------------------------
# Enhanced PDF Generation with Reportlab (9 Chapters)
# ---------------------------------------------------------------------------

def _generate_pdf(report_data: Dict[str, Any], output_path: Path) -> bool:
    """Generate 9-chapter FDA/ISO/WHO standard PDF report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm, cm
        from reportlab.lib.colors import HexColor, black, white, grey, Color
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            PageBreak, Image as RLImage, KeepTogether, HRFlowable,
        )
        from reportlab.graphics.shapes import Drawing, Rect, String, Circle
        from reportlab.graphics.charts.piecharts import Pie
        from reportlab.graphics import renderPDF
    except Exception:
        return False

    font_name, cjk_font = _register_fonts()

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
    )

    page_width = A4[0] - 4 * cm

    # Styles
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontName=cjk_font,
                                  fontSize=24, leading=32, alignment=TA_CENTER,
                                  spaceAfter=6 * mm, textColor=HexColor("#0f172a"))
    subtitle_style = ParagraphStyle("ReportSubtitle", parent=styles["Normal"], fontName=cjk_font,
                                     fontSize=13, leading=18, alignment=TA_CENTER,
                                     spaceAfter=12 * mm, textColor=HexColor("#475569"))
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontName=cjk_font,
                               fontSize=18, leading=24, spaceBefore=10 * mm, spaceAfter=5 * mm,
                               textColor=HexColor("#0f172a"), borderWidth=0)
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontName=cjk_font,
                               fontSize=14, leading=20, spaceBefore=6 * mm, spaceAfter=3 * mm,
                               textColor=HexColor("#1e40af"))
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"], fontName=cjk_font,
                               fontSize=12, leading=16, spaceBefore=4 * mm, spaceAfter=2 * mm,
                               textColor=HexColor("#334155"))
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontName=cjk_font,
                                 fontSize=10, leading=16, spaceAfter=3 * mm)
    small_style = ParagraphStyle("Small", parent=styles["Normal"], fontName=cjk_font,
                                  fontSize=8, leading=11, textColor=HexColor("#64748b"))
    score_style = ParagraphStyle("Score", parent=styles["Normal"], fontName=cjk_font,
                                  fontSize=36, leading=44, alignment=TA_CENTER,
                                  textColor=HexColor("#10b981"))

    story = []

    # ═══════════════════════════════════════════════════════════════
    # Chapter 1: Cover Page
    # ═══════════════════════════════════════════════════════════════
    story.append(Spacer(1, 50 * mm))

    # Lab name / logo placeholder
    cover_header = ParagraphStyle("CoverHeader", parent=styles["Normal"], fontName=cjk_font,
                                   fontSize=14, alignment=TA_CENTER, textColor=HexColor("#64748b"))
    story.append(Paragraph("LabSOPGuard 智能实验室合规监控系统", cover_header))
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph("实验SOP合规分析报告", title_style))
    story.append(Paragraph("Standard Operating Procedure Compliance Analysis Report", subtitle_style))
    story.append(Spacer(1, 10 * mm))

    # Separator line
    story.append(HRFlowable(width="60%", thickness=1, color=HexColor("#cbd5e1"),
                             spaceAfter=10 * mm, spaceBefore=5 * mm))

    # Metadata table
    exp_type = f"{report_data['experiment_type_zh']} / {report_data['experiment_type'].replace('_', ' ')}"
    meta_data = [
        ["实验编号 / Experiment ID", report_data["task_id"]],
        ["实验类型 / Type", exp_type],
        ["分析状态 / Status", report_data["status"]],
        ["生成日期 / Date", report_data["generated_at"][:10]],
        ["关键帧数 / Keyframes", str(report_data["keyframe_count"])],
        ["违规总数 / Violations", str(report_data["alarm_count"])],
    ]
    meta_table = Table(meta_data, colWidths=[page_width * 0.40, page_width * 0.60])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), cjk_font), ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#64748b")),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"), ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, HexColor("#cbd5e1")),
    ]))
    story.append(meta_table)

    # Confidential mark
    story.append(Spacer(1, 20 * mm))
    conf_style = ParagraphStyle("Conf", parent=small_style, alignment=TA_CENTER,
                                 textColor=HexColor("#ef4444"), fontSize=9)
    story.append(Paragraph("⚠ 机密文件 / CONFIDENTIAL", conf_style))
    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 2: Executive Summary
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("一、执行摘要 / Executive Summary", h1_style))

    # Compliance Score Display
    compliance_ratio = report_data.get("compliance_ratio", 0)
    passed = report_data.get("compliance_passed", False)
    score_color = "#10b981" if compliance_ratio >= 80 else "#f59e0b" if compliance_ratio >= 60 else "#ef4444"
    grade = "优秀" if compliance_ratio >= 90 else "良好" if compliance_ratio >= 80 else "合格" if compliance_ratio >= 60 else "不合格"
    pass_text = "通过 / PASSED" if passed else "未通过 / FAILED"

    score_data = [[
        Paragraph(f'<font color="{score_color}" size="36"><b>{compliance_ratio:.1f}%</b></font>', body_style),
        Paragraph(f'<font color="{score_color}" size="18"><b>{grade}</b></font>', body_style),
        Paragraph(f'<font color="{score_color}" size="12">{pass_text}</font>', body_style),
    ]]
    score_table = Table(score_data, colWidths=[page_width * 0.33, page_width * 0.33, page_width * 0.34])
    score_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 0), (-1, -1), HexColor("#f0fdf4" if passed else "#fef2f2")),
        ("BOX", (0, 0), (-1, -1), 2, HexColor(score_color)),
        ("TOPPADDING", (0, 0), (-1, -1), 12), ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 5 * mm))

    # Key metrics
    sev = report_data.get("severity_counts", {})
    metrics_data = [
        ["严重 Critical", str(sev.get("critical", 0)), "#dc2626"],
        ["一般 Major", str(sev.get("major", 0)), "#f97316"],
        ["轻微 Minor", str(sev.get("minor", 0)), "#eab308"],
        ["步骤完成", f"{len(report_data.get('observed_steps', []))}/{len(report_data.get('expected_steps', []))}", "#3b82f6"],
    ]
    metrics_row = []
    for label, val, color in metrics_data:
        metrics_row.append(Paragraph(
            f'<font color="{color}" size="11"><b>{val}</b></font><br/><font size="8" color="#64748b">{label}</font>',
            ParagraphStyle("metric", parent=body_style, alignment=TA_CENTER),
        ))
    metrics_table = Table([metrics_row], colWidths=[page_width / 4] * 4)
    metrics_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 5 * mm))

    # LLM-generated executive summary
    llm_texts = report_data.get("llm_texts", {})
    exec_summary = llm_texts.get("executive_summary", "")
    if not exec_summary:
        exec_summary = generate_executive_summary_text(report_data)
    for line in exec_summary.split("\n"):
        line = line.strip()
        if line:
            story.append(Paragraph(line, body_style))

    # AI analysis summary
    if report_data.get("overall_summary"):
        story.append(Spacer(1, 3 * mm))
        story.append(Paragraph("AI分析摘要", h2_style))
        ai_text = report_data["overall_summary"][:2000]
        for line in ai_text.split("\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line, body_style))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 3: Experimental Timeline
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("二、实验时间线 / Experimental Timeline", h1_style))
    story.append(Paragraph("下图展示了实验流程中各SOP步骤的时间分布与违规节点。", body_style))

    # Embed Gantt chart if available
    chart_paths = report_data.get("chart_paths", [])
    gantt_chart = next((c for c in chart_paths if "gantt" in c.lower()), None)
    if gantt_chart and Path(gantt_chart).exists():
        try:
            img = RLImage(gantt_chart, width=page_width, height=page_width * 0.35)
            story.append(img)
        except Exception:
            story.append(Paragraph("[时间线图表生成失败]", small_style))
    else:
        story.append(Paragraph("[时间线图表未生成]", small_style))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 4: Per-Step Detail
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("三、分步骤详情 / Step-by-Step Details", h1_style))

    step_summaries = llm_texts.get("step_summaries", {})
    for step in report_data.get("step_details", []):
        idx = step["index"]
        name = step["step_name"]
        status = step["status"]
        status_color = "#10b981" if status == "compliant" else "#ef4444" if status == "missing" else "#f59e0b"
        status_text = "合规" if status == "compliant" else "缺失" if status == "missing" else "警告"

        story.append(Paragraph(
            f'步骤 {idx}: {name} — <font color="{status_color}">{status_text}</font>', h2_style))

        # Time range
        start_min = step.get("start_sec", 0) / 60
        end_min = step.get("end_sec", 0) / 60
        story.append(Paragraph(f"时间范围: {start_min:.1f} - {end_min:.1f} 分钟", small_style))

        # Step summary (LLM or fallback)
        summary = step_summaries.get(idx, "")
        if not summary:
            if status == "compliant":
                summary = f"步骤{idx}「{name}」已按SOP规范正确执行。"
            else:
                summary = f"步骤{idx}「{name}」未检测到或存在合规问题。"
        story.append(Paragraph(summary, body_style))

        # Annotated frame for this step
        step_frames = step.get("frames", [])
        if step_frames:
            for fp in step_frames[:1]:
                if Path(fp).exists():
                    try:
                        img = RLImage(fp, width=page_width * 0.6, height=page_width * 0.34)
                        story.append(img)
                    except Exception:
                        pass

        story.append(Spacer(1, 5 * mm))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 5: Violation Detail Cards
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("四、违规详细记录 / Violation Details", h1_style))

    violation_descs = llm_texts.get("violation_descriptions", [])
    alarms = report_data.get("alarms", [])
    if alarms:
        for i, alarm in enumerate(alarms[:30], 1):
            vtype = alarm.get("alarm_type", "未知")
            severity = alarm.get("severity", "minor")
            timestamp = alarm.get("timestamp", 0)
            ts_str = f"{timestamp:.1f}s" if isinstance(timestamp, (int, float)) else "N/A"

            sev_color = {"critical": "#dc2626", "major": "#f97316", "minor": "#eab308", "info": "#64748b"}.get(severity, "#64748b")
            sev_label = {"critical": "严重", "major": "一般", "minor": "轻微", "info": "信息"}.get(severity, severity)

            # Violation card header
            card_data = [[
                Paragraph(f'<font color="{sev_color}" size="11"><b>#{i} {vtype}</b></font>', body_style),
                Paragraph(f'<font color="{sev_color}" size="9">{sev_label}</font>', body_style),
                Paragraph(f'<font size="9" color="#64748b">{ts_str}</font>', body_style),
            ]]
            card_table = Table(card_data, colWidths=[page_width * 0.55, page_width * 0.20, page_width * 0.25])
            card_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#fef2f2" if severity == "critical" else "#fff7ed" if severity == "major" else "#fefce8")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor(sev_color)),
                ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]))
            story.append(card_table)

            # LLM or template description
            desc_text = ""
            # violation_descs may be a list of dicts (from generate_violation_descriptions)
            # or a list of strings (from app_integrated pipeline)
            if violation_descs and isinstance(violation_descs[0], dict):
                for vd in violation_descs:
                    if vd.get("alarm") == alarm:
                        desc_text = vd.get("llm_description", "")
                        break
            elif violation_descs and isinstance(violation_descs[0], str):
                desc_text = violation_descs[i - 1] if (i - 1) < len(violation_descs) else ""
            if not desc_text:
                desc_zh = alarm.get("description_zh", "")
                desc_en = alarm.get("description_en", "")
                desc_text = desc_zh or desc_en or f"检测到{vtype}违规。"

            for line in desc_text.split("\n"):
                line = line.strip()
                if line:
                    story.append(Paragraph(line, ParagraphStyle("violation_desc", parent=body_style,
                                                                  leftIndent=8, fontSize=9)))

            # Annotated frame
            frame_path = alarm.get("frame_path", "")
            if frame_path and Path(frame_path).exists():
                try:
                    img = RLImage(frame_path, width=page_width * 0.5, height=page_width * 0.28)
                    story.append(img)
                except Exception:
                    pass

            story.append(Spacer(1, 4 * mm))
    else:
        story.append(Paragraph("未检测到违规 / No violations detected.", body_style))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 6: Statistical Analysis
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("五、统计分析 / Statistical Analysis", h1_style))

    # Embed all available charts
    if chart_paths:
        for cp in chart_paths:
            if Path(cp).exists() and "gantt" not in cp.lower():
                try:
                    img = RLImage(cp, width=page_width * 0.9, height=page_width * 0.4)
                    story.append(img)
                    story.append(Spacer(1, 5 * mm))
                except Exception:
                    pass
    else:
        story.append(Paragraph("[统计图表未生成]", small_style))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 7: Improvement Recommendations
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("六、改进建议 / Improvement Recommendations", h1_style))

    recs = llm_texts.get("recommendations", [])
    if not recs:
        recs = generate_recommendations(
            report_data.get("violation_types", {}),
            report_data.get("severity_counts", {}),
        )

    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}. {rec}", body_style))

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 8: Appendix
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("七、附录 / Appendix", h1_style))

    # Keyframe gallery
    story.append(Paragraph("关键帧图库 / Keyframe Gallery", h2_style))
    keyframe_images = report_data.get("keyframe_images", [])
    if keyframe_images:
        kf_per_row = 2
        kf_w = (page_width - 5 * mm) / kf_per_row
        kf_h = kf_w * 0.56
        for row_start in range(0, min(len(keyframe_images), 20), kf_per_row):
            row_imgs = keyframe_images[row_start:row_start + kf_per_row]
            row_elements = []
            for fp in row_imgs:
                if Path(fp).exists():
                    try:
                        row_elements.append(RLImage(fp, width=kf_w, height=kf_h))
                    except Exception:
                        row_elements.append(Paragraph("[Image unavailable]", small_style))
                else:
                    row_elements.append(Paragraph("[Image unavailable]", small_style))
            while len(row_elements) < kf_per_row:
                row_elements.append("")
            if len(row_elements) == kf_per_row:
                t = Table([row_elements], colWidths=[kf_w] * kf_per_row)
                t.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(t)
                story.append(Spacer(1, 2 * mm))
    else:
        story.append(Paragraph("无关键帧图像 / No keyframe images available.", body_style))

    # Module status
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("模块执行状态 / Module Status", h2_style))
    module_status = report_data.get("module_status", {})
    if module_status:
        mod_data = [["模块 / Module", "状态 / Status", "说明 / Message"]]
        for mod_name, mod_info in module_status.items():
            if isinstance(mod_info, dict):
                status = mod_info.get("status", "unknown")
                msg = mod_info.get("message", "")
                mod_data.append([mod_name, status, Paragraph(msg[:120], small_style)])
        mod_table = Table(mod_data, colWidths=[page_width * 0.20, page_width * 0.15, page_width * 0.65])
        mod_table.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), cjk_font), ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1e40af")),
            ("TEXTCOLOR", (0, 0), (-1, 0), white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, HexColor("#f8fafc")]),
        ]))
        story.append(mod_table)

    story.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════
    # Chapter 9: Signature Page
    # ═══════════════════════════════════════════════════════════════
    story.append(Paragraph("八、签字页 / Signature Page", h1_style))
    story.append(Spacer(1, 15 * mm))

    sig_data = [
        ["系统生成标识", "LabSOPGuard v2.0 智能分析系统"],
        ["报告生成时间", report_data["generated_at"][:19].replace("T", " ")],
        ["分析模型版本", "YOLO26s-pose + Chemistry Analyzer v1.0"],
        ["", ""],
        ["审核人 / Reviewed by", "________________________"],
        ["审核日期 / Date", "________________________"],
        ["", ""],
        ["批准人 / Approved by", "________________________"],
        ["批准日期 / Date", "________________________"],
    ]
    sig_table = Table(sig_data, colWidths=[page_width * 0.35, page_width * 0.65])
    sig_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), cjk_font), ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), HexColor("#64748b")),
        ("ALIGN", (0, 0), (0, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, HexColor("#e2e8f0")),
    ]))
    story.append(sig_table)

    # Footer
    story.append(Spacer(1, 20 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=HexColor("#cbd5e1"), spaceAfter=5 * mm))
    story.append(Paragraph(
        "本报告由 LabSOPGuard 实验室SOP合规智能监控系统自动生成，仅供内部参考。",
        ParagraphStyle("FooterZH", parent=small_style, alignment=TA_CENTER),
    ))
    story.append(Paragraph(
        "Auto-generated by LabSOPGuard SOP Compliance Intelligent Monitoring System.",
        ParagraphStyle("FooterEN", parent=small_style, alignment=TA_CENTER),
    ))

    # Build PDF
    try:
        doc.build(story)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# TXT Fallback (Structured Bilingual)
# ---------------------------------------------------------------------------

def _generate_txt_fallback(report_data: Dict[str, Any], output_path: Path) -> Path:
    """Generate structured bilingual TXT report as PDF fallback."""
    lines = []
    lines.append("=" * 70)
    lines.append("实验SOP合规分析报告 / SOP Compliance Analysis Report")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"实验编号: {report_data['task_id']}")
    lines.append(f"实验类型: {report_data['experiment_type_zh']} / {report_data['experiment_type']}")
    lines.append(f"合规得分: {report_data.get('compliance_ratio', 0):.1f}%")
    lines.append(f"违规总数: {report_data['alarm_count']}")
    lines.append(f"生成时间: {report_data['generated_at'][:19]}")
    lines.append("")

    # Executive summary
    lines.append("-" * 70)
    lines.append("执行摘要 / Executive Summary")
    lines.append("-" * 70)
    llm_exec = report_data.get("llm_texts", {}).get("executive_summary", "")
    if llm_exec:
        lines.append(llm_exec)
    else:
        lines.append(f"合规得分 {report_data.get('compliance_ratio', 0):.1f}%，检测到{report_data['alarm_count']}条违规。")
    lines.append("")

    # Steps
    lines.append("-" * 70)
    lines.append("步骤合规性 / Step Compliance")
    lines.append("-" * 70)
    for step in report_data.get("step_details", []):
        status = "[OK]" if step["status"] == "compliant" else "[MISSING]"
        lines.append(f"  {step['index']}. {status} {step['step_name']}")
    lines.append("")

    # Alarms
    lines.append("-" * 70)
    lines.append("违规详情 / Violation Details")
    lines.append("-" * 70)
    for i, alarm in enumerate(report_data.get("alarms", [])[:30], 1):
        sev = alarm.get("severity", "")
        ts = alarm.get("timestamp", 0)
        ts_str = f"{ts:.1f}s" if isinstance(ts, (int, float)) else "N/A"
        vtype = alarm.get("alarm_type", "unknown")
        desc = alarm.get("description_zh", "") or alarm.get("description_en", "")
        lines.append(f"  [{sev}] {ts_str} - {vtype}")
        if desc:
            lines.append(f"    {desc}")
        lines.append("")

    # Recommendations
    lines.append("-" * 70)
    lines.append("改进建议 / Recommendations")
    lines.append("-" * 70)
    recs = generate_recommendations(
        report_data.get("violation_types", {}),
        report_data.get("severity_counts", {}),
    )
    for i, rec in enumerate(recs, 1):
        lines.append(f"  {i}. {rec}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("本报告由 LabSOPGuard 系统自动生成。")
    lines.append("=" * 70)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def generate_integrated_pdf(report_data: Dict[str, Any], output_path: str | Path) -> str:
    """Generate 9-chapter FDA/ISO/WHO standard PDF report.

    Falls back to structured TXT if reportlab is unavailable.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    pdf_success = _generate_pdf(report_data, out)
    if pdf_success:
        return str(out)

    txt_path = out.with_suffix(".txt")
    return str(_generate_txt_fallback(report_data, txt_path))


def generate_report_from_task(
    task_id: str,
    outputs_root: Path,
) -> str:
    """Convenience function: generate report from task outputs directory."""
    task_dirs = sorted(outputs_root.glob(f"*{task_id}*"), key=lambda p: p.name, reverse=True)
    if not task_dirs:
        raise FileNotFoundError(f"No output directory found for task {task_id}")

    output_dir = task_dirs[0]

    def _load_json(name: str) -> Dict[str, Any]:
        p = output_dir / name
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    task_result = _load_json("task_result.json")
    alarm_log = _load_json("alarm_log.json")
    scene_profile = _load_json("scene_profile.json")
    chemistry_analysis = _load_json("chemistry_analysis.json")
    keyframe_meta = _load_json("part1_keyframes.json").get("keyframes", [])

    # Load chart paths
    chart_dir = output_dir / "charts"
    chart_paths = []
    if chart_dir.exists():
        chart_paths = sorted(str(p) for p in chart_dir.glob("chart_*.png"))

    # Load annotated frames
    annotated_frames = sorted(str(p) for p in output_dir.glob("*_annotated.jpg"))

    # Load LLM texts if available
    llm_path = output_dir / "llm_report_texts.json"
    llm_texts = {}
    if llm_path.exists():
        try:
            llm_texts = json.loads(llm_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    report_data = _build_report_data(
        task_result=task_result,
        scene_profile=scene_profile,
        chemistry_analysis=chemistry_analysis,
        alarm_log=alarm_log,
        keyframe_meta=keyframe_meta,
        output_dir=output_dir,
        chart_paths=chart_paths,
        annotated_frame_paths=annotated_frames,
        llm_texts=llm_texts,
    )

    pdf_path = output_dir / "integrated_analysis_report.pdf"
    return generate_integrated_pdf(report_data, pdf_path)
