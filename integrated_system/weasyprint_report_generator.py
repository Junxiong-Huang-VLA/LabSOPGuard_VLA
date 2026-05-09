"""
WeasyPrint + Jinja2 Professional PDF Report Generator for LabSOPGuard.

Renders HTML templates with Jinja2, then converts to PDF via WeasyPrint.
Falls back to reportlab-based generator when WeasyPrint is unavailable.

Template: integrated_system/templates/report_template.html
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _build_template_context(
    task_result: Dict[str, Any],
    scene_profile: Dict[str, Any],
    alarm_log: Dict[str, Any],
    chemistry_analysis: Optional[Dict[str, Any]],
    charts: Dict[str, str],
    keyframe_images: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    """Build Jinja2 template context from analysis results."""
    task_id = task_result.get("task_id", "unknown")
    status = task_result.get("status", "unknown")
    video_path = task_result.get("video_path", "unknown")

    exp_type = scene_profile.get("experiment_type", "general_lab_operation")
    exp_type_zh = scene_profile.get("experiment_type_zh", "一般实验室操作")

    alarms = alarm_log.get("alarms", [])
    alarm_count = len(alarms)
    expected_steps = alarm_log.get("expected_steps", [])
    observed_steps = alarm_log.get("observed_steps", [])
    compliance = alarm_log.get("compliance_summary", {})

    # Count severities
    critical_count = len([a for a in alarms if a.get("severity") == "critical"])
    high_count = len([a for a in alarms if a.get("severity") == "high"])

    total_expected = compliance.get("total_expected", len(expected_steps))
    total_observed = compliance.get("total_observed", len(observed_steps))
    compliance_ratio = total_observed / max(total_expected, 1)

    # Build step status list
    observed_set = set(observed_steps)
    steps = []
    for i, step_id in enumerate(expected_steps):
        step_info = _find_step_info(step_id, scene_profile)
        is_observed = step_id in observed_set
        steps.append({
            "step_id": step_id,
            "name_zh": step_info.get("name_zh", step_id),
            "name_en": step_info.get("name_en", step_id.replace("_", " ")),
            "status_zh": "已完成" if is_observed else "未检测到",
            "status_class": "status-ok" if is_observed else "status-missing",
            "notes": "",
        })

    # AI summary
    overall_summary = ""
    summary_path = output_dir / "overall_summary.txt"
    if summary_path.exists():
        overall_summary = summary_path.read_text(encoding="utf-8")

    # Summary text (ZH + EN)
    summary_zh = (
        f"本次分析针对{exp_type_zh}实验视频进行智能分析。"
        f"共提取{task_result.get('keyframe_count', 0)}个关键帧，"
        f"识别出{total_observed}/{total_expected}个实验步骤，"
        f"检测到{alarm_count}条报警信息，"
        f"其中严重{critical_count}条、高风险{high_count}条。"
    )
    summary_en = (
        f"This report analyzes a {exp_type.replace('_', ' ')} experiment video. "
        f"{task_result.get('keyframe_count', 0)} keyframes were extracted, "
        f"{total_observed}/{total_expected} steps identified, "
        f"with {alarm_count} alarms detected "
        f"({critical_count} critical, {high_count} high)."
    )

    # Module status
    module_status = task_result.get("module_status", {})
    modules = []
    for mod_name, mod_info in module_status.items():
        if isinstance(mod_info, dict):
            modules.append({
                "name": mod_name,
                "status": mod_info.get("status", "unknown"),
                "message": mod_info.get("message", ""),
            })

    # Timeline events (from alarms + observations)
    timeline_events = []
    for a in alarms:
        timeline_events.append({
            "timestamp": a.get("timestamp", 0),
            "step_id": a.get("alarm_type", ""),
            "event_type": "alarm",
            "description": a.get("description_en", a.get("description_zh", "")),
        })
    if chemistry_analysis:
        for obs in chemistry_analysis.get("observations", []):
            timeline_events.append({
                "timestamp": obs.get("timestamp_sec", 0),
                "step_id": obs.get("expected_step_id", ""),
                "event_type": "observation",
                "description": obs.get("operation_description_en", ""),
            })
    timeline_events.sort(key=lambda e: e.get("timestamp", 0))

    # Keyframe images
    kf_images = []
    for i, kf in enumerate(keyframe_images[:8]):
        kf_images.append({
            "path": kf.get("path", ""),
            "index": i + 1,
            "timestamp": kf.get("timestamp", 0),
        })

    return {
        "task_id": task_id,
        "status": status,
        "video_path": video_path,
        "experiment_type": exp_type,
        "experiment_type_zh": exp_type_zh,
        "generated_at": datetime.utcnow().isoformat()[:19].replace("T", " "),
        "keyframe_count": task_result.get("keyframe_count", 0),
        "alarm_count": alarm_count,
        "critical_count": critical_count,
        "high_count": high_count,
        "compliance_ratio": compliance_ratio,
        "summary_zh": summary_zh,
        "summary_en": summary_en,
        "overall_summary": overall_summary,
        "steps": steps,
        "alarms": [
            {
                "alarm_type": a.get("alarm_type", "unknown"),
                "severity": a.get("severity", "medium"),
                "severity_en": a.get("severity_en", a.get("severity", "")),
                "timestamp": a.get("timestamp", -1),
                "description_zh": a.get("description_zh", ""),
                "description_en": a.get("description_en", ""),
            }
            for a in alarms
        ],
        "timeline_events": timeline_events,
        "keyframe_images": kf_images,
        "modules": modules,
        "charts": charts,
    }


def _find_step_info(step_id: str, scene_profile: Dict[str, Any]) -> Dict[str, str]:
    """Find step name info from scene profile."""
    for step in scene_profile.get("expected_steps", []):
        if step.get("step_id") == step_id:
            return step
    return {"name_zh": step_id, "name_en": step_id.replace("_", " ")}


def generate_weasyprint_pdf(
    context: Dict[str, Any],
    output_path: Path,
    template_path: Optional[Path] = None,
) -> bool:
    """Generate PDF using WeasyPrint + Jinja2.

    Returns True on success, False on failure.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
        from weasyprint import HTML
    except ImportError:
        return False

    if template_path is None:
        template_path = Path(__file__).parent / "templates" / "report_template.html"

    if not template_path.exists():
        return False

    # Render HTML template
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    html_content = template.render(**context)

    # Save intermediate HTML (for debugging)
    html_path = output_path.with_suffix(".html")
    html_path.write_text(html_content, encoding="utf-8")

    # Convert to PDF
    try:
        HTML(string=html_content, base_url=str(template_path.parent)).write_pdf(str(output_path))
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def generate_report_weasyprint(
    task_result: Dict[str, Any],
    scene_profile: Dict[str, Any],
    alarm_log: Dict[str, Any],
    output_dir: Path,
    chemistry_analysis: Optional[Dict[str, Any]] = None,
    charts: Optional[Dict[str, str]] = None,
    keyframe_images: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate professional PDF report using WeasyPrint + Jinja2.

    Falls back to reportlab generator if WeasyPrint unavailable.

    Returns:
        Path to generated report file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = charts or {}
    keyframe_images = keyframe_images or []

    # Build template context
    context = _build_template_context(
        task_result=task_result,
        scene_profile=scene_profile,
        alarm_log=alarm_log,
        chemistry_analysis=chemistry_analysis,
        charts=charts,
        keyframe_images=keyframe_images,
        output_dir=output_dir,
    )

    pdf_path = output_dir / "integrated_analysis_report.pdf"

    # Try WeasyPrint
    success = generate_weasyprint_pdf(context, pdf_path)
    if success:
        return str(pdf_path)

    # Fallback to reportlab
    from integrated_system.scientific_report_generator import (
        _build_report_data,
        generate_integrated_pdf,
    )

    report_data = _build_report_data(
        task_result=task_result,
        scene_profile=scene_profile,
        chemistry_analysis=chemistry_analysis or {},
        alarm_log=alarm_log,
        keyframe_meta=[],
        output_dir=output_dir,
    )

    return generate_integrated_pdf(report_data, pdf_path)
