from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

_log = logging.getLogger(__name__)


def _write_text_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def generate_compliance_report(report_data: Dict[str, Any], output_pdf_path: str) -> str:
    """Generate PDF if reportlab exists; fallback to txt report."""
    out = Path(output_pdf_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"Report Title: {report_data.get('title', 'Lab SOP Compliance Report')}",
        f"Session ID: {report_data.get('session_id', 'unknown')}",
        f"Compliance Ratio: {report_data.get('compliance_ratio', 0.0):.2f}",
        f"Violation Count: {len(report_data.get('violations', []))}",
        "",
        "Violations:",
    ]
    for v in report_data.get("violations", []):
        summary_lines.append(f"- [{v.get('severity','na')}] {v.get('rule_id','na')}: {v.get('message','')}")
    content = "\n".join(summary_lines)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(out), pagesize=A4)
        text = c.beginText(40, 800)
        for line in summary_lines:
            text.textLine(line)
        c.drawText(text)
        c.save()
        return str(out)
    except ImportError:
        _log.warning("reportlab not installed; falling back to plain-text report at %s", out.with_suffix('.txt'))
        fallback = out.with_suffix('.txt')
        _write_text_report(fallback, content)
        return str(fallback)
    except Exception as exc:
        _log.warning("PDF generation failed (%s); falling back to plain-text report.", exc)
        fallback = out.with_suffix('.txt')
        _write_text_report(fallback, content)
        return str(fallback)
