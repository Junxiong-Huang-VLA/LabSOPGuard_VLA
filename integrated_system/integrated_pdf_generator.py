from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _build_lines(report_data: Dict[str, Any]) -> List[str]:
    outputs = report_data.get("outputs", {}) if isinstance(report_data.get("outputs"), dict) else {}
    overall_summary = str(report_data.get("overall_summary", "") or "").strip()
    if not overall_summary:
        overall_summary = "No overall summary generated."

    lines: List[str] = [
        "Integrated Analysis Report",
        "",
        f"Generated At: {datetime.utcnow().isoformat()}Z",
        f"Task ID: {report_data.get('task_id', 'unknown')}",
        f"Input Video: {report_data.get('video_path', 'unknown')}",
        f"Status: {report_data.get('status', 'unknown')}",
        "",
        "Hand Detection Summary:",
        json.dumps(report_data.get("hand_summary", {}), ensure_ascii=False),
        "",
        "Keyframe Analysis:",
        overall_summary,
        "",
        "Step Check:",
        f"Alarm Count: {len(report_data.get('alarms', []))}",
    ]

    for a in report_data.get("alarms", []):
        lines.append(
            f"- [{a.get('severity','na')}] {a.get('alarm_type','na')} frame={a.get('frame','na')} {a.get('description','')}"
        )

    lines.extend(["", "Output Files:"])
    for k, v in outputs.items():
        lines.append(f"- {k}: {v}")
    return lines


def _register_font():
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except Exception:
        return None

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        pass

    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for path in candidates:
        if path.exists():
            try:
                pdfmetrics.registerFont(TTFont("IntegratedCN", str(path)))
                return "IntegratedCN"
            except Exception:
                continue
    return None


def generate_integrated_pdf(report_data: Dict[str, Any], output_path: str | Path) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = _build_lines(report_data)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(out), pagesize=A4)
        width, height = A4
        y = height - 40
        font_name = _register_font() or "Helvetica"
        c.setFont(font_name, 11)

        for line in lines:
            if y < 40:
                c.showPage()
                c.setFont(font_name, 11)
                y = height - 40
            c.drawString(40, y, str(line)[:130])
            y -= 18

        c.save()
        return str(out)
    except Exception:
        fallback = out.with_suffix(".txt")
        lines.append("")
        lines.append("NOTE: PDF generation failed, exported as TXT fallback.")
        fallback.write_text("\n".join(lines), encoding="utf-8")
        return str(fallback)
