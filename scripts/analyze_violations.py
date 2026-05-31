from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def analyze(export_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rule_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    sample_rows: List[Dict[str, Any]] = []

    for row in export_rows:
        sample_id = str(row.get("sample_id", ""))
        status = row.get("status", {}) if isinstance(row.get("status", {}), dict) else {}
        compliance_ratio = _to_float(status.get("compliance_ratio", 0.0), 0.0)
        violations = row.get("violations", []) if isinstance(row.get("violations", []), list) else []

        sample_rule_counter: Counter[str] = Counter()
        for v in violations:
            if not isinstance(v, dict):
                continue
            rule_id = str(v.get("rule_id", "unknown"))
            severity = str(v.get("severity", "unknown"))
            sample_rule_counter[rule_id] += 1
            rule_counter[rule_id] += 1
            severity_counter[severity] += 1

        sample_rows.append(
            {
                "sample_id": sample_id,
                "compliance_ratio": compliance_ratio,
                "violation_count": len(violations),
                "rule_breakdown": dict(sample_rule_counter),
                "completed_steps": status.get("completed_steps", []),
                "pending_steps": status.get("pending_steps", []),
            }
        )

    sample_rows.sort(key=lambda x: (-int(x["violation_count"]), x["sample_id"]))

    return {
        "total_samples": len(export_rows),
        "samples_with_violations": sum(1 for r in sample_rows if int(r["violation_count"]) > 0),
        "total_violations": int(sum(int(r["violation_count"]) for r in sample_rows)),
        "rule_distribution": dict(rule_counter),
        "severity_distribution": dict(severity_counter),
        "samples": sample_rows,
    }


def render_markdown(analysis: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Violation Analysis Report")
    lines.append("")
    lines.append(f"- total_samples: {analysis.get('total_samples', 0)}")
    lines.append(f"- samples_with_violations: {analysis.get('samples_with_violations', 0)}")
    lines.append(f"- total_violations: {analysis.get('total_violations', 0)}")
    lines.append("")

    lines.append("## Rule Distribution")
    rd = analysis.get("rule_distribution", {})
    if isinstance(rd, dict) and rd:
        for k, v in sorted(rd.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Severity Distribution")
    sd = analysis.get("severity_distribution", {})
    if isinstance(sd, dict) and sd:
        for k, v in sorted(sd.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Per Sample")
    samples = analysis.get("samples", [])
    if isinstance(samples, list) and samples:
        for row in samples:
            sid = row.get("sample_id", "")
            vc = row.get("violation_count", 0)
            cr = row.get("compliance_ratio", 0.0)
            lines.append(f"- {sid}: violations={vc}, compliance_ratio={cr:.3f}")
    else:
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze exported SOP violations")
    parser.add_argument("--export-summary-json", default="outputs/predictions/export_summary.json")
    parser.add_argument("--out-json", default="outputs/reports/violation_analysis.json")
    parser.add_argument("--out-md", default="outputs/reports/violation_analysis.md")
    args = parser.parse_args()

    data = _load_json(args.export_summary_json)
    if not isinstance(data, list):
        raise ValueError("export summary should be a JSON array")

    analysis = analyze(data)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render_markdown(analysis), encoding="utf-8")

    print(f"analysis json: {out_json}")
    print(f"analysis md: {out_md}")
    print(f"samples: {analysis.get('total_samples', 0)}")
    print(f"violations: {analysis.get('total_violations', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
