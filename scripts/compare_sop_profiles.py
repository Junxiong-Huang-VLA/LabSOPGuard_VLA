from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


PROFILE_RULES = {
    "strict": "configs/sop/rules.yaml",
    "mvp": "configs/sop/rules_mvp.yaml",
}


def _run(cmd: List[str], cwd: Path, dry_run: bool) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return 0
    p = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(p.returncode)


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except Exception:
        return 0


def _profile_block(profile: str, analysis: Dict[str, Any] | None, diagnostics: Dict[str, Any] | None) -> Dict[str, Any]:
    a = analysis or {}
    d = diagnostics or {}
    return {
        "profile": profile,
        "total_samples": _to_int(a.get("total_samples", 0)),
        "samples_with_violations": _to_int(a.get("samples_with_violations", 0)),
        "total_violations": _to_int(a.get("total_violations", 0)),
        "rule_distribution": a.get("rule_distribution", {}),
        "missing_ppe_item_distribution": d.get("missing_ppe_item_distribution", {}),
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# SOP Profile Compare Report")
    lines.append("")
    lines.append(f"- manifest_csv: {report.get('manifest_csv', '')}")
    lines.append(f"- valid_only: {report.get('valid_only', False)}")
    lines.append("")

    for key in ["strict", "mvp"]:
        row = report.get(key, {})
        lines.append(f"## {key.upper()}")
        lines.append(f"- total_samples: {row.get('total_samples', 0)}")
        lines.append(f"- samples_with_violations: {row.get('samples_with_violations', 0)}")
        lines.append(f"- total_violations: {row.get('total_violations', 0)}")
        rd = row.get("rule_distribution", {})
        lines.append("- rule_distribution:")
        if isinstance(rd, dict) and rd:
            for rk, rv in sorted(rd.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
                lines.append(f"  - {rk}: {rv}")
        else:
            lines.append("  - none")
        mp = row.get("missing_ppe_item_distribution", {})
        lines.append("- missing_ppe_item_distribution:")
        if isinstance(mp, dict) and mp:
            for mk, mv in sorted(mp.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
                lines.append(f"  - {mk}: {mv}")
        else:
            lines.append("  - none")
        lines.append("")

    delta = report.get("delta", {})
    lines.append("## DELTA (mvp - strict)")
    lines.append(f"- total_violations_delta: {delta.get('total_violations_delta', 0)}")
    lines.append(f"- samples_with_violations_delta: {delta.get('samples_with_violations_delta', 0)}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict/mvp profile compare and generate report")
    parser.add_argument("--manifest-csv", default="data/interim/video_manifest.csv")
    parser.add_argument("--valid-only", action="store_true")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--alert-config", default="configs/alerts/alerting.yaml")
    parser.add_argument("--monitor-out-root", default="outputs/predictions/profile_compare")
    parser.add_argument("--report-out-root", default="outputs/reports/profile_compare")
    parser.add_argument("--out-json", default="outputs/reports/profile_compare/compare_profiles.json")
    parser.add_argument("--out-md", default="outputs/reports/profile_compare/compare_profiles.md")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    py = sys.executable

    monitor_root = project_root / args.monitor_out_root
    report_root = project_root / args.report_out_root
    monitor_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    valid_flag = ["--valid-only"] if args.valid_only else []

    for profile in ["strict", "mvp"]:
        rules = PROFILE_RULES[profile]
        batch_monitor_dir = monitor_root / profile / "batch_monitor"
        export_json = report_root / f"export_summary_{profile}.json"
        export_csv = report_root / f"export_summary_{profile}.csv"
        export_events_jsonl = report_root / f"export_events_{profile}.jsonl"
        export_events_csv = report_root / f"export_events_{profile}.csv"
        analysis_json = report_root / f"violation_analysis_{profile}.json"
        analysis_md = report_root / f"violation_analysis_{profile}.md"
        diagnostics_json = report_root / f"violation_diagnostics_{profile}.json"
        diagnostics_md = report_root / f"violation_diagnostics_{profile}.md"

        cmd_monitor = [
            py,
            "scripts/run_monitor.py",
            "--manifest-csv",
            args.manifest_csv,
            "--rules",
            rules,
            "--alert-config",
            args.alert_config,
            *valid_flag,
            "--max-frames",
            str(args.max_frames),
            "--target-fps",
            str(args.target_fps),
            "--batch-output-dir",
            str(batch_monitor_dir),
        ]
        if _run(cmd_monitor, cwd=project_root, dry_run=args.dry_run) != 0:
            return 1

        cmd_export = [
            py,
            "scripts/export_results.py",
            "--manifest-csv",
            args.manifest_csv,
            "--rules",
            rules,
            "--alert-config",
            args.alert_config,
            *valid_flag,
            "--max-frames",
            str(args.max_frames),
            "--target-fps",
            str(args.target_fps),
            "--out-json",
            str(export_json),
            "--out-csv",
            str(export_csv),
            "--out-events-jsonl",
            str(export_events_jsonl),
            "--out-events-csv",
            str(export_events_csv),
        ]
        if _run(cmd_export, cwd=project_root, dry_run=args.dry_run) != 0:
            return 1

        cmd_analysis = [
            py,
            "scripts/analyze_violations.py",
            "--export-summary-json",
            str(export_json),
            "--out-json",
            str(analysis_json),
            "--out-md",
            str(analysis_md),
        ]
        if _run(cmd_analysis, cwd=project_root, dry_run=args.dry_run) != 0:
            return 1

        cmd_diag = [
            py,
            "scripts/analyze_violation_diagnostics.py",
            "--batch-monitor-dir",
            str(batch_monitor_dir),
            "--rules",
            rules,
            "--out-json",
            str(diagnostics_json),
            "--out-md",
            str(diagnostics_md),
        ]
        if _run(cmd_diag, cwd=project_root, dry_run=args.dry_run) != 0:
            return 1

    if args.dry_run:
        print("[INFO] dry-run mode: compare report generation skipped.")
        return 0

    strict_a = _load_json(report_root / "violation_analysis_strict.json")
    strict_d = _load_json(report_root / "violation_diagnostics_strict.json")
    mvp_a = _load_json(report_root / "violation_analysis_mvp.json")
    mvp_d = _load_json(report_root / "violation_diagnostics_mvp.json")

    strict_block = _profile_block("strict", strict_a, strict_d)
    mvp_block = _profile_block("mvp", mvp_a, mvp_d)

    report = {
        "manifest_csv": args.manifest_csv,
        "valid_only": bool(args.valid_only),
        "strict": strict_block,
        "mvp": mvp_block,
        "delta": {
            "total_violations_delta": mvp_block["total_violations"] - strict_block["total_violations"],
            "samples_with_violations_delta": mvp_block["samples_with_violations"] - strict_block["samples_with_violations"],
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(report), encoding="utf-8")

    print(f"compare json: {out_json}")
    print(f"compare md: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
