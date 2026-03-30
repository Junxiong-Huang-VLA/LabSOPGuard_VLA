from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


STAGES = ["scan", "infer", "monitor", "export", "analyze", "audit"]
SOP_PROFILE_TO_RULES = {
    "strict": "configs/sop/rules.yaml",
    "mvp": "configs/sop/rules_mvp.yaml",
}


def _run(cmd: List[str], cwd: Path, dry_run: bool = False) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    if dry_run:
        return 0
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    return int(result.returncode)


def _normalize_stage(value: str | None, fallback: str) -> str:
    if not value:
        return fallback
    text = value.strip().lower()
    if text not in STAGES:
        raise ValueError(f"Invalid stage '{value}', choose from: {', '.join(STAGES)}")
    return text


def _build_stage_plan(
    from_stage: str,
    to_stage: str,
    skip_scan: bool,
    skip_infer: bool,
    skip_monitor: bool,
    skip_export: bool,
    skip_analyze: bool,
    skip_audit: bool,
) -> List[str]:
    start = STAGES.index(from_stage)
    end = STAGES.index(to_stage)
    if start > end:
        raise ValueError(f"--from-stage '{from_stage}' cannot be after --to-stage '{to_stage}'.")
    selected = STAGES[start : end + 1]
    skip_map = {
        "scan": skip_scan,
        "infer": skip_infer,
        "monitor": skip_monitor,
        "export": skip_export,
        "analyze": skip_analyze,
        "audit": skip_audit,
    }
    return [s for s in selected if not skip_map[s]]


def _ensure_runtime_dependencies(stage_plan: List[str]) -> None:
    need_ultralytics = any(s in stage_plan for s in ("infer", "monitor", "export"))
    if not need_ultralytics:
        return
    if importlib.util.find_spec("ultralytics") is None:
        raise RuntimeError(
            "Missing dependency 'ultralytics' in current interpreter. "
            "Use conda env python, e.g.: "
            "'conda run -n LabSOPGuard python scripts/run_0to1_pipeline.py ...'"
        )


def _resolve_rules_path(rules: str, sop_profile: str) -> str:
    profile = str(sop_profile or "").strip().lower()
    if profile:
        if profile not in SOP_PROFILE_TO_RULES:
            raise ValueError(f"Invalid --sop-profile '{sop_profile}', choose from: strict, mvp")
        return SOP_PROFILE_TO_RULES[profile]
    return rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="0-to-1 pipeline: scan -> infer -> monitor -> export"
    )
    parser.add_argument("--dataset-root", default=r"D:\labdata")
    parser.add_argument("--rules", default="configs/sop/rules.yaml")
    parser.add_argument("--alert-config", default="configs/alerts/alerting.yaml")
    parser.add_argument("--sop-profile", default="", help="Optional: strict|mvp")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--valid-only", action="store_true", default=True)
    parser.add_argument("--all-samples", action="store_true", help="Disable --valid-only filter.")

    parser.add_argument("--scan-max-frames-per-video", type=int, default=5)
    parser.add_argument("--scan-interval-sec", type=float, default=1.0)
    parser.add_argument("--scan-frames-root", default="data/interim/frames")
    parser.add_argument("--scan-manifest-csv", default="data/interim/video_manifest.csv")
    parser.add_argument("--scan-report-json", default="outputs/reports/video_scan_report.json")

    parser.add_argument("--infer-max-frames", type=int, default=120)
    parser.add_argument("--infer-target-fps", type=float, default=10.0)
    parser.add_argument("--infer-out-dir", default="outputs/predictions/batch_infer")

    parser.add_argument("--monitor-max-frames", type=int, default=120)
    parser.add_argument("--monitor-target-fps", type=float, default=10.0)
    parser.add_argument("--monitor-out-dir", default="outputs/predictions/batch_monitor")

    parser.add_argument("--export-max-frames", type=int, default=120)
    parser.add_argument("--export-target-fps", type=float, default=10.0)
    parser.add_argument("--export-out-json", default="outputs/predictions/export_summary.json")
    parser.add_argument("--export-out-csv", default="outputs/reports/export_summary.csv")
    parser.add_argument("--export-events-jsonl", default="outputs/predictions/export_events.jsonl")
    parser.add_argument("--export-events-csv", default="outputs/reports/export_events.csv")
    parser.add_argument("--analyze-in-json", default="outputs/predictions/export_summary.json")
    parser.add_argument("--analyze-out-json", default="outputs/reports/violation_analysis.json")
    parser.add_argument("--analyze-out-md", default="outputs/reports/violation_analysis.md")
    parser.add_argument("--diagnostics-out-json", default="outputs/reports/violation_diagnostics.json")
    parser.add_argument("--diagnostics-out-md", default="outputs/reports/violation_diagnostics.md")
    parser.add_argument("--compare-profiles", action="store_true")
    parser.add_argument("--compare-out-json", default="outputs/reports/profile_compare/compare_profiles.json")
    parser.add_argument("--compare-out-md", default="outputs/reports/profile_compare/compare_profiles.md")
    parser.add_argument("--build-hardcase-pack", action="store_true")
    parser.add_argument("--hardcase-rule-id", default="missing_ppe")
    parser.add_argument("--hardcase-context", type=int, default=3)
    parser.add_argument("--hardcase-out-dir", default="data/interim/hardcases/ppe_missing")
    parser.add_argument("--hardcase-out-csv", default="data/interim/hardcases/ppe_missing_manifest.csv")

    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-monitor", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-analyze", action="store_true")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--from-stage", default="scan", help="scan|infer|monitor|export|analyze|audit")
    parser.add_argument("--to-stage", default="audit", help="scan|infer|monitor|export|analyze|audit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument(
        "--run-meta-json",
        default="outputs/reports/run_0to1_meta.json",
        help="Path to save stage execution metadata.",
    )
    parser.add_argument("--audit-batch-monitor-dir", default="outputs/predictions/batch_monitor")
    parser.add_argument("--audit-out-dir", default="outputs/reports/audit_assets")
    parser.add_argument("--audit-max-snaps-per-sample", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all_samples:
        args.valid_only = False

    from_stage = _normalize_stage(args.from_stage, "scan")
    to_stage = _normalize_stage(args.to_stage, "audit")
    resolved_rules = _resolve_rules_path(args.rules, args.sop_profile)
    stage_plan = _build_stage_plan(
        from_stage=from_stage,
        to_stage=to_stage,
        skip_scan=args.skip_scan,
        skip_infer=args.skip_infer,
        skip_monitor=args.skip_monitor,
        skip_export=args.skip_export,
        skip_analyze=args.skip_analyze,
        skip_audit=args.skip_audit,
    )
    _ensure_runtime_dependencies(stage_plan)

    project_root = Path(__file__).resolve().parents[1]
    python = sys.executable
    start_dt = datetime.now()
    print(f"[INFO] 0-to-1 pipeline started at {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] stages: {', '.join(stage_plan) if stage_plan else '(none)'}")

    valid_only_flag = ["--valid-only"] if args.valid_only else []
    recursive_flag = ["--recursive"] if args.recursive else []
    stage_records: List[Dict[str, object]] = []

    if "scan" in stage_plan:
        cmd = [
            python,
            "scripts/scan_and_extract_frames.py",
            "--dataset-root",
            args.dataset_root,
            *recursive_flag,
            "--max-frames-per-video",
            str(args.scan_max_frames_per_video),
            "--interval-sec",
            str(args.scan_interval_sec),
            "--manifest-csv",
            args.scan_manifest_csv,
            "--report-json",
            args.scan_report_json,
            "--frames-root",
            args.scan_frames_root,
            "--verbose",
        ]
        t0 = time.perf_counter()
        code = _run(cmd, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {"stage": "scan", "return_code": code, "duration_sec": round(time.perf_counter() - t0, 3)}
        )
        if code != 0:
            raise RuntimeError(f"Stage failed: scan (exit={code})")

    if "infer" in stage_plan:
        cmd = [
            python,
            "scripts/infer.py",
            "--manifest-csv",
            args.scan_manifest_csv,
            "--rules",
            resolved_rules,
            "--alert-config",
            args.alert_config,
            *valid_only_flag,
            "--max-frames",
            str(args.infer_max_frames),
            "--target-fps",
            str(args.infer_target_fps),
            "--batch-output-dir",
            args.infer_out_dir,
        ]
        t0 = time.perf_counter()
        code = _run(cmd, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {"stage": "infer", "return_code": code, "duration_sec": round(time.perf_counter() - t0, 3)}
        )
        if code != 0:
            raise RuntimeError(f"Stage failed: infer (exit={code})")

    if "monitor" in stage_plan:
        cmd = [
            python,
            "scripts/run_monitor.py",
            "--manifest-csv",
            args.scan_manifest_csv,
            "--rules",
            resolved_rules,
            "--alert-config",
            args.alert_config,
            *valid_only_flag,
            "--max-frames",
            str(args.monitor_max_frames),
            "--target-fps",
            str(args.monitor_target_fps),
            "--batch-output-dir",
            args.monitor_out_dir,
        ]
        t0 = time.perf_counter()
        code = _run(cmd, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {"stage": "monitor", "return_code": code, "duration_sec": round(time.perf_counter() - t0, 3)}
        )
        if code != 0:
            raise RuntimeError(f"Stage failed: monitor (exit={code})")

    if "export" in stage_plan:
        cmd = [
            python,
            "scripts/export_results.py",
            "--manifest-csv",
            args.scan_manifest_csv,
            "--rules",
            resolved_rules,
            "--alert-config",
            args.alert_config,
            *valid_only_flag,
            "--max-frames",
            str(args.export_max_frames),
            "--target-fps",
            str(args.export_target_fps),
            "--out-json",
            args.export_out_json,
            "--out-csv",
            args.export_out_csv,
            "--out-events-jsonl",
            args.export_events_jsonl,
            "--out-events-csv",
            args.export_events_csv,
        ]
        t0 = time.perf_counter()
        code = _run(cmd, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {"stage": "export", "return_code": code, "duration_sec": round(time.perf_counter() - t0, 3)}
        )
        if code != 0:
            raise RuntimeError(f"Stage failed: export (exit={code})")

    if "analyze" in stage_plan:
        cmd = [
            python,
            "scripts/analyze_violations.py",
            "--export-summary-json",
            args.analyze_in_json,
            "--out-json",
            args.analyze_out_json,
            "--out-md",
            args.analyze_out_md,
        ]
        t0 = time.perf_counter()
        code = _run(cmd, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {"stage": "analyze", "return_code": code, "duration_sec": round(time.perf_counter() - t0, 3)}
        )
        if code != 0:
            raise RuntimeError(f"Stage failed: analyze (exit={code})")

        cmd_diag = [
            python,
            "scripts/analyze_violation_diagnostics.py",
            "--batch-monitor-dir",
            args.monitor_out_dir,
            "--rules",
            resolved_rules,
            "--out-json",
            args.diagnostics_out_json,
            "--out-md",
            args.diagnostics_out_md,
        ]
        t1 = time.perf_counter()
        code_diag = _run(cmd_diag, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {
                "stage": "analyze_diagnostics",
                "return_code": code_diag,
                "duration_sec": round(time.perf_counter() - t1, 3),
            }
        )
        if code_diag != 0:
            raise RuntimeError(f"Stage failed: analyze_diagnostics (exit={code_diag})")

    if "audit" in stage_plan:
        cmd = [
            python,
            "scripts/build_audit_assets.py",
            "--batch-monitor-dir",
            args.audit_batch_monitor_dir,
            "--out-dir",
            args.audit_out_dir,
            "--max-snaps-per-sample",
            str(args.audit_max_snaps_per_sample),
        ]
        t0 = time.perf_counter()
        code = _run(cmd, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {"stage": "audit", "return_code": code, "duration_sec": round(time.perf_counter() - t0, 3)}
        )
        if code != 0:
            raise RuntimeError(f"Stage failed: audit (exit={code})")

    if args.compare_profiles:
        cmd = [
            python,
            "scripts/compare_sop_profiles.py",
            "--manifest-csv",
            args.scan_manifest_csv,
            "--max-frames",
            str(args.monitor_max_frames),
            "--target-fps",
            str(args.monitor_target_fps),
            "--alert-config",
            args.alert_config,
            "--out-json",
            args.compare_out_json,
            "--out-md",
            args.compare_out_md,
        ]
        if args.valid_only:
            cmd.append("--valid-only")
        if args.dry_run:
            cmd.append("--dry-run")

        t2 = time.perf_counter()
        code_cmp = _run(cmd, cwd=project_root, dry_run=False)
        stage_records.append(
            {
                "stage": "compare_profiles",
                "return_code": code_cmp,
                "duration_sec": round(time.perf_counter() - t2, 3),
            }
        )
        if code_cmp != 0:
            raise RuntimeError(f"Stage failed: compare_profiles (exit={code_cmp})")

    if args.build_hardcase_pack:
        cmd_hc = [
            python,
            "scripts/build_ppe_hardcase_pack.py",
            "--diagnostics-json",
            args.diagnostics_out_json,
            "--batch-monitor-summary",
            str(Path(args.monitor_out_dir) / "summary.json"),
            "--rule-id",
            args.hardcase_rule_id,
            "--context",
            str(args.hardcase_context),
            "--out-dir",
            args.hardcase_out_dir,
            "--out-csv",
            args.hardcase_out_csv,
        ]
        t3 = time.perf_counter()
        code_hc = _run(cmd_hc, cwd=project_root, dry_run=args.dry_run)
        stage_records.append(
            {
                "stage": "build_hardcase_pack",
                "return_code": code_hc,
                "duration_sec": round(time.perf_counter() - t3, 3),
            }
        )
        if code_hc != 0:
            raise RuntimeError(f"Stage failed: build_hardcase_pack (exit={code_hc})")

    end_dt = datetime.now()
    run_meta = {
        "started_at": start_dt.isoformat(timespec="seconds"),
        "finished_at": end_dt.isoformat(timespec="seconds"),
        "duration_sec": round((end_dt - start_dt).total_seconds(), 3),
        "project_root": str(project_root),
        "python": python,
        "dry_run": bool(args.dry_run),
        "dataset_root": args.dataset_root,
        "rules": resolved_rules,
        "alert_config": args.alert_config,
        "sop_profile": args.sop_profile or None,
        "manifest_csv": args.scan_manifest_csv,
        "compare_profiles": bool(args.compare_profiles),
        "compare_out_json": args.compare_out_json if args.compare_profiles else None,
        "compare_out_md": args.compare_out_md if args.compare_profiles else None,
        "build_hardcase_pack": bool(args.build_hardcase_pack),
        "hardcase_out_csv": args.hardcase_out_csv if args.build_hardcase_pack else None,
        "valid_only": bool(args.valid_only),
        "from_stage": from_stage,
        "to_stage": to_stage,
        "executed_stages": stage_plan,
        "stage_records": stage_records,
    }
    meta_path = project_root / args.run_meta_json
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] run metadata saved: {meta_path}")

    print(f"[INFO] 0-to-1 pipeline finished at {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
