from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.soak_test import load_soak_test_config, make_dry_run_config, preflight_soak_test_sources, run_soak_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-camera 24h soak test.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "runtime" / "multicam_soak.yaml"),
        help="YAML config path for real camera sources.",
    )
    parser.add_argument("--duration-sec", type=float, default=None, help="Override configured duration.")
    parser.add_argument("--output-root", default=None, help="Override output root.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run two synthetic cameras for a short local validation without real hardware.",
    )
    parser.add_argument("--preflight", action="store_true", help="Only open configured sources and read one frame.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        config = make_dry_run_config(duration_sec=args.duration_sec or 3.0, output_root=args.output_root or ".runtime/soak_tests")
    else:
        config = load_soak_test_config(args.config)
        if args.duration_sec is not None:
            config.duration_sec = args.duration_sec
        if args.output_root:
            config.output_root = args.output_root

    report = preflight_soak_test_sources(config) if args.preflight else run_soak_test(config)
    print(json.dumps({"status": report["status"], "artifacts": report.get("artifact_paths"), "report": report}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
