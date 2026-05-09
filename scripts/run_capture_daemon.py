from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.capture_daemon import CaptureDaemon, CaptureDaemonOptions, run_capture_daemon
from labsopguard.soak_test import make_dry_run_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised multi-camera capture.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "runtime" / "multicam_soak.yaml"),
        help="Capture YAML config path.",
    )
    parser.add_argument("--once", action="store_true", help="Run one capture cycle and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Run supervised synthetic cameras without real hardware.")
    parser.add_argument("--duration-sec", type=float, default=3.0, help="Dry-run duration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dry_run:
        daemon = CaptureDaemon(
            make_dry_run_config(duration_sec=args.duration_sec),
            CaptureDaemonOptions(run_once=True, status_path=".runtime/capture_daemon/dry_run_status.json"),
        )
        report = daemon.run()
    else:
        report = run_capture_daemon(args.config, run_once=args.once)
    print(json.dumps({"status": report.get("status"), "report": report}, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
