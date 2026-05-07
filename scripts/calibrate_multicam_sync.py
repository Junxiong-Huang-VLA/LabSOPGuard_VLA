from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.sync_calibration import build_sync_calibration_report_from_file, write_sync_calibration_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a multi-camera sync calibration report.")
    parser.add_argument("--video-inputs", required=True, help="Path to video_inputs.json or experiment.json.")
    parser.add_argument("--output", default=None, help="Output sync_calibration_report.json path.")
    parser.add_argument("--auto-flash", action="store_true", help="Detect flash anchors from local video files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_sync_calibration_report_from_file(args.video_inputs, auto_flash=args.auto_flash)
    if args.output:
        write_sync_calibration_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
