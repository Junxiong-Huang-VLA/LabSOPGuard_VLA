from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lab_vla.core.runtime import run_lab_vla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified lab_vla runtime.")
    parser.add_argument("--runtime-config", default="configs/runtime/lab_vla_runtime.yaml")
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_lab_vla(runtime_config_path=args.runtime_config)
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("runtime complete, see outputs/runtime/lab_vla_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
