from __future__ import annotations

import argparse
import json
from pathlib import Path

from lab_vla.core.runtime import run_lab_vla


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab VLA unified runtime entry.")
    parser.add_argument(
        "--runtime-config",
        default="configs/runtime/lab_vla_runtime.yaml",
        help="Runtime config yaml path.",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="Print final summary json to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_lab_vla(runtime_config_path=args.runtime_config)
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        out = Path(args.runtime_config).resolve().parents[2] / "outputs/runtime/lab_vla_summary.json"
        print(f"runtime summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

