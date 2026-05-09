from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.workspace_governance import build_workspace_governance_report, write_workspace_governance_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate workspace governance report.")
    parser.add_argument("--workspace-root", default=str(WORKSPACE_ROOT))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "docs" / "workspace_governance_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_workspace_governance_report(args.workspace_root)
    write_workspace_governance_report(report, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
