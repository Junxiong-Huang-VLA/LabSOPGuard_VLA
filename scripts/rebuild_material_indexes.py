from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from labsopguard.material_maintenance import rebuild_workspace_material_index, scan_experiment_material_health


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild LabSOPGuard material indexes.")
    parser.add_argument("--experiments-root", default=str(PROJECT_ROOT / "outputs" / "experiments"))
    parser.add_argument("--workspace-index", default=str(PROJECT_ROOT / "outputs" / "materials" / "material_index.sqlite"))
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--force-experiment-indexes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.health_only:
        report = scan_experiment_material_health(args.experiments_root)
    else:
        report = rebuild_workspace_material_index(
            args.experiments_root,
            args.workspace_index,
            force_experiment_indexes=args.force_experiment_indexes,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
