"""CLI: validate (and optionally rewrite) a material database consistency report.

Usage:
  python scripts/validate_material_db_consistency.py --material-root <dir> [--write]

Read-only by default (prints the report). With --write, regenerates
reports/database_consistency_validation_report.json from LIVE data — it never
mutates the materials, stream, or SQLite index. This is the tool that fixes a
stale report whose contents no longer match the database.

No GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a standalone script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from key_action_indexer.material_db_consistency import (  # noqa: E402
    validate_material_database,
    write_consistency_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-root", required=True, type=Path)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the consistency report from live data (does not mutate DB).",
    )
    args = parser.parse_args(argv)

    root = args.material_root
    if not root.exists():
        print(f"material root not found: {root}", file=sys.stderr)
        return 2

    if args.write:
        result = write_consistency_report(root)
    else:
        result = validate_material_database(root)

    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    # exit non-zero on inconsistency so CI / callers can gate on it
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
