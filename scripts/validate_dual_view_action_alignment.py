#!/usr/bin/env python
"""Validate dual-view action alignment from small JSONL artifacts.

The script is intentionally data-only: it reads candidate/material/timing JSONL
files and never opens videos or calls ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from key_action_indexer.dual_view_action_validation import (  # noqa: E402
    validate_dual_view_action_alignment_files,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Candidate action JSONL")
    parser.add_argument("--materials", default=None, help="Material asset JSONL")
    parser.add_argument("--timing", default=None, help="YOLO timing JSONL")
    parser.add_argument("--config", default=None, help="Optional run/config JSON")
    parser.add_argument("--output", default=None, help="Optional report JSON path")
    parser.add_argument("--max-alignment-delta-sec", type=float, default=1.0)
    parser.add_argument("--require-formal-event", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    args = parser.parse_args(argv)

    report = validate_dual_view_action_alignment_files(
        args.candidates,
        materials_path=args.materials,
        timing_path=args.timing,
        config_path=args.config,
        max_alignment_delta_sec=args.max_alignment_delta_sec,
        require_formal_event=args.require_formal_event,
    )
    if args.output:
        write_json(args.output, report)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = report["summary"]
        print(
            "status={status} formal_events={events} rejected={rejected}".format(
                status=report["status"],
                events=summary["formal_event_count"],
                rejected=summary["rejected_count"],
            )
        )
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
