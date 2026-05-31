from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_name.common.io_utils import read_jsonl
from project_name.common.logging_utils import setup_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate SOP monitoring outputs")
    parser.add_argument("--pred", default="outputs/predictions/test_predictions.jsonl")
    args = parser.parse_args()

    rows = read_jsonl(args.pred)
    logger = setup_logger("evaluate")
    if not rows:
        metrics = {"sessions": 0, "violation_rate": 0.0, "avg_compliance_ratio": 0.0}
    else:
        violation_counts = [len(r.get("violations", [])) for r in rows]
        compliance = [float(r.get("status", {}).get("compliance_ratio", 0.0)) for r in rows]
        metrics = {
            "sessions": len(rows),
            "violation_rate": float(sum(1 for c in violation_counts if c > 0) / len(rows)),
            "avg_compliance_ratio": float(sum(compliance) / len(compliance)),
        }

    out = Path("outputs/reports/eval_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("metrics=%s", metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
