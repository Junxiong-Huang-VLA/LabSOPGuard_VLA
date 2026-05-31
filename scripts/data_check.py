from __future__ import annotations

import argparse
import json
from pathlib import Path

from project_name.common.config import load_yaml
from project_name.common.io_utils import read_jsonl
from project_name.common.logging_utils import setup_logger


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate SOP monitoring dataset")
    parser.add_argument("--config", default="configs/data/dataset.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    logger = setup_logger("data_check")
    required = cfg["validation"]["required_fields"]
    rows = read_jsonl(cfg["dataset"]["annotation_file"])

    errors = []
    for idx, row in enumerate(rows):
        miss = [k for k in required if k not in row]
        if miss:
            errors.append({"index": idx, "sample_id": row.get("sample_id", "unknown"), "missing": miss})

    report = {"total": len(rows), "invalid": len(errors), "errors": errors}
    out = Path("outputs/reports/data_check_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if errors:
        logger.error("Invalid rows=%d report=%s", len(errors), out)
        return 1

    logger.info("Dataset valid. total=%d", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
