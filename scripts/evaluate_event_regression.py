from __future__ import annotations

import argparse
import json
from pathlib import Path

from labsopguard.evaluation.event_regression import evaluate_dataset_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate physical event predictions against a lab action dataset.")
    parser.add_argument("--dataset", required=True, help="Path to lab_action_dataset.v1 JSON.")
    parser.add_argument("--outputs-root", required=True, help="Root containing per-video/experiment physical_events.json outputs.")
    parser.add_argument("--output", default=None, help="Optional path for the regression report JSON.")
    parser.add_argument("--iou-threshold", type=float, default=0.35)
    args = parser.parse_args()

    report = evaluate_dataset_outputs(args.dataset, args.outputs_root, iou_threshold=args.iou_threshold)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
