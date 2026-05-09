#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from multimodal_eval_common import PROJECT_ROOT, ensure_reports_dir

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from labsopguard.qwen_writeback import QwenFrameWritebackConfig, writeback_qwen_frame_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Qwen single-frame analysis into material_stream.analysis and rebuild material index.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--flash-model", default="qwen3.6-flash")
    parser.add_argument("--review-model", default="qwen3.6-plus")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-eval-cache", action="store_true", help="Do not reuse reports/multimodal_image_eval_details.json.")
    parser.add_argument("--force-live", action="store_true", help="Ignore cache and call DashScope for every frame.")
    args = parser.parse_args()

    config = QwenFrameWritebackConfig(
        enabled=True,
        flash_model=args.flash_model,
        review_model=args.review_model,
        limit=args.limit,
        force_live=args.force_live,
        use_eval_cache=not args.no_eval_cache,
        eval_cache_path=ensure_reports_dir() / "multimodal_image_eval_details.json",
    )
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / args.exp_id
    summary = writeback_qwen_frame_analysis(exp_dir, exp_id=args.exp_id, config=config)
    report_copy = ensure_reports_dir() / f"qwen_frame_writeback_{args.exp_id}.json"
    report_copy.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["report_copy"] = str(report_copy)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["flash_written"] > 0 and not summary["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
