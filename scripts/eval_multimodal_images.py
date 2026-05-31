#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from multimodal_eval_common import (
    PROJECT_ROOT,
    compact,
    dashscope_call_multimodal,
    ensure_reports_dir,
    extract_json_object,
    file_uri,
    model_list,
    pass_fail_from_structured,
    read_json,
    write_matrix_rows,
)


PROMPT = """你是实验过程理解评测员。请只输出 JSON，不要 Markdown。
根据这张实验图像回答：
1. 画面中有哪些关键器材、容器、手部或操作对象？
2. 当前画面可能对应实验步骤中的什么动作？
3. 是否存在显著风险点、状态变化或异常？
4. 用中文输出结构化结果。

JSON schema:
{
  "objects": ["关键器材/容器/手部/操作对象"],
  "actions": ["可能动作"],
  "scene_summary": "一句话中文场景概述",
  "risk_flags": ["风险或异常，没有则为空数组"],
  "state_changes": ["可见状态变化，没有则为空数组"],
  "confidence": 0.0
}
"""


def collect_image_samples(exp_id: str, limit: int) -> List[Dict[str, Any]]:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    material_stream = read_json(exp_dir / "material_stream.json", [])
    samples: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for item in material_stream:
        path = item.get("frame_bgr_path")
        if not path:
            continue
        frame_path = Path(path)
        if not frame_path.is_absolute():
            frame_path = PROJECT_ROOT / frame_path
        if not frame_path.exists() or str(frame_path) in seen:
            continue
        seen.add(str(frame_path))
        expected = {
            "object_labels": item.get("object_labels", []),
            "detected_activities": item.get("detected_activities", []),
            "timestamp_sec": item.get("timestamp_sec"),
            "transcript_segment": item.get("transcript_segment"),
        }
        samples.append(
            {
                "sample_id": f"img_{len(samples) + 1:03d}",
                "path": frame_path,
                "expected": expected,
            }
        )
        if len(samples) >= limit:
            break

    if len(samples) < limit:
        for frame_path in (exp_dir / "artifacts" / "frames").glob("**/*.jpg"):
            if str(frame_path) in seen:
                continue
            seen.add(str(frame_path))
            samples.append({"sample_id": f"img_{len(samples) + 1:03d}", "path": frame_path, "expected": {}})
            if len(samples) >= limit:
                break
    return samples


def evaluate(exp_id: str, limit: int, models: List[str], matrix_path: Path) -> Dict[str, Any]:
    samples = collect_image_samples(exp_id, limit)
    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    for sample in samples:
        for model in models:
            try:
                result = dashscope_call_multimodal(
                    model=model,
                    content=[
                        {"image": file_uri(sample["path"])},
                        {"text": PROMPT},
                    ],
                )
                structured = extract_json_object(result["raw_response"])
                pass_fail, notes = pass_fail_from_structured(
                    structured,
                    ["objects", "actions", "scene_summary", "risk_flags"],
                )
                actual = structured or {"raw_response": result["raw_response"]}
                response_time_ms = result["response_time_ms"]
            except Exception as exc:
                structured = {}
                actual = {"error": str(exc)}
                pass_fail = "fail"
                notes = type(exc).__name__
                response_time_ms = 0
                result = {"raw_response": "", "response_time_ms": 0}

            row = {
                "task_type": "single_frame_understanding",
                "sample_id": sample["sample_id"],
                "model_name": model,
                "input_path": str(sample["path"]),
                "expected": sample["expected"],
                "actual": actual,
                "pass_fail": pass_fail,
                "response_time_ms": response_time_ms,
                "notes": notes,
            }
            rows.append(row)
            detailed.append(
                {
                    **row,
                    "raw_response": result.get("raw_response", ""),
                    "structured_result": structured,
                    "prompt_type": "objects_actions_risk_structured",
                }
            )

    write_matrix_rows(matrix_path, rows, append=matrix_path.exists())
    out_json = ensure_reports_dir() / "multimodal_image_eval_details.json"
    out_json.write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for row in rows if row["pass_fail"] == "pass")
    return {
        "task_type": "single_frame_understanding",
        "sample_count": len(samples),
        "model_count": len(models),
        "case_count": len(rows),
        "pass_count": passed,
        "pass_rate": round(passed / max(len(rows), 1), 4),
        "details_path": str(out_json),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qwen multimodal image understanding on experiment frames.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--models", default=",".join(model_list()))
    parser.add_argument("--matrix", default=str(ensure_reports_dir() / "multimodal_eval_matrix.csv"))
    parser.add_argument("--reset-matrix", action="store_true")
    args = parser.parse_args()
    matrix = Path(args.matrix)
    if args.reset_matrix and matrix.exists():
        matrix.unlink()
    summary = evaluate(args.exp_id, args.limit, [m.strip() for m in args.models.split(",") if m.strip()], matrix)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
