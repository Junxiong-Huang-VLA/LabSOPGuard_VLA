#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2

from multimodal_eval_common import (
    PROJECT_ROOT,
    dashscope_call_multimodal,
    ensure_reports_dir,
    extract_json_object,
    file_uri,
    model_list,
    pass_fail_from_structured,
    read_json,
    write_matrix_rows,
)


PROMPT = """你是实验视频理解评测员。请只输出 JSON，不要 Markdown。
请分析这段实验 clip：
1. 用一句话总结视频内容
2. 按时间顺序提取步骤列表
3. 提取关键对象
4. 判断显著变化点
5. 给出最值得人工复核的时刻

JSON schema:
{
  "summary": "一句话中文总结",
  "steps": [{"time_hint": "约x秒", "action": "动作"}],
  "key_objects": ["对象"],
  "change_points": [{"time_hint": "约x秒", "change": "变化"}],
  "review_moments": [{"time_hint": "约x秒", "reason": "复核原因"}],
  "confidence": 0.0
}
"""


def duration_sec(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return round(float(frames / fps), 3) if fps else 0.0


def collect_video_samples(exp_id: str, limit: int) -> List[Dict[str, Any]]:
    exp_dir = PROJECT_ROOT / "outputs" / "experiments" / exp_id
    preprocessing = read_json(exp_dir / "preprocessing.json", {})
    clips = []
    for clip in preprocessing.get("key_clips", []) or []:
        path = clip.get("file_path")
        if path and Path(path).exists():
            clips.append(
                {
                    "sample_id": f"clip_{len(clips) + 1:03d}",
                    "path": Path(path),
                    "expected": {
                        "clip_id": clip.get("clip_id"),
                        "reason": clip.get("reason"),
                        "camera_id": clip.get("camera_id"),
                        "anchor_timestamp_sec": clip.get("anchor_timestamp_sec"),
                    },
                }
            )
        if len(clips) >= limit:
            break
    if len(clips) < limit:
        for path in (exp_dir / "clips").glob("*.mp4"):
            if any(item["path"] == path for item in clips):
                continue
            clips.append({"sample_id": f"clip_{len(clips) + 1:03d}", "path": path, "expected": {}})
            if len(clips) >= limit:
                break
    return clips


def evaluate(exp_id: str, limit: int, models: List[str], matrix_path: Path) -> Dict[str, Any]:
    samples = collect_video_samples(exp_id, limit)
    rows: List[Dict[str, Any]] = []
    detailed: List[Dict[str, Any]] = []
    for sample in samples:
        dur = duration_sec(sample["path"])
        for model in models:
            try:
                result = dashscope_call_multimodal(
                    model=model,
                    content=[
                        {"video": file_uri(sample["path"])},
                        {"text": PROMPT},
                    ],
                    timeout=120,
                    retries=2,
                    compress_video=True,
                )
                structured = extract_json_object(result["raw_response"])
                pass_fail, notes = pass_fail_from_structured(
                    structured,
                    ["summary", "steps", "key_objects", "change_points", "review_moments"],
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
                "task_type": "short_clip_understanding",
                "sample_id": sample["sample_id"],
                "model_name": model,
                "input_path": str(sample["path"]),
                "expected": {**sample["expected"], "duration_sec": dur},
                "actual": actual,
                "pass_fail": pass_fail,
                "response_time_ms": response_time_ms,
                "notes": notes,
            }
            rows.append(row)
            detailed.append({**row, "duration_sec": dur, "raw_response": result.get("raw_response", ""), "structured_result": structured})
    write_matrix_rows(matrix_path, rows, append=matrix_path.exists())
    out_json = ensure_reports_dir() / "multimodal_video_eval_details.json"
    out_json.write_text(json.dumps(detailed, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(1 for row in rows if row["pass_fail"] == "pass")
    return {
        "task_type": "short_clip_understanding",
        "sample_count": len(samples),
        "model_count": len(models),
        "case_count": len(rows),
        "pass_count": passed,
        "pass_rate": round(passed / max(len(rows), 1), 4),
        "details_path": str(out_json),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Qwen multimodal short-clip understanding.")
    parser.add_argument("--exp-id", default="final_acceptance_e2e")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--models", default=",".join(model_list()))
    parser.add_argument("--matrix", default=str(ensure_reports_dir() / "multimodal_eval_matrix.csv"))
    args = parser.parse_args()
    summary = evaluate(args.exp_id, args.limit, [m.strip() for m in args.models.split(",") if m.strip()], Path(args.matrix))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
