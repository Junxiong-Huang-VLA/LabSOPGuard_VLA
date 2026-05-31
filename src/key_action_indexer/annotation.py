from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import read_jsonl, write_jsonl


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _micro_source(session_dir: Path, source_path: str | Path | None = None) -> Path:
    if source_path is not None:
        return Path(source_path)
    corrected = session_dir / "metadata" / "micro_segments_corrected.jsonl"
    if corrected.exists():
        return corrected
    return session_dir / "metadata" / "micro_segments.jsonl"


def _safe(value: Any) -> str:
    return "" if value is None else str(value)


def _nested(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


def _review_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    interaction = _nested(row, "interaction")
    text = _nested(row, "text_description")
    quality = _nested(row, "quality")
    keyframes = _nested(row, "keyframes")
    third = _nested(row, "third_person")
    first = _nested(row, "first_person")
    return {
        "review_id": f"review_{index:06d}",
        "micro_segment_id": _safe(row.get("micro_segment_id")),
        "parent_segment_id": _safe(row.get("parent_segment_id")),
        "display_id": _safe(row.get("display_id") or row.get("micro_segment_id")),
        "start_sec": float(row.get("start_sec", 0.0) or 0.0),
        "end_sec": float(row.get("end_sec", 0.0) or 0.0),
        "duration_sec": float(row.get("duration_sec", 0.0) or 0.0),
        "global_start_time": _safe(row.get("global_start_time")),
        "global_end_time": _safe(row.get("global_end_time")),
        "action_type": _safe(text.get("action_type")),
        "primary_object": _safe(interaction.get("primary_object")),
        "interaction_type": _safe(interaction.get("interaction_type")),
        "max_interaction_score": interaction.get("max_interaction_score"),
        "quality": _safe(quality.get("confidence")),
        "warnings": quality.get("warnings") or [],
        "summary": _safe(text.get("summary")),
        "peak_frame": keyframes.get("peak_frame"),
        "contact_frame": keyframes.get("contact_frame"),
        "release_frame": keyframes.get("release_frame"),
        "first_person_clip": first.get("clip_path") if isinstance(first, dict) else None,
        "third_person_clip": third.get("clip_path"),
        "manual_action": "",
        "manual_primary_object": "",
        "manual_start_sec": "",
        "manual_end_sec": "",
        "accept": "",
        "review_note": "",
    }


def _template_row(row: dict[str, Any], index: int, *, full_window: bool = False) -> dict[str, Any]:
    if full_window:
        return {
            "micro_segment_id": f"gt_micro_{index:03d}",
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "primary_object": row["primary_object"] or "edit_me",
            "interaction_type": row["interaction_type"] or "edit_me",
            "action_type": row["action_type"] or "edit_me",
            "source_prediction_id": row["micro_segment_id"],
            "note": "edit me",
        }
    return {
        "micro_segment_id": row["micro_segment_id"],
        "parent_segment_id": row["parent_segment_id"],
        "start_sec": row["start_sec"],
        "end_sec": row["end_sec"],
        "action_type": row["action_type"],
        "primary_object": row["primary_object"],
        "interaction_type": row["interaction_type"],
        "summary": row["summary"],
        "operation": "update",
        "note": "",
    }


def _markdown(rows: list[dict[str, Any]], *, full_window: bool = False) -> str:
    lines = [
        "# Micro Segment Review",
        "",
        "Use `manual_micro_segments.template.jsonl` as the starting point for corrections. Set `operation` to `update`, `insert`, or `delete`.",
        "",
    ]
    if full_window:
        lines.extend(
            [
                "Full-window GT annotation rules:",
                "",
                "1. Do not only label predicted micro-segments.",
                "2. Label every visible real micro interaction inside the labeled window.",
                "3. Edit start/end/object/action when a prediction is inaccurate.",
                "4. Add new GT rows when the system missed a real interaction.",
                "5. Do not copy false-positive predictions into GT.",
                "",
            ]
        )
    lines.extend(
        [
            "| # | micro_segment_id | parent | time | object | action | quality | review |",
            "|---:|---|---|---:|---|---|---|---|",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        time_range = f"{row['start_sec']:.3f}-{row['end_sec']:.3f}s"
        review = "accept / edit / delete"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    _safe(row["micro_segment_id"]),
                    _safe(row["parent_segment_id"]),
                    time_range,
                    _safe(row["primary_object"]),
                    _safe(row["action_type"]),
                    _safe(row["quality"]),
                    review,
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_micro_annotation_pack(
    session_dir: str | Path,
    *,
    source_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    full_window: bool = False,
    window_start_sec: float | None = None,
    window_end_sec: float | None = None,
) -> dict[str, Any]:
    session = Path(session_dir)
    source = _micro_source(session, source_path)
    if not source.exists():
        raise FileNotFoundError(f"micro segments not found: {source}")
    target_dir = Path(output_dir) if output_dir else session / "annotation" / "micro_review"
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = [_review_row(row, index) for index, row in enumerate(read_jsonl(source), start=1)]
    review_jsonl = target_dir / "micro_review.jsonl"
    template_jsonl = target_dir / "manual_micro_segments.template.jsonl"
    eval_config_template = target_dir / "manual_micro_eval_config.template.json"
    review_md = target_dir / "micro_review.md"
    write_jsonl(review_jsonl, rows)
    write_jsonl(template_jsonl, [_template_row(row, index, full_window=full_window) for index, row in enumerate(rows, start=1)])
    review_md.write_text(_markdown(rows, full_window=full_window), encoding="utf-8")

    if full_window:
        if window_start_sec is None:
            window_start_sec = min((float(row["start_sec"]) for row in rows), default=0.0)
        if window_end_sec is None:
            window_end_sec = max((float(row["end_sec"]) for row in rows), default=0.0)
        manifest_path = session / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.exists() else {}
        eval_config = {
            "session_id": manifest.get("session_id", session.name),
            "labeled_windows": [
                {
                    "window_id": "win_001",
                    "start_sec": float(window_start_sec),
                    "end_sec": float(window_end_sec),
                    "description": "full parent segment labeled",
                }
            ],
            "gt_completeness": "complete",
            "notes": "All visible micro interactions in this parent segment are labeled.",
        }
        _write_json(eval_config_template, eval_config)

    summary = {
        "session_dir": str(session),
        "source": str(source),
        "output_dir": str(target_dir),
        "micro_review_md": str(review_md),
        "micro_review_jsonl": str(review_jsonl),
        "manual_template": str(template_jsonl),
        "eval_config_template": str(eval_config_template) if full_window else None,
        "full_window": bool(full_window),
        "micro_segment_count": len(rows),
    }
    _write_json(target_dir / "micro_review_summary.json", summary)
    return summary
