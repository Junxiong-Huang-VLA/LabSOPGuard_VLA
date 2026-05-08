from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any

from .schemas import read_jsonl


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _session_rel(path: str | Path, root: Path, page_dir: Path) -> str:
    candidate = Path(path)
    root_abs = root.resolve()
    page_abs = page_dir.resolve()
    if candidate.is_absolute():
        try:
            candidate.resolve().relative_to(root_abs)
            return Path(os.path.relpath(candidate.resolve(), page_abs)).as_posix()
        except ValueError:
            return candidate.as_uri()
    try:
        return ("../" + candidate.relative_to(root).as_posix()).replace("\\", "/")
    except ValueError:
        return ("../" + candidate.as_posix()).replace("\\", "/")


def _video_tag(src: str, label: str) -> str:
    if not src:
        return f'<div class="video-missing">{html.escape(label)} not available</div>'
    return (
        '<div class="video-block">'
        f'<div class="video-label">{html.escape(label)}</div>'
        f'<video controls preload="metadata" src="{html.escape(src)}"></video>'
        "</div>"
    )


def generate_frontend_viewer(session_dir: str | Path, output_path: str | Path | None = None) -> Path:
    root = Path(session_dir)
    frontend_dir = root / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    target = Path(output_path) if output_path else frontend_dir / "index.html"
    page_dir = target.parent

    manifest = _read_json(root / "manifest.json", {})
    summary = _read_json(root / "pipeline_summary.json", {})
    video_info = _read_json(root / "video_info.json", {})
    detector_config = _read_json(root / "metadata" / "detector_config.json", {})
    segments = read_jsonl(root / "metadata" / "key_action_segments.jsonl") if (root / "metadata" / "key_action_segments.jsonl").exists() else []
    detected = read_jsonl(root / "cv_outputs" / "detected_segments.jsonl") if (root / "cv_outputs" / "detected_segments.jsonl").exists() else []

    max_end = max((float(row.get("end_sec", 0.0)) for row in detected), default=1.0)
    total_duration = float(summary.get("total_action_duration_sec", 0.0) or 0.0)
    debug_paths = {
        "ROI Preview": "../debug/roi_preview.jpg",
        "Frame Scores": "../debug/frame_scores.png",
        "Contact Sheet": "../debug/segments_contact_sheet.jpg",
    }

    segment_html = []
    for segment in segments:
        third = segment.get("third_person") or {}
        first = segment.get("first_person") or {}
        text_desc = segment.get("text_description") or {}
        index_text = ((segment.get("index") or {}).get("index_text") or "").strip()
        third_src = _session_rel(third.get("clip_path"), root, page_dir) if third.get("clip_path") else ""
        first_src = _session_rel(first.get("clip_path"), root, page_dir) if first.get("clip_path") else ""
        segment_html.append(
            f"""
      <section class="segment" id="{html.escape(str(segment.get('segment_id')))}">
        <div class="segment-head">
          <div>
            <h2>{html.escape(str(segment.get('segment_id')))}</h2>
            <p>{html.escape(str(segment.get('global_start_time')))} - {html.escape(str(segment.get('global_end_time')))}</p>
          </div>
          <div class="badge">{html.escape(str(text_desc.get('action_type', 'unknown_operation')))}</div>
        </div>
        <div class="video-grid">
          {_video_tag(third_src, "Top / detection view")}
          {_video_tag(first_src, "Bottom / synced view")}
        </div>
        <dl class="meta-grid">
          <div><dt>Duration</dt><dd>{float(segment.get('duration_sec', 0.0)):.2f}s</dd></div>
          <div><dt>Top Local Time</dt><dd>{float(third.get('local_start_sec', 0.0)):.2f}s - {float(third.get('local_end_sec', 0.0)):.2f}s</dd></div>
          <div><dt>Bottom Local Time</dt><dd>{float(first.get('local_start_sec', 0.0)):.2f}s - {float(first.get('local_end_sec', 0.0)):.2f}s</dd></div>
          <div><dt>Avg Active Score</dt><dd>{float((segment.get('cv_detection') or {}).get('avg_active_score', 0.0)):.3f}</dd></div>
        </dl>
        <details>
          <summary>Index text and retrieval metadata</summary>
          <pre>{html.escape(index_text)}</pre>
        </details>
      </section>
"""
        )

    timeline_marks = []
    for row in detected:
        start = float(row.get("start_sec", 0.0))
        end = float(row.get("end_sec", start))
        left = max(0.0, min(100.0, start / max_end * 100.0))
        width = max(0.3, min(100.0 - left, (end - start) / max_end * 100.0))
        timeline_marks.append(
            f'<span class="mark" style="left:{left:.3f}%;width:{width:.3f}%;" title="{html.escape(str(row.get("segment_id")))} {start:.2f}s-{end:.2f}s"></span>'
        )

    video_lines = []
    for name, info in (video_info.get("video_sources") or {}).items():
        if not isinstance(info, dict):
            continue
        video_lines.append(
            f"<li><strong>{html.escape(name)}</strong>: {html.escape(str(info.get('width')))}x{html.escape(str(info.get('height')))}, "
            f"{html.escape(str(info.get('fps')))} FPS, {html.escape(str(info.get('duration_sec')))}s</li>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(manifest.get('session_id', 'experiment')))} Viewer</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172026;
      --muted: #5e6972;
      --line: #d9dee4;
      --panel: #f7f8fa;
      --accent: #1769aa;
      --active: #f0a130;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 28px 36px 18px;
      border-bottom: 1px solid var(--line);
      background: #f3f6f8;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }}
    h2 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.5; }}
    main {{ padding: 22px 36px 36px; max-width: 1440px; margin: 0 auto; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
    }}
    .metric span {{ display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }}
    .metric strong {{ font-size: 20px; }}
    .timeline {{
      position: relative;
      height: 24px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #edf1f4;
      margin: 12px 0 24px;
      overflow: hidden;
    }}
    .mark {{ position: absolute; top: 0; bottom: 0; background: var(--active); opacity: 0.85; }}
    .debug-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(220px, 1fr));
      gap: 14px;
      margin: 18px 0 28px;
    }}
    .debug-grid a {{
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      text-decoration: none;
      color: var(--ink);
      background: #fff;
    }}
    .debug-grid img {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #f1f3f5;
      border: 1px solid var(--line);
    }}
    .debug-grid span {{ display: block; margin-top: 8px; font-weight: 700; }}
    .segment {{
      border-top: 1px solid var(--line);
      padding: 24px 0;
    }}
    .segment-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .badge {{
      border: 1px solid #b8c7d6;
      color: var(--accent);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 13px;
      white-space: nowrap;
      background: #f7fbff;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      gap: 14px;
    }}
    .video-block {{
      border: 1px solid var(--line);
      background: #0f151a;
      border-radius: 8px;
      overflow: hidden;
    }}
    .video-label {{
      color: #e9eef2;
      padding: 8px 10px;
      font-size: 13px;
      background: #172026;
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
    }}
    .video-missing {{
      border: 1px dashed var(--line);
      border-radius: 8px;
      min-height: 220px;
      display: grid;
      place-items: center;
      color: var(--muted);
      background: var(--panel);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(150px, 1fr));
      gap: 10px;
      margin: 14px 0;
    }}
    .meta-grid div {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }}
    dt {{ font-size: 12px; color: var(--muted); }}
    dd {{ margin: 5px 0 0; font-weight: 700; }}
    details {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
    }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 700; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      color: #25313a;
      background: #f6f8fa;
      padding: 12px;
      border-radius: 6px;
      overflow: auto;
    }}
    ul {{ margin: 8px 0 0; padding-left: 18px; color: var(--muted); }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .summary, .debug-grid, .video-grid, .meta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(str(manifest.get('session_id', 'experiment')))}</h1>
    <p>Dual-view key action indexing output. Detection uses the top view; clips are aligned by global time.</p>
  </header>
  <main>
    <section class="summary">
      <div class="metric"><span>Segments</span><strong>{len(segments)}</strong></div>
      <div class="metric"><span>Total Action Duration</span><strong>{total_duration:.2f}s</strong></div>
      <div class="metric"><span>Start Time</span><strong>{html.escape(str(manifest.get('session_start_time', 'N/A')))}</strong></div>
      <div class="metric"><span>Detector</span><strong>ROI motion</strong></div>
    </section>

    <section>
      <h2>Timeline</h2>
      <div class="timeline">{''.join(timeline_marks)}</div>
      <ul>{''.join(video_lines)}</ul>
    </section>

    <section>
      <h2>Debug Outputs</h2>
      <div class="debug-grid">
        {''.join(f'<a href="{html.escape(src)}" target="_blank"><img src="{html.escape(src)}" alt="{html.escape(label)}"><span>{html.escape(label)}</span></a>' for label, src in debug_paths.items())}
      </div>
    </section>

    {''.join(segment_html) if segment_html else '<section class="segment"><h2>No detected segments</h2></section>'}

    <section class="segment">
      <h2>Detector Config</h2>
      <pre>{html.escape(json.dumps(detector_config, ensure_ascii=False, indent=2))}</pre>
    </section>
  </main>
</body>
</html>
"""
    target.write_text(html_text, encoding="utf-8")
    return target
