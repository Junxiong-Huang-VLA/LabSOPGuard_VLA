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


def _read_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    return f"{number:.{digits}f}"


def _metadata_lookup(mapping: dict[str, Any], segment_id: str) -> dict[str, Any]:
    key = str(segment_id or "").strip()
    return mapping.get(key, {})


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
    alignment_health = _read_json(root / "metadata" / "alignment_health.json", {})
    segments = read_jsonl(root / "metadata" / "key_action_segments.jsonl") if (root / "metadata" / "key_action_segments.jsonl").exists() else []
    detected = read_jsonl(root / "cv_outputs" / "detected_segments.jsonl") if (root / "cv_outputs" / "detected_segments.jsonl").exists() else []

    max_end = max((float(row.get("end_sec", 0.0)) for row in detected), default=1.0)
    total_duration = float(summary.get("total_action_duration_sec", 0.0) or 0.0)
    debug_paths = {
        "ROI Preview": "../debug/roi_preview.jpg",
        "Frame Scores": "../debug/frame_scores.png",
        "Contact Sheet": "../debug/segments_contact_sheet.jpg",
    }
    vector_metadata_rows = read_jsonl(root / "metadata" / "vector_metadata.jsonl") if (root / "metadata" / "vector_metadata.jsonl").exists() else []
    vector_metadata_map = {str(row.get("segment_id")): row for row in vector_metadata_rows if isinstance(row, dict) and row.get("segment_id") is not None}
    micro_rows = read_jsonl(root / "metadata" / "micro_segments.jsonl") if (root / "metadata" / "micro_segments.jsonl").exists() else []
    micros_by_parent: dict[str, list[dict]] = {}
    for micro in micro_rows:
        if isinstance(micro, dict):
            parent_id = str(micro.get("parent_segment_id") or "")
            micros_by_parent.setdefault(parent_id, []).append(micro)
    detector_summary = summary.get("detector_summary") if isinstance(summary.get("detector_summary"), dict) else {}
    detector_label = str(detector_summary.get("detector_backend", detector_config.get("detector_backend", "motion")))
    detector_summary_reason = str(detector_summary.get("reason_code", ""))
    detector_summary_fallback_used = bool(detector_summary.get("fallback_used", False))
    detector_summary_fallback_reason = detector_summary.get("fallback_reason", "")

    ah_summary = alignment_health.get("summary") if isinstance(alignment_health.get("summary"), dict) else {}
    ah_status = str(ah_summary.get("status", "no_data"))
    ah_mean_offset = _read_float(ah_summary.get("mean_offset_ms"), 1)
    ah_jitter = _read_float(ah_summary.get("jitter_ms"), 1)
    ah_drift_events = int(ah_summary.get("drift_events", 0))
    ah_alerts = alignment_health.get("alerts", []) if isinstance(alignment_health.get("alerts"), list) else []
    ah_status_color = {"healthy": "#2e7d32", "warning": "#f57c00", "drift_alert": "#c62828", "no_data": "#757575"}.get(ah_status, "#757575")

    segment_html = []
    for segment in segments:
        segment_id = str(segment.get("segment_id"))
        vector_metadata = _metadata_lookup(vector_metadata_map, segment_id)
        third = segment.get("third_person") or {}
        first = segment.get("first_person") or {}
        text_desc = segment.get("text_description") or {}
        index_text = ((segment.get("index") or {}).get("index_text") or "").strip()
        third_src = _session_rel(third.get("clip_path"), root, page_dir) if third.get("clip_path") else ""
        first_src = _session_rel(first.get("clip_path"), root, page_dir) if first.get("clip_path") else ""
        decision_trace = segment.get("decision_trace", vector_metadata.get("decision_trace"))
        if isinstance(decision_trace, list):
            decision_trace_text = " | ".join(str(item) for item in decision_trace if str(item))
        elif isinstance(decision_trace, tuple):
            decision_trace_text = " | ".join(str(item) for item in list(decision_trace) if str(item))
        else:
            decision_trace_text = str(decision_trace or "")
        if not decision_trace_text:
            decision_trace_text = "-"
        detector_backend = segment.get("detector_backend", vector_metadata.get("detector_backend", "motion"))
        detector_source_view = segment.get("detector_source_view", vector_metadata.get("detector_source_view", "third_person"))
        fallback_used = bool(segment.get("fallback_used", vector_metadata.get("fallback_used", False)))
        fallback_reason = segment.get("fallback_reason", vector_metadata.get("fallback_reason", ""))
        decision_path = segment.get("decision_path", vector_metadata.get("decision_path", ""))
        reason_code = segment.get("reason_code", vector_metadata.get("reason_code", ""))
        raw_score = segment.get("raw_score", vector_metadata.get("raw_score"))
        final_score = segment.get("final_score", vector_metadata.get("final_score"))
        avg_active_score = (segment.get("cv_detection") or {}).get("avg_active_score")

        micro_list = micros_by_parent.get(str(segment.get("segment_id")), [])
        micro_html_items = []
        for micro in micro_list:
            m_interaction = micro.get("interaction") if isinstance(micro.get("interaction"), dict) else {}
            m_quality = micro.get("quality") if isinstance(micro.get("quality"), dict) else {}
            m_keyframes = micro.get("keyframes") if isinstance(micro.get("keyframes"), dict) else {}
            m_primary = str(m_interaction.get("primary_object") or "-")
            m_score = _read_float(m_interaction.get("max_interaction_score"))
            m_confidence = str(m_quality.get("confidence") or "-")
            m_trigger = _read_float(m_interaction.get("contact_start_sec"), 2)
            m_peak_frame = str(m_keyframes.get("peak_frame") or "")
            m_peak_src = _session_rel(m_peak_frame, root, page_dir) if m_peak_frame else ""
            m_border_color = "#c62828" if m_confidence == "low" else "#f57c00" if m_confidence == "medium" else "#2e7d32"
            peak_img = f'<img src="{html.escape(m_peak_src)}" style="height:48px;border-radius:4px;margin-right:8px">' if m_peak_src else ""
            micro_html_items.append(
                f'<li style="border-left:3px solid {m_border_color};padding:4px 8px;margin:4px 0">'
                f'{peak_img}'
                f'<strong>{html.escape(m_primary)}</strong> '
                f'score={m_score} confidence={html.escape(m_confidence)} '
                f'trigger={m_trigger}s</li>'
            )
        micro_section = ""
        if micro_html_items:
            micro_section = (
                '<details style="margin-top:10px"><summary>Micro-Segments ('
                + str(len(micro_html_items))
                + ')</summary><ul style="list-style:none;padding:0">'
                + "".join(micro_html_items)
                + "</ul></details>"
            )

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
          <div><dt>Duration</dt><dd>{_read_float(segment.get('duration_sec', None), 2)}s</dd></div>
          <div><dt>Top Local Time</dt><dd>{_read_float(third.get('local_start_sec'), 2)}s - {_read_float(third.get('local_end_sec'), 2)}s</dd></div>
          <div><dt>Bottom Local Time</dt><dd>{_read_float(first.get('local_start_sec'), 2)}s - {_read_float(first.get('local_end_sec'), 2)}s</dd></div>
          <div><dt>Avg Active Score</dt><dd>{_read_float(avg_active_score)}</dd></div>
          <div><dt>Detector</dt><dd>{html.escape(str(detector_backend))} / {html.escape(str(detector_source_view))}</dd></div>
          <div><dt>Fallback</dt><dd>{'used' if bool(fallback_used) else 'not used'}</dd></div>
          <div><dt>Decision Path</dt><dd>{html.escape(str(decision_path))} / {html.escape(str(reason_code))}</dd></div>
          <div><dt>Score</dt><dd>raw={_read_float(raw_score)} final={_read_float(final_score)}</dd></div>
        </dl>
        <p class="trace">Decision Trace: {html.escape(decision_trace_text)}</p>
        <p class="trace">Fallback Reason: {html.escape(str(fallback_reason or '-'))}</p>
        {micro_section}
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

    alert_html = []
    for item in ah_alerts[:5]:
        w_start = item.get("window_start", "?")
        w_end = item.get("window_end", "?")
        drift_val = _read_float(item.get("drift_sec"), 3)
        alert_html.append(
            f'<div class="metric" style="border-color:#c62828;background:#fff5f5">'
            f'<span>Drift Alert (window {w_start}-{w_end})</span>'
            f'<strong style="color:#c62828">{drift_val}s drift</strong></div>'
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
    .trace {{
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 13px;
    }}
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
    <p style="font-size:12px;color:var(--muted)">Run ID: {html.escape(str(summary.get('run_id', '-')))} | <a href="../run_manifest.json" style="color:var(--accent)">Run Manifest</a> | <a href="../metadata/alignment_report.json" style="color:var(--accent)">Alignment Report</a></p>
  </header>
    <main>
     <section class="summary">
       <div class="metric"><span>Segments</span><strong>{len(segments)}</strong></div>
       <div class="metric"><span>Total Action Duration</span><strong>{total_duration:.2f}s</strong></div>
       <div class="metric"><span>Start Time</span><strong>{html.escape(str(manifest.get('session_start_time', 'N/A')))}</strong></div>
       <div class="metric"><span>Detector</span><strong>{html.escape(detector_label)}</strong></div>
       <div class="metric"><span>Decision</span><strong>{html.escape(detector_summary_reason or '-')}</strong></div>
       <div class="metric"><span>Summary Fallback</span><strong>{'used' if detector_summary_fallback_used else 'not used'}</strong></div>
       <div class="metric"><span>Summary Fallback Reason</span><strong>{html.escape(str(detector_summary_fallback_reason or '-'))}</strong></div>
       <div class="metric" style="border-color:{ah_status_color}"><span>Alignment Health</span><strong style="color:{ah_status_color}">{html.escape(ah_status)}</strong></div>
       <div class="metric"><span>Mean Offset</span><strong>{ah_mean_offset} ms</strong></div>
       <div class="metric"><span>Jitter</span><strong>{ah_jitter} ms</strong></div>
       <div class="metric"><span>Drift Events</span><strong style="color:{'#c62828' if ah_drift_events > 0 else 'inherit'}">{ah_drift_events}</strong></div>
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

    {''.join(alert_html) if alert_html else ''}

    {''.join(segment_html) if segment_html else '<section class="segment"><h2>No detected segments</h2></section>'}

    <section class="segment">
      <h2>Detector Config</h2>
      <pre>{html.escape(json.dumps(detector_config, ensure_ascii=False, indent=2))}</pre>
    </section>
  </main>
  <script>
  (function(){{
    const segments = document.querySelectorAll('.segment[id]');
    const controls = document.createElement('div');
    controls.style.cssText = 'position:sticky;top:0;z-index:99;background:#fff;padding:8px 36px;border-bottom:1px solid #d9dee4;display:flex;gap:12px;align-items:center';
    controls.innerHTML = '<input id="seg-search" type="text" placeholder="Search segments..." style="flex:1;padding:6px 10px;border:1px solid #ccc;border-radius:6px;font-size:14px">'
      + '<select id="seg-filter" style="padding:6px 10px;border:1px solid #ccc;border-radius:6px;font-size:14px"><option value="">All types</option></select>'
      + '<select id="seg-sort" style="padding:6px 10px;border:1px solid #ccc;border-radius:6px;font-size:14px"><option value="time">By time</option><option value="score">By score</option></select>';
    document.querySelector('main').prepend(controls);
    const types = new Set();
    segments.forEach(s => {{const b = s.querySelector('.badge'); if(b) types.add(b.textContent.trim())}});
    const sel = document.getElementById('seg-filter');
    types.forEach(t => {{const o = document.createElement('option'); o.value=t; o.textContent=t; sel.appendChild(o)}});
    function applyFilter(){{
      const q = document.getElementById('seg-search').value.toLowerCase();
      const f = sel.value;
      segments.forEach(s => {{
        const text = s.textContent.toLowerCase();
        const badge = (s.querySelector('.badge')||{{}}).textContent||'';
        const matchQ = !q || text.includes(q);
        const matchF = !f || badge.trim()===f;
        s.style.display = (matchQ && matchF) ? '' : 'none';
      }});
    }}
    document.getElementById('seg-search').addEventListener('input', applyFilter);
    sel.addEventListener('change', applyFilter);
  }})();
  </script>
</body>
</html>
"""
    target.write_text(html_text, encoding="utf-8")
    return target
