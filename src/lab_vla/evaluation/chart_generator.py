"""
Chart & Timeline Visualization Generator for LabSOPGuard.

Generates:
- Experiment timeline with events, anomalies, and step transitions
- Severity distribution charts (pie/bar)
- Compliance progress over time
- Anomaly distribution heatmap by time window
- Per-step duration analysis

Uses Matplotlib (always available) with optional Plotly for interactive HTML.
Outputs: PNG images for PDF embedding, HTML for interactive dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Color Palette
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "critical": "#d32f2f",
    "high": "#f57c00",
    "medium": "#fbc02d",
    "low": "#388e3c",
    "info": "#1976d2",
}

STEP_COLORS = [
    "#1565c0", "#2e7d32", "#f57c00", "#c62828", "#6a1b9a",
    "#00838f", "#4e342e", "#37474f", "#ad1457", "#283593",
]


# ---------------------------------------------------------------------------
# Matplotlib Charts (always available)
# ---------------------------------------------------------------------------

def _generate_timeline_chart_matplotlib(
    events: List[Dict[str, Any]],
    anomalies: List[Dict[str, Any]],
    step_sequence: List[str],
    output_path: Path,
    title: str = "Experiment Timeline",
) -> Path:
    """Generate timeline chart using Matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                     gridspec_kw={"hspace": 0.3})

    # --- Top: Step timeline ---
    max_time = max((e.get("timestamp", 0) for e in events), default=10.0)
    max_time = max(max_time, 1.0)

    # Draw step bars
    step_id_to_idx = {s: i for i, s in enumerate(step_sequence)}
    for i, event in enumerate(events):
        ts = event.get("timestamp", 0)
        step_id = event.get("step_id", event.get("expected_step_id", ""))
        idx = step_id_to_idx.get(step_id, 0)
        color = STEP_COLORS[idx % len(STEP_COLORS)]
        ax1.barh(idx, min(0.3, max_time * 0.02), left=ts, height=0.6, color=color, alpha=0.8)

    # Draw anomalies
    for anom in anomalies:
        ts = anom.get("timestamp", anom.get("timestamp_sec", 0))
        if ts < 0:
            continue
        sev = anom.get("severity", "medium")
        color = SEVERITY_COLORS.get(sev, "#999")
        ax1.axvline(x=ts, color=color, linestyle="--", alpha=0.6, linewidth=1)

    ax1.set_yticks(range(len(step_sequence)))
    ax1.set_yticklabels([s.replace("_", " ").title() for s in step_sequence], fontsize=9)
    ax1.set_xlabel("Time (seconds)", fontsize=10)
    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_xlim(0, max_time * 1.05)
    ax1.grid(axis="x", alpha=0.3)

    # Legend
    legend_elements = [
        mpatches.Patch(color=SEVERITY_COLORS["critical"], label="Critical"),
        mpatches.Patch(color=SEVERITY_COLORS["high"], label="High"),
        mpatches.Patch(color=SEVERITY_COLORS["medium"], label="Medium"),
        mpatches.Patch(color=SEVERITY_COLORS["low"], label="Low"),
        Line2D([0], [0], color="#999", linestyle="--", label="Anomaly"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # --- Bottom: Anomaly density ---
    if anomalies:
        anomaly_times = [a.get("timestamp", a.get("timestamp_sec", 0)) for a in anomalies if a.get("timestamp", a.get("timestamp_sec", 0)) >= 0]
        if anomaly_times:
            bins = np.linspace(0, max_time, min(20, max(5, int(max_time / 2))))
            ax2.hist(anomaly_times, bins=bins, color="#e53935", alpha=0.7, edgecolor="white")
            ax2.set_ylabel("Anomaly Count", fontsize=10)
            ax2.set_xlabel("Time (seconds)", fontsize=10)
            ax2.set_title("Anomaly Distribution", fontsize=11)
            ax2.set_xlim(0, max_time * 1.05)
            ax2.grid(axis="y", alpha=0.3)
    else:
        ax2.text(max_time / 2, 0.5, "No anomalies detected", ha="center", va="center",
                 fontsize=11, color="#666", transform=ax2.transAxes)
        ax2.set_xlim(0, max_time * 1.05)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _generate_severity_chart_matplotlib(
    anomalies: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """Generate severity distribution pie + bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Count by severity
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for a in anomalies:
        sev = a.get("severity", "medium")
        if sev in counts:
            counts[sev] += 1

    labels = [k for k, v in counts.items() if v > 0]
    values = [v for v in counts.values() if v > 0]
    colors = [SEVERITY_COLORS[l] for l in labels]

    if not values:
        values = [1]
        labels = ["No Anomalies"]
        colors = ["#e0e0e0"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        values, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10},
    )
    ax1.set_title("Severity Distribution", fontsize=13, fontweight="bold")

    # Bar chart
    all_sevs = ["critical", "high", "medium", "low"]
    bar_values = [counts[s] for s in all_sevs]
    bar_colors = [SEVERITY_COLORS[s] for s in all_sevs]
    bars = ax2.bar(all_sevs, bar_values, color=bar_colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, bar_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_title("Anomalies by Severity", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _generate_compliance_chart_matplotlib(
    completed_steps: List[str],
    total_steps: int,
    step_sequence: List[str],
    output_path: Path,
) -> Path:
    """Generate compliance progress chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    x_labels = [s.replace("_", " ").title() for s in step_sequence]
    x = range(len(step_sequence))
    completed_set = set(completed_steps)

    colors = []
    for s in step_sequence:
        if s in completed_set:
            colors.append("#4caf50")  # Green: completed
        else:
            colors.append("#ffcdd2")  # Light red: missing

    bars = ax.bar(x, [1] * len(step_sequence), color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"Step Compliance: {len(completed_steps)}/{total_steps} completed ({len(completed_steps)/max(total_steps,1)*100:.0f}%)",
                 fontsize=13, fontweight="bold")

    # Legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color="#4caf50", label="Completed"),
        mpatches.Patch(color="#ffcdd2", label="Missing"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def generate_charts(
    output_dir: Path,
    alarm_log: Dict[str, Any],
    scene_profile: Dict[str, Any],
    chemistry_analysis: Optional[Dict[str, Any]] = None,
    prego_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Generate all visualization charts.

    Returns dict of chart_name -> file_path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    charts = {}

    # Extract data
    alarms = alarm_log.get("alarms", [])
    expected_steps = alarm_log.get("expected_steps", [])
    observed_steps = alarm_log.get("observed_steps", [])
    step_sequence = scene_profile.get("expected_step_ids", expected_steps)

    # Build events from observed steps + timestamps
    events = []
    for obs_step in observed_steps:
        # Find matching alarm/timestamp
        ts = 0.0
        for a in alarms:
            if a.get("alarm_type") == obs_step or obs_step in str(a.get("evidence", {})):
                ts = a.get("timestamp", 0)
                break
        events.append({"step_id": obs_step, "timestamp": ts})

    # Also add timestamps from chemistry observations if available
    if chemistry_analysis:
        for obs in chemistry_analysis.get("observations", []):
            events.append({
                "step_id": obs.get("expected_step_id", ""),
                "timestamp": obs.get("timestamp_sec", 0),
            })

    # Sort events by time
    events.sort(key=lambda e: e.get("timestamp", 0))

    # 1. Timeline chart
    try:
        timeline_path = _generate_timeline_chart_matplotlib(
            events, alarms, step_sequence,
            output_dir / "chart_timeline.png",
            title=f"Experiment Timeline ({scene_profile.get('experiment_type_zh', 'Unknown')})",
        )
        charts["timeline"] = str(timeline_path)
    except Exception as exc:
        print(f"Timeline chart failed: {exc}")

    # 2. Severity distribution
    try:
        severity_path = _generate_severity_chart_matplotlib(
            alarms, output_dir / "chart_severity.png"
        )
        charts["severity"] = str(severity_path)
    except Exception as exc:
        print(f"Severity chart failed: {exc}")

    # 3. Compliance progress
    try:
        compliance_path = _generate_compliance_chart_matplotlib(
            observed_steps, len(step_sequence), step_sequence,
            output_dir / "chart_compliance.png",
        )
        charts["compliance"] = str(compliance_path)
    except Exception as exc:
        print(f"Compliance chart failed: {exc}")

    # Save chart index
    index_path = output_dir / "charts_index.json"
    index_path.write_text(json.dumps(charts, ensure_ascii=False, indent=2), encoding="utf-8")

    return charts
