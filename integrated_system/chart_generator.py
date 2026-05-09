"""
Chart Generator
===============
使用 matplotlib 生成专业统计图表（150 DPI PNG），嵌入PDF报告：
- 违规类型分布柱状图
- 严重等级饼图
- 各步骤合规率横向条形图
- 违规时间分布直方图
- Gantt式实验流程时间线
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# 中文字体配置
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 配色方案
COLORS = {
    "compliant": "#10b981",    # 绿色
    "violation": "#ef4444",    # 红色
    "warning": "#f59e0b",      # 橙色
    "info": "#3b82f6",         # 蓝色
    "critical": "#dc2626",     # 深红
    "major": "#f97316",        # 深橙
    "minor": "#eab308",        # 黄色
    "bg": "#1a2235",           # 暗色背景
    "text": "#e2e8f0",         # 浅色文字
    "grid": "#334155",         # 网格线
}

SEVERITY_COLORS = ["#dc2626", "#f97316", "#eab308", "#64748b"]
SEVERITY_LABELS = ["严重 (Critical)", "一般 (Major)", "轻微 (Minor)", "信息 (Info)"]


@dataclass
class ChartOutput:
    """图表输出"""
    file_path: str
    chart_type: str
    width: int
    height: int


class ChartGenerator:
    """统计图表生成器"""

    def __init__(self, output_dir: Path, dpi: int = 150):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def generate_all(
        self,
        violations: List[Dict[str, Any]],
        steps: List[Dict[str, Any]],
        timeline_data: Optional[Dict[str, Any]] = None,
    ) -> List[ChartOutput]:
        """生成全部图表"""
        charts = []

        c1 = self.generate_violation_type_chart(violations)
        if c1:
            charts.append(c1)

        c2 = self.generate_severity_pie_chart(violations)
        if c2:
            charts.append(c2)

        c3 = self.generate_step_compliance_bar(steps)
        if c3:
            charts.append(c3)

        c4 = self.generate_violation_timeline_histogram(violations)
        if c4:
            charts.append(c4)

        c5 = self.generate_gantt_timeline(steps, violations)
        if c5:
            charts.append(c5)

        return charts

    def generate_violation_type_chart(
        self, violations: List[Dict[str, Any]]
    ) -> Optional[ChartOutput]:
        """违规类型分布柱状图"""
        type_counts: Dict[str, int] = {}
        for v in violations:
            vtype = v.get("type", v.get("violation_type", "未知"))
            type_counts[vtype] = type_counts.get(vtype, 0) + 1

        if not type_counts:
            return None

        fig, ax = self._create_dark_figure(10, 5)

        types = list(type_counts.keys())
        counts = list(type_counts.values())
        colors = [COLORS["violation"], COLORS["warning"], COLORS["info"],
                  COLORS["critical"], COLORS["major"]][:len(types)]

        bars = ax.barh(types, counts, color=colors[:len(types)], height=0.6, edgecolor="none")

        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=10, color=COLORS["text"], fontweight="bold")

        ax.set_xlabel("违规次数", fontsize=11, color=COLORS["text"])
        ax.set_title("违规类型分布", fontsize=14, color=COLORS["text"], fontweight="bold", pad=12)
        ax.invert_yaxis()

        fpath = self.output_dir / "chart_violation_types.png"
        fig.savefig(str(fpath), dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return ChartOutput(str(fpath), "violation_types", 10, 5)

    def generate_severity_pie_chart(
        self, violations: List[Dict[str, Any]]
    ) -> Optional[ChartOutput]:
        """严重等级饼图"""
        severity_counts: Dict[str, int] = {}
        for v in violations:
            sev = v.get("severity", "minor").lower()
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        if not severity_counts:
            return None

        fig, ax = self._create_dark_figure(7, 5)

        sev_order = ["critical", "major", "minor", "info"]
        labels = []
        sizes = []
        colors = []
        for s in sev_order:
            if s in severity_counts:
                labels.append({"critical": "严重", "major": "一般", "minor": "轻微", "info": "信息"}.get(s, s))
                sizes.append(severity_counts[s])
                colors.append(COLORS.get(s, COLORS["info"]))

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.0f%%",
            startangle=90, textprops={"color": COLORS["text"], "fontsize": 11},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")

        ax.set_title("违规严重等级分布", fontsize=14, color=COLORS["text"], fontweight="bold", pad=12)

        fpath = self.output_dir / "chart_severity_pie.png"
        fig.savefig(str(fpath), dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return ChartOutput(str(fpath), "severity_pie", 7, 5)

    def generate_step_compliance_bar(
        self, steps: List[Dict[str, Any]]
    ) -> Optional[ChartOutput]:
        """各步骤合规率横向条形图"""
        if not steps:
            return None

        fig, ax = self._create_dark_figure(10, max(4, len(steps) * 0.6))

        step_names = []
        compliance_rates = []
        bar_colors = []

        for step in steps:
            name = step.get("name", step.get("step_name", "未知步骤"))
            expected = step.get("expected_count", 1)
            observed = step.get("observed_count", 0)
            rate = min(100.0, (observed / max(expected, 1)) * 100)
            step_names.append(name)
            compliance_rates.append(rate)
            bar_colors.append(COLORS["compliant"] if rate >= 80 else
                              COLORS["warning"] if rate >= 50 else COLORS["violation"])

        y_pos = np.arange(len(step_names))
        bars = ax.barh(y_pos, compliance_rates, color=bar_colors, height=0.5, edgecolor="none")

        for bar, rate in zip(bars, compliance_rates):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{rate:.0f}%", va="center", fontsize=10, color=COLORS["text"], fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(step_names, fontsize=10)
        ax.set_xlim(0, 110)
        ax.set_xlabel("合规率 (%)", fontsize=11, color=COLORS["text"])
        ax.set_title("各步骤合规率", fontsize=14, color=COLORS["text"], fontweight="bold", pad=12)
        ax.invert_yaxis()

        fpath = self.output_dir / "chart_step_compliance.png"
        fig.savefig(str(fpath), dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return ChartOutput(str(fpath), "step_compliance", 10, max(4, len(steps) * 0.6))

    def generate_violation_timeline_histogram(
        self, violations: List[Dict[str, Any]]
    ) -> Optional[ChartOutput]:
        """违规时间分布直方图"""
        timestamps = []
        for v in violations:
            ts = v.get("timestamp_sec", v.get("timestamp", 0))
            if isinstance(ts, (int, float)):
                timestamps.append(ts / 60.0)  # 转换为分钟

        if not timestamps:
            return None

        fig, ax = self._create_dark_figure(10, 4)

        max_time = max(timestamps) if timestamps else 60
        bins = np.linspace(0, max_time + 1, min(20, int(max_time / 2) + 2))

        ax.hist(timestamps, bins=bins, color=COLORS["violation"], alpha=0.8,
                edgecolor=COLORS["text"], linewidth=0.5)
        ax.set_xlabel("实验时间 (分钟)", fontsize=11, color=COLORS["text"])
        ax.set_ylabel("违规次数", fontsize=11, color=COLORS["text"])
        ax.set_title("违规时间分布", fontsize=14, color=COLORS["text"], fontweight="bold", pad=12)

        fpath = self.output_dir / "chart_violation_histogram.png"
        fig.savefig(str(fpath), dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return ChartOutput(str(fpath), "violation_histogram", 10, 4)

    def generate_gantt_timeline(
        self,
        steps: List[Dict[str, Any]],
        violations: List[Dict[str, Any]],
    ) -> Optional[ChartOutput]:
        """Gantt式实验流程时间线"""
        if not steps:
            return None

        fig, ax = self._create_dark_figure(12, max(4, len(steps) * 0.8))

        for i, step in enumerate(steps):
            name = step.get("name", step.get("step_name", f"步骤{i+1}"))
            start = step.get("start_sec", step.get("start_time", i * 60)) / 60.0
            end = step.get("end_sec", step.get("end_time", (i + 1) * 60)) / 60.0
            status = step.get("status", "compliant")

            color = COLORS["compliant"] if status == "compliant" else \
                    COLORS["violation"] if status == "violation" else COLORS["warning"]

            ax.barh(i, end - start, left=start, height=0.5, color=color, alpha=0.85,
                    edgecolor="none")

            # 步骤名称标签
            ax.text(start + (end - start) / 2, i, name,
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        # 标记违规时间点
        for v in violations:
            ts = v.get("timestamp_sec", v.get("timestamp", 0)) / 60.0
            sev = v.get("severity", "minor")
            marker_color = COLORS["critical"] if sev == "critical" else \
                           COLORS["warning"] if sev == "major" else COLORS["info"]
            ax.axvline(x=ts, color=marker_color, linestyle="--", alpha=0.6, linewidth=1)

        y_labels = [step.get("name", step.get("step_name", f"步骤{i+1}"))
                    for i, step in enumerate(steps)]
        ax.set_yticks(range(len(steps)))
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.set_xlabel("实验时间 (分钟)", fontsize=11, color=COLORS["text"])
        ax.set_title("实验流程时间线", fontsize=14, color=COLORS["text"], fontweight="bold", pad=12)
        ax.invert_yaxis()

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS["compliant"], label="合规"),
            Patch(facecolor=COLORS["warning"], label="警告"),
            Patch(facecolor=COLORS["violation"], label="违规"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
                  facecolor=COLORS["bg"], edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

        fpath = self.output_dir / "chart_gantt_timeline.png"
        fig.savefig(str(fpath), dpi=self.dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return ChartOutput(str(fpath), "gantt_timeline", 12, max(4, len(steps) * 0.8))

    def _create_dark_figure(self, width: int, height: int):
        """创建暗色主题图表"""
        fig, ax = plt.subplots(figsize=(width, height))
        fig.patch.set_facecolor(COLORS["bg"])
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text"], labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(COLORS["grid"])
        ax.spines["left"].set_color(COLORS["grid"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.grid(axis="x" if width > height else "y", color=COLORS["grid"], alpha=0.3, linewidth=0.5)
        return fig, ax
