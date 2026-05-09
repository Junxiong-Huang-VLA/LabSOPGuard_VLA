"""
PDF报告生成器 - 基于WeasyPrint + Jinja2的专业合规报告
符合FDA ORA-LAB、ISO/IEC 17025:2017和WHO LQMS标准
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import pandas as pd
import numpy as np
import cv2
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """PDF合规报告生成器"""

    def __init__(self, template_dir: str = "templates"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.template_dir = self.project_root / template_dir
        self.output_dir = self.project_root / "outputs" / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 初始化Jinja2环境
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )

        # 配置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def generate_report(self, report_data: Dict[str, Any], output_path: str) -> str:
        """
        生成PDF合规报告

        Args:
            report_data: 报告数据
            output_path: 输出文件路径

        Returns:
            生成的PDF文件路径
        """
        try:
            logger.info(f"开始生成PDF报告: {report_data.get('task_id', 'unknown')}")

            # 1. 生成图表
            charts = self._generate_charts(report_data)

            # 2. 处理违规截图
            annotated_frames = self._process_violation_frames(report_data)

            # 3. 准备模板数据
            template_data = self._prepare_template_data(report_data, charts, annotated_frames)

            # 4. 渲染HTML
            html_content = self._render_html_template(template_data)

            # 5. 生成PDF
            pdf_path = self._generate_pdf(html_content, output_path)

            logger.info(f"PDF报告生成完成: {pdf_path}")
            return pdf_path

        except Exception as e:
            logger.error(f"PDF报告生成失败: {e}")
            raise

    def _generate_charts(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """生成统计图表"""
        charts = {}
        violations = report_data.get("violations", [])

        if not violations:
            return charts

        # 1. 违规类型分布柱状图
        rule_counts = {}
        for v in violations:
            rule_id = v.get("rule_id", "unknown")
            rule_counts[rule_id] = rule_counts.get(rule_id, 0) + 1

        if rule_counts:
            fig, ax = plt.subplots(figsize=(10, 6))
            rules = list(rule_counts.keys())
            counts = list(rule_counts.values())

            bars = ax.bar(rules, counts, color=['#ff4444', '#ff8800', '#ffcc00', '#00cc00'][:len(rules)])
            ax.set_title('违规类型分布', fontsize=16, fontweight='bold')
            ax.set_xlabel('违规类型', fontsize=12)
            ax.set_ylabel('次数', fontsize=12)
            plt.xticks(rotation=45, ha='right')

            # 添加数值标签
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            chart_path = self.output_dir / "violation_types.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            charts["violation_types"] = str(chart_path)

        # 2. 严重等级饼图
        severity_counts = {"Critical": 0, "Major": 0, "Minor": 0}
        for v in violations:
            severity = v.get("severity", "Minor")
            if severity in severity_counts:
                severity_counts[severity] += 1

        if any(severity_counts.values()):
            fig, ax = plt.subplots(figsize=(8, 8))
            labels = [k for k, v in severity_counts.items() if v > 0]
            sizes = [v for v in severity_counts.values() if v > 0]
            colors = ['#ff4444', '#ff8800', '#ffcc00'][:len(labels)]

            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                             autopct='%1.1f%%', startangle=90)
            ax.set_title('违规严重等级分布', fontsize=16, fontweight='bold')

            # 美化字体
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

            plt.tight_layout()
            chart_path = self.output_dir / "severity_distribution.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            charts["severity_distribution"] = str(chart_path)

        # 3. 时间分布直方图
        timestamps = [v.get("timestamp_sec", 0) for v in violations if "timestamp_sec" in v]
        if timestamps:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(timestamps, bins=min(20, len(timestamps)), alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_title('违规时间分布', fontsize=16, fontweight='bold')
            ax.set_xlabel('时间 (秒)', fontsize=12)
            ax.set_ylabel('违规次数', fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            chart_path = self.output_dir / "time_distribution.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            charts["time_distribution"] = str(chart_path)

        # 4. 合规率趋势图 (如果有步骤数据)
        if len(violations) > 1:
            df = pd.DataFrame(violations)
            if "timestamp_sec" in df.columns and "severity" in df.columns:
                # 计算滚动合规率
                df = df.sort_values("timestamp_sec")
                df["violation_score"] = df["severity"].map({"Critical": 3, "Major": 2, "Minor": 1})
                df["rolling_violations"] = df["violation_score"].rolling(window=5, min_periods=1).sum()
                df["compliance_rate"] = 100 - (df["rolling_violations"] / df["rolling_violations"].max() * 100)

                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df["timestamp_sec"], df["compliance_rate"], linewidth=2, color='green')
                ax.fill_between(df["timestamp_sec"], df["compliance_rate"], alpha=0.3, color='lightgreen')
                ax.set_title('合规率趋势', fontsize=16, fontweight='bold')
                ax.set_xlabel('时间 (秒)', fontsize=12)
                ax.set_ylabel('合规率 (%)', fontsize=12)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                chart_path = self.output_dir / "compliance_trend.png"
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()
                charts["compliance_trend"] = str(chart_path)

        return charts

    def _process_violation_frames(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """处理违规截图"""
        frames = []
        violations = report_data.get("violations", [])

        for i, violation in enumerate(violations[:10]):  # 最多处理10张截图
            frame_info = {
                "index": i + 1,
                "timestamp": violation.get("timestamp_sec", 0),
                "rule_id": violation.get("rule_id", "unknown"),
                "severity": violation.get("severity", "Minor"),
                "message": violation.get("message", "")
            }

            # 如果有截图路径，处理标注
            screenshot_path = violation.get("screenshot_path")
            if screenshot_path and Path(screenshot_path).exists():
                try:
                    # 读取并标注图片
                    img = cv2.imread(screenshot_path)
                    if img is not None:
                        # 添加违规信息标注
                        annotated_img = self._annotate_frame(img, violation)

                        # 保存标注后的图片
                        annotated_path = self.output_dir / f"annotated_frame_{i+1}.jpg"
                        cv2.imwrite(str(annotated_path), annotated_img)

                        frame_info["annotated_path"] = str(annotated_path)
                        frame_info["original_path"] = screenshot_path

                except Exception as e:
                    logger.warning(f"处理截图失败: {e}")

            frames.append(frame_info)

        return frames

    def _annotate_frame(self, frame, violation: Dict[str, Any]):
        """标注违规帧"""
        annotated = frame.copy()
        height, width = frame.shape[:2]

        # 添加红色边框
        cv2.rectangle(annotated, (0, 0), (width-1, height-1), (0, 0, 255), 3)

        # 添加违规信息文本
        severity = violation.get("severity", "Minor")
        rule_id = violation.get("rule_id", "unknown")
        timestamp = violation.get("timestamp_sec", 0)

        # 根据严重等级设置颜色
        if severity == "Critical":
            color = (0, 0, 255)  # 红色
        elif severity == "Major":
            color = (0, 140, 255)  # 橙色
        else:
            color = (0, 255, 255)  # 黄色

        # 添加半透明背景
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # 添加文本
        cv2.putText(annotated, f"违规: {rule_id}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(annotated, f"等级: {severity} | 时间: {timestamp:.1f}s", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    def _prepare_template_data(self, report_data: Dict[str, Any],
                             charts: Dict[str, str], frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """准备模板数据"""
        violations = report_data.get("violations", [])
        summary = report_data.get("summary", {})

        # 计算合规分数
        total_violations = len(violations)
        if total_violations == 0:
            compliance_score = 100.0
        else:
            # 根据严重等级加权计算
            severity_weights = {"Critical": 10, "Major": 5, "Minor": 1}
            total_weight = sum(severity_weights.get(v.get("severity", "Minor"), 1) for v in violations)
            compliance_score = max(0, 100 - total_weight)

        return {
            "report_title": "实验室SOP合规监控报告",
            "task_id": report_data.get("task_id", "unknown"),
            "video_path": report_data.get("video_path", ""),
            "generated_at": report_data.get("generated_at", datetime.now().isoformat()),
            "summary": {
                "compliance_score": compliance_score,
                "total_violations": total_violations,
                "critical_count": summary.get("critical_count", 0),
                "major_count": summary.get("major_count", 0),
                "minor_count": summary.get("minor_count", 0),
                "status": "通过" if compliance_score >= 80 else "不通过"
            },
            "violations": violations,
            "charts": charts,
            "frames": frames,
            "company_info": {
                "name": "实验室SOP合规监控系统",
                "version": "1.0.0",
                "standard": "ISO/IEC 17025:2017"
            }
        }

    def _render_html_template(self, template_data: Dict[str, Any]) -> str:
        """渲染HTML模板"""
        try:
            template = self.jinja_env.get_template("compliance_report.html")
            return template.render(**template_data)
        except Exception as e:
            logger.warning(f"模板加载失败，使用内置模板: {e}")
            return self._generate_builtin_template(template_data)

    def _generate_builtin_template(self, data: Dict[str, Any]) -> str:
        """生成内置HTML模板"""
        summary = data.get("summary", {})
        violations = data.get("violations", [])
        charts = data.get("charts", {})
        frames = data.get("frames", [])

        html = f'''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <title>{data.get("report_title", "合规报告")}</title>
            <style>
                body {{ font-family: 'SimHei', 'DejaVu Sans', sans-serif; margin: 40px; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; margin-bottom: 30px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
                .score {{ font-size: 48px; font-weight: bold; color: {"green" if summary.get("compliance_score", 0) >= 80 else "red"}; }}
                .violation-card {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .critical {{ border-left: 5px solid #ff4444; }}
                .major {{ border-left: 5px solid #ff8800; }}
                .minor {{ border-left: 5px solid #ffcc00; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                .frame {{ margin: 15px 0; text-align: center; }}
                .frame img {{ max-width: 100%; border: 1px solid #ddd; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{data.get("report_title", "合规报告")}</h1>
                <p>任务ID: {data.get("task_id", "unknown")} | 生成时间: {data.get("generated_at", "")}</p>
                <p>视频文件: {data.get("video_path", "unknown")}</p>
            </div>

            <div class="summary">
                <h2>执行摘要</h2>
                <div style="text-align: center;">
                    <div class="score">{summary.get("compliance_score", 0):.1f}%</div>
                    <p>合规状态: {summary.get("status", "未知")}</p>
                </div>
                <table>
                    <tr><th>指标</th><th>数值</th></tr>
                    <tr><td>总违规次数</td><td>{summary.get("total_violations", 0)}</td></tr>
                    <tr><td>严重违规</td><td>{summary.get("critical_count", 0)}</td></tr>
                    <tr><td>重要违规</td><td>{summary.get("major_count", 0)}</td></tr>
                    <tr><td>轻微违规</td><td>{summary.get("minor_count", 0)}</td></tr>
                </table>
            </div>

            <h2>统计图表</h2>
        '''

        # 添加图表
        for chart_name, chart_path in charts.items():
            html += f'<div class="chart"><img src="file://{chart_path}" alt="{chart_name}"></div>'

        # 添加违规详情
        if violations:
            html += '<h2>违规详细记录</h2>'
            for i, violation in enumerate(violations, 1):
                severity_class = violation.get("severity", "minor").lower()
                html += f'''
                <div class="violation-card {severity_class}">
                    <h3>违规 #{i}</h3>
                    <p><strong>时间戳:</strong> {violation.get("timestamp_sec", 0):.1f}s</p>
                    <p><strong>规则ID:</strong> {violation.get("rule_id", "unknown")}</p>
                    <p><strong>严重等级:</strong> {violation.get("severity", "Minor")}</p>
                    <p><strong>描述:</strong> {violation.get("message", "")}</p>
                </div>
                '''

        # 添加标注截图
        if frames:
            html += '<h2>违规截图</h2>'
            for frame in frames:
                if "annotated_path" in frame:
                    html += f'''
                    <div class="frame">
                        <img src="file://{frame["annotated_path"]}" alt="违规截图 {frame["index"]}">
                        <p>截图 {frame["index"]} - {frame.get("rule_id", "unknown")} ({frame.get("severity", "Minor")})</p>
                    </div>
                    '''

        html += '''
            <div style="margin-top: 40px; text-align: center; color: #666; font-size: 12px;">
                <p>本报告由实验室SOP合规智能监控系统自动生成</p>
                <p>符合 ISO/IEC 17025:2017 标准</p>
            </div>
        </body>
        </html>
        '''

        return html

    def _generate_pdf(self, html_content: str, output_path: str) -> str:
        """生成PDF文件"""
        try:
            HTML(string=html_content).write_pdf(output_path)
            return output_path
        except Exception as e:
            logger.error(f"PDF生成失败: {e}")
            raise

# 使用示例
if __name__ == "__main__":
    # 测试数据
    test_data = {
        "task_id": "test_001",
        "video_path": "/path/to/video.mp4",
        "generated_at": datetime.now().isoformat(),
        "violations": [
            {
                "frame_id": 100,
                "timestamp_sec": 3.33,
                "rule_id": "missing_goggles",
                "severity": "Critical",
                "message": "操作人员未佩戴护目镜",
                "screenshot_path": "/path/to/screenshot.jpg"
            },
            {
                "frame_id": 200,
                "timestamp_sec": 6.67,
                "rule_id": "missing_gloves",
                "severity": "Major",
                "message": "操作人员未佩戴手套"
            }
        ],
        "summary": {
            "total_violations": 2,
            "critical_count": 1,
            "major_count": 1,
            "minor_count": 0
        }
    }

    generator = PDFReportGenerator()
    pdf_path = generator.generate_report(test_data, "test_report.pdf")
    print(f"测试报告生成完成: {pdf_path}")