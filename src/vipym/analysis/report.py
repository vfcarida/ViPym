"""Comprehensive HTML Report Generator for Compression Experiments and Pareto Decisions.

Embeds interactive Plotly visualizations, ranked deployment recommendations, enterprise financial projections,
and quality retention heatmaps into a self-contained, publication-grade HTML document.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

from vipym.analysis.pareto import ParetoPoint
from vipym.analysis.recommender import RecommendationReport
from vipym.analysis.visualizations import (
    create_cost_comparison_bar_chart,
    create_pareto_scatter_plot,
    create_quality_retention_heatmap,
)
from vipym.core.logger import get_logger

logger = get_logger(__name__)


class HTMLReportGenerator:
    """Generates standalone interactive HTML decision and compression reports."""

    def __init__(self, title: str = "ViPym Compression & Pareto Decision Report") -> None:
        self.title = title

    def generate_report(
        self,
        points: list[ParetoPoint],
        recommendation: RecommendationReport,
        cost_comparison: list[dict[str, Any]] | None = None,
        output_path: str | Path = "compression-report.html",
    ) -> str:
        """Render and save complete HTML report to file."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Render Plotly interactive charts to HTML fragments
        pareto_chart_html = ""
        heatmap_chart_html = ""
        cost_chart_html = ""

        pareto_fig = create_pareto_scatter_plot(points)
        if pareto_fig:
            pareto_chart_html = pareto_fig.to_html(full_html=False, include_plotlyjs="cdn")

        heatmap_fig = create_quality_retention_heatmap(points)
        if heatmap_fig:
            heatmap_chart_html = heatmap_fig.to_html(full_html=False, include_plotlyjs=False)

        if cost_comparison:
            cost_fig = create_cost_comparison_bar_chart(cost_comparison)
            if cost_fig:
                cost_chart_html = cost_fig.to_html(full_html=False, include_plotlyjs=False)

        # 2. Build Recommendation Rows
        rec_rows_html = ""
        for rank, p in enumerate(recommendation.ranked_options, 1):
            qual = (
                f"{p.relative_quality_score * 100:.1f}%"
                if p.relative_quality_score is not None
                else f"{p.quality_score * 100:.1f}%"
            )
            is_top = p.configuration_name == (
                recommendation.recommended_variant.configuration_name if recommendation.recommended_variant else ""
            )
            row_class = "top-row" if is_top else ""
            badge_class = "badge-pareto" if p.is_pareto_optimal else "badge-dominated"
            badge_text = "Pareto Optimal" if p.is_pareto_optimal else "Dominated"

            rec_rows_html += f"""
            <tr class="{row_class}">
                <td style="font-weight: bold; text-align: center;">{rank}</td>
                <td style="font-weight: 600;">{p.configuration_name}</td>
                <td><code>{p.compression_method}</code></td>
                <td><span class="{badge_class}">{badge_text}</span></td>
                <td><span class="qual-badge">{qual}</span></td>
                <td style="color: #38BDF8; font-weight: 600;">${p.cost_per_1m_tokens:.2f}</td>
                <td>{p.throughput_tok_s:.0f} tok/s</td>
                <td>{p.latency_p50_ms:.1f} ms</td>
                <td><code>{p.hardware_recommendation or '1x H100'}</code></td>
            </tr>
            """

        # 3. Build Cost Comparison Rows
        cost_rows_html = ""
        if cost_comparison:
            for row in cost_comparison:
                savings = row.get("savings_vs_claude_pct", 0.0)
                savings_str = f"+{savings:.1f}%" if savings > 0 else f"{savings:.1f}%"
                savings_class = "savings-positive" if savings > 0 else "savings-neutral"

                cost_rows_html += f"""
                <tr>
                    <td style="font-weight: 600;">{row['name']}</td>
                    <td>{row.get('type', 'N/A')}</td>
                    <td><code>{row.get('hardware', 'N/A')}</code></td>
                    <td>${row['cost_per_1m_tokens']:.2f}</td>
                    <td style="font-weight: bold;">${row['monthly_spend_usd']:,.0f}</td>
                    <td>${row['annual_spend_usd']:,.0f}</td>
                    <td><span class="{savings_class}">{savings_str}</span></td>
                </tr>
                """

        # 4. Top Recommendation Summary Card
        top_card_html = ""
        if recommendation.recommended_variant:
            top = recommendation.recommended_variant
            top_qual = (
                f"{top.relative_quality_score * 100:.1f}%"
                if top.relative_quality_score is not None
                else f"{top.quality_score * 100:.1f}%"
            )
            top_card_html = f"""
            <div class="recommendation-card">
                <div class="rec-header">
                    <span class="rec-title">🏆 Top Recommendation: {top.configuration_name}</span>
                    <span class="rec-method">Method: {top.compression_method}</span>
                </div>
                <div class="rec-grid">
                    <div class="rec-stat">
                        <div class="stat-val">{top_qual}</div>
                        <div class="stat-lbl">SE Quality Retained</div>
                    </div>
                    <div class="rec-stat">
                        <div class="stat-val" style="color: #38BDF8;">${top.cost_per_1m_tokens:.2f}</div>
                        <div class="stat-lbl">Cost per 1M Tokens</div>
                    </div>
                    <div class="rec-stat">
                        <div class="stat-val" style="color: #34D399;">{top.throughput_tok_s:.0f}</div>
                        <div class="stat-lbl">Tokens / Second</div>
                    </div>
                    <div class="rec-stat">
                        <div class="stat-val" style="color: #F472B6;">{top.hardware_recommendation or '1x H100'}</div>
                        <div class="stat-lbl">Hardware Instance</div>
                    </div>
                </div>
                <div class="rec-summary-text">{recommendation.executive_summary}</div>
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        :root {{
            --bg-primary: #0F172A;
            --bg-secondary: #1E293B;
            --bg-card: #1E293B;
            --border-color: #334155;
            --text-primary: #F8FAFC;
            --text-secondary: #94A3B8;
            --accent-green: #10B981;
            --accent-blue: #38BDF8;
            --accent-pink: #F472B6;
            --accent-purple: #818CF8;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            padding: 30px;
            line-height: 1.6;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{ font-size: 28px; font-weight: 700; color: #FFFFFF; }}
        .header .meta {{ color: var(--text-secondary); font-size: 14px; }}
        .recommendation-card {{
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 2px solid var(--accent-green);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 30px;
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.15);
        }}
        .rec-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }}
        .rec-title {{ font-size: 22px; font-weight: 700; color: #FFFFFF; }}
        .rec-method {{ background: #334155; padding: 4px 10px; border-radius: 6px; font-size: 13px; }}
        .rec-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 16px;
        }}
        .rec-stat {{
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            text-align: center;
        }}
        .stat-val {{ font-size: 26px; font-weight: 800; color: var(--accent-green); }}
        .stat-lbl {{ font-size: 13px; color: var(--text-secondary); margin-top: 4px; text-transform: uppercase; }}
        .rec-summary-text {{ font-size: 15px; color: #E2E8F0; border-top: 1px solid var(--border-color); padding-top: 14px; }}
        .chart-section {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 30px;
            margin-bottom: 35px;
        }}
        .chart-box {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: #0F172A;
            color: var(--text-secondary);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}
        tr:hover {{ background-color: rgba(51, 65, 85, 0.4); }}
        .top-row {{ background-color: rgba(16, 185, 129, 0.1); }}
        .badge-pareto {{
            background-color: rgba(16, 185, 129, 0.2);
            color: var(--accent-green);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 12px;
        }}
        .badge-dominated {{
            background-color: rgba(148, 163, 184, 0.15);
            color: var(--text-secondary);
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .qual-badge {{
            background-color: rgba(129, 140, 248, 0.2);
            color: var(--accent-purple);
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
        }}
        .savings-positive {{ color: var(--accent-green); font-weight: 700; }}
        .savings-neutral {{ color: var(--text-secondary); }}
        code {{
            background-color: #0F172A;
            padding: 2px 6px;
            border-radius: 4px;
            color: #38BDF8;
            font-family: monospace;
            font-size: 13px;
        }}
        .section-title {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 15px;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>⚡ ViPym Compression & Pareto Decision Engine</h1>
            <div class="meta">Automated Multi-Objective Optimization Report</div>
        </div>
        <div class="meta">Generated: {now_str}</div>
    </div>

    {top_card_html}

    <div class="chart-section">
        <div class="chart-box">
            {pareto_chart_html}
        </div>
    </div>

    <div class="chart-box" style="margin-bottom: 30px;">
        <div class="section-title">📊 Ranked Deployment Candidates (Pareto Frontier)</div>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Model Variant</th>
                    <th>Method</th>
                    <th>Status</th>
                    <th>SE Quality</th>
                    <th>Cost ($ / 1M)</th>
                    <th>Throughput</th>
                    <th>p50 Latency</th>
                    <th>Hardware</th>
                </tr>
            </thead>
            <tbody>
                {rec_rows_html}
            </tbody>
        </table>
    </div>

    <div class="chart-section" style="grid-template-columns: 1fr 1fr;">
        <div class="chart-box">
            {heatmap_chart_html}
        </div>
        <div class="chart-box">
            {cost_chart_html}
        </div>
    </div>

    <div class="chart-box">
        <div class="section-title">💰 Enterprise Fleet Financial Comparison (15,000 Developers @ 33B Tokens/Mo)</div>
        <table>
            <thead>
                <tr>
                    <th>Deployment Option</th>
                    <th>Type</th>
                    <th>Infrastructure</th>
                    <th>Cost / 1M Tokens</th>
                    <th>Monthly Spend</th>
                    <th>Annual Spend</th>
                    <th>Savings vs Claude</th>
                </tr>
            </thead>
            <tbody>
                {cost_rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
        """

        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(html_content, encoding="utf-8")
        logger.info(f"HTML decision report written to {out_file.resolve()}")

        return html_content
