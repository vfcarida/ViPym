"""Interactive Visualizations for Pareto Frontier, Cost Comparison, and Quality Heatmaps.

Generates interactive Plotly charts with modern dark styling and rich annotations:
- 2D/3D Multi-Objective Pareto Scatter Plots (Quality vs Cost vs Latency)
- Quality Retention Breakdown Heatmap (Suites × Model Variants)
- Enterprise Serving Cost Comparison Bar Charts vs Hosted APIs (Claude, GPT-4o, DeepSeek)
"""

from __future__ import annotations

from typing import Any

from vipym.analysis.pareto import ParetoPoint

try:
    import plotly.graph_objects as go
    import plotly.io as pio

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def create_pareto_scatter_plot(
    points: list[ParetoPoint],
    x_metric: str = "cost_per_1m_tokens",
    y_metric: str = "quality_score",
) -> Any:
    """Generate interactive 2D Pareto frontier scatter plot."""
    if not HAS_PLOTLY or not points:
        return None

    pareto_pts = [p for p in points if p.is_pareto_optimal]
    non_pareto_pts = [p for p in points if not p.is_pareto_optimal]

    # Sort pareto points by cost for clean line connection
    pareto_pts_sorted = sorted(pareto_pts, key=lambda p: getattr(p, x_metric, 0.0))

    fig = go.Figure()

    # 1. Add Non-Pareto dominated points
    if non_pareto_pts:
        fig.add_trace(
            go.Scatter(
                x=[getattr(p, x_metric, 0.0) for p in non_pareto_pts],
                y=[getattr(p, y_metric, 0.0) for p in non_pareto_pts],
                mode="markers+text",
                name="Dominated Variants",
                text=[p.configuration_name for p in non_pareto_pts],
                textposition="top center",
                marker=dict(
                    size=14,
                    color="#64748B",  # Slate gray
                    opacity=0.6,
                    line=dict(width=1, color="#94A3B8"),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + f"{x_metric}: $%{{x:.2f}}<br>"
                    + f"{y_metric}: %{{y:.3f}}<br>"
                    + "<extra></extra>"
                ),
            )
        )

    # 2. Add Pareto-Optimal points
    if pareto_pts_sorted:
        fig.add_trace(
            go.Scatter(
                x=[getattr(p, x_metric, 0.0) for p in pareto_pts_sorted],
                y=[getattr(p, y_metric, 0.0) for p in pareto_pts_sorted],
                mode="markers+text",
                name="Pareto Optimal Frontier",
                text=[p.configuration_name for p in pareto_pts_sorted],
                textposition="bottom right",
                marker=dict(
                    size=18,
                    color="#10B981",  # Emerald green
                    symbol="diamond",
                    line=dict(width=2, color="#FFFFFF"),
                ),
                hovertemplate=(
                    "<b>%{text} (Pareto Optimal)</b><br>"
                    + f"{x_metric}: $%{{x:.2f}}<br>"
                    + f"{y_metric}: %{{y:.3f}}<br>"
                    + "<extra></extra>"
                ),
            )
        )

        # 3. Add Pareto frontier connecting line
        fig.add_trace(
            go.Scatter(
                x=[getattr(p, x_metric, 0.0) for p in pareto_pts_sorted],
                y=[getattr(p, y_metric, 0.0) for p in pareto_pts_sorted],
                mode="lines",
                name="Frontier Curve",
                line=dict(color="#10B981", width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title="<b>Multi-Objective Pareto Frontier: SE Quality vs Serving Cost</b>",
        xaxis=dict(
            title=f"<b>Cost ($ / 1M Tokens)</b> [{x_metric}]",
            gridcolor="#334155",
            showgrid=True,
        ),
        yaxis=dict(
            title=f"<b>Quality Score</b> [{y_metric}]",
            gridcolor="#334155",
            showgrid=True,
        ),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(15, 23, 42, 0.8)"),
    )

    return fig


def create_quality_retention_heatmap(
    points: list[ParetoPoint],
) -> Any:
    """Generate heatmap of benchmark suite scores across compressed model variants."""
    if not HAS_PLOTLY or not points:
        return None

    variants = [p.configuration_name for p in points]
    suites = ["swebench", "aider_edit", "bigcodebench", "testgeneval", "crqbench"]
    suite_labels = ["SWE-bench", "Aider Edit", "BigCodeBench", "TestGenEval", "CRQBench Review"]

    z_data: list[list[float]] = []
    for s in suites:
        row: list[float] = []
        for p in points:
            score = p.suite_scores.get(s, p.quality_score)
            row.append(score)
        z_data.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=variants,
            y=suite_labels,
            colorscale="Viridis",
            text=[[f"{val:.2f}" for val in row] for row in z_data],
            texttemplate="%{text}",
            textfont=dict(size=12, color="#FFFFFF"),
            colorbar=dict(title="Score"),
        )
    )

    fig.update_layout(
        title="<b>SE Benchmark Capability Retention Across Model Variants</b>",
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        margin=dict(l=100, r=40, t=60, b=60),
    )

    return fig


def create_cost_comparison_bar_chart(
    comparison_matrix: list[dict[str, Any]],
) -> Any:
    """Generate enterprise monthly spend comparison bar chart against hosted commercial APIs."""
    if not HAS_PLOTLY or not comparison_matrix:
        return None

    names = [row["name"] for row in comparison_matrix]
    monthly_spends = [row["monthly_spend_usd"] for row in comparison_matrix]
    colors = ["#EF4444" if "API" in row.get("type", "") else "#3B82F6" for row in comparison_matrix]

    fig = go.Figure(
        data=go.Bar(
            x=names,
            y=monthly_spends,
            marker=dict(color=colors),
            text=[f"${val:,.0f}" for val in monthly_spends],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Enterprise Monthly Spend (15,000 Developers): Self-Hosted vs Hosted APIs</b>",
        xaxis=dict(title="<b>Deployment Target</b>", tickangle=-20),
        yaxis=dict(title="<b>Monthly Cost ($ USD)</b>", gridcolor="#334155"),
        template="plotly_dark",
        paper_bgcolor="#0F172A",
        plot_bgcolor="#1E293B",
        font=dict(family="Inter, sans-serif", color="#F8FAFC"),
        margin=dict(l=80, r=40, t=60, b=100),
    )

    return fig
