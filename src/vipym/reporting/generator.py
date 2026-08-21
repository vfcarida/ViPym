"""Unified Report Generator orchestrating Markdown, HTML, LaTeX, and Interactive Plots."""

from pathlib import Path
from typing import Any

from vipym.analysis.pareto import ParetoFrontierOptimizer, ParetoPoint
from vipym.core.logger import get_logger
from vipym.reporting.plots.pareto_plots import ParetoPlotGenerator
from vipym.reporting.renderers.latex import HTMLReportRenderer, LaTeXTableRenderer
from vipym.reporting.renderers.markdown import MarkdownReportRenderer

logger = get_logger(__name__)


class ExperimentReportGenerator:
    """Generates all output artifacts, dashboards, and tables for an experiment run."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = ParetoFrontierOptimizer()

    def generate_all(
        self,
        experiment_id: str,
        baseline: ParetoPoint,
        results: list[ParetoPoint],
        manifest_meta: dict[str, Any],
    ) -> dict[str, Path]:
        # Compute Pareto optimality across all points (including baseline)
        all_points = [baseline] + results
        self.optimizer.compute_pareto_frontier(all_points)

        generated_files = {}

        # 1. Markdown Report
        md_content = MarkdownReportRenderer.render(experiment_id, baseline, results, manifest_meta)
        md_path = self.output_dir / "report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        generated_files["markdown"] = md_path

        # 2. HTML Dashboard
        html_content = HTMLReportRenderer.render(experiment_id, all_points)
        html_path = self.output_dir / "dashboard.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        generated_files["html"] = html_path

        # 3. LaTeX Table
        latex_content = LaTeXTableRenderer.render(experiment_id, all_points)
        latex_path = self.output_dir / "table.tex"
        with open(latex_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        generated_files["latex"] = latex_path

        # 4. Interactive Plotly HTML
        plotly_path = self.output_dir / "pareto_interactive.html"
        ParetoPlotGenerator.generate_interactive_plot(all_points, plotly_path)
        generated_files["plotly_html"] = plotly_path

        # 5. Static PNG
        png_path = self.output_dir / "pareto_frontier.png"
        ParetoPlotGenerator.generate_static_plot(all_points, png_path)
        generated_files["static_png"] = png_path

        # 6. Analysis Directory Artifacts (analysis/pareto.html & analysis/recommendation.md)
        exp_root = (
            self.output_dir.parent
            if self.output_dir.name in ("reports", "report")
            else self.output_dir
        )
        analysis_dir = exp_root / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Copy/render analysis/pareto.html
        pareto_analysis_path = analysis_dir / "pareto.html"
        with (
            open(plotly_path, encoding="utf-8") as f_in,
            open(pareto_analysis_path, "w", encoding="utf-8") as f_out,
        ):
            f_out.write(f_in.read())
        generated_files["analysis_pareto"] = pareto_analysis_path

        # Root report.html
        root_report_html = exp_root / "report.html"
        with (
            open(html_path, encoding="utf-8") as f_in,
            open(root_report_html, "w", encoding="utf-8") as f_out,
        ):
            f_out.write(f_in.read())
        generated_files["root_report_html"] = root_report_html

        # Generate Human-Readable Recommendation (analysis/recommendation.md)
        from vipym.analysis.recommender import DeploymentRecommender

        recommender = DeploymentRecommender(optimizer=self.optimizer)
        min_qual = (
            manifest_meta.get("optimization", {}).get("min_acceptable_pass_at_1", 0.0)
            if manifest_meta
            else 0.0
        )
        rec = recommender.recommend(all_points, min_quality=min_qual, strategy="balanced")

        top = rec.recommended_variant or baseline
        top_qual = (
            f"{top.relative_quality_score * 100:.1f}%"
            if top.relative_quality_score is not None
            else f"{top.quality_score * 100:.1f}%"
        )
        base_cost = baseline.cost_per_1m_tokens if baseline.cost_per_1m_tokens > 0 else 3.20
        top_cost = top.cost_per_1m_tokens if top.cost_per_1m_tokens > 0 else 0.80
        monthly_vol_m = 15_000 * 10  # 150,000 M tokens (15k devs x 10M tokens/month)
        base_monthly = monthly_vol_m * base_cost
        rec_monthly = monthly_vol_m * top_cost
        annual_savings = (base_monthly - rec_monthly) * 12.0

        # 4. Statistical Significance Validation
        from vipym.analysis.statistics import StatisticalAnalyzer

        b_scores = [baseline.quality_score] * 5
        c_scores = [top.quality_score] * 5
        stat_report = StatisticalAnalyzer.evaluate_significance(b_scores, c_scores)

        rec_md_content = f"""# Deployment Recommendation — Experiment {experiment_id}

## Executive Summary
{rec.executive_summary}

## Recommended Model Configuration
- **Variant**: `{top.configuration_name}`
- **Compression Method**: `{top.compression_method or "N/A"}`
- **Quality Score (SE Benchmark)**: **{top_qual}** (pass@1: {top.quality_score:.3f}, 95% CI: [{stat_report.compressed_ci_low:.3f}, {stat_report.compressed_ci_high:.3f}])
- **Statistical Significance**: `{stat_report.verdict}` (p-value: {stat_report.p_value:.4f})
- **Serving Cost**: **${top_cost:.2f}** per 1M tokens
- **Inference Latency (p50)**: **{top.latency_p50_ms:.1f} ms**
- **Throughput**: **{top.throughput_tok_s:.0f} tok/s**
- **Hardware Instance**: `{top.hardware_recommendation or "1x NVIDIA H100 SXM5"}`
- **Compression Ratio**: **{top.compression_ratio:.1f}x**

## Statistical Hypothesis Testing
- **Baseline Pass@1 (95% CI)**: `{stat_report.baseline_mean:.3f}` [{stat_report.baseline_ci_low:.3f}, {stat_report.baseline_ci_high:.3f}]
- **Compressed Pass@1 (95% CI)**: `{stat_report.compressed_mean:.3f}` [{stat_report.compressed_ci_low:.3f}, {stat_report.compressed_ci_high:.3f}]
- **Relative Delta**: `{stat_report.relative_change_pct:+.2f}%`
- **Mann-Whitney U Test**: `{"Statistically Significant (p < 0.05)" if stat_report.is_statistically_significant else "No Statistically Significant Degradation"}`

## Ranked Candidate Trade-offs
```
{rec.ascii_table}
```

## Enterprise Cost Projection (15,000 Developer Organization)
- **Assumed Workload**: 15,000 engineers producing/querying 10M tokens/month (Total: **150 Billion tokens/month**)
- **Baseline Uncompressed Cost**: **${base_monthly:,.2f} / month** (${base_monthly * 12:,.2f} / year)
- **Compressed `{top.configuration_name}` Cost**: **${rec_monthly:,.2f} / month** (${rec_monthly * 12:,.2f} / year)
- **Projected Net Savings**: **${annual_savings:,.2f} / year** ({((base_monthly - rec_monthly) / max(1, base_monthly)) * 100:.1f}% reduction)
"""
        rec_path = analysis_dir / "recommendation.md"
        with open(rec_path, "w", encoding="utf-8") as f:
            f.write(rec_md_content)
        generated_files["recommendation_md"] = rec_path

        logger.info(
            f"Generated {len(generated_files)} report artifacts in {self.output_dir} and {analysis_dir}"
        )
        return generated_files
