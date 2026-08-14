"""Unified Report Generator orchestrating Markdown, HTML, LaTeX, and Interactive Plots."""

from pathlib import Path
from typing import Any, Dict, List
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
        results: List[ParetoPoint],
        manifest_meta: Dict[str, Any],
    ) -> Dict[str, Path]:
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

        logger.info(f"Generated {len(generated_files)} report artifacts in {self.output_dir}")
        return generated_files
