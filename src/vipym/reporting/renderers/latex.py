"""LaTeX and HTML report renderers."""

from typing import Any, Dict, List
from vipym.analysis.pareto import ParetoPoint


class LaTeXTableRenderer:
    """Renders camera-ready LaTeX table for academic research papers."""

    @staticmethod
    def render(experiment_id: str, results: List[ParetoPoint]) -> str:
        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{ViPym Compression Trade-off Comparison for " + experiment_id + r"}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Configuration} & \textbf{Pass@1 (\%)} & \textbf{Latency (ms)} & \textbf{VRAM (GB)} & \textbf{Ratio} & \textbf{Pareto} \\",
            r"\midrule",
        ]
        for r in results:
            pareto_str = r"\checkmark" if r.is_pareto_optimal else ""
            lines.append(
                f"{r.configuration_name.replace('_', r'\_')} & {r.quality_score*100:.1f} & {r.latency_p50_ms:.1f} & {r.peak_vram_gb:.1f} & {r.compression_ratio:.1f}x & {pareto_str} \\\\"
            )
        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\label{tab:vipym_results}",
            r"\end{table*}",
        ])
        return "\n".join(lines)


class HTMLReportRenderer:
    """Renders standalone interactive HTML dashboard."""

    @staticmethod
    def render(experiment_id: str, results: List[ParetoPoint]) -> str:
        rows = "".join(
            f"<tr><td><code>{r.configuration_name}</code></td><td>{r.quality_score*100:.1f}%</td><td>{r.latency_p50_ms:.1f} ms</td><td>{r.peak_vram_gb:.1f} GB</td><td>{r.compression_ratio:.1f}x</td><td>{'⭐ Yes' if r.is_pareto_optimal else 'No'}</td></tr>"
            for r in results
        )
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ViPym Report: {experiment_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #0f172a; color: #f8fafc; }}
        h1 {{ color: #38bdf8; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: #1e293b; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0284c7; color: white; }}
        tr:hover {{ background: #334155; }}
        code {{ color: #f43f5e; }}
    </style>
</head>
<body>
    <h1>ViPym Experiment Report: {experiment_id}</h1>
    <p>Shrinking LLMs, Preserving Intelligence — Multidimensional Pareto Analysis</p>
    <table>
        <thead>
            <tr><th>Configuration</th><th>Pass@1</th><th>P50 Latency</th><th>Peak VRAM</th><th>Compression Ratio</th><th>Pareto Optimal</th></tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
</body>
</html>"""
