"""Interactive and static Pareto Plot generators (Plotly & Matplotlib)."""

from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from vipym.analysis.pareto import ParetoPoint


class ParetoPlotGenerator:
    """Generates interactive Plotly and static Matplotlib Pareto Frontier charts."""

    @staticmethod
    def generate_interactive_plot(points: List[ParetoPoint], output_html: Optional[Path] = None) -> go.Figure:
        data = [
            {
                "Config": p.configuration_name,
                "Pass@1 (%)": p.quality_score * 100,
                "Peak VRAM (GB)": p.peak_vram_gb,
                "P50 Latency (ms)": p.latency_p50_ms,
                "Est. Cost ($)": p.cost_usd,
                "Pareto": "Optimal" if p.is_pareto_optimal else "Suboptimal",
            }
            for p in points
        ]
        import pandas as pd
        df = pd.DataFrame(data)

        fig = px.scatter(
            df,
            x="Peak VRAM (GB)",
            y="Pass@1 (%)",
            color="Pareto",
            size="Est. Cost ($)",
            hover_name="Config",
            hover_data=["P50 Latency (ms)", "Est. Cost ($)"],
            title="ViPym Pareto Frontier: Intelligence Retention vs VRAM Footprint",
            template="plotly_dark",
        )

        if output_html:
            Path(output_html).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_html))

        return fig

    @staticmethod
    def generate_static_plot(points: List[ParetoPoint], output_png: Path) -> None:
        plt.figure(figsize=(8, 6))
        for p in points:
            color = "gold" if p.is_pareto_optimal else "steelblue"
            marker = "*" if p.is_pareto_optimal else "o"
            size = 140 if p.is_pareto_optimal else 70
            plt.scatter(p.peak_vram_gb, p.quality_score * 100, c=color, marker=marker, s=size)
            plt.text(p.peak_vram_gb + 0.5, p.quality_score * 100, p.configuration_name, fontsize=8)

        plt.xlabel("Peak VRAM Footprint (GB)")
        plt.ylabel("Pass@1 Accuracy (%)")
        plt.title("ViPym: Capability Retention vs VRAM Footprint")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        Path(output_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(output_png), dpi=300)
        plt.close()
