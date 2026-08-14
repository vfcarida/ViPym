"""Markdown experiment report renderer."""

from typing import Any

from vipym.analysis.pareto import ParetoPoint


class MarkdownReportRenderer:
    """Renders comprehensive, publication-quality Markdown reports."""

    @staticmethod
    def render(
        experiment_id: str,
        baseline: ParetoPoint,
        results: list[ParetoPoint],
        manifest_meta: dict[str, Any],
    ) -> str:
        lines = [
            f"# ViPym Experiment Report: `{experiment_id}`",
            "",
            "> **Research Question:** *How much can we shrink this LLM before reduction in intelligence becomes unacceptable?*",
            "",
            "## 1. System & Reproducibility Summary",
            "",
            f"* **Timestamp:** {manifest_meta.get('timestamp_utc', 'N/A')}",
            f"* **ViPym Version:** `{manifest_meta.get('vipym_version', '0.1.0')}`",
            f"* **Git Commit:** `{manifest_meta.get('git_commit_sha', 'N/A')}`",
            f"* **PyTorch / Transformers / vLLM:** `{manifest_meta.get('torch_version', 'N/A')}` / `{manifest_meta.get('transformers_version', 'N/A')}` / `{manifest_meta.get('vllm_version', 'N/A')}`",
            f"* **GPU Architecture:** {', '.join(manifest_meta.get('gpu_devices', ['CPU/Mock']))}",
            "",
            "## 2. Benchmark Comparison Table",
            "",
            "| Configuration | Scheme | Pass@1 (%) | P50 Latency (ms) | Peak VRAM (GB) | Est. Cost ($) | Pareto Optimal? |",
            "|---|---|---|---|---|---|---|",
        ]

        # Add baseline row
        lines.append(
            f"| **Baseline** | Native | **{baseline.quality_score * 100:.1f}%** | {baseline.latency_p50_ms:.1f} | {baseline.peak_vram_gb:.1f} | ${baseline.cost_usd:.2f} | - |"
        )

        for pt in results:
            pareto_mark = "🌟 **Yes**" if pt.is_pareto_optimal else "No"
            lines.append(
                f"| `{pt.configuration_name}` | Auto | {pt.quality_score * 100:.1f}% | {pt.latency_p50_ms:.1f} | {pt.peak_vram_gb:.1f} | ${pt.cost_usd:.2f} | {pareto_mark} |"
            )

        lines.extend(
            [
                "",
                "## 3. Key Findings & Recommendation",
                "",
                "Configurations marked with 🌟 represent non-dominated Pareto efficient trade-offs across capability, memory, and operational cost.",
            ]
        )

        return "\n".join(lines)
