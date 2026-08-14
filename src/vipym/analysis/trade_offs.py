"""Trade-off marginal analysis and knee-point calculation."""

from vipym.analysis.pareto import ParetoPoint


class TradeOffAnalyzer:
    """Calculates marginal capability retention per unit of VRAM/Latency reduction."""

    @staticmethod
    def calculate_marginal_efficiency(
        baseline: ParetoPoint, compressed: ParetoPoint
    ) -> dict[str, float]:
        vram_saved_gb = max(0.0, baseline.peak_vram_gb - compressed.peak_vram_gb)
        quality_delta = compressed.quality_score - baseline.quality_score
        cost_saved_usd = max(0.0, baseline.cost_usd - compressed.cost_usd)

        retention_rate = (compressed.quality_score / max(1e-6, baseline.quality_score)) * 100.0

        return {
            "vram_saved_gb": vram_saved_gb,
            "cost_saved_usd": cost_saved_usd,
            "quality_retention_percent": retention_rate,
            "quality_delta": quality_delta,
            "retention_per_gb_saved": (retention_rate / max(0.1, vram_saved_gb))
            if vram_saved_gb > 0
            else 0.0,
        }
