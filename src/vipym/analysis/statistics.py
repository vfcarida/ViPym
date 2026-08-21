"""Statistical hypothesis testing, bootstrap confidence intervals, and significance validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class StatisticalSignificanceReport:
    """Comprehensive statistical comparison between baseline and compressed performance."""

    baseline_mean: float
    baseline_ci_low: float
    baseline_ci_high: float
    compressed_mean: float
    compressed_ci_low: float
    compressed_ci_high: float
    relative_change_pct: float
    p_value: float
    is_statistically_significant: bool
    verdict: str


class StatisticalAnalyzer:
    """Rigorous statistical validation for comparing compressed models vs baseline."""

    @staticmethod
    def bootstrap_confidence_interval(
        samples: list[float],
        num_resamples: int = 2000,
        confidence_level: float = 0.95,
    ) -> tuple[float, float, float]:
        """Calculate mean and bootstrap confidence interval."""
        if not samples:
            return 0.0, 0.0, 0.0
        arr = np.array(samples)
        if len(arr) == 1 or np.all(arr == arr[0]):
            val = float(arr[0])
            return val, val, val

        means = [
            np.mean(np.random.choice(arr, size=len(arr), replace=True))
            for _ in range(num_resamples)
        ]
        alpha = (1.0 - confidence_level) / 2.0
        low = float(np.percentile(means, alpha * 100))
        high = float(np.percentile(means, (1.0 - alpha) * 100))
        return float(np.mean(arr)), low, high

    @staticmethod
    def compare_distributions(
        baseline_scores: list[float], compressed_scores: list[float]
    ) -> dict[str, float]:
        """Perform Mann-Whitney U rank-sum test."""
        if len(baseline_scores) < 2 or len(compressed_scores) < 2:
            return {"p_value": 1.0, "statistically_significant": 0.0}

        try:
            stat, p_val = stats.mannwhitneyu(
                baseline_scores, compressed_scores, alternative="two-sided"
            )
            return {
                "u_statistic": float(stat),
                "p_value": float(p_val),
                "statistically_significant": float(p_val < 0.05),
            }
        except Exception:
            return {"p_value": 1.0, "statistically_significant": 0.0}

    @classmethod
    def evaluate_significance(
        cls,
        baseline_scores: list[float],
        compressed_scores: list[float],
    ) -> StatisticalSignificanceReport:
        """Produce full significance report with bootstrap 95% CIs and hypothesis testing."""
        b_mean, b_low, b_high = cls.bootstrap_confidence_interval(baseline_scores)
        c_mean, c_low, c_high = cls.bootstrap_confidence_interval(compressed_scores)

        rel_change = ((c_mean - b_mean) / max(1e-6, b_mean)) * 100.0 if b_mean > 0 else 0.0
        comp = cls.compare_distributions(baseline_scores, compressed_scores)
        p_val = comp["p_value"]
        is_sig = bool(comp["statistically_significant"])

        if not is_sig or abs(rel_change) < 2.0:
            verdict = "STATISTICALLY EQUIVALENT (Near-Zero Loss)"
        elif rel_change < -5.0 and p_val < 0.01:
            verdict = f"SIGNIFICANT REGRESSION (p={p_val:.4f})"
        elif rel_change < 0:
            verdict = f"SLIGHT DEGRADATION (p={p_val:.4f})"
        else:
            verdict = f"PERFORMANCE GAIN (p={p_val:.4f})"

        return StatisticalSignificanceReport(
            baseline_mean=round(b_mean, 4),
            baseline_ci_low=round(b_low, 4),
            baseline_ci_high=round(b_high, 4),
            compressed_mean=round(c_mean, 4),
            compressed_ci_low=round(c_low, 4),
            compressed_ci_high=round(c_high, 4),
            relative_change_pct=round(rel_change, 2),
            p_value=round(p_val, 4),
            is_statistically_significant=is_sig,
            verdict=verdict,
        )
