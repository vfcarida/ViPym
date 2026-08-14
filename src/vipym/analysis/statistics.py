"""Statistical hypothesis testing and bootstrap confidence interval calculations."""

import numpy as np
from scipy import stats


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
        """Perform Mann-Whitney U and Wilcoxon rank-sum tests."""
        if len(baseline_scores) < 2 or len(compressed_scores) < 2:
            return {"p_value": 1.0, "statistically_significant": 0.0}

        stat, p_val = stats.mannwhitneyu(
            baseline_scores, compressed_scores, alternative="two-sided"
        )
        return {
            "u_statistic": float(stat),
            "p_value": float(p_val),
            "statistically_significant": float(p_val < 0.05),
        }
