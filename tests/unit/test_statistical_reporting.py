"""Unit tests for StatisticalAnalyzer and statistical reporting integration."""

from __future__ import annotations

from pathlib import Path

from vipym.analysis.pareto import ParetoPoint
from vipym.analysis.statistics import StatisticalAnalyzer, StatisticalSignificanceReport
from vipym.reporting.generator import ExperimentReportGenerator


class TestStatisticalReporting:
    def test_bootstrap_confidence_interval(self):
        """Verify bootstrap confidence interval calculation."""
        scores = [0.80, 0.82, 0.79, 0.81, 0.80, 0.83, 0.78]
        mean, low, high = StatisticalAnalyzer.bootstrap_confidence_interval(
            scores, num_resamples=500
        )
        assert 0.75 < mean < 0.85
        assert low <= mean <= high

    def test_evaluate_significance_identical(self):
        """Verify statistical equivalence on identical baseline and compressed scores."""
        scores = [0.85, 0.85, 0.84, 0.86, 0.85]
        report = StatisticalAnalyzer.evaluate_significance(scores, scores)
        assert isinstance(report, StatisticalSignificanceReport)
        assert not report.is_statistically_significant
        assert "STATISTICALLY EQUIVALENT" in report.verdict

    def test_evaluate_significance_regression(self):
        """Verify significant regression is detected when compression severely degrades performance."""
        b_scores = [0.90, 0.92, 0.89, 0.91, 0.90]
        c_scores = [0.40, 0.42, 0.39, 0.41, 0.40]
        report = StatisticalAnalyzer.evaluate_significance(b_scores, c_scores)
        assert report.relative_change_pct < -50.0
        assert report.p_value < 0.05
        assert report.is_statistically_significant

    def test_report_generator_includes_statistical_validation(self, tmp_path: Path):
        """Verify ExperimentReportGenerator writes statistical hypothesis tests to recommendation.md."""
        gen = ExperimentReportGenerator(output_dir=tmp_path / "reports")
        baseline = ParetoPoint(
            experiment_id="test-exp",
            configuration_name="Baseline",
            quality_score=0.85,
            latency_p50_ms=45.0,
            peak_vram_gb=16.0,
            cost_usd=2.50,
            compression_ratio=1.0,
        )
        compressed = [
            ParetoPoint(
                experiment_id="test-exp",
                configuration_name="Compressed-W4A16",
                quality_score=0.84,
                latency_p50_ms=28.0,
                peak_vram_gb=4.0,
                cost_usd=0.80,
                compression_ratio=4.0,
            )
        ]

        files = gen.generate_all(
            experiment_id="test-exp",
            baseline=baseline,
            results=compressed,
            manifest_meta={},
        )

        rec_file = files["recommendation_md"]
        assert rec_file.exists()
        content = rec_file.read_text(encoding="utf-8")
        assert "Statistical Hypothesis Testing" in content
        assert "Mann-Whitney U Test" in content
        assert "95% CI" in content
