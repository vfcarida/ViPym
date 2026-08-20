"""Unit tests for P012 — Pareto Frontier, Cost Modeling, Recommender, and HTML Reporting.

Test classes:
  TestParetoFrontier            — Multi-objective dominance, fast non-dominated sorting (NSGA-II), Utopia distance
  TestDeploymentCostModel       — Hardware rate modeling, enterprise fleet projections, API reference comparison
  TestDeploymentRecommender     — Ranked recommendations, constraint filtering, ASCII report tables
  TestReportAndVisualizations   — Plotly interactive charts and standalone HTML report generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vipym.analysis.cost_model import (
    COMMERCIAL_APIS,
    DeploymentCostModel,
    EnterpriseWorkloadConfig,
    STANDARD_INSTANCES,
)
from vipym.analysis.pareto import ParetoFrontierOptimizer, ParetoPoint
from vipym.analysis.recommender import DeploymentRecommender, RecommendationReport
from vipym.analysis.report import HTMLReportGenerator
from vipym.analysis.visualizations import (
    create_cost_comparison_bar_chart,
    create_pareto_scatter_plot,
    create_quality_retention_heatmap,
)
from vipym.config import CostAssumptionConfig
from vipym.metrics.cost import AWSTraceableCostModel


# ============================================================
# TestParetoFrontier
# ============================================================


class TestParetoFrontier:
    def test_pareto_dominance_logic(self):
        optimizer = ParetoFrontierOptimizer()
        points = [
            ParetoPoint(
                experiment_id="exp1",
                configuration_name="Model_A",
                quality_score=0.85,
                latency_p50_ms=50.0,
                peak_vram_gb=16.0,
                cost_per_1m_tokens=2.0,
                throughput_tok_s=1000.0,
                compression_ratio=1.0,
            ),
            ParetoPoint(
                experiment_id="exp1",
                configuration_name="Model_B_Dominated",
                quality_score=0.70,  # Lower quality
                latency_p50_ms=60.0,  # Higher latency
                peak_vram_gb=20.0,  # Higher vram
                cost_per_1m_tokens=3.0,  # Higher cost
                throughput_tok_s=800.0,  # Lower throughput
                compression_ratio=0.8,
            ),
            ParetoPoint(
                experiment_id="exp1",
                configuration_name="Model_C_Optimal_Light",
                quality_score=0.80,
                latency_p50_ms=20.0,  # Faster
                peak_vram_gb=4.0,  # Less VRAM
                cost_per_1m_tokens=0.5,  # Cheaper
                throughput_tok_s=2500.0,
                compression_ratio=4.0,
            ),
        ]

        pareto_set = optimizer.compute_pareto_frontier(points)
        names = {p.configuration_name for p in pareto_set}

        assert "Model_A" in names
        assert "Model_C_Optimal_Light" in names
        assert "Model_B_Dominated" not in names

    def test_fast_non_dominated_sort_nsga2(self):
        optimizer = ParetoFrontierOptimizer()
        p1 = ParetoPoint(
            experiment_id="e1",
            configuration_name="P1_Best",
            quality_score=0.90,
            cost_per_1m_tokens=1.0,
            latency_p50_ms=30.0,
            peak_vram_gb=8.0,
            throughput_tok_s=1500.0,
        )
        p2 = ParetoPoint(
            experiment_id="e1",
            configuration_name="P2_Middle",
            quality_score=0.85,
            cost_per_1m_tokens=1.5,
            latency_p50_ms=40.0,
            peak_vram_gb=12.0,
            throughput_tok_s=1200.0,
        )
        p3 = ParetoPoint(
            experiment_id="e1",
            configuration_name="P3_Worst",
            quality_score=0.75,
            cost_per_1m_tokens=2.5,
            latency_p50_ms=60.0,
            peak_vram_gb=16.0,
            throughput_tok_s=900.0,
        )

        fronts = optimizer.fast_non_dominated_sort([p1, p2, p3])
        assert len(fronts) >= 2
        # Front 1 must have P1
        assert any(p.configuration_name == "P1_Best" for p in fronts[0])

    def test_filter_by_constraints(self):
        optimizer = ParetoFrontierOptimizer()
        points = [
            ParetoPoint(
                experiment_id="e",
                configuration_name="Good",
                quality_score=0.88,
                cost_per_1m_tokens=1.2,
                latency_p50_ms=35.0,
                peak_vram_gb=16.0,
            ),
            ParetoPoint(
                experiment_id="e",
                configuration_name="LowQuality",
                quality_score=0.60,
                cost_per_1m_tokens=0.2,
                latency_p50_ms=15.0,
                peak_vram_gb=4.0,
            ),
            ParetoPoint(
                experiment_id="e",
                configuration_name="TooExpensive",
                quality_score=0.95,
                cost_per_1m_tokens=5.0,
                latency_p50_ms=25.0,
                peak_vram_gb=32.0,
            ),
        ]

        filtered = optimizer.filter_by_constraints(
            points,
            min_quality=0.70,
            max_cost=2.0,
            max_latency_ms=40.0,
        )
        assert len(filtered) == 1
        assert filtered[0].configuration_name == "Good"

    def test_find_closest_to_utopia(self):
        optimizer = ParetoFrontierOptimizer()
        points = [
            ParetoPoint(
                experiment_id="e",
                configuration_name="Balanced_Top",
                quality_score=0.89,
                cost_per_1m_tokens=0.5,
                latency_p50_ms=25.0,
                peak_vram_gb=8.0,
            ),
            ParetoPoint(
                experiment_id="e",
                configuration_name="HighCost_HighQual",
                quality_score=0.91,
                cost_per_1m_tokens=4.0,
                latency_p50_ms=80.0,
                peak_vram_gb=32.0,
            ),
        ]
        best = optimizer.find_closest_to_utopia(points)
        assert best is not None
        assert best.configuration_name == "Balanced_Top"


# ============================================================
# TestDeploymentCostModel
# ============================================================


class TestDeploymentCostModel:
    def test_compute_cost_per_1m_tokens(self):
        model = DeploymentCostModel()
        # $32.77/hr on p5.48xlarge (8xH100) running at 2000 tok/s
        # 2000 tok/s * 3600s = 7.2M tok/hr
        # Cost/1M = ($32.77 / 7.2M) * 1M = $4.551
        cost_1m = model.compute_cost_per_1m_tokens(32.77, 2000.0)
        assert cost_1m == pytest.approx(4.551, abs=1e-2)

    def test_compute_variant_cost_and_tiers(self):
        model = DeploymentCostModel()
        breakdown_od = model.compute_variant_cost(
            variant_name="K3-AWQ-4bit",
            hardware_instance="g5.xlarge",
            throughput_tok_s=1000.0,
            pricing_tier="ondemand",
        )
        breakdown_spot = model.compute_variant_cost(
            variant_name="K3-AWQ-4bit",
            hardware_instance="g5.xlarge",
            throughput_tok_s=1000.0,
            pricing_tier="spot",
        )

        assert breakdown_od.hourly_rate_usd == pytest.approx(1.006, abs=1e-3)
        assert breakdown_spot.hourly_rate_usd == pytest.approx(0.352, abs=1e-3)
        assert breakdown_spot.cost_per_1m_tokens < breakdown_od.cost_per_1m_tokens

    def test_enterprise_workload_projection(self):
        workload = EnterpriseWorkloadConfig(
            developers=15000,
            requests_per_dev_per_day=50,
            avg_tokens_per_request=2000,
            working_days_per_month=22,
        )
        # Total monthly tokens = 15000 * 50 * 2000 * 22 = 33,000,000,000 (33B)
        assert workload.total_monthly_tokens == 33_000_000_000

        model = DeploymentCostModel(workload_config=workload)
        claude_spend = model.project_commercial_api_monthly_spend("claude_sonnet")
        assert claude_spend > 0.0

        gpt4o_spend = model.project_commercial_api_monthly_spend("gpt_4o")
        assert gpt4o_spend > 0.0

    def test_compute_annual_roi(self):
        model = DeploymentCostModel()
        roi = model.compute_annual_roi(
            self_hosted_monthly_spend=15_000.0,
            baseline_api_name="claude_sonnet",
        )
        assert roi["annual_savings_usd"] > 0.0
        assert roi["savings_percentage"] > 50.0

    def test_legacy_aws_cost_model(self):
        cost_cfg = CostAssumptionConfig(
            aws_ec2_hourly_rate=32.77,
            s3_storage_cost_per_gb_month=0.023,
            data_transfer_per_gb=0.09,
        )
        cost_model = AWSTraceableCostModel(cost_cfg)
        breakdown = cost_model.estimate_cost(
            duration_hours=2.0,
            storage_gb=100.0,
            data_transfer_gb=10.0,
            input_tokens=1_000_000,
            output_tokens=500_000,
            successful_tasks=50,
        )
        assert breakdown.total_cost_usd > 60.0


# ============================================================
# TestDeploymentRecommender
# ============================================================


class TestDeploymentRecommender:
    @pytest.fixture
    def sample_candidates(self) -> list[ParetoPoint]:
        return [
            ParetoPoint(
                experiment_id="e1",
                configuration_name="K3-AWQ-4bit",
                compression_method="awq",
                quality_score=0.97,
                relative_quality_score=0.97,
                cost_per_1m_tokens=1.20,
                throughput_tok_s=1800.0,
                latency_p50_ms=35.0,
                peak_vram_gb=28.0,
                hardware_recommendation="2x H100",
            ),
            ParetoPoint(
                experiment_id="e1",
                configuration_name="K3-Pruned50%",
                compression_method="prune_and_merge",
                quality_score=0.95,
                relative_quality_score=0.95,
                cost_per_1m_tokens=1.80,
                throughput_tok_s=1400.0,
                latency_p50_ms=45.0,
                peak_vram_gb=38.0,
                hardware_recommendation="3x H100",
            ),
            ParetoPoint(
                experiment_id="e1",
                configuration_name="K3-Distill32B",
                compression_method="distillation",
                quality_score=0.84,
                relative_quality_score=0.84,
                cost_per_1m_tokens=0.40,
                throughput_tok_s=3200.0,
                latency_p50_ms=18.0,
                peak_vram_gb=14.0,
                hardware_recommendation="1x H100",
            ),
            ParetoPoint(
                experiment_id="e1",
                configuration_name="K3-Distill14B",
                compression_method="distillation",
                quality_score=0.72,
                relative_quality_score=0.72,
                cost_per_1m_tokens=0.12,
                throughput_tok_s=5000.0,
                latency_p50_ms=10.0,
                peak_vram_gb=6.0,
                hardware_recommendation="1x A10G",
            ),
        ]

    def test_recommend_balanced_strategy(self, sample_candidates):
        recommender = DeploymentRecommender()
        report = recommender.recommend(
            candidates=sample_candidates,
            min_quality=0.80,
            strategy="balanced",
        )

        assert isinstance(report, RecommendationReport)
        assert report.recommended_variant is not None
        assert report.recommended_variant.configuration_name in {"K3-AWQ-4bit", "K3-Distill32B"}
        assert len(report.ranked_options) == 3  # K3-Distill14B filtered out (< 0.80)
        assert "ViPym Compression Recommendations" in report.ascii_table

    def test_recommend_cost_first(self, sample_candidates):
        recommender = DeploymentRecommender()
        report = recommender.recommend(
            candidates=sample_candidates,
            min_quality=0.70,
            strategy="cost_first",
        )
        assert report.recommended_variant is not None
        # Lowest cost is K3-Distill14B ($0.12)
        assert report.recommended_variant.configuration_name == "K3-Distill14B"

    def test_recommend_quality_first(self, sample_candidates):
        recommender = DeploymentRecommender()
        report = recommender.recommend(
            candidates=sample_candidates,
            min_quality=0.70,
            strategy="quality_first",
        )
        assert report.recommended_variant is not None
        # Highest quality is K3-AWQ-4bit (0.97)
        assert report.recommended_variant.configuration_name == "K3-AWQ-4bit"

    def test_recommend_empty_candidates(self):
        recommender = DeploymentRecommender()
        report = recommender.recommend(candidates=[])
        assert report.recommended_variant is None
        assert report.ranked_options == []


# ============================================================
# TestReportAndVisualizations
# ============================================================


class TestReportAndVisualizations:
    def test_generate_plotly_figures(self):
        points = [
            ParetoPoint(
                experiment_id="e1",
                configuration_name="Model_1",
                quality_score=0.90,
                cost_per_1m_tokens=1.5,
                latency_p50_ms=25.0,
                is_pareto_optimal=True,
                suite_scores={"swebench": 0.45, "aider_edit": 0.82},
            ),
            ParetoPoint(
                experiment_id="e1",
                configuration_name="Model_2",
                quality_score=0.75,
                cost_per_1m_tokens=0.5,
                latency_p50_ms=12.0,
                is_pareto_optimal=True,
                suite_scores={"swebench": 0.32, "aider_edit": 0.70},
            ),
        ]

        scatter = create_pareto_scatter_plot(points)
        assert scatter is not None

        heatmap = create_quality_retention_heatmap(points)
        assert heatmap is not None

        bar = create_cost_comparison_bar_chart(
            [
                {"name": "API: Claude", "type": "Commercial API", "monthly_spend_usd": 150000.0},
                {"name": "Model_1", "type": "Self-Hosted", "monthly_spend_usd": 30000.0},
            ]
        )
        assert bar is not None

    def test_html_report_generation(self, tmp_path):
        points = [
            ParetoPoint(
                experiment_id="e1",
                configuration_name="K3-AWQ-4bit",
                compression_method="awq",
                quality_score=0.97,
                relative_quality_score=0.97,
                cost_per_1m_tokens=1.20,
                throughput_tok_s=1800.0,
                latency_p50_ms=35.0,
                hardware_recommendation="2x H100",
                is_pareto_optimal=True,
            )
        ]
        recommender = DeploymentRecommender()
        rec_report = recommender.recommend(points)

        report_gen = HTMLReportGenerator()
        out_file = tmp_path / "test_report.html"
        html = report_gen.generate_report(
            points=points,
            recommendation=rec_report,
            cost_comparison=[
                {
                    "name": "K3-AWQ-4bit",
                    "type": "Self-Hosted",
                    "hardware": "2x H100",
                    "cost_per_1m_tokens": 1.20,
                    "monthly_spend_usd": 39600.0,
                    "annual_spend_usd": 475200.0,
                    "savings_vs_claude_pct": 82.5,
                }
            ],
            output_path=out_file,
        )

        assert out_file.exists()
        assert "ViPym Compression & Pareto Decision Engine" in html
        assert "K3-AWQ-4bit" in html
