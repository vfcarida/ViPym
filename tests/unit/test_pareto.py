"""Unit tests for Pareto Frontier and Cost modeling."""

from vipym.analysis.pareto import ParetoFrontierOptimizer, ParetoPoint
from vipym.config import CostAssumptionConfig
from vipym.metrics.cost import AWSTraceableCostModel


def test_pareto_frontier_optimizer():
    optimizer = ParetoFrontierOptimizer()
    points = [
        ParetoPoint(
            experiment_id="exp1",
            configuration_name="Model_A",
            quality_score=0.85,
            latency_p50_ms=50.0,
            peak_vram_gb=16.0,
            cost_usd=2.0,
            compression_ratio=1.0,
        ),
        ParetoPoint(
            experiment_id="exp1",
            configuration_name="Model_B_Dominated",
            quality_score=0.70,  # lower quality
            latency_p50_ms=60.0,  # higher latency
            peak_vram_gb=20.0,  # higher vram
            cost_usd=3.0,
            compression_ratio=0.8,
        ),
        ParetoPoint(
            experiment_id="exp1",
            configuration_name="Model_C_Optimal_Light",
            quality_score=0.80,
            latency_p50_ms=20.0,  # significantly faster
            peak_vram_gb=4.0,  # much less vram
            cost_usd=0.5,
            compression_ratio=4.0,
        ),
    ]

    pareto_set = optimizer.compute_pareto_frontier(points)
    names = {p.configuration_name for p in pareto_set}

    assert "Model_A" in names
    assert "Model_C_Optimal_Light" in names
    assert "Model_B_Dominated" not in names


def test_aws_cost_model():
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
    assert breakdown.cost_per_successful_task_usd > 0.0
