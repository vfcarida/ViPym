"""Unit tests verifying the centralized cloud pricing catalog and backward compatibility."""

import pytest

from vipym.analysis.cost_model import (
    COMMERCIAL_APIS as COST_MODEL_APIS,
)
from vipym.analysis.cost_model import (
    STANDARD_INSTANCES as COST_MODEL_INSTANCES,
)
from vipym.analysis.cost_model import (
    DeploymentCostModel,
)
from vipym.analysis.cost_model import (
    InstancePricing as CostModelInstancePricing,
)
from vipym.cost import (
    COMMERCIAL_API_PRICING,
    COMMERCIAL_APIS,
    DEFAULT_HARDWARE_RATES,
    STANDARD_INSTANCES,
    InstancePricing,
    get_instance_hourly_rate,
)
from vipym.telemetry.cost_tracker import (
    COMMERCIAL_API_PRICING as TELEMETRY_API_PRICING,
)
from vipym.telemetry.cost_tracker import (
    DEFAULT_HARDWARE_RATES as TELEMETRY_HARDWARE_RATES,
)
from vipym.telemetry.cost_tracker import (
    InferenceCostTracker,
)


def test_backward_compatibility_reexports():
    """Ensure modules re-export identical references for full backward compatibility."""
    assert STANDARD_INSTANCES is COST_MODEL_INSTANCES
    assert COMMERCIAL_APIS is COST_MODEL_APIS
    assert InstancePricing is CostModelInstancePricing
    assert DEFAULT_HARDWARE_RATES is TELEMETRY_HARDWARE_RATES
    assert COMMERCIAL_API_PRICING is TELEMETRY_API_PRICING


def test_standard_instances_structure_and_pricing_tiers():
    """Verify all standard instances have valid positive pricing tiers."""
    assert len(STANDARD_INSTANCES) >= 10
    required_instances = [
        "p5.48xlarge",
        "p5e.48xlarge",
        "p4de.24xlarge",
        "g6.12xlarge",
        "g5.12xlarge",
        "g5.xlarge",
    ]
    for name in required_instances:
        assert name in STANDARD_INSTANCES
        inst = STANDARD_INSTANCES[name]
        assert inst.instance_type == name
        assert inst.gpu_count > 0
        assert inst.hourly_rate_ondemand > 0
        assert inst.hourly_rate_spot > 0
        assert inst.hourly_rate_reserved_1yr > 0
        assert inst.hourly_rate_reserved_3yr > 0
        # Spot and reserved rates must be strictly cheaper than on-demand
        assert inst.hourly_rate_spot < inst.hourly_rate_ondemand
        assert inst.hourly_rate_reserved_1yr < inst.hourly_rate_ondemand
        assert inst.hourly_rate_reserved_3yr < inst.hourly_rate_ondemand


def test_get_rate_method_variations():
    """Verify InstancePricing.get_rate handles tier string variations."""
    p5 = STANDARD_INSTANCES["p5.48xlarge"]
    assert p5.get_rate("ondemand") == p5.hourly_rate_ondemand
    assert p5.get_rate("spot") == p5.hourly_rate_spot
    assert p5.get_rate("SPOT") == p5.hourly_rate_spot
    assert p5.get_rate("1yr") == p5.hourly_rate_reserved_1yr
    assert p5.get_rate("1_year") == p5.hourly_rate_reserved_1yr
    assert p5.get_rate("reserved") == p5.hourly_rate_reserved_1yr
    assert p5.get_rate("3yr") == p5.hourly_rate_reserved_3yr
    assert p5.get_rate("3_year") == p5.hourly_rate_reserved_3yr
    # Fallback default tier
    assert p5.get_rate("unknown_tier") == p5.hourly_rate_ondemand


def test_get_instance_hourly_rate():
    """Verify get_instance_hourly_rate lookup and case-insensitivity."""
    assert get_instance_hourly_rate("p5.48xlarge") == 32.77
    assert get_instance_hourly_rate("P5.48XLARGE") == 32.77
    assert get_instance_hourly_rate("g5.xlarge") == 1.006
    # Fallback for unknown instance
    assert get_instance_hourly_rate("nonexistent.instance", fallback_rate=99.9) == 99.9


def test_commercial_api_catalogs_consistency():
    """Verify commercial API rates exist and match between models."""
    essential_models = ["gpt-4o", "claude-3.5-sonnet", "deepseek-v3"]
    for model in essential_models:
        assert model in COMMERCIAL_APIS
        assert model in COMMERCIAL_API_PRICING
        assert (
            COMMERCIAL_APIS[model]["cost_1m_input"]
            == COMMERCIAL_API_PRICING[model]["prompt_per_1m"]
        )
        assert (
            COMMERCIAL_APIS[model]["cost_1m_output"]
            == COMMERCIAL_API_PRICING[model]["completion_per_1m"]
        )
        assert COMMERCIAL_APIS[model]["cost_1m_input"] > 0
        assert COMMERCIAL_APIS[model]["cost_1m_output"] > 0


def test_deployment_cost_model_with_centralized_catalog():
    """Verify DeploymentCostModel calculates cost per million tokens correctly."""
    model = DeploymentCostModel()
    # 1000 tokens/sec on g5.xlarge ($1.006/hr)
    breakdown = model.compute_variant_cost(
        variant_name="gpt2-int4",
        hardware_instance="g5.xlarge",
        throughput_tok_s=1000.0,
        pricing_tier="ondemand",
    )
    assert breakdown.hourly_rate_usd == 1.006
    # tokens per hour = 3,600,000 -> cost per 1M tokens = 1.006 / 3.6 = ~0.27944
    expected_cost_1m = 1.006 / 3.6
    assert pytest.approx(breakdown.cost_per_1m_tokens, rel=1e-3) == expected_cost_1m
    assert breakdown.monthly_enterprise_spend_usd > 0
    assert breakdown.annual_enterprise_spend_usd == breakdown.monthly_enterprise_spend_usd * 12.0


def test_inference_cost_tracker_with_centralized_catalog():
    """Verify InferenceCostTracker correctly uses centralized rates and computes savings."""
    tracker = InferenceCostTracker(
        model_variant="deepseek-quant",
        hardware_type="A100-80GB",
        baseline_api="gpt-4o",
    )
    # Record 100k prompt tokens, 50k completion tokens in 100 seconds
    tracker.record_usage(prompt_tokens=100_000, completion_tokens=50_000, duration_seconds=100.0)
    report = tracker.get_report()

    assert report.total_tokens == 150_000
    assert report.total_time_seconds == 100.0
    assert report.hourly_hardware_cost_usd == 2.50
    # Expected hardware cost: (100 / 3600) * 2.50 = 0.06944
    assert pytest.approx(report.hardware_cost_usd, rel=1e-3) == (100.0 / 3600.0) * 2.50
    # Baseline gpt-4o: prompt: 0.1 * 2.50 = 0.25, completion: 0.05 * 10.0 = 0.50 -> total 0.75
    assert pytest.approx(report.baseline_api_cost_usd, rel=1e-3) == 0.75
    assert report.cost_savings_percentage > 85.0
