"""Enterprise Hardware Cost Modeling, Workload Projection, and API Comparison Engine.

Calculates:
- Hardware serving cost ($/1M tokens) across AWS/GCP/Azure GPU instances and pricing tiers (On-Demand, Spot, 1Yr/3Yr Reserved).
- Enterprise monthly & annual cost projections for large developer fleets (e.g. 15,000 developers).
- Real-time comparison and ROI savings against commercial hosted APIs (Claude 3.5 Sonnet, GPT-4o, DeepSeek).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from vipym.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InstancePricing:
    """Cloud GPU instance specification and pricing tiers."""

    instance_type: str
    gpu_type: str
    gpu_count: int
    hourly_rate_ondemand: float
    hourly_rate_spot: float
    hourly_rate_reserved_1yr: float
    hourly_rate_reserved_3yr: float

    def get_rate(self, pricing_tier: str = "ondemand") -> float:
        tier = pricing_tier.lower()
        if "spot" in tier:
            return self.hourly_rate_spot
        if "3yr" in tier or "3_year" in tier:
            return self.hourly_rate_reserved_3yr
        if "1yr" in tier or "1_year" in tier or "reserved" in tier:
            return self.hourly_rate_reserved_1yr
        return self.hourly_rate_ondemand


# Standard AWS cloud instance catalog
STANDARD_INSTANCES: dict[str, InstancePricing] = {
    "p5.48xlarge": InstancePricing(
        instance_type="p5.48xlarge",
        gpu_type="H100-80GB",
        gpu_count=8,
        hourly_rate_ondemand=32.77,
        hourly_rate_spot=11.47,
        hourly_rate_reserved_1yr=21.30,
        hourly_rate_reserved_3yr=14.50,
    ),
    "p4de.24xlarge": InstancePricing(
        instance_type="p4de.24xlarge",
        gpu_type="A100-80GB",
        gpu_count=8,
        hourly_rate_ondemand=40.96,
        hourly_rate_spot=14.33,
        hourly_rate_reserved_1yr=26.62,
        hourly_rate_reserved_3yr=18.10,
    ),
    "g6.12xlarge": InstancePricing(
        instance_type="g6.12xlarge",
        gpu_type="L40S-48GB",
        gpu_count=4,
        hourly_rate_ondemand=4.944,
        hourly_rate_spot=1.73,
        hourly_rate_reserved_1yr=3.21,
        hourly_rate_reserved_3yr=2.18,
    ),
    "g5.12xlarge": InstancePricing(
        instance_type="g5.12xlarge",
        gpu_type="A10G-24GB",
        gpu_count=4,
        hourly_rate_ondemand=5.672,
        hourly_rate_spot=1.985,
        hourly_rate_reserved_1yr=3.687,
        hourly_rate_reserved_3yr=2.507,
    ),
    "g5.xlarge": InstancePricing(
        instance_type="g5.xlarge",
        gpu_type="A10G-24GB",
        gpu_count=1,
        hourly_rate_ondemand=1.006,
        hourly_rate_spot=0.352,
        hourly_rate_reserved_1yr=0.654,
        hourly_rate_reserved_3yr=0.445,
    ),
    "default": InstancePricing(
        instance_type="default",
        gpu_type="H100-80GB",
        gpu_count=1,
        hourly_rate_ondemand=2.50,
        hourly_rate_spot=0.875,
        hourly_rate_reserved_1yr=1.625,
        hourly_rate_reserved_3yr=1.105,
    ),
}

# Commercial LLM API catalog ($ / 1M tokens)
COMMERCIAL_APIS: dict[str, dict[str, float]] = {
    "claude_sonnet": {
        "cost_1m_input": 3.00,
        "cost_1m_output": 15.00,
    },
    "gpt_4o": {
        "cost_1m_input": 2.50,
        "cost_1m_output": 10.00,
    },
    "deepseek_v3": {
        "cost_1m_input": 0.14,
        "cost_1m_output": 0.28,
    },
    "deepseek_r1": {
        "cost_1m_input": 0.55,
        "cost_1m_output": 2.19,
    },
}


@dataclass
class EnterpriseWorkloadConfig:
    """Enterprise scale workload assumptions for SE code intelligence."""

    developers: int = 15000
    requests_per_dev_per_day: int = 50
    avg_tokens_per_request: int = 2000
    working_days_per_month: int = 22
    prompt_token_ratio: float = 0.70  # 70% input (context/files), 30% output (completion)

    @property
    def total_monthly_tokens(self) -> int:
        return (
            self.developers
            * self.requests_per_dev_per_day
            * self.avg_tokens_per_request
            * self.working_days_per_month
        )

    @property
    def monthly_prompt_tokens(self) -> int:
        return int(self.total_monthly_tokens * self.prompt_token_ratio)

    @property
    def monthly_completion_tokens(self) -> int:
        return self.total_monthly_tokens - self.monthly_prompt_tokens


@dataclass
class VariantCostBreakdown:
    """Cost breakdown for a specific model variant."""

    variant_name: str
    hardware_instance: str
    gpu_type: str
    pricing_tier: str
    hourly_rate_usd: float
    throughput_tok_s: float
    cost_per_1m_tokens: float
    monthly_enterprise_spend_usd: float
    annual_enterprise_spend_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeploymentCostModel:
    """Cost modeling and ROI projection calculator."""

    def __init__(
        self,
        instance_catalog: dict[str, InstancePricing] | None = None,
        workload_config: EnterpriseWorkloadConfig | None = None,
    ) -> None:
        self.catalog = instance_catalog or STANDARD_INSTANCES
        self.workload = workload_config or EnterpriseWorkloadConfig()

    def compute_cost_per_1m_tokens(
        self,
        hourly_rate: float,
        throughput_tok_s: float,
    ) -> float:
        """Compute cost per 1M tokens based on instance hourly cost and throughput."""
        if throughput_tok_s <= 0:
            return 0.0
        tokens_per_hour = throughput_tok_s * 3600.0
        return (hourly_rate / tokens_per_hour) * 1_000_000.0

    def compute_variant_cost(
        self,
        variant_name: str,
        hardware_instance: str,
        throughput_tok_s: float,
        pricing_tier: str = "ondemand",
    ) -> VariantCostBreakdown:
        """Compute detailed cost metrics for a compressed model variant."""
        inst = self.catalog.get(hardware_instance, self.catalog["default"])
        hourly_rate = inst.get_rate(pricing_tier)

        cost_1m = self.compute_cost_per_1m_tokens(hourly_rate, throughput_tok_s)
        monthly_spend = (self.workload.total_monthly_tokens / 1_000_000.0) * cost_1m
        annual_spend = monthly_spend * 12.0

        return VariantCostBreakdown(
            variant_name=variant_name,
            hardware_instance=inst.instance_type,
            gpu_type=inst.gpu_type,
            pricing_tier=pricing_tier,
            hourly_rate_usd=hourly_rate,
            throughput_tok_s=throughput_tok_s,
            cost_per_1m_tokens=cost_1m,
            monthly_enterprise_spend_usd=monthly_spend,
            annual_enterprise_spend_usd=annual_spend,
        )

    def project_commercial_api_monthly_spend(
        self,
        api_name: str = "claude_sonnet",
        workload: EnterpriseWorkloadConfig | None = None,
    ) -> float:
        """Calculate total enterprise monthly spend if using a commercial hosted API."""
        cfg = workload or self.workload
        pricing = COMMERCIAL_APIS.get(api_name.lower(), COMMERCIAL_APIS["claude_sonnet"])

        input_cost = (cfg.monthly_prompt_tokens / 1_000_000.0) * pricing["cost_1m_input"]
        output_cost = (cfg.monthly_completion_tokens / 1_000_000.0) * pricing["cost_1m_output"]
        return input_cost + output_cost

    def compute_annual_roi(
        self,
        self_hosted_monthly_spend: float,
        baseline_api_name: str = "claude_sonnet",
    ) -> dict[str, Any]:
        """Compute annual cost savings and percentage ROI vs commercial API."""
        api_monthly = self.project_commercial_api_monthly_spend(baseline_api_name)
        api_annual = api_monthly * 12.0
        self_hosted_annual = self_hosted_monthly_spend * 12.0

        annual_savings = max(0.0, api_annual - self_hosted_annual)
        savings_pct = (
            ((api_annual - self_hosted_annual) / api_annual * 100.0) if api_annual > 0 else 0.0
        )

        return {
            "baseline_api": baseline_api_name,
            "api_monthly_spend_usd": api_monthly,
            "api_annual_spend_usd": api_annual,
            "self_hosted_monthly_spend_usd": self_hosted_monthly_spend,
            "self_hosted_annual_spend_usd": self_hosted_annual,
            "annual_savings_usd": annual_savings,
            "savings_percentage": savings_pct,
        }

    def generate_comparison_matrix(
        self,
        variants_cost: list[VariantCostBreakdown],
    ) -> list[dict[str, Any]]:
        """Generate unified cost comparison rows across variants and reference commercial APIs."""
        rows: list[dict[str, Any]] = []

        # 1. Add reference APIs
        for api_name, _pricing in COMMERCIAL_APIS.items():
            monthly_cost = self.project_commercial_api_monthly_spend(api_name)
            blended_1m = monthly_cost / (self.workload.total_monthly_tokens / 1_000_000.0)
            rows.append(
                {
                    "name": f"API: {api_name}",
                    "type": "Commercial API",
                    "hardware": "Cloud Hosted",
                    "cost_per_1m_tokens": blended_1m,
                    "monthly_spend_usd": monthly_cost,
                    "annual_spend_usd": monthly_cost * 12.0,
                    "savings_vs_claude_pct": (
                        (
                            1.0
                            - monthly_cost
                            / self.project_commercial_api_monthly_spend("claude_sonnet")
                        )
                        * 100.0
                    ),
                }
            )

        # 2. Add compressed self-hosted variants
        claude_monthly = self.project_commercial_api_monthly_spend("claude_sonnet")
        for v in variants_cost:
            savings = (
                ((1.0 - v.monthly_enterprise_spend_usd / claude_monthly) * 100.0)
                if claude_monthly > 0
                else 0.0
            )
            rows.append(
                {
                    "name": v.variant_name,
                    "type": "Self-Hosted Compressed",
                    "hardware": f"{v.hardware_instance} ({v.gpu_type})",
                    "cost_per_1m_tokens": v.cost_per_1m_tokens,
                    "monthly_spend_usd": v.monthly_enterprise_spend_usd,
                    "annual_spend_usd": v.annual_enterprise_spend_usd,
                    "savings_vs_claude_pct": savings,
                }
            )

        return rows
