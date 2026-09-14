"""Standard cloud instance catalog and pricing presets.

Centralized canonical pricing data for:
- GPU hardware instance tiers (On-Demand, Spot, 1-Year Reserved, 3-Year Reserved)
- Hourly dedicated GPU hardware rates
- Commercial hosted LLM APIs (GPT-4o, Claude 3.5 Sonnet, DeepSeek V3 / R1)
"""

from __future__ import annotations

from dataclasses import dataclass


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


# Standard cloud GPU instance catalog
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
    "p5e.48xlarge": InstancePricing(
        instance_type="p5e.48xlarge",
        gpu_type="H200-141GB",
        gpu_count=8,
        hourly_rate_ondemand=43.16,
        hourly_rate_spot=15.10,
        hourly_rate_reserved_1yr=28.05,
        hourly_rate_reserved_3yr=19.08,
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
    "p4d.24xlarge": InstancePricing(
        instance_type="p4d.24xlarge",
        gpu_type="A100-40GB",
        gpu_count=8,
        hourly_rate_ondemand=32.77,
        hourly_rate_spot=11.47,
        hourly_rate_reserved_1yr=21.30,
        hourly_rate_reserved_3yr=14.50,
    ),
    "g6.48xlarge": InstancePricing(
        instance_type="g6.48xlarge",
        gpu_type="L4-24GB",
        gpu_count=8,
        hourly_rate_ondemand=13.34,
        hourly_rate_spot=4.67,
        hourly_rate_reserved_1yr=8.67,
        hourly_rate_reserved_3yr=5.90,
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
    "g5.48xlarge": InstancePricing(
        instance_type="g5.48xlarge",
        gpu_type="A10G-24GB",
        gpu_count=8,
        hourly_rate_ondemand=16.29,
        hourly_rate_spot=5.70,
        hourly_rate_reserved_1yr=10.59,
        hourly_rate_reserved_3yr=7.20,
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
    "g5.2xlarge": InstancePricing(
        instance_type="g5.2xlarge",
        gpu_type="A10G-24GB",
        gpu_count=1,
        hourly_rate_ondemand=1.21,
        hourly_rate_spot=0.424,
        hourly_rate_reserved_1yr=0.787,
        hourly_rate_reserved_3yr=0.535,
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

# Published on-demand GPU pricing benchmarks (USD per hour)
INSTANCE_PRICING_CATALOG: dict[str, float] = {
    inst_name: inst.hourly_rate_ondemand
    for inst_name, inst in STANDARD_INSTANCES.items()
    if inst_name != "default"
}

# Standard hourly single-GPU rental rates ($/hour)
DEFAULT_HARDWARE_RATES: dict[str, float] = {
    "A100-80GB": 2.50,
    "H100-80GB": 3.50,
    "A10G": 1.00,
    "L40S": 1.50,
    "default": 2.50,
}

# Commercial LLM API catalog ($ / 1M tokens)
COMMERCIAL_APIS: dict[str, dict[str, float]] = {
    "claude_sonnet": {
        "cost_1m_input": 3.00,
        "cost_1m_output": 15.00,
    },
    "claude-3.5-sonnet": {
        "cost_1m_input": 3.00,
        "cost_1m_output": 15.00,
    },
    "gpt_4o": {
        "cost_1m_input": 2.50,
        "cost_1m_output": 10.00,
    },
    "gpt-4o": {
        "cost_1m_input": 2.50,
        "cost_1m_output": 10.00,
    },
    "deepseek_v3": {
        "cost_1m_input": 0.14,
        "cost_1m_output": 0.28,
    },
    "deepseek-v3": {
        "cost_1m_input": 0.14,
        "cost_1m_output": 0.28,
    },
    "deepseek_r1": {
        "cost_1m_input": 0.55,
        "cost_1m_output": 2.19,
    },
    "deepseek-r1": {
        "cost_1m_input": 0.55,
        "cost_1m_output": 2.19,
    },
}

# Commercial LLM API pricing mapping ($ / 1M tokens) with prompt/completion keys
COMMERCIAL_API_PRICING: dict[str, dict[str, float]] = {
    k: {
        "prompt_per_1m": v["cost_1m_input"],
        "completion_per_1m": v["cost_1m_output"],
    }
    for k, v in COMMERCIAL_APIS.items()
}


def get_instance_hourly_rate(instance_type: str, fallback_rate: float = 32.77) -> float:
    """Retrieve standard on-demand hourly compute rate for an instance type."""
    inst_lower = instance_type.lower()
    if inst_lower in STANDARD_INSTANCES:
        return STANDARD_INSTANCES[inst_lower].hourly_rate_ondemand
    return INSTANCE_PRICING_CATALOG.get(inst_lower, fallback_rate)
