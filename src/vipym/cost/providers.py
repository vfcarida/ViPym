"""Standard cloud instance catalog and pricing presets."""

from typing import Dict

# Published on-demand GPU pricing benchmarks (USD per hour)
INSTANCE_PRICING_CATALOG: Dict[str, float] = {
    # AWS Accelerated Computing Instances
    "p5.48xlarge": 32.77,       # 8x H100 SXM5 80GB
    "p5e.48xlarge": 43.16,      # 8x H200 SXM5 141GB
    "p4de.24xlarge": 40.96,     # 8x A100 SXM4 80GB
    "p4d.24xlarge": 32.77,      # 8x A100 SXM4 40GB
    "g6.48xlarge": 13.34,       # 8x L4 24GB
    "g5.48xlarge": 16.29,       # 8x A10G 24GB
    "g5.12xlarge": 5.67,        # 4x A10G 24GB
    "g5.2xlarge": 1.21,         # 1x A10G 24GB
    "g5.xlarge": 1.006,         # 1x A10G 24GB (Smoke testing)
}


def get_instance_hourly_rate(instance_type: str, fallback_rate: float = 32.77) -> float:
    """Retrieve standard on-demand hourly compute rate for an instance type."""
    return INSTANCE_PRICING_CATALOG.get(instance_type.lower(), fallback_rate)
