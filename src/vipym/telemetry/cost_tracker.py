"""Real-time Inference Cost Tracker and Commercial API Pricing Comparison.

Computes operational cost based on dedicated hardware instance pricing (e.g. A100/H100 hourly rates)
and evaluates cost savings relative to commercial hosted LLM APIs (GPT-4o, Claude 3.5 Sonnet, DeepSeek).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from vipym.core.logger import get_logger

logger = get_logger(__name__)

# Standard hourly hardware rental rates ($/hour)
DEFAULT_HARDWARE_RATES: dict[str, float] = {
    "A100-80GB": 2.50,
    "H100-80GB": 3.50,
    "A10G": 1.00,
    "L40S": 1.50,
    "default": 2.50,
}

# Commercial LLM API pricing ($ / 1 Million tokens)
COMMERCIAL_API_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "prompt_per_1m": 2.50,
        "completion_per_1m": 10.00,
    },
    "claude-3.5-sonnet": {
        "prompt_per_1m": 3.00,
        "completion_per_1m": 15.00,
    },
    "deepseek-v3": {
        "prompt_per_1m": 0.14,
        "completion_per_1m": 0.28,
    },
}


@dataclass
class CostSummaryReport:
    """Detailed financial report on inference cost and API savings."""

    model_variant: str
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_time_seconds: float
    hourly_hardware_cost_usd: float
    hardware_cost_usd: float
    cost_per_1m_tokens: float
    cost_per_1m_prompt_tokens: float
    cost_per_1m_completion_tokens: float
    baseline_api_name: str
    baseline_api_cost_usd: float
    cost_savings_percentage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class InferenceCostTracker:
    """Tracks token volume, execution time, and computes operational hardware vs API costs."""

    def __init__(
        self,
        model_variant: str = "default",
        hardware_type: str = "default",
        hourly_hardware_rate: float | None = None,
        baseline_api: str = "gpt-4o",
    ) -> None:
        self.model_variant = model_variant
        self.hardware_type = hardware_type
        self.hourly_rate = (
            hourly_hardware_rate
            if hourly_hardware_rate is not None
            else DEFAULT_HARDWARE_RATES.get(hardware_type, DEFAULT_HARDWARE_RATES["default"])
        )
        self.baseline_api = baseline_api.lower()

        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_time_seconds: float = 0.0

    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        duration_seconds: float,
    ) -> None:
        """Accumulate token counts and inference execution time."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_time_seconds += duration_seconds

    def get_report(self) -> CostSummaryReport:
        """Calculate hardware operational cost and savings vs baseline API."""
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens

        # Dedicated Hardware Cost
        gpu_hours = self.total_time_seconds / 3600.0
        hardware_cost = gpu_hours * self.hourly_rate

        cost_per_1m = (hardware_cost / total_tokens * 1_000_000.0) if total_tokens > 0 else 0.0
        cost_per_1m_prompt = (
            (hardware_cost / self.total_prompt_tokens * 1_000_000.0) if self.total_prompt_tokens > 0 else 0.0
        )
        cost_per_1m_comp = (
            (hardware_cost / self.total_completion_tokens * 1_000_000.0)
            if self.total_completion_tokens > 0
            else 0.0
        )

        # Baseline Commercial API Cost
        api_rates = COMMERCIAL_API_PRICING.get(self.baseline_api, COMMERCIAL_API_PRICING["gpt-4o"])
        p_cost = (self.total_prompt_tokens / 1_000_000.0) * api_rates["prompt_per_1m"]
        c_cost = (self.total_completion_tokens / 1_000_000.0) * api_rates["completion_per_1m"]
        baseline_api_cost = p_cost + c_cost

        # Savings percentage
        if baseline_api_cost > 0:
            savings = max(0.0, (1.0 - (hardware_cost / baseline_api_cost)) * 100.0)
        else:
            savings = 0.0

        return CostSummaryReport(
            model_variant=self.model_variant,
            total_prompt_tokens=self.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens,
            total_tokens=total_tokens,
            total_time_seconds=self.total_time_seconds,
            hourly_hardware_cost_usd=self.hourly_rate,
            hardware_cost_usd=hardware_cost,
            cost_per_1m_tokens=cost_per_1m,
            cost_per_1m_prompt_tokens=cost_per_1m_prompt,
            cost_per_1m_completion_tokens=cost_per_1m_comp,
            baseline_api_name=self.baseline_api,
            baseline_api_cost_usd=baseline_api_cost,
            cost_savings_percentage=savings,
        )

    def reset(self) -> None:
        """Reset accumulated usage counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_time_seconds = 0.0
