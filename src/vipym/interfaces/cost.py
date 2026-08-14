"""Interfaces for Cost Modeling."""

from abc import ABC, abstractmethod

import pydantic


class CostBreakdown(pydantic.BaseModel):
    """Traceable financial cost breakdown."""

    compression_cost_usd: float = 0.0
    inference_evaluation_cost_usd: float = 0.0
    storage_cost_usd: float = 0.0
    data_transfer_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    cost_per_1m_input_tokens_usd: float = 0.0
    cost_per_1m_output_tokens_usd: float = 0.0
    cost_per_successful_task_usd: float = 0.0


class CostModel(ABC):
    """Abstract interface for traceable AWS cloud cost estimation."""

    @abstractmethod
    def estimate_cost(
        self,
        duration_hours: float,
        storage_gb: float,
        data_transfer_gb: float,
        input_tokens: int,
        output_tokens: int,
        successful_tasks: int,
    ) -> CostBreakdown:
        pass
