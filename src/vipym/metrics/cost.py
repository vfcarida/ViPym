"""Traceable AWS Cloud Cost Model."""

from vipym.core.config import CostAssumptionConfig
from vipym.interfaces.cost import CostBreakdown, CostModel


class AWSTraceableCostModel(CostModel):
    """Calculates financial cost for an experiment based on explicit AWS rates."""

    def __init__(self, config: CostAssumptionConfig) -> None:
        self.config = config

    def estimate_cost(
        self,
        duration_hours: float,
        storage_gb: float,
        data_transfer_gb: float,
        input_tokens: int,
        output_tokens: int,
        successful_tasks: int,
    ) -> CostBreakdown:
        # Compute cost
        hourly_rate = self.config.aws_ec2_hourly_rate
        compute_cost = duration_hours * hourly_rate

        # Storage cost (prorated for duration in month: 730 hours/month)
        monthly_storage_rate = self.config.s3_storage_cost_per_gb_month
        storage_cost = (storage_gb * monthly_storage_rate) * (duration_hours / 730.0)

        # Data transfer egress
        transfer_cost = data_transfer_gb * self.config.data_transfer_per_gb

        total_cost = compute_cost + storage_cost + transfer_cost

        # Per-token costs
        total_tokens = input_tokens + output_tokens
        cost_per_1m_in = (
            compute_cost * (input_tokens / max(1, total_tokens)) / max(1, input_tokens)
        ) * 1_000_000
        cost_per_1m_out = (
            compute_cost * (output_tokens / max(1, total_tokens)) / max(1, output_tokens)
        ) * 1_000_000
        cost_per_task = total_cost / max(1, successful_tasks)

        return CostBreakdown(
            compression_cost_usd=compute_cost * 0.5,  # Estimated split
            inference_evaluation_cost_usd=compute_cost * 0.5,
            storage_cost_usd=storage_cost,
            data_transfer_cost_usd=transfer_cost,
            total_cost_usd=total_cost,
            cost_per_1m_input_tokens_usd=cost_per_1m_in,
            cost_per_1m_output_tokens_usd=cost_per_1m_out,
            cost_per_successful_task_usd=cost_per_task,
        )
