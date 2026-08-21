"""Traceable Cloud Cost Calculator."""

from vipym.config.schema import CostAssumptionConfig
from vipym.interfaces.cost import CostBreakdown, CostModel


class CloudCostCalculator(CostModel):
    """Calculates granular compute, storage, transfer, per-token, and per-task cloud costs."""

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
        compression_duration_hours: float = 0.0,
        eval_duration_hours: float = 0.0,
    ) -> CostBreakdown:
        hourly_rate = self.config.aws_ec2_hourly_rate
        total_compute_cost = duration_hours * hourly_rate

        comp_cost = (
            compression_duration_hours * hourly_rate
            if compression_duration_hours > 0
            else total_compute_cost * 0.5
        )
        eval_cost = (
            eval_duration_hours * hourly_rate
            if eval_duration_hours > 0
            else total_compute_cost * 0.5
        )

        # Storage (prorated by monthly duration)
        monthly_storage_rate = self.config.s3_storage_cost_per_gb_month
        storage_cost = (storage_gb * monthly_storage_rate) * (max(0.1, duration_hours) / 730.0)

        # Data transfer egress
        transfer_cost = data_transfer_gb * self.config.data_transfer_per_gb

        total_cost = total_compute_cost + storage_cost + transfer_cost

        # Granular per-unit costs
        total_tokens = input_tokens + output_tokens
        cost_per_1m_in = (
            (eval_cost * (input_tokens / max(1, total_tokens)) / max(1, input_tokens)) * 1_000_000
            if input_tokens > 0
            else 0.0
        )
        cost_per_1m_out = (
            (eval_cost * (output_tokens / max(1, total_tokens)) / max(1, output_tokens)) * 1_000_000
            if output_tokens > 0
            else 0.0
        )
        cost_per_task = total_cost / max(1, successful_tasks)

        return CostBreakdown(
            compression_cost_usd=round(comp_cost, 4),
            inference_evaluation_cost_usd=round(eval_cost, 4),
            storage_cost_usd=round(storage_cost, 4),
            data_transfer_cost_usd=round(transfer_cost, 4),
            total_cost_usd=round(total_cost, 4),
            cost_per_1m_input_tokens_usd=round(cost_per_1m_in, 4),
            cost_per_1m_output_tokens_usd=round(cost_per_1m_out, 4),
            cost_per_successful_task_usd=round(cost_per_task, 4),
        )

    def compute_serving_cost_per_1m_tokens(self, throughput_tokens_per_sec: float) -> float:
        """Estimate serving cost per 1M generated tokens based on GPU hourly rate and throughput."""
        hourly_rate = self.config.aws_ec2_hourly_rate
        if throughput_tokens_per_sec <= 0:
            return 2.50
        tokens_per_hour = throughput_tokens_per_sec * 3600.0
        return max(0.01, round((hourly_rate / tokens_per_hour) * 1_000_000, 4))
