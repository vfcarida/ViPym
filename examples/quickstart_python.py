"""ViPym Python API Quickstart Example."""

from pathlib import Path
from vipym.config.schema import (
    CostAssumptionConfig,
    EvaluationConfig,
    ModelConfig,
    ServingConfig,
    ViPymExperimentConfig,
)
from vipym.experiments.runner import ResumableExperimentRunner


def main():
    config = ViPymExperimentConfig(
        experiment_id="python-api-demo",
        model=ModelConfig(id="HuggingFaceTB/SmolLM-135M", revision="main"),
        compression_pipeline=[
            {
                "stage_id": "awq_stage",
                "method": "awq",
                "scheme": "W4A16",
                "parameters": {"bits": 4, "group_size": 128},
            }
        ],
        serving=ServingConfig(backend="vllm", tensor_parallel_size=1),
        evaluation=EvaluationConfig(suites=["humaneval"], task_limit=1),
        cost_assumptions=CostAssumptionConfig(aws_ec2_hourly_rate=0.0),
    )

    runner = ResumableExperimentRunner(config=config, artifacts_dir="./artifacts")
    summary = runner.run()

    print(f"Experiment {summary.experiment_id} finished in state {summary.final_state}")
    print(f"Baseline Score: {summary.baseline_point.quality_score * 100:.1f}%")
    for pt in summary.compressed_points:
        print(f"Config: {pt.configuration_name} | Score: {pt.quality_score * 100:.1f}% | Pareto: {pt.is_pareto_optimal}")


if __name__ == "__main__":
    main()
