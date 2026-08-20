"""End-to-end smoke test validating complete baseline -> compression -> eval -> pareto -> reporting."""

import tempfile
from pathlib import Path

import pytest

from vipym.core.config import (
    CostAssumptionConfig,
    EvaluationConfig,
    ModelConfig,
    ServingConfig,
    ViPymExperimentConfig,
)
from vipym.core.constants import ExecutionStatus
from vipym.core.runner import ViPymRunner


@pytest.mark.smoke
def test_end_to_end_smoke_pipeline(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ViPymExperimentConfig(
            experiment_id="smoke-test-run",
            model=ModelConfig(id="HuggingFaceTB/SmolLM-135M", revision="main"),
            compression_pipeline=[
                {
                    "stage_id": "awq_w4a16",
                    "method": "awq",
                    "scheme": "W4A16",
                    "parameters": {"bits": 4, "group_size": 128},
                }
            ],
            serving=ServingConfig(backend="vllm", tensor_parallel_size=1),
            evaluation=EvaluationConfig(
                suites=["humaneval"],
                task_limit=1,
                timeout_per_task_sec=5,
                allow_unsafe_execution=True,
            ),
            cost_assumptions=CostAssumptionConfig(aws_ec2_hourly_rate=0.0),
        )

        runner = ViPymRunner(config=config, artifacts_dir=tmpdir)
        result = runner.run()

        assert result.status == ExecutionStatus.COMPLETED
        assert result.baseline_point is not None
        assert len(result.compressed_points) == 1
        assert "markdown" in result.generated_report_files
        assert Path(result.generated_report_files["markdown"]).exists()
        assert Path(result.generated_report_files["html"]).exists()
        assert Path(result.generated_report_files["plotly_html"]).exists()
