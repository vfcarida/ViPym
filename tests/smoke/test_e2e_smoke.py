"""End-to-end smoke test validating complete baseline -> compression -> eval -> pareto -> reporting."""

import tempfile
from pathlib import Path

import pytest

from vipym.config.schema import (
    CostAssumptionConfig,
    EvaluationConfig,
    ModelConfig,
    ServingConfig,
    ViPymExperimentConfig,
)
from vipym.experiments.runner import ResumableExperimentRunner
from vipym.experiments.state import ExperimentState


@pytest.mark.smoke
def test_end_to_end_smoke_pipeline(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ViPymExperimentConfig(
            experiment_id="smoke-test-run",
            model=ModelConfig(id="openai-community/gpt2", revision="main"),
            compression_pipeline=[
                {
                    "stage_id": "wanda_prune",
                    "method": "prune_wanda",
                    "scheme": "UNSTRUCTURED_SPARSITY",
                    "parameters": {"sparsity": 0.5},
                }
            ],
            serving=ServingConfig(backend="hf", tensor_parallel_size=1, max_model_len=1024),
            evaluation=EvaluationConfig(
                suites=["humaneval"],
                task_limit=1,
                timeout_per_task_sec=15,
                max_new_tokens=64,
                allow_unsafe_execution=True,
                isolate_with_gvisor=False,
            ),
            cost_assumptions=CostAssumptionConfig(aws_ec2_hourly_rate=0.5),
        )

        runner = ResumableExperimentRunner(config=config, artifacts_dir=Path(tmpdir))
        result = runner.run(resume=False)

        assert result.final_state == ExperimentState.REPORT_COMPLETED
        assert result.experiment_id == "smoke-test-run"
        assert result.baseline_point is not None
        assert len(result.compressed_points) == 1
        assert (Path(tmpdir) / "smoke-test-run" / "report.html").exists()
        assert (Path(tmpdir) / "smoke-test-run" / "analysis" / "pareto.html").exists()
        assert (Path(tmpdir) / "smoke-test-run" / "analysis" / "recommendation.md").exists()
