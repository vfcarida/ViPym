"""Unit tests for ViPymRunner and ResumableExperimentRunner consolidation."""

import tempfile
from pathlib import Path

from vipym.config import (
    CostAssumptionConfig,
    EvaluationConfig,
    ExperimentState,
    ModelConfig,
    ServingConfig,
    ViPymExperimentConfig,
)
from vipym.core.constants import ExecutionStatus
from vipym.core.runner import ExperimentExecutionResult, ViPymRunner
from vipym.experiments.runner import ExperimentRunSummary, ResumableExperimentRunner


def _make_test_config(experiment_id: str) -> ViPymExperimentConfig:
    return ViPymExperimentConfig(
        experiment_id=experiment_id,
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


def test_vipym_runner_wrapper_execution(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config("vipym-wrapper-test")
        runner = ViPymRunner(config=config, artifacts_dir=tmpdir)

        # Verify property forwarding
        assert runner.config == config
        assert runner.artifacts_dir == Path(tmpdir) / "vipym-wrapper-test"
        assert runner.manifest is not None
        assert runner.cost_model is not None

        result = runner.run()

        # Verify result and status property compatibility
        assert isinstance(result, ExperimentRunSummary)
        assert result.status == ExecutionStatus.COMPLETED
        assert result.final_state == ExperimentState.REPORT_COMPLETED
        assert result.baseline_point is not None
        assert len(result.compressed_points) == 1

        # Verify checkpoint.json and state.json are NOT persisted when checkpoint_enabled=False
        exp_dir = Path(tmpdir) / "vipym-wrapper-test"
        assert not (exp_dir / "checkpoint.json").exists()
        assert not (exp_dir / "state.json").exists()

        # Manifest, reports, and summary JSONs should still exist
        assert (exp_dir / "manifest.json").exists()
        assert (exp_dir / "reports").exists()


def test_resumable_runner_with_checkpoint_enabled(monkeypatch):
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    with tempfile.TemporaryDirectory() as tmpdir:
        config = _make_test_config("resumable-checkpoint-test")
        runner = ResumableExperimentRunner(
            config=config,
            artifacts_dir=tmpdir,
            checkpoint_enabled=True,
        )
        result = runner.run()

        assert result.status == ExecutionStatus.COMPLETED
        exp_dir = Path(tmpdir) / "resumable-checkpoint-test"
        assert (exp_dir / "checkpoint.json").exists()
        assert (exp_dir / "state.json").exists()
        assert (exp_dir / "manifest.json").exists()


def test_experiment_run_summary_status_property():
    from vipym.analysis.pareto import ParetoPoint

    dummy_point = ParetoPoint(
        experiment_id="test",
        configuration_name="Base",
        quality_score=1.0,
        latency_p50_ms=10.0,
        peak_vram_gb=1.0,
        cost_usd=0.1,
        compression_ratio=1.0,
    )

    summary_completed = ExperimentRunSummary(
        experiment_id="test",
        manifest_id="man-1",
        final_state=ExperimentState.REPORT_COMPLETED,
        baseline_point=dummy_point,
        compressed_points=[],
        generated_report_files={},
        total_duration_sec=1.0,
        total_cost_usd=0.0,
    )
    assert summary_completed.status == ExecutionStatus.COMPLETED

    summary_failed = ExperimentRunSummary(
        experiment_id="test",
        manifest_id="man-1",
        final_state=ExperimentState.FAILED,
        baseline_point=dummy_point,
        compressed_points=[],
        generated_report_files={},
        total_duration_sec=1.0,
        total_cost_usd=0.0,
    )
    assert summary_failed.status == ExecutionStatus.FAILED

    summary_running = ExperimentRunSummary(
        experiment_id="test",
        manifest_id="man-1",
        final_state=ExperimentState.BASELINE_RUNNING,
        baseline_point=dummy_point,
        compressed_points=[],
        generated_report_files={},
        total_duration_sec=1.0,
        total_cost_usd=0.0,
    )
    assert summary_running.status == ExecutionStatus.RUNNING
