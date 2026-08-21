"""Global shared pytest fixtures for ViPym test suites."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from vipym.analysis.pareto import ParetoPoint
from vipym.config.schema import (
    CompressionStageConfig,
    CostAssumptionConfig,
    EvaluationConfig,
    ModelConfig,
    OptimizationConfig,
    ServingConfig,
    ViPymExperimentConfig,
)


@pytest.fixture
def sample_experiment_config(tmp_path: Path) -> ViPymExperimentConfig:
    """Generate a valid, lightweight test experiment configuration."""
    return ViPymExperimentConfig(
        experiment_id="test-exp-conftest",
        seed=42,
        description="Fixture-generated experiment configuration for unit tests",
        model=ModelConfig(id="openai-community/gpt2", revision="main"),
        compression_pipeline=[
            CompressionStageConfig(
                stage_id="stage_prune",
                method="prune_wanda",
                scheme="UNSTRUCTURED_SPARSITY",
                dependencies=[],
                parameters={"sparsity": 0.50},
            ),
            CompressionStageConfig(
                stage_id="stage_quant",
                method="gptq",
                scheme="W4A16",
                dependencies=["stage_prune"],
                parameters={"bits": 4, "group_size": 64},
            ),
        ],
        serving=ServingConfig(
            backend="hf",
            tensor_parallel_size=1,
            max_model_len=512,
        ),
        evaluation=EvaluationConfig(
            suites=["humaneval"],
            timeout_per_task_sec=10,
            max_new_tokens=64,
            allow_unsafe_execution=True,
            isolate_with_gvisor=False,
            task_limit=2,
        ),
        cost_assumptions=CostAssumptionConfig(
            provider="custom",
            aws_ec2_hourly_rate=0.50,
        ),
        optimization=OptimizationConfig(
            objectives=["maximize_quality", "minimize_cost"],
            min_acceptable_pass_at_1=0.0,
        ),
    )


@pytest.fixture
def sample_pareto_points() -> list[ParetoPoint]:
    """Provide standard Pareto points for testing optimizer and report generators."""
    return [
        ParetoPoint(
            experiment_id="exp-test",
            configuration_name="Baseline (FP16)",
            quality_score=0.75,
            latency_p50_ms=45.0,
            peak_vram_gb=16.0,
            cost_usd=2.50,
            compression_ratio=1.0,
        ),
        ParetoPoint(
            experiment_id="exp-test",
            configuration_name="Compressed (AWQ-W4A16)",
            quality_score=0.73,
            latency_p50_ms=28.0,
            peak_vram_gb=4.5,
            cost_usd=0.85,
            compression_ratio=3.5,
        ),
        ParetoPoint(
            experiment_id="exp-test",
            configuration_name="Compressed (Wanda+GPTQ)",
            quality_score=0.70,
            latency_p50_ms=22.0,
            peak_vram_gb=3.2,
            cost_usd=0.60,
            compression_ratio=5.0,
        ),
    ]


@pytest.fixture
def allow_unsafe_sandbox(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Enable bare-subprocess execution flag for fast unit tests without Docker."""
    monkeypatch.setenv("VIPYM_ALLOW_UNSAFE", "1")
    yield
