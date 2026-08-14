"""Unit tests for configuration schemas and validation."""

import pytest
from pydantic import ValidationError
from vipym.core.config import (
    CostAssumptionConfig,
    ModelConfig,
    ServingConfig,
    ViPymExperimentConfig,
)


def test_valid_experiment_config():
    cfg = ViPymExperimentConfig(
        experiment_id="test-exp-01",
        model=ModelConfig(id="HuggingFaceTB/SmolLM-135M"),
        serving=ServingConfig(backend="vllm", tensor_parallel_size=2),
    )
    assert cfg.experiment_id == "test-exp-01"
    assert cfg.model.id == "HuggingFaceTB/SmolLM-135M"
    assert cfg.serving.tensor_parallel_size == 2


def test_invalid_experiment_id_pattern():
    with pytest.raises(ValidationError):
        ViPymExperimentConfig(
            experiment_id="invalid exp id with spaces!",
            model=ModelConfig(id="test/model"),
        )


def test_duplicate_stage_ids_rejected():
    with pytest.raises(ValidationError):
        ViPymExperimentConfig(
            experiment_id="valid-exp",
            model=ModelConfig(id="test/model"),
            compression_pipeline=[
                {"stage_id": "quant_1", "method": "awq", "scheme": "W4A16"},
                {"stage_id": "quant_1", "method": "gptq", "scheme": "W4A16"},
            ],
        )
