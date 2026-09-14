"""Unit tests verifying dynamic custom plugin registration and YAML schema validation."""

from pathlib import Path
from typing import Any

import pydantic
import pytest
import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.config.schema import CompressionStageConfig, ViPymExperimentConfig
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability


class MockCustomPluginMethod(CompressionMethod):
    """Test custom compression plugin implementation."""

    def __init__(self, target_bits: int = 3) -> None:
        self.target_bits = target_bits

    @property
    def name(self) -> str:
        return f"mock_custom_int{self.target_bits}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={ComputeArchitecture.DENSE},
            supported_dtypes={SupportedDtype.INT4, SupportedDtype.FP16},
            supports_moe=False,
            requires_calibration=False,
            supported_runtimes={"vllm", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./custom_out")
        out.mkdir(parents=True, exist_ok=True)
        return CompressionArtifact(
            output_path=out,
            format="custom-safetensors",
            compressed_size_bytes=1000,
            applied_methods=[self.name],
        )


@pytest.fixture(autouse=True)
def register_mock_plugin():
    CompressionRegistry.register("custom_quant_int3", MockCustomPluginMethod)
    yield
    # Cleanup registration
    CompressionRegistry._registry.pop("custom_quant_int3", None)


def test_custom_plugin_stage_config_validation():
    stage = CompressionStageConfig(
        stage_id="stage_custom",
        method="custom_quant_int3",
        scheme="CUSTOM_3BIT_SCHEME",
        parameters={"target_bits": 3},
    )
    assert stage.stage_id == "stage_custom"
    assert stage.method == "custom_quant_int3"
    assert stage.scheme == "CUSTOM_3BIT_SCHEME"
    assert stage.parameters["target_bits"] == 3


def test_custom_plugin_case_insensitivity():
    stage = CompressionStageConfig(
        stage_id="stage_custom_upper",
        method="CUSTOM_QUANT_INT3",
        scheme="custom_scheme",
    )
    assert stage.method == "custom_quant_int3"


def test_custom_plugin_in_full_experiment_config():
    config_dict = {
        "experiment_id": "test-custom-plugin-run",
        "model": {"id": "openai-community/gpt2", "revision": "main"},
        "compression_pipeline": [
            {
                "stage_id": "stage_1",
                "method": "custom_quant_int3",
                "scheme": "CUSTOM_3BIT",
                "parameters": {"target_bits": 3},
            }
        ],
        "serving": {"backend": "hf", "tensor_parallel_size": 1},
        "evaluation": {"suites": ["humaneval"], "task_limit": 1},
    }
    cfg = ViPymExperimentConfig.from_dict(config_dict)
    assert cfg.experiment_id == "test-custom-plugin-run"
    assert len(cfg.compression_pipeline) == 1
    assert cfg.compression_pipeline[0].method == "custom_quant_int3"


def test_unknown_compression_method_raises_validation_error():
    with pytest.raises(pydantic.ValidationError) as excinfo:
        CompressionStageConfig(
            stage_id="stage_invalid",
            method="completely_unknown_nonexistent_method_xyz",
            scheme="W4A16",
        )
    err_msg = str(excinfo.value)
    assert "Unknown compression method" in err_msg
    assert "Must be a built-in method or registered via CompressionRegistry" in err_msg
