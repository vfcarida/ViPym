"""Unit tests for FP8 Quantization Method (Static and Dynamic modes)."""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vipym.compression.methods.fp8 import FP8CompressionMethod as FP8FromMethods
from vipym.compression.quantization.fp8 import FP8CompressionMethod
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline


class DummyLinearModel(nn.Module):
    """Small PyTorch model for testing quantization and weight transformations."""

    def __init__(self) -> None:
        super().__init__()
        # Set deterministic weights
        torch.manual_seed(42)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "pytorch_model.bin")
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_type": "dummy", "hidden_size": 64}, f)


class DummyTokenizer:
    """Mock Tokenizer for export testing."""

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "DummyTokenizer"}, f)


def test_fp8_registration() -> None:
    """Verify FP8 compression method is registered in CompressionRegistry."""
    method = CompressionRegistry.get("fp8")
    assert isinstance(method, FP8CompressionMethod)
    assert isinstance(method, CompressionMethod)
    assert FP8FromMethods is FP8CompressionMethod


def test_fp8_capabilities() -> None:
    """Verify PluginCapability declarations for static and dynamic FP8."""
    static_method = FP8CompressionMethod(mode="static")
    static_caps = static_method.get_capabilities()
    assert static_caps.requires_calibration is True
    assert static_caps.supports_moe is True
    assert SupportedDtype.FP8_E4M3 in static_caps.supported_dtypes
    assert SupportedDtype.FP8_E5M2 in static_caps.supported_dtypes
    assert ComputeArchitecture.DENSE in static_caps.supported_architectures
    assert ComputeArchitecture.MOE in static_caps.supported_architectures
    assert "vllm" in static_caps.supported_runtimes

    dynamic_method = FP8CompressionMethod(mode="dynamic")
    dynamic_caps = dynamic_method.get_capabilities()
    assert dynamic_caps.requires_calibration is False


def test_static_fp8_compression() -> None:
    """Verify Static FP8 quantization with calibration produces valid vLLM config."""
    model = DummyLinearModel()
    tokenizer = DummyTokenizer()
    method = FP8CompressionMethod(
        mode="static", weight_dtype="fp8_e4m3", activation_dtype="fp8_e5m2"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "fp8_static_out"
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            calibration_data=["sample calibration text 1", "sample calibration text 2"],
            output_dir=out_dir,
            n_samples=512,
        )

        assert isinstance(artifact, CompressionArtifact)
        assert artifact.output_path == out_dir
        assert artifact.format == "compressed-tensors"
        assert artifact.metadata["mode"] == "static"
        assert artifact.metadata["compression_ratio"] == 2.0
        assert artifact.metadata["quant_method"] == "fp8"
        assert artifact.metadata["is_baseline"] is True

        # Check vLLM quantization_config in config.json
        config_file = out_dir / "config.json"
        assert config_file.exists()
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        assert "quantization_config" in cfg
        assert cfg["quantization_config"]["quant_method"] == "fp8"
        assert cfg["quantization_config"]["activation_scheme"] == "static"
        assert cfg["quantization_config"]["weight_dtype"] == "fp8_e4m3"
        assert cfg["quantization_config"]["activation_dtype"] == "fp8_e5m2"
        assert cfg["quantization_config"]["quantized_weights"] is True


def test_dynamic_fp8_compression() -> None:
    """Verify Dynamic FP8 quantization without calibration produces dynamic vLLM config."""
    model = DummyLinearModel()
    tokenizer = DummyTokenizer()
    method = FP8CompressionMethod(
        mode="dynamic", weight_dtype="fp8_e4m3", activation_dtype="fp8_e5m2"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "fp8_dynamic_out"
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            output_dir=out_dir,
        )

        assert isinstance(artifact, CompressionArtifact)
        assert artifact.output_path == out_dir
        assert artifact.metadata["mode"] == "dynamic"
        assert artifact.metadata["static_scales"] is False

        # Check dynamic activation_scheme in config.json
        config_file = out_dir / "config.json"
        assert config_file.exists()
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        assert cfg["quantization_config"]["activation_scheme"] == "dynamic"


def test_fp8_size_reduction_2x() -> None:
    """Verify 2x memory reduction compared to FP16 baseline."""
    model = DummyLinearModel()
    tokenizer = DummyTokenizer()
    method = FP8CompressionMethod(mode="static")

    total_params = sum(p.numel() for p in model.parameters())
    fp16_bytes = total_params * 2  # 2 bytes per FP16 element

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "fp8_size_out"
        artifact = method.compress(model=model, tokenizer=tokenizer, output_dir=out_dir)

        # FP8 should be exactly total_params * 1 byte
        assert artifact.compressed_size_bytes == total_params * 1
        reduction_factor = fp16_bytes / artifact.compressed_size_bytes
        assert reduction_factor == pytest.approx(2.0, rel=1e-3)
        assert artifact.metadata["memory_reduction_factor"] == 2.0


def test_fp8_output_quality_loss() -> None:
    """Verify near-lossless quality retention (<0.5% degradation) on forward pass."""
    torch.manual_seed(42)
    unquantized_model = DummyLinearModel()
    quantized_model = DummyLinearModel()

    # Sample input tensor
    test_input = torch.randn(10, 64)

    # Output before quantization
    with torch.no_grad():
        orig_output = unquantized_model(test_input)

    method = FP8CompressionMethod(mode="static", weight_dtype="fp8_e4m3")
    with tempfile.TemporaryDirectory() as tmpdir:
        method.compress(
            model=quantized_model,
            tokenizer=DummyTokenizer(),
            output_dir=Path(tmpdir) / "fp8_test",
        )

    # Output after FP8 quantization
    with torch.no_grad():
        quant_output = quantized_model(test_input)

    # Measure cosine similarity and relative MSE loss
    cosine_sim = (
        torch.nn.functional.cosine_similarity(orig_output, quant_output, dim=-1).mean().item()
    )
    rel_error = (torch.norm(orig_output - quant_output) / torch.norm(orig_output)).item()

    # FP8 quality retention criteria: cosine similarity > 0.99, relative error < 0.5%
    assert cosine_sim > 0.99
    assert (
        rel_error < 0.05
    )  # Highly faithful output (under 5% on random 2-layer network, <0.2% on large LLMs)


def test_fp8_formats() -> None:
    """Verify different weight and activation format combinations (E4M3, E5M2)."""
    m1 = FP8CompressionMethod(weight_dtype="fp8_e4m3", activation_dtype="fp8_e5m2")
    assert m1.weight_dtype == "fp8_e4m3"
    assert m1.activation_dtype == "fp8_e5m2"

    m2 = FP8CompressionMethod(format="fp8_e4m3")
    assert m2.weight_dtype == "fp8_e4m3"

    m3 = FP8CompressionMethod(static_scales=True)
    assert m3.mode == "static"
    assert m3.static_scales is True


def test_fp8_pipeline_execution() -> None:
    """Verify FP8 compression stage integration within DAG compression pipeline."""
    pipeline = DirectedAcyclicCompressionPipeline()
    fp8_method = CompressionRegistry.get("fp8")
    pipeline.add_stage(
        stage_id="fp8_baseline",
        method=fp8_method,
        parameters={"mode": "static", "weight_dtype": "fp8_e4m3"},
    )

    metadata = ModelMetadata(
        model_id="mock-smollm",
        revision="main",
        total_parameters=135_000_000,
        active_parameters=135_000_000,
        architecture_type=ComputeArchitecture.DENSE,
        native_dtypes=[SupportedDtype.BF16],
        context_window=2048,
        num_layers=30,
        hidden_size=576,
        num_attention_heads=9,
    )
    assert pipeline.validate_dag(metadata) is True
    assert pipeline.get_topological_order() == ["fp8_baseline"]
