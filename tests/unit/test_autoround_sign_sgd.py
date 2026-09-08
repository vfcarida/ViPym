"""Unit tests for AutoRound Sign-SGD layer-wise rounding optimization."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from vipym.compression.quantization.autoround import AutoRoundCompressionMethod
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact


class TinyModel(nn.Module):
    def __init__(self, in_features: int = 64, hidden: int = 64, out_features: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def test_autoround_registry_and_capabilities():
    entry = CompressionRegistry.get("autoround")
    assert isinstance(entry, AutoRoundCompressionMethod) or entry is AutoRoundCompressionMethod

    method = AutoRoundCompressionMethod(bits=4, group_size=128, iters=30)
    caps = method.get_capabilities()
    assert ComputeArchitecture.DENSE in caps.supported_architectures
    assert ComputeArchitecture.MOE in caps.supported_architectures
    assert SupportedDtype.INT4 in caps.supported_dtypes
    assert caps.requires_calibration is True


def test_autoround_sign_sgd_layer_optimization():
    method = AutoRoundCompressionMethod(bits=4, group_size=64, iters=40, lr=0.05)

    torch.manual_seed(42)
    weight = torch.randn(32, 64)
    inputs = torch.randn(20, 64)

    opt_weight, metrics = method._optimize_layer_sign_sgd(
        weight=weight,
        inputs=inputs,
        bits=4,
        group_size=64,
        iters=40,
        lr=0.05,
    )

    assert opt_weight.shape == weight.shape
    assert opt_weight.dtype == weight.dtype
    assert metrics["initial_loss"] > 0.0
    # Sign-SGD should achieve lower or equal reconstruction loss compared to naive rounding
    assert metrics["final_loss"] <= metrics["initial_loss"] + 1e-6
    assert metrics["loss_reduction"] >= 0.0


def test_autoround_rtn_fallback_without_calibration_inputs():
    method = AutoRoundCompressionMethod(bits=4, group_size=64, iters=40)

    weight = torch.randn(16, 64)
    opt_weight, metrics = method._optimize_layer_sign_sgd(
        weight=weight,
        inputs=None,
        bits=4,
        group_size=64,
        iters=40,
        lr=0.05,
    )

    assert opt_weight.shape == weight.shape
    assert metrics["loss_reduction"] == 0.0


def test_autoround_compress_pipeline_execution():
    method = AutoRoundCompressionMethod(bits=4, group_size=64, iters=15)
    model = TinyModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "quantized_autoround"
        calib = ["def foo(): return 42", "def bar(): return 100"]

        artifact = method.compress(
            model=model,
            tokenizer=None,
            calibration_data=calib,
            output_dir=out_dir,
        )

        assert isinstance(artifact, CompressionArtifact)
        assert artifact.format == "compressed-tensors"
        assert artifact.applied_methods == [method.name]
        assert artifact.metadata["native_sign_sgd"] is True
        assert artifact.metadata["layers_optimized"] == 2
        assert artifact.metadata["bits"] == 4
        assert artifact.metadata["iters"] == 15

        meta_file = out_dir / "autoround_metadata.json"
        assert meta_file.exists()
        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta_data["layers_optimized"] == 2
        assert "mean_loss_reduction" in meta_data
