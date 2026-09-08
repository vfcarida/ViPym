"""Unit tests for Production FP8 KV-Cache Quantization Method with Empirical Scale Profiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from vipym.compression.kv_cache.fp8_kv import KVCacheQuantizationMethod
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact
from vipym.interfaces.model import ModelMetadata


class DummyAttentionModel(nn.Module):
    """Synthetic model with attention Key and Value projection layers."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(k + v)

    def save_pretrained(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "model.pt")
        (p / "config.json").write_text(
            json.dumps({"model_type": "dummy_attn", "hidden_size": 64}),
            encoding="utf-8",
        )


class DummyTokenizer:
    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[101, 102, 103]])}

    def save_pretrained(self, path: Path | str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "tokenizer.json").write_text("{}", encoding="utf-8")


class TestFP8KVCacheQuantization:
    def test_registry_discovery(self):
        """Verify KV cache quantization methods are registered in CompressionRegistry."""
        method_e4m3 = CompressionRegistry.get("kv_cache_fp8")
        assert isinstance(method_e4m3, KVCacheQuantizationMethod)
        assert method_e4m3.kv_dtype == "fp8_e4m3"

        method_e5m2 = CompressionRegistry.get("kv_cache_fp8_e5m2")
        assert isinstance(method_e5m2, KVCacheQuantizationMethod)
        assert method_e5m2.kv_dtype == "fp8_e5m2"

        method_int4 = CompressionRegistry.get("kv_cache_int4")
        assert isinstance(method_int4, KVCacheQuantizationMethod)
        assert method_int4.kv_dtype == "int4"

    def test_capabilities(self):
        """Verify plugin capabilities declare correct KV dtypes and runtimes."""
        method = KVCacheQuantizationMethod("fp8_e4m3")
        caps = method.get_capabilities()
        assert caps.supports_moe is True
        assert SupportedDtype.FP8_E4M3 in caps.supported_dtypes
        assert "vllm" in caps.supported_runtimes
        assert "sglang" in caps.supported_runtimes

    def test_calibration_forward_hooks_and_scale_export(self, tmp_path: Path):
        """Verify empirical scale profiling via forward hooks and export of kv_cache_scales.json."""
        model = DummyAttentionModel(hidden_dim=64)
        tokenizer = DummyTokenizer()
        output_dir = tmp_path / "kv_out_empirical"

        # Calibration tensors with known range
        calibration_data = [
            torch.randn(1, 16, 64) * 5.0,
            torch.randn(1, 16, 64) * 10.0,
        ]

        method = KVCacheQuantizationMethod(kv_dtype="fp8_e4m3")
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            calibration_data=calibration_data,
            output_dir=output_dir,
        )

        assert isinstance(artifact, CompressionArtifact)
        assert output_dir.exists()

        # Verify kv_cache_scales.json exists and contains layer scales
        scales_file = output_dir / "kv_cache_scales.json"
        assert scales_file.exists()
        scales_data = json.loads(scales_file.read_text(encoding="utf-8"))
        assert scales_data["kv_cache_dtype"] == "fp8_e4m3"
        assert scales_data["scales_count"] >= 2
        assert "k_proj.scale" in scales_data["scales"]
        assert "v_proj.scale" in scales_data["scales"]
        assert scales_data["scales"]["k_proj.scale"] > 0.0

        # Verify serving_config.json
        serving_file = output_dir / "serving_config.json"
        assert serving_file.exists()
        serving_data = json.loads(serving_file.read_text(encoding="utf-8"))
        assert serving_data["kv_cache_dtype"] == "fp8"
        assert serving_data["kv_cache_scheme"] == "fp8_e4m3"
        assert serving_data["calculate_kv_scales"] is False

        # Verify config.json was updated with quantization_config
        config_file = output_dir / "config.json"
        assert config_file.exists()
        cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert "quantization_config" in cfg_data
        assert cfg_data["quantization_config"]["kv_cache_dtype"] == "fp8_e4m3"

    def test_fallback_without_calibration_data(self, tmp_path: Path):
        """Verify safe fallback scales when calibration data is omitted."""
        model = DummyAttentionModel(hidden_dim=64)
        tokenizer = DummyTokenizer()
        output_dir = tmp_path / "kv_out_fallback"

        method = KVCacheQuantizationMethod(kv_dtype="fp8_e5m2")
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            calibration_data=None,
            output_dir=output_dir,
        )

        assert artifact.metadata["kv_cache_dtype"] == "fp8_e5m2"
        scales_file = output_dir / "kv_cache_scales.json"
        assert scales_file.exists()
        scales_data = json.loads(scales_file.read_text(encoding="utf-8"))
        assert scales_data["scales"]["k_proj.scale"] == pytest.approx(16.0 / 57344.0, rel=1e-3)

    def test_applicability_validation(self):
        """Verify applicability checks on model metadata."""
        method = KVCacheQuantizationMethod()
        metadata = ModelMetadata(
            model_id="mock-smollm",
            revision="main",
            total_parameters=135_000_000,
            active_parameters=135_000_000,
            architecture_type=ComputeArchitecture.DENSE,
            native_dtypes=[SupportedDtype.FP16],
            context_window=2048,
            num_layers=30,
            hidden_size=576,
            num_attention_heads=9,
        )
        method.validate_applicability(metadata)
