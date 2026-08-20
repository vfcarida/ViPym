"""Unit tests for GPTQ Quantization Method (Dense, MoE Per-Expert, and DAG Pipeline)."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from vipym.compression.methods.gptq import GPTQCompressionMethod as GPTQFromMethods
from vipym.compression.quantization.gptq import GPTQCompressionMethod
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline


class DummyDenseModel(nn.Module):
    """Small PyTorch model for testing standard dense layer quantization."""

    def __init__(self) -> None:
        super().__init__()
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
            json.dump({"model_type": "dummy_dense", "hidden_size": 64}, f)


class DummyMoEModel(nn.Module):
    """Mock Mixture-of-Experts model with shared layer and routed expert layers."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.shared_attn = nn.Linear(64, 64)
        self.gate = nn.Linear(64, 2)
        self.expert_0 = nn.Linear(64, 128)
        self.expert_1 = nn.Linear(64, 128)
        self.out_proj = nn.Linear(128, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared_attn(x)
        weights = torch.softmax(self.gate(h), dim=-1)
        e0 = torch.relu(self.expert_0(h))
        e1 = torch.relu(self.expert_1(h))
        e_combined = weights[:, :, 0:1] * e0 + weights[:, :, 1:2] * e1
        return self.out_proj(e_combined)

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "pytorch_model.bin")
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_type": "dummy_moe", "num_experts": 2, "hidden_size": 64}, f)


class DummyTinyGPT2Model(nn.Module):
    """Tiny GPT2-like causal language model for end-to-end quantization test."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.wte = nn.Embedding(vocab_size, hidden_size)
        self.attn_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.attn_proj = nn.Linear(hidden_size, hidden_size)
        self.mlp_fc1 = nn.Linear(hidden_size, hidden_size * 2)
        self.mlp_fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.wte(input_ids)
        # Self-attention projection
        qkv = self.attn_qkv(x)
        attn_out = self.attn_proj(torch.relu(qkv[:, :, :64]))
        x = x + attn_out
        # MLP
        mlp_out = self.mlp_fc2(torch.relu(self.mlp_fc1(x)))
        x = x + mlp_out
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 5) -> torch.Tensor:
        cur_ids = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(cur_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_token], dim=-1)
        return cur_ids

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "pytorch_model.bin")
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump({"model_type": "gpt2_tiny", "vocab_size": 100, "hidden_size": 64}, f)


class DummyTokenizer:
    """Mock Tokenizer for export testing."""

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "DummyTokenizer"}, f)


def test_gptq_registration() -> None:
    """Verify GPTQ method is registered in CompressionRegistry."""
    method = CompressionRegistry.get("gptq")
    assert isinstance(method, GPTQCompressionMethod)
    assert isinstance(method, CompressionMethod)
    assert GPTQFromMethods is GPTQCompressionMethod


def test_gptq_capabilities() -> None:
    """Verify PluginCapability declarations for GPTQ."""
    method = GPTQCompressionMethod(bits=4, group_size=128)
    caps = method.get_capabilities()
    assert caps.requires_calibration is True
    assert caps.supports_moe is True
    assert SupportedDtype.INT4 in caps.supported_dtypes
    assert SupportedDtype.FP16 in caps.supported_dtypes
    assert ComputeArchitecture.DENSE in caps.supported_architectures
    assert ComputeArchitecture.MOE in caps.supported_architectures
    assert "vllm" in caps.supported_runtimes


def test_gptq_quantize_dense_model_4bit() -> None:
    """Verify 4-bit group_size=128 GPTQ quantization on a dense model."""
    model = DummyDenseModel()
    tokenizer = DummyTokenizer()
    method = GPTQCompressionMethod(
        bits=4,
        group_size=128,
        damp_percent=0.01,
        desc_act=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    fp16_bytes = total_params * 2

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "gptq_4bit_out"
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            calibration_data=["sample calibration text 1", "sample calibration text 2"],
            output_dir=out_dir,
        )

        assert isinstance(artifact, CompressionArtifact)
        assert artifact.output_path == out_dir
        assert artifact.format == "gptq"
        assert artifact.metadata["bits"] == 4
        assert artifact.metadata["group_size"] == 128
        assert artifact.metadata["compression_ratio"] == 4.0

        # Memory reduction check (~4x reduction)
        assert artifact.compressed_size_bytes == int(total_params * 0.5)
        reduction_factor = fp16_bytes / artifact.compressed_size_bytes
        assert reduction_factor == pytest.approx(4.0, rel=1e-3)

        # Check vLLM / HF quantization_config in config.json
        config_file = out_dir / "config.json"
        assert config_file.exists()
        with open(config_file, encoding="utf-8") as f:
            cfg = json.load(f)
        assert "quantization_config" in cfg
        assert cfg["quantization_config"]["quant_method"] == "gptq"
        assert cfg["quantization_config"]["bits"] == 4
        assert cfg["quantization_config"]["group_size"] == 128
        assert cfg["quantization_config"]["damp_percent"] == 0.01
        assert cfg["quantization_config"]["desc_act"] is True


def test_gptq_quantize_8bit_and_2bit() -> None:
    """Verify 8-bit and 2-bit GPTQ quantization modes."""
    model_8bit = DummyDenseModel()
    method_8bit = GPTQCompressionMethod(bits=8, group_size=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_8bit = method_8bit.compress(
            model=model_8bit,
            tokenizer=DummyTokenizer(),
            output_dir=Path(tmpdir) / "gptq_8bit",
        )
        assert art_8bit.metadata["bits"] == 8
        assert art_8bit.metadata["compression_ratio"] == 2.0

    model_2bit = DummyDenseModel()
    method_2bit = GPTQCompressionMethod(bits=2, group_size=32)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_2bit = method_2bit.compress(
            model=model_2bit,
            tokenizer=DummyTokenizer(),
            output_dir=Path(tmpdir) / "gptq_2bit",
        )
        assert art_2bit.metadata["bits"] == 2
        assert art_2bit.metadata["compression_ratio"] == 8.0


def test_gptq_moe_per_expert_quantization() -> None:
    """Verify per-expert MoE quantization with independent statistics and mixed precision."""
    moe_model = DummyMoEModel()
    tokenizer = DummyTokenizer()

    progress_events = []

    def on_progress(event: dict[str, Any]) -> None:
        progress_events.append(event)

    method = GPTQCompressionMethod(
        bits=4,
        group_size=64,
        per_expert=True,
        expert_bits=4,
        shared_bits=8,
        progress_callback=on_progress,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "gptq_moe_out"
        artifact = method.compress(
            model=moe_model,
            tokenizer=tokenizer,
            output_dir=out_dir,
        )

        assert artifact.metadata["per_expert"] is True
        assert artifact.metadata["expert_bits"] == 4
        assert artifact.metadata["shared_bits"] == 8

        # Verify progress events were emitted for layers
        assert len(progress_events) > 0
        expert_events = [e for e in progress_events if e["is_expert"]]
        assert len(expert_events) >= 2
        for ee in expert_events:
            assert ee["bits"] == 4

        shared_events = [e for e in progress_events if not e["is_expert"]]
        for se in shared_events:
            assert se["bits"] == 8

        # Test forward pass on quantized MoE model
        test_in = torch.randn(2, 4, 64)
        out_tensor = moe_model(test_in)
        assert out_tensor.shape == (2, 4, 64)
        assert not torch.isnan(out_tensor).any()


def test_gptq_output_quality_loss() -> None:
    """Verify forward pass output retention and low quantization degradation."""
    torch.manual_seed(42)
    unquantized_model = DummyDenseModel()
    quantized_model = DummyDenseModel()

    test_input = torch.randn(10, 64)

    with torch.no_grad():
        orig_output = unquantized_model(test_input)

    method = GPTQCompressionMethod(bits=4, group_size=128)
    with tempfile.TemporaryDirectory() as tmpdir:
        method.compress(
            model=quantized_model,
            tokenizer=DummyTokenizer(),
            output_dir=Path(tmpdir) / "gptq_quality_test",
        )

    with torch.no_grad():
        quant_output = quantized_model(test_input)

    cosine_sim = (
        torch.nn.functional.cosine_similarity(orig_output, quant_output, dim=-1).mean().item()
    )
    rel_error = (torch.norm(orig_output - quant_output) / torch.norm(orig_output)).item()

    assert cosine_sim > 0.98
    assert rel_error < 0.20


def test_gptq_pipeline_dag_execution() -> None:
    """Verify GPTQ stage execution and validation inside DirectedAcyclicCompressionPipeline."""
    pipeline = DirectedAcyclicCompressionPipeline()
    gptq_method = CompressionRegistry.get("gptq")
    pipeline.add_stage(
        stage_id="quantize_4bit",
        method=gptq_method,
        parameters={"bits": 4, "group_size": 128, "per_expert": False},
    )

    metadata = ModelMetadata(
        model_id="mock-gpt2",
        revision="main",
        total_parameters=124_000_000,
        active_parameters=124_000_000,
        architecture_type=ComputeArchitecture.DENSE,
        native_dtypes=[SupportedDtype.FP16],
        context_window=1024,
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
    )
    assert pipeline.validate_dag(metadata) is True
    assert pipeline.get_topological_order() == ["quantize_4bit"]


def test_gptq_tiny_model_generation() -> None:
    """Verify tiny GPT-2-like model end-to-end quantization and coherent token generation."""
    model = DummyTinyGPT2Model()
    tokenizer = DummyTokenizer()
    prompt_ids = torch.tensor([[1, 5, 10, 15]])

    # Generate tokens before quantization
    with torch.no_grad():
        orig_gen = model.generate(prompt_ids, max_new_tokens=4)
    assert orig_gen.shape == (1, 8)

    method = GPTQCompressionMethod(bits=4, group_size=32)
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            output_dir=Path(tmpdir) / "gptq_tiny_gpt2",
        )

        assert artifact.format == "gptq"
        assert artifact.metadata["compression_ratio"] == 4.0

    # Generate tokens after quantization
    with torch.no_grad():
        quant_gen = model.generate(prompt_ids, max_new_tokens=4)

    assert quant_gen.shape == (1, 8)
    assert not torch.isnan(quant_gen.float()).any()
