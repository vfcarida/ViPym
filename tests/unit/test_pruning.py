"""Unit tests for Unstructured & Semi-Structured Pruning Methods (SparseGPT and Wanda)."""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from vipym.compression.methods.gptq import GPTQCompressionMethod
from vipym.compression.methods.pruning import (
    SparseGPTPruningMethod,
    UnifiedPruningMethod,
    WandaPruningMethod,
)
from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata
from vipym.pipelines.dag import DirectedAcyclicCompressionPipeline


class DummyDenseModel(nn.Module):
    """Dense PyTorch model for testing pruning methods."""

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
    """Mock Mixture-of-Experts model for testing differential per-expert sparsity."""

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
    """Tiny GPT2-like causal language model for generation tests."""

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
        qkv = self.attn_qkv(x)
        attn_out = self.attn_proj(torch.relu(qkv[:, :, :64]))
        x = x + attn_out
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
    """Mock Tokenizer."""

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump({"tokenizer_class": "DummyTokenizer"}, f)


def test_pruning_registration() -> None:
    """Verify Wanda, SparseGPT, and Unified pruning are registered in CompressionRegistry."""
    wanda = CompressionRegistry.get("wanda")
    sparsegpt = CompressionRegistry.get("sparsegpt")
    unified = CompressionRegistry.get("pruning")

    assert isinstance(wanda, WandaPruningMethod)
    assert isinstance(sparsegpt, SparseGPTPruningMethod)
    assert isinstance(unified, UnifiedPruningMethod)
    assert issubclass(WandaPruningMethod, CompressionMethod)
    assert issubclass(SparseGPTPruningMethod, CompressionMethod)


def test_wanda_unstructured_50pct_sparsity() -> None:
    """Verify Wanda unstructured pruning achieves target 50% sparsity."""
    model = DummyDenseModel()
    tokenizer = DummyTokenizer()
    method = WandaPruningMethod(sparsity=0.50, prune_type="unstructured")

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "wanda_50"
        artifact = method.compress(model=model, tokenizer=tokenizer, output_dir=out_dir)

        assert isinstance(artifact, CompressionArtifact)
        assert artifact.format == "safetensors"
        assert artifact.metadata["algorithm"] == "wanda"

        # Check measured zero parameters in linear layers
        total_linear_params = sum(m.weight.numel() for m in [model.fc1, model.fc2])
        zero_linear_params = sum((m.weight == 0).sum().item() for m in [model.fc1, model.fc2])
        actual_sparsity = zero_linear_params / total_linear_params
        assert actual_sparsity == pytest.approx(0.50, abs=0.02)

        # Verify forward pass
        test_in = torch.randn(4, 64)
        out = model(test_in)
        assert out.shape == (4, 64)
        assert not torch.isnan(out).any()


def test_wanda_2_4_structured_sparsity() -> None:
    """Verify Wanda 2:4 semi-structured pruning keeps exactly 2 non-zeros in each group of 4."""
    model = DummyDenseModel()
    tokenizer = DummyTokenizer()
    method = WandaPruningMethod(prune_type="2:4")

    with tempfile.TemporaryDirectory() as tmpdir:
        method.compress(model=model, tokenizer=tokenizer, output_dir=Path(tmpdir) / "wanda_24")

        # Verify 2:4 pattern for fc1
        w_fc1 = model.fc1.weight.data
        blocks_4 = w_fc1.view(-1, 4)
        zeros_per_block = (blocks_4 == 0).sum(dim=-1)
        assert (zeros_per_block == 2).all().item()

        # Verify 2:4 pattern for fc2
        w_fc2 = model.fc2.weight.data
        blocks_4_fc2 = w_fc2.view(-1, 4)
        zeros_per_block_fc2 = (blocks_4_fc2 == 0).sum(dim=-1)
        assert (zeros_per_block_fc2 == 2).all().item()


def test_sparsegpt_hessian_reconstruction() -> None:
    """Verify SparseGPT second-order inverse Hessian pruning and output quality."""
    torch.manual_seed(42)
    unpruned = DummyDenseModel()
    sparse_model = DummyDenseModel()
    tokenizer = DummyTokenizer()

    test_input = torch.randn(8, 64)
    with torch.no_grad():
        orig_out = unpruned(test_input)

    method = SparseGPTPruningMethod(sparsity=0.50, prune_type="unstructured", damp_percent=0.01)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = method.compress(
            model=sparse_model,
            tokenizer=tokenizer,
            output_dir=Path(tmpdir) / "sparsegpt_out",
        )

        assert artifact.metadata["algorithm"] == "sparsegpt"
        assert artifact.metadata["measured_sparsity"] == pytest.approx(0.50, abs=0.03)

    with torch.no_grad():
        sparse_out = sparse_model(test_input)

    # SparseGPT OBS reconstruction preserves output fidelity (>0.85 cosine similarity)
    cos_sim = torch.nn.functional.cosine_similarity(orig_out, sparse_out, dim=-1).mean().item()
    assert cos_sim > 0.85


def test_moe_per_expert_sparsity() -> None:
    """Verify differential per-expert sparsity on MoE model (25% shared, 60% experts)."""
    moe_model = DummyMoEModel()
    tokenizer = DummyTokenizer()

    method = WandaPruningMethod(
        per_expert_sparsity=True,
        shared_sparsity=0.25,
        expert_sparsity=0.60,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = method.compress(
            model=moe_model,
            tokenizer=tokenizer,
            output_dir=Path(tmpdir) / "moe_sparse",
        )
        assert artifact.metadata["per_expert_sparsity"] is True

        # Check shared layer sparsity
        shared_zeros = (moe_model.shared_attn.weight == 0).sum().item()
        shared_sp = shared_zeros / moe_model.shared_attn.weight.numel()
        assert shared_sp == pytest.approx(0.25, abs=0.03)

        # Check expert layers sparsity
        for exp in [moe_model.expert_0, moe_model.expert_1]:
            exp_zeros = (exp.weight == 0).sum().item()
            exp_sp = exp_zeros / exp.weight.numel()
            assert exp_sp == pytest.approx(0.60, abs=0.03)


class MockModelAdapter:
    """Mock adapter for DAG execution test."""

    def __init__(self, model: nn.Module, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def load_for_compression(self, model_id: str, revision: str = "main") -> nn.Module:
        return self.model

    def get_tokenizer(self, model_id: str, revision: str = "main") -> Any:
        return self.tokenizer


def test_composable_pruning_and_quantization_dag() -> None:
    """Verify DAG pipeline combining 50% Wanda Pruning (Stage 1) + 4-bit GPTQ Quantization (Stage 2)."""
    pipeline = DirectedAcyclicCompressionPipeline()

    wanda_method = CompressionRegistry.get("wanda")
    gptq_method = CompressionRegistry.get("gptq")
    assert isinstance(gptq_method, GPTQCompressionMethod)

    pipeline.add_stage(
        stage_id="stage_01_prune",
        method=wanda_method,
        parameters={"sparsity": 0.50, "prune_type": "unstructured"},
    )
    pipeline.add_stage(
        stage_id="stage_02_quantize",
        method=gptq_method,
        dependencies=["stage_01_prune"],
        parameters={"bits": 4, "group_size": 32},
    )

    metadata = ModelMetadata(
        model_id="mock-smollm-135m",
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

    assert pipeline.validate_dag(metadata) is True
    assert pipeline.get_topological_order() == ["stage_01_prune", "stage_02_quantize"]

    # Execute 2-stage compression pipeline on model
    model = DummyDenseModel()
    tokenizer = DummyTokenizer()
    adapter = MockModelAdapter(model=model, tokenizer=tokenizer)

    with tempfile.TemporaryDirectory() as tmpdir:
        final_artifact = pipeline.execute(
            model_adapter=adapter,  # type: ignore[arg-type]
            model_id="mock-smollm-135m",
            output_dir=Path(tmpdir) / "dag_output",
        )

        assert isinstance(final_artifact, CompressionArtifact)
        assert final_artifact.format == "gptq"
        assert len(final_artifact.applied_methods) == 2
        assert "stage_01_prune" in pipeline.nodes
        assert "stage_02_quantize" in pipeline.nodes
        assert pipeline.nodes["stage_01_prune"].executed is True
        assert pipeline.nodes["stage_02_quantize"].executed is True


def test_tiny_gpt2_pruning_generation() -> None:
    """Verify tiny GPT-2-like model generation after 50% unstructured Wanda pruning."""
    model = DummyTinyGPT2Model()
    tokenizer = DummyTokenizer()
    prompt_ids = torch.tensor([[1, 5, 10, 15]])

    with torch.no_grad():
        orig_gen = model.generate(prompt_ids, max_new_tokens=4)
    assert orig_gen.shape == (1, 8)

    method = WandaPruningMethod(sparsity=0.50)
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact = method.compress(
            model=model,
            tokenizer=tokenizer,
            output_dir=Path(tmpdir) / "pruned_gpt2",
        )
        assert artifact.metadata["algorithm"] == "wanda"

    with torch.no_grad():
        quant_gen = model.generate(prompt_ids, max_new_tokens=4)

    assert quant_gen.shape == (1, 8)
    assert not torch.isnan(quant_gen.float()).any()
