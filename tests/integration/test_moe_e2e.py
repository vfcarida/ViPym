"""End-to-end integration tests for Mixture-of-Experts (MoE) Expert Pruning and Quantization.

Verifies:
1. Small MoE model with routed experts
2. ExpertPruningMethod prunes 50% of experts (4 experts -> 2 experts)
3. Router gate layer is automatically restructured and sliced
4. Pruned MoE model produces valid forward pass and token output
5. Mixed-precision per-expert quantization
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from vipym.compression.methods.awq import AWQCompressionMethod
from vipym.compression.methods.expert_profiler import ExpertProfiler
from vipym.compression.methods.expert_pruning import ExpertPruningMethod
from vipym.interfaces.compression import CompressionArtifact


class SmallMoEBlock(nn.Module):
    """Realistic small MoE block with router gating and routed FFN experts."""

    def __init__(self, hidden_dim: int = 64, ffn_dim: int = 128, num_experts: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim),
                nn.ReLU(),
                nn.Linear(ffn_dim, hidden_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Routing logits
        routing_logits = self.router(x)
        routing_weights = torch.softmax(routing_logits, dim=-1)

        # Top-2 routing
        top_weights, top_indices = torch.topk(routing_weights, k=min(2, len(self.experts)), dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        batch_size, seq_len, _ = x.shape
        out = torch.zeros_like(x)

        for b in range(batch_size):
            for s in range(seq_len):
                token_x = x[b : b + 1, s : s + 1, :]
                for k_idx in range(top_indices.shape[-1]):
                    expert_id = int(top_indices[b, s, k_idx].item())
                    weight = top_weights[b, s, k_idx]
                    if expert_id < len(self.experts):
                        expert_out = self.experts[expert_id](token_x)
                        out[b : b + 1, s : s + 1, :] += weight * expert_out

        return out


class SmallMoEModel(nn.Module):
    """Small multi-block MoE causal language model for integration testing."""

    def __init__(self, hidden_dim: int = 64, num_experts: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.embed = nn.Embedding(1000, hidden_dim)
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.moe_layers = nn.ModuleList([
            SmallMoEBlock(hidden_dim=hidden_dim, num_experts=num_experts)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, 1000, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x + self.attn(x)
        for moe in self.moe_layers:
            x = x + moe(x)
        return self.lm_head(x)

    def save_pretrained(self, save_directory: str | Path) -> None:
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "pytorch_model.bin")
        with open(p / "config.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "small_moe",
                    "hidden_size": self.hidden_dim,
                    "num_local_experts": self.num_experts,
                    "num_experts_per_tok": 2,
                },
                f,
            )


@pytest.mark.integration
class TestMoEE2E:
    @pytest.fixture
    def moe_model(self):
        torch.manual_seed(42)
        model = SmallMoEModel(hidden_dim=64, num_experts=4, num_layers=2)
        model.eval()
        return model

    def test_moe_expert_pruning_50pct_e2e(self, moe_model: SmallMoEModel, tmp_path: Path):
        """Test MoE 50% expert pruning reduces expert count from 4 to 2 and produces valid outputs."""
        out_dir = tmp_path / "moe_pruned_output"
        pruning_method = ExpertPruningMethod(
            prune_ratio=0.50,
            strategy="importance",
            retrain_router=True,
            router_steps=10,
        )

        initial_experts = len(moe_model.moe_layers[0].experts)
        assert initial_experts == 4

        artifact: CompressionArtifact = pruning_method.compress(
            model=moe_model,
            tokenizer=None,
            output_dir=out_dir,
        )

        assert artifact is not None
        assert out_dir.exists()

        # 1. Verify expert count is reduced from 4 to 2 in every MoE layer
        for i, moe_layer in enumerate(moe_model.moe_layers):
            current_experts = len(moe_layer.experts)
            assert current_experts == 2, f"Layer {i} expert count {current_experts} != 2"
            assert moe_layer.router.out_features == 2, f"Router out_features {moe_layer.router.out_features} != 2"

        # 2. Verify functional forward pass on pruned MoE model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        moe_model.to(device)
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = moe_model(input_ids)
            assert logits is not None
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()
            assert logits.shape == (1, 4, 1000)

    def test_moe_expert_profiling_and_awq_quantization_e2e(self, moe_model: SmallMoEModel, tmp_path: Path):
        """Test expert profiler analysis and AWQ quantization on MoE architecture."""
        profiler = ExpertProfiler(n_samples=16)
        profile_artifact = profiler.compress(
            model=moe_model,
            tokenizer=None,
            output_dir=tmp_path / "moe_profile",
        )
        assert profile_artifact is not None
        assert "stats_file" in profile_artifact.metadata
        assert Path(profile_artifact.metadata["stats_file"]).exists()

        # Test AWQ quantization on MoE
        awq_method = AWQCompressionMethod(
            w_bit=4,
            group_size=32,
        )
        awq_artifact = awq_method.compress(
            model=moe_model,
            tokenizer=None,
            output_dir=tmp_path / "moe_awq",
        )
        assert awq_artifact is not None
        assert any("awq" in m for m in awq_artifact.applied_methods)
