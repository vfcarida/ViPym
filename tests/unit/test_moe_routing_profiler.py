"""Unit tests for Authentic MoE Routing Profiler and Co-Activation Matrix generation."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from vipym.compression.methods.expert_profiler import (
    ExpertProfiler,
    _find_moe_blocks,
    _get_expert_modules,
)
from vipym.interfaces.compression import CompressionArtifact


class ToyMoEBlock(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, Seq, Hidden)
        logits = self.gate(x)
        weights = torch.softmax(logits, dim=-1)
        topk_weights, topk_indices = torch.topk(weights, k=2, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x)
        for i, exp in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1).unsqueeze(-1)
            out = out + mask.float() * exp(x)
        return out


class ToyMoEModel(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_experts: int = 4):
        super().__init__()
        self.embed = nn.Embedding(500, hidden_dim)
        self.block = ToyMoEBlock(hidden_dim=hidden_dim, num_experts=num_experts)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        return self.block(h)

    def save_pretrained(self, path: Path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "model.pt")


def test_find_moe_blocks_and_experts():
    model = ToyMoEModel()
    blocks = _find_moe_blocks(model)
    assert len(blocks) == 1
    assert blocks[0][0] == "block"

    experts = _get_expert_modules(blocks[0][1])
    assert len(experts) == 4


def test_moe_profiler_with_real_calibration_data():
    profiler = ExpertProfiler(n_samples=32)
    model = ToyMoEModel()

    # Pass integer token calibration batches
    calib_tokens = torch.randint(0, 500, (4, 16))

    stats = profiler.profile_model(model=model, calibration_data=calib_tokens)

    assert stats["num_moe_layers"] == 1
    assert "block" in stats["layers"]

    layer_stats = stats["layers"]["block"]
    assert layer_stats["num_experts"] == 4
    assert len(layer_stats["experts"]) == 4

    # Check importance ranking
    ranking = layer_stats["importance_ranking"]
    assert len(ranking) == 4
    assert set(ranking) == {0, 1, 2, 3}

    # Verify Co-Activation matrix exists and is 4x4
    co_act = layer_stats["co_activation_matrix"]
    assert len(co_act) == 4
    assert all(len(row) == 4 for row in co_act)
    # Diagonal should be normalized to 1.0
    for i in range(4):
        assert co_act[i][i] == 1.0


def test_moe_profiler_compress_exports_stats_file():
    profiler = ExpertProfiler(n_samples=16, output="test_expert_stats.json")
    model = ToyMoEModel()
    calib_tokens = torch.randint(0, 500, (2, 8))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "expert_profile_test"
        artifact = profiler.compress(
            model=model,
            tokenizer=None,
            calibration_data=calib_tokens,
            output_dir=out_dir,
        )

        assert isinstance(artifact, CompressionArtifact)
        stats_file = out_dir / "test_expert_stats.json"
        assert stats_file.exists()

        data = json.loads(stats_file.read_text(encoding="utf-8"))
        assert data["num_moe_layers"] == 1
        assert "co_activation_matrix" in data["layers"]["block"]


def test_moe_profiler_fallback_without_calibration_data():
    profiler = ExpertProfiler(n_samples=16)
    model = ToyMoEModel()

    stats = profiler.profile_model(model=model, calibration_data=None)

    layer_stats = stats["layers"]["block"]
    # Should fall back to uniform distribution without random noise
    freqs = [exp["frequency"] for exp in layer_stats["experts"]]
    assert freqs == [0.25, 0.25, 0.25, 0.25]
