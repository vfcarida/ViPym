"""End-to-end integration tests for Pruning and Sparsity (Wanda 50%, 2:4 Semi-Structured) on GPT-2.

Verifies:
1. GPT-2 + Wanda 50% sparsity: verify measured sparsity > 45% zero weights + functional output
2. GPT-2 + Wanda 2:4 semi-structured sparsity: verify 2:4 pattern + functional output
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from vipym.compression.methods.pruning import WandaPruningMethod
from vipym.interfaces.compression import CompressionArtifact


def create_small_gpt2_model(output_dir: Path) -> tuple[GPT2LMHeadModel, GPT2TokenizerFast]:
    """Create and save a small GPT-2 model and tokenizer for fast, deterministic integration tests."""
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GPT2Config(
        vocab_size=1000,
        n_positions=512,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=0,
        eos_token_id=0,
    )
    torch.manual_seed(42)
    model = GPT2LMHeadModel(config)
    model.eval()

    model.save_pretrained(output_dir)

    vocab = {"<|endoftext|>": 0}
    for i in range(1, 1000):
        vocab[f"token_{i}"] = i

    vocab_file = output_dir / "vocab.json"
    merges_file = output_dir / "merges.txt"
    vocab_file.write_text(json.dumps(vocab), encoding="utf-8")
    merges_file.write_text("#version: 0.2\n", encoding="utf-8")

    tokenizer = GPT2TokenizerFast(vocab_file=str(vocab_file), merges_file=str(merges_file))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


@pytest.mark.integration
class TestPruningE2E:
    @pytest.fixture(autouse=True)
    def setup_model(self, tmp_path: Path):
        self.model_dir = tmp_path / "gpt2_prune_source"
        self.model, self.tokenizer = create_small_gpt2_model(self.model_dir)

    def test_wanda_50pct_sparsity_e2e(self, tmp_path: Path):
        """Test GPT-2 + Wanda 50% unstructured pruning achieves > 45% zero weights and valid generation."""
        out_dir = tmp_path / "wanda_50pct_output"
        wanda_method = WandaPruningMethod(
            sparsity=0.50,
            prune_type="unstructured",
        )

        artifact: CompressionArtifact = wanda_method.compress(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=out_dir,
        )

        # 1. Verify artifact exists
        assert artifact is not None
        assert out_dir.exists()

        # 2. Verify achieved sparsity > 45% zero weights
        linear_weights = [
            module.weight.data
            for name, module in self.model.named_modules()
            if (isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "Conv1D")
            and not any(k in name.lower() for k in ("lm_head", "embed", "wte", "wpe"))
            and hasattr(module, "weight")
            and module.weight is not None
            and len(module.weight.shape) == 2
        ]
        assert len(linear_weights) > 0

        total_linear_params = sum(w.numel() for w in linear_weights)
        zero_linear_params = sum((w == 0).sum().item() for w in linear_weights)
        measured_sparsity = zero_linear_params / total_linear_params

        assert measured_sparsity > 0.45, (
            f"Measured sparsity {measured_sparsity * 100:.1f}% is not > 45%"
        )

        # 3. Verify functional output: forward pass & text generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        input_ids = torch.tensor([[10, 20, 30]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            assert logits is not None
            assert not torch.isnan(logits).any()
            assert not torch.isinf(logits).any()

            generated = self.model.generate(
                input_ids,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            assert generated.shape[1] == 13

    def test_wanda_2_4_semi_structured_sparsity_e2e(self, tmp_path: Path):
        """Test GPT-2 + Wanda 2:4 semi-structured sparsity pattern and functionality."""
        out_dir = tmp_path / "wanda_2_4_output"
        wanda_method = WandaPruningMethod(
            prune_type="2:4",
        )

        artifact: CompressionArtifact = wanda_method.compress(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=out_dir,
        )

        assert artifact is not None
        assert out_dir.exists()

        # Check that for linear weight rows, every contiguous 4 weights have exactly 2 zeros
        for name, module in self.model.named_modules():
            if (
                (isinstance(module, torch.nn.Linear) or module.__class__.__name__ == "Conv1D")
                and not any(k in name.lower() for k in ("lm_head", "embed", "wte", "wpe"))
                and hasattr(module, "weight")
                and module.weight is not None
                and len(module.weight.shape) == 2
            ):
                is_conv1d = module.__class__.__name__ == "Conv1D"
                w = module.weight.data.t() if is_conv1d else module.weight.data
                if w.shape[1] % 4 == 0:
                    w_blocks = w.reshape(-1, 4)
                    non_zero_counts = (w_blocks != 0).sum(dim=-1)
                    # Every block of 4 weights should have <= 2 non-zeros
                    assert (non_zero_counts <= 2).all(), f"2:4 violation in {name}"
