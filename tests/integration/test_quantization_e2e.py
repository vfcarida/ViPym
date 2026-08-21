"""End-to-end integration tests for Quantization (GPTQ, AWQ, FP8) on GPT-2 architecture.

Verifies:
1. GPT-2 + GPTQ 4-bit: size reduction (<30% of original FP32) + functional generation output
2. GPT-2 + AWQ 4-bit: size reduction (<30% of original FP32) + functional generation output
3. GPT-2 + FP8 (E4M3): size reduction (~50% of original) + functional generation output
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from vipym.compression.methods.awq import AWQCompressionMethod
from vipym.compression.methods.fp8 import FP8CompressionMethod
from vipym.compression.methods.gptq import GPTQCompressionMethod
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

    # Save model
    model.save_pretrained(output_dir)

    # Create and save a simple vocabulary and tokenizer
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
class TestQuantizationE2E:
    @pytest.fixture(autouse=True)
    def setup_model(self, tmp_path: Path):
        self.model_dir = tmp_path / "gpt2_small"
        self.model, self.tokenizer = create_small_gpt2_model(self.model_dir)

        # Calculate original baseline uncompressed parameter size in FP32 bytes
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.fp32_baseline_bytes = self.total_params * 4

    def test_gptq_4bit_quantization_e2e(self, tmp_path: Path):
        """Test GPT-2 + GPTQ 4-bit quantization size reduction and functional generation."""
        out_dir = tmp_path / "gptq_4bit_output"
        gptq_method = GPTQCompressionMethod(
            bits=4,
            group_size=64,
            desc_act=False,
            sym=True,
        )

        artifact: CompressionArtifact = gptq_method.compress(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=out_dir,
        )

        # 1. Verify artifact exists and size reduction < 30% of original FP32 baseline
        assert artifact is not None
        assert out_dir.exists()
        assert any("gptq" in m for m in artifact.applied_methods)

        compressed_ratio = artifact.compressed_size_bytes / self.fp32_baseline_bytes
        # 4-bit is nominally 12.5% of FP32 + group scales / metadata, well under 30%
        assert compressed_ratio < 0.30, f"Compressed ratio {compressed_ratio:.2f} is not < 0.30"

        # 2. Verify config.json contains valid quantization_config
        config_file = out_dir / "config.json"
        assert config_file.exists()
        cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert "quantization_config" in cfg_data
        assert cfg_data["quantization_config"]["bits"] == 4
        assert cfg_data["quantization_config"]["quant_method"] == "gptq"

        # 3. Verify functional output: model forward pass & token generation
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        input_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            assert logits is not None
            assert not torch.isnan(logits).any(), "Quantized model produced NaN logits"
            assert not torch.isinf(logits).any(), "Quantized model produced Inf logits"
            assert logits.shape[0] == 1 and logits.shape[1] == 4

            # Test generation capability
            generated = self.model.generate(
                input_ids,
                max_new_tokens=10,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            assert generated.shape[1] == 14
            assert generated.dtype == torch.long

    def test_awq_4bit_quantization_e2e(self, tmp_path: Path):
        """Test GPT-2 + AWQ 4-bit quantization size reduction and functional generation."""
        out_dir = tmp_path / "awq_4bit_output"
        awq_method = AWQCompressionMethod(
            w_bit=4,
            group_size=64,
            zero_point=True,
        )

        artifact: CompressionArtifact = awq_method.compress(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=out_dir,
        )

        # 1. Verify size reduction < 30% of original FP32 baseline
        assert artifact is not None
        assert out_dir.exists()
        assert any("awq" in m for m in artifact.applied_methods)

        compressed_ratio = artifact.compressed_size_bytes / self.fp32_baseline_bytes
        assert compressed_ratio < 0.30, f"AWQ compressed ratio {compressed_ratio:.2f} is not < 0.30"

        # 2. Verify config.json
        config_file = out_dir / "config.json"
        assert config_file.exists()
        cfg_data = json.loads(config_file.read_text(encoding="utf-8"))
        assert "quantization_config" in cfg_data
        assert cfg_data["quantization_config"]["bits"] == 4
        assert cfg_data["quantization_config"]["quant_method"] == "awq"

        # 3. Verify functional output
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        input_ids = torch.tensor([[5, 15, 25]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            assert logits is not None
            assert not torch.isnan(logits).any()

            generated = self.model.generate(
                input_ids,
                max_new_tokens=8,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            assert generated.shape[1] == 11

    def test_fp8_quantization_e2e(self, tmp_path: Path):
        """Test GPT-2 + FP8 quantization size reduction and functional generation."""
        out_dir = tmp_path / "fp8_output"
        fp8_method = FP8CompressionMethod(
            mode="static",
            weight_dtype="fp8_e4m3",
        )

        artifact: CompressionArtifact = fp8_method.compress(
            model=self.model,
            tokenizer=self.tokenizer,
            output_dir=out_dir,
        )

        # 1. Verify size reduction (~50% of FP16 / 25% of FP32 baseline)
        assert artifact is not None
        assert out_dir.exists()
        assert "fp8_static_fp8_e4m3" in artifact.applied_methods

        compressed_ratio = artifact.compressed_size_bytes / self.fp32_baseline_bytes
        assert compressed_ratio < 0.35, f"FP8 compressed ratio {compressed_ratio:.2f} is not < 0.35"

        # 2. Verify functional output
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits
            assert logits is not None
            assert not torch.isnan(logits).any()
