"""KV Cache FP8 Quantization and Tuning Example.

Demonstrates how to calibrate, configure, and profile FP8 KV cache quantization
to reduce memory footprint by 50% and increase maximum serving batch capacity.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from vipym.compression.kv_cache.fp8_kv import FP8KVCacheQuantizer


class SimpleAttentionBlock(nn.Module):
    """Synthetic multi-head attention module for demonstration."""

    def __init__(self, hidden_size: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) * 0.1
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


def calculate_kv_cache_memory_mb(
    num_layers: int = 32,
    num_heads: int = 32,
    head_dim: int = 128,
    context_length: int = 8192,
    batch_size: int = 16,
    bytes_per_elem: int = 2,  # 2 for FP16/BF16, 1 for FP8
) -> float:
    """Calculate KV Cache memory footprint in Megabytes."""
    # 2 for Key and Value
    total_elements = 2 * num_layers * num_heads * head_dim * context_length * batch_size
    total_bytes = total_elements * bytes_per_elem
    return total_bytes / (1024.0 * 1024.0)


def main() -> None:
    print("=" * 70)
    print("ViPym FP8 KV-Cache Quantization & Memory Profiling")
    print("=" * 70)

    # 1. Theoretical Memory Analysis
    batch_size = 16
    context_len = 8192
    mem_fp16_mb = calculate_kv_cache_memory_mb(
        context_length=context_len, batch_size=batch_size, bytes_per_elem=2
    )
    mem_fp8_mb = calculate_kv_cache_memory_mb(
        context_length=context_len, batch_size=batch_size, bytes_per_elem=1
    )
    savings_pct = (1.0 - (mem_fp8_mb / mem_fp16_mb)) * 100.0

    print(f"Context Window: {context_len} tokens | Batch Size: {batch_size}")
    print(f"  FP16 KV Cache Memory: {mem_fp16_mb:.1f} MB ({mem_fp16_mb / 1024.0:.2f} GB)")
    print(f"  FP8  KV Cache Memory: {mem_fp8_mb:.1f} MB ({mem_fp8_mb / 1024.0:.2f} GB)")
    print(f"  Memory Savings:       {savings_pct:.1f}% reduction")
    print(f"  Effective Concurrency: ~{mem_fp16_mb / mem_fp8_mb:.1f}x higher request concurrency")

    # 2. Calibrate and export FP8 KV cache configuration
    print("\nCalibrating FP8 Key & Value projection scaling factors...")
    model = nn.Sequential(SimpleAttentionBlock(), SimpleAttentionBlock())
    quantizer = FP8KVCacheQuantizer(kv_dtype="fp8_e4m3", scaling_granularity="per_tensor")

    with TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "fp8_kv_export"
        artifact = quantizer.compress(
            model=model,
            tokenizer=None,
            calibration_data=None,  # Fallback to analytical scaling
            output_dir=out_dir,
        )

        print(f"Exported artifact format: {artifact.format}")
        print(f"Output directory:         {artifact.output_path}")
        print(f"Applied methods:          {artifact.applied_methods}")
        if artifact.metadata and "vllm_config" in artifact.metadata:
            print(f"vLLM runtime flags:       {artifact.metadata['vllm_config']}")

    print("\n=" * 70)
    print("KV Cache FP8 tuning completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
