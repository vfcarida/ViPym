# ViPym Experiment Report: `se-lifecycle-demo-gpt2`

> **Research Question:** *How much can we shrink this LLM before reduction in intelligence becomes unacceptable?*

## 1. System & Reproducibility Summary

* **Timestamp:** N/A
* **ViPym Version:** `0.1.0`
* **Git Commit:** `ed45fa8ad69ad391b83e400e3c1d9e203c613415`
* **PyTorch / Transformers / vLLM:** `2.13.0+cpu` / `5.15.1` / `None`
* **GPU Architecture:** 

## 2. Benchmark Comparison Table

| Configuration | Scheme | Pass@1 (%) | P50 Latency (ms) | Peak VRAM (GB) | Est. Cost ($) | Pareto Optimal? |
|---|---|---|---|---|---|---|
| **Baseline** | Native | **0.0%** | 45.0 | 0.3 | $2.50 | - |
| `Compressed (wanda_unstructured_50pct+gptq_w4a16_g128)` | Auto | 0.0% | 28.0 | 0.1 | $1.20 | 🌟 **Yes** |

## 3. Key Findings & Recommendation

Configurations marked with 🌟 represent non-dominated Pareto efficient trade-offs across capability, memory, and operational cost.