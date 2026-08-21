# Deployment Recommendation — Experiment se-lifecycle-demo-gpt2

## Executive Summary
Recommended configuration: 'Compressed (wanda_unstructured_50pct+gptq_w4a16_g128)' (none). Achieves 0.0% quality at $1.20/1M tokens with 0 tok/s on standard GPU. Selected from 2 candidates under 'balanced' strategy (min_quality=0.00).

## Recommended Model Configuration
- **Variant**: `Compressed (wanda_unstructured_50pct+gptq_w4a16_g128)`
- **Compression Method**: `none`
- **Quality Score (SE Benchmark)**: **0.0%** (pass@1: 0.000)
- **Serving Cost**: **$1.20** per 1M tokens
- **Inference Latency (p50)**: **28.0 ms**
- **Throughput**: **0 tok/s**
- **Hardware Instance**: `1x NVIDIA H100 SXM5`
- **Compression Ratio**: **4.0x**

## Ranked Candidate Trade-offs
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                      ViPym Compression Recommendations                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Constraint: SE Quality >= 0%                                                 ║
╠════════╦═════════════════════╦═══════╦══════════╦════════════╦═══════════════╣
║ Rank   ║ Variant             ║ Qual% ║ $/1M Tok ║ Tok/s      ║ HW Instance   ║
╠════════╬═════════════════════╬═══════╬══════════╬════════════╬═══════════════╣
║ 1      ║ Compressed (wanda_unstructured_50pct+gptq_w4a16_g128) ║ 0%    ║ $1.20    ║ 0          ║ 1x H100       ║
║ 2      ║ Baseline            ║ 0%    ║ $2.50    ║ 0          ║ 1x H100       ║
╚════════╩═════════════════════╩═══════╩══════════╩════════════╩═══════════════╝
```

## Enterprise Cost Projection (15,000 Developer Organization)
- **Assumed Workload**: 15,000 engineers producing/querying 10M tokens/month (Total: **150 Billion tokens/month**)
- **Baseline Uncompressed Cost**: **$375,000.00 / month** ($4,500,000.00 / year)
- **Compressed `Compressed (wanda_unstructured_50pct+gptq_w4a16_g128)` Cost**: **$180,000.00 / month** ($2,160,000.00 / year)
- **Projected Net Savings**: **$2,340,000.00 / year** (52.0% reduction)
