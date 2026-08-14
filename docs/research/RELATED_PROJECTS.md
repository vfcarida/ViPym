# ViPym — Related Open-Source Projects & Comparative Analysis

| Project | Organization | Scope & Focus | ViPym Distinction |
|---|---|---|---|
| **llm-compressor** | Neural Magic / vLLM | Standalone quantization library for vLLM models (AWQ, GPTQ, SmoothQuant, FP8). | ViPym provides end-to-end DAG orchestration, immutable baseline validation, isolated benchmark sandboxing, multi-objective Pareto analysis, and cloud cost accounting. |
| **AutoAWQ / AutoGPTQ** | Community / Casper-hansen | Single-algorithm quantization tooling. | ViPym coordinates multi-stage compression DAGs (e.g. QuaRot transform $\to$ 2:4 Sparsity $\to$ AWQ $\to$ FP8 KV-Cache) with unified reporting. |
| **vLLM / SGLang** | UC Berkeley / LMSYS | High-throughput LLM serving runtimes. | ViPym integrates vLLM and SGLang as pluggable execution engines for pre- and post-compression latency and throughput benchmarking. |
| **EvalPlus / BigCode-Evaluation-Harness** | UIUC / BigCode | Standalone code generation benchmark suites. | ViPym wraps benchmarks in hardened gVisor sandboxes with credential scrubbing, resource quotas, and contamination audits. |
| **Stanford HELM** | Stanford CRFM | Comprehensive holistic evaluation of language models. | ViPym focuses specifically on the compression-vs-intelligence Pareto frontier, tracking hardware VRAM, cloud cost, and compression ratios alongside task capability. |
