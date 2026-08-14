# Changelog

All notable changes to ViPym will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-14

### Added
- Initial release of **ViPym: Shrinking LLMs, Preserving Intelligence**.
- Decoupled Pipeline-as-a-DAG compression execution engine with Kahn's topological sort.
- Compression adapters: AWQ, GPTQ, SmoothQuant, AutoRound, RTN, FP8, MXFP4/MXFP8, QuaRot, SpinQuant, 2:4 Sparsity, Magnitude Pruning, Wanda, Sequence Distillation, Logit Distillation, and FP8/INT4 KV-Cache quantization.
- Serving runtime adapters for vLLM, SGLang, and Hugging Face.
- Sandboxed evaluation subsystem with gVisor / Docker container isolation and AST validation.
- Coding benchmark adapters: HumanEval, HumanEval+, MBPP, MBPP+, LiveCodeBench, and SWE-bench.
- Contamination auditing engine with n-gram overlap and release cutoff filtering.
- Multi-objective Pareto frontier calculation engine across Capability, Latency, Peak VRAM, and Cost.
- Traceable AWS Cloud Cost Model.
- Resumable experiment lifecycle state machine.
- Comprehensive Typer CLI with `run`, `validate`, `baseline`, `compress`, `evaluate`, `benchmark`, `analyze`, `report`, `doctor`, and `inspect-model`.
- Reference experiment matrix for **Moonshot AI Kimi K3** (2.8T MoE / 104B active).
