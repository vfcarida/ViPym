# Changelog

All notable changes to ViPym will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-11

### Added
- **ViPym Studio ASGI Architecture**: Modernized high-concurrency web dashboard powered by FastAPI and Uvicorn (`src/vipym/studio/app.py`, `src/vipym/studio/server.py`).
- **Live Model Playground**: Real-time token generation interface with Server-Sent Events (SSE) typewriter streaming (`POST /api/inference/generate`), sub-millisecond TTFT measurement, and tokens/sec throughput dials.
- **Real-Time Observability**: Authenticated WebSocket push stream (`/ws/progress`) for layer-by-layer progress telemetry and VRAM tracking.
- **Topological DAG & MoE Endpoints**: Added `GET /api/dag/graph` for Kahn's DAG visualization and `GET /api/moe/matrix` for MoE expert co-activation heatmaps.
- **Multi-Format Report Export**: On-demand export via `GET /api/reports/{id}/export` supporting Markdown, LaTeX publication tables, and raw JSON.
- **Multi-Experiment Grid Sweep Engine**: New CLI command `vipym sweep` (`src/vipym/experiments/sweep.py`) with Cartesian hyperparameter expansion, state checkpointing (`state.json` / `points/*.json`), and automated non-dominated Pareto frontier discovery.
- **AutoRound Sign-SGD Quantization**: Implemented Sign-SGD optimization with gradient descent tuning for weight rounding scale calibration (`src/vipym/compression/quantization/autoround.py`).
- **FP8 KV-Cache Quantization**: Hardware-aligned key-value cache compression with per-tensor and per-channel scaling (`src/vipym/compression/kv_cache/fp8_kv.py`).
- **Multi-Parent DAG Merging**: Extended `DirectedAcyclicCompressionPipeline` to pass structured upstream artifacts and models into fusion stages (cross-branch model merging, expert fusion).
- **Hermetic Benchmark Isolation**: Added native offline mock mode (`VIPYM_OFFLINE=1`) ensuring 100% deterministic test execution without external Hugging Face dataset network calls.
- **CLI Ergonomics**: Added positional argument support to `vipym validate` matching `vipym run` (`vipym validate recipe.yaml`).
- **Documentation Build Integrity**: Achieved zero-warning `mkdocs build --strict` compliance and added community governance guides under `docs/community/`.
- **Packaging & CI/CD**: Added partitioned PyPI extras (`[studio]`, `[torch]`, `[compression]`, `[serving]`, `[dev]`, `[eval]`, `[all]`) and automated OIDC Trusted Publishing release workflow.

### Fixed
- Fixed network flakiness in BigCodeBench unit tests by introducing offline fallback to embedded canonical tasks.
- Fixed relative source code links in documentation to ensure clean MkDocs compilation.

---

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
