# ViPym — Target System Architecture & Module Boundaries

The target architecture enforces clean separation of concerns, eliminating duplicate modules while preserving 100% backward compatibility for legacy imports.

```text
vipym/
├── cli/              # User-facing Typer CLI entrypoints & doctor diagnostic suite
├── config/           # Pydantic v2 configuration schemas, exceptions, constants
├── models/           # Foundational model adapters (HF, Kimi K3, LLaMA-3) & ModelRegistry
├── compression/      # Quantization, Sparsity, Pruning, Distillation, Transforms, KV-Cache
├── pipelines/        # Directed Acyclic Graph (DAG) topological execution engine
├── inference/        # Pluggable serving engines (vLLM, SGLang, HuggingFace fallback)
├── evaluation/       # Benchmark suites (HumanEval, MBPP, LiveCodeBench, SWE-bench) & sandboxes
├── security/         # Threat model, environment sanitizer, gVisor/Docker isolation runner
├── metrics/          # Telemetry collector (Peak VRAM, Host RSS, TTFT, ITL), Quality evaluators
├── cost/             # CloudCostCalculator, instance pricing catalog (AWS EC2 presets)
├── artifacts/        # LocalArtifactStore & S3 chunked artifact serialization
├── experiments/      # Lifecycle state machine (12 states), CheckpointManager, ReproducibilityManifest
├── analysis/         # Multi-objective ParetoFrontierOptimizer, Bootstrap confidence intervals
├── reporting/        # HTML dashboard, Markdown reports, LaTeX tables, Plotly WebGL plots
├── aws/              # Ephemeral EC2 lifecycle manager, S3 transfers, CloudWatch metrics, IAM
├── core/             # Backward-compatible proxy module re-exporting from modular subpackages
└── utils/            # GPU discovery, hardware topology, deterministic hashing
```

## Architectural Design Principles

1. **Topological Pipeline Independence:** Compression steps are composed via Kahn's DAG algorithm rather than linear pipelines.
2. **Deterministic Reproducibility:** Every run outputs an immutable `manifest.json` with git SHA, CUDA versions, hardware topology, and configuration hashes.
3. **Fail-Safe Resumability:** In the event of hardware preemption or out-of-memory errors, `vipym run` resumes from the last completed stage snapshot.
4. **Sandboxed Untrusted Code Execution:** All generated benchmark programs execute in hardened gVisor or Docker sandboxes with no network access, isolated memory, and scrubbed credentials.
