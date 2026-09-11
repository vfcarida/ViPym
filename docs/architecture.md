# ViPym System Architecture & Design Overview

ViPym is architected around **strict decoupled symmetry**: compression stages operate independently of benchmark domains, and evaluation engines operate independently of physical tensor storage formats.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         CLI / REST API / Studio UI                       │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
    ┌─────────────────────────┐             ┌─────────────────────────┐
    │     Control Plane       │             │   Plugin Registries     │
    │ ├─ Pydantic Schema      │             │ ├─ CompressionRegistry  │
    │ ├─ Manifest Generator   │             │ ├─ EvaluationRegistry   │
    │ ├─ Resumable FSM        │             │ ├─ ModelRegistry        │
    │ └─ Kahn's DAG Planner   │             │ └─ InferenceRegistry    │
    └────────────┬────────────┘             └────────────┬────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                      Compute & Compression Engine                    │
    │ ├─ Rotational Transforms (QuaRot, SpinQuant)                         │
    │ ├─ Sparsity & Pruning (Wanda, 2:4 Semi-Structured, Magnitude)        │
    │ ├─ Quantization (AWQ, GPTQ, FP8, MXFP4, AutoRound, SmoothQuant)      │
    │ ├─ MoE Surgery (Expert Profiling, Pruning, Merging)                  │
    │ └─ Cross-Architecture Student Distillation                           │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                   Sandboxed Evaluation & Observability               │
    │ ├─ High-Throughput Serving Backends (vLLM, SGLang, HF)               │
    │ ├─ Zero-Trust Docker / gVisor Container Sandbox                      │
    │ ├─ Software Engineering Suites (HumanEval+, BigCodeBench, SWE-bench) │
    │ └─ Telemetry Profiler (TTFT, Throughput, VRAM, Structlog Events)     │
    └────────────────────────────────┬─────────────────────────────────────┘
                                     ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │               Pareto Frontier Optimization & Reporting               │
    │ ├─ Non-Dominated Sorting & Utopia Distance Analysis                  │
    │ ├─ Cloud ROI Modeling (15,000 Developer Scale)                       │
    │ ├─ Automated Human-Readable Deployment Recommendations               │
    │ └─ Unified Artifacts (Interactive Plotly HTML, LaTeX, Markdown)      │
    └──────────────────────────────────────────────────────────────────────┘
```

---

## 1. Core Subsystems

### 1. Control Plane & FSM State Machine
- **Pydantic v2 Configuration Engine** ([src/vipym/config/schema.py](../src/vipym/config/schema.py)): Validates recipe files and parameter bounds before compute allocation.
- **12-State Resumable FSM** ([src/vipym/experiments/state.py](../src/vipym/experiments/state.py)): Checkpoints progress at every major milestone (`VALIDATED`, `BASELINE_COMPLETED`, `COMPRESSION_COMPLETED`, `EVALUATION_COMPLETED`, `ANALYSIS_COMPLETED`, `REPORT_COMPLETED`), allowing multi-hour experiments to resume seamlessly after interruptions.
- **DAG Pipeline Planner** ([src/vipym/pipelines/dag.py](../src/vipym/pipelines/dag.py)): Resolves stage dependencies using Kahn's topological sort with cycle detection. Supports branching and multi-parent fusion stages (e.g. cross-branch model merging and expert fusion), passing upstream parent artifacts and models as structured mappings.

### 2. Compression Engine
- **Transforms**: Outlier-suppression rotations (`quarot`, `spinquant`).
- **Pruning**: Activation-aware sparsity (`wanda`, `sparsegpt`, `prune_nm`, `prune_magnitude`).
- **Quantization**: Second-order and activation-aware integer and floating-point formats (`awq`, `gptq`, `fp8`, `mxfp`, `smoothquant`, `autoround`). AutoRound employs Sign-SGD gradient optimization to tune rounding scales.
- **KV-Cache Compression**: FP8 and INT4 key-value cache quantization (`fp8_kv`, `int4_kv`) to maximize generation throughput and context length.
- **MoE Surgery**: Expert router frequency profiling, surgical pruning, and cluster merging.
- **Distillation**: Cross-architecture student-teacher training.

### 3. Sandboxed Evaluation Runner
- **gVisor / Docker Isolation** ([src/vipym/evaluation/sandbox/docker_sandbox.py](../src/vipym/evaluation/sandbox/docker_sandbox.py)): Executes untrusted generated code in isolated containers with total network lockdown (`--network=none`), memory caps, and process timeouts.
- **Inference Engines**: Integration with `vllm`, `sglang`, and `hf` engines for realistic throughput and latency measurements.
- **Hermetic Benchmark Isolation**: Native offline mock mode (`VIPYM_OFFLINE=1`) ensuring deterministic test execution without external Hugging Face dataset network requests.

### 4. Analysis, Pareto Optimization & Reporting
- **Pareto Frontier Optimizer** ([src/vipym/analysis/pareto.py](../src/vipym/analysis/pareto.py)): Multi-objective non-dominated sorting over `(Quality, VRAM, Latency, $/1M tokens)`.
- **Deployment Recommender** ([src/vipym/analysis/recommender.py](../src/vipym/analysis/recommender.py)): Synthesizes ranked deployment strategies and ROI projections.
- **Unified Generator** ([src/vipym/reporting/generator.py](../src/vipym/reporting/generator.py)): Produces Plotly interactive 3D/2D visualizers, standalone HTML dashboards, LaTeX publication tables, and Markdown executive summaries.

### 5. Multi-Experiment Grid Sweep Engine
- **Hyperparameter Exploration** ([src/vipym/experiments/sweep.py](../src/vipym/experiments/sweep.py)): Executes multi-dimensional parameter grids across compression bit-widths, pruning ratios, and calibration sets using Cartesian expansion.
- **Resumable State Checkpoints**: Persists evaluation progress in `state.json` with individual point results stored in `points/`, allowing preempted or interrupted sweeps to safely resume.
- **Automated Pareto Discovery**: Automatically filters dominated parameter combinations and exports the global non-dominated frontier to `pareto_frontier.json` and a Rich terminal summary table.

### 6. ViPym Studio (ASGI Web Architecture)
- **High-Concurrency ASGI Backend** ([src/vipym/studio/server.py](../src/vipym/studio/server.py), [src/vipym/studio/app.py](../src/vipym/studio/app.py)): Built on FastAPI and Uvicorn with asynchronous non-blocking request handling.
- **Live Model Playground**: Interactive token generation interface with Server-Sent Events (SSE) streaming for real-time typewriter output, TTFT, and throughput metrics.
- **Real-Time Observability**: Authenticated WebSocket (`/ws/progress`) pushing layer-by-layer compression metrics and memory consumption.
- **Security Controls**: Bearer token authentication (`VIPYM_API_TOKEN`), rate limiting (100 req/min), audit logging, and read-only mode (`--read-only`).

