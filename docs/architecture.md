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
- **Pydantic v2 Configuration Engine** ([src/vipym/config/schema.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/config/schema.py)): Validates recipe files and parameter bounds before compute allocation.
- **12-State Resumable FSM** ([src/vipym/experiments/state.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/experiments/state.py)): Checkpoints progress at every major milestone (`VALIDATED`, `BASELINE_COMPLETED`, `COMPRESSION_COMPLETED`, `EVALUATION_COMPLETED`, `ANALYSIS_COMPLETED`, `REPORT_COMPLETED`), allowing multi-hour experiments to resume seamlessly after interruptions.
- **DAG Pipeline Planner** ([src/vipym/compression/dag.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/compression/dag.py)): Resolves stage dependencies using Kahn's topological sort.

### 2. Compression Engine
- **Transforms**: Outlier-suppression rotations (`quarot`, `spinquant`).
- **Pruning**: Activation-aware sparsity (`wanda`, `sparsegpt`, `prune_nm`, `prune_magnitude`).
- **Quantization**: Second-order and activation-aware integer and floating-point formats (`awq`, `gptq`, `fp8`, `mxfp`, `smoothquant`).
- **MoE Surgery**: Expert router frequency profiling, surgical pruning, and cluster merging.
- **Distillation**: Cross-architecture student-teacher training.

### 3. Sandboxed Evaluation Runner
- **gVisor / Docker Isolation** ([src/vipym/evaluation/sandbox/docker_sandbox.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/evaluation/sandbox/docker_sandbox.py)): Executes untrusted generated code in isolated containers with total network lockdown (`--network=none`), memory caps, and process timeouts.
- **Inference Engines**: Integration with `vllm`, `sglang`, and `hf` engines for realistic throughput and latency measurements.

### 4. Analysis, Pareto Optimization & Reporting
- **Pareto Frontier Optimizer** ([src/vipym/analysis/pareto.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/analysis/pareto.py)): Multi-objective non-dominated sorting over `(Quality, VRAM, Latency, $/1M tokens)`.
- **Deployment Recommender** ([src/vipym/analysis/recommender.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/analysis/recommender.py)): Synthesizes ranked deployment strategies and ROI projections.
- **Unified Generator** ([src/vipym/reporting/generator.py](file:///c:/Users/vinicius/Documents/GeminiCodes/ViPym/src/vipym/reporting/generator.py)): Produces Plotly interactive 3D/2D visualizers, standalone HTML dashboards, LaTeX publication tables, and Markdown executive summaries.
